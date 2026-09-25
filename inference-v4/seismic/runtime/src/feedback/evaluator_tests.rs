//! A downstream evaluator selects ordinary domain coordinates, then executes
//! admitted native artifacts through the compiler's production command path.
struct FixedTestResources<'a, B>(&'a [RuntimeBuffer<B>]);
impl<B> seismic_compiler::executable::ExecutionResources<B> for FixedTestResources<'_, B> {
    fn retire_completed_instances(
        &mut self,
        _: &[seismic_compiler::executable::ExecutableAllocationId],
    ) {
    }
    fn buffer(
        &self,
        id: seismic_compiler::executable::ExecutableAllocationId,
    ) -> &RuntimeBuffer<B> {
        &self.0[id.ordinal()]
    }
    fn acquire_instance(
        &mut self,
        id: seismic_compiler::executable::ExecutableAllocationId,
        bytes: u64,
        _: u64,
    ) -> Result<(), seismic_compiler::errors::ExecutionError> {
        assert!(
            self.0[id.ordinal()].accessible_bytes >= bytes,
            "fixed test backing is insufficient"
        );
        Ok(())
    }
}

use seismic_compiler::candidate_domain::{
    BodyMapping, ConstructionCoordinate, Materialization, NonEmpty,
};
use seismic_compiler::errors::PreparationError;
use seismic_compiler::evaluation::{
    CandidateEvaluator, EvaluationIdentity, EvaluationProvenance, EvaluationSession,
    PreparedCandidateId, RealizationAdmission,
};
use seismic_compiler::executable::{
    execute_variant, DeviceService, ExecutableCommand, ExecutionEnvironment, NativeExecution,
    NativeExecutor, NativeSubmission, RuntimeBuffer,
};
use seismic_compiler::planning::{
    OptimizationCompletion, PlanningBudgetReport, PlanningLimit, PlanningReport, SelectionPolicy,
};
use seismic_compiler::prepared::SelectionFunction;
use seismic_compiler::refinement::{ChoiceKind, PhysicalChoice};
use seismic_compiler::{prepare_with_evaluator, PlanningBudget, PreparationBudget};
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::ElementBindings;
use seismic_lang::expr::compiled::{constant_predicate, InvocationValues};
use seismic_lang::precision::PrecisionPolicy;
use seismic_lang::{registry, types::DType};
use seismic_metal::{DeviceHandle, Metal, MetalBuffer, MetalDevice, MetalExecutor, Pipeline};
use seismic_native_target::{
    DeviceDescription, NativeCompilationError, NativeCompiler, NativeKernelReflection, TargetFamily,
};
use std::sync::atomic::{AtomicU32, Ordering};

struct SelectedBody {
    mapping: BodyMapping,
    workgroup: i64,
    selected: Option<PreparedCandidateId>,
}
impl<T: TargetFamily, C: NativeCompiler<T>> CandidateEvaluator<T, C> for SelectedBody {
    fn evaluate(
        &mut self,
        session: &mut EvaluationSession<'_, T, C>,
    ) -> Result<(SelectionPolicy, EvaluationIdentity, PlanningReport), PreparationError> {
        let general = session
            .domain()
            .canonicalize(session.domain().universal_proposal())
            .unwrap();
        let RealizationAdmission::Ready(general) = session.realize_checked(&general)? else {
            panic!("general candidate must be ready")
        };
        let body = session
            .domain()
            .root_selections()
            .into_iter()
            .find(|body| body.mapping == self.mapping)
            .expect("selected authored or independent body exists");
        let mut construction = ConstructionCoordinate::root(body);
        loop {
            let allowance = session.construction_allowance();
            match session.advance(&construction, allowance).state {
                Materialization::Ready(ready) => {
                    construction = ready;
                    break;
                }
                Materialization::Choice(choice) => {
                    let selected = choice
                        .alternatives
                        .iter()
                        .find(|body| body.mapping == BodyMapping::Sequential)
                        .cloned()
                        .unwrap_or_else(|| choice.alternatives[0].clone());
                    construction = construction.select(&choice, selected);
                }
                other => panic!("chosen source construction did not complete: {other:?}"),
            }
        }
        let axes = {
            let read = session.domain().read_materialized(&construction).unwrap();
            read.candidate()
                .choices()
                .iter()
                .enumerate()
                .map(|(ordinal, choice)| {
                    let allowed = read.arena().decision_domain(choice.decision()).values();
                    let selected = if choice.kind() == ChoiceKind::WorkgroupSize
                        && allowed.contains(&self.workgroup)
                    {
                        self.workgroup
                    } else {
                        allowed[0]
                    };
                    (
                        PhysicalChoice {
                            ordinal: ordinal as u32,
                            kind: choice.kind(),
                        },
                        selected,
                    )
                })
                .collect()
        };
        let point = session
            .domain()
            .canonicalize(session.domain().proposal(construction, axes))
            .unwrap();
        let selected = match session.realize_checked(&point)? {
            RealizationAdmission::Ready(id) => id,
            other => panic!(
                "chosen coordinate did not pass ordinary numerical/native admission: {other:?}"
            ),
        };
        self.selected = Some(selected);
        let policy = session.policy(
            NonEmpty::new(vec![(general, ()), (selected, ())]).unwrap(),
            |_, candidates| {
                let selected = candidates.index(1).unwrap();
                SelectionFunction::ordered_decision(
                    candidates,
                    vec![(constant_predicate(true), selected)],
                )
            },
        )?;
        Ok((
            policy,
            EvaluationIdentity::new(
                session.domain().device_identity().clone(),
                EvaluationProvenance::new([71; 32], [19; 32]),
            ),
            PlanningReport {
                optimization: OptimizationCompletion::Limited(PlanningLimit::CandidateConstruction),
                budget: PlanningBudgetReport::default(),
            },
        ))
    }
}

#[derive(Default)]
struct RecordingCompiler {
    width: AtomicU32,
}
impl NativeCompiler<Metal> for RecordingCompiler {
    type Context = DeviceHandle;
    type Candidate = seismic_metal::NativeCandidate;
    type Handle = Pipeline;
    fn form(
        &self,
        context: &DeviceHandle,
        target: &DeviceDescription<Metal>,
        kernel: &seismic_ir::kernel::Kernel<Metal>,
        layout: &seismic_ir::physical_target::KernelEmissionLayout,
    ) -> Result<Self::Candidate, NativeCompilationError> {
        seismic_metal::native_compiler().form(context, target, kernel, layout)
    }
    fn reflect(
        &self,
        target: &DeviceDescription<Metal>,
        kernel: &seismic_ir::kernel::Kernel<Metal>,
        layout: &seismic_ir::physical_target::KernelEmissionLayout,
        candidate: Self::Candidate,
    ) -> Result<NativeKernelReflection<Metal, Pipeline>, NativeCompilationError> {
        let result = seismic_metal::native_compiler().reflect(target, kernel, layout, candidate)?;
        if let Some(width) = result.description().launch.subgroup_width {
            self.width.store(width, Ordering::Relaxed);
        }
        Ok(result)
    }
}

struct AuditSubmission {
    inner: seismic_metal::executor::MetalSubmission,
    status: Option<(MetalBuffer, u64, u64)>,
}
impl NativeSubmission<Metal> for AuditSubmission {
    type Handle = Pipeline;
    type Device = MetalDevice;
    type Execution = seismic_metal::executor::MetalExecution;
    fn execute(
        &mut self,
        command: &ExecutableCommand<Metal>,
        env: &mut ExecutionEnvironment<'_, Metal, Pipeline, MetalDevice>,
    ) -> Result<(), seismic_compiler::errors::ExecutionError> {
        if let ExecutableCommand::ScalarRead { source, .. } = command {
            if source.representation == registry::dense(DType::U32) {
                let view = env.resolve_view(source)?;
                self.status = Some((
                    view.buffer.clone(),
                    view.byte_offset,
                    view.extents.iter().product::<u64>() * 4,
                ));
            }
        }
        self.inner.execute(command, env)
    }
    fn complete_prefix(&mut self) -> Result<(), seismic_compiler::errors::ExecutionError> {
        self.inner.complete_prefix()
    }
    fn submit(self) -> Self::Execution {
        self.inner.submit()
    }
}

#[test]
#[ignore = "requires Metal device"]
fn authored_failure_before_subgroup_preserves_prefix_and_reconverges_after_if() {
    let device = MetalDevice::open(DeviceHandle::system_default().unwrap()).unwrap();
    let target = seismic_metal::profile::open_device(&device).unwrap();
    let registry = seismic_metal::profile::registry();
    let module=check_source(SourceSet::new(vec![SourceFile { path:"cohort-failure.seismic".into(),text:r#"fn probe(indices: &tensor[64] i32, later: &tensor[64] i32, input: &tensor[1] f32, out: &mut tensor[64] f32):
    parallel for i in 0..64:
        out[i] = 7.0
        let value = input[indices[i]]
        if i < 32:
            out[i] = value + 1.0
        else:
            out[i] = value + 2.0
        let next = input[later[i]]
        out[i] = next

lower probe(indices: &tensor[64] i32, later: &tensor[64] i32, input: &tensor[1] f32, out: &mut tensor[64] f32)
    for metal requires metal.subgroup:
    parallel for i in 0..64:
        out[i] = 7.0
        let value = input[indices[i]]
        let lane = metal.subgroup.lane_index()
        let first = metal.subgroup.shuffle(value, lane)
        if i < 32:
            out[i] = first + 1.0
        else:
            out[i] = first + 2.0
        let next = input[later[i]]
        let second = metal.subgroup.shuffle(next, lane)
        out[i] = second
"#.into() }])).unwrap();
    let entry = module
        .entry(
            module.entry_named("probe").unwrap(),
            &ElementBindings::default(),
        )
        .unwrap();
    let mut budget = PreparationBudget::default();
    budget.construction_wall_time = std::time::Duration::from_secs(30);
    budget.native_compile_wall_time = std::time::Duration::from_secs(30);
    let compiler = RecordingCompiler::default();
    let mut session = EvaluationSession::new(
        entry,
        &target,
        &registry,
        &compiler,
        device.handle(),
        &PrecisionPolicy::Exact,
        &budget,
        &PlanningBudget::default(),
    )
    .unwrap();
    let mut evaluator = SelectedBody {
        mapping: BodyMapping::Authored,
        workgroup: 64,
        selected: None,
    };
    let prepared = prepare_with_evaluator(&mut session, &mut evaluator).unwrap();
    assert_eq!(prepared.variants().len(), 2);
    let id = evaluator
        .selected
        .expect("strategy prepared its authored candidate");
    let width = compiler.width.load(Ordering::Relaxed) as usize;
    assert!(width > 0 && width <= 64 && 64 % width == 0);
    let values = InvocationValues::new();
    assert_eq!(prepared.select(&values).as_usize(), 1);
    let variant = &prepared.variants().as_slice()[1];
    assert_eq!(
        variant.identity(),
        session.executable(id).unwrap().identity()
    );
    let authored_body = session
        .domain()
        .root_selections()
        .into_iter()
        .find(|body| body.mapping == BodyMapping::Authored)
        .unwrap()
        .body;
    drop(session);
    let executor = MetalExecutor::new(device.clone());
    for failing in [false, true] {
        let mut values = InvocationValues::new();
        assert!(variant.guard().evaluate(&values).unwrap());
        let buffers = variant
            .allocations()
            .iter()
            .enumerate().map(|(ordinal, allocation)| {
                let bytes: u64 = std::iter::once(allocation.byte_candidates().first())
                    .chain(allocation.byte_candidates().rest())
                    .map(|n| n.evaluate(&values).unwrap())
                    .max()
                    .unwrap()
                    .try_into()
                    .expect("fixture allocation must fit the device byte address domain");
                let buffer = device.allocate(bytes.max(1), allocation.alignment).unwrap();
                device
                    .write(
                        &buffer,
                        0,
                        &vec![
                            0;
                            usize::try_from(bytes).expect("fixture allocation fits host memory")
                        ],
                    )
                    .unwrap();
                RuntimeBuffer {
                    tensor: variant.bindings().arguments.iter().flatten().find(|view| view.allocation_index() == ordinal).map(|view| seismic_compiler::executable::RuntimeTensorGeometry {
                        representation: view.representation,
                        extents: view.extents.iter().map(|value| value.evaluate_u64(&values).unwrap()).collect(),
                        strides: view.strides.iter().map(|value| value.evaluate_u64(&values).unwrap()).collect(),
                    }),
                    buffer,
                    base_offset: 0,
                    accessible_bytes: bytes,
                }
            })
            .collect::<Vec<_>>();
        let mut indices = vec![0i32; 64];
        let mut later = vec![0i32; 64];
        if failing {
            indices[0] = 1;
            later[..width].fill(1);
        }
        for (argument, bytes) in [
            indices
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
            later.into_iter().flat_map(i32::to_le_bytes).collect(),
            3.0f32.to_le_bytes().to_vec(),
            vec![0; 64 * 4],
        ]
        .into_iter()
        .enumerate()
        {
            let view = variant.bindings().arguments[argument].as_ref().unwrap();
            device
                .write(
                    &buffers[view.allocation_index()].buffer,
                    view.byte_offset
                        .evaluate(&values)
                        .unwrap()
                        .try_into()
                        .expect("fixture offset must fit the device byte address domain"),
                    &bytes,
                )
                .unwrap();
        }
        let mut submission = AuditSubmission {
            inner: executor.begin_submission().unwrap(),
            status: None,
        };
        let result = execute_variant(
            variant,
            &mut submission,
            &device,
            &mut FixedTestResources(&buffers),
            &mut values,
        );
        let status = submission
            .status
            .clone()
            .expect("source status read executes");
        let mut execution = submission.submit();
        execution.complete().unwrap();
        if failing {
            let Err(seismic_compiler::errors::ExecutionError::DataCheckFailed(failure)) = result
            else {
                panic!("expected a typed source failure, got {result:?}");
            };
            assert_eq!(failure.failure.event.body(), authored_body);
            assert_eq!(
                failure.failure.cause,
                seismic_lang::failure::SourceFailureCause::Check(
                    seismic_lang::entry::CheckReason::IndexBound
                )
            );
        } else {
            result.unwrap();
        }
        let output = variant.bindings().arguments[3].as_ref().unwrap();
        let mut bytes = vec![0u8; 64 * 4];
        device
            .read(
                &buffers[output.allocation_index()].buffer,
                output
                    .byte_offset
                    .evaluate(&values)
                    .unwrap()
                    .try_into()
                    .expect("fixture offset must fit the device byte address domain"),
                &mut bytes,
            )
            .unwrap();
        let actual = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            (0..64)
                .map(|i| if failing && i < width { 7.0 } else { 3.0 })
                .collect::<Vec<_>>()
        );
        let mut bytes = vec![0u8; status.2 as usize];
        device.read(&status.0, status.1, &mut bytes).unwrap();
        let failures = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            failures.iter().filter(|v| **v != 0).count(),
            usize::from(failing),
            "stopped peers must not invent a later failure: {failures:?}"
        );
    }
}

#[test]
fn cpu_public_evaluator_selects_and_executes_an_independent_body() {
    let opened = seismic_cpu::open_host().unwrap();
    let registry = seismic_cpu::registry();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "public-evaluator.seismic".into(),
        text: r#"fn probe(input: &tensor[4] i32, out: &mut tensor[4] i32):
    parallel for i in 0..4:
        out[i] = input[i] + 1
"#
        .into(),
    }]))
    .unwrap();
    let entry = module
        .entry(
            module.entry_named("probe").unwrap(),
            &ElementBindings::default(),
        )
        .unwrap();
    let mut session = EvaluationSession::new(
        entry,
        &opened.device,
        &registry,
        seismic_cpu::native_compiler(),
        &(),
        &PrecisionPolicy::Exact,
        &PreparationBudget::default(),
        &PlanningBudget::default(),
    )
    .unwrap();
    let mut evaluator = SelectedBody {
        mapping: BodyMapping::Independent,
        workgroup: 1,
        selected: None,
    };
    let prepared = prepare_with_evaluator(&mut session, &mut evaluator).unwrap();
    drop(session);
    let mut values = InvocationValues::new();
    assert_eq!(prepared.select(&values).as_usize(), 1);
    let variant = &prepared.variants().as_slice()[1];
    let buffers = variant
        .allocations()
        .iter()
        .enumerate().map(|(ordinal, allocation)| {
            let bytes: u64 = std::iter::once(allocation.byte_candidates().first())
                .chain(allocation.byte_candidates().rest())
                .map(|n| n.evaluate(&values).unwrap())
                .max()
                .unwrap()
                .try_into()
                .expect("fixture allocation must fit the device byte address domain");
            let buffer = opened
                .service
                .allocate(bytes.max(1), allocation.alignment)
                .unwrap();
            opened
                .service
                .write(
                    &buffer,
                    0,
                    &vec![0; usize::try_from(bytes).expect("fixture allocation fits host memory")],
                )
                .unwrap();
            RuntimeBuffer {
                tensor: variant.bindings().arguments.iter().flatten().find(|view| view.allocation_index() == ordinal).map(|view| seismic_compiler::executable::RuntimeTensorGeometry {
                    representation: view.representation,
                    extents: view.extents.iter().map(|value| value.evaluate_u64(&values).unwrap()).collect(),
                    strides: view.strides.iter().map(|value| value.evaluate_u64(&values).unwrap()).collect(),
                }),
                buffer,
                base_offset: 0,
                accessible_bytes: bytes,
            }
        })
        .collect::<Vec<_>>();
    let input = variant.bindings().arguments[0].as_ref().unwrap();
    opened
        .service
        .write(
            &buffers[input.allocation_index()].buffer,
            input
                .byte_offset
                .evaluate(&values)
                .unwrap()
                .try_into()
                .expect("fixture offset must fit the device byte address domain"),
            &[1i32, 2, 3, 4]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut submission = opened.executor.begin_submission().unwrap();
    let issued = execute_variant(
        variant,
        &mut submission,
        &opened.service,
        &mut FixedTestResources(&buffers),
        &mut values,
    );
    let mut execution = submission.submit();
    execution.complete().unwrap();
    issued.unwrap();
    let output = variant.bindings().arguments[1].as_ref().unwrap();
    let mut bytes = [0; 16];
    opened
        .service
        .read(
            &buffers[output.allocation_index()].buffer,
            output
                .byte_offset
                .evaluate(&values)
                .unwrap()
                .try_into()
                .expect("fixture offset must fit the device byte address domain"),
            &mut bytes,
        )
        .unwrap();
    assert_eq!(
        bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>(),
        vec![2, 3, 4, 5]
    );
}
