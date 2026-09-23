//! Host-only composition fixture: the production builders, traversal, Metal
//! transfer and emitter all consume the same kernel, without opening a device.
use crate::services::{CONTROL, SERVICE_SUBMISSION};
const INTEGER: ServiceClassId = ServiceClassId::new("metal.integer");
use crate::{Metal, MetalFacts};
use seismic_estimator::*;
use seismic_ir::{
    construction::Construction,
    kernel::{
        ops::{BinaryOp, ClosedOpView, ConstantValue, ValueType},
        Kernel,
    },
    schedule::{Launch, LaunchParticipation},
    storage::LaunchLocalLayout,
    target::*,
};
use seismic_lang::{
    expr::{Assignment, ExprArena},
    types::DType,
};

fn facts() -> MetalFacts {
    MetalFacts {
        device_name: "fixture".into(),
        architecture: "fixture".into(),
        operating_system: "fixture".into(),
        registry_id: 0,
        unified_memory: true,
        families: Default::default(),
        language: crate::facts::LanguageVersion::V3_0,
        max_threads_per_threadgroup: [1; 3],
        max_threadgroup_bytes: 1024,
        max_buffer_bytes: 1024,
        buffer_alignment: 16,
        scalar_collective_dtypes: Default::default(),
        bfloat_arithmetic: false,
        matrix_dtypes: Default::default(),
        matrix_combinations: Default::default(),
        argument_table_entries: 31,
        reserved_argument_entries: 4,
        backend_revision: "fixture",
    }
}
fn emission(kernel: &Kernel<Metal>) -> KernelEmissionLayout {
    KernelEmissionLayout {
        words: KernelWordLayout::for_kernel(kernel),
        bindings: vec![],
        locals: vec![],
        addressable_resources: vec![],
        scalar_args: vec![],
        result_types: vec![],
    }
}

#[test]
fn repeat_carry_permutations_read_the_old_state() {
    use seismic_lang::registry::IntrinsicUniformity;
    use std::collections::HashMap;

    for count in [2, 3, 40] {
        let facts = facts();
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Metal>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut builder = construction.portable_kernel(&mut arena, &facts, &[], &vectors);
        let start = builder.index_constant(0);
        let end = builder.index_constant(1);
        let initial = (1..=count)
            .map(|i| builder.constant(ConstantValue::U32(i as u32), ValueType::Scalar(DType::U32)))
            .collect();
        builder.repeat(
            start,
            end,
            initial,
            &vec![IntrinsicUniformity::Workgroup; count],
            |_, _, mut values| {
                values.rotate_left(1);
                values
            },
        );
        builder.close();
        let kernel = &construction.kernels()[0];
        let source = crate::render::render(0, kernel, &emission(kernel)).source;
        let (prefix, body) = source.split_once("for (").unwrap();
        let mut state = HashMap::<String, u32>::new();
        let mut initialized = Vec::new();
        for line in prefix.lines() {
            let Some((left, right)) = line.trim().split_once(" = ") else {
                continue;
            };
            let destination = left.split_whitespace().last().unwrap();
            let right = right.trim_end_matches(';');
            if let Some(value) = state.get(right).copied().or_else(|| {
                right
                    .strip_prefix("uint(")?
                    .strip_suffix("u)")?
                    .parse()
                    .ok()
            }) {
                state.insert(destination.to_string(), value);
                initialized.push(destination.to_string());
            }
        }
        let carries = &initialized[initialized.len() - count..];
        for line in body
            .split_once('{')
            .unwrap()
            .1
            .split('}')
            .next()
            .unwrap()
            .lines()
        {
            let Some((left, right)) = line.trim().split_once(" = ") else {
                continue;
            };
            let destination = left.split_whitespace().last().unwrap();
            let value = state[right.trim_end_matches(';')];
            state.insert(destination.to_string(), value);
        }
        for (i, carry) in carries.iter().enumerate() {
            assert_eq!(state[carry], ((i + 1) % count + 1) as u32, "{source}");
        }
    }
}

struct Model {
    definitions: Vec<ServiceDefinition>,
}
impl ServiceModel for Model {
    fn service(&self, class: ServiceClassId) -> &ServiceDefinition {
        self.definitions.iter().find(|d| d.class == class).unwrap()
    }
    fn maximum_relative_error_basis_points(&self) -> u16 {
        0
    }
}
impl ExecutionModel<Metal> for Model {
    fn emission_layout(&self, kernel: &Kernel<Metal>) -> KernelEmissionLayout {
        emission(kernel)
    }
    fn operation_cost(
        &self,
        arena: &mut ExprArena,
        kernel: &Kernel<Metal>,
        _: &KernelEmissionLayout,
        _: &Launch<Metal>,
        _: &LaunchLocalLayout,
        op: ClosedOpView<'_, Metal>,
    ) -> Result<OperationCost, ModelLimitation> {
        seismic_estimator_metal::operation_cost(arena, kernel, op)
            .map(|cost| cost.map_services(|service| ServiceClassId::new(service.stable_name())))
    }
}
#[test]
fn same_closed_kernel_reaches_estimation_and_metal_emission_without_a_device() {
    let model = Model {
        definitions: [SERVICE_SUBMISSION, CONTROL, INTEGER]
            .into_iter()
            .map(|class| ServiceDefinition {
                class,
                correlation: ServiceCorrelationId::new("fixture"),
                qualification: ServiceQualificationDomain {
                    minimum_units: 1,
                    maximum_units: 1,
                    maximum_concurrent_uses: 1,
                },
                accuracy: ServiceAccuracyClass::Compute,
                topology: ResourceTopology {
                    resources: 1,
                    max_concurrency: 1,
                },
                dependency_latency: DurationInterval::new(7, 7, 1),
                saturated_capacity: ServiceCurve {
                    setup: DurationInterval::new(0, 0, 1),
                    regimes: vec![],
                },
                provenance: FactProvenance::Derived {
                    rule: "synthetic integration coefficient",
                    inputs: Box::new([]),
                },
            })
            .collect(),
    };
    let mut arena = ExprArena::default();
    let mut c = Construction::<Metal>::new(&mut arena, vec![], false, 0);
    let facts = facts();
    let vectors = VectorSupport::default();
    let mut builder = c.portable_kernel(&mut arena, &facts, &[], &vectors);
    let a = builder.constant(ConstantValue::F32(2.0), ValueType::Scalar(DType::F32));
    let b = builder.constant(ConstantValue::F32(3.0), ValueType::Scalar(DType::F32));
    builder.binary(BinaryOp::Add, a, b);
    let id = builder.close();
    let one = arena.nat(1);
    let empty = arena.bool(false);
    let mut builder = c.schedule(&mut arena, 0);
    let launch = builder.launch(Launch {
        kernel: id,
        descriptor: crate::MetalLaunchMode,
        grid: [one; 3],
        workgroup: [one; 3],
        empty,
        parallel_extent: None,
        logical_base: None,
    });
    builder.step_launch(launch);
    let token = builder.close();
    let analyzed = c
        .close(token)
        .normalize_launches(&mut arena, u64::MAX, 64)
        .unwrap()
        .analyze_allocations();
    let planned = analyzed.apply_allocation_plan(
        &mut arena,
        seismic_ir::construction::AllocationPlan::distinct(),
    );
    let executable = planned.finish().close_execution(
        &mut arena,
        LocalRealizationPolicy {
            workgroup: LocalRealization::NativeDynamic,
            participant: LocalRealization::NativeStatic,
            register: LocalRealization::NativeStatic,
        },
        &crate::profile::MetalKernelAbi,
    );
    let prediction = estimate(&model, &mut arena, executable.view()).unwrap();
    let duration = arena
        .eval_duration(prediction.estimate(), &Assignment::new())
        .unwrap();
    // Source addition now contributes its actual word recipe; the old
    // three-scalar-op fixture cost would hide that constructed work.
    assert!(duration.upper().numerator() > 28 * u128::from(duration.upper().denominator()));
    let kernel = executable.kernels().kernel(id);
    let rendered = crate::render::render(0, kernel, &emission(kernel));
    assert!(rendered.source.contains("as_type<float>("));
    assert!(prediction
        .contributions()
        .iter()
        .any(|term| term.class() == INTEGER));
}

#[test]
fn checked_source_refines_without_a_profile_or_native_context() {
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "refinement-fixture.seismic".into(),
        text: "fn inner[N](x: &tensor[N] f32, factor: f32, output: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        output[i] = x[i] * factor\n\nfn scale[N](x: &tensor[N] f32, factor: f32, output: &mut tensor[N] f32):\n    inner(x, factor, output)\n".into(),
    }])).unwrap();
    let entry = module
        .entry(
            module.entry_named("scale").unwrap(),
            &seismic_lang::entry::ElementBindings::new(),
        )
        .unwrap();
    use seismic_compiler::candidate_domain::{
        construct_candidate_domain, ConstructionAllowance, ConstructionCoordinate, Materialization,
    };
    let target = crate::profile::device_description_from_facts(facts()).unwrap();
    let budget = seismic_compiler::preparation_budget::PreparationBudget::default();
    let precision = Default::default();
    let mut domain = construct_candidate_domain(
        entry,
        &target,
        crate::profile::registry(),
        &precision,
        &budget,
    )
    .unwrap();
    let mut paths = domain
        .root_selections()
        .into_iter()
        .map(ConstructionCoordinate::root)
        .collect::<Vec<_>>();
    let mut completed = 0;
    while let Some(path) = paths.pop() {
        match domain
            .advance(
                &path,
                ConstructionAllowance {
                    work_units: 100_000,
                    wall_time: std::time::Duration::from_secs(30),
                },
            )
            .state
        {
            Materialization::Choice(choice) => paths.extend(
                choice
                    .alternatives
                    .iter()
                    .map(|body| path.select(&choice, *body)),
            ),
            Materialization::Ready(identity) => {
                let read = domain.read_materialized(&identity).unwrap();
                assert!(read.candidate().kernels().kernels().next().is_some());
                assert!(!read.candidate().schedule().launches().is_empty());
                completed += 1;
            }
            other => panic!("fixture construction did not finish: {other:?}"),
        }
    }
    assert!(completed > 1);
}
