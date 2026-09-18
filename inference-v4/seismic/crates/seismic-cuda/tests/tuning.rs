use seismic_accounting::{schedule, selection, workload};
use seismic_compiler::tuner::{self, Backend as _, Preparation};
use seismic_cuda::{execution::Execution, model, tuning};
use seismic_lang::{
    Scope,
    lowered_ir::LoweredIr,
    program::{SourceFile, compile},
};
use std::{cell::RefCell, collections::HashMap};

const COPY: &str = "fn kernel(x: tensor[2] f32, out: tensor[2] f32):\n  for i in parallel:\n    y = tile[1] f32\n    for j in owned(y): y[j] = x[i] + 1.0\n    store(y,out[i:i+1])\n";
const TWO_PHASES: &str = "fn kernel(x: tensor[1] f32, middle: tensor[1] f32, out: tensor[1] f32):\n  for i in parallel:\n    y = tile[1] f32\n    for j in owned(y): y[j] = x[i] + 1.0\n    store(y,middle[i:i+1])\n  for i in parallel:\n    y = tile[1] f32\n    for j in owned(y): y[j] = middle[i] * 2.0\n    store(y,out[i:i+1])\n";
const LIMITS: workload::DerivationLimits = workload::DerivationLimits {
    instructions: 100_000,
    operations: 100_000,
};
const BUDGET: selection::Budget = selection::Budget {
    nodes: 100,
    schedule_assignments: 100_000,
};

fn lowered(source: &str) -> LoweredIr {
    let program = compile(
        &[SourceFile {
            path: "tuning.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap();
    seismic_lang::lower::lower(&program, "kernel", "cuda", &HashMap::new()).unwrap()
}
fn device() -> seismic_cuda::DeviceInfo {
    // Deliberately synthetic geometry keeps this complete finite-domain oracle small.
    seismic_cuda::DeviceInfo {
        name: "synthetic CUDA geometry".into(),
        compute_capability: (8, 0),
        driver_version: 0,
        max_threads_per_block: 2,
        max_grid_x: 8,
        warp_size: 2,
        multiprocessors: 1,
        global_memory_bytes: 4096,
        l2_cache_bytes: 0,
        max_threads_per_multiprocessor: 4,
        registers_32bit_per_multiprocessor: 1024,
        shared_bytes_per_multiprocessor: 1024,
    }
}
fn executions(
    function: &LoweredIr,
    device: &seismic_cuda::DeviceInfo,
) -> Vec<(Vec<usize>, Vec<Execution>)> {
    fn visit(
        function: &LoweredIr,
        device: &seismic_cuda::DeviceInfo,
        path: &mut Vec<usize>,
        out: &mut Vec<(Vec<usize>, Vec<Execution>)>,
    ) {
        match tuning::prepare(function, device, path).unwrap() {
            Preparation::Choice { alternatives, .. } => {
                for i in 0..alternatives.len() {
                    path.push(i);
                    visit(function, device, path, out);
                    path.pop();
                }
            }
            Preparation::Execution(execution) => out.push((path.clone(), execution)),
            Preparation::Infeasible(_) => {}
        }
    }
    let mut out = Vec::new();
    visit(function, device, &mut Vec::new(), &mut out);
    out
}
fn profile(executions: &[Execution], device: &seismic_cuda::DeviceInfo) -> model::CudaHardware {
    let mut primitives = Vec::new();
    for execution in executions {
        for requirement in model::requirements(execution) {
            if let model::Requirement::Instruction(primitive) = requirement {
                if !primitives.contains(&primitive) {
                    primitives.push(primitive);
                }
            }
        }
    }
    model::CudaHardware {
        identity: "synthetic one-tick PTX services".into(),
        scope: model::Scope::HypotheticalInstructionPreservingPtxV1,
        timebase: schedule::Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1_000_000_000,
        },
        execution_units: device.multiprocessors as usize,
        warp_width: device.warp_size,
        cohorts: model::CohortPolicy::LowestPosition,
        internal_alignment: 256,
        resources: vec![model::Resource {
            name: "issue".into(),
            scope: model::ResourceScope::Device,
            capacity: 1,
            unit: schedule::CapacityUnit::Slots,
        }],
        timings: primitives
            .into_iter()
            .map(|primitive| model::PrimitiveTiming {
                primitive,
                latency: model::Ticks::Fixed(1),
                reservations: vec![model::Reservation {
                    resource: 0,
                    offset: 0,
                    duration: model::Ticks::Fixed(1),
                    units: model::Amount::fixed(1),
                }],
            })
            .collect(),
        block_residency: vec![],
        per_unit_residency: vec![model::UnitResidency {
            name: "blocks".into(),
            capacity: 2,
            units_per_block: model::Amount::fixed(1),
        }],
    }
}
fn workload(execution: &Execution) -> workload::ScalarWorkload {
    workload::ScalarWorkload {
        identity: "fixed independent bindings".into(),
        allocations: execution
            .program()
            .buffers
            .iter()
            .enumerate()
            .map(|(i, b)| workload::Allocation {
                id: i as u64,
                bytes: b.bytes as u64,
                alignment: 256,
                known_bytes: Default::default(),
            })
            .collect(),
        buffers: execution
            .program()
            .buffers
            .iter()
            .enumerate()
            .map(|(i, b)| workload::BufferBinding {
                allocation: i as u64,
                offset: 0,
                bytes: b.bytes as u64,
            })
            .collect(),
        scalars: vec![],
    }
}

fn select() -> tuner::TunedIr<Vec<Execution>, tuning::Conditions> {
    let function = lowered(COPY);
    let device = device();
    let executions = executions(&function, &device);
    assert_eq!(executions.len(), 4); // two dispatches, all two legal block sizes
    let all = executions
        .iter()
        .flat_map(|(_, e)| e.clone())
        .collect::<Vec<_>>();
    let hardware = profile(&all, &device);
    let backend = tuning::Backend::new(&device, &hardware).unwrap();
    let workload = workload(&all[0]);
    let expected = executions
        .iter()
        .map(|(_, e)| {
            let model = backend.analyze(e, &workload, LIMITS).unwrap();
            let solution = model.solve(100_000).unwrap();
            assert!(solution.is_optimal());
            solution.schedule().completion
        })
        .min()
        .unwrap();
    let request = tuner::Request {
        input: tuner::Input::Lowered(&function),
        backend: &backend,
        workload: &workload,
        derivation_limits: LIMITS,
    };
    let tuner::Outcome::Optimal(selected) = tuner::tune(&request, BUDGET).unwrap() else {
        panic!("complete CUDA choice space did not resolve")
    };
    assert_eq!(selected.modeled_cost().upper(), expected);
    let (_, original) = executions
        .iter()
        .find(|(path, _)| path == selected.selected_path())
        .unwrap();
    for (a, b) in original.iter().zip(selected.execution()) {
        assert_eq!(a.target_plan(), b.target_plan());
        assert_eq!(a.dispatch(), b.dispatch());
    }
    selected
}

#[test]
fn compiler_selects_the_complete_cuda_form_and_retains_target_ir() {
    select();
}

#[test]
fn retained_block_family_refines_complete_phase_domains_without_repreparing_plans() {
    let function = lowered(TWO_PHASES);
    let device = device();
    let leaves = executions(&function, &device);
    assert_eq!(leaves.len(), 6); // one sequential phase or two parallel phases
    let all = leaves
        .iter()
        .flat_map(|(_, e)| e.clone())
        .collect::<Vec<_>>();
    let backend = tuning::Backend::new(&device, &profile(&all, &device)).unwrap();
    let dispatch = 1;
    let Preparation::Choice { alternatives, .. } =
        tuning::prepare(&function, &device, &[dispatch]).unwrap()
    else {
        panic!("expected the retained first-phase block domain")
    };
    let owner = alternatives
        .owner::<selection::IntegerRange<tuning::BlockChoice>>()
        .unwrap();
    assert_eq!((owner.first(), owner.last()), (1, 2));
    assert_eq!(owner.decision.family().phase_count(), 2);
    assert!(owner.decision.family().selected_blocks().is_empty());
    let mut refined = Vec::<Vec<Execution>>::new();
    for first in 0..alternatives.len() {
        let Some(Preparation::Choice {
            alternatives: second,
            ..
        }) = backend.refine(&alternatives, first).unwrap()
        else {
            panic!("expected the retained second-phase block domain")
        };
        let next = second
            .owner::<selection::IntegerRange<tuning::BlockChoice>>()
            .unwrap();
        assert_eq!(next.decision.phase, 1);
        assert_eq!((next.first(), next.last()), (1, 2));
        assert_eq!(
            next.decision.family().selected_blocks(),
            &[first as u32 + 1]
        );
        for phase in 0..2 {
            assert!(std::ptr::eq(
                owner.decision.family().target(phase).unwrap(),
                next.decision.family().target(phase).unwrap()
            ));
        }
        for index in 0..second.len() {
            let Some(Preparation::Execution(execution)) = backend.refine(&second, index).unwrap()
            else {
                panic!("all phase dimensions are resolved")
            };
            let (_, expected) = leaves
                .iter()
                .find(|(path, _)| path == &[dispatch, first, index])
                .unwrap();
            for (phase, (actual, expected)) in execution.iter().zip(expected).enumerate() {
                assert_eq!(actual.target_plan(), expected.target_plan());
                assert_eq!(actual.dispatch(), expected.dispatch());
                assert_eq!(actual.storage(), expected.storage());
                assert_eq!(actual.program().conditions, expected.program().conditions);
                assert!(std::ptr::eq(
                    owner.decision.family().target(phase).unwrap(),
                    actual.target_plan()
                ));
                if let Some(sibling) = refined.first() {
                    assert!(std::ptr::eq(actual.program(), sibling[phase].program()));
                }
            }
            refined.push(execution);
        }
    }
    assert_eq!(refined.len(), 4);
    assert!(backend.refine(&alternatives, alternatives.len()).is_err());
    let shortened =
        selection::Domain::new(selection::IntegerRange::new(owner.decision.clone(), 1, 1).unwrap())
            .unwrap();
    assert!(backend.refine(&shortened, 0).is_err());
    let mut wrong_phase = owner.clone();
    wrong_phase.decision.phase = 1;
    assert!(
        backend
            .refine(&selection::Domain::new(wrong_phase).unwrap(), 0)
            .is_err()
    );
}

/// Fixes parallel dispatch, then fails if search reconstructs the captured
/// two-phase family through the source path instead of refining its owner.
struct RetainedBackend {
    inner: tuning::Backend,
    prepared: RefCell<Vec<Vec<usize>>>,
    families: RefCell<Vec<selection::Domain>>,
    refined: RefCell<usize>,
}
impl tuner::Backend for RetainedBackend {
    type Execution = Vec<Execution>;
    type Conditions = tuning::Conditions;
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn conditions(&self) -> Self::Conditions {
        self.inner.conditions()
    }
    fn description(&self) -> tuner::Description {
        self.inner.description()
    }
    fn prepare(
        &self,
        function: &LoweredIr,
        path: &[usize],
    ) -> Result<Preparation<Self::Execution>, String> {
        assert!(
            path.is_empty(),
            "repeated source/scalar/PTX preparation for {path:?}"
        );
        self.prepared.borrow_mut().push(path.to_vec());
        let result = self.inner.prepare(function, &[1])?;
        if let Preparation::Choice { alternatives, .. } = &result {
            if alternatives
                .owner::<selection::IntegerRange<tuning::BlockChoice>>()
                .is_some()
            {
                self.families.borrow_mut().push(alternatives.clone());
            }
        }
        Ok(result)
    }
    fn refine(
        &self,
        alternatives: &selection::Domain,
        index: usize,
    ) -> Result<Option<Preparation<Self::Execution>>, String> {
        let result = self.inner.refine(alternatives, index)?;
        if result.is_some() {
            *self.refined.borrow_mut() += 1;
        }
        Ok(result)
    }
    fn relax(
        &self,
        alternatives: &selection::Domain,
        indices: std::ops::Range<usize>,
        workload: &workload::ScalarWorkload,
    ) -> Result<Option<schedule::Demand>, String> {
        self.inner.relax(alternatives, indices, workload)
    }
    fn analyze(
        &self,
        execution: &Self::Execution,
        workload: &workload::ScalarWorkload,
        limits: workload::DerivationLimits,
    ) -> Result<schedule::Model, workload::DerivationError> {
        let mut model = self.inner.analyze(execution, workload, limits)?;
        // A synthetic fully blocking warp: each instruction completes before
        // the next issues. Keep the derived operations/resources intact while
        // making this launch-choice oracle independent of schedule branching.
        for operation in &mut model.operations {
            for &predecessor in &operation.start_predecessors {
                if !operation.predecessors.contains(&predecessor) {
                    operation.predecessors.push(predecessor);
                }
            }
        }
        model.identity.push_str(":blocking-warp-test");
        Ok(model)
    }
    fn materialize(
        &self,
        execution: &Self::Execution,
        objective: &selection::Objective,
    ) -> Result<Self::Execution, String> {
        self.inner.materialize(execution, objective)
    }
    fn check_materialization(
        &self,
        source: &Self::Execution,
        selected: &Self::Execution,
        objective: &selection::Objective,
    ) -> Result<(), String> {
        for (source, selected) in source.iter().zip(selected) {
            assert!(std::ptr::eq(source.program(), selected.program()));
            assert!(std::ptr::eq(source.target_plan(), selected.target_plan()));
        }
        self.inner
            .check_materialization(source, selected, objective)
    }
}

#[test]
fn interrupted_two_phase_search_reuses_retained_families_and_finds_exact_optimum() {
    let function = lowered(TWO_PHASES);
    let device = device();
    let leaves = executions(&function, &device)
        .into_iter()
        .filter(|(path, _)| path[0] == 1)
        .map(|(path, execution)| (path[1..].to_vec(), execution))
        .collect::<Vec<_>>();
    assert_eq!(leaves.len(), 4);
    let all = leaves
        .iter()
        .flat_map(|(_, e)| e.clone())
        .collect::<Vec<_>>();
    let mut hardware = profile(&all, &device);
    hardware.per_unit_residency[0].capacity = 1;
    let backend = RetainedBackend {
        inner: tuning::Backend::new(&device, &hardware).unwrap(),
        prepared: RefCell::default(),
        families: RefCell::default(),
        refined: RefCell::default(),
    };
    let invocation = workload(&all[0]);
    let expected = leaves
        .iter()
        .map(|(_, execution)| {
            let solution = backend
                .analyze(execution, &invocation, LIMITS)
                .unwrap()
                .solve(100_000)
                .unwrap();
            assert!(solution.is_optimal());
            solution.schedule().completion
        })
        .min()
        .unwrap();
    let request = tuner::Request {
        input: tuner::Input::Lowered(&function),
        backend: &backend,
        workload: &invocation,
        derivation_limits: LIMITS,
    };
    let tuner::Outcome::Incomplete(progress) = tuner::tune(
        &request,
        selection::Budget {
            nodes: 1,
            schedule_assignments: 0,
        },
    )
    .unwrap() else {
        panic!("expected pending block regions")
    };
    assert!(!progress.frontier().is_empty());
    assert_eq!(backend.families.borrow().len(), 1);
    let tuner::Outcome::Incomplete(progress) = tuner::resume(
        &request,
        progress,
        selection::Budget {
            nodes: 1,
            schedule_assignments: 0,
        },
    )
    .unwrap() else {
        panic!("expected the pending second-phase domain")
    };
    assert!(!progress.frontier().is_empty());
    assert!(progress.feasible_upper().is_none());
    assert_eq!(backend.families.borrow().len(), 1);
    assert_eq!(backend.prepared.borrow().len(), 1);
    let refinements = *backend.refined.borrow();
    let tuner::Outcome::Optimal(selected) = tuner::resume(
        &request,
        progress,
        selection::Budget {
            nodes: 100,
            schedule_assignments: 100_000,
        },
    )
    .unwrap() else {
        panic!("complete CUDA family did not resolve")
    };
    assert_eq!(backend.prepared.borrow().len(), 1);
    assert!(
        *backend.refined.borrow() > refinements,
        "resumed search must continue refining the retained owner"
    );
    assert_eq!(selected.modeled_cost().upper(), expected);
    let (_, expected) = leaves
        .iter()
        .find(|(path, _)| path == selected.selected_path())
        .unwrap();
    for (phase, (actual, expected)) in selected.execution().iter().zip(expected).enumerate() {
        assert_eq!(actual.target_plan(), expected.target_plan());
        assert_eq!(actual.dispatch(), expected.dispatch());
        assert!(backend.families.borrow().iter().any(|domain| {
            domain
                .owner::<selection::IntegerRange<tuning::BlockChoice>>()
                .unwrap()
                .decision
                .family()
                .target(phase)
                .is_some_and(|target| std::ptr::eq(target, actual.target_plan()))
        }));
    }
}

#[test]
fn every_block_interval_relaxes_all_of_its_members() {
    let function = lowered(COPY);
    let mut device = device();
    device.max_threads_per_block = 5;
    let leaves = executions(&function, &device);
    let all = leaves
        .iter()
        .flat_map(|(_, e)| e.clone())
        .collect::<Vec<_>>();
    let hardware = profile(&all, &device);
    let backend = tuning::Backend::new(&device, &hardware).unwrap();
    let invocation = workload(&all[0]);
    let Preparation::Choice { alternatives, .. } =
        tuning::prepare(&function, &device, &[1]).unwrap()
    else {
        panic!()
    };
    let owner = alternatives
        .owner::<selection::IntegerRange<tuning::BlockChoice>>()
        .unwrap();
    assert_eq!(owner.decision.family().phase_count(), 1);
    for start in 0..alternatives.len() {
        for end in start + 1..=alternatives.len() {
            let lower = backend
                .relax(&alternatives, start..end, &invocation)
                .unwrap()
                .unwrap()
                .lower_bound()
                .unwrap();
            assert!(lower > 0);
            for index in start..end {
                let (_, selected) = leaves.iter().find(|(path, _)| path == &[1, index]).unwrap();
                assert_eq!(
                    owner.decision.family().target(0).unwrap(),
                    selected[0].target_plan()
                );
                let solution = backend
                    .analyze(selected, &invocation, LIMITS)
                    .unwrap()
                    .solve(100_000)
                    .unwrap();
                assert!(
                    lower <= solution.schedule().completion,
                    "region {start}..{end} overstates member {index}"
                );
            }
        }
    }
    let function = lowered(&COPY.replace("tensor[2]", "tensor[0]"));
    let Preparation::Choice { alternatives, .. } =
        tuning::prepare(&function, &device, &[1]).unwrap()
    else {
        panic!()
    };
    assert_eq!(
        backend
            .relax(&alternatives, 0..alternatives.len(), &invocation)
            .unwrap()
            .unwrap()
            .lower_bound()
            .unwrap(),
        0
    );
}

#[test]
fn unsupported_service_aborts_analysis_and_cannot_exclude_a_branch() {
    let function = lowered(COPY);
    let device = device();
    let executions = executions(&function, &device);
    let all = executions
        .iter()
        .flat_map(|(_, e)| e.clone())
        .collect::<Vec<_>>();
    let mut hardware = profile(&all, &device);
    hardware.timings.clear();
    let backend = tuning::Backend::new(&device, &hardware).unwrap();
    let workload = workload(&all[0]);
    let request = tuner::Request {
        input: tuner::Input::Lowered(&function),
        backend: &backend,
        workload: &workload,
        derivation_limits: LIMITS,
    };
    let error = match tuner::tune(&request, BUDGET) {
        Err(e) => e,
        _ => panic!("missing service became a search result"),
    };
    assert!(error.contains("missing CUDA hardware timing"), "{error}");
}

#[test]
fn phases_propagate_writes_through_aliased_parameter_names() {
    let source = "fn kernel(x: tensor[1] i32, flag: tensor[1] i32, alias: tensor[1] i32, out: tensor[1] i32):\n  for i in parallel:\n    y = tile[1] i32\n    for j in owned(y): y[j] = 1\n    store(y,flag[i:i+1])\n  for i in parallel:\n    y = tile[1] i32\n    for j in owned(y):\n      if alias[i] == 0: y[j] = 0\n      else: y[j] = x[i] + x[i] * x[i]\n    store(y,out[i:i+1])\n";
    let function = lowered(source);
    let device = device();
    let Preparation::Execution(phases) = tuning::prepare(&function, &device, &[1, 0, 0]).unwrap()
    else {
        panic!("expected two resolved phases")
    };
    assert_eq!(phases.len(), 2);
    let hardware = profile(&phases, &device);
    let mut invocation = workload(&phases[0]);
    let flag = phases[0]
        .program()
        .buffers
        .iter()
        .position(|b| b.parameter == "flag")
        .unwrap();
    let alias = phases[0]
        .program()
        .buffers
        .iter()
        .position(|b| b.parameter == "alias")
        .unwrap();
    invocation.buffers[alias] = invocation.buffers[flag].clone();
    let flag_id = invocation.buffers[flag].allocation;
    let set_flag = |invocation: &mut workload::ScalarWorkload, value: i32| {
        invocation
            .allocations
            .iter_mut()
            .find(|a| a.id == flag_id)
            .unwrap()
            .known_bytes = value
            .to_le_bytes()
            .into_iter()
            .enumerate()
            .map(|(i, b)| (i as u64, b))
            .collect();
    };
    set_flag(&mut invocation, 0);
    let initial_zero = model::derive_sequence(&phases, &hardware, &invocation, LIMITS).unwrap();
    set_flag(&mut invocation, 1);
    let initial_one = model::derive_sequence(&phases, &hardware, &invocation, LIMITS).unwrap();
    assert_eq!(
        initial_zero, initial_one,
        "phase one overwrites initial flag through its alias"
    );
    // An unknown store must invalidate initial knowledge, not silently retain it.
    let unknown = lowered(&source.replace("y[j] = 1\n", "y[j] = x[i]\n"));
    let Preparation::Execution(phases) = tuning::prepare(&unknown, &device, &[1, 0, 0]).unwrap()
    else {
        panic!()
    };
    let hardware = profile(&phases, &device);
    let error = model::derive_sequence(&phases, &hardware, &invocation, LIMITS).unwrap_err();
    assert!(
        error.to_string().contains("known integer or predicate"),
        "{error}"
    );
    let phase_two = initial_one
        .operations
        .iter()
        .position(|o| o.name.starts_with("phase1:"))
        .unwrap();
    assert_eq!(
        initial_one.operations[phase_two].predecessors,
        vec![phase_two - 1]
    );
    assert!(
        initial_one
            .lifetimes
            .iter()
            .filter(|l| l.begin.operation < phase_two)
            .all(|l| l.end.operation < phase_two)
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn selected_cuda_execution_runs_without_repreparing_its_ir() {
    let selected = select();
    let (execution, _artifact) = selected.into_parts();
    let emitted = execution
        .iter()
        .map(|e| seismic_cuda::ptx::print(e.target_plan()))
        .collect::<Vec<_>>();
    let device = seismic_cuda::Device::open(0).unwrap();
    let mut kernel = device.compile_executions(execution).unwrap();
    assert_eq!(
        kernel.ptx_sources().collect::<Vec<_>>(),
        emitted.iter().map(String::as_str).collect::<Vec<_>>()
    );
    let input = [1.0f32, 2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let mut output = vec![0; 8];
    let x = device.buffer_from(&input).unwrap();
    let out = device.buffer(8).unwrap();
    kernel.execute(&[x, out.clone()], &[], false).unwrap();
    out.read(&mut output).unwrap();
    let actual = output
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual, vec![2.0, 3.0]);
}
