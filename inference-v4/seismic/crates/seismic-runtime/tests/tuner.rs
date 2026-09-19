//! The fixture machine is intentionally hypothetical. These tests verify the
//! source -> choices -> derived model -> checked selection -> native artifact
//! connection and applicability checks; they do not qualify physical machine timing.
use seismic_accounting::{
    execution_model::*, schedule::*, workload::DerivationLimits,
};
use seismic_lang::{
    Scope as LanguageScope,
    program::{SourceFile, compile},
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{
    Device,
    execution::Execution,
    tuner::{self, Form, Hardware, ImplementationConditions, Input, Outcome, Request},
};
use std::collections::HashMap;

fn source_input(program: &seismic_lang::program::Program) -> Input<'_> {
    static SHAPES: std::sync::LazyLock<HashMap<String, i64>> = std::sync::LazyLock::new(HashMap::new);
    static ELEMENTS: std::sync::LazyLock<HashMap<String, seismic_lang::types::Elem>> = std::sync::LazyLock::new(HashMap::new);
    static OPTIONS: std::sync::LazyLock<seismic_lang::lower::Options> = std::sync::LazyLock::new(Default::default);
    Input::Portable { program, entry: &program.functions[0].name, shapes: &SHAPES, elements: &ELEMENTS, options: &OPTIONS }
}

use automatic_hardware::{cpu as contract, cuda as cuda_hardware};
#[path = "support/automatic_hardware.rs"]
mod automatic_hardware;

fn settings() -> tuner::Settings {
    seismic_runtime::tuner::Settings { limits: seismic_runtime::tuner::Limits { work: 100_000, ..Default::default() }, ..Default::default() }
}

fn floating_literal_resume_identity(initial: u64, changed: u64) {
    use seismic_lang::ir::{ExprKind, StmtKind};
    let mut program = compile(&[SourceFile {
        path: "literal-identity.seismic.portable".into(),
        scope: LanguageScope::Portable,
        text: "fn write(out:tensor[1] f32):\n  value = 0.0\n  y = tile[1] f32\n  for i in owned(y):\n    y[i] = value\n  store(y,out)\n".into(),
    }], &[]).unwrap();
    let set_literal = |program: &mut seismic_lang::program::Program, bits| {
        let StmtKind::Assign { value, .. } = &mut program.functions[0].body[0].kind else {
            panic!("fixture literal assignment");
        };
        let ExprKind::Float(number) = &mut value.kind else {
            panic!("fixture floating literal");
        };
        *number = f64::from_bits(bits);
    };
    let shapes = HashMap::new();
    let elements = HashMap::new();
    let options = seismic_lang::lower::Options::default();
    let baseline = seismic_lang::lower::lower(&program, "write", "cpu", &shapes).unwrap();
    let hardware = Hardware::Cpu(contract(&baseline));
    // Keep every other source fact, including spans, identical. These are typed
    // IR literals: identity must compare their representation, not IEEE equality.
    set_literal(&mut program, initial);
    let mut altered = program.clone();
    set_literal(&mut altered, changed);
    let device = Device::cpu();
    let facts = device.facts();
    let workload =
        tuner::workload("literal identity", &[device.buffer(4).unwrap()], &[], &[]).unwrap();
    let portable = |program| Input::Portable {
        program,
        entry: "write",
        shapes: &shapes,
        elements: &elements,
        options: &options,
    };
    for (input, changed_input) in [
        (portable(&program), portable(&altered)),
    ] {
        let mut request = Request {
            input,
            device: &facts,
            form: Form::CpuScalar,
            hardware: &hardware,
            workload: &workload,
            derivation_limits: DerivationLimits {
                instructions: 100,
                operations: 100,
            },
        };
        let limited = seismic_runtime::tuner::Settings { limits: seismic_runtime::tuner::Limits { work: 0, ..Default::default() }, ..Default::default() };
        let Outcome::Incomplete(progress) = tuner::tune(&request, limited.clone()).unwrap() else {
            panic!("zero-node budget retains source without selecting an execution");
        };
        let Outcome::Incomplete(progress) = tuner::resume(&request, progress, limited.limits.clone()).unwrap()
        else {
            panic!("identical floating literal must permit resumption");
        };
        request.input = changed_input;
        assert!(
            matches!(tuner::resume(&request, progress, limited.limits.clone()), Err(error) if error.contains("inputs changed")),
            "changed literal bits must invalidate retained choices"
        );
    }
}

#[test]
fn source_identity_distinguishes_signed_zero_literals() {
    floating_literal_resume_identity(0.0_f64.to_bits(), (-0.0_f64).to_bits());
}

#[test]
fn source_identity_retains_identical_nan_and_distinguishes_payloads() {
    floating_literal_resume_identity(0x7ff8_0000_0000_0001, 0x7ff8_0000_0000_0002);
}

#[test]
fn construction_limits_require_a_new_export_while_search_limits_resume() {
    let program = compile(&[SourceFile {
        path: "construction.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn write(out:tensor[1] f32):\n  out[0] = 2.0\n".into(),
    }], &[]).unwrap();
    let lowered = seismic_lang::lower::lower(&program, "write", "cpu", &HashMap::new()).unwrap();
    let hardware = Hardware::Cpu(contract(&lowered));
    let device = Device::cpu();
    let facts = device.facts();
    let workload = tuner::workload("write", &[device.buffer(4).unwrap()], &[], &[]).unwrap();
    let mut request = Request { input: source_input(&program), device: &facts,
        form: Form::CpuScalar, hardware: &hardware, workload: &workload,
        derivation_limits: DerivationLimits { instructions: 1, operations: 1 } };
    let Outcome::Incomplete(progress) = tuner::tune(&request, settings()).unwrap() else {
        panic!("construction exhaustion cannot establish an optimum or infeasibility");
    };
    assert!(progress.feasible_upper().is_none());
    let model = progress.model().clone();
    let Outcome::Incomplete(progress) = tuner::resume(&request, progress, settings().limits).unwrap() else { panic!(); };
    assert!(std::sync::Arc::ptr_eq(&model, progress.model()));
    request.derivation_limits = DerivationLimits { instructions: 10_000, operations: 10_000 };
    assert!(matches!(tuner::resume(&request, progress, settings().limits), Err(e) if e.contains("construction limits")));
    let Outcome::Optimal(selected) = tuner::tune(&request, settings()).unwrap() else {
        panic!("new construction allowance requires a fresh complete export");
    };
    assert!(selected.modeled_cost().is_exact());
}

#[test]
fn source_to_native_keeps_checked_model_and_enforces_workload_conditions() {
    let program = compile(&[SourceFile { path: "tuner.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn copy(x: tensor[7] f32, out: tensor[7] f32, enabled: bool):\n  if enabled:\n    a = load(x)\n    b = a\n    store(b, out)\n".into() }], &[]).unwrap();
    let shapes = HashMap::new();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cpu", &shapes).unwrap();
    let model = Hardware::Cpu(contract(&lowered));
    let device = Device::cpu();
    let hardware = device.facts();
    let input: Vec<_> = (0..7).flat_map(|n| (n as f32).to_le_bytes()).collect();
    let source = device.buffer_from(&input).unwrap();
    let target = device.buffer(input.len()).unwrap();
    let schema = seismic_cpu::prepare(&lowered, LoadStrategy::Materialize)
        .unwrap()
        .scalars;
    let bindings = vec![source.clone(), target.clone()];
    let workload = tuner::workload(
        "fixed scalars and allocation geometry",
        &bindings,
        &schema,
        &[1.0],
    )
    .unwrap();
    let elements = HashMap::new();
    let options = seismic_lang::lower::Options::default();
    let request = Request {
        input: Input::Portable {
            program: &program,
            entry: "copy",
            shapes: &shapes,
            elements: &elements,
            options: &options,
        },
        device: &hardware,
        form: Form::CpuScalar,
        hardware: &model,
        workload: &workload,
        derivation_limits: DerivationLimits {
            instructions: 10_000,
            operations: 30_000,
        },
    };
    let Outcome::Optimal(tuned) = tuner::tune(&request, settings()).unwrap() else {
        panic!("expected exact conditional optimum")
    };
    let ImplementationConditions::Cpu(conditions) = tuned.conditions().implementation() else {
        panic!("CPU conditions")
    };
    assert_eq!(conditions.hardware.scope, Scope::HypotheticalDirectScalarV1);
    assert!(tuned.modeled_cost().is_exact());
    let Execution::Cpu(selected) = tuned.execution() else {
        panic!("CPU execution")
    };
    assert!(
        selected
            .loads
            .iter()
            .all(|d| d.mode == seismic_lang::ir::LoadMode::Borrow)
    );
    let emitted_order = seismic_realization::scheduling::Order::current(selected);
    assert_eq!(
        emitted_order.blocks,
        static_order::orders(tuned.objective().flat().unwrap().0, tuned.objective().flat().unwrap().1).unwrap()
    );
    let selected_analysis = tuned.objective().flat().unwrap().0.clone();
    let selected_schedule = tuned.objective().flat().unwrap().1.clone();
    let mut kernel = device.compile_tuned(tuned).unwrap();
    let retained = kernel.tuning();
    assert_eq!(retained.objective().flat().unwrap().0, &selected_analysis);
    assert_eq!(retained.objective().flat().unwrap().1, &selected_schedule);
    assert!(retained.modeled_cost().is_exact());
    kernel.execute(&bindings, &[1.0]).unwrap();
    let mut output = vec![0; input.len()];
    target.read(&mut output).unwrap();
    assert_eq!(output, input);
    assert!(
        kernel
            .execute(&bindings, &[0.0])
            .unwrap_err()
            .contains("scalar bindings")
    );
    assert!(
        kernel
            .execute(&[source.clone(), source], &[1.0])
            .unwrap_err()
            .contains("alias relationships")
    );
    let larger = device
        .buffer(input.len() + 4)
        .unwrap()
        .view(4..input.len() + 4)
        .unwrap();
    assert!(
        kernel
            .execute(&[larger, target], &[1.0])
            .unwrap_err()
            .contains("allocation/view")
    );
}
#[test]
fn nested_views_derive_canonical_alias_geometry_and_resume_binds_all_inputs() {
    let program = compile(
        &[SourceFile {
            path: "copy.seismic.portable".into(),
            scope: LanguageScope::Portable,
            text:
                "fn copy(x: tensor[2] f32, out: tensor[2] f32):\n  a = load(x)\n  store(a, out)\n"
                    .into(),
        }],
        &[],
    )
    .unwrap();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cpu", &HashMap::new()).unwrap();
    let model = Hardware::Cpu(contract(&lowered));
    let device = Device::cpu();
    let hardware = device.facts();
    let allocation = device.buffer(32).unwrap();
    let a = allocation.view(4..24).unwrap().view(4..12).unwrap();
    let b = allocation.view(20..28).unwrap();
    assert_eq!(a.allocation_offset(), 8);
    let workload = tuner::workload("views", &[a, b], &[], &[]).unwrap();
    assert_eq!(workload.allocations.len(), 1);
    assert_eq!(
        workload.buffers[0].allocation,
        workload.buffers[1].allocation
    );
    assert_eq!(workload.buffers[0].offset, 8);
    let request = Request {
        input: source_input(&program),
        device: &hardware,
        form: Form::CpuScalar,
        hardware: &model,
        workload: &workload,
        derivation_limits: DerivationLimits {
            instructions: 10_000,
            operations: 30_000,
        },
    };
    let mut limited = settings();
    limited.limits.work = 0;
    let Outcome::Incomplete(progress) = tuner::tune(&request, limited.clone()).unwrap() else {
        panic!()
    };
    assert!(progress.feasible_upper().is_none());
    let retained_model = progress.model().clone();
    assert!(!retained_model.variables.is_empty());
    let Outcome::Optimal(tuned) = tuner::resume(&request, progress, settings().limits).unwrap() else {
        panic!()
    };
    let Outcome::Incomplete(second) = tuner::tune(&request, limited.clone()).unwrap() else {
        panic!()
    };
    let mut changed = model.clone();
    let Hardware::Cpu(hardware) = &mut changed else {
        unreachable!()
    };
    hardware.resources[0].capacity = 2;
    let altered = Request {
        hardware: &changed,
        ..request
    };
    assert!(
        matches!(tuner::resume(&altered,second,settings().limits),Err(error) if error.contains("inputs changed"))
    );
    assert!(tuned.modeled_cost().is_exact());
}

fn gpu_program() -> seismic_lang::program::Program {
    compile(&[SourceFile { path: "gpu-tuner.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn copy(x: tensor[1] f32, out: tensor[1] f32, enabled: bool):\n  if enabled:\n    value = load(x)\n    store(value, out)\n".into() }], &[]).unwrap()
}

#[test]
fn cuda_uses_compiler_selection_and_rejects_a_different_native_device() {
    let program = gpu_program();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cuda", &HashMap::new()).unwrap();
    let facts = seismic_runtime::DeviceFacts::Cuda(seismic_cuda::DeviceInfo {
        name: "one-thread hypothetical device".into(),
        compute_capability: (8, 0),
        driver_version: 0,
        max_threads_per_block: 1,
        max_grid_x: 1,
        warp_size: 32,
        multiprocessors: 1,
        global_memory_bytes: 1024,
        l2_cache_bytes: 0,
        max_threads_per_multiprocessor: 32,
        registers_32bit_per_multiprocessor: 65536,
        shared_bytes_per_multiprocessor: 65536,
    });
    let hardware = cuda_hardware(&lowered, &facts);
    // Only geometry is used by the IR-only compiler: it needs no GPU or native
    // candidate. Actual device compatibility is established at native binding.
    let host = Device::cpu();
    let schema = vec![seismic_lang::abi::ScalarParameter::plain(
        "enabled",
        seismic_lang::types::DType::Bool,
    )];
    let workload = tuner::workload(
        "two separate words",
        &[host.buffer(4).unwrap(), host.buffer(4).unwrap()],
        &schema,
        &[1.0],
    )
    .unwrap();
    let request = Request {
        input: source_input(&program),
        device: &facts,
        form: Form::CudaScalar,
        hardware: &hardware,
        workload: &workload,
        derivation_limits: DerivationLimits {
            instructions: 100_000,
            operations: 100_000,
        },
    };
    let Outcome::Optimal(tuned) = tuner::tune(&request, settings()).unwrap() else {
        panic!("CUDA conditional optimum")
    };
    let Execution::Cuda(phases) = tuned.execution() else {
        panic!("selected CUDA execution")
    };
    assert_eq!(phases.len(), 1);
    assert_eq!(tuned.conditions().device(), &facts);
    assert!(matches!(
        tuned.conditions().implementation(),
        ImplementationConditions::Cuda(_)
    ));
    assert!(tuned.objective().flat().unwrap().0.unmapped.is_empty());
    assert!(tuned.modeled_cost().is_exact());
    assert!(matches!(host.compile_tuned(tuned), Err(e) if e.contains("device differs")));
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_tuned_native_artifact_retains_conditions_and_enforces_bindings() {
    let device = Device::cuda(0).unwrap();
    let facts = device.facts();
    let program = gpu_program();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cuda", &HashMap::new()).unwrap();
    let hardware = cuda_hardware(&lowered, &facts);
    let schema = vec![seismic_lang::abi::ScalarParameter::plain(
        "enabled",
        seismic_lang::types::DType::Bool,
    )];
    let source = device.buffer_from(&7f32.to_le_bytes()).unwrap();
    let target = device.buffer(4).unwrap();
    let bindings = vec![source.clone(), target.clone()];
    let workload = tuner::workload("native CUDA copy", &bindings, &schema, &[1.0]).unwrap();
    let request = Request {
        input: source_input(&program),
        device: &facts,
        form: Form::CudaScalar,
        hardware: &hardware,
        workload: &workload,
        derivation_limits: DerivationLimits {
            instructions: 1_000_000,
            operations: 1_000_000,
        },
    };
    let Outcome::Optimal(tuned) = tuner::tune(
        &request,
        seismic_runtime::tuner::Settings { limits: seismic_runtime::tuner::Limits { work: 100_000, ..Default::default() }, ..Default::default() },
    )
    .unwrap() else {
        panic!("CUDA conditional optimum")
    };
    let selected_model = tuned.objective().flat().unwrap().0.clone();
    let mut kernel = device.compile_tuned(tuned).unwrap();
    assert_eq!(kernel.tuning().objective().flat().unwrap().0, &selected_model);
    kernel.execute(&bindings, &[1.0]).unwrap();
    let mut result = [0; 4];
    target.read(&mut result).unwrap();
    assert_eq!(result, 7f32.to_le_bytes());
    assert!(
        kernel
            .execute(&bindings, &[0.0])
            .unwrap_err()
            .contains("scalar bindings")
    );
    assert!(
        kernel
            .execute(&[source.clone(), source], &[1.0])
            .unwrap_err()
            .contains("alias relationships")
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_tuned_native_artifact_preserves_publication_and_binding_conditions() {
    let program = compile(&[SourceFile { path: "metal-tuner.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn evaluate(out: tensor[2] f32):\n  y = tile[2] f32\n  for i in owned(y): y[i] = 3.0\n  store(y,out)\n".into() }], &[]).unwrap();
    let lowered =
        seismic_lang::lower::lower(&program, "evaluate", "metal", &HashMap::new()).unwrap();
    let device = Device::metal().unwrap();
    let facts = device.facts();
    let hardware = automatic_hardware::metal(&lowered);
    let backing = device.buffer_from(&[0xa5; 16]).unwrap();
    let out = backing.view(4..12).unwrap();
    let workload = tuner::workload("native Metal output", &[out.clone()], &[], &[]).unwrap();
    let request = Request {
        input: source_input(&program),
        device: &facts,
        form: Form::Metal,
        hardware: &hardware,
        workload: &workload,
        derivation_limits: DerivationLimits {
            instructions: 1_000_000,
            operations: 1_000_000,
        },
    };
    let Outcome::Optimal(tuned) = tuner::tune(
        &request,
        seismic_runtime::tuner::Settings { limits: seismic_runtime::tuner::Limits { work: 100_000, ..Default::default() }, ..Default::default() },
    )
    .unwrap() else {
        panic!("Metal conditional optimum")
    };
    assert!(matches!(tuned.execution(), Execution::Metal(_)));
    let selected_objective = tuned.objective().clone();
    let mut kernel = device.compile_tuned(tuned).unwrap();
    assert_eq!(kernel.tuning().objective(), &selected_objective);
    assert!(matches!(
        kernel.tuning().conditions().implementation(),
        ImplementationConditions::Metal(_)
    ));
    kernel.execute(&[out], &[]).unwrap();
    let mut result = [0; 16];
    backing.read(&mut result).unwrap();
    assert_eq!(&result[..4], &[0xa5; 4]);
    assert_eq!(
        &result[4..12],
        &[3f32.to_le_bytes(), 3f32.to_le_bytes()].concat()
    );
    assert_eq!(&result[12..], &[0xa5; 4]);
    assert!(
        kernel
            .execute(&[backing.view(0..8).unwrap()], &[])
            .unwrap_err()
            .contains("allocation/view")
    );
}

#[test]
fn packet_table_lookup_has_a_complete_model_without_weight_values() {
    use seismic_lang::{
        interp::{Rng, TensorData},
        lower::{Options, lower_selected},
        lowered_ir::{Alternative, DecisionKind},
        repr,
    };
    let program = compile(&[SourceFile {
        path: "packet-table.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn decode(x:tensor[32] iq4g32,out:tensor[32] f32):\n  values = load(x)\n  result = tile[32] f32\n  for i in owned(result): result[i] = values[i]\n  store(result,out)\n".into(),
    }], &[]).unwrap();
    let lowered = lower_selected(&program, "decode", "cpu", &HashMap::new(), &HashMap::new(), &Options::default(), &mut |d| {
        Ok(match d.kind {
            DecisionKind::Representation { .. } => Alternative::DecodedPackets,
            DecisionKind::PacketDecode { .. } => Alternative::PacketWidth(7),
            _ => d.alternatives.get(0).unwrap(),
        })
    }).unwrap();
    let prepared = seismic_cpu::prepare(&lowered, LoadStrategy::BorrowProvenReadOnly).unwrap();
    let hardware = contract(&lowered);
    let device = Device::cpu();
    let source = TensorData::random_packed(&mut Rng(0x329018), repr::lookup("iq4g32").unwrap(), vec![32]);
    let mut bindings = source.device_bytes().iter().map(|p| device.buffer_from(p).unwrap()).collect::<Vec<_>>();
    let output = device.buffer(32 * 4).unwrap();
    bindings.push(output.clone());
    // Workload captures only allocation/scalar facts. The model may not inspect
    // the packed code values or branch according to native observations.
    let workload = tuner::workload("unknown table codes", &bindings, &[], &[]).unwrap();
    let derived = derive_scalar(&prepared, &hardware, &workload, DerivationLimits {
        instructions: 100_000, operations: 100_000,
    }).unwrap();
    assert!(derived.model.unmapped.is_empty(), "{:?}", derived.model.unmapped);
    assert!(derived.accesses.iter().any(|a| a.write));
    // Native execution is separately covered by completed-selection tests.
    // This fixture checks that unknown packed values have a complete IR account.
}

#[test]
fn runtime_rejects_preselected_frontends_and_fixed_decomposition() {
    let program = gpu_program();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cpu", &HashMap::new()).unwrap();
    let device = Device::cpu();
    let facts = device.facts();
    let hardware = Hardware::Cpu(contract(&lowered));
    let buffers = [device.buffer(4).unwrap(), device.buffer(4).unwrap()];
    let workload = tuner::workload("gate", &buffers, &[], &[]).unwrap();
    let mut request = Request { input: Input::Lowered(&lowered), device: &facts, form: Form::CpuScalar,
        hardware: &hardware, workload: &workload, derivation_limits: DerivationLimits { instructions: 1, operations: 1 } };
    assert!(matches!(tuner::tune(&request, settings()), Err(e) if e.contains("requires portable source")));
    let options = seismic_lang::lower::Options { piece: Some(1), ..Default::default() };
    let shapes = HashMap::new(); let elements = HashMap::new();
    request.input = Input::Portable { program: &program, entry: "copy", shapes: &shapes, elements: &elements, options: &options };
    assert!(matches!(tuner::tune(&request, settings()), Err(e) if e.contains("fixed stream piece")));
}

#[test]
fn automatic_selection_checks_control_contents_before_native_execution() {
    let program = compile(&[SourceFile {
        path: "control.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn count(control: tensor[1] i32, out: tensor[1] f32):\n  y = tile[1] f32\n  for i in owned(y):\n    if control[0] == 3:\n      y[i] = 3.0\n    else:\n      y[i] = 4.0\n  store(y, out)\n".into(),
    }], &[]).unwrap();
    let lowered = seismic_lang::lower::lower(&program, "count", "cpu", &HashMap::new()).unwrap();
    let model = Hardware::Cpu(contract(&lowered));
    let device = Device::cpu();
    let facts = device.facts();
    let control = device.buffer_from(&3i32.to_le_bytes()).unwrap();
    let out = device.buffer(4).unwrap();
    let buffers = vec![control.clone(), out.clone()];
    let mut workload = tuner::workload("control bytes", &buffers, &[], &[]).unwrap();
    let uncaptured = Request {
        input: source_input(&program), device: &facts, form: Form::CpuScalar,
        hardware: &model, workload: &workload,
        derivation_limits: DerivationLimits { instructions: 100_000, operations: 100_000 },
    };
    let Outcome::Incomplete(progress) = tuner::tune(&uncaptured, settings()).unwrap() else {
        panic!("unknown branch analysis must remain incomplete, not infeasible");
    };
    assert!(progress.unsupported_analyses().any(|(_, reason)| reason.contains("data-dependent branch")));
    tuner::capture_contents(&mut workload, &buffers, &[0]).unwrap();
    let request = Request {
        input: source_input(&program), device: &facts, form: Form::CpuScalar,
        hardware: &model, workload: &workload,
        derivation_limits: DerivationLimits { instructions: 100_000, operations: 100_000 },
    };
    let Outcome::Optimal(tuned) = tuner::tune(&request, settings()).unwrap() else { panic!("control data must admit complete selection"); };
    let mut kernel = device.compile_tuned(tuned).unwrap();
    kernel.execute(&buffers, &[]).unwrap();
    let mut actual = [0; 4]; out.read(&mut actual).unwrap();
    assert_eq!(f32::from_le_bytes(actual), 3.0);
    control.write(&4i32.to_le_bytes()).unwrap();
    assert!(kernel.execute(&buffers, &[]).unwrap_err().contains("contents differ"));
    out.read(&mut actual).unwrap();
    assert_eq!(f32::from_le_bytes(actual), 3.0, "rejected inputs must not execute");
    control.write(&3i32.to_le_bytes()).unwrap();
    let mut writable_workload = workload.clone();
    tuner::capture_contents(&mut writable_workload, &buffers, &[1]).unwrap();
    let writable_request = Request { workload: &writable_workload, ..request };
    let Outcome::Optimal(tuned) = tuner::tune(&writable_request, settings()).unwrap() else { panic!("bounded writable case must complete analysis"); };
    let mut kernel = device.compile_tuned(tuned).unwrap();
    assert!(kernel.execute(&buffers, &[]).unwrap_err().contains("may be modified"));
}

#[test]
fn content_conditions_cannot_be_invalidated_inside_a_batch() {
    use seismic_runtime::plan::{Bindings, PlanCompiler, Settings, Submission};
    let program = compile(&[SourceFile {
        path: "contents.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn copy(x: tensor[1] f32, out: tensor[1] f32):\n  a = load(x)\n  store(a, out)\n".into(),
    }], &[]).unwrap();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cpu", &HashMap::new()).unwrap();
    let device = Device::cpu();
    let settings = Settings {
        hardware: Hardware::Cpu(contract(&lowered)), form: Form::CpuScalar,
        derivation_limits: DerivationLimits { instructions: 100_000, operations: 100_000 }, search: settings(),
    };
    let mut compiler = PlanCompiler::new(&device, &program, settings);
    let plan = compiler.compile_entry("copy", &HashMap::new(), &HashMap::new(), &Default::default()).unwrap();
    struct Binding { input: seismic_runtime::Buffer, out: seismic_runtime::Buffer, known: bool }
    impl Bindings for Binding {
        fn buffer(&self, root: &str, _: &str) -> Option<&seismic_runtime::Buffer> { match root { "x" => Some(&self.input), "out" => Some(&self.out), _ => None } }
        fn scalar(&self, _: &str) -> Option<f64> { None }
        fn known_buffer(&self, root: &str, _: &str) -> bool { self.known && root == "x" }
    }
    let source = device.buffer_from(&1f32.to_le_bytes()).unwrap();
    let changed = device.buffer_from(&2f32.to_le_bytes()).unwrap();
    let output = device.buffer_from(&0f32.to_le_bytes()).unwrap();
    // Compile the conditioned reader first. If an unconditioned copy already
    // exists it can safely serve the reader without a captured-byte assumption.
    let reader = plan.prepare(&Binding { input: source.clone(), out: output, known: true }).unwrap();
    let mut batch = Submission::default();
    batch.append(plan.prepare(&Binding { input: changed, out: source.clone(), known: false }).unwrap());
    batch.append(reader);
    assert!(batch.execute_batched().unwrap_err().contains("may modify"));
    let mut bytes = [0; 4]; source.read(&mut bytes).unwrap();
    assert_eq!(f32::from_le_bytes(bytes), 1.0, "invalid batch must not submit its first kernel");
}

#[test]
fn completed_artifact_reuses_weaker_content_contract() {
    use seismic_runtime::plan::{Bindings, PlanCompiler, Settings};
    let program = gpu_program();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cpu", &HashMap::new()).unwrap();
    let device = Device::cpu();
    let settings = Settings { hardware: Hardware::Cpu(contract(&lowered)), form: Form::CpuScalar,
        derivation_limits: DerivationLimits { instructions: 100_000, operations: 100_000 }, search: settings() };
    let mut compiler = PlanCompiler::new(&device, &program, settings);
    let mut plan = compiler.compile_entry("copy", &HashMap::new(), &HashMap::new(), &Default::default()).unwrap();
    struct Binding { input: seismic_runtime::Buffer, out: seismic_runtime::Buffer, known: bool, enabled: bool }
    impl Bindings for Binding {
        fn buffer(&self, root: &str, _: &str) -> Option<&seismic_runtime::Buffer> {
            match root { "x" => Some(&self.input), "out" => Some(&self.out), _ => None }
        }
        fn scalar(&self, name: &str) -> Option<f64> { (name == "enabled").then_some(if self.enabled { 1. } else { 0. }) }
        fn known_buffer(&self, root: &str, _: &str) -> bool { self.known && root == "x" }
    }
    let mut binding = Binding { input: device.buffer(4).unwrap(), out: device.buffer(4).unwrap(), known: false, enabled: true };
    binding.input.write(&3f32.to_le_bytes()).unwrap();
    plan.execute(&binding).unwrap();
    assert_eq!(plan.kernel_count(), 1);
    binding.known = true;
    for value in [7f32, 11f32] {
        binding.input.write(&value.to_le_bytes()).unwrap();
        plan.execute(&binding).unwrap();
        let mut output = [0; 4];
        binding.out.read(&mut output).unwrap();
        assert_eq!(f32::from_le_bytes(output), value);
        assert_eq!(plan.kernel_count(), 1, "additional facts cannot require retuning a valid broader artifact");
    }
    binding.enabled = false;
    plan.execute(&binding).unwrap();
    assert_eq!(plan.kernel_count(), 2, "changed scalar conditions must not reuse the old artifact");

}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_automatic_selection_reuses_checked_indirect_input_domains() {
    use seismic_metal::{execution, model};
    use seismic_runtime::{Buffer, plan::{Bindings, PlanCompiler, Settings}};
    use tuner::{BufferIntegerDomain, IntegerRange};
    let program = compile(&[SourceFile {
        path: "varying-gather.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn gather(table: tensor[4, 2] f32, tokens: tensor[1] i32, out: tensor[2] f32, bias: i32):\n  values = load(table[tokens[0]])\n  y = tile[2] f32\n  for i in owned(y): y[i] = values[i] + f32(bias)\n  store(y, out)\n".into(),
    }], &[]).unwrap();
    let lowered = seismic_lang::lower::lower(&program, "gather", "metal", &HashMap::new()).unwrap();
    let mut keys = Vec::new();
    for placement in [
        seismic_realization::dispatch::TilePlacement::Replicated,
        seismic_realization::dispatch::TilePlacement::Distributed,
        seismic_realization::dispatch::TilePlacement::GroupShared,
    ] {
        let execution = execution::prepare_storage_selected(&lowered, execution::Config::default(), &mut |_| Ok(placement.clone())).unwrap();
        let required = model::requirements(&execution).unwrap();
        assert!(required.unmapped.is_empty(), "{:?}", required.unmapped);
        for key in required.primitives { if !keys.contains(&key) { keys.push(key); } }
    }
    let hardware = Hardware::Metal(model::Hardware {
        identity: "functional varying-input fixture; native timing unqualified".into(),
        timebase: Timebase { seconds_numerator: 1, seconds_denominator: 1 },
        resources: vec![Resource { name: "service".into(), capacity: 1024, unit: CapacityUnit::Slots }],
        resident_groups: 2,
        resident_shared_bytes: 65536,
        timings: keys.into_iter().map(|primitive| model::Timing {
            primitive, latency: 1,
            services: vec![model::Service { resource: 0, offset: 0, duration: 1, units: model::Units::PerLane(1) }],
        }).collect(),
    });
    let device = Device::metal().unwrap();
    let settings = Settings { hardware, form: Form::Metal,
        derivation_limits: DerivationLimits { instructions: 1_000_000, operations: 1_000_000 },
        search: seismic_runtime::tuner::Settings { limits: seismic_runtime::tuner::Limits { work: 100_000, ..Default::default() }, ..Default::default() } };
    let mut compiler = PlanCompiler::new(&device, &program, settings);
    let mut plan = compiler.compile_entry("gather", &HashMap::new(), &HashMap::new(), &Default::default()).unwrap();
    struct Inputs { table: Buffer, tokens: Buffer, out: Buffer, bias: f64 }
    impl Bindings for Inputs {
        fn buffer(&self, root: &str, _: &str) -> Option<&Buffer> {
            match root { "table" => Some(&self.table), "tokens" => Some(&self.tokens), "out" => Some(&self.out), _ => None }
        }
        fn scalar(&self, name: &str) -> Option<f64> { (name == "bias").then_some(self.bias) }
        fn known_buffer(&self, root: &str, _: &str) -> bool { root == "tokens" }
        fn scalar_domain(&self, name: &str) -> Option<IntegerRange> {
            (name == "bias").then_some(IntegerRange { min: 0, max: 8, stride: 1 })
        }
        fn buffer_domains(&self, root: &str, _: &str) -> Vec<BufferIntegerDomain> {
            if root != "tokens" { return Vec::new(); }
            vec![BufferIntegerDomain { offset: 0, bytes: 4, signed: true, range: IntegerRange { min: 0, max: 3, stride: 1 } }]
        }
    }
    let mut inputs = Inputs {
        table: device.buffer_from(&[10f32, 11., 20., 21., 30., 31., 40., 41.].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>()).unwrap(),
        tokens: device.buffer_from(&0i32.to_le_bytes()).unwrap(),
        out: device.buffer(8).unwrap(), bias: 0.,
    };
    for (token, bias) in [(0i32, 0.), (3, 5.), (1, 8.), (0, 2.)] {
        inputs.tokens.write(&token.to_le_bytes()).unwrap();
        inputs.bias = bias;
        plan.execute(&inputs).unwrap();
        let mut output = [0; 8];
        inputs.out.read(&mut output).unwrap();
        let expected = (token + 1) as f32 * 10. + bias as f32;
        assert_eq!(output.to_vec(), [expected, expected + 1.].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>());
        assert_eq!(plan.kernel_count(), 1, "changed admitted input must reuse completed selection");
    }
    let mut submission = plan.prepare(&inputs).unwrap();
    inputs.tokens.write(&4i32.to_le_bytes()).unwrap();
    assert!(submission.execute_batched().unwrap_err().contains("integer input domain"));
    assert!(plan.prepare(&inputs).err().unwrap().contains("integer input domain"));
    assert_eq!(plan.kernel_count(), 1);
}
