use cranelift_codegen::isa::CallConv;
use seismic_lang::program::{collect_files, compile};
use seismic_realization::Dispatch;
use std::{collections::HashMap, path::PathBuf};

#[path = "support/remainder.rs"]
mod remainder;

#[test]
fn signed_remainder_expansion_is_visible_in_terminal_requirements() {
    use cranelift_codegen::ir::{types, Opcode};
    use seismic_cuda::ptx::*;
    for ty in [types::I8, types::I16, types::I32, types::I64] {
        let pairs = remainder::operands(ty);
        let program = remainder::program(ty, pairs.len());
        let plan = prepare(&program).unwrap();
        let data_type = if ty == types::I64 { DataType::U64 } else { DataType::U32 };
        for instruction in program.function.layout.blocks()
            .flat_map(|b| program.function.layout.block_insts(b))
            .filter(|&i| program.function.dfg.insts[i].opcode() == Opcode::Srem)
        {
            let expansion = plan.instructions().filter(|i| matches!(i.origin,
                Origin::Ssa { instruction: source, .. } if source == instruction.as_u32())).collect::<Vec<_>>();
            assert!(expansion.iter().any(|i| matches!(i.operation,
                Operation::Binary { operation: Binary::Remainder, data_type: t, .. } if t == data_type)));
            assert!(expansion.iter().any(|i| matches!(i.operation, Operation::Select { .. })));
            assert!(!expansion.iter().any(|i| matches!(i.operation,
                Operation::Binary { operation: Binary::Remainder, data_type: DataType::S32 | DataType::S64, .. })));
            for operation in expansion {
                assert!(plan.requirements().any(|r| r == Requirement::Instruction(operation.operation.primitive())));
            }
        }
    }
}

#[test]
fn portable_projection_and_norm_emit_direct_ptx() {
    let p = compile(
        &collect_files(&[
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../seismic-std/lib")
        ])
        .unwrap(),
        &["cuda".into(), "metal".into(), "cpu".into()],
    )
    .unwrap_or_else(|e| panic!("{e:?}"));
    for (name, shapes) in [
        (
            "projection",
            HashMap::from([("N".into(), 7), ("K".into(), 64)]),
        ),
        (
            "rms_norm",
            HashMap::from([("R".into(), 3), ("W".into(), 17)]),
        ),
    ] {
        let function = p.functions.iter().find(|f| f.name == name).unwrap();
        let elements = function
            .elem_params
            .iter()
            .map(|p| {
                (
                    p.clone(),
                    seismic_lang::types::Elem::Dtype(seismic_lang::types::DType::BF16),
                )
            })
            .collect();
        let lowered = seismic_lang::lower::lower_specialized(
            &p,
            name,
            "cuda",
            &shapes,
            &elements,
            &Default::default(),
        )
        .unwrap();
        let program =
            seismic_compiler::scalar_with(&lowered, CallConv::SystemV, Dispatch::ParallelRoot)
                .unwrap();
        let plan = seismic_cuda::ptx::prepare(&program).unwrap();
        let ptx = seismic_cuda::ptx::print(&plan);
        assert!(ptx.contains(".visible .entry seismic_kernel"));
        assert!(ptx.contains("st.global.u32 [%status_addr]"));
        if let Ok(dir) = std::env::var("SEISMIC_PTX_OUTPUT") {
            std::fs::write(PathBuf::from(dir).join(format!("{name}.ptx")), ptx).unwrap();
        }
    }
}

fn scalar(source: &str) -> seismic_realization::ScalarProgram {
    let program = seismic_lang::program::compile(
        &[seismic_lang::program::SourceFile {
            path: "target.seismic.portable".into(),
            scope: seismic_lang::Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap_or_else(|errors| panic!("{errors:?}"));
    let lowered = seismic_lang::lower::lower(&program, "kernel", "cuda", &HashMap::new()).unwrap();
    seismic_compiler::scalar_with(&lowered, CallConv::SystemV, Dispatch::ParallelRoot).unwrap()
}

#[test]
fn selected_plan_owns_abi_dispatch_control_and_complete_helper_requirement() {
    use seismic_cuda::{
        execution::{Execution, Limits},
        ptx::*,
    };
    let program = scalar(
        "fn kernel(x: tensor[4] f32, y: tensor[4] f32):\n  for i in parallel:\n    a = tile[1] f32\n    for j in owned(a): a[j] = exp(x[i])\n    store(a,y[i:i+1])\n",
    );
    let execution = Execution::new(
        program,
        32,
        Limits {
            max_threads_per_block: 1024,
            max_grid_x: u32::MAX,
        },
    )
    .unwrap();
    let plan = execution.target_plan();
    plan.validate().unwrap();
    assert_eq!(plan.domain().work_items, 4);
    assert_eq!(execution.dispatch().groups, 1);
    assert!(plan.instructions().any(|i| i.origin == Origin::Dispatch
        && i.predicate.is_some()
        && matches!(i.operation, Operation::Return)));
    assert!(
        plan.instructions()
            .any(|i| i.origin == Origin::InvocationAbi
                && matches!(
                    i.operation,
                    Operation::Load {
                        space: Space::Parameter,
                        ..
                    }
                ))
    );
    assert!(plan.instructions().any(|i| matches!(
        i.operation,
        Operation::Store {
            space: Space::Global,
            ..
        }
    ) && matches!(i.effects().control, Control::Next)));
    let call = plan
        .instructions()
        .find(|i| matches!(i.operation, Operation::Call { .. }))
        .unwrap();
    assert_eq!(call.requirements().count(), 2);
    assert!(call.requirements().any(|r| matches!(
        r,
        Requirement::WholeBody(Helper {
            function: seismic_realization::MathFunction::Exp,
            library: Library::ExpLibm
        })
    )));
    assert_eq!(call.effects().parameter_reads.len(), 1);
    assert_eq!(call.effects().parameter_writes.len(), 1);
    assert_eq!(plan.libraries(), &[Library::ExpLibm]);
    let printed = print(plan);
    assert_eq!(printed, print(plan));
    assert!(printed.contains("call.uni (call_result_"));
    assert!(printed.contains(".func (.param .b32 result) seismic_exp"));
}

#[test]
fn branch_edge_copies_and_primitive_effects_survive_planning() {
    use seismic_cuda::ptx::*;
    let program = scalar(
        "fn kernel(x: tensor[1] f32, y: tensor[1] f32):\n  a = tile[1] f32\n  for j in owned(a):\n    n = 0.0\n    for k in range(0,3): n += x[0]\n    if n > 0.0: a[j] = n\n    else: a[j] = 0.0\n  store(a,y)\n",
    );
    let plan = prepare(&program).unwrap();
    assert!(
        plan.body()
            .iter()
            .any(|item| matches!(item, Item::Label(Label::Edge(_))))
    );
    assert!(
        plan.registers()
            .iter()
            .any(|r| matches!(r.name, RegisterName::Temporary(_)))
    );
    for instruction in plan.instructions() {
        let effects = instruction.effects();
        for register in effects
            .register_reads
            .iter()
            .chain(&effects.register_writes)
        {
            assert!(register.0 < plan.registers().len());
        }
        assert!(
            instruction
                .requirements()
                .any(|r| matches!(r, Requirement::Instruction(_)))
        );
    }
    assert!(plan.instructions().any(|i| {
        matches!(i.operation, Operation::Compare { .. })
            && i.effects()
                .register_writes
                .iter()
                .all(|r| plan.registers()[r.0].class == RegisterClass::Predicate)
    }));
}
