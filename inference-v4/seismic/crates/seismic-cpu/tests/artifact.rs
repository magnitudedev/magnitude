use seismic_lang::{
    lowered_ir::LoweredIr,
    program::{compile, SourceFile},
    Scope,
};
use seismic_realization::LoadStrategy;
use std::collections::HashMap;
fn program(n: i64, expression: &str) -> LoweredIr {
    let source = format!("fn evaluate[N](x: tensor[N] f32, out: tensor[N] f32):\n  a = load(x)\n  y = tile[N] f32\n  for i in owned(y): y[i] = {expression}\n  store(y,out)\n");
    let program = compile(
        &[SourceFile {
            path: "native.seismic.portable".into(),
            text: source,
            scope: Scope::Portable,
        }],
        &[],
    )
    .unwrap();
    seismic_lang::lower::lower(
        &program,
        "evaluate",
        "cpu",
        &HashMap::from([("N".into(), n)]),
    )
    .unwrap()
}
#[test]
fn inspection_has_no_invocation_allocation_and_retains_native_attribution() {
    let lowered = program(1 << 36, "a[i] * 3.0 + 1.0");
    let image = seismic_cpu::compile_artifact(&lowered, LoadStrategy::Materialize).unwrap();
    assert!(image.scratch_bytes > 1 << 36);
    assert!(!image.machine_code.is_empty());
    assert_eq!(image.machine_code.len(), image.unrelocated_code.len());
    assert!(!image.vcode.is_empty());
    assert!(!image.block_starts.is_empty());
    assert!(!image.origins.is_empty());
    for origin in &image.origins {
        assert!(origin.start < origin.end);
        assert!(origin.end as usize <= image.machine_code.len());
        if let Some(id) = origin.ssa_instruction {
            // Source tokens refer to input SSA IDs, preserved in the retained IR.
            assert!(image.ir.contains(&format!("@{id:04x}")));
        }
    }
}
#[test]
fn inspected_compilation_matches_execution_and_exposes_external_math() {
    for expression in ["a[i] * 3.0 + 1.0", "exp(a[i])"] {
        let lowered = program(4, expression);
        for loads in [
            LoadStrategy::Materialize,
            LoadStrategy::BorrowProvenReadOnly,
        ] {
            let inspected = seismic_cpu::compile_artifact(&lowered, loads).unwrap();
            let mut executable = seismic_cpu::compile_candidate(&lowered, loads).unwrap();
            let executed = executable.native_artifact();
            assert_eq!(inspected.unrelocated_code, executed.unrelocated_code);
            assert_eq!(inspected.ir, executed.ir);
            assert_eq!(inspected.optimized_ir, executed.optimized_ir);
            assert_eq!(inspected.frame_bytes, executed.frame_bytes);
            if expression.starts_with("exp") {
                assert!(!inspected.imports.is_empty());
                assert!(!inspected.relocations.is_empty());
            }
            let input = [0.0f32, 1.0, -1.0, 2.0];
            let mut bytes: Vec<_> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut output = [0u8; 16];
            executable.run(&mut [&mut bytes, &mut output], &[]).unwrap();
            for (bytes, x) in output.chunks_exact(4).zip(input) {
                let got = f32::from_le_bytes(bytes.try_into().unwrap());
                let expected = if expression.starts_with("exp") {
                    x.exp()
                } else {
                    x * 3.0 + 1.0
                };
                assert_eq!(got, expected);
            }
        }
    }
}
