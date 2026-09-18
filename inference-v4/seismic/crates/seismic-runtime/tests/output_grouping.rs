use seismic_lang::{
    Scope,
    lower::{self, Options},
    lowered_ir::{Alternative, DecisionKind, LoweredIr},
    program::{SourceFile, compile},
    types::{DType, Elem},
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{Candidate, Device};
use std::collections::HashMap;
fn program(text: &str, backend: &str, implementation: &str) -> seismic_lang::program::Program {
    compile(
        &[
            SourceFile {
                path: "group.seismic.portable".into(),
                scope: Scope::Portable,
                text: text.into(),
            },
            SourceFile {
                path: format!("group.seismic.{backend}").into(),
                scope: Scope::Backend(backend.into()),
                text: implementation.into(),
            },
        ],
        &[],
    )
    .unwrap()
}
fn lower(
    program: &seismic_lang::program::Program,
    name: &str,
    backend: &str,
    widths: &[(&str, i64)],
    piece: Option<i64>,
    shapes: &HashMap<String, i64>,
    elements: &HashMap<String, Elem>,
) -> LoweredIr {
    lower::lower_selected(
        program,
        name,
        backend,
        shapes,
        elements,
        &Options {
            piece,
            ..Default::default()
        },
        &mut |d| {
            Ok(match &d.kind {
                DecisionKind::OutputGroup { parameter, .. } => Alternative::OutputWidth(
                    widths
                        .iter()
                        .find(|(p, _)| *p == parameter)
                        .map_or(1, |(_, n)| *n),
                ),
                _ => d.alternatives.get(0).unwrap(),
            })
        },
    )
    .unwrap()
}
fn execute(ir: &LoweredIr, input: &[f32], length: usize) -> Vec<f32> {
    execute_on(
        &Device::cpu(),
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
        ir,
        input,
        length,
    )
}
fn execute_on(
    device: &Device,
    candidate: Candidate,
    ir: &LoweredIr,
    input: &[f32],
    length: usize,
) -> Vec<f32> {
    let mut kernel = device.compile(ir, candidate).unwrap();
    let input = device
        .buffer_from(
            &input
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let out = device.buffer(length * 4).unwrap();
    kernel.execute(&[input, out.clone()], &[]).unwrap();
    let mut bytes = vec![0; length * 4];
    out.read(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
const GENERIC: &str = r#"
construct affine[N](x:tile[N] f32,scale:f32,y:tile[N] f16):
  for i in owned(y): y[i] = f16(f32(y[i]) + x[i] * scale)
fn evaluate(x:tensor[7] f32,out:tensor[7] f32):
  for row in parallel:
    raw = load(x[row:row+1])
    y = tile[1] f16
    bias = f32(row) * 0.125
    for i in owned(y): y[i] = f16(bias)
    affine(raw,1.003,y)
    for i in owned(y): y[i] = f16(f32(y[i]) + bias)
    store(y,out[row:row+1])
"#;
#[test]
fn every_width_retains_generic_calls_and_exact_nondivisor_tails() {
    let p = program(GENERIC, "cpu", "lower affine: portable\n");
    let input = (0..7).map(|i| i as f32 * 1.0003 - 2.).collect::<Vec<_>>();
    let expected = input
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let bias = i as f32 * 0.125;
            seismic_lang::numeric::f16_round(
                seismic_lang::numeric::f16_round(bias + x * 1.003) + bias,
            )
        })
        .collect::<Vec<_>>();
    for width in 1..=7 {
        let ir = lower(
            &p,
            "evaluate",
            "cpu",
            &[("N", width)],
            None,
            &HashMap::new(),
            &HashMap::new(),
        );
        let groups = ir
            .decisions
            .iter()
            .filter(|d| matches!(d.domain.kind, DecisionKind::OutputGroup { .. }))
            .collect::<Vec<_>>();
        assert_eq!(groups.len(), 1, "width {width}: {:#?}", ir.decisions);
        assert_eq!(
            groups[0].domain.alternatives.iter().collect::<Vec<_>>(),
            (1..=7).map(Alternative::OutputWidth).collect::<Vec<_>>()
        );
        let mut shapes = ir
            .selections
            .iter()
            .filter(|s| s.construct == "affine")
            .map(|s| s.shape_args[0])
            .collect::<Vec<_>>();
        shapes.sort();
        let mut wanted = vec![width];
        if 7 % width != 0 {
            wanted.push(7 % width);
        }
        wanted.sort();
        assert_eq!(shapes, wanted, "width {width}");
        assert_eq!(execute(&ir, &input, 7), expected, "width {width}");
    }
}

#[test]
fn independent_axis_rectangles_preserve_per_output_seed_and_scalar_epilogue() {
    let text = r#"
construct transform[M,N](x:tile[M,N] f32,y:tile[M,N] f32):
  for i,j in owned(y): y[i,j] = x[i,j] * 2.0 + y[i,j]
fn evaluate(x:tensor[5,7] f32,out:tensor[5,7] f32):
  for row,col in parallel:
    a = load(x[row:row+1,col:col+1])
    y = tile[1,1] f32
    bias = f32(row * 7 + col)
    for i,j in owned(y): y[i,j] = bias
    transform(a,y)
    for i,j in owned(y): y[i,j] = y[i,j] + bias * 0.25
    store(y,out[row:row+1,col:col+1])
"#;
    let p = program(text, "cpu", "lower transform: portable\n");
    let ir = lower(
        &p,
        "evaluate",
        "cpu",
        &[("M", 2), ("N", 3)],
        None,
        &HashMap::new(),
        &HashMap::new(),
    );
    let mut actual = ir
        .selections
        .iter()
        .filter(|s| s.construct == "transform")
        .map(|s| s.shape_args.clone())
        .collect::<Vec<_>>();
    actual.sort();
    assert_eq!(actual, vec![vec![1, 1], vec![1, 3], vec![2, 1], vec![2, 3]]);
    let input = (0..35).map(|i| i as f32 * 0.25).collect::<Vec<_>>();
    let expected = input
        .iter()
        .enumerate()
        .map(|(i, x)| x * 2. + i as f32 * 1.25)
        .collect::<Vec<_>>();
    assert_eq!(execute(&ir, &input, 35), expected);
}

fn linear(backend: &str, m: i64, n: i64, k: i64, width: i64, piece: i64) -> LoweredIr {
    let text = format!(
        "{}\n{}",
        include_str!("../../../../seismic-std/lib/constructs/matmul.seismic.portable"),
        include_str!("../../../../seismic-std/lib/kernels/linear.seismic.portable")
    );
    let p = program(
        &text,
        backend,
        if backend == "metal" {
            include_str!("../../../../seismic-std/lib/constructs/matmul.seismic.metal")
        } else {
            "lower matmul: portable\n"
        },
    );
    lower(
        &p,
        "linear",
        backend,
        &[("M", width), ("N", width)],
        Some(piece),
        &HashMap::from([("M".into(), m), ("N".into(), n), ("K".into(), k)]),
        &HashMap::from([
            ("T".into(), Elem::Dtype(DType::F32)),
            ("U".into(), Elem::Dtype(DType::F32)),
            ("V".into(), Elem::Dtype(DType::F32)),
        ]),
    )
}
#[test]
fn std_linear_combines_output_width_and_contraction_capacity() {
    for width in [1, 2, 3] {
        for piece in [5, 2, 1] {
            let ir = linear("cpu", 3, 3, 5, width, piece);
            assert_eq!(
                ir.decisions
                    .iter()
                    .filter(|d| matches!(d.domain.kind, DecisionKind::OutputGroup { .. }))
                    .count(),
                2
            );
            let calls = ir
                .selections
                .iter()
                .filter(|s| s.construct == "matmul")
                .collect::<Vec<_>>();
            assert!(
                calls
                    .iter()
                    .any(|s| s.shape_args == vec![width, width, piece])
            );
            let device = Device::cpu();
            let mut kernel = device
                .compile(
                    &ir,
                    Candidate::Cpu {
                        loads: LoadStrategy::Materialize,
                    },
                )
                .unwrap();
            let a = (0..15).map(|i| (i % 7) as f32 - 3.).collect::<Vec<_>>();
            let b = (0..15).map(|i| (i % 5) as f32 - 2.).collect::<Vec<_>>();
            let upload = |x: &[f32]| {
                device
                    .buffer_from(&x.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                    .unwrap()
            };
            let out = device.buffer(36).unwrap();
            kernel
                .execute(&[upload(&a), upload(&b), out.clone()], &[])
                .unwrap();
            let mut bytes = [0; 36];
            out.read(&mut bytes).unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    let expected = (0..5).map(|k| a[i * 5 + k] * b[j * 5 + k]).sum::<f32>();
                    assert_eq!(
                        f32::from_le_bytes(
                            bytes[(i * 3 + j) * 4..(i * 3 + j + 1) * 4]
                                .try_into()
                                .unwrap()
                        ),
                        expected,
                        "width {width}, piece {piece}, {i},{j}"
                    );
                }
            }
        }
    }
}

#[test]
fn computed_operand_copies_keep_mutations_and_intermediate_rounding() {
    let text = r#"
construct twice[N](x:tile[N] f16,y:tile[N] f32):
  for i in owned(y): y[i] = y[i] + f32(x[i]) * 2.0
fn evaluate(x:tensor[7] f32,out:tensor[7] f32):
  for row in parallel:
    raw = load(x[row:row+1])
    for i in owned(raw): raw[i] = raw[i] * 1.0003
    a = tile[1] f16
    for i in owned(a):
      scalar = raw[i] * 1.007
      a[i] = f16(scalar)
    y = tile[1] f32
    for i in owned(y): y[i] = f32(row)
    twice(a,y)
    published = tile[1] f32
    for i in owned(published): published[i] = y[i] + 0.5
    store(published,out[row:row+1])
"#;
    let p = program(text, "cpu", "lower twice: portable\n");
    let ir = lower(
        &p,
        "evaluate",
        "cpu",
        &[("N", 3)],
        None,
        &HashMap::new(),
        &HashMap::new(),
    );
    assert!(
        ir.selections
            .iter()
            .any(|s| s.construct == "twice" && s.shape_args == vec![3])
    );
    let input = (0..7).map(|i| i as f32 * 0.4 - 1.).collect::<Vec<_>>();
    let expected = input
        .iter()
        .enumerate()
        .map(|(i, x)| seismic_lang::numeric::f16_round((x * 1.0003) * 1.007) * 2. + i as f32 + 0.5)
        .collect::<Vec<_>>();
    assert_eq!(execute(&ir, &input, 7), expected);
}

#[test]
fn coordinate_dependent_contract_or_scalar_parameter_is_not_grouped() {
    for (definition, call) in [
        (
            "construct transform[N](x:tile[N] f32,gain:f32,y:tile[N] f32):\n  for i in owned(y): y[i] = x[i] + f32(i)\n",
            "transform(a,1.0,y)",
        ),
        (
            "construct transform[N](x:tile[N] f32,gain:f32,y:tile[N] f32):\n  for i in owned(y): y[i] = x[i] * gain\n",
            "transform(a,f32(row),y)",
        ),
    ] {
        let text = format!(
            "{definition}\nfn evaluate(x:tensor[7] f32,out:tensor[7] f32):\n  for row in parallel:\n    a = load(x[row:row+1])\n    y = tile[1] f32\n    for i in owned(y): y[i] = 0.0\n    {call}\n    store(y,out[row:row+1])\n"
        );
        let p = program(&text, "cpu", "lower transform: portable\n");
        let ir = lower(
            &p,
            "evaluate",
            "cpu",
            &[],
            None,
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(
            !ir.decisions
                .iter()
                .any(|d| matches!(d.domain.kind, DecisionKind::OutputGroup { .. })),
            "unsupported axes must remain explicitly uncovered"
        );
    }
}

#[test]
fn std_packed_projection_widens_retained_call_and_shares_activation_snapshot() {
    let text = format!(
        "{}\n{}",
        include_str!("../../../../seismic-std/lib/constructs/matmul.seismic.portable"),
        include_str!("../../../../seismic-std/lib/kernels/projection.seismic.portable")
    );
    let p = program(&text, "cpu", "lower matmul: portable\n");
    let ir = lower(
        &p,
        "projection",
        "cpu",
        &[("N", 3)],
        Some(64),
        &HashMap::from([("N".into(), 7), ("K".into(), 64)]),
        &HashMap::new(),
    );
    let mut calls = ir
        .selections
        .iter()
        .filter(|s| s.construct == "matmul")
        .map(|s| s.shape_args.clone())
        .collect::<Vec<_>>();
    calls.sort();
    assert_eq!(calls, vec![vec![1, 1, 64], vec![1, 3, 64]]);
    use seismic_lang::ir::{Builtin, ExprKind, StmtKind};
    let mut activation_loads = 0;
    for s in &ir.body {
        if let StmtKind::Parallel { body, .. } = &s.kind {
            for s in body {
                if let StmtKind::Assign { value, .. } = &s.kind {
                    if matches!(
                        value.kind,
                        ExprKind::Builtin {
                            name: Builtin::Load,
                            ..
                        }
                    ) && value
                        .ty
                        .shaped()
                        .is_some_and(|s| s.elem == Elem::Dtype(DType::BF16))
                    {
                        activation_loads += 1;
                    }
                }
            }
        }
    }
    assert_eq!(
        activation_loads, 2,
        "one invariant activation snapshot per full/tail rectangle"
    );
    // Packed inputs remain encoded snapshots with their representation-owned
    // packet geometry; this also checks compilation of grouped packed tails.
    seismic_cpu::compile_artifact(&ir, LoadStrategy::Materialize).unwrap();
}

#[test]
#[cfg(target_os = "macos")]
fn grouped_std_linear_selects_actual_metal_matrix_instructions() {
    let ir = linear("metal", 9, 11, 17, 8, 8);
    assert!(
        ir.selections
            .iter()
            .any(|s| s.construct == "matmul" && s.shape_args == vec![8, 8, 8])
    );
    let emitted = seismic_metal::msl::emit(&ir).unwrap();
    assert!(emitted.source.contains("simdgroup_multiply_accumulate"));
}

#[test]
fn source_alias_contract_survives_whole_axis_grouping_and_metadata_queries() {
    for shifted in [false, true] {
        for width in [1, 4] {
            let text = format!(
                r#"
construct copy[N](x:tile[N] f32,y:tile[N] f32):
  for i in owned(y): y[i] = x[i] * 2.0
fn evaluate(x:tensor[{}] f32,out:tensor[4] f32):
  for row in parallel:
    a = load(x[{}:{}])
    y = tile[1] f32
    observed = extent(out,0)
    for i in owned(y): y[i] = f32(observed) * 0.0
    copy(a,y)
    store(y,out[row:row+1])
"#,
                if shifted { 5 } else { 4 },
                if shifted { "row+1" } else { "row" },
                if shifted { "row+2" } else { "row+1" }
            );
            let p = program(&text, "cpu", "lower copy: portable\n");
            let ir = lower(
                &p,
                "evaluate",
                "cpu",
                &[("N", width)],
                None,
                &HashMap::new(),
                &HashMap::new(),
            );
            assert!(
                ir.selections
                    .iter()
                    .any(|s| s.construct == "copy" && s.shape_args == vec![width])
            );
            assert_eq!(ir.alias_requirements.len(), 1);
            assert_eq!(ir.alias_requirements[0].exact_allowed, !shifted);
            let device = Device::cpu();
            let mut kernel = device
                .compile(
                    &ir,
                    Candidate::Cpu {
                        loads: LoadStrategy::Materialize,
                    },
                )
                .unwrap();
            let values = if shifted {
                vec![1f32, 2., 3., 4., 5.]
            } else {
                vec![1f32, 2., 3., 4.]
            };
            let initial = values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let shared = device.buffer_from(&initial).unwrap();
            let result = kernel.execute(&[shared.clone(), shared.clone()], &[]);
            let mut actual = vec![0; initial.len()];
            shared.read(&mut actual).unwrap();
            if shifted {
                assert!(
                    result.is_err(),
                    "shifted source alias admitted at width {width}"
                );
                assert_eq!(actual, initial, "rejection must precede output writes");
                let mut native = seismic_cpu::compile(&ir).unwrap();
                let shared = seismic_cpu::Buffer::from_bytes(&initial).unwrap();
                assert!(
                    native
                        .run_resident(&[shared.clone(), shared.clone()], &[])
                        .is_err(),
                    "direct CPU admission lost the source condition at width {width}"
                );
                shared.read(&mut actual).unwrap();
                assert_eq!(actual, initial);
            } else {
                result.unwrap();
                assert_eq!(
                    actual,
                    values
                        .iter()
                        .flat_map(|v| (v * 2.).to_le_bytes())
                        .collect::<Vec<_>>()
                );
            }
            let out = device.buffer(16).unwrap();
            let x = device.buffer_from(&initial).unwrap();
            kernel.execute(&[x, out.clone()], &[]).unwrap();
            let mut actual = [0; 16];
            out.read(&mut actual).unwrap();
            let expected = (0..4)
                .map(|i| values[i + usize::from(shifted)] * 2.)
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            assert_eq!(actual.as_slice(), expected.as_slice());
        }
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn grouped_std_linear_native_matrix_and_all_axis_tails() {
    let ir = linear("metal", 9, 11, 17, 8, 8);
    let device = Device::metal().unwrap();
    let mut kernel = device
        .compile(&ir, Candidate::Metal(Default::default()))
        .unwrap();
    let a = (0..9 * 17).map(|i| (i % 7) as f32 - 3.).collect::<Vec<_>>();
    let b = (0..11 * 17)
        .map(|i| (i % 5) as f32 - 2.)
        .collect::<Vec<_>>();
    let upload = |x: &[f32]| {
        device
            .buffer_from(&x.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let out = device.buffer(9 * 11 * 4).unwrap();
    kernel
        .execute(&[upload(&a), upload(&b), out.clone()], &[])
        .unwrap();
    let mut bytes = vec![0; 9 * 11 * 4];
    out.read(&mut bytes).unwrap();
    for i in 0..9 {
        for j in 0..11 {
            let expected = (0..17).map(|k| a[i * 17 + k] * b[j * 17 + k]).sum::<f32>();
            assert_eq!(
                f32::from_le_bytes(
                    bytes[(i * 11 + j) * 4..(i * 11 + j + 1) * 4]
                        .try_into()
                        .unwrap()
                ),
                expected,
                "{i},{j}"
            );
        }
    }
}

#[test]
fn modified_loaded_operand_uses_its_value_snapshot() {
    let text = r#"
construct twice[N](x:tile[N] f32,y:tile[N] f32):
  for i in owned(y): y[i] = x[i] * 2.0
fn evaluate(x:tensor[7] f32,out:tensor[7] f32):
  for row in parallel:
    a = load(x[row:row+1])
    for i in owned(a): a[i] = a[i] + f32(row)
    y = tile[1] f32
    for i in owned(y): y[i] = 0.0
    twice(a,y)
    store(y,out[row:row+1])
"#;
    let p = program(text, "cpu", "lower twice: portable\n");
    let ir = lower(
        &p,
        "evaluate",
        "cpu",
        &[("N", 3)],
        None,
        &HashMap::new(),
        &HashMap::new(),
    );
    assert!(
        ir.selections
            .iter()
            .any(|s| s.construct == "twice" && s.shape_args == vec![3])
    );
    let input = (0..7).map(|i| i as f32 * 0.25).collect::<Vec<_>>();
    let expected = input
        .iter()
        .enumerate()
        .map(|(i, x)| (x + i as f32) * 2.)
        .collect::<Vec<_>>();
    assert_eq!(execute(&ir, &input, 7), expected);
}

#[test]
fn escaping_scalar_effects_and_same_root_cross_item_reads_do_not_form_groups() {
    let definition =
        "construct copy[N](x:tile[N] f32,y:tile[N] f32):\n  for i in owned(y): y[i] = x[i]\n";
    for source in [
        r#"
fn evaluate(x:tensor[4] f32,out:tensor[4] f32,observed:tensor[1] f32):
  counter = 0.0
  for row in parallel:
    a = load(x[row:row+1])
    y = tile[1] f32
    counter = f32(row)
    for i in owned(y): y[i] = 0.0
    copy(a,y)
    store(y,out[row:row+1])
  seen = tile[1] f32
  for i in owned(seen): seen[i] = counter
  store(seen,observed)
"#,
        r#"
fn evaluate(x:tensor[5] f32):
  for row in parallel:
    a = load(x[4-row:5-row])
    y = tile[1] f32
    for i in owned(y): y[i] = 0.0
    copy(a,y)
    store(y,x[row:row+1])
"#,
    ] {
        let p = program(
            &format!("{definition}\n{source}"),
            "cpu",
            "lower copy: portable\n",
        );
        let ir = lower(
            &p,
            "evaluate",
            "cpu",
            &[],
            None,
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(
            !ir.decisions
                .iter()
                .any(|d| matches!(d.domain.kind, DecisionKind::OutputGroup { .. }))
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_grouped_generic_values_tails_and_source_alias_contract() {
    let device = Device::cuda(0).unwrap();
    let candidate = || Candidate::Cuda {
        options: seismic_realization::ScalarOptions {
            dispatch: seismic_realization::Dispatch::ParallelRoot,
            loads: LoadStrategy::Materialize,
        },
        threads_per_block: 32,
    };
    let p = program(GENERIC, "cuda", "lower affine: portable\n");
    let input = (0..7).map(|i| i as f32 * 1.0003 - 2.).collect::<Vec<_>>();
    let expected = input
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let bias = i as f32 * 0.125;
            seismic_lang::numeric::f16_round(
                seismic_lang::numeric::f16_round(bias + x * 1.003) + bias,
            )
        })
        .collect::<Vec<_>>();
    for width in [1, 3, 7] {
        let ir = lower(
            &p,
            "evaluate",
            "cuda",
            &[("N", width)],
            None,
            &HashMap::new(),
            &HashMap::new(),
        );
        assert_eq!(execute_on(&device, candidate(), &ir, &input, 7), expected);
    }
    let source = r#"
construct copy[N](x:tile[N] f32,y:tile[N] f32):
  for i in owned(y): y[i] = x[i] * 2.0
fn evaluate(x:tensor[5] f32,out:tensor[4] f32):
  for row in parallel:
    a = load(x[row+1:row+2])
    y = tile[1] f32
    for i in owned(y): y[i] = 0.0
    copy(a,y)
    store(y,out[row:row+1])
"#;
    let p = program(source, "cuda", "lower copy: portable\n");
    for width in [1, 4] {
        let ir = lower(
            &p,
            "evaluate",
            "cuda",
            &[("N", width)],
            None,
            &HashMap::new(),
            &HashMap::new(),
        );
        let mut kernel = device.compile(&ir, candidate()).unwrap();
        let initial = [1f32, 2., 3., 4., 5.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let shared = device.buffer_from(&initial).unwrap();
        assert!(
            kernel
                .execute(&[shared.clone(), shared.clone()], &[])
                .is_err()
        );
        let mut actual = vec![0; initial.len()];
        shared.read(&mut actual).unwrap();
        assert_eq!(actual, initial);
        assert_eq!(
            execute_on(&device, candidate(), &ir, &[1., 2., 3., 4., 5.], 4),
            vec![4., 6., 8., 10.]
        );
    }
}
