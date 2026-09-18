//! Ordered retained calls share grouped operands while preserving local state.
use seismic_lang::{
    ir::{Builtin, Expr, ExprKind, StmtKind},
    lower::{lower_selected, Options},
    lowered_ir::{Alternative, DecisionKind, LoweredIr},
    program::{compile, Program, SourceFile},
    Scope,
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{Candidate, Device};

const CONTRACTION: &str = r#"
fn merge[S](a:tile[S] f32,b:tile[S] f32,y:tile[S] f32):
  for i in owned(y): y[i] = a[i] + b[i]
fn step[S](s:tile[S] f32,a:tile[S] f32,b:tile[S] f32,y:tile[S] f32):
  for i in owned(y): y[i] = fma(a[i],b[i],s[i])
construct first[M,N,K](a:tile[M,K] f32,b:tile[N,K] f32,out:tile[M,N] f32):
  for i,j in owned(out):
    left = a[i:i+1,:]
    right = b[j:j+1,:]
    s = tile[1] f32
    zero = tile[1] f32
    for t in owned(s): s[t] = out[i,j]
    for t in owned(zero): zero[t] = 0.0
    reduce((left,right),1,merge,into=(s,),step=step,identity=(zero,),ordered=true)
    out[i,j] = s[0]
"#;
fn program(text: &str, backend: &str, lowering: &str) -> Program {
    compile(
        &[
            SourceFile {
                path: "multiple.seismic.portable".into(),
                scope: Scope::Portable,
                text: text.into(),
            },
            SourceFile {
                path: format!("multiple.seismic.{backend}").into(),
                scope: Scope::Backend(backend.into()),
                text: lowering.into(),
            },
        ],
        &[],
    )
    .unwrap()
}
fn source(mode: &str) -> String {
    source_with_k(mode, 3)
}
fn source_with_k(mode: &str, k: usize) -> String {
    let second = CONTRACTION[CONTRACTION.find("construct first").unwrap()..]
        .replace("construct first", "construct second");
    let middle = match mode {
        "dependent" => "    for i,j in owned(u): u[i,j] = f32(f16(g[i,j])) + bias\n",
        "same" => "    for i,j in owned(g): g[i,j] = f32(f16(g[i,j])) + bias\n",
        "mutated" => "    for i,k in owned(a): a[i,k] = a[i,k] + 0.25\n",
        _ => "",
    };
    let output = if mode == "same" { "g" } else { "u" };
    let publication = if mode == "same" {
        "    for i,j in owned(g): g[i,j] = f32(f16(g[i,j])) + bias\n"
    } else {
        "    for i,j in owned(u): u[i,j] = f32(f16(u[i,j])) + f32(f16(g[i,j])) * 0.75 + bias\n"
    };
    format!("{CONTRACTION}{second}\nfn evaluate(x:tensor[1,{k}] f32,w0:tensor[5,{k}] f32,w1:tensor[5,{k}] f32,seed:tensor[1,5] f32,out:tensor[1,5] f32):\n  for column in parallel:\n    a = load(x)\n    b = load(w0[column:column+1,:])\n    c = load(w1[column:column+1,:])\n    g = load(seed[:,column:column+1])\n    u = tile[1,1] f32\n    bias = f32(column) * 0.125\n    for i,j in owned(u): u[i,j] = -bias\n    first(a,b,g)\n{middle}    second(a,c,{output})\n{publication}    store({output},out[:,column:column+1])\n")
}
fn lower(program: &Program, backend: &str, width: i64, capacities: [i64; 2]) -> LoweredIr {
    lower_with_composition(program, backend, width, capacities, None)
}
fn lower_with_composition(
    program: &Program,
    backend: &str,
    width: i64,
    capacities: [i64; 2],
    composition: Option<(i64, bool)>,
) -> LoweredIr {
    let rectangles = if width == 1 || 5 % width == 0 { 1 } else { 2 };
    let mut stream = 0;
    lower_selected(
        program,
        "evaluate",
        backend,
        &Default::default(),
        &Default::default(),
        &Options::default(),
        &mut |decision| {
            Ok(match &decision.kind {
                DecisionKind::OutputGroup { calls, extent, .. } => {
                    assert_eq!(*extent, 5);
                    assert_eq!(calls.len(), 2);
                    assert!(calls[0].position < calls[1].position);
                    assert!(calls.iter().all(|call| call.parameter == "N"));
                    assert_eq!(
                        decision.alternatives.iter().collect::<Vec<_>>(),
                        (1..=5).map(Alternative::OutputWidth).collect::<Vec<_>>()
                    );
                    Alternative::OutputWidth(width)
                }
                DecisionKind::Stream { maximum, .. } => {
                    // Retained-call partitioning visits both source calls in each
                    // rectangle before body selection introduces primitive folds.
                    let capacity = if stream < 2 * rectangles {
                        capacities[stream % 2]
                    } else {
                        *maximum
                    };
                    stream += 1;
                    Alternative::StreamCapacity(capacity.min(*maximum))
                }
                DecisionKind::Producer { ty, .. }
                    if composition.is_some_and(|(k, _)| {
                        ty.shaped().is_some_and(|shape| {
                            shape.shape.last().and_then(|n| n.as_constant()) == Some(k)
                        })
                    }) && decision.alternatives.contains(&Alternative::Recompute) =>
                {
                    // Project the source's contraction dimension. Keep each
                    // bounded piece materialized for both retained consumers.
                    Alternative::Recompute
                }
                DecisionKind::StreamFusion { .. } if composition.is_some_and(|(_, fuse)| fuse) => {
                    Alternative::Fuse
                }
                DecisionKind::Intermediate { .. }
                    if composition.is_some()
                        && decision.alternatives.contains(&Alternative::RetainLocal) =>
                {
                    Alternative::RetainLocal
                }
                _ => decision.alternatives.get(0).unwrap(),
            })
        },
    )
    .unwrap()
}
fn values(mode: &str) -> (Vec<Vec<f32>>, Vec<f32>) {
    values_with_k(mode, 3)
}
fn values_with_k(mode: &str, k: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let x = (0..k)
        .map(|i| [1.0003f32, -0.7231, 0.1259][i % 3])
        .collect::<Vec<_>>();
    let w0 = (0..5 * k)
        .map(|i| i as f32 * 0.173 - 0.31)
        .collect::<Vec<_>>();
    let w1 = (0..5 * k)
        .map(|i| 0.217 - i as f32 * 0.091)
        .collect::<Vec<_>>();
    let seed = (0..5)
        .map(|i| 0.371 + i as f32 * 0.0313)
        .collect::<Vec<_>>();
    let rounded = seismic_lang::numeric::f16_round;
    let expected = (0..5)
        .map(|column| {
            let bias = column as f32 * 0.125;
            let g = (0..k).fold(seed[column], |sum, q| x[q].mul_add(w0[column * k + q], sum));
            let initial = if mode == "same" || mode == "dependent" {
                rounded(g) + bias
            } else {
                -bias
            };
            let u = (0..k).fold(initial, |sum, q| {
                let a = if mode == "mutated" { x[q] + 0.25 } else { x[q] };
                a.mul_add(w1[column * k + q], sum)
            });
            if mode == "same" {
                rounded(u) + bias
            } else {
                rounded(u) + rounded(g) * 0.75 + bias
            }
        })
        .collect();
    (vec![x, w0, w1, seed], expected)
}
fn execute(device: &Device, candidate: Candidate, ir: &LoweredIr, mode: &str) {
    let (values, expected) = values(mode);
    execute_values(device, candidate, ir, mode, &values, &expected);
}
fn execute_values(
    device: &Device,
    candidate: Candidate,
    ir: &LoweredIr,
    mode: &str,
    values: &[Vec<f32>],
    expected: &[f32],
) {
    let mut kernel = device.compile(ir, candidate).unwrap();
    execute_kernel_values(device, &mut kernel, mode, values, expected);
}
fn execute_kernel_values(
    device: &Device,
    kernel: &mut seismic_runtime::Kernel,
    mode: &str,
    values: &[Vec<f32>],
    expected: &[f32],
) {
    let mut inputs = values
        .iter()
        .map(|values| {
            device
                .buffer_from(
                    &values
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    let output = device.buffer(20).unwrap();
    inputs.push(output.clone());
    kernel.execute(&inputs, &[]).unwrap();
    let mut bytes = [0; 20];
    output.read(&mut bytes).unwrap();
    let actual = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual.as_slice(), expected, "{mode}");
}
#[test]
fn every_small_width_and_two_contraction_capacities_preserve_call_state() {
    let device = Device::cpu();
    for mode in ["independent", "dependent", "same"] {
        let program = program(
            &source(mode),
            "cpu",
            "lower first: portable\nlower second: portable\n",
        );
        for width in 1..=5 {
            for first in 1..=3 {
                for second in 1..=3 {
                    let ir = lower(&program, "cpu", width, [first, second]);
                    assert!(ir
                        .decisions
                        .iter()
                        .any(|d| matches!(d.domain.kind, DecisionKind::OutputGroup { .. })));
                    for (name, capacity) in [("first", first), ("second", second)] {
                        assert!(
                            ir.selections.iter().any(
                                |s| s.construct == name && s.shape_args == [1, width, capacity]
                            ),
                            "{mode} width{width} capacities{first},{second}: {:?}",
                            ir.selections
                        );
                    }
                    execute(
                        &device,
                        Candidate::Cpu {
                            loads: LoadStrategy::Materialize,
                        },
                        &ir,
                        mode,
                    );
                }
            }
        }
    }
}
fn tensor_root(expression: &Expr) -> Option<usize> {
    match &expression.kind {
        ExprKind::Var(v) => Some(*v),
        ExprKind::Index { base, .. } | ExprKind::Transpose(base) => tensor_root(base),
        _ => None,
    }
}
#[test]
fn unchanged_shared_operand_is_loaded_once_and_mutation_invalidates_reuse() {
    let device = Device::cpu();
    for mode in ["independent", "mutated"] {
        let program = program(
            &source(mode),
            "cpu",
            "lower first: portable\nlower second: portable\n",
        );
        let ir = lower(&program, "cpu", 2, [3, 3]);
        if mode == "independent" {
            let snapshots = ir.body.iter().filter_map(|s| match &s.kind { StmtKind::Parallel { body, .. } => Some(body), _ => None }).flatten().filter(|s| matches!(&s.kind,
                StmtKind::Assign { value: Expr { kind: ExprKind::Builtin { name: Builtin::Load, args }, .. }, .. } if tensor_root(&args[0]) == Some(0)
            )).count();
            assert_eq!(
                snapshots, 2,
                "one shared activation snapshot in each full/tail rectangle"
            );
        }
        execute(
            &device,
            Candidate::Cpu {
                loads: LoadStrategy::Materialize,
            },
            &ir,
            mode,
        );
    }
}
#[test]
fn checked_call_writes_prevent_sharing_coordinate_dependent_scalar_arguments() {
    let text = source("independent").replace("    second(a,c,u)", "    scale = g[0,0]\n    broadcast(scale,u)") + "\nconstruct broadcast[M,N](scale:f32,out:tile[M,N] f32):\n  for i,j in owned(out): out[i,j] = scale\n";
    let program = program(
        &text,
        "cpu",
        "lower first: portable\nlower second: portable\nlower broadcast: portable\n",
    );
    let ir = lower_selected(
        &program,
        "evaluate",
        "cpu",
        &Default::default(),
        &Default::default(),
        &Options::default(),
        &mut |d| Ok(d.alternatives.get(0).unwrap()),
    )
    .unwrap();
    assert!(!ir
        .decisions
        .iter()
        .any(|d| matches!(d.domain.kind, DecisionKind::OutputGroup { .. })));
}

fn native(device: Device, candidate: Candidate) {
    for mode in ["independent", "dependent", "same", "mutated"] {
        let program = program(
            &source(mode),
            device.backend(),
            "lower first: portable\nlower second: portable\n",
        );
        for width in [2, 5] {
            let ir = lower(&program, device.backend(), width, [2, 3]);
            execute(&device, candidate.clone(), &ir, mode);
        }
    }
}
#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_multi_call_grouping_preserves_state_rounding_and_tails() {
    native(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_multi_call_grouping_preserves_state_rounding_and_tails() {
    native(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::Sequential,
                loads: LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
    );
}

#[test]
fn source_alias_admission_survives_two_calls_and_whole_axis_grouping() {
    let device = Device::cpu();
    for shifted in [false, true] {
        let length = if shifted { 6 } else { 5 };
        let start = if shifted { "column+1" } else { "column" };
        let end = if shifted { "column+2" } else { "column+1" };
        let text = format!("construct add[N](a:tile[N] f32,y:tile[N] f32):\n  for i in owned(y): y[i] = y[i] + a[i] * 2.0\nfn evaluate(x:tensor[{length}] f32,out:tensor[5] f32):\n  for column in parallel:\n    a = load(x[{start}:{end}])\n    y = tile[1] f32\n    for i in owned(y): y[i] = f32(column)\n    add(a,y)\n    add(a,y)\n    store(y,out[column:column+1])\n");
        let program = program(&text, "cpu", "lower add: portable\n");
        for width in [1, 5] {
            let ir = lower(&program, "cpu", width, [3, 3]);
            assert_eq!(ir.alias_requirements.len(), 1);
            assert_eq!(ir.alias_requirements[0].exact_allowed, !shifted);
            let values = (0..length).map(|i| i as f32 + 1.0).collect::<Vec<_>>();
            let initial = values
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>();
            let shared = device.buffer_from(&initial).unwrap();
            let mut kernel = device
                .compile(
                    &ir,
                    Candidate::Cpu {
                        loads: LoadStrategy::Materialize,
                    },
                )
                .unwrap();
            let result = kernel.execute(&[shared.clone(), shared.clone()], &[]);
            let mut actual = vec![0; initial.len()];
            shared.read(&mut actual).unwrap();
            if shifted {
                assert!(result.is_err());
                assert_eq!(
                    actual, initial,
                    "source alias rejection must precede both calls' writes"
                );
                let mut native = seismic_cpu::compile(&ir).unwrap();
                let shared = seismic_cpu::Buffer::from_bytes(&initial).unwrap();
                assert!(native
                    .run_resident(&[shared.clone(), shared.clone()], &[])
                    .is_err());
            } else {
                result.unwrap();
                assert_eq!(
                    actual,
                    (0..5)
                        .flat_map(|i| (i as f32 + values[i] * 4.0).to_le_bytes())
                        .collect::<Vec<_>>()
                );
            }
            let input = device.buffer_from(&initial).unwrap();
            let output = device.buffer(20).unwrap();
            kernel.execute(&[input, output.clone()], &[]).unwrap();
            let mut actual = [0; 20];
            output.read(&mut actual).unwrap();
            assert_eq!(
                actual.as_slice(),
                (0..5)
                    .flat_map(|i| (i as f32 + values[i + usize::from(shifted)] * 4.0).to_le_bytes())
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[cfg(target_os = "macos")]
fn two_grouped_std_contractions() -> LoweredIr {
    let text = "fn evaluate(x:tensor[9,17] f32,w0:tensor[11,17] f32,w1:tensor[11,17] f32,out:tensor[9,11] f32):\n  for row,column in parallel:\n    a = load(x[row:row+1,:])\n    b = load(w0[column:column+1,:])\n    c = load(w1[column:column+1,:])\n    g = tile[1,1] f32\n    u = tile[1,1] f32\n    for i,j in owned(g): g[i,j] = f32(row) * 0.125\n    for i,j in owned(u): u[i,j] = f32(column) * 0.25\n    matmul(a,b,g)\n    matmul(a,c,u)\n    for i,j in owned(u): u[i,j] = f32(f16(u[i,j])) + f32(f16(g[i,j]))\n    store(u,out[row:row+1,column:column+1])\n";
    let program = compile(
        &[
            SourceFile {
                path: "two_matrices.seismic.portable".into(),
                scope: Scope::Portable,
                text: text.into(),
            },
            SourceFile {
                path: "matmul.seismic.portable".into(),
                scope: Scope::Portable,
                text: include_str!(
                    "../../../../seismic-std/lib/constructs/matmul.seismic.portable"
                )
                .into(),
            },
            SourceFile {
                path: "matmul.seismic.metal".into(),
                scope: Scope::Backend("metal".into()),
                text: include_str!("../../../../seismic-std/lib/constructs/matmul.seismic.metal")
                    .into(),
            },
        ],
        &[],
    )
    .unwrap();
    let ir = lower_selected(
        &program,
        "evaluate",
        "metal",
        &Default::default(),
        &Default::default(),
        &Options {
            piece: Some(8),
            ..Default::default()
        },
        &mut |decision| {
            Ok(match &decision.kind {
                DecisionKind::OutputGroup { calls, .. } => {
                    assert_eq!(calls.len(), 2);
                    Alternative::OutputWidth(8)
                }
                _ => decision.alternatives.get(0).unwrap(),
            })
        },
    )
    .unwrap();
    ir
}

#[test]
#[cfg(target_os = "macos")]
fn two_grouped_std_contractions_retain_actual_metal_matrix_covers() {
    let ir = two_grouped_std_contractions();
    assert_eq!(
        ir.selections
            .iter()
            .filter(|s| s.construct == "matmul" && s.shape_args == [8, 8, 8])
            .count(),
        2
    );
    let emitted = seismic_metal::msl::emit(&ir).unwrap();
    assert!(
        emitted
            .source
            .matches("simdgroup_multiply_accumulate")
            .count()
            >= 2
    );
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_two_std_contractions_preserve_seeds_casts_and_all_axis_tails() {
    let ir = two_grouped_std_contractions();
    let device = Device::metal().unwrap();
    let mut kernel = device
        .compile(&ir, Candidate::Metal(Default::default()))
        .unwrap();
    // Binary fractions keep every multiply and partial accumulation exact,
    // while the publication casts still round the seeded results to f16.
    let a = (0..9 * 17)
        .map(|i| ((i % 7) as f32 - 3.) * 0.0625)
        .collect::<Vec<_>>();
    let b = (0..11 * 17)
        .map(|i| ((i % 5) as f32 - 2.) * 0.03125)
        .collect::<Vec<_>>();
    let c = (0..11 * 17)
        .map(|i| ((i % 11) as f32 - 5.) * 0.0625)
        .collect::<Vec<_>>();
    let upload = |values: &[f32]| {
        device
            .buffer_from(
                &values
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
    };
    let output = device.buffer(9 * 11 * 4).unwrap();
    kernel
        .execute(&[upload(&a), upload(&b), upload(&c), output.clone()], &[])
        .unwrap();
    let mut bytes = vec![0; 9 * 11 * 4];
    output.read(&mut bytes).unwrap();
    for row in 0..9 {
        for column in 0..11 {
            let first = (0..17).fold(row as f32 * 0.125, |sum, k| {
                a[row * 17 + k].mul_add(b[column * 17 + k], sum)
            });
            let second = (0..17).fold(column as f32 * 0.25, |sum, k| {
                a[row * 17 + k].mul_add(c[column * 17 + k], sum)
            });
            let expected =
                seismic_lang::numeric::f16_round(first) + seismic_lang::numeric::f16_round(second);
            let offset = (row * 11 + column) * 4;
            let actual = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            assert_eq!(actual, expected, "{row},{column}");
        }
    }
}

#[test]
fn repeated_group_parameter_requires_complete_rectangular_initialization() {
    for source in [
        r#"
construct transform[N](a:tile[N,N] f32,y:tile[N] f32):
  for i in owned(y): y[i] = y[i] + a[i,i]
fn evaluate(x:tensor[5,5] f32,out:tensor[5] f32):
  for column in parallel:
    a = load(x[column:column+1,column:column+1])
    y = tile[1] f32
    for i in owned(y): y[i] = 0.0
    transform(a,y)
    transform(a,y)
    store(y,out[column:column+1])
"#,
        r#"
construct transform[N](y:tile[N,N] f32):
  for i,j in owned(y): y[i,j] = y[i,j] + 1.0
fn evaluate(out:tensor[5,1] f32):
  for column in parallel:
    y = tile[1,1] f32
    for i,j in owned(y): y[i,j] = 0.0
    transform(y)
    transform(y)
    store(y,out[column:column+1,:])
"#,
    ] {
        let program = program(source, "cpu", "lower transform: portable\n");
        let ir = lower_selected(
            &program,
            "evaluate",
            "cpu",
            &Default::default(),
            &Default::default(),
            &Options::default(),
            &mut |decision| Ok(decision.alternatives.get(0).unwrap()),
        )
        .unwrap();
        assert!(
            !ir.decisions
                .iter()
                .any(|decision| matches!(decision.domain.kind, DecisionKind::OutputGroup { .. })),
            "diagonal source blocks do not initialize the entire widened formal"
        );
    }
}

/// Source parameter identity and checked extent distinguish actual shared
/// bounded loads from equal values or one retained full-source snapshot.
fn direct_source_loads(body: &[seismic_lang::ir::Stmt]) -> Vec<(usize, i64)> {
    body.iter()
        .filter_map(|statement| {
            let StmtKind::Assign { value, .. } = &statement.kind else {
                return None;
            };
            let view = match &value.kind {
                ExprKind::Builtin {
                    name: Builtin::Load,
                    args,
                } => args.first()?,
                ExprKind::Load { view, .. } => view,
                _ => return None,
            };
            let root = tensor_root(view).filter(|root| *root < 3)?;
            let extent = value.ty.shaped()?.shape.last()?.as_constant()?;
            Some((root, extent))
        })
        .collect()
}

fn contraction_stream_loads(ir: &LoweredIr, k: i64) -> Vec<Vec<(usize, i64)>> {
    ir.body
        .iter()
        .filter_map(|statement| {
            let StmtKind::Parallel { body, .. } = &statement.kind else {
                return None;
            };
            Some(
                body.iter()
                    .filter_map(|statement| {
                        let StmtKind::Range { lo, hi, body, .. } = &statement.kind else {
                            return None;
                        };
                        if lo.as_constant() != Some(0) || hi.as_constant() != Some(k / 3) {
                            return None;
                        }
                        let mut loads = direct_source_loads(body);
                        if loads.is_empty() {
                            return None;
                        }
                        loads.sort_unstable();
                        Some(loads)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .flatten()
        .collect()
}

fn shared_bounded_pieces(device: &Device, candidate: Candidate) {
    use seismic_lang::types::Ty;
    use seismic_runtime::execution::Execution;
    let mut storage = Vec::new();
    // All sizes retain the same one-element K tail; output width2 also leaves
    // one output tail. Only the number of repeated contraction pieces grows.
    for k in [7, 13, 25] {
        let program = program(
            &source_with_k("independent", k),
            device.backend(),
            "lower first: portable\nlower second: portable\n",
        );
        let (inputs, expected) = values_with_k("independent", k);
        for fuse in [false, true] {
            let ir = lower_with_composition(
                &program,
                device.backend(),
                2,
                [3, 3],
                Some((k as i64, fuse)),
            );
            let streams = contraction_stream_loads(&ir, k as i64);
            let execution = Execution::prepare(&ir, candidate.clone(), &device.facts()).unwrap();
            if fuse {
                assert_eq!(streams, vec![vec![(0, 3), (1, 3), (2, 3)]; 2],
                    "each full/output-tail rectangle must consume one activation piece and two weight pieces");
                assert_eq!(
                    ir.decisions
                        .iter()
                        .filter(|decision| matches!(
                            decision.domain.kind,
                            DecisionKind::StreamFusion { .. }
                        ) && decision.selected == Alternative::Fuse)
                        .count(),
                    2
                );
                for statement in &ir.body {
                    if let StmtKind::Parallel { body, .. } = &statement.kind {
                        let mut tail = direct_source_loads(body)
                            .into_iter()
                            .filter(|(_, extent)| *extent == 1)
                            .collect::<Vec<_>>();
                        tail.sort_unstable();
                        assert_eq!(
                            tail,
                            vec![(0, 1), (1, 1), (2, 1)],
                            "the exact K tail must share the same activation snapshot"
                        );
                    }
                }
                let data = seismic_lang::demand::data_variables(&ir.body);
                for (id, variable) in ir.vars.iter().enumerate() {
                    if matches!(&variable.ty, Ty::Tile(shape)
                        if shape.shape.last().and_then(|n| n.as_constant()) == Some(k as i64))
                    {
                        assert!(
                            !data.contains(&id),
                            "full contraction source remains live: {}",
                            variable.name
                        );
                    }
                }
                storage.push(prepared_storage(&execution));
            } else {
                assert_eq!(
                    streams.len(),
                    4,
                    "both independent streams remain separately selectable"
                );
                assert!(streams.iter().all(|loads| loads.len() == 2));
            }
            let mut kernel = device.compile_execution(execution).unwrap();
            execute_kernel_values(device, &mut kernel, "independent", &inputs, &expected);
        }
    }
    assert!(
        storage.windows(2).all(|pair| pair[0] == pair[1]),
        "retained source storage grows with K at fixed capacity: {storage:?}"
    );
}

fn bounded_state_dependencies(device: &Device, candidate: Candidate) {
    for mode in ["dependent", "same", "mutated"] {
        let program = program(
            &source_with_k(mode, 7),
            device.backend(),
            "lower first: portable\nlower second: portable\n",
        );
        let ir = lower_with_composition(&program, device.backend(), 2, [3, 3], Some((7, true)));
        if mode != "mutated" {
            assert!(
                !ir.decisions.iter().any(|decision| matches!(
                    decision.domain.kind,
                    DecisionKind::StreamFusion { .. }
                ) && decision.selected == Alternative::Fuse),
                "a later seed must observe the completed first contraction"
            );
        }
        let (inputs, expected) = values_with_k(mode, 7);
        if mode == "mutated" {
            assert_ne!(
                expected,
                values_with_k("independent", 7).1,
                "the oracle must distinguish the two reaching activation versions"
            );
        }
        // Independent states may still interleave after a value-copy mutation;
        // reusing the unchanged activation as the second operand is forbidden.
        execute_values(device, candidate.clone(), &ir, mode, &inputs, &expected);
    }
}

fn prepared_storage(
    execution: &seismic_runtime::execution::Execution,
) -> Vec<seismic_accounting::quantity::Count> {
    use seismic_accounting::quantity::Count;
    use seismic_runtime::execution::Account;
    let quantities = match execution.account().unwrap() {
        Account::CpuScalarIr { account, .. } => {
            vec![Count::Exact(account.scratch_bytes_per_invocation)]
        }
        Account::Cuda(phases) => phases
            .iter()
            .map(|phase| Count::Exact(phase.scalar_ir().scratch_bytes_per_invocation))
            .collect(),
        #[cfg(target_os = "macos")]
        Account::MetalStorage { account, .. } => {
            let mut quantities = vec![account.retained_scratch_bytes];
            for launch in account.launches {
                quantities.extend([
                    launch.declared_private_array_bytes_per_lane,
                    launch.declared_shared_array_bytes_per_group,
                    launch.declared_fragment_payload_bytes_per_subgroup,
                ]);
            }
            quantities
        }
    };
    assert!(
        quantities
            .iter()
            .all(|quantity| matches!(quantity, Count::Exact(_))),
        "prepared storage must have an exact account: {quantities:?}"
    );
    quantities
}

#[test]
fn equal_capacity_grouped_contractions_share_bounded_activation_pieces_and_tails() {
    shared_bounded_pieces(
        &Device::cpu(),
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

#[test]
fn bounded_composition_preserves_dependent_seeds_and_mutated_activation_versions() {
    bounded_state_dependencies(
        &Device::cpu(),
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_grouped_contractions_share_bounded_pieces_and_preserve_state() {
    let device = Device::metal().unwrap();
    let candidate = Candidate::Metal(Default::default());
    shared_bounded_pieces(&device, candidate.clone());
    bounded_state_dependencies(&device, candidate);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_grouped_contractions_share_bounded_pieces_and_preserve_state() {
    let device = Device::cuda(0).unwrap();
    let candidate = Candidate::Cuda {
        options: seismic_realization::ScalarOptions {
            dispatch: seismic_realization::Dispatch::Sequential,
            loads: LoadStrategy::Materialize,
        },
        threads_per_block: 32,
    };
    shared_bounded_pieces(&device, candidate.clone());
    bounded_state_dependencies(&device, candidate);
}
