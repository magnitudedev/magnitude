use seismic_lang::{
    Scope,
    ir::{Builtin, ExprKind, Stmt, StmtKind},
    lower::{self, Options},
    lowered_ir::{Alternative, DecisionKind},
    program::{SourceFile, compile},
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{Candidate, Device};
use std::collections::HashMap;
const BODY: &str = r#"
fn merge[S](a:tile[S] f32,b:tile[S] f32,y:tile[S] f32):
  for i in owned(y): y[i] = a[i] + b[i]
fn step[S](s:tile[S] f32,a:tile[S] f16,b:tile[S] f32,y:tile[S] f32):
  for i in owned(y): y[i] = fma(f32(a[i]), b[i], s[i])
construct contraction[M,N,K](a:tile[M,K] f16,b:tile[N,K] f32,out:tile[M,N] f32):
  for i,j in owned(out):
    left = a[i:i+1,:]
    right = b[j:j+1,:]
    s = tile[1] f32
    z = tile[1] f32
    for t in owned(s): s[t] = out[i,j]
    for t in owned(z): z[t] = 0.0
    reduce((left,right),1,merge,into=(s,),step=step,identity=(z,),ordered=true)
    out[i,j] = s[0]
fn evaluate(x:tensor[1,5] f32,w:tensor[1,5] f32,y:tensor[1,1] f32):
  for row in parallel:
    raw = load(x[row:row+1,:])
    a = tile[1,5] f16
    for i,k in owned(a):
      tmp = raw[i,k] * 1.0003
      a[i,k] = f16(tmp)
    b = load(w)
    out = tile[1,1] f32
    for i,j in owned(out): out[i,j] = 3.0
    contraction(a,b,out)
    store(out,y[row:row+1,:])
"#;
fn load_sizes(body: &[Stmt], out: &mut Vec<i64>) {
    for s in body {
        match &s.kind {
            StmtKind::Assign { value, .. }
                if matches!(
                    value.kind,
                    ExprKind::Builtin {
                        name: Builtin::Load,
                        ..
                    }
                ) =>
            {
                out.push(
                    value
                        .ty
                        .shaped()
                        .unwrap()
                        .shape
                        .iter()
                        .map(|n| n.as_constant().unwrap())
                        .product(),
                )
            }
            StmtKind::Range { body, .. }
            | StmtKind::Owned { body, .. }
            | StmtKind::Parallel { body, .. } => load_sizes(body, out),
            _ => {}
        }
    }
}
fn exercise(device: Device, backend: &str, candidate: Candidate) {
    let program = compile(
        &[
            SourceFile {
                path: "logical.seismic.portable".into(),
                scope: Scope::Portable,
                text: BODY.into(),
            },
            SourceFile {
                path: format!("logical.seismic.{backend}").into(),
                scope: Scope::Backend(backend.into()),
                text: "lower contraction: portable\n".into(),
            },
        ],
        &[],
    )
    .unwrap();
    let mut expected = None;
    for capacity in 1..=5 {
        let f = lower::lower_selected(
            &program,
            "evaluate",
            backend,
            &HashMap::new(),
            &HashMap::new(),
            &Options {
                piece: Some(capacity),
                ..Default::default()
            },
            &mut |d| {
                if matches!(d.kind, DecisionKind::Producer { .. })
                    && d.alternatives.contains(&Alternative::Recompute)
                {
                    Ok(Alternative::Recompute)
                } else {
                    Ok(d.alternatives.get(0).unwrap())
                }
            },
        )
        .unwrap();
        assert!(
            f.decisions
                .iter()
                .any(|d| matches!(d.domain.kind, DecisionKind::Stream { maximum: 5, .. })),
            "{:?}",
            f.decisions
        );
        assert!(
            f.selections
                .iter()
                .filter(|s| s.construct == "contraction")
                .all(|s| s.shape_args[2] <= capacity)
        );
        let mut sizes = Vec::new();
        load_sizes(&f.body, &mut sizes);
        assert!(
            sizes.iter().all(|&n| n <= capacity),
            "capacity {capacity}: {sizes:?}\n{:#?}",
            f.body
        );
        let mut kernel = device.compile(&f, candidate.clone()).unwrap();
        let floats = |v: &[f32]| {
            device
                .buffer_from(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .unwrap()
        };
        let out = device.buffer(4).unwrap();
        kernel
            .execute(
                &[
                    floats(&[1.0001, 2.001, 1000.125, 0.0003, -2.75]),
                    floats(&[0.25, -0.5, 1.5, 2., -1.]),
                    out.clone(),
                ],
                &[],
            )
            .unwrap();
        let mut bytes = [0u8; 4];
        out.read(&mut bytes).unwrap();
        if let Some(expected) = expected {
            assert_eq!(bytes, expected);
        } else {
            expected = Some(bytes);
        }
    }
}

#[test]
fn partition_dependent_computations_do_not_acquire_a_contraction_domain() {
    for body in [
        BODY.replace("out[i,j] = s[0]","out[i,j] = s[0] * 2.0"),
        BODY.replace("s[t] = out[i,j]","s[t] = out[i,j] + 1.0"),
        BODY.replace("left = a[i:i+1,:]","left = tile[1,K] f16\n    for p,k in owned(left): left[p,k] = a[i,0]"),
        BODY.replace("left = a[i:i+1,:]","left = tile[1,K] f16\n    for p,k in owned(left): left[p,k] = f16(f32(a[i,k]) * f32(K))"),
    ] {
        let program=compile(&[SourceFile{path:"dependent.seismic.portable".into(),scope:Scope::Portable,text:body},SourceFile{path:"dependent.seismic.cpu".into(),scope:Scope::Backend("cpu".into()),text:"lower contraction: portable\n".into()}],&[]).unwrap();
        let f=lower::lower_specialized(&program,"evaluate","cpu",&HashMap::new(),&HashMap::new(),&Options::default()).unwrap();
        assert!(!f.decisions.iter().take_while(|d|!matches!(d.domain.kind,DecisionKind::Construct{..})).any(|d|matches!(d.domain.kind,DecisionKind::Stream{..})),"{:?}",f.decisions);
        assert!(f.selections.iter().filter(|s|s.construct=="contraction").all(|s|s.shape_args[2]==5));
    }
}

#[test]
fn logical_contraction_derives_bounded_inputs_before_selecting_the_backend_body() {
    exercise(
        Device::cpu(),
        "cpu",
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}
#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_logical_contraction_preserves_all_partition_sizes() {
    exercise(
        Device::metal().unwrap(),
        "metal",
        Candidate::Metal(Default::default()),
    );
}

fn dynamic_computed_stream(device: Device, backend: &str, candidate: Candidate) {
    let source = r#"
fn consume[N](raw:tile[N] f32,out:tensor[1] f32):
  values = tile[N] f16
  for k in owned(values):
    shifted = raw[k] + f32(k)
    values[k] = f16(shifted)
  result = tile[1] f32
  for i in owned(result): result[i] = 0.0
  result[0] = reduce(values,0,sum,ordered=true)
  store(result,out)
fn evaluate(x:tensor[17] f32,visible:tensor[2] i32,out:tensor[1] f32):
  raw = load(x[visible[0]:visible[1]])
  consume(raw,out)
"#;
    let program = compile(
        &[SourceFile {
            path: "dynamic.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap();
    for capacity in [1, 3, 7, 17] {
        let f = lower::lower_selected(
            &program,
            "evaluate",
            backend,
            &HashMap::new(),
            &HashMap::new(),
            &Options {
                piece: Some(capacity),
                ..Default::default()
            },
            &mut |d| {
                Ok(
                    if matches!(d.kind, DecisionKind::Producer { .. })
                        && d.alternatives.contains(&Alternative::Recompute)
                    {
                        Alternative::Recompute
                    } else {
                        d.alternatives.get(0).unwrap()
                    },
                )
            },
        )
        .unwrap();
        let streams = f
            .body
            .iter()
            .filter_map(|s| match &s.kind {
                StmtKind::LoadLoop { offset, views, .. } => Some((offset, views)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(streams.len(), 1, "{:#?}", f.body);
        assert!(streams[0].0.is_some());
        assert!(
            streams[0]
                .1
                .iter()
                .all(|v| matches!(v.ty, seismic_lang::types::Ty::Tensor(_))),
            "{:#?}",
            f.body
        );
        assert!(!f.body.iter().any(|s|matches!(&s.kind,StmtKind::Assign{value,..} if matches!(value.kind,ExprKind::Builtin{name:Builtin::Load,..}))),"full logical input must be eliminated");
        let mut kernel = device.compile(&f, candidate.clone()).unwrap();
        let x = device
            .buffer_from(
                &[1.0003f32; 17]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let output = device.buffer(4).unwrap();
        for (start, end) in [(0i32, 0i32), (2, 3), (3, 16), (0, 17)] {
            let visible = device
                .buffer_from(
                    &[start, end]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            kernel
                .execute(&[x.clone(), visible, output.clone()], &[])
                .unwrap();
            let mut bytes = [0; 4];
            output.read(&mut bytes).unwrap();
            let n = (end - start) as f32;
            assert_eq!(
                f32::from_le_bytes(bytes),
                n * (n + 1.0) / 2.0,
                "capacity {capacity}, window {start}:{end}"
            );
        }
    }
}
#[test]
fn dynamic_computed_stream_preserves_logical_indices_and_rounding() {
    dynamic_computed_stream(
        Device::cpu(),
        "cpu",
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
    dynamic_computed_stream(
        Device::cpu(),
        "cpu",
        Candidate::Cpu {
            loads: LoadStrategy::BorrowProvenReadOnly,
        },
    );
}
#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_dynamic_computed_stream_preserves_logical_indices_and_rounding() {
    dynamic_computed_stream(
        Device::metal().unwrap(),
        "metal",
        Candidate::Metal(Default::default()),
    );
}

#[test]
fn reduction_producer_regions_are_projected_without_full_output_storage() {
    let source = r#"
fn evaluate(x:tensor[7,3] f32,out:tensor[1] f32):
  input = load(x)
  values = tile[7] f32
  for n in owned(values): values[n] = 0.0
  for n in owned(values):
    row = input[n,:]
    values[n] = reduce(row,0,sum,ordered=true) * 2.0 + f32(n)
  result = tile[1] f32
  for i in owned(result): result[i] = 0.0
  result[0] = reduce(values,0,sum,ordered=true)
  store(result,out)
"#;
    let program = compile(
        &[SourceFile {
            path: "regions.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap();
    let device = Device::cpu();
    for capacity in [1, 2, 3, 7] {
        let f = lower::lower_selected(
            &program,
            "evaluate",
            "cpu",
            &HashMap::new(),
            &HashMap::new(),
            &Options {
                piece: Some(capacity),
                ..Default::default()
            },
            &mut |d| {
                Ok(
                    if matches!(d.kind, DecisionKind::Producer { .. })
                        && d.alternatives.contains(&Alternative::Recompute)
                    {
                        Alternative::Recompute
                    } else {
                        d.alternatives.get(0).unwrap()
                    },
                )
            },
        )
        .unwrap();
        if capacity < 7 {
            assert!(!f.body.iter().any(|s|matches!(&s.kind,StmtKind::Assign{value,..} if matches!(&value.kind,ExprKind::TileAlloc{shape,..} if shape.len()==1 && shape[0].as_constant()==Some(7)))),"{:#?}",f.body);
        }
        let mut kernel = device
            .compile(
                &f,
                Candidate::Cpu {
                    loads: LoadStrategy::Materialize,
                },
            )
            .unwrap();
        let x = device
            .buffer_from(
                &[1f32; 21]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let out = device.buffer(4).unwrap();
        kernel.execute(&[x, out.clone()], &[]).unwrap();
        let mut bytes = [0; 4];
        out.read(&mut bytes).unwrap();
        assert_eq!(f32::from_le_bytes(bytes), 63.0, "capacity {capacity}");
    }
}

fn dynamic_region_stream(device: Device, backend: &str, candidate: Candidate) {
    let source = r#"
fn consume[N](input:tile[N,3] f32,out:tensor[1] f32):
  values = tile[N] f32
  for n in owned(values): values[n] = 0.0
  for n in owned(values):
    row = input[n,:]
    values[n] = reduce(row,0,sum,ordered=true) * 2.0 + f32(n)
  result = tile[1] f32
  for i in owned(result): result[i] = 0.0
  result[0] = reduce(values,0,sum,ordered=true)
  store(result,out)
fn evaluate(x:tensor[17,3] f32,visible:tensor[2] i32,out:tensor[1] f32):
  input = load(x[visible[0]:visible[1],:])
  consume(input,out)
"#;
    let program = compile(
        &[SourceFile {
            path: "dynamic-region.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap();
    for capacity in [1, 3, 7, 17] {
        let f = lower::lower_selected(
            &program,
            "evaluate",
            backend,
            &HashMap::new(),
            &HashMap::new(),
            &Options {
                piece: Some(capacity),
                ..Default::default()
            },
            &mut |d| {
                Ok(
                    if matches!(d.kind, DecisionKind::Producer { .. })
                        && d.alternatives.contains(&Alternative::Recompute)
                    {
                        Alternative::Recompute
                    } else {
                        d.alternatives.get(0).unwrap()
                    },
                )
            },
        )
        .unwrap();
        assert!(f.body.iter().any(|s|matches!(&s.kind,StmtKind::LoadLoop{domain,views,..} if views.is_empty() && matches!(domain.view.ty,seismic_lang::types::Ty::Tensor(_)))),"computed region must use actual domain metadata, without a dummy value load: {:#?}",f.body);
        let mut kernel = device.compile(&f, candidate.clone()).unwrap();
        let x = device
            .buffer_from(
                &[1f32; 51]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let out = device.buffer(4).unwrap();
        for (start, end) in [(0i32, 0i32), (2, 3), (3, 16), (0, 17)] {
            let visible = device
                .buffer_from(
                    &[start, end]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            kernel
                .execute(&[x.clone(), visible, out.clone()], &[])
                .unwrap();
            let mut bytes = [0; 4];
            out.read(&mut bytes).unwrap();
            let n = (end - start) as f32;
            assert_eq!(
                f32::from_le_bytes(bytes),
                6.0 * n + n * (n - 1.0) / 2.0,
                "capacity {capacity}, {start}:{end}"
            );
        }
    }
}
#[test]
fn dynamic_nested_reduction_producer_has_no_dummy_load() {
    dynamic_region_stream(
        Device::cpu(),
        "cpu",
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}
#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_dynamic_nested_reduction_producer_has_no_dummy_load() {
    dynamic_region_stream(
        Device::metal().unwrap(),
        "metal",
        Candidate::Metal(Default::default()),
    );
}

fn projected_matrix(device: Device, backend: &str, candidate: Candidate) {
    let portable = include_str!("../../../../seismic-std/lib/constructs/matmul.seismic.portable");
    let implementation = if backend == "metal" {
        include_str!("../../../../seismic-std/lib/constructs/matmul.seismic.metal")
    } else {
        "lower matmul: portable\n"
    };
    let source = r#"
fn evaluate(x:tensor[8,8] f32,w:tensor[16,8] f32,out:tensor[8] f32):
  a = load(x)
  b = load(w)
  scores = tile[8,16] f32
  for i,j in owned(scores): scores[i,j] = 0.0
  matmul(a,b,scores)
  sums = reduce(scores,1,sum,ordered=true)
  store(sums,out)
"#;
    let program = compile(
        &[
            SourceFile {
                path: "matrix.seismic.portable".into(),
                scope: Scope::Portable,
                text: format!("{portable}\n{source}"),
            },
            SourceFile {
                path: format!("matrix.seismic.{backend}").into(),
                scope: Scope::Backend(backend.into()),
                text: implementation.into(),
            },
        ],
        &[],
    )
    .unwrap();
    let f = lower::lower_selected(
        &program,
        "evaluate",
        backend,
        &HashMap::new(),
        &HashMap::new(),
        &Options {
            piece: Some(8),
            ..Default::default()
        },
        &mut |d| {
            Ok(
                if matches!(d.kind, DecisionKind::Producer { .. })
                    && d.alternatives.contains(&Alternative::Recompute)
                {
                    Alternative::Recompute
                } else {
                    d.alternatives.get(0).unwrap()
                },
            )
        },
    )
    .unwrap();
    let matrices = f
        .selections
        .iter()
        .filter(|s| s.construct == "matmul")
        .collect::<Vec<_>>();
    assert!(!matrices.is_empty());
    assert!(
        matrices.iter().all(|s| s.shape_args == vec![8, 8, 8]),
        "{matrices:#?}\n{:#?}",
        f.body
    );
    fn full_storage(body: &[Stmt]) -> bool {
        body.iter().any(|s|match &s.kind{
        StmtKind::Assign{value,..} if matches!(&value.kind,ExprKind::TileAlloc{shape,..} if shape.iter().map(|n|n.as_constant()).collect::<Vec<_>>()==vec![Some(8),Some(16)])=>true,
        StmtKind::Owned{body,..}|StmtKind::Range{body,..}|StmtKind::Parallel{body,..}|StmtKind::LoadLoop{body,..}=>full_storage(body),
        StmtKind::If{then,els,..}=>full_storage(then)||full_storage(els),
        StmtKind::Reduction(r)=>r.bodies().any(full_storage),_=>false})
    }
    assert!(
        !full_storage(&f.body),
        "whole logical score allocation survived projection"
    );
    if backend == "metal" {
        assert!(
            matrices
                .iter()
                .all(|s| matches!(s.choice, seismic_lang::lowered_ir::Choice::Block(_)))
        );
    }
    let mut kernel = device.compile(&f, candidate).unwrap();
    let a = (0..64).map(|i| (i % 7) as f32 - 3.0).collect::<Vec<_>>();
    let b = (0..128).map(|i| (i % 5) as f32 - 2.0).collect::<Vec<_>>();
    let upload = |v: &[f32]| {
        device
            .buffer_from(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let out = device.buffer(32).unwrap();
    kernel
        .execute(&[upload(&a), upload(&b), out.clone()], &[])
        .unwrap();
    let mut bytes = [0u8; 32];
    out.read(&mut bytes).unwrap();
    for i in 0..8 {
        let expected = (0..16)
            .map(|j| (0..8).map(|k| a[i * 8 + k] * b[j * 8 + k]).sum::<f32>())
            .sum::<f32>();
        let actual = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(actual, expected, "row {i}");
    }
}
#[test]
fn bounded_producer_call_selects_its_backend_body_after_output_projection() {
    projected_matrix(
        Device::cpu(),
        "cpu",
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}
#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn bounded_matrix_producer_preserves_actual_metal_matrix_cover() {
    projected_matrix(
        Device::metal().unwrap(),
        "metal",
        Candidate::Metal(Default::default()),
    );
}

fn dynamic_attention(device: Device, backend: &str, candidate: Candidate) {
    let program = compile(
        &[
            SourceFile {
                path: "attention.seismic.portable".into(),
                scope: Scope::Portable,
                text: include_str!(
                    "../../../../seismic-std/lib/kernels/attention.seismic.portable"
                )
                .into(),
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
                path: format!("matmul.seismic.{backend}").into(),
                scope: Scope::Backend(backend.into()),
                text: if backend == "metal" {
                    include_str!("../../../../seismic-std/lib/constructs/matmul.seismic.metal")
                        .into()
                } else {
                    "lower matmul: portable\n".into()
                },
            },
        ],
        &[],
    )
    .unwrap();
    let mut expected_storage = None;
    for history_capacity in [32, 64, 128] {
        let f = lower::lower_selected(
            &program,
            "attention",
            backend,
            &HashMap::from([
                ("Q".into(), 1),
                ("T".into(), history_capacity),
                ("H".into(), 2),
                ("KV".into(), 1),
                ("W".into(), 8),
            ]),
            &HashMap::from([(
                "A".into(),
                seismic_lang::types::Elem::Dtype(seismic_lang::types::DType::F32),
            )]),
            &Options {
                piece: Some(7),
                ..Default::default()
            },
            &mut |d| {
                Ok(
                    if matches!(d.kind, DecisionKind::Producer { .. })
                        && d.alternatives.contains(&Alternative::Recompute)
                    {
                        Alternative::Recompute
                    } else {
                        d.alternatives.get(0).unwrap()
                    },
                )
            },
        )
        .unwrap();
        let history = f
            .vars
            .iter()
            .filter(|v| v.name == "kt")
            .map(|v| v.ty.shaped().unwrap().shape[0].clone())
            .collect::<Vec<_>>();
        // The variable table retains both checked and specialized declarations.
        // Check every full-domain symbol, not the first diagnostic-name match.
        assert!(!history.is_empty());
        assert!(history.iter().all(|n| n.as_constant().is_none()));
        assert!(f.decisions.iter().all(|decision| {
            !matches!(&decision.domain.kind, DecisionKind::Construct { name, shape_args, .. }
                if name == "matmul" && history.contains(&shape_args[2]))
        }), "dynamic contraction body was selected before its history domain was partitioned");
        fn allocations(body: &[Stmt], data: &std::collections::HashSet<usize>, found: &mut Vec<Vec<seismic_lang::sym::Sym>>) {
            for s in body {
                match &s.kind {
                    StmtKind::Assign { target, value, .. } => {
                        if matches!(target.kind, ExprKind::Var(v) if !data.contains(&v)) { continue; }
                        if matches!(
                            value.kind,
                            ExprKind::TileAlloc { .. }
                                | ExprKind::Load { .. }
                                | ExprKind::Builtin {
                                    name: Builtin::Load,
                                    ..
                                }
                        ) {
                            found.push(value.ty.shaped().unwrap().shape.clone());
                        }
                    }
                    StmtKind::Owned { body, .. }
                    | StmtKind::Range { body, .. }
                    | StmtKind::Parallel { body, .. }
                    | StmtKind::LoadLoop { body, .. } => allocations(body, data, found),
                    StmtKind::If { then, els, .. } => {
                        allocations(then, data, found);
                        allocations(els, data, found);
                    }
                    StmtKind::Reduction(r) => {
                        for b in r.bodies() {
                            allocations(b, data, found);
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut allocated = Vec::new();
        allocations(&f.body, &seismic_lang::demand::data_variables(&f.body), &mut allocated);
        assert!(
            !allocated.iter().any(|shape| history.iter().any(|n| shape.contains(n))),
            "whole-domain allocation: {allocated:?}"
        );
        let execution =
            seismic_runtime::execution::Execution::prepare(&f, candidate.clone(), &device.facts())
                .unwrap();
        use seismic_accounting::quantity::Count;
        use seismic_runtime::execution::Account;
        let storage = match execution.account().unwrap() {
            Account::CpuScalarIr { account, .. } => {
                vec![Count::Exact(account.scratch_bytes_per_invocation)]
            }
            Account::Cuda(phases) => phases.iter().map(|phase| {
                Count::Exact(phase.scalar_ir().scratch_bytes_per_invocation)
            }).collect(),
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
            storage.iter().all(|n| matches!(n, Count::Exact(_))),
            "selected allocation sizes must be exact: {storage:?}"
        );
        if let Some(expected) = &expected_storage {
            assert_eq!(
                &storage, expected,
                "selected storage scales with history capacity {history_capacity}"
            );
        } else {
            expected_storage = Some(storage);
        }
        let mut kernel = device.compile_execution(execution).unwrap();
        let q = (0..16)
            .map(|i| (i % 7) as f32 / 7. - 0.4)
            .collect::<Vec<_>>();
        let k = (0..history_capacity * 8)
            .map(|i| (i % 11) as f32 / 11. - 0.3)
            .collect::<Vec<_>>();
        let v = (0..history_capacity * 8)
            .map(|i| (i % 13) as f32 / 13. - 0.5)
            .collect::<Vec<_>>();
        let upload = |v: &[f32]| {
            device
                .buffer_from(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .unwrap()
        };
        let visible = device
            .buffer_from(&[3i32.to_le_bytes(), 17i32.to_le_bytes()].concat())
            .unwrap();
        let out = device.buffer(64).unwrap();
        kernel
            .execute(
                &[upload(&q), upload(&k), upload(&v), visible, out.clone()],
                &[0.25],
            )
            .unwrap();
        let mut bytes = [0u8; 64];
        out.read(&mut bytes).unwrap();
        for h in 0..2 {
            let weights = (3..17)
                .map(|t| {
                    ((0..8)
                        .map(|w| q[h * 8 + w] as f64 * k[t * 8 + w] as f64)
                        .sum::<f64>()
                        * 0.25)
                        .exp()
                })
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f64>();
            for w in 0..8 {
                let expected = weights
                    .iter()
                    .enumerate()
                    .map(|(i, a)| a * v[(i + 3) * 8 + w] as f64)
                    .sum::<f64>()
                    / denominator;
                let actual = f32::from_le_bytes(
                    bytes[(h * 8 + w) * 4..(h * 8 + w + 1) * 4]
                        .try_into()
                        .unwrap(),
                );
                assert!(
                    (actual as f64 - expected).abs() < 1e-5,
                    "{h},{w}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn dynamic_attention_recomputes_bounded_score_producers() {
    dynamic_attention(
        Device::cpu(),
        "cpu",
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_dynamic_attention_recomputes_bounded_score_producers() {
    dynamic_attention(
        Device::cuda(0).unwrap(),
        "cuda",
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
    );
}
#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_dynamic_attention_recomputes_bounded_score_producers() {
    dynamic_attention(
        Device::metal().unwrap(),
        "metal",
        Candidate::Metal(Default::default()),
    );
}
