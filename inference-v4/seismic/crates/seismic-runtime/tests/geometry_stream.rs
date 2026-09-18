//! Stream snapshots retain geometry and guards without element storage.
use seismic_lang::{
    Scope,
    ir::*,
    lower::{Options, lower_selected},
    lowered_ir::{Alternative, DecisionKind, LoweredIr},
    program::{Program, SourceFile, compile},
    sym::{Atom, Sym},
    types::Ty,
};
use seismic_realization::{CallConv, Dispatch, LoadStrategy, ScalarOptions};
use seismic_runtime::{Candidate, Device};

fn program(source: &str, backend: Option<&str>) -> Program {
    let mut sources = vec![SourceFile {
        path: "geometry_stream.seismic.portable".into(),
        scope: Scope::Portable,
        text: source.into(),
    }];
    if let Some(backend) = backend {
        sources.push(SourceFile {
            path: format!("geometry_stream.seismic.{backend}").into(),
            scope: Scope::Backend(backend.into()),
            text: "lower fold: portable\n".into(),
        });
    }
    compile(&sources, &[]).unwrap()
}
fn encode(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn output(buffer: &seismic_runtime::Buffer) -> i32 {
    let mut bytes = [0; 4];
    buffer.read(&mut bytes).unwrap();
    i32::from_le_bytes(bytes)
}
// The query observes the unchanged width while decomposition partitions K.
const ORDINARY: &str = r#"
fn merge[M](a:tile[M] i32,b:tile[M] i32,out:tile[M] i32):
  for j in owned(out): out[j] = a[j] + b[j]
construct fold[K,M](a:tile[K,1] i32,shape_only:tile[K,M] i32,acc:tile[1] i32):
  values = tile[K,1] i32
  for k,j in owned(values): values[k,j] = a[k,j] + extent(shape_only,1)
  reduce((values,),0,merge,into=(acc,),ordered=true)
fn evaluate(x:tensor[8,4] i32,bounds:tensor[2] i32,out:tensor[1] i32):
  controls = load(bounds)
  window = x[controls[0]:controls[1],:]
  a = load(window[:,0:1])
  shape_only = load(window[:,1:3])
  for i in owned(controls): controls[i] = 0
  acc = tile[1] i32
  for i in owned(acc): acc[i] = 0
  fold(a,shape_only,acc)
  store(acc,out)
"#;
fn stream_bindings(body: &[Stmt], found: &mut Vec<VarId>) {
    for statement in body {
        match &statement.kind {
            StmtKind::LoadLoop { vars, body, .. } => {
                found.extend(vars);
                stream_bindings(body, found);
            }
            StmtKind::Owned { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::Parallel { body, .. }
            | StmtKind::Lanes { body, .. } => stream_bindings(body, found),
            StmtKind::If { then, els, .. } => {
                stream_bindings(then, found);
                stream_bindings(els, found);
            }
            _ => {}
        }
    }
}
fn ordinary(device: &Device, candidate: &Candidate) {
    for piece in [1, 3, 8] {
        let ir = lower_selected(
            &program(ORDINARY, Some(device.backend())),
            "evaluate",
            device.backend(),
            &Default::default(),
            &Default::default(),
            &Options::default(),
            &mut |decision| {
                Ok(match decision.kind {
                    DecisionKind::Stream { maximum, .. } => {
                        Alternative::StreamCapacity(piece.min(maximum))
                    }
                    _ => decision.alternatives.get(0).unwrap(),
                })
            },
        )
        .unwrap();
        let data = seismic_lang::demand::data_variables(&ir.body);
        let mut bindings = Vec::new();
        stream_bindings(&ir.body, &mut bindings);
        assert!(
            bindings.iter().any(|v| !data.contains(v)),
            "ordinary construct must retain an actual geometry-only stream binding"
        );
        assert!(
            bindings.iter().any(|v| data.contains(v)),
            "the numerical operand must still transfer elements"
        );
        let mut kernel = device.compile(&ir, candidate.clone()).unwrap();
        let input = device
            .buffer_from(&encode(&(0..32).collect::<Vec<_>>()))
            .unwrap();
        let bounds = device.buffer(8).unwrap();
        let result = device.buffer(4).unwrap();
        for (start, end) in [(0, 8), (1, 6), (-9, 3), (4, 99), (6, 2), (0, 0)] {
            bounds.write(&encode(&[start, end])).unwrap();
            kernel
                .execute(&[input.clone(), bounds.clone(), result.clone()], &[])
                .unwrap();
            let end = end.clamp(0, 8);
            let start = start.clamp(0, end);
            assert_eq!(
                output(&result),
                (start..end).map(|row| row * 4 + 2).sum::<i32>(),
                "piece={piece}, window={start}:{end}"
            );
        }
    }
}

// Build an explicit internal stream from checked ordinary loads. This isolates
// runtime guards that source shape checking often already proves.
fn explicit(source: &str, backend: &str, capacity: Option<i64>, count: usize) -> LoweredIr {
    let mut function = program(source, None)
        .functions
        .into_iter()
        .find(|f| f.name == "evaluate")
        .unwrap();
    let at = function
        .body
        .iter()
        .position(|s| matches!(s.kind, StmtKind::Range { .. }))
        .unwrap();
    let statement = function.body.remove(at);
    let StmtKind::Range {
        var: offset,
        mut body,
        ..
    } = statement.kind
    else {
        unreachable!()
    };
    let mut bindings = Vec::new();
    let mut views = Vec::new();
    let piece = Atom::Param("geometry_piece".into());
    for statement in body.drain(..count) {
        let StmtKind::Assign { target, value, .. } = statement.kind else {
            panic!("fixture expects load assignments")
        };
        let ExprKind::Var(variable) = target.kind else {
            unreachable!()
        };
        let ExprKind::Builtin {
            name: Builtin::Load,
            args,
        } = value.kind
        else {
            panic!("fixture expects logical loads")
        };
        if capacity.is_some() {
            let Ty::Tile(shape) = &mut function.vars[variable].ty else {
                unreachable!()
            };
            shape.shape[0] = Sym::atom(piece.clone());
        }
        bindings.push(variable);
        views.push(args.into_iter().next().unwrap());
    }
    fn expression(e: &mut Expr, vars: &[Var], bindings: &[VarId]) {
        match &mut e.kind {
            ExprKind::Var(v) if bindings.contains(v) => e.ty = vars[*v].ty.clone(),
            ExprKind::Index { base, indices } => {
                expression(base, vars, bindings);
                for index in indices {
                    match index {
                        Index::Point(e) => expression(e, vars, bindings),
                        Index::Slice { start, end } => {
                            for e in start.iter_mut().chain(end) {
                                expression(e, vars, bindings);
                            }
                        }
                    }
                }
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                expression(lhs, vars, bindings);
                expression(rhs, vars, bindings);
            }
            ExprKind::Builtin { args, .. }
            | ExprKind::Call { args, .. }
            | ExprKind::Intrinsic { args, .. }
            | ExprKind::Tuple(args) => {
                for a in args {
                    expression(a, vars, bindings);
                }
            }
            ExprKind::Cast { expr, .. }
            | ExprKind::Unary { expr, .. }
            | ExprKind::Transpose(expr)
            | ExprKind::Load { view: expr, .. }
            | ExprKind::Accessor { base: expr, .. }
            | ExprKind::Lanes { base: expr, .. } => expression(expr, vars, bindings),
            _ => {}
        }
    }
    fn statements(body: &mut [Stmt], vars: &[Var], bindings: &[VarId]) {
        for statement in body {
            match &mut statement.kind {
                StmtKind::Assign { target, value, .. } => {
                    expression(target, vars, bindings);
                    expression(value, vars, bindings);
                }
                StmtKind::Expr(e) => expression(e, vars, bindings),
                StmtKind::Owned { tile, body, .. } => {
                    expression(tile, vars, bindings);
                    statements(body, vars, bindings);
                }
                _ => panic!("unsupported explicit geometry fixture statement"),
            }
        }
    }
    statements(&mut body, &function.vars, &bindings);
    function.body.insert(
        at,
        Stmt {
            id: None,
            span: statement.span,
            kind: StmtKind::LoadLoop {
                domain: IterationDomain {
                    view: views[0].clone(),
                    axis: 0,
                },
                offset: Some(offset),
                axes: vec![0; bindings.len()],
                vars: bindings,
                views,
                piece,
                capacity,
                modes: None,
                body,
            },
        },
    );
    LoweredIr {
        name: function.name,
        backend: backend.into(),
        params: function.params,
        index_params: function.index_params,
        vars: function.vars,
        body: function.body,
        shapes: Default::default(),
        selections: vec![],
        decisions: vec![],
        ownership: Default::default(),
        alias_requirements: vec![],
    }
}
const GUARDED: &str = r#"
fn evaluate(x:tensor[8,4] i32,bounds:tensor[3] i32,out:tensor[1] i32):
  controls = load(bounds)
  acc = tile[1] i32
  for i in owned(acc): acc[i] = 0
  for start in range(1):
    chunk = load(x[:controls[0],1:3])
    peer = load(x[controls[2],:controls[1]])
    acc[0] += extent(chunk,0) + extent(peer,0)
    for i in owned(controls): controls[i] = 0
  store(acc,out)
"#;
fn guards(device: &Device, candidate: &Candidate) {
    for capacity in [None, Some(1), Some(3)] {
        let ir = explicit(GUARDED, device.backend(), capacity, 2);
        let data = seismic_lang::demand::data_variables(&ir.body);
        for name in ["chunk", "peer"] {
            let v = ir.vars.iter().position(|v| v.name == name).unwrap();
            assert!(
                !data.contains(&v),
                "{name} has geometry but no element consumer"
            );
        }
        let controls = ir.vars.iter().position(|v| v.name == "controls").unwrap();
        assert!(
            data.contains(&controls),
            "nested endpoint reads retain their elements"
        );
        let mut kernel = device.compile(&ir, candidate.clone()).unwrap();
        let input = device.buffer_from(&encode(&[0; 32])).unwrap();
        let bounds = device.buffer(12).unwrap();
        let result = device.buffer(4).unwrap();
        // Recovery is observable; failed completion need not roll back effects.
        for (values, expected) in [
            ([3, 3, 1], Some(6)),
            ([3, 2, 1], None),
            ([4, 4, 2], Some(8)),
            ([0, 1, 1], None),
            ([0, 0, -1], None),
            ([0, 0, 1], Some(0)),
            ([1, 1, 8], None),
            ([1, 1, 0], Some(2)),
        ] {
            bounds.write(&encode(&values)).unwrap();
            let execution = kernel.execute(&[input.clone(), bounds.clone(), result.clone()], &[]);
            assert_eq!(
                execution.is_ok(),
                expected.is_some(),
                "capacity={capacity:?}, values={values:?}: {execution:?}"
            );
            if let Some(expected) = expected {
                assert_eq!(output(&result), expected);
            }
        }
    }
}
const LAYOUT: &str = r#"
fn evaluate(x:tensor[8,4] i32,out:tensor[1] i32):
  acc = tile[1] i32
  for i in owned(acc): acc[i] = 0
  for start in range(1):
    chunk = load(x[:,1:3])
    acc[0] += extent(chunk,0)
  store(acc,out)
"#;
const NO_STORAGE: &str = r#"
fn evaluate(x:tensor[8192,4] i32):
  for start in range(1):
    chunk = load(x[:,1:3])
    length = extent(chunk,0)
"#;
// Tile reshape is an internal view operation; author-facing reshape accepts
// tensor views. Build the typed internal query after obtaining a checked load.
fn layout_ir(backend: &str) -> LoweredIr {
    let mut ir = explicit(LAYOUT, backend, None, 1);
    let stream = ir
        .body
        .iter_mut()
        .find(|s| matches!(s.kind, StmtKind::LoadLoop { .. }))
        .unwrap();
    let StmtKind::LoadLoop { body, .. } = &mut stream.kind else {
        unreachable!()
    };
    let StmtKind::Assign { value, .. } = &mut body[0].kind else {
        unreachable!()
    };
    let ExprKind::Builtin {
        name: Builtin::Extent,
        args,
    } = &mut value.kind
    else {
        unreachable!()
    };
    let chunk = args[0].clone();
    let mut shape = chunk.ty.shaped().unwrap().clone();
    shape.shape = vec![Sym::constant(16)];
    args[0] = Expr {
        kind: ExprKind::Builtin {
            name: Builtin::Reshape,
            args: vec![chunk],
        },
        ty: Ty::Tile(shape),
        sym: None,
        span: value.span,
    };
    value.sym = Some(Sym::constant(16));
    ir
}
fn layout(device: &Device, candidate: &Candidate) {
    let ir = layout_ir(device.backend());
    let mut kernel = device.compile(&ir, candidate.clone()).unwrap();
    let input = device.buffer_from(&encode(&[0; 32])).unwrap();
    let result = device.buffer(4).unwrap();
    kernel.execute(&[input, result.clone()], &[]).unwrap();
    assert_eq!(output(&result), 16);
}
#[test]
fn cpu_geometry_only_streams_preserve_guards_snapshots_and_owning_layout() {
    for loads in [
        LoadStrategy::Materialize,
        LoadStrategy::BorrowProvenReadOnly,
    ] {
        let device = Device::cpu();
        let candidate = Candidate::Cpu { loads };
        ordinary(&device, &candidate);
        guards(&device, &candidate);
        layout(&device, &candidate);
        for capacity in [None, Some(3)] {
            let ir = explicit(NO_STORAGE, "cpu", capacity, 1);
            let scalar = seismic_compiler::scalar_candidate(
                &ir,
                CallConv::SystemV,
                ScalarOptions {
                    dispatch: Dispatch::Sequential,
                    loads,
                },
            )
            .unwrap();
            assert_eq!(
                scalar.scratch_bytes, 0,
                "capacity={capacity:?}, loads={loads:?}"
            );
        }
    }
}
#[test]
fn cpu_geometry_only_stream_sources_retain_layout_failure() {
    let source = LAYOUT.replace("load(x[:,1:3])", "load(reshape(x[:,1:3],(16,)))");
    let ir = explicit(&source, "cpu", None, 1);
    let Err(error) = seismic_compiler::scalar(&ir, CallConv::SystemV) else {
        panic!("source layout failure disappeared");
    };
    assert!(error.contains("reshape"), "{error}");
}
#[cfg(target_os = "macos")]
#[test]
fn metal_geometry_only_streams_require_no_selected_element_storage() {
    for (capacity, loads) in [None, Some(3)].into_iter().flat_map(|capacity| {
        [
            LoadStrategy::Materialize,
            LoadStrategy::BorrowProvenReadOnly,
        ]
        .map(|loads| (capacity, loads))
    }) {
        let ir = explicit(NO_STORAGE, "metal", capacity, 1);
        let execution = seismic_metal::execution::prepare_storage_selected(
            &ir,
            seismic_metal::execution::Config {
                loads,
                ..Default::default()
            },
            &mut |decision| panic!("geometry-only stream requested placement: {decision:?}"),
        )
        .unwrap();
        let emitted = seismic_metal::msl::emit_execution(&execution).unwrap();
        assert!(
            emitted
                .launches
                .iter()
                .all(|launch| launch.tiles.is_empty())
        );
    }
    let source = LAYOUT.replace("load(x[:,1:3])", "load(reshape(x[:,1:3],(16,)))");
    let ir = explicit(&source, "metal", None, 1);
    let execution = seismic_metal::execution::prepare_storage_selected(
        &ir,
        Default::default(),
        &mut |decision| Ok(decision.alternatives[0].clone()),
    );
    let error = match execution {
        Ok(execution) => seismic_metal::msl::emit_execution(&execution).unwrap_err(),
        Err(error) => error,
    };
    assert!(error.contains("reshape"), "{error}");
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_geometry_only_streams_execute() {
    let device = Device::metal().unwrap();
    for loads in [
        LoadStrategy::Materialize,
        LoadStrategy::BorrowProvenReadOnly,
    ] {
        let candidate = Candidate::Metal(seismic_metal::execution::Config {
            loads,
            ..Default::default()
        });
        ordinary(&device, &candidate);
        guards(&device, &candidate);
        layout(&device, &candidate);
    }
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_geometry_only_streams_execute() {
    for loads in [
        LoadStrategy::Materialize,
        LoadStrategy::BorrowProvenReadOnly,
    ] {
        let device = Device::cuda(0).unwrap();
        let candidate = Candidate::Cuda {
            options: ScalarOptions {
                dispatch: Dispatch::Sequential,
                loads,
            },
            threads_per_block: 32,
        };
        ordinary(&device, &candidate);
        guards(&device, &candidate);
        layout(&device, &candidate);
    }
}
