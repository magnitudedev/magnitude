use seismic_lang::{
    ir::{ExprKind, StmtKind},
    lower::{lower, Options},
    normalize,
    program::{compile, SourceFile},
    types::{DType, Ty},
    verify::{self, Stage},
    Scope,
};
use std::collections::HashMap;

fn function() -> seismic_lang::lowered_ir::LoweredIr {
    let source = SourceFile {
        path: "verify.seismic.portable".into(), scope: Scope::Portable,
        text: "fn copy(x: tensor[4] f32, out: tensor[4] f32):\n  value = load(x)\n  store(value, out)\n".into(),
    };
    let program = compile(&[source], &["cpu".into()]).unwrap();
    lower(&program, "copy", "cpu", &HashMap::new()).unwrap()
}
#[test]
fn stage_boundary_rejects_lost_scope_type_and_snapshot_provenance() {
    let source = function();
    verify::lowered(&source, Stage::Expanded).unwrap();
    let mut escaped = source.clone();
    escaped.body.remove(0);
    assert!(verify::lowered(&escaped, Stage::Expanded)
        .unwrap_err()
        .contains("scope"));
    let mut mistyped = source.clone();
    let StmtKind::Assign { value, .. } = &mut mistyped.body[0].kind else {
        panic!()
    };
    let ExprKind::Builtin { args, .. } = &mut value.kind else {
        panic!()
    };
    args[0].ty = Ty::Scalar(DType::I32);
    assert!(verify::lowered(&mistyped, Stage::Expanded)
        .unwrap_err()
        .contains("binding has"));
    assert!(verify::lowered(&source, Stage::Executable)
        .unwrap_err()
        .contains("unresolved"));
    let mut selected = source;
    normalize::select_loads(&mut selected.body, true);
    verify::lowered(&selected, Stage::Executable).unwrap();
    // Publication to the source backing after capture invalidates borrowing,
    // even though every retained type and variable identity remains valid.
    let mut write = selected.body[1].clone();
    let StmtKind::Expr(expr) = &mut write.kind else {
        panic!()
    };
    let ExprKind::Builtin { args, .. } = &mut expr.kind else {
        panic!()
    };
    let StmtKind::Assign { value, .. } = &selected.body[0].kind else {
        panic!()
    };
    let ExprKind::Load { view, .. } = &value.kind else {
        panic!()
    };
    args[1] = (**view).clone();
    selected.body.insert(1, write);
    let StmtKind::Assign { value, .. } = &mut selected.body[0].kind else {
        panic!()
    };
    let ExprKind::Load { mode, .. } = &mut value.kind else {
        panic!()
    };
    *mode = seismic_lang::ir::LoadMode::Borrow;
    assert!(verify::lowered(&selected, Stage::Executable)
        .unwrap_err()
        .contains("borrowing lifetime"));
}
#[test]
fn stage_boundary_keeps_assignment_conversion_and_shape_contracts() {
    let source = SourceFile { path: "conversion.seismic.portable".into(), scope: Scope::Portable,
        text: "fn convert(x: tensor[4] f32, out: tensor[4] bf16):\n  value = load(x)\n  store(value, out)\n".into() };
    let program = compile(&[source], &["cpu".into()]).unwrap();
    let mut lowered = seismic_lang::lower::lower_with(
        &program,
        "convert",
        "cpu",
        &HashMap::new(),
        &Options::default(),
    )
    .unwrap();
    normalize::select_loads(&mut lowered.body, false);
    verify::lowered(&lowered, Stage::Executable).unwrap();
    let StmtKind::Expr(expr) = &mut lowered.body[1].kind else {
        panic!()
    };
    let ExprKind::Builtin { args, .. } = &mut expr.kind else {
        panic!()
    };
    let Ty::Tile(tile) = &mut args[0].ty else {
        panic!()
    };
    tile.shape[0] = seismic_lang::sym::Sym::constant(3);
    assert!(verify::lowered(&lowered, Stage::Executable).is_err());
}

#[test]
fn scalar_reduction_views_keep_zero_rank_storage_distinct_from_scalar_reads() {
    let program = compile(&[SourceFile {
        path: "scalar_reduction.seismic.portable".into(), scope: Scope::Portable,
        text: "fn sum(x: tensor[3] f32, out: tensor[1] f32):\n  a = load(x)\n  y = tile[1] f32\n  for i in owned(y): y[i] = reduce(a,0,sum,ordered=true)\n  store(y,out)\n".into(),
    }], &[]).unwrap();
    let lowered = lower(&program, "sum", "cpu", &HashMap::new()).unwrap();
    let mut expanded = seismic_lang::reduction::structured::materialize(&lowered).unwrap();
    normalize::select_loads(&mut expanded.body, false);
    normalize::bind_values(&mut expanded.body, &mut expanded.vars);
    verify::lowered(&expanded, Stage::Executable).unwrap();

    // Test the view contract independently of reduction materialization: full
    // coordinates may denote a zero-rank tile, but must preserve decoded dtype.
    use seismic_lang::{
        ir::{Expr, Index, Stmt},
        sym::Sym,
        types::Elem,
    };
    let mut source = function();
    normalize::select_loads(&mut source.body, false);
    let (id, variable) = source
        .vars
        .iter()
        .enumerate()
        .find(|(_, v)| v.name == "value")
        .unwrap();
    let span = variable.span;
    let mut shape = variable.ty.shaped().unwrap().clone();
    shape.shape.clear();
    let view = Expr {
        kind: ExprKind::Index {
            base: Box::new(Expr {
                kind: ExprKind::Var(id),
                ty: variable.ty.clone(),
                sym: None,
                span,
            }),
            indices: vec![Index::Point(Expr {
                kind: ExprKind::Int(0),
                ty: Ty::Scalar(DType::I32),
                sym: Some(Sym::constant(0)),
                span,
            })],
        },
        ty: Ty::Tile(shape),
        sym: None,
        span,
    };
    source.body.push(Stmt {
        id: None,
        kind: StmtKind::Expr(view),
        span,
    });
    verify::lowered(&source, Stage::Executable).unwrap();
    let StmtKind::Expr(view) = &mut source.body.last_mut().unwrap().kind else {
        unreachable!()
    };
    let Ty::Tile(shape) = &mut view.ty else {
        unreachable!()
    };
    shape.elem = Elem::Dtype(DType::I32);
    assert!(verify::lowered(&source, Stage::Executable)
        .unwrap_err()
        .contains("indexed scalar type"));
}

#[test]
fn generic_scalar_indices_follow_bound_storage_types_through_nested_calls() {
    use seismic_lang::{
        lower::{
            alternatives::{Space, Specialization},
            lower_specialized,
        },
        types::Elem,
    };
    let program = compile(&[SourceFile {
        path: "generic_index.seismic.portable".into(), scope: Scope::Portable,
        text: "fn leaf[N](x: tensor[N] T, out: tensor[N] U):\n  a = load(x)\n  y = tile[N] U\n  for i in owned(y): y[i] = f32(a[i]) * 2.0\n  store(y,out)\nfn entry[N](x: tensor[N] T, out: tensor[N] U):\n  leaf(x,out)\n".into(),
    }], &[]).unwrap();
    let shapes = HashMap::from([("N".into(), 3)]);
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        for entry in ["leaf", "entry"] {
            let elements = HashMap::from([
                ("T".into(), Elem::Dtype(dtype)),
                ("U".into(), Elem::Dtype(dtype)),
            ]);
            let options = Options::default();
            let baseline =
                lower_specialized(&program, entry, "cpu", &shapes, &elements, &options).unwrap();
            verify::lowered(&baseline, Stage::Expanded).unwrap();
            let mut count = 0;
            let mut space = Space::new(Specialization {
                program: &program,
                entry,
                backend: "cpu",
                shapes: &shapes,
                elements: &elements,
                options: &options,
            });
            for attempt in space.by_ref() {
                let function = attempt
                    .result
                    .unwrap_or_else(|error| panic!("{dtype:?} {entry}: {error}"));
                verify::lowered(&function, Stage::Expanded).unwrap();
                count += 1;
            }
            assert!(space.exhausted());
            assert!(count > 0);
        }
    }
}
