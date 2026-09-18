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
