//! Internal LoadLoop fixtures. Source programs contain logical loads; this
//! factory explicitly constructs the execution IR exercised by split tests.
use seismic_lang::{
    Scope,
    ir::*,
    lowered_ir::LoweredIr,
    program::{SourceFile, compile},
    sym::{Atom, Sym},
    types::Ty,
};

/// Checked common IR before portable normalization, for tests of a particular
/// internal primitive's execution realization rather than source selection.
#[allow(dead_code)]
pub fn common_ir(program: &seismic_lang::program::Program) -> LoweredIr {
    let f = program
        .functions
        .iter()
        .find(|f| f.name == "evaluate")
        .unwrap();
    assert!(f.shape_params.is_empty() && f.elem_params.is_empty());
    LoweredIr {
        name: f.name.clone(),
        backend: "metal".into(),
        ownership: Default::default(),
        alias_requirements: Vec::new(),
        params: f.params.clone(),
        index_params: f.index_params.clone(),
        vars: f.vars.clone(),
        body: f.body.clone(),
        shapes: Default::default(),
        selections: vec![],
        decisions: vec![],
    }
}
#[allow(dead_code)]
pub fn checked(text: &str) -> LoweredIr {
    let p = compile(
        &[SourceFile {
            path: "internal_fixture.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    let f = p
        .functions
        .into_iter()
        .find(|f| f.name == "evaluate")
        .unwrap();
    assert!(f.shape_params.is_empty() && f.elem_params.is_empty());
    LoweredIr {
        name: f.name,
        backend: "metal".into(),
        ownership: Default::default(),
        alias_requirements: Vec::new(),
        params: f.params,
        index_params: f.index_params,
        vars: f.vars,
        body: f.body,
        shapes: Default::default(),
        selections: vec![],
        decisions: vec![],
    }
}

#[allow(dead_code)]
pub fn streamed(text: &str, piece: i64, names: &[&str]) -> LoweredIr {
    assert!(piece > 0);
    let p = compile(
        &[SourceFile {
            path: "internal_stream_fixture.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    let mut f = p
        .functions
        .into_iter()
        .find(|f| f.name == "evaluate")
        .unwrap();
    assert!(f.shape_params.is_empty() && f.elem_params.is_empty());
    let mut found = 0;
    fn block(
        body: &mut Vec<Stmt>,
        vars: &mut [Var],
        piece: i64,
        names: &[&str],
        found: &mut usize,
    ) {
        let mut i = 0;
        while i < body.len() {
            let selected = |s: &Stmt| match &s.kind {
                StmtKind::Assign {
                    target:
                        Expr {
                            kind: ExprKind::Var(v),
                            ..
                        },
                    value:
                        Expr {
                            kind:
                                ExprKind::Builtin {
                                    name: Builtin::Load,
                                    ..
                                },
                            ..
                        },
                    ..
                } => names.contains(&vars[*v].name.as_str()),
                _ => false,
            };
            if selected(&body[i]) {
                let mut count = 1;
                while i + count < body.len() && selected(&body[i + count]) {
                    count += 1;
                }
                let assignments = body.drain(i..i + count).collect::<Vec<_>>();
                let mut updates = body.drain(i..i + count).collect::<Vec<_>>();
                let span = assignments[0].span;
                let atom = Atom::Param(format!("$fixture_piece_{}", *found));
                *found += 1;
                let mut bindings = Vec::new();
                let mut views = Vec::new();
                for s in assignments {
                    let StmtKind::Assign { target, value, .. } = s.kind else {
                        unreachable!()
                    };
                    let ExprKind::Var(v) = target.kind else {
                        unreachable!()
                    };
                    let ExprKind::Builtin { args, .. } = value.kind else {
                        unreachable!()
                    };
                    let Ty::Tile(t) = &mut vars[v].ty else {
                        unreachable!()
                    };
                    t.shape[0] = Sym::atom(atom.clone());
                    bindings.push(v);
                    views.push(args[0].clone());
                }
                fn expr(e: &mut Expr, vars: &[Var], bindings: &[usize]) {
                    match &mut e.kind {
                        ExprKind::Var(v) if bindings.contains(v) => e.ty = vars[*v].ty.clone(),
                        ExprKind::Index { base, indices } => {
                            expr(base, vars, bindings);
                            for i in indices {
                                match i {
                                    Index::Point(e) => expr(e, vars, bindings),
                                    Index::Slice { start, end } => {
                                        for e in start.iter_mut().chain(end.iter_mut()) {
                                            expr(e, vars, bindings);
                                        }
                                    }
                                }
                            }
                        }
                        ExprKind::Unary { expr: e, .. }
                        | ExprKind::Cast { expr: e, .. }
                        | ExprKind::Transpose(e)
                        | ExprKind::Accessor { base: e, .. } => expr(e, vars, bindings),
                        ExprKind::Binary { lhs, rhs, .. } => {
                            expr(lhs, vars, bindings);
                            expr(rhs, vars, bindings);
                        }
                        ExprKind::Builtin { args, .. }
                        | ExprKind::Intrinsic { args, .. }
                        | ExprKind::Call { args, .. }
                        | ExprKind::Tuple(args) => {
                            for a in args {
                                expr(a, vars, bindings);
                            }
                        }
                        _ => {}
                    }
                }
                for s in &mut updates {
                    let StmtKind::Assign { target, value, .. } = &mut s.kind else {
                        panic!("internal stream fixture must supply one scalar update per binding")
                    };
                    expr(target, vars, &bindings);
                    expr(value, vars, &bindings);
                }
                body.insert(
                    i,
                    Stmt {
                        id: None,
                        span,
                        kind: StmtKind::LoadLoop {
                            offset: None,
                            domain: IterationDomain {
                                view: views[0].clone(),
                                axis: 0,
                            },
                            axes: vec![0; bindings.len()],
                            vars: bindings,
                            views,
                            piece: atom,
                            capacity: Some(piece),
                            modes: None,
                            body: updates,
                        },
                    },
                );
            } else {
                match &mut body[i].kind {
                    StmtKind::Parallel { body, .. }
                    | StmtKind::Owned { body, .. }
                    | StmtKind::Range { body, .. } => block(body, vars, piece, names, found),
                    StmtKind::If { then, els, .. } => {
                        block(then, vars, piece, names, found);
                        block(els, vars, piece, names, found);
                    }
                    _ => {}
                }
            }
            i += 1;
        }
    }
    block(&mut f.body, &mut f.vars, piece, names, &mut found);
    assert!(found > 0);
    LoweredIr {
        name: f.name,
        backend: "metal".into(),
        ownership: Default::default(),
        alias_requirements: Vec::new(),
        params: f.params,
        index_params: f.index_params,
        vars: f.vars,
        body: f.body,
        shapes: Default::default(),
        selections: vec![],
        decisions: vec![],
    }
}
