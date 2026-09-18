//! Covering several work items with one, by widening what depends on the index.
//!
//! A `parallel` index names independent work. When one item covers `factor` values of that
//! index, the naive realization repeats the whole body once per value, which shares nothing
//! and costs parallelism. The useful realization instead notes that a body divides in two:
//! the part that depends on the index, and the part that does not. The independent part is
//! computed once. The dependent part is widened, so a tile that held one value now holds
//! `factor`, and every read of the index becomes a read of that item's `factor` values.
//!
//! This is what makes one item covering four output rows read the shared activation once
//! rather than four times, which a statement-moving rewrite cannot express.

use crate::ir::{Expr, ExprKind, Index, Stmt, StmtKind, Var, VarId, VarKind};
use crate::sym::{Atom, Sym};
use crate::types::{Shaped, Ty};
use std::collections::HashSet;

/// How a body divides when one item covers several values of an index.
#[derive(Clone, Debug)]
pub struct Widening {
    /// Statements to emit once, in order, with the index unbound.
    pub shared: Vec<usize>,
    /// Statements to emit widened, in order.
    pub widened: Vec<usize>,
    /// Variables whose value now holds `factor` of what it held, one per covered index.
    pub wide_vars: Vec<VarId>,
    pub factor: i64,
}

/// Divide a `parallel` body for an item covering `factor` values of `inner`.
///
/// A statement is shared when it neither reads the index nor reads anything a widened
/// statement wrote. Everything else is widened, and every variable a widened statement writes
/// becomes wide. The two sets partition the body, so nothing is duplicated or lost.
pub fn plan(body: &[Stmt], inner: VarId, atom: &Atom, factor: i64) -> Widening {
    plan_at(body, inner, atom, factor)
}

/// The same division applied inside compound statements. A loop whose views mix an
/// index-dependent operand with shared ones is itself dependent, but its body still divides:
/// the statements reading only the shared operands are computed once per iteration rather
/// than once per covered index. Recursing is what lets the shared activation stream be read
/// once while the weight reads widen.
pub fn plan_at(body: &[Stmt], inner: VarId, atom: &Atom, factor: i64) -> Widening {
    let writes_of: Vec<HashSet<VarId>> = body
        .iter()
        .map(|s| {
            let mut w = HashSet::new();
            crate::rewrite::writes(s, &mut w);
            w
        })
        .collect();
    plan_with_writes(body, inner, atom, factor, &writes_of)
}

/// Reuse the same dependency closure when a caller has checked write effects
/// for retained operations that have not yet expanded into assignments.
pub(crate) fn plan_with_writes(
    body: &[Stmt],
    inner: VarId,
    atom: &Atom,
    factor: i64,
    writes_of: &[HashSet<VarId>],
) -> Widening {
    assert_eq!(body.len(), writes_of.len());
    // Grow the dependent set to a fixpoint: a statement is dependent if it reads the index or
    // any value a dependent statement produced.
    let mut dependent = vec![false; body.len()];
    let mut tainted: HashSet<VarId> = HashSet::new();
    tainted.insert(inner);
    loop {
        let mut changed = false;
        for (i, s) in body.iter().enumerate() {
            if dependent[i] {
                continue;
            }
            if crate::rewrite::depends(s, &tainted, atom) || crate::effects::tensor_effect(s) {
                dependent[i] = true;
                for v in &writes_of[i] {
                    tainted.insert(*v);
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // A variable written by a dependent statement holds one value per covered index.
    let mut wide_vars: Vec<VarId> = Vec::new();
    for (i, _) in body.iter().enumerate() {
        if dependent[i] {
            for v in &writes_of[i] {
                if !wide_vars.contains(v) {
                    wide_vars.push(*v);
                }
            }
        }
    }
    wide_vars.sort();
    Widening {
        shared: (0..body.len()).filter(|i| !dependent[*i]).collect(),
        widened: (0..body.len()).filter(|i| dependent[*i]).collect(),
        wide_vars,
        factor,
    }
}

/// Rewrite a `parallel` body so one item covers `factor` values of `inner`.
///
/// Every statement that depends on the index is replicated `factor` times, with the index
/// replaced by each covered value and the variables it writes given one copy per value.
/// Statements that do not depend on the index are left alone, so their work is done once.
/// This runs at every level, so a loop that mixes a shared operand with an index-dependent
/// one keeps its shared reads single while the dependent reads multiply.
pub fn apply(
    body: &[Stmt],
    inner: VarId,
    atom: &Atom,
    factor: i64,
    vars: &mut Vec<Var>,
    base: &Expr,
) -> Vec<Stmt> {
    let mut copies: Vec<std::collections::HashMap<VarId, VarId>> =
        vec![std::collections::HashMap::new(); factor as usize];
    rewrite_block(body, inner, atom, factor, vars, base, &mut copies)
}

/// Cover a bounded axis, including a final partial work item. Full items retain
/// shared producers. The partial item guards each complete original occurrence,
/// so no invalid address, effect or collective is evaluated for a padded output.
pub fn apply_bounded(
    body: &[Stmt],
    inner: VarId,
    atom: &Atom,
    factor: i64,
    extent: i64,
    vars: &mut Vec<Var>,
    base: &Expr,
) -> Vec<Stmt> {
    if extent % factor == 0 {
        return apply(body, inner, atom, factor, vars, base);
    }
    use crate::{ast::BinaryOp, types::DType};
    let literal = |n| Expr {
        kind: ExprKind::Int(n),
        ty: Ty::Scalar(DType::I32),
        sym: Some(Sym::constant(n)),
        span: base.span,
    };
    let below = |value: Expr, n: i64| Expr {
        kind: ExprKind::Binary {
            op: BinaryOp::Lt,
            lhs: Box::new(value),
            rhs: Box::new(literal(n)),
        },
        ty: Ty::Scalar(DType::Bool),
        sym: None,
        span: base.span,
    };
    let full = apply(body, inner, atom, factor, vars, base);
    let mut tail = Vec::new();
    for k in 0..factor {
        let mut copies = std::collections::HashMap::new();
        let value = offset(base, k);
        let occurrence = body
            .iter()
            .map(|s| substitute(s, inner, atom, &value, vars, &mut copies))
            .collect();
        // Compare the base against extent-k before evaluating base+k. This
        // preserves the checked source index width even near its maximum.
        tail.push(Stmt {
            id: None,
            kind: StmtKind::If {
                cond: below(base.clone(), extent - k),
                then: occurrence,
                els: Vec::new(),
            },
            span: base.span,
        });
    }
    vec![Stmt {
        id: None,
        kind: StmtKind::If {
            cond: below(base.clone(), extent - factor + 1),
            then: full,
            els: tail,
        },
        span: base.span,
    }]
}

fn rewrite_block(
    body: &[Stmt],
    inner: VarId,
    atom: &Atom,
    factor: i64,
    vars: &mut Vec<Var>,
    base: &Expr,
    copies: &mut [std::collections::HashMap<VarId, VarId>],
) -> Vec<Stmt> {
    // What the index-dependent statements of this block write, so reads of those variables
    // are redirected to the copy belonging to each covered value.
    let mut tainted: HashSet<VarId> = HashSet::new();
    tainted.insert(inner);
    loop {
        let mut changed = false;
        for s in body {
            if crate::rewrite::depends(s, &tainted, atom) || crate::effects::tensor_effect(s) {
                let mut w = HashSet::new();
                crate::rewrite::writes(s, &mut w);
                for v in w {
                    if tainted.insert(v) {
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut out = Vec::new();
    for s in body {
        if !crate::rewrite::depends(s, &tainted, atom) && !crate::effects::tensor_effect(s) {
            out.push(s.clone());
            continue;
        }
        // A compound statement whose own operands are shared divides internally instead of
        // being replicated whole.
        if let Some(inner_stmt) = descend(s, inner, atom, factor, vars, base, copies) {
            out.push(inner_stmt);
            continue;
        }
        for k in 0..factor as usize {
            let index_value = offset(base, k as i64);
            out.push(substitute(
                s,
                inner,
                atom,
                &index_value,
                vars,
                &mut copies[k],
            ));
        }
    }
    out
}

/// If a compound statement's own operands do not depend on the index, rewrite its body
/// instead of replicating the statement.
fn descend(
    s: &Stmt,
    inner: VarId,
    atom: &Atom,
    factor: i64,
    vars: &mut Vec<Var>,
    base: &Expr,
    copies: &mut [std::collections::HashMap<VarId, VarId>],
) -> Option<Stmt> {
    let mut shallow = HashSet::new();
    shallow.insert(inner);
    let rewritten = match &s.kind {
        StmtKind::LoadLoop {
            offset,
            vars: lv,
            views,
            domain,
            axes,
            piece,
            capacity,
            modes,
            body,
        } => {
            // The loop itself is shared only when no view reads the index.
            if std::iter::once(&domain.view)
                .chain(views)
                .any(|v| expr_depends_shallow(v, &shallow, atom))
            {
                return None;
            }
            StmtKind::LoadLoop {
                offset: *offset,
                modes: modes.clone(),
                vars: lv.clone(),
                views: views.clone(),
                domain: domain.clone(),
                axes: axes.clone(),
                piece: piece.clone(),
                capacity: *capacity,
                body: rewrite_block(body, inner, atom, factor, vars, base, copies),
            }
        }
        StmtKind::Range { var, lo, hi, body } => {
            if lo.atoms().contains(atom) || hi.atoms().contains(atom) {
                return None;
            }
            StmtKind::Range {
                var: *var,
                lo: lo.clone(),
                hi: hi.clone(),
                body: rewrite_block(body, inner, atom, factor, vars, base, copies),
            }
        }
        _ => return None,
    };
    Some(Stmt {
        id: None,
        kind: rewritten,
        span: s.span,
    })
}

/// Whether an expression reads the index directly, without following variables.
fn expr_depends_shallow(e: &Expr, tainted: &HashSet<VarId>, atom: &Atom) -> bool {
    let mut st: Vec<&Expr> = Vec::new();
    st.push(e);
    while let Some(x) = st.pop() {
        if let Some(sym) = &x.sym {
            if sym.atoms().contains(atom) {
                return true;
            }
        }
        match &x.kind {
            ExprKind::Var(v) if tainted.contains(v) => return true,
            ExprKind::Index { base, indices } => {
                st.push(base);
                for i in indices {
                    match i {
                        Index::Point(p) => st.push(p),
                        Index::Slice { start, end } => {
                            if let Some(a) = start {
                                st.push(a)
                            }
                            if let Some(b) = end {
                                st.push(b)
                            }
                        }
                    }
                }
            }
            ExprKind::Load { view: x2, .. }
            | ExprKind::Transpose(x2)
            | ExprKind::Accessor { base: x2, .. }
            | ExprKind::Lanes { base: x2, .. }
            | ExprKind::Unary { expr: x2, .. }
            | ExprKind::Cast { expr: x2, .. } => st.push(x2),
            ExprKind::Builtin { args, .. }
            | ExprKind::Intrinsic { args, .. }
            | ExprKind::Call { args, .. }
            | ExprKind::Tuple(args) => st.extend(args.iter()),
            ExprKind::Binary { lhs, rhs, .. } => {
                st.push(lhs);
                st.push(rhs);
            }
            _ => {}
        }
    }
    false
}

/// `base + k`, the k-th index this item covers.
fn offset(base: &Expr, k: i64) -> Expr {
    if k == 0 {
        return base.clone();
    }
    let n = Expr {
        kind: ExprKind::Int(k),
        ty: base.ty.clone(),
        sym: Some(Sym::constant(k)),
        span: base.span,
    };
    let sym = base.sym.as_ref().map(|s| s.add(&Sym::constant(k)));
    Expr {
        kind: ExprKind::Binary {
            op: crate::ast::BinaryOp::Add,
            lhs: Box::new(base.clone()),
            rhs: Box::new(n),
        },
        ty: base.ty.clone(),
        sym,
        span: base.span,
    }
}

/// One covered value's copy of a statement: the index takes that value, and each variable the
/// statement writes gets its own copy so the values stay live together.
fn substitute(
    s: &Stmt,
    inner: VarId,
    atom: &Atom,
    value: &Expr,
    vars: &mut Vec<Var>,
    copy: &mut std::collections::HashMap<VarId, VarId>,
) -> Stmt {
    let mut w = HashSet::new();
    crate::rewrite::writes(s, &mut w);
    // Variables a loop inside this statement binds are defined by that loop, so each copy of
    // the statement needs its own, exactly as it needs its own copy of what it writes.
    bound_vars(s, &mut w);
    let mut ordered: Vec<VarId> = w.into_iter().collect();
    ordered.sort();
    for v in ordered {
        if v == inner || copy.contains_key(&v) {
            continue;
        }
        let fresh = vars.len();
        let mut var = vars[v].clone();
        var.name = format!("{}#{}", var.name, fresh);
        vars.push(var);
        copy.insert(v, fresh);
    }
    map_stmt(s, inner, atom, value, copy)
}

pub(crate) fn copy_bindings(
    body: &[Stmt],
    vars: &mut Vec<Var>,
) -> (Vec<Stmt>, std::collections::HashMap<VarId, VarId>) {
    let mut copies = std::collections::HashMap::new();
    let atom = Atom::Param(format!("copy#{}", vars.len()));
    let Some(first) = body.first() else {
        return (Vec::new(), copies);
    };
    let value = Expr {
        kind: ExprKind::Int(0),
        ty: Ty::Scalar(crate::types::DType::I32),
        sym: Some(Sym::constant(0)),
        span: first.span,
    };
    let result = body
        .iter()
        .map(|s| substitute(s, usize::MAX, &atom, &value, vars, &mut copies))
        .collect();
    (result, copies)
}

/// Variables bound by loops inside a statement.
fn bound_vars(s: &Stmt, out: &mut HashSet<VarId>) {
    match &s.kind {
        StmtKind::Reduction(r) => {
            for merge in r.implementations() {
                for p in merge.left.iter().chain(&merge.right).chain(&merge.output) {
                    if let ExprKind::Var(v) = p.kind {
                        out.insert(v);
                    }
                }
                for s in &merge.body {
                    bound_vars(s, out);
                }
            }
        }
        StmtKind::LoadLoop {
            vars, offset, body, ..
        } => {
            if let Some(v) = offset {
                out.insert(*v);
            }
            for v in vars {
                out.insert(*v);
            }
            for b in body {
                bound_vars(b, out);
            }
        }
        StmtKind::Owned { body, .. } => {
            // An `owned` loop's indices address a tile's own elements, so every copy uses the
            // same ones; renaming them would break the match between a distributed tile and
            // the loop that owns it.
            for b in body {
                bound_vars(b, out);
            }
        }
        StmtKind::Range { var, body, .. } | StmtKind::Lanes { var, body, .. } => {
            out.insert(*var);
            for b in body {
                bound_vars(b, out);
            }
        }
        StmtKind::If { then, els, .. } => {
            for b in then.iter().chain(els.iter()) {
                bound_vars(b, out);
            }
        }
        StmtKind::Parallel { body, .. } => {
            for b in body {
                bound_vars(b, out);
            }
        }
        StmtKind::Assign { .. } | StmtKind::Expr(_) => {}
    }
}

fn map_stmt(
    s: &Stmt,
    inner: VarId,
    atom: &Atom,
    value: &Expr,
    copy: &std::collections::HashMap<VarId, VarId>,
) -> Stmt {
    let kind = match &s.kind {
        StmtKind::Reduction(r) => {
            let mut r = r.clone();
            for e in r.operands_mut() {
                *e = map_expr(e, inner, atom, value, copy);
            }
            r.merge = map_expr(&r.merge, inner, atom, value, copy);
            if let Some(step) = &mut r.step {
                step.call = map_expr(&step.call, inner, atom, value, copy);
            }
            for merge in r.implementations_mut() {
                for e in merge
                    .left
                    .iter_mut()
                    .chain(&mut merge.right)
                    .chain(&mut merge.output)
                {
                    *e = map_expr(e, inner, atom, value, copy);
                }
                merge.body = merge
                    .body
                    .iter()
                    .map(|s| map_stmt(s, inner, atom, value, copy))
                    .collect();
            }
            StmtKind::Reduction(r)
        }
        StmtKind::Assign {
            target,
            op,
            value: v,
        } => StmtKind::Assign {
            target: map_expr(target, inner, atom, value, copy),
            op: *op,
            value: map_expr(v, inner, atom, value, copy),
        },
        StmtKind::Expr(e) => StmtKind::Expr(map_expr(e, inner, atom, value, copy)),
        StmtKind::Owned {
            vars: ov,
            tile,
            body,
        } => StmtKind::Owned {
            vars: ov.clone(),
            tile: map_expr(tile, inner, atom, value, copy),
            body: body
                .iter()
                .map(|b| map_stmt(b, inner, atom, value, copy))
                .collect(),
        },
        StmtKind::Range { var, lo, hi, body } => StmtKind::Range {
            var: copy.get(var).copied().unwrap_or(*var),
            lo: map_sym(lo, atom, value),
            hi: map_sym(hi, atom, value),
            body: body
                .iter()
                .map(|b| map_stmt(b, inner, atom, value, copy))
                .collect(),
        },
        StmtKind::Lanes {
            var,
            extent,
            width,
            body,
        } => StmtKind::Lanes {
            var: copy.get(var).copied().unwrap_or(*var),
            extent: map_sym(extent, atom, value),
            width: *width,
            body: body
                .iter()
                .map(|b| map_stmt(b, inner, atom, value, copy))
                .collect(),
        },
        StmtKind::LoadLoop {
            offset,
            vars: lv,
            views,
            domain,
            axes,
            piece,
            capacity,
            modes,
            body,
        } => StmtKind::LoadLoop {
            offset: offset.map(|v| copy.get(&v).copied().unwrap_or(v)),
            modes: modes.clone(),
            vars: lv
                .iter()
                .map(|v| copy.get(v).copied().unwrap_or(*v))
                .collect(),
            views: views
                .iter()
                .map(|v| map_expr(v, inner, atom, value, copy))
                .collect(),
            domain: crate::ir::IterationDomain {
                view: map_expr(&domain.view, inner, atom, value, copy),
                axis: domain.axis,
            },
            axes: axes.clone(),
            piece: piece.clone(),
            capacity: *capacity,
            body: body
                .iter()
                .map(|b| map_stmt(b, inner, atom, value, copy))
                .collect(),
        },
        StmtKind::If { cond, then, els } => StmtKind::If {
            cond: map_expr(cond, inner, atom, value, copy),
            then: then
                .iter()
                .map(|b| map_stmt(b, inner, atom, value, copy))
                .collect(),
            els: els
                .iter()
                .map(|b| map_stmt(b, inner, atom, value, copy))
                .collect(),
        },
        StmtKind::Parallel { .. } => s.kind.clone(),
    };
    Stmt {
        id: None,
        kind,
        span: s.span,
    }
}

/// Replace an index's expression and symbolic uses while preserving binders.
/// The replacement has a symbolic value, so control bounds and element addresses
/// observe the same coordinate as ordinary scalar arithmetic.
pub(crate) fn replace_index(body: &mut [Stmt], index: VarId, atom: &Atom, value: &Expr) {
    debug_assert!(value.sym.is_some());
    let copy = std::collections::HashMap::new();
    for statement in body {
        *statement = map_stmt(statement, index, atom, value, &copy);
    }
}

fn map_sym(s: &Sym, atom: &Atom, value: &Expr) -> Sym {
    match &value.sym {
        Some(v) if s.atoms().contains(atom) => s.subst(atom, v),
        _ => s.clone(),
    }
}

fn map_expr(
    e: &Expr,
    inner: VarId,
    atom: &Atom,
    value: &Expr,
    copy: &std::collections::HashMap<VarId, VarId>,
) -> Expr {
    if matches!(e.kind, ExprKind::Var(v) if v == inner) {
        return value.clone();
    }
    let sym = e.sym.as_ref().map(|s| map_sym(s, atom, value));
    let kind = match &e.kind {
        ExprKind::Var(v) => ExprKind::Var(copy.get(v).copied().unwrap_or(*v)),
        ExprKind::Index { base, indices } => ExprKind::Index {
            base: Box::new(map_expr(base, inner, atom, value, copy)),
            indices: indices
                .iter()
                .map(|i| match i {
                    Index::Point(p) => Index::Point(map_expr(p, inner, atom, value, copy)),
                    Index::Slice { start, end } => Index::Slice {
                        start: start
                            .as_ref()
                            .map(|x| map_expr(x, inner, atom, value, copy)),
                        end: end.as_ref().map(|x| map_expr(x, inner, atom, value, copy)),
                    },
                })
                .collect(),
        },
        ExprKind::Load { view, mode } => ExprKind::Load {
            view: Box::new(map_expr(view, inner, atom, value, copy)),
            mode: *mode,
        },
        ExprKind::Transpose(x) => {
            ExprKind::Transpose(Box::new(map_expr(x, inner, atom, value, copy)))
        }
        ExprKind::Accessor { base, name } => ExprKind::Accessor {
            base: Box::new(map_expr(base, inner, atom, value, copy)),
            name: name.clone(),
        },
        ExprKind::Lanes { base, extent } => ExprKind::Lanes {
            base: Box::new(map_expr(base, inner, atom, value, copy)),
            extent: map_sym(extent, atom, value),
        },
        ExprKind::Builtin { name, args } => ExprKind::Builtin {
            name: *name,
            args: args
                .iter()
                .map(|a| map_expr(a, inner, atom, value, copy))
                .collect(),
        },
        ExprKind::Intrinsic { op: name, args } => ExprKind::Intrinsic {
            op: *name,
            args: args
                .iter()
                .map(|a| map_expr(a, inner, atom, value, copy))
                .collect(),
        },
        ExprKind::Call {
            callee,
            shape_args,
            elem_args,
            args,
        } => ExprKind::Call {
            callee: callee.clone(),
            shape_args: shape_args.iter().map(|s| map_sym(s, atom, value)).collect(),
            elem_args: elem_args.clone(),
            args: args
                .iter()
                .map(|a| map_expr(a, inner, atom, value, copy))
                .collect(),
        },
        ExprKind::Tuple(items) => ExprKind::Tuple(
            items
                .iter()
                .map(|a| map_expr(a, inner, atom, value, copy))
                .collect(),
        ),
        ExprKind::Unary { op, expr } => ExprKind::Unary {
            op: *op,
            expr: Box::new(map_expr(expr, inner, atom, value, copy)),
        },
        ExprKind::Cast { dtype, expr } => ExprKind::Cast {
            dtype: *dtype,
            expr: Box::new(map_expr(expr, inner, atom, value, copy)),
        },
        ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
            op: *op,
            lhs: Box::new(map_expr(lhs, inner, atom, value, copy)),
            rhs: Box::new(map_expr(rhs, inner, atom, value, copy)),
        },
        ExprKind::TileAlloc { shape, dtype } => ExprKind::TileAlloc {
            shape: shape.iter().map(|s| map_sym(s, atom, value)).collect(),
            dtype: dtype.clone(),
        },
        other => other.clone(),
    };
    Expr {
        kind,
        ty: e.ty.clone(),
        sym,
        span: e.span,
    }
}

/// Widen a tile's type: it holds `factor` of what it held, along a new leading axis.
pub fn widen_ty(ty: &Ty, factor: i64) -> Ty {
    match ty {
        Ty::Tile(s) => {
            let mut shape = vec![Sym::constant(factor)];
            shape.extend(s.shape.iter().cloned());
            Ty::Tile(Shaped {
                shape,
                elem: s.elem.clone(),
                packed_axis: s.packed_axis.map(|a| a + 1),
            })
        }
        // A scalar becomes a tile of `factor` scalars.
        Ty::Scalar(d) => Ty::Tile(Shaped::new(
            vec![Sym::constant(factor)],
            crate::types::Elem::Dtype(*d),
        )),
        other => other.clone(),
    }
}

/// Every variable a body reads, for deciding what must be available outside the widened part.
pub fn reads(s: &Stmt, out: &mut HashSet<VarId>) {
    fn expr(e: &Expr, out: &mut HashSet<VarId>) {
        match &e.kind {
            ExprKind::Var(v) => {
                out.insert(*v);
            }
            ExprKind::Index { base, indices } => {
                expr(base, out);
                for i in indices {
                    match i {
                        Index::Point(p) => expr(p, out),
                        Index::Slice { start, end } => {
                            if let Some(x) = start {
                                expr(x, out)
                            }
                            if let Some(x) = end {
                                expr(x, out)
                            }
                        }
                    }
                }
            }
            ExprKind::Load { view: x, .. }
            | ExprKind::Transpose(x)
            | ExprKind::Accessor { base: x, .. }
            | ExprKind::Lanes { base: x, .. }
            | ExprKind::Unary { expr: x, .. }
            | ExprKind::Cast { expr: x, .. } => expr(x, out),
            ExprKind::Builtin { args, .. }
            | ExprKind::Intrinsic { args, .. }
            | ExprKind::Call { args, .. }
            | ExprKind::Tuple(args) => {
                for a in args {
                    expr(a, out);
                }
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                expr(lhs, out);
                expr(rhs, out);
            }
            _ => {}
        }
    }
    match &s.kind {
        StmtKind::Assign { target, value, .. } => {
            expr(target, out);
            expr(value, out);
        }
        StmtKind::Expr(e) => expr(e, out),
        StmtKind::Reduction(r) => {
            for e in r.operands() {
                expr(e, out);
            }
            for body in r.bodies() {
                for s in body {
                    reads(s, out);
                }
            }
        }
        StmtKind::Owned { tile, body, .. } => {
            expr(tile, out);
            for b in body {
                reads(b, out);
            }
        }
        StmtKind::Range { body, .. }
        | StmtKind::Lanes { body, .. }
        | StmtKind::Parallel { body, .. } => {
            for b in body {
                reads(b, out);
            }
        }
        StmtKind::LoadLoop {
            domain,
            views,
            body,
            ..
        } => {
            expr(&domain.view, out);
            for v in views {
                expr(v, out);
            }
            for b in body {
                reads(b, out);
            }
        }
        StmtKind::If { cond, then, els } => {
            expr(cond, out);
            for b in then.iter().chain(els.iter()) {
                reads(b, out);
            }
        }
    }
}

#[allow(dead_code)]
fn _unused(_v: &Var, _k: &VarKind) {}
