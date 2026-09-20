//! Read-only traversals of `sir`: use counts, the folding rule of execution-unit
//! normalization (language.md section 7), scope paths and terminal control flow.
use crate::family::ScopeStep;
use crate::sir::*;
use crate::types::Ty;
use std::collections::BTreeSet;

/// Operand expressions of `e`; the body of a region expression is not an operand.
pub(super) fn children(e: &Expr) -> Vec<&Expr> {
    match &e.kind {
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Var(_)
        | ExprKind::ShapeParam(_)
        | ExprKind::TileAlloc
        | ExprKind::CoordOf(_) => Vec::new(),
        ExprKind::Tuple(items) => items.iter().collect(),
        ExprKind::Range { lo, hi } => vec![lo, hi],
        ExprKind::Field { base, .. }
        | ExprKind::Transpose(base)
        | ExprKind::Reshape { base, .. }
        | ExprKind::Load(base)
        | ExprKind::Decode(base)
        | ExprKind::Cast { expr: base, .. }
        | ExprKind::Unary { expr: base, .. }
        | ExprKind::ExtentOf { base, .. }
        | ExprKind::Accessor { base, .. }
        | ExprKind::Geometry { base, .. }
        | ExprKind::Filled { like: base, .. }
        | ExprKind::Reduce { value: base, .. } => vec![base],
        ExprKind::Member { result, .. } => vec![result],
        ExprKind::Index { base, indices } => {
            let mut out: Vec<&Expr> = vec![base];
            for index in indices {
                match index {
                    Index::Point(p) => out.push(p),
                    Index::Range { start, end } => out.extend(start.iter().chain(end)),
                    Index::Coord(_) | Index::Slice(_) => {}
                }
            }
            out
        }
        ExprKind::Binary { lhs, rhs, .. } => vec![lhs, rhs],
        ExprKind::Select { cond, then, els } => vec![cond, then, els],
        ExprKind::Math { args, .. }
        | ExprKind::Intrinsic { args, .. }
        | ExprKind::Call { args, .. } => args.iter().collect(),
        ExprKind::Atomic { place, value, .. } => vec![place, value],
        ExprKind::Region(region) => match &region.source {
            RegionSource::Results(source) => vec![source],
            RegionSource::Domains => Vec::new(),
        },
    }
}

pub(super) fn each_expr<'a>(e: &'a Expr, f: &mut dyn FnMut(&'a Expr)) {
    f(e);
    for child in children(e) {
        each_expr(child, f);
    }
}

/// Expressions a statement evaluates in its own block (nested blocks excluded).
pub(super) fn direct_exprs(s: &Stmt) -> Vec<&Expr> {
    match &s.kind {
        StmtKind::Bind { value, .. } => vec![value],
        StmtKind::Assign { target, value, .. } => vec![target, value],
        StmtKind::Region(region) => match &region.source {
            RegionSource::Results(source) => vec![source],
            RegionSource::Domains => Vec::new(),
        },
        StmtKind::Stages(_) => Vec::new(),
        StmtKind::Range { lo, hi, value, .. } => {
            let mut expressions = vec![lo, hi];
            expressions.extend(value);
            expressions
        }
        StmtKind::Coordinates { of, .. } => vec![of],
        StmtKind::Members { .. } => Vec::new(),
        StmtKind::If { cond, .. } => vec![cond],
        StmtKind::Publish { value, destination } => vec![value, destination],
        StmtKind::Yield(values) | StmtKind::Return(values) => values.iter().collect(),
        StmtKind::Expr(e) => vec![e],
    }
}

fn region_blocks<'a>(region: &'a Region, out: &mut Vec<&'a Block>) {
    out.push(&region.body);
    if let Some(merge) = &region.merge {
        each_expr(&merge.identity, &mut |e| {
            if let ExprKind::Region(r) = &e.kind {
                region_blocks(r, out);
            }
        });
        out.push(&merge.body);
    }
}

/// Every block nested in a statement, including bodies of region expressions.
pub(super) fn nested_blocks(s: &Stmt) -> Vec<&Block> {
    let mut out = Vec::new();
    for e in direct_exprs(s) {
        each_expr(e, &mut |x| {
            if let ExprKind::Region(r) = &x.kind {
                region_blocks(r, &mut out);
            }
        });
    }
    match &s.kind {
        StmtKind::Region(region) => region_blocks(region, &mut out),
        StmtKind::Stages(stages) => out.extend(stages.iter().map(|s| &s.body)),
        StmtKind::Range { body, .. }
        | StmtKind::Coordinates { body, .. }
        | StmtKind::Members { body, .. } => out.push(body),
        StmtKind::If { then, els, .. } => {
            out.push(then);
            out.push(els);
        }
        _ => {}
    }
    out
}

pub(super) fn each_block<'a>(block: &'a Block, f: &mut dyn FnMut(&'a Block)) {
    f(block);
    for s in block {
        for nested in nested_blocks(s) {
            each_block(nested, f);
        }
    }
}

fn merge_identities<'a>(s: &'a Stmt, f: &mut dyn FnMut(&'a Expr)) {
    let mut regions: Vec<&'a Region> = Vec::new();
    for e in direct_exprs(s) {
        each_expr(e, &mut |x| {
            if let ExprKind::Region(r) = &x.kind {
                regions.push(r);
            }
        });
    }
    if let StmtKind::Region(r) = &s.kind {
        regions.push(r);
    }
    for region in regions {
        if let Some(merge) = &region.merge {
            each_expr(&merge.identity, f);
        }
    }
}

/// References of every variable in the whole body.
pub(super) fn uses(body: &Body) -> Vec<usize> {
    let mut counts = vec![0usize; body.vars.len()];
    let mut count = |e: &Expr| {
        let var = match &e.kind {
            ExprKind::Var(v) | ExprKind::CoordOf(v) => Some(*v),
            _ => None,
        };
        if let Some(slot) = var.and_then(|v| counts.get_mut(v)) {
            *slot += 1;
        }
        if let ExprKind::Index { indices, .. } = &e.kind {
            for index in indices {
                if let Index::Coord(v) = index {
                    if let Some(slot) = counts.get_mut(*v) {
                        *slot += 1;
                    }
                }
            }
        }
    };
    each_block(&body.block, &mut |block| {
        for s in block {
            for e in direct_exprs(s) {
                each_expr(e, &mut count);
            }
            merge_identities(s, &mut count);
        }
    });
    counts
}

/// The family's purity rule (family/normalize/units.rs): no call, region, target operation
/// or allocation anywhere in the value.
fn pure(e: &Expr) -> bool {
    let mut pure = true;
    each_expr(e, &mut |x| {
        pure &= !matches!(
            x.kind,
            ExprKind::Call { .. }
                | ExprKind::Region(_)
                | ExprKind::Intrinsic { .. }
                | ExprKind::Atomic { .. }
                | ExprKind::TileAlloc
        )
    });
    pure
}

/// The statement that is the only consumer of the pure tile-valued `let` at `i`.
fn consumer(body: &Body, block: &Block, uses: &[usize], i: usize) -> Option<usize> {
    let StmtKind::Bind {
        pattern: Pattern::Var(var),
        value,
    } = &block[i].kind
    else {
        return None;
    };
    if body.vars.get(*var)?.kind != VarKind::Value
        || !matches!(value.ty, Ty::Tile(_))
        || !pure(value)
        || uses.get(*var) != Some(&1)
    {
        return None;
    }
    let direct = |s: &Stmt| {
        let mut n = 0;
        for e in direct_exprs(s) {
            each_expr(e, &mut |x| {
                n += usize::from(matches!(x.kind, ExprKind::Var(v) if v == *var))
            });
        }
        n
    };
    let later = &block[i + 1..];
    let j = later.iter().position(|s| direct(s) == 1)?;
    matches!(
        later[j].kind,
        StmtKind::Bind { .. }
            | StmtKind::Assign { .. }
            | StmtKind::Publish { .. }
            | StmtKind::Expr(_)
            | StmtKind::Yield(_)
            | StmtKind::Return(_)
    )
    .then_some(i + 1 + j)
}

/// Folding of a block without a sequence: the family's adjacency rule, last statement first.
fn fold_block(body: &Body, block: &Block, uses: &[usize], out: &mut BTreeSet<VarId>) {
    let mut head: Vec<usize> = (0..block.len()).collect();
    let mut start: Vec<usize> = (0..block.len()).collect();
    for i in (0..block.len()).rev() {
        let Some(consumer) = consumer(body, block, uses, i) else {
            continue;
        };
        let h = head[consumer];
        if matches!(block[h].kind, StmtKind::Stages(_)) || start[h] != i + 1 {
            continue;
        }
        head[i] = h;
        start[h] = i;
        if let StmtKind::Bind {
            pattern: Pattern::Var(v),
            ..
        } = &block[i].kind
        {
            out.insert(*v);
        }
    }
}

/// The `let`s normalization folded into their consumer. A block with a sequence takes the
/// folding from its unit ranges (every statement before the last of a range is folded);
/// a block without one has fewer than two units and follows the same adjacency rule.
pub(super) fn folded(
    body: &Body,
    uses: &[usize],
    sequences: &[(&Block, &[std::ops::Range<usize>])],
) -> Result<BTreeSet<VarId>, String> {
    let mut out = BTreeSet::new();
    let mut failure = None;
    each_block(&body.block, &mut |block| {
        let Some((_, ranges)) = sequences.iter().find(|(b, _)| std::ptr::eq(*b, block)) else {
            fold_block(body, block, uses, &mut out);
            return;
        };
        for range in ranges.iter() {
            for s in block
                .get(range.start..range.end.saturating_sub(1))
                .unwrap_or_default()
            {
                match &s.kind {
                    StmtKind::Bind {
                        pattern: Pattern::Var(v),
                        ..
                    } => {
                        out.insert(*v);
                    }
                    _ => {
                        failure = Some(format!(
                            "a unit range {range:?} folds a statement that is not a `let`"
                        ))
                    }
                }
            }
        }
    });
    match failure {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

fn find_region<'a>(block: &'a Block, id: crate::types::RegionId) -> Option<&'a Region> {
    for s in block {
        if let StmtKind::Region(r) = &s.kind {
            if r.id == id {
                return Some(r);
            }
        }
        let mut found = None;
        for e in direct_exprs(s) {
            each_expr(e, &mut |x| {
                if let ExprKind::Region(r) = &x.kind {
                    if r.id == id {
                        found = Some(&**r);
                    }
                }
            });
        }
        if found.is_some() {
            return found;
        }
    }
    None
}

/// The block a sequence scope path names. `Stage(k)` counts stages cumulatively within
/// the block; `Then`/`Else`/`Loop` carry the raw ordinal of their statement.
pub(super) fn scope<'a>(body: &'a Body, steps: &[ScopeStep]) -> Result<&'a Block, String> {
    let mut block = &body.block;
    for step in steps {
        let statement = |n: usize| {
            block
                .get(n)
                .ok_or_else(|| format!("scope step {step:?} names no statement"))
        };
        block = match step {
            ScopeStep::Region(id) => {
                &find_region(block, *id)
                    .ok_or_else(|| format!("scope step {step:?} names no region of its block"))?
                    .body
            }
            ScopeStep::Stage(k) => {
                let stage = block
                    .iter()
                    .filter_map(|s| match &s.kind {
                        StmtKind::Stages(stages) => Some(stages.iter()),
                        _ => None,
                    })
                    .flatten()
                    .nth(*k);
                &stage
                    .ok_or_else(|| format!("scope step {step:?} names no stage of its block"))?
                    .body
            }
            ScopeStep::Then(n) | ScopeStep::Else(n) => match &statement(*n)?.kind {
                StmtKind::If { then, els, .. } => {
                    if matches!(step, ScopeStep::Then(_)) {
                        then
                    } else {
                        els
                    }
                }
                _ => return Err(format!("scope step {step:?} does not name an `if`")),
            },
            ScopeStep::Loop(n) => match &statement(*n)?.kind {
                StmtKind::Range { body, .. }
                | StmtKind::Coordinates { body, .. }
                | StmtKind::Members { body, .. } => body,
                _ => return Err(format!("scope step {step:?} does not name a loop")),
            },
        };
    }
    Ok(block)
}

/// Whether some path through the statements ends its result boundary here.
pub(super) fn exits(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match &s.kind {
        StmtKind::Yield(_) | StmtKind::Return(_) => true,
        StmtKind::If { then, els, .. } => exits(then) || exits(els),
        _ => false,
    })
}

/// Whether every path through the statements ends its result boundary.
pub(super) fn terminates(stmts: &[Stmt]) -> bool {
    match stmts.last().map(|s| &s.kind) {
        Some(StmtKind::Yield(_) | StmtKind::Return(_)) => true,
        Some(StmtKind::If { then, els, .. }) => terminates(then) && terminates(els),
        _ => false,
    }
}

/// Statements that only define per-coordinate values: safe under an `Owned` element
/// domain, whose coordinates carry no order. Every access to a tile the loop writes must
/// address exactly the current coordinate.
pub(super) fn coordinate_local(block: &Block, body: &Body) -> bool {
    fn pattern(p: &Pattern, out: &mut BTreeSet<VarId>) {
        match p {
            Pattern::Var(v) => {
                out.insert(*v);
            }
            Pattern::Tuple(items) => items.iter().for_each(|p| pattern(p, out)),
        }
    }
    fn current(indices: &[Index], body: &Body) -> bool {
        indices.iter().all(|i| match i {
            Index::Coord(_) => true,
            Index::Point(Expr {
                kind: ExprKind::Var(v),
                ..
            }) => body
                .vars
                .get(*v)
                .is_some_and(|v| v.kind == VarKind::Coordinate),
            _ => false,
        })
    }
    fn local(
        block: &Block,
        body: &Body,
        bound: &mut BTreeSet<VarId>,
        written: &mut BTreeSet<VarId>,
    ) -> bool {
        block.iter().all(|s| {
            let effect_free = direct_exprs(s).into_iter().all(|e| {
                let mut ok = true;
                each_expr(e, &mut |x| {
                    ok &= !matches!(
                        x.kind,
                        ExprKind::Call { .. }
                            | ExprKind::Region(_)
                            | ExprKind::Intrinsic { .. }
                            | ExprKind::Atomic { .. }
                    )
                });
                ok
            });
            effect_free
                && match &s.kind {
                    StmtKind::Bind { pattern: p, .. } => {
                        pattern(p, bound);
                        true
                    }
                    StmtKind::Assign { target, .. } => match &target.kind {
                        ExprKind::Var(v) => bound.contains(v),
                        ExprKind::Index { base, indices } => match base.kind {
                            ExprKind::Var(v) if current(indices, body) => {
                                written.insert(v);
                                true
                            }
                            _ => false,
                        },
                        _ => false,
                    },
                    StmtKind::Range { body: inner, .. }
                    | StmtKind::Coordinates { body: inner, .. }
                    | StmtKind::Members { body: inner, .. } => local(inner, body, bound, written),
                    StmtKind::If { then, els, .. } => {
                        local(then, body, bound, written) && local(els, body, bound, written)
                    }
                    _ => false,
                }
        })
    }
    fn reads_current(e: &Expr, body: &Body, written: &BTreeSet<VarId>) -> bool {
        match &e.kind {
            ExprKind::Var(v) => !written.contains(v),
            ExprKind::Index { base, indices } if matches!(base.kind, ExprKind::Var(v) if written.contains(&v)) => {
                current(indices, body)
                    && children(e)
                        .into_iter()
                        .skip(1)
                        .all(|c| reads_current(c, body, written))
            }
            _ => children(e)
                .into_iter()
                .all(|c| reads_current(c, body, written)),
        }
    }
    let (mut bound, mut written) = (BTreeSet::new(), BTreeSet::new());
    if !local(block, body, &mut bound, &mut written) {
        return false;
    }
    let mut ok = true;
    each_block(block, &mut |b| {
        for s in b {
            ok &= direct_exprs(s)
                .into_iter()
                .all(|e| reads_current(e, body, &written));
        }
    });
    ok
}
