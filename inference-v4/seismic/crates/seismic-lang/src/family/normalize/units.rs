//! Execution units of a definition body (language.md section 7).
//!
//! Every single-consumer pure tile-valued `let` adjacent to its consumer's unit joins that
//! unit; every remaining statement is one unit (a `stage` run is one unit per stage).
//! Units partition the block's statement ordinals in authored order.
//!
//! Conventions consumers (backend mappings, `instantiate`) rely on:
//! - `Unit::statements` are RAW ordinals into the `sir` block (`Body::block`, a region /
//!   stage / loop / branch body) exactly as the checker emitted it. Nothing is removed or
//!   renumbered: a folded `let` stays at its ordinal and is INCLUDED in its consumer's range,
//!   so a unit covering a consumer at ordinal `j` with `k` folded lets is `j-k..j+1`. The
//!   consumer is the last statement of the range; every earlier statement of the range is a
//!   folded `let`. The ranges of a block's units are contiguous, ascending and cover
//!   `0..block.len()` exactly; the units of one `stage` run all carry that statement's
//!   one-ordinal range.
//! - A single-consumer pure tile-valued `let` that is not adjacent to its consumer's unit is
//!   not folded: it is its own unit with its ordinary kind.
//! - `ScopeStep::Stage(i)` and `UnitKind::Stage(i)` count stages cumulatively within the
//!   block across all its `Stages` statements (the index within the run when there is one).
//! - `ScopeStep::Then(n)`, `Else(n)`, `Loop(n)` carry the raw statement ordinal `n` of the
//!   `if` / loop in its enclosing block. `ScopeStep::Region(id)` names the region, in
//!   statement or expression position. Merge bodies have no scope step and no sequence.
//! - `UnitKind::Call(occurrence)` is the unit of every `let`/assignment/expression/`yield`/
//!   `return` statement whose own expressions contain a call, to a lowering boundary or a
//!   plain helper alike (helper extraction does not change unit structure); with several
//!   calls in one statement it names the first in evaluation order. A `publish` stays
//!   `Publish`. `Family::root_regions` tells a backend when a selected candidate of that
//!   occurrence is nothing but root-level parallel regions, so the `Call` unit can be treated
//!   like `Region` units.
//! - A `Sequence` exists only for a block with at least two units; a block with fewer has
//!   no grouping decision and no cover in a witness.

use super::super::construct::walk;
use super::super::{OccurrenceId, ScopeStep, Unit, UnitKind};
use crate::sir::{Block, Body, Expr, ExprKind, Index, Pattern, Stmt, StmtKind, VarKind};
use crate::syntax::ast::RegionMode;
use crate::types::{Extent, Ty};

/// Units of one block. `UnitKind::Call` carries the body's `CallId` in place of the
/// occurrence; construction substitutes the candidate's child occurrence.
pub struct BlockUnits {
    pub scope: Vec<ScopeStep>,
    pub units: Vec<Unit>,
}

/// Blocks with at least two units, root first, nested blocks in authored order.
pub fn sequences(body: &Body) -> Vec<BlockUnits> {
    let mut out = Vec::new();
    collect(body, &body.block, &mut Vec::new(), false, &mut out);
    out
}

fn collect(body: &Body, block: &Block, scope: &mut Vec<ScopeStep>, pipeline: bool, out: &mut Vec<BlockUnits>) {
    let units = units(body, block, pipeline);
    if units.len() >= 2 {
        out.push(BlockUnits { scope: scope.clone(), units });
    }
    let mut stage = 0;
    for (ordinal, s) in block.iter().enumerate() {
        let mut enter = |step: ScopeStep, inner: &Block, pipeline: bool, out: &mut Vec<BlockUnits>| {
            scope.push(step);
            collect(body, inner, scope, pipeline, out);
            scope.pop();
        };
        for r in walk::expression_regions(s) {
            enter(ScopeStep::Region(r.id), &r.body, r.mode == RegionMode::Pipeline, out);
        }
        match &s.kind {
            StmtKind::Region(r) => enter(ScopeStep::Region(r.id), &r.body, r.mode == RegionMode::Pipeline, out),
            StmtKind::Stages(stages) => {
                for st in stages {
                    enter(ScopeStep::Stage(stage), &st.body, false, out);
                    stage += 1;
                }
            }
            StmtKind::Range { body: inner, .. } | StmtKind::Coordinates { body: inner, .. } | StmtKind::Members { body: inner, .. } => {
                enter(ScopeStep::Loop(ordinal), inner, false, out)
            }
            StmtKind::If { then, els, .. } => {
                enter(ScopeStep::Then(ordinal), then, false, out);
                enter(ScopeStep::Else(ordinal), els, false, out);
            }
            _ => {}
        }
    }
}

fn units(body: &Body, block: &Block, pipeline: bool) -> Vec<Unit> {
    // Per statement: its unit kinds with completion (several only for a stage run).
    let mut stage = 0;
    let mut kinds: Vec<Vec<(UnitKind, bool)>> = block
        .iter()
        .map(|s| match &s.kind {
            StmtKind::Stages(stages) => stages
                .iter()
                .map(|_| {
                    stage += 1;
                    (UnitKind::Stage(stage - 1), !pipeline)
                })
                .collect(),
            _ => vec![(classify(s), false)],
        })
        .collect();
    // Fold adjacent single-consumer lets into the unit of their consumer, last statement first.
    let mut head: Vec<usize> = (0..block.len()).collect();
    let mut start: Vec<usize> = (0..block.len()).collect();
    for i in (0..block.len()).rev() {
        let Some(consumer) = consumer(body, block, i) else { continue };
        let h = head[consumer];
        if kinds[h].len() != 1 || start[h] != i + 1 {
            continue;
        }
        if kinds[h][0].0 == UnitKind::Elementwise && kinds[i][0].0 != UnitKind::Elementwise {
            kinds[h][0].0 = UnitKind::Local;
        }
        head[i] = h;
        start[h] = i;
    }
    let mut out = Vec::new();
    for i in 0..block.len() {
        if head[i] != i {
            continue;
        }
        for (kind, completion_after) in kinds[i].drain(..) {
            out.push(Unit { statements: start[i]..i + 1, kind, completion_after });
        }
    }
    out
}

/// The statement of `block` that is the only consumer of the pure tile-valued `let` at `i`.
fn consumer(body: &Body, block: &Block, i: usize) -> Option<usize> {
    let StmtKind::Bind { pattern: Pattern::Var(var), value } = &block[i].kind else { return None };
    if body.vars[*var].kind != VarKind::Value || !matches!(value.ty, Ty::Tile(_)) || !pure(value) {
        return None;
    }
    let uses = |s: &Stmt, nested: bool| {
        let mut n = 0;
        walk::stmt(s, nested, &mut |e| n += usize::from(matches!(e.kind, ExprKind::Var(v) if v == *var)));
        n
    };
    let later = &block[i + 1..];
    if later.iter().map(|s| uses(s, true)).sum::<usize>() != 1 {
        return None;
    }
    let j = later.iter().position(|s| uses(s, false) == 1)?;
    matches!(
        later[j].kind,
        StmtKind::Bind { .. } | StmtKind::Assign { .. } | StmtKind::Publish { .. } | StmtKind::Expr(_) | StmtKind::Yield(_) | StmtKind::Return(_)
    )
    .then_some(i + 1 + j)
}

fn classify(s: &Stmt) -> UnitKind {
    let call = || {
        let mut found = None;
        walk::stmt(s, false, &mut |e| {
            if let ExprKind::Call { call, .. } = &e.kind {
                if found.is_none() {
                    found = Some(UnitKind::Call(OccurrenceId(call.0)));
                }
            }
        });
        found
    };
    let tile = |elementwise: bool| if elementwise { UnitKind::Elementwise } else { UnitKind::Local };
    match &s.kind {
        StmtKind::Region(r) => UnitKind::Region(r.id),
        StmtKind::Publish { .. } => UnitKind::Publish,
        StmtKind::Bind { value, .. } => match &value.kind {
            ExprKind::Region(r) => UnitKind::Region(r.id),
            _ => call().unwrap_or_else(|| tile(elementwise(value))),
        },
        StmtKind::Assign { target, value, .. } => {
            call().unwrap_or_else(|| tile(matches!(target.ty, Ty::Tile(_)) && elementwise(value)))
        }
        StmtKind::Expr(_) | StmtKind::Yield(_) | StmtKind::Return(_) => call().unwrap_or(UnitKind::Local),
        StmtKind::Stages(_) | StmtKind::Range { .. } | StmtKind::Coordinates { .. } | StmtKind::Members { .. } | StmtKind::If { .. } => UnitKind::Local,
    }
}

fn pure(e: &Expr) -> bool {
    let mut pure = true;
    walk::expr(e, false, &mut |e| {
        pure &= !matches!(
            e.kind,
            ExprKind::Call { .. } | ExprKind::Region(_) | ExprKind::Intrinsic { .. } | ExprKind::Atomic { .. } | ExprKind::TileAlloc
        )
    });
    pure
}

/// Tile-valued arithmetic, casts, select and math over operands with identical axes.
fn elementwise(e: &Expr) -> bool {
    match &e.ty {
        Ty::Tile(shape) => operand(e, &shape.axes),
        _ => false,
    }
}

fn operand(e: &Expr, axes: &[Extent]) -> bool {
    let shape = match &e.ty {
        Ty::Scalar(_) | Ty::Index(_) => return scalar(e),
        Ty::Tile(s) | Ty::View(s) | Ty::Tensor(s) => s,
        _ => return false,
    };
    if shape.axes != axes {
        return false;
    }
    match &e.kind {
        ExprKind::Var(_) | ExprKind::Field { .. } | ExprKind::Member { .. } => true,
        ExprKind::Index { base, indices } => {
            matches!(base.kind, ExprKind::Var(_) | ExprKind::Field { .. } | ExprKind::Member { .. } | ExprKind::Index { .. })
                && indices.iter().all(|index| match index {
                    Index::Point(p) => scalar(p),
                    Index::Range { start, end } => [start, end].into_iter().flatten().all(scalar),
                    Index::Coord(_) | Index::Slice(_) => true,
                })
        }
        ExprKind::Decode(inner) | ExprKind::Cast { expr: inner, .. } | ExprKind::Unary { expr: inner, .. } => operand(inner, axes),
        ExprKind::Binary { lhs, rhs, .. } => operand(lhs, axes) && operand(rhs, axes),
        ExprKind::Math { args, .. } => args.iter().all(|a| operand(a, axes)),
        ExprKind::Select { cond, then, els } => operand(cond, axes) && operand(then, axes) && operand(els, axes),
        _ => false,
    }
}

/// Scalar broadcast operand: no reduction, call, region or target operation inside.
fn scalar(e: &Expr) -> bool {
    let mut plain = true;
    walk::expr(e, false, &mut |e| {
        plain &= !matches!(
            e.kind,
            ExprKind::Reduce { .. } | ExprKind::Call { .. } | ExprKind::Region(_) | ExprKind::Intrinsic { .. } | ExprKind::Atomic { .. }
        )
    });
    plain
}
