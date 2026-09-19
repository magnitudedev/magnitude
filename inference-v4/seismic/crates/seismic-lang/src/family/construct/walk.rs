//! Read-only traversal of a structured body. `nested == false` stays inside the current
//! block's own statements: region, stage, loop and branch bodies are not entered.

use crate::sir::{Block, Expr, ExprKind, Index, Region, RegionSource, Stmt, StmtKind};

pub fn expr<'a>(e: &'a Expr, nested: bool, f: &mut dyn FnMut(&'a Expr)) {
    f(e);
    match &e.kind {
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Var(_)
        | ExprKind::ShapeParam(_)
        | ExprKind::TileAlloc
        | ExprKind::CoordOf(_) => {}
        ExprKind::Tuple(items)
        | ExprKind::Math { args: items, .. }
        | ExprKind::Call { args: items, .. }
        | ExprKind::Intrinsic { args: items, .. } => {
            for item in items {
                expr(item, nested, f);
            }
        }
        ExprKind::Field { base, .. }
        | ExprKind::Filled { like: base, .. }
        | ExprKind::Member { result: base, .. }
        | ExprKind::Transpose(base)
        | ExprKind::Reshape { base, .. }
        | ExprKind::Load(base)
        | ExprKind::Decode(base)
        | ExprKind::Cast { expr: base, .. }
        | ExprKind::Unary { expr: base, .. }
        | ExprKind::Reduce { value: base, .. }
        | ExprKind::ExtentOf { base, .. }
        | ExprKind::Accessor { base, .. }
        | ExprKind::Geometry { base, .. } => expr(base, nested, f),
        ExprKind::Index { base, indices } => {
            expr(base, nested, f);
            for index in indices {
                match index {
                    Index::Point(point) => expr(point, nested, f),
                    Index::Range { start, end } => {
                        for bound in [start, end].into_iter().flatten() {
                            expr(bound, nested, f);
                        }
                    }
                    Index::Coord(_) | Index::Slice(_) => {}
                }
            }
        }
        ExprKind::Binary { lhs, rhs, .. } | ExprKind::Atomic { place: lhs, value: rhs, .. } => {
            expr(lhs, nested, f);
            expr(rhs, nested, f);
        }
        ExprKind::Select { cond, then, els } => {
            expr(cond, nested, f);
            expr(then, nested, f);
            expr(els, nested, f);
        }
        ExprKind::Region(r) => region(r, nested, f),
    }
}

fn region<'a>(r: &'a Region, nested: bool, f: &mut dyn FnMut(&'a Expr)) {
    if let RegionSource::Results(source) = &r.source {
        expr(source, nested, f);
    }
    if nested {
        block(&r.body, nested, f);
        if let Some(merge) = &r.merge {
            expr(&merge.identity, nested, f);
            block(&merge.body, nested, f);
        }
    }
}

pub fn stmt<'a>(s: &'a Stmt, nested: bool, f: &mut dyn FnMut(&'a Expr)) {
    match &s.kind {
        StmtKind::Bind { value, .. } | StmtKind::Expr(value) => expr(value, nested, f),
        StmtKind::Assign { target, value, .. } | StmtKind::Publish { value, destination: target } => {
            expr(target, nested, f);
            expr(value, nested, f);
        }
        StmtKind::Region(r) => region(r, nested, f),
        StmtKind::Stages(stages) => {
            if nested {
                for stage in stages {
                    block(&stage.body, nested, f);
                }
            }
        }
        StmtKind::Range { lo, hi, body, .. } => {
            expr(lo, nested, f);
            expr(hi, nested, f);
            if nested {
                block(body, nested, f);
            }
        }
        StmtKind::Coordinates { of, body, .. } => {
            expr(of, nested, f);
            if nested {
                block(body, nested, f);
            }
        }
        StmtKind::Members { body, .. } => {
            if nested {
                block(body, nested, f);
            }
        }
        StmtKind::If { cond, then, els } => {
            expr(cond, nested, f);
            if nested {
                block(then, nested, f);
                block(els, nested, f);
            }
        }
        StmtKind::Yield(values) | StmtKind::Return(values) => {
            for value in values {
                expr(value, nested, f);
            }
        }
    }
}

pub fn block<'a>(b: &'a Block, nested: bool, f: &mut dyn FnMut(&'a Expr)) {
    for s in b {
        stmt(s, nested, f);
    }
}

/// Regions in expression position among the statement's own expressions.
pub fn expression_regions(s: &Stmt) -> Vec<&Region> {
    let mut out = Vec::new();
    stmt(s, false, &mut |e| {
        if let ExprKind::Region(r) = &e.kind {
            out.push(&**r);
        }
    });
    out
}

/// Every statement of a block at all depths, including bodies of expression regions.
pub fn stmts<'a>(b: &'a Block, f: &mut dyn FnMut(&'a Stmt)) {
    fn region_blocks<'a>(r: &'a Region, f: &mut dyn FnMut(&'a Stmt)) {
        stmts(&r.body, f);
        if let Some(merge) = &r.merge {
            stmts(&merge.body, f);
        }
    }
    for s in b {
        f(s);
        for r in expression_regions(s) {
            region_blocks(r, f);
        }
        match &s.kind {
            StmtKind::Region(r) => region_blocks(r, f),
            StmtKind::Stages(stages) => stages.iter().for_each(|stage| stmts(&stage.body, f)),
            StmtKind::Range { body, .. } | StmtKind::Coordinates { body, .. } | StmtKind::Members { body, .. } => stmts(body, f),
            StmtKind::If { then, els, .. } => {
                stmts(then, f);
                stmts(els, f);
            }
            _ => {}
        }
    }
}

/// Every region of a block, statement and expression position, all depths.
pub fn regions(b: &Block) -> Vec<&Region> {
    let mut out = Vec::new();
    stmts(b, &mut |s| {
        if let StmtKind::Region(r) = &s.kind {
            out.push(r);
        }
        out.extend(expression_regions(s));
    });
    out
}
