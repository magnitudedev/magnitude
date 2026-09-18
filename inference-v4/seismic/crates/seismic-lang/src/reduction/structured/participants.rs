//! A selected contiguous fold tree mapped to a complete subgroup. This refines
//! the existing source operation; step and merge arithmetic remain ordinary IR.
use super::*;
mod wavefront;
use crate::{ast::BinaryOp, intrinsics::Operation, lowered_ir::LoweredIr, types::Elem};
use std::collections::{HashMap, HashSet};

/// Lexical reduction sites that this finite participant form can represent.
/// Leaves, including the seed, occupy cyclic lane/slot coordinates. Each
/// participant may retain several contiguous segments in separate private slots.
pub fn candidates(function: &LoweredIr, lanes: u32) -> Vec<usize> {
    candidate_sites(function, lanes, false, false)
}
pub fn wavefront_candidates(function: &LoweredIr, lanes: u32) -> Vec<usize> {
    candidate_sites(function, lanes, true, false)
}
pub fn root_seed_candidates(function: &LoweredIr, lanes: u32) -> Vec<usize> {
    candidate_sites(function, lanes, false, true)
}
fn candidate_sites(function: &LoweredIr, lanes: u32, wavefront: bool, root_seed: bool) -> Vec<usize> {
    let mut sites = Vec::new();
    let mut site = 0;
    fn visit(
        body: &[Stmt],
        site: &mut usize,
        sites: &mut Vec<usize>,
        lanes: u32,
        independent: bool,
        wavefront: bool,
        root_seed: bool,
    ) {
        for s in body {
            match &s.kind {
                StmtKind::Reduction(r) => {
                    if independent && applicable(r, lanes) && (!wavefront || matches!(r.tree, Some(Tree::Pairwise | Tree::SeedThenPairwise))) && (!root_seed || r.tree == Some(Tree::SeedThenPairwise)) {
                        sites.push(*site);
                    }
                    *site += 1;
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::LoadLoop { body, .. } => visit(body, site, sites, lanes, independent, wavefront, root_seed),
                StmtKind::Lanes { body, .. } => visit(body, site, sites, lanes, false, wavefront, root_seed),
                StmtKind::If { then, els, .. } => {
                    visit(then, site, sites, lanes, independent, wavefront, root_seed);
                    visit(els, site, sites, lanes, independent, wavefront, root_seed);
                }
                _ => {}
            }
        }
    }
    visit(&function.body, &mut site, &mut sites, lanes, true, wavefront, root_seed);
    sites
}
fn local_expr(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Intrinsic { .. } | ExprKind::Lanes { .. } | ExprKind::Call { .. } => false,
        ExprKind::Load { view, .. }
        | ExprKind::Transpose(view)
        | ExprKind::Accessor { base: view, .. }
        | ExprKind::Unary { expr: view, .. }
        | ExprKind::Cast { expr: view, .. } => local_expr(view),
        ExprKind::Index { base, indices } => {
            local_expr(base)
                && indices.iter().all(|i| match i {
                    Index::Point(e) => local_expr(e),
                    Index::Slice { start, end } => start.iter().chain(end).all(local_expr),
                })
        }
        ExprKind::Binary { lhs, rhs, .. } => local_expr(lhs) && local_expr(rhs),
        ExprKind::Builtin { args, .. } | ExprKind::Tuple(args) => args.iter().all(local_expr),
        _ => true,
    }
}
fn local_body(body: &[Stmt]) -> bool {
    body.iter().all(|s| match &s.kind {
        StmtKind::Lanes { .. } | StmtKind::Parallel { .. } | StmtKind::LoadLoop { .. } => false,
        StmtKind::Owned { body, .. } | StmtKind::Range { body, .. } => local_body(body),
        StmtKind::Reduction(r) => r.operands().all(local_expr) && r.bodies().all(local_body),
        StmtKind::If { cond, then, els } => local_expr(cond) && local_body(then) && local_body(els),
        StmtKind::Assign { target, value, .. } => local_expr(target) && local_expr(value),
        StmtKind::Expr(e) => local_expr(e),
    })
}
fn applicable(r: &Reduction, lanes: u32) -> bool {
    if lanes < 2
        || !lanes.is_power_of_two()
        || r.ordered
        || r.step.is_none()
        || r.implementation.is_none()
        || !matches!(r.tree, Some(Tree::Pairwise | Tree::Explicit | Tree::SeedThenPairwise))
    {
        return false;
    }
    let Some(n) = r.extent().as_constant() else {
        return false;
    };
    let Some(segment) = r.segment.filter(|&s| s > 0 && s <= n) else {
        return false;
    };
    let count = n / segment + i64::from(n % segment != 0);
    // This realization uses ordinary signed-I32 loop and tree coordinates.
    // Include the padded segment end and inactive neighbor calculations, not
    // just the largest live leaf, in that representability check.
    count.checked_add(1).is_some_and(|leaves| leaves <= (i64::from(i32::MAX) + 1) / 2)
        && i128::from(count) * i128::from(segment) <= i128::from(i32::MAX)
        && i64::from(lanes) <= (i64::from(i32::MAX) + 1) / 2
        && r.bodies().all(local_body)
        && r.state.iter().all(|s| {
            s.ty.shaped()
                .is_some_and(|s| s.elem == Elem::Dtype(DType::F32))
        })
}

/// Selected site IDs come from `candidates` on this exact retained computation.
/// Unselected reductions retain their ordinary single-participant realization.
#[derive(Clone, Debug)]
pub struct Refinement {
    pub function: LoweredIr,
    /// Tile values with distinct state in each participant. Placement must
    /// preserve this ownership; cooperation on one shared value is not valid.
    pub private_values: Vec<usize>,
}
/// Where segment computation takes place relative to the seed leaf. Both
/// mappings publish the identical logical leaves before applying the merge tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedPlacement {
    LeadingLeaf,
    InsertAfterSegments,
    /// The selected tree combines the unchanged seed with the completed input root.
    AtRoot,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completion { RetainLeaves, CompleteWaves }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub site: usize,
    pub seed: SeedPlacement,
    pub completion: Completion,
}
pub fn apply(
    function: &LoweredIr,
    selected: &[Selection],
    lanes: u32,
) -> Result<Refinement, String> {
    let admitted = candidates(function, lanes);
    let chosen: HashMap<_, _> = selected.iter().map(|s| (s.site, *s)).collect();
    if chosen.len() != selected.len() || chosen.keys().any(|i| !admitted.contains(i)) {
        return Err("participant fold site is outside the selected reduction family".into());
    }
    let mut result = function.clone();
    let mut site = 0;
    let mut private_values = HashSet::new();
    fn block(
        body: &mut Vec<Stmt>,
        vars: &mut Vec<Var>,
        site: &mut usize,
        chosen: &HashMap<usize, Selection>,
        lanes: u32,
        private_values: &mut HashSet<usize>,
    ) -> Result<(), String> {
        let mut output = Vec::new();
        for mut stmt in std::mem::take(body) {
            match &mut stmt.kind {
                StmtKind::Reduction(r) => {
                    let id = *site;
                    *site += 1;
                    if let Some(selection) = chosen.get(&id) {
                        if (selection.seed == SeedPlacement::AtRoot) != (r.tree == Some(Tree::SeedThenPairwise)) {
                            return Err("participant seed placement does not implement the selected tree".into());
                        }
                        let generated = match selection.completion {
                            Completion::RetainLeaves => expand(r, vars, lanes, selection.seed)?,
                            Completion::CompleteWaves => wavefront::expand(r, vars, lanes, selection.seed)?,
                        };
                        private_tiles(&generated, private_values);
                        output.extend(generated);
                        continue;
                    }
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::LoadLoop { body, .. }
                | StmtKind::Lanes { body, .. } => {
                    block(body, vars, site, chosen, lanes, private_values)?
                }
                StmtKind::If { then, els, .. } => {
                    block(then, vars, site, chosen, lanes, private_values)?;
                    block(els, vars, site, chosen, lanes, private_values)?;
                }
                _ => {}
            }
            output.push(stmt);
        }
        *body = output;
        Ok(())
    }
    block(
        &mut result.body,
        &mut result.vars,
        &mut site,
        &chosen,
        lanes,
        &mut private_values,
    )?;
    let mut private_values: Vec<_> = private_values.into_iter().collect();
    private_values.sort_unstable();
    Ok(Refinement {
        function: result,
        private_values,
    })
}
fn private_tiles(body: &[Stmt], values: &mut HashSet<usize>) {
    for s in body {
        match &s.kind {
            StmtKind::Assign { target, .. } => {
                let mut root = target;
                while let ExprKind::Index { base, .. }
                | ExprKind::Transpose(base)
                | ExprKind::Accessor { base, .. } = &root.kind
                {
                    root = base;
                }
                if let ExprKind::Var(id) = root.kind {
                    if matches!(root.ty, Ty::Tile(_)) {
                        values.insert(id);
                    }
                }
            }
            StmtKind::Owned { body, .. }
            | StmtKind::Parallel { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::Lanes { body, .. }
            | StmtKind::LoadLoop { body, .. } => private_tiles(body, values),
            StmtKind::If { then, els, .. } => {
                private_tiles(then, values);
                private_tiles(els, values);
            }
            _ => {}
        }
    }
}
fn intrinsic(operation: Operation, args: Vec<Expr>, dtype: DType, span: Span) -> Expr {
    Expr {
        kind: ExprKind::Intrinsic {
            op: operation,
            args,
        },
        ty: Ty::Scalar(dtype),
        sym: None,
        span,
    }
}
fn binary(op: BinaryOp, lhs: Expr, rhs: Expr, span: Span) -> Expr {
    let comparison = matches!(
        op,
        BinaryOp::Eq | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::And
    );
    Expr {
        kind: ExprKind::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        ty: Ty::Scalar(if comparison { DType::Bool } else { DType::I32 }),
        sym: None,
        span,
    }
}
fn assign(target: Expr, value: Expr, span: Span) -> Stmt {
    stmt(
        StmtKind::Assign {
            target,
            op: AssignOp::Assign,
            value,
        },
        span,
    )
}
fn shuffle_copy(b: &mut Builder<'_>, target: &Expr, value: &Expr, lane: &Expr) -> Stmt {
    let mut copy = b.copy(target, value);
    let StmtKind::Owned { body, .. } = &mut copy.kind else {
        unreachable!()
    };
    let StmtKind::Assign { value, .. } = &mut body[0].kind else {
        unreachable!()
    };
    *value = intrinsic(
        Operation::ShuffleIndex,
        vec![value.clone(), lane.clone()],
        DType::F32,
        b.span,
    );
    copy
}
fn expand(
    r: &Reduction,
    vars: &mut Vec<Var>,
    lanes: u32,
    seed: SeedPlacement,
) -> Result<Vec<Stmt>, String> {
    if !applicable(r, lanes) {
        return Err("unsupported selected participant fold".into());
    }
    let parts = r.segments(vars)?;
    let span = r.span;
    let mut b = Builder { vars, span };
    let leaf_ids: HashSet<_> = parts
        .leaves
        .iter()
        .filter_map(|e| match e.kind {
            ExprKind::Var(id) => Some(id),
            _ => None,
        })
        .collect();
    // Replace the logical leaf array with private slots distributed cyclically
    // across participants. A single slot reuses the existing partial directly.
    let mut body: Vec<_> = parts.setup.into_iter().filter(|s| {
        !matches!(&s.kind, StmtKind::Assign { target: Expr { kind: ExprKind::Var(id), .. }, .. } if leaf_ids.contains(id))
    }).collect();
    let leaves = parts
        .count
        .checked_add(i64::from(seed != SeedPlacement::AtRoot))
        .ok_or("participant leaf count overflow")?;
    let lanes = i64::from(lanes);
    let slots = leaves / lanes + i64::from(leaves % lanes != 0);
    slots
        .checked_mul(lanes)
        .ok_or("participant slot extent overflow")?;
    let storage = if slots == 1 {
        parts.partial.clone()
    } else {
        parts
            .partial
            .iter()
            .map(|value| {
                let mut shape = value.ty.shaped().unwrap().clone();
                shape.shape.insert(0, Sym::constant(slots));
                b.alloc(&Ty::Tile(shape), &mut body)
            })
            .collect::<Vec<_>>()
    };
    let at = |field: &Expr, index: &Expr| {
        if slots == 1 {
            field.clone()
        } else {
            slice(field, 0, index, span)
        }
    };
    let lane = b.local(Ty::Scalar(DType::I32));
    body.push(assign(
        lane.clone(),
        intrinsic(Operation::LaneIndex, vec![], DType::I32, span),
        span,
    ));
    let slot = b.index();
    let logical_value = binary(
        BinaryOp::Add,
        binary(BinaryOp::Mul, slot.clone(), integer(lanes, span), span),
        lane.clone(),
        span,
    );
    let logical = b.local(Ty::Scalar(DType::I32));
    let mut fill = Vec::new();
    fill.push(assign(logical.clone(), logical_value, span));
    for (dst, identity) in parts.partial.iter().zip(&r.step.as_ref().unwrap().identity) {
        fill.push(b.copy(dst, identity));
    }
    let mut segment_body = vec![assign(
        parts.index.clone(),
        match seed {
            SeedPlacement::LeadingLeaf => {
                binary(BinaryOp::Sub, logical.clone(), integer(1, span), span)
            }
            SeedPlacement::InsertAfterSegments | SeedPlacement::AtRoot => logical.clone(),
        },
        span,
    )];
    segment_body.extend(parts.body);
    let valid = match seed {
        SeedPlacement::LeadingLeaf => binary(
            BinaryOp::And,
            binary(BinaryOp::Gt, logical.clone(), integer(0, span), span),
            binary(
                BinaryOp::Le,
                logical.clone(),
                integer(parts.count, span),
                span,
            ),
            span,
        ),
        SeedPlacement::InsertAfterSegments | SeedPlacement::AtRoot => binary(
            BinaryOp::Lt,
            logical.clone(),
            integer(parts.count, span),
            span,
        ),
    };
    fill.push(stmt(
        StmtKind::If {
            cond: valid,
            then: segment_body,
            els: vec![],
        },
        span,
    ));
    if seed == SeedPlacement::LeadingLeaf {
        let seed = parts
            .partial
            .iter()
            .zip(&r.state)
            .map(|(dst, src)| b.copy(dst, src))
            .collect();
        fill.push(stmt(
            StmtKind::If {
                cond: binary(BinaryOp::Eq, logical.clone(), integer(0, span), span),
                then: seed,
                els: vec![],
            },
            span,
        ));
    }
    if slots > 1 {
        for (field, value) in storage.iter().zip(&parts.partial) {
            fill.push(b.copy(&at(field, &slot), value));
        }
    }
    body.push(b.range(&slot, Sym::constant(slots), fill));

    if seed == SeedPlacement::InsertAfterSegments {
        // Segment j was computed at (j / lanes, j % lanes). Shift retained
        // leaves to j + 1 before merging, inserting the unchanged seed at 0.
        // Descending slots preserve the preceding slot until all readers use it.
        let shifted = parts
            .partial
            .iter()
            .map(|value| b.alloc(&value.ty, &mut body))
            .collect::<Vec<_>>();
        let previous = parts
            .partial
            .iter()
            .map(|value| b.alloc(&value.ty, &mut body))
            .collect::<Vec<_>>();
        let previous_lane = binary(
            BinaryOp::Rem,
            binary(BinaryOp::Add, lane.clone(), integer(lanes - 1, span), span),
            integer(lanes, span),
            span,
        );
        let insert = |b: &mut Builder<'_>, target_slot: &Expr, source_slot: Option<&Expr>| {
            let mut transfer = Vec::new();
            for ((dst, prev), field) in shifted.iter().zip(&previous).zip(&storage) {
                transfer.push(shuffle_copy(
                    b,
                    dst,
                    &at(field, target_slot),
                    &previous_lane,
                ));
                if let Some(source_slot) = source_slot {
                    // Every lane reads the same slot before either shuffle.
                    transfer.push(shuffle_copy(
                        b,
                        prev,
                        &at(field, source_slot),
                        &integer(lanes - 1, span),
                    ));
                }
            }
            let lane_zero = shifted
                .iter()
                .zip(if source_slot.is_some() {
                    &previous
                } else {
                    &r.state
                })
                .map(|(dst, src)| b.copy(dst, src))
                .collect();
            transfer.push(stmt(
                StmtKind::If {
                    cond: binary(BinaryOp::Eq, lane.clone(), integer(0, span), span),
                    then: lane_zero,
                    els: vec![],
                },
                span,
            ));
            for (field, value) in storage.iter().zip(&shifted) {
                transfer.push(b.copy(&at(field, target_slot), value));
            }
            transfer
        };
        if slots > 1 {
            let reverse = b.index();
            let target = binary(
                BinaryOp::Sub,
                integer(slots - 1, span),
                reverse.clone(),
                span,
            );
            let source = binary(BinaryOp::Sub, target.clone(), integer(1, span), span);
            let transfer = insert(&mut b, &target, Some(&source));
            body.push(b.range(&reverse, Sym::constant(slots - 1), transfer));
        }
        body.extend(insert(&mut b, &integer(0, span), None));
    }

    let merge = r.implementation.as_ref().unwrap();
    for parameter in merge.left.iter().chain(&merge.right).chain(&merge.output) {
        body.push(b.allocate(parameter));
    }
    let combine = |b: &mut Builder<'_>,
                   target_slot: &Expr,
                   source_slot: &Expr,
                   source_lane: Expr,
                   active: Expr| {
        let left = storage.iter().map(|field| at(field, target_slot)).collect::<Vec<_>>();
        let right = storage.iter().map(|field| at(field, source_slot)).collect::<Vec<_>>();
        combine_values(b, merge, &left, &right, &left, &source_lane, active)
    };
    match r.tree.unwrap() {
        Tree::Pairwise | Tree::SeedThenPairwise => {
            let mut stride = 1_i64;
            while stride < leaves {
                let slot = b.index();
                let logical = binary(
                    BinaryOp::Add,
                    binary(BinaryOp::Mul, slot.clone(), integer(lanes, span), span),
                    lane.clone(),
                    span,
                );
                // Below subgroup width, participating pairs stay in one slot.
                // Above it, partners occupy a uniformly offset slot on the same
                // lane. Ascending slots never overwrite an unread right child.
                let offset = stride / lanes;
                let count = slots - offset;
                let neighbor = binary(BinaryOp::Add, logical.clone(), integer(stride, span), span);
                let in_range = binary(BinaryOp::Lt, neighbor, integer(leaves, span), span);
                let leader = binary(
                    BinaryOp::Eq,
                    binary(
                        BinaryOp::Rem,
                        logical.clone(),
                        integer(
                            stride
                                .checked_mul(2)
                                .ok_or("participant tree stride overflow")?,
                            span,
                        ),
                        span,
                    ),
                    integer(0, span),
                    span,
                );
                let source_lane = binary(
                    BinaryOp::Rem,
                    binary(
                        BinaryOp::Add,
                        lane.clone(),
                        integer(stride % lanes, span),
                        span,
                    ),
                    integer(lanes, span),
                    span,
                );
                let source_slot = if offset == 0 {
                    slot.clone()
                } else {
                    // Preserve symbolic view coordinates for the uniform loop.
                    symbol(slot.sym.as_ref().unwrap().add(&Sym::constant(offset)), span)
                };
                let computation = combine(
                    &mut b,
                    &slot,
                    &source_slot,
                    source_lane,
                    binary(BinaryOp::And, leader, in_range, span),
                );
                body.push(b.range(&slot, Sym::constant(count), computation));
                stride = stride
                    .checked_mul(2)
                    .ok_or("participant tree stride overflow")?;
            }
        }
        Tree::Explicit => {
            for branch in r.branches.iter().rev() {
                body.extend(combine(
                    &mut b,
                    &integer(branch.start / lanes, span),
                    &integer(branch.cut / lanes, span),
                    integer(branch.cut % lanes, span),
                    binary(
                        BinaryOp::Eq,
                        lane.clone(),
                        integer(branch.start % lanes, span),
                        span,
                    ),
                ));
            }
        }
        Tree::Ordered => unreachable!(),
    }
    if seed == SeedPlacement::AtRoot {
        let root = storage.iter().map(|field| at(field, &integer(0, span))).collect::<Vec<_>>();
        body.extend(combine_seed_root(&mut b, r, &root, &lane));
    }
    for (dst, field) in r.state.iter().zip(&storage) {
        body.push(shuffle_copy(
            &mut b,
            dst,
            &at(field, &integer(0, span)),
            &integer(0, span),
        ));
    }
    Ok(body)
}

/// Source slots are uniform across participants before exchange. The callback
/// sees complete left/right snapshots and publishes only under its leader mask.
fn combine_values(b: &mut Builder<'_>, merge: &Merge, left: &[Expr], right: &[Expr], target: &[Expr], source_lane: &Expr, active: Expr) -> Vec<Stmt> {
    let mut computation = Vec::new();
    for ((l,r),(a,c)) in merge.left.iter().zip(&merge.right).zip(left.iter().zip(right)) {
        computation.push(b.copy(l,a));
        computation.push(shuffle_copy(b,r,c,source_lane));
    }
    let mut arithmetic = merge.body.clone();
    for (field,result) in target.iter().zip(&merge.output) { arithmetic.push(b.copy(field,result)); }
    computation.push(stmt(StmtKind::If { cond: active, then: arithmetic, els: vec![] }, b.span));
    computation
}

/// The seed remains the leftmost source leaf. Apply its one root merge only at
/// lane zero, then use the ordinary final broadcast for the completed result.
fn combine_seed_root(b: &mut Builder<'_>, r: &Reduction, root: &[Expr], lane: &Expr) -> Vec<Stmt> {
    combine_values(b, r.implementation.as_ref().unwrap(), &r.state, root, root,
        &integer(0, r.span), binary(BinaryOp::Eq, lane.clone(), integer(0, r.span), r.span))
}
