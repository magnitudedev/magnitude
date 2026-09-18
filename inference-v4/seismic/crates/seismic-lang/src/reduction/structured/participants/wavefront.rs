//! Complete the fixed pairwise subtrees within each subgroup-sized leaf wave.
//! Their roots occupy distributed slots; earlier leaves do not stay live until
//! the entire input has been traversed. This is scheduling of the existing tree.
use super::*;

pub(super) fn expand(
    r: &Reduction,
    vars: &mut Vec<Var>,
    lanes: u32,
    seed: SeedPlacement,
) -> Result<Vec<Stmt>, String> {
    if !applicable(r, lanes) || !matches!(r.tree, Some(Tree::Pairwise | Tree::SeedThenPairwise)) {
        return Err("wave completion requires a selected pairwise participant fold".into());
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
    let mut body: Vec<_> = parts.setup.into_iter().filter(|s| {
        !matches!(&s.kind, StmtKind::Assign { target: Expr { kind: ExprKind::Var(id), .. }, .. } if leaf_ids.contains(id))
    }).collect();
    let leaves = parts
        .count
        .checked_add(i64::from(seed != SeedPlacement::AtRoot))
        .ok_or("participant leaf count overflow")?;
    let lanes = i64::from(lanes);
    let waves = leaves / lanes + i64::from(leaves % lanes != 0);
    let slots = waves / lanes + i64::from(waves % lanes != 0);
    let roots = parts
        .partial
        .iter()
        .map(|value| {
            let mut shape = value.ty.shaped().unwrap().clone();
            if slots > 1 {
                shape.shape.insert(0, Sym::constant(slots));
            }
            b.alloc(&Ty::Tile(shape), &mut body)
        })
        .collect::<Vec<_>>();
    let at = |field: &Expr, index: &Expr| {
        if slots == 1 {
            field.clone()
        } else {
            slice(field, 0, index, span)
        }
    };
    let identity = &r.step.as_ref().unwrap().identity;
    let initial = b.index();
    let initialize = roots
        .iter()
        .zip(identity)
        .map(|(field, value)| b.copy(&at(field, &initial), value))
        .collect();
    body.push(b.range(&initial, Sym::constant(slots), initialize));
    let merge = r.implementation.as_ref().unwrap();
    for parameter in merge.left.iter().chain(&merge.right).chain(&merge.output) {
        body.push(b.allocate(parameter));
    }
    let lane = b.local(Ty::Scalar(DType::I32));
    body.push(assign(
        lane.clone(),
        intrinsic(Operation::LaneIndex, vec![], DType::I32, span),
        span,
    ));
    let wave = b.index();
    let logical = b.local(Ty::Scalar(DType::I32));
    let mut fill = vec![assign(
        logical.clone(),
        binary(
            BinaryOp::Add,
            binary(BinaryOp::Mul, wave.clone(), integer(lanes, span), span),
            lane.clone(),
            span,
        ),
        span,
    )];
    for (target, value) in parts.partial.iter().zip(identity) {
        fill.push(b.copy(target, value));
    }
    let mut segment = vec![assign(
        parts.index.clone(),
        match seed {
            SeedPlacement::LeadingLeaf => {
                binary(BinaryOp::Sub, logical.clone(), integer(1, span), span)
            }
            SeedPlacement::InsertAfterSegments | SeedPlacement::AtRoot => logical.clone(),
        },
        span,
    )];
    segment.extend(parts.body);
    let valid = match seed {
        SeedPlacement::LeadingLeaf => binary(
            BinaryOp::And,
            binary(BinaryOp::Gt, logical.clone(), integer(0, span), span),
            binary(BinaryOp::Lt, logical.clone(), integer(leaves, span), span),
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
            then: segment,
            els: vec![],
        },
        span,
    ));
    match seed {
        SeedPlacement::AtRoot => {},
        SeedPlacement::LeadingLeaf => {
            let original = parts
                .partial
                .iter()
                .zip(&r.state)
                .map(|(a, c)| b.copy(a, c))
                .collect();
            fill.push(stmt(
                StmtKind::If {
                    cond: binary(BinaryOp::Eq, logical.clone(), integer(0, span), span),
                    then: original,
                    els: vec![],
                },
                span,
            ));
        }
        SeedPlacement::InsertAfterSegments => {
            // The previous wave's last segment becomes this wave's first leaf.
            // All lanes retain the same carry before any current-wave exchange.
            let carry = parts
                .partial
                .iter()
                .map(|e| b.alloc(&e.ty, &mut body))
                .collect::<Vec<_>>();
            let next = parts
                .partial
                .iter()
                .map(|e| b.alloc(&e.ty, &mut body))
                .collect::<Vec<_>>();
            let shifted = parts
                .partial
                .iter()
                .map(|e| b.alloc(&e.ty, &mut body))
                .collect::<Vec<_>>();
            for (target, value) in carry.iter().zip(&r.state) {
                body.push(b.copy(target, value));
            }
            let previous_lane = binary(
                BinaryOp::Rem,
                binary(BinaryOp::Add, lane.clone(), integer(lanes - 1, span), span),
                integer(lanes, span),
                span,
            );
            for ((shifted, next), value) in shifted.iter().zip(&next).zip(&parts.partial) {
                fill.push(shuffle_copy(&mut b, shifted, value, &previous_lane));
                fill.push(shuffle_copy(&mut b, next, value, &integer(lanes - 1, span)));
            }
            let first = shifted
                .iter()
                .zip(&carry)
                .map(|(a, c)| b.copy(a, c))
                .collect();
            fill.push(stmt(
                StmtKind::If {
                    cond: binary(BinaryOp::Eq, lane.clone(), integer(0, span), span),
                    then: first,
                    els: vec![],
                },
                span,
            ));
            for (target, value) in parts.partial.iter().zip(&shifted) {
                fill.push(b.copy(target, value));
            }
            for (target, value) in carry.iter().zip(&next) {
                fill.push(b.copy(target, value));
            }
        }
    }
    // These are exactly the lower levels of the selected pairwise tree. Wave
    // boundaries are powers of two, so no edge crosses a boundary at these levels.
    let mut stride = 1;
    while stride < lanes.min(leaves) {
        let leader = binary(
            BinaryOp::Eq,
            binary(BinaryOp::Rem, lane.clone(), integer(stride * 2, span), span),
            integer(0, span),
            span,
        );
        let valid = binary(
            BinaryOp::Lt,
            binary(BinaryOp::Add, logical.clone(), integer(stride, span), span),
            integer(leaves, span),
            span,
        );
        let source_lane = binary(
            BinaryOp::Rem,
            binary(BinaryOp::Add, lane.clone(), integer(stride, span), span),
            integer(lanes, span),
            span,
        );
        fill.extend(combine_values(
            &mut b,
            merge,
            &parts.partial,
            &parts.partial,
            &parts.partial,
            &source_lane,
            binary(BinaryOp::And, leader, valid, span),
        ));
        stride *= 2;
    }
    // Publish only after all lanes have supplied the root. A shuffle within the
    // destination-lane branch would exclude the actual source lane.
    for (target, value) in merge.left.iter().zip(&parts.partial) {
        fill.push(shuffle_copy(&mut b, target, value, &integer(0, span)));
    }
    let root_slot = binary(BinaryOp::Div, wave.clone(), integer(lanes, span), span);
    let owner = binary(BinaryOp::Rem, wave.clone(), integer(lanes, span), span);
    let publish = roots
        .iter()
        .zip(&merge.left)
        .map(|(target, value)| b.copy(&at(target, &root_slot), value))
        .collect();
    fill.push(stmt(
        StmtKind::If {
            cond: binary(BinaryOp::Eq, lane.clone(), owner, span),
            then: publish,
            els: vec![],
        },
        span,
    ));
    body.push(b.range(&wave, Sym::constant(waves), fill));
    // Complete the remaining pairwise levels over the same ordered wave roots.
    let mut stride = 1i64;
    while stride < waves {
        let slot = b.index();
        let logical = binary(
            BinaryOp::Add,
            binary(BinaryOp::Mul, slot.clone(), integer(lanes, span), span),
            lane.clone(),
            span,
        );
        let offset = stride / lanes;
        let source = if offset == 0 {
            slot.clone()
        } else {
            symbol(slot.sym.as_ref().unwrap().add(&Sym::constant(offset)), span)
        };
        let left = roots
            .iter()
            .map(|field| at(field, &slot))
            .collect::<Vec<_>>();
        let right = roots
            .iter()
            .map(|field| at(field, &source))
            .collect::<Vec<_>>();
        let lane_source = binary(
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
        let leader = binary(
            BinaryOp::Eq,
            binary(
                BinaryOp::Rem,
                logical.clone(),
                integer(
                    stride.checked_mul(2).ok_or("wave tree stride overflow")?,
                    span,
                ),
                span,
            ),
            integer(0, span),
            span,
        );
        let valid = binary(
            BinaryOp::Lt,
            binary(BinaryOp::Add, logical, integer(stride, span), span),
            integer(waves, span),
            span,
        );
        let computation = combine_values(
            &mut b,
            merge,
            &left,
            &right,
            &left,
            &lane_source,
            binary(BinaryOp::And, leader, valid, span),
        );
        body.push(b.range(&slot, Sym::constant(slots - offset), computation));
        stride = stride.checked_mul(2).ok_or("wave tree stride overflow")?;
    }
    if seed == SeedPlacement::AtRoot {
        let root = roots.iter().map(|field| at(field, &integer(0, span))).collect::<Vec<_>>();
        body.extend(combine_seed_root(&mut b, r, &root, &lane));
    }
    for (target, value) in r.state.iter().zip(&roots) {
        body.push(shuffle_copy(
            &mut b,
            target,
            &at(value, &integer(0, span)),
            &integer(0, span),
        ));
    }
    Ok(body)
}
