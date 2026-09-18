//! Fuse independent serial regions over the same checked constant interval.
//! This exposes common operand preparation across ordinary library calls while
//! preserving each region's iteration order and typed intrinsic effects.
use super::*;

pub(super) fn select(
    body: &mut Vec<Stmt>,
    vars: &mut [Var],
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    let mut first = 0;
    while first < body.len() {
        let mut second = first + 1;
        while second < body.len() {
            let Some((lo, hi, replacement, atoms)) = join(body, vars, first, second) else {
                second += 1;
                continue;
            };
            let decision = Decision {
                kind: DecisionKind::RangeFusion {
                    first,
                    second,
                    lo,
                    hi,
                },
                alternatives: vec![Alternative::Separate, Alternative::Fuse].into(),
            };
            match select(&decision)? {
                Alternative::Separate => second += 1,
                Alternative::Fuse => {
                    body.splice(first..=second, replacement);
                    for variable in vars.iter_mut() {
                        map_ty(&mut variable.ty, &atoms);
                    }
                    // Initializers may have moved before the fused region.
                    // Revisit the replaced interval to find its new position.
                    second = first + 1;
                }
                _ => return Err("invalid serial region fusion choice".into()),
            }
        }
        first += 1;
    }
    for statement in body {
        let mut error = None;
        nested_mut(statement, &mut |nested| {
            if error.is_none() {
                error = self::select(nested, vars, select).err();
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
    }
    Ok(())
}

type Joined = (Sym, Sym, Vec<Stmt>, Vec<(Atom, Sym)>);
fn join(body: &[Stmt], vars: &[Var], first: usize, second: usize) -> Option<Joined> {
    let (
        StmtKind::Range {
            var: a,
            lo,
            hi,
            body: left,
        },
        StmtKind::Range {
            var: b,
            lo: blo,
            hi: bhi,
            body: right,
        },
    ) = (&body[first].kind, &body[second].kind)
    else {
        return None;
    };
    // Runtime endpoint capture is not reordered by this cover. Static bounds
    // already have their complete checked meaning in the owning IR.
    if lo != blo
        || hi != bhi
        || lo.as_constant().is_none()
        || hi.as_constant().is_none()
        || !independent(&body[first], &body[second])
        || Accesses::of(left).unknown
        || Accesses::of(right).unknown
    {
        return None;
    }
    let (mut before, after) = intervening(body, first, second)?;
    let rename = HashMap::from([(*b, *a)]);
    let atoms = index_substitutions(&rename, vars);
    let mut merged = left.clone();
    let mut right = right.clone();
    for statement in &mut right {
        remap(statement, &rename, &atoms);
    }
    merged.extend(right);
    let mut region = body[first].clone();
    region.kind = StmtKind::Range {
        var: *a,
        lo: lo.clone(),
        hi: hi.clone(),
        body: merged,
    };
    before.push(region);
    before.extend(after);
    Some((lo.clone(), hi.clone(), before, atoms))
}

/// Move prerequisites across the first independent operation and remaining
/// statements across the second without crossing a dependency or external effect.
pub(super) fn intervening(
    body: &[Stmt],
    first: usize,
    second: usize,
) -> Option<(Vec<Stmt>, Vec<Stmt>)> {
    let (ar, aw, br, bw) = (
        used(std::slice::from_ref(&body[first])),
        written(std::slice::from_ref(&body[first])),
        used(std::slice::from_ref(&body[second])),
        written(std::slice::from_ref(&body[second])),
    );
    let mut before = Vec::new();
    let mut after = Vec::new();
    // Move prerequisites of the second region before the first only when they
    // commute with the first region and every statement left after it.
    // Walk backwards so prerequisites of prerequisites move with their users.
    let mut needed = br.clone();
    let mut move_before = vec![false; second - first - 1];
    for (at, statement) in body[first + 1..second].iter().enumerate().rev() {
        let reads = used(std::slice::from_ref(statement));
        let writes = written(std::slice::from_ref(statement));
        if writes.iter().any(|v| needed.contains(v)) {
            move_before[at] = true;
            needed.extend(reads);
        }
    }
    for (at, statement) in body[first + 1..second].iter().enumerate() {
        if crate::effects::tensor_effect(statement)
            || Accesses::of(std::slice::from_ref(statement)).unknown
        {
            return None;
        }
        let reads = used(std::slice::from_ref(statement));
        let writes = written(std::slice::from_ref(statement));
        if move_before[at] {
            if reads.iter().any(|v| aw.contains(v))
                || writes.iter().any(|v| ar.contains(v) || aw.contains(v))
                || after.iter().any(|s| !independent(s, statement))
            {
                return None;
            }
            before.push(statement.clone());
        } else {
            if reads.iter().any(|v| bw.contains(v))
                || writes.iter().any(|v| br.contains(v) || bw.contains(v))
            {
                return None;
            }
            after.push(statement.clone());
        }
    }
    Some((before, after))
}
