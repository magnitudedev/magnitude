//! Reuse equal, complete pure tile producers at one lexical scope. This extends
//! snapshot sharing to preparation/conversion, preserving the producer's stored
//! precision and selecting its longer lifetime through the existing domain.
use super::super::*;

struct Producer {
    output: VarId,
    indices: Vec<VarId>,
    definition: Vec<Stmt>,
    dependencies: HashSet<VarId>,
}

pub(in crate::composition) fn select(
    f: &mut LoweredIr,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    let uses = counts(&f.body);
    block(&mut f.body, &f.vars, &uses, select)
}

fn counts(body: &[Stmt]) -> HashMap<VarId, usize> {
    let mut counts = HashMap::new();
    for statement in body {
        visit_stmt(statement, &mut |e| {
            walk(e, &mut |e| {
                if let ExprKind::Var(id) = e.kind {
                    *counts.entry(id).or_default() += 1;
                }
            })
        });
    }
    counts
}

fn producer(
    body: &[Stmt],
    at: usize,
    vars: &[Var],
    global: &HashMap<VarId, usize>,
) -> Option<Producer> {
    let StmtKind::Assign {
        target,
        op: AssignOp::Assign,
        value: Expr {
            kind: ExprKind::TileAlloc { .. },
            ..
        },
    } = &body.get(at)?.kind
    else {
        return None;
    };
    let ExprKind::Var(output) = target.kind else {
        return None;
    };
    if !matches!(vars[output].kind, VarKind::Local) {
        return None;
    }
    let StmtKind::Owned {
        vars: indices,
        tile,
        body: source_definition,
    } = &body.get(at + 1)?.kind
    else {
        return None;
    };
    if !matches!(tile.kind, ExprKind::Var(id) if id == output) {
        return None;
    }
    let (target, definition) =
        crate::lower::producer_definition(source_definition, output, vars, &body[..at], &body[at + 2..])?;
    let ExprKind::Index { indices: at, .. } = target.kind else {
        return None;
    };
    if at.len() != indices.len() || !at.iter().zip(indices).all(|(at, &id)| {
        matches!(at, Index::Point(e) if e.sym.is_some() && e.sym == variable(id, vars).sym)
    }) { return None; }
    fn previous_value(body: &[Stmt], output: VarId) -> bool {
        body.iter().any(|s| match &s.kind {
            StmtKind::Assign { value, .. } => mentions(value, output),
            StmtKind::If { cond, then, els } => mentions(cond, output) || previous_value(then, output) || previous_value(els, output),
            _ => true,
        })
    }
    if previous_value(&definition, output) { return None; }

    // The extraction above resolves pure scalar temporaries. Their writes must
    // not escape through an ancestor or another branch of the enclosing entry.
    let local = counts(source_definition);
    for id in written(source_definition).into_iter().filter(|&id| id != output) {
        if global.get(&id) != local.get(&id) {
            return None;
        }
    }
    let mut dependencies = HashSet::new();
    for statement in &definition {
        visit_stmt(statement, &mut |e| walk(e, &mut |e| {
            if let ExprKind::Var(id) = e.kind { dependencies.insert(id); }
        }));
    }
    dependencies.remove(&output);
    for id in indices {
        dependencies.remove(id);
    }
    Some(Producer {
        output,
        indices: indices.clone(),
        definition,
        dependencies,
    })
}

fn block(
    body: &mut Vec<Stmt>,
    vars: &[Var],
    global: &HashMap<VarId, usize>,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    for statement in body.iter_mut() {
        let mut error = None;
        nested_mut(statement, &mut |nested| {
            if error.is_none() {
                error = block(nested, vars, global, select).err();
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
    }
    let mut first = 0;
    while first + 1 < body.len() {
        let Some(left) = producer(body, first, vars, global) else {
            first += 1;
            continue;
        };
        // Reusing an immutable stored result must not turn the second snapshot
        // into an alias of subsequently mutated storage.
        if body[first + 2..]
            .iter()
            .any(|s| crate::effects::tile_mutated(s, left.output))
        {
            first += 2;
            continue;
        }
        let mut second = first + 2;
        while second + 1 < body.len() {
            if crate::effects::tensor_effect(&body[second])
                || left
                    .dependencies
                    .iter()
                    .any(|&v| crate::effects::tile_mutated(&body[second], v))
            {
                break;
            }
            let Some(right) = producer(body, second, vars, global) else {
                second += 1;
                continue;
            };
            let lexical = counts(body);
            if vars[left.output].ty != vars[right.output].ty
                || left.indices.len() != right.indices.len()
                || global.get(&right.output) != lexical.get(&right.output)
                || body[second + 2..]
                    .iter()
                    .any(|s| crate::effects::tile_mutated(s, right.output))
            {
                second += 1;
                continue;
            }
            let mut rename = right
                .indices
                .iter()
                .copied()
                .zip(left.indices.iter().copied())
                .collect::<HashMap<_, _>>();
            rename.insert(right.output, left.output);
            let atoms = index_substitutions(&rename, vars);
            let mut definition = right.definition.clone();
            for statement in &mut definition { remap(statement, &rename, &atoms); }
            if !same_definition(&left.definition, &definition) {
                second += 1;
                continue;
            }
            let domain = Decision {
                kind: DecisionKind::Intermediate {
                    variable: right.output,
                    publication: second,
                },
                alternatives: vec![Alternative::Materialize, Alternative::RetainLocal].into(),
            };
            match select(&domain)? {
                Alternative::Materialize => second += 2,
                Alternative::RetainLocal => {
                    let rename = HashMap::from([(right.output, left.output)]);
                    for statement in &mut body[second + 2..] {
                        remap(statement, &rename, &[]);
                    }
                    body.drain(second..second + 2);
                }
                _ => return Err("invalid shared value producer choice".into()),
            }
        }
        first += 2;
    }
    Ok(())
}

fn same_definition(left: &[Stmt], right: &[Stmt]) -> bool {
    use crate::normalize::value_identity;
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| match (&a.kind, &b.kind) {
        (StmtKind::Assign { target: at, op: ao, value: av }, StmtKind::Assign { target: bt, op: bo, value: bv }) => ao == bo && value_identity(at) == value_identity(bt) && value_identity(av) == value_identity(bv),
        (StmtKind::If { cond: ac, then: at, els: ae }, StmtKind::If { cond: bc, then: bt, els: be }) => value_identity(ac) == value_identity(bc) && same_definition(at, bt) && same_definition(ae, be),
        _ => false,
    })
}
