//! Product of independent retained folds with an identical selected schedule.
//! State fields keep their original step and merge bodies. Only read-only
//! snapshots shared by those bodies may use one prepared leaf parameter.
use super::*;
use crate::reduction::structured::{Callback, Merge, Reduction};

pub(super) fn select(
    body: &mut Vec<Stmt>,
    vars: &[Var],
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    let mut first = 0;
    while first < body.len() {
        let mut second = first + 1;
        while second < body.len() {
            let Some(mut reduction) = product(&body[first], &body[second], vars) else {
                second += 1;
                continue;
            };
            let Some((mut before, after)) = ranges::intervening(body, first, second) else {
                second += 1;
                continue;
            };
            let decision = Decision {
                kind: DecisionKind::ReductionFusion {
                    first,
                    second,
                    extent: reduction.extent().clone(),
                },
                alternatives: vec![Alternative::Separate, Alternative::Fuse].into(),
            };
            match select(&decision)? {
                Alternative::Separate => second += 1,
                Alternative::Fuse => {
                    reduction.span = body[first].span;
                    before.push(Stmt {
                        id: body[first].id,
                        span: body[first].span,
                        kind: StmtKind::Reduction(Box::new(reduction)),
                    });
                    before.extend(after);
                    body.splice(first..=second, before);
                    second = first + 1;
                }
                _ => return Err("invalid reduction fusion choice".into()),
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

fn append(left: &mut Merge, right: &Merge) {
    left.left.extend(right.left.clone());
    left.right.extend(right.right.clone());
    left.output.extend(right.output.clone());
    left.body.extend(right.body.clone());
}

fn product(a: &Stmt, b: &Stmt, vars: &[Var]) -> Option<Reduction> {
    let (StmtKind::Reduction(a_fold), StmtKind::Reduction(b_fold)) = (&a.kind, &b.kind) else {
        return None;
    };
    if !independent(a, b)
        || a_fold.axis != b_fold.axis
        || a_fold.extent() != b_fold.extent()
        || a_fold.tree.is_none()
        || a_fold.tree != b_fold.tree
        || a_fold.preparation_window != b_fold.preparation_window
        || a_fold.unroll != b_fold.unroll
        || a_fold.segment != b_fold.segment
        || a_fold.branches != b_fold.branches
        || Accesses::of(std::slice::from_ref(a)).unknown
        || Accesses::of(std::slice::from_ref(b)).unknown
    {
        return None;
    }
    // Interleaving must not change which numerical or bounds failure occurs.
    // Helpers whose checked expressions may fail remain separate.
    let mut failure = false;
    for statement in a_fold.bodies().chain(b_fold.bodies()).flatten() {
        visit_stmt(statement, &mut |e| {
            failure |= !crate::effects::expression_can_be_omitted(e);
        });
    }
    if failure {
        return None;
    }
    let (a_step, b_step) = (a_fold.step.as_ref()?, b_fold.step.as_ref()?);
    let (a_impl, b_impl) = (
        a_step.implementation.as_ref()?,
        b_step.implementation.as_ref()?,
    );
    // Named values already denote captured snapshots. More complicated operands
    // retain their evaluation order in separate reductions.
    if a_fold
        .inputs
        .iter()
        .chain(&b_fold.inputs)
        .any(|e| !matches!(e.kind, ExprKind::Var(_)))
    {
        return None;
    }
    let mut result = (**a_fold).clone();
    result.ordered |= b_fold.ordered;
    result.merge = Callback::Product;
    result.state.extend(b_fold.state.clone());
    append(
        result.implementation.as_mut()?,
        b_fold.implementation.as_ref()?,
    );
    let step = result.step.as_mut()?;
    step.call = Callback::Product;
    step.identity.extend(b_step.identity.clone());
    let implementation = step.implementation.as_mut()?;
    let mut right = b_impl.clone();
    let mut rename = HashMap::new();
    let mut keep = Vec::new();
    for (at, input) in b_fold.inputs.iter().enumerate() {
        let common = a_fold.inputs.iter().enumerate().find(|(i, old)| {
            crate::normalize::value_identity(old) == crate::normalize::value_identity(input)
                && read_only(&a_impl.right[*i], &a_impl.body)
                && read_only(&b_impl.right[at], &b_impl.body)
        });
        if let Some((i, _)) = common {
            let (ExprKind::Var(from), ExprKind::Var(to)) =
                (&b_impl.right[at].kind, &a_impl.right[i].kind)
            else {
                return None;
            };
            rename.insert(*from, *to);
        } else {
            result.inputs.push(input.clone());
            result.preparation.push(b_fold.preparation[at]);
            keep.push(right.right[at].clone());
        }
    }
    let atoms = index_substitutions(&rename, vars);
    for statement in &mut right.body {
        remap(statement, &rename, &atoms);
    }
    right.right = keep;
    append(implementation, &right);
    Some(result)
}

pub(crate) fn read_only(parameter: &Expr, body: &[Stmt]) -> bool {
    let ExprKind::Var(root) = parameter.kind else {
        return false;
    };
    let mut aliases = HashSet::from([root]);
    // Include aliases formed through nested ordinary views. Overapproximating
    // derived tile values is safe: it only retains duplicate leaf preparation.
    loop {
        let previous = aliases.len();
        fn collect(body: &[Stmt], aliases: &mut HashSet<VarId>) {
            for statement in body {
                if let StmtKind::Assign { target, value, .. } = &statement.kind {
                    if let ExprKind::Var(target) = target.kind {
                        if value.ty.shaped().is_some()
                            && aliases.iter().any(|v| mentions(value, *v))
                        {
                            aliases.insert(target);
                        }
                    }
                }
                match &statement.kind {
                    StmtKind::Range { body, .. }
                    | StmtKind::Parallel { body, .. }
                    | StmtKind::Owned { body, .. }
                    | StmtKind::Lanes { body, .. }
                    | StmtKind::LoadLoop { body, .. } => collect(body, aliases),
                    StmtKind::If { then, els, .. } => {
                        collect(then, aliases);
                        collect(els, aliases);
                    }
                    StmtKind::Reduction(r) => {
                        for body in r.bodies() {
                            collect(body, aliases);
                        }
                    }
                    _ => {}
                }
            }
        }
        collect(body, &mut aliases);
        if aliases.len() == previous {
            break;
        }
    }
    !written(body).iter().any(|v| aliases.contains(v))
}
