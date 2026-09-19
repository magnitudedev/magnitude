//! Local, retained topology alternatives from the same legality proofs used by
//! concrete composition. No selector callback or recursive path walk is used.
use super::*;

pub(crate) fn project_producer(
    view: &Expr, target: &Expr, variable: VarId, indices: &[VarId], value: &Expr, vars: &mut Vec<Var>,
) -> Result<Option<Vec<Stmt>>, String> {
    producers::pointwise_projection(view, target, variable, indices, value, vars)
}

pub(crate) struct Fusion {
    pub domain: Decision,
    pub arms: Vec<Vec<Stmt>>,
    pub coverage: Vec<(usize, String)>,
}

/// The ordinary parallel-composition rule forbids publication to externally
/// visible roots. This rejection is independent of loop widths, layouts and
/// other local implementation choices, so retained callers can prove it before
/// closing their bodies. Unknown effects alone are not an inapplicability proof.
pub(crate) fn forbids_parallel_fusion(function: &LoweredIr, body: &[Stmt]) -> bool {
    let private = function.ownership.roots(function);
    Accesses::of(body).accesses.iter().any(|access|
        access.write && !private.contains(&access.view.root))
}

/// Preserve variable identities across local arms. A type whose index changes
/// belongs to that alternative's lexical storage, so clone just that binding.
fn localize(body: &mut [Stmt], vars: &mut Vec<Var>, atoms: &[(Atom, Sym)], outside: &HashSet<VarId>) -> Result<(), String> {
    let mut rename = HashMap::new();
    let mut local = used(body); local.extend(written(body));
    let existing = vars.len();
    for id in 0..existing {
        if !local.contains(&id) { continue; }
        let mut ty = vars[id].ty.clone(); map_ty(&mut ty, atoms);
        if ty != vars[id].ty && matches!(vars[id].kind, VarKind::Local) {
            if outside.contains(&id) {
                return Err(format!("fusion changes the type of variable {id} across its local region boundary"));
            }
            let mut variable = vars[id].clone(); variable.ty = ty;
            let new = vars.len(); variable.name = format!("{}_variant_{new}", variable.name);
            vars.push(variable); rename.insert(id, new);
        }
    }
    for statement in body { remap(statement, &rename, atoms); }
    Ok(())
}

fn outside(body: &[Stmt], first: usize, second: usize) -> HashSet<VarId> {
    let mut variables = used(&body[..first]);
    variables.extend(written(&body[..first]));
    variables.extend(used(&body[second + 1..]));
    variables.extend(written(&body[second + 1..]));
    variables
}

pub(crate) fn range(body: &[Stmt], vars: &mut Vec<Var>, first: usize, second: usize) -> Option<Fusion> {
    let (lo, hi, mut joined, atoms) = ranges::join(body, vars, first, second)?;
    let coverage = localize(&mut joined, vars, &atoms, &outside(body, first, second)).err().map(|reason| (1, reason)).into_iter().collect();
    Some(Fusion { domain: Decision { kind: DecisionKind::RangeFusion { first, second, lo, hi },
        alternatives: vec![Alternative::Separate, Alternative::Fuse].into() },
        arms: vec![body[first..=second].to_vec(), joined], coverage })
}

pub(crate) fn stream(body: &[Stmt], vars: &mut Vec<Var>, first: usize, second: usize) -> Option<Fusion> {
    let (extent, mut joined, atoms) = join_streams(body, vars, first, second).or_else(|| join_ranges(body, vars, first, second))?;
    let coverage = localize(&mut joined, vars, &atoms, &outside(body, first, second)).err().map(|reason| (1, reason)).into_iter().collect();
    Some(Fusion { domain: Decision { kind: DecisionKind::StreamFusion { first, second, extent },
        alternatives: vec![Alternative::Separate, Alternative::Fuse].into() },
        arms: vec![body[first..=second].to_vec(), joined], coverage })
}

pub(crate) fn parallel(function: &LoweredIr, vars: &mut Vec<Var>, first: usize, second: usize) -> Option<Fusion> {
    let (StmtKind::Parallel { extents: left, .. }, StmtKind::Parallel { extents: right, .. }) = (&function.body[first].kind, &function.body[second].kind) else { return None; };
    let mut variants = vec![(false, function.clone())];
    if left.len() == right.len() + 1 && left[..right.len()] == *right {
        let mut phase = function.clone(); phase.body = vec![function.body[second].clone()];
        if let Ok(partition) = crate::partition::pointwise(&phase, 1) {
            if matches!(&partition.function.body[0].kind, StmtKind::Parallel { extents, .. } if extents == left) {
                let mut candidate = function.clone(); candidate.vars = partition.function.vars; candidate.body[second] = partition.function.body[0].clone();
                variants.push((true, candidate));
            }
        }
    }
    let private = function.ownership.roots(function);
    let mut alternatives = vec![Alternative::Separate];
    let mut arms = vec![function.body[first..=second].to_vec()];
    let mut coverage = Vec::new();
    let outside = outside(&function.body, first, second);
    for (refine, candidate) in variants {
        let StmtKind::Parallel { extents, .. } = &candidate.body[second].kind else { unreachable!() };
        let shared = left.iter().zip(extents).take_while(|(a, b)| a == b).count();
        for prefix in 0..=shared {
            let Some((phase, atoms)) = join_parallel(&candidate, first, second, prefix, &private) else { continue; };
            let mut body = vec![phase]; body.extend_from_slice(&candidate.body[first + 1..second]);
            let mut rename = HashMap::new();
            for id in function.vars.len()..candidate.vars.len() {
                rename.insert(id, vars.len()); vars.push(candidate.vars[id].clone());
            }
            for statement in &mut body { remap(statement, &rename, &[]); }
            if let Err(reason) = localize(&mut body, vars, &atoms, &outside) { coverage.push((arms.len(), reason)); }
            alternatives.push(Alternative::ParallelFusion { shared_axes: prefix, refine_consumer: refine }); arms.push(body);
        }
    }
    (alternatives.len() > 1).then_some(Fusion { domain: Decision { kind: DecisionKind::ParallelFusion {
        boundary: first, other: second, left_domain: left.clone(), right_domain: right.clone(),
    }, alternatives: alternatives.into() }, arms, coverage })
}
