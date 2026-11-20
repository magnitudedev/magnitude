//! Conditional independence from the complete residual factor scopes.
use crate::model::{Factor, VarId};
use crate::Domain;
use std::collections::BTreeMap;

pub(crate) fn components(
    factor_scopes: &[Vec<VarId>],
    ids: &[usize],
    domains: &[Domain],
) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..domains.len()).collect();
    fn root(parent: &mut [usize], mut v: usize) -> usize {
        while parent[v] != v {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        v
    }
    let mut first_active = Vec::with_capacity(ids.len());
    for &id in ids {
        let mut active_vars = factor_scopes[id]
            .iter()
            .copied()
            .filter(|v| !domains[v.0].is_singleton());
        let first = active_vars.next();
        if let Some(first) = first {
            for v in active_vars {
                let a = root(&mut parent, first.0);
                let b = root(&mut parent, v.0);
                parent[b] = a;
            }
        }
        first_active.push(first);
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&id, first) in ids.iter().zip(first_active) {
        // A zero-scope coverage obligation is its own child, not erased.
        let group = first.map_or(domains.len() + id, |v| root(&mut parent, v.0));
        groups.entry(group).or_default().push(id);
    }
    groups.into_values().collect()
}

pub(crate) fn scope(factor_scopes: &[Vec<VarId>], ids: &[usize]) -> Vec<VarId> {
    let mut scope = Vec::new();
    for &id in ids {
        scope.extend_from_slice(&factor_scopes[id]);
    }
    scope.sort_unstable();
    scope.dedup();
    scope
}

pub(crate) fn branch_variable(
    factors: &[Factor],
    factor_scopes: &[Vec<VarId>],
    ids: &[usize],
    domains: &[Domain],
) -> Option<VarId> {
    // Resolve conditional construction before its private choices. Complete
    // scopes still couple all possible effects until an alternative is fixed.
    let mut guards = BTreeMap::<VarId, usize>::new();
    for &id in ids {
        let factor = &factors[id];
        if factor
            .guards
            .iter()
            .any(|g| !domains[g.variable.0].contains(g.value))
        {
            continue;
        }
        for guard in &factor.guards {
            if !domains[guard.variable.0].is_singleton() {
                *guards.entry(guard.variable).or_default() += 1;
            }
        }
    }
    if let Some((&variable, _)) = guards
        .iter()
        .max_by_key(|(v, n)| (**n, std::cmp::Reverse(**v)))
    {
        return Some(variable);
    }
    let mut degree = BTreeMap::<VarId, usize>::new();
    for &id in ids {
        for &v in &factor_scopes[id] {
            if !domains[v.0].is_singleton() {
                *degree.entry(v).or_default() += 1;
            }
        }
    }
    let maximum = degree.values().copied().max()?;
    let peers: Vec<_> = degree
        .into_iter()
        .filter(|(_, d)| *d == maximum)
        .map(|(v, _)| v)
        .collect();
    // Stable middle separator among equally connected variables. No domain score
    // or source identity changes admissibility or bound-based exclusion.
    peers.get(peers.len() / 2).copied()
}
