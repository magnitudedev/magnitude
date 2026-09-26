//! Conditional independence from the complete residual factor scopes.
use crate::model::{Factor, VarId};
use crate::Domain;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn components(factors: &[Factor], ids: &[usize], domains: &[Domain]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..domains.len()).collect();
    fn root(parent: &mut [usize], mut v: usize) -> usize {
        while parent[v] != v {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        v
    }
    let mut scopes = Vec::with_capacity(ids.len());
    for &id in ids {
        let vars: Vec<_> = factors[id]
            .scope()
            .into_iter()
            .filter(|v| !domains[v.0].is_singleton())
            .collect();
        if let Some(first) = vars.first() {
            for v in &vars[1..] {
                let a = root(&mut parent, first.0);
                let b = root(&mut parent, v.0);
                parent[b] = a;
            }
        }
        scopes.push(vars);
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&id, scope) in ids.iter().zip(scopes) {
        // A zero-scope coverage obligation is its own child, not erased.
        let group = scope
            .first()
            .map_or(domains.len() + id, |v| root(&mut parent, v.0));
        groups.entry(group).or_default().push(id);
    }
    groups.into_values().collect()
}

pub(crate) fn scope(factors: &[Factor], ids: &[usize]) -> Vec<VarId> {
    ids.iter()
        .flat_map(|&id| factors[id].scope())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn branch_variable(
    factors: &[Factor],
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
        for v in factors[id].scope() {
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
