//! State elimination by reference: the paths between kept states of a graph
//! whose edges are terminals. Removing a state q adds, for each neighbour
//! pair p→q→r, one terminal `p→r | p→q (q→q)* q→r` that names its operands,
//! so each elimination costs output proportional to its neighbour pairs,
//! never to the size of the expressions it composes.
use crate::terminal::{Allowance, TermId, Terms};
use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
};

/// Edges between kept states. A state whose elimination the allowance
/// cannot pay for is kept as well and reported.
pub(crate) struct Eliminated {
    pub edges: Vec<(u32, u32, TermId)>,
    pub kept: Vec<u32>,
}

pub(crate) fn eliminate(
    states: u32,
    edges: impl IntoIterator<Item = (u32, u32, TermId)>,
    keep: &[bool],
    terms: &mut Terms,
    allowance: &mut Allowance,
) -> Eliminated {
    let states = states as usize;
    let mut outgoing: Vec<BTreeMap<u32, TermId>> = vec![BTreeMap::new(); states];
    let mut incoming: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); states];
    for (from, to, term) in edges {
        let merged = match outgoing[from as usize].get(&to) {
            Some(&existing) => terms.alt([existing, term]),
            None => term,
        };
        outgoing[from as usize].insert(to, merged);
        incoming[to as usize].insert(from);
    }
    let degree = |state: u32, incoming: &[BTreeSet<u32>], outgoing: &[BTreeMap<u32, TermId>]| {
        let ins = incoming[state as usize]
            .iter()
            .filter(|&&p| p != state)
            .count();
        let outs = outgoing[state as usize]
            .keys()
            .filter(|&&r| r != state)
            .count();
        ins * outs
    };
    // Cheapest first; an entry whose degree changed since it was queued is
    // requeued with the current degree.
    let mut remaining = vec![false; states];
    let mut queue = BinaryHeap::new();
    for state in (0..states as u32).filter(|&state| !keep[state as usize]) {
        remaining[state as usize] = true;
        queue.push(Reverse((degree(state, &incoming, &outgoing), state)));
    }
    let mut kept = Vec::new();
    while let Some(Reverse((queued, state))) = queue.pop() {
        if !remaining[state as usize] {
            continue;
        }
        let current = degree(state, &incoming, &outgoing);
        if current != queued {
            queue.push(Reverse((current, state)));
            continue;
        }
        remaining[state as usize] = false;
        if !allowance.spend(1 + current) {
            kept.push(state);
            continue;
        }
        let mut after = std::mem::take(&mut outgoing[state as usize]);
        let looped = after.remove(&state);
        let before = std::mem::take(&mut incoming[state as usize]);
        for target in after.keys() {
            incoming[*target as usize].remove(&state);
        }
        let star = looped.map(|looped| terms.repeat(looped, 0, None));
        let touched = before
            .iter()
            .chain(after.keys())
            .copied()
            .filter(|&neighbour| neighbour != state && remaining[neighbour as usize])
            .collect::<BTreeSet<_>>();
        for &source in before.iter().filter(|&&source| source != state) {
            let prefix = outgoing[source as usize]
                .remove(&state)
                .expect("incoming edges have outgoing counterparts");
            for (&target, &suffix) in &after {
                let path = match star {
                    Some(star) => terms.seq([prefix, star, suffix]),
                    None => terms.seq([prefix, suffix]),
                };
                let merged = match outgoing[source as usize].get(&target) {
                    Some(&existing) => terms.alt([existing, path]),
                    None => path,
                };
                outgoing[source as usize].insert(target, merged);
                incoming[target as usize].insert(source);
            }
        }
        for neighbour in touched {
            queue.push(Reverse((
                degree(neighbour, &incoming, &outgoing),
                neighbour,
            )));
        }
    }
    let edges = outgoing
        .into_iter()
        .enumerate()
        .flat_map(|(from, targets)| {
            targets
                .into_iter()
                .map(move |(to, term)| (from as u32, to, term))
        })
        .collect();
    Eliminated { edges, kept }
}
