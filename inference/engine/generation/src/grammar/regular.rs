//! Bounded language-preserving resolution of regular productions into
//! reference-free expressions. Right-linear recursive regions (scanners such
//! as "anything until a delimiter") are resolved by state elimination; other
//! non-recursive rules by inlining. Rules outside these bounds stay grammar
//! rules.
use super::{Expr, Rules};
use std::collections::{BTreeMap, BTreeSet};

type Edges = BTreeMap<String, Vec<(Vec<Expr>, Option<String>)>>;
const STATES: usize = 64;
/// Node bound for one resolved expression and for an elimination graph.
pub(super) const EXPRESSION: usize = 32 * 1024;
const GRAPH: usize = 128 * 1024;

fn regions(rules: &Rules) -> (Edges, BTreeSet<String>) {
    let mut edges = Edges::new();
    for (name, expr) in rules {
        let alternatives = match expr {
            Expr::Alternative(parts) => parts.as_slice(),
            _ => std::slice::from_ref(expr),
        };
        let mut paths = Vec::new();
        let mut valid = true;
        for alternative in alternatives {
            let mut nodes = match alternative {
                Expr::Sequence(parts) => parts.clone(),
                _ => vec![alternative.clone()],
            };
            let target = match nodes.last() {
                Some(Expr::Reference(name)) => Some(name.clone()),
                _ => None,
            };
            if target.is_some() {
                nodes.pop();
            }
            let mut references = BTreeSet::new();
            for node in &nodes {
                node.references(&mut references);
            }
            if !references.is_empty() {
                valid = false;
                break;
            }
            paths.push((nodes, target));
        }
        if valid {
            edges.insert(name.clone(), paths);
        }
    }
    loop {
        let rejected = edges
            .iter()
            .filter(|(_, paths)| {
                paths
                    .iter()
                    .any(|(_, target)| target.as_ref().is_some_and(|t| !edges.contains_key(t)))
            })
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        if rejected.is_empty() {
            break;
        }
        for name in rejected {
            edges.remove(&name);
        }
    }
    let mut entries = BTreeSet::new();
    if edges.contains_key("root") {
        entries.insert("root".into());
    }
    for (name, expr) in rules {
        if !edges.contains_key(name) {
            let mut references = BTreeSet::new();
            expr.references(&mut references);
            entries.extend(
                references
                    .into_iter()
                    .filter(|name| edges.contains_key(name)),
            );
        }
    }
    (edges, entries)
}
fn reachable(edges: &Edges, entry: &str) -> Option<BTreeSet<String>> {
    let mut region = BTreeSet::new();
    let mut pending = vec![entry.to_string()];
    while let Some(name) = pending.pop() {
        if region.insert(name.clone()) {
            if region.len() > STATES {
                return None;
            }
            pending.extend(edges[&name].iter().filter_map(|(_, target)| target.clone()));
        }
    }
    Some(region)
}
fn bounded(expr: Expr) -> Option<Expr> {
    (expr.size() <= EXPRESSION).then_some(expr)
}

/// Concatenation that drops empty parts and flattens nested sequences.
pub(super) fn sequence(parts: impl IntoIterator<Item = Expr>) -> Expr {
    let mut output = Vec::new();
    for part in parts {
        match part {
            Expr::Sequence(inner) => output.extend(inner),
            part => output.push(part),
        }
    }
    if output.len() == 1 {
        output.pop().unwrap()
    } else {
        Expr::Sequence(output)
    }
}

/// Union that keeps equal operands once and flattens nested alternatives.
pub(super) fn union(left: Option<Expr>, right: Expr) -> Expr {
    let Some(left) = left else {
        return right;
    };
    let mut parts = match left {
        Expr::Alternative(parts) => parts,
        left => vec![left],
    };
    for part in match right {
        Expr::Alternative(inner) => inner,
        right => vec![right],
    } {
        if !parts.contains(&part) {
            parts.push(part);
        }
    }
    if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        Expr::Alternative(parts)
    }
}

pub(super) fn star(expr: Expr) -> Expr {
    if expr.is_empty() {
        expr
    } else {
        Expr::Repeat(Box::new(expr), 0, None)
    }
}

/// Which states elimination may leave in place.
pub(super) enum Keep<'a> {
    /// Eliminate every state.
    Nothing,
    /// Keep a state whose elimination would grow the graph by more than
    /// `growth` nodes (it joins many long paths), unless it is `forced`.
    Hubs {
        growth: usize,
        forced: &'a BTreeSet<usize>,
    },
}

/// State elimination over a labeled graph: states in `states` are removed
/// cheapest first, and each path through a removed state becomes an edge
/// between its remaining neighbours. Returns the graph and the states kept.
pub(super) fn eliminate_states(
    graph: BTreeMap<(usize, usize), Expr>,
    states: impl IntoIterator<Item = usize>,
    keep: Keep<'_>,
    budget: usize,
) -> Option<(BTreeMap<(usize, usize), Expr>, BTreeSet<usize>)> {
    type Outgoing = BTreeMap<usize, BTreeMap<usize, Expr>>;
    type Incoming = BTreeMap<usize, BTreeSet<usize>>;
    let mut outgoing = Outgoing::new();
    let mut incoming = Incoming::new();
    let mut size = 0;
    for ((from, to), expr) in graph {
        size += expr.size();
        incoming.entry(to).or_default().insert(from);
        outgoing.entry(from).or_default().insert(to, expr);
    }
    // Nodes added by eliminating `state` minus nodes removed with it.
    let growth = |state: usize, outgoing: &Outgoing, incoming: &Incoming| -> i64 {
        let edges = outgoing.get(&state);
        let looped = edges
            .and_then(|edges| edges.get(&state))
            .map_or(0, Expr::size);
        let outs = edges.map_or(Vec::new(), |edges| {
            edges
                .iter()
                .filter(|(to, _)| **to != state)
                .map(|(_, expr)| expr.size())
                .collect()
        });
        let ins = incoming.get(&state).map_or(Vec::new(), |sources| {
            sources
                .iter()
                .filter(|from| **from != state)
                .map(|from| outgoing[from][&state].size())
                .collect()
        });
        let (count_in, count_out) = (ins.len() as i64, outs.len() as i64);
        let (size_in, size_out) = (ins.iter().sum::<usize>() as i64, outs.iter().sum::<usize>() as i64);
        let repeat = if looped > 0 { looped as i64 + 1 } else { 0 };
        count_out * size_in + count_in * size_out + count_in * count_out * repeat
            - size_in
            - size_out
            - looped as i64
    };
    let mut remaining = states.into_iter().collect::<BTreeSet<_>>();
    let mut kept = BTreeSet::new();
    let mut queue = remaining
        .iter()
        .map(|&state| std::cmp::Reverse((growth(state, &outgoing, &incoming), state)))
        .collect::<std::collections::BinaryHeap<_>>();
    while let Some(std::cmp::Reverse((cost, state))) = queue.pop() {
        if !remaining.contains(&state) {
            continue;
        }
        let current = growth(state, &outgoing, &incoming);
        if current != cost {
            queue.push(std::cmp::Reverse((current, state)));
            continue;
        }
        remaining.remove(&state);
        if let Keep::Hubs { growth, forced } = keep {
            if current > growth as i64 && !forced.contains(&state) {
                kept.insert(state);
                continue;
            }
        }
        let mut after = outgoing.remove(&state).unwrap_or_default();
        let looped = after.remove(&state);
        size -= looped.as_ref().map_or(0, Expr::size);
        let repeat = looped.map_or(Expr::Sequence(Vec::new()), star);
        let sources = incoming.remove(&state).unwrap_or_default();
        for target in after.keys() {
            incoming.get_mut(target).unwrap().remove(&state);
        }
        let mut touched = after.keys().copied().collect::<BTreeSet<_>>();
        for &source in sources.iter().filter(|&&source| source != state) {
            touched.insert(source);
            let before = outgoing.get_mut(&source).unwrap().remove(&state).unwrap();
            size -= before.size();
            for (&target, suffix) in &after {
                let path = bounded(sequence([before.clone(), repeat.clone(), suffix.clone()]))?;
                let edges = outgoing.entry(source).or_default();
                let previous = edges.remove(&target);
                size -= previous.as_ref().map_or(0, Expr::size);
                let combined = bounded(union(previous, path))?;
                size += combined.size();
                edges.insert(target, combined);
                incoming.entry(target).or_default().insert(source);
            }
        }
        size -= after.values().map(Expr::size).sum::<usize>();
        if size > budget {
            return None;
        }
        for neighbour in touched.into_iter().filter(|s| remaining.contains(s)) {
            queue.push(std::cmp::Reverse((
                growth(neighbour, &outgoing, &incoming),
                neighbour,
            )));
        }
    }
    let graph = outgoing
        .into_iter()
        .flat_map(|(from, edges)| edges.into_iter().map(move |(to, expr)| ((from, to), expr)))
        .collect();
    Some((graph, kept))
}

fn eliminate(edges: &Edges, entry: &str) -> Option<Expr> {
    let region = reachable(edges, entry)?;
    let ids: BTreeMap<_, _> = region
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i + 2))
        .collect();
    let mut graph =
        BTreeMap::<(usize, usize), Expr>::from([((0, ids[entry]), Expr::Sequence(Vec::new()))]);
    for name in &region {
        for (nodes, target) in &edges[name] {
            let key = (ids[name], target.as_ref().map_or(1, |t| ids[t]));
            let previous = graph.remove(&key);
            graph.insert(key, union(previous, sequence(nodes.iter().cloned())));
        }
    }
    let (mut graph, _) = eliminate_states(graph, ids.values().copied(), Keep::Nothing, GRAPH)?;
    graph.remove(&(0, 1)).and_then(bounded)
}

/// Every rule whose language is regular within the bounds, as a
/// reference-free expression.
pub(super) fn regular_rules(rules: &Rules) -> BTreeMap<String, Expr> {
    let (edges, entries) = regions(rules);
    let mut resolved = BTreeMap::new();
    for entry in entries {
        let expr = eliminate(&edges, &entry);
        resolved.insert(entry, expr);
    }
    fn rule(
        name: &str,
        rules: &Rules,
        resolved: &mut BTreeMap<String, Option<Expr>>,
        visiting: &mut BTreeSet<String>,
    ) -> Option<Expr> {
        if let Some(expr) = resolved.get(name) {
            return expr.clone();
        }
        if visiting.len() > super::MAX_DEPTH || !visiting.insert(name.into()) {
            return None;
        }
        let expr = expression(&rules[name], rules, resolved, visiting);
        visiting.remove(name);
        // A failure below a rule that is still being visited may be caused
        // by that ancestor's recursion, so only complete resolutions of an
        // unvisited stack are final.
        if expr.is_some() || visiting.is_empty() {
            resolved.insert(name.into(), expr.clone());
        }
        expr
    }
    fn expression(
        expr: &Expr,
        rules: &Rules,
        resolved: &mut BTreeMap<String, Option<Expr>>,
        visiting: &mut BTreeSet<String>,
    ) -> Option<Expr> {
        match expr {
            Expr::Reference(name) => rule(name, rules, resolved, visiting),
            Expr::Sequence(parts) => bounded(sequence(
                parts
                    .iter()
                    .map(|p| expression(p, rules, resolved, visiting))
                    .collect::<Option<Vec<_>>>()?,
            )),
            Expr::Alternative(parts) => {
                let mut result = None;
                for part in parts {
                    result = Some(union(result, expression(part, rules, resolved, visiting)?));
                }
                bounded(result?)
            }
            Expr::Repeat(part, min, max) => bounded(Expr::Repeat(
                Box::new(expression(part, rules, resolved, visiting)?),
                *min,
                *max,
            )),
            _ => Some(expr.clone()),
        }
    }
    for name in rules.keys() {
        rule(name, rules, &mut resolved, &mut BTreeSet::new());
    }
    resolved
        .into_iter()
        .filter_map(|(name, expr)| expr.map(|expr| (name, expr)))
        .collect()
}

/// Left-orient right-linear scanner regions of a grammar rendered without
/// lexical compilation, so Earley recognition of long scans stays linear.
pub(super) fn orient(rules: &mut Rules) {
    let (edges, entries) = regions(rules);
    let mut added = 0;
    for (index, entry) in entries.into_iter().enumerate() {
        let Some(region) = reachable(&edges, &entry) else {
            continue;
        };
        if added + region.len() > 4096 {
            continue;
        }
        let mut remaining = region.clone();
        while !remaining.is_empty() {
            let leaves = remaining
                .iter()
                .filter(|name| {
                    edges[*name]
                        .iter()
                        .all(|(_, target)| target.as_ref().is_none_or(|t| !remaining.contains(t)))
                })
                .cloned()
                .collect::<Vec<_>>();
            if leaves.is_empty() {
                break;
            }
            for leaf in leaves {
                remaining.remove(&leaf);
            }
        }
        if remaining.is_empty() {
            continue;
        }
        let names: BTreeMap<_, _> = region
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), format!("scan{index}state{i}")))
            .collect();
        let mut incoming: BTreeMap<_, Vec<Expr>> =
            region.iter().map(|n| (n.clone(), Vec::new())).collect();
        incoming
            .get_mut(&entry)
            .unwrap()
            .push(Expr::Sequence(Vec::new()));
        let mut endings = Vec::new();
        for name in &region {
            for (nodes, target) in &edges[name] {
                let mut path = vec![Expr::Reference(names[name].clone())];
                path.extend(nodes.clone());
                match target {
                    Some(target) => incoming.get_mut(target).unwrap().push(Expr::Sequence(path)),
                    None => endings.push(Expr::Sequence(path)),
                }
            }
        }
        if endings.is_empty() {
            continue;
        }
        for name in &region {
            rules.insert(
                names[name].clone(),
                Expr::Alternative(incoming.remove(name).unwrap()),
            );
        }
        rules.insert(entry, Expr::Alternative(endings));
        added += region.len();
    }
    let mut reachable = BTreeSet::new();
    let mut pending = vec!["root".to_string()];
    while let Some(name) = pending.pop() {
        if reachable.insert(name.clone()) {
            let mut refs = BTreeSet::new();
            rules[&name].references(&mut refs);
            pending.extend(refs);
        }
    }
    rules.retain(|name, _| reachable.contains(name));
}
