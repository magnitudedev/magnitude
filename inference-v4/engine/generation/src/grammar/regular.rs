//! Bounded language-preserving transformations of regular scanner productions.
//! Partial regex substitution is deliberately avoided: greedy lexeme boundaries
//! could remove valid CFG parses. Only the entire completion becomes one lexeme.
use super::{Expr, Rules};
use std::collections::{BTreeMap, BTreeSet};

type Edges = BTreeMap<String, Vec<(Vec<Expr>, Option<String>)>>;
const STATES: usize = 64;
const EXPRESSION: usize = 256 * 1024;
const GRAPH: usize = 1024 * 1024;

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
fn bounded(text: String) -> Option<String> {
    (text.len() <= EXPRESSION).then_some(text)
}
fn sequence(parts: impl IntoIterator<Item = String>) -> Option<String> {
    let mut output = String::new();
    for part in parts {
        if part == "\"\"" {
            continue;
        }
        if !output.is_empty() {
            output.push(' ');
        }
        output.push_str(&part);
        if output.len() > EXPRESSION {
            return None;
        }
    }
    if output.is_empty() {
        output = "\"\"".into();
    }
    Some(output)
}
fn union(left: Option<&String>, right: &str) -> Option<String> {
    match left {
        None => bounded(right.to_owned()),
        Some(left) if left == right => Some(left.clone()),
        Some(left) => bounded(format!("({left} | {right})")),
    }
}
fn eliminate(edges: &Edges, entry: &str) -> Option<String> {
    let region = reachable(edges, entry)?;
    let ids: BTreeMap<_, _> = region
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i + 2))
        .collect();
    let mut graph = BTreeMap::<(usize, usize), String>::from([((0, ids[entry]), "\"\"".into())]);
    let mut size = 2usize;
    fn add(
        graph: &mut BTreeMap<(usize, usize), String>,
        size: &mut usize,
        a: usize,
        b: usize,
        text: String,
    ) -> Option<()> {
        let previous = graph.get(&(a, b));
        let combined = union(previous, &text)?;
        *size = *size - previous.map_or(0, String::len) + combined.len();
        if *size > GRAPH {
            return None;
        }
        graph.insert((a, b), combined);
        Some(())
    }
    for name in &region {
        for (nodes, target) in &edges[name] {
            let text = sequence(nodes.iter().map(Expr::render))?;
            add(
                &mut graph,
                &mut size,
                ids[name],
                target.as_ref().map_or(1, |t| ids[t]),
                text,
            )?;
        }
    }
    let mut remaining = ids.values().copied().collect::<BTreeSet<_>>();
    while !remaining.is_empty() {
        let state = *remaining.iter().min_by_key(|&&state| {
            let incoming = graph
                .keys()
                .filter(|&&(a, b)| b == state && a != state)
                .count();
            let outgoing = graph
                .keys()
                .filter(|&&(a, b)| a == state && b != state)
                .count();
            (incoming * outgoing, state)
        })?;
        let incoming = graph
            .iter()
            .filter(|((a, b), _)| *b == state && *a != state)
            .map(|((a, _), s)| (*a, s.clone()))
            .collect::<Vec<_>>();
        let outgoing = graph
            .iter()
            .filter(|((a, b), _)| *a == state && *b != state)
            .map(|((_, b), s)| (*b, s.clone()))
            .collect::<Vec<_>>();
        let repeat = match graph.get(&(state, state)) {
            Some(text) if text != "\"\"" => bounded(format!("({text})*"))?,
            _ => "\"\"".into(),
        };
        for (source, before) in &incoming {
            for (target, after) in &outgoing {
                let text = sequence([before.clone(), repeat.clone(), after.clone()])?;
                add(&mut graph, &mut size, *source, *target, text)?;
            }
        }
        graph.retain(|&(a, b), _| a != state && b != state);
        size = graph.values().map(String::len).sum();
        remaining.remove(&state);
    }
    graph.remove(&(0, 1))
}

pub(super) fn whole_completion(rules: &Rules) -> Option<String> {
    let (edges, entries) = regions(rules);
    let mut resolved = BTreeMap::new();
    for entry in entries {
        resolved.insert(entry.clone(), eliminate(&edges, &entry)?);
    }
    fn rule(
        name: &str,
        rules: &Rules,
        resolved: &mut BTreeMap<String, String>,
        visiting: &mut BTreeSet<String>,
    ) -> Option<String> {
        if let Some(text) = resolved.get(name) {
            return Some(text.clone());
        }
        if visiting.len() > super::MAX_DEPTH || !visiting.insert(name.into()) {
            return None;
        }
        let text = expression(&rules[name], rules, resolved, visiting)?;
        visiting.remove(name);
        resolved.insert(name.into(), text.clone());
        Some(text)
    }
    fn expression(
        expr: &Expr,
        rules: &Rules,
        resolved: &mut BTreeMap<String, String>,
        visiting: &mut BTreeSet<String>,
    ) -> Option<String> {
        match expr {
            Expr::Reference(name) => rule(name, rules, resolved, visiting),
            Expr::Sequence(parts) => sequence(
                parts
                    .iter()
                    .map(|p| expression(p, rules, resolved, visiting))
                    .collect::<Option<Vec<_>>>()?,
            ),
            Expr::Alternative(parts) => {
                let mut result = None;
                for part in parts {
                    result = Some(union(
                        result.as_ref(),
                        &expression(part, rules, resolved, visiting)?,
                    )?);
                }
                result
            }
            Expr::Repeat(part, min, max) => {
                let text = expression(part, rules, resolved, visiting)?;
                let suffix = match (*min, *max) {
                    (0, None) => "*".into(),
                    (1, None) => "+".into(),
                    (0, Some(1)) => "?".into(),
                    (min, Some(max)) if min == max => format!("{{{min}}}"),
                    (min, Some(max)) => format!("{{{min},{max}}}"),
                    (min, None) => format!("{{{min},}}"),
                };
                bounded(format!("({text}){suffix}"))
            }
            _ => bounded(expr.render()),
        }
    }
    rule("root", rules, &mut resolved, &mut BTreeSet::new())
}

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
