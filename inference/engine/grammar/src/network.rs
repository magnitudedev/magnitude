//! The rule network: GBNF rules with resolved references, reduced to the
//! rules reachable from `root` whose language is non-empty. Every remaining
//! alternative can complete, so a mask never admits a dead end.
use crate::{
    gbnf::{Expr, Rules},
    GrammarError,
};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) type RuleId = usize;
pub(crate) type RepeatId = usize;

const SURROGATES: (u32, u32) = (0xD800, 0xDFFF);
const MAX_SCALAR: u32 = 0x10FFFF;

/// A set of Unicode scalar values as sorted, disjoint, non-adjacent ranges.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct CharSet(Vec<(u32, u32)>);

impl CharSet {
    fn new(mut ranges: Vec<(u32, u32)>) -> Self {
        ranges.sort_unstable();
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
        for (start, end) in ranges {
            match merged.last_mut() {
                Some(last) if start <= last.1.saturating_add(1) => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }
        // Surrogates are not scalar values; a range spanning them keeps
        // only its scalar parts.
        let mut ranges = Vec::with_capacity(merged.len());
        for (start, end) in merged {
            if end < SURROGATES.0 || start > SURROGATES.1 {
                ranges.push((start, end));
                continue;
            }
            if start < SURROGATES.0 {
                ranges.push((start, SURROGATES.0 - 1));
            }
            if end > SURROGATES.1 {
                ranges.push((SURROGATES.1 + 1, end));
            }
        }
        Self(ranges)
    }
    pub(crate) fn any() -> Self {
        Self::new(vec![(0, MAX_SCALAR)])
    }
    fn complement(&self) -> Self {
        let mut gaps = Vec::new();
        let mut next = 0u32;
        for &(start, end) in &self.0 {
            if start > next {
                gaps.push((next, start - 1));
            }
            next = end + 1;
        }
        if next <= MAX_SCALAR {
            gaps.push((next, MAX_SCALAR));
        }
        Self::new(gaps)
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub(crate) fn ranges(&self) -> &[(u32, u32)] {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Node {
    Chars(CharSet),
    /// Non-empty text.
    Literal(String),
    /// Concatenation; the empty sequence is the empty string.
    Seq(Vec<Node>),
    Alt(Vec<Node>),
    Repeat {
        id: RepeatId,
        part: Box<Node>,
        min: u32,
        max: Option<u32>,
    },
    Rule(RuleId),
}

impl Node {
    pub(crate) fn empty() -> Self {
        Self::Seq(Vec::new())
    }
    pub(crate) fn references(&self, rules: &mut BTreeSet<RuleId>) {
        match self {
            Self::Rule(rule) => {
                rules.insert(*rule);
            }
            Self::Seq(parts) | Self::Alt(parts) => {
                for part in parts {
                    part.references(rules);
                }
            }
            Self::Repeat { part, .. } => part.references(rules),
            Self::Chars(_) | Self::Literal(_) => {}
        }
    }
    fn size(&self) -> usize {
        match self {
            Self::Seq(parts) | Self::Alt(parts) => 1 + parts.iter().map(Self::size).sum::<usize>(),
            Self::Repeat { part, .. } => 1 + part.size(),
            Self::Chars(_) | Self::Literal(_) | Self::Rule(_) => 1,
        }
    }
    fn productive(&self, rules: &[bool]) -> bool {
        match self {
            Self::Chars(set) => !set.is_empty(),
            Self::Literal(_) => true,
            Self::Seq(parts) => parts.iter().all(|part| part.productive(rules)),
            Self::Alt(parts) => parts.iter().any(|part| part.productive(rules)),
            Self::Repeat { part, min, .. } => *min == 0 || part.productive(rules),
            Self::Rule(rule) => rules[*rule],
        }
    }
    /// The same language without unproductive alternatives. Only called on
    /// productive nodes.
    fn prune(self, rules: &[bool]) -> Self {
        match self {
            Self::Seq(parts) => Self::Seq(parts.into_iter().map(|p| p.prune(rules)).collect()),
            Self::Alt(parts) => {
                let mut parts = parts
                    .into_iter()
                    .filter(|part| part.productive(rules))
                    .map(|part| part.prune(rules))
                    .collect::<Vec<_>>();
                if parts.len() == 1 {
                    parts.pop().unwrap()
                } else {
                    Self::Alt(parts)
                }
            }
            Self::Repeat { part, .. } if !part.productive(rules) => Self::empty(),
            Self::Repeat { id, part, min, max } => Self::Repeat {
                id,
                part: Box::new(part.prune(rules)),
                min,
                max,
            },
            node => node,
        }
    }
    fn renumber(&mut self, map: &BTreeMap<RuleId, RuleId>) {
        match self {
            Self::Rule(rule) => *rule = map[rule],
            Self::Seq(parts) | Self::Alt(parts) => {
                for part in parts {
                    part.renumber(map);
                }
            }
            Self::Repeat { part, .. } => part.renumber(map),
            Self::Chars(_) | Self::Literal(_) => {}
        }
    }
}

pub(crate) struct Network {
    pub bodies: Vec<Node>,
    pub root: RuleId,
    /// Node count of the pruned network; the unit of the size allowance.
    pub symbols: usize,
}

struct Resolver<'a> {
    ids: &'a BTreeMap<&'a str, RuleId>,
    repeats: RepeatId,
}

impl Resolver<'_> {
    fn node(&mut self, expr: &Expr) -> Node {
        match expr {
            Expr::Literal(text) if text.is_empty() => Node::empty(),
            Expr::Literal(text) => Node::Literal(text.clone()),
            Expr::Class { negated, ranges } => {
                let set = CharSet::new(
                    ranges
                        .iter()
                        .map(|(start, end)| (*start as u32, *end as u32))
                        .collect(),
                );
                Node::Chars(if *negated { set.complement() } else { set })
            }
            Expr::Any => Node::Chars(CharSet::any()),
            Expr::Reference(name) => Node::Rule(self.ids[name.as_str()]),
            Expr::Sequence(parts) => Node::Seq(parts.iter().map(|p| self.node(p)).collect()),
            Expr::Alternative(parts) => Node::Alt(parts.iter().map(|p| self.node(p)).collect()),
            Expr::Repeat(part, min, max) => {
                let part = Box::new(self.node(part));
                self.repeats += 1;
                Node::Repeat {
                    id: self.repeats - 1,
                    part,
                    min: *min,
                    max: *max,
                }
            }
        }
    }
}

impl Network {
    pub(crate) fn build(rules: &Rules) -> Result<Self, GrammarError> {
        let ids: BTreeMap<&str, RuleId> = rules
            .keys()
            .enumerate()
            .map(|(id, name)| (name.as_str(), id))
            .collect();
        let mut resolver = Resolver {
            ids: &ids,
            repeats: 0,
        };
        let bodies = rules
            .values()
            .map(|expr| resolver.node(expr))
            .collect::<Vec<_>>();
        let root = ids["root"];

        let mut productive = vec![false; bodies.len()];
        loop {
            let next = bodies
                .iter()
                .map(|body| body.productive(&productive))
                .collect::<Vec<_>>();
            if next == productive {
                break;
            }
            productive = next;
        }
        if !productive[root] {
            return Err(GrammarError::Language(
                "the grammar's root rule matches no text".into(),
            ));
        }
        let bodies = bodies
            .into_iter()
            .zip(&productive)
            .map(|(body, &productive_rule)| productive_rule.then(|| body))
            .collect::<Vec<_>>();
        let bodies = bodies
            .into_iter()
            .map(|body| body.map(|body| body.prune(&productive)))
            .collect::<Vec<_>>();

        let mut reachable = BTreeSet::from([root]);
        let mut pending = vec![root];
        while let Some(rule) = pending.pop() {
            let mut references = BTreeSet::new();
            bodies[rule]
                .as_ref()
                .expect("reachable rules are productive")
                .references(&mut references);
            for reference in references {
                if reachable.insert(reference) {
                    pending.push(reference);
                }
            }
        }
        let map: BTreeMap<RuleId, RuleId> = reachable
            .iter()
            .enumerate()
            .map(|(new, &old)| (old, new))
            .collect();
        let mut kept_bodies = Vec::with_capacity(map.len());
        for (index, body) in bodies.into_iter().enumerate() {
            if !map.contains_key(&index) {
                continue;
            }
            let mut body = body.expect("reachable rules are productive");
            body.renumber(&map);
            kept_bodies.push(body);
        }
        let symbols = kept_bodies.iter().map(Node::size).sum();
        Ok(Self {
            bodies: kept_bodies,
            root: map[&root],
            symbols,
        })
    }

    /// Strongly connected components of the reference graph, each listed
    /// after every component it references.
    pub(crate) fn components(&self) -> Vec<Vec<RuleId>> {
        let edges = self
            .bodies
            .iter()
            .map(|body| {
                let mut references = BTreeSet::new();
                body.references(&mut references);
                references.into_iter().collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // Iterative Tarjan: rule chains may be deeper than the call stack.
        let count = self.bodies.len();
        let mut index = vec![usize::MAX; count];
        let mut low = vec![0; count];
        let mut on_stack = vec![false; count];
        let mut stack = Vec::new();
        let mut components = Vec::new();
        let mut next = 0;
        for start in 0..count {
            if index[start] != usize::MAX {
                continue;
            }
            let mut frames = vec![(start, 0usize)];
            index[start] = next;
            low[start] = next;
            next += 1;
            stack.push(start);
            on_stack[start] = true;
            while let Some(&mut (rule, ref mut edge)) = frames.last_mut() {
                if let Some(&target) = edges[rule].get(*edge) {
                    *edge += 1;
                    if index[target] == usize::MAX {
                        index[target] = next;
                        low[target] = next;
                        next += 1;
                        stack.push(target);
                        on_stack[target] = true;
                        frames.push((target, 0));
                    } else if on_stack[target] {
                        low[rule] = low[rule].min(index[target]);
                    }
                    continue;
                }
                frames.pop();
                if let Some(&(parent, _)) = frames.last() {
                    low[parent] = low[parent].min(low[rule]);
                }
                if low[rule] == index[rule] {
                    let mut component = Vec::new();
                    loop {
                        let member = stack.pop().unwrap();
                        on_stack[member] = false;
                        component.push(member);
                        if member == rule {
                            break;
                        }
                    }
                    component.sort_unstable();
                    components.push(component);
                }
            }
        }
        components
    }
}
