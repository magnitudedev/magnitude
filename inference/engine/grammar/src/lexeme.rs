//! The lexical plan: Earley rules as graphs between boundaries, whose edges
//! are lexemes (all regular paths between two boundaries), calls, or ε.
//!
//! An Earley rule's skeleton is built from its body. Regular pieces are
//! terminal references; structural rules are flattened into the skeleton
//! within the flattening allowance; recursive rules, unflattened structural
//! rules and large repetitions of non-regular parts are calls. Other
//! repetitions stay in the skeleton as loops, so lexemes run through them.
//! Boundaries are the rule's entry, exit and call endpoints; the regular
//! paths between them are found by elimination by reference.
use crate::{
    elimination,
    network::{Network, Node, RepeatId, RuleId},
    terminal::{is_regular, Allowance, Class, Term, TermId, Terms, EMPTY},
};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

/// Repetitions of non-regular parts with bounds up to this are unrolled in
/// the skeleton; larger ones are calls repeated by llguidance.
const UNROLL: u32 = 8;
/// Flattening may make skeletons at most this multiple of the network.
const FLATTEN_FACTOR: usize = 4;
/// Flattening may at most double the call sites of the unflattened
/// network: each copied call site copies the lexemes that end and start
/// there. Flattening exists to keep a scanner and its delimiter in one
/// lexeme across a call, which needs few copies.
const CALL_FACTOR: u64 = 2;
const CALL_FLOOR: u64 = 16;
/// The most lexemes one boundary may start. Lexemes starting together are
/// lexed as one set by llguidance and certified against each other.
const FANOUT: usize = 128;

fn unrolled(min: u32, max: Option<u32>) -> Option<u64> {
    match max {
        None if min <= UNROLL => Some(u64::from(min) + 1),
        Some(max) if max <= UNROLL => Some(u64::from(max)),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Key {
    Rule(RuleId),
    Repeat(RepeatId),
    /// The character-level rendering of a region terminal.
    Twin(TermId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EdgeKind {
    /// `lexeme` is `region` without the empty string.
    Lexeme {
        lexeme: TermId,
        region: TermId,
    },
    Call {
        rule: usize,
        min: u32,
        max: Option<u32>,
    },
    Eps,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Edge {
    pub from: u32,
    pub to: u32,
    pub kind: EdgeKind,
}

/// An Earley rule: boundary 0 is its entry, boundary 1 its exit.
pub(crate) struct Earley {
    pub boundaries: u32,
    pub edges: Vec<Edge>,
}

pub(crate) struct Plan {
    pub rules: Vec<Earley>,
    index: HashMap<Key, usize>,
    pub unflattened: usize,
    pub hubs: usize,
}

enum Raw {
    /// A terminal; `EMPTY` is ε.
    Term(TermId),
    Call(Key, u32, Option<u32>),
}

struct Skeleton {
    states: u32,
    edges: Vec<(u32, u32, Raw)>,
    /// Loop heads of repetitions whose part contains a call: boundaries, so
    /// a lexeme spans one iteration rather than any number of the part's
    /// regular iterations around the calls.
    heads: Vec<u32>,
}

struct Builder<'n, 'b> {
    network: &'n Network,
    regular: &'b [Option<TermId>],
    classes: &'b [Class],
    flattened: &'b BTreeSet<RuleId>,
    repeats: &'b mut BTreeMap<RepeatId, &'n Node>,
    terms: &'b mut Terms,
    skeleton: Skeleton,
}

impl<'n> Builder<'n, '_> {
    fn state(&mut self) -> u32 {
        self.skeleton.states += 1;
        self.skeleton.states - 1
    }
    fn eps(&mut self, from: u32, to: u32) {
        self.skeleton.edges.push((from, to, Raw::Term(EMPTY)));
    }
    /// A call between private endpoints, so paths that bypass it never end a
    /// lexeme where the call starts or ends.
    fn call(&mut self, from: u32, to: u32, key: Key, min: u32, max: Option<u32>) {
        let (enter, exit) = (self.state(), self.state());
        self.eps(from, enter);
        self.skeleton
            .edges
            .push((enter, exit, Raw::Call(key, min, max)));
        self.eps(exit, to);
    }
    fn build(&mut self, node: &'n Node, from: u32, to: u32) {
        if is_regular(node, self.regular) {
            let term = self.terms.of_node(node, self.regular);
            self.skeleton.edges.push((from, to, Raw::Term(term)));
            return;
        }
        match node {
            Node::Rule(rule) => match self.classes[*rule] {
                Class::Structural if self.flattened.contains(rule) => {
                    self.build(&self.network.bodies[*rule], from, to)
                }
                Class::Structural | Class::Recursive => {
                    self.call(from, to, Key::Rule(*rule), 1, Some(1))
                }
                Class::Regular(_) => unreachable!("regular references are terminals"),
            },
            Node::Seq(parts) => {
                let mut current = from;
                for (index, part) in parts.iter().enumerate() {
                    let next = if index + 1 == parts.len() {
                        to
                    } else {
                        self.state()
                    };
                    self.build(part, current, next);
                    current = next;
                }
            }
            Node::Alt(parts) => {
                for part in parts {
                    self.build(part, from, to);
                }
            }
            Node::Repeat { id, part, min, max } => {
                if unrolled(*min, *max).is_none() {
                    self.repeats.insert(*id, part);
                    self.call(from, to, Key::Repeat(*id), *min, *max);
                    return;
                }
                let mut current = from;
                for _ in 0..*min {
                    let next = self.state();
                    self.build(part, current, next);
                    current = next;
                }
                match max {
                    None => {
                        let repeat = self.state();
                        self.skeleton.heads.push(repeat);
                        self.eps(current, repeat);
                        self.build(part, repeat, repeat);
                        self.eps(repeat, to);
                    }
                    Some(max) => {
                        for _ in *min..*max {
                            let next = self.state();
                            self.eps(current, to);
                            self.build(part, current, next);
                            current = next;
                        }
                        self.eps(current, to);
                    }
                }
            }
            Node::Chars(_) | Node::Literal(_) => unreachable!("terminals are regular"),
        }
    }
}

/// What a node contributes to skeletons, with flattened structural rules
/// and unrolled repetitions expanded. Every copied call site is a boundary
/// pair whose lexemes are copied too, so calls are bounded as well as
/// edges. Saturates: only compared against the allowance.
#[derive(Clone, Copy, Default)]
struct Size {
    edges: u64,
    calls: u64,
}

impl Size {
    fn plus(self, other: Self) -> Self {
        Self {
            edges: self.edges.saturating_add(other.edges),
            calls: self.calls.saturating_add(other.calls),
        }
    }
    fn times(self, copies: u64) -> Self {
        Self {
            edges: self.edges.saturating_mul(copies),
            calls: self.calls.saturating_mul(copies),
        }
    }
    fn cost(self) -> u64 {
        self.edges.saturating_add(self.calls)
    }
}

const CALL: Size = Size { edges: 3, calls: 1 };

fn size(
    node: &Node,
    network: &Network,
    regular: &[Option<TermId>],
    classes: &[Class],
    flattened: &BTreeSet<RuleId>,
    memo: &mut HashMap<RuleId, Size>,
) -> Size {
    if is_regular(node, regular) {
        return Size { edges: 1, calls: 0 };
    }
    match node {
        Node::Rule(rule) if classes[*rule] == Class::Structural && flattened.contains(rule) => {
            if let Some(&size) = memo.get(rule) {
                return size;
            }
            let body = size(
                &network.bodies[*rule],
                network,
                regular,
                classes,
                flattened,
                memo,
            );
            memo.insert(*rule, body);
            body
        }
        Node::Seq(parts) | Node::Alt(parts) => parts.iter().fold(Size::default(), |total, part| {
            total.plus(size(part, network, regular, classes, flattened, memo))
        }),
        Node::Repeat { part, min, max, .. } => match unrolled(*min, *max) {
            Some(copies) => size(part, network, regular, classes, flattened, memo)
                .times(copies)
                .plus(Size {
                    edges: copies + 1,
                    calls: 0,
                }),
            None => CALL,
        },
        _ => CALL,
    }
}

/// Bodies that become Earley rules: the root, recursive and unflattened
/// structural rules, and every repetition too large to unroll.
fn earley_bodies<'n>(
    network: &'n Network,
    regular: &[Option<TermId>],
    classes: &[Class],
    flattened: &BTreeSet<RuleId>,
) -> Vec<&'n Node> {
    fn repeated<'n>(
        node: &'n Node,
        regular: &[Option<TermId>],
        parts: &mut BTreeMap<RepeatId, &'n Node>,
    ) {
        if is_regular(node, regular) {
            return;
        }
        match node {
            Node::Seq(children) | Node::Alt(children) => {
                for child in children {
                    repeated(child, regular, parts);
                }
            }
            Node::Repeat { id, part, min, max } => {
                if unrolled(*min, *max).is_none() {
                    parts.insert(*id, part);
                }
                repeated(part, regular, parts);
            }
            _ => {}
        }
    }
    let mut bodies = Vec::new();
    let mut parts = BTreeMap::new();
    for (rule, body) in network.bodies.iter().enumerate() {
        if rule == network.root
            || classes[rule] == Class::Recursive
            || (classes[rule] == Class::Structural && !flattened.contains(&rule))
        {
            bodies.push(body);
        }
        repeated(body, regular, &mut parts);
    }
    bodies.extend(parts.into_values());
    bodies
}

/// How many times each flattened structural rule's body is copied.
fn copies(
    network: &Network,
    regular: &[Option<TermId>],
    classes: &[Class],
    flattened: &BTreeSet<RuleId>,
) -> BTreeMap<RuleId, u64> {
    fn count(
        node: &Node,
        weight: u64,
        regular: &[Option<TermId>],
        classes: &[Class],
        flattened: &BTreeSet<RuleId>,
        references: &mut BTreeMap<RuleId, u64>,
    ) {
        if is_regular(node, regular) {
            return;
        }
        match node {
            Node::Rule(rule) if classes[*rule] == Class::Structural && flattened.contains(rule) => {
                let entry = references.entry(*rule).or_default();
                *entry = entry.saturating_add(weight);
            }
            Node::Seq(parts) | Node::Alt(parts) => {
                for part in parts {
                    count(part, weight, regular, classes, flattened, references);
                }
            }
            Node::Repeat { part, min, max, .. } => {
                // A repetition too large to unroll is its own Earley body.
                if let Some(copies) = unrolled(*min, *max) {
                    count(
                        part,
                        weight.saturating_mul(copies),
                        regular,
                        classes,
                        flattened,
                        references,
                    );
                }
            }
            _ => {}
        }
    }
    // Direct references from Earley bodies, then propagate through
    // flattened rules in reverse topological order (callers first).
    let mut totals = BTreeMap::new();
    for body in earley_bodies(network, regular, classes, flattened) {
        count(body, 1, regular, classes, flattened, &mut totals);
    }
    for component in network.components().into_iter().rev() {
        let rule = component[0];
        if !(classes[rule] == Class::Structural && flattened.contains(&rule)) {
            continue;
        }
        let weight = totals.get(&rule).copied().unwrap_or(0);
        if weight > 0 {
            count(
                &network.bodies[rule],
                weight,
                regular,
                classes,
                flattened,
                &mut totals,
            );
        }
    }
    totals
}

/// The largest flattened rule that is copied, if any. Keeping the largest
/// copied structure as a call removes the most copying per new boundary,
/// and places boundaries around whole constructs rather than at the start
/// of small leaves whose text siblings share.
fn costliest(
    network: &Network,
    regular: &[Option<TermId>],
    classes: &[Class],
    flattened: &BTreeSet<RuleId>,
) -> Option<RuleId> {
    let mut memo = HashMap::new();
    copies(network, regular, classes, flattened)
        .iter()
        .filter(|(_, &count)| count > 1)
        .max_by_key(|(rule, &count)| {
            let body = size(
                &network.bodies[**rule],
                network,
                regular,
                classes,
                flattened,
                &mut memo,
            );
            (body.cost(), count, std::cmp::Reverse(**rule))
        })
        .map(|(rule, _)| *rule)
}

/// Flatten every structural rule the allowance permits, except `kept`:
/// while the skeletons would exceed it, keep as a call the rule whose
/// copies cost the most.
fn choose_flattened(
    network: &Network,
    regular: &[Option<TermId>],
    classes: &[Class],
    kept: &BTreeSet<RuleId>,
) -> BTreeSet<RuleId> {
    let total = |flattened: &BTreeSet<RuleId>, memo: &mut HashMap<RuleId, Size>| {
        earley_bodies(network, regular, classes, flattened)
            .into_iter()
            .fold(Size::default(), |total, body| {
                total.plus(size(body, network, regular, classes, flattened, memo))
            })
    };
    // Without flattening, every reference is one call: the baseline.
    let baseline = total(&BTreeSet::new(), &mut HashMap::new());
    let edge_limit = Allowance::limit(network.symbols, FLATTEN_FACTOR) as u64;
    let call_limit = CALL_FACTOR * baseline.calls.max(CALL_FLOOR);
    let mut flattened: BTreeSet<RuleId> = (0..network.bodies.len())
        .filter(|&rule| {
            classes[rule] == Class::Structural && rule != network.root && !kept.contains(&rule)
        })
        .collect();
    loop {
        let current = total(&flattened, &mut HashMap::new());
        if current.edges <= edge_limit && current.calls <= call_limit {
            return flattened;
        }
        match costliest(network, regular, classes, &flattened) {
            Some(rule) => {
                flattened.remove(&rule);
            }
            // Nothing is copied: the skeletons are the network's own size,
            // apart from unrolled repetitions, which are bounded per node.
            None => return flattened,
        }
    }
}

impl Plan {
    /// Flatten what the allowance permits; while some boundary would start
    /// more than `FANOUT` lexemes, keep the costliest copied rule as a call.
    pub(crate) fn build(
        network: &Network,
        classes: &[Class],
        terms: &mut Terms,
        allowance: &mut Allowance,
    ) -> Self {
        let regular = classes
            .iter()
            .map(|class| match class {
                Class::Regular(term) => Some(*term),
                _ => None,
            })
            .collect::<Vec<_>>();
        let structural = (0..network.bodies.len())
            .filter(|&rule| classes[rule] == Class::Structural && rule != network.root)
            .count();
        let mut kept = BTreeSet::new();
        loop {
            let flattened = choose_flattened(network, &regular, classes, &kept);
            let mut attempt = allowance.clone();
            let mut plan =
                Self::assemble(network, &regular, classes, &flattened, terms, &mut attempt);
            plan.unflattened = structural - flattened.len();
            let next = if plan.fanout() > FANOUT {
                costliest(network, &regular, classes, &flattened)
            } else {
                None
            };
            match next {
                Some(rule) => {
                    kept.insert(rule);
                }
                None => {
                    *allowance = attempt;
                    return plan;
                }
            }
        }
    }

    /// The most lexemes that start at one boundary.
    fn fanout(&self) -> usize {
        self.rules
            .iter()
            .flat_map(|rule| {
                let mut starts = vec![0usize; rule.boundaries as usize];
                for edge in &rule.edges {
                    if let EdgeKind::Lexeme { .. } = edge.kind {
                        starts[edge.from as usize] += 1;
                    }
                }
                starts
            })
            .max()
            .unwrap_or(0)
    }

    fn assemble(
        network: &Network,
        regular: &[Option<TermId>],
        classes: &[Class],
        flattened: &BTreeSet<RuleId>,
        terms: &mut Terms,
        allowance: &mut Allowance,
    ) -> Self {
        let mut plan = Self {
            rules: Vec::new(),
            index: HashMap::new(),
            unflattened: 0,
            hubs: 0,
        };
        let mut repeats: BTreeMap<RepeatId, &Node> = BTreeMap::new();
        let mut pending = VecDeque::from([Key::Rule(network.root)]);
        plan.index.insert(Key::Rule(network.root), 0);
        plan.rules.push(Earley {
            boundaries: 0,
            edges: Vec::new(),
        });
        while let Some(key) = pending.pop_front() {
            let body: &Node = match key {
                Key::Rule(rule) => &network.bodies[rule],
                Key::Repeat(id) => repeats[&id],
                Key::Twin(_) => unreachable!("twins are added by certification"),
            };
            let mut discovered = BTreeMap::new();
            let skeleton = {
                let mut builder = Builder {
                    network,
                    regular,
                    classes,
                    flattened,
                    repeats: &mut discovered,
                    terms,
                    skeleton: Skeleton {
                        states: 2,
                        edges: Vec::new(),
                        heads: Vec::new(),
                    },
                };
                builder.build(body, 0, 1);
                builder.skeleton
            };
            repeats.extend(discovered);
            let mut resolve = |key: Key, plan: &mut Plan| -> usize {
                if let Some(&index) = plan.index.get(&key) {
                    return index;
                }
                let index = plan.rules.len();
                plan.index.insert(key, index);
                plan.rules.push(Earley {
                    boundaries: 0,
                    edges: Vec::new(),
                });
                pending.push_back(key);
                index
            };
            let (boundaries, edges, hubs) =
                regions(skeleton, terms, allowance, |key| resolve(key, &mut plan));
            let rule = plan.index[&key];
            plan.rules[rule].boundaries = boundaries;
            plan.rules[rule].edges = edges;
            plan.hubs += hubs;
        }
        plan
    }

    /// Render the lexeme at `rules[rule].edges[edge]` at character level.
    pub(crate) fn characters(&mut self, rule: usize, edge: usize, terms: &mut Terms) {
        let EdgeKind::Lexeme { region, .. } = self.rules[rule].edges[edge].kind else {
            unreachable!("only lexemes are rendered at character level")
        };
        let twin = self.twin(region, terms);
        self.rules[rule].edges[edge].kind = EdgeKind::Call {
            rule: twin,
            min: 1,
            max: Some(1),
        };
    }

    /// The Earley rule spelling a terminal with one-character lexemes.
    fn twin(&mut self, term: TermId, terms: &mut Terms) -> usize {
        if let Some(&index) = self.index.get(&Key::Twin(term)) {
            return index;
        }
        let index = self.rules.len();
        self.index.insert(Key::Twin(term), index);
        self.rules.push(Earley {
            boundaries: 2,
            edges: Vec::new(),
        });
        let mut boundaries = 2;
        let mut edges = Vec::new();
        let atom = |part: TermId, from: u32, to: u32, plan: &mut Plan, terms: &mut Terms| {
            let kind = if terms.single_character(part) {
                EdgeKind::Lexeme {
                    lexeme: part,
                    region: part,
                }
            } else {
                EdgeKind::Call {
                    rule: plan.twin(part, terms),
                    min: 1,
                    max: Some(1),
                }
            };
            Edge { from, to, kind }
        };
        let mut chain = |parts: Vec<TermId>, plan: &mut Plan, terms: &mut Terms| {
            let mut current = 0;
            for (position, &part) in parts.iter().enumerate() {
                let next = if position + 1 == parts.len() {
                    1
                } else {
                    boundaries += 1;
                    boundaries - 1
                };
                edges.push(atom(part, current, next, plan, terms));
                current = next;
            }
        };
        match terms.get(term).clone() {
            Term::Chars(_) => chain(vec![term], self, terms),
            Term::Literal(text) => {
                let characters = text
                    .chars()
                    .map(|character| terms.literal(&character.to_string()))
                    .collect();
                chain(characters, self, terms);
            }
            Term::Seq(parts) if parts.is_empty() => edges.push(Edge {
                from: 0,
                to: 1,
                kind: EdgeKind::Eps,
            }),
            Term::Seq(parts) => chain(parts, self, terms),
            Term::Alt(parts) => {
                for part in parts {
                    edges.push(atom(part, 0, 1, self, terms));
                }
            }
            Term::Repeat(part, min, max) => edges.push(Edge {
                from: 0,
                to: 1,
                kind: EdgeKind::Call {
                    rule: self.twin(part, terms),
                    min,
                    max,
                },
            }),
            Term::NonEmpty(_) => unreachable!("regions never contain non-empty restrictions"),
        }
        self.rules[index].boundaries = boundaries;
        self.rules[index].edges = edges;
        index
    }
}

/// Collapse a skeleton to its boundaries: calls keep their endpoints; every
/// regular path between two boundaries becomes one lexeme. Returns the
/// boundary count, the edges, and how many states the allowance kept.
fn regions(
    skeleton: Skeleton,
    terms: &mut Terms,
    allowance: &mut Allowance,
    mut resolve: impl FnMut(Key) -> usize,
) -> (u32, Vec<Edge>, usize) {
    let states = skeleton.states as usize;
    let mut boundary = vec![false; states];
    boundary[0] = true;
    boundary[1] = true;
    for &head in &skeleton.heads {
        boundary[head as usize] = true;
    }
    let mut calls = Vec::new();
    let mut regular = Vec::new();
    for (from, to, raw) in skeleton.edges {
        match raw {
            Raw::Term(term) => regular.push((from, to, term)),
            Raw::Call(key, min, max) => {
                boundary[from as usize] = true;
                boundary[to as usize] = true;
                calls.push((from, to, key, min, max));
            }
        }
    }
    let eliminated = elimination::eliminate(skeleton.states, regular, &boundary, terms, allowance);
    for &hub in &eliminated.kept {
        boundary[hub as usize] = true;
    }
    let numbers: Vec<Option<u32>> = {
        let mut next = 2;
        (0..states)
            .map(|state| match state {
                0 | 1 => Some(state as u32),
                _ if boundary[state] => {
                    next += 1;
                    Some(next - 1)
                }
                _ => None,
            })
            .collect()
    };
    let count = numbers.iter().flatten().count() as u32;
    let number = |state: u32| numbers[state as usize].expect("edges join boundaries");
    let mut edges = Vec::new();
    for (from, to, key, min, max) in calls {
        edges.push(Edge {
            from: number(from),
            to: number(to),
            kind: EdgeKind::Call {
                rule: resolve(key),
                min,
                max,
            },
        });
    }
    let mut eps = BTreeSet::new();
    for (from, to, region) in eliminated.edges {
        let (from, to) = (number(from), number(to));
        if terms.nullable(region) && from != to {
            eps.insert((from, to));
        }
        if region != EMPTY {
            edges.push(Edge {
                from,
                to,
                kind: EdgeKind::Lexeme {
                    lexeme: terms.non_empty(region),
                    region,
                },
            });
        }
    }
    edges.extend(eps.into_iter().map(|(from, to)| Edge {
        from,
        to,
        kind: EdgeKind::Eps,
    }));
    (count, edges, eliminated.kept.len())
}
