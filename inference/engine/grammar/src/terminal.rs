//! Terminals: named regular languages defined by reference to other
//! terminals. Composition never copies an operand, so a terminal used in many
//! places costs one definition, and llguidance compiles each name once.
use crate::network::{CharSet, Network, Node, RuleId};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(crate) type TermId = u32;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Term {
    Chars(CharSet),
    /// Non-empty text.
    Literal(String),
    /// Concatenation of at least two operands; `Seq([])` is the empty string.
    Seq(Vec<TermId>),
    /// Union of at least two operands, none of them the empty string.
    Alt(Vec<TermId>),
    Repeat(TermId, u32, Option<u32>),
    /// The operand's language without the empty string.
    NonEmpty(TermId),
}

/// Bytes that may begin a non-empty word.
pub(crate) type Bytes = [u64; 4];

pub(crate) fn has_byte(bytes: &Bytes, byte: u8) -> bool {
    bytes[byte as usize / 64] & (1 << (byte % 64)) != 0
}
pub(crate) fn intersects(a: &Bytes, b: &Bytes) -> bool {
    a.iter().zip(b).any(|(x, y)| x & y != 0)
}
pub(crate) fn union(a: &mut Bytes, b: &Bytes) {
    for (x, y) in a.iter_mut().zip(b) {
        *x |= y;
    }
}

fn set_byte(bytes: &mut Bytes, byte: u8) {
    bytes[byte as usize / 64] |= 1 << (byte % 64);
}

/// UTF-8 lead bytes of a scalar range: within one encoding length, lead
/// bytes grow monotonically with the scalar.
fn lead_bytes(set: &CharSet) -> Bytes {
    let mut bytes = [0; 4];
    let lead = |scalar: u32| {
        let mut buffer = [0; 4];
        char::from_u32(scalar)
            .expect("character sets hold scalar values")
            .encode_utf8(&mut buffer);
        buffer[0]
    };
    for &(start, end) in set.ranges() {
        for (low, high) in [
            (0, 0x7F),
            (0x80, 0x7FF),
            (0x800, 0xFFFF),
            (0x10000, 0x10FFFF),
        ] {
            let (from, to) = (start.max(low), end.min(high));
            if from > to {
                continue;
            }
            // Skip into the scalar range if the segment starts inside the
            // surrogates; character sets exclude them.
            let from = if (0xD800..=0xDFFF).contains(&from) {
                0xE000
            } else {
                from
            };
            let to = if (0xD800..=0xDFFF).contains(&to) {
                0xD7FF
            } else {
                to
            };
            if from > to {
                continue;
            }
            for byte in lead(from)..=lead(to) {
                set_byte(&mut bytes, byte);
            }
        }
    }
    bytes
}

pub(crate) struct Terms {
    list: Vec<Term>,
    index: HashMap<Term, TermId>,
    nullable: Vec<bool>,
    first: Vec<Bytes>,
}

pub(crate) const EMPTY: TermId = 0;

impl Terms {
    pub(crate) fn new() -> Self {
        let mut terms = Self {
            list: Vec::new(),
            index: HashMap::new(),
            nullable: Vec::new(),
            first: Vec::new(),
        };
        let empty = terms.intern(Term::Seq(Vec::new()));
        debug_assert_eq!(empty, EMPTY);
        terms
    }
    pub(crate) fn get(&self, id: TermId) -> &Term {
        &self.list[id as usize]
    }
    pub(crate) fn nullable(&self, id: TermId) -> bool {
        self.nullable[id as usize]
    }
    pub(crate) fn first(&self, id: TermId) -> &Bytes {
        &self.first[id as usize]
    }
    /// Every word is exactly one scalar value.
    pub(crate) fn single_character(&self, id: TermId) -> bool {
        match self.get(id) {
            Term::Chars(_) => true,
            Term::Literal(text) => text.chars().nth(1).is_none(),
            _ => false,
        }
    }

    fn intern(&mut self, term: Term) -> TermId {
        if let Some(&id) = self.index.get(&term) {
            return id;
        }
        let (nullable, first) = match &term {
            Term::Chars(set) => (false, lead_bytes(set)),
            Term::Literal(text) => {
                let mut first = [0; 4];
                set_byte(&mut first, text.as_bytes()[0]);
                (false, first)
            }
            Term::Seq(parts) => {
                let mut first = [0; 4];
                let mut nullable = true;
                for &part in parts {
                    if nullable {
                        union(&mut first, &self.first[part as usize]);
                    }
                    nullable &= self.nullable[part as usize];
                }
                (nullable, first)
            }
            Term::Alt(parts) => {
                let mut first = [0; 4];
                for &part in parts {
                    union(&mut first, &self.first[part as usize]);
                }
                (
                    parts.iter().any(|&part| self.nullable[part as usize]),
                    first,
                )
            }
            Term::Repeat(part, min, _) => (
                *min == 0 || self.nullable[*part as usize],
                self.first[*part as usize],
            ),
            Term::NonEmpty(part) => (false, self.first[*part as usize]),
        };
        let id = TermId::try_from(self.list.len()).expect("terminal count fits u32");
        self.list.push(term.clone());
        self.index.insert(term, id);
        self.nullable.push(nullable);
        self.first.push(first);
        id
    }

    pub(crate) fn chars(&mut self, set: CharSet) -> TermId {
        self.intern(Term::Chars(set))
    }
    pub(crate) fn literal(&mut self, text: &str) -> TermId {
        if text.is_empty() {
            EMPTY
        } else {
            self.intern(Term::Literal(text.into()))
        }
    }
    pub(crate) fn seq(&mut self, parts: impl IntoIterator<Item = TermId>) -> TermId {
        let mut flat = Vec::new();
        for part in parts {
            match self.get(part) {
                Term::Seq(inner) => flat.extend(inner.iter().copied()),
                _ => flat.push(part),
            }
        }
        match flat.len() {
            0 => EMPTY,
            1 => flat[0],
            _ => self.intern(Term::Seq(flat)),
        }
    }
    /// Union of a non-empty list of operands.
    pub(crate) fn alt(&mut self, parts: impl IntoIterator<Item = TermId>) -> TermId {
        let mut flat = BTreeSet::new();
        let mut empty = false;
        let mut operands = 0;
        for part in parts {
            operands += 1;
            match self.get(part) {
                Term::Alt(inner) => flat.extend(inner.iter().copied()),
                _ if part == EMPTY => empty = true,
                _ => {
                    flat.insert(part);
                }
            }
        }
        assert!(
            operands > 0,
            "a union needs an operand; no operand is the empty language"
        );
        let union = match flat.len() {
            0 => EMPTY,
            1 => *flat.first().unwrap(),
            _ => self.intern(Term::Alt(flat.into_iter().collect())),
        };
        if empty {
            self.repeat(union, 0, Some(1))
        } else {
            union
        }
    }
    pub(crate) fn repeat(&mut self, part: TermId, min: u32, max: Option<u32>) -> TermId {
        if part == EMPTY || max == Some(0) {
            return EMPTY;
        }
        if (min, max) == (1, Some(1)) {
            return part;
        }
        if (min, max) == (0, Some(1)) && self.nullable(part) {
            return part;
        }
        self.intern(Term::Repeat(part, min, max))
    }
    pub(crate) fn non_empty(&mut self, part: TermId) -> TermId {
        if self.nullable(part) {
            self.intern(Term::NonEmpty(part))
        } else {
            part
        }
    }

    /// A node whose references all name regular rules, as a terminal.
    pub(crate) fn of_node(&mut self, node: &Node, regular: &[Option<TermId>]) -> TermId {
        match node {
            Node::Chars(set) => self.chars(set.clone()),
            Node::Literal(text) => self.literal(text),
            Node::Seq(parts) => {
                let parts = parts
                    .iter()
                    .map(|part| self.of_node(part, regular))
                    .collect::<Vec<_>>();
                self.seq(parts)
            }
            Node::Alt(parts) => {
                let parts = parts
                    .iter()
                    .map(|part| self.of_node(part, regular))
                    .collect::<Vec<_>>();
                self.alt(parts)
            }
            Node::Repeat { part, min, max, .. } => {
                let part = self.of_node(part, regular);
                self.repeat(part, *min, *max)
            }
            Node::Rule(rule) => regular[*rule].expect("reference to a regular rule"),
        }
    }
}

/// Whether every reference in a node names a regular rule.
pub(crate) fn is_regular(node: &Node, regular: &[Option<TermId>]) -> bool {
    match node {
        Node::Chars(_) | Node::Literal(_) => true,
        Node::Seq(parts) | Node::Alt(parts) => parts.iter().all(|part| is_regular(part, regular)),
        Node::Repeat { part, .. } => is_regular(part, regular),
        Node::Rule(rule) => regular[*rule].is_some(),
    }
}

/// Output the compiler may create beyond one term per network symbol.
#[derive(Clone)]
pub(crate) struct Allowance {
    remaining: usize,
}

impl Allowance {
    pub(crate) fn limit(symbols: usize, factor: usize) -> usize {
        factor * symbols.max(ALLOWANCE_FLOOR)
    }
    pub(crate) fn new(symbols: usize) -> Self {
        Self {
            remaining: Self::limit(symbols, ALLOWANCE_FACTOR),
        }
    }
    pub(crate) fn spend(&mut self, amount: usize) -> bool {
        if amount > self.remaining {
            return false;
        }
        self.remaining -= amount;
        true
    }
}

/// Terminals created by elimination may exceed the pruned network by at
/// most this factor.
pub(crate) const ALLOWANCE_FACTOR: usize = 16;
const ALLOWANCE_FLOOR: usize = 256;

/// How a rule is lowered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    /// A terminal: acyclic over regular rules, or a right-linear component.
    Regular(TermId),
    /// Not recursive, but references a recursive rule.
    Structural,
    /// In a recursive component that is not a right-linear scanner.
    Recursive,
}

/// Classify every rule, callees first, turning regular rules and scanner
/// components into terminals.
pub(crate) fn classify(
    network: &Network,
    terms: &mut Terms,
    allowance: &mut Allowance,
) -> (Vec<Class>, usize) {
    let mut regular: Vec<Option<TermId>> = vec![None; network.bodies.len()];
    let mut classes = vec![Class::Structural; network.bodies.len()];
    let mut scanners = 0;
    for component in network.components() {
        let recursive = component.len() > 1 || {
            let mut references = BTreeSet::new();
            network.bodies[component[0]].references(&mut references);
            references.contains(&component[0])
        };
        if !recursive {
            let rule = component[0];
            let body = &network.bodies[rule];
            if is_regular(body, &regular) {
                let term = terms.of_node(body, &regular);
                regular[rule] = Some(term);
                classes[rule] = Class::Regular(term);
            }
            continue;
        }
        match scanner(network, &component, terms, &regular, allowance) {
            Some(solved) => {
                scanners += 1;
                for (rule, term) in solved {
                    regular[rule] = Some(term);
                    classes[rule] = Class::Regular(term);
                }
            }
            None => {
                for rule in component {
                    classes[rule] = Class::Recursive;
                }
            }
        }
    }
    (classes, scanners)
}

/// Solve a right-linear component: every in-component reference is the last
/// element of its alternative, and all other references are regular. Each
/// rule's language is found by eliminating the others (Arden's rule), with
/// every intermediate expression a new terminal naming its operands.
fn scanner(
    network: &Network,
    component: &[RuleId],
    terms: &mut Terms,
    regular: &[Option<TermId>],
    allowance: &mut Allowance,
) -> Option<Vec<(RuleId, TermId)>> {
    let members: BTreeSet<RuleId> = component.iter().copied().collect();
    let local: BTreeMap<RuleId, usize> = component
        .iter()
        .enumerate()
        .map(|(index, &rule)| (rule, index))
        .collect();
    // Equation per member: X_i = Σ coefficient_ij X_j + constant_i.
    let mut coefficients: Vec<BTreeMap<usize, Vec<TermId>>> =
        vec![BTreeMap::new(); component.len()];
    let mut constants: Vec<Vec<TermId>> = vec![Vec::new(); component.len()];
    for (index, &rule) in component.iter().enumerate() {
        let body = &network.bodies[rule];
        let alternatives = match body {
            Node::Alt(parts) => parts.as_slice(),
            _ => std::slice::from_ref(body),
        };
        for alternative in alternatives {
            let (prefix, tail) = match alternative {
                Node::Rule(target) if members.contains(target) => (&[][..], Some(*target)),
                Node::Seq(parts) => match parts.last() {
                    Some(Node::Rule(target)) if members.contains(target) => {
                        (&parts[..parts.len() - 1], Some(*target))
                    }
                    _ => (parts.as_slice(), None),
                },
                _ => (std::slice::from_ref(alternative), None),
            };
            if !prefix.iter().all(|part| is_regular(part, regular)) {
                return None;
            }
            let prefix = prefix
                .iter()
                .map(|part| terms.of_node(part, regular))
                .collect::<Vec<_>>();
            let prefix = terms.seq(prefix);
            match tail {
                Some(target) => coefficients[index]
                    .entry(local[&target])
                    .or_default()
                    .push(prefix),
                None => constants[index].push(prefix),
            }
        }
    }
    let mut coefficients = coefficients
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(target, parts)| (target, terms.alt(parts)))
                .collect::<BTreeMap<_, _>>()
        })
        .collect::<Vec<_>>();
    // `None` is the empty language: a rule with no alternative leaving the
    // component.
    let mut constants = constants
        .into_iter()
        .map(|parts| (!parts.is_empty()).then(|| terms.alt(parts)))
        .collect::<Vec<Option<TermId>>>();

    // Eliminate cheapest variables first; each keeps its Arden-normalized
    // equation for back-substitution.
    let mut remaining: BTreeSet<usize> = (0..component.len()).collect();
    let mut order = Vec::with_capacity(component.len());
    loop {
        let Some(variable) = remaining.iter().copied().min_by_key(|&variable| {
            let incoming = remaining
                .iter()
                .filter(|&&row| row != variable && coefficients[row].contains_key(&variable))
                .count();
            let outgoing = coefficients[variable]
                .keys()
                .filter(|&&target| target != variable)
                .count();
            (incoming * (outgoing + 1), variable)
        }) else {
            break;
        };
        remaining.remove(&variable);
        // X = a X + rest  ⇒  X = a* rest
        if let Some(looped) = coefficients[variable].remove(&variable) {
            let star = terms.repeat(looped, 0, None);
            let targets = coefficients[variable].keys().copied().collect::<Vec<_>>();
            for target in targets {
                let coefficient = coefficients[variable][&target];
                let scaled = terms.seq([star, coefficient]);
                coefficients[variable].insert(target, scaled);
            }
            constants[variable] = constants[variable].map(|constant| terms.seq([star, constant]));
            if !allowance.spend(1 + coefficients[variable].len()) {
                return None;
            }
        }
        let rows = remaining
            .iter()
            .copied()
            .filter(|row| coefficients[*row].contains_key(&variable))
            .collect::<Vec<_>>();
        for row in rows {
            let factor = coefficients[row].remove(&variable).unwrap();
            let substituted = coefficients[variable].clone();
            if !allowance.spend(1 + substituted.len()) {
                return None;
            }
            for (target, coefficient) in substituted {
                let path = terms.seq([factor, coefficient]);
                let merged = match coefficients[row].get(&target) {
                    Some(&existing) => terms.alt([existing, path]),
                    None => path,
                };
                coefficients[row].insert(target, merged);
            }
            if let Some(constant) = constants[variable] {
                let path = terms.seq([factor, constant]);
                constants[row] = Some(match constants[row] {
                    Some(existing) => terms.alt([existing, path]),
                    None => path,
                });
            }
        }
        order.push(variable);
    }
    // Back-substitute in reverse elimination order: every coefficient of an
    // eliminated variable names a variable eliminated after it. A variable
    // whose language is empty contributes no path.
    let mut solved: Vec<Option<TermId>> = vec![None; component.len()];
    for &variable in order.iter().rev() {
        let mut parts = constants[variable].into_iter().collect::<Vec<_>>();
        for (&target, &coefficient) in &coefficients[variable] {
            if let Some(rest) = solved[target] {
                parts.push(terms.seq([coefficient, rest]));
            }
        }
        solved[variable] = (!parts.is_empty()).then(|| terms.alt(parts));
    }
    Some(
        component
            .iter()
            .enumerate()
            .map(|(index, &rule)| {
                (
                    rule,
                    solved[index].expect("pruned rules have non-empty languages"),
                )
            })
            .collect(),
    )
}
