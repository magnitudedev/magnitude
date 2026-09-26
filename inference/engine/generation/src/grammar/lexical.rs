//! Lexical compilation: every regular stretch between calls of recursive rules
//! becomes one lexeme, so the Earley parser only sees the grammar's recursive
//! skeleton and free text (reasoning, JSON strings, raw arguments) is scanned
//! by the lexer instead of one parser row per byte.
//!
//! llguidance lexes greedily without backtracking: a lexeme ends only when no
//! allowed lexeme can consume the next byte. That is language-preserving only
//! when no complete lexeme can be continued, by itself or by a lexeme allowed
//! at the same position, with a byte that may start the lexeme after it. The
//! compiler proves this for every lexeme; if it cannot, it declines and the
//! grammar is rendered without lexical compilation.
use super::regular::{eliminate_states, sequence, union, Keep, EXPRESSION};
use super::{Expr, Rules};
use derivre::{RegexAst, RegexBuilder};
use std::collections::{BTreeMap, BTreeSet};

/// Bound on automaton states built for one emitted rule.
const STATES: usize = 64 * 1024;
/// A state whose elimination would copy more than this many expression nodes
/// stays a lexeme boundary (a hub), so long shared paths such as reasoning
/// are not copied into every lexeme through it.
const HUB_GROWTH: usize = 256;
/// Bound on distinct sets of lexemes that may be allowed together.
const ALLOWED_SETS: usize = 4096;
/// Derivative fuel for one emptiness proof.
const FUEL: u64 = 2_000_000;

#[derive(Clone, Debug)]
enum Item {
    Epsilon,
    Lexeme(usize),
    Call(String),
}

/// A rule's language as an automaton whose edges are lexemes, epsilons, and
/// calls of recursive rules. State 0 enters the rule, state 1 exits it.
struct Rule {
    edges: Vec<(usize, usize, Item)>,
}

struct Builder<'a> {
    rules: &'a Rules,
    regular: &'a BTreeMap<String, Expr>,
    recursive: &'a BTreeSet<String>,
    states: usize,
    lexemes: BTreeMap<(usize, usize), Expr>,
    calls: Vec<(usize, usize, String)>,
}

impl Builder<'_> {
    fn state(&mut self) -> Option<usize> {
        self.states += 1;
        (self.states <= STATES).then_some(self.states - 1)
    }
    fn lexeme(&mut self, from: usize, to: usize, expr: Expr) {
        let previous = self.lexemes.remove(&(from, to));
        self.lexemes.insert((from, to), union(previous, expr));
    }
    /// The expression as one regular language, when every reference in it
    /// resolves to a regular rule.
    fn regular(&self, expr: &Expr) -> Option<Expr> {
        let resolved = match expr {
            Expr::Reference(name) => self.regular.get(name)?.clone(),
            Expr::Sequence(parts) => sequence(
                parts
                    .iter()
                    .map(|part| self.regular(part))
                    .collect::<Option<Vec<_>>>()?,
            ),
            Expr::Alternative(parts) => {
                let mut result = None;
                for part in parts {
                    result = Some(union(result, self.regular(part)?));
                }
                result?
            }
            Expr::Repeat(part, min, max) => Expr::Repeat(Box::new(self.regular(part)?), *min, *max),
            _ => expr.clone(),
        };
        (resolved.size() <= EXPRESSION).then_some(resolved)
    }
    fn build(&mut self, expr: &Expr, from: usize, to: usize) -> Option<()> {
        if let Some(regular) = self.regular(expr) {
            self.lexeme(from, to, regular);
            return Some(());
        }
        match expr {
            Expr::Reference(name) if self.recursive.contains(name) => {
                // Private endpoints: paths that bypass this call must not be
                // cut into separate lexemes where it starts or ends.
                let (enter, exit) = (self.state()?, self.state()?);
                self.lexeme(from, enter, Expr::Sequence(Vec::new()));
                self.calls.push((enter, exit, name.clone()));
                self.lexeme(exit, to, Expr::Sequence(Vec::new()));
            }
            Expr::Reference(name) => self.build(&self.rules[name], from, to)?,
            Expr::Sequence(parts) => {
                let mut current = from;
                for (index, part) in parts.iter().enumerate() {
                    let next = if index + 1 == parts.len() {
                        to
                    } else {
                        self.state()?
                    };
                    self.build(part, current, next)?;
                    current = next;
                }
            }
            Expr::Alternative(parts) => {
                for part in parts {
                    self.build(part, from, to)?;
                }
            }
            Expr::Repeat(part, min, max) => {
                let mut current = from;
                for _ in 0..*min {
                    let next = self.state()?;
                    self.build(part, current, next)?;
                    current = next;
                }
                match max {
                    None => {
                        let repeat = self.state()?;
                        self.lexeme(current, repeat, Expr::Sequence(Vec::new()));
                        self.build(part, repeat, repeat)?;
                        self.lexeme(repeat, to, Expr::Sequence(Vec::new()));
                    }
                    Some(max) => {
                        for _ in *min..*max {
                            let next = self.state()?;
                            self.lexeme(current, to, Expr::Sequence(Vec::new()));
                            self.build(part, current, next)?;
                            current = next;
                        }
                        self.lexeme(current, to, Expr::Sequence(Vec::new()));
                    }
                }
            }
            Expr::Literal(_) | Expr::Class { .. } | Expr::Any => {
                unreachable!("terminals are regular")
            }
        }
        Some(())
    }
}

/// Rules that take part in a reference cycle.
fn recursive_rules(rules: &Rules) -> BTreeSet<String> {
    let references: BTreeMap<_, _> = rules
        .iter()
        .map(|(name, expr)| {
            let mut names = BTreeSet::new();
            expr.references(&mut names);
            (name.clone(), names)
        })
        .collect();
    rules
        .keys()
        .filter(|start| {
            let mut seen = BTreeSet::new();
            let mut pending = references[*start].iter().cloned().collect::<Vec<_>>();
            while let Some(name) = pending.pop() {
                if &name == *start {
                    return true;
                }
                if seen.insert(name.clone()) {
                    pending.extend(references[&name].iter().cloned());
                }
            }
            false
        })
        .cloned()
        .collect()
}

struct Grammar {
    rules: BTreeMap<String, Rule>,
    lexemes: Vec<Expr>,
    /// The rule and state each lexeme ends in.
    ends: Vec<(String, usize)>,
    /// States kept to share long paths instead of copying them into every
    /// lexeme through them; lexemes may end there only if proven safe.
    hubs: BTreeSet<(String, usize)>,
}

/// Automata for the root and every recursive rule it calls. States in
/// `forced` are merged into the lexemes through them even when they are hubs.
fn automata(
    rules: &Rules,
    regular: &BTreeMap<String, Expr>,
    forced: &BTreeSet<(String, usize)>,
) -> Option<Grammar> {
    let recursive = recursive_rules(rules);
    let mut grammar = Grammar {
        rules: BTreeMap::new(),
        lexemes: Vec::new(),
        ends: Vec::new(),
        hubs: BTreeSet::new(),
    };
    let mut pending = vec!["root".to_string()];
    while let Some(name) = pending.pop() {
        if grammar.rules.contains_key(&name) {
            continue;
        }
        let mut builder = Builder {
            rules,
            regular,
            recursive: &recursive,
            states: 2,
            lexemes: BTreeMap::new(),
            calls: Vec::new(),
        };
        builder.build(&rules[&name], 0, 1)?;
        let anchors = builder
            .calls
            .iter()
            .flat_map(|(from, to, _)| [*from, *to])
            .chain([0, 1])
            .collect::<BTreeSet<_>>();
        let inner = (0..builder.states).filter(|state| !anchors.contains(state));
        let forced_here = forced
            .iter()
            .filter(|(rule, _)| *rule == name)
            .map(|(_, state)| *state)
            .collect();
        let (eliminated, kept) = eliminate_states(
            builder.lexemes,
            inner,
            Keep::Hubs {
                growth: HUB_GROWTH,
                forced: &forced_here,
            },
            16 * EXPRESSION,
        )?;
        let mut paths = eliminated
            .into_iter()
            .map(|((from, to), expr)| (from, to, Some(expr), None))
            .collect::<Vec<_>>();
        for (from, to, callee) in builder.calls {
            pending.push(callee.clone());
            paths.push((from, to, None, Some(callee)));
        }
        // Keep only states on some entry-to-exit path.
        let shape = paths
            .iter()
            .map(|(from, to, _, _)| (*from, *to))
            .collect::<Vec<_>>();
        let forward = closure(&shape, 0, false);
        let backward = closure(&shape, 1, true);
        if !forward.contains(&1) {
            return None;
        }
        let mut edges = Vec::new();
        for (from, to, expr, callee) in paths {
            if !(forward.contains(&from) && backward.contains(&to)) {
                continue;
            }
            let item = match (expr, callee) {
                (Some(expr), None) if expr.is_empty() => Item::Epsilon,
                (Some(expr), None) => {
                    grammar.lexemes.push(expr);
                    grammar.ends.push((name.clone(), to));
                    Item::Lexeme(grammar.lexemes.len() - 1)
                }
                (None, Some(callee)) => Item::Call(callee),
                _ => unreachable!("a path is a lexeme or a call"),
            };
            edges.push((from, to, item));
        }
        grammar
            .hubs
            .extend(kept.into_iter().map(|state| (name.clone(), state)));
        grammar.rules.insert(name, Rule { edges });
    }
    Some(grammar)
}

fn closure(edges: &[(usize, usize)], start: usize, reverse: bool) -> BTreeSet<usize> {
    let mut seen = BTreeSet::from([start]);
    let mut pending = vec![start];
    while let Some(state) = pending.pop() {
        for (from, to) in edges {
            let (source, target) = if reverse { (to, from) } else { (from, to) };
            if *source == state && seen.insert(*target) {
                pending.push(*target);
            }
        }
    }
    seen
}

type Bytes = [bool; 256];

/// The lexemes that may come next from a state, and whether the rule may end
/// without consuming another lexeme.
#[derive(Clone, Default, PartialEq, Eq)]
struct Next {
    lexemes: BTreeSet<usize>,
    ends: bool,
}

/// Lexemes that may be allowed together at one input position.
type Allowed = BTreeSet<usize>;

struct Analysis<'a> {
    grammar: &'a Grammar,
    nullable_lexemes: Vec<bool>,
    nullable_rules: BTreeSet<String>,
    first: BTreeMap<String, BTreeSet<usize>>,
    /// For each rule, the lexemes that may follow its end, one set per
    /// calling context (the empty set is the end of the completion).
    returns: BTreeMap<String, BTreeSet<Allowed>>,
}

impl Analysis<'_> {
    fn transparent(&self, item: &Item) -> bool {
        match item {
            Item::Epsilon => true,
            Item::Lexeme(id) => self.nullable_lexemes[*id],
            Item::Call(name) => self.nullable_rules.contains(name),
        }
    }
    fn next(&self, rule: &str, state: usize) -> Next {
        let edges = &self.grammar.rules[rule].edges;
        let mut next = Next::default();
        let mut seen = BTreeSet::from([state]);
        let mut pending = vec![state];
        while let Some(state) = pending.pop() {
            next.ends |= state == 1;
            for (_, to, item) in edges.iter().filter(|(from, _, _)| *from == state) {
                match item {
                    Item::Lexeme(id) => {
                        next.lexemes.insert(*id);
                    }
                    Item::Call(name) => next.lexemes.extend(self.first[name].iter().copied()),
                    Item::Epsilon => {}
                }
                if self.transparent(item) && seen.insert(*to) {
                    pending.push(*to);
                }
            }
        }
        next
    }
    /// Whether some position predicts one rule from two call sites at once,
    /// so a single lexeme can be active in two calling contexts.
    fn predicts_twice(&self) -> bool {
        self.grammar.rules.iter().any(|(_, rule)| {
            rule.edges.iter().any(|(start, _, _)| {
                let mut sites = BTreeSet::new();
                let mut callees = BTreeSet::new();
                let mut seen = BTreeSet::from([*start]);
                let mut pending = vec![*start];
                while let Some(state) = pending.pop() {
                    for (index, (_, to, item)) in rule
                        .edges
                        .iter()
                        .enumerate()
                        .filter(|(_, (from, _, _))| *from == state)
                    {
                        if let Item::Call(callee) = item {
                            if sites.insert(index) && !callees.insert(callee) {
                                return true;
                            }
                        }
                        if self.transparent(item) && seen.insert(*to) {
                            pending.push(*to);
                        }
                    }
                }
                false
            })
        })
    }
    /// What may be allowed right after a lexeme or call ending in `state` of
    /// `rule`, one set per calling context of `rule`.
    fn after(&self, rule: &str, state: usize) -> BTreeSet<Allowed> {
        let next = self.next(rule, state);
        if !next.ends {
            return BTreeSet::from([next.lexemes]);
        }
        self.returns[rule]
            .iter()
            .map(|context| next.lexemes.union(context).copied().collect())
            .collect()
    }
}

fn analyze(grammar: &Grammar, nullable_lexemes: Vec<bool>) -> Option<Analysis<'_>> {
    let mut analysis = Analysis {
        grammar,
        nullable_lexemes,
        nullable_rules: BTreeSet::new(),
        first: grammar.rules.keys().map(|n| (n.clone(), BTreeSet::new())).collect(),
        returns: grammar.rules.keys().map(|n| (n.clone(), BTreeSet::new())).collect(),
    };
    loop {
        let nullable = grammar
            .rules
            .keys()
            .filter(|name| analysis.next(name, 0).ends)
            .cloned()
            .collect::<BTreeSet<_>>();
        if nullable == analysis.nullable_rules {
            break;
        }
        analysis.nullable_rules = nullable;
    }
    loop {
        let first = grammar
            .rules
            .keys()
            .map(|name| (name.clone(), analysis.next(name, 0).lexemes))
            .collect::<BTreeMap<_, _>>();
        if first == analysis.first {
            break;
        }
        analysis.first = first;
    }
    analysis
        .returns
        .get_mut("root")
        .unwrap()
        .insert(Allowed::new());
    loop {
        let mut returns = analysis.returns.clone();
        for (name, rule) in &grammar.rules {
            for (_, to, item) in &rule.edges {
                if let Item::Call(callee) = item {
                    let after = analysis.after(name, *to);
                    returns.get_mut(callee).unwrap().extend(after);
                }
            }
        }
        if returns == analysis.returns {
            break;
        }
        if returns.values().map(BTreeSet::len).sum::<usize>() > ALLOWED_SETS {
            return None;
        }
        analysis.returns = returns;
    }
    Some(analysis)
}

fn any_bytes(min: u32) -> RegexAst {
    RegexAst::Repeat(Box::new(RegexAst::ByteSet(vec![u32::MAX; 8])), min, u32::MAX)
}

fn byte_set(bytes: &Bytes) -> RegexAst {
    let mut set = vec![0u32; 8];
    for (byte, present) in bytes.iter().enumerate() {
        if *present {
            set[byte / 32] |= 1 << (byte % 32);
        }
    }
    RegexAst::ByteSet(set)
}

fn nonempty(builder: &mut RegexBuilder, expr: RegexAst) -> Option<bool> {
    let expr = builder.mk(&expr).ok()?;
    builder
        .to_regex_limited(expr, FUEL)
        .ok()
        .map(|mut regex| !regex.always_empty())
}

/// Why greedy lexing of a grammar was not proven language-preserving.
enum Unproven {
    /// Some text completing this lexeme may be continued by an allowed
    /// lexeme with a byte that may start the next one.
    Overrun(usize),
    /// The proof exceeded its bounds.
    Bounds,
}

/// Prove that greedy lexing loses no string of the grammar.
fn greedy_safe(grammar: &Grammar) -> Result<(), Unproven> {
    let mut builder = RegexBuilder::new();
    let compiled = grammar
        .lexemes
        .iter()
        .map(|expr| builder.mk(&expr.regex()).map_err(|_| Unproven::Bounds))
        .collect::<Result<Vec<_>, _>>()?;
    let proper = compiled
        .iter()
        .map(|e| {
            builder
                .mk(&RegexAst::And(vec![RegexAst::ExprRef(*e), any_bytes(1)]))
                .map_err(|_| Unproven::Bounds)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut first = Vec::with_capacity(proper.len());
    for expr in &proper {
        let mut regex = builder
            .to_regex_limited(*expr, FUEL)
            .map_err(|_| Unproven::Bounds)?;
        let initial = regex.initial_state();
        let mut bytes = [false; 256];
        if !initial.is_dead() {
            for (byte, present) in bytes.iter_mut().enumerate() {
                *present = !regex.transition(initial, byte as u8).is_dead();
            }
        }
        first.push(bytes);
    }
    let analysis = analyze(
        grammar,
        compiled.iter().map(|e| builder.is_nullable(*e)).collect(),
    )
    .ok_or(Unproven::Bounds)?;
    // What may be allowed after each lexeme, per calling context.
    let mut follow = vec![BTreeSet::<Allowed>::new(); grammar.lexemes.len()];
    for (name, rule) in &grammar.rules {
        for (_, to, item) in &rule.edges {
            if let Item::Lexeme(id) = item {
                follow[*id].extend(analysis.after(name, *to));
            }
        }
    }
    let follow_bytes = follow
        .iter()
        .map(|sets| {
            let mut bytes = [false; 256];
            for id in sets.iter().flatten() {
                for (byte, present) in first[*id].iter().enumerate() {
                    bytes[byte] |= present;
                }
            }
            bytes
        })
        .collect::<Vec<_>>();
    let overlaps = |a: &Bytes, b: &Bytes| a.iter().zip(b).any(|(x, y)| *x && *y);

    // Sets of lexemes the parser may allow at one position: the start, what
    // follows each lexeme in each context, and the union of what follows
    // lexemes that were allowed together and match a common text. Once two
    // parses can be alive at one position, one lexeme may be active in all
    // of its contexts at once, so its contexts are also taken together.
    let mut allowed = BTreeSet::from([analysis.next("root", 0).lexemes]);
    allowed.extend(follow.iter().flatten().cloned());
    let mut ambiguous = analysis.predicts_twice();
    let mut shared = BTreeMap::<(usize, usize), bool>::new();
    loop {
        let was_ambiguous = ambiguous;
        let before = allowed.len();
        if ambiguous {
            allowed.extend(
                follow
                    .iter()
                    .map(|sets| sets.iter().flatten().copied().collect::<Allowed>()),
            );
        }
        if allowed.len() > ALLOWED_SETS {
            return Err(Unproven::Bounds);
        }
        let mut added = Vec::new();
        for set in &allowed {
            for &a in set {
                for &b in set.range(a + 1..) {
                    if !overlaps(&first[a], &first[b]) {
                        continue;
                    }
                    let common = match shared.get(&(a, b)) {
                        Some(common) => *common,
                        None => {
                            let common = nonempty(
                                &mut builder,
                                RegexAst::And(vec![
                                    RegexAst::ExprRef(proper[a]),
                                    RegexAst::ExprRef(proper[b]),
                                ]),
                            )
                            .unwrap_or(true);
                            shared.insert((a, b), common);
                            common
                        }
                    };
                    if common {
                        ambiguous = true;
                        for left in &follow[a] {
                            for right in &follow[b] {
                                let union: Allowed = left.union(right).copied().collect();
                                if !allowed.contains(&union) {
                                    added.push(union);
                                }
                            }
                        }
                    }
                }
            }
        }
        allowed.extend(added);
        if allowed.len() == before && ambiguous == was_ambiguous {
            break;
        }
    }

    let mut proven = BTreeSet::new();
    for set in &allowed {
        for &ended in set {
            if !follow_bytes[ended].iter().any(|present| *present) {
                continue;
            }
            for &continued in set {
                if !overlaps(&first[ended], &first[continued]) || !proven.insert((ended, continued))
                {
                    continue;
                }
                // Some text `x` completes `ended`, and `continued` can consume
                // `x` followed by a byte that may start the next lexeme.
                let overrun = RegexAst::And(vec![
                    RegexAst::ExprRef(compiled[continued]),
                    RegexAst::Concat(vec![
                        RegexAst::ExprRef(proper[ended]),
                        byte_set(&follow_bytes[ended]),
                        any_bytes(0),
                    ]),
                ]);
                match nonempty(&mut builder, overrun) {
                    Some(false) => {}
                    Some(true) => return Err(Unproven::Overrun(ended)),
                    None => return Err(Unproven::Bounds),
                }
            }
        }
    }
    Ok(())
}

fn render(grammar: &Grammar) -> String {
    let mut output = String::from("%llguidance {}\nstart: root\n");
    for (name, rule) in &grammar.rules {
        let state = |s: usize| format!("{name}__{s}");
        output.push_str(&format!("{name}: {}\n", state(1)));
        let mut productions = BTreeMap::<usize, Vec<String>>::new();
        productions.entry(0).or_default().push("\"\"".into());
        for (from, to, item) in &rule.edges {
            let symbol = match item {
                Item::Epsilon => String::new(),
                Item::Lexeme(id) => format!(" L{id}"),
                Item::Call(callee) => format!(" {callee}"),
            };
            productions
                .entry(*to)
                .or_default()
                .push(format!("{}{symbol}", state(*from)));
        }
        for (target, alternatives) in productions {
            output.push_str(&format!("{}: {}\n", state(target), alternatives.join(" | ")));
        }
    }
    for (id, expr) in grammar.lexemes.iter().enumerate() {
        output.push_str(&format!("L{id}: {}\n", expr.render()));
    }
    output
}

/// Lark for the grammar with regular stretches as lexemes, or `None` when the
/// lexical form cannot be proven to accept exactly the grammar's language.
/// A hub where greedy lexing could overrun a lexeme's end is merged into the
/// lexemes through it, and the proof is repeated.
pub(super) fn compile(rules: &Rules, regular: &BTreeMap<String, Expr>) -> Option<String> {
    let mut forced = BTreeSet::new();
    loop {
        let grammar = automata(rules, regular, &forced)?;
        match greedy_safe(&grammar) {
            Ok(()) => return Some(render(&grammar)),
            Err(Unproven::Overrun(lexeme)) if grammar.hubs.contains(&grammar.ends[lexeme]) => {
                forced.insert(grammar.ends[lexeme].clone());
            }
            Err(Unproven::Overrun(_) | Unproven::Bounds) => return None,
        }
    }
}
