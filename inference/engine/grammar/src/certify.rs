//! Greedy-exactness certification.
//!
//! llguidance lexes greedily without backtracking: a lexeme ends only when
//! the next byte kills every allowed lexeme, and all lexemes accepting there
//! are emitted. A string is lost exactly when a completed lexeme `a` is
//! abandoned because a lexeme `b` allowed with it consumes a byte that may
//! follow `a`. This module finds every such pair over a sound
//! over-approximation of the lexemes an Earley row can allow together, and
//! renders the lexemes involved at character level until none remain.
//! One-character lexemes never extend one another, so the loop terminates.
//!
//! An Earley row allows the lexemes of its own position and, where that
//! position can finish its rule, those of the caller that is actually
//! waiting. Allowed sets are therefore kept per calling context rather than
//! merged across every caller of a rule.
use crate::{
    lexeme::{EdgeKind, Plan},
    terminal::{has_byte, intersects, union, Bytes, Term, TermId, Terms},
};
use llguidance::derivre::{raw::RelevanceCache, ExprRef, RegexAst, RegexBuilder};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Derivative work allowed for one emptiness question.
const FUEL: u64 = 200_000;
/// Derivative work allowed per compilation. A question left unanswered
/// counts as positive, which only makes the rendering finer.
const TOTAL_FUEL: u64 = 20_000_000;
/// Calling contexts kept apart per rule; beyond this they are merged, which
/// only over-approximates what is allowed together. Contexts are call
/// sites, which flattening bounds, so this bound is only a backstop.
const CONTEXTS: usize = 1024;

#[derive(Default)]
pub(crate) struct Certification {
    pub questions: usize,
    pub exhausted: usize,
    pub characters: usize,
}

struct Occurrence {
    rule: usize,
    edge: usize,
    lexeme: TermId,
    end: usize,
    single: bool,
}

type Set = Vec<u64>;

fn set_new(size: usize) -> Set {
    vec![0; size.div_ceil(64)]
}
fn set_insert(set: &mut Set, index: usize) {
    set[index / 64] |= 1 << (index % 64);
}
fn set_union(target: &mut Set, source: &Set) -> bool {
    let mut changed = false;
    for (t, s) in target.iter_mut().zip(source) {
        let merged = *t | s;
        changed |= merged != *t;
        *t = merged;
    }
    changed
}
fn set_or(a: &Set, b: &Set) -> Set {
    a.iter().zip(b).map(|(x, y)| x | y).collect()
}
fn set_iter(set: &Set) -> impl Iterator<Item = usize> + '_ {
    set.iter().enumerate().flat_map(|(word, &bits)| {
        (0..64)
            .filter(move |bit| bits & (1 << bit) != 0)
            .map(move |bit| word * 64 + bit)
    })
}

fn subset(a: &Set, b: &Set) -> bool {
    a.iter().zip(b).all(|(x, y)| x & !y == 0)
}

/// Add `set` to `sets` unless an existing context already allows all of it
/// (it could add no pair); past the context bound, merge.
fn add_context(sets: &mut Vec<Set>, set: Set) -> bool {
    if sets.iter().any(|existing| subset(&set, existing)) {
        return false;
    }
    sets.retain(|existing| !subset(existing, &set));
    if sets.len() < CONTEXTS {
        sets.push(set);
        return true;
    }
    let mut merged = set;
    for existing in sets.iter() {
        set_union(&mut merged, existing);
    }
    let changed = sets.len() != 1 || sets[0] != merged;
    *sets = vec![merged];
    changed
}

/// derivre expressions for terminals and batched emptiness questions.
struct Regexes {
    builder: RegexBuilder,
    exprs: Vec<Option<ExprRef>>,
    fuel: u64,
}

impl Regexes {
    fn new() -> Self {
        Self {
            builder: RegexBuilder::new(),
            exprs: Vec::new(),
            fuel: TOTAL_FUEL,
        }
    }
    fn byte_set(bytes: &Bytes) -> RegexAst {
        let mut words = vec![0u32; 8];
        for byte in 0..=255u8 {
            if has_byte(bytes, byte) {
                words[byte as usize / 32] |= 1 << (byte % 32);
            }
        }
        RegexAst::ByteSet(words)
    }
    fn any_bytes(min: u32) -> RegexAst {
        RegexAst::Repeat(Box::new(Self::byte_set(&[u64::MAX; 4])), min, u32::MAX)
    }
    /// Terminals are interned after their operands, so building in id
    /// order never recurses.
    fn expr(&mut self, terms: &Terms, id: TermId) -> ExprRef {
        while self.exprs.len() <= id as usize {
            self.exprs.push(None);
        }
        if let Some(expr) = self.exprs[id as usize] {
            return expr;
        }
        for current in 0..=id {
            if self.exprs[current as usize].is_some() {
                continue;
            }
            let child = |term: TermId, exprs: &[Option<ExprRef>]| {
                RegexAst::ExprRef(exprs[term as usize].expect("operands precede their terminal"))
            };
            let ast = match terms.get(current) {
                Term::Chars(set) => {
                    let mut class = String::from("[");
                    for &(start, end) in set.ranges() {
                        class.push_str(&format!("\\x{{{start:x}}}-\\x{{{end:x}}}"));
                    }
                    class.push(']');
                    RegexAst::Regex(class)
                }
                Term::Literal(text) => RegexAst::Literal(text.clone()),
                Term::Seq(parts) => {
                    RegexAst::Concat(parts.iter().map(|&p| child(p, &self.exprs)).collect())
                }
                Term::Alt(parts) => {
                    RegexAst::Or(parts.iter().map(|&p| child(p, &self.exprs)).collect())
                }
                Term::Repeat(part, min, max) => RegexAst::Repeat(
                    Box::new(child(*part, &self.exprs)),
                    *min,
                    max.unwrap_or(u32::MAX),
                ),
                Term::NonEmpty(part) => {
                    RegexAst::And(vec![child(*part, &self.exprs), Self::any_bytes(1)])
                }
            };
            let expr = self
                .builder
                .mk(&ast)
                .expect("terminal expressions are well formed");
            self.exprs[current as usize] = Some(expr);
        }
        self.exprs[id as usize].unwrap()
    }
    fn union(&mut self, terms: &Terms, parts: &[TermId]) -> RegexAst {
        RegexAst::Or(
            parts
                .iter()
                .map(|&part| RegexAst::ExprRef(self.expr(terms, part)))
                .collect(),
        )
    }
    /// Some text of `a` is also text of one of `others`.
    fn shared(&mut self, terms: &Terms, a: TermId, others: &[TermId]) -> ExprRef {
        let ast = RegexAst::And(vec![
            RegexAst::ExprRef(self.expr(terms, a)),
            self.union(terms, others),
        ]);
        self.builder.mk(&ast).expect("intersection of terminals")
    }
    /// Some text of `ended`, then a byte of `follow`, begins text of one of
    /// `continued`.
    fn overrun(
        &mut self,
        terms: &Terms,
        ended: TermId,
        follow: &Bytes,
        continued: &[TermId],
    ) -> ExprRef {
        let ast = RegexAst::And(vec![
            self.union(terms, continued),
            RegexAst::Concat(vec![
                RegexAst::ExprRef(self.expr(terms, ended)),
                Self::byte_set(follow),
                Self::any_bytes(0),
            ]),
        ]);
        self.builder.mk(&ast).expect("overrun of terminals")
    }
    /// Non-emptiness of each expression; unanswerable questions count as
    /// non-empty.
    fn nonempty(&mut self, questions: &[ExprRef], report: &mut Certification) -> Vec<bool> {
        if questions.is_empty() {
            return Vec::new();
        }
        report.questions += questions.len();
        let mut exprs = self.builder.exprset().clone();
        let mut relevance = RelevanceCache::new();
        questions
            .iter()
            .map(|&question| {
                let answer = if self.fuel == 0 {
                    None
                } else {
                    let before = exprs.cost();
                    let answer = relevance
                        .is_non_empty_limited(&mut exprs, question, FUEL.min(self.fuel))
                        .ok();
                    self.fuel = self.fuel.saturating_sub(exprs.cost() - before);
                    answer
                };
                if answer.is_none() {
                    report.exhausted += 1;
                }
                answer.unwrap_or(true)
            })
            .collect()
    }
}

struct Analysis {
    occurrences: Vec<Occurrence>,
    /// Sets of occurrences an Earley row at each position may allow, one
    /// per calling context.
    allowed: Vec<Vec<Set>>,
    /// First bytes of everything allowed at each position.
    next: Vec<Bytes>,
}

/// Allowed sets, given the positions known to be active together.
///
/// Earley shares one callee instance among every item that predicts it at
/// the same input position, and completing it resumes all of them. Call
/// sites predicted together — within one prediction closure, or from two
/// positions active together — therefore contribute their continuations to
/// one context of the callee.
fn analyze(plan: &Plan, terms: &Terms, together: &BTreeSet<(usize, usize)>) -> Analysis {
    let mut offsets = Vec::with_capacity(plan.rules.len());
    let mut positions = 0usize;
    for rule in &plan.rules {
        offsets.push(positions);
        positions += rule.boundaries as usize;
    }
    let position = |rule: usize, boundary: u32| offsets[rule] + boundary as usize;
    let rule_of = {
        let mut rule_of = vec![0; positions];
        for (index, rule) in plan.rules.iter().enumerate() {
            for boundary in 0..rule.boundaries {
                rule_of[position(index, boundary)] = index;
            }
        }
        rule_of
    };

    let mut occurrences = Vec::new();
    let mut direct = vec![Vec::new(); positions];
    for (rule_index, rule) in plan.rules.iter().enumerate() {
        for (edge_index, edge) in rule.edges.iter().enumerate() {
            if let EdgeKind::Lexeme { lexeme, .. } = edge.kind {
                direct[position(rule_index, edge.from)].push(occurrences.len());
                occurrences.push(Occurrence {
                    rule: rule_index,
                    edge: edge_index,
                    lexeme,
                    end: position(rule_index, edge.to),
                    single: terms.single_character(lexeme),
                });
            }
        }
    }
    let empty = set_new(occurrences.len());

    // Rules that can match the empty text.
    let mut nullable = vec![false; plan.rules.len()];
    loop {
        let mut changed = false;
        for (index, rule) in plan.rules.iter().enumerate() {
            if nullable[index] {
                continue;
            }
            let mut reached = vec![false; rule.boundaries as usize];
            reached[0] = true;
            loop {
                let mut grew = false;
                for edge in &rule.edges {
                    let passes = match edge.kind {
                        EdgeKind::Eps => true,
                        EdgeKind::Call { rule, min, .. } => min == 0 || nullable[rule],
                        EdgeKind::Lexeme { .. } => false,
                    };
                    if passes && reached[edge.from as usize] && !reached[edge.to as usize] {
                        reached[edge.to as usize] = true;
                        grew = true;
                    }
                }
                if !grew {
                    break;
                }
            }
            if reached[1] {
                nullable[index] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Within one calling context: moves that consume nothing stay inside
    // the rule (ε, skipping a nullable call); a call contributes the
    // callee's first lexemes.
    let mut moves: Vec<Vec<usize>> = vec![Vec::new(); positions];
    let mut calls: Vec<Vec<usize>> = vec![Vec::new(); positions];
    for (index, rule) in plan.rules.iter().enumerate() {
        for edge in &rule.edges {
            let (from, to) = (position(index, edge.from), position(index, edge.to));
            match edge.kind {
                EdgeKind::Eps => moves[from].push(to),
                EdgeKind::Call {
                    rule: callee, min, ..
                } => {
                    calls[from].push(callee);
                    if min == 0 || nullable[callee] {
                        moves[from].push(to);
                    }
                }
                EdgeKind::Lexeme { .. } => {}
            }
        }
    }
    let mut local: Vec<Set> = vec![empty.clone(); positions];
    let mut exits = vec![false; positions];
    for (index, _) in plan.rules.iter().enumerate() {
        exits[position(index, 1)] = true;
    }
    loop {
        let mut changed = false;
        for p in (0..positions).rev() {
            let mut set = local[p].clone();
            for &occurrence in &direct[p] {
                set_insert(&mut set, occurrence);
            }
            for &callee in &calls[p] {
                let first = local[position(callee, 0)].clone();
                set_union(&mut set, &first);
            }
            let mut exit = exits[p];
            for &q in &moves[p] {
                let reached = local[q].clone();
                set_union(&mut set, &reached);
                exit |= exits[q];
            }
            if set != local[p] || exit != exits[p] {
                local[p] = set;
                exits[p] = exit;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Call sites, and those each position's prediction closure reaches.
    struct Site {
        caller: usize,
        callee: usize,
        to: usize,
        max: Option<u32>,
    }
    let mut sites = Vec::new();
    let mut site_starts = vec![Vec::new(); positions];
    for (index, rule) in plan.rules.iter().enumerate() {
        for edge in &rule.edges {
            if let EdgeKind::Call {
                rule: callee, max, ..
            } = edge.kind
            {
                site_starts[position(index, edge.from)].push(sites.len());
                sites.push(Site {
                    caller: index,
                    callee,
                    to: position(index, edge.to),
                    max,
                });
            }
        }
    }
    let mut predicted: Vec<Set> = vec![set_new(sites.len()); positions];
    loop {
        let mut changed = false;
        for p in (0..positions).rev() {
            let mut set = predicted[p].clone();
            for &site in &site_starts[p] {
                set_insert(&mut set, site);
            }
            for &callee in &calls[p] {
                let entered = predicted[position(callee, 0)].clone();
                set_union(&mut set, &entered);
            }
            for &q in &moves[p] {
                let reached = predicted[q].clone();
                set_union(&mut set, &reached);
            }
            if set != predicted[p] {
                predicted[p] = set;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Pairs of sites that may wait on one instance of their callee.
    let mut shared: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut pair_sites = |active: Set| {
        let members = set_iter(&active).collect::<Vec<_>>();
        for (index, &a) in members.iter().enumerate() {
            for &b in &members[index + 1..] {
                if sites[a].callee == sites[b].callee {
                    shared.insert((a, b));
                }
            }
        }
    };
    for set in &predicted {
        pair_sites(set.clone());
    }
    for &(p, q) in together {
        pair_sites(set_or(&predicted[p], &predicted[q]));
    }

    // What may follow each rule's completion, one set per calling context.
    let mut after: Vec<Vec<Set>> = vec![Vec::new(); plan.rules.len()];
    after[0].push(empty.clone());
    loop {
        let continuations = |site: &Site, after: &Vec<Vec<Set>>| {
            let mut base = local[site.to].clone();
            if site.max != Some(1) {
                set_union(&mut base, &local[position(site.callee, 0)]);
            }
            if exits[site.to] {
                after[site.caller]
                    .iter()
                    .map(|context| set_or(&base, context))
                    .collect::<Vec<_>>()
            } else {
                vec![base]
            }
        };
        let mut changed = false;
        for site in &sites {
            for context in continuations(site, &after) {
                changed |= add_context(&mut after[site.callee], context);
            }
        }
        for &(a, b) in &shared {
            let (left, right) = (
                continuations(&sites[a], &after),
                continuations(&sites[b], &after),
            );
            for x in &left {
                for y in &right {
                    changed |= add_context(&mut after[sites[a].callee], set_or(x, y));
                }
            }
        }
        if !changed {
            break;
        }
    }

    let allowed = (0..positions)
        .map(|p| {
            if !exits[p] {
                return vec![local[p].clone()];
            }
            let contexts = &after[rule_of[p]];
            if contexts.is_empty() {
                return vec![local[p].clone()];
            }
            let mut sets = Vec::new();
            for context in contexts {
                let set = set_or(&local[p], context);
                if !sets.contains(&set) {
                    sets.push(set);
                }
            }
            sets
        })
        .collect::<Vec<Vec<Set>>>();
    let next = allowed
        .iter()
        .map(|sets| {
            let mut bytes = [0; 4];
            for set in sets {
                for occurrence in set_iter(set) {
                    union(&mut bytes, terms.first(occurrences[occurrence].lexeme));
                }
            }
            bytes
        })
        .collect();
    Analysis {
        occurrences,
        allowed,
        next,
    }
}

fn ordered(a: TermId, b: TermId) -> (TermId, TermId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Every set of occurrences allowed together — each position's sets, and
/// unions of sets at positions active together because two lexemes matched
/// the same text — and those pairs of positions.
fn simultaneity(
    analysis: &Analysis,
    terms: &Terms,
    regexes: &mut Regexes,
    overlaps: &mut HashMap<(TermId, TermId), bool>,
    report: &mut Certification,
) -> (Vec<Set>, BTreeSet<(usize, usize)>) {
    let occurrences = &analysis.occurrences;
    let lexeme = |index: usize| occurrences[index].lexeme;
    let mut sets: Vec<Set> = Vec::new();
    let mut known: HashSet<Set> = HashSet::new();
    for position_sets in &analysis.allowed {
        for set in position_sets {
            if known.insert(set.clone()) {
                sets.push(set.clone());
            }
        }
    }
    let mut together: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut scanned = 0;
    while scanned < sets.len() {
        let batch = sets[scanned..].to_vec();
        scanned = sets.len();
        // Which lexemes of each set share text with another of the set.
        let mut pending: Vec<(TermId, Vec<TermId>)> = Vec::new();
        let mut seen = HashSet::new();
        for set in &batch {
            let members = set_iter(set).collect::<Vec<_>>();
            for &a in &members {
                let others = members
                    .iter()
                    .map(|&b| lexeme(b))
                    .filter(|&b| {
                        b != lexeme(a)
                            && intersects(terms.first(lexeme(a)), terms.first(b))
                            && !overlaps.contains_key(&ordered(lexeme(a), b))
                    })
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                if !others.is_empty() && seen.insert((lexeme(a), others.clone())) {
                    pending.push((lexeme(a), others));
                }
            }
        }
        let questions = pending
            .iter()
            .map(|(a, others)| regexes.shared(terms, *a, others))
            .collect::<Vec<_>>();
        let answers = regexes.nonempty(&questions, report);
        // Resolve positive unions pair by pair.
        let mut pairs = Vec::new();
        let mut queued = HashSet::new();
        for ((a, others), positive) in pending.into_iter().zip(answers) {
            for b in others {
                let key = ordered(a, b);
                if positive {
                    if !overlaps.contains_key(&key) && queued.insert(key) {
                        pairs.push(key);
                    }
                } else {
                    overlaps.insert(key, false);
                }
            }
        }
        let questions = pairs
            .iter()
            .map(|&(a, b)| regexes.shared(terms, a, &[b]))
            .collect::<Vec<_>>();
        for (key, answer) in pairs.into_iter().zip(regexes.nonempty(&questions, report)) {
            overlaps.insert(key, answer);
        }
        for set in &batch {
            let members = set_iter(set).collect::<Vec<_>>();
            for (index, &a) in members.iter().enumerate() {
                for &b in &members[index + 1..] {
                    let same_text = lexeme(a) == lexeme(b)
                        || overlaps
                            .get(&ordered(lexeme(a), lexeme(b)))
                            .copied()
                            .unwrap_or(false);
                    if !same_text {
                        continue;
                    }
                    let (p, q) = (occurrences[a].end, occurrences[b].end);
                    let pair = if p <= q { (p, q) } else { (q, p) };
                    if p == q || !together.insert(pair) {
                        continue;
                    }
                    for left in &analysis.allowed[p] {
                        for right in &analysis.allowed[q] {
                            let combined = set_or(left, right);
                            if known.insert(combined.clone()) {
                                sets.push(combined);
                            }
                        }
                    }
                }
            }
        }
    }
    (sets, together)
}

/// Lexeme occurrences that must be rendered at character level, as
/// (rule, edge) pairs; empty when the plan is certified.
fn conflicts(
    plan: &Plan,
    terms: &Terms,
    regexes: &mut Regexes,
    overlaps: &mut HashMap<(TermId, TermId), bool>,
    overruns: &mut HashMap<(TermId, TermId, Bytes), bool>,
    report: &mut Certification,
) -> BTreeSet<(usize, usize)> {
    // Positions active together change which call sites share a callee, so
    // analysis and simultaneity are solved together.
    let mut together: BTreeSet<(usize, usize)> = BTreeSet::new();
    let (analysis, sets) = loop {
        let analysis = analyze(plan, terms, &together);
        let (sets, found) = simultaneity(&analysis, terms, regexes, overlaps, report);
        if found.is_subset(&together) {
            break (analysis, sets);
        }
        together.extend(found);
    };
    let occurrences = &analysis.occurrences;
    let lexeme = |index: usize| occurrences[index].lexeme;
    // Overrun questions: `a` ended, and a lexeme allowed with it continues
    // over a byte that may follow `a`.
    let mut pending: Vec<(TermId, Bytes, Vec<TermId>)> = Vec::new();
    let mut seen = HashSet::new();
    for set in &sets {
        let members = set_iter(set).collect::<Vec<_>>();
        for &a in &members {
            let follow = analysis.next[occurrences[a].end];
            if follow == [0; 4] {
                continue;
            }
            let continued = members
                .iter()
                .filter(|&&b| !occurrences[b].single)
                .map(|&b| lexeme(b))
                .filter(|&b| {
                    intersects(terms.first(lexeme(a)), terms.first(b))
                        && !overruns.contains_key(&(lexeme(a), b, follow))
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            if !continued.is_empty() && seen.insert((lexeme(a), follow, continued.clone())) {
                pending.push((lexeme(a), follow, continued));
            }
        }
    }
    let questions = pending
        .iter()
        .map(|(a, follow, continued)| regexes.overrun(terms, *a, follow, continued))
        .collect::<Vec<_>>();
    let answers = regexes.nonempty(&questions, report);
    let mut pairs = Vec::new();
    let mut queued = HashSet::new();
    for ((a, follow, continued), positive) in pending.into_iter().zip(answers) {
        for b in continued {
            let key = (a, b, follow);
            if positive {
                if !overruns.contains_key(&key) && queued.insert(key) {
                    pairs.push(key);
                }
            } else {
                overruns.insert(key, false);
            }
        }
    }
    let questions = pairs
        .iter()
        .map(|(a, b, follow)| regexes.overrun(terms, *a, follow, &[*b]))
        .collect::<Vec<_>>();
    for (key, answer) in pairs.into_iter().zip(regexes.nonempty(&questions, report)) {
        overruns.insert(key, answer);
    }

    let mut marked = BTreeSet::new();
    for set in &sets {
        let members = set_iter(set).collect::<Vec<_>>();
        for &a in &members {
            let follow = analysis.next[occurrences[a].end];
            for &b in &members {
                if occurrences[b].single {
                    continue;
                }
                let key = (lexeme(a), lexeme(b), follow);
                if overruns.get(&key).copied().unwrap_or(false) {
                    for occurrence in [a, b] {
                        if !occurrences[occurrence].single {
                            marked.insert((
                                occurrences[occurrence].rule,
                                occurrences[occurrence].edge,
                            ));
                        }
                    }
                }
            }
        }
    }
    marked
}

/// Render conflicting lexemes at character level until the plan is
/// certified.
pub(crate) fn certify(plan: &mut Plan, terms: &mut Terms) -> Certification {
    let mut report = Certification::default();
    let mut regexes = Regexes::new();
    let mut overlaps = HashMap::new();
    let mut overruns = HashMap::new();
    loop {
        let marked = conflicts(
            plan,
            terms,
            &mut regexes,
            &mut overlaps,
            &mut overruns,
            &mut report,
        );
        if marked.is_empty() {
            return report;
        }
        report.characters += marked.len();
        for (rule, edge) in marked {
            plan.characters(rule, edge, terms);
        }
    }
}
