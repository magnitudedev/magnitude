//! The checker's private prover over `expr` nodes (spec §24.5 R15).
//!
//! Every symbolic integer of the checker is an `IntExpr` in the definition's
//! arena. A proof normalizes the nodes it inspects into a transient
//! polynomial form over symbol atoms (with floor quotients and remainders as
//! opaque atoms), rewrites under the facts, and discards the form when it
//! returns. The form is never stored on a checked datum, never crosses a
//! module boundary, and every value the checker keeps is re-interned into the
//! arena through [`intern`].
//!
//! Dimension and Index symbols carry explicit nonnegative bounds. Ordinary
//! signed Integer/I32 symbols do not. The prover establishes `e >= 0` and `e == 0` under facts by
//! rewriting quotients and remainders through `x = c * (x / c) + x % c` and
//! bounding atoms that carry an upper bound. It is sound and incomplete: a
//! failed proof is a residual the caller reports.

use crate::expr::poly::{intern, normalize_int, single_atom_of, Atom, Poly};
use crate::expr::{AnyExpr, ExprArena, IntExpr, SymbolId};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Public (crate) surface over nodes
// ---------------------------------------------------------------------------

/// The constant value of a node, when it has one.
pub(crate) fn constant(arena: &ExprArena, e: IntExpr) -> Option<i64> {
    normalize_int(arena, e).as_constant()
}

pub(crate) fn is_zero(arena: &ExprArena, e: IntExpr) -> bool {
    normalize_int(arena, e).is_zero()
}

/// Whether two nodes denote the same integer as polynomials (no facts).
pub(crate) fn same(arena: &ExprArena, a: IntExpr, b: IntExpr) -> bool {
    let (a, b) = (normalize_int(arena, a), normalize_int(arena, b));
    a.valid && b.valid && a == b
}

/// Every symbol mentioned by `e`.
pub(crate) fn symbols(arena: &ExprArena, e: IntExpr) -> Vec<SymbolId> {
    arena.free_symbols(AnyExpr::Int(e))
}

pub(crate) fn mentions(arena: &ExprArena, e: IntExpr, symbol: SymbolId) -> bool {
    arena.free_symbols(AnyExpr::Int(e)).contains(&symbol)
}

/// The canonical (normalized, re-interned) form of a node. Two nodes with
/// equal polynomial meaning canonicalize to the same handle.
pub(crate) fn canonical(arena: &mut ExprArena, e: IntExpr) -> IntExpr {
    let p = normalize_int(arena, e);
    if p.valid {
        intern(arena, &p)
    } else {
        e
    }
}

/// Recompose a checked logical mixed-radix address. Quotient/remainder
/// decomposition is lossless: `n % d = n - (n / d) * d`. This reuses the
/// ordinary exact polynomial owner, so coefficient overflow cannot establish
/// coverage. Callers supply checked integer addresses, whose divisors are
/// positive on their execution path.
pub(crate) fn recompose_address(arena: &mut ExprArena, e: IntExpr) -> IntExpr {
    let mut p = normalize_int(arena, e);
    for atom in p.atoms() {
        if let Atom::Rem(numerator, denominator) = &atom {
            let quotient = numerator.quot(denominator);
            p = p.subst(&atom, &numerator.sub(&quotient.mul(denominator)));
        }
    }
    if p.valid {
        intern(arena, &p)
    } else {
        e
    }
}

/// `e / c` when every coefficient is divisible by `c`.
pub(crate) fn div_exact(arena: &mut ExprArena, e: IntExpr, c: i64) -> Option<IntExpr> {
    let p = normalize_int(arena, e).div_exact(c)?;
    Some(intern(arena, &p))
}

/// `e / d` when `d` is a nonzero constant or monomial that divides every term
/// of `e`, so the quotient is exact for every assignment.
pub(crate) fn divide_exact(arena: &mut ExprArena, e: IntExpr, d: IntExpr) -> Option<IntExpr> {
    let divisor = normalize_int(arena, d);
    if divisor.terms.len() != 1 {
        return None;
    }
    let p = normalize_int(arena, e).divide_exactly(&divisor)?;
    p.valid.then(|| intern(arena, &p))
}

/// If `e` is `c * symbol + rest` with `rest` free of `symbol`, `(c, rest)`.
pub(crate) fn linear_in(
    arena: &mut ExprArena,
    e: IntExpr,
    symbol: SymbolId,
) -> Option<(i64, IntExpr)> {
    let (c, rest) = normalize_int(arena, e).linear_in(&Atom::Symbol(symbol))?;
    Some((c, intern(arena, &rest)))
}

/// See [`Poly::linear_coefficient`].
pub(crate) fn linear_coefficient(
    arena: &mut ExprArena,
    e: IntExpr,
    symbol: SymbolId,
) -> Option<IntExpr> {
    let c = normalize_int(arena, e).linear_coefficient(symbol)?;
    Some(intern(arena, &c))
}

/// Simultaneous substitution of symbols, also inside quotients and remainders.
pub(crate) fn substitute(
    arena: &mut ExprArena,
    e: IntExpr,
    map: &dyn Fn(SymbolId) -> Option<IntExpr>,
) -> IntExpr {
    let p = normalize_int(arena, e);
    if !p.valid {
        return e;
    }
    let substituted = substitute_poly(arena, &p, map);
    if substituted.valid {
        intern(arena, &substituted)
    } else {
        e
    }
}

fn substitute_poly(arena: &ExprArena, p: &Poly, map: &dyn Fn(SymbolId) -> Option<IntExpr>) -> Poly {
    let mut out = Poly::constant(0);
    for (monomial, coefficient) in &p.terms {
        let mut term = Poly::constant(*coefficient);
        for (atom, power) in monomial {
            let factor = match atom {
                Atom::Symbol(s) => match map(*s) {
                    Some(value) => normalize_int(arena, value),
                    None => Poly::atom(atom.clone()),
                },
                Atom::Quot(n, d) => {
                    substitute_poly(arena, n, map).quot(&substitute_poly(arena, d, map))
                }
                Atom::Rem(n, d) => {
                    substitute_poly(arena, n, map).rem(&substitute_poly(arena, d, map))
                }
                Atom::Foreign(_) => Poly::atom(atom.clone()),
            };
            for _ in 0..*power {
                term = term.mul(&factor);
            }
        }
        out = out.add(&term);
    }
    out
}

/// Renders a node with symbol names, in normalized form.
pub(crate) fn display(arena: &ExprArena, e: IntExpr, name: &dyn Fn(SymbolId) -> String) -> String {
    format_poly(&normalize_int(arena, e), name)
}

fn format_atom(a: &Atom, name: &dyn Fn(SymbolId) -> String) -> String {
    match a {
        Atom::Symbol(s) => name(*s),
        Atom::Quot(n, d) => format!("({} / {})", grouped(n, name), grouped(d, name)),
        Atom::Rem(n, d) => format!("({} % {})", grouped(n, name), grouped(d, name)),
        Atom::Foreign(f) => format!("{:?}", f.node),
    }
}

fn grouped(s: &Poly, name: &dyn Fn(SymbolId) -> String) -> String {
    if s.terms.len() > 1 {
        format!("({})", format_poly(s, name))
    } else {
        format_poly(s, name)
    }
}

fn format_poly(p: &Poly, name: &dyn Fn(SymbolId) -> String) -> String {
    if p.terms.is_empty() {
        return "0".to_string();
    }
    let mut out = String::new();
    let mut first = true;
    for (m, c) in &p.terms {
        let sign = if *c < 0 { "-" } else { "+" };
        if !first || *c < 0 {
            if first {
                out.push('-');
            } else {
                out.push_str(&format!(" {sign} "));
            }
        }
        first = false;
        let mag = c.unsigned_abs();
        if m.is_empty() {
            out.push_str(&mag.to_string());
            continue;
        }
        if mag != 1 {
            out.push_str(&format!("{mag} * "));
        }
        let mut factors = Vec::new();
        for (a, k) in m {
            for _ in 0..*k {
                factors.push(format_atom(a, name));
            }
        }
        out.push_str(&factors.join(" * "));
    }
    out
}

// ---------------------------------------------------------------------------
// Facts
// ---------------------------------------------------------------------------

/// Known bounds on symbols, and equalities that hold. Signed symbols have no
/// implicit lower bound; nonnegative origins install an explicit bound.
#[derive(Clone, Debug, Default)]
pub(crate) struct Facts {
    /// symbol -> inclusive upper bound
    upper: BTreeMap<SymbolId, IntExpr>,
    /// symbol -> inclusive lower bound, when established by its origin
    lower: BTreeMap<SymbolId, IntExpr>,
    /// Further bounds from path conditions; every entry is valid on its own.
    extra_upper: Vec<(SymbolId, IntExpr)>,
    extra_lower: Vec<(SymbolId, IntExpr)>,
    /// Expressions known to be zero.
    zero: Vec<IntExpr>,
    /// Expressions known nonnegative that bound no unit-coefficient symbol
    /// (`H / KV >= 1`, `T*P*P*C >= 1`).
    assumed: Vec<IntExpr>,
    /// Expressions known to be nonzero (`where M != 0`).
    nonzero: Vec<IntExpr>,
    /// Proof results under exactly these facts (rule 7). Clones of unchanged
    /// facts share it; every change starts a new one.
    memo: Rc<RefCell<HashMap<Poly, Memo>>>,
}

/// What is known about one subgoal at the fixed depths searched so far. The
/// search is monotone in its depth, so a success holds at every greater depth
/// and a failure at every smaller one.
#[derive(Clone, Copy, Debug)]
struct Memo {
    /// The least depth at which the goal was proved.
    proved: Option<usize>,
    /// The greatest depth at which the goal failed.
    failed: usize,
}

impl Facts {
    pub(crate) fn new() -> Facts {
        Facts::default()
    }

    /// Every change to the facts invalidates the proofs made under them.
    fn changed(&mut self) -> &mut Self {
        self.memo = Rc::default();
        self
    }

    pub(crate) fn set_range_lower(&mut self, symbol: SymbolId, lo: IntExpr) {
        self.changed().lower.insert(symbol, lo);
    }

    pub(crate) fn set_range(&mut self, symbol: SymbolId, lo: IntExpr, hi: IntExpr) {
        let facts = self.changed();
        facts.lower.insert(symbol, lo);
        facts.upper.insert(symbol, hi);
    }

    pub(crate) fn assume_zero(&mut self, arena: &ExprArena, e: IntExpr) {
        if !is_zero(arena, e) {
            self.changed().zero.push(e);
        }
    }

    /// Assume `e != 0`.
    pub(crate) fn assume_nonzero(&mut self, arena: &ExprArena, e: IntExpr) {
        if constant(arena, e).is_none() {
            self.changed().nonzero.push(e);
        }
    }

    /// Whether `e != 0` holds: `e` is at least one or at most minus one, or
    /// it is an assumed nonzero expression up to sign.
    pub(crate) fn nonzero(&self, arena: &mut ExprArena, e: IntExpr) -> bool {
        let one = arena.int(1);
        let positive = arena.int_sub(e, one);
        let zero = arena.int(0);
        let negated = arena.int_sub(zero, e);
        let negative = arena.int_sub(negated, one);
        if nonneg(arena, self, positive) || nonneg(arena, self, negative) {
            return true;
        }
        let value = normalize_int(arena, e);
        value.valid
            && self.nonzero.iter().any(|assumed| {
                let assumed = normalize_int(arena, *assumed);
                assumed == value || assumed == value.neg()
            })
    }

    pub(crate) fn upper_of(&self, symbol: SymbolId) -> Option<IntExpr> {
        self.upper.get(&symbol).copied()
    }

    /// A path condition `symbol <= hi`.
    pub(crate) fn add_upper(&mut self, symbol: SymbolId, hi: IntExpr) {
        self.changed().extra_upper.push((symbol, hi));
    }

    /// A path condition `symbol >= lo`.
    pub(crate) fn add_lower(&mut self, symbol: SymbolId, lo: IntExpr) {
        self.changed().extra_lower.push((symbol, lo));
    }

    /// The bounds of a joined value `joined`, which equals `value` of one of
    /// the two `arms` under that arm's facts (L30, L32 (b)). A candidate `e`
    /// bounds `joined` below when `value - e >= 0` is proved in both arms, and
    /// above when `e - value >= 0` is. The candidates are both arm values,
    /// zero, and the fact bounds of every symbol of each arm value, restricted
    /// to expressions whose symbols are all `in_scope` after the join.
    pub(crate) fn join_bounds(
        &mut self,
        arena: &mut ExprArena,
        joined: SymbolId,
        arms: [(&Facts, IntExpr); 2],
        in_scope: &dyn Fn(SymbolId) -> bool,
    ) {
        let mut candidates = vec![arms[0].1, arms[1].1, arena.int(0)];
        for (facts, value) in arms {
            for symbol in symbols(arena, value) {
                candidates.extend(facts.lower.get(&symbol).copied());
                candidates.extend(facts.upper.get(&symbol).copied());
                for (bounded, bound) in facts.extra_lower.iter().chain(&facts.extra_upper) {
                    if *bounded == symbol {
                        candidates.push(*bound);
                    }
                }
            }
        }
        let mut distinct: Vec<IntExpr> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if symbols(arena, candidate)
                .into_iter()
                .all(|symbol| in_scope(symbol))
                && !distinct.iter().any(|kept| same(arena, *kept, candidate))
            {
                distinct.push(candidate);
            }
        }
        for candidate in distinct {
            if arms
                .iter()
                .all(|(facts, value)| le(arena, facts, candidate, *value))
            {
                self.add_lower(joined, candidate);
            }
            if arms
                .iter()
                .all(|(facts, value)| le(arena, facts, *value, candidate))
            {
                self.add_upper(joined, candidate);
            }
        }
    }

    /// Assume `e >= 0`: a bound on every symbol with a unit coefficient, and
    /// the whole expression when it bounds no such symbol.
    pub(crate) fn assume_nonnegative(&mut self, arena: &mut ExprArena, e: IntExpr) {
        let mut bounded = false;
        for symbol in symbols(arena, e) {
            match linear_in(arena, e, symbol) {
                Some((1, rest)) => {
                    let zero = arena.int(0);
                    let lower = arena.int_sub(zero, rest);
                    self.add_lower(symbol, lower);
                    bounded = true;
                }
                Some((-1, rest)) => {
                    self.add_upper(symbol, rest);
                    bounded = true;
                }
                _ => {}
            }
        }
        if !bounded && constant(arena, e).is_none() {
            self.changed().assumed.push(e);
        }
    }

    fn uppers_of(&self, arena: &ExprArena, symbol: SymbolId) -> Vec<Poly> {
        let mut out: Vec<Poly> = self
            .upper
            .get(&symbol)
            .map(|e| normalize_int(arena, *e))
            .into_iter()
            .collect();
        out.extend(
            self.extra_upper
                .iter()
                .filter(|(x, _)| *x == symbol)
                .map(|(_, s)| normalize_int(arena, *s)),
        );
        out
    }

    fn lowers_of(&self, arena: &ExprArena, symbol: SymbolId) -> Vec<Poly> {
        let mut out: Vec<Poly> = self
            .lower
            .get(&symbol)
            .map(|e| normalize_int(arena, *e))
            .filter(|l| !l.is_zero())
            .into_iter()
            .collect();
        out.extend(
            self.extra_lower
                .iter()
                .filter(|(x, _)| *x == symbol)
                .map(|(_, s)| normalize_int(arena, *s))
                .filter(|s| !s.is_zero()),
        );
        out
    }

    fn lower_poly(&self, arena: &ExprArena, symbol: SymbolId) -> Option<Poly> {
        self.lower.get(&symbol).map(|e| normalize_int(arena, *e))
    }

    fn upper_poly(&self, arena: &ExprArena, symbol: SymbolId) -> Option<Poly> {
        self.upper.get(&symbol).map(|e| normalize_int(arena, *e))
    }

    fn zeros(&self, arena: &ExprArena) -> Vec<Poly> {
        self.zero.iter().map(|e| normalize_int(arena, *e)).collect()
    }

    fn assumed(&self, arena: &ExprArena) -> Vec<Poly> {
        self.assumed
            .iter()
            .map(|e| normalize_int(arena, *e))
            .filter(|p| p.valid && !p.is_zero())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The prover
// ---------------------------------------------------------------------------

/// The rewriting depth of one proof. It is part of the documented rule set:
/// `nonneg(e)` is the fixed-depth search below, a function of the facts and
/// `e` alone, whatever order its rules are explored in (rule 7). Every
/// std/engine and checker-test proof needs at most depth 3; the search is
/// exponential in the depth, so 8 keeps a margin within the check-cost gate.
const DEPTH: usize = 8;

struct Engine<'a> {
    arena: &'a ExprArena,
    facts: &'a Facts,
    zeros: Vec<Poly>,
    assumed: Vec<Poly>,
}

/// The direction in which an atom may be replaced by one of its bounds
/// without raising the value of the expression that contains it (rule 6).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Monotone {
    /// Every monomial `c * a * R` with `a` has `c > 0` and `R >= 0`.
    Lower,
    /// Every monomial `c * a * R` with `a` has `c < 0` and `R >= 0`.
    Upper,
}

/// Prove `e >= 0` under the facts.
pub(crate) fn nonneg(arena: &ExprArena, facts: &Facts, e: IntExpr) -> bool {
    let engine = Engine::new(arena, facts);
    let p = normalize_int(arena, e);
    p.valid && engine.nonneg_steps(p, DEPTH)
}

/// Prove `e == 0` under the zero facts.
pub(crate) fn zero(arena: &ExprArena, facts: &Facts, e: IntExpr) -> bool {
    let engine = Engine::new(arena, facts);
    let p = engine.apply_zero_facts(normalize_int(arena, e));
    p.valid && (p.is_zero() || engine.zeros.contains(&p) || engine.zeros.contains(&p.neg()))
}

/// `a <= b`
pub(crate) fn le(arena: &mut ExprArena, facts: &Facts, a: IntExpr, b: IntExpr) -> bool {
    let d = arena.int_sub(b, a);
    nonneg(arena, facts, d)
}

/// `a < b`
pub(crate) fn lt(arena: &mut ExprArena, facts: &Facts, a: IntExpr, b: IntExpr) -> bool {
    let d = arena.int_sub(b, a);
    let one = arena.int(1);
    let d = arena.int_sub(d, one);
    nonneg(arena, facts, d)
}

impl<'a> Engine<'a> {
    fn new(arena: &'a ExprArena, facts: &'a Facts) -> Engine<'a> {
        Engine {
            arena,
            facts,
            zeros: facts.zeros(arena),
            assumed: facts.assumed(arena),
        }
    }

    /// The memoized search: each subgoal is decided once per depth range
    /// under these facts.
    fn nonneg_steps(&self, e: Poly, depth: usize) -> bool {
        if !e.valid || depth == 0 {
            return false;
        }
        if let Some(memo) = self.facts.memo.borrow().get(&e) {
            if memo.proved.is_some_and(|proved| proved <= depth) {
                return true;
            }
            if depth <= memo.failed {
                return false;
            }
        }
        let proved = self.nonneg_rules(e.clone(), depth);
        let mut memo = self.facts.memo.borrow_mut();
        let known = memo.entry(e).or_insert(Memo {
            proved: None,
            failed: 0,
        });
        if proved {
            known.proved = Some(known.proved.map_or(depth, |least| least.min(depth)));
        } else {
            known.failed = known.failed.max(depth);
        }
        proved
    }

    fn nonneg_rules(&self, mut e: Poly, depth: usize) -> bool {
        e = self.apply_zero_facts(e);
        // A positive coefficient only helps when every factor is known
        // nonnegative. Signed source values have no such default fact.
        if self.nonnegative_polynomial(&e, &mut HashSet::new()) {
            return true;
        }
        // For a positive common divisor, floor is monotone and commutes with
        // integer translation: c + floor(a/d) >= floor(b/d) follows from
        // a + c*d >= b.
        for positive in e.atoms() {
            let Atom::Quot(a, d) = &positive else {
                continue;
            };
            if !self.positive_denominator_on_defined_atom(d) {
                continue;
            }
            let positive_term = Poly::atom(positive.clone());
            if e.linear_in(&positive).is_none_or(|(c, _)| c != 1) {
                continue;
            }
            for negative in e.atoms() {
                let Atom::Quot(b, other_d) = &negative else {
                    continue;
                };
                if d != other_d || e.linear_in(&negative).is_none_or(|(c, _)| c != -1) {
                    continue;
                }
                let constant = Poly::constant(e.constant_term());
                let rest = e
                    .sub(&positive_term)
                    .add(&Poly::atom(negative.clone()))
                    .sub(&constant);
                if (rest.is_zero() || self.nonneg_steps(rest, depth - 1))
                    && self.nonneg_steps(a.add(&constant.mul(d)).sub(b), depth - 1)
                {
                    return true;
                }
            }
        }
        // Rewrite a quotient or remainder atom through the division identity.
        for atom in e.atoms() {
            match &atom {
                Atom::Quot(n, d) => {
                    if let Some(rewritten) = self.rewrite_numerator(&e, n, d, &atom) {
                        if self.nonneg_steps(rewritten, depth - 1) {
                            return true;
                        }
                    }
                }
                Atom::Rem(n, d) => {
                    if let Some(rewritten) =
                        self.rewrite_numerator(&e, n, d, &Atom::Quot(n.clone(), d.clone()))
                    {
                        if self.nonneg_steps(rewritten, depth - 1) {
                            return true;
                        }
                    }
                }
                Atom::Symbol(_) | Atom::Foreign(_) => {}
            }
        }
        // A quotient is monotone in its numerator.
        for atom in e.atoms() {
            if let Atom::Quot(n, d) = &atom {
                if !self.positive_denominator_on_defined_atom(d) {
                    continue;
                }
                let Some(direction) = self.monotone(&e, &atom) else {
                    continue;
                };
                let (lo, hi) = self.interval_over(n, &|_| true);
                let bound = match direction {
                    Monotone::Upper => self.quot_on_positive_denominator(&hi, d),
                    Monotone::Lower => self.quot_on_positive_denominator(&lo, d),
                };
                if bound != Poly::atom(atom.clone()) {
                    let substituted = e.subst(&atom, &bound);
                    if substituted != e && self.nonneg_steps(substituted, depth - 1) {
                        return true;
                    }
                }
            }
        }
        // A quotient with unit coefficient over a positive denominator:
        // `n / d + r >= 0` iff `n + r*d >= 0`, because `r` is an integer.
        for atom in e.atoms() {
            let Atom::Quot(n, d) = &atom else {
                continue;
            };
            let Some((1, rest)) = e.linear_in(&atom) else {
                continue;
            };
            let positive = self.positive_denominator_on_defined_atom(d)
                || self.nonneg_steps(d.sub(&Poly::constant(1)), depth - 1);
            if positive && self.nonneg_steps(n.add(&rest.mul(d)), depth - 1) {
                return true;
            }
        }
        // A positive monomial of atoms with nonnegative lower bounds is at
        // least the product of those bounds, whatever the powers.
        let lowered = self.lower_positive_monomials(&e);
        if lowered != e && self.nonneg_steps(lowered, depth - 1) {
            return true;
        }
        // Bound an atom by its upper bound where it only ever lowers the
        // value, or by a nonzero lower bound where it only raises it.
        let mut substitutions = Vec::new();
        for atom in e.atoms() {
            let Some(direction) = self.monotone(&e, &atom) else {
                continue;
            };
            let bounds: Vec<Poly> = match direction {
                Monotone::Upper => match &atom {
                    Atom::Rem(_, d) if self.positive_denominator_on_defined_atom(d) => {
                        vec![d.sub(&Poly::constant(1))]
                    }
                    Atom::Symbol(s) => self.facts.uppers_of(self.arena, *s),
                    Atom::Rem(..) | Atom::Quot(..) | Atom::Foreign(_) => Vec::new(),
                },
                Monotone::Lower => match &atom {
                    Atom::Symbol(s) => self.facts.lowers_of(self.arena, *s),
                    _ => Vec::new(),
                },
            };
            for b in bounds {
                let substituted = e.subst(&atom, &b);
                if substituted != e {
                    substitutions.push(substituted);
                }
            }
        }
        // Try every immediate bound before recursively combining bounds.
        if substitutions.iter().any(|s| {
            self.nonnegative_polynomial(&self.apply_zero_facts(s.clone()), &mut HashSet::new())
        }) {
            return true;
        }
        for substituted in substitutions {
            if self.nonneg_steps(substituted, depth - 1) {
                return true;
            }
        }
        // `e = (e - f) + f` for an assumed nonnegative `f`.
        for assumed in self.assumed.clone() {
            if self.nonneg_steps(e.sub(&assumed), depth - 1) {
                return true;
            }
        }
        false
    }

    /// Rule 6: the bound of `atom` that cannot raise `e`, when one exists.
    /// `atom` occurs to the first power in every monomial `c * atom * R` that
    /// contains it, every such `c` has one sign, and every cofactor `R` is
    /// structurally nonnegative: each of its atoms is nonnegative by the
    /// implicit rules or a nonnegative lower bound, or occurs to an even power.
    fn monotone(&self, e: &Poly, atom: &Atom) -> Option<Monotone> {
        let mut direction = None;
        for (monomial, coefficient) in &e.terms {
            let Some(power) = monomial.get(atom) else {
                continue;
            };
            if *power != 1 {
                return None;
            }
            let cofactor_nonnegative = monomial.iter().all(|(factor, power)| {
                factor == atom
                    || power % 2 == 0
                    || self.nonnegative_atom(factor, &mut HashSet::new())
            });
            if !cofactor_nonnegative {
                return None;
            }
            let sign = if *coefficient > 0 {
                Monotone::Lower
            } else {
                Monotone::Upper
            };
            if direction.replace(sign).is_some_and(|prior| prior != sign) {
                return None;
            }
        }
        direction
    }

    /// Replace every positive-coefficient monomial whose atoms all have a
    /// nonnegative lower bound by the product of those bounds. Each atom is
    /// its own bound when it is nonnegative and has no better one.
    fn lower_positive_monomials(&self, e: &Poly) -> Poly {
        let mut out = Poly::default();
        for (monomial, coefficient) in &e.terms {
            let mut term = Poly::constant(*coefficient);
            let mut lowered = *coefficient > 0;
            if lowered {
                for (atom, power) in monomial {
                    let bound = match atom {
                        Atom::Symbol(symbol) => self
                            .facts
                            .lowers_of(self.arena, *symbol)
                            .into_iter()
                            .find(|lower| self.nonnegative_polynomial(lower, &mut HashSet::new())),
                        _ => None,
                    };
                    let bound = match bound {
                        Some(bound) => bound,
                        None if self.nonnegative_atom(atom, &mut HashSet::new()) => {
                            Poly::atom(atom.clone())
                        }
                        None => {
                            lowered = false;
                            break;
                        }
                    };
                    for _ in 0..*power {
                        term = term.mul(&bound);
                    }
                }
            }
            if !lowered {
                term = Poly::default();
                term.insert(monomial.clone(), *coefficient);
            }
            out = out.add(&term);
        }
        out
    }

    /// Replace a symbol that is the numerator of `q = n / d` by `d * q + r`.
    fn rewrite_numerator(&self, e: &Poly, n: &Poly, d: &Poly, q: &Atom) -> Option<Poly> {
        let symbol = match single_atom_of(n) {
            Some(Atom::Symbol(s)) => Atom::Symbol(s),
            _ => return None,
        };
        let r = Poly::atom(Atom::Rem(Box::new(n.clone()), Box::new(d.clone())));
        let replacement = d.mul(&Poly::atom(q.clone())).add(&r);
        let out = e.subst(&symbol, &replacement);
        if out == *e {
            None
        } else {
            Some(out)
        }
    }

    fn apply_zero_facts(&self, e: Poly) -> Poly {
        let mut e = e;
        for z in &self.zeros {
            // A zero fact of the form `atom == 0` lets us drop that atom.
            if let Some(atom) = single_atom_of(z) {
                e = e.subst(&atom, &Poly::constant(0));
            }
        }
        e
    }

    /// Interval arithmetic over symbolic ends, bounding only the symbols
    /// `bound` accepts.
    fn interval_over(&self, e: &Poly, bound: &dyn Fn(SymbolId) -> bool) -> (Poly, Poly) {
        let mut lo = Poly::default();
        let mut hi = Poly::default();
        for (m, c) in &e.terms {
            // Endpoint multiplication is monotone only when every factor is
            // nonnegative. Preserve a signed monomial exactly; even bounding
            // a different nonnegative factor can reverse its inequality.
            if m.keys()
                .any(|a| !self.nonnegative_atom(a, &mut HashSet::new()))
            {
                let mut exact = Poly::default();
                exact.insert(m.clone(), *c);
                lo = lo.add(&exact);
                hi = hi.add(&exact);
                continue;
            }
            let mut lo_term = Poly::constant(*c);
            let mut hi_term = Poly::constant(*c);
            for (a, k) in m {
                let (l, u) = self.atom_bounds_over(a, bound);
                for _ in 0..*k {
                    if *c >= 0 {
                        lo_term = lo_term.mul(&l);
                        match &u {
                            Some(u) => hi_term = hi_term.mul(u),
                            None => hi_term = hi_term.mul(&Poly::atom(a.clone())),
                        }
                    } else {
                        hi_term = hi_term.mul(&l);
                        match &u {
                            Some(u) => lo_term = lo_term.mul(u),
                            None => lo_term = lo_term.mul(&Poly::atom(a.clone())),
                        }
                    }
                }
            }
            lo = lo.add(&lo_term);
            hi = hi.add(&hi_term);
        }
        (lo, hi)
    }

    fn atom_bounds_over(&self, a: &Atom, bound: &dyn Fn(SymbolId) -> bool) -> (Poly, Option<Poly>) {
        match a {
            Atom::Rem(_, d) => {
                let upper = self
                    .positive_denominator_on_defined_atom(d)
                    .then(|| d.sub(&Poly::constant(1)));
                (Poly::constant(0), upper)
            }
            Atom::Quot(n, d) => {
                if !self.nonnegative_atom(a, &mut HashSet::new()) {
                    let exact = Poly::atom(a.clone());
                    return (exact.clone(), Some(exact));
                }
                let (lo, hi) = self.interval_over(n, bound);
                // Accepted integer division has a positive denominator, so
                // floor division is monotone even when that denominator is a
                // symbolic shape extent. Keeping the quotient here is what
                // proves row-major bounds such as
                // `(M*N - 1) / N == M - 1`.
                let lower = self.quot_on_positive_denominator(&lo, d);
                if !self.nonnegative_polynomial(&lower, &mut HashSet::new()) {
                    let exact = Poly::atom(a.clone());
                    return (exact.clone(), Some(exact));
                }
                (lower, Some(self.quot_on_positive_denominator(&hi, d)))
            }
            Atom::Symbol(s) => {
                if bound(*s) {
                    match self.facts.lower_poly(self.arena, *s) {
                        Some(lower) if self.nonnegative_polynomial(&lower, &mut HashSet::new()) => {
                            (lower, self.facts.upper_poly(self.arena, *s))
                        }
                        None => (Poly::symbol(*s), Some(Poly::symbol(*s))),
                        Some(_) => (Poly::symbol(*s), Some(Poly::symbol(*s))),
                    }
                } else {
                    (Poly::symbol(*s), Some(Poly::symbol(*s)))
                }
            }
            Atom::Foreign(f) => {
                if matches!(
                    f.node,
                    AnyExpr::Nat(_) | AnyExpr::Bool(_) | AnyExpr::Duration(_)
                ) {
                    (Poly::constant(0), None)
                } else {
                    let exact = Poly::atom(a.clone());
                    (exact.clone(), Some(exact))
                }
            }
        }
    }

    fn nonnegative_polynomial(&self, value: &Poly, visiting: &mut HashSet<SymbolId>) -> bool {
        value.valid
            && value.terms.iter().all(|(factors, coefficient)| {
                *coefficient >= 0
                    && factors
                        .keys()
                        .all(|factor| self.nonnegative_atom(factor, visiting))
            })
    }

    /// A quotient/remainder atom is evaluated only when its denominator is
    /// nonzero. A denominator already known nonnegative is therefore positive
    /// on every execution where that atom has a value.
    fn positive_denominator_on_defined_atom(&self, denominator: &Poly) -> bool {
        self.nonnegative_polynomial(denominator, &mut HashSet::new())
    }

    /// On a defined quotient with a nonnegative denominator, the denominator
    /// is positive and Euclidean `-1 / d` equals `-1`. Keep this rewrite in
    /// the proof engine, where the atom's definedness is available; the
    /// unconditional polynomial normalizer cannot make it.
    fn quot_on_positive_denominator(&self, numerator: &Poly, denominator: &Poly) -> Poly {
        let quotient = numerator.quot(denominator);
        if !self.positive_denominator_on_defined_atom(denominator) {
            return quotient;
        }
        let minus_one = Poly::constant(-1);
        let boundary = Atom::Quot(Box::new(minus_one.clone()), Box::new(denominator.clone()));
        quotient.subst(&boundary, &minus_one)
    }

    fn nonnegative_atom(&self, atom: &Atom, visiting: &mut HashSet<SymbolId>) -> bool {
        match atom {
            Atom::Symbol(symbol) => {
                if self.arena.symbol_sort(*symbol) == crate::expr::SymbolSort::Nat {
                    return true;
                }
                if !visiting.insert(*symbol) {
                    return false;
                }
                let proved = self
                    .facts
                    .lower
                    .get(symbol)
                    .into_iter()
                    .chain(
                        self.facts
                            .extra_lower
                            .iter()
                            .filter(|(bounded, _)| bounded == symbol)
                            .map(|(_, lower)| lower),
                    )
                    .any(|lower| {
                        self.nonnegative_polynomial(&normalize_int(self.arena, *lower), visiting)
                    });
                visiting.remove(symbol);
                proved
            }
            Atom::Rem(..) => true,
            Atom::Foreign(f) => {
                matches!(
                    f.node,
                    AnyExpr::Nat(_) | AnyExpr::Bool(_) | AnyExpr::Duration(_)
                )
            }
            Atom::Quot(numerator, denominator) => {
                self.nonnegative_polynomial(numerator, visiting)
                    && self.positive_denominator_on_defined_atom(denominator)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine's interval arithmetic over every symbol, re-interned.
    fn interval(arena: &mut ExprArena, facts: &Facts, e: IntExpr) -> (IntExpr, IntExpr) {
        let p = normalize_int(arena, e);
        let (lo, hi) = Engine::new(arena, facts).interval_over(&p, &|_| true);
        (intern(arena, &lo), intern(arena, &hi))
    }

    /// A positive dimension symbol, as the checker installs it.
    fn dimension(arena: &mut ExprArena, facts: &mut Facts) -> IntExpr {
        let (_, symbol, value) = arena.nat_loop_binder();
        let one = arena.int(1);
        facts.set_range_lower(symbol, one);
        arena.int_from_nat(value)
    }

    #[test]
    fn positive_monomials_are_bounded_by_their_atoms_lower_bounds() {
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let a = dimension(&mut arena, &mut facts);
        let one = arena.int(1);
        let square = arena.int_mul(a, a);
        let square = arena.int_sub(square, one);
        assert!(nonneg(&arena, &facts, square), "A*A >= 1");
        let (t, p, c) = (
            dimension(&mut arena, &mut facts),
            dimension(&mut arena, &mut facts),
            dimension(&mut arena, &mut facts),
        );
        let patch = arena.int_mul(t, p);
        let patch = arena.int_mul(patch, p);
        let patch = arena.int_mul(patch, c);
        let patch = arena.int_sub(patch, one);
        assert!(nonneg(&arena, &facts, patch), "T*P*P*C >= 1");
        let (_, _, signed) = arena.loop_binder();
        let signed_square = arena.int_mul(signed, a);
        let signed_square = arena.int_sub(signed_square, one);
        assert!(
            !nonneg(&arena, &facts, signed_square),
            "a signed factor has no bound"
        );
    }

    #[test]
    fn quotients_are_bounded_through_their_numerator() {
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let h = dimension(&mut arena, &mut facts);
        let kv = dimension(&mut arena, &mut facts);
        let one = arena.int(1);
        let ratio = arena.int_div(h, kv);
        let at_least_one = arena.int_sub(ratio, one);
        assert!(
            !nonneg(&arena, &facts, at_least_one),
            "H / KV >= 1 needs H >= KV"
        );
        let difference = arena.int_sub(h, kv);
        facts.assume_nonnegative(&mut arena, difference);
        assert!(
            nonneg(&arena, &facts, at_least_one),
            "H >= KV proves H / KV >= 1"
        );
    }

    #[test]
    fn assumed_expressions_discharge_larger_ones() {
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let h = dimension(&mut arena, &mut facts);
        let kv = dimension(&mut arena, &mut facts);
        let one = arena.int(1);
        let ratio = arena.int_div(h, kv);
        let at_least_one = arena.int_sub(ratio, one);
        facts.assume_nonnegative(&mut arena, at_least_one);
        let two = arena.int(2);
        let doubled = arena.int_mul(ratio, two);
        let doubled = arena.int_sub(doubled, two);
        assert!(nonneg(&arena, &facts, at_least_one));
        assert!(
            nonneg(&arena, &facts, doubled),
            "2*(H/KV) - 2 >= 0 from H/KV >= 1"
        );
    }

    #[test]
    fn exact_division_by_constants_and_monomials() {
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let (a, b) = (
            dimension(&mut arena, &mut facts),
            dimension(&mut arena, &mut facts),
        );
        let two = arena.int(2);
        let three = arena.int(3);
        let ab = arena.int_mul(a, b);
        let twice = arena.int_mul(ab, two);
        let sum = arena.int_add(twice, ab);
        assert!(divide_exact(&mut arena, sum, b).is_some_and(|q| {
            let expected = arena.int_mul(a, three);
            same(&arena, q, expected)
        }));
        let odd = arena.int_add(ab, three);
        assert!(divide_exact(&mut arena, odd, b).is_none());
        let twice_a = arena.int_mul(a, two);
        assert!(divide_exact(&mut arena, twice_a, two).is_some_and(|q| same(&arena, q, a)));
        let zero = arena.int(0);
        assert!(divide_exact(&mut arena, twice_a, zero).is_none());
    }

    #[test]
    fn quotient_differences_follow_their_numerators() {
        // `token / 32 < (V + 31) / 32` for `0 <= token <= V - 1`.
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let v = dimension(&mut arena, &mut facts);
        let (_, symbol, token) = arena.loop_binder();
        let zero = arena.int(0);
        let one = arena.int(1);
        let last = arena.int_sub(v, one);
        facts.set_range(symbol, zero, last);
        let (thirty_one, thirty_two) = (arena.int(31), arena.int(32));
        let rows = arena.int_add(v, thirty_one);
        let rows = arena.int_div(rows, thirty_two);
        let row = arena.int_div(token, thirty_two);
        assert!(lt(&mut arena, &facts, row, rows));
    }

    #[test]
    fn t7_lower_substitution_ignores_signed_cofactor() {
        // i in [1, 9], k <= -1: i*k - k = (i - 1)*k is negative for i >= 2.
        let mut arena = ExprArena::new();
        let (_, i_symbol, i) = arena.loop_binder();
        let (_, k_symbol, k) = arena.loop_binder();
        let mut facts = Facts::new();
        let one = arena.int(1);
        let nine = arena.int(9);
        facts.set_range(i_symbol, one, nine);
        let minus_one = arena.int(-1);
        facts.add_upper(k_symbol, minus_one);
        let product = arena.int_mul(i, k);
        let e = arena.int_sub(product, k);
        assert!(
            !nonneg(&arena, &facts, e),
            "prover claims (i - 1) * k >= 0 for i in [1, 9], k <= -1"
        );
        // With a nonnegative cofactor the lower bound of `i` still applies.
        let (_, n_symbol, n) = arena.loop_binder();
        let zero = arena.int(0);
        facts.set_range_lower(n_symbol, zero);
        let product = arena.int_mul(i, n);
        let e = arena.int_sub(product, n);
        assert!(
            nonneg(&arena, &facts, e),
            "(i - 1) * n >= 0 for i >= 1, n >= 0"
        );
    }

    #[test]
    fn proofs_do_not_depend_on_earlier_goals() {
        // Rule 7: the memo shared by earlier goals never changes a verdict.
        let goals = |arena: &mut ExprArena, facts: &mut Facts| {
            let h = dimension(arena, facts);
            let kv = dimension(arena, facts);
            let difference = arena.int_sub(h, kv);
            facts.assume_nonnegative(arena, difference);
            let one = arena.int(1);
            let ratio = arena.int_div(h, kv);
            let at_least_one = arena.int_sub(ratio, one);
            let (_, g_symbol, g) = arena.loop_binder();
            let zero = arena.int(0);
            let last = arena.int_sub(kv, one);
            facts.set_range(g_symbol, zero, last);
            let scaled = arena.int_mul(g, ratio);
            let negative = arena.int_sub(zero, scaled);
            vec![at_least_one, negative, scaled, difference]
        };
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let forward = goals(&mut arena, &mut facts);
        let first: Vec<bool> = forward.iter().map(|g| nonneg(&arena, &facts, *g)).collect();
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let backward = goals(&mut arena, &mut facts);
        let mut second: Vec<bool> = backward
            .iter()
            .rev()
            .map(|g| nonneg(&arena, &facts, *g))
            .collect();
        second.reverse();
        assert_eq!(first, second);
        assert_eq!(first, vec![true, false, true, true]);
    }

    #[test]
    fn nonzero_hypotheses_and_strict_signs() {
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let (_, _, m) = arena.loop_binder();
        assert!(!facts.nonzero(&mut arena, m));
        facts.assume_nonzero(&arena, m);
        assert!(facts.nonzero(&mut arena, m));
        let zero = arena.int(0);
        let negated = arena.int_sub(zero, m);
        assert!(facts.nonzero(&mut arena, negated), "nonzero up to sign");
        let n = dimension(&mut arena, &mut facts);
        assert!(facts.nonzero(&mut arena, n), "a dimension is at least one");
        let (_, _, other) = arena.loop_binder();
        assert!(!facts.nonzero(&mut arena, other));
    }

    #[test]
    fn join_bounds_keep_exactly_the_bounds_both_arms_prove() {
        // `let mut j = 0; if c: j = 9` bounds `j` to [0, 9], so `j <= 7`
        // (a read of `tensor[8]`) is not proved.
        let mut arena = ExprArena::new();
        let facts = Facts::new();
        let (zero, nine, seven) = (arena.int(0), arena.int(9), arena.int(7));
        let (_, joined_symbol, joined) = arena.loop_binder();
        let mut after = facts.clone();
        after.join_bounds(
            &mut arena,
            joined_symbol,
            [(&facts, nine), (&facts, zero)],
            &|_| true,
        );
        assert!(le(&mut arena, &after, zero, joined));
        assert!(le(&mut arena, &after, joined, nine));
        assert!(!le(&mut arena, &after, joined, seven));
        // `let mut j = i; if c: j = t` with `i, t` in [0, N - 1] keeps
        // `0 <= j <= N - 1` through the fact bounds of each arm value.
        let mut facts = Facts::new();
        let n = dimension(&mut arena, &mut facts);
        let one = arena.int(1);
        let last = arena.int_sub(n, one);
        let (_, i_symbol, i) = arena.loop_binder();
        let (_, t_symbol, t) = arena.loop_binder();
        facts.set_range(i_symbol, zero, last);
        let mut then_facts = facts.clone();
        then_facts.set_range(t_symbol, zero, last);
        let (_, joined_symbol, joined) = arena.loop_binder();
        let mut after = facts.clone();
        after.join_bounds(
            &mut arena,
            joined_symbol,
            [(&then_facts, t), (&facts, i)],
            &|symbol| symbol != t_symbol,
        );
        assert!(le(&mut arena, &after, zero, joined));
        assert!(le(&mut arena, &after, joined, last));
        assert!(
            !le(&mut arena, &after, joined, i),
            "`i` is only the else value"
        );
    }

    #[test]
    fn interval_multiplication_preserves_signed_factors() {
        let mut arena = ExprArena::new();
        let (_, symbol, value) = arena.loop_binder();
        let lower = arena.int(-5);
        let upper = arena.int(3);
        let square = arena.int_mul(value, value);
        let mut facts = Facts::new();
        facts.set_range(symbol, lower, upper);
        let (lo, hi) = interval(&mut arena, &facts, square);
        assert!(same(&arena, lo, square));
        assert!(same(&arena, hi, square));

        let (_, natural_symbol, natural) = arena.nat_loop_binder();
        let natural = arena.int_from_nat(natural);
        let product = arena.int_mul(value, natural);
        let zero = arena.int(0);
        let three = arena.int(3);
        facts.set_range(natural_symbol, zero, three);
        let (lo, hi) = interval(&mut arena, &facts, product);
        assert!(same(&arena, lo, product));
        assert!(same(&arena, hi, product));
    }

    #[test]
    fn interval_multiplication_still_bounds_nonnegative_factors() {
        let mut arena = ExprArena::new();
        let (_, symbol, value) = arena.nat_loop_binder();
        let value = arena.int_from_nat(value);
        let lower = arena.int(0);
        let upper = arena.int(3);
        let square = arena.int_mul(value, value);
        let mut facts = Facts::new();
        facts.set_range(symbol, lower, upper);
        let (lo, hi) = interval(&mut arena, &facts, square);
        assert!(is_zero(&arena, lo));
        let nine = arena.int(9);
        assert!(same(&arena, hi, nine));
    }

    #[test]
    fn signed_opaque_scalar_round_trips_without_a_zero_bound() {
        let mut arena = ExprArena::new();
        let (_, _, input) = arena.loop_binder();
        let one = arena.int(1);
        let wrapped = arena.scalar_integer(
            crate::reference_math::ScalarOp::Binary(crate::syntax::ast::BinaryOp::Add),
            &[
                (crate::types::DType::I32, input),
                (crate::types::DType::I32, one),
            ],
        );
        let normalized = normalize_int(&arena, wrapped);
        let restored = intern(&mut arena, &normalized);
        assert!(same(&arena, restored, wrapped));
        assert!(!nonneg(&arena, &Facts::new(), wrapped));
        let (lo, hi) = interval(&mut arena, &Facts::new(), wrapped);
        assert!(same(&arena, lo, wrapped));
        assert!(same(&arena, hi, wrapped));
    }
}
