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
//! Every symbol is a nonnegative integer: dimensions and bounded runtime
//! integers (index parameters, loop binders, view lengths) are never
//! negative. The prover establishes `e >= 0` and `e == 0` under facts by
//! rewriting quotients and remainders through `x = c * (x / c) + x % c` and
//! bounding atoms that carry an upper bound. It is sound and incomplete: a
//! failed proof is a residual the caller reports.

use crate::expr::{AnyExpr, BinaryOp, ExprArena, IntExpr, NodeView, SymbolId, UnaryOp};
use std::collections::{BTreeMap, HashSet};

// ---------------------------------------------------------------------------
// Transient normal form
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Atom {
    Symbol(SymbolId),
    /// floor(num / den), den > 0 in normalized form
    Quot(Box<Poly>, Box<Poly>),
    /// num mod den, den > 0
    Rem(Box<Poly>, Box<Poly>),
    /// A node outside the polynomial fragment, as a nonnegative unknown. An
    /// integer-sorted foreign node is the difference of its positive and
    /// negative parts (`negative == true` names the subtracted part).
    Foreign(Foreign),
}

/// A foreign node keyed by its interned identity. Equal nodes are one key,
/// so two normalizations of the same node agree.
#[derive(Clone, Copy, Debug)]
struct Foreign {
    key: u64,
    negative: bool,
    node: AnyExpr,
}

impl PartialEq for Foreign {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.negative == other.negative
    }
}
impl Eq for Foreign {}
impl PartialOrd for Foreign {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Foreign {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.key, self.negative)
            .cmp(&(other.key, other.negative))
            // A hash is an ordering accelerator, never identity. Arena-owned
            // expression handles have a unique debug form (sort, owner,
            // ordinal), which closes the otherwise-unsound collision case.
            .then_with(|| format!("{:?}", self.node).cmp(&format!("{:?}", other.node)))
    }
}
impl std::hash::Hash for Foreign {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.node, self.negative).hash(state);
    }
}

fn foreign(node: AnyExpr, negative: bool) -> Atom {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    node.hash(&mut hasher);
    Atom::Foreign(Foreign {
        key: hasher.finish(),
        negative,
        node,
    })
}

type Monomial = BTreeMap<Atom, u32>;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Poly {
    /// monomial -> coefficient; the empty monomial is the constant term.
    terms: BTreeMap<Monomial, i64>,
    /// False means normalization exceeded the exact coefficient domain. An
    /// invalid polynomial can never discharge a proof obligation.
    valid: bool,
}

impl Default for Poly {
    fn default() -> Self {
        Self {
            terms: BTreeMap::new(),
            valid: true,
        }
    }
}

impl Poly {
    fn constant(c: i64) -> Poly {
        let mut terms = BTreeMap::new();
        if c != 0 {
            terms.insert(Monomial::new(), c);
        }
        Poly { terms, valid: true }
    }

    fn atom(a: Atom) -> Poly {
        let mut m = Monomial::new();
        m.insert(a, 1);
        let mut terms = BTreeMap::new();
        terms.insert(m, 1);
        Poly { terms, valid: true }
    }

    fn symbol(s: SymbolId) -> Poly {
        Poly::atom(Atom::Symbol(s))
    }

    fn is_zero(&self) -> bool {
        self.valid && self.terms.is_empty()
    }

    fn as_constant(&self) -> Option<i64> {
        if !self.valid {
            return None;
        }
        if self.terms.is_empty() {
            return Some(0);
        }
        if self.terms.len() == 1 {
            if let Some(c) = self.terms.get(&Monomial::new()) {
                return Some(*c);
            }
        }
        None
    }

    fn constant_term(&self) -> i64 {
        self.terms.get(&Monomial::new()).copied().unwrap_or(0)
    }

    fn insert(&mut self, m: Monomial, c: i64) {
        if !self.valid {
            return;
        }
        if c == 0 {
            return;
        }
        let entry = self.terms.entry(m.clone()).or_insert(0);
        let Some(sum) = entry.checked_add(c) else {
            self.valid = false;
            self.terms.clear();
            return;
        };
        *entry = sum;
        if *entry == 0 {
            self.terms.remove(&m);
        }
    }

    fn add(&self, other: &Poly) -> Poly {
        if !self.valid || !other.valid {
            return Poly {
                terms: BTreeMap::new(),
                valid: false,
            };
        }
        let mut out = self.clone();
        for (m, c) in &other.terms {
            out.insert(m.clone(), *c);
        }
        out
    }

    fn neg(&self) -> Poly {
        if !self.valid {
            return Poly {
                terms: BTreeMap::new(),
                valid: false,
            };
        }
        let mut out = Poly::default();
        for (monomial, coefficient) in &self.terms {
            let Some(coefficient) = coefficient.checked_neg() else {
                return Poly {
                    terms: BTreeMap::new(),
                    valid: false,
                };
            };
            out.terms.insert(monomial.clone(), coefficient);
        }
        out
    }

    fn sub(&self, other: &Poly) -> Poly {
        self.add(&other.neg())
    }

    fn mul(&self, other: &Poly) -> Poly {
        if !self.valid || !other.valid {
            return Poly {
                terms: BTreeMap::new(),
                valid: false,
            };
        }
        let mut out = Poly::default();
        for (m1, c1) in &self.terms {
            for (m2, c2) in &other.terms {
                let mut m = m1.clone();
                for (a, k) in m2 {
                    let entry = m.entry(a.clone()).or_insert(0);
                    let Some(sum) = entry.checked_add(*k) else {
                        return Poly {
                            terms: BTreeMap::new(),
                            valid: false,
                        };
                    };
                    *entry = sum;
                }
                let Some(coefficient) = c1.checked_mul(*c2) else {
                    return Poly {
                        terms: BTreeMap::new(),
                        valid: false,
                    };
                };
                out.insert(m, coefficient);
            }
        }
        out
    }

    fn scale(&self, c: i64) -> Poly {
        self.mul(&Poly::constant(c))
    }

    /// If `self` is `c * atom + rest` with `rest` free of `atom`, return
    /// `(c, rest)`.
    fn linear_in(&self, atom: &Atom) -> Option<(i64, Poly)> {
        if !self.valid {
            return None;
        }
        let mut c = 0i64;
        let mut rest = Poly::default();
        for (m, k) in &self.terms {
            match m.get(atom) {
                None => {
                    rest.terms.insert(m.clone(), *k);
                }
                Some(1) if m.len() == 1 => c = c.checked_add(*k)?,
                Some(_) => return None,
            }
        }
        if c == 0 {
            None
        } else {
            Some((c, rest))
        }
    }

    /// `self / c` when every coefficient is divisible by `c`.
    fn div_exact(&self, c: i64) -> Option<Poly> {
        if !self.valid || c == 0 {
            return None;
        }
        let mut out = Poly::default();
        for (m, k) in &self.terms {
            if k % c != 0 {
                return None;
            }
            out.terms.insert(m.clone(), k / c);
        }
        Some(out)
    }

    /// floor(self / den). Requires den to be provably positive; the caller
    /// checks.
    fn quot(&self, den: &Poly) -> Poly {
        if !self.valid || !den.valid {
            return Poly {
                terms: BTreeMap::new(),
                valid: false,
            };
        }
        if let (Some(a), Some(b)) = (self.as_constant(), den.as_constant()) {
            if b != 0 {
                return Poly::constant(a.div_euclid(b));
            }
        }
        if den.as_constant() == Some(1) {
            return self.clone();
        }
        if *self == *den {
            return Poly::constant(1);
        }
        if let Some(q) = self.divide_exactly(den) {
            return q;
        }
        // `H / (H / KV)` is `KV`: the denominator is itself a quotient whose
        // numerator is this one, and the division is exact.
        if let Some(Atom::Quot(n, d)) = single_atom_of(den) {
            if *n == *self && n.divide_exactly(&d).is_some() {
                return *d;
            }
        }
        // Pull out the part of the numerator that is an exact multiple of a
        // constant denominator.
        if let Some(c) = den.as_constant() {
            if c > 0 {
                let mut exact = Poly::default();
                let mut rest = Poly::default();
                for (m, k) in &self.terms {
                    if k % c == 0 {
                        exact.insert(m.clone(), k / c);
                    } else {
                        rest.insert(m.clone(), *k);
                    }
                }
                if !exact.is_zero() {
                    if rest.is_zero() {
                        return exact;
                    }
                    // Integer division is Euclidean and every accepted shape
                    // divisor is positive.  Therefore `-1 / d == -1` for
                    // every such `d`.  This is the boundary term in the
                    // canonical bounds `(m*d - 1) / d == m - 1`.
                    if rest.as_constant() == Some(-1) {
                        return exact.sub(&Poly::constant(1));
                    }
                    return exact.add(&Poly::atom(Atom::Quot(
                        Box::new(rest),
                        Box::new(den.clone()),
                    )));
                }
            }
        }
        // A single-atom denominator: pull out the monomials it divides.
        if let Some(atom) = single_atom_of(den) {
            let mut exact = Poly::default();
            let mut rest = Poly::default();
            for (m, k) in &self.terms {
                match m.get(&atom) {
                    Some(deg) if *deg >= 1 => {
                        let mut reduced = m.clone();
                        if *deg == 1 {
                            reduced.remove(&atom);
                        } else {
                            reduced.insert(atom.clone(), deg - 1);
                        }
                        exact.insert(reduced, *k);
                    }
                    _ => rest.insert(m.clone(), *k),
                }
            }
            if !exact.is_zero() {
                if rest.is_zero() {
                    return exact;
                }
                if rest.as_constant() == Some(-1) {
                    return exact.sub(&Poly::constant(1));
                }
                return exact.add(&Poly::atom(Atom::Quot(
                    Box::new(rest),
                    Box::new(den.clone()),
                )));
            }
        }
        Poly::atom(Atom::Quot(Box::new(self.clone()), Box::new(den.clone())))
    }

    /// `self / den` when the division is exact as polynomials.
    fn divide_exactly(&self, den: &Poly) -> Option<Poly> {
        if !self.valid || !den.valid {
            return None;
        }
        let (dm, dc) = den.terms.iter().next()?;
        if den.terms.len() == 1 {
            let mut out = Poly::default();
            for (m, c) in &self.terms {
                if c % dc != 0 {
                    return None;
                }
                let mut reduced = m.clone();
                for (a, k) in dm {
                    match reduced.get(a) {
                        Some(have) if have > k => {
                            reduced.insert(a.clone(), have - k);
                        }
                        Some(have) if have == k => {
                            reduced.remove(a);
                        }
                        _ => return None,
                    }
                }
                out.insert(reduced, c / dc);
            }
            return Some(out);
        }
        // Multi-term denominator: a single-monomial quotient covers the cases
        // that arise (`(H*R + H*S) / (R + S)` is `H`).
        for (m, c) in &self.terms {
            if c % dc != 0 {
                continue;
            }
            let mut q_mono = m.clone();
            let mut ok = true;
            for (a, k) in dm {
                match q_mono.get(a) {
                    Some(have) if have > k => {
                        q_mono.insert(a.clone(), have - k);
                    }
                    Some(have) if have == k => {
                        q_mono.remove(a);
                    }
                    _ => ok = false,
                }
            }
            if !ok {
                continue;
            }
            let mut q = Poly::default();
            q.insert(q_mono, c / dc);
            if q.mul(den) == *self {
                return Some(q);
            }
        }
        None
    }

    /// self mod den, in [0, den).
    fn rem(&self, den: &Poly) -> Poly {
        if !self.valid || !den.valid {
            return Poly {
                terms: BTreeMap::new(),
                valid: false,
            };
        }
        if let (Some(a), Some(b)) = (self.as_constant(), den.as_constant()) {
            if b != 0 {
                return Poly::constant(a.rem_euclid(b));
            }
        }
        if den.as_constant() == Some(1) {
            return Poly::constant(0);
        }
        if let Some(c) = den.as_constant() {
            if c > 0 {
                let mut rest = Poly::default();
                for (m, k) in &self.terms {
                    if k % c != 0 {
                        rest.insert(m.clone(), *k);
                    }
                }
                if rest.is_zero() {
                    return Poly::constant(0);
                }
                return Poly::atom(Atom::Rem(Box::new(rest), Box::new(den.clone())));
            }
        }
        Poly::atom(Atom::Rem(Box::new(self.clone()), Box::new(den.clone())))
    }

    fn atoms(&self) -> Vec<Atom> {
        let mut out = Vec::new();
        for m in self.terms.keys() {
            for a in m.keys() {
                if !out.contains(a) {
                    out.push(a.clone());
                }
            }
        }
        out
    }

    /// Substitute an atom by an expression.
    fn subst(&self, atom: &Atom, value: &Poly) -> Poly {
        if !self.valid || !value.valid {
            return Poly {
                terms: BTreeMap::new(),
                valid: false,
            };
        }
        let mut out = Poly::default();
        for (m, c) in &self.terms {
            let mut term = Poly::constant(*c);
            for (a, k) in m {
                let factor = if a == atom {
                    value.clone()
                } else {
                    Poly::atom(a.clone())
                };
                for _ in 0..*k {
                    term = term.mul(&factor);
                }
            }
            out = out.add(&term);
        }
        out
    }

    /// Every symbol mentioned, including inside quotients and remainders.
    fn symbols(&self, out: &mut Vec<SymbolId>) {
        for a in self.atoms() {
            a.symbols(out);
        }
    }

    /// Whether `target` is mentioned anywhere, including inside quotient and
    /// remainder atoms (`i / 2` mentions `i` without being linear in it).
    fn mentions(&self, target: SymbolId) -> bool {
        self.atoms().iter().any(|atom| atom.mentions(target))
    }

    /// The coefficient of `symbol` when `self` is `c * symbol + rest` with
    /// neither `c` nor `rest` mentioning `symbol` (zero when absent); `None`
    /// when nonlinear in `symbol` or mentioning it inside a quotient or
    /// remainder. `c` may be symbolic.
    fn linear_coefficient(&self, symbol: SymbolId) -> Option<Poly> {
        if !self.valid {
            return None;
        }
        let atom = Atom::Symbol(symbol);
        let mut coefficient = Poly::constant(0);
        let mut rest = Poly::constant(0);
        for (monomial, k) in &self.terms {
            let mut term = Poly::constant(*k);
            let mut power = 0u32;
            for (factor, multiplicity) in monomial {
                if *factor == atom {
                    power = *multiplicity;
                    continue;
                }
                for _ in 0..*multiplicity {
                    term = term.mul(&Poly::atom(factor.clone()));
                }
            }
            match power {
                0 => rest = rest.add(&term),
                1 => coefficient = coefficient.add(&term),
                _ => return None,
            }
        }
        if coefficient.mentions(symbol) || rest.mentions(symbol) {
            None
        } else {
            Some(coefficient)
        }
    }
}

impl Atom {
    fn symbols(&self, out: &mut Vec<SymbolId>) {
        match self {
            Atom::Symbol(s) => {
                if !out.contains(s) {
                    out.push(*s);
                }
            }
            Atom::Quot(n, d) | Atom::Rem(n, d) => {
                n.symbols(out);
                d.symbols(out);
            }
            Atom::Foreign(_) => {}
        }
    }

    fn mentions(&self, target: SymbolId) -> bool {
        match self {
            Atom::Symbol(s) => *s == target,
            Atom::Quot(n, d) | Atom::Rem(n, d) => n.mentions(target) || d.mentions(target),
            Atom::Foreign(_) => false,
        }
    }
}

/// The atom of a polynomial that is exactly one atom with coefficient 1.
fn single_atom_of(s: &Poly) -> Option<Atom> {
    if s.terms.len() != 1 {
        return None;
    }
    let (m, k) = s.terms.iter().next()?;
    if *k != 1 || m.len() != 1 {
        return None;
    }
    let (atom, power) = m.iter().next()?;
    (*power == 1).then(|| atom.clone())
}

// ---------------------------------------------------------------------------
// Bridges between arena nodes and the normal form
// ---------------------------------------------------------------------------

/// Normalizes an arena node. Total over every node the arena can hold: sorts
/// other than `Int` are read through their integer meaning (`Nat` nodes are
/// nonnegative integers; `Bool` nodes are `1`/`0`), and operations the
/// polynomial cannot express (min, max, select, alignment, ceiling division,
/// folds, membership) become opaque atoms through a fresh quotient-free
/// symbol-less encoding: they are treated as the node's own symbol set, which
/// keeps every proof sound (an opaque atom is only ever bounded below by 0).
fn normalize(arena: &ExprArena, node: AnyExpr) -> Poly {
    match arena.view(node) {
        NodeView::NatConst(c) => match i64::try_from(c) {
            Ok(c) => Poly::constant(c),
            Err(_) => opaque(node),
        },
        NodeView::IntConst(c) => Poly::constant(c),
        NodeView::BoolConst(b) => Poly::constant(i64::from(b)),
        NodeView::ScalarConst { .. } => opaque(node),
        NodeView::Symbol(s) => Poly::symbol(s),
        NodeView::Unary { op, operand } => match op {
            UnaryOp::NatFromInt | UnaryOp::IntFromNat => normalize(arena, operand),
            UnaryOp::Not => Poly::constant(1).sub(&normalize(arena, operand)),
        },
        NodeView::Binary { op, lhs, rhs } => {
            let l = normalize(arena, lhs);
            let r = normalize(arena, rhs);
            match op {
                BinaryOp::Add => l.add(&r),
                BinaryOp::Sub => l.sub(&r),
                BinaryOp::Mul => l.mul(&r),
                BinaryOp::Div => l.quot(&r),
                BinaryOp::Rem => l.rem(&r),
                // Ceiling division `(a + b - 1) / b`.
                BinaryOp::CeilDiv => l.add(&r).sub(&Poly::constant(1)).quot(&r),
                // `align_up(a, u) = ((a + u - 1) / u) * u`.
                BinaryOp::AlignUp => l.add(&r).sub(&Poly::constant(1)).quot(&r).mul(&r),
                BinaryOp::Min
                | BinaryOp::Max
                | BinaryOp::And
                | BinaryOp::Or
                | BinaryOp::Implies
                | BinaryOp::Iff => opaque(node),
            }
        }
        NodeView::Nary { op, operands } => match op {
            crate::expr::NaryOp::Product => operands
                .iter()
                .fold(Poly::constant(1), |acc, o| acc.mul(&normalize(arena, *o))),
            _ => opaque(node),
        },
        NodeView::Select { .. }
        | NodeView::Cmp { .. }
        | NodeView::In { .. }
        | NodeView::Fold { .. }
        | NodeView::Duration(_)
        | NodeView::DurationScale { .. } => opaque(node),
    }
}

/// An operation outside the polynomial fragment: a nonnegative unknown for
/// `Nat`/`Bool`/`Cost` sorts, the difference of two nonnegative unknowns for
/// `Int`/`Scalar` sorts. Sound (nothing is provable about it beyond its
/// sort) and never produced by the checker's own constructions, which only
/// build constants, symbols and `+ - * / %`.
fn opaque(node: AnyExpr) -> Poly {
    match node {
        AnyExpr::Int(_) | AnyExpr::Scalar(_) => {
            Poly::atom(foreign(node, false)).sub(&Poly::atom(foreign(node, true)))
        }
        AnyExpr::Nat(_) | AnyExpr::Bool(_) | AnyExpr::Duration(_) => {
            Poly::atom(foreign(node, false))
        }
    }
}

fn normalize_int(arena: &ExprArena, e: IntExpr) -> Poly {
    normalize(arena, AnyExpr::Int(e))
}

/// Interns a normal form back into the arena.
fn intern(arena: &mut ExprArena, p: &Poly) -> IntExpr {
    assert!(p.valid, "invalid proof normal form cannot be materialized");
    let mut acc: Option<IntExpr> = None;
    for (monomial, coefficient) in &p.terms {
        let mut term = arena.int(*coefficient);
        for (atom, power) in monomial {
            let factor = intern_atom(arena, atom);
            for _ in 0..*power {
                term = arena.int_mul(term, factor);
            }
        }
        acc = Some(match acc {
            None => term,
            Some(sum) => arena.int_add(sum, term),
        });
    }
    match acc {
        Some(e) => e,
        None => arena.int(0),
    }
}

fn intern_atom(arena: &mut ExprArena, atom: &Atom) -> IntExpr {
    match atom {
        Atom::Symbol(s) => arena.int_symbol(*s),
        Atom::Foreign(f) => {
            // The positive part re-interns the node itself; the negative part
            // is the node's subtraction from that (`node - node`), which is
            // exact because `pos - neg == node` by construction.
            let node = match f.node {
                AnyExpr::Int(e) => e,
                AnyExpr::Nat(n) => arena.int_from_nat(n),
                AnyExpr::Bool(b) => {
                    let one = arena.int(1);
                    let zero = arena.int(0);
                    arena.int_select(b, one, zero)
                }
                AnyExpr::Duration(_) | AnyExpr::Scalar(_) => arena.int(0),
            };
            if f.negative {
                arena.int(0)
            } else {
                node
            }
        }
        Atom::Quot(n, d) => {
            let n = intern(arena, n);
            let d = intern(arena, d);
            arena.int_div(n, d)
        }
        Atom::Rem(n, d) => {
            let n = intern(arena, n);
            let d = intern(arena, d);
            arena.int_rem(n, d)
        }
    }
}

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

/// floor(a / b); the caller has proved `b >= 1`.
pub(crate) fn quot(arena: &mut ExprArena, a: IntExpr, b: IntExpr) -> IntExpr {
    let p = normalize_int(arena, a).quot(&normalize_int(arena, b));
    if p.valid {
        intern(arena, &p)
    } else {
        arena.int_div(a, b)
    }
}

pub(crate) fn rem(arena: &mut ExprArena, a: IntExpr, b: IntExpr) -> IntExpr {
    let p = normalize_int(arena, a).rem(&normalize_int(arena, b));
    if p.valid {
        intern(arena, &p)
    } else {
        arena.int_rem(a, b)
    }
}

/// `e / c` when every coefficient is divisible by `c`.
pub(crate) fn div_exact(arena: &mut ExprArena, e: IntExpr, c: i64) -> Option<IntExpr> {
    let p = normalize_int(arena, e).div_exact(c)?;
    Some(intern(arena, &p))
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

/// Known bounds on symbols, and equalities that hold. Every symbol is at
/// least zero; `lower` records a tighter bound.
#[derive(Clone, Debug, Default)]
pub(crate) struct Facts {
    /// symbol -> inclusive upper bound
    upper: BTreeMap<SymbolId, IntExpr>,
    /// symbol -> inclusive lower bound (default 0)
    lower: BTreeMap<SymbolId, IntExpr>,
    /// Further bounds from path conditions; every entry is valid on its own.
    extra_upper: Vec<(SymbolId, IntExpr)>,
    extra_lower: Vec<(SymbolId, IntExpr)>,
    /// Expressions known to be zero.
    zero: Vec<IntExpr>,
}

impl Facts {
    pub(crate) fn new() -> Facts {
        Facts::default()
    }

    pub(crate) fn set_range_lower(&mut self, symbol: SymbolId, lo: IntExpr) {
        self.lower.insert(symbol, lo);
    }

    pub(crate) fn set_range(&mut self, symbol: SymbolId, lo: IntExpr, hi: IntExpr) {
        self.lower.insert(symbol, lo);
        self.upper.insert(symbol, hi);
    }

    pub(crate) fn assume_zero(&mut self, arena: &ExprArena, e: IntExpr) {
        if !is_zero(arena, e) {
            self.zero.push(e);
        }
    }

    pub(crate) fn lower_of(&self, arena: &mut ExprArena, symbol: SymbolId) -> IntExpr {
        match self.lower.get(&symbol) {
            Some(lo) => *lo,
            None => arena.int(0),
        }
    }

    pub(crate) fn upper_of(&self, symbol: SymbolId) -> Option<IntExpr> {
        self.upper.get(&symbol).copied()
    }

    /// A path condition `symbol <= hi`.
    pub(crate) fn add_upper(&mut self, symbol: SymbolId, hi: IntExpr) {
        self.extra_upper.push((symbol, hi));
    }

    /// A path condition `symbol >= lo`.
    pub(crate) fn add_lower(&mut self, symbol: SymbolId, lo: IntExpr) {
        self.extra_lower.push((symbol, lo));
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

    fn lower_poly(&self, arena: &ExprArena, symbol: SymbolId) -> Poly {
        self.lower
            .get(&symbol)
            .map(|e| normalize_int(arena, *e))
            .unwrap_or_else(|| Poly::constant(0))
    }

    fn upper_poly(&self, arena: &ExprArena, symbol: SymbolId) -> Option<Poly> {
        self.upper.get(&symbol).map(|e| normalize_int(arena, *e))
    }

    fn zeros(&self, arena: &ExprArena) -> Vec<Poly> {
        self.zero.iter().map(|e| normalize_int(arena, *e)).collect()
    }
}

// ---------------------------------------------------------------------------
// The prover
// ---------------------------------------------------------------------------

/// Total rewriting steps one proof may spend across every branch.
const MAX_STEPS: usize = 4000;

struct Engine<'a> {
    arena: &'a ExprArena,
    facts: &'a Facts,
    zeros: Vec<Poly>,
    steps: usize,
    seen: HashSet<Poly>,
}

/// Prove `e >= 0` under the facts.
pub(crate) fn nonneg(arena: &ExprArena, facts: &Facts, e: IntExpr) -> bool {
    let mut engine = Engine::new(arena, facts);
    let p = normalize_int(arena, e);
    p.valid && engine.nonneg_steps(p, 24)
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

/// Interval arithmetic that bounds only the symbols `bound` accepts; other
/// symbols stay symbolic. Returns `(lo, hi)`.
pub(crate) fn interval_over(
    arena: &mut ExprArena,
    facts: &Facts,
    e: IntExpr,
    bound: &dyn Fn(SymbolId) -> bool,
) -> (IntExpr, IntExpr) {
    let engine = Engine::new(arena, facts);
    let p = normalize_int(arena, e);
    if !p.valid {
        return (e, e);
    }
    let (lo, hi) = engine.interval_over(&p, bound);
    if lo.valid && hi.valid {
        (intern(arena, &lo), intern(arena, &hi))
    } else {
        (e, e)
    }
}

/// Whether the binders in `radix` (magnitude of coefficient, range width)
/// can be ordered so that each magnitude exceeds `reach`, the largest value
/// the binders before it can contribute: then `sum(c_k * v_k)` is injective
/// over the box of ranges, as digits of a mixed radix are. The last binder
/// needs no width.
fn mixed_radix(
    radix: &[(Poly, Option<Poly>)],
    proves: &dyn Fn(&Poly) -> bool,
    used: &mut [bool],
    reach: Poly,
    one: &Poly,
) -> bool {
    let remaining = used.iter().filter(|u| !**u).count();
    if remaining == 0 {
        return true;
    }
    for i in 0..radix.len() {
        if used[i] {
            continue;
        }
        let (magnitude, width) = &radix[i];
        if !proves(&magnitude.sub(one).sub(&reach)) {
            continue;
        }
        used[i] = true;
        let ok = if remaining == 1 {
            true
        } else {
            match width {
                Some(width) => {
                    mixed_radix(radix, proves, used, reach.add(&width.mul(magnitude)), one)
                }
                None => false,
            }
        };
        if ok {
            return true;
        }
        used[i] = false;
    }
    false
}

/// One axis of a write, for the disjoint-visit proof.
pub(crate) enum WriteAxis {
    Opaque,
    /// The point index.
    Point(IntExpr),
    /// `start .. end`.
    Slice {
        start: IntExpr,
        end: IntExpr,
    },
}

/// Prove that distinct visits of the enclosing `parallel for` loops in
/// `binders` write distinct elements through `axes`.
///
/// A binder is proven by a point axis whose index is affine in it with a
/// nonzero coefficient once every other binder on that axis is already
/// proven; several unproven binders on one axis are proven together when
/// their coefficients form a mixed radix over the binders' ranges. A slice
/// `c*v + d : c*v + d + len` proves `v` when `len <= c`. Data-dependent,
/// nonlinear, and unbounded indices prove nothing. Returns the first binder
/// (by position) that stays unproven.
pub(crate) fn disjoint_visits(
    arena: &ExprArena,
    facts: &Facts,
    axes: &[WriteAxis],
    binders: &[SymbolId],
) -> Result<(), usize> {
    enum Axis {
        Opaque,
        Point(Vec<Poly>),
        Slice {
            binder: usize,
            coefficient: Poly,
            length: Poly,
        },
    }
    let mut normalized = Vec::with_capacity(axes.len());
    for axis in axes {
        normalized.push(match axis {
            WriteAxis::Opaque => Axis::Opaque,
            WriteAxis::Point(p) => {
                let p = normalize_int(arena, *p);
                let mut coefficients = Vec::with_capacity(binders.len());
                let mut opaque = false;
                for binder in binders {
                    match p.linear_coefficient(*binder) {
                        Some(c) => coefficients.push(c),
                        None => {
                            opaque = true;
                            break;
                        }
                    }
                }
                if opaque {
                    Axis::Opaque
                } else {
                    Axis::Point(coefficients)
                }
            }
            WriteAxis::Slice { start, end } => {
                let start = normalize_int(arena, *start);
                let length = normalize_int(arena, *end).sub(&start);
                let mut found = None;
                let mut opaque = false;
                for (k, binder) in binders.iter().enumerate() {
                    if length.mentions(*binder) {
                        opaque = true;
                        break;
                    }
                    match start.linear_coefficient(*binder) {
                        Some(c) if c.is_zero() => {}
                        Some(c) if found.is_none() => found = Some((k, c)),
                        _ => {
                            opaque = true;
                            break;
                        }
                    }
                }
                match found {
                    Some((binder, coefficient)) if !opaque => Axis::Slice {
                        binder,
                        coefficient,
                        length,
                    },
                    _ => Axis::Opaque,
                }
            }
        });
    }
    // A quotient/remainder pair is a lossless mixed-radix decomposition of
    // its numerator: `(x / d, x % d)` uniquely determines `x` for positive
    // `d`. Treat the pair as a virtual point axis so flattened row-major
    // writes retain the same injectivity proof as writing through `x`.
    for (left_index, left) in axes.iter().enumerate() {
        let WriteAxis::Point(left) = left else {
            continue;
        };
        let left = normalize_int(arena, *left);
        let Some(left_atom) = single_atom_of(&left) else {
            continue;
        };
        for right in axes.iter().skip(left_index + 1) {
            let WriteAxis::Point(right) = right else {
                continue;
            };
            let right = normalize_int(arena, *right);
            let Some(right_atom) = single_atom_of(&right) else {
                continue;
            };
            let numerator = match (&left_atom, &right_atom) {
                (Atom::Quot(left_num, left_den), Atom::Rem(right_num, right_den))
                | (Atom::Rem(left_num, left_den), Atom::Quot(right_num, right_den))
                    if left_num == right_num && left_den == right_den =>
                {
                    let mut engine = Engine::new(arena, facts);
                    if !engine.nonneg_steps(left_den.sub(&Poly::constant(1)), 24) {
                        continue;
                    }
                    left_num.as_ref()
                }
                _ => continue,
            };
            let mut coefficients = Vec::with_capacity(binders.len());
            let mut opaque = false;
            for binder in binders {
                match numerator.linear_coefficient(*binder) {
                    Some(coefficient) => coefficients.push(coefficient),
                    None => {
                        opaque = true;
                        break;
                    }
                }
            }
            if !opaque {
                normalized.push(Axis::Point(coefficients));
            }
        }
    }
    let one = Poly::constant(1);
    // The write executes inside every capturing loop, so each of their ranges
    // is nonempty: `upper - lower >= 0` is a fact the goal may spend.
    let nonempty: Vec<Poly> = binders
        .iter()
        .filter_map(|binder| {
            let upper = facts.upper_poly(arena, *binder)?;
            Some(upper.sub(&facts.lower_poly(arena, *binder)))
        })
        .collect();
    let proves = |goal: &Poly| -> bool {
        let mut engine = Engine::new(arena, facts);
        if engine.nonneg_steps(goal.clone(), 24) {
            return true;
        }
        nonempty.iter().any(|fact| {
            let mut engine = Engine::new(arena, facts);
            engine.nonneg_steps(goal.sub(fact), 24)
        })
    };
    let magnitude = |c: &Poly| -> Option<Poly> {
        if proves(&c.sub(&one)) {
            Some(c.clone())
        } else if proves(&c.neg().sub(&one)) {
            Some(c.neg())
        } else {
            None
        }
    };
    let mut proven = vec![false; binders.len()];
    loop {
        let mut progress = false;
        for axis in &normalized {
            match axis {
                Axis::Opaque => {}
                Axis::Slice {
                    binder,
                    coefficient,
                    length,
                } => {
                    if !proven[*binder] && proves(&coefficient.sub(length)) {
                        proven[*binder] = true;
                        progress = true;
                    }
                }
                Axis::Point(coefficients) => {
                    let mut pending: Vec<(usize, Poly)> = Vec::new();
                    let mut usable = true;
                    for (k, c) in coefficients.iter().enumerate() {
                        if proven[k] || c.is_zero() {
                            continue;
                        }
                        match magnitude(c) {
                            Some(m) => pending.push((k, m)),
                            None => {
                                usable = false;
                                break;
                            }
                        }
                    }
                    if !usable || pending.is_empty() {
                        continue;
                    }
                    let radix: Vec<(Poly, Option<Poly>)> = pending
                        .iter()
                        .map(|(k, m)| {
                            let width = facts
                                .upper_poly(arena, binders[*k])
                                .map(|upper| upper.sub(&facts.lower_poly(arena, binders[*k])));
                            (m.clone(), width)
                        })
                        .collect();
                    let mut used = vec![false; radix.len()];
                    if mixed_radix(&radix, &proves, &mut used, Poly::constant(0), &one) {
                        for (k, _) in &pending {
                            proven[*k] = true;
                        }
                        progress = true;
                    }
                }
            }
        }
        if !progress {
            break;
        }
    }
    match proven.iter().position(|p| !p) {
        Some(k) => Err(k),
        None => Ok(()),
    }
}

impl<'a> Engine<'a> {
    fn new(arena: &'a ExprArena, facts: &'a Facts) -> Engine<'a> {
        Engine {
            arena,
            facts,
            zeros: facts.zeros(arena),
            steps: 0,
            seen: HashSet::new(),
        }
    }

    fn nonneg_steps(&mut self, mut e: Poly, depth: usize) -> bool {
        if !e.valid {
            return false;
        }
        if depth == 0 || self.steps >= MAX_STEPS {
            return false;
        }
        self.steps += 1;
        if !self.seen.insert(e.clone()) {
            return false;
        }
        e = self.apply_zero_facts(e);
        // Trivially nonnegative: every coefficient nonnegative (atoms are
        // nonnegative).
        if e.terms.values().all(|c| *c >= 0) {
            return true;
        }
        // For a positive common divisor, floor is monotone and commutes with
        // integer translation: c + floor(a/d) >= floor(b/d) follows from
        // a + c*d >= b.
        for positive in e.atoms() {
            let Atom::Quot(a, d) = &positive else {
                continue;
            };
            if d.as_constant().is_none_or(|n| n <= 0) {
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
                let (lo, hi) = self.interval_over(n, &|_| true);
                for (bound, want_upper) in [(hi.quot(d), true), (lo.quot(d), false)] {
                    let signs: Vec<i64> = e
                        .terms
                        .iter()
                        .filter(|(m, _)| m.contains_key(&atom))
                        .map(|(_, c)| *c)
                        .collect();
                    let usable = if want_upper {
                        signs.iter().all(|c| *c < 0)
                    } else {
                        signs.iter().all(|c| *c > 0)
                    };
                    if usable && bound != Poly::atom(atom.clone()) {
                        let substituted = e.subst(&atom, &bound);
                        if substituted != e && self.nonneg_steps(substituted, depth - 1) {
                            return true;
                        }
                    }
                }
            }
        }
        // Bound an atom by its upper bound where it only ever lowers the
        // value, or by a nonzero lower bound where it only raises it.
        let mut candidates: Vec<(Atom, bool)> = Vec::new();
        for atom in e.atoms() {
            let signs: Vec<i64> = e
                .terms
                .iter()
                .filter(|(m, _)| m.contains_key(&atom))
                .map(|(_, c)| *c)
                .collect();
            let degree_one = e
                .terms
                .iter()
                .filter(|(m, _)| m.contains_key(&atom))
                .all(|(m, _)| m.get(&atom) == Some(&1));
            if !degree_one {
                continue;
            }
            if signs.iter().all(|c| *c < 0) {
                candidates.push((atom.clone(), true));
            } else if signs.iter().all(|c| *c > 0) {
                candidates.push((atom.clone(), false));
            }
        }
        let mut substitutions = Vec::new();
        for (atom, upper) in candidates {
            let bounds: Vec<Poly> = if upper {
                match &atom {
                    Atom::Rem(_, d) => vec![d.sub(&Poly::constant(1))],
                    Atom::Symbol(s) => self.facts.uppers_of(self.arena, *s),
                    Atom::Quot(..) | Atom::Foreign(_) => Vec::new(),
                }
            } else {
                match &atom {
                    Atom::Symbol(s) => self.facts.lowers_of(self.arena, *s),
                    _ => Vec::new(),
                }
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
            self.apply_zero_facts(s.clone())
                .terms
                .values()
                .all(|c| *c >= 0)
        }) {
            return true;
        }
        for substituted in substitutions {
            if self.nonneg_steps(substituted, depth - 1) {
                return true;
            }
        }
        false
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
            Atom::Rem(_, d) => (Poly::constant(0), Some(d.sub(&Poly::constant(1)))),
            Atom::Quot(n, d) => {
                let (lo, hi) = self.interval_over(n, bound);
                // Accepted integer division has a positive denominator, so
                // floor division is monotone even when that denominator is a
                // symbolic shape extent. Keeping the quotient here is what
                // proves row-major bounds such as
                // `(M*N - 1) / N == M - 1`.
                (lo.quot(d), Some(hi.quot(d)))
            }
            Atom::Symbol(s) => {
                if bound(*s) {
                    (
                        self.facts.lower_poly(self.arena, *s),
                        self.facts.upper_poly(self.arena, *s),
                    )
                } else {
                    (Poly::symbol(*s), Some(Poly::symbol(*s)))
                }
            }
            Atom::Foreign(_) => (Poly::constant(0), None),
        }
    }
}
