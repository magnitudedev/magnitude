//! The one polynomial normal form over arena integer nodes. The checker's
//! prover and the arena's subtraction and addition constructors both
//! normalize through it (A1 §2.2.6 item 9).
//!
//! A normalization turns the nodes it inspects into a transient polynomial
//! over symbol atoms, with floor quotients and remainders as opaque atoms. The
//! form is never stored; every value kept is re-interned into the arena.

use super::{AnyExpr, BinaryOp, ExprArena, IntExpr, NatExpr, NodeView, SymbolId, UnaryOp};
use std::collections::{BTreeMap, HashMap};
// ---------------------------------------------------------------------------
// Transient normal form
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Atom {
    Symbol(SymbolId),
    /// Euclidean quotient; positive-divisor rules require a proof below.
    Quot(Box<Poly>, Box<Poly>),
    /// Nonnegative Euclidean remainder; the divisor may be signed.
    Rem(Box<Poly>, Box<Poly>),
    /// A node outside the polynomial fragment, retaining its actual sort.
    Foreign(Foreign),
}

/// A foreign node keyed by its interned identity. Equal nodes are one key,
/// so two normalizations of the same node agree.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Foreign {
    key: u64,
    pub(crate) node: AnyExpr,
}

impl PartialEq for Foreign {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
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
        self.key
            .cmp(&other.key)
            // A hash is an ordering accelerator, never identity. Arena-owned
            // expression handles have a unique debug form (sort, owner,
            // ordinal), which closes the otherwise-unsound collision case.
            .then_with(|| format!("{:?}", self.node).cmp(&format!("{:?}", other.node)))
    }
}
impl std::hash::Hash for Foreign {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.node.hash(state);
    }
}

pub(crate) fn foreign(node: AnyExpr) -> Atom {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    node.hash(&mut hasher);
    Atom::Foreign(Foreign {
        key: hasher.finish(),
        node,
    })
}

pub(crate) type Monomial = BTreeMap<Atom, u32>;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Poly {
    /// monomial -> coefficient; the empty monomial is the constant term.
    pub(crate) terms: BTreeMap<Monomial, i64>,
    /// False means normalization exceeded the exact coefficient domain. An
    /// invalid polynomial can never discharge a proof obligation.
    pub(crate) valid: bool,
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
    pub(crate) fn constant(c: i64) -> Poly {
        let mut terms = BTreeMap::new();
        if c != 0 {
            terms.insert(Monomial::new(), c);
        }
        Poly { terms, valid: true }
    }

    pub(crate) fn atom(a: Atom) -> Poly {
        let mut m = Monomial::new();
        m.insert(a, 1);
        let mut terms = BTreeMap::new();
        terms.insert(m, 1);
        Poly { terms, valid: true }
    }

    pub(crate) fn symbol(s: SymbolId) -> Poly {
        Poly::atom(Atom::Symbol(s))
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.valid && self.terms.is_empty()
    }

    pub(crate) fn as_constant(&self) -> Option<i64> {
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

    pub(crate) fn constant_term(&self) -> i64 {
        self.terms.get(&Monomial::new()).copied().unwrap_or(0)
    }

    pub(crate) fn insert(&mut self, m: Monomial, c: i64) {
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

    pub(crate) fn add(&self, other: &Poly) -> Poly {
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

    pub(crate) fn neg(&self) -> Poly {
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

    pub(crate) fn sub(&self, other: &Poly) -> Poly {
        self.add(&other.neg())
    }

    pub(crate) fn mul(&self, other: &Poly) -> Poly {
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

    /// If `self` is `c * atom + rest` with `rest` free of `atom`, return
    /// `(c, rest)`.
    pub(crate) fn linear_in(&self, atom: &Atom) -> Option<(i64, Poly)> {
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
    pub(crate) fn div_exact(&self, c: i64) -> Option<Poly> {
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

    /// Euclidean quotient. Sign-dependent rewrites belong to the prover,
    /// which can establish a divisor's sign on the atom's defined path.
    pub(crate) fn quot(&self, den: &Poly) -> Poly {
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
                return exact.add(&Poly::atom(Atom::Quot(
                    Box::new(rest),
                    Box::new(den.clone()),
                )));
            }
        }
        Poly::atom(Atom::Quot(Box::new(self.clone()), Box::new(den.clone())))
    }

    /// `self / den` when the division is exact as polynomials.
    pub(crate) fn divide_exactly(&self, den: &Poly) -> Option<Poly> {
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

    /// Euclidean remainder, in [0, abs(den)) for a nonzero divisor.
    pub(crate) fn rem(&self, den: &Poly) -> Poly {
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

    pub(crate) fn atoms(&self) -> Vec<Atom> {
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
    pub(crate) fn subst(&self, atom: &Atom, value: &Poly) -> Poly {
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

    /// Whether `target` is mentioned anywhere, including inside quotient and
    /// remainder atoms (`i / 2` mentions `i` without being linear in it).
    pub(crate) fn mentions(&self, target: SymbolId) -> bool {
        self.atoms().iter().any(|atom| atom.mentions(target))
    }

    /// The coefficient of `symbol` when `self` is `c * symbol + rest` with
    /// neither `c` nor `rest` mentioning `symbol` (zero when absent); `None`
    /// when nonlinear in `symbol` or mentioning it inside a quotient or
    /// remainder. `c` may be symbolic.
    pub(crate) fn linear_coefficient(&self, symbol: SymbolId) -> Option<Poly> {
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
    pub(crate) fn mentions(&self, target: SymbolId) -> bool {
        match self {
            Atom::Symbol(s) => *s == target,
            Atom::Quot(n, d) | Atom::Rem(n, d) => n.mentions(target) || d.mentions(target),
            Atom::Foreign(_) => false,
        }
    }
}

/// The atom of a polynomial that is exactly one atom with coefficient 1.
pub(crate) fn single_atom_of(s: &Poly) -> Option<Atom> {
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
/// symbol-less encoding. Only sorts with a nonnegative value domain receive
/// an implicit zero bound.
pub(crate) fn normalize(arena: &ExprArena, node: AnyExpr) -> Poly {
    normalize_shared(arena, node, &mut HashMap::new())
}

/// [`normalize`] with every shared subnode normalized once, so the cost is
/// linear in the size of the node DAG.
fn normalize_shared(arena: &ExprArena, node: AnyExpr, memo: &mut HashMap<AnyExpr, Poly>) -> Poly {
    if let Some(known) = memo.get(&node) {
        return known.clone();
    }
    let value = match arena.view(node) {
        NodeView::NatConst(c) => match i64::try_from(c) {
            Ok(c) => Poly::constant(c),
            Err(_) => opaque(node),
        },
        NodeView::IntConst(c) => Poly::constant(c),
        NodeView::BoolConst(b) => Poly::constant(i64::from(b)),
        NodeView::ScalarConst { .. } => opaque(node),
        NodeView::ScalarInteger { .. } => opaque(node),
        NodeView::Symbol(s) => Poly::symbol(s),
        NodeView::Unary { op, operand } => match op {
            UnaryOp::NatFromInt | UnaryOp::IntFromNat | UnaryOp::IntFromScalar => {
                normalize_shared(arena, operand, memo)
            }
            UnaryOp::Not => Poly::constant(1).sub(&normalize_shared(arena, operand, memo)),
            UnaryOp::ScalarIntegerDefined => opaque(node),
        },
        NodeView::Binary { op, lhs, rhs } => {
            let l = normalize_shared(arena, lhs, memo);
            let r = normalize_shared(arena, rhs, memo);
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
            crate::expr::NaryOp::Product => operands.iter().fold(Poly::constant(1), |acc, o| {
                acc.mul(&normalize_shared(arena, *o, memo))
            }),
            _ => opaque(node),
        },
        NodeView::Select { .. }
        | NodeView::Cmp { .. }
        | NodeView::In { .. }
        | NodeView::Fold { .. }
        | NodeView::Duration(_)
        | NodeView::DurationScale { .. } => opaque(node),
    };
    memo.insert(node, value.clone());
    value
}

/// An operation outside the polynomial fragment: a nonnegative unknown for
/// `Nat`/`Bool`/`Cost` sorts, and a signed unknown for `Int`/`Scalar` sorts.
/// The atom re-interns to the identical source node, so proof bounds never
/// depend on synthetic parts that cannot be represented in the arena.
pub(crate) fn opaque(node: AnyExpr) -> Poly {
    Poly::atom(foreign(node))
}

pub(crate) fn normalize_int(arena: &ExprArena, e: IntExpr) -> Poly {
    normalize(arena, AnyExpr::Int(e))
}

/// Interns a normal form back into the arena.
pub(crate) fn intern(arena: &mut ExprArena, p: &Poly) -> IntExpr {
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
            Some(sum) => arena.inner.int_add(sum, term),
        });
    }
    match acc {
        Some(e) => e,
        None => arena.int(0),
    }
}

pub(crate) fn intern_atom(arena: &mut ExprArena, atom: &Atom) -> IntExpr {
    match atom {
        Atom::Symbol(s) => match arena.symbol_sort(*s) {
            crate::expr::SymbolSort::Nat => {
                let natural = arena.nat_symbol(*s);
                arena.int_from_nat(natural)
            }
            crate::expr::SymbolSort::Int => arena.int_symbol(*s),
            crate::expr::SymbolSort::Scalar(crate::types::DType::I32) => {
                let value = arena.scalar_symbol::<crate::expr::I32>(*s);
                arena.int_from_scalar(value)
            }
            crate::expr::SymbolSort::Scalar(crate::types::DType::U32) => {
                let value = arena.scalar_symbol::<crate::expr::U32>(*s);
                arena.int_from_scalar(value)
            }
            _ => panic!("checked integer algebra contains a non-integer symbol"),
        },
        Atom::Foreign(f) => match f.node {
            AnyExpr::Int(e) => e,
            AnyExpr::Nat(n) => arena.int_from_nat(n),
            AnyExpr::Bool(b) => {
                let one = arena.int(1);
                let zero = arena.int(0);
                arena.int_select(b, one, zero)
            }
            AnyExpr::Scalar(s) => arena.int_from_scalar_value(s),
            AnyExpr::Duration(_) => {
                unreachable!("duration cannot occur in an integer expression")
            }
        },
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

/// Interns a normal form whose coefficients are all nonnegative and whose
/// atoms are all natural-valued as a natural node.
fn intern_nat(arena: &mut ExprArena, p: &Poly) -> NatExpr {
    assert!(p.valid, "invalid normal form cannot be materialized");
    let mut acc: Option<NatExpr> = None;
    for (monomial, coefficient) in &p.terms {
        let coefficient = u64::try_from(*coefficient)
            .unwrap_or_else(|_| panic!("natural normal form has a negative coefficient"));
        let mut term = arena.nat(coefficient);
        for (atom, power) in monomial {
            let factor = match atom {
                Atom::Symbol(symbol) => arena.nat_symbol(*symbol),
                Atom::Foreign(Foreign {
                    node: AnyExpr::Nat(node),
                    ..
                }) => *node,
                _ => unreachable!("natural normal form has a non-natural atom"),
            };
            for _ in 0..*power {
                term = arena.nat_mul(term, factor);
            }
        }
        acc = Some(match acc {
            None => term,
            Some(sum) => arena.nat_add(sum, term),
        });
    }
    acc.unwrap_or_else(|| arena.nat(0))
}

/// Whether a normal form can be re-interned by the arena's constructors
/// without dropping a side condition or binding a loop binder into the
/// result: it is valid, has no quotient or remainder atom, and no atom
/// mentions a `LoopBinder` (A1 §7 "Polynomial normalization").
fn rebuildable(arena: &ExprArena, p: &Poly) -> bool {
    p.valid
        && p.atoms().iter().all(|atom| match atom {
            Atom::Symbol(symbol) => {
                !matches!(arena.symbol_kind(*symbol), super::SymbolKind::LoopBinder(_))
            }
            Atom::Foreign(foreign) => arena.free_symbols(foreign.node).iter().all(|symbol| {
                !matches!(arena.symbol_kind(*symbol), super::SymbolKind::LoopBinder(_))
            }),
            Atom::Quot(..) | Atom::Rem(..) => false,
        })
}

fn natural_atom(arena: &ExprArena, atom: &Atom) -> bool {
    match atom {
        Atom::Symbol(symbol) => arena.symbol_sort(*symbol) == super::SymbolSort::Nat,
        Atom::Foreign(foreign) => matches!(foreign.node, AnyExpr::Nat(_)),
        Atom::Quot(..) | Atom::Rem(..) => false,
    }
}

/// `a - b` in the one normal form, when both operands are total, combining
/// them cancels a term, and the normal form is rebuildable with nonnegative
/// coefficients over natural atoms. The value and the side condition `b <= a` then hold identically.
pub(crate) fn natural_difference(arena: &mut ExprArena, a: NatExpr, b: NatExpr) -> Option<NatExpr> {
    if !arena.is_total(AnyExpr::Nat(a)) || !arena.is_total(AnyExpr::Nat(b)) {
        return None;
    }
    let mut memo = HashMap::new();
    let left = normalize_shared(arena, AnyExpr::Nat(a), &mut memo);
    let right = normalize_shared(arena, AnyExpr::Nat(b), &mut memo);
    let difference = left.sub(&right);
    let natural = difference.terms.values().all(|coefficient| *coefficient >= 0)
        && difference.atoms().iter().all(|atom| natural_atom(arena, atom));
    (natural && cancels(&left, &right, &difference) && rebuildable(arena, &difference))
        .then(|| intern_nat(arena, &difference))
}

/// Whether combining the operands merged or cancelled a term. Otherwise the
/// constructor keeps today's node, so a sum with nothing to cancel keeps its
/// authored structure.
fn cancels(left: &Poly, right: &Poly, combined: &Poly) -> bool {
    combined.terms.len() < left.terms.len() + right.terms.len()
}

/// `a + b` (`negate_right == false`) or `a - b` in the one normal form, when
/// both operands are total, combining them cancels a term, and the normal
/// form is rebuildable.
pub(crate) fn integer_sum(
    arena: &mut ExprArena,
    a: IntExpr,
    b: IntExpr,
    negate_right: bool,
) -> Option<IntExpr> {
    if !arena.is_total(AnyExpr::Int(a)) || !arena.is_total(AnyExpr::Int(b)) {
        return None;
    }
    let mut memo = HashMap::new();
    let left = normalize_shared(arena, AnyExpr::Int(a), &mut memo);
    let right = normalize_shared(arena, AnyExpr::Int(b), &mut memo);
    let sum = if negate_right { left.sub(&right) } else { left.add(&right) };
    (cancels(&left, &right, &sum) && rebuildable(arena, &sum)).then(|| intern(arena, &sum))
}

#[cfg(test)]
mod tests {
    use crate::expr::{AnyExpr, ExprArena, NodeView, SymbolSort};

    #[test]
    fn natural_difference_cancels_to_the_remaining_term() {
        let mut arena = ExprArena::new();
        let (_, a) = arena.target_constant(SymbolSort::Nat);
        let (_, g) = arena.target_constant(SymbolSort::Nat);
        let a = arena.nat_symbol(a);
        let g = arena.nat_symbol(g);
        let one = arena.nat(1);
        let next = arena.nat_add(a, one);
        let end = arena.nat_mul(next, g);
        let start = arena.nat_mul(a, g);
        assert_eq!(arena.nat_sub(end, start), g);
    }

    #[test]
    fn unrelated_difference_keeps_its_side_condition() {
        let mut arena = ExprArena::new();
        let (_, n) = arena.target_constant(SymbolSort::Nat);
        let (_, m) = arena.target_constant(SymbolSort::Nat);
        let n = arena.nat_symbol(n);
        let m = arena.nat_symbol(m);
        let difference = arena.nat_sub(n, m);
        assert!(matches!(
            arena.view(AnyExpr::Nat(difference)),
            NodeView::Binary { op: crate::expr::BinaryOp::Sub, .. }
        ));
        assert!(!arena.is_total(AnyExpr::Nat(difference)));
    }

    #[test]
    fn partial_operands_are_never_normalized() {
        let mut arena = ExprArena::new();
        let (_, x) = arena.target_constant(SymbolSort::Nat);
        let x = arena.nat_symbol(x);
        let five = arena.nat(5);
        let partial = arena.nat_sub(x, five);
        let difference = arena.nat_sub(partial, partial);
        assert!(matches!(
            arena.view(AnyExpr::Nat(difference)),
            NodeView::Binary { op: crate::expr::BinaryOp::Sub, .. }
        ));
        assert!(!arena.is_total(AnyExpr::Nat(difference)));
    }

    #[test]
    fn loop_binder_results_keep_the_authored_node() {
        let mut arena = ExprArena::new();
        let (_, _, i) = arena.loop_binder();
        let (_, g) = arena.target_constant(SymbolSort::Int);
        let g = arena.int_symbol(g);
        let twice = arena.int_add(i, i);
        let difference = arena.int_sub(twice, g);
        // `i + i - g` keeps `i`, so it is not rebuilt.
        assert!(matches!(
            arena.view(AnyExpr::Int(difference)),
            NodeView::Binary { op: crate::expr::BinaryOp::Sub, .. }
        ));
        let cancelled = arena.int_sub(twice, i);
        assert!(matches!(
            arena.view(AnyExpr::Int(cancelled)),
            NodeView::Binary { op: crate::expr::BinaryOp::Sub, .. }
        ));
        let one = arena.int(1);
        let shifted = arena.int_add(g, one);
        assert_eq!(arena.int_sub(shifted, one), g);
    }
}
