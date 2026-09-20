//! Symbolic integers.
//!
//! A `Sym` is a normalized polynomial with integer coefficients over atoms.
//! An atom is a named parameter, a floor quotient `x / c`, or a remainder
//! `x % c`. Every atom is a nonnegative integer; shape parameters and loop
//! indices are never negative in this language.
//!
//! The prover establishes `e >= 0` and `e == 0` under a set of facts by
//! rewriting quotients and remainders through `x = c * (x / c) + x % c` and
//! bounding atoms that carry an upper bound. It is sound and incomplete: a
//! failed proof is a residual the caller reports or records as a precondition.

use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Atom {
    Param(String),
    /// floor(num / den), den > 0 in normalized form
    Quot(Box<Sym>, Box<Sym>),
    /// num mod den, den > 0
    Rem(Box<Sym>, Box<Sym>),
}

/// A monomial is a sorted product of atoms with multiplicities.
pub type Monomial = BTreeMap<Atom, u32>;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Sym {
    /// monomial -> coefficient; the empty monomial is the constant term.
    terms: BTreeMap<Monomial, i64>,
}

impl Sym {
    /// Structural traversal for native code generation and resource analysis.
    /// Consumers must not recover this structure by parsing the display string.
    pub fn monomials(&self) -> impl Iterator<Item = (&Monomial, i64)> {
        self.terms
            .iter()
            .map(|(monomial, coefficient)| (monomial, *coefficient))
    }

    /// A divisor of every integer value of this polynomial, independent of
    /// parameter bounds. Zero denotes the identically-zero polynomial.
    pub fn coefficient_divisor(&self) -> u64 {
        fn gcd(mut a: u64, mut b: u64) -> u64 {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        }
        self.terms.values().fold(0, |g, c| gcd(g, c.unsigned_abs()))
    }

    /// If `self` is `c * atom + rest` with `rest` free of `atom`, return `(c, rest)`.
    pub fn linear_in(&self, atom: &Atom) -> Option<(i64, Sym)> {
        let mut c = 0i64;
        let mut rest = Sym::default();
        for (m, k) in &self.terms {
            match m.get(atom) {
                None => {
                    rest.terms.insert(m.clone(), *k);
                }
                Some(1) if m.len() == 1 => c += *k,
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
    pub fn div_exact(&self, c: i64) -> Option<Sym> {
        let mut out = Sym::default();
        for (m, k) in &self.terms {
            if k % c != 0 {
                return None;
            }
            out.terms.insert(m.clone(), k / c);
        }
        Some(out)
    }

    pub fn constant(c: i64) -> Sym {
        let mut terms = BTreeMap::new();
        if c != 0 {
            terms.insert(Monomial::new(), c);
        }
        Sym { terms }
    }

    pub fn param(name: &str) -> Sym {
        Sym::atom(Atom::Param(name.to_string()))
    }

    pub fn atom(a: Atom) -> Sym {
        let mut m = Monomial::new();
        m.insert(a, 1);
        let mut terms = BTreeMap::new();
        terms.insert(m, 1);
        Sym { terms }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn as_constant(&self) -> Option<i64> {
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

    pub fn constant_term(&self) -> i64 {
        self.terms.get(&Monomial::new()).copied().unwrap_or(0)
    }

    fn insert(&mut self, m: Monomial, c: i64) {
        if c == 0 {
            return;
        }
        let entry = self.terms.entry(m).or_insert(0);
        *entry += c;
        if *entry == 0 {
            let key = self
                .terms
                .iter()
                .find(|(_, v)| **v == 0)
                .map(|(k, _)| k.clone());
            if let Some(k) = key {
                self.terms.remove(&k);
            }
        }
    }

    pub fn add(&self, other: &Sym) -> Sym {
        let mut out = self.clone();
        for (m, c) in &other.terms {
            out.insert(m.clone(), *c);
        }
        out
    }

    pub fn neg(&self) -> Sym {
        Sym {
            terms: self.terms.iter().map(|(m, c)| (m.clone(), -c)).collect(),
        }
    }

    pub fn sub(&self, other: &Sym) -> Sym {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &Sym) -> Sym {
        let mut out = Sym::default();
        for (m1, c1) in &self.terms {
            for (m2, c2) in &other.terms {
                let mut m = m1.clone();
                for (a, k) in m2 {
                    *m.entry(a.clone()).or_insert(0) += k;
                }
                out.insert(m, c1 * c2);
            }
        }
        out
    }

    pub fn scale(&self, c: i64) -> Sym {
        self.mul(&Sym::constant(c))
    }

    /// floor(self / den). Requires den to be provably positive; the caller checks.
    pub fn quot(&self, den: &Sym) -> Sym {
        if let (Some(a), Some(b)) = (self.as_constant(), den.as_constant()) {
            if b != 0 {
                return Sym::constant(a.div_euclid(b));
            }
        }
        if den.as_constant() == Some(1) {
            return self.clone();
        }
        if *self == *den {
            return Sym::constant(1);
        }
        // `(c * d) / d` is `c` when the numerator is a product of the denominator: divide
        // term by term where the denominator is a single monomial, and otherwise try to
        // factor the denominator out of the whole numerator.
        if let Some(q) = self.divide_exactly(den) {
            return q;
        }
        // `H / (H / KV)` is `KV`: the denominator is itself a quotient whose numerator is
        // this one, and the division is exact.
        if let Some(atom) = single_atom_of(den) {
            if let Atom::Quot(n, d) = &atom {
                if **n == *self && n.divide_exactly(d).is_some() {
                    return (**d).clone();
                }
            }
        }
        // Pull out the part of the numerator that is an exact multiple of a constant denominator.
        if let Some(c) = den.as_constant() {
            if c > 0 {
                let mut exact = Sym::default();
                let mut rest = Sym::default();
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
                    return exact.add(&Sym::atom(Atom::Quot(
                        Box::new(rest),
                        Box::new(den.clone()),
                    )));
                }
            }
        }
        // A single-atom denominator: pull out the monomials it divides. A constant remainder
        // of -1 over a shape parameter (>= 1) floors to -1.
        if let Some(atom) = single_atom_of(den) {
            let mut exact = Sym::default();
            let mut rest = Sym::default();
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
                if rest.as_constant() == Some(-1)
                    && matches!(&atom, Atom::Param(p) if !p.contains('#'))
                {
                    return exact.sub(&Sym::constant(1));
                }
                return exact.add(&Sym::atom(Atom::Quot(
                    Box::new(rest),
                    Box::new(den.clone()),
                )));
            }
        }
        Sym::atom(Atom::Quot(Box::new(self.clone()), Box::new(den.clone())))
    }

    /// `self / den` when the division is exact as polynomials: every monomial of `self` is
    /// divisible by a single-monomial denominator, or `self` factors as `den * q`.
    pub fn divide_exactly(&self, den: &Sym) -> Option<Sym> {
        if den.is_zero() {
            return None;
        }
        // Single-monomial denominator: divide each term of the numerator by it.
        if den.terms.len() == 1 {
            let (dm, dc) = den.terms.iter().next().unwrap();
            let mut out = Sym::default();
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
        // Multi-term denominator: try each candidate quotient monomial in turn. The quotient
        // of a product of sums has degree equal to the difference, so a single monomial
        // quotient covers the cases that arise here (`(H*R + H*S) / (R + S)` is `H`).
        let (dm, dc) = den.terms.iter().next().unwrap();
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
            let mut q = Sym::default();
            q.insert(q_mono, c / dc);
            if q.mul(den) == *self {
                return Some(q);
            }
        }
        None
    }

    /// self mod den, in [0, den).
    pub fn rem(&self, den: &Sym) -> Sym {
        if let (Some(a), Some(b)) = (self.as_constant(), den.as_constant()) {
            if b != 0 {
                return Sym::constant(a.rem_euclid(b));
            }
        }
        if den.as_constant() == Some(1) {
            return Sym::constant(0);
        }
        if let Some(c) = den.as_constant() {
            if c > 0 {
                let mut rest = Sym::default();
                for (m, k) in &self.terms {
                    if k % c != 0 {
                        rest.insert(m.clone(), *k);
                    }
                }
                if rest.is_zero() {
                    return Sym::constant(0);
                }
                return Sym::atom(Atom::Rem(Box::new(rest), Box::new(den.clone())));
            }
        }
        Sym::atom(Atom::Rem(Box::new(self.clone()), Box::new(den.clone())))
    }

    pub fn atoms(&self) -> Vec<Atom> {
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
    pub fn subst(&self, atom: &Atom, value: &Sym) -> Sym {
        let mut out = Sym::default();
        for (m, c) in &self.terms {
            let mut term = Sym::constant(*c);
            for (a, k) in m {
                let factor = if a == atom {
                    value.clone()
                } else {
                    Sym::atom(a.clone())
                };
                for _ in 0..*k {
                    term = term.mul(&factor);
                }
            }
            out = out.add(&term);
        }
        out
    }

    /// Evaluate with concrete values for every parameter atom. `None` if one is missing.
    pub fn eval(&self, env: &dyn Fn(&str) -> Option<i64>) -> Option<i64> {
        let mut total: i64 = 0;
        for (m, c) in &self.terms {
            let mut term = *c;
            for (a, k) in m {
                let v = a.eval(env)?;
                for _ in 0..*k {
                    term = term.checked_mul(v)?;
                }
            }
            total = total.checked_add(term)?;
        }
        Some(total)
    }

    /// Checked numeric enclosure under bounded parameter values. Unlike
    /// substituting capacities into an expression, this also bounds remainders
    /// and negative terms. Every intermediate in the structural evaluation must
    /// fit i64; unknown bounds or a potentially zero divisor remain unsupported.
    pub fn eval_interval(&self, env: &dyn Fn(&str) -> Option<(i64, i64)>) -> Option<(i64, i64)> {
        self.eval_interval_with(env, &|_| None)
    }

    /// Intersect proven expression bounds at every node, including inside
    /// quotient and remainder operands. Facts must hold throughout the domain
    /// supplied by the caller; contradictory facts produce no enclosure.
    pub fn eval_interval_with(
        &self,
        env: &dyn Fn(&str) -> Option<(i64, i64)>,
        facts: &dyn Fn(&Sym) -> Option<(i64, i64)>,
    ) -> Option<(i64, i64)> {
        fn refine(
            expression: &Sym,
            mut range: (i64, i64),
            facts: &dyn Fn(&Sym) -> Option<(i64, i64)>,
        ) -> Option<(i64, i64)> {
            if let Some((lo, hi)) = facts(expression) {
                range.0 = range.0.max(lo);
                range.1 = range.1.min(hi);
            }
            (range.0 <= range.1).then_some(range)
        }
        fn atom(
            a: &Atom,
            env: &dyn Fn(&str) -> Option<(i64, i64)>,
            facts: &dyn Fn(&Sym) -> Option<(i64, i64)>,
        ) -> Option<(i64, i64)> {
            let range = match a {
                Atom::Param(name) => env(name)?,
                Atom::Quot(n, d) | Atom::Rem(n, d) => {
                    let (nl, nh) = n.eval_interval_with(env, facts)?;
                    let (dl, dh) = d.eval_interval_with(env, facts)?;
                    if nl < 0 || dl <= 0 {
                        return None;
                    }
                    if matches!(a, Atom::Quot(..)) {
                        (nl / dh, nh / dl)
                    } else if dl == dh && nl / dl == nh / dl {
                        (nl % dl, nh % dl)
                    } else {
                        let mut step = 1;
                        if dl == dh {
                            let (mut a, mut b) = (n.coefficient_divisor(), dl as u64);
                            while b != 0 {
                                (a, b) = (b, a % b);
                            }
                            step = i64::try_from(a).ok()?;
                        }
                        (0, nh.min(dh - step))
                    }
                }
            };
            let range = refine(&Sym::atom(a.clone()), range, facts)?;
            (range.0 >= 0 && range.0 <= range.1).then_some(range)
        }
        let (mut low, mut high) = (0i64, 0i64);
        for (monomial, coefficient) in &self.terms {
            let (mut lo, mut hi) = (*coefficient, *coefficient);
            for (a, power) in monomial {
                let (al, ah) = atom(a, env, facts)?;
                for _ in 0..*power {
                    let products = [
                        lo.checked_mul(al)?,
                        lo.checked_mul(ah)?,
                        hi.checked_mul(al)?,
                        hi.checked_mul(ah)?,
                    ];
                    lo = *products.iter().min()?;
                    hi = *products.iter().max()?;
                }
            }
            low = low.checked_add(lo)?;
            high = high.checked_add(hi)?;
        }
        refine(self, (low, high), facts)
    }

    pub fn params(&self) -> Vec<String> {
        let mut out = Vec::new();
        fn visit(a: &Atom, out: &mut Vec<String>) {
            match a {
                Atom::Param(p) => {
                    if !out.contains(p) {
                        out.push(p.clone());
                    }
                }
                Atom::Quot(n, d) | Atom::Rem(n, d) => {
                    for x in n.atoms().iter().chain(d.atoms().iter()) {
                        visit(x, out);
                    }
                }
            }
        }
        for a in self.atoms() {
            visit(&a, &mut out);
        }
        out
    }
}

impl Atom {
    /// Whether this atom mentions a loop-scoped symbol (named with `#`), directly or inside
    /// a quotient or remainder.
    pub fn mentions_loop(&self) -> bool {
        match self {
            Atom::Param(p) => p.contains('#'),
            Atom::Quot(n, d) | Atom::Rem(n, d) => n
                .params()
                .iter()
                .chain(d.params().iter())
                .any(|p| p.contains('#')),
        }
    }

    pub fn eval(&self, env: &dyn Fn(&str) -> Option<i64>) -> Option<i64> {
        match self {
            Atom::Param(p) => env(p),
            Atom::Quot(n, d) => {
                let d = d.eval(env)?;
                if d <= 0 {
                    return None;
                }
                Some(n.eval(env)?.div_euclid(d))
            }
            Atom::Rem(n, d) => {
                let d = d.eval(env)?;
                if d <= 0 {
                    return None;
                }
                Some(n.eval(env)?.rem_euclid(d))
            }
        }
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Atom::Param(p) => write!(f, "{p}"),
            Atom::Quot(n, d) => write!(f, "({} / {})", grouped(n), grouped(d)),
            Atom::Rem(n, d) => write!(f, "({} % {})", grouped(n), grouped(d)),
        }
    }
}

/// The atom of a symbol that is exactly one atom with coefficient 1.
fn single_atom_of(s: &Sym) -> Option<Atom> {
    if s.terms.len() != 1 {
        return None;
    }
    let (m, k) = s.terms.iter().next().unwrap();
    if *k != 1 || m.len() != 1 || *m.values().next().unwrap() != 1 {
        return None;
    }
    Some(m.keys().next().unwrap().clone())
}

/// A sum with more than one term is parenthesized where it appears as an operand.
fn grouped(s: &Sym) -> String {
    if s.terms.len() > 1 {
        format!("({s})")
    } else {
        format!("{s}")
    }
}

impl fmt::Display for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        let mut first = true;
        for (m, c) in &self.terms {
            let sign = if *c < 0 { "-" } else { "+" };
            if !first || *c < 0 {
                if first {
                    write!(f, "-")?;
                } else {
                    write!(f, " {sign} ")?;
                }
            }
            first = false;
            let mag = c.abs();
            if m.is_empty() {
                write!(f, "{mag}")?;
                continue;
            }
            if mag != 1 {
                write!(f, "{mag} * ")?;
            }
            let mut factors = Vec::new();
            for (a, k) in m {
                for _ in 0..*k {
                    factors.push(a.to_string());
                }
            }
            write!(f, "{}", factors.join(" * "))?;
        }
        Ok(())
    }
}

/// Known bounds on atoms, and equalities that hold.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    /// atom -> inclusive upper bound
    upper: BTreeMap<Atom, Sym>,
    /// atom -> inclusive lower bound (default 0)
    lower: BTreeMap<Atom, Sym>,
    /// Further bounds from path conditions; every entry is valid on its own. Kept in order
    /// so a branch can roll back exactly what it added.
    extra_upper: Vec<(Atom, Sym)>,
    extra_lower: Vec<(Atom, Sym)>,
    /// expressions known to be zero
    zero: Vec<Sym>,
}

impl Facts {
    pub fn new() -> Facts {
        Facts::default()
    }

    pub fn set_range_lower(&mut self, atom: Atom, lo: Sym) {
        self.lower.insert(atom, lo);
    }

    pub fn set_range(&mut self, atom: Atom, lo: Sym, hi: Sym) {
        self.lower.insert(atom.clone(), lo);
        self.upper.insert(atom, hi);
    }

    pub fn assume_zero(&mut self, e: Sym) {
        if !e.is_zero() {
            self.zero.push(e);
        }
    }

    pub fn assume_divisible(&mut self, e: &Sym, c: &Sym) {
        self.assume_zero(e.rem(c));
    }

    pub fn lower_of(&self, a: &Atom) -> Sym {
        self.lower
            .get(a)
            .cloned()
            .unwrap_or_else(|| Sym::constant(0))
    }

    pub fn upper_of(&self, a: &Atom) -> Option<Sym> {
        self.upper.get(a).cloned()
    }

    /// A path condition `atom <= hi`.
    pub fn add_upper(&mut self, atom: Atom, hi: Sym) {
        self.extra_upper.push((atom, hi));
    }

    /// A path condition `atom >= lo`.
    pub fn add_lower(&mut self, atom: Atom, lo: Sym) {
        self.extra_lower.push((atom, lo));
    }

    /// Every known upper bound of an atom.
    pub fn uppers_of(&self, a: &Atom) -> Vec<Sym> {
        let mut out: Vec<Sym> = self.upper.get(a).cloned().into_iter().collect();
        out.extend(
            self.extra_upper
                .iter()
                .filter(|(x, _)| x == a)
                .map(|(_, s)| s.clone()),
        );
        out
    }

    /// Every known nonzero lower bound of an atom.
    pub fn lowers_of(&self, a: &Atom) -> Vec<Sym> {
        let mut out: Vec<Sym> = self
            .lower
            .get(a)
            .cloned()
            .into_iter()
            .filter(|l| !l.is_zero())
            .collect();
        out.extend(
            self.extra_lower
                .iter()
                .filter(|(x, s)| x == a && !s.is_zero())
                .map(|(_, s)| s.clone()),
        );
        out
    }
}

/// Inclusive interval with symbolic ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interval {
    pub lo: Sym,
    pub hi: Sym,
}

impl Interval {
    pub fn point(s: Sym) -> Interval {
        Interval {
            lo: s.clone(),
            hi: s,
        }
    }
}

pub struct Prover<'a> {
    pub facts: &'a Facts,
    steps: std::cell::Cell<usize>,
    seen: std::cell::RefCell<std::collections::HashSet<Sym>>,
}

/// Total rewriting steps one proof may spend across every branch.
const MAX_STEPS: usize = 4000;

impl<'a> Prover<'a> {
    pub fn new(facts: &'a Facts) -> Prover<'a> {
        Prover {
            facts,
            steps: std::cell::Cell::new(0),
            seen: std::cell::RefCell::new(std::collections::HashSet::new()),
        }
    }

    /// Prove `e >= 0`.
    pub fn nonneg(&self, e: &Sym) -> bool {
        self.steps.set(0);
        self.seen.borrow_mut().clear();
        self.nonneg_steps(e.clone(), 24)
    }

    fn nonneg_steps(&self, mut e: Sym, depth: usize) -> bool {
        if depth == 0 || self.steps.get() >= MAX_STEPS {
            return false;
        }
        self.steps.set(self.steps.get() + 1);
        if !self.seen.borrow_mut().insert(e.clone()) {
            return false;
        }
        e = self.apply_zero_facts(e);
        // Trivially nonnegative: every coefficient nonnegative (atoms are nonnegative).
        if e.terms.values().all(|c| *c >= 0) {
            return true;
        }
        // For a positive common divisor, floor is monotone and commutes with
        // integer translation: c + floor(a/d) >= floor(b/d) follows from
        // a + c*d >= b. Keep every other term as a separately proved remainder.
        // This preserves correlations between row/block indices and padded byte
        // extents that independent interval bounds otherwise discard.
        for positive in e.atoms() {
            let Atom::Quot(a, d) = &positive else {
                continue;
            };
            if d.as_constant().is_none_or(|n| n <= 0) {
                continue;
            }
            let positive_term = Sym::atom(positive.clone());
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
                let constant = Sym::constant(e.constant_term());
                let rest = e
                    .sub(&positive_term)
                    .add(&Sym::atom(negative.clone()))
                    .sub(&constant);
                if (rest.is_zero() || self.nonneg_steps(rest, depth - 1))
                    && self.nonneg_steps(a.add(&constant.mul(d)).sub(b), depth - 1)
                {
                    return true;
                }
            }
        }
        // Rewrite a quotient or remainder atom through the division identity, then retry.
        for atom in e.atoms() {
            match &atom {
                Atom::Quot(n, d) => {
                    // n = d * q + r, so q = (n - r) / d cannot be substituted linearly; instead
                    // substitute occurrences of the numerator's params is not possible. Use bounds:
                    // q <= n / d (real) is not linear either. Substitute n's representation instead:
                    // introduce r = n % d and rewrite n := d * q + r everywhere.
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
                Atom::Param(_) => {}
            }
        }
        // A quotient is monotone in its numerator: q <= hi(n) / d and q >= lo(n) / d.
        for atom in e.atoms() {
            if let Atom::Quot(n, d) = &atom {
                let iv = self.interval(n);
                for (bound, want_upper) in [(iv.hi.quot(d), true), (iv.lo.quot(d), false)] {
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
                    if usable && bound != Sym::atom(atom.clone()) {
                        let substituted = e.subst(&atom, &bound);
                        if substituted != e && self.nonneg_steps(substituted, depth - 1) {
                            return true;
                        }
                    }
                }
            }
        }
        // Bound an atom by its upper bound where it only ever lowers the value (every term
        // containing it is negative), or by a nonzero lower bound where it only raises it.
        // Atoms are nonnegative, so this holds inside products too.
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
            let atom = &atom;
            let c: i64 = if upper { -1 } else { 1 };
            let bounds: Vec<Sym> = if c < 0 {
                match atom {
                    Atom::Rem(_, d) => vec![d.sub(&Sym::constant(1))],
                    _ => self.facts.uppers_of(atom),
                }
            } else {
                self.facts.lowers_of(atom)
            };
            for b in bounds {
                let substituted = e.subst(atom, &b);
                if substituted != e {
                    substitutions.push(substituted);
                }
            }
        }
        // Try every immediate bound before recursively combining bounds. Otherwise
        // cyclic path facts can exhaust the proof budget before the exact defining
        // loop bound is considered (e.g. i < 2*P+S and P <= i).
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

    /// Replace a parameter that is the numerator of `q = n / d` by `d * q + r`.
    fn rewrite_numerator(&self, e: &Sym, n: &Sym, d: &Sym, q: &Atom) -> Option<Sym> {
        // Only handle a numerator that is a single parameter; that covers shape parameters.
        let param = match n.atoms().as_slice() {
            [Atom::Param(p)] if n == &Sym::param(p) => Atom::Param(p.clone()),
            _ => return None,
        };
        let r = Sym::atom(Atom::Rem(Box::new(n.clone()), Box::new(d.clone())));
        let replacement = d.mul(&Sym::atom(q.clone())).add(&r);
        let out = e.subst(&param, &replacement);
        if out == *e {
            None
        } else {
            Some(out)
        }
    }

    fn apply_zero_facts(&self, e: Sym) -> Sym {
        let mut e = e;
        for z in &self.facts.zero {
            // A zero fact of the form `atom == 0` lets us drop that atom.
            if let [atom] = z.atoms().as_slice() {
                if *z == Sym::atom(atom.clone()) {
                    e = e.subst(atom, &Sym::constant(0));
                }
            }
        }
        e
    }

    pub fn zero(&self, e: &Sym) -> bool {
        let e = self.apply_zero_facts(e.clone());
        e.is_zero() || self.facts.zero.contains(&e) || self.facts.zero.contains(&e.neg())
    }

    /// `a <= b`
    pub fn le(&self, a: &Sym, b: &Sym) -> bool {
        self.nonneg(&b.sub(a))
    }

    /// `a < b`
    pub fn lt(&self, a: &Sym, b: &Sym) -> bool {
        self.nonneg(&b.sub(a).sub(&Sym::constant(1)))
    }

    /// Bounds of one atom: facts for parameters, [0, d-1] for remainders, and for a quotient
    /// with a constant divisor the quotients of the numerator's bounds.
    fn atom_bounds(&self, a: &Atom) -> (Sym, Option<Sym>) {
        match a {
            Atom::Rem(_, d) => (Sym::constant(0), Some(d.sub(&Sym::constant(1)))),
            Atom::Quot(n, d) => {
                // floor(n / d) with d >= 1 never exceeds n; with a constant d the numerator's
                // bounds divide through.
                let inner = self.interval(n);
                if d.as_constant().is_some() {
                    (inner.lo.quot(d), Some(inner.hi.quot(d)))
                } else {
                    (Sym::constant(0), Some(inner.hi))
                }
            }
            _ => (self.facts.lower_of(a), self.facts.upper_of(a)),
        }
    }

    /// Interval arithmetic over symbolic ends, using the facts for atom bounds.
    pub fn interval(&self, e: &Sym) -> Interval {
        self.interval_over(e, &|_| true)
    }

    /// Interval arithmetic that bounds only the atoms `bound` accepts; other atoms stay symbolic.
    /// Used to eliminate loop indices from a constraint while keeping shape parameters exact.
    pub fn interval_over(&self, e: &Sym, bound: &dyn Fn(&Atom) -> bool) -> Interval {
        let mut lo = Sym::default();
        let mut hi = Sym::default();
        for (m, c) in &e.terms {
            let mut lo_term = Sym::constant(*c);
            let mut hi_term = Sym::constant(*c);
            for (a, k) in m {
                let (l, u) = if bound(a) {
                    self.atom_bounds_over(a, bound)
                } else {
                    (Sym::atom(a.clone()), Some(Sym::atom(a.clone())))
                };
                for _ in 0..*k {
                    if *c >= 0 {
                        lo_term = lo_term.mul(&l);
                        match &u {
                            Some(u) => hi_term = hi_term.mul(u),
                            None => hi_term = hi_term.mul(&Sym::atom(a.clone())),
                        }
                    } else {
                        hi_term = hi_term.mul(&l);
                        match &u {
                            Some(u) => lo_term = lo_term.mul(u),
                            None => lo_term = lo_term.mul(&Sym::atom(a.clone())),
                        }
                    }
                }
            }
            lo = lo.add(&lo_term);
            hi = hi.add(&hi_term);
        }
        Interval { lo, hi }
    }

    fn atom_bounds_over(&self, a: &Atom, bound: &dyn Fn(&Atom) -> bool) -> (Sym, Option<Sym>) {
        match a {
            Atom::Rem(_, d) => (Sym::constant(0), Some(d.sub(&Sym::constant(1)))),
            Atom::Quot(n, d) => {
                let inner = self.interval_over(n, bound);
                if d.as_constant().is_some() {
                    (inner.lo.quot(d), Some(inner.hi.quot(d)))
                } else {
                    (Sym::constant(0), Some(inner.hi))
                }
            }
            _ => (self.facts.lower_of(a), self.facts.upper_of(a)),
        }
    }

    #[allow(dead_code)]
    fn interval_old(&self, e: &Sym) -> Interval {
        // lo: substitute every atom with negative coefficient by its upper bound if any; positive by lower.
        // hi: the reverse. Nonlinear monomials are handled by treating each atom of the monomial the same way,
        // which is valid because all atoms are nonnegative.
        let mut lo = Sym::default();
        let mut hi = Sym::default();
        let mut unbounded_hi = false;
        for (m, c) in &e.terms {
            let mut lo_term = Sym::constant(*c);
            let mut hi_term = Sym::constant(*c);
            for (a, k) in m {
                let (l, u) = self.atom_bounds(a);
                for _ in 0..*k {
                    if *c >= 0 {
                        lo_term = lo_term.mul(&l);
                        match &u {
                            Some(u) => hi_term = hi_term.mul(u),
                            None => {
                                hi_term = hi_term.mul(&Sym::atom(a.clone()));
                                unbounded_hi = true;
                            }
                        }
                    } else {
                        hi_term = hi_term.mul(&l);
                        match &u {
                            Some(u) => lo_term = lo_term.mul(u),
                            None => lo_term = lo_term.mul(&Sym::atom(a.clone())),
                        }
                    }
                }
            }
            lo = lo.add(&lo_term);
            hi = hi.add(&hi_term);
        }
        let _ = unbounded_hi;
        Interval { lo, hi }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Sym {
        Sym::param(s)
    }

    #[test]
    fn normalization() {
        let m = p("M");
        let e = m.add(&m).sub(&m.scale(2));
        assert!(e.is_zero());
        assert_eq!(p("a").mul(&p("b")), p("b").mul(&p("a")));
        assert_eq!(p("M").scale(8).quot(&Sym::constant(8)), p("M"));
        assert_eq!(p("M").scale(8).rem(&Sym::constant(8)), Sym::constant(0));
        assert_eq!(
            p("M")
                .scale(8)
                .add(&Sym::constant(3))
                .quot(&Sym::constant(8))
                .to_string(),
            "M + (3 / 8)"
        );
    }

    #[test]
    fn tiled_index_stays_in_bounds() {
        // i in [0, M/8 - 1], k in [0, 7]: i*8 + k <= M - 1
        let mut facts = Facts::new();
        let q = p("M").quot(&Sym::constant(8));
        facts.set_range(
            Atom::Param("i".into()),
            Sym::constant(0),
            q.sub(&Sym::constant(1)),
        );
        facts.set_range(Atom::Param("k".into()), Sym::constant(0), Sym::constant(7));
        let prover = Prover::new(&facts);
        let idx = p("i").scale(8).add(&p("k"));
        let iv = prover.interval(&idx);
        assert!(prover.lt(&iv.hi, &p("M")), "hi = {}", iv.hi);
    }

    #[test]
    fn group_slice_stays_in_bounds() {
        // kv in [0, H/G - 1]: (kv + 1) * G <= H
        let mut facts = Facts::new();
        let q = p("H").quot(&p("G"));
        facts.set_range(
            Atom::Param("kv".into()),
            Sym::constant(0),
            q.sub(&Sym::constant(1)),
        );
        let prover = Prover::new(&facts);
        let end = p("kv").add(&Sym::constant(1)).mul(&p("G"));
        let iv = prover.interval(&end);
        assert!(prover.le(&iv.hi, &p("H")), "hi = {}", iv.hi);
    }

    #[test]
    fn divisibility_fact_closes_the_gap() {
        // Without M % 8 == 0, 8 * (M / 8) == M is unprovable; with it, it is.
        let mut facts = Facts::new();
        let e = p("M").quot(&Sym::constant(8)).scale(8).sub(&p("M"));
        assert!(!Prover::new(&facts).nonneg(&e));
        facts.assume_divisible(&p("M"), &Sym::constant(8));
        let prover = Prover::new(&facts);
        assert!(prover.nonneg(&e), "e = {e}");
        assert!(prover.nonneg(&e.neg()));
    }

    #[test]
    fn unprovable_is_reported() {
        let facts = Facts::new();
        let prover = Prover::new(&facts);
        assert!(!prover.lt(&p("i"), &p("M")));
    }
    #[test]
    fn direct_loop_bound_survives_cyclic_path_bounds() {
        let mut facts = Facts::new();
        let extent = p("P").scale(2).add(&p("S"));
        facts.set_range(
            Atom::Param("i".into()),
            Sym::constant(0),
            extent.sub(&Sym::constant(1)),
        );
        facts.add_lower(Atom::Param("i".into()), p("P"));
        facts.add_upper(Atom::Param("P".into()), p("i"));
        assert!(Prover::new(&facts).lt(&p("i"), &extent));
        assert!(!Prover::new(&facts).lt(&p("i"), &p("P")));
    }

    #[test]
    fn padded_byte_words_preserve_quotient_correlations() {
        let mut facts = Facts::new();
        facts.set_range_lower(Atom::Param("B".into()), Sym::constant(1));
        facts.set_range(
            Atom::Param("block".into()),
            Sym::constant(0),
            p("B").sub(&Sym::constant(1)),
        );
        facts.set_range(Atom::Param("g".into()), Sym::constant(0), Sym::constant(15));
        let divisor = Sym::constant(4);
        let extent = p("B").scale(210).add(&Sym::constant(3)).quot(&divisor);
        let at = p("block")
            .scale(210)
            .add(&Sym::constant(192))
            .add(&p("g"))
            .quot(&divisor);
        assert!(Prover::new(&facts).lt(&at, &extent));
        let invalid = p("block")
            .scale(210)
            .add(&Sym::constant(210))
            .add(&p("g"))
            .quot(&divisor);
        assert!(!Prover::new(&facts).lt(&invalid, &extent));
    }

    #[test]
    fn checked_extent_intervals_bound_tails_and_reject_undefined_arithmetic() {
        let n = p("N");
        let group = Sym::constant(32);
        let groups = n.quot(&group);
        let tail = n.rem(&group);
        let bounds = |name: &str| (name == "N").then_some((0, 129));
        assert_eq!(groups.eval_interval(&bounds), Some((0, 4)));
        assert_eq!(tail.eval_interval(&bounds), Some((0, 31)));
        for expression in [groups, tail, Sym::constant(129).sub(&n), n.mul(&n)] {
            let (lo, hi) = expression.eval_interval(&bounds).unwrap();
            for value in 0..=129 {
                let actual = expression.eval(&|_| Some(value)).unwrap();
                assert!(lo <= actual && actual <= hi);
            }
        }
        assert_eq!(n.scale(i64::MAX).eval_interval(&bounds), None);
        assert_eq!(Sym::constant(32).quot(&n).eval_interval(&bounds), None);
        assert_eq!(n.eval_interval(&|_| None), None);
    }

    #[test]
    fn remainder_enclosures_preserve_integer_packet_divisibility() {
        for multiplier in 1..=32i64 {
            for modulus in 1..=65i64 {
                let expression = p("i").scale(multiplier).rem(&Sym::constant(modulus));
                let (lo, hi) = expression.eval_interval(&|_| Some((0, 255))).unwrap();
                for i in 0..=255 {
                    assert!((lo..=hi).contains(&((i * multiplier) % modulus)));
                }
            }
        }
        let offset = p("i").scale(16).rem(&Sym::constant(64));
        assert_eq!(offset.eval_interval(&|_| Some((0, 255))), Some((0, 48)));
        assert_eq!(
            offset
                .scale(4)
                .quot(&Sym::constant(32))
                .eval_interval(&|_| Some((0, 255))),
            Some((0, 6))
        );
    }
}
