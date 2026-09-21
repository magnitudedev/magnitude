//! Exact and bounded specialization domains (package W1).
//!
//! `SpecializationDomain::new` is the only constructor of a compiler entry
//! specialization. It validates a complete binding for every entry shape and
//! element parameter before logical construction. There is no compiler entry
//! accepting a raw shape integer map.
//!
//! Exact bindings normalize to `ExtentExpr::Static`. Bounded bindings
//! normalize to one retained invocation-sourced runtime extent
//! (`RuntimeScalarExpr::ShapeField`) carrying `max` as capacity and
//! `expected` as cost weight; the actual value is an invocation shape field.
//! `expected` never affects semantics, geometry, or resources.
//!
//! # Domain arithmetic
//!
//! A checked shape expression is a [`Sym`]: a normalized polynomial with `i64`
//! coefficients over atoms, where an atom is a shape parameter, a floor
//! quotient, or a euclidean remainder. Every `Sym` constructor the checker
//! uses (`constant`, `param`, `add`, `sub`, `neg`, `scale`, `mul`, `quot`,
//! `rem`) produces that normal form, so the domain transfer is defined
//! exhaustively over the normal form rather than over a syntax tree:
//!
//! | `Sym` operation | normal-form shape | domain-transfer rule ([`enclosure`]) |
//! |---|---|---|
//! | `constant(c)` | empty monomial with coefficient `c` | `[c, c]` |
//! | `param(N)` | atom `Param(N)` | `[min, max]` of the binding (`[v, v]` for `Exact(v)`); a name the domain does not bind is [`IntervalFailure::ForeignSymbol`] |
//! | `add`, `sub`, `neg` | signed sum of monomials | interval sum of the monomial enclosures (`sub`/`neg` are folded into coefficient signs) |
//! | `scale(c)`, `mul` | monomial `c * a1^k1 * ... * an^kn` | corner products of `[c, c]` with each atom enclosure `k` times |
//! | `quot(n, d)` | atom `Quot(n, d)` | floor division over the positive part of the divisor's enclosure; the extremes of a function monotone in each argument lie on corners |
//! | `rem(n, d)` | atom `Rem(n, d)` | exact `[nl mod d, nh mod d]` when the divisor is constant and the quotient does not change over the domain; otherwise `[0, min(nh, dh - 1, d - gcd(k, d))]` where `k` divides every value of `n` |
//!
//! Partial arithmetic never widens silently: a divisor whose enclosure
//! reaches a non-positive value marks the enclosure `partial` (defined on
//! part of the domain only), a divisor that is never positive makes the
//! expression [`Enclosure::Undefined`], and every intermediate is computed
//! with checked `i128` arithmetic ([`IntervalFailure::Unrepresentable`] on
//! overflow). No value is ever substituted by a capacity.
//!
//! The enclosure is sound and incomplete: it contains every value the
//! expression takes on the domain, but correlated occurrences of one
//! parameter (`N * N - 2 * N + 1`) are enclosed independently, so a verdict
//! may be `Mixed` where the predicate is in fact always true. A `Mixed`
//! verdict is always conservative: it never admits an alternative.

use crate::sir::{Predicate, Program};
use crate::sym::{Atom, Sym};
use crate::types::Elem;
use std::collections::BTreeMap;
use std::fmt;

/// Identity of one invocation shape field: the actual runtime value of one
/// bounded entry shape parameter, validated once by `CompiledPlan::prepare`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShapeFieldId(pub u32);

/// How one entry shape parameter is bound for one compiled artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShapeBinding {
    Exact(u64),
    Bounded { min: u64, max: u64, expected: u64 },
}

impl ShapeBinding {
    /// The capacity which bounds storage, native resources, address
    /// arithmetic, and admissible mapping rules.
    pub fn capacity(self) -> u64 {
        match self {
            ShapeBinding::Exact(value) => value,
            ShapeBinding::Bounded { max, .. } => max,
        }
    }

    pub fn domain(self) -> ShapeDomain {
        match self {
            ShapeBinding::Exact(value) => ShapeDomain::Exact(value),
            ShapeBinding::Bounded { min, max, .. } => ShapeDomain::Bounded { min, max },
        }
    }
}

/// The admissible domain of one shape field, retained in the invocation
/// contract (package A1 consumes it; W1 owns it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShapeDomain {
    Exact(u64),
    Bounded { min: u64, max: u64 },
}

impl ShapeDomain {
    pub fn contains(self, value: u64) -> bool {
        match self {
            ShapeDomain::Exact(exact) => value == exact,
            ShapeDomain::Bounded { min, max } => min <= value && value <= max,
        }
    }

    pub fn min(self) -> u64 {
        match self {
            ShapeDomain::Exact(value) => value,
            ShapeDomain::Bounded { min, .. } => min,
        }
    }

    pub fn max(self) -> u64 {
        match self {
            ShapeDomain::Exact(value) => value,
            ShapeDomain::Bounded { max, .. } => max,
        }
    }
}

/// One bounded entry shape parameter retained by the logical program: its
/// interface name and finite domain. Exact parameters have no field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeField {
    pub id: ShapeFieldId,
    pub name: String,
    pub domain: ShapeDomain,
    pub expected: u64,
}

/// Why a specialization domain could not be constructed. Every variant is a
/// source/semantic diagnostic of the caller's request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecializationError {
    /// The entry declares a shape parameter the request does not bind.
    UnboundShape { name: String },
    /// The request binds a shape parameter the entry does not declare.
    UnknownShape { name: String },
    /// The entry declares an element parameter the request does not bind.
    UnboundElem { name: String },
    /// The request binds an element parameter the entry does not declare.
    UnknownElem { name: String },
    /// A bounded binding violates `min <= expected <= max`.
    InvalidBounds {
        name: String,
        min: u64,
        max: u64,
        expected: u64,
    },
    /// The entry family does not exist in the checked program.
    UnknownEntry { name: String },
    /// Several disjoint overload families share the entry name; a name-only
    /// request cannot select one.
    AmbiguousEntry { name: String },
    /// An element parameter is bound to another parameter instead of a
    /// concrete dtype or representation.
    NonConcreteElem { name: String, bound_to: String },
}

impl fmt::Display for SpecializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnboundShape { name } => write!(f, "shape parameter `{name}` is unbound"),
            Self::UnknownShape { name } => write!(f, "shape parameter `{name}` is not declared"),
            Self::UnboundElem { name } => write!(f, "element parameter `{name}` is unbound"),
            Self::UnknownElem { name } => write!(f, "element parameter `{name}` is not declared"),
            Self::InvalidBounds {
                name,
                min,
                max,
                expected,
            } => write!(
                f,
                "shape parameter `{name}` bounds violate min <= expected <= max: {min} <= {expected} <= {max}"
            ),
            Self::UnknownEntry { name } => write!(f, "entry `{name}` is not a checked family"),
            Self::AmbiguousEntry { name } => write!(
                f,
                "entry `{name}` names several disjoint overload families; a name-only request cannot select one"
            ),
            Self::NonConcreteElem { name, bound_to } => write!(
                f,
                "element parameter `{name}` is bound to parameter `{bound_to}`, not a concrete element"
            ),
        }
    }
}

impl std::error::Error for SpecializationError {}

/// A complete, validated binding of every entry shape and element parameter.
/// Part of the logical identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecializationDomain {
    entry: String,
    shapes: BTreeMap<String, ShapeBinding>,
    elems: BTreeMap<String, Elem>,
    /// The entry contract's shape parameters in declared order. `shapes`
    /// binds exactly this set; this vector only fixes the order in which
    /// bounded parameters become shape fields.
    shape_order: Vec<String>,
}

impl SpecializationDomain {
    /// The only constructor. Validates against the checked entry family:
    /// every declared shape/element parameter is bound exactly once, no
    /// undeclared parameter is bound, and every bounded domain satisfies
    /// `min <= expected <= max`. Zero is retained when the domain admits it;
    /// it is not globally forbidden here (the source operation's own
    /// predicates decide).
    ///
    /// Every alternative of a family shares the family's contract, so the
    /// contract definition's declared parameter lists are the completeness
    /// authority for the whole family. "Bound exactly once" holds by the
    /// map type (one key per name) together with the two inclusion checks.
    pub fn new(
        program: &Program,
        entry: &str,
        shapes: BTreeMap<String, ShapeBinding>,
        elems: BTreeMap<String, Elem>,
    ) -> Result<SpecializationDomain, SpecializationError> {
        let mut candidates = program
            .families
            .iter()
            .filter(|family| family.name == entry);
        let family = match (candidates.next(), candidates.next()) {
            (None, _) => {
                return Err(SpecializationError::UnknownEntry {
                    name: entry.to_string(),
                })
            }
            (Some(_), Some(_)) => {
                return Err(SpecializationError::AmbiguousEntry {
                    name: entry.to_string(),
                })
            }
            (Some(family), None) => family,
        };
        let contract = program.definition(family.contract);

        for name in &contract.shape_params {
            if !shapes.contains_key(name) {
                return Err(SpecializationError::UnboundShape { name: name.clone() });
            }
        }
        for name in shapes.keys() {
            if !contract.shape_params.contains(name) {
                return Err(SpecializationError::UnknownShape { name: name.clone() });
            }
        }
        for name in &contract.elem_params {
            if !elems.contains_key(name) {
                return Err(SpecializationError::UnboundElem { name: name.clone() });
            }
        }
        for (name, elem) in &elems {
            if !contract.elem_params.contains(name) {
                return Err(SpecializationError::UnknownElem { name: name.clone() });
            }
            match elem {
                Elem::Dtype(_) | Elem::Repr(_) => {}
                Elem::Param(bound_to) => {
                    return Err(SpecializationError::NonConcreteElem {
                        name: name.clone(),
                        bound_to: bound_to.clone(),
                    })
                }
            }
        }
        for (name, binding) in &shapes {
            match *binding {
                ShapeBinding::Exact(_) => {}
                ShapeBinding::Bounded { min, max, expected } => {
                    if !(min <= expected && expected <= max) {
                        return Err(SpecializationError::InvalidBounds {
                            name: name.clone(),
                            min,
                            max,
                            expected,
                        });
                    }
                }
            }
        }

        Ok(SpecializationDomain {
            entry: entry.to_string(),
            shapes,
            elems,
            shape_order: contract.shape_params.clone(),
        })
    }

    pub fn entry(&self) -> &str {
        &self.entry
    }

    pub fn shapes(&self) -> &BTreeMap<String, ShapeBinding> {
        &self.shapes
    }

    pub fn elems(&self) -> &BTreeMap<String, Elem> {
        &self.elems
    }

    /// The binding of one declared shape parameter. Total after validation:
    /// asking for a name the entry does not declare is a compiler defect.
    pub fn binding(&self, name: &str) -> ShapeBinding {
        match self.shapes.get(name) {
            Some(binding) => *binding,
            None => panic!(
                "compiler defect (W1): shape parameter `{name}` is not declared by entry `{}`",
                self.entry
            ),
        }
    }

    /// The concrete element of one declared element parameter. Total after
    /// validation: asking for a name the entry does not declare is a
    /// compiler defect.
    pub fn elem(&self, name: &str) -> &Elem {
        match self.elems.get(name) {
            Some(elem) => elem,
            None => panic!(
                "compiler defect (W1): element parameter `{name}` is not declared by entry `{}`",
                self.entry
            ),
        }
    }

    /// The bounded parameters `(name, min, max, expected)` in the entry's
    /// declared parameter order.
    fn bounded_parameters(&self) -> impl Iterator<Item = (&str, u64, u64, u64)> {
        self.shape_order
            .iter()
            .filter_map(move |name| match self.binding(name) {
                ShapeBinding::Exact(_) => None,
                ShapeBinding::Bounded { min, max, expected } => {
                    Some((name.as_str(), min, max, expected))
                }
            })
    }

    /// The bounded parameters in the entry's declared parameter order. The
    /// position of a parameter in this iteration is its `ShapeFieldId`.
    pub fn bounded(&self) -> impl Iterator<Item = (&str, ShapeBinding)> {
        self.bounded_parameters()
            .map(|(name, min, max, expected)| (name, ShapeBinding::Bounded { min, max, expected }))
    }

    /// The retained shape fields: one per bounded parameter, identified by
    /// its position in [`SpecializationDomain::bounded`].
    pub fn shape_fields(&self) -> Vec<ShapeField> {
        self.bounded_parameters()
            .enumerate()
            .map(|(index, (name, min, max, expected))| ShapeField {
                id: ShapeFieldId(index as u32),
                name: name.to_string(),
                domain: ShapeDomain::Bounded { min, max },
                expected,
            })
            .collect()
    }

    /// Canonical, self-delimiting encoding for the logical identity hash:
    /// the entry, then every `(name, binding)` in name order, then every
    /// `(name, elem)` in name order. `expected` is part of the identity
    /// because planning prices the domain with it, so two requests that
    /// differ only in `expected` produce different artifacts.
    pub fn identity_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_str(&mut out, &self.entry);
        put_u64(&mut out, self.shapes.len() as u64);
        for (name, binding) in &self.shapes {
            put_str(&mut out, name);
            match *binding {
                ShapeBinding::Exact(value) => {
                    out.push(0);
                    put_u64(&mut out, value);
                }
                ShapeBinding::Bounded { min, max, expected } => {
                    out.push(1);
                    put_u64(&mut out, min);
                    put_u64(&mut out, max);
                    put_u64(&mut out, expected);
                }
            }
        }
        put_u64(&mut out, self.elems.len() as u64);
        for (name, elem) in &self.elems {
            put_str(&mut out, name);
            match elem {
                Elem::Dtype(dtype) => {
                    out.push(0);
                    put_str(&mut out, dtype.name());
                }
                Elem::Repr(repr) => {
                    out.push(1);
                    put_str(&mut out, repr);
                }
                Elem::Param(param) => {
                    // Rejected by `new`; a validated domain has no such binding.
                    panic!(
                        "compiler defect (W1): validated domain binds element parameter to `{param}`"
                    )
                }
            }
        }
        out
    }
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, text: &str) {
    put_u64(out, text.len() as u64);
    out.extend_from_slice(text.as_bytes());
}

/// The verdict of one source implementation predicate over a domain. An
/// alternative is admissible only under `Always`; `Mixed` names a predicate
/// the caller must partition its workload envelope on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PredicateVerdict {
    Always,
    Never,
    /// True on part of the domain only. The alternative is inadmissible for
    /// this domain; a finite disjoint partition of classes is required.
    Mixed,
}

// ---------------------------------------------------------------------------
// Domain arithmetic
// ---------------------------------------------------------------------------

/// The values a checked shape expression takes over a domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Enclosure {
    /// The expression is defined on at least one point of the domain, and
    /// `[low, high]` contains its value at every point where it is defined.
    Defined {
        low: i128,
        high: i128,
        /// A quotient or remainder divisor reaches a non-positive value on
        /// part of the domain: the expression is undefined there.
        partial: bool,
    },
    /// A quotient or remainder divisor is never positive: the expression is
    /// undefined on every point of the domain.
    Undefined,
}

/// Why a checked shape expression has no enclosure over a domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntervalFailure {
    /// The expression mentions a symbol the domain does not bind: a shape
    /// parameter of another scope or a loop-scoped symbol.
    ForeignSymbol { name: String },
    /// An intermediate value does not fit the `i128` evaluation width.
    Unrepresentable,
}

impl fmt::Display for IntervalFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignSymbol { name } => {
                write!(f, "symbol `{name}` is not a shape parameter of the specialization domain")
            }
            Self::Unrepresentable => {
                write!(f, "the expression exceeds the 128-bit evaluation width over the domain")
            }
        }
    }
}

/// The enclosure of a checked shape expression over a domain, by the
/// domain-transfer rules in the module documentation.
pub fn enclosure(sym: &Sym, domain: &SpecializationDomain) -> Result<Enclosure, IntervalFailure> {
    let mut low: i128 = 0;
    let mut high: i128 = 0;
    let mut partial = false;
    for (monomial, coefficient) in sym.monomials() {
        // `scale`/`mul`: the coefficient times each atom enclosure, power times.
        let (mut term_low, mut term_high) = (i128::from(coefficient), i128::from(coefficient));
        for (atom, power) in monomial {
            let (atom_low, atom_high) = match atom_enclosure(atom, domain)? {
                Enclosure::Undefined => return Ok(Enclosure::Undefined),
                Enclosure::Defined {
                    low: atom_low,
                    high: atom_high,
                    partial: atom_partial,
                } => {
                    partial |= atom_partial;
                    (atom_low, atom_high)
                }
            };
            for _ in 0..*power {
                (term_low, term_high) = mul_interval((term_low, term_high), (atom_low, atom_high))?;
            }
        }
        // `add`/`sub`/`neg`: the signed sum of monomials.
        low = low.checked_add(term_low).ok_or(IntervalFailure::Unrepresentable)?;
        high = high.checked_add(term_high).ok_or(IntervalFailure::Unrepresentable)?;
    }
    Ok(Enclosure::Defined { low, high, partial })
}

/// Corner-product enclosure of the product of two intervals.
fn mul_interval(a: (i128, i128), b: (i128, i128)) -> Result<(i128, i128), IntervalFailure> {
    let corners = [
        a.0.checked_mul(b.0),
        a.0.checked_mul(b.1),
        a.1.checked_mul(b.0),
        a.1.checked_mul(b.1),
    ];
    let mut low = i128::MAX;
    let mut high = i128::MIN;
    for corner in corners {
        let value = corner.ok_or(IntervalFailure::Unrepresentable)?;
        low = low.min(value);
        high = high.max(value);
    }
    Ok((low, high))
}

fn atom_enclosure(atom: &Atom, domain: &SpecializationDomain) -> Result<Enclosure, IntervalFailure> {
    match atom {
        // `param`: the binding's finite domain.
        Atom::Param(name) => match domain.shapes.get(name) {
            Some(binding) => {
                let shape = binding.domain();
                Ok(Enclosure::Defined {
                    low: i128::from(shape.min()),
                    high: i128::from(shape.max()),
                    partial: false,
                })
            }
            None => Err(IntervalFailure::ForeignSymbol { name: name.clone() }),
        },
        Atom::Quot(numerator, divisor) | Atom::Rem(numerator, divisor) => {
            let is_quotient = matches!(atom, Atom::Quot(..));
            let (numerator_low, numerator_high, numerator_partial) =
                match enclosure(numerator, domain)? {
                    Enclosure::Undefined => return Ok(Enclosure::Undefined),
                    Enclosure::Defined { low, high, partial } => (low, high, partial),
                };
            let (divisor_low, divisor_high, divisor_partial) = match enclosure(divisor, domain)? {
                Enclosure::Undefined => return Ok(Enclosure::Undefined),
                Enclosure::Defined { low, high, partial } => (low, high, partial),
            };
            // Floor quotient and euclidean remainder are defined only for a
            // positive divisor. A divisor that is never positive makes the
            // atom undefined everywhere; one that is non-positive somewhere
            // makes it partial, and the enclosure below covers the positive
            // part of the divisor's domain only.
            if divisor_high <= 0 {
                return Ok(Enclosure::Undefined);
            }
            let partial = numerator_partial || divisor_partial || divisor_low <= 0;
            let divisor_low = divisor_low.max(1);
            let (low, high) = if is_quotient {
                // `quot`: floor(n / d) is monotone in `n` for fixed `d` and
                // monotone in `d` for fixed `n`, so its extremes over the box
                // lie on the corners.
                (
                    numerator_low
                        .div_euclid(divisor_low)
                        .min(numerator_low.div_euclid(divisor_high)),
                    numerator_high
                        .div_euclid(divisor_low)
                        .max(numerator_high.div_euclid(divisor_high)),
                )
            } else {
                // `rem`: exact when the divisor is one constant and the
                // quotient is the same at both ends of the numerator (the
                // remainder is then monotone over the numerator's range);
                // otherwise every remainder lies in `[0, d - 1]`, below a
                // nonnegative numerator, and is a multiple of `gcd(k, d)`
                // for the integer `k` dividing every value of the numerator.
                let constant_divisor = divisor_low == divisor_high;
                if constant_divisor
                    && numerator_low.div_euclid(divisor_low)
                        == numerator_high.div_euclid(divisor_low)
                {
                    (
                        numerator_low.rem_euclid(divisor_low),
                        numerator_high.rem_euclid(divisor_low),
                    )
                } else {
                    let mut high = divisor_high - 1;
                    if numerator_low >= 0 {
                        high = high.min(numerator_high);
                    }
                    if constant_divisor {
                        let step = gcd(i128::from(numerator.coefficient_divisor()), divisor_low);
                        high = high.min(divisor_low - step);
                    }
                    (0, high)
                }
            };
            Ok(Enclosure::Defined { low, high, partial })
        }
    }
}

/// Greatest common divisor of two nonnegative values; `gcd(0, d) == d`.
fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Finite interval of a checked symbolic shape expression under the domain.
/// Taking the minimum and maximum over every parameter's corners is not
/// sufficient for non-monotone expressions; this uses the interval
/// arithmetic of [`enclosure`] over per-atom intervals. `None` when the
/// expression mentions a symbol outside the domain, when the arithmetic is
/// undefined anywhere in the domain, when an intermediate is unrepresentable,
/// or when the expression is not a `u64` extent (negative or above
/// `u64::MAX`) somewhere in the domain. No case is ever widened or clamped.
pub fn shape_interval(sym: &Sym, domain: &SpecializationDomain) -> Option<(u64, u64)> {
    match enclosure(sym, domain) {
        Ok(Enclosure::Defined {
            low,
            high,
            partial: false,
        }) => match (u64::try_from(low), u64::try_from(high)) {
            (Ok(low), Ok(high)) => Some((low, high)),
            (Err(_), _) | (_, Err(_)) => None,
        },
        Ok(Enclosure::Defined { partial: true, .. })
        | Ok(Enclosure::Undefined)
        | Err(IntervalFailure::ForeignSymbol { .. })
        | Err(IntervalFailure::Unrepresentable) => None,
    }
}

/// Verdict of one implementation predicate over the whole domain.
///
/// `Always` only when the predicate holds at every point of the domain,
/// `Never` when it holds at no point, otherwise `Mixed`. The predicate's
/// expression must already be over the domain's shape parameter names (the
/// caller substitutes an occurrence's shape arguments first, exactly as
/// `family::applicable` does).
///
/// - A point where the expression is undefined (non-positive divisor) never
///   satisfies a predicate, so a `partial` enclosure is never `Always`; it
///   is `Never` when the defined part is `Never`, else `Mixed`. An
///   expression undefined everywhere is `Never` for all three predicates.
/// - Divisibility `Zero(N % c)` over a bounded `N` is `Always` only when the
///   remainder enclosure is exactly `[0, 0]`, which by the `rem` rule
///   requires either a numerator constant over the domain or a single-valued
///   divisor dividing every value of the numerator; otherwise it is `Mixed`
///   unless `0` is excluded from the enclosure.
/// - An expression that is not decidable over the domain, because it
///   mentions a foreign symbol or exceeds the evaluation width, is `Mixed`:
///   inadmissible, never silently admitted (the current
///   `family::predicates_hold` rejects such predicates as undecidable).
pub fn predicate_verdict(predicate: &Predicate, domain: &SpecializationDomain) -> PredicateVerdict {
    let (sym, test) = match predicate {
        Predicate::NonNegative(sym) => (sym, Test::NonNegative),
        Predicate::Zero(sym) => (sym, Test::Zero),
        Predicate::NonZero(sym) => (sym, Test::NonZero),
    };
    match enclosure(sym, domain) {
        Err(IntervalFailure::ForeignSymbol { .. }) | Err(IntervalFailure::Unrepresentable) => {
            PredicateVerdict::Mixed
        }
        Ok(Enclosure::Undefined) => PredicateVerdict::Never,
        Ok(Enclosure::Defined { low, high, partial }) => {
            let (always, never) = match test {
                Test::NonNegative => (low >= 0, high < 0),
                Test::Zero => (low == 0 && high == 0, low > 0 || high < 0),
                Test::NonZero => (low > 0 || high < 0, low == 0 && high == 0),
            };
            if never {
                PredicateVerdict::Never
            } else if always && !partial {
                PredicateVerdict::Always
            } else {
                PredicateVerdict::Mixed
            }
        }
    }
}

/// The three source predicate tests over an enclosure.
#[derive(Clone, Copy)]
enum Test {
    NonNegative,
    Zero,
    NonZero,
}
