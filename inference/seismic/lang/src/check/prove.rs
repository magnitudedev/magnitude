//! The checker's single private arithmetic derivation owner (spec §24.5 R15).
//!
//! Every symbolic integer of the checker is an `IntExpr` in the definition's
//! arena. A proof normalizes the nodes it inspects into a transient
//! polynomial form over symbol atoms (with floor quotients and remainders as
//! opaque atoms), rewrites under the facts, and discards the form when it
//! returns. The form is never stored on a checked datum, never crosses a
//! module boundary, and every value the checker keeps is re-interned into the
//! arena through [`intern`].
//!
//! Dimensions and Index quantities carry their declared bounds. Source words
//! carry their dtype bounds independently of exact operation meaning. The owner
//! establishes `e >= 0` and `e == 0` under scoped premises by
//! rewriting quotients and remainders through `x = c * (x / c) + x % c` and
//! bounding atoms that carry an upper bound. It is sound and incomplete: a
//! unresolved derivation is a residual for the consumer's static/runtime policy,
//! never a refutation. Source value construction does not invoke this search.

use crate::expr::poly::{intern, normalize_int, single_atom_of, Atom, Monomial, Poly};
use crate::expr::{
    AnyExpr, ArenaId, BinaryOp, BoolExpr, CmpOp, ExprArena, IntExpr, NaryOp, NodeView, ScalarOperand, SymbolId,
};
use crate::types::{DType, ValueType};
use std::cell::{OnceCell, RefCell};
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

/// Exact reached scalar meaning. Unchanged facts share an arena-bound DAG memo;
/// new facts or successful proof evidence invalidate reached reductions.
pub(crate) fn canonical_with_facts(arena: &mut ExprArena, facts: &Facts, e: IntExpr) -> IntExpr {
    canonical_many_with_facts(arena, facts, &[e])[0]
}

pub(crate) fn canonical_many_with_facts(
    arena: &mut ExprArena,
    facts: &Facts,
    expressions: &[IntExpr],
) -> Vec<IntExpr> {
    let mut normal = ReachedScalarNormalForm::new(arena, facts);
    expressions.iter().map(|e| normal.value(*e).0).collect()
}

/// One fact world's exact equalities: the one owner shared by proof and
/// reached scalar normalization. What an equation implies about one quotient
/// or remainder atom depends on that atom and the equations alone, so each
/// is derived once per world.
#[derive(Debug)]
struct ExactZeros {
    /// The arena whose normal forms the equations are.
    arena: ArenaId,
    zeros: Rc<Vec<Poly>>,
    zero_atoms: Rc<HashSet<Atom>>,
    /// Quotient atom -> its quotient/remainder-free exact replacement.
    quotients: RefCell<HashMap<Atom, Option<Poly>>>,
    /// Remainder atom -> whether an equation establishes it is zero.
    remainders: RefCell<HashMap<Atom, bool>>,
    /// The equations' pivot rows, reduced once on first use.
    span: OnceCell<BTreeMap<Monomial, Poly>>,
}

impl ExactZeros {
    fn new(arena: ArenaId, zeros: Vec<Poly>) -> ExactZeros {
        let zero_atoms = zeros.iter().filter_map(single_atom_of).collect();
        ExactZeros {
            arena,
            zeros: Rc::new(zeros),
            zero_atoms: Rc::new(zero_atoms),
            quotients: RefCell::default(),
            remainders: RefCell::default(),
            span: OnceCell::new(),
        }
    }

    /// Whether `apply` can change a monomial containing `atom`.
    fn active(&self, atom: &Atom) -> bool {
        if self.zero_atoms.contains(atom) {
            return true;
        }
        if self.zeros.is_empty() {
            return false;
        }
        match atom {
            Atom::Quot(numerator, denominator) => {
                denominator.as_constant().is_some_and(|d| d > 0)
                    && self.quotient(atom, numerator, denominator).is_some()
            }
            Atom::Rem(numerator, denominator) => denominator
                .as_constant()
                .filter(|d| *d > 0)
                .is_some_and(|divisor| self.zero_remainder(atom, numerator, divisor)),
            Atom::Symbol(_) | Atom::Foreign(_) => false,
        }
    }

    /// `goal == 0` is a linear combination of the equations.
    fn spans(&self, goal: Poly) -> bool {
        in_zero_span(goal, self.span.get_or_init(|| zero_span_basis(&self.zeros)))
    }

    /// Exact stored equations may expose an integral quotient: one
    /// quotient/remainder-free replacement, never a new division.
    fn quotient(&self, atom: &Atom, numerator: &Poly, denominator: &Poly) -> Option<Poly> {
        if let Some(known) = self.quotients.borrow().get(atom) {
            return known.clone();
        }
        let candidate = self
            .zeros
            .iter()
            .flat_map(|zero| [numerator.add(zero), numerator.sub(zero)])
            .find_map(|equivalent| {
                let quotient = equivalent.divide_exactly(denominator)?;
                (quotient.valid
                    && quotient
                        .atoms()
                        .iter()
                        .all(|atom| !matches!(atom, Atom::Quot(..) | Atom::Rem(..))))
                .then_some(quotient)
            });
        self.quotients
            .borrow_mut()
            .insert(atom.clone(), candidate.clone());
        candidate
    }

    /// A known multiple of a positive constant is also a multiple of every
    /// positive divisor of that constant. Require identical exact numerators;
    /// unrelated remainders contribute no facts.
    fn zero_remainder(&self, atom: &Atom, n: &Poly, divisor: i64) -> bool {
        if let Some(known) = self.remainders.borrow().get(atom) {
            return *known;
        }
        let zeros = &self.zeros;
        let implied = zeros.iter().any(|zero| {
            let Some(Atom::Rem(known_n, known_d)) = single_atom_of(zero) else {
                return false;
            };
            let Some(multiple) = known_d.as_constant().filter(|value| *value > 0) else {
                return false;
            };
            if multiple % divisor != 0 {
                return false;
            }
            if *known_n == *n {
                return true;
            }
            // For example W=16*C and W%32=0 imply C%2=0.
            // Cancellation uses only an established exact zero relation.
            let factor = Poly::constant(multiple / divisor);
            let difference = known_n.sub(&n.mul(&factor));
            difference.valid
                && (difference.is_zero()
                    || zeros.contains(&difference)
                    || zeros.contains(&difference.neg()))
        });
        self.remainders.borrow_mut().insert(atom.clone(), implied);
        implied
    }

    fn apply(&self, e: Poly) -> Poly {
        let mut e = e;
        // Substituting zero for an opaque atom removes precisely the
        // monomials containing it. Do this once, rather than rebuilding every
        // polynomial for every zero fact, including atoms absent from it.
        // A symbol inside a quotient or remainder is still a different atom.
        if !e.valid && !self.zero_atoms.is_empty() {
            e.terms.clear();
        } else {
            e.terms
                .retain(|monomial, _| !monomial.keys().any(|atom| self.zero_atoms.contains(atom)));
        }
        if self.zeros.is_empty() {
            return e;
        }
        for atom in e.atoms() {
            let Atom::Quot(numerator, denominator) = &atom else {
                continue;
            };
            if denominator.as_constant().is_none_or(|d| d <= 0) {
                continue;
            }
            if let Some(candidate) = self.quotient(&atom, numerator, denominator) {
                let substituted = e.subst(&atom, &candidate);
                if substituted.valid {
                    e = substituted;
                }
            }
        }
        for atom in e.atoms() {
            let Atom::Rem(n, d) = &atom else {
                continue;
            };
            let Some(divisor) = d.as_constant().filter(|value| *value > 0) else {
                continue;
            };
            if self.zero_remainder(&atom, n, divisor) {
                if e.valid {
                    e.terms.retain(|monomial, _| !monomial.contains_key(&atom));
                } else {
                    e.terms.clear();
                }
            }
        }
        e
    }
}

struct ReachedScalarNormalForm<'a> {
    arena: &'a mut ExprArena,
    facts: &'a Facts,
    world: ReachedScalarWorld,
    required: bool,
}

#[derive(Clone, Debug)]
struct ReachedScalarWorld {
    exact: Rc<ExactZeros>,
    memo: Rc<RefCell<HashMap<IntExpr, (IntExpr, bool)>>>,
}

#[cfg(test)]
#[test]
fn reached_scalar_cache_separates_fact_worlds_and_arenas() {
    let mut arena = ExprArena::new();
    let symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
    let x = arena.int_symbol(symbol);
    let zero = arena.int(0);
    let minimum = arena.int_min(x, zero);
    let facts = Facts::new();
    let original = canonical_with_facts(&mut arena, &facts, minimum);
    let unchanged = facts.clone();
    assert_eq!(canonical_with_facts(&mut arena, &unchanged, minimum), original);
    assert!(Rc::ptr_eq(&facts.caches(), &unchanged.caches()));
    let mut positive = facts.clone();
    positive.set_range_lower(symbol, zero);
    assert_eq!(canonical_with_facts(&mut arena, &positive, minimum), zero);
    assert_eq!(canonical_with_facts(&mut arena, &facts, minimum), original);
    assert!(!same(&arena, original, zero));

    let mut other = ExprArena::new();
    let other_symbol = other.checker_variable(crate::expr::SymbolSort::Int);
    let other_x = other.int_symbol(other_symbol);
    let other_zero = other.int(0);
    let other_minimum = other.int_min(other_x, other_zero);
    let other_result = canonical_with_facts(&mut other, &facts, other_minimum);
    assert!(same(&other, other_result, other_minimum));
    assert_eq!(facts.caches().scalar_cache.borrow().len(), 2);
}

#[cfg(test)]
#[test]
fn learned_proof_revisits_cached_reached_minimum() {
    let mut arena = ExprArena::new();
    let symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
    let n = arena.int_symbol(symbol);
    let zero = arena.int(0);
    let eight = arena.int(8);
    let quotient = arena.int_div(n, eight);
    let multiple = arena.int_mul(eight, quotient);
    let remainder = arena.int_sub(n, multiple);
    let minimum = arena.int_min(remainder, zero);
    let mut facts = Facts::new();
    facts.set_range_lower(symbol, zero);
    assert!(!established_nonnegative(&arena, &facts, remainder));
    let before = canonical_with_facts(&mut arena, &facts, minimum);
    assert!(!same(&arena, before, zero));
    assert!(nonneg(&mut arena, &facts, remainder));
    assert_eq!(canonical_with_facts(&mut arena, &facts, minimum), zero);
    // A failed proof neither introduces a sign nor removes the new reduction.
    let excessive = arena.int_sub(remainder, eight);
    assert!(!nonneg(&mut arena, &facts, excessive));
    assert_eq!(canonical_with_facts(&mut arena, &facts, minimum), zero);
}

impl<'a> ReachedScalarNormalForm<'a> {
    fn new(arena: &'a mut ExprArena, facts: &'a Facts) -> Self {
        // The constant handle includes the arena's identity. In particular,
        // empty facts reused with two arenas cannot share reached node results.
        let owner = arena.int(0);
        let existing = facts.caches().scalar_cache.borrow().get(&owner).cloned();
        let world = existing.unwrap_or_else(|| {
            let world = ReachedScalarWorld {
                exact: Engine::new(arena, facts).exact,
                memo: Rc::default(),
            };
            facts.caches().scalar_cache.borrow_mut().insert(owner, world.clone());
            world
        });
        Self {
            arena,
            facts,
            world,
            required: false,
        }
    }

    fn for_obligation(arena: &'a mut ExprArena, facts: &'a Facts) -> Self {
        let mut normal = Self::new(arena, facts);
        // Mandatory candidate proofs can learn signs while this traversal runs.
        // Do not publish a partially reached traversal into the optional cache.
        normal.world.memo = Rc::default();
        normal.required = true;
        normal
    }

    fn nonnegative(&mut self, value: IntExpr) -> bool {
        if self.required { nonneg(self.arena, self.facts, value) }
        else { established_nonnegative(self.arena, self.facts, value) }
    }

    fn word_range(&mut self, value: IntExpr, dtype: DType) -> bool {
        let (lo, hi) = match dtype {
            DType::I32 => (i64::from(i32::MIN), i64::from(i32::MAX)),
            DType::U32 => (0, i64::from(u32::MAX)),
            _ => return false,
        };
        let lower = self.arena.int(lo);
        let upper = self.arena.int(hi);
        let from_lower = self.arena.int_sub(value, lower);
        let to_upper = self.arena.int_sub(upper, value);
        self.nonnegative(from_lower) && self.nonnegative(to_upper)
    }

    fn typed_value(&mut self, original: IntExpr, operation: crate::reference_math::ScalarOp,
        operands: &[ScalarOperand]) -> (IntExpr, bool)
    {
        use crate::reference_math::ScalarOp;
        use crate::intrinsics::MathOp;
        use crate::syntax::ast::{BinaryOp as SourceBinary, UnaryOp as SourceUnary};
        let mut reached = Vec::with_capacity(operands.len());
        let mut children_defined = true;
        for operand in operands {
            match *operand {
                ScalarOperand::Word { dtype, value } => {
                    let (value, defined) = self.value(value);
                    children_defined &= defined;
                    reached.push(ScalarOperand::Word { dtype, value });
                }
                ScalarOperand::Scalar(value) => {
                    children_defined &= self.arena.is_total(AnyExpr::Scalar(value));
                    reached.push(ScalarOperand::Scalar(value));
                }
            }
        }
        let defined = self.arena.scalar_integer_defined(original);
        let continued = self.facts.boolean(self.arena, defined);
        if !(self.arena.is_total(original.into()) || continued) || !(children_defined || continued) {
            return (original, false);
        }
        let types = reached.iter().map(|operand| operand.dtype(self.arena)).collect::<Vec<_>>();
        let dtype = crate::reference_math::scalar_recipe(operation, &types).output().ty();
        let words = reached.iter().map(|operand| match operand {
            ScalarOperand::Word { value, .. } => Some(*value),
            ScalarOperand::Scalar(_) => None,
        }).collect::<Option<Vec<_>>>();
        let operands_in_range = reached.iter().all(|operand| match *operand {
            ScalarOperand::Word { dtype, value } => self.word_range(value, dtype),
            ScalarOperand::Scalar(_) => false,
        });
        if !operands_in_range { return (self.arena.scalar_integer(operation, &reached), true); }
        let candidate = match (operation, words.as_deref()) {
            (ScalarOp::Math(MathOp::Min), Some([a, b])) => Some(self.arena.int_min(*a, *b)),
            (ScalarOp::Math(MathOp::Max), Some([a, b])) => Some(self.arena.int_max(*a, *b)),
            (ScalarOp::Binary(op @ (SourceBinary::Add | SourceBinary::Sub | SourceBinary::Mul)), Some([a, b])) => {
                // Candidate bounds do not mention this operation's result binding
                // or assume the typed result range: wrapping remains explicit.
                let value = match op {
                    SourceBinary::Add => self.arena.int_add(*a, *b),
                    SourceBinary::Sub => self.arena.int_sub(*a, *b),
                    _ => self.arena.int_mul(*a, *b),
                };
                self.word_range(value, dtype).then_some(value)
            }
            (ScalarOp::Unary(SourceUnary::Neg), Some([a])) => {
                let zero = self.arena.int(0);
                let value = self.arena.int_sub(zero, *a);
                self.word_range(value, dtype).then_some(value)
            }
            (ScalarOp::Math(MathOp::Abs), Some([a])) => {
                let zero = self.arena.int(0);
                let negative = self.arena.int_sub(zero, *a);
                let value = self.arena.int_max(*a, negative);
                self.word_range(value, dtype).then_some(value)
            }
            (ScalarOp::Binary(op @ (SourceBinary::Div | SourceBinary::Rem)), Some([a, b])) => {
                let one = self.arena.int(1);
                let positive = self.arena.int_sub(*b, one);
                if self.nonnegative(*a) && self.nonnegative(positive) {
                    Some(if op == SourceBinary::Div { self.arena.int_div(*a, *b) } else { self.arena.int_rem(*a, *b) })
                } else { None }
            }
            (ScalarOp::Cast(target), Some([a])) => {
                if self.word_range(*a, target) { Some(*a) }
                else {
                    // The wrap is ordinary remainder/selection arithmetic;
                    // reach it so an established source interval decides it.
                    let wrapped = self.arena.int_to_word(*a, target);
                    let (reached, defined) = self.value(wrapped);
                    Some(if defined { reached } else { wrapped })
                }
            }
            _ => None,
        };
        (candidate.unwrap_or_else(|| self.arena.scalar_integer(operation, &reached)), true)
    }

    /// The value of an integer ordering these facts establish either way.
    fn decided_comparison(&mut self, condition: BoolExpr) -> Option<bool> {
        let NodeView::Cmp {
            op,
            lhs: AnyExpr::Int(lhs),
            rhs: AnyExpr::Int(rhs),
        } = self.arena.view(AnyExpr::Bool(condition))
        else {
            return None;
        };
        let (lhs, lhs_defined) = self.value(lhs);
        let (rhs, rhs_defined) = self.value(rhs);
        if !(lhs_defined && rhs_defined) {
            return None;
        }
        let (when_true, when_false) = match op {
            CmpOp::Ge => ((rhs, lhs, false), (lhs, rhs, true)),
            CmpOp::Lt => ((lhs, rhs, true), (rhs, lhs, false)),
            CmpOp::Gt => ((rhs, lhs, true), (lhs, rhs, false)),
            CmpOp::Le => ((lhs, rhs, false), (rhs, lhs, true)),
            CmpOp::Eq | CmpOp::Ne => return None,
        };
        // `(a, b, strict)`: `a < b` when strict, `a <= b` otherwise.
        let mut holds = |(a, b, strict): (IntExpr, IntExpr, bool)| {
            let mut gap = self.arena.int_sub(b, a);
            if strict {
                let one = self.arena.int(1);
                gap = self.arena.int_sub(gap, one);
            }
            established_nonnegative(self.arena, self.facts, gap)
        };
        if holds(when_true) {
            Some(true)
        } else if holds(when_false) {
            Some(false)
        } else {
            None
        }
    }

    fn exact_zeros(&self, value: Poly) -> Poly {
        self.world.exact.apply(value)
    }

    fn value(&mut self, original: IntExpr) -> (IntExpr, bool) {
        if let Some(value) = self.world.memo.borrow().get(&original) {
            return *value;
        }
        let (node, defined) = match self.arena.view(AnyExpr::Int(original)) {
            NodeView::Symbol(symbol) if self.facts.definitions.contains_key(&symbol) => {
                let meaning = self.facts.definitions[&symbol];
                self.value(meaning)
            }
            NodeView::ScalarInteger { operation, operands } => {
                let operands = operands.to_vec();
                self.typed_value(original, operation, &operands)
            }
            NodeView::Binary {
                op,
                lhs: AnyExpr::Int(a),
                rhs: AnyExpr::Int(b),
            } if matches!(
                op,
                BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Min
                    | BinaryOp::Max
                    | BinaryOp::Div
                    | BinaryOp::Rem
            ) =>
            {
                let (a, a_defined) = self.value(a);
                let (b, b_defined) = self.value(b);
                let divisor = constant(self.arena, b);
                let defined = a_defined
                    && b_defined
                    && (!matches!(op, BinaryOp::Div | BinaryOp::Rem)
                        || divisor.is_some_and(|d| d != 0));
                if !defined {
                    (original, false)
                } else {
                    let node = match op {
                        BinaryOp::Add => self.arena.int_add(a, b),
                        BinaryOp::Sub => self.arena.int_sub(a, b),
                        BinaryOp::Mul => self.arena.int_mul(a, b),
                        BinaryOp::Min | BinaryOp::Max => {
                            let left = normalize_int(self.arena, a);
                            let right = normalize_int(self.arena, b);
                            let left_first = established_polynomial_nonnegative(
                                self.arena,
                                self.facts,
                                right.sub(&left),
                            );
                            let right_first = !left_first
                                && established_polynomial_nonnegative(
                                    self.arena,
                                    self.facts,
                                    left.sub(&right),
                                );
                            match (op, left_first, right_first) {
                                (BinaryOp::Min, true, _) | (BinaryOp::Max, _, true) => a,
                                (BinaryOp::Min, _, true) | (BinaryOp::Max, true, _) => b,
                                (BinaryOp::Min, _, _) => self.arena.int_min(a, b),
                                _ => self.arena.int_max(a, b),
                            }
                        }
                        BinaryOp::Div if divisor.is_some_and(|d| d > 0) => self.arena.int_div(a, b),
                        BinaryOp::Rem if divisor.is_some_and(|d| d > 0) => {
                            let numerator = normalize_int(self.arena, a);
                            let upper = normalize_int(self.arena, b)
                                .sub(&Poly::constant(1))
                                .sub(&numerator);
                            let divisor = normalize_int(self.arena, b);
                            if established_polynomial_nonnegative(
                                self.arena,
                                self.facts,
                                numerator.clone(),
                            ) && established_polynomial_nonnegative(self.arena, self.facts, upper)
                            {
                                a
                            } else if established_polynomial_nonnegative(
                                self.arena,
                                self.facts,
                                numerator.add(&divisor),
                            ) && established_polynomial_nonnegative(
                                self.arena,
                                self.facts,
                                Poly::constant(-1).sub(&numerator),
                            ) {
                                // Euclidean remainder of `-d <= n < 0` is `n + d`.
                                self.arena.int_add(a, b)
                            } else {
                                self.arena.int_rem(a, b)
                            }
                        }
                        // Unknown/signed divisors retain the original operation;
                        // no unsigned interval rule is applied to their meaning.
                        _ => original,
                    };
                    (node, true)
                }
            }
            // A selection whose integer comparison these facts decide is the
            // chosen operand (for example a fixed-width wrap of an in-range
            // value is that value).
            NodeView::Select {
                cond,
                then: AnyExpr::Int(then),
                otherwise: AnyExpr::Int(otherwise),
            } => match self.decided_comparison(cond) {
                Some(true) => self.value(then),
                Some(false) => self.value(otherwise),
                None => (original, self.arena.is_total(AnyExpr::Int(original))),
            },
            _ => (original, self.arena.is_total(AnyExpr::Int(original))),
        };
        let value = if defined {
            let p = self.exact_zeros(normalize_int(self.arena, node));
            if p.valid {
                intern(self.arena, &p)
            } else {
                original
            }
        } else {
            original
        };
        let result = (value, defined);
        self.world.memo.borrow_mut().insert(original, result);
        result
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
    /// A source occurrence's oriented exact definition. Equality views are
    /// derived from this record; no second mutable equation list owns it.
    definitions: BTreeMap<SymbolId, IntExpr>,
    /// Expressions known nonnegative that bound no unit-coefficient symbol
    /// (`H / KV >= 1`, `T*P*P*C >= 1`).
    assumed: Vec<IntExpr>,
    /// Expressions known to be nonzero (`where M != 0`).
    nonzero: Vec<IntExpr>,
    /// Whole boolean facts; a disjunction never grants either arm alone.
    boolean: Vec<BoolExpr>,
    /// The caches of every world with exactly these facts (`Facts::caches`).
    /// Unset until first use after a change.
    caches: RefCell<Option<Rc<WorldCaches>>>,
}

/// Results derived under one exact fact content. Every world with that
/// content shares them: each entry holds under those facts alone, whichever
/// world derived it, so reuse never grants an unestablished proposition.
#[derive(Debug, Default)]
struct WorldCaches {
    /// Proof results under exactly these facts (rule 7).
    memo: RefCell<HashMap<Poly, Memo>>,
    /// Exact reached scalar reductions, separated by arena identity.
    scalar_cache: RefCell<HashMap<IntExpr, ReachedScalarWorld>>,
    /// The exact premise index, prepared once per arena.
    prepared: RefCell<HashMap<ArenaId, PreparedFacts>>,
    /// The established interval premises, selected once per arena. Selection
    /// consults proved memo entries, so a newly proved goal discards it along
    /// with the reached scalar results.
    interval: RefCell<HashMap<ArenaId, Rc<IntervalPremises>>>,
    /// Whether the facts directly contradict each other.
    contradiction: RefCell<HashMap<ArenaId, bool>>,
    /// Comparison condition -> its value decided by these facts' established
    /// evidence, or `None` when undecided.
    comparisons: RefCell<HashMap<crate::initialization::Condition, Option<bool>>>,
}

/// A fast multiplicative hasher for fact-content keys. Equal keys are still
/// confirmed by full comparison, so the hash only selects a bucket.
#[derive(Default)]
struct ContentHasher(u64);
impl std::hash::Hasher for ContentHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.write_u64(u64::from(*byte));
        }
    }
    fn write_u8(&mut self, value: u8) {
        self.write_u64(u64::from(value));
    }
    fn write_u32(&mut self, value: u32) {
        self.write_u64(u64::from(value));
    }
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

thread_local! {
    /// Exact equation indexes, shared by every world of one arena with the
    /// same zero facts and definitions. Bounded like `WORLDS`.
    static EXACT: RefCell<(usize, HashMap<u64, Vec<(Facts, Rc<ExactZeros>)>>)> =
        RefCell::default();
}

/// Fact contents seen on this thread, by content hash, with their caches.
/// Bounded: past the limit the table restarts, which only forgoes reuse.
const WORLD_LIMIT: usize = 1024;
thread_local! {
    static WORLDS: RefCell<(usize, HashMap<u64, Vec<(Facts, Rc<WorldCaches>)>>)> =
        RefCell::default();
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

    /// Publish an immutable source version. A changed meaning or cycle rejects
    /// the internal publication invariant; owners must mint a new version.
    pub(crate) fn define_scalar(&mut self, arena: &ExprArena, binding: SymbolId, meaning: IntExpr) -> bool {
        if let Some(existing) = self.definitions.get(&binding) {
            return *existing == meaning;
        }
        let mut pending = symbols(arena, meaning);
        let mut seen = HashSet::new();
        while let Some(symbol) = pending.pop() {
            if symbol == binding { return matches!(arena.view(meaning.into()), NodeView::Symbol(symbol) if symbol == binding); }
            if seen.insert(symbol) {
                if let Some(definition) = self.definitions.get(&symbol) {
                    pending.extend(symbols(arena, *definition));
                }
            }
        }
        self.changed().definitions.insert(binding, meaning);
        true
    }

    fn scalar_defined(&self, arena: &ExprArena, value: IntExpr) -> bool {
        if arena.is_total(value.into()) {
            return true;
        }
        // Successful exits can carry this premise inside a disjunction of
        // conjunctive paths. Reuse the Boolean implication owner: one unknown
        // alternative must never establish another alternative's continuation.
        let mut pending = self.boolean.clone();
        let mut seen = HashSet::new();
        while let Some(predicate) = pending.pop() {
            if !seen.insert(predicate) {
                continue;
            }
            if matches!(arena.view(predicate.into()),
                NodeView::Unary { op: crate::expr::UnaryOp::ScalarIntegerDefined, operand: AnyExpr::Int(operand) } if operand == value)
                && self.boolean.iter().any(|known| Self::boolean_implies(arena, *known, predicate))
            {
                return true;
            }
            if let Some((_, operands)) = Self::boolean_operands(arena, predicate) {
                pending.extend(operands);
            }
        }
        false
    }

    /// Premise of this source continuation, not a global totality assertion.
    pub(crate) fn assume_scalar_defined(&mut self, arena: &mut ExprArena, value: IntExpr) {
        if matches!(arena.view(value.into()), NodeView::ScalarInteger { .. }) {
            let defined = arena.scalar_integer_defined(value);
            self.assume_boolean(defined);
        }
    }

    pub(crate) fn assume_boolean(&mut self, value: BoolExpr) {
        if !self.boolean.contains(&value) {
            self.changed().boolean.push(value);
        }
    }

    /// Structural implication of whole boolean domains, without selecting an
    /// unknown disjunct. In particular a smaller allowed set implies a larger
    /// one, while an allowed set never implies one of its individual members.
    fn boolean_operands(arena: &ExprArena, value: BoolExpr) -> Option<(NaryOp, Vec<BoolExpr>)> {
        let (op, values) = match arena.view(value.into()) {
            NodeView::Binary {
                op,
                lhs: AnyExpr::Bool(lhs),
                rhs: AnyExpr::Bool(rhs),
            } if matches!(op, BinaryOp::And | BinaryOp::Or) => (
                if op == BinaryOp::And {
                    NaryOp::All
                } else {
                    NaryOp::Any
                },
                vec![lhs, rhs],
            ),
            NodeView::Nary { op, operands } if matches!(op, NaryOp::All | NaryOp::Any) => {
                let values = operands
                    .iter()
                    .map(|value| match value {
                        AnyExpr::Bool(value) => Some(*value),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()?;
                (op, values)
            }
            _ => return None,
        };
        let mut flattened = Vec::new();
        for value in values {
            match Self::boolean_operands(arena, value) {
                Some((child_op, children)) if child_op == op => flattened.extend(children),
                _ => flattened.push(value),
            }
        }
        Some((op, flattened))
    }

    fn boolean_implies(arena: &ExprArena, known: BoolExpr, required: BoolExpr) -> bool {
        if known == required {
            return true;
        }
        if let Some((NaryOp::Any, operands)) = Self::boolean_operands(arena, known) {
            return operands
                .iter()
                .all(|value| Self::boolean_implies(arena, *value, required));
        }
        if let Some((NaryOp::All, operands)) = Self::boolean_operands(arena, required) {
            return operands
                .iter()
                .all(|value| Self::boolean_implies(arena, known, *value));
        }
        if let Some((NaryOp::All, operands)) = Self::boolean_operands(arena, known) {
            return operands
                .iter()
                .any(|value| Self::boolean_implies(arena, *value, required));
        }
        if let Some((NaryOp::Any, operands)) = Self::boolean_operands(arena, required) {
            return operands
                .iter()
                .any(|value| Self::boolean_implies(arena, known, *value));
        }
        false
    }

    /// Integer equalities to every point of a finite contiguous interval are
    /// exhaustive when the existing numeric facts bound the same quantity.
    /// A gap in the allowed values is never filled by interval reasoning.
    fn equality_interval(&self, arena: &mut ExprArena, value: BoolExpr) -> bool {
        fn collect(arena: &ExprArena, value: BoolExpr, out: &mut Vec<(IntExpr, i64)>) -> bool {
            if let Some((NaryOp::Any, values)) = Facts::boolean_operands(arena, value) {
                return values.into_iter().all(|value| collect(arena, value, out));
            }
            let NodeView::Cmp {
                op: CmpOp::Eq,
                lhs: AnyExpr::Int(lhs),
                rhs: AnyExpr::Int(rhs),
            } = arena.view(value.into())
            else {
                return false;
            };
            if let Some(value) = constant(arena, rhs) {
                out.push((lhs, value));
                true
            } else if let Some(value) = constant(arena, lhs) {
                out.push((rhs, value));
                true
            } else {
                false
            }
        }
        let mut alternatives = Vec::new();
        if !collect(arena, value, &mut alternatives) {
            return false;
        }
        let Some((expression, _)) = alternatives.first().copied() else {
            return false;
        };
        if alternatives
            .iter()
            .any(|(candidate, _)| *candidate != expression)
        {
            return false;
        }
        let mut allowed = alternatives
            .into_iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>();
        allowed.sort_unstable();
        allowed.dedup();
        let minimum = allowed[0];
        let maximum = *allowed.last().unwrap();
        if i128::from(maximum) - i128::from(minimum) != (allowed.len() - 1) as i128 {
            return false;
        }
        let lower = arena.int(minimum);
        let upper = arena.int(maximum);
        derive_le(arena, self, lower, expression).is_established() && derive_le(arena, self, expression, upper).is_established()
    }

    fn boolean_false(&self, arena: &mut ExprArena, value: BoolExpr) -> bool {
        if let Some((op, values)) = Self::boolean_operands(arena, value) {
            return match op {
                NaryOp::All => values.iter().any(|value| self.boolean_false(arena, *value)),
                NaryOp::Any => values.iter().all(|value| self.boolean_false(arena, *value)),
                _ => unreachable!("boolean operands contain only conjunction/disjunction"),
            };
        }
        match arena.view(value.into()) {
            NodeView::BoolConst(value) => !value,
            NodeView::Cmp {
                op,
                lhs: AnyExpr::Int(lhs),
                rhs: AnyExpr::Int(rhs),
            } => {
                let op = match op {
                    CmpOp::Eq => CmpOp::Ne,
                    CmpOp::Ne => CmpOp::Eq,
                    CmpOp::Ge => CmpOp::Lt,
                    CmpOp::Gt => CmpOp::Le,
                    CmpOp::Le => CmpOp::Gt,
                    CmpOp::Lt => CmpOp::Ge,
                };
                let opposite = arena.int_cmp(op, lhs, rhs);
                self.boolean(arena, opposite)
            }

            _ => false,
        }
    }

    /// Derive numeric facts only from an established whole domain. A
    /// disjunction contributes its remaining arm only after every other arm
    /// has been proved false from the reached branch's existing facts.
    pub(crate) fn refine_booleans(&mut self, arena: &mut ExprArena) {
        fn refine(
            facts: &mut Facts,
            arena: &mut ExprArena,
            value: BoolExpr,
            derived: &mut std::collections::HashSet<BoolExpr>,
        ) -> bool {
            if let Some((op, values)) = Facts::boolean_operands(arena, value) {
                return match op {
                    NaryOp::All => {
                        let mut changed = false;
                        for value in values {
                            changed |= refine(facts, arena, value, derived);
                        }
                        changed
                    }
                    NaryOp::Any => {
                        let total = values.len();
                        let surviving = values
                            .into_iter()
                            .filter(|value| !facts.boolean_false(arena, *value))
                            .collect::<Vec<_>>();
                        if let [value] = surviving.as_slice() {
                            refine(facts, arena, *value, derived)
                        } else if !surviving.is_empty() && surviving.len() < total {
                            // Keep the remaining whole domain, never select
                            // an unknown alternative. Excluding F=0 from
                            // {0,1,2} establishes {1,2}, not either member.
                            let remaining = arena.any(&surviving);
                            if derived.insert(remaining) {
                                facts.assume_boolean(remaining);
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                    _ => unreachable!("boolean operands contain only conjunction/disjunction"),
                };
            }
            match arena.view(value.into()) {
                NodeView::Cmp {
                    op,
                    lhs: AnyExpr::Int(lhs),
                    rhs: AnyExpr::Int(rhs),
                } => {
                    if !derived.insert(value) {
                        return false;
                    }
                    let delta = arena.int_sub(lhs, rhs);
                    match op {
                        CmpOp::Eq => facts.assume_zero(arena, delta),
                        CmpOp::Ne => facts.assume_nonzero(arena, delta),
                        CmpOp::Ge => facts.assume_nonnegative(arena, delta),
                        CmpOp::Le => {
                            let reverse = arena.int_sub(rhs, lhs);
                            facts.assume_nonnegative(arena, reverse);
                        }
                        CmpOp::Gt | CmpOp::Lt => {
                            let delta = if op == CmpOp::Lt {
                                arena.int_sub(rhs, lhs)
                            } else {
                                delta
                            };
                            let one = arena.int(1);
                            let strict = arena.int_sub(delta, one);
                            facts.assume_nonnegative(arena, strict);
                        }
                    }
                    true
                }
                _ => false,
            }
        }
        let known = self.boolean.clone();
        let mut derived = std::collections::HashSet::new();
        loop {
            let mut changed = false;
            for value in &known {
                changed |= refine(self, arena, *value, &mut derived);
            }
            if !changed {
                break;
            }
        }
    }

    pub(crate) fn boolean(&self, arena: &mut ExprArena, value: BoolExpr) -> bool {
        let known = arena.all(&self.boolean);
        if arena.entails(known, value)
            || self
                .boolean
                .iter()
                .any(|known| Self::boolean_implies(arena, *known, value))
        {
            return true;
        }
        if let Some((op, values)) = Self::boolean_operands(arena, value) {
            if op == NaryOp::Any
                && (self.equality_interval(arena, value) || finite_case_boolean(arena, self, value))
            {
                return true;
            }
            return match op {
                NaryOp::All => values.iter().all(|value| self.boolean(arena, *value)),
                NaryOp::Any => values.iter().any(|value| self.boolean(arena, *value)),
                _ => unreachable!("boolean operands contain only conjunction/disjunction"),
            };
        }
        match arena.view(AnyExpr::Bool(value)) {
            NodeView::BoolConst(value) => value,

            NodeView::Cmp {
                op,
                lhs: AnyExpr::Int(lhs),
                rhs: AnyExpr::Int(rhs),
            } => match op {
                CmpOp::Ge => {
                    let delta = arena.int_sub(lhs, rhs);
                    nonneg(arena, self, delta)
                }
                CmpOp::Gt => derive_lt(arena, self, rhs, lhs).is_established(),
                CmpOp::Le => derive_le(arena, self, lhs, rhs).is_established(),
                CmpOp::Lt => derive_lt(arena, self, lhs, rhs).is_established(),
                CmpOp::Eq => {
                    let delta = arena.int_sub(lhs, rhs);
                    zero(arena, self, delta)
                }
                CmpOp::Ne => {
                    let delta = arena.int_sub(lhs, rhs);
                    self.nonzero(arena, delta)
                }
            },
            _ => false,
        }
    }

    /// Every change to the facts invalidates the proofs made under them.
    fn changed(&mut self) -> &mut Self {
        *self.caches.get_mut() = None;
        self
    }

    /// The shared caches of this exact fact content.
    fn caches(&self) -> Rc<WorldCaches> {
        if let Some(caches) = &*self.caches.borrow() {
            return caches.clone();
        }
        let key = {
            use std::hash::{Hash, Hasher};
            let mut hasher = ContentHasher::default();
            self.upper.hash(&mut hasher);
            self.lower.hash(&mut hasher);
            self.extra_upper.hash(&mut hasher);
            self.extra_lower.hash(&mut hasher);
            self.zero.hash(&mut hasher);
            self.definitions.hash(&mut hasher);
            self.assumed.hash(&mut hasher);
            self.nonzero.hash(&mut hasher);
            self.boolean.hash(&mut hasher);
            hasher.finish()
        };
        let caches = WORLDS.with(|worlds| {
            let mut worlds = worlds.borrow_mut();
            if worlds.0 >= WORLD_LIMIT {
                *worlds = Default::default();
            }
            if let Some((_, caches)) = worlds
                .1
                .get(&key)
                .and_then(|bucket| bucket.iter().find(|(known, _)| known.same_content(self)))
            {
                return caches.clone();
            }
            let caches = Rc::new(WorldCaches::default());
            let content = Facts {
                caches: RefCell::new(None),
                ..self.clone()
            };
            worlds.0 += 1;
            worlds.1.entry(key).or_default().push((content, caches.clone()));
            caches
        });
        *self.caches.borrow_mut() = Some(caches.clone());
        caches
    }

    /// The exact equation index of these zero facts and definitions.
    fn exact_zeros(&self, arena: &ExprArena) -> Rc<ExactZeros> {
        let key = {
            use std::hash::{Hash, Hasher};
            let mut hasher = ContentHasher::default();
            arena.identity().hash(&mut hasher);
            self.zero.hash(&mut hasher);
            self.definitions.hash(&mut hasher);
            hasher.finish()
        };
        let known = EXACT.with(|exact| {
            exact.borrow().1.get(&key).and_then(|bucket| {
                bucket
                    .iter()
                    .find(|(facts, exact)| {
                        exact.arena == arena.identity()
                            && facts.zero == self.zero
                            && facts.definitions == self.definitions
                    })
                    .map(|(_, exact)| exact.clone())
            })
        });
        if let Some(known) = known {
            return known;
        }
        let made = Rc::new(ExactZeros::new(arena.identity(), self.zeros(arena)));
        EXACT.with(|exact| {
            let mut exact = exact.borrow_mut();
            if exact.0 >= WORLD_LIMIT {
                *exact = Default::default();
            }
            exact.0 += 1;
            let content = Facts {
                zero: self.zero.clone(),
                definitions: self.definitions.clone(),
                ..Facts::default()
            };
            exact.1.entry(key).or_default().push((content, made.clone()));
        });
        made
    }

    /// A comparison's value already decided under these facts.
    pub(crate) fn comparison(
        &self,
        condition: &crate::initialization::Condition,
    ) -> Option<Option<bool>> {
        self.caches().comparisons.borrow().get(condition).copied()
    }

    pub(crate) fn remember_comparison(
        &self,
        condition: crate::initialization::Condition,
        value: Option<bool>,
    ) {
        self.caches()
            .comparisons
            .borrow_mut()
            .insert(condition, value);
    }

    fn same_content(&self, other: &Facts) -> bool {
        self.upper == other.upper
            && self.lower == other.lower
            && self.extra_upper == other.extra_upper
            && self.extra_lower == other.extra_lower
            && self.zero == other.zero
            && self.definitions == other.definitions
            && self.assumed == other.assumed
            && self.nonzero == other.nonzero
            && self.boolean == other.boolean
    }

    /// Newly established signs can select an exact min/max operand or reduce
    /// a remainder, or a stronger interval endpoint. Keep immutable equality
    /// metadata, but revisit DAG results and endpoint selection.
    fn learned_sign(&self) {
        let caches = self.caches();
        for world in caches.scalar_cache.borrow().values() {
            world.memo.borrow_mut().clear();
        }
        caches.interval.borrow_mut().clear();
    }

    /// Replace the lower endpoint owned by a binding or a newly entered world.
    /// Independently conjoined premises remain valid in that world.
    pub(crate) fn set_range_lower(&mut self, symbol: SymbolId, lo: IntExpr) {
        if self.lower.get(&symbol) != Some(&lo) {
            self.changed().lower.insert(symbol, lo);
        }
    }

    /// Install the range owned by a binding or quantified scope. This is a
    /// deliberate replacement, unlike adding a path or type premise.
    pub(crate) fn set_range(&mut self, symbol: SymbolId, lo: IntExpr, hi: IntExpr) {
        if self.lower.get(&symbol) != Some(&lo) || self.upper.get(&symbol) != Some(&hi) {
            let facts = self.changed();
            facts.lower.insert(symbol, lo);
            facts.upper.insert(symbol, hi);
        }
    }

    pub(crate) fn assume_zero(&mut self, arena: &ExprArena, e: IntExpr) {
        if !is_zero(arena, e) && !self.zero.contains(&e) {
            self.changed().zero.push(e);
        }
    }

    /// Assume `e != 0`.
    pub(crate) fn assume_nonzero(&mut self, arena: &ExprArena, e: IntExpr) {
        if constant(arena, e).is_none() && !self.nonzero.contains(&e) {
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

    /// A path condition `symbol <= hi`.
    pub(crate) fn add_upper(&mut self, symbol: SymbolId, hi: IntExpr) {
        if !self.extra_upper.contains(&(symbol, hi)) {
            self.changed().extra_upper.push((symbol, hi));
        }
    }

    /// A path condition `symbol >= lo`.
    pub(crate) fn add_lower(&mut self, symbol: SymbolId, lo: IntExpr) {
        if !self.extra_lower.contains(&(symbol, lo)) {
            self.changed().extra_lower.push((symbol, lo));
        }
    }

    /// Conjoin the bounds of a value of type `ty` held by `symbol` (L32 (a)):
    /// `index[B]` is `0 <= v <= B - 1`, an `i32` word is its two's-complement
    /// range and a `u32` word `0 <= v <= 2^32 - 1`. Every other type states
    /// nothing. Publishing a known value again does not replace the bounds
    /// already established by its source position. Missing primary endpoints
    /// are seeded for fresh bindings; the type premises are retained separately
    /// so later deliberate range replacement cannot erase them.
    pub(crate) fn assume_type(&mut self, arena: &mut ExprArena, symbol: SymbolId, ty: &ValueType) {
        let (lo, hi) = match ty {
            ValueType::Index { bound } => {
                let one = arena.int(1);
                (arena.int(0), arena.int_sub(*bound, one))
            }
            ValueType::Scalar(DType::I32) => (
                arena.int(i64::from(i32::MIN)),
                arena.int(i64::from(i32::MAX)),
            ),
            ValueType::Scalar(DType::U32) => (arena.int(0), arena.int(i64::from(u32::MAX))),
            _ => return,
        };
        let seed_lower = !self.lower.contains_key(&symbol);
        let seed_upper = !self.upper.contains_key(&symbol);
        let add_lower = !self.extra_lower.contains(&(symbol, lo));
        let add_upper = !self.extra_upper.contains(&(symbol, hi));
        if seed_lower || seed_upper || add_lower || add_upper {
            let facts = self.changed();
            if seed_lower {
                facts.lower.insert(symbol, lo);
            }
            if seed_upper {
                facts.upper.insert(symbol, hi);
            }
            if add_lower {
                facts.extra_lower.push((symbol, lo));
            }
            if add_upper {
                facts.extra_upper.push((symbol, hi));
            }
        }
    }

    /// All conjunctive bounds, omitting a currently equal primary/extra pair
    /// during consumption without discarding either premise's ownership.
    fn lower_bounds(&self) -> impl Iterator<Item = (SymbolId, IntExpr)> + '_ {
        self.lower
            .iter()
            .map(|(symbol, bound)| (*symbol, *bound))
            .chain(self.extra_lower.iter().copied().filter(|(symbol, bound)| {
                self.lower.get(symbol) != Some(bound)
            }))
    }

    fn upper_bounds(&self) -> impl Iterator<Item = (SymbolId, IntExpr)> + '_ {
        self.upper
            .iter()
            .map(|(symbol, bound)| (*symbol, *bound))
            .chain(self.extra_upper.iter().copied().filter(|(symbol, bound)| {
                self.upper.get(symbol) != Some(bound)
            }))
    }

    /// Optional common-bound strengthening for a joined integer. Candidate
    /// inequalities must be established in both arms; unknown candidates
    /// add no facts. Later required checks retain the complete prover.
    /// Candidates are both arm values, zero, and fact bounds of their symbols,
    /// restricted to expressions in scope after the join.
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
            if arms.iter().all(|(facts, value)| {
                let difference = arena.int_sub(*value, candidate);
                established_nonnegative(arena, facts, difference)
            }) {
                self.add_lower(joined, candidate);
            }
            if arms.iter().all(|(facts, value)| {
                let difference = arena.int_sub(candidate, *value);
                established_nonnegative(arena, facts, difference)
            }) {
                self.add_upper(joined, candidate);
            }
        }
    }

    /// Assume `e >= 0`: a bound on every symbol with a unit coefficient, and
    /// the whole expression when it bounds no such symbol.
    pub(crate) fn assume_nonnegative(&mut self, arena: &mut ExprArena, e: IntExpr) {
        // Normalize using facts that held before this new assumption. This
        // preserves the reached proposition while exposing exact modular
        // round trips to ordinary bound/guard ownership.
        let e = canonical_with_facts(arena, self, e);
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
        if !bounded && constant(arena, e).is_none() && !self.assumed.contains(&e) {
            self.changed().assumed.push(e);
        }
    }

    fn uppers_of(&self, arena: &ExprArena, symbol: SymbolId) -> Vec<Poly> {
        self.upper
            .get(&symbol)
            .copied()
            .into_iter()
            .chain(self.extra_upper.iter().filter_map(|(bounded, bound)| {
                (*bounded == symbol && self.upper.get(&symbol) != Some(bound)).then_some(*bound)
            }))
            .map(|bound| normalize_int(arena, bound))
            .collect()
    }

    fn lowers_of(&self, arena: &ExprArena, symbol: SymbolId) -> Vec<Poly> {
        self.lower
            .get(&symbol)
            .copied()
            .into_iter()
            .chain(self.extra_lower.iter().filter_map(|(bounded, bound)| {
                (*bounded == symbol && self.lower.get(&symbol) != Some(bound)).then_some(*bound)
            }))
            .map(|bound| normalize_int(arena, bound))
            .filter(|bound| !bound.is_zero())
            .collect()
    }

    fn lower_poly(&self, arena: &ExprArena, symbol: SymbolId) -> Option<Poly> {
        self.lower.get(&symbol).map(|e| normalize_int(arena, *e))
    }

    fn upper_poly(&self, arena: &ExprArena, symbol: SymbolId) -> Option<Poly> {
        self.upper.get(&symbol).map(|e| normalize_int(arena, *e))
    }

    fn zeros(&self, arena: &ExprArena) -> Vec<Poly> {
        self.zero.iter().map(|e| normalize_int(arena, *e)).chain(
            self.definitions.iter().map(|(binding, meaning)| Poly::symbol(*binding).sub(&normalize_int(arena, *meaning)))
        ).collect()
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
/// `e` alone, whatever order its rules are explored in (rule 7). The search
/// includes exact packing bounds and is exponential in depth. Keep this
/// fixed limit rather than expanding it to admit one source obligation;
/// unresolved goals remain diagnostics, never assumed facts.
const DEPTH: usize = 8;

#[derive(Clone, Debug)]
struct PreparedFacts {
    exact: Rc<ExactZeros>,
    assumed: Rc<Vec<Poly>>,
    /// Structural signs decided from an empty cycle guard. They are a
    /// function of the unchanged facts alone, independent of proof memo.
    structural: Rc<RefCell<HashMap<Poly, bool>>>,
    atom_signs: Rc<RefCell<HashMap<Atom, bool>>>,
    /// Symbol -> its normalized upper or lower bounds, in fact order.
    uppers: Rc<RefCell<HashMap<SymbolId, Rc<Vec<Poly>>>>>,
    lowers: Rc<RefCell<HashMap<SymbolId, Rc<Vec<Poly>>>>>,
}
struct Engine<'a> {
    arena: &'a ExprArena,
    facts: &'a Facts,
    exact: Rc<ExactZeros>,
    zeros: Rc<Vec<Poly>>,
    assumed: Rc<Vec<Poly>>,
    prepared: PreparedFacts,
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

/// An admission request establishes a proposition or leaves it unresolved.
/// Unresolved is not a proof of its negation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Derivation { Established, Unresolved }
impl Derivation {
    pub(crate) fn is_established(self) -> bool { self == Self::Established }
    fn from_established(value: bool) -> Self { if value { Self::Established } else { Self::Unresolved } }
}
pub(crate) fn derive_nonnegative(arena: &mut ExprArena, facts: &Facts, value: IntExpr) -> Derivation {
    Derivation::from_established(nonneg(arena, facts, value))
}
pub(crate) fn derive_zero(arena: &mut ExprArena, facts: &Facts, value: IntExpr) -> Derivation {
    Derivation::from_established(zero(arena, facts, value))
}

/// Prove `e >= 0` under the facts.
fn nonneg(arena: &mut ExprArena, facts: &Facts, e: IntExpr) -> bool {
    let reached = ReachedScalarNormalForm::for_obligation(arena, facts).value(e).0;
    if nonneg_reached(arena, facts, reached) {
        return true;
    }
    // Recorded bounds keep their source form. When reaching rewrote the goal
    // (for example a selected min operand), the source form can still meet
    // those bounds term for term.
    !same(arena, reached, e)
        && normalize_int(arena, e).valid
        && nonneg_reached(arena, facts, e)
}

fn nonneg_reached(arena: &mut ExprArena, facts: &Facts, e: IntExpr) -> bool {
    let p = normalize_int(arena, e);
    // A shallow existing proof must not wait behind a failing deep branch.
    // Failed ranges are memoized; the final depth remains the same rule set.
    if !p.valid {
        return false;
    }
    let known = facts.caches().memo.borrow().get(&p).copied();
    if known.is_some_and(|memo| memo.proved.is_some_and(|depth| depth <= 1)) {
        return true;
    }
    // Recursive-search failure does not include the independent finite-domain
    // alternative. Preserve that alternative even when every depth is cached;
    // only avoid constructing the engine that would return the same failure.
    let engine = if known.is_some_and(|memo| memo.failed >= DEPTH) {
        None
    } else {
        Some(Engine::new(arena, facts))
    };
    if engine
        .as_ref()
        .is_some_and(|engine| engine.nonneg_steps(p.clone(), 1))
    {
        return true;
    }
    // Mandatory admission consumes the same established relations as optional
    // projection before extending them with the bounded derivation strategy.
    // Keep constant, direct and cached successes above this interval summary.
    // A successfully evaluated opaque word recipe retains its declared result
    // range even when it cannot project to mathematical Int.
    if established_polynomial_nonnegative(arena, facts, p.clone()) {
        return true;
    }
    if finite_domain_nonnegative(arena, facts, e) {
        // The exhaustive result is established under exactly these facts.
        // Keep it in the same success memo as every other positive proof.
        let caches = facts.caches();
        let mut memo = caches.memo.borrow_mut();
        let learned = memo.get(&p).is_none_or(|known| known.proved.is_none());
        memo.insert(
            p,
            Memo {
                proved: Some(1),
                failed: 0,
            },
        );
        drop(memo);
        if learned {
            facts.learned_sign();
        }
        return true;
    }
    engine.is_some_and(|engine| (2..=DEPTH).any(|depth| engine.nonneg_steps(p.clone(), depth)))
}

/// Whether `facts` hold on no execution, by direct evidence only: a nonzero
/// constant in the span of the exact equations, a nonzero premise that the
/// equations make zero, a negative constant assumption, or a symbol whose
/// lower bound structurally exceeds one of its upper bounds. This never
/// searches a derivation; an undetected contradiction leaves the world
/// reachable, which is always sound.
pub(crate) fn directly_contradictory(arena: &ExprArena, facts: &Facts) -> bool {
    if let Some(known) = facts.caches().contradiction.borrow().get(&arena.identity()) {
        return *known;
    }
    let engine = Engine::new(arena, facts);
    let contradictory = engine.exact.spans(Poly::constant(1))
        || facts.nonzero.iter().any(|value| {
            let value = engine.apply_zero_facts(normalize_int(arena, *value));
            value.is_zero() || (value.valid && engine.exact.spans(value))
        })
        || engine
            .assumed
            .iter()
            .any(|assumed| assumed.as_constant().is_some_and(|constant| constant < 0))
        || {
            let mut uppers = HashMap::<SymbolId, Vec<IntExpr>>::new();
            for (symbol, upper) in facts.upper_bounds() {
                uppers.entry(symbol).or_default().push(upper);
            }
            facts.lower_bounds().any(|(symbol, lower)| {
                let Some(uppers) = uppers.get(&symbol) else {
                    return false;
                };
                let lower = normalize_int(arena, lower);
                uppers.iter().any(|upper| {
                    let gap = engine.apply_zero_facts(
                        lower
                            .sub(&normalize_int(arena, *upper))
                            .sub(&Poly::constant(1)),
                    );
                    gap.valid && engine.nonnegative_polynomial(&gap, &mut HashSet::new())
                })
            })
        };
    facts
        .caches()
        .contradiction
        .borrow_mut()
        .insert(arena.identity(), contradictory);
    contradictory
}

pub(crate) fn established_nonnegative(arena: &ExprArena, facts: &Facts, e: IntExpr) -> bool {
    established_polynomial_nonnegative(arena, facts, normalize_int(arena, e))
}

fn established_polynomial_nonnegative(arena: &ExprArena, facts: &Facts, raw: Poly) -> bool {
    let engine = Engine::new(arena, facts);
    let canonical = engine.apply_zero_facts(raw.clone());
    if !raw.valid || !canonical.valid {
        return false;
    }
    let caches = facts.caches();
    let memo = caches.memo.borrow();
    if [&raw, &canonical]
        .iter()
        .any(|p| memo.get(*p).is_some_and(|m| m.proved.is_some()))
    {
        return true;
    }
    drop(memo);
    let premises = IntervalPremises::new(&engine);
    if let Some(known) = premises.decided.borrow().get(&canonical) {
        return *known;
    }
    let decided = interval_nonnegative(arena, &engine, &premises, canonical.clone());
    premises.decided.borrow_mut().insert(canonical, decided);
    decided
}

/// The interval and established-premise rules of
/// `established_polynomial_nonnegative` for one zero-reduced goal.
fn interval_nonnegative(
    arena: &ExprArena,
    engine: &Engine,
    premises: &IntervalPremises,
    canonical: Poly,
) -> bool {
    let known = |p: Poly| premises.known(engine, p);
    let tight_engine = Engine::new(arena, &premises.tightened);
    let (lower, _) = tight_engine.interval_over(&canonical, &|_| true);
    let lower = tight_engine.apply_zero_facts(lower);
    if tight_engine.nonnegative_polynomial(&lower, &mut HashSet::new()) {
        return true;
    }
    if known(canonical.clone()) || known(lower.clone()) {
        return true;
    }
    // Positive constant scaling preserves a polynomial's sign. Remove only
    // the common integer coefficient factor from the goal or its already
    // established interval lower bound; do not search new substitutions.
    let primitive = |mut value: Poly| {
        let factor = value.terms.values().fold(0u64, |mut factor, coefficient| {
            let mut remainder = coefficient.unsigned_abs();
            while remainder != 0 {
                (factor, remainder) = (remainder, factor % remainder);
            }
            factor
        });
        if factor <= 1 {
            return None;
        }
        for coefficient in value.terms.values_mut() {
            *coefficient = (i128::from(*coefficient) / i128::from(factor)) as i64;
        }
        Some(value)
    };
    let factored_known = |value: Poly| {
        if primitive(value.clone()).is_some_and(&known) {
            return true;
        }
        let Some(first) = value.terms.keys().next() else {
            return false;
        };
        let mut common = first.clone();
        for monomial in value.terms.keys().skip(1) {
            common.retain(|atom, power| {
                *power = (*power).min(monomial.get(atom).copied().unwrap_or(0));
                *power > 0
            });
        }
        // A common nonnegative factor times an established nonnegative
        // residual is nonnegative. An unknown signed factor is never removed.
        common.retain(|atom, _| engine.nonnegative_atom(atom, &mut HashSet::new()));
        if common.is_empty() {
            return false;
        }
        let mut residual = value;
        residual.terms = residual
            .terms
            .into_iter()
            .map(|(mut monomial, coefficient)| {
                for (atom, power) in &common {
                    let remaining = monomial[atom] - power;
                    if remaining == 0 {
                        monomial.remove(atom);
                    } else {
                        monomial.insert(atom.clone(), remaining);
                    }
                }
                (monomial, coefficient)
            })
            .collect();
        known(residual.clone()) || primitive(residual).is_some_and(&known)
    };
    factored_known(canonical) || factored_known(lower)
}

/// The goal-independent premises of `established_polynomial_nonnegative`
/// under one fact world: every established bound, and the interval world
/// whose structural endpoints are the strongest established ones.
#[derive(Debug)]
struct IntervalPremises {
    established: Vec<Poly>,
    /// Whether each established bound is zero-inert (`Engine::inert`).
    inert: Vec<bool>,
    /// Whether every reached assumption is zero-inert.
    assumed_inert: bool,
    /// Monomial -> the inert bounds with a nonzero coefficient on it.
    bounds_with: HashMap<Monomial, Vec<usize>>,
    /// The bounds that are not inert.
    active_bounds: Vec<usize>,
    /// Monomial -> the inert bounds with a positive coefficient on it.
    positive_on: HashMap<Monomial, Vec<usize>>,
    /// The inert bounds without a positive coefficient.
    nonpositive_bounds: Vec<usize>,
    /// The inert bounds without a nonconstant monomial.
    constant_bounds: Vec<usize>,
    tightened: Facts,
    /// Zero-reduced goal -> its interval/established-premise verdict, which
    /// is a function of these same facts and proved memo entries.
    decided: RefCell<HashMap<Poly, bool>>,
}

impl IntervalPremises {
    /// The premises of `engine`'s fact world. They depend only on its facts
    /// and proved memo entries; `Facts::changed` and `Facts::learned_sign`
    /// discard the shared result whenever either can differ.
    fn new(engine: &Engine) -> Rc<IntervalPremises> {
        let (arena, facts) = (engine.arena, engine.facts);
        if let Some(existing) = facts.caches().interval.borrow().get(&arena.identity()) {
            return existing.clone();
        }
        // Each extra bound is already a path or type premise. Select a stronger endpoint
        // only with direct established evidence; this never searches a proof.
        // Unit-symbol bounds retain the original inequality just as an
        // irreducible assumption does, even when two valid bounds are unordered.
        let mut established = (*engine.assumed).clone();
        for (symbol, bound) in facts.lower_bounds() {
            established.push(Poly::symbol(symbol).sub(&normalize_int(arena, bound)));
        }
        for (symbol, bound) in facts.upper_bounds() {
            established.push(normalize_int(arena, bound).sub(&Poly::symbol(symbol)));
        }
        let inert = established
            .iter()
            .map(|bound| engine.inert(bound))
            .collect::<Vec<_>>();
        let mut bounds_with = HashMap::<Monomial, Vec<usize>>::new();
        let mut positive_on = HashMap::<Monomial, Vec<usize>>::new();
        let mut active_bounds = Vec::new();
        let mut nonpositive_bounds = Vec::new();
        let mut constant_bounds = Vec::new();
        for (index, bound) in established.iter().enumerate() {
            if !inert[index] {
                active_bounds.push(index);
                continue;
            }
            for (monomial, coefficient) in &bound.terms {
                if *coefficient != 0 {
                    bounds_with.entry(monomial.clone()).or_default().push(index);
                }
                if *coefficient > 0 {
                    positive_on.entry(monomial.clone()).or_default().push(index);
                }
            }
            if bound.terms.values().all(|coefficient| *coefficient <= 0) {
                nonpositive_bounds.push(index);
            }
            if bound
                .terms
                .iter()
                .all(|(monomial, coefficient)| monomial.is_empty() || *coefficient == 0)
            {
                constant_bounds.push(index);
            }
        }
        let mut premises = IntervalPremises {
            inert,
            assumed_inert: engine.assumed.iter().all(|assumed| engine.inert(assumed)),
            bounds_with,
            active_bounds,
            positive_on,
            nonpositive_bounds,
            constant_bounds,
            established,
            tightened: facts.clone(),
            decided: RefCell::default(),
        };
        // This private interval world changes structural bounds below. It must
        // not publish those reductions into the original world's shared caches.
        // Every extra bound is itself a premise, so each choice is sound; the
        // candidate replaces the current endpoint when it is directly at least
        // as strong, without consulting the other bounds.
        premises.tightened.changed();
        for (symbol, candidate) in &facts.extra_lower {
            if premises.tightened.lower.get(symbol) == Some(candidate) {
                continue;
            }
            let use_candidate = premises.tightened.lower.get(symbol).is_none_or(|current| {
                directly_known(
                    engine,
                    normalize_int(arena, *candidate).sub(&normalize_int(arena, *current)),
                )
            });
            if use_candidate {
                premises.tightened.lower.insert(*symbol, *candidate);
            }
        }
        for (symbol, candidate) in &facts.extra_upper {
            if premises.tightened.upper.get(symbol) == Some(candidate) {
                continue;
            }
            let use_candidate = premises.tightened.upper.get(symbol).is_none_or(|current| {
                directly_known(
                    engine,
                    normalize_int(arena, *current).sub(&normalize_int(arena, *candidate)),
                )
            });
            if use_candidate {
                premises.tightened.upper.insert(*symbol, *candidate);
            }
        }
        let premises = Rc::new(premises);
        facts
            .caches()
            .interval
            .borrow_mut()
            .insert(arena.identity(), premises.clone());
        premises
    }

    /// `p >= 0` by direct established evidence under `engine`'s facts.
    fn known(&self, engine: &Engine, p: Poly) -> bool {
        let p = engine.apply_zero_facts(p);
        p.valid
            && (engine.assumed.contains(&p)
                || engine.nonnegative_polynomial(&p, &mut HashSet::new())
                || self.discharged_by_bound(engine, &p)
                || engine
                    .facts
                    .caches()
                    .memo
                    .borrow()
                    .get(&p)
                    .is_some_and(|m| m.proved.is_some()))
    }

    /// Some established bound `b` makes `p - b` structurally nonnegative.
    fn discharged_by_bound(&self, engine: &Engine, p: &Poly) -> bool {
        let inert = self.assumed_inert && engine.inert(p);
        let structural = |index: usize| {
            let bound = &self.established[index];
            if inert && self.inert[index] {
                engine.inert_difference_nonnegative(p, bound)
            } else {
                engine.nonnegative_polynomial(
                    &engine.apply_zero_facts(p.sub(bound)),
                    &mut HashSet::new(),
                )
            }
        };
        match self.candidates(engine, p) {
            Some(candidates) => candidates.into_iter().any(structural),
            None => (0..self.established.len()).any(structural),
        }
    }

    /// The bounds that can discharge `p`, in bound order: every other
    /// bound provably cannot. For inert `p` and bound `b`, every coefficient
    /// of `p - b` is nonnegative over nonnegative atoms only if `b` matches
    /// `p` exactly on each monomial of `p` with an atom of unknown sign, has
    /// at most `p`'s coefficient on each negative monomial of `p`, and has
    /// positive coefficients only on monomials of `p`. The gap of `p - b` to
    /// an assumption `a` is constant only if `b` has exactly `p - a`'s
    /// coefficient on a monomial where they differ (or, when they differ on
    /// none, `b` is constant). `None` keeps every bound.
    fn candidates(&self, engine: &Engine, p: &Poly) -> Option<Vec<usize>> {
        if !self.assumed_inert
            || !engine.inert(p)
            || p.terms.values().any(|coefficient| *coefficient == 0)
        {
            return None;
        }
        let coefficient = |bound: usize, monomial: &Monomial| {
            self.established[bound]
                .terms
                .get(monomial)
                .copied()
                .unwrap_or(0)
        };
        let with = |monomial: &Monomial| {
            self.bounds_with
                .get(monomial)
                .map_or(&[][..], Vec::as_slice)
        };
        let mut candidates = self.active_bounds.clone();
        // Structural coefficients.
        let unsigned = p.terms.iter().find(|(monomial, _)| {
            !monomial
                .keys()
                .all(|atom| engine.nonnegative_atom(atom, &mut HashSet::new()))
        });
        let negative = p.terms.iter().find(|(_, c)| **c < 0);
        match (unsigned, negative) {
            (Some((monomial, value)), _) => candidates.extend(
                with(monomial)
                    .iter()
                    .filter(|bound| coefficient(**bound, monomial) == *value),
            ),
            (None, Some((monomial, value))) => candidates.extend(
                with(monomial)
                    .iter()
                    .filter(|bound| coefficient(**bound, monomial) <= *value),
            ),
            (None, None) => {
                candidates.extend_from_slice(&self.nonpositive_bounds);
                for (monomial, value) in &p.terms {
                    if let Some(bounds) = self.positive_on.get(monomial) {
                        candidates.extend(
                            bounds
                                .iter()
                                .filter(|bound| coefficient(**bound, monomial) <= *value),
                        );
                    }
                }
            }
        }
        // Constant assumption gaps.
        for assumed in engine.assumed.iter() {
            let differing = p
                .terms
                .keys()
                .chain(assumed.terms.keys())
                .find(|monomial| {
                    !monomial.is_empty()
                        && p.terms.get(*monomial).copied().unwrap_or(0)
                            != assumed.terms.get(*monomial).copied().unwrap_or(0)
                });
            match differing {
                Some(monomial) => {
                    let p_value = p.terms.get(monomial).copied().unwrap_or(0);
                    let a_value = assumed.terms.get(monomial).copied().unwrap_or(0);
                    match p_value.checked_sub(a_value) {
                        Some(required) => candidates.extend(
                            with(monomial)
                                .iter()
                                .filter(|bound| coefficient(**bound, monomial) == required),
                        ),
                        // Out-of-range coefficients keep every bound for this gap.
                        None => candidates.extend(with(monomial)),
                    }
                }
                None => candidates.extend_from_slice(&self.constant_bounds),
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        Some(candidates)
    }
}

/// `p >= 0` as a reached assumption or by its structural sign alone.
fn directly_known(engine: &Engine, p: Poly) -> bool {
    let p = engine.apply_zero_facts(p);
    p.valid
        && (engine.assumed.contains(&p) || engine.nonnegative_polynomial(&p, &mut HashSet::new()))
}

/// The terms of `left - right`, exactly as `Poly::sub` builds them from the
/// monomial-ordered terms of a polynomial with validity `valid`; `None` when
/// that difference is invalid.
fn coefficient_difference<'a>(
    valid: bool,
    left: impl Iterator<Item = (&'a Monomial, &'a i64)>,
    right: &'a Poly,
) -> Option<Vec<(&'a Monomial, i64)>> {
    if !valid || !right.valid {
        return None;
    }
    let mut out = Vec::with_capacity(left.size_hint().0 + right.terms.len());
    let mut left = left.peekable();
    let mut right = right.terms.iter().peekable();
    loop {
        let order = match (left.peek(), right.peek()) {
            (None, None) => return Some(out),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (Some((l, _)), Some((r, _))) => l.cmp(r),
        };
        match order {
            std::cmp::Ordering::Less => {
                let (monomial, coefficient) = left.next().expect("peeked left term");
                out.push((monomial, *coefficient));
            }
            std::cmp::Ordering::Greater => {
                let (monomial, coefficient) = right.next().expect("peeked right term");
                let negated = coefficient.checked_neg()?;
                if negated != 0 {
                    out.push((monomial, negated));
                }
            }
            std::cmp::Ordering::Equal => {
                let (monomial, kept) = left.next().expect("peeked left term");
                let (_, coefficient) = right.next().expect("peeked right term");
                let negated = coefficient.checked_neg()?;
                if negated == 0 {
                    out.push((monomial, *kept));
                    continue;
                }
                let sum = kept.checked_add(negated)?;
                if sum != 0 {
                    out.push((monomial, sum));
                }
            }
        }
    }
}

/// An already established or structural nonnegative fact, without searching
/// arithmetic alternatives. Optional inference can use this cheaper evidence;
/// required proof obligations still use the complete fixed-depth `nonneg`.
pub(crate) fn directly_nonnegative(arena: &ExprArena, facts: &Facts, e: IntExpr) -> bool {
    let engine = Engine::new(arena, facts);
    let p = engine.apply_zero_facts(normalize_int(arena, e));
    p.valid
        && (engine.assumed.contains(&p) || engine.nonnegative_polynomial(&p, &mut HashSet::new()))
}

/// Prove `e == 0` under the zero facts.
fn zero(arena: &mut ExprArena, facts: &Facts, e: IntExpr) -> bool {
    let reached = canonical_with_facts(arena, facts, e);
    let engine = Engine::new(arena, facts);
    let p = engine.apply_zero_facts(normalize_int(arena, reached));
    p.valid
        && (p.is_zero()
            || engine.zeros.contains(&p)
            || engine.zeros.contains(&p.neg())
            || engine.exact.spans(p))
}

/// The pivot rows that decide membership in the span of `equations`.
fn zero_span_basis(equations: &[Poly]) -> BTreeMap<Monomial, Poly> {
    let mut basis = BTreeMap::new();
    for equation in equations.iter().filter(|p| p.valid) {
        let row = reduce_by_zero_span_basis(equation.clone(), &basis);
        if !row.valid {
            continue;
        }
        if let Some((pivot, _)) = row.terms.last_key_value() {
            basis.insert(pivot.clone(), row);
        }
    }
    basis
}

/// Eliminate the basis pivots from `row` in descending pivot order. An
/// elimination changes only monomials below its pivot (each pivot is its
/// equation's greatest monomial), so visiting just the pivots `row` contains
/// performs exactly the eliminations of a scan over every pivot.
fn reduce_by_zero_span_basis(mut row: Poly, basis: &BTreeMap<Monomial, Poly>) -> Poly {
    let mut below: Option<Monomial> = None;
    loop {
        let next = match &below {
            None => row.terms.keys().rev().find(|m| basis.contains_key(*m)),
            Some(bound) => row
                .terms
                .range::<Monomial, _>(..bound)
                .rev()
                .map(|(m, _)| m)
                .find(|m| basis.contains_key(*m)),
        };
        let Some(pivot) = next.cloned() else {
            return row;
        };
        let equation = &basis[&pivot];
        let coefficient = row.terms[&pivot];
        let factor = equation.terms[&pivot];
        row = row
            .mul(&Poly::constant(factor))
            .sub(&equation.mul(&Poly::constant(coefficient)));
        if !row.valid {
            return row;
        }
        below = Some(pivot);
    }
}

fn in_zero_span(goal: Poly, basis: &BTreeMap<Monomial, Poly>) -> bool {
    let result = reduce_by_zero_span_basis(goal, basis);
    result.valid && result.is_zero()
}

#[cfg(test)]
#[test]
fn affine_storage_equalities_compose_without_unrelated_assumptions() {
    let mut arena = ExprArena::new();
    let symbols = (0..3)
        .map(|_| arena.checker_variable(crate::expr::SymbolSort::Int))
        .collect::<Vec<_>>();
    let w = arena.int_symbol(symbols[0]);
    let kp = arena.int_symbol(symbols[1]);
    let vp = arena.int_symbol(symbols[2]);
    let sixteen = arena.int(16);
    let sized = arena.int_mul(sixteen, kp);
    let actual = arena.int_mul(sixteen, vp);
    let geometry = arena.int_sub(sized, w);
    let alias = arena.int_sub(kp, vp);
    let goal = arena.int_sub(actual, w);
    let equations = vec![
        normalize_int(&arena, geometry),
        normalize_int(&arena, alias),
    ];
    let identity = arena.identity();
    let established_zero_span =
        |goal: Poly, equations: &[Poly]| ExactZeros::new(identity, equations.to_vec()).spans(goal);
    assert!(established_zero_span(
        normalize_int(&arena, goal),
        &equations
    ));
    let one = arena.int(1);
    let off = arena.int_add(goal, one);
    assert!(!established_zero_span(
        normalize_int(&arena, off),
        &equations
    ));
    assert!(!established_zero_span(
        normalize_int(&arena, goal),
        &equations[..1]
    ));
    assert!(!established_zero_span(
        normalize_int(&arena, goal),
        &equations[1..]
    ));
}

/// `a <= b`
pub(crate) fn derive_le(arena: &mut ExprArena, facts: &Facts, a: IntExpr, b: IntExpr) -> Derivation {
    let d = arena.int_sub(b, a);
    derive_nonnegative(arena, facts, d)
}

/// `a < b`
pub(crate) fn derive_lt(arena: &mut ExprArena, facts: &Facts, a: IntExpr, b: IntExpr) -> Derivation {
    let d = arena.int_sub(b, a);
    let one = arena.int(1);
    let d = arena.int_sub(d, one);
    derive_nonnegative(arena, facts, d)
}

impl<'a> Engine<'a> {
    fn new(arena: &'a ExprArena, facts: &'a Facts) -> Engine<'a> {
        let existing = facts.caches().prepared.borrow().get(&arena.identity()).cloned();
        let prepared = existing.unwrap_or_else(|| {
            let exact = facts.exact_zeros(arena);
            let assumed = facts
                .assumed(arena)
                .into_iter()
                .map(|assumed| exact.apply(assumed))
                .collect();
            let prepared = PreparedFacts {
                exact,
                assumed: Rc::new(assumed),
                structural: Rc::default(),
                atom_signs: Rc::default(),
                uppers: Rc::default(),
                lowers: Rc::default(),
            };
            facts.caches().prepared.borrow_mut().insert(arena.identity(), prepared.clone());
            prepared
        });
        Engine {
            arena,
            facts,
            zeros: prepared.exact.zeros.clone(),
            exact: prepared.exact.clone(),
            assumed: prepared.assumed.clone(),
            prepared,
        }
    }

    fn uppers_of(&self, symbol: SymbolId) -> Rc<Vec<Poly>> {
        if let Some(known) = self.prepared.uppers.borrow().get(&symbol) {
            return known.clone();
        }
        let bounds = Rc::new(self.facts.uppers_of(self.arena, symbol));
        self.prepared.uppers.borrow_mut().insert(symbol, bounds.clone());
        bounds
    }

    fn lowers_of(&self, symbol: SymbolId) -> Rc<Vec<Poly>> {
        if let Some(known) = self.prepared.lowers.borrow().get(&symbol) {
            return known.clone();
        }
        let bounds = Rc::new(self.facts.lowers_of(self.arena, symbol));
        self.prepared.lowers.borrow_mut().insert(symbol, bounds.clone());
        bounds
    }

    /// Exact zero application leaves every polynomial over `p`'s atoms
    /// unchanged: none is a zero atom, a reducible quotient or a remainder
    /// established zero.
    fn inert(&self, p: &Poly) -> bool {
        p.terms
            .keys()
            .all(|monomial| monomial.keys().all(|atom| !self.exact.active(atom)))
    }

    /// `nonnegative_polynomial(apply_zero_facts(p - a))` from an empty cycle
    /// guard, for inert `p`, `a` and reached assumptions. Zero application
    /// is then the identity on the difference and on its assumption gaps, so
    /// both structural rules are decided on its coefficients directly.
    fn inert_difference_nonnegative(&self, p: &Poly, a: &Poly) -> bool {
        let Some(difference) = coefficient_difference(p.valid, p.terms.iter(), a) else {
            return false;
        };
        self.assumed.iter().any(|assumed| {
            coefficient_difference(true, difference.iter().map(|(m, c)| (*m, c)), assumed)
                .is_some_and(|gap| match gap.as_slice() {
                    [] => true,
                    [(monomial, constant)] => monomial.is_empty() && *constant >= 0,
                    _ => false,
                })
        }) || difference.iter().all(|(monomial, coefficient)| {
            *coefficient >= 0
                && monomial
                    .keys()
                    .all(|atom| self.nonnegative_atom(atom, &mut HashSet::new()))
        })
    }

    /// The memoized search: each subgoal is decided once per depth range
    /// under these facts.
    fn nonneg_steps(&self, e: Poly, depth: usize) -> bool {
        if !e.valid || depth == 0 {
            return false;
        }
        if let Some(memo) = self.facts.caches().memo.borrow().get(&e) {
            if memo.proved.is_some_and(|proved| proved <= depth) {
                return true;
            }
            if depth <= memo.failed {
                return false;
            }
        }
        let proved = self.nonneg_rules(e.clone(), depth);
        let caches = self.facts.caches();
        let mut memo = caches.memo.borrow_mut();
        let known = memo.entry(e).or_insert(Memo {
            proved: None,
            failed: 0,
        });
        let learned = proved && known.proved.is_none();
        if proved {
            known.proved = Some(known.proved.map_or(depth, |least| least.min(depth)));
        } else {
            known.failed = known.failed.max(depth);
        }
        drop(memo);
        if learned {
            self.facts.learned_sign();
        }
        proved
    }

    fn nonneg_rules(&self, mut e: Poly, depth: usize) -> bool {
        e = self.apply_zero_facts(e);

        // A reached assumption already establishes this exact proposition.
        // Consult it before speculative arithmetic rewrites; the final rule
        // below would rediscover the same fact through `e - assumed == 0`.
        if self.assumed.contains(&e) {
            return true;
        }
        // Try existing direct proofs before any recursive alternative. The
        // assumption subtraction below is the same rule used at the end;
        // its structural subgoal needs one less depth, exactly as before.
        if self.nonnegative_polynomial(&e, &mut HashSet::new()) {
            return true;
        }
        if depth > 1
            && self.assumed.iter().any(|assumed| {
                self.nonnegative_polynomial(
                    &self.apply_zero_facts(e.sub(assumed)),
                    &mut HashSet::new(),
                )
            })
        {
            return true;
        }
        // Composite zero consequences are useful for remaining required
        // obligations, after the inexpensive existing positive rules.
        if self.exact.spans(e.clone()) {
            return true;
        }
        if depth > 0 {
            for atom in e.atoms() {
                let Atom::Quot(n, d) = &atom else {
                    continue;
                };
                if !established_polynomial_nonnegative(
                    self.arena,
                    self.facts,
                    d.sub(&Poly::constant(1)),
                ) {
                    continue;
                }
                let mut coefficient = Poly::default();
                let mut rest = Poly::default();
                let mut linear = true;
                for (monomial, value) in &e.terms {
                    let mut monomial = monomial.clone();
                    match monomial.remove(&atom) {
                        None => rest.insert(monomial, *value),
                        Some(1) => coefficient.insert(monomial, *value),
                        Some(_) => {
                            linear = false;
                            break;
                        }
                    }
                }
                let expected = Poly::constant(1).sub(d);
                let same = self.apply_zero_facts(coefficient.sub(&expected));
                if linear
                    && established_polynomial_nonnegative(
                        self.arena,
                        self.facts,
                        n.sub(&Poly::constant(1)),
                    )
                    && same.valid
                    && (same.is_zero() || self.exact.spans(same))
                    && self.nonneg_steps(rest.sub(&n.sub(&Poly::constant(1))), depth - 1)
                {
                    return true;
                }
            }
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
                    Atom::Symbol(s) => (*self.uppers_of(*s)).clone(),
                    Atom::Foreign(f) => match self.arena.view(f.node) {
                        NodeView::Binary {
                            op: BinaryOp::Min,
                            lhs: AnyExpr::Int(lhs),
                            rhs: AnyExpr::Int(rhs),
                        } => vec![
                            normalize_int(self.arena, lhs),
                            normalize_int(self.arena, rhs),
                        ],
                        _ => Vec::new(),
                    },
                    Atom::Rem(..) | Atom::Quot(..) => Vec::new(),
                },
                Monotone::Lower => match &atom {
                    Atom::Symbol(s) => (*self.lowers_of(*s)).clone(),
                    Atom::Foreign(f) => match self.arena.view(f.node) {
                        NodeView::Binary {
                            op: BinaryOp::Max,
                            lhs: AnyExpr::Int(lhs),
                            rhs: AnyExpr::Int(rhs),
                        } => vec![
                            normalize_int(self.arena, lhs),
                            normalize_int(self.arena, rhs),
                        ],
                        _ => Vec::new(),
                    },
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
        // For positive d,k and integer r, r-k*floor(n/d)>=0 iff
        // n <= d*(floor(r/k)+1)-1. Both quotients are Euclidean;
        // checked fixed-word division reaches this fragment only for
        // nonnegative numerators and positive denominators.
        for atom in e.atoms() {
            let Atom::Quot(n, d) = &atom else {
                continue;
            };
            let Some((coefficient, rest)) = e.linear_in(&atom) else {
                continue;
            };
            let Some(scale) = coefficient.checked_neg().filter(|value| *value > 0) else {
                continue;
            };
            // Introduce at most one floor over a quotient-free polynomial.
            // Reapplying this rewrite to residual quotient atoms would grow
            // nested floors without a decreasing symbolic measure.
            if d.as_constant().is_none_or(|divisor| divisor <= 0)
                || rest
                    .atoms()
                    .iter()
                    .any(|atom| !matches!(atom, Atom::Symbol(_)))
            {
                continue;
            }
            // Remove the known constant residue before introducing a floor.
            // If the remaining polynomial is an established multiple of k,
            // floor((multiple + residue)/k) is exactly multiple/k, including
            // a negative constant offset such as C-1 with even C.
            let residue = rest.constant_term().rem_euclid(scale);
            let aligned = rest.sub(&Poly::constant(residue));
            let scale = Poly::constant(scale);
            if !self.apply_zero_facts(aligned.rem(&scale)).is_zero() {
                continue;
            }
            let bound = aligned.quot(&scale);
            if self.positive_denominator_on_defined_atom(d)
                && self.nonneg_steps(
                    d.mul(&bound.add(&Poly::constant(1)))
                        .sub(&Poly::constant(1))
                        .sub(n),
                    depth - 1,
                )
            {
                return true;
            }
        }
        // An exact remainder inside its established fundamental interval
        // is its numerator. The subgoals have no quotient/remainder atoms,
        // so this cannot introduce recursive speculative floor expressions.
        for atom in e.atoms() {
            let Atom::Rem(numerator, denominator) = &atom else {
                continue;
            };
            if denominator.as_constant().is_none_or(|value| value <= 0)
                || numerator
                    .atoms()
                    .iter()
                    .any(|atom| matches!(atom, Atom::Quot(..) | Atom::Rem(..)))
            {
                continue;
            }
            let upper = denominator.sub(&Poly::constant(1)).sub(numerator);
            if self.nonneg_steps(numerator.as_ref().clone(), depth - 1)
                && self.nonneg_steps(upper, depth - 1)
                && self.nonneg_steps(e.subst(&atom, numerator), depth - 1)
            {
                return true;
            }
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
        for substituted in substitutions {
            if self.nonneg_steps(substituted, depth - 1) {
                return true;
            }
        }
        // `e = (e - f) + f` for an assumed nonnegative `f`.
        for assumed in Rc::clone(&self.assumed).iter() {
            if self.nonneg_steps(e.sub(assumed), depth - 1) {
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
                            .lowers_of(*symbol)
                            .iter()
                            .find(|lower| self.nonnegative_polynomial(lower, &mut HashSet::new()))
                            .cloned(),
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
        let r = Poly::atom(Atom::Rem(Rc::new(n.clone()), Rc::new(d.clone())));
        let replacement = d.mul(&Poly::atom(q.clone())).add(&r);
        let out = e.subst(&symbol, &replacement);
        if out == *e {
            None
        } else {
            Some(out)
        }
    }

    fn apply_zero_facts(&self, e: Poly) -> Poly {
        self.exact.apply(e)
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
            // A signed affine term is monotone according to its coefficient;
            // positivity is needed only for products and higher powers.
            let affine = m.len() == 1 && m.values().next() == Some(&1);
            if !affine
                && m.keys()
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

    fn word_dtype(&self, node: AnyExpr) -> Option<DType> {
        match self.arena.view(node) {
            NodeView::ScalarInteger { operation, operands } => {
                let AnyExpr::Int(value) = node else { return None; };
                if !self.facts.scalar_defined(self.arena, value) { return None; }
                let types = operands.iter().map(|operand| operand.dtype(self.arena)).collect::<Vec<_>>();
                Some(crate::reference_math::scalar_recipe(operation, &types).output().ty())
            }
            NodeView::Unary { op: crate::expr::UnaryOp::IntFromScalar, operand: AnyExpr::Scalar(value) } => {
                Some(ScalarOperand::Scalar(value).dtype(self.arena))
            }
            _ => None,
        }
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
                if let Some(dtype) = self.word_dtype(f.node) {
                    return match dtype {
                        DType::I32 => (Poly::constant(i64::from(i32::MIN)), Some(Poly::constant(i64::from(i32::MAX)))),
                        DType::U32 => (Poly::constant(0), Some(Poly::constant(i64::from(u32::MAX)))),
                        _ => (Poly::atom(a.clone()), Some(Poly::atom(a.clone()))),
                    };
                }
                if let NodeView::Binary {
                    op,
                    lhs: AnyExpr::Int(lhs),
                    rhs: AnyExpr::Int(rhs),
                } = self.arena.view(f.node)
                {
                    let left = normalize_int(self.arena, lhs);
                    let right = normalize_int(self.arena, rhs);
                    if op == BinaryOp::Min {
                        let lower = if self.nonnegative_polynomial(&left, &mut HashSet::new())
                            && self.nonnegative_polynomial(&right, &mut HashSet::new())
                        {
                            Poly::constant(0)
                        } else {
                            Poly::atom(a.clone())
                        };
                        let (_, left_upper) = self.interval_over(&left, bound);
                        let (_, right_upper) = self.interval_over(&right, bound);
                        // Either operand independently bounds min from above.
                        // Prefer an established tighter endpoint, or a concrete
                        // bound when symbolic endpoints are not ordered.
                        let upper = if self.nonnegative_polynomial(
                            &self.apply_zero_facts(left_upper.sub(&right_upper)),
                            &mut HashSet::new(),
                        ) || (right_upper.as_constant().is_some()
                            && left_upper.as_constant().is_none())
                        {
                            right_upper
                        } else {
                            left_upper
                        };
                        return (lower, Some(upper));
                    }
                    if op == BinaryOp::Max {
                        let (left_lower, _) = self.interval_over(&left, bound);
                        let (right_lower, _) = self.interval_over(&right, bound);
                        // Either operand independently bounds max from below.
                        let lower = if self.nonnegative_polynomial(
                            &self.apply_zero_facts(right_lower.sub(&left_lower)),
                            &mut HashSet::new(),
                        ) || (right_lower.as_constant().is_some()
                            && left_lower.as_constant().is_none())
                        {
                            right_lower
                        } else {
                            left_lower
                        };
                        return (lower, Some(Poly::atom(a.clone())));
                    }
                }
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
        // The search restores `visiting` on return, so from an empty guard
        // the result depends on the value and the unchanged facts alone.
        if !visiting.is_empty() {
            return self.structural_nonnegative(value, visiting);
        }
        if let Some(known) = self.prepared.structural.borrow().get(value) {
            return *known;
        }
        let known = self.structural_nonnegative(value, visiting);
        self.prepared.structural.borrow_mut().insert(value.clone(), known);
        known
    }

    fn structural_nonnegative(&self, value: &Poly, visiting: &mut HashSet<SymbolId>) -> bool {
        value.valid
            && (self.assumed.iter().any(|assumed| {
                // Exact reached facts imply their weaker constant bounds,
                // without recursive sign queries or another proof search.
                self.apply_zero_facts(value.sub(assumed))
                    .as_constant()
                    .is_some_and(|gap| gap >= 0)
            }) || value.terms.iter().all(|(factors, coefficient)| {
                *coefficient >= 0
                    && factors
                        .keys()
                        .all(|factor| self.nonnegative_atom(factor, visiting))
            }))
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
        let boundary = Atom::Quot(Rc::new(minus_one.clone()), Rc::new(denominator.clone()));
        quotient.subst(&boundary, &minus_one)
    }

    fn nonnegative_atom(&self, atom: &Atom, visiting: &mut HashSet<SymbolId>) -> bool {
        // As for polynomials: from an empty guard the sign is a function of
        // the atom and the unchanged facts alone.
        if !visiting.is_empty() {
            return self.atom_sign(atom, visiting);
        }
        if let Some(known) = self.prepared.atom_signs.borrow().get(atom) {
            return *known;
        }
        let known = self.atom_sign(atom, visiting);
        self.prepared
            .atom_signs
            .borrow_mut()
            .insert(atom.clone(), known);
        known
    }

    fn atom_sign(&self, atom: &Atom, visiting: &mut HashSet<SymbolId>) -> bool {
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
                            .filter(|(bounded, lower)| {
                                bounded == symbol && self.facts.lower.get(symbol) != Some(lower)
                            })
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
                if self.word_dtype(f.node) == Some(DType::U32) { return true; }
                if let NodeView::Binary {
                    op,
                    lhs: AnyExpr::Int(lhs),
                    rhs: AnyExpr::Int(rhs),
                } = self.arena.view(f.node)
                {
                    let left =
                        self.nonnegative_polynomial(&normalize_int(self.arena, lhs), visiting);
                    let right =
                        self.nonnegative_polynomial(&normalize_int(self.arena, rhs), visiting);
                    if op == BinaryOp::Min {
                        return left && right;
                    }
                    if op == BinaryOp::Max {
                        return left || right;
                    }
                }
                matches!(
                    f.node,
                    AnyExpr::Nat(_) | AnyExpr::Bool(_) | AnyExpr::Duration(_)
                )
            }
            Atom::Quot(numerator, denominator) => {
                self.nonnegative_polynomial(numerator, visiting)
                    // Denominator positivity is another dependency of this same sign
                    // query. Keep active bound symbols so cyclic premises cannot
                    // justify themselves; other bounds remain available.
                    && self.nonnegative_polynomial(denominator, visiting)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_atom_deletion_matches_substitution_without_entering_opaque_atoms() {
        let mut arena = ExprArena::new();
        let (_, x_symbol, x) = arena.loop_binder();
        let (_, y_symbol, y) = arena.loop_binder();
        let (_, _, absent) = arena.loop_binder();
        let xp = Poly::symbol(x_symbol);
        let yp = Poly::symbol(y_symbol);
        let opaque = Poly::atom(Atom::Quot(
            Rc::new(xp.mul(&yp)),
            Rc::new(Poly::constant(3)),
        ))
        .add(&Poly::atom(Atom::Rem(
            Rc::new(xp.mul(&yp)),
            Rc::new(Poly::constant(5)),
        )));
        let mixed = xp
            .mul(&xp)
            .mul(&xp)
            .mul(&yp)
            .mul(&Poly::constant(-7))
            .add(&yp.mul(&yp).mul(&Poly::constant(i64::MAX)))
            .add(&opaque)
            .add(&Poly::constant(-11));
        for zero_values in [vec![], vec![absent], vec![x, absent], vec![x, y, absent]] {
            let mut facts = Facts::new();
            for value in zero_values {
                facts.assume_zero(&mut arena, value);
            }
            let engine = Engine::new(&arena, &facts);
            for input in [mixed.clone(), opaque.clone(), Poly::constant(-1)] {
                let mut previous = input.clone();
                for zero in engine.zeros.iter() {
                    if let Some(atom) = single_atom_of(zero) {
                        previous = previous.subst(&atom, &Poly::constant(0));
                    }
                }
                assert_eq!(engine.apply_zero_facts(input), previous);
            }
            let invalid = Poly {
                terms: mixed.terms.clone(),
                valid: false,
            };
            assert!(!engine.apply_zero_facts(invalid).valid);
        }
    }

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
    fn optional_direct_inference_does_not_replace_required_search() {
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let h = dimension(&mut arena, &mut facts);
        let kv = dimension(&mut arena, &mut facts);
        let quotient = arena.int_div(h, kv);
        assert!(directly_nonnegative(&arena, &facts, quotient));
        let zero = arena.int(0);
        let negative = arena.int_sub(zero, quotient);
        assert!(!directly_nonnegative(&arena, &facts, negative));

        let (_, _, x) = arena.loop_binder();
        let (_, _, y) = arena.loop_binder();
        let difference = arena.int_sub(x, y);
        facts.assume_nonnegative(&mut arena, difference);
        assert!(!directly_nonnegative(&arena, &facts, difference));
        assert!(nonneg(&mut arena, &facts, difference));
    }

    #[test]
    fn positive_monomials_are_bounded_by_their_atoms_lower_bounds() {
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let a = dimension(&mut arena, &mut facts);
        let one = arena.int(1);
        let square = arena.int_mul(a, a);
        let square = arena.int_sub(square, one);
        assert!(nonneg(&mut arena, &facts, square), "A*A >= 1");
        let (t, p, c) = (
            dimension(&mut arena, &mut facts),
            dimension(&mut arena, &mut facts),
            dimension(&mut arena, &mut facts),
        );
        let patch = arena.int_mul(t, p);
        let patch = arena.int_mul(patch, p);
        let patch = arena.int_mul(patch, c);
        let patch = arena.int_sub(patch, one);
        assert!(nonneg(&mut arena, &facts, patch), "T*P*P*C >= 1");
        let (_, _, signed) = arena.loop_binder();
        let signed_square = arena.int_mul(signed, a);
        let signed_square = arena.int_sub(signed_square, one);
        assert!(
            !nonneg(&mut arena, &facts, signed_square),
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
            !nonneg(&mut arena, &facts, at_least_one),
            "H / KV >= 1 needs H >= KV"
        );
        let difference = arena.int_sub(h, kv);
        facts.assume_nonnegative(&mut arena, difference);
        assert!(
            nonneg(&mut arena, &facts, at_least_one),
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
        assert!(nonneg(&mut arena, &facts, at_least_one));
        assert!(
            nonneg(&mut arena, &facts, doubled),
            "2*(H/KV) - 2 >= 0 from H/KV >= 1"
        );
    }

    #[test]
    fn reached_branch_retains_every_unexcluded_recipe_alternative() {
        use crate::checked::{check_source, SourceFile, SourceSet};
        let source = r#"fn packed[M, C, F](x: &tensor[M] f32) -> f32 where C > 0 and (F == 1 or F == 2):
    return x[0]

fn run[M, C, F](x: &tensor[M] f32) -> f32 where C >= 0 and F >= 0 and (F == 0 or F == 1 or F == 2) and (F == 0 or C > 0):
    let mut value = 0.0
    if M <= 64 or F == 0:
        value = x[0]
    else:
        value = packed[C = C, F = F](x)
    return value
"#;
        check_source(SourceSet::new(vec![SourceFile {
            path: "reached-recipes.seismic".into(),
            text: source.into(),
        }]))
        .unwrap();
        for invalid in [
            source.replace(" and (F == 0 or C > 0)", ""),
            source.replace("C > 0 and (F == 1 or F == 2)", "C > 0 and F == 1"),
        ] {
            assert!(check_source(SourceSet::new(vec![SourceFile {
                path: "invalid-recipes.seismic".into(),
                text: invalid,
            }]))
            .is_err());
        }
    }

    #[test]
    fn clamped_team_recipe_preserves_positive_helper_contract() {
        use crate::checked::{check_source, SourceFile, SourceSet};
        let source = r#"fn tile[W, TEAMS](x: &tensor[W] f32) -> f32:
    return x[0]

fn run[W](x: &tensor[W] f32) -> f32:
    return tile[TEAMS = max(1, 256 / min(W, 256))](x)
"#;
        check_source(SourceSet::new(vec![SourceFile {
            path: "teams.seismic".into(),
            text: source.into(),
        }]))
        .unwrap();
        let invalid = source.replace("max(1, 256 / min(W, 256))", "max(0, 256 / (W + 1))");
        assert!(check_source(SourceSet::new(vec![SourceFile {
            path: "invalid-teams.seismic".into(),
            text: invalid,
        }]))
        .is_err());
    }

    #[test]
    fn exact_reached_assumptions_need_no_arithmetic_search() {
        let mut arena = ExprArena::new();
        let x_symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
        let y_symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
        let x = arena.int_symbol(x_symbol);
        let y = arena.int_symbol(y_symbol);
        let eight = arena.int(8);
        let two = arena.int(2);
        let divided = arena.int_div(y, eight);
        let doubled = arena.int_mul(two, divided);
        let reached = arena.int_sub(x, doubled);
        let mut facts = Facts::new();
        facts.assume_nonnegative(&mut arena, reached);
        let goal = normalize_int(&arena, reached);
        assert!(Engine::new(&arena, &facts).nonneg_steps(goal.clone(), 1));
        assert!(!Engine::new(&arena, &Facts::new()).nonneg_steps(goal.clone(), 1));
        assert!(!Engine::new(&arena, &facts).nonneg_steps(goal.sub(&Poly::constant(1)), 1));
        assert!(!Engine::new(&arena, &facts).nonneg_steps(goal.neg(), 1));
    }

    #[test]
    fn direct_structural_proofs_precede_recursive_alternatives_without_losing_vacuity() {
        let mut arena = ExprArena::new();
        let x_symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
        let y_symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
        let z_symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
        let x = arena.int_symbol(x_symbol);
        let y = arena.int_symbol(y_symbol);
        let z = arena.int_symbol(z_symbol);
        let eight = arena.int(8);
        let two = arena.int(2);
        let one = arena.int(1);
        let seven = arena.int(7);
        let divided = arena.int_div(y, eight);
        let doubled = arena.int_mul(two, divided);
        let reached = arena.int_sub(x, doubled);
        let square = arena.int_mul(z, z);
        let unrelated = arena.int_sub(square, seven);
        let mut facts = Facts::new();
        facts.assume_nonnegative(&mut arena, unrelated);
        facts.assume_nonnegative(&mut arena, reached);
        let goal = arena.int_add(reached, one);
        assert!(nonneg(&mut arena, &facts, goal));
        let normalized = normalize_int(&arena, goal);
        assert!(facts.caches().memo.borrow()[&normalized].proved.unwrap() <= 2);
        let too_strong = arena.int_sub(reached, one);
        assert!(!nonneg(&mut arena, &facts, too_strong));

        // There is deliberately no negative pruning: contradictory reached
        // assumptions still supply the existing vacuous derivation.
        let minus_one = arena.int(-1);
        let zero = arena.int(0);
        let negated = arena.int_sub(zero, unrelated);
        let contrary = arena.int_sub(negated, one);
        facts.assume_nonnegative(&mut arena, contrary);
        assert!(nonneg(&mut arena, &facts, minus_one));
    }

    #[test]
    fn min_max_bounds_use_only_the_ordering_the_operator_establishes() {
        let mut arena = ExprArena::new();
        let facts = Facts::new();
        let (_, _, signed) = arena.loop_binder();
        let one = arena.int(1);
        let maximum = arena.int_max(one, signed);
        let minimum = arena.int_min(one, signed);
        assert!(derive_le(&mut arena, &facts, one, maximum).is_established());
        assert!(derive_le(&mut arena, &facts, signed, maximum).is_established());
        assert!(derive_le(&mut arena, &facts, minimum, one).is_established());
        assert!(derive_le(&mut arena, &facts, minimum, signed).is_established());
        assert!(!derive_le(&mut arena, &facts, one, minimum).is_established());
        assert!(!derive_le(&mut arena, &facts, maximum, one).is_established());
        let two = arena.int(2);
        assert!(!derive_le(&mut arena, &facts, two, maximum).is_established());
    }

    #[test]
    fn packed_group_bounds_keep_the_last_legal_element() {
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let n = dimension(&mut arena, &mut facts);
        let (_, symbol, i) = arena.loop_binder();
        let four = arena.int(4);
        let extent = arena.int_mul(n, four);
        let one = arena.int(1);
        let last = arena.int_sub(extent, one);
        let start = arena.int(0);
        facts.set_range(symbol, start, last);
        let group = arena.int_div(i, four);
        assert!(derive_lt(&mut arena, &facts, group, n).is_established());
        facts.set_range(symbol, start, extent);
        assert!(!derive_lt(&mut arena, &facts, group, n).is_established());
    }

    #[test]
    fn paired_codec_coefficients_follow_exact_storage_geometry() {
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let w = dimension(&mut arena, &mut facts);
        let c = dimension(&mut arena, &mut facts);
        let sixteen = arena.int(16);
        let storage = arena.int_mul(c, sixteen);
        let relation = arena.int_sub(w, storage);
        facts.assume_zero(&mut arena, relation);
        // Match the source checker's equality fact: both directions are
        // established nonnegative bounds as well as a zero relation.
        facts.assume_nonnegative(&mut arena, relation);
        let origin = arena.int(0);
        let reverse = arena.int_sub(origin, relation);
        facts.assume_nonnegative(&mut arena, reverse);
        let thirty_two = arena.int(32);
        let aligned = arena.int_rem(w, thirty_two);
        facts.assume_zero(&mut arena, aligned);
        let two = arena.int(2);
        let parity = arena.int_rem(c, two);
        assert!(zero(&mut arena, &facts, parity));
        let (_, symbol, i) = arena.loop_binder();
        let one = arena.int(1);
        let last = arena.int_sub(w, one);
        let start = arena.int(0);
        facts.set_range(symbol, start, last);
        let group = arena.int_div(i, thirty_two);
        let coefficient = arena.int_mul(group, two);
        assert!(
            derive_lt(&mut arena, &facts, coefficient, c).is_established(),
            "first coefficient has odd residual C-1"
        );
        let second = arena.int_add(coefficient, one);
        assert!(derive_lt(&mut arena, &facts, second, c).is_established());
        facts.set_range(symbol, start, w);
        assert!(!derive_lt(&mut arena, &facts, coefficient, c).is_established());
        assert!(!derive_lt(&mut arena, &facts, second, c).is_established());
    }

    #[test]
    fn constant_divisibility_implies_only_actual_divisors() {
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let k = dimension(&mut arena, &mut facts);
        let large = arena.int(256);
        let known = arena.int_rem(k, large);
        facts.assume_zero(&mut arena, known);
        let small = arena.int(32);
        let implied = arena.int_rem(k, small);
        assert!(zero(&mut arena, &facts, implied));
        let unrelated = arena.int(48);
        let not_implied = arena.int_rem(k, unrelated);
        assert!(!zero(&mut arena, &facts, not_implied));
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
        assert!(derive_lt(&mut arena, &facts, row, rows).is_established());
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
            !nonneg(&mut arena, &facts, e),
            "prover claims (i - 1) * k >= 0 for i in [1, 9], k <= -1"
        );
        // With a nonnegative cofactor the lower bound of `i` still applies.
        let (_, n_symbol, n) = arena.loop_binder();
        let zero = arena.int(0);
        facts.set_range_lower(n_symbol, zero);
        let product = arena.int_mul(i, n);
        let e = arena.int_sub(product, n);
        assert!(
            nonneg(&mut arena, &facts, e),
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
        let first: Vec<bool> = forward.iter().map(|g| nonneg(&mut arena, &facts, *g)).collect();
        let mut arena = ExprArena::new();
        let mut facts = Facts::new();
        let backward = goals(&mut arena, &mut facts);
        let mut second: Vec<bool> = backward
            .iter()
            .rev()
            .map(|g| nonneg(&mut arena, &facts, *g))
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
        facts.assume_nonzero(&mut arena, m);
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
        assert!(derive_le(&mut arena, &after, zero, joined).is_established());
        assert!(derive_le(&mut arena, &after, joined, nine).is_established());
        assert!(!derive_le(&mut arena, &after, joined, seven).is_established());
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
        assert!(derive_le(&mut arena, &after, zero, joined).is_established());
        assert!(derive_le(&mut arena, &after, joined, last).is_established());
        assert!(
            !derive_le(&mut arena, &after, joined, i).is_established(),
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
        assert!(is_zero(&mut arena, lo));
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
                ScalarOperand::Word { dtype: DType::I32, value: input },
                ScalarOperand::Word { dtype: DType::I32, value: one },
            ],
        );
        let normalized = normalize_int(&arena, wrapped);
        let restored = intern(&mut arena, &normalized);
        assert!(same(&arena, restored, wrapped));
        assert!(!nonneg(&mut arena, &Facts::new(), wrapped));
        let (lo, hi) = interval(&mut arena, &Facts::new(), wrapped);
        let minimum = arena.int(i64::from(i32::MIN));
        let maximum = arena.int(i64::from(i32::MAX));
        assert!(same(&arena, lo, minimum));
        assert!(same(&arena, hi, maximum));
    }
}

#[cfg(test)]
#[test]
fn storage_equality_is_available_to_the_required_quotient_bound() {
    let mut arena = ExprArena::new();
    let mut facts = Facts::new();
    let (_, ws, w) = arena.loop_binder();
    let (_, ns, n) = arena.loop_binder();
    let zero = arena.int(0);
    let one = arena.int(1);
    facts.set_range_lower(ws, one);
    facts.set_range_lower(ns, one);
    let eight = arena.int(8);
    let actual_width = arena.int_mul(eight, n);
    let equality = arena.int_sub(actual_width, w);
    facts.assume_zero(&mut arena, equality);
    facts.assume_nonnegative(&mut arena, equality);
    let neg = arena.int_sub(zero, equality);
    facts.assume_nonnegative(&mut arena, neg);
    let (_, is, i) = arena.loop_binder();
    let last = arena.int_sub(w, one);
    facts.set_range(is, zero, last);
    let index = arena.int_div(i, eight);
    let goal = arena.int_sub(n, one);
    let goal = arena.int_sub(goal, index);
    assert!(nonneg(&mut arena, &facts, goal));
    let invalid = arena.int_sub(goal, one);
    assert!(!nonneg(&mut arena, &facts, invalid));
}

#[cfg(test)]
#[test]
fn euclidean_product_bound_requires_positive_numerator_and_denominator() {
    let mut arena = ExprArena::new();
    let mut facts = Facts::new();
    let (_, ns, n) = arena.loop_binder();
    let (_, ds, d) = arena.loop_binder();
    let one = arena.int(1);
    let zero = arena.int(0);
    facts.set_range_lower(ns, one);
    facts.set_range_lower(ds, one);
    let q = arena.int_div(n, d);
    let dm = arena.int_sub(d, one);
    let product = arena.int_mul(dm, q);
    let upper = arena.int_sub(n, one);
    let goal = arena.int_sub(upper, product);
    assert!(nonneg(&mut arena, &facts, goal));
    let tighter = arena.int_sub(goal, one);
    assert!(!nonneg(&mut arena, &facts, tighter));
    let mut no_positive_n = facts.clone();
    no_positive_n.set_range_lower(ns, zero);
    assert!(!nonneg(&mut arena, &no_positive_n, goal));
    let mut no_positive_d = Facts::new();
    no_positive_d.set_range_lower(ns, one);
    assert!(!nonneg(&mut arena, &no_positive_d, goal));
}

fn finite_choice(arena: &ExprArena, value: BoolExpr) -> Option<(SymbolId, Vec<i64>)> {
    if let Some((NaryOp::Any, parts)) = Facts::boolean_operands(arena, value) {
        let mut combined: Option<(SymbolId, Vec<i64>)> = None;
        for part in parts {
            let (symbol, values) = finite_choice(arena, part)?;
            match &mut combined {
                None => combined = Some((symbol, values)),
                Some((prior, all)) if *prior == symbol => all.extend(values),
                _ => return None,
            }
        }
        return combined;
    }
    let NodeView::Cmp {
        op: CmpOp::Eq,
        lhs: AnyExpr::Int(lhs),
        rhs: AnyExpr::Int(rhs),
    } = arena.view(value.into())
    else {
        return None;
    };
    let (expression, value) = if let Some(value) = constant(arena, rhs) {
        (lhs, value)
    } else {
        (rhs, constant(arena, lhs)?)
    };
    let Atom::Symbol(symbol) = single_atom_of(&normalize_int(arena, expression))? else {
        return None;
    };
    Some((symbol, vec![value]))
}
fn collect_finite_choices(
    arena: &ExprArena,
    value: BoolExpr,
    needed: &[SymbolId],
    domains: &mut BTreeMap<SymbolId, Vec<i64>>,
) {
    if let Some((NaryOp::All, parts)) = Facts::boolean_operands(arena, value) {
        for part in parts {
            collect_finite_choices(arena, part, needed, domains);
        }
    } else if let Some((symbol, mut values)) = finite_choice(arena, value) {
        if !needed.contains(&symbol) {
            return;
        }
        values.sort_unstable();
        values.dedup();
        match domains.get_mut(&symbol) {
            Some(existing) => existing.retain(|value| values.contains(value)),
            None => {
                domains.insert(symbol, values);
            }
        }
    }
}

/// Keep finite recipe correlations while leaving ordinary symbolic coefficients
/// symbolic. Every case transfers the original facts and uses the existing
/// required proof; a missing finite domain or budget overflow proves nothing.
fn finite_case_boolean(source: &ExprArena, facts: &Facts, goal: BoolExpr) -> bool {
    let needed = source.free_symbols(goal.into());
    let mut domains = BTreeMap::new();
    for known in &facts.boolean {
        collect_finite_choices(source, *known, &needed, &mut domains);
    }
    let mut selected = Vec::new();
    let mut cases = 1usize;
    for symbol in needed {
        let Some(values) = domains.remove(&symbol) else {
            continue;
        };
        if values.is_empty()
            || !matches!(
                source.symbol_sort(symbol),
                crate::expr::SymbolSort::Int | crate::expr::SymbolSort::Nat
            )
        {
            return false;
        }
        let Some(count) = cases
            .checked_mul(values.len())
            .filter(|count| *count <= 256)
        else {
            return false;
        };
        cases = count;
        selected.push((symbol, values));
    }
    if selected.is_empty() {
        return false;
    }
    fn every(
        source: &ExprArena,
        facts: &Facts,
        goal: BoolExpr,
        remaining: &[(SymbolId, Vec<i64>)],
        assignments: &mut BTreeMap<SymbolId, i64>,
    ) -> bool {
        if let Some(((symbol, values), rest)) = remaining.split_first() {
            return values.iter().all(|value| {
                assignments.insert(*symbol, *value);
                every(source, facts, goal, rest, assignments)
            });
        }
        let mut destination = ExprArena::new();
        let mut symbols = BTreeMap::new();
        // Unsupported sorts conservatively leave this proof to the normal path.
        let mut needed = source.free_symbols(goal.into());
        for expression in facts
            .zero
            .iter()
            .chain(&facts.assumed)
            .chain(&facts.nonzero)
            .chain(facts.lower.values())
            .chain(facts.upper.values())
            .chain(facts.extra_lower.iter().map(|(_, bound)| bound))
            .chain(facts.extra_upper.iter().map(|(_, bound)| bound))
        {
            needed.extend(source.free_symbols((*expression).into()));
        }
        needed.extend(facts.lower.keys().chain(facts.upper.keys()).copied());
        needed.extend(
            facts
                .extra_lower
                .iter()
                .chain(&facts.extra_upper)
                .map(|(symbol, _)| *symbol),
        );
        for expression in &facts.boolean {
            needed.extend(source.free_symbols((*expression).into()));
        }
        needed.sort_unstable();
        needed.dedup();
        for symbol in needed {
            let sort = source.symbol_sort(symbol);
            let expression = match (sort, assignments.get(&symbol)) {
                (crate::expr::SymbolSort::Int, Some(value)) => destination.int(*value).into(),
                (crate::expr::SymbolSort::Nat, Some(value)) => {
                    let Ok(value) = u64::try_from(*value) else {
                        return false;
                    };
                    destination.nat(value).into()
                }
                (crate::expr::SymbolSort::Int, None) => {
                    let fresh = destination.checker_variable(sort);
                    destination.int_symbol(fresh).into()
                }
                (crate::expr::SymbolSort::Nat, None) => {
                    let fresh = destination.checker_variable(sort);
                    destination.nat_symbol(fresh).into()
                }
                _ => return false,
            };
            symbols.insert(symbol, expression);
        }
        let mut map = |symbol: SymbolId, _: &mut ExprArena| symbols[&symbol];
        let mut transferred = Facts::new();
        for (symbol, bound) in facts.lower_bounds() {
            let value = match symbols[&symbol] {
                AnyExpr::Int(value) => value,
                AnyExpr::Nat(value) => destination.int_from_nat(value),
                _ => return false,
            };
            let bound = super::xfer::transfer_int(source, bound, &mut destination, &mut map);
            let difference = destination.int_sub(value, bound);
            transferred.assume_nonnegative(&mut destination, difference);
        }
        for (symbol, bound) in facts.upper_bounds() {
            let value = match symbols[&symbol] {
                AnyExpr::Int(value) => value,
                AnyExpr::Nat(value) => destination.int_from_nat(value),
                _ => return false,
            };
            let bound = super::xfer::transfer_int(source, bound, &mut destination, &mut map);
            let difference = destination.int_sub(bound, value);
            transferred.assume_nonnegative(&mut destination, difference);
        }
        for (binding, meaning) in &facts.definitions {
            let AnyExpr::Int(value) = symbols[binding] else { return false; };
            let meaning = super::xfer::transfer_int(source, *meaning, &mut destination, &mut map);
            let difference = destination.int_sub(value, meaning);
            transferred.assume_zero(&destination, difference);
        }
        for expression in &facts.zero {
            let value = super::xfer::transfer_int(source, *expression, &mut destination, &mut map);
            transferred.assume_zero(&destination, value);
        }
        for expression in &facts.assumed {
            let value = super::xfer::transfer_int(source, *expression, &mut destination, &mut map);
            transferred.assume_nonnegative(&mut destination, value);
        }
        for expression in &facts.nonzero {
            let value = super::xfer::transfer_int(source, *expression, &mut destination, &mut map);
            transferred.assume_nonzero(&destination, value);
        }
        for expression in &facts.boolean {
            let value = super::xfer::transfer_bool(source, *expression, &mut destination, &mut map);
            transferred.assume_boolean(value);
        }
        transferred.refine_booleans(&mut destination);
        let value = super::xfer::transfer_bool(source, goal, &mut destination, &mut map);
        transferred.boolean(&mut destination, value)
    }
    every(source, facts, goal, &selected, &mut BTreeMap::new())
}

/// Exact exhaustive evidence from whole established finite domains. A budget
/// limit or an undefined member leaves the ordinary proof authoritative.
fn finite_domain_nonnegative(arena: &ExprArena, facts: &Facts, goal: IntExpr) -> bool {
    const MAX_CASES: usize = 256;
    let needed = symbols(arena, goal);
    if needed.is_empty() {
        return false;
    }
    let mut domains = BTreeMap::new();
    for known in &facts.boolean {
        collect_finite_choices(arena, *known, &needed, &mut domains);
    }
    let mut cases = 1usize;
    let mut selected = Vec::new();
    for symbol in needed {
        if !matches!(
            arena.symbol_sort(symbol),
            crate::expr::SymbolSort::Int | crate::expr::SymbolSort::Nat
        ) {
            return false;
        }
        let Some(values) = domains.remove(&symbol) else {
            return false;
        };
        // An inconsistent domain is not used as a shortcut for vacuity here.
        if values.is_empty() {
            return false;
        }
        let Some(count) = cases
            .checked_mul(values.len())
            .filter(|count| *count <= MAX_CASES)
        else {
            return false;
        };
        cases = count;
        selected.push((symbol, values));
    }
    fn every(
        arena: &ExprArena,
        goal: IntExpr,
        domains: &[(SymbolId, Vec<i64>)],
        values: &mut crate::expr::Assignment,
    ) -> bool {
        let Some(((symbol, choices), remaining)) = domains.split_first() else {
            return arena
                .eval_int_i64(goal, values)
                .is_ok_and(|value| value >= 0);
        };
        choices.iter().all(|value| {
            let assigned = match arena.symbol_sort(*symbol) {
                crate::expr::SymbolSort::Int => crate::expr::SymbolValue::Int((*value).into()),
                crate::expr::SymbolSort::Nat => {
                    let Ok(value) = u64::try_from(*value) else {
                        return false;
                    };
                    crate::expr::SymbolValue::Nat(value.into())
                }
                _ => return false,
            };
            values.bind(*symbol, assigned);
            every(arena, goal, remaining, values)
        })
    }
    every(arena, goal, &selected, &mut crate::expr::Assignment::new())
}

#[cfg(test)]
#[test]
fn repeated_fact_installation_retains_same_world_evidence() {
    let mut arena = ExprArena::new();
    let x_symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
    let y_symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
    let x = arena.int_symbol(x_symbol);
    let y = arena.int_symbol(y_symbol);
    let zero = arena.int(0);
    let one = arena.int(1);
    let sixteen = arena.int(16);
    let thirty_two = arena.int(32);
    let boolean = arena.int_cmp(CmpOp::Ge, x, zero);
    let nonzero = arena.int_add(x, one);
    let square = arena.int_mul(x, x);
    let composite = arena.int_sub(square, one);
    let mut facts = Facts::new();
    facts.set_range(x_symbol, zero, sixteen);
    facts.add_lower(x_symbol, zero);
    facts.add_upper(x_symbol, sixteen);
    facts.assume_boolean(boolean);
    facts.assume_zero(&arena, y);
    facts.assume_nonzero(&arena, nonzero);
    facts.assume_nonnegative(&mut arena, composite);
    let minimum = arena.int_min(x, sixteen);
    assert_eq!(canonical_with_facts(&mut arena, &facts, minimum), x);
    let minus_one = arena.int(-1);
    assert!(!nonneg(&mut arena, &facts, minus_one));
    let unchanged = facts.clone();
    facts.set_range(x_symbol, zero, sixteen);
    facts.set_range_lower(x_symbol, zero);
    facts.add_lower(x_symbol, zero);
    facts.add_upper(x_symbol, sixteen);
    facts.assume_boolean(boolean);
    facts.assume_zero(&arena, y);
    facts.assume_nonzero(&arena, nonzero);
    facts.assume_nonnegative(&mut arena, composite);
    assert!(Rc::ptr_eq(&facts.caches(), &unchanged.caches()));
    assert!(Rc::ptr_eq(&facts.caches(), &unchanged.caches()));
    assert_eq!(facts.extra_lower.len(), 1);
    assert_eq!(facts.extra_upper.len(), 1);
    assert_eq!(facts.zero.len(), 1);
    assert_eq!(facts.nonzero.len(), 1);
    assert_eq!(facts.boolean.len(), 1);
    assert_eq!(facts.assumed.len(), 1);
    // Real replacement is a new world even when it weakens a primary bound.
    facts.set_range(x_symbol, zero, thirty_two);
    assert!(!Rc::ptr_eq(&facts.caches(), &unchanged.caches()));
    assert!(!Rc::ptr_eq(&facts.caches(), &unchanged.caches()));
    assert_eq!(unchanged.upper[&x_symbol], sixteen);
    assert_eq!(facts.extra_upper, vec![(x_symbol, sixteen)]);
    // An equal extra assumption must not hide a real primary replacement.
    let replaced = facts.clone();
    facts.set_range(x_symbol, zero, sixteen);
    assert_eq!(facts.upper[&x_symbol], sixteen);
    assert!(!Rc::ptr_eq(&facts.caches(), &replaced.caches()));
    assert!(!Rc::ptr_eq(&facts.caches(), &replaced.caches()));
}

#[cfg(test)]
#[test]
fn type_premises_preserve_scoped_ranges_and_repeated_publication_caches() {
    let mut arena = ExprArena::new();
    let symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
    let value = arena.int_symbol(symbol);
    let zero = arena.int(0);
    let one = arena.int(1);
    let two = arena.int(2);
    let three = arena.int(3);
    let four = arena.int(4);
    let mut facts = Facts::new();
    // The loop establishes its own start before a local occurrence publishes
    // the same immutable value's broader index type.
    facts.set_range(symbol, one, three);
    facts.assume_type(&mut arena, symbol, &ValueType::Index { bound: four });
    assert_eq!(facts.lower[&symbol], one);
    assert_eq!(facts.upper[&symbol], three);
    assert!(facts.extra_lower.contains(&(symbol, zero)));
    assert!(facts.extra_upper.contains(&(symbol, three)));
    let above_start = arena.int_sub(value, one);
    assert!(nonneg(&mut arena, &facts, above_start));
    let below_last = arena.int_sub(two, value);
    assert!(!nonneg(&mut arena, &facts, below_last));
    let minimum = arena.int_min(value, one);
    assert_eq!(canonical_with_facts(&mut arena, &facts, minimum), one);
    let reached = facts.clone();
    facts.assume_type(&mut arena, symbol, &ValueType::Index { bound: four });
    assert!(Rc::ptr_eq(&facts.caches(), &reached.caches()));
    assert!(Rc::ptr_eq(&facts.caches(), &reached.caches()));
    assert!(Rc::ptr_eq(&facts.caches(), &reached.caches()));
    assert_eq!(
        facts.lower_bounds().collect::<Vec<_>>(),
        vec![(symbol, one), (symbol, zero)]
    );
    assert_eq!(facts.upper_bounds().collect::<Vec<_>>(), vec![(symbol, three)]);

    // A further type premise is conjoined, even when the primary range is
    // weaker. The existing consumers must retain both kinds of evidence.
    facts.assume_type(&mut arena, symbol, &ValueType::Index { bound: two });
    assert_eq!(facts.upper[&symbol], three);
    let at_most_one = arena.int_sub(one, value);
    assert!(nonneg(&mut arena, &facts, at_most_one));

    // Explicit world-range replacement still replaces its primary endpoints;
    // independent type premises are not silently erased with them.
    let minus_one = arena.int(-1);
    facts.set_range(symbol, minus_one, four);
    assert_eq!(facts.lower[&symbol], minus_one);
    assert_eq!(facts.upper[&symbol], four);
    assert!(facts.extra_lower.contains(&(symbol, zero)));
    assert!(facts.extra_upper.contains(&(symbol, one)));

    let fresh = arena.checker_variable(crate::expr::SymbolSort::Int);
    let mut fresh_facts = Facts::new();
    fresh_facts.assume_type(&mut arena, fresh, &ValueType::Scalar(DType::I32));
    assert_eq!(
        constant(&arena, fresh_facts.lower[&fresh]),
        Some(i64::from(i32::MIN))
    );
    assert_eq!(
        constant(&arena, fresh_facts.upper[&fresh]),
        Some(i64::from(i32::MAX))
    );
    assert_eq!(fresh_facts.lower_bounds().count(), 1);
    assert_eq!(fresh_facts.upper_bounds().count(), 1);
    let signed = arena.int_symbol(fresh);
    assert!(!nonneg(&mut arena, &fresh_facts, signed));
}

#[cfg(test)]
#[test]
fn extra_bounds_survive_equal_primary_bounds_and_later_replacement() {
    let mut arena = ExprArena::new();
    let symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
    let x = arena.int_symbol(symbol);
    let one = arena.int(1);
    let sixteen = arena.int(16);
    let minus_ten = arena.int(-10);
    let thirty_two = arena.int(32);
    let seventeen = arena.int(17);
    let zero = arena.int(0);
    let mut facts = Facts::new();
    facts.set_range(symbol, one, sixteen);
    facts.add_lower(symbol, one);
    facts.add_upper(symbol, sixteen);
    facts.set_range(symbol, minus_ten, thirty_two);
    assert!(derive_le(&mut arena, &facts, one, x).is_established());
    assert!(derive_le(&mut arena, &facts, x, sixteen).is_established());
    assert!(!derive_le(&mut arena, &facts, x, zero).is_established());
    assert!(!derive_le(&mut arena, &facts, seventeen, x).is_established());
}

#[cfg(test)]
#[test]
fn cached_recursive_failure_keeps_the_finite_domain_alternative() {
    let mut arena = ExprArena::new();
    let (_, _, x) = arena.loop_binder();
    let zero = arena.int(0);
    let one = arena.int(1);
    let x0 = arena.int_cmp(CmpOp::Eq, x, zero);
    let x1 = arena.int_cmp(CmpOp::Eq, x, one);
    let domain = arena.any(&[x0, x1]);
    let complement = arena.int_sub(one, x);
    let goal = arena.int_mul(x, complement);
    let negative = arena.int_sub(goal, one);
    let undefined = arena.int_div(one, x);
    let mut facts = Facts::new();
    facts.assume_boolean(domain);
    for value in [goal, negative, undefined] {
        facts.caches().memo.borrow_mut().insert(
            normalize_int(&arena, value),
            Memo {
                proved: None,
                failed: DEPTH,
            },
        );
    }
    assert!(nonneg(&mut arena, &facts, goal));
    assert!(!nonneg(&mut arena, &facts, negative));
    assert!(!nonneg(&mut arena, &facts, undefined));
    // A finite proof installs the same shallow success and learned-sign event.
    assert_eq!(
        facts.caches().memo.borrow()[&normalize_int(&arena, goal)].proved,
        Some(1)
    );
    assert!(nonneg(&mut arena, &facts, goal));
}

#[cfg(test)]
#[test]
fn finite_choice_proofs_require_every_defined_admitted_member() {
    let mut arena = ExprArena::new();
    let mut facts = Facts::new();
    let (_, _, x) = arena.loop_binder();
    let (_, _, y) = arena.loop_binder();
    let zero = arena.int(0);
    let one = arena.int(1);
    let two = arena.int(2);
    let x0 = arena.int_cmp(CmpOp::Eq, x, zero);
    let x1 = arena.int_cmp(CmpOp::Eq, x, one);
    let domain = arena.any(&[x0, x1]);
    facts.assume_boolean(domain);
    let complement = arena.int_sub(one, x);
    let goal = arena.int_mul(x, complement);
    assert!(finite_domain_nonnegative(&arena, &facts, goal));
    let negative = arena.int_sub(goal, one);
    assert!(!finite_domain_nonnegative(&arena, &facts, negative));
    assert!(!finite_domain_nonnegative(&arena, &facts, y));
    let undefined = arena.int_div(one, x);
    assert!(!finite_domain_nonnegative(&arena, &facts, undefined));
    let mut partial = Facts::new();
    partial.assume_boolean(x0);
    let unrelated = arena.int_cmp(CmpOp::Eq, y, two);
    let disjunction = arena.any(&[x0, unrelated]);
    let mut mixed = Facts::new();
    mixed.assume_boolean(disjunction);
    assert!(!finite_domain_nonnegative(&arena, &mixed, x));
    // No unknown disjunct is silently selected; one negative admitted member rejects.
    let minus_one = arena.int(-1);
    let xm = arena.int_cmp(CmpOp::Eq, x, minus_one);
    let negative_domain = arena.any(&[xm, x1]);
    let mut rejected = Facts::new();
    rejected.assume_boolean(negative_domain);
    assert!(!finite_domain_nonnegative(&arena, &rejected, x));
    assert!(finite_domain_nonnegative(&arena, &partial, x));
}

#[cfg(test)]
#[test]
fn finite_recipe_selectors_keep_exact_nonnegative_and_upper_bounds() {
    let mut arena = ExprArena::new();
    let mut facts = Facts::new();
    let zero = arena.int(0);
    let one = arena.int(1);
    let two = arena.int(2);
    let mut variables = Vec::new();
    for index in 0..5 {
        let (_, _, value) = arena.loop_binder();
        let a = arena.int_cmp(CmpOp::Eq, value, zero);
        let b = arena.int_cmp(CmpOp::Eq, value, one);
        let domain = if index == 0 {
            arena.any(&[a, b])
        } else {
            let c = arena.int_cmp(CmpOp::Eq, value, two);
            arena.any(&[a, b, c])
        };
        facts.assume_boolean(domain);
        variables.push(value);
    }
    let mut selected = variables[4];
    for value in variables[1..4].iter().rev() {
        let present = arena.int_min(one, *value);
        let absent = arena.int_sub(one, present);
        let later = arena.int_mul(absent, selected);
        selected = arena.int_add(*value, later);
    }
    let enabled = arena.int_sub(one, variables[0]);
    let selected = arena.int_mul(enabled, selected);
    assert!(finite_domain_nonnegative(&arena, &facts, selected));
    let upper = arena.int_sub(two, selected);
    assert!(finite_domain_nonnegative(&arena, &facts, upper));
    let too_small = arena.int_sub(one, selected);
    assert!(!finite_domain_nonnegative(&arena, &facts, too_small));
    let mut many = Facts::new();
    let mut sum = zero;
    for _ in 0..9 {
        let (_, _, value) = arena.loop_binder();
        let a = arena.int_cmp(CmpOp::Eq, value, zero);
        let b = arena.int_cmp(CmpOp::Eq, value, one);
        let domain = arena.any(&[a, b]);
        many.assume_boolean(domain);
        sum = arena.int_add(sum, value);
    }
    assert!(!finite_domain_nonnegative(&arena, &many, sum));
}

#[cfg(test)]
#[test]
fn grouped_coefficients_keep_the_last_legal_lane_and_reject_the_next() {
    let mut arena = ExprArena::new();
    let mut facts = Facts::new();
    let (_, ws, width) = arena.loop_binder();
    let (_, cs, coefficients) = arena.loop_binder();
    let (_, ls, lanes) = arena.loop_binder();
    let (_, is, lane) = arena.loop_binder();
    let zero = arena.int(0);
    let one = arena.int(1);
    let two = arena.int(2);
    let sixteen = arena.int(16);
    let thirty_two = arena.int(32);
    facts.set_range_lower(ws, thirty_two);
    facts.set_range_lower(cs, two);
    facts.set_range_lower(ls, one);
    let last_lane = arena.int_sub(lanes, one);
    facts.set_range(is, zero, last_lane);
    let stored = arena.int_mul(sixteen, coefficients);
    let equality = arena.int_sub(stored, width);
    facts.assume_zero(&mut arena, equality);
    facts.assume_nonnegative(&mut arena, equality);
    let negative_equality = arena.int_sub(zero, equality);
    facts.assume_nonnegative(&mut arena, negative_equality);
    let groups_remainder = arena.int_rem(width, thirty_two);
    facts.assume_zero(&mut arena, groups_remainder);
    let stride = arena.int_div(width, lanes);
    let offset = arena.int_mul(lane, stride);
    let final_column = arena.int_sub(width, one);
    let offset_bound = arena.int_sub(final_column, offset);
    let parity = arena.int_rem(coefficients, two);
    assert!(nonneg(&mut arena, &facts, offset_bound));
    assert!(self::zero(&mut arena, &facts, parity));
    let group = arena.int_div(offset, thirty_two);
    let twice_group = arena.int_mul(two, group);
    let last_pair = arena.int_sub(coefficients, two);
    let bound = arena.int_sub(last_pair, twice_group);
    assert!(nonneg(&mut arena, &facts, bound));
    let next = arena.int_sub(bound, one);
    assert!(!nonneg(&mut arena, &facts, next));
}

#[cfg(test)]
#[test]
fn composite_width_group_bound_uses_its_established_positive_domain() {
    let mut arena = ExprArena::new();
    let mut facts = Facts::new();
    let (_, ps, p) = arena.loop_binder();
    let (_, ss, s) = arena.loop_binder();
    let (_, cs, coefficients) = arena.loop_binder();
    let (_, ls, lanes) = arena.loop_binder();
    let (_, is, lane) = arena.loop_binder();
    let zero = arena.int(0);
    let one = arena.int(1);
    let two = arena.int(2);
    let sixteen = arena.int(16);
    let thirty_two = arena.int(32);
    facts.set_range_lower(ps, zero);
    facts.set_range_lower(ss, zero);
    facts.set_range_lower(cs, one);
    facts.set_range_lower(ls, one);
    let doubled = arena.int_mul(two, p);
    let width = arena.int_add(doubled, s);
    let width_domain = arena.int_sub(width, thirty_two);
    facts.assume_nonnegative(&mut arena, width_domain);
    let last_lane = arena.int_sub(lanes, one);
    facts.set_range(is, zero, last_lane);
    let stored = arena.int_mul(sixteen, coefficients);
    let equality = arena.int_sub(stored, width);
    facts.assume_zero(&mut arena, equality);
    facts.assume_nonnegative(&mut arena, equality);
    let negative_equality = arena.int_sub(zero, equality);
    facts.assume_nonnegative(&mut arena, negative_equality);
    let groups_remainder = arena.int_rem(width, thirty_two);
    facts.assume_zero(&mut arena, groups_remainder);
    let stride = arena.int_div(width, lanes);
    let offset = arena.int_mul(lane, stride);
    let positive_width = arena.int_sub(width, one);
    assert!(established_nonnegative(&arena, &facts, positive_width));
    let group = arena.int_div(offset, thirty_two);
    let twice_group = arena.int_mul(two, group);
    let last_pair = arena.int_sub(coefficients, two);
    let bound = arena.int_sub(last_pair, twice_group);
    assert!(nonneg(&mut arena, &facts, bound));
    let too_far = arena.int_sub(bound, one);
    assert!(!nonneg(&mut arena, &facts, too_far));
    let fixed_lanes = arena.int_sub(lanes, thirty_two);
    facts.assume_zero(&mut arena, fixed_lanes);
    facts.assume_nonnegative(&mut arena, fixed_lanes);
    let negative_fixed = arena.int_sub(zero, fixed_lanes);
    facts.assume_nonnegative(&mut arena, negative_fixed);
    // The fixed-lane case must prove from its own context, not reuse the
    // earlier general-lane successful memo entry.
    facts.changed();
    assert!(nonneg(&mut arena, &facts, bound));
    assert!(!nonneg(&mut arena, &facts, too_far));
}

#[cfg(test)]
#[test]
fn paired_group_last_coefficient_follows_the_exact_remainder_bound() {
    let mut arena = ExprArena::new();
    let mut facts = Facts::new();
    let (_, cs, coefficients) = arena.loop_binder();
    let (_, gs, group) = arena.loop_binder();
    let zero = arena.int(0);
    let one = arena.int(1);
    let two = arena.int(2);
    facts.set_range_lower(cs, one);
    let count = arena.int_div(coefficients, two);
    let last = arena.int_sub(count, one);
    facts.set_range(gs, zero, last);
    let paired = arena.int_mul(two, group);
    let last_coefficient = arena.int_sub(coefficients, two);
    let bound = arena.int_sub(last_coefficient, paired);
    assert!(nonneg(&mut arena, &facts, bound));
    let outside = arena.int_sub(bound, one);
    assert!(!nonneg(&mut arena, &facts, outside));
}

#[cfg(test)]
#[test]
fn euclidean_remainder_bound_requires_its_exact_defined_domain() {
    let mut arena = ExprArena::new();
    let mut facts = Facts::new();
    let (_, ns, n) = arena.loop_binder();
    let (_, ds, d) = arena.loop_binder();
    let zero = arena.int(0);
    let one = arena.int(1);
    facts.set_range_lower(ns, zero);
    facts.set_range_lower(ds, one);
    let quotient = arena.int_div(n, d);
    let product = arena.int_mul(d, quotient);
    let remainder = arena.int_sub(n, product);
    assert!(nonneg(&mut arena, &facts, remainder));
    let excessive = arena.int_sub(remainder, one);
    assert!(!nonneg(&mut arena, &facts, excessive));
    let wrong_multiplier = arena.int_add(d, one);
    let wrong_product = arena.int_mul(wrong_multiplier, quotient);
    let wrong = arena.int_sub(n, wrong_product);
    assert!(!nonneg(&mut arena, &facts, wrong));
    let mut no_domain = Facts::new();
    no_domain.set_range_lower(ds, one);
    // Mathematical Int division is Euclidean even for a negative numerator.
    // The actual signed scalar recipe also retains Euclidean division.
    assert!(nonneg(&mut arena, &no_domain, remainder));
    let negative = arena.int(-5);
    let two = arena.int(2);
    let signed_quotient = arena.scalar_integer(
        crate::reference_math::ScalarOp::Binary(crate::syntax::ast::BinaryOp::Div),
        &[ScalarOperand::Word { dtype: DType::I32, value: negative }, ScalarOperand::Word { dtype: DType::I32, value: two }],
    );
    let signed_product = arena.int_mul(two, signed_quotient);
    let signed_remainder = arena.int_sub(negative, signed_product);
    assert_eq!(
        arena
            .eval_int_i64(signed_remainder, &crate::expr::Assignment::new())
            .unwrap(),
        1
    );
    assert!(nonneg(&mut arena, &Facts::new(), signed_remainder));
    let minimum = arena.int(i64::from(i32::MIN));
    let negative_one = arena.int(-1);
    let overflow = arena.scalar_integer(
        crate::reference_math::ScalarOp::Binary(crate::syntax::ast::BinaryOp::Div),
        &[ScalarOperand::Word { dtype: DType::I32, value: minimum }, ScalarOperand::Word { dtype: DType::I32, value: negative_one }],
    );
    assert!(arena
        .eval_int_i64(overflow, &crate::expr::Assignment::new())
        .is_err());
    assert!(!nonneg(&mut arena, &Facts::new(), overflow));
    let mut no_denominator = Facts::new();
    no_denominator.set_range_lower(ns, zero);
    // The original quotient/remainder definedness proof remains authoritative.
    assert!(nonneg(&mut arena, &no_denominator, remainder));
    let undefined = arena.scalar_integer(
        crate::reference_math::ScalarOp::Binary(crate::syntax::ast::BinaryOp::Div),
        &[ScalarOperand::Word { dtype: DType::I32, value: one }, ScalarOperand::Word { dtype: DType::I32, value: zero }],
    );
    assert!(arena
        .eval_int_i64(undefined, &crate::expr::Assignment::new())
        .is_err());
    assert!(!nonneg(&mut arena, &Facts::new(), undefined));
}

#[cfg(test)]
#[test]
fn established_evidence_preserves_positive_integer_scaling_only() {
    let mut arena = ExprArena::new();
    let mut facts = Facts::new();
    let a = arena.checker_variable(crate::expr::SymbolSort::Int);
    let b = arena.checker_variable(crate::expr::SymbolSort::Int);
    let a = arena.int_symbol(a);
    let b = arena.int_symbol(b);
    let distance = arena.int_sub(b, a);
    let one = arena.int(1);
    let distance = arena.int_sub(distance, one);
    facts.assume_nonnegative(&mut arena, distance);
    let two = arena.int(2);
    let scaled = arena.int_mul(two, distance);
    assert!(established_nonnegative(&arena, &facts, scaled));
    let zero = arena.int(0);
    let reversed = arena.int_sub(zero, scaled);
    assert!(!established_nonnegative(&arena, &facts, reversed));
    let strict = arena.int_sub(scaled, one);
    assert!(!established_nonnegative(&arena, &facts, strict));
    assert!(established_nonnegative(&arena, &facts, zero));
    let negative_a = arena.int_sub(zero, a);
    facts.assume_nonnegative(&mut arena, negative_a);
    let minimum = arena.int(i64::MIN);
    let minimum_scaled = arena.int_mul(minimum, a);
    assert!(established_nonnegative(&arena, &facts, minimum_scaled));
    let (_, _, width) = arena.nat_loop_binder();
    let width = arena.int_from_nat(width);
    let row_major = arena.int_mul(width, distance);
    assert!(established_nonnegative(&arena, &facts, row_major));
    let signed = arena.checker_variable(crate::expr::SymbolSort::Int);
    let signed = arena.int_symbol(signed);
    let unknown_factor = arena.int_mul(signed, distance);
    assert!(!established_nonnegative(&arena, &facts, unknown_factor));
}

#[cfg(test)]
#[test]
fn fundamental_remainder_equality_requires_actual_interval() {
    let mut arena = ExprArena::new();
    let symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
    let x = arena.int_symbol(symbol);
    let origin = arena.int(0);
    let four = arena.int(4);
    let modulus = arena.int(1_i64 << 32);
    let half = arena.int(1_i64 << 31);
    let shifted = arena.int_add(x, half);
    let remainder = arena.int_rem(shifted, modulus);
    let restored = arena.int_sub(remainder, half);
    let difference = arena.int_sub(restored, x);
    let mut bounded = Facts::new();
    bounded.set_range(symbol, origin, four);
    assert!(zero(&mut arena, &bounded, difference));
    let canonical = canonical_with_facts(&mut arena, &bounded, restored);
    assert!(same(&arena, canonical, x));
    assert!(!zero(&mut arena, &Facts::new(), difference));
    let at_modulus = arena.int_rem(modulus, modulus);
    let upper_difference = arena.int_sub(at_modulus, modulus);
    assert!(!zero(&mut arena, &bounded, upper_difference));
    let negative = arena.int(-1);
    let negative_remainder = arena.int_rem(negative, modulus);
    let negative_difference = arena.int_sub(negative_remainder, negative);
    assert!(!zero(&mut arena, &bounded, negative_difference));
}

#[cfg(test)]
#[test]
fn established_signed_affine_endpoints_keep_nonlinear_sign_guards() {
    let mut arena = ExprArena::new();
    let symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
    let x = arena.int_symbol(symbol);
    let zero = arena.int(0);
    let four = arena.int(4);
    let minimum = arena.int_min(x, four);
    let maximum = arena.int_max(x, zero);
    let upper = arena.int_sub(four, minimum);
    assert!(established_nonnegative(&arena, &Facts::new(), upper));
    assert!(established_nonnegative(&arena, &Facts::new(), maximum));
    let invalid_upper = arena.int_sub(minimum, four);
    let invalid_lower = arena.int_sub(zero, maximum);
    assert!(!established_nonnegative(
        &arena,
        &Facts::new(),
        invalid_upper
    ));
    assert!(!established_nonnegative(
        &arena,
        &Facts::new(),
        invalid_lower
    ));
    let other = arena.checker_variable(crate::expr::SymbolSort::Int);
    let y = arena.int_symbol(other);
    let product = arena.int_mul(x, y);
    let mut facts = Facts::new();
    let minus_two = arena.int(-2);
    let minus_one = arena.int(-1);
    let two = arena.int(2);
    facts.set_range(symbol, minus_two, minus_one);
    facts.set_range(other, zero, two);
    let polynomial = normalize_int(&arena, product);
    let engine = Engine::new(&arena, &facts);
    let (lower, upper) = engine.interval_over(&polynomial, &|_| true);
    assert_eq!(lower, polynomial);
    assert_eq!(upper, polynomial);
    assert!(!established_nonnegative(&arena, &facts, product));
}

#[test]
fn reached_composite_nonnegative_fact_supplies_nested_min_operand() {
    let mut arena = ExprArena::new();
    let symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
    let requested = arena.int_symbol(symbol);
    let thirty_two = arena.int(32);
    let sixteen = arena.int(16);
    let old = arena.int_min(requested, thirty_two);
    let smaller = arena.int_min(old, sixteen);
    let mut facts = Facts::new();
    assert!(!established_nonnegative(&arena, &facts, smaller));
    facts.assume_nonnegative(&mut arena, old);
    assert!(established_nonnegative(&arena, &facts, smaller));
    assert!(nonneg(&mut arena, &facts, smaller));
    // Reaching a fact about another composite value cannot confer the sign
    // of this operand or of an unrelated signed product.
    let other_symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
    let other = arena.int_symbol(other_symbol);
    let other_old = arena.int_min(other, thirty_two);
    let other_smaller = arena.int_min(other_old, sixteen);
    assert!(!established_nonnegative(&arena, &facts, other_smaller));
    let product = arena.int_mul(smaller, other);
    assert!(!established_nonnegative(&arena, &facts, product));
}

#[cfg(test)]
#[test]
fn exact_quotient_normal_form_requires_established_integral_equations() {
    let mut arena = ExprArena::new();
    let (_, _, w) = arena.loop_binder();
    let (_, _, c) = arena.loop_binder();
    let (_, _, unrelated) = arena.loop_binder();
    let four = arena.int(4);
    let multiple = arena.int_mul(four, c);
    let equality = arena.int_sub(multiple, w);
    let quotient = arena.int_div(w, four);
    let difference = arena.int_sub(quotient, c);
    let mut facts = Facts::new();
    assert!(!zero(&mut arena, &facts, difference));
    facts.assume_zero(&mut arena, equality);
    assert!(zero(&mut arena, &facts, difference));
    let wrong = arena.int_sub(quotient, unrelated);
    assert!(!zero(&mut arena, &facts, wrong));
    let zero_value = arena.int(0);
    let neg_w = arena.int_sub(zero_value, w);
    let neg_c = arena.int_sub(zero_value, c);
    let signed_quotient = arena.int_div(neg_w, four);
    let signed_difference = arena.int_sub(signed_quotient, neg_c);
    assert!(zero(&mut arena, &facts, signed_difference));
    let one = arena.int(1);
    let offset = arena.int_add(w, one);
    let inexact = arena.int_div(offset, four);
    let expected_offset = arena.int_add(c, one);
    let bad_difference = arena.int_sub(inexact, expected_offset);
    assert!(!zero(&mut arena, &facts, bad_difference));
}

#[cfg(test)]
#[test]
fn exact_quotient_normal_form_keeps_undefined_unproved_and_nested_atoms() {
    let mut arena = ExprArena::new();
    let (_, _, w) = arena.loop_binder();
    let (_, _, c) = arena.loop_binder();
    let four = arena.int(4);
    let multiple = arena.int_mul(four, c);
    let equality = arena.int_sub(multiple, w);
    let numerator = normalize_int(&arena, w);
    let raw = |denominator: Poly| {
        Poly::atom(Atom::Quot(
            Rc::new(numerator.clone()),
            Rc::new(denominator),
        ))
    };
    let mut inequality_only = Facts::new();
    inequality_only.assume_nonnegative(&mut arena, equality);
    let positive = raw(Poly::constant(4));
    assert_eq!(
        Engine::new(&arena, &inequality_only).apply_zero_facts(positive.clone()),
        positive
    );
    let mut facts = Facts::new();
    facts.assume_zero(&mut arena, equality);
    let engine = Engine::new(&arena, &facts);
    for denominator in [
        Poly::constant(0),
        Poly::constant(-4),
        normalize_int(&arena, c),
    ] {
        let quotient = raw(denominator);
        assert_eq!(engine.apply_zero_facts(quotient.clone()), quotient);
    }
    let invalid = Poly::constant(i64::MAX).add(&Poly::constant(1));
    assert!(!invalid.valid);
    assert_eq!(engine.apply_zero_facts(invalid.clone()), invalid);
    let two = arena.int(2);
    let nested = arena.int_div(c, two);
    let nested_multiple = arena.int_mul(four, nested);
    let nested_equality = arena.int_sub(nested_multiple, w);
    let mut nested_facts = Facts::new();
    nested_facts.assume_zero(&mut arena, nested_equality);
    assert_eq!(
        Engine::new(&arena, &nested_facts).apply_zero_facts(positive.clone()),
        positive
    );
}

#[cfg(test)]
#[test]
fn joined_affine_values_keep_established_common_bounds_and_declared_type() {
    let mut arena = ExprArena::new();
    let (_, ns, n) = arena.loop_binder();
    let (_, xs, x) = arena.loop_binder();
    let (_, ys, y) = arena.loop_binder();
    let (_, js, joined) = arena.loop_binder();
    let zero = arena.int(0);
    let one = arena.int(1);
    let mut facts = Facts::new();
    facts.set_range_lower(ns, one);
    facts.set_range(xs, one, n);
    facts.set_range(ys, one, n);
    let left = arena.int_sub(x, one);
    let right = arena.int_sub(y, one);
    let mut after = Facts::new();
    after.assume_type(&mut arena, js, &ValueType::Scalar(DType::I32));
    after.join_bounds(&mut arena, js, [(&facts, left), (&facts, right)], &|s| {
        s == ns
    });
    assert!(derive_le(&mut arena, &after, zero, joined).is_established());
    assert!(derive_le(&mut arena, &after, joined, n).is_established());
    let negative = arena.int(-1);
    assert!(!derive_le(&mut arena, &after, joined, negative).is_established());
    let low = arena.int(i64::from(i32::MIN));
    let high = arena.int(i64::from(i32::MAX));
    assert!(derive_le(&mut arena, &after, low, joined).is_established());
    assert!(derive_le(&mut arena, &after, joined, high).is_established());
}

#[cfg(test)]
#[test]
fn joined_signed_arms_do_not_infer_nonnegative_from_one_arm() {
    let mut arena = ExprArena::new();
    let arm_facts = Facts::new();
    let negative = arena.int(-1);
    let positive = arena.int(9);
    let zero = arena.int(0);
    let (_, joined_symbol, joined) = arena.loop_binder();
    let mut after = Facts::new();
    after.assume_type(&mut arena, joined_symbol, &ValueType::Scalar(DType::I32));
    after.join_bounds(
        &mut arena,
        joined_symbol,
        [(&arm_facts, negative), (&arm_facts, positive)],
        &|_| true,
    );
    assert!(
        !derive_le(&mut arena, &after, zero, joined).is_established(),
        "one nonnegative arm cannot grant a joined lower bound"
    );
    assert!(derive_le(&mut arena, &after, negative, joined).is_established());
    assert!(derive_le(&mut arena, &after, joined, positive).is_established());
    let type_lower = arena.int(i64::from(i32::MIN));
    let type_upper = arena.int(i64::from(i32::MAX));
    assert!(derive_le(&mut arena, &after, type_lower, joined).is_established());
    assert!(derive_le(&mut arena, &after, joined, type_upper).is_established());
}

#[test]
fn reached_scalar_normal_form_keeps_definedness_and_word_wrap() {
    let mut arena = ExprArena::new();
    let symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
    let requested = arena.int_symbol(symbol);
    let zero = arena.int(0);
    let one = arena.int(1);
    let sixteen = arena.int(16);
    let thirty_two = arena.int(32);
    let modulus = arena.int(4294967296);
    let half = arena.int(2147483648);
    let old = arena.int_min(requested, thirty_two);
    let count = arena.int_min(old, sixteen);
    let unsigned = arena.int_rem(count, modulus);
    let shifted = arena.int_add(unsigned, half);
    let rem = arena.int_rem(shifted, modulus);
    let signed = arena.int_sub(rem, half);
    for strict in [false, true] {
        let mut facts = Facts::new();
        let reached = if strict { arena.int_sub(old, one) } else { old };
        facts.assume_nonnegative(&mut arena, reached);
        let canonical = canonical_with_facts(&mut arena, &facts, signed);
        assert!(same(&arena, canonical, count));
    }
    let mut empty = Facts::new();
    empty.assume_zero(&arena, old);
    let normalized = canonical_with_facts(&mut arena, &empty, signed);
    assert_eq!(constant(&arena, normalized), Some(0));
    let unknown = canonical_with_facts(&mut arena, &Facts::new(), signed);
    assert!(!same(&arena, unknown, count));
    let outside = arena.int(4294967297);
    let wrapped = arena.int_rem(outside, modulus);
    let normalized = canonical_with_facts(&mut arena, &Facts::new(), wrapped);
    assert_eq!(constant(&arena, normalized), Some(1));
    let negative = arena.int(-1);
    let negative_cast = arena.int_rem(negative, modulus);
    let normalized = canonical_with_facts(&mut arena, &Facts::new(), negative_cast);
    assert_eq!(constant(&arena, normalized), Some(4294967295));
    // Undefined children must remain in the expression, even when algebra
    // would erase their numeric coefficient or choose another Min operand.
    let undefined = arena.int_div(requested, zero);
    let product = arena.int_mul(zero, undefined);
    let minimum = arena.int_min(undefined, undefined);
    for expression in [undefined, product, minimum] {
        let normalized = canonical_with_facts(&mut arena, &Facts::new(), expression);
        assert_eq!(normalized, expression);
        assert!(!arena.is_total(AnyExpr::Int(normalized)));
    }
    let unknown_divisor = arena.checker_variable(crate::expr::SymbolSort::Int);
    let unknown_divisor = arena.int_symbol(unknown_divisor);
    let division = arena.int_div(requested, unknown_divisor);
    assert_eq!(
        canonical_with_facts(&mut arena, &Facts::new(), division),
        division
    );
}


#[cfg(test)]
#[test]
fn typed_recipe_projection_preserves_wrap_and_independent_candidate_bounds() {
    use crate::reference_math::ScalarOp;
    use crate::syntax::ast::BinaryOp as SourceBinary;
    let mut arena = ExprArena::new();
    let input = arena.checker_variable(crate::expr::SymbolSort::Int);
    let x = arena.int_symbol(input);
    let zero = arena.int(0);
    let sixteen = arena.int(16);
    let one = arena.int(1);
    let add = arena.scalar_integer(ScalarOp::Binary(SourceBinary::Add), &[
        ScalarOperand::Word { dtype: DType::I32, value: x },
        ScalarOperand::Word { dtype: DType::I32, value: one },
    ]);
    let result = arena.checker_variable(crate::expr::SymbolSort::Int);
    let published = arena.int_symbol(result);
    let mut facts = Facts::new();
    facts.set_range(input, zero, sixteen);
    facts.assume_type(&mut arena, result, &ValueType::Scalar(DType::I32));
    assert!(facts.define_scalar(&arena, result, add));
    facts.assume_scalar_defined(&mut arena, add);
    let seventeen = arena.int(17);
    assert!(derive_le(&mut arena, &facts, published, seventeen).is_established());
    let increment = arena.int_sub(published, x);
    assert!(derive_nonnegative(&mut arena, &facts, increment).is_established());

    // The result's declared I32 range does not establish no-wrap arithmetic.
    let mut overflowing = Facts::new();
    let maximum = arena.int(i64::from(i32::MAX));
    overflowing.set_range(input, maximum, maximum);
    overflowing.assume_type(&mut arena, result, &ValueType::Scalar(DType::I32));
    assert!(overflowing.define_scalar(&arena, result, add));
    assert_eq!(derive_nonnegative(&mut arena, &overflowing, increment), Derivation::Unresolved);
    let literal = arena.scalar_integer(ScalarOp::Binary(SourceBinary::Add), &[
        ScalarOperand::Word { dtype: DType::I32, value: maximum },
        ScalarOperand::Word { dtype: DType::I32, value: one },
    ]);
    assert_eq!(constant(&arena, literal), Some(i64::from(i32::MIN)));
}

#[cfg(test)]
#[test]
fn source_definitions_reject_cycles_and_preserve_partial_continuations() {
    use crate::reference_math::ScalarOp;
    use crate::syntax::ast::BinaryOp as SourceBinary;
    let mut arena = ExprArena::new();
    let a = arena.checker_variable(crate::expr::SymbolSort::Int);
    let b = arena.checker_variable(crate::expr::SymbolSort::Int);
    let av = arena.int_symbol(a);
    let bv = arena.int_symbol(b);
    let mut facts = Facts::new();
    assert!(facts.define_scalar(&arena, a, bv));
    assert!(!facts.define_scalar(&arena, b, av));
    let one = arena.int(1);
    assert!(!facts.define_scalar(&arena, a, one));
    assert!(!facts.define_scalar(&arena, a, av));
    assert_eq!(canonical_with_facts(&mut arena, &facts, av), bv);
    let zero = arena.int(0);
    let failed = arena.scalar_integer(ScalarOp::Binary(SourceBinary::Div), &[
        ScalarOperand::Word { dtype: DType::I32, value: one },
        ScalarOperand::Word { dtype: DType::I32, value: zero },
    ]);
    assert!(!arena.is_total(failed.into()));
    let cancelled = arena.int_mul(zero, failed);
    assert_eq!(canonical_with_facts(&mut arena, &facts, cancelled), cancelled);
    assert_eq!(derive_nonnegative(&mut arena, &facts, failed), Derivation::Unresolved);
    let defined = arena.scalar_integer_defined(failed);
    let branch_symbol = arena.checker_variable(crate::expr::SymbolSort::Int);
    let branch_value = arena.int_symbol(branch_symbol);
    let branch = arena.int_cmp(CmpOp::Gt, branch_value, zero);
    let opposite = arena.not(branch);
    let first = arena.all(&[branch, defined]);
    let second = arena.all(&[opposite, defined]);
    let exits = arena.any(&[first, second]);
    let mut all_exits = facts.clone();
    all_exits.assume_boolean(exits);
    assert!(all_exits.scalar_defined(&arena, failed));
    let conditional_exits = arena.any(&[first, opposite]);
    let mut conditional = facts.clone();
    conditional.assume_boolean(conditional_exits);
    assert!(!conditional.scalar_defined(&arena, failed));
    let before = canonical_with_facts(&mut arena, &facts, failed);
    let mut continued = facts.clone();
    continued.assume_scalar_defined(&mut arena, failed);
    // A scoped continuation premise cannot change global evaluation/totality.
    assert!(!arena.is_total(failed.into()));
    assert_eq!(canonical_with_facts(&mut arena, &facts, failed), before);
    assert!(arena.eval_int_i64(failed, &crate::expr::Assignment::new()).is_err());
}


#[cfg(test)]
#[test]
fn required_derivation_consumes_established_partial_word_bounds() {
    use crate::reference_math::ScalarOp;
    use crate::syntax::ast::BinaryOp as SourceBinary;
    let mut arena = ExprArena::new();
    let denominator = arena.checker_variable(crate::expr::SymbolSort::Int);
    let denominator_value = arena.int_symbol(denominator);
    let one = arena.int(1);
    let recipe = arena.scalar_integer(ScalarOp::Binary(SourceBinary::Div), &[
        ScalarOperand::Word { dtype: DType::I32, value: one },
        ScalarOperand::Word { dtype: DType::I32, value: denominator_value },
    ]);
    let minimum = arena.int(i64::from(i32::MIN));
    let maximum = arena.int(i64::from(i32::MAX));
    let mut facts = Facts::new();
    facts.assume_type(&mut arena, denominator, &ValueType::Scalar(DType::I32));
    let lower = arena.int_sub(recipe, minimum);
    assert!(!established_nonnegative(&arena, &facts, lower));
    assert_eq!(derive_le(&mut arena, &facts, minimum, recipe), Derivation::Unresolved);
    facts.assume_scalar_defined(&mut arena, recipe);
    assert!(established_nonnegative(&arena, &facts, lower));
    assert_eq!(derive_le(&mut arena, &facts, minimum, recipe), Derivation::Established);
    assert_eq!(derive_le(&mut arena, &facts, recipe, maximum), Derivation::Established);
    // Successful evaluation gives the typed range, not a positive denominator
    // or a nonnegative signed result. The operation remains globally partial.
    let zero = arena.int(0);
    assert_eq!(derive_le(&mut arena, &facts, zero, recipe), Derivation::Unresolved);
    assert!(!arena.is_total(recipe.into()));
}

#[cfg(test)]
#[test]
fn established_sign_rejects_cyclic_quotient_bounds_and_uses_other_premises() {
    let mut arena = ExprArena::new();
    let x = arena.checker_variable(crate::expr::SymbolSort::Int);
    let y = arena.checker_variable(crate::expr::SymbolSort::Int);
    let xv = arena.int_symbol(x);
    let yv = arena.int_symbol(y);
    let one = arena.int(1);
    let self_quotient = arena.int_div(one, xv);
    let mut self_cycle = Facts::new();
    self_cycle.set_range_lower(x, self_quotient);
    // x >= 1/x does not establish x >= 0: x=-1 satisfies it.
    assert!(!established_nonnegative(&arena, &self_cycle, xv));
    let signed_quotient = arena.int_div(one, xv);
    assert!(!established_nonnegative(&arena, &self_cycle, signed_quotient));

    let reciprocal_y = arena.int_div(one, yv);
    let mut mutual = Facts::new();
    mutual.set_range_lower(x, reciprocal_y);
    mutual.set_range_lower(y, self_quotient);
    // x=y=-1 satisfies both premises, so neither cyclic chain gives a sign.
    assert!(!established_nonnegative(&arena, &mutual, xv));
    assert!(!established_nonnegative(&arena, &mutual, yv));
    let quotient = arena.int_div(xv, yv);
    assert!(!established_nonnegative(&arena, &mutual, quotient));

    // A cyclic first bound must not prevent another independent premise from
    // establishing positivity and then the quotient's denominator sign.
    mutual.add_lower(x, one);
    assert!(established_nonnegative(&arena, &mutual, xv));
    assert!(established_nonnegative(&arena, &mutual, yv));
    assert!(established_nonnegative(&arena, &mutual, quotient));
    let zero = arena.int(0);
    let negative_x = arena.int_sub(zero, xv);
    assert!(!established_nonnegative(&arena, &mutual, negative_x));
}
