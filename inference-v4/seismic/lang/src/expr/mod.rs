//! The one symbolic expression and constraint system (spec §5).
//!
//! One hash-consed typed DAG per entry. The same interned node drives solver
//! constraints, partial evaluation, applicability guards, runtime layout and
//! geometry, allocation sizes, numerical bounds, and duration comparison. There is
//! no second AST: consumers compile nodes into evaluators (`compiled`) but
//! never translate them into an independently editable representation.
//!
//! Handles are `Copy` indices into one [`ExprArena`]; a handle is valid only
//! for the arena that produced it. Mixing arenas is a panic (§13.3.2).
//!
//! Semantics: `NatExpr` and `IntExpr` denote mathematical integers. Runtime
//! representability is a domain restriction (§5.2), never wrapping.
//!
//! Ownership: W3 owns the internals of this module; the public surface below
//! is frozen by W0.

use crate::ids::{DimensionId, ParameterId};
use crate::types::DType;
use std::fmt;
pub use num_bigint::{BigInt, BigUint};
use std::marker::PhantomData;
use std::num::NonZeroU64;

/// Identity of the arena that minted a handle. It is carried by every
/// arena-local identifier, so equal ordinals from different entries can
/// never compare equal or be accepted by the wrong arena.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArenaId(NonZeroU64);

impl fmt::Debug for ArenaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "arena#{}", self.0.get())
    }
}

pub mod compiled;
pub(crate) mod poly;

/// A typed handle to one interned node.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Expr<Sort> {
    owner: ArenaId,
    index: u32,
    sort: PhantomData<Sort>,
}

impl<Sort> fmt::Debug for Expr<Sort> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}.expr#{}", self.owner, self.index)
    }
}

/// Sort markers.
pub mod sort {
    /// Non-negative sizes, indices, strides, byte counts, geometry.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Nat {}
    /// Signed integer arithmetic where semantics require it.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Int {}
    /// Predicates and logical composition.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Bool {}
    /// Typed numeric scalar expressions (numerical bounds, scalar parameters).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Scalar<T>(std::marker::PhantomData<T>);
    /// Physical-duration intervals in nanoseconds.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Duration {}
}

pub type NatExpr = Expr<sort::Nat>;
pub type IntExpr = Expr<sort::Int>;
pub type BoolExpr = Expr<sort::Bool>;
pub type ScalarExpr<T> = Expr<sort::Scalar<T>>;
pub type DurationExpr = Expr<sort::Duration>;

/// One typed symbol. Symbols are the only free variables (§5.1).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolId {
    owner: ArenaId,
    index: u32,
}

impl fmt::Debug for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}.sym#{}", self.owner, self.index)
    }
}

/// What a symbol stands for. Construction is available only through the
/// typed allocator corresponding to each semantic category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    /// A source declaration's symbolic shape parameter before entry
    /// monomorphization. Checker-private and forbidden from every public
    /// predicate level; the entry builder replaces it with CallDimension.
    TemplateDimension(u32),
    /// A call-schema dimension: symbolic at compile time, bound by tensors at
    /// invocation.
    CallDimension(DimensionId),
    /// A scalar or range parameter component of the call schema (`Nat`,
    /// `Int`, or a scalar dtype).
    CallScalar(ScalarArgument),
    /// Element stride of axis `u32` of root tensor parameter leaf `ParameterId`; bound by
    /// `validate_invocation` from the argument's `TensorLayout::strides()`.
    CallStride(ParameterId, u32),
    /// A runtime scalar SSA value in a monomorphized semantic function.
    /// Checker-private and rejected by entry/target predicates.
    RuntimeValue(crate::ids::SemanticValueId),
    /// A constant of the target profile, fixed before planning.
    TargetConstant(TargetConstantId),
    /// A finite compile-time decision owned by one implementation.
    Decision(DecisionId),
    /// A lexical loop binder inside a schedule or kernel.
    LoopBinder(LoopBinderId),
    /// A schedule-level mutable scalar slot (§8.1), written by commands at
    /// runtime and rebound before any dependent predicate or range is
    /// evaluated. Indexed by the owning schedule's slot ordinal.
    ScheduleSlot(u32),
    /// A checker-internal fresh variable. Never in an entry arena;
    /// `is_invocation() == false`.
    ProofVariable(u32),
}

/// One scalar component of a call-schema parameter: the value of a scalar or
/// `index` parameter, or one endpoint of a `range` parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScalarArgument {
    pub parameter: ParameterId,
    pub component: ScalarComponent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarComponent {
    Value,
    RangeStart,
    RangeEnd,
}

impl SymbolKind {
    /// Single owner of "evaluable from the invocation and fixed target/member facts".
    pub fn is_invocation(self) -> bool {
        matches!(
            self,
            Self::CallDimension(_)
                | Self::CallScalar(_)
                | Self::CallStride(..)
                | Self::TargetConstant(_)
                | Self::Decision(_)
        )
    }
}

/// A target-profile constant symbol, allocated by the compiler when a profile
/// is bound to an arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TargetConstantId {
    owner: ArenaId,
    index: u32,
}

/// A finite decision symbol, allocated by the compiler's implementation
/// builder. Its finite domain is recorded in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DecisionId {
    owner: ArenaId,
    index: u32,
}

/// Arena-owned lexical binder used by symbolic folds and frozen structured
/// schedules. Semantic source binders are mapped to these during lowering;
/// compiler-synthesized loops allocate them directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoopBinderId {
    owner: ArenaId,
    index: u32,
}

/// The sort of a symbol's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SymbolSort {
    Nat,
    Int,
    Scalar(DType),
}

/// A finite explicit domain for a decision symbol (§10.2).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FiniteDomain {
    values: Vec<i64>,
}

impl FiniteDomain {
    /// A non-empty ascending set. Empty or unsorted inputs are a construction
    /// error at the builder, never a runtime state.
    pub fn new(mut values: Vec<i64>) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        values.dedup();
        Some(Self { values })
    }

    pub fn values(&self) -> &[i64] {
        &self.values
    }
}

/// Reduction operator of a symbolic sum/product over a lexical binder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FoldOp {
    Sum,
    Product,
    Max,
}

/// The arena: one per `LogicalEntry`, moved (not copied) through
/// `PlanSpace`, shared read-only by `FrozenPlan` and evaluators afterwards.
///
/// Every constructor interns: structurally equal nodes are one handle.
/// Divisions record their nonzero-divisor side condition, which is exposed
/// through [`ExprArena::side_conditions`] and must be conjoined into any
/// predicate that claims totality.
#[derive(Debug)]
pub struct ExprArena {
    inner: internals::Arena,
}

impl Default for ExprArena {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprArena {
    pub fn new() -> Self {
        Self {
            inner: internals::Arena::new(),
        }
    }

    // ----- symbols ---------------------------------------------------------

    pub fn call_dimension(&mut self, dimension: DimensionId) -> (SymbolId, NatExpr) {
        self.inner.call_dimension(dimension)
    }

    pub(crate) fn template_dimension(&mut self, ordinal: u32) -> (SymbolId, IntExpr) {
        self.inner.template_dimension(ordinal)
    }

    pub(crate) fn runtime_value(
        &mut self,
        value: crate::ids::SemanticValueId,
    ) -> (SymbolId, IntExpr) {
        self.inner.runtime_value(value)
    }

    pub fn call_scalar(&mut self, argument: ScalarArgument, sort: SymbolSort) -> SymbolId {
        self.inner.call_scalar(argument, sort)
    }

    /// A fresh checker-internal proof variable.
    pub(crate) fn proof_variable(&mut self, sort: SymbolSort) -> SymbolId {
        self.inner.proof_variable(sort)
    }

    /// The `Nat` symbol for the element stride of `axis` of root tensor
    /// parameter leaf `parameter`. Interned: one symbol per `(parameter, axis)`.
    pub fn call_stride_symbol(&mut self, parameter: ParameterId, axis: u32) -> SymbolId {
        self.inner.call_stride_symbol(parameter, axis)
    }

    /// Allocates one target-profile constant and its symbol in this arena.
    /// The caller binds the symbol to the discovered target value before
    /// planning. There is no string-based target-constant namespace.
    pub fn target_constant(&mut self, sort: SymbolSort) -> (TargetConstantId, SymbolId) {
        self.inner.target_constant(sort)
    }

    /// The unique symbol carrying this target constant's concrete value.
    pub fn target_constant_symbol(&self, constant: TargetConstantId) -> SymbolId {
        self.inner.target_constant_symbol(constant)
    }

    /// Allocates one finite decision and records its exact domain.
    pub fn decision(&mut self, domain: FiniteDomain) -> DecisionId {
        self.inner.decision(domain)
    }

    pub fn loop_binder(&mut self) -> (LoopBinderId, SymbolId, IntExpr) {
        self.inner.loop_binder()
    }

    pub fn nat_loop_binder(&mut self) -> (LoopBinderId, SymbolId, NatExpr) {
        self.inner.nat_loop_binder()
    }

    pub fn schedule_slot(&mut self, ordinal: u32, sort: SymbolSort) -> SymbolId {
        self.inner.schedule_slot(ordinal, sort)
    }

    /// Rebase a consumed child's slot into its final schedule namespace.
    /// Expression nodes keep their SymbolId; only stable structural hashing
    /// changes to the slot's final position in the owning construction.
    pub fn rebase_schedule_slot(&mut self, symbol: SymbolId, ordinal: u32) {
        self.inner.rebase_schedule_slot(symbol, ordinal)
    }

    pub fn symbol_kind(&self, symbol: SymbolId) -> SymbolKind {
        self.inner.symbol_kind(symbol)
    }

    pub fn symbol_sort(&self, symbol: SymbolId) -> SymbolSort {
        self.inner.symbol_sort(symbol)
    }

    pub fn decision_domain(&self, decision: DecisionId) -> &FiniteDomain {
        self.inner.decision_domain(decision)
    }

    /// The unique `Int` symbol carrying this decision's selected value.
    pub fn decision_symbol(&self, decision: DecisionId) -> SymbolId {
        self.inner.decision_symbol(decision)
    }

    pub fn symbols(&self) -> impl Iterator<Item = SymbolId> + '_ {
        self.inner.symbols()
    }

    // ----- Nat -------------------------------------------------------------

    pub fn nat(&mut self, value: u64) -> NatExpr {
        self.inner.nat_const(value)
    }
    pub fn nat_symbol(&mut self, symbol: SymbolId) -> NatExpr {
        self.inner.nat_symbol(symbol)
    }
    pub fn nat_add(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        self.inner.nat_add(a, b)
    }
    pub fn nat_mul(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        self.inner.nat_mul(a, b)
    }
    /// `a - b` under the side condition `b <= a`, recorded as a side
    /// condition of the node.
    pub fn nat_sub(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        if let Some(difference) = poly::natural_difference(self, a, b) {
            return difference;
        }
        self.inner.nat_sub(a, b)
    }
    /// Floor division; records `b != 0`.
    pub fn nat_div(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        self.inner.nat_div(a, b)
    }
    /// Ceiling division; records `b != 0`.
    pub fn nat_ceil_div(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        self.inner.nat_ceil_div(a, b)
    }
    /// Remainder; records `b != 0`.
    pub fn nat_rem(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        self.inner.nat_rem(a, b)
    }
    pub fn nat_min(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        self.inner.nat_min(a, b)
    }
    pub fn nat_max(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        self.inner.nat_max(a, b)
    }
    /// Round `a` up to a multiple of `unit`; records `unit != 0`.
    pub fn nat_align_up(&mut self, a: NatExpr, unit: NatExpr) -> NatExpr {
        self.inner.nat_align_up(a, unit)
    }
    pub fn nat_select(&mut self, cond: BoolExpr, then: NatExpr, otherwise: NatExpr) -> NatExpr {
        self.inner.nat_select(cond, then, otherwise)
    }
    /// Product of shape extents.
    pub fn nat_product(&mut self, factors: &[NatExpr]) -> NatExpr {
        self.inner.nat_product(factors)
    }
    /// `fold_{binder in 0..extent} body`; `body` may mention `binder`.
    pub fn nat_fold(
        &mut self,
        op: FoldOp,
        binder: LoopBinderId,
        extent: NatExpr,
        body: NatExpr,
    ) -> NatExpr {
        self.inner.nat_fold(op, binder, extent, body)
    }
    /// `fold_{binder in start..start+extent} body`; `body` may mention `binder`.
    pub fn nat_fold_range(
        &mut self,
        op: FoldOp,
        binder: LoopBinderId,
        start: NatExpr,
        extent: NatExpr,
        body: NatExpr,
    ) -> NatExpr {
        self.inner.nat_fold_range(op, binder, start, extent, body)
    }
    /// Exact cast from a proven non-negative `Int`; records `i >= 0`.
    pub fn nat_from_int(&mut self, i: IntExpr) -> NatExpr {
        self.inner.nat_from_int(i)
    }

    // ----- Int -------------------------------------------------------------

    pub fn int(&mut self, value: i64) -> IntExpr {
        self.inner.int_const(value)
    }
    pub fn int_symbol(&mut self, symbol: SymbolId) -> IntExpr {
        self.inner.int_symbol(symbol)
    }
    pub fn int_from_nat(&mut self, n: NatExpr) -> IntExpr {
        self.inner.int_from_nat(n)
    }
    /// Injects an actual I32/U32 scalar value into mathematical Int, preserving
    /// its signedness and current bits. This does not recover producer algebra.
    pub fn int_from_scalar<T: ScalarSort>(&mut self, value: ScalarExpr<T>) -> IntExpr {
        self.int_from_scalar_value(value.erase())
    }
    pub(crate) fn int_from_scalar_value(&mut self, value: ErasedScalarExpr) -> IntExpr {
        self.inner.int_from_scalar(value)
    }
    /// The exact integer value of a typed I32/U32 source operation. Operands
    /// enter their declared word representation before the shared scalar recipe
    /// runs. Partial operations remain partial; this is not a success assertion.
    pub fn scalar_integer(
        &mut self,
        operation: crate::reference_math::ScalarOp,
        operands: &[(DType, IntExpr)],
    ) -> IntExpr {
        self.inner.scalar_integer(operation, operands)
    }
    pub(crate) fn scalar_integer_defined(&mut self, value: IntExpr) -> BoolExpr {
        self.inner.scalar_integer_defined(value)
    }
    /// `a + b`, rebuilt in the one polynomial normal form when both operands
    /// are total (`poly::integer_sum`).
    pub fn int_add(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        if let Some(sum) = poly::integer_sum(self, a, b, false) {
            return sum;
        }
        self.inner.int_add(a, b)
    }
    /// `a - b`, rebuilt in the one polynomial normal form when both operands
    /// are total (`poly::integer_sum`).
    pub fn int_sub(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        if let Some(difference) = poly::integer_sum(self, a, b, true) {
            return difference;
        }
        self.inner.int_sub(a, b)
    }
    pub fn int_mul(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        self.inner.int_mul(a, b)
    }
    /// Euclidean division; records `b != 0`. The remainder is nonnegative.
    pub fn int_div(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        self.inner.int_div(a, b)
    }
    pub fn int_rem(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        self.inner.int_rem(a, b)
    }
    pub fn int_min(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        self.inner.int_min(a, b)
    }
    pub fn int_max(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        self.inner.int_max(a, b)
    }
    pub fn int_select(&mut self, cond: BoolExpr, then: IntExpr, otherwise: IntExpr) -> IntExpr {
        self.inner.int_select(cond, then, otherwise)
    }

    // ----- Bool ------------------------------------------------------------

    pub fn bool(&mut self, value: bool) -> BoolExpr {
        self.inner.bool_const(value)
    }
    pub fn and(&mut self, a: BoolExpr, b: BoolExpr) -> BoolExpr {
        self.inner.and(a, b)
    }
    pub fn or(&mut self, a: BoolExpr, b: BoolExpr) -> BoolExpr {
        self.inner.or(a, b)
    }
    pub fn not(&mut self, a: BoolExpr) -> BoolExpr {
        self.inner.not(a)
    }
    pub fn implies(&mut self, a: BoolExpr, b: BoolExpr) -> BoolExpr {
        self.inner.implies(a, b)
    }
    /// Conservatively prove that whenever `a` evaluates to true, `b` is
    /// defined and true. This does not prove `a` total and must not replace
    /// an executable implication, whose evaluation can still fail in `a`.
    pub fn entails(&self, a: BoolExpr, b: BoolExpr) -> bool {
        self.inner.entails(a, b)
    }
    pub fn iff(&mut self, a: BoolExpr, b: BoolExpr) -> BoolExpr {
        self.inner.iff(a, b)
    }
    pub fn all(&mut self, terms: &[BoolExpr]) -> BoolExpr {
        self.inner.all(terms)
    }
    pub fn any(&mut self, terms: &[BoolExpr]) -> BoolExpr {
        self.inner.any(terms)
    }
    pub fn nat_cmp(&mut self, op: CmpOp, a: NatExpr, b: NatExpr) -> BoolExpr {
        self.inner.nat_cmp(op, a, b)
    }
    pub fn int_cmp(&mut self, op: CmpOp, a: IntExpr, b: IntExpr) -> BoolExpr {
        self.inner.int_cmp(op, a, b)
    }
    /// Finite membership of a `Nat` in an explicit set.
    pub fn nat_in(&mut self, a: NatExpr, values: &[u64]) -> BoolExpr {
        self.inner.nat_in(a, values)
    }
    /// Finite membership of a decision in a subset of its domain.
    pub fn decision_in(&mut self, decision: DecisionId, values: &[i64]) -> BoolExpr {
        self.inner.decision_in(decision, values)
    }
    /// Equality of a decision with one value of its domain.
    pub fn decision_is(&mut self, decision: DecisionId, value: i64) -> BoolExpr {
        self.inner.decision_is(decision, value)
    }
    /// Equality of a decision with one value of its domain, as an `Int`
    /// expression for use in arithmetic.
    pub fn decision_value(&mut self, decision: DecisionId) -> IntExpr {
        self.inner.decision_value(decision)
    }

    // ----- Scalar ----------------------------------------------------------

    pub fn scalar_symbol<T: ScalarSort>(&mut self, symbol: SymbolId) -> ScalarExpr<T> {
        self.inner.scalar_symbol::<T>(symbol)
    }
    pub fn scalar_const<T: ScalarSort>(&mut self, value: T::Value) -> ScalarExpr<T> {
        self.inner.scalar_const::<T>(value)
    }
    pub fn scalar_cmp<T: ScalarSort>(
        &mut self,
        op: CmpOp,
        a: ScalarExpr<T>,
        b: ScalarExpr<T>,
    ) -> BoolExpr {
        self.inner.scalar_cmp::<T>(op, a, b)
    }

    // ----- Duration --------------------------------------------------------

    /// A physical-duration interval. Terms are additive rational nanoseconds.
    pub fn duration(&mut self, terms: &[DurationTerm]) -> DurationExpr {
        self.inner.duration(terms)
    }
    pub fn duration_add(&mut self, a: DurationExpr, b: DurationExpr) -> DurationExpr {
        self.inner.duration_add(a, b)
    }
    pub fn duration_select(
        &mut self,
        cond: BoolExpr,
        then: DurationExpr,
        otherwise: DurationExpr,
    ) -> DurationExpr {
        self.inner.duration_select(cond, then, otherwise)
    }
    /// Scales a duration by an exact dynamic multiplicity.
    pub fn duration_scale(&mut self, duration: DurationExpr, by: NatExpr) -> DurationExpr {
        self.inner.duration_scale(duration, by)
    }
    /// Sums a duration body over a zero-based loop binder. Unlike
    /// `duration_scale`, this preserves exact binder-dependent demand (for
    /// example, a final partial launch chunk) by folding each additive demand
    /// term before rebuilding the duration interval.
    pub fn duration_sum(
        &mut self,
        binder: LoopBinderId,
        extent: NatExpr,
        duration: DurationExpr,
    ) -> DurationExpr {
        self.inner.duration_sum(binder, extent, duration)
    }
    /// Sums a duration body over a general half-open binder range.
    pub fn duration_sum_range(
        &mut self,
        binder: LoopBinderId,
        start: NatExpr,
        extent: NatExpr,
        duration: DurationExpr,
    ) -> DurationExpr {
        self.inner
            .duration_sum_range(binder, start, extent, duration)
    }

    // ----- named roots -----------------------------------------------------

    /// Registers a named derived root (layout stride, launch geometry, byte
    /// size) so downstream artifacts reference it by handle and consumers can
    /// enumerate what a plan evaluates.
    pub fn root(&mut self, name: RootName, node: AnyExpr) -> RootId {
        self.inner.root(name, node)
    }
    pub fn roots(&self) -> impl Iterator<Item = (RootId, &RootName, AnyExpr)> + '_ {
        self.inner.roots()
    }

    /// Deterministic content identity for an ordered set of named roots.
    /// Arena-owner nonces and raw interning indices are excluded; operation
    /// structure, constants, semantic symbol kinds, symbol ordinals, decision
    /// domains and root names are included.
    pub fn canonical_digest(&self, roots: &[RootId]) -> ExprDigest {
        self.inner.canonical_digest(roots)
    }

    // ----- analysis --------------------------------------------------------

    /// The conjunction of every side condition (nonzero divisors, exact
    /// subtraction, non-negative casts) recorded beneath `node`.
    pub fn side_conditions(&mut self, node: AnyExpr) -> BoolExpr {
        self.inner.side_conditions(node)
    }

    /// The symbols mentioned beneath `node`.
    pub fn free_symbols(&self, node: AnyExpr) -> Vec<SymbolId> {
        self.inner.free_symbols(node)
    }

    /// Partially evaluates `node` under `assignment` (decision values and any
    /// bound invocation symbols), returning a node whose remaining free
    /// symbols are exactly the unbound ones. Interned like every other node.
    pub fn partial<Sort>(
        &mut self,
        node: Expr<Sort>,
        assignment: &PartialAssignment,
    ) -> Expr<Sort> {
        self.inner.partial(node, assignment)
    }

    /// Resolves semantic integer values to their already-lowered mathematical
    /// expressions in this arena. The owning lowering must preserve dominance
    /// and lexical scope; unresolved values remain runtime dependencies.
    pub fn resolve_runtime_values(
        &mut self,
        node: IntExpr,
        values: &[(crate::ids::SemanticValueId, IntExpr)],
    ) -> IntExpr {
        self.inner.resolve_runtime_values(node, values)
    }

    /// Replaces natural symbols by expressions that equal them where `node`
    /// is evaluated (for example a product slot proved equal to its source).
    pub fn substitute_nat<Sort>(
        &mut self,
        node: Expr<Sort>,
        values: &[(SymbolId, NatExpr)],
    ) -> Expr<Sort> {
        self.inner.substitute_nat(node, values)
    }

    /// The `NatExpr` counterpart of [`ExprArena::resolve_runtime_values`].
    pub fn resolve_runtime_nat(
        &mut self,
        node: NatExpr,
        values: &[(crate::ids::SemanticValueId, IntExpr)],
    ) -> NatExpr {
        self.inner.resolve_runtime_values(node, values)
    }

    /// True iff evaluation cannot fail for any assignment of its free symbols.
    pub fn is_total(&self, expression: AnyExpr) -> bool {
        self.inner.expression_total(expression)
    }

    /// Total evaluation under a complete assignment.
    pub fn eval_nat(&self, node: NatExpr, values: &Assignment) -> Result<BigUint, EvalError> {
        self.inner.eval_nat(node, values)
    }
    pub fn eval_int(&self, node: IntExpr, values: &Assignment) -> Result<BigInt, EvalError> {
        self.inner.eval_int(node, values)
    }
    /// Checked projection for a consumer whose physical natural representation is u64.
    pub fn eval_nat_u64(&self, node: NatExpr, values: &Assignment) -> Result<u64, EvalError> {
        self.eval_nat(node, values)?.try_into().map_err(|_| EvalError::Unrepresentable)
    }
    /// Checked projection for a consumer whose physical signed representation is i64.
    pub fn eval_int_i64(&self, node: IntExpr, values: &Assignment) -> Result<i64, EvalError> {
        self.eval_int(node, values)?.try_into().map_err(|_| EvalError::Unrepresentable)
    }
    /// Embed an arbitrary exact constant in the same expression DAG.
    pub fn nat_exact(&mut self, value: BigUint) -> NatExpr { self.inner.nat_exact(value) }
    pub fn int_exact(&mut self, value: BigInt) -> IntExpr { self.inner.int_exact(value) }
    pub fn eval_bool(&self, node: BoolExpr, values: &Assignment) -> Result<bool, EvalError> {
        self.inner.eval_bool(node, values)
    }
    pub fn eval_duration(
        &self,
        node: DurationExpr,
        values: &Assignment,
    ) -> Result<DurationEstimate, EvalError> {
        self.inner.eval_duration(node, values)
    }

    /// Compiles a node into a self-contained evaluator that no longer needs
    /// the arena. Used by `ExecutableVariant` for guards, durations, layouts and
    /// geometry (§5.3).
    pub fn compile_nat(&self, node: NatExpr) -> compiled::Compiled<BigUint> {
        self.inner.compile_nat(node)
    }
    pub fn compile_int(&self, node: IntExpr) -> compiled::Compiled<BigInt> {
        self.inner.compile_int(node)
    }
    pub fn compile_bool(&self, node: BoolExpr) -> compiled::Compiled<bool> {
        self.inner.compile_bool(node)
    }
    /// Compiles a solver-side predicate whose remaining symbols are finite
    /// decisions. The distinct result type cannot enter invocation APIs.
    #[doc(hidden)]
    pub fn compile_decision_bool(&self, node: BoolExpr) -> compiled::CompiledDecisionPredicate {
        self.inner.compile_decision_bool(node)
    }
    pub fn compile_duration(&self, node: DurationExpr) -> compiled::Compiled<DurationEstimate> {
        self.inner.compile_duration(node)
    }
    /// Compiles after fixing target constants and finite decisions without
    /// mutating this arena. Fixed symbols are captured by the evaluator and
    /// are absent from its runtime binding table.
    pub fn compile_nat_with(
        &self,
        node: NatExpr,
        fixed: &PartialAssignment,
    ) -> compiled::Compiled<BigUint> {
        self.inner.compile_nat_with(node, fixed)
    }
    pub fn compile_int_with(
        &self,
        node: IntExpr,
        fixed: &PartialAssignment,
    ) -> compiled::Compiled<BigInt> {
        self.inner.compile_int_with(node, fixed)
    }
    pub fn compile_bool_with(
        &self,
        node: BoolExpr,
        fixed: &PartialAssignment,
    ) -> compiled::Compiled<bool> {
        self.inner.compile_bool_with(node, fixed)
    }
    pub fn compile_duration_with(
        &self,
        node: DurationExpr,
        fixed: &PartialAssignment,
    ) -> compiled::Compiled<DurationEstimate> {
        self.inner.compile_duration_with(node, fixed)
    }

    /// Structural view of one node, for solver export and printing. This is a
    /// read-only projection of the interned node, not a second AST.
    pub fn view(&self, node: AnyExpr) -> NodeView<'_> {
        self.inner.view(node)
    }
}

/// Comparison operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Scalar sorts admitted in `ScalarExpr<T>`.
pub trait ScalarSort: 'static + Copy + fmt::Debug + private::Sealed {
    type Value: Copy + fmt::Debug + PartialEq;
    const DTYPE: DType;
    #[doc(hidden)]
    fn encode(value: Self::Value) -> u32;
}

mod private {
    pub trait Sealed {}
}

macro_rules! scalar_sort {
    ($name:ident, $value:ty, $dtype:expr, $encode:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {}
        impl private::Sealed for $name {}
        impl ScalarSort for $name {
            type Value = $value;
            const DTYPE: DType = $dtype;
            fn encode(value: Self::Value) -> u32 {
                ($encode)(value)
            }
        }
    };
}

scalar_sort!(F32, f32, DType::F32, f32::to_bits);
scalar_sort!(F16, u16, DType::F16, u32::from);
scalar_sort!(BF16, u16, DType::BF16, u32::from);
scalar_sort!(BoolScalar, bool, DType::Bool, u32::from);
scalar_sort!(I32, i32, DType::I32, |value: i32| value as u32);
scalar_sort!(U32, u32, DType::U32, |value: u32| value);

/// One additive contribution to a physical duration interval, in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DurationTerm {
    /// Exact structural multiplicity derived from executable semantics.
    pub demand: NatExpr,
    /// Lower service time per unit of demand.
    pub lower_numerator: u64,
    /// Upper service time per unit of demand.
    pub upper_numerator: u64,
    /// Non-zero constant denominator.
    pub denominator: u64,
}

/// One exact non-negative rational duration in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RationalDuration {
    numerator: u128,
    denominator: u64,
}

impl PartialOrd for RationalDuration {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RationalDuration {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        compare_nonnegative_rationals(
            self.numerator,
            u128::from(self.denominator),
            other.numerator,
            u128::from(other.denominator),
        )
    }
}

/// Compares two non-negative rational numbers without a cross product. Duration
/// numerators are `u128` and denominators are `u64`; multiplying them would
/// silently overflow for valid large shapes. Continued-fraction comparison
/// is exact and bounded by the Euclidean algorithm.
fn compare_nonnegative_rationals(
    mut left: u128,
    mut left_denominator: u128,
    mut right: u128,
    mut right_denominator: u128,
) -> std::cmp::Ordering {
    assert!(left_denominator != 0 && right_denominator != 0);
    let mut reversed = false;
    loop {
        let left_integer = left / left_denominator;
        let right_integer = right / right_denominator;
        let integer_order = left_integer.cmp(&right_integer);
        if integer_order != std::cmp::Ordering::Equal {
            return if reversed {
                integer_order.reverse()
            } else {
                integer_order
            };
        }

        let left_remainder = left % left_denominator;
        let right_remainder = right % right_denominator;
        match (left_remainder == 0, right_remainder == 0) {
            (true, true) => return std::cmp::Ordering::Equal,
            (true, false) => {
                let ordering = std::cmp::Ordering::Less;
                return if reversed {
                    ordering.reverse()
                } else {
                    ordering
                };
            }
            (false, true) => {
                let ordering = std::cmp::Ordering::Greater;
                return if reversed {
                    ordering.reverse()
                } else {
                    ordering
                };
            }
            (false, false) => {
                left = left_denominator;
                left_denominator = left_remainder;
                right = right_denominator;
                right_denominator = right_remainder;
                reversed = !reversed;
            }
        }
    }
}

impl RationalDuration {
    pub fn numerator(self) -> u128 {
        self.numerator
    }
    pub fn denominator(self) -> u64 {
        self.denominator
    }
}

/// Evaluated physical-duration interval. Both bounds use one exact unit and
/// retain the uncertainty acquired by the target profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DurationEstimate {
    lower: RationalDuration,
    upper: RationalDuration,
}

impl DurationEstimate {
    pub fn lower(self) -> RationalDuration {
        self.lower
    }
    pub fn upper(self) -> RationalDuration {
        self.upper
    }
    pub fn overlaps(self, other: Self) -> bool {
        self.lower <= other.upper && other.lower <= self.upper
    }
}

/// A sort-erased handle, for analysis entry points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnyExpr {
    Nat(NatExpr),
    Int(IntExpr),
    Bool(BoolExpr),
    Duration(DurationExpr),
    Scalar(ErasedScalarExpr),
}

/// Sort-erased scalar handle. Its fields remain private so `AnyExpr` does not
/// become a handle-forging back door.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ErasedScalarExpr {
    owner: ArenaId,
    index: u32,
}

impl From<NatExpr> for AnyExpr {
    fn from(e: NatExpr) -> Self {
        AnyExpr::Nat(e)
    }
}
impl From<IntExpr> for AnyExpr {
    fn from(e: IntExpr) -> Self {
        AnyExpr::Int(e)
    }
}
impl From<BoolExpr> for AnyExpr {
    fn from(e: BoolExpr) -> Self {
        AnyExpr::Bool(e)
    }
}
impl From<DurationExpr> for AnyExpr {
    fn from(e: DurationExpr) -> Self {
        AnyExpr::Duration(e)
    }
}
impl<T: ScalarSort> From<ScalarExpr<T>> for AnyExpr {
    fn from(e: ScalarExpr<T>) -> Self {
        AnyExpr::Scalar(ErasedScalarExpr {
            owner: e.owner,
            index: e.index,
        })
    }
}

impl<T: ScalarSort> Expr<sort::Scalar<T>> {
    pub(crate) fn erase(self) -> ErasedScalarExpr {
        ErasedScalarExpr {
            owner: self.owner,
            index: self.index,
        }
    }
}

/// Name of a derived root.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RootName {
    /// An actual scalar/quantity operand of a structured region result.
    RegionOperand { operand: u32 },
    /// Byte size of one global allocation.
    AllocationBytes {
        allocation: u32,
    },
    /// Byte offset of a buffer view within its allocation.
    ViewOffset {
        view: u32,
    },
    /// Element stride of one axis of a buffer view.
    ViewStride {
        view: u32,
        axis: u32,
    },
    /// Extent of one axis of a buffer view.
    ViewExtent {
        view: u32,
        axis: u32,
    },
    /// Launch grid of one kernel, one axis.
    LaunchGrid {
        launch: u32,
        axis: u8,
    },
    /// Workgroup size of one kernel, one axis.
    Workgroup {
        launch: u32,
        axis: u8,
    },
    /// Whether a launch is empty and therefore skipped.
    LaunchEmpty {
        launch: u32,
    },
    /// The complete hard-constraint conjunction of an implementation.
    HardConstraints,
    /// Predicate of one structured schedule branch.
    ScheduleCondition {
        control: u32,
    },
    /// Inclusive start of one structured schedule repeat.
    RepeatStart {
        repeat: u32,
    },
    /// Exclusive end of one structured schedule repeat.
    RepeatEnd {
        repeat: u32,
    },
    /// The finite choice symbol controlling one schedule `Choose`.
    ScheduleChoice {
        control: u32,
    },
    /// One arena-valued natural argument of a kernel.
    KernelNatArgument {
        kernel: u32,
        argument: u32,
    },
    /// One arena-valued scalar argument of a kernel.
    KernelScalarArgument {
        kernel: u32,
        argument: u32,
    },
    /// One axis of a schedule scalar-read index.
    ScalarReadIndex {
        step: u32,
        axis: u32,
    },
    /// Actual value expression of one reached host schedule operation.
    HostEvaluation {
        step: u32,
    },
    /// Declared public extent checked by one reached result publication.
    PublishedExtent {
        step: u32,
        axis: u32,
    },
    /// One extent of a kernel-local allocation.
    LocalExtent {
        kernel: u32,
        local: u32,
        axis: u32,
    },
    LocalOffset {
        launch: u32,
        local: u32,
    },
    LocalStride {
        launch: u32,
        local: u32,
        axis: u32,
    },
    LocalClassBytes {
        launch: u32,
        class: u8,
    },
    /// Canonical native-unit offset of one addressable intrinsic lease.
    AddressableResourceOffset {
        kernel: u32,
        lease: u32,
    },
    /// Native-unit extent of one addressable intrinsic lease.
    AddressableResourceUnits {
        kernel: u32,
        lease: u32,
    },
    /// Exact physical invocation-scratch bytes used to realize one local
    /// address-space class for a launch.
    LaunchScratchBytes {
        launch: u32,
        class: u8,
    },
    /// One compiler-owned backend ABI allocation of a launch.
    LaunchAbiBytes {
        launch: u32,
        allocation: u32,
    },
    /// One intrinsic-declared resource expression.
    IntrinsicWorkgroupBytes {
        kernel: u32,
        resource: u32,
    },
    IntrinsicParticipantBytes {
        kernel: u32,
        resource: u32,
    },
    IntrinsicRegisterBytes {
        kernel: u32,
        resource: u32,
    },
    /// An implementation's applicability guard.
    Guard,
    /// An implementation's modeled physical duration.
    Duration,
    /// A numerical error bound of one output.
    ErrorBound {
        output: u32,
    },
    /// Exact finite-choice predicate guarding one composed child numerical
    /// transfer. This is part of implementation identity; the arena-local
    /// decision handle itself is never persisted.
    NumericalCondition {
        child: u32,
    },
    /// Dynamic multiplicity of one operation in the recursively composed
    /// numerical transfer.
    NumericalOperationMultiplicity {
        operation: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RootId {
    owner: ArenaId,
    index: u32,
}

/// Stable SHA-256 content identity of canonical expression roots.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprDigest([u8; 32]);

impl ExprDigest {
    pub fn bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Debug for ExprDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// A read-only structural projection of one node.
#[derive(Clone, Copy, Debug)]
pub enum NodeView<'a> {
    NatConst(u64),
    IntConst(i64),
    BoolConst(bool),
    ScalarConst {
        dtype: DType,
        bits: u32,
    },
    ScalarInteger {
        operation: crate::reference_math::ScalarOp,
        operands: &'a [(DType, IntExpr)],
    },
    Symbol(SymbolId),
    Unary {
        op: UnaryOp,
        operand: AnyExpr,
    },
    Binary {
        op: BinaryOp,
        lhs: AnyExpr,
        rhs: AnyExpr,
    },
    Nary {
        op: NaryOp,
        operands: &'a [AnyExpr],
    },
    Select {
        cond: BoolExpr,
        then: AnyExpr,
        otherwise: AnyExpr,
    },
    Cmp {
        op: CmpOp,
        lhs: AnyExpr,
        rhs: AnyExpr,
    },
    In {
        operand: AnyExpr,
        values: &'a [i64],
    },
    Fold {
        op: FoldOp,
        binder: LoopBinderId,
        start: NatExpr,
        extent: NatExpr,
        body: NatExpr,
    },
    Duration(&'a [DurationTerm]),
    DurationScale {
        duration: DurationExpr,
        by: NatExpr,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Not,
    NatFromInt,
    IntFromNat,
    IntFromScalar,
    /// Recipe-derived definedness of a ScalarInteger node.
    ScalarIntegerDefined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    CeilDiv,
    Rem,
    Min,
    Max,
    AlignUp,
    And,
    Or,
    Implies,
    Iff,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NaryOp {
    All,
    Any,
    Product,
    DurationAdd,
}

/// Values for a subset of symbols.
#[derive(Clone, Debug, Default)]
pub struct PartialAssignment {
    values: Vec<(SymbolId, SymbolValue)>,
}

impl PartialAssignment {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn bind(&mut self, symbol: SymbolId, value: SymbolValue) {
        self.values.retain(|(s, _)| *s != symbol);
        self.values.push((symbol, value));
    }
    pub fn get(&self, symbol: SymbolId) -> Option<SymbolValue> {
        self.values
            .iter()
            .find(|(s, _)| *s == symbol)
            .map(|(_, v)| v.clone())
    }
    pub fn iter(&self) -> impl Iterator<Item = (SymbolId, SymbolValue)> + '_ {
        self.values.iter().cloned()
    }
}

/// Values for every symbol a node mentions. Missing symbols are an
/// [`EvalError::Unbound`].
pub type Assignment = PartialAssignment;

#[derive(Clone, Debug, PartialEq)]
pub enum SymbolValue {
    Nat(BigUint),
    Int(BigInt),
    F32(f32),
    F16(u16),
    BF16(u16),
    Bool(bool),
    I32(i32),
    U32(u32),
}

impl SymbolValue {
    pub fn retained_metadata_bytes(&self) -> usize {
        let bits = match self {
            Self::Nat(value) => value.bits(),
            Self::Int(value) => value.bits(),
            _ => 0,
        };
        std::mem::size_of::<Self>().saturating_add(
            usize::try_from(bits.div_ceil(8)).unwrap_or(usize::MAX))
    }
    /// Encode an existing single-word native ABI slot. Quantity values are
    /// checked before encoding; this is never a mathematical truncation.
    pub fn try_word64(self) -> Result<u64, EvalError> {
        Ok(match self {
            Self::Nat(value) => value.try_into().map_err(|_| EvalError::Unrepresentable)?,
            Self::Int(value) => i64::try_from(value).map_err(|_| EvalError::Unrepresentable)? as u64,
            Self::F32(value) => u64::from(value.to_bits()),
            Self::F16(value) | Self::BF16(value) => u64::from(value),
            Self::I32(value) => u64::from(value as u32),
            Self::U32(value) => u64::from(value),
            Self::Bool(value) => u64::from(value),
        })
    }
}

/// Evaluation failure. `Unbound` is a caller bug (a compiled evaluator is
/// always paired with the values its schema demands); the others are the
/// domain restriction of §5.2 surfacing as data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalError {
    Unbound(SymbolId),
    DivisionByZero,
    NegativeNat,
    Unrepresentable,
    ScalarFailure(crate::reference_math::ScalarFailure),
}

/// Exact semantic entry domain. Only caller-provided dimensions and scalars
/// may occur; target facts and compiler/runtime state are excluded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntryPredicate {
    node: BoolExpr,
}

impl EntryPredicate {
    pub fn new(arena: &ExprArena, node: BoolExpr) -> Result<Self, PredicateLevelError> {
        predicate(arena, node, |kind| {
            matches!(
                kind,
                SymbolKind::CallDimension(_) | SymbolKind::CallScalar(_)
            )
        })
        .map(|()| Self { node })
    }
    pub fn node(self) -> BoolExpr {
        self.node
    }
}

/// Target-specific shape domain and executable-variant guard. Arbitrary data
/// scalars belong to invocation validation and cannot influence physical-plan
/// legality. Target constants are fixed before solver export; decisions, loop
/// binders and slots are forbidden.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TargetPredicate {
    node: BoolExpr,
}

impl TargetPredicate {
    pub fn new(arena: &ExprArena, node: BoolExpr) -> Result<Self, PredicateLevelError> {
        predicate(arena, node, |kind| {
            matches!(
                kind,
                SymbolKind::CallDimension(_) | SymbolKind::TargetConstant(_)
            )
        })
        .map(|()| Self { node })
    }
    pub fn node(self) -> BoolExpr {
        self.node
    }
}

/// Invocation-level guard: every free symbol is evaluable from the invocation
/// and fixed target/member facts ([`SymbolKind::is_invocation`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InvocationPredicate {
    node: BoolExpr,
}

impl InvocationPredicate {
    /// The only constructor. Err iff some free symbol has `is_invocation() == false`.
    pub fn new(arena: &ExprArena, node: BoolExpr) -> Result<Self, PredicateLevelError> {
        predicate(arena, node, SymbolKind::is_invocation).map(|()| Self { node })
    }
    pub fn node(self) -> BoolExpr {
        self.node
    }
}

/// Runtime structured-control predicate. Decisions are forbidden because
/// freezing must resolve them; call scalars, lexical binders, and mutable
/// schedule slots are legitimate runtime inputs evaluated by the schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SchedulePredicate {
    node: BoolExpr,
}

impl SchedulePredicate {
    pub fn new(arena: &ExprArena, node: BoolExpr) -> Result<Self, PredicateLevelError> {
        predicate(arena, node, |kind| {
            matches!(
                kind,
                SymbolKind::CallDimension(_)
                    | SymbolKind::CallScalar(_)
                    | SymbolKind::TargetConstant(_)
                    | SymbolKind::LoopBinder(_)
                    | SymbolKind::ScheduleSlot(_)
            )
        })
        .map(|()| Self { node })
    }
    pub fn node(self) -> BoolExpr {
        self.node
    }
}

fn predicate(
    arena: &ExprArena,
    node: BoolExpr,
    allowed: impl Fn(SymbolKind) -> bool,
) -> Result<(), PredicateLevelError> {
    let offending = arena
        .free_symbols(AnyExpr::Bool(node))
        .into_iter()
        .find(|symbol| !allowed(arena.symbol_kind(*symbol)));
    match offending {
        Some(symbol) => Err(PredicateLevelError::ForbiddenSymbol(symbol)),
        None => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredicateLevelError {
    ForbiddenSymbol(SymbolId),
}

mod internals;

#[cfg(test)]
mod exact_value_tests {
    use super::*;

    #[test]
    fn exact_quantities_survive_symbols_compilation_and_specialization() {
        let mut arena = ExprArena::default();
        let (_, symbol) = arena.target_constant(SymbolSort::Int);
        let input = arena.int_symbol(symbol);
        let square = arena.int_mul(input, input);
        let restored = arena.int_div(square, input);
        let huge = -(BigInt::from(1u32) << 160usize) + BigInt::from(7u32);
        let mut values = Assignment::new();
        values.bind(symbol, SymbolValue::Int(huge.clone()));
        let mut invocation = compiled::InvocationValues::new();
        invocation.bind(symbol, SymbolValue::Int(huge.clone()));
        assert_eq!(arena.eval_int(restored, &values).unwrap(), huge);
        assert_eq!(arena.compile_int(restored).evaluate(&invocation).unwrap(), huge);
        let partial = arena.partial(restored, &values);
        assert_eq!(arena.eval_int(partial, &Assignment::new()).unwrap(), huge);
        let captured = arena.compile_int_with(restored, &values);
        assert!(captured.reads().is_empty());
        assert_eq!(captured.evaluate(&compiled::InvocationValues::new()).unwrap(), huge);
        assert_eq!(captured.evaluate_i64(&compiled::InvocationValues::new()), Err(EvalError::Unrepresentable));
    }

    #[test]
    fn exact_constants_and_checked_projections_have_distinct_domains() {
        let mut arena = ExprArena::default();
        let huge = (BigUint::from(1u32) << 130usize) + BigUint::from(3u32);
        let value = arena.nat_exact(huge.clone());
        let values = compiled::InvocationValues::new();
        let compiled = arena.compile_nat(value);
        assert_eq!(compiled.evaluate(&values).unwrap(), huge);
        assert_eq!(compiled.evaluate_u64(&values), Err(EvalError::Unrepresentable));
        assert_eq!(SymbolValue::Nat(huge).try_word64(), Err(EvalError::Unrepresentable));
        let negative = arena.int_exact(-(BigInt::from(1u32) << 100usize));
        let invalid = arena.nat_from_int(negative);
        assert_eq!(arena.eval_nat(invalid, &Assignment::new()), Err(EvalError::NegativeNat));
    }
}

#[cfg(test)]
mod invocation_tests {
    use super::*;
    use crate::ids::{FunctionId, ProgramId, SchemaId, SemanticValueId};

    #[test]
    fn call_stride_symbols_are_interned_nat_invocation_symbols() {
        let mut arena = ExprArena::new();
        let schema = SchemaId::fresh();
        let parameter = ParameterId::new(schema, 0);
        let other = ParameterId::new(schema, 1);
        let stride = arena.call_stride_symbol(parameter, 1);
        assert_eq!(arena.call_stride_symbol(parameter, 1), stride);
        assert_ne!(arena.call_stride_symbol(parameter, 0), stride);
        assert_ne!(arena.call_stride_symbol(other, 1), stride);
        assert_eq!(arena.symbol_kind(stride), SymbolKind::CallStride(parameter, 1));
        assert_eq!(arena.symbol_sort(stride), SymbolSort::Nat);
        assert!(arena.symbol_kind(stride).is_invocation());

        let axis0 = arena.call_stride_symbol(parameter, 0);
        let first = arena.nat_symbol(stride);
        let second = arena.nat_symbol(axis0);
        let first_root = arena.root(RootName::ViewStride { view: 0, axis: 0 }, first.into());
        let second_root = arena.root(RootName::ViewStride { view: 0, axis: 0 }, second.into());
        assert_ne!(
            arena.canonical_digest(&[first_root]),
            arena.canonical_digest(&[second_root])
        );
    }

    #[test]
    fn invocation_predicate_admits_exactly_invocation_symbols() {
        let mut arena = ExprArena::new();
        let schema = SchemaId::fresh();
        let (_, dimension) = arena.call_dimension(DimensionId::new(schema, 0));
        let stride = arena.call_stride_symbol(ParameterId::new(schema, 0), 0);
        let stride = arena.nat_symbol(stride);
        let scalar = arena.call_scalar(
            ScalarArgument { parameter: ParameterId::new(schema, 1), component: ScalarComponent::Value },
            SymbolSort::Nat,
        );
        let scalar = arena.nat_symbol(scalar);
        let (_, constant) = arena.target_constant(SymbolSort::Nat);
        let constant = arena.nat_symbol(constant);
        let decision = arena.decision(FiniteDomain::new(vec![1, 2]).unwrap());
        let chosen = arena.decision_is(decision, 2);
        let product = arena.nat_mul(dimension, stride);
        let bounded = arena.nat_cmp(CmpOp::Le, product, constant);
        let with_scalar = arena.nat_cmp(CmpOp::Lt, scalar, dimension);
        let admitted = arena.all(&[bounded, with_scalar, chosen]);
        let predicate = InvocationPredicate::new(&arena, admitted).unwrap();
        assert_eq!(predicate.node(), admitted);

        let (_, binder, _) = arena.nat_loop_binder();
        let slot = arena.schedule_slot(0, SymbolSort::Nat);
        let function = FunctionId::new(ProgramId::fresh(), 0);
        let (runtime, _) = arena.runtime_value(SemanticValueId::new(function, 0));
        let (template, _) = arena.template_dimension(0);
        for forbidden in [binder, slot, runtime, template] {
            assert!(!arena.symbol_kind(forbidden).is_invocation());
            let term = match arena.symbol_sort(forbidden) {
                SymbolSort::Nat => arena.nat_symbol(forbidden),
                _ => {
                    let value = arena.int_symbol(forbidden);
                    arena.nat_from_int(value)
                }
            };
            let condition = arena.nat_cmp(CmpOp::Lt, term, dimension);
            let condition = arena.and(admitted, condition);
            assert_eq!(
                InvocationPredicate::new(&arena, condition),
                Err(PredicateLevelError::ForbiddenSymbol(forbidden))
            );
        }
    }

    #[test]
    fn totality_and_runtime_nat_resolution_are_public() {
        let mut arena = ExprArena::new();
        let (_, dimension) = arena.call_dimension(DimensionId::new(SchemaId::fresh(), 0));
        let one = arena.nat(1);
        let total = arena.nat_add(dimension, one);
        let partial = arena.nat_div(one, dimension);
        assert!(arena.is_total(total.into()));
        assert!(!arena.is_total(partial.into()));

        let function = FunctionId::new(ProgramId::fresh(), 0);
        let value = SemanticValueId::new(function, 0);
        let (_, runtime) = arena.runtime_value(value);
        let node = arena.nat_from_int(runtime);
        let node = arena.nat_add(node, one);
        let five = arena.int(5);
        let resolved = arena.resolve_runtime_nat(node, &[(value, five)]);
        assert!(arena.free_symbols(resolved.into()).is_empty());
        assert_eq!(arena.eval_nat_u64(resolved, &Assignment::new()).unwrap(), 6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_word_projection_folds_the_actual_wrapped_result() {
        use crate::reference_math::ScalarOp;
        use crate::syntax::ast::BinaryOp;
        let mut arena = ExprArena::new();
        let max = arena.int(i64::from(i32::MAX));
        let one = arena.int(1);
        let wrapped = arena.scalar_integer(
            ScalarOp::Binary(BinaryOp::Add),
            &[(DType::I32, max), (DType::I32, one)],
        );
        assert!(matches!(
            arena.view(wrapped.into()),
            NodeView::IntConst(-2147483648)
        ));
        let unsigned = arena.int(i64::from(u32::MAX));
        let signed = arena.scalar_integer(ScalarOp::Cast(DType::I32), &[(DType::U32, unsigned)]);
        assert!(matches!(arena.view(signed.into()), NodeView::IntConst(-1)));
    }

    #[test]
    fn symbolic_word_projection_preserves_conversion_before_arithmetic() {
        use crate::reference_math::ScalarOp;
        use crate::syntax::ast::BinaryOp;
        let mut arena = ExprArena::new();
        let (_, symbol) = arena.target_constant(SymbolSort::Nat);
        let input = arena.nat_symbol(symbol);
        let input = arena.int_from_nat(input);
        let two = arena.int(2);
        let projected = arena.scalar_integer(
            ScalarOp::Binary(BinaryOp::Div),
            &[(DType::U32, input), (DType::U32, two)],
        );
        let mut values = Assignment::new();
        values.bind(symbol, SymbolValue::Nat(((1_u64 << 32) + 2).into()));
        // U32(2^32 + 2) / U32(2) is 1; narrowing the mathematical quotient
        // afterward would instead yield 2^31 + 1.
        assert_eq!(arena.eval_int_i64(projected, &values).unwrap(), 1);
        let compiled = arena.compile_int(projected);
        let mut invocation = compiled::InvocationValues::new();
        invocation.bind(symbol, SymbolValue::Nat(((1_u64 << 32) + 2).into()));
        assert_eq!(compiled.evaluate_i64(&invocation).unwrap(), 1);
        let partial = arena.partial(projected, &values);
        assert_eq!(arena.eval_int_i64(partial, &Assignment::new()).unwrap(), 1);
    }

    #[test]
    fn partial_word_recipe_has_derived_definedness_and_no_numeric_placeholder() {
        use crate::reference_math::{ScalarFailure, ScalarOp};
        use crate::syntax::ast::BinaryOp;
        let mut arena = ExprArena::new();
        let (_, symbol) = arena.target_constant(SymbolSort::Int);
        let denominator = arena.int_symbol(symbol);
        let numerator = arena.int(i64::from(i32::MIN));
        let quotient = arena.scalar_integer(
            ScalarOp::Binary(BinaryOp::Div),
            &[(DType::I32, numerator), (DType::I32, denominator)],
        );
        let defined = arena.side_conditions(quotient.into());
        let guard_defined = arena.side_conditions(defined.into());
        for (input, failure) in [
            (0, ScalarFailure::IntegerDivisionByZero),
            (-1, ScalarFailure::SignedDivisionOverflow),
        ] {
            let mut values = Assignment::new();
            values.bind(symbol, SymbolValue::Int((input).into()));
            assert_eq!(
                arena.eval_int_i64(quotient, &values),
                Err(EvalError::ScalarFailure(failure))
            );
            assert!(!arena.eval_bool(defined, &values).unwrap());
            assert!(arena.eval_bool(guard_defined, &values).unwrap());
            let specialized = arena.partial(defined, &values);
            assert!(!arena.eval_bool(specialized, &Assignment::new()).unwrap());
        }
        let mut values = Assignment::new();
        values.bind(symbol, SymbolValue::Int((3).into()));
        assert!(arena.eval_bool(defined, &values).unwrap());
        assert_eq!(arena.eval_int_i64(quotient, &values).unwrap(), -715827883);
    }

    #[test]
    fn mathematical_intermediates_exceed_host_words_without_changing_results() {
        let mut arena = ExprArena::new();
        let (_, symbol) = arena.target_constant(SymbolSort::Nat);
        let x = arena.nat_symbol(symbol);
        let square = arena.nat_mul(x, x);
        let cube = arena.nat_mul(square, x);
        let recovered = arena.nat_div(cube, square);
        let zero = arena.nat(0);
        let present = arena.nat_cmp(CmpOp::Gt, x, zero);
        let lazy = arena.nat_select(present, recovered, zero);
        let same = arena.nat_cmp(CmpOp::Eq, lazy, x);
        let compiled = arena.compile_nat(lazy);
        let compiled_same = arena.compile_bool(same);
        for actual in [0, 1, 1_u64 << 63, u64::MAX] {
            let mut values = Assignment::new();
            values.bind(symbol, SymbolValue::Nat((actual).into()));
            let mut invocation = compiled::InvocationValues::new();
            invocation.bind(symbol, SymbolValue::Nat((actual).into()));
            assert_eq!(arena.eval_nat_u64(lazy, &values).unwrap(), actual);
            assert_eq!(compiled.evaluate_u64(&invocation).unwrap(), actual);
            assert!(arena.eval_bool(same, &values).unwrap());
            assert!(compiled_same.evaluate(&invocation).unwrap());
        }
    }

    #[test]
    fn signed_division_folding_and_both_evaluators_share_euclidean_meaning() {
        let mut arena = ExprArena::new();
        let (_, numerator) = arena.target_constant(SymbolSort::Int);
        let (_, denominator) = arena.target_constant(SymbolSort::Int);
        let a = arena.int_symbol(numerator);
        let b = arena.int_symbol(denominator);
        let quotient = arena.int_div(a, b);
        let remainder = arena.int_rem(a, b);
        let compiled_q = arena.compile_int(quotient);
        let compiled_r = arena.compile_int(remainder);
        for (a, b) in [(-7_i64, 3_i64), (-7, -3), (7, -3), (7, 3)] {
            let mut values = Assignment::new();
            values.bind(numerator, SymbolValue::Int((a).into()));
            values.bind(denominator, SymbolValue::Int((b).into()));
            let mut invocation = compiled::InvocationValues::new();
            invocation.bind(numerator, SymbolValue::Int((a).into()));
            invocation.bind(denominator, SymbolValue::Int((b).into()));
            let expected_q = a.div_euclid(b);
            let expected_r = a.rem_euclid(b);
            assert_eq!(arena.eval_int_i64(quotient, &values).unwrap(), expected_q);
            assert_eq!(arena.eval_int_i64(remainder, &values).unwrap(), expected_r);
            assert_eq!(compiled_q.evaluate_i64(&invocation).unwrap(), expected_q);
            assert_eq!(compiled_r.evaluate_i64(&invocation).unwrap(), expected_r);
            let a = arena.int(a);
            let b = arena.int(b);
            let q = arena.int_div(a, b);
            let r = arena.int_rem(a, b);
            assert_eq!(arena.eval_int_i64(q, &Assignment::new()).unwrap(), expected_q);
            assert_eq!(arena.eval_int_i64(r, &Assignment::new()).unwrap(), expected_r);
        }
        // The mathematical quotient is representable as a Nat even though it
        // exceeds the signed word used for these two source constants.
        let min = arena.int(i64::MIN);
        let minus_one = arena.int(-1);
        let q = arena.int_div(min, minus_one);
        let q = arena.nat_from_int(q);
        assert_eq!(arena.eval_nat_u64(q, &Assignment::new()).unwrap(), 1_u64 << 63);
    }

    #[test]
    fn integer_word_projection_preserves_signedness_and_partial_substitution() {
        let mut arena = ExprArena::new();
        let (_, unsigned) = arena.target_constant(SymbolSort::Scalar(crate::types::DType::U32));
        let (_, signed) = arena.target_constant(SymbolSort::Scalar(crate::types::DType::I32));
        let unsigned_expr = arena.scalar_symbol::<U32>(unsigned);
        let signed_expr = arena.scalar_symbol::<I32>(signed);
        let unsigned_expr = arena.int_from_scalar(unsigned_expr);
        let signed_expr = arena.int_from_scalar(signed_expr);
        let mut values = Assignment::new();
        values.bind(unsigned, SymbolValue::U32(u32::MAX));
        values.bind(signed, SymbolValue::I32(-1));
        assert_eq!(
            arena.eval_int_i64(unsigned_expr, &values).unwrap(),
            i64::from(u32::MAX)
        );
        assert_eq!(arena.eval_int_i64(signed_expr, &values).unwrap(), -1);
        let mut fixed = PartialAssignment::new();
        fixed.bind(unsigned, SymbolValue::U32(u32::MAX));
        fixed.bind(signed, SymbolValue::I32(-1));
        let specialized = arena.partial(unsigned_expr, &fixed);
        assert!(matches!(
            arena.view(specialized.into()),
            NodeView::IntConst(4294967295)
        ));
        assert_eq!(
            arena
                .compile_int_with(signed_expr, &fixed)
                .evaluate_i64(&compiled::InvocationValues::new())
                .unwrap(),
            -1
        );
    }

    #[test]
    fn folds_and_signed_intermediates_remain_exact_beyond_i128() {
        let mut arena = ExprArena::new();
        let n = arena.nat(u64::MAX);
        let large = arena.nat_product(&[n, n, n, n]);
        let (binder, _, _) = arena.nat_loop_binder();
        let three = arena.nat(3);
        let total = arena.nat_fold(FoldOp::Sum, binder, three, large);
        let recovered = arena.nat_div(total, large);
        assert_eq!(arena.eval_nat_u64(recovered, &Assignment::new()).unwrap(), 3);
        assert_eq!(
            arena
                .compile_nat(recovered)
                .evaluate_u64(&compiled::InvocationValues::new())
                .unwrap(),
            3
        );
        let signed = arena.int_from_nat(large);
        let zero = arena.int(0);
        let negative = arena.int_sub(zero, signed);
        let quotient = arena.int_div(negative, signed);
        assert_eq!(arena.eval_int_i64(quotient, &Assignment::new()).unwrap(), -1);
    }

    #[test]
    fn runtime_resolution_uses_semantic_identity_and_preserves_lexical_binders() {
        use crate::ids::{FunctionId, ProgramId, SemanticValueId};
        let mut arena = ExprArena::new();
        let function = FunctionId::new(ProgramId::fresh(), 0);
        let value = SemanticValueId::new(function, 0);
        let (_, first) = arena.runtime_value(value);
        let (_, second) = arena.runtime_value(value);
        let sum = arena.int_add(first, second);
        let (binder, symbol, coordinate) = arena.loop_binder();
        let body = arena.int_add(sum, coordinate);
        let body = arena.nat_from_int(body);
        let count = arena.nat(3);
        let fold = arena.nat_fold(FoldOp::Sum, binder, count, body);
        let expression = arena.int_from_nat(fold);
        let seven = arena.int(7);
        let resolved = arena.resolve_runtime_values(expression, &[(value, seven)]);
        assert!(arena.free_symbols(resolved.into()).is_empty());
        assert_eq!(arena.eval_int_i64(resolved, &Assignment::new()).unwrap(), 45);
        assert!(!arena.free_symbols(resolved.into()).contains(&symbol));
        assert!(arena.eval_int_i64(expression, &Assignment::new()).is_err());
    }

    #[test]
    fn constant_bounds_close_total_geometry_without_erasing_partial_expressions() {
        let mut arena = ExprArena::new();
        let (_, symbol) = arena.target_constant(SymbolSort::Nat);
        let x = arena.nat_symbol(symbol);
        let cap = arena.nat(32);
        let remainder = arena.nat_rem(x, cap);
        let condition = arena.nat_cmp(CmpOp::Gt, x, cap);
        let geometry = arena.nat_select(condition, cap, remainder);
        let bounded = arena.nat_cmp(CmpOp::Le, geometry, cap);
        assert!(matches!(
            arena.view(bounded.into()),
            NodeView::BoolConst(true)
        ));
        let partial = arena.nat_sub(x, cap);
        let geometry = arena.nat_min(partial, cap);
        let bounded = arena.nat_cmp(CmpOp::Le, geometry, cap);
        assert!(!matches!(
            arena.view(bounded.into()),
            NodeView::BoolConst(true)
        ));
        let mut values = Assignment::new();
        values.bind(symbol, SymbolValue::Nat((0u64).into()));
        assert!(arena.eval_bool(bounded, &values).is_err());
    }

    #[test]
    fn duration_sum_binds_and_sums_dynamic_demand_exactly() {
        let mut arena = ExprArena::default();
        let (binder, symbol, iteration) = arena.nat_loop_binder();
        let one = arena.nat(1);
        let demand = arena.nat_add(iteration, one);
        let body = arena.duration(&[DurationTerm {
            demand,
            lower_numerator: 2,
            upper_numerator: 4,
            denominator: 1,
        }]);
        let extent = arena.nat(3);
        let total = arena.duration_sum(binder, extent, body);

        assert!(!arena
            .free_symbols(AnyExpr::Duration(total))
            .contains(&symbol));
        let value = arena.eval_duration(total, &Assignment::new()).unwrap();
        assert_eq!(value.lower().numerator(), 12);
        assert_eq!(value.lower().denominator(), 1);
        assert_eq!(value.upper().numerator(), 24);
        assert_eq!(value.upper().denominator(), 1);
    }

    #[test]
    fn duration_sum_range_binds_the_original_nonzero_iteration_values() {
        let mut arena = ExprArena::default();
        let (binder, symbol, iteration) = arena.nat_loop_binder();
        let body = arena.duration(&[DurationTerm {
            demand: iteration,
            lower_numerator: 1,
            upper_numerator: 1,
            denominator: 1,
        }]);
        let start = arena.nat(4);
        let extent = arena.nat(3);
        let total = arena.duration_sum_range(binder, start, extent, body);

        assert!(!arena
            .free_symbols(AnyExpr::Duration(total))
            .contains(&symbol));
        let value = arena.eval_duration(total, &Assignment::new()).unwrap();
        assert_eq!(value.lower().numerator(), 15);
        assert_eq!(value.upper().numerator(), 15);
    }
    #[test]
    fn deep_predicates_evaluate_without_recursive_boolean_frames() {
        let mut arena = ExprArena::new();
        let (_, symbol) = arena.target_constant(SymbolSort::Nat);
        let n = arena.nat_symbol(symbol);
        let mut predicate = arena.bool(true);
        for value in 0..1024 {
            let bound = arena.nat(value);
            let term = arena.nat_cmp(CmpOp::Ge, n, bound);
            predicate = arena.and(predicate, term);
        }
        let zero = arena.nat(0);
        let invalid = arena.nat_div(n, zero);
        let invalid = arena.nat_cmp(CmpOp::Eq, invalid, zero);
        let guarded = arena.or(predicate, invalid);
        let mut assignment = Assignment::new();
        assignment.bind(symbol, SymbolValue::Nat((1024u64).into()));
        assert_eq!(arena.eval_bool(guarded, &assignment).unwrap(), true);
        assignment.bind(symbol, SymbolValue::Nat((0u64).into()));
        assert!(arena.eval_bool(guarded, &assignment).is_err());
    }

    #[test]
    fn minimum_envelopes_preserve_resource_bounds_and_definedness() {
        let mut arena = ExprArena::new();
        let (_, symbol) = arena.target_constant(SymbolSort::Nat);
        let groups = arena.nat_symbol(symbol);
        let cap = arena.nat(32);
        let width = arena.nat(8);
        let limit = arena.nat(4096);
        let original = arena.nat_mul(groups, width);
        let accepted = arena.nat_cmp(CmpOp::Le, original, limit);
        let envelope = arena.nat_min(groups, cap);
        let reserved = arena.nat_mul(envelope, width);
        let required = arena.nat_cmp(CmpOp::Le, reserved, limit);
        assert!(arena.entails(accepted, required));
        let zero = arena.nat(0);
        let count = arena.nat_ceil_div(groups, cap);
        let nonempty = arena.nat_cmp(CmpOp::Gt, count, zero);
        let conditional = arena.implies(nonempty, required);
        assert!(arena.entails(accepted, conditional));
        let empty_reservation = arena.nat_select(nonempty, reserved, zero);
        let conditional_bound = arena.nat_cmp(CmpOp::Le, empty_reservation, limit);
        assert!(arena.entails(accepted, conditional_bound));

        let invalid = arena.nat_div(groups, zero);
        let partial_condition = arena.nat_cmp(CmpOp::Gt, invalid, zero);
        let conditional = arena.implies(partial_condition, required);
        assert!(!arena.entails(accepted, conditional));
        let invalid_reservation = arena.nat_select(partial_condition, reserved, zero);
        let invalid_bound = arena.nat_cmp(CmpOp::Le, invalid_reservation, limit);
        assert!(!arena.entails(accepted, invalid_bound));

        let partial = arena.nat_sub(groups, cap);
        let envelope = arena.nat_min(groups, partial);
        let reserved = arena.nat_mul(envelope, width);
        let required = arena.nat_cmp(CmpOp::Le, reserved, limit);
        assert!(!arena.entails(accepted, required));
        let mut assignment = Assignment::new();
        assignment.bind(symbol, SymbolValue::Nat((0u64).into()));
        assert!(arena.eval_bool(accepted, &assignment).unwrap());
        assert!(arena.eval_bool(required, &assignment).is_err());
    }

    #[test]
    fn entailment_does_not_erase_partial_predicate_evaluation() {
        let mut arena = ExprArena::new();
        let (_, symbol) = arena.target_constant(SymbolSort::Int);
        let signed = arena.int_symbol(symbol);
        let n = arena.nat_from_int(signed);
        let limit = arena.nat(32);
        let accepted = arena.nat_cmp(CmpOp::Le, n, limit);
        assert!(arena.entails(accepted, accepted));
        let implication = arena.implies(accepted, accepted);
        let mut assignment = Assignment::new();
        assignment.bind(symbol, SymbolValue::Int((-1).into()));
        assert!(arena.eval_bool(implication, &assignment).is_err());
        let zero = arena.nat(0);
        let division = arena.nat_div(n, zero);
        let invalid = arena.nat_cmp(CmpOp::Le, division, limit);
        assert!(!arena.entails(accepted, invalid));
        assignment.bind(symbol, SymbolValue::Int((16).into()));
        assert!(arena.eval_bool(implication, &assignment).unwrap());
    }

    #[test]
    fn nonnegative_signed_shape_arithmetic_has_canonical_natural_form() {
        let mut arena = ExprArena::new();
        let (_, symbol) = arena.target_constant(SymbolSort::Nat);
        let n = arena.nat_symbol(symbol);
        let signed = arena.int_from_nat(n);
        let two = arena.int(2);
        let sum = arena.int_add(signed, two);
        let product = arena.int_mul(sum, signed);
        let actual = arena.nat_from_int(product);
        let two = arena.nat(2);
        let sum = arena.nat_add(n, two);
        let expected = arena.nat_mul(sum, n);
        assert_eq!(actual, expected);
        let minus_one = arena.int(-1);
        let possibly_negative = arena.int_add(signed, minus_one);
        let checked = arena.nat_from_int(possibly_negative);
        let mut assignment = Assignment::new();
        assignment.bind(symbol, SymbolValue::Nat((0u64).into()));
        assert!(arena.eval_nat_u64(checked, &assignment).is_err());
        assignment.bind(symbol, SymbolValue::Nat((2u64).into()));
        assert_eq!(arena.eval_nat_u64(checked, &assignment).unwrap(), 1);
    }

    #[test]
    fn product_bound_can_drop_only_proven_positive_factors() {
        let mut arena = ExprArena::new();
        let (_, ns) = arena.target_constant(SymbolSort::Nat);
        let (_, ws) = arena.target_constant(SymbolSort::Nat);
        let n = arena.nat_symbol(ns);
        let w = arena.nat_symbol(ws);
        let four = arena.nat(4);
        let limit = arena.nat(256);
        let total = arena.nat_product(&[n, w, four]);
        let row = arena.nat_mul(w, four);
        let available = arena.nat_cmp(CmpOp::Le, total, limit);
        let required = arena.nat_cmp(CmpOp::Le, row, limit);
        let unproved = arena.implies(available, required);
        assert!(!matches!(
            arena.view(unproved.into()),
            NodeView::BoolConst(true)
        ));
        let int_n = arena.int_from_nat(n);
        let one = arena.int(1);
        let nonzero = arena.int_cmp(CmpOp::Ge, int_n, one);
        let domain = arena.all(&[nonzero, available]);
        let proved = arena.implies(domain, required);
        assert!(matches!(
            arena.view(proved.into()),
            NodeView::BoolConst(true)
        ));
    }

    #[test]
    fn natural_bound_implication_composes_packet_rounding_and_storage_width() {
        let mut arena = ExprArena::new();
        let (_, xs) = arena.target_constant(SymbolSort::Nat);
        let (_, ys) = arena.target_constant(SymbolSort::Nat);
        let x = arena.nat_symbol(xs);
        let y = arena.nat_symbol(ys);
        let divisor = arena.nat(64);
        let two = arena.nat(2);
        let four = arena.nat(4);
        let limit = arena.nat(256);
        let packets = arena.nat_ceil_div(x, divisor);
        let width_bound = arena.nat_cmp(CmpOp::Le, x, limit);
        let packet_bound = arena.nat_cmp(CmpOp::Le, packets, limit);
        let proof = arena.implies(width_bound, packet_bound);
        assert!(matches!(
            arena.view(proof.into()),
            NodeView::BoolConst(true)
        ));
        let wide = arena.nat_product(&[y, x, four]);
        let rounded = arena.nat_product(&[two, packets, y]);
        let available = arena.nat_cmp(CmpOp::Le, wide, limit);
        let required = arena.nat_cmp(CmpOp::Le, rounded, limit);
        let proof = arena.implies(available, required);
        assert!(matches!(
            arena.view(proof.into()),
            NodeView::BoolConst(true)
        ));
        let reversed = arena.implies(required, available);
        assert!(!matches!(
            arena.view(reversed.into()),
            NodeView::BoolConst(true)
        ));
        let partial = arena.nat_sub(x, four);
        let partial_bound = arena.nat_cmp(CmpOp::Le, partial, limit);
        let partial_proof = arena.implies(width_bound, partial_bound);
        assert!(!matches!(
            arena.view(partial_proof.into()),
            NodeView::BoolConst(true)
        ));
    }
}
