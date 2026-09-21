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
    /// A semantic scalar parameter of the call schema (`Nat`, `Int`, or a
    /// scalar dtype).
    CallScalar(ParameterId),
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

    pub fn call_scalar(&mut self, parameter: ParameterId, sort: SymbolSort) -> SymbolId {
        self.inner.call_scalar(parameter, sort)
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
    pub fn int_add(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        self.inner.int_add(a, b)
    }
    pub fn int_sub(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        self.inner.int_sub(a, b)
    }
    pub fn int_mul(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        self.inner.int_mul(a, b)
    }
    /// Truncating division; records `b != 0`.
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

    /// Total evaluation under a complete assignment.
    pub fn eval_nat(&self, node: NatExpr, values: &Assignment) -> Result<u64, EvalError> {
        self.inner.eval_nat(node, values)
    }
    pub fn eval_int(&self, node: IntExpr, values: &Assignment) -> Result<i64, EvalError> {
        self.inner.eval_int(node, values)
    }
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
    pub fn compile_nat(&self, node: NatExpr) -> compiled::Compiled<u64> {
        self.inner.compile_nat(node)
    }
    pub fn compile_int(&self, node: IntExpr) -> compiled::Compiled<i64> {
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
    ) -> compiled::Compiled<u64> {
        self.inner.compile_nat_with(node, fixed)
    }
    pub fn compile_int_with(
        &self,
        node: IntExpr,
        fixed: &PartialAssignment,
    ) -> compiled::Compiled<i64> {
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
    /// Multiplicity attached to one numerical fact.
    NumericalMultiplicity {
        kernel: u32,
        fact: u32,
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
            .map(|(_, v)| *v)
    }
    pub fn iter(&self) -> impl Iterator<Item = (SymbolId, SymbolValue)> + '_ {
        self.values.iter().copied()
    }
}

/// Values for every symbol a node mentions. Missing symbols are an
/// [`EvalError::Unbound`].
pub type Assignment = PartialAssignment;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SymbolValue {
    Nat(u64),
    Int(i64),
    F32(f32),
    F16(u16),
    BF16(u16),
    Bool(bool),
    I32(i32),
    U32(u32),
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
mod tests {
    use super::*;

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
}
