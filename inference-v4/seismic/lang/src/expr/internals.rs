//! W3-owned internals of the expression arena. The public surface is in
//! `expr/mod.rs` and is frozen; everything here is replaceable by W3.
//!
//! Guarantees established here:
//! - structural hash-consing: equal nodes are one index;
//! - constructors fold constants and neutral elements, and never discard a
//!   sub-expression whose side conditions could fail (a discarded subtree is
//!   either total or never evaluated under the node's short-circuit/select
//!   semantics), so a folded node has exactly the side conditions of the
//!   node it replaces;
//! - `partial` rebuilds through the same constructors and is interned;
//! - evaluation uses exact mathematical integer arithmetic;
//!   `and`/`or`/`implies`/`all`/`any` short-circuit left to right and
//!   `select` evaluates only the taken branch, so a predicate that guards a
//!   partial operation (`b != 0 and a / b < n`) is total;
//! - compiled evaluators are self-contained programs over a compacted copy
//!   of the reachable sub-DAG (`Send + Sync`, no arena reference);
//! - fold binders are bound variables: they are excluded from
//!   `free_symbols`, are never substituted by `partial`, and are rebound on
//!   every iteration during evaluation;
//! - every evaluated duration is normalized to the arena-wide common
//!   denominator before interval addition, and duration-bound ordering uses
//!   exact rational comparison.
//!
//! Symbols are allocated fresh on every `symbol` call: the call schema mints
//! several symbols of one kind (a range parameter has a start and an end),
//! so kind does not identify a symbol. The decision symbol used by
//! `decision_value`/`decision_is`/`decision_in` is the one minted by
//! `declare_decision`.
//!
//! Panics in this file are all §13.3.2: a handle or id that is not one this
//! arena allocated (an out-of-range node index, symbol index, or an
//! undeclared decision), plus one §13.3.3 site in `scalar_bits`.

use super::{
    compiled::{Compiled, CompiledDecisionPredicate},
    AnyExpr, ArenaId, Assignment, BinaryOp, BoolExpr, CmpOp, DecisionId, DurationEstimate,
    DurationExpr, DurationTerm, ErasedScalarExpr, EvalError, Expr, ExprDigest, FiniteDomain,
    FoldOp, IntExpr, LoopBinderId, NaryOp, NatExpr, NodeView, PartialAssignment, RootId, RootName,
    ScalarExpr, ScalarSort, SymbolId, SymbolKind, SymbolSort, SymbolValue, TargetConstantId,
    UnaryOp,
};
use crate::reference_math::{self, ReferenceScalar, ScalarOp};
use crate::types::DType;
use num_bigint::{BigInt, BigUint};
use num_traits::{Euclid, One, Signed, Zero};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

fn handle<Sort>(owner: ArenaId, index: u32) -> Expr<Sort> {
    Expr {
        owner,
        index,
        sort: std::marker::PhantomData,
    }
}

fn raw(e: AnyExpr) -> (ArenaId, u32) {
    match e {
        AnyExpr::Nat(x) => (x.owner, x.index),
        AnyExpr::Int(x) => (x.owner, x.index),
        AnyExpr::Bool(x) => (x.owner, x.index),
        AnyExpr::Duration(x) => (x.owner, x.index),
        AnyExpr::Scalar(value) => (value.owner, value.index),
    }
}

fn ordinal(e: AnyExpr) -> u32 {
    raw(e).1
}

const OUT_OF_ARENA: &str = "ExprArena: expression handle outside its owning arena (§13.3.2)";
const SYMBOL_OUT_OF_ARENA: &str = "ExprArena: symbol id outside its owning arena (§13.3.2)";
const UNDECLARED_DECISION: &str =
    "ExprArena: decision id was never declared in its owning arena (§13.3.2)";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Sort {
    Nat,
    Int,
    Bool,
    Scalar(DType),
    Duration,
}

/// One interned node. Operands are sort-carrying handles so `view` is a
/// direct projection.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Node {
    NatConst(u64),
    IntConst(i64),
    BoolConst(bool),
    ScalarConst {
        dtype: DType,
        bits: u32,
    },
    ScalarInteger {
        operation: ScalarOp,
        operands: Box<[(DType, IntExpr)]>,
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
        operands: Box<[AnyExpr]>,
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
        values: Box<[i64]>,
    },
    Fold {
        op: FoldOp,
        binder: LoopBinderId,
        start: NatExpr,
        extent: NatExpr,
        body: NatExpr,
    },
    Duration(Box<[DurationTerm]>),
    DurationScale {
        duration: DurationExpr,
        by: NatExpr,
    },
}

#[derive(Clone, Copy, Debug)]
struct SymbolRecord {
    kind: SymbolKind,
    sort: SymbolSort,
}

#[derive(Debug)]
pub(super) struct Arena {
    id: ArenaId,
    symbols: Vec<SymbolRecord>,
    target_constants: Vec<SymbolId>,
    decision_count: u32,
    binder_symbols: HashMap<LoopBinderId, Vec<SymbolId>>,
    loop_binder_count: u32,
    decisions: HashMap<DecisionId, (FiniteDomain, SymbolId)>,
    nodes: Vec<Node>,
    sorts: Vec<Sort>,
    /// Every side condition beneath the node is trivially true (conservative).
    total: Vec<bool>,
    /// Free symbols beneath the node, sorted; fold binders are bound.
    free: Vec<Vec<SymbolId>>,
    interned: HashMap<Node, u32>,
    roots: Vec<(RootName, AnyExpr)>,
}

// ---------------------------------------------------------------------------
// construction
// ---------------------------------------------------------------------------

impl Arena {
    pub(super) fn new() -> Self {
        static NEXT_ARENA: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_ARENA
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .unwrap_or_else(|_| panic!("ExprArena identity space exhausted"));
        Self {
            id: ArenaId(
                NonZeroU64::new(id)
                    .unwrap_or_else(|| panic!("ExprArena identity allocator produced zero")),
            ),
            symbols: Vec::new(),
            target_constants: Vec::new(),
            decision_count: 0,
            binder_symbols: HashMap::new(),
            loop_binder_count: 0,
            decisions: HashMap::new(),
            nodes: Vec::new(),
            sorts: Vec::new(),
            total: Vec::new(),
            free: Vec::new(),
            interned: HashMap::new(),
            roots: Vec::new(),
        }
    }

    fn h<Sort>(&self, index: u32) -> Expr<Sort> {
        handle(self.id, index)
    }

    fn index(&self, expression: AnyExpr) -> u32 {
        let (owner, index) = raw(expression);
        if owner != self.id {
            panic!("{OUT_OF_ARENA}: expected {:?}, found {owner:?}", self.id);
        }
        index
    }

    fn expr_index<Sort>(&self, expression: Expr<Sort>) -> u32 {
        if expression.owner != self.id {
            panic!(
                "{OUT_OF_ARENA}: expected {:?}, found {:?}",
                self.id, expression.owner
            );
        }
        expression.index
    }

    fn symbol_index(&self, symbol: SymbolId) -> usize {
        if symbol.owner != self.id {
            panic!(
                "{SYMBOL_OUT_OF_ARENA}: expected {:?}, found {:?}",
                self.id, symbol.owner
            );
        }
        symbol.index as usize
    }

    fn decision_index(&self, decision: DecisionId) -> u32 {
        if decision.owner != self.id {
            panic!(
                "{UNDECLARED_DECISION}: expected {:?}, found {:?}",
                self.id, decision.owner
            );
        }
        decision.index
    }

    fn loop_binder_index(&self, binder: LoopBinderId) -> u32 {
        if binder.owner != self.id {
            panic!(
                "ExprArena: loop-binder id outside its owning arena (§13.3.2): expected {:?}, found {:?}",
                self.id, binder.owner
            );
        }
        if binder.index >= self.loop_binder_count {
            panic!("ExprArena: loop-binder id was never allocated in its owning arena (§13.3.2)");
        }
        binder.index
    }

    fn node(&self, i: u32) -> &Node {
        self.nodes.get(i as usize).expect(OUT_OF_ARENA)
    }
    fn sort(&self, i: u32) -> Sort {
        *self.sorts.get(i as usize).expect(OUT_OF_ARENA)
    }
    fn is_total(&self, i: u32) -> bool {
        *self.total.get(i as usize).expect(OUT_OF_ARENA)
    }
    fn expr_total<Sort>(&self, expression: Expr<Sort>) -> bool {
        self.is_total(self.expr_index(expression))
    }
    fn erased(&self, i: u32) -> AnyExpr {
        match self.sort(i) {
            Sort::Nat => AnyExpr::Nat(handle(self.id, i)),
            Sort::Int => AnyExpr::Int(handle(self.id, i)),
            Sort::Bool => AnyExpr::Bool(handle(self.id, i)),
            Sort::Duration => AnyExpr::Duration(handle(self.id, i)),
            Sort::Scalar(_) => AnyExpr::Scalar(ErasedScalarExpr {
                owner: self.id,
                index: i,
            }),
        }
    }
    fn record(&self, symbol: SymbolId) -> SymbolRecord {
        *self
            .symbols
            .get(self.symbol_index(symbol))
            .expect(SYMBOL_OUT_OF_ARENA)
    }

    fn intern(&mut self, node: Node, sort: Sort) -> u32 {
        if let Some(&i) = self.interned.get(&node) {
            return i;
        }
        let total = self.node_total(&node);
        let free = self.node_free(&node);
        let i = u32::try_from(self.nodes.len())
            .unwrap_or_else(|_| panic!("ExprArena node identity space exhausted"));
        self.nodes.push(node.clone());
        self.sorts.push(sort);
        self.total.push(total);
        self.free.push(free);
        self.interned.insert(node, i);
        i
    }

    fn children(node: &Node) -> Vec<AnyExpr> {
        match node {
            Node::NatConst(_)
            | Node::IntConst(_)
            | Node::BoolConst(_)
            | Node::ScalarConst { .. }
            | Node::Symbol(_) => Vec::new(),
            Node::Unary { operand, .. } => vec![*operand],
            Node::Binary { lhs, rhs, .. } | Node::Cmp { lhs, rhs, .. } => vec![*lhs, *rhs],
            Node::Nary { operands, .. } => operands.to_vec(),
            Node::ScalarInteger { operands, .. } => {
                operands.iter().map(|(_, value)| (*value).into()).collect()
            }
            Node::Select {
                cond,
                then,
                otherwise,
            } => vec![AnyExpr::Bool(*cond), *then, *otherwise],
            Node::In { operand, .. } => vec![*operand],
            Node::Fold {
                start,
                extent,
                body,
                ..
            } => vec![
                AnyExpr::Nat(*start),
                AnyExpr::Nat(*extent),
                AnyExpr::Nat(*body),
            ],
            Node::Duration(terms) => terms.iter().map(|term| AnyExpr::Nat(term.demand)).collect(),
            Node::DurationScale { duration, by } => {
                vec![AnyExpr::Duration(*duration), AnyExpr::Nat(*by)]
            }
        }
    }

    fn node_total(&self, node: &Node) -> bool {
        if let Node::Unary {
            op: UnaryOp::ScalarIntegerDefined,
            operand,
        } = node
        {
            // Definedness observes this recipe's failure, not a numeric
            // placeholder. Its operands must still themselves be defined.
            return Self::children(self.node(self.index(*operand)))
                .into_iter()
                .all(|child| self.is_total(self.index(child)));
        }
        let children_total = Self::children(node)
            .into_iter()
            .all(|c| self.is_total(self.index(c)));
        if !children_total {
            return false;
        }
        match node {
            Node::ScalarInteger {
                operation,
                operands,
            } => {
                let types = operands.iter().map(|(dtype, _)| *dtype).collect::<Vec<_>>();
                reference_math::scalar_recipe(*operation, &types)
                    .failures()
                    .is_empty()
            }
            Node::Unary {
                op: UnaryOp::NatFromInt,
                ..
            } => false,
            Node::Binary { op, lhs, rhs } => match op {
                BinaryOp::Sub => self.sort(self.index(*lhs)) == Sort::Int,
                BinaryOp::Div | BinaryOp::CeilDiv | BinaryOp::Rem | BinaryOp::AlignUp => {
                    self.nonzero_const(self.index(*rhs))
                }
                _ => true,
            },
            Node::Duration(terms) => terms
                .iter()
                .all(|term| term.denominator != 0 && term.lower_numerator <= term.upper_numerator),
            _ => true,
        }
    }

    fn nonzero_const(&self, i: u32) -> bool {
        match self.node(i) {
            Node::NatConst(v) => *v != 0,
            Node::IntConst(v) => *v != 0,
            _ => false,
        }
    }

    fn node_free(&self, node: &Node) -> Vec<SymbolId> {
        let mut free: Vec<SymbolId> = match node {
            Node::Symbol(s) => vec![*s],
            Node::Fold {
                binder,
                start,
                extent,
                body,
                ..
            } => {
                let mut free = self.free[self.expr_index(*start) as usize].clone();
                free.extend(self.free[self.expr_index(*extent) as usize].iter().copied());
                let mut body_free = self.free[self.expr_index(*body) as usize].clone();
                if let Some(bound) = self.binder_symbols.get(binder) {
                    body_free.retain(|symbol| !bound.contains(symbol));
                }
                free.extend(body_free);
                free
            }
            _ => Self::children(node)
                .into_iter()
                .flat_map(|c| self.free[self.index(c) as usize].iter().copied())
                .collect(),
        };
        free.sort_unstable();
        free.dedup();
        free
    }

    fn mentions_binder(&self, i: u32, binder: LoopBinderId) -> bool {
        match self.binder_symbols.get(&binder) {
            Some(bound) => self.free[i as usize].iter().any(|s| bound.contains(s)),
            None => false,
        }
    }

    // ----- symbols ---------------------------------------------------------

    fn symbol(&mut self, kind: SymbolKind, sort: SymbolSort) -> SymbolId {
        let id = SymbolId {
            owner: self.id,
            index: u32::try_from(self.symbols.len())
                .unwrap_or_else(|_| panic!("ExprArena symbol identity space exhausted")),
        };
        self.symbols.push(SymbolRecord { kind, sort });
        if let SymbolKind::LoopBinder(binder) = kind {
            self.binder_symbols.entry(binder).or_default().push(id);
        }
        id
    }
    pub(super) fn call_dimension(
        &mut self,
        dimension: crate::ids::DimensionId,
    ) -> (SymbolId, NatExpr) {
        let symbol = self.symbol(SymbolKind::CallDimension(dimension), SymbolSort::Nat);
        let expression = self.nat_symbol(symbol);
        (symbol, expression)
    }

    pub(super) fn template_dimension(&mut self, ordinal: u32) -> (SymbolId, IntExpr) {
        let symbol = self.symbol(SymbolKind::TemplateDimension(ordinal), SymbolSort::Int);
        let expression = self.int_symbol(symbol);
        (symbol, expression)
    }
    pub(super) fn runtime_value(
        &mut self,
        value: crate::ids::SemanticValueId,
    ) -> (SymbolId, IntExpr) {
        let symbol = self.symbol(SymbolKind::RuntimeValue(value), SymbolSort::Int);
        (symbol, self.int_symbol(symbol))
    }
    pub(super) fn call_scalar(
        &mut self,
        parameter: crate::ids::ParameterId,
        sort: SymbolSort,
    ) -> SymbolId {
        self.symbol(SymbolKind::CallScalar(parameter), sort)
    }
    pub(super) fn target_constant(&mut self, sort: SymbolSort) -> (TargetConstantId, SymbolId) {
        let index = u32::try_from(self.target_constants.len())
            .unwrap_or_else(|_| panic!("ExprArena target-constant identity space exhausted"));
        let id = TargetConstantId {
            owner: self.id,
            index,
        };
        let symbol = self.symbol(SymbolKind::TargetConstant(id), sort);
        self.target_constants.push(symbol);
        (id, symbol)
    }
    pub(super) fn target_constant_symbol(&self, constant: TargetConstantId) -> SymbolId {
        if constant.owner != self.id {
            panic!(
                "ExprArena: target-constant id outside its owning arena (§13.3.2): expected {:?}, found {:?}",
                self.id, constant.owner
            );
        }
        *self.target_constants.get(constant.index as usize)
            .unwrap_or_else(|| panic!("ExprArena: target-constant id was never allocated in its owning arena (§13.3.2)"))
    }
    pub(super) fn decision(&mut self, domain: FiniteDomain) -> DecisionId {
        let decision = DecisionId {
            owner: self.id,
            index: self.decision_count,
        };
        self.decision_count = self
            .decision_count
            .checked_add(1)
            .expect("ExprArena decision identity space exhausted");
        let symbol = self.symbol(SymbolKind::Decision(decision), SymbolSort::Int);
        self.decisions.insert(decision, (domain, symbol));
        decision
    }
    pub(super) fn loop_binder(&mut self) -> (LoopBinderId, SymbolId, IntExpr) {
        let binder = LoopBinderId {
            owner: self.id,
            index: self.loop_binder_count,
        };
        self.loop_binder_count = self
            .loop_binder_count
            .checked_add(1)
            .expect("ExprArena loop-binder identity space exhausted");
        let symbol = self.symbol(SymbolKind::LoopBinder(binder), SymbolSort::Int);
        let expression = self.int_symbol(symbol);
        (binder, symbol, expression)
    }

    pub(super) fn nat_loop_binder(&mut self) -> (LoopBinderId, SymbolId, NatExpr) {
        let binder = LoopBinderId {
            owner: self.id,
            index: self.loop_binder_count,
        };
        self.loop_binder_count = self
            .loop_binder_count
            .checked_add(1)
            .expect("ExprArena loop-binder identity space exhausted");
        let symbol = self.symbol(SymbolKind::LoopBinder(binder), SymbolSort::Nat);
        let expression = self.nat_symbol(symbol);
        (binder, symbol, expression)
    }
    pub(super) fn schedule_slot(&mut self, ordinal: u32, sort: SymbolSort) -> SymbolId {
        self.symbol(SymbolKind::ScheduleSlot(ordinal), sort)
    }
    pub(super) fn rebase_schedule_slot(&mut self, symbol: SymbolId, ordinal: u32) {
        let index = self.symbol_index(symbol);
        let record = self.symbols.get_mut(index).expect(SYMBOL_OUT_OF_ARENA);
        assert!(matches!(record.kind, SymbolKind::ScheduleSlot(_)), "only a schedule slot may be rebased");
        record.kind = SymbolKind::ScheduleSlot(ordinal);
    }
    pub(super) fn symbol_kind(&self, symbol: SymbolId) -> SymbolKind {
        self.record(symbol).kind
    }
    pub(super) fn symbol_sort(&self, symbol: SymbolId) -> SymbolSort {
        self.record(symbol).sort
    }
    pub(super) fn decision_domain(&self, decision: DecisionId) -> &FiniteDomain {
        self.decision_index(decision);
        &self.decisions.get(&decision).expect(UNDECLARED_DECISION).0
    }
    pub(super) fn decision_symbol(&self, decision: DecisionId) -> SymbolId {
        self.decision_index(decision);
        self.decisions.get(&decision).expect(UNDECLARED_DECISION).1
    }
    pub(super) fn symbols(&self) -> impl Iterator<Item = SymbolId> + '_ {
        (0..self.symbols.len()).map(|index| SymbolId {
            owner: self.id,
            index: u32::try_from(index)
                .unwrap_or_else(|_| panic!("ExprArena symbol identity space exhausted")),
        })
    }

    // ----- constant inspection ---------------------------------------------

    fn nat_of(&self, expression: NatExpr) -> Option<u64> {
        match self.node(self.expr_index(expression)) {
            Node::NatConst(v) => Some(*v),
            _ => None,
        }
    }
    fn int_of(&self, expression: IntExpr) -> Option<i64> {
        match self.node(self.expr_index(expression)) {
            Node::IntConst(v) => Some(*v),
            _ => None,
        }
    }
    fn bool_of(&self, expression: BoolExpr) -> Option<bool> {
        match self.node(self.expr_index(expression)) {
            Node::BoolConst(v) => Some(*v),
            _ => None,
        }
    }

    fn symbol_node(&mut self, symbol: SymbolId) -> u32 {
        let sort = match self.record(symbol).sort {
            SymbolSort::Nat => Sort::Nat,
            SymbolSort::Int => Sort::Int,
            SymbolSort::Scalar(dtype) => Sort::Scalar(dtype),
        };
        self.intern(Node::Symbol(symbol), sort)
    }

    // ----- Nat -------------------------------------------------------------

    pub(super) fn nat_exact(&mut self, value: BigUint) -> NatExpr {
        // Constant leaves retain their compact native encoding; arbitrary
        // constants use the same exact arithmetic DAG, not an auxiliary map.
        let mut result = self.nat_const(0);
        let radix = self.nat_const(1_u64 << 32);
        for digit in value.to_u32_digits().into_iter().rev() {
            let shifted = self.nat_mul(result, radix);
            let digit = self.nat_const(u64::from(digit));
            result = self.nat_add(shifted, digit);
        }
        result
    }
    pub(super) fn int_exact(&mut self, value: BigInt) -> IntExpr {
        let magnitude = self.nat_exact(value.magnitude().clone());
        let magnitude = self.int_from_nat(magnitude);
        if value.is_negative() {
            let zero = self.int_const(0);
            self.int_sub(zero, magnitude)
        } else { magnitude }
    }

    pub(super) fn nat_const(&mut self, v: u64) -> NatExpr {
        handle(self.id, self.intern(Node::NatConst(v), Sort::Nat))
    }
    pub(super) fn nat_symbol(&mut self, s: SymbolId) -> NatExpr {
        if self.record(s).sort != SymbolSort::Nat {
            panic!("ExprArena: {s:?} is not a Nat symbol");
        }
        handle(self.id, self.symbol_node(s))
    }
    fn nat_binary(&mut self, op: BinaryOp, a: NatExpr, b: NatExpr) -> NatExpr {
        handle(
            self.id,
            self.intern(
                Node::Binary {
                    op,
                    lhs: AnyExpr::Nat(a),
                    rhs: AnyExpr::Nat(b),
                },
                Sort::Nat,
            ),
        )
    }
    pub(super) fn nat_add(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        match (self.nat_of(a), self.nat_of(b)) {
            (Some(x), Some(y)) => {
                if let Some(v) = x.checked_add(y) {
                    return self.nat_const(v);
                }
            }
            (Some(0), _) => return b,
            (_, Some(0)) => return a,
            _ => (),
        }
        self.nat_binary(BinaryOp::Add, a, b)
    }
    pub(super) fn nat_mul(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        match (self.nat_of(a), self.nat_of(b)) {
            (Some(x), Some(y)) => {
                if let Some(v) = x.checked_mul(y) {
                    return self.nat_const(v);
                }
            }
            (Some(1), _) => return b,
            (_, Some(1)) => return a,
            (Some(0), _) if self.expr_total(b) => return a,
            (_, Some(0)) if self.expr_total(a) => return b,
            _ => (),
        }
        self.nat_binary(BinaryOp::Mul, a, b)
    }
    pub(super) fn nat_sub(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        match (self.nat_of(a), self.nat_of(b)) {
            (Some(x), Some(y)) if x >= y => return self.nat_const(x - y),
            (_, Some(0)) => return a,
            _ => (),
        }
        self.nat_binary(BinaryOp::Sub, a, b)
    }
    pub(super) fn nat_div(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        match (self.nat_of(a), self.nat_of(b)) {
            (Some(x), Some(y)) if y != 0 => return self.nat_const(x / y),
            (_, Some(1)) => return a,
            _ => (),
        }
        self.nat_binary(BinaryOp::Div, a, b)
    }
    pub(super) fn nat_ceil_div(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        match (self.nat_of(a), self.nat_of(b)) {
            (Some(x), Some(y)) if y != 0 => return self.nat_const(x.div_ceil(y)),
            (_, Some(1)) => return a,
            _ => (),
        }
        self.nat_binary(BinaryOp::CeilDiv, a, b)
    }
    pub(super) fn nat_rem(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        match (self.nat_of(a), self.nat_of(b)) {
            (Some(x), Some(y)) if y != 0 => return self.nat_const(x % y),
            (_, Some(1)) if self.expr_total(a) => return self.nat_const(0),
            _ => (),
        }
        self.nat_binary(BinaryOp::Rem, a, b)
    }
    pub(super) fn nat_min(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        self.expr_index(a);
        self.expr_index(b);
        if a == b {
            return a;
        }
        if let (Some(x), Some(y)) = (self.nat_of(a), self.nat_of(b)) {
            return self.nat_const(x.min(y));
        }
        self.nat_binary(BinaryOp::Min, a, b)
    }
    pub(super) fn nat_max(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        self.expr_index(a);
        self.expr_index(b);
        if self.nat_of(a) == Some(0) {
            return b;
        }
        if self.nat_of(b) == Some(0) {
            return a;
        }
        if a == b {
            return a;
        }
        if let (Some(x), Some(y)) = (self.nat_of(a), self.nat_of(b)) {
            return self.nat_const(x.max(y));
        }
        self.nat_binary(BinaryOp::Max, a, b)
    }
    pub(super) fn nat_align_up(&mut self, a: NatExpr, u: NatExpr) -> NatExpr {
        match (self.nat_of(a), self.nat_of(u)) {
            (Some(x), Some(y)) if y != 0 => {
                if let Some(v) = x.div_ceil(y).checked_mul(y) {
                    return self.nat_const(v);
                }
            }
            (_, Some(1)) => return a,
            _ => (),
        }
        self.nat_binary(BinaryOp::AlignUp, a, u)
    }
    pub(super) fn nat_select(&mut self, c: BoolExpr, t: NatExpr, e: NatExpr) -> NatExpr {
        self.expr_index(t);
        self.expr_index(e);
        match self.bool_of(c) {
            Some(true) => return t,
            Some(false) => return e,
            None => (),
        }
        if t == e && self.expr_total(c) {
            return t;
        }
        handle(
            self.id,
            self.intern(
                Node::Select {
                    cond: c,
                    then: AnyExpr::Nat(t),
                    otherwise: AnyExpr::Nat(e),
                },
                Sort::Nat,
            ),
        )
    }
    pub(super) fn nat_product(&mut self, factors: &[NatExpr]) -> NatExpr {
        let mut constant = 1_u64;
        let mut symbolic = Vec::new();
        let mut overflow = false;
        for &f in factors {
            match self.nat_of(f) {
                Some(v) => match constant.checked_mul(v) {
                    Some(c) => constant = c,
                    None => {
                        overflow = true;
                        symbolic.push(f);
                    }
                },
                None => symbolic.push(f),
            }
        }
        if !overflow {
            if constant == 0 && symbolic.iter().all(|f| self.expr_total(*f)) {
                return self.nat_const(0);
            }
            if symbolic.is_empty() {
                return self.nat_const(constant);
            }
            if constant == 1 && symbolic.len() == 1 {
                return symbolic[0];
            }
            if constant != 1 {
                let c = self.nat_const(constant);
                symbolic.insert(0, c);
            }
        } else {
            let c = self.nat_const(constant);
            symbolic.insert(0, c);
        }
        let operands: Box<[AnyExpr]> = symbolic.iter().map(|f| AnyExpr::Nat(*f)).collect();
        handle(
            self.id,
            self.intern(
                Node::Nary {
                    op: NaryOp::Product,
                    operands,
                },
                Sort::Nat,
            ),
        )
    }
    pub(super) fn nat_fold(
        &mut self,
        op: FoldOp,
        binder: LoopBinderId,
        extent: NatExpr,
        body: NatExpr,
    ) -> NatExpr {
        let start = self.nat_const(0);
        self.nat_fold_range(op, binder, start, extent, body)
    }
    pub(super) fn nat_fold_range(
        &mut self,
        op: FoldOp,
        binder: LoopBinderId,
        start: NatExpr,
        extent: NatExpr,
        body: NatExpr,
    ) -> NatExpr {
        self.loop_binder_index(binder);
        self.expr_index(start);
        let body_index = self.expr_index(body);
        if self.nat_of(extent) == Some(0) {
            return self.nat_const(match op {
                FoldOp::Sum | FoldOp::Max => 0,
                FoldOp::Product => 1,
            });
        }
        if !self.mentions_binder(body_index, binder) {
            match op {
                FoldOp::Sum if self.expr_total(body) => return self.nat_mul(extent, body),
                FoldOp::Max => {
                    let zero = self.nat_const(0);
                    let positive = self.nat_cmp(CmpOp::Gt, extent, zero);
                    return self.nat_select(positive, body, zero);
                }
                _ => (),
            }
        }
        handle(
            self.id,
            self.intern(
                Node::Fold {
                    op,
                    binder,
                    start,
                    extent,
                    body,
                },
                Sort::Nat,
            ),
        )
    }
    // Preserve natural shape arithmetic through the signed source-expression
    // layer. Only operations on independently nonnegative operands qualify;
    // arbitrary signed expressions retain their checked conversion.
    fn natural_integer(&mut self, i: IntExpr) -> Option<NatExpr> {
        match self.node(self.expr_index(i)).clone() {
            Node::IntConst(value) => u64::try_from(value).ok().map(|n| self.nat_const(n)),
            Node::Unary {
                op: UnaryOp::IntFromNat,
                operand: AnyExpr::Nat(n),
            } => Some(n),
            Node::Binary {
                op: op @ (BinaryOp::Add | BinaryOp::Mul),
                lhs: AnyExpr::Int(a),
                rhs: AnyExpr::Int(b),
            } => {
                let a = self.natural_integer(a)?;
                let b = self.natural_integer(b)?;
                Some(match op {
                    BinaryOp::Add => self.nat_add(a, b),
                    BinaryOp::Mul => self.nat_mul(a, b),
                    _ => unreachable!(),
                })
            }
            _ => None,
        }
    }

    pub(super) fn nat_from_int(&mut self, i: IntExpr) -> NatExpr {
        if let Some(natural) = self.natural_integer(i) {
            return natural;
        }
        if let Some(v) = self.int_of(i) {
            if let Ok(n) = u64::try_from(v) {
                return self.nat_const(n);
            }
        }
        if let Node::Unary {
            op: UnaryOp::IntFromNat,
            operand: AnyExpr::Nat(n),
        } = *self.node(self.expr_index(i))
        {
            return n;
        }
        handle(
            self.id,
            self.intern(
                Node::Unary {
                    op: UnaryOp::NatFromInt,
                    operand: AnyExpr::Int(i),
                },
                Sort::Nat,
            ),
        )
    }

    // ----- Int -------------------------------------------------------------

    pub(super) fn int_const(&mut self, v: i64) -> IntExpr {
        handle(self.id, self.intern(Node::IntConst(v), Sort::Int))
    }
    pub(super) fn int_symbol(&mut self, s: SymbolId) -> IntExpr {
        if self.record(s).sort != SymbolSort::Int {
            panic!("ExprArena: {s:?} is not an Int symbol");
        }
        handle(self.id, self.symbol_node(s))
    }
    pub(super) fn int_from_nat(&mut self, n: NatExpr) -> IntExpr {
        if let Some(v) = self.nat_of(n) {
            if let Ok(i) = i64::try_from(v) {
                return self.int_const(i);
            }
        }
        handle(
            self.id,
            self.intern(
                Node::Unary {
                    op: UnaryOp::IntFromNat,
                    operand: AnyExpr::Nat(n),
                },
                Sort::Int,
            ),
        )
    }
    fn int_binary(&mut self, op: BinaryOp, a: IntExpr, b: IntExpr) -> IntExpr {
        handle(
            self.id,
            self.intern(
                Node::Binary {
                    op,
                    lhs: AnyExpr::Int(a),
                    rhs: AnyExpr::Int(b),
                },
                Sort::Int,
            ),
        )
    }
    pub(super) fn int_from_scalar(&mut self, value: ErasedScalarExpr) -> IntExpr {
        let ordinal = self.index(AnyExpr::Scalar(value));
        assert!(
            matches!(self.sort(ordinal), Sort::Scalar(DType::I32 | DType::U32)),
            "mathematical integer injection requires an integer word"
        );
        if let Node::ScalarConst { dtype, bits } = *self.node(ordinal) {
            let integer = match dtype {
                DType::I32 => i64::from(bits as i32),
                DType::U32 => i64::from(bits),
                _ => unreachable!(),
            };
            return self.int_const(integer);
        }
        handle(
            self.id,
            self.intern(
                Node::Unary {
                    op: UnaryOp::IntFromScalar,
                    operand: AnyExpr::Scalar(value),
                },
                Sort::Int,
            ),
        )
    }
    pub(super) fn scalar_integer(
        &mut self,
        operation: ScalarOp,
        operands: &[(DType, IntExpr)],
    ) -> IntExpr {
        let types = operands
            .iter()
            .map(|(dtype, value)| {
                self.expr_index(*value);
                assert!(dtype.is_int(), "scalar integer operand must be I32/U32");
                *dtype
            })
            .collect::<Vec<_>>();
        let recipe = reference_math::scalar_recipe(operation, &types);
        assert!(
            recipe.output().ty().is_int(),
            "scalar integer result must be I32/U32"
        );
        let constants = operands
            .iter()
            .map(|(dtype, value)| {
                self.int_of(*value)
                    .map(|integer| integer_word(*dtype, &integer.into()))
            })
            .collect::<Option<Vec<_>>>();
        if let Some(inputs) = constants {
            if let Ok(value) = reference_math::evaluate(&recipe, &inputs) {
                return self.int_const(scalar_integer_value(value));
            }
        }
        handle(
            self.id,
            self.intern(
                Node::ScalarInteger {
                    operation,
                    operands: operands.into(),
                },
                Sort::Int,
            ),
        )
    }
    pub(super) fn scalar_integer_defined(&mut self, value: IntExpr) -> BoolExpr {
        let index = self.expr_index(value);
        if self.is_total(index) {
            return self.bool_const(true);
        }
        assert!(
            matches!(self.node(index), Node::ScalarInteger { .. }),
            "scalar definedness must refer to its actual recipe node"
        );
        handle(
            self.id,
            self.intern(
                Node::Unary {
                    op: UnaryOp::ScalarIntegerDefined,
                    operand: value.into(),
                },
                Sort::Bool,
            ),
        )
    }
    pub(super) fn int_add(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        match (self.int_of(a), self.int_of(b)) {
            (Some(x), Some(y)) => {
                if let Some(v) = x.checked_add(y) {
                    return self.int_const(v);
                }
            }
            (Some(0), _) => return b,
            (_, Some(0)) => return a,
            _ => (),
        }
        self.int_binary(BinaryOp::Add, a, b)
    }
    pub(super) fn int_sub(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        match (self.int_of(a), self.int_of(b)) {
            (Some(x), Some(y)) => {
                if let Some(v) = x.checked_sub(y) {
                    return self.int_const(v);
                }
            }
            (_, Some(0)) => return a,
            _ => (),
        }
        self.int_binary(BinaryOp::Sub, a, b)
    }
    pub(super) fn int_mul(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        match (self.int_of(a), self.int_of(b)) {
            (Some(x), Some(y)) => {
                if let Some(v) = x.checked_mul(y) {
                    return self.int_const(v);
                }
            }
            (Some(1), _) => return b,
            (_, Some(1)) => return a,
            (Some(0), _) if self.expr_total(b) => return a,
            (_, Some(0)) if self.expr_total(a) => return b,
            _ => (),
        }
        self.int_binary(BinaryOp::Mul, a, b)
    }
    pub(super) fn int_div(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        match (self.int_of(a), self.int_of(b)) {
            (Some(x), Some(y)) if y != 0 => {
                if let Some(v) = x.checked_div_euclid(y) {
                    return self.int_const(v);
                }
            }
            (_, Some(1)) => return a,
            _ => (),
        }
        self.int_binary(BinaryOp::Div, a, b)
    }
    pub(super) fn int_rem(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        match (self.int_of(a), self.int_of(b)) {
            (Some(x), Some(y)) if y != 0 => {
                if let Some(v) = x.checked_rem_euclid(y) {
                    return self.int_const(v);
                }
            }
            (_, Some(1)) if self.expr_total(a) => return self.int_const(0),
            _ => (),
        }
        self.int_binary(BinaryOp::Rem, a, b)
    }
    pub(super) fn int_min(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        self.expr_index(a);
        self.expr_index(b);
        if a == b {
            return a;
        }
        if let (Some(x), Some(y)) = (self.int_of(a), self.int_of(b)) {
            return self.int_const(x.min(y));
        }
        self.int_binary(BinaryOp::Min, a, b)
    }
    pub(super) fn int_max(&mut self, a: IntExpr, b: IntExpr) -> IntExpr {
        self.expr_index(a);
        self.expr_index(b);
        if a == b {
            return a;
        }
        if let (Some(x), Some(y)) = (self.int_of(a), self.int_of(b)) {
            return self.int_const(x.max(y));
        }
        self.int_binary(BinaryOp::Max, a, b)
    }
    pub(super) fn int_select(&mut self, c: BoolExpr, t: IntExpr, e: IntExpr) -> IntExpr {
        self.expr_index(t);
        self.expr_index(e);
        match self.bool_of(c) {
            Some(true) => return t,
            Some(false) => return e,
            None => (),
        }
        if t == e && self.expr_total(c) {
            return t;
        }
        handle(
            self.id,
            self.intern(
                Node::Select {
                    cond: c,
                    then: AnyExpr::Int(t),
                    otherwise: AnyExpr::Int(e),
                },
                Sort::Int,
            ),
        )
    }

    // ----- Bool ------------------------------------------------------------

    pub(super) fn bool_const(&mut self, v: bool) -> BoolExpr {
        handle(self.id, self.intern(Node::BoolConst(v), Sort::Bool))
    }
    fn bool_binary(&mut self, op: BinaryOp, a: BoolExpr, b: BoolExpr) -> BoolExpr {
        handle(
            self.id,
            self.intern(
                Node::Binary {
                    op,
                    lhs: AnyExpr::Bool(a),
                    rhs: AnyExpr::Bool(b),
                },
                Sort::Bool,
            ),
        )
    }
    /// `a and b`, short-circuit: `b` is evaluated only when `a` holds.
    pub(super) fn and(&mut self, a: BoolExpr, b: BoolExpr) -> BoolExpr {
        match (self.bool_of(a), self.bool_of(b)) {
            (Some(true), _) => return b,
            (Some(false), _) => return a,
            (_, Some(true)) => return a,
            (_, Some(false)) if self.expr_total(a) => return b,
            _ => (),
        }
        if a == b {
            return a;
        }
        self.bool_binary(BinaryOp::And, a, b)
    }
    /// `a or b`, short-circuit: `b` is evaluated only when `a` fails.
    pub(super) fn or(&mut self, a: BoolExpr, b: BoolExpr) -> BoolExpr {
        match (self.bool_of(a), self.bool_of(b)) {
            (Some(true), _) => return a,
            (Some(false), _) => return b,
            (_, Some(false)) => return a,
            (_, Some(true)) if self.expr_total(a) => return b,
            _ => (),
        }
        if a == b {
            return a;
        }
        self.bool_binary(BinaryOp::Or, a, b)
    }
    pub(super) fn not(&mut self, a: BoolExpr) -> BoolExpr {
        if let Some(v) = self.bool_of(a) {
            return self.bool_const(!v);
        }
        if let Node::Unary {
            op: UnaryOp::Not,
            operand: AnyExpr::Bool(inner),
        } = *self.node(self.expr_index(a))
        {
            return inner;
        }
        handle(
            self.id,
            self.intern(
                Node::Unary {
                    op: UnaryOp::Not,
                    operand: AnyExpr::Bool(a),
                },
                Sort::Bool,
            ),
        )
    }
    /// `a implies b`: `b` is evaluated only when `a` holds.
    pub(super) fn implies(&mut self, a: BoolExpr, b: BoolExpr) -> BoolExpr {
        match (self.bool_of(a), self.bool_of(b)) {
            (Some(false), _) => return self.bool_const(true),
            (Some(true), _) => return b,
            (_, Some(false)) => return self.not(a),
            (_, Some(true)) if self.expr_total(a) => return b,
            _ => (),
        }
        if a == b && self.expr_total(a) {
            return self.bool_const(true);
        }
        // Conjunction construction is intentionally short-circuiting and
        // therefore preserves insertion order. Universal closure still needs
        // the order-independent propositional fact that a total conjunction
        // entails any conjunction made solely from its exact conjuncts.
        // Restricting this fold to a total antecedent preserves partial
        // expression semantics while avoiding a general theorem prover.
        if self.expr_total(a) && self.conjunction_contains(a, b) {
            return self.bool_const(true);
        }
        self.bool_binary(BinaryOp::Implies, a, b)
    }

    pub(super) fn entails(&self, antecedent: BoolExpr, consequent: BoolExpr) -> bool {
        self.bool_of(antecedent) == Some(false)
            || self.bool_of(consequent) == Some(true)
            || self.conjunction_contains(antecedent, consequent)
    }

    fn conjunction_contains(&self, antecedent: BoolExpr, consequent: BoolExpr) -> bool {
        fn collect(arena: &Arena, expression: BoolExpr, out: &mut Vec<BoolExpr>) {
            match *arena.node(arena.expr_index(expression)) {
                Node::Binary {
                    op: BinaryOp::And,
                    lhs: AnyExpr::Bool(lhs),
                    rhs: AnyExpr::Bool(rhs),
                } => {
                    collect(arena, lhs, out);
                    collect(arena, rhs, out);
                }
                _ => out.push(expression),
            }
        }

        let mut available = Vec::new();
        let mut required = Vec::new();
        collect(self, antecedent, &mut available);
        collect(self, consequent, &mut required);
        required.into_iter().all(|term| {
            available.contains(&term)
                || available
                    .iter()
                    .any(|candidate| self.nat_upper_bound_dominates(*candidate, term, &available))
                || match self.node(self.expr_index(term)) {
                    Node::Binary {
                        op: BinaryOp::Implies,
                        lhs: AnyExpr::Bool(condition),
                        rhs: AnyExpr::Bool(body),
                    } if self.expr_total(*condition) => self.entails(antecedent, *body),
                    _ => false,
                }
        })
    }

    /// Exact monotone ordering of total natural expressions. This is a
    /// structural proof, never a sampled bound or a target-specific assumption.
    fn nat_upper_bound_dominates(
        &self,
        available: BoolExpr,
        required: BoolExpr,
        assumptions: &[BoolExpr],
    ) -> bool {
        let (
            Node::Cmp {
                op: CmpOp::Le,
                lhs: AnyExpr::Nat(available_lhs),
                rhs: AnyExpr::Nat(available_rhs),
            },
            Node::Cmp {
                op: CmpOp::Le,
                lhs: AnyExpr::Nat(required_lhs),
                rhs: AnyExpr::Nat(required_rhs),
            },
        ) = (
            self.node(self.expr_index(available)),
            self.node(self.expr_index(required)),
        )
        else {
            return false;
        };
        if available_rhs != required_rhs {
            return false;
        }

        self.nat_no_larger(*required_lhs, *available_lhs, assumptions)
    }

    fn nat_no_larger(&self, smaller: NatExpr, larger: NatExpr, assumptions: &[BoolExpr]) -> bool {
        if smaller == larger {
            return true;
        }
        if !self.expr_total(smaller) || !self.expr_total(larger) {
            return false;
        }
        if let (Some(a), Some(b)) = (self.nat_of(smaller), self.nat_of(larger)) {
            return a <= b;
        }
        if self.nat_of(smaller) == Some(0) {
            return true;
        }
        if let Node::Select {
            then: AnyExpr::Nat(then),
            otherwise: AnyExpr::Nat(otherwise),
            ..
        } = self.node(self.expr_index(smaller))
        {
            // Totality above includes the condition and both arms. Either arm
            // is therefore bounded without discarding a partial evaluation.
            return self.nat_no_larger(*then, larger, assumptions)
                && self.nat_no_larger(*otherwise, larger, assumptions);
        }
        if let Node::Binary {
            op: BinaryOp::Min,
            lhs: AnyExpr::Nat(lhs),
            rhs: AnyExpr::Nat(rhs),
        } = self.node(self.expr_index(smaller))
        {
            // Both operands are total (checked above), so either operand is
            // a sound upper bound without suppressing a partial evaluation.
            if self.nat_no_larger(*lhs, larger, assumptions)
                || self.nat_no_larger(*rhs, larger, assumptions)
            {
                return true;
            }
        }
        if let Node::Binary {
            op: BinaryOp::Div | BinaryOp::CeilDiv,
            lhs: AnyExpr::Nat(numerator),
            rhs: AnyExpr::Nat(divisor),
        } = self.node(self.expr_index(smaller))
        {
            if self.nat_of(*divisor).is_some_and(|divisor| divisor >= 1)
                && self.nat_no_larger(*numerator, larger, assumptions)
            {
                return true;
            }
        }
        fn product(
            arena: &Arena,
            expression: NatExpr,
            coefficient: &mut u64,
            factors: &mut Vec<NatExpr>,
        ) -> bool {
            match arena.node(arena.expr_index(expression)) {
                Node::NatConst(value) => {
                    let Some(next) = coefficient.checked_mul(*value) else {
                        return false;
                    };
                    *coefficient = next;
                    true
                }
                Node::Binary {
                    op: BinaryOp::Mul,
                    lhs: AnyExpr::Nat(a),
                    rhs: AnyExpr::Nat(b),
                } => {
                    product(arena, *a, coefficient, factors)
                        && product(arena, *b, coefficient, factors)
                }
                Node::Nary {
                    op: NaryOp::Product,
                    operands,
                } => operands.iter().all(|operand| {
                    let AnyExpr::Nat(value) = operand else {
                        return false;
                    };
                    product(arena, *value, coefficient, factors)
                }),
                _ => {
                    factors.push(expression);
                    true
                }
            }
        }
        let mut ac = 1;
        let mut bc = 1;
        let mut af = Vec::new();
        let mut bf = Vec::new();
        if !product(self, smaller, &mut ac, &mut af)
            || !product(self, larger, &mut bc, &mut bf)
            || ac > bc
            || af.len() > bf.len()
        {
            return false;
        }
        // Atom pairs without any decomposition cannot yield a new proof.
        if af.as_slice() == [smaller] && bf.as_slice() == [larger] {
            return false;
        }
        for factor in af {
            let found = bf.iter().position(|other| factor == *other).or_else(|| {
                bf.iter()
                    .position(|other| self.nat_no_larger(factor, *other, assumptions))
            });
            let Some(index) = found else {
                return false;
            };
            bf.swap_remove(index);
        }
        bf.into_iter()
            .all(|factor| self.nat_positive_under(factor, assumptions))
    }

    fn nat_positive_under(&self, value: NatExpr, assumptions: &[BoolExpr]) -> bool {
        if self.nat_of(value).is_some_and(|n| n > 0) {
            return true;
        }
        assumptions.iter().any(|assumption| {
            let Node::Cmp { op, lhs, rhs } = self.node(self.expr_index(*assumption)) else {
                return false;
            };
            let natural = match lhs {
                AnyExpr::Nat(n) => Some(*n),
                AnyExpr::Int(i) => match self.node(self.expr_index(*i)) {
                    Node::Unary {
                        op: UnaryOp::IntFromNat,
                        operand: AnyExpr::Nat(n),
                    } => Some(*n),
                    _ => None,
                },
                _ => None,
            };
            let bound = match rhs {
                AnyExpr::Nat(n) => self.nat_of(*n).map(i128::from),
                AnyExpr::Int(i) => self.int_of(*i).map(i128::from),
                _ => None,
            };
            natural == Some(value)
                && match (op, bound) {
                    (CmpOp::Ge, Some(n)) => n >= 1,
                    (CmpOp::Gt, Some(n)) => n >= 0,
                    _ => false,
                }
        })
    }
    pub(super) fn iff(&mut self, a: BoolExpr, b: BoolExpr) -> BoolExpr {
        match (self.bool_of(a), self.bool_of(b)) {
            (Some(x), Some(y)) => return self.bool_const(x == y),
            (Some(true), _) => return b,
            (_, Some(true)) => return a,
            (Some(false), _) => return self.not(b),
            (_, Some(false)) => return self.not(a),
            _ => (),
        }
        if a == b && self.expr_total(a) {
            return self.bool_const(true);
        }
        self.bool_binary(BinaryOp::Iff, a, b)
    }
    pub(super) fn all(&mut self, terms: &[BoolExpr]) -> BoolExpr {
        let mut acc = self.bool_const(true);
        for &t in terms {
            acc = self.and(acc, t);
        }
        acc
    }
    pub(super) fn any(&mut self, terms: &[BoolExpr]) -> BoolExpr {
        let mut acc = self.bool_const(false);
        for &t in terms {
            acc = self.or(acc, t);
        }
        acc
    }
    fn cmp_node(&mut self, op: CmpOp, lhs: AnyExpr, rhs: AnyExpr) -> BoolExpr {
        handle(self.id, self.intern(Node::Cmp { op, lhs, rhs }, Sort::Bool))
    }
    pub(super) fn nat_cmp(&mut self, op: CmpOp, a: NatExpr, b: NatExpr) -> BoolExpr {
        if let (Some(x), Some(y)) = (self.nat_of(a), self.nat_of(b)) {
            return self.bool_const(compare(op, x, y));
        }
        if a == b && self.expr_total(a) {
            return self.bool_const(reflexive(op));
        }
        if self.expr_total(a) {
            if let (Some(upper), Some(rhs)) = (self.nat_constant_upper(a), self.nat_of(b)) {
                match op {
                    CmpOp::Le if upper <= rhs => return self.bool_const(true),
                    CmpOp::Lt if upper < rhs => return self.bool_const(true),
                    CmpOp::Gt if upper <= rhs => return self.bool_const(false),
                    CmpOp::Ge if upper < rhs => return self.bool_const(false),
                    _ => {}
                }
            }
        }
        self.cmp_node(op, AnyExpr::Nat(a), AnyExpr::Nat(b))
    }
    /// A conservative constant upper bound from total expression structure.
    /// No invocation assumption, sampled value or backend fact enters it.
    fn nat_constant_upper(&self, expression: NatExpr) -> Option<u64> {
        match self.node(self.expr_index(expression)) {
            Node::NatConst(value) => Some(*value),
            Node::Binary {
                op: BinaryOp::Rem,
                rhs: AnyExpr::Nat(divisor),
                ..
            } => self.nat_of(*divisor)?.checked_sub(1),
            Node::Binary {
                op: BinaryOp::Min,
                lhs: AnyExpr::Nat(a),
                rhs: AnyExpr::Nat(b),
            } => match (self.nat_constant_upper(*a), self.nat_constant_upper(*b)) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
            Node::Binary {
                op: BinaryOp::Max,
                lhs: AnyExpr::Nat(a),
                rhs: AnyExpr::Nat(b),
            } => Some(
                self.nat_constant_upper(*a)?
                    .max(self.nat_constant_upper(*b)?),
            ),
            Node::Select {
                then: AnyExpr::Nat(a),
                otherwise: AnyExpr::Nat(b),
                ..
            } => Some(
                self.nat_constant_upper(*a)?
                    .max(self.nat_constant_upper(*b)?),
            ),
            _ => None,
        }
    }

    pub(super) fn int_cmp(&mut self, op: CmpOp, a: IntExpr, b: IntExpr) -> BoolExpr {
        if let (Some(x), Some(y)) = (self.int_of(a), self.int_of(b)) {
            return self.bool_const(compare(op, x, y));
        }
        if a == b && self.expr_total(a) {
            return self.bool_const(reflexive(op));
        }
        self.cmp_node(op, AnyExpr::Int(a), AnyExpr::Int(b))
    }
    fn in_node(&mut self, operand: AnyExpr, values: &[i64]) -> BoolExpr {
        let mut values = values.to_vec();
        values.sort_unstable();
        values.dedup();
        // A constant operand folds; a `Nat` constant above `i64::MAX` is
        // outside every representable member set.
        let constant: Option<Option<i64>> = match operand {
            AnyExpr::Nat(n) => self.nat_of(n).map(|v| i64::try_from(v).ok()),
            AnyExpr::Int(i) => self.int_of(i).map(Some),
            _ => None,
        };
        if let Some(c) = constant {
            let member = c.is_some_and(|c| values.binary_search(&c).is_ok());
            return self.bool_const(member);
        }
        if values.is_empty() && self.is_total(self.index(operand)) {
            return self.bool_const(false);
        }
        handle(
            self.id,
            self.intern(
                Node::In {
                    operand,
                    values: values.into_boxed_slice(),
                },
                Sort::Bool,
            ),
        )
    }
    pub(super) fn nat_in(&mut self, a: NatExpr, values: &[u64]) -> BoolExpr {
        if values.iter().any(|&v| i64::try_from(v).is_err()) {
            // `NodeView::In` carries `i64` members, so a set with a member
            // above `i64::MAX` is expressed as the equivalent disjunction of
            // equalities against `NatConst` members instead.
            let terms: Vec<BoolExpr> = values
                .iter()
                .map(|&v| {
                    let c = self.nat_const(v);
                    self.nat_cmp(CmpOp::Eq, a, c)
                })
                .collect();
            return self.any(&terms);
        }
        let values: Vec<i64> = values.iter().map(|&v| v as i64).collect();
        self.in_node(AnyExpr::Nat(a), &values)
    }
    pub(super) fn decision_in(&mut self, d: DecisionId, values: &[i64]) -> BoolExpr {
        let operand = self.decision_value(d);
        self.in_node(AnyExpr::Int(operand), values)
    }
    pub(super) fn decision_is(&mut self, d: DecisionId, v: i64) -> BoolExpr {
        let operand = self.decision_value(d);
        let value = self.int_const(v);
        self.int_cmp(CmpOp::Eq, operand, value)
    }
    pub(super) fn decision_value(&mut self, d: DecisionId) -> IntExpr {
        let symbol = self.decision_symbol(d);
        self.int_symbol(symbol)
    }

    // ----- Scalar ----------------------------------------------------------

    pub(super) fn scalar_symbol<T: ScalarSort>(&mut self, s: SymbolId) -> ScalarExpr<T> {
        if self.record(s).sort != SymbolSort::Scalar(T::DTYPE) {
            panic!("ExprArena: {s:?} is not a {:?} scalar symbol", T::DTYPE);
        }
        handle(self.id, self.symbol_node(s))
    }
    pub(super) fn scalar_const<T: ScalarSort>(&mut self, v: T::Value) -> ScalarExpr<T> {
        handle(
            self.id,
            self.intern(
                Node::ScalarConst {
                    dtype: T::DTYPE,
                    bits: scalar_bits::<T>(v),
                },
                Sort::Scalar(T::DTYPE),
            ),
        )
    }
    pub(super) fn scalar_cmp<T: ScalarSort>(
        &mut self,
        op: CmpOp,
        a: ScalarExpr<T>,
        b: ScalarExpr<T>,
    ) -> BoolExpr {
        let a = self.expr_index(a);
        let b = self.expr_index(b);
        self.scalar_cmp_raw(op, a, b)
    }
    fn scalar_cmp_raw(&mut self, op: CmpOp, a: u32, b: u32) -> BoolExpr {
        if let (
            Node::ScalarConst {
                dtype: da,
                bits: ba,
            },
            Node::ScalarConst {
                dtype: db,
                bits: bb,
            },
        ) = (self.node(a).clone(), self.node(b).clone())
        {
            if da == db {
                let x = ScalarVal::decode(da, ba);
                let y = ScalarVal::decode(db, bb);
                return self.bool_const(x.compare(op, y));
            }
        }
        self.cmp_node(
            op,
            AnyExpr::Scalar(ErasedScalarExpr {
                owner: self.id,
                index: a,
            }),
            AnyExpr::Scalar(ErasedScalarExpr {
                owner: self.id,
                index: b,
            }),
        )
    }

    // ----- Duration --------------------------------------------------------

    pub(super) fn duration(&mut self, terms: &[DurationTerm]) -> DurationExpr {
        for term in terms {
            self.expr_index(term.demand);
        }
        for term in terms {
            assert!(
                term.denominator != 0,
                "DurationExpr term has a zero denominator"
            );
            assert!(
                term.lower_numerator <= term.upper_numerator,
                "DurationExpr term has an inverted interval"
            );
        }
        handle(
            self.id,
            self.intern(
                Node::Duration(terms.to_vec().into_boxed_slice()),
                Sort::Duration,
            ),
        )
    }
    pub(super) fn duration_add(&mut self, a: DurationExpr, b: DurationExpr) -> DurationExpr {
        let mut operands: Vec<AnyExpr> = Vec::new();
        for duration in [a, b] {
            match self.node(self.expr_index(duration)) {
                Node::Nary {
                    op: NaryOp::DurationAdd,
                    operands: inner,
                } => operands.extend(inner.iter().copied()),
                _ => operands.push(AnyExpr::Duration(duration)),
            }
        }
        handle(
            self.id,
            self.intern(
                Node::Nary {
                    op: NaryOp::DurationAdd,
                    operands: operands.into_boxed_slice(),
                },
                Sort::Duration,
            ),
        )
    }
    pub(super) fn duration_select(
        &mut self,
        c: BoolExpr,
        t: DurationExpr,
        e: DurationExpr,
    ) -> DurationExpr {
        self.expr_index(t);
        self.expr_index(e);
        match self.bool_of(c) {
            Some(true) => return t,
            Some(false) => return e,
            None => (),
        }
        if t == e && self.expr_total(c) {
            return t;
        }
        handle(
            self.id,
            self.intern(
                Node::Select {
                    cond: c,
                    then: AnyExpr::Duration(t),
                    otherwise: AnyExpr::Duration(e),
                },
                Sort::Duration,
            ),
        )
    }
    pub(super) fn duration_scale(&mut self, duration: DurationExpr, by: NatExpr) -> DurationExpr {
        self.expr_index(duration);
        if self.nat_of(by) == Some(1) {
            return duration;
        }
        handle(
            self.id,
            self.intern(Node::DurationScale { duration, by }, Sort::Duration),
        )
    }

    pub(super) fn duration_sum(
        &mut self,
        binder: LoopBinderId,
        extent: NatExpr,
        duration: DurationExpr,
    ) -> DurationExpr {
        let start = self.nat_const(0);
        self.duration_sum_range(binder, start, extent, duration)
    }

    pub(super) fn duration_sum_range(
        &mut self,
        binder: LoopBinderId,
        start: NatExpr,
        extent: NatExpr,
        duration: DurationExpr,
    ) -> DurationExpr {
        self.loop_binder_index(binder);
        self.expr_index(start);
        self.expr_index(extent);
        self.expr_index(duration);

        fn additive_terms(arena: &mut Arena, duration: DurationExpr) -> Vec<DurationTerm> {
            match arena.node(arena.expr_index(duration)).clone() {
                Node::Duration(terms) => terms.into_vec(),
                Node::Nary {
                    op: NaryOp::DurationAdd,
                    operands,
                } => operands
                    .iter()
                    .flat_map(|operand| {
                        let AnyExpr::Duration(duration) = operand else {
                            unreachable!("DurationAdd contains a non-duration operand")
                        };
                        additive_terms(arena, *duration)
                    })
                    .collect(),
                Node::DurationScale { duration, by } => additive_terms(arena, duration)
                    .into_iter()
                    .map(|mut term| {
                        term.demand = arena.nat_mul(term.demand, by);
                        term
                    })
                    .collect(),
                Node::Select {
                    cond,
                    then: AnyExpr::Duration(then),
                    otherwise: AnyExpr::Duration(otherwise),
                } => {
                    let zero = arena.nat_const(0);
                    let mut terms = additive_terms(arena, then);
                    for term in &mut terms {
                        term.demand = arena.nat_select(cond, term.demand, zero);
                    }
                    let mut otherwise_terms = additive_terms(arena, otherwise);
                    for term in &mut otherwise_terms {
                        term.demand = arena.nat_select(cond, zero, term.demand);
                    }
                    terms.extend(otherwise_terms);
                    terms
                }
                _ => unreachable!("duration expression is not an additive duration form"),
            }
        }

        let terms = additive_terms(self, duration)
            .into_iter()
            .map(|mut term| {
                term.demand = self.nat_fold_range(FoldOp::Sum, binder, start, extent, term.demand);
                term
            })
            .collect::<Vec<_>>();
        self.duration(&terms)
    }

    // ----- roots -----------------------------------------------------------

    pub(super) fn root(&mut self, name: RootName, node: AnyExpr) -> RootId {
        self.index(node);
        let id = RootId {
            owner: self.id,
            index: u32::try_from(self.roots.len())
                .unwrap_or_else(|_| panic!("ExprArena root identity space exhausted")),
        };
        self.roots.push((name, node));
        id
    }
    pub(super) fn roots(&self) -> impl Iterator<Item = (RootId, &RootName, AnyExpr)> + '_ {
        self.roots.iter().enumerate().map(|(i, (name, node))| {
            (
                RootId {
                    owner: self.id,
                    index: u32::try_from(i)
                        .unwrap_or_else(|_| panic!("ExprArena root identity space exhausted")),
                },
                name,
                *node,
            )
        })
    }

    pub(super) fn canonical_digest(&self, roots: &[RootId]) -> ExprDigest {
        let mut memo = HashMap::<u32, [u8; 32]>::new();
        let mut digest = Sha256::new();
        digest.update(b"seismic-expr-roots-v2");
        digest.update((roots.len() as u64).to_le_bytes());
        for root in roots {
            if root.owner != self.id {
                panic!(
                    "ExprArena: root id outside its owning arena (§13.3.2): expected {:?}, found {:?}",
                    self.id, root.owner
                );
            }
            let (name, node) = self.roots.get(root.index as usize).expect(OUT_OF_ARENA);
            hash_root_name(&mut digest, name);
            digest.update(self.node_digest(self.index(*node), &mut memo));
        }
        ExprDigest(digest.finalize().into())
    }

    fn node_digest(&self, root: u32, memo: &mut HashMap<u32, [u8; 32]>) -> [u8; 32] {
        if let Some(value) = memo.get(&root) {
            return *value;
        }
        let mut stack = vec![(root, false)];
        while let Some((index, expanded)) = stack.pop() {
            if memo.contains_key(&index) {
                continue;
            }
            if !expanded {
                stack.push((index, true));
                for child in Self::children(self.node(index)).into_iter().rev() {
                    let child = self.index(child);
                    if !memo.contains_key(&child) {
                        stack.push((child, false));
                    }
                }
                continue;
            }
            let mut digest = Sha256::new();
            digest.update(b"seismic-expr-node-v2");
            self.hash_node(&mut digest, self.node(index), memo);
            memo.insert(index, digest.finalize().into());
        }
        memo[&root]
    }

    fn hash_node(&self, digest: &mut Sha256, node: &Node, memo: &HashMap<u32, [u8; 32]>) {
        macro_rules! child {
            ($expression:expr) => {
                digest.update(memo[&self.index($expression)])
            };
        }
        match node {
            Node::NatConst(value) => {
                digest.update([0]);
                digest.update(value.to_le_bytes());
            }
            Node::IntConst(value) => {
                digest.update([1]);
                digest.update(value.to_le_bytes());
            }
            Node::BoolConst(value) => {
                digest.update([2, u8::from(*value)]);
            }
            Node::ScalarConst { dtype, bits } => {
                digest.update([3, dtype_tag(*dtype)]);
                digest.update(bits.to_le_bytes());
            }
            Node::ScalarInteger {
                operation,
                operands,
            } => {
                digest.update([14]);
                digest.update(reference_math::digest());
                match operation {
                    ScalarOp::Binary(op) => {
                        digest.update([0]);
                        digest.update(op.text().as_bytes());
                    }
                    ScalarOp::Unary(op) => {
                        digest.update([1]);
                        digest.update(op.text().as_bytes());
                    }
                    ScalarOp::Math(op) => {
                        digest.update([2]);
                        digest.update(op.name().as_bytes());
                    }
                    ScalarOp::Cast(dtype) => {
                        digest.update([3, dtype_tag(*dtype)]);
                    }
                }
                digest.update((operands.len() as u64).to_le_bytes());
                for (dtype, value) in operands.iter() {
                    digest.update([dtype_tag(*dtype)]);
                    child!(AnyExpr::Int(*value));
                }
            }
            Node::Symbol(symbol) => {
                digest.update([4]);
                hash_symbol_kind(digest, self.record(*symbol).kind);
                digest.update([symbol_sort_tag(self.record(*symbol).sort)]);
                if let SymbolKind::Decision(decision) = self.record(*symbol).kind {
                    let domain = self.decision_domain(decision);
                    digest.update((domain.values().len() as u64).to_le_bytes());
                    for value in domain.values() {
                        digest.update(value.to_le_bytes());
                    }
                }
            }
            Node::Unary { op, operand } => {
                digest.update([5, unary_tag(*op)]);
                child!(*operand);
            }
            Node::Binary { op, lhs, rhs } => {
                digest.update([6, binary_tag(*op)]);
                child!(*lhs);
                child!(*rhs);
            }
            Node::Nary { op, operands } => {
                digest.update([7, nary_tag(*op)]);
                digest.update((operands.len() as u64).to_le_bytes());
                for operand in operands.iter() {
                    child!(*operand);
                }
            }
            Node::Select {
                cond,
                then,
                otherwise,
            } => {
                digest.update([8]);
                child!(AnyExpr::Bool(*cond));
                child!(*then);
                child!(*otherwise);
            }
            Node::Cmp { op, lhs, rhs } => {
                digest.update([9, cmp_tag(*op)]);
                child!(*lhs);
                child!(*rhs);
            }
            Node::In { operand, values } => {
                digest.update([10]);
                child!(*operand);
                digest.update((values.len() as u64).to_le_bytes());
                for value in values.iter() {
                    digest.update(value.to_le_bytes());
                }
            }
            Node::Fold {
                op,
                binder,
                start,
                extent,
                body,
            } => {
                digest.update([11, fold_tag(*op)]);
                digest.update(binder.index.to_le_bytes());
                child!(AnyExpr::Nat(*start));
                child!(AnyExpr::Nat(*extent));
                child!(AnyExpr::Nat(*body));
            }
            Node::Duration(terms) => {
                digest.update([12]);
                digest.update((terms.len() as u64).to_le_bytes());
                for term in terms.iter() {
                    child!(AnyExpr::Nat(term.demand));
                    digest.update(term.lower_numerator.to_le_bytes());
                    digest.update(term.upper_numerator.to_le_bytes());
                    digest.update(term.denominator.to_le_bytes());
                }
            }
            Node::DurationScale { duration, by } => {
                digest.update([13]);
                child!(AnyExpr::Duration(*duration));
                child!(AnyExpr::Nat(*by));
            }
        }
    }
}

fn hash_root_name(digest: &mut Sha256, name: &RootName) {
    match name {
        RootName::AllocationBytes { allocation } => {
            digest.update([0]);
            digest.update(allocation.to_le_bytes());
        }
        RootName::ViewOffset { view } => {
            digest.update([1]);
            digest.update(view.to_le_bytes());
        }
        RootName::ViewStride { view, axis } => {
            digest.update([2]);
            digest.update(view.to_le_bytes());
            digest.update(axis.to_le_bytes());
        }
        RootName::ViewExtent { view, axis } => {
            digest.update([3]);
            digest.update(view.to_le_bytes());
            digest.update(axis.to_le_bytes());
        }
        RootName::LaunchGrid { launch, axis } => {
            digest.update([4]);
            digest.update(launch.to_le_bytes());
            digest.update([*axis]);
        }
        RootName::Workgroup { launch, axis } => {
            digest.update([5]);
            digest.update(launch.to_le_bytes());
            digest.update([*axis]);
        }
        RootName::Guard => digest.update([8]),
        RootName::Duration => digest.update([9]),
        RootName::ErrorBound { output } => {
            digest.update([10]);
            digest.update(output.to_le_bytes());
        }
        RootName::HardConstraints => digest.update([11]),
        RootName::ScheduleCondition { control } => {
            digest.update([12]);
            digest.update(control.to_le_bytes());
        }
        RootName::RepeatStart { repeat } => {
            digest.update([13]);
            digest.update(repeat.to_le_bytes());
        }
        RootName::RepeatEnd { repeat } => {
            digest.update([14]);
            digest.update(repeat.to_le_bytes());
        }
        RootName::LaunchEmpty { launch } => {
            digest.update([15]);
            digest.update(launch.to_le_bytes());
        }
        RootName::KernelNatArgument { kernel, argument } => {
            digest.update([16]);
            digest.update(kernel.to_le_bytes());
            digest.update(argument.to_le_bytes());
        }
        RootName::ScalarReadIndex { step, axis } => {
            digest.update([17]);
            digest.update(step.to_le_bytes());
            digest.update(axis.to_le_bytes());
        }
        RootName::HostEvaluation { step } => {
            digest.update([23]);
            digest.update(step.to_le_bytes());
        }
        RootName::PublishedExtent { step, axis } => {
            digest.update([33]);
            digest.update(step.to_le_bytes());
            digest.update(axis.to_le_bytes());
        }
        RootName::LocalExtent {
            kernel,
            local,
            axis,
        } => {
            digest.update([18]);
            digest.update(kernel.to_le_bytes());
            digest.update(local.to_le_bytes());
            digest.update(axis.to_le_bytes());
        }
        RootName::ScheduleChoice { control } => {
            digest.update([19]);
            digest.update(control.to_le_bytes());
        }
        RootName::IntrinsicWorkgroupBytes { kernel, resource } => {
            digest.update([20]);
            digest.update(kernel.to_le_bytes());
            digest.update(resource.to_le_bytes());
        }
        RootName::IntrinsicParticipantBytes { kernel, resource } => {
            digest.update([21]);
            digest.update(kernel.to_le_bytes());
            digest.update(resource.to_le_bytes());
        }
        RootName::IntrinsicRegisterBytes { kernel, resource } => {
            digest.update([22]);
            digest.update(kernel.to_le_bytes());
            digest.update(resource.to_le_bytes());
        }
        RootName::KernelScalarArgument { kernel, argument } => {
            digest.update([24]);
            digest.update(kernel.to_le_bytes());
            digest.update(argument.to_le_bytes());
        }
        RootName::LocalOffset { launch, local } => {
            digest.update([25]);
            digest.update(launch.to_le_bytes());
            digest.update(local.to_le_bytes());
        }
        RootName::LocalStride {
            launch,
            local,
            axis,
        } => {
            digest.update([26]);
            digest.update(launch.to_le_bytes());
            digest.update(local.to_le_bytes());
            digest.update(axis.to_le_bytes());
        }
        RootName::LocalClassBytes { launch, class } => {
            digest.update([27]);
            digest.update(launch.to_le_bytes());
            digest.update([*class]);
        }
        RootName::LaunchScratchBytes { launch, class } => {
            digest.update([28]);
            digest.update(launch.to_le_bytes());
            digest.update([*class]);
        }
        RootName::AddressableResourceOffset { kernel, lease } => {
            digest.update([29]);
            digest.update(kernel.to_le_bytes());
            digest.update(lease.to_le_bytes());
        }
        RootName::AddressableResourceUnits { kernel, lease } => {
            digest.update([30]);
            digest.update(kernel.to_le_bytes());
            digest.update(lease.to_le_bytes());
        }
        RootName::LaunchAbiBytes { launch, allocation } => {
            digest.update([31]);
            digest.update(launch.to_le_bytes());
            digest.update(allocation.to_le_bytes());
        }
        RootName::NumericalCondition { child } => {
            digest.update([31]);
            digest.update(child.to_le_bytes());
        }
        RootName::RegionOperand { operand } => {
            digest.update([34]);
            digest.update(operand.to_le_bytes());
        }
        RootName::NumericalOperationMultiplicity { operation } => {
            digest.update([32]);
            digest.update(operation.to_le_bytes());
        }
    }
}

fn hash_symbol_kind(digest: &mut Sha256, kind: SymbolKind) {
    match kind {
        SymbolKind::TemplateDimension(ordinal) => {
            digest.update([0]);
            digest.update(ordinal.to_le_bytes());
        }
        SymbolKind::CallDimension(id) => {
            digest.update([1]);
            digest.update((id.index() as u64).to_le_bytes());
        }
        SymbolKind::CallScalar(id) => {
            digest.update([2]);
            digest.update((id.index() as u64).to_le_bytes());
        }
        SymbolKind::RuntimeValue(id) => {
            digest.update([7]);
            digest.update((id.function().index() as u64).to_le_bytes());
            digest.update((id.index() as u64).to_le_bytes());
        }
        SymbolKind::TargetConstant(id) => {
            digest.update([3]);
            digest.update(id.index.to_le_bytes());
        }
        SymbolKind::Decision(id) => {
            digest.update([4]);
            digest.update(id.index.to_le_bytes());
        }
        SymbolKind::LoopBinder(id) => {
            digest.update([5]);
            digest.update(id.index.to_le_bytes());
        }
        SymbolKind::ScheduleSlot(ordinal) => {
            digest.update([6]);
            digest.update(ordinal.to_le_bytes());
        }
    }
}

fn dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::Bool => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        DType::F32 => 3,
        DType::I32 => 4,
        DType::U32 => 5,
    }
}
fn symbol_sort_tag(sort: SymbolSort) -> u8 {
    match sort {
        SymbolSort::Nat => 0,
        SymbolSort::Int => 1,
        SymbolSort::Scalar(dtype) => 2 + dtype_tag(dtype),
    }
}
fn unary_tag(op: UnaryOp) -> u8 {
    match op {
        UnaryOp::Not => 0,
        UnaryOp::NatFromInt => 1,
        UnaryOp::IntFromNat => 2,
        UnaryOp::IntFromScalar => 3,
        UnaryOp::ScalarIntegerDefined => 4,
    }
}
fn binary_tag(op: BinaryOp) -> u8 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::CeilDiv => 4,
        BinaryOp::Rem => 5,
        BinaryOp::Min => 6,
        BinaryOp::Max => 7,
        BinaryOp::AlignUp => 8,
        BinaryOp::And => 9,
        BinaryOp::Or => 10,
        BinaryOp::Implies => 11,
        BinaryOp::Iff => 12,
    }
}
fn nary_tag(op: NaryOp) -> u8 {
    match op {
        NaryOp::All => 0,
        NaryOp::Any => 1,
        NaryOp::Product => 2,
        NaryOp::DurationAdd => 3,
    }
}
fn cmp_tag(op: CmpOp) -> u8 {
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    }
}
fn fold_tag(op: FoldOp) -> u8 {
    match op {
        FoldOp::Sum => 0,
        FoldOp::Product => 1,
        FoldOp::Max => 2,
    }
}

fn compare<T: Ord>(op: CmpOp, x: T, y: T) -> bool {
    match op {
        CmpOp::Eq => x == y,
        CmpOp::Ne => x != y,
        CmpOp::Lt => x < y,
        CmpOp::Le => x <= y,
        CmpOp::Gt => x > y,
        CmpOp::Ge => x >= y,
    }
}

fn reflexive(op: CmpOp) -> bool {
    matches!(op, CmpOp::Eq | CmpOp::Le | CmpOp::Ge)
}

fn scalar_bits<T: ScalarSort>(value: T::Value) -> u32 {
    T::encode(value)
}

fn symbol_value_matches(sort: SymbolSort, value: SymbolValue) -> bool {
    matches!(
        (sort, value),
        (SymbolSort::Nat, SymbolValue::Nat(_))
            | (SymbolSort::Int, SymbolValue::Int(_))
            | (SymbolSort::Scalar(DType::F32), SymbolValue::F32(_))
            | (SymbolSort::Scalar(DType::F16), SymbolValue::F16(_))
            | (SymbolSort::Scalar(DType::BF16), SymbolValue::BF16(_))
            | (SymbolSort::Scalar(DType::Bool), SymbolValue::Bool(_))
            | (SymbolSort::Scalar(DType::I32), SymbolValue::I32(_))
            | (SymbolSort::Scalar(DType::U32), SymbolValue::U32(_))
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ScalarVal {
    F32(f32),
    F16(u16),
    BF16(u16),
    Bool(bool),
    I32(i32),
    U32(u32),
}

impl ScalarVal {
    fn decode(dtype: DType, bits: u32) -> Self {
        match dtype {
            DType::I32 => Self::I32(bits as i32),
            DType::U32 => Self::U32(bits),
            DType::F32 => Self::F32(f32::from_bits(bits)),
            DType::F16 => Self::F16(bits as u16),
            DType::BF16 => Self::BF16(bits as u16),
            DType::Bool => Self::Bool(bits != 0),
        }
    }
    fn from_value(value: SymbolValue) -> Option<Self> {
        match value {
            SymbolValue::F32(v) => Some(Self::F32(v)),
            SymbolValue::F16(v) => Some(Self::F16(v)),
            SymbolValue::BF16(v) => Some(Self::BF16(v)),
            SymbolValue::Bool(v) => Some(Self::Bool(v)),
            SymbolValue::I32(v) => Some(Self::I32(v)),
            SymbolValue::U32(v) => Some(Self::U32(v)),
            SymbolValue::Nat(_) | SymbolValue::Int(_) => None,
        }
    }
    fn matches(self, dtype: DType) -> bool {
        matches!(
            (self, dtype),
            (Self::I32(_), DType::I32)
                | (Self::U32(_), DType::U32)
                | (Self::F32(_), DType::F32)
                | (Self::F16(_), DType::F16)
                | (Self::BF16(_), DType::BF16)
                | (Self::Bool(_), DType::Bool)
        )
    }
    fn bits(self) -> u32 {
        match self {
            Self::F32(value) => value.to_bits(),
            Self::F16(value) | Self::BF16(value) => value as u32,
            Self::Bool(value) => u32::from(value),
            Self::I32(value) => value as u32,
            Self::U32(value) => value,
        }
    }
    /// Comparison with IEEE semantics for floats: every ordered comparison
    /// and equality with a NaN is false, inequality is true.
    fn compare(self, op: CmpOp, other: Self) -> bool {
        match (self, other) {
            (Self::I32(x), Self::I32(y)) => compare(op, x, y),
            (Self::U32(x), Self::U32(y)) => compare(op, x, y),
            (Self::Bool(x), Self::Bool(y)) => compare(op, x, y),
            (Self::F32(x), Self::F32(y)) => match op {
                CmpOp::Eq => x == y,
                CmpOp::Ne => x != y,
                CmpOp::Lt => x < y,
                CmpOp::Le => x <= y,
                CmpOp::Gt => x > y,
                CmpOp::Ge => x >= y,
            },
            (Self::F16(x), Self::F16(y)) => compare_float(op, f16_to_f32(x), f16_to_f32(y)),
            (Self::BF16(x), Self::BF16(y)) => compare_float(
                op,
                f32::from_bits((x as u32) << 16),
                f32::from_bits((y as u32) << 16),
            ),
            // Mixed sorts cannot be constructed through the typed surface;
            // treated as unequal so evaluation stays total.
            _ => matches!(op, CmpOp::Ne),
        }
    }
}

fn compare_float(op: CmpOp, x: f32, y: f32) -> bool {
    match op {
        CmpOp::Eq => x == y,
        CmpOp::Ne => x != y,
        CmpOp::Lt => x < y,
        CmpOp::Le => x <= y,
        CmpOp::Gt => x > y,
        CmpOp::Ge => x >= y,
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;
    let encoded = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = (fraction << shift) & 0x03ff;
            sign | ((113 - shift) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(encoded)
}

// ---------------------------------------------------------------------------
// analysis: view, free symbols, side conditions, partial evaluation
// ---------------------------------------------------------------------------

impl Arena {
    pub(super) fn view(&self, node: AnyExpr) -> NodeView<'_> {
        match self.node(self.index(node)) {
            Node::NatConst(v) => NodeView::NatConst(*v),
            Node::IntConst(v) => NodeView::IntConst(*v),
            Node::BoolConst(v) => NodeView::BoolConst(*v),
            Node::ScalarConst { dtype, bits } => NodeView::ScalarConst {
                dtype: *dtype,
                bits: *bits,
            },
            Node::ScalarInteger {
                operation,
                operands,
            } => NodeView::ScalarInteger {
                operation: *operation,
                operands,
            },
            Node::Symbol(s) => NodeView::Symbol(*s),
            Node::Unary { op, operand } => NodeView::Unary {
                op: *op,
                operand: *operand,
            },
            Node::Binary { op, lhs, rhs } => NodeView::Binary {
                op: *op,
                lhs: *lhs,
                rhs: *rhs,
            },
            Node::Nary { op, operands } => NodeView::Nary {
                op: *op,
                operands: &operands[..],
            },
            Node::Select {
                cond,
                then,
                otherwise,
            } => NodeView::Select {
                cond: *cond,
                then: *then,
                otherwise: *otherwise,
            },
            Node::Cmp { op, lhs, rhs } => NodeView::Cmp {
                op: *op,
                lhs: *lhs,
                rhs: *rhs,
            },
            Node::In { operand, values } => NodeView::In {
                operand: *operand,
                values: &values[..],
            },
            Node::Fold {
                op,
                binder,
                start,
                extent,
                body,
            } => NodeView::Fold {
                op: *op,
                binder: *binder,
                start: *start,
                extent: *extent,
                body: *body,
            },
            Node::Duration(terms) => NodeView::Duration(&terms[..]),
            Node::DurationScale { duration, by } => NodeView::DurationScale {
                duration: *duration,
                by: *by,
            },
        }
    }

    pub(super) fn free_symbols(&self, node: AnyExpr) -> Vec<SymbolId> {
        self.free
            .get(self.index(node) as usize)
            .expect(OUT_OF_ARENA)
            .clone()
    }

    /// The conjunction of every side condition beneath `node`, respecting
    /// short-circuit and select semantics: a sub-expression's conditions are
    /// required only where it is evaluated.
    pub(super) fn side_conditions(&mut self, node: AnyExpr) -> BoolExpr {
        let mut memo = HashMap::new();
        self.side_conditions_of(self.index(node), &mut memo)
    }

    fn side_conditions_of(&mut self, i: u32, memo: &mut HashMap<u32, BoolExpr>) -> BoolExpr {
        if self.is_total(i) {
            return self.bool_const(true);
        }
        if let Some(&sc) = memo.get(&i) {
            return sc;
        }
        let node = self.node(i).clone();
        let sc = match node {
            Node::NatConst(_)
            | Node::IntConst(_)
            | Node::BoolConst(_)
            | Node::ScalarConst { .. }
            | Node::Symbol(_) => self.bool_const(true),
            Node::ScalarInteger { operands, .. } => {
                let mut conditions = operands
                    .iter()
                    .map(|(_, value)| self.side_conditions_of(self.expr_index(*value), memo))
                    .collect::<Vec<_>>();
                conditions.push(self.scalar_integer_defined(self.h(i)));
                self.all(&conditions)
            }
            Node::Unary {
                op: UnaryOp::ScalarIntegerDefined,
                operand,
            } => {
                let conditions = Self::children(self.node(self.index(operand)))
                    .into_iter()
                    .map(|child| self.side_conditions_of(self.index(child), memo))
                    .collect::<Vec<_>>();
                self.all(&conditions)
            }
            Node::Unary { op, operand } => {
                let inner = self.side_conditions_of(self.index(operand), memo);
                match (op, operand) {
                    (UnaryOp::NatFromInt, AnyExpr::Int(int)) => {
                        let zero = self.int_const(0);
                        let non_negative = self.int_cmp(CmpOp::Ge, int, zero);
                        self.and(inner, non_negative)
                    }
                    _ => inner,
                }
            }
            Node::Binary { op, lhs, rhs } => {
                let l = self.side_conditions_of(self.index(lhs), memo);
                match (op, lhs, rhs) {
                    (BinaryOp::And | BinaryOp::Implies, AnyExpr::Bool(a), AnyExpr::Bool(b)) => {
                        let r = self.side_conditions_of(self.expr_index(b), memo);
                        let guarded = self.implies(a, r);
                        self.and(l, guarded)
                    }
                    (BinaryOp::Or, AnyExpr::Bool(a), AnyExpr::Bool(b)) => {
                        let r = self.side_conditions_of(self.expr_index(b), memo);
                        let not_a = self.not(a);
                        let guarded = self.implies(not_a, r);
                        self.and(l, guarded)
                    }
                    (BinaryOp::Sub, AnyExpr::Nat(a), AnyExpr::Nat(b)) => {
                        let r = self.side_conditions_of(self.expr_index(b), memo);
                        let both = self.and(l, r);
                        let ordered = self.nat_cmp(CmpOp::Le, b, a);
                        self.and(both, ordered)
                    }
                    (
                        BinaryOp::Div | BinaryOp::CeilDiv | BinaryOp::Rem | BinaryOp::AlignUp,
                        _,
                        AnyExpr::Nat(b),
                    ) => {
                        let r = self.side_conditions_of(self.expr_index(b), memo);
                        let both = self.and(l, r);
                        let zero = self.nat_const(0);
                        let nonzero = self.nat_cmp(CmpOp::Ne, b, zero);
                        self.and(both, nonzero)
                    }
                    (BinaryOp::Div | BinaryOp::Rem, _, AnyExpr::Int(b)) => {
                        let r = self.side_conditions_of(self.expr_index(b), memo);
                        let both = self.and(l, r);
                        let zero = self.int_const(0);
                        let nonzero = self.int_cmp(CmpOp::Ne, b, zero);
                        self.and(both, nonzero)
                    }
                    _ => {
                        let r = self.side_conditions_of(self.index(rhs), memo);
                        self.and(l, r)
                    }
                }
            }
            Node::Nary { op, operands } => match op {
                NaryOp::All | NaryOp::Any => {
                    // Sequential short circuit, right to left.
                    let mut acc = self.bool_const(true);
                    for operand in operands.iter().rev() {
                        let AnyExpr::Bool(term) = *operand else {
                            continue;
                        };
                        let own = self.side_conditions_of(self.expr_index(term), memo);
                        let guard = if op == NaryOp::All {
                            term
                        } else {
                            self.not(term)
                        };
                        let rest = self.implies(guard, acc);
                        acc = self.and(own, rest);
                    }
                    acc
                }
                NaryOp::Product | NaryOp::DurationAdd => {
                    let mut acc = self.bool_const(true);
                    for operand in operands.iter() {
                        let own = self.side_conditions_of(self.index(*operand), memo);
                        acc = self.and(acc, own);
                    }
                    acc
                }
            },
            Node::Select {
                cond,
                then,
                otherwise,
            } => {
                let c = self.side_conditions_of(self.expr_index(cond), memo);
                let t = self.side_conditions_of(self.index(then), memo);
                let e = self.side_conditions_of(self.index(otherwise), memo);
                let when_then = self.implies(cond, t);
                let not_cond = self.not(cond);
                let when_else = self.implies(not_cond, e);
                let both = self.and(when_then, when_else);
                self.and(c, both)
            }
            Node::Cmp { lhs, rhs, .. } => {
                let l = self.side_conditions_of(self.index(lhs), memo);
                let r = self.side_conditions_of(self.index(rhs), memo);
                self.and(l, r)
            }
            Node::In { operand, .. } => self.side_conditions_of(self.index(operand), memo),
            Node::Fold {
                binder,
                start,
                extent,
                body,
                ..
            } => {
                let s = self.side_conditions_of(self.expr_index(start), memo);
                let e = self.side_conditions_of(self.expr_index(extent), memo);
                let e = self.and(s, e);
                let b = self.side_conditions_of(self.expr_index(body), memo);
                if self.bool_of(b) == Some(true) {
                    e
                } else {
                    // `for all i < extent: b(i)` as `max_i [not b(i)] == 0`.
                    let zero = self.nat_const(0);
                    let one = self.nat_const(1);
                    let violation = self.nat_select(b, zero, one);
                    let worst = self.nat_fold_range(FoldOp::Max, binder, start, extent, violation);
                    let none = self.nat_cmp(CmpOp::Eq, worst, zero);
                    self.and(e, none)
                }
            }
            Node::Duration(terms) => {
                let valid = terms.iter().all(|term| {
                    term.denominator != 0 && term.lower_numerator <= term.upper_numerator
                });
                let mut acc = self.bool_const(valid);
                for term in terms.iter() {
                    let own = self.side_conditions_of(self.expr_index(term.demand), memo);
                    acc = self.and(acc, own);
                }
                acc
            }
            Node::DurationScale { duration, by } => {
                let c = self.side_conditions_of(self.expr_index(duration), memo);
                let b = self.side_conditions_of(self.expr_index(by), memo);
                self.and(c, b)
            }
        };
        memo.insert(i, sc);
        sc
    }

    pub(super) fn partial<Sort>(&mut self, node: Expr<Sort>, a: &PartialAssignment) -> Expr<Sort> {
        for (symbol, value) in a.iter() {
            let sort = self.record(symbol).sort;
            if !symbol_value_matches(sort, value) {
                panic!("ExprArena partial assignment value does not match its symbol sort");
            }
        }
        let mut memo = HashMap::new();
        let mut shadow: Vec<SymbolId> = Vec::new();
        let node = self.expr_index(node);
        handle(
            self.id,
            self.partial_of(node, a, &HashMap::new(), &mut shadow, &mut memo),
        )
    }

    pub(super) fn resolve_runtime_values(
        &mut self,
        node: IntExpr,
        values: &[(crate::ids::SemanticValueId, IntExpr)],
    ) -> IntExpr {
        let mut expressions = HashMap::new();
        let node = self.expr_index(node);
        for symbol in &self.free[node as usize] {
            if let SymbolKind::RuntimeValue(value) = self.record(*symbol).kind {
                if let Some((_, expression)) =
                    values.iter().find(|(candidate, _)| *candidate == value)
                {
                    expressions.insert(*symbol, self.expr_index(*expression));
                }
            }
        }
        let result = self.partial_of(
            node,
            &PartialAssignment::new(),
            &expressions,
            &mut Vec::new(),
            &mut HashMap::new(),
        );
        self.h(result)
    }

    fn partial_of(
        &mut self,
        i: u32,
        a: &PartialAssignment,
        expressions: &HashMap<SymbolId, u32>,
        shadow: &mut Vec<SymbolId>,
        memo: &mut HashMap<u32, u32>,
    ) -> u32 {
        let free = &self.free[i as usize];
        if !free
            .iter()
            .any(|s| a.get(*s).is_some() || expressions.contains_key(s))
        {
            return i;
        }
        if shadow.is_empty() {
            if let Some(&r) = memo.get(&i) {
                return r;
            }
        }
        let node = self.node(i).clone();
        let result = match node {
            Node::NatConst(_)
            | Node::IntConst(_)
            | Node::BoolConst(_)
            | Node::ScalarConst { .. } => i,
            Node::Symbol(s) => {
                if shadow.contains(&s) {
                    i
                } else if let Some(&value) = expressions.get(&s) {
                    value
                } else {
                    match (self.record(s).sort, a.get(s)) {
                        (SymbolSort::Nat, Some(SymbolValue::Nat(v))) => self.nat_exact(v).index,
                        (SymbolSort::Int, Some(SymbolValue::Int(v))) => self.int_exact(v).index,
                        (SymbolSort::Scalar(dtype), Some(value)) => {
                            match ScalarVal::from_value(value) {
                                Some(scalar) if scalar.matches(dtype) => {
                                    let bits = scalar.bits();
                                    self.intern(
                                        Node::ScalarConst { dtype, bits },
                                        Sort::Scalar(dtype),
                                    )
                                }
                                _ => i,
                            }
                        }
                        _ => i,
                    }
                }
            }
            Node::Unary { op, operand } => {
                let o = self.partial_of(self.index(operand), a, expressions, shadow, memo);
                match op {
                    UnaryOp::Not => self.not(self.h(o)).index,
                    UnaryOp::ScalarIntegerDefined => self.scalar_integer_defined(self.h(o)).index,
                    UnaryOp::NatFromInt => self.nat_from_int(self.h(o)).index,
                    UnaryOp::IntFromNat => self.int_from_nat(self.h(o)).index,
                    UnaryOp::IntFromScalar => {
                        self.int_from_scalar(ErasedScalarExpr {
                            owner: self.id,
                            index: o,
                        })
                        .index
                    }
                }
            }
            Node::ScalarInteger {
                operation,
                operands,
            } => {
                let operands = operands
                    .iter()
                    .map(|(dtype, value)| {
                        let node =
                            self.partial_of(self.expr_index(*value), a, expressions, shadow, memo);
                        (*dtype, self.h(node))
                    })
                    .collect::<Vec<_>>();
                self.scalar_integer(operation, &operands).index
            }
            Node::Binary { op, lhs, rhs } => {
                let l = self.partial_of(self.index(lhs), a, expressions, shadow, memo);
                let r = self.partial_of(self.index(rhs), a, expressions, shadow, memo);
                match self.sort(i) {
                    Sort::Nat => {
                        match op {
                            BinaryOp::Add => self.nat_add(self.h(l), self.h(r)),
                            BinaryOp::Sub => self.nat_sub(self.h(l), self.h(r)),
                            BinaryOp::Mul => self.nat_mul(self.h(l), self.h(r)),
                            BinaryOp::Div => self.nat_div(self.h(l), self.h(r)),
                            BinaryOp::CeilDiv => self.nat_ceil_div(self.h(l), self.h(r)),
                            BinaryOp::Rem => self.nat_rem(self.h(l), self.h(r)),
                            BinaryOp::Min => self.nat_min(self.h(l), self.h(r)),
                            BinaryOp::Max => self.nat_max(self.h(l), self.h(r)),
                            BinaryOp::AlignUp => self.nat_align_up(self.h(l), self.h(r)),
                            BinaryOp::And | BinaryOp::Or | BinaryOp::Implies | BinaryOp::Iff => {
                                self.nat_binary(op, self.h(l), self.h(r))
                            }
                        }
                        .index
                    }
                    Sort::Int => {
                        match op {
                            BinaryOp::Add => self.int_add(self.h(l), self.h(r)),
                            BinaryOp::Sub => self.int_sub(self.h(l), self.h(r)),
                            BinaryOp::Mul => self.int_mul(self.h(l), self.h(r)),
                            BinaryOp::Div => self.int_div(self.h(l), self.h(r)),
                            BinaryOp::Rem => self.int_rem(self.h(l), self.h(r)),
                            BinaryOp::Min => self.int_min(self.h(l), self.h(r)),
                            BinaryOp::Max => self.int_max(self.h(l), self.h(r)),
                            _ => self.int_binary(op, self.h(l), self.h(r)),
                        }
                        .index
                    }
                    Sort::Bool => {
                        match op {
                            BinaryOp::And => self.and(self.h(l), self.h(r)),
                            BinaryOp::Or => self.or(self.h(l), self.h(r)),
                            BinaryOp::Implies => self.implies(self.h(l), self.h(r)),
                            BinaryOp::Iff => self.iff(self.h(l), self.h(r)),
                            _ => self.bool_binary(op, self.h(l), self.h(r)),
                        }
                        .index
                    }
                    Sort::Scalar(_) | Sort::Duration => i,
                }
            }
            Node::Nary { op, operands } => {
                let rebuilt: Vec<u32> = operands
                    .iter()
                    .map(|o| self.partial_of(self.index(*o), a, expressions, shadow, memo))
                    .collect();
                match op {
                    NaryOp::All => {
                        let terms: Vec<BoolExpr> = rebuilt.into_iter().map(|i| self.h(i)).collect();
                        self.all(&terms).index
                    }
                    NaryOp::Any => {
                        let terms: Vec<BoolExpr> = rebuilt.into_iter().map(|i| self.h(i)).collect();
                        self.any(&terms).index
                    }
                    NaryOp::Product => {
                        let factors: Vec<NatExpr> =
                            rebuilt.into_iter().map(|i| self.h(i)).collect();
                        self.nat_product(&factors).index
                    }
                    NaryOp::DurationAdd => {
                        let terms = rebuilt
                            .into_iter()
                            .map(|i| handle::<super::sort::Duration>(self.id, i))
                            .collect::<Vec<_>>();
                        match terms.split_first() {
                            None => i,
                            Some((first, rest)) => {
                                let mut sum = *first;
                                for term in rest {
                                    sum = self.duration_add(sum, *term);
                                }
                                sum.index
                            }
                        }
                    }
                }
            }
            Node::Select {
                cond,
                then,
                otherwise,
            } => {
                let c_index = self.partial_of(cond.index, a, expressions, shadow, memo);
                let c = self.h(c_index);
                let t = self.partial_of(self.index(then), a, expressions, shadow, memo);
                let e = self.partial_of(self.index(otherwise), a, expressions, shadow, memo);
                match self.sort(i) {
                    Sort::Nat => self.nat_select(c, self.h(t), self.h(e)).index,
                    Sort::Int => self.int_select(c, self.h(t), self.h(e)).index,
                    Sort::Duration => self.duration_select(c, self.h(t), self.h(e)).index,
                    Sort::Bool | Sort::Scalar(_) => i,
                }
            }
            Node::Cmp { op, lhs, rhs } => {
                let l = self.partial_of(self.index(lhs), a, expressions, shadow, memo);
                let r = self.partial_of(self.index(rhs), a, expressions, shadow, memo);
                match lhs {
                    AnyExpr::Nat(_) => self.nat_cmp(op, self.h(l), self.h(r)).index,
                    AnyExpr::Int(_) => self.int_cmp(op, self.h(l), self.h(r)).index,
                    AnyExpr::Scalar(_) => self.scalar_cmp_raw(op, l, r).index,
                    AnyExpr::Bool(_) | AnyExpr::Duration(_) => i,
                }
            }
            Node::In { operand, values } => {
                let o = self.partial_of(self.index(operand), a, expressions, shadow, memo);
                self.in_node(self.erased(o), &values).index
            }
            Node::Fold {
                op,
                binder,
                start,
                extent,
                body,
            } => {
                let s = self.partial_of(start.index, a, expressions, shadow, memo);
                let e = self.partial_of(extent.index, a, expressions, shadow, memo);
                let bound: Vec<SymbolId> = self
                    .binder_symbols
                    .get(&binder)
                    .cloned()
                    .unwrap_or_default();
                let before = shadow.len();
                shadow.extend(bound);
                let b = self.partial_of(body.index, a, expressions, shadow, memo);
                shadow.truncate(before);
                self.nat_fold_range(op, binder, self.h(s), self.h(e), self.h(b))
                    .index
            }
            Node::Duration(terms) => {
                let rebuilt: Vec<DurationTerm> = terms
                    .iter()
                    .map(|term| DurationTerm {
                        demand: {
                            let demand =
                                self.partial_of(term.demand.index, a, expressions, shadow, memo);
                            self.h(demand)
                        },
                        lower_numerator: term.lower_numerator,
                        upper_numerator: term.upper_numerator,
                        denominator: term.denominator,
                    })
                    .collect();
                self.duration(&rebuilt).index
            }
            Node::DurationScale { duration, by } => {
                let duration = self.partial_of(duration.index, a, expressions, shadow, memo);
                let b = self.partial_of(by.index, a, expressions, shadow, memo);
                self.duration_scale(self.h(duration), self.h(b)).index
            }
        };
        if shadow.is_empty() {
            memo.insert(i, result);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// evaluation
// ---------------------------------------------------------------------------

/// A self-contained copy of one reachable sub-DAG with compact indices.
#[derive(Debug)]
struct Program {
    nodes: Vec<Node>,
    sorts: Vec<Sort>,
    /// A node is memoised within one evaluation when it mentions no binder.
    memoizable: Vec<bool>,
    symbol_sorts: HashMap<SymbolId, SymbolSort>,
    binder_symbols: HashMap<LoopBinderId, Vec<SymbolId>>,
    /// Common denominator of every duration term of the arena, `None` when
    /// it does not fit `u64`.
    common_denominator: Option<u64>,
    root: u32,
    reads: Vec<SymbolId>,
}

impl Program {
    fn retained_metadata_bytes(&self) -> usize {
        use std::mem::{size_of, size_of_val};
        let nested: usize = self
            .nodes
            .iter()
            .map(|node| match node {
                Node::Nary { operands, .. } => size_of_val(operands.as_ref()),
                Node::ScalarInteger { operands, .. } => size_of_val(operands.as_ref()),
                Node::In { values, .. } => size_of_val(values.as_ref()),
                Node::Duration(terms) => size_of_val(terms.as_ref()),
                _ => 0,
            })
            .sum();
        size_of::<Self>()
            + self.nodes.capacity() * size_of::<Node>()
            + self.sorts.capacity() * size_of::<Sort>()
            + self.memoizable.capacity()
            + self.symbol_sorts.capacity() * (size_of::<(SymbolId, SymbolSort)>() + 1)
            + self.binder_symbols.capacity() * (size_of::<(LoopBinderId, Vec<SymbolId>)>() + 1)
            + self
                .binder_symbols
                .values()
                .map(|symbols| symbols.capacity() * size_of::<SymbolId>())
                .sum::<usize>()
            + self.reads.capacity() * size_of::<SymbolId>()
            + nested
    }
}

#[derive(Clone, Debug)]
enum Val {
    Int(BigInt),
    Bool(bool),
    Scalar(ScalarVal),
    Duration {
        lower: u128,
        upper: u128,
        denominator: u64,
    },
}

impl Arena {
    fn extract(&self, root: u32) -> Program {
        let mut map: HashMap<u32, u32> = HashMap::new();
        let mut order: Vec<u32> = Vec::new();
        // Post-order over the DAG: children before parents.
        let mut stack: Vec<(u32, bool)> = vec![(root, false)];
        while let Some((i, expanded)) = stack.pop() {
            if map.contains_key(&i) {
                continue;
            }
            if expanded {
                let compact = u32::try_from(order.len())
                    .unwrap_or_else(|_| panic!("compiled expression identity space exhausted"));
                map.insert(i, compact);
                order.push(i);
                continue;
            }
            stack.push((i, true));
            for child in Self::children(self.node(i)) {
                if !map.contains_key(&self.index(child)) {
                    stack.push((self.index(child), false));
                }
            }
        }
        let remap = |e: AnyExpr| -> AnyExpr {
            let new = map[&self.index(e)];
            match e {
                AnyExpr::Nat(_) => AnyExpr::Nat(self.h(new)),
                AnyExpr::Int(_) => AnyExpr::Int(self.h(new)),
                AnyExpr::Bool(_) => AnyExpr::Bool(self.h(new)),
                AnyExpr::Duration(_) => AnyExpr::Duration(self.h(new)),
                AnyExpr::Scalar(_) => AnyExpr::Scalar(ErasedScalarExpr {
                    owner: self.id,
                    index: new,
                }),
            }
        };
        let mut nodes = Vec::with_capacity(order.len());
        let mut sorts = Vec::with_capacity(order.len());
        let mut memoizable = Vec::with_capacity(order.len());
        let mut symbol_sorts = HashMap::new();
        let mut binder_symbols = HashMap::new();
        for &i in &order {
            let node = match self.node(i).clone() {
                Node::ScalarInteger {
                    operation,
                    operands,
                } => Node::ScalarInteger {
                    operation,
                    operands: operands
                        .iter()
                        .map(|(dtype, value)| (*dtype, self.h(map[&self.expr_index(*value)])))
                        .collect(),
                },
                Node::Unary { op, operand } => Node::Unary {
                    op,
                    operand: remap(operand),
                },
                Node::Binary { op, lhs, rhs } => Node::Binary {
                    op,
                    lhs: remap(lhs),
                    rhs: remap(rhs),
                },
                Node::Nary { op, operands } => Node::Nary {
                    op,
                    operands: operands.iter().map(|o| remap(*o)).collect(),
                },
                Node::Select {
                    cond,
                    then,
                    otherwise,
                } => Node::Select {
                    cond: self.h(map[&self.expr_index(cond)]),
                    then: remap(then),
                    otherwise: remap(otherwise),
                },
                Node::Cmp { op, lhs, rhs } => Node::Cmp {
                    op,
                    lhs: remap(lhs),
                    rhs: remap(rhs),
                },
                Node::In { operand, values } => Node::In {
                    operand: remap(operand),
                    values,
                },
                Node::Fold {
                    op,
                    binder,
                    start,
                    extent,
                    body,
                } => {
                    let bound = self
                        .binder_symbols
                        .get(&binder)
                        .cloned()
                        .unwrap_or_default();
                    for s in &bound {
                        symbol_sorts.insert(*s, self.record(*s).sort);
                    }
                    binder_symbols.insert(binder, bound);
                    Node::Fold {
                        op,
                        binder,
                        start: self.h(map[&self.expr_index(start)]),
                        extent: self.h(map[&self.expr_index(extent)]),
                        body: self.h(map[&self.expr_index(body)]),
                    }
                }
                Node::Duration(terms) => Node::Duration(
                    terms
                        .iter()
                        .map(|term| DurationTerm {
                            demand: self.h(map[&self.expr_index(term.demand)]),
                            lower_numerator: term.lower_numerator,
                            upper_numerator: term.upper_numerator,
                            denominator: term.denominator,
                        })
                        .collect(),
                ),
                Node::DurationScale { duration, by } => Node::DurationScale {
                    duration: self.h(map[&self.expr_index(duration)]),
                    by: self.h(map[&self.expr_index(by)]),
                },
                leaf @ (Node::NatConst(_)
                | Node::IntConst(_)
                | Node::BoolConst(_)
                | Node::ScalarConst { .. }
                | Node::Symbol(_)) => {
                    if let Node::Symbol(s) = leaf {
                        symbol_sorts.insert(s, self.record(s).sort);
                    }
                    leaf
                }
            };
            nodes.push(node);
            sorts.push(self.sort(i));
            memoizable.push(
                !self.free[i as usize]
                    .iter()
                    .any(|s| matches!(self.record(*s).kind, SymbolKind::LoopBinder(_))),
            );
        }
        let common_denominator = nodes
            .iter()
            .filter_map(|node| match node {
                Node::Duration(terms) => Some(terms.iter()),
                _ => None,
            })
            .flatten()
            .try_fold(1_u64, |acc, term| {
                let divisor = gcd(acc, term.denominator);
                (acc / divisor).checked_mul(term.denominator)
            });
        Program {
            nodes,
            sorts,
            memoizable,
            symbol_sorts,
            binder_symbols,
            common_denominator,
            root: map[&root],
            reads: self.free[root as usize].clone(),
        }
    }

    fn evaluate(&self, root: u32, values: &Assignment) -> Result<Val, EvalError> {
        let program = self.extract(root);
        Evaluator::new(&program, &|s| values.get(s)).run(program.root)
    }

    pub(super) fn eval_nat(&self, n: NatExpr, v: &Assignment) -> Result<BigUint, EvalError> {
        to_nat(self.evaluate(self.expr_index(n), v)?)
    }
    pub(super) fn eval_int(&self, n: IntExpr, v: &Assignment) -> Result<BigInt, EvalError> {
        to_int(self.evaluate(self.expr_index(n), v)?)
    }
    pub(super) fn eval_bool(&self, n: BoolExpr, v: &Assignment) -> Result<bool, EvalError> {
        to_bool(self.evaluate(self.expr_index(n), v)?)
    }
    pub(super) fn eval_duration(
        &self,
        n: DurationExpr,
        v: &Assignment,
    ) -> Result<DurationEstimate, EvalError> {
        to_duration(self.evaluate(self.expr_index(n), v)?)
    }

    fn compile<T: 'static>(
        &self,
        root: u32,
        convert: fn(Val) -> Result<T, EvalError>,
    ) -> Compiled<T> {
        self.compile_with(root, &PartialAssignment::new(), convert)
    }

    fn compile_with<T: 'static>(
        &self,
        root: u32,
        fixed: &PartialAssignment,
        convert: fn(Val) -> Result<T, EvalError>,
    ) -> Compiled<T> {
        let program = self.extract(root);
        for (symbol, value) in fixed.iter() {
            let record = self.record(symbol);
            if !symbol_value_matches(record.sort, value) {
                panic!("fixed expression binding has the wrong symbol sort");
            }
        }
        let captured = program
            .reads
            .iter()
            .filter_map(|symbol| fixed.get(*symbol).map(|value| (*symbol, value)))
            .collect::<Vec<_>>();
        let reads = program
            .reads
            .iter()
            .copied()
            .filter(|symbol| fixed.get(*symbol).is_none())
            .collect::<Vec<_>>();
        if reads
            .iter()
            .any(|symbol| matches!(self.record(*symbol).kind, SymbolKind::Decision(_)))
        {
            panic!("compiled evaluator retains an unfixed finite decision");
        }
        let retained_bytes = program
            .retained_metadata_bytes()
            .saturating_add(captured.capacity() * std::mem::size_of::<(SymbolId, SymbolValue)>())
            .saturating_add(captured.iter().fold(0usize, |bytes, (_, value)| bytes.saturating_add(
                value.retained_metadata_bytes().saturating_sub(std::mem::size_of::<SymbolValue>()))));
        Compiled::new(
            reads,
            retained_bytes,
            Box::new(move |values| {
                let value = Evaluator::new(&program, &|symbol| {
                    captured
                        .iter()
                        .find_map(|(fixed, value)| (*fixed == symbol).then(|| value.clone()))
                        .or_else(|| values.get(symbol))
                })
                .run(program.root)?;
                convert(value)
            }),
        )
    }
    pub(super) fn compile_nat(&self, n: NatExpr) -> Compiled<BigUint> {
        self.compile(self.expr_index(n), to_nat)
    }
    pub(super) fn compile_int(&self, n: IntExpr) -> Compiled<BigInt> {
        self.compile(self.expr_index(n), to_int)
    }
    pub(super) fn compile_bool(&self, n: BoolExpr) -> Compiled<bool> {
        self.compile(self.expr_index(n), to_bool)
    }
    pub(super) fn compile_decision_bool(&self, n: BoolExpr) -> CompiledDecisionPredicate {
        let program = self.extract(self.expr_index(n));
        CompiledDecisionPredicate::new(Box::new(move |values| {
            let value = Evaluator::new(&program, &|symbol| values.get(symbol)).run(program.root)?;
            to_bool(value)
        }))
    }
    pub(super) fn compile_duration(&self, n: DurationExpr) -> Compiled<DurationEstimate> {
        self.compile(self.expr_index(n), to_duration)
    }
    pub(super) fn compile_nat_with(&self, n: NatExpr, fixed: &PartialAssignment) -> Compiled<BigUint> {
        self.compile_with(self.expr_index(n), fixed, to_nat)
    }
    pub(super) fn compile_int_with(&self, n: IntExpr, fixed: &PartialAssignment) -> Compiled<BigInt> {
        self.compile_with(self.expr_index(n), fixed, to_int)
    }
    pub(super) fn compile_bool_with(
        &self,
        n: BoolExpr,
        fixed: &PartialAssignment,
    ) -> Compiled<bool> {
        self.compile_with(self.expr_index(n), fixed, to_bool)
    }
    pub(super) fn compile_duration_with(
        &self,
        n: DurationExpr,
        fixed: &PartialAssignment,
    ) -> Compiled<DurationEstimate> {
        self.compile_with(self.expr_index(n), fixed, to_duration)
    }
}

fn to_nat(v: Val) -> Result<BigUint, EvalError> {
    match v {
        Val::Int(i) => i.to_biguint().ok_or(EvalError::NegativeNat),
        _ => Err(EvalError::Unrepresentable),
    }
}
fn to_int(v: Val) -> Result<BigInt, EvalError> {
    match v {
        Val::Int(i) => Ok(i),
        _ => Err(EvalError::Unrepresentable),
    }
}
fn to_bool(v: Val) -> Result<bool, EvalError> {
    match v {
        Val::Bool(b) => Ok(b),
        _ => Err(EvalError::Unrepresentable),
    }
}
fn to_duration(v: Val) -> Result<DurationEstimate, EvalError> {
    match v {
        Val::Duration {
            lower,
            upper,
            denominator,
        } => Ok(DurationEstimate {
            lower: super::RationalDuration {
                numerator: lower,
                denominator,
            },
            upper: super::RationalDuration {
                numerator: upper,
                denominator,
            },
        }),
        _ => Err(EvalError::Unrepresentable),
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

struct Evaluator<'a> {
    program: &'a Program,
    lookup: &'a dyn Fn(SymbolId) -> Option<SymbolValue>,
    memo: Vec<Option<Val>>,
    /// Active fold bindings, innermost last.
    bound: Vec<(SymbolId, BigInt)>,
}

impl<'a> Evaluator<'a> {
    fn new(program: &'a Program, lookup: &'a dyn Fn(SymbolId) -> Option<SymbolValue>) -> Self {
        Self {
            program,
            lookup,
            memo: vec![None; program.nodes.len()],
            bound: Vec::new(),
        }
    }

    fn run(&mut self, i: u32) -> Result<Val, EvalError> {
        if let Some(v) = &self.memo[i as usize] {
            return Ok(v.clone());
        }
        let value = if matches!(
            self.program.nodes[i as usize],
            Node::Binary {
                op: BinaryOp::And | BinaryOp::Or | BinaryOp::Implies,
                ..
            }
        ) {
            Val::Bool(self.short_circuit(i)?)
        } else {
            self.compute(i)?
        };
        if self.program.memoizable[i as usize] {
            self.memo[i as usize] = Some(value.clone());
        }
        Ok(value)
    }

    /// Predicate conjunctions grow with program size. Evaluate their control
    /// flow on an explicit stack, retaining left-to-right definedness and
    /// short-circuiting without consuming one host frame per conjunct.
    fn short_circuit(&mut self, root: u32) -> Result<bool, EvalError> {
        enum Pending {
            Left { id: u32, op: BinaryOp, right: u32 },
            Right(u32),
        }
        let mut pending = Vec::new();
        let mut current = root;
        'evaluate: loop {
            let mut value = if let Some(value) = &self.memo[current as usize] {
                to_bool(value.clone())?
            } else if let Node::Binary {
                op: op @ (BinaryOp::And | BinaryOp::Or | BinaryOp::Implies),
                lhs: AnyExpr::Bool(left),
                rhs: AnyExpr::Bool(right),
            } = self.program.nodes[current as usize]
            {
                pending.push(Pending::Left {
                    id: current,
                    op,
                    right: right.index,
                });
                current = left.index;
                continue;
            } else {
                to_bool(self.run(current)?)?
            };
            while let Some(frame) = pending.pop() {
                let id = match frame {
                    Pending::Left { id, op, right } => {
                        let needs_right = match op {
                            BinaryOp::And | BinaryOp::Implies => value,
                            BinaryOp::Or => !value,
                            _ => unreachable!(),
                        };
                        if needs_right {
                            pending.push(Pending::Right(id));
                            current = right;
                            continue 'evaluate;
                        }
                        value = matches!(op, BinaryOp::Or | BinaryOp::Implies);
                        id
                    }
                    Pending::Right(id) => id,
                };
                if self.program.memoizable[id as usize] {
                    self.memo[id as usize] = Some(Val::Bool(value));
                }
            }
            return Ok(value);
        }
    }

    fn int(&mut self, e: AnyExpr) -> Result<BigInt, EvalError> {
        match self.run(ordinal(e))? {
            Val::Int(v) => Ok(v),
            _ => Err(EvalError::Unrepresentable),
        }
    }
    fn boolean(&mut self, e: BoolExpr) -> Result<bool, EvalError> {
        match self.run(e.index)? {
            Val::Bool(v) => Ok(v),
            _ => Err(EvalError::Unrepresentable),
        }
    }
    fn duration(&mut self, e: DurationExpr) -> Result<(u128, u128, u64), EvalError> {
        match self.run(e.index)? {
            Val::Duration {
                lower,
                upper,
                denominator,
            } => Ok((lower, upper, denominator)),
            _ => Err(EvalError::Unrepresentable),
        }
    }

    fn symbol(&self, s: SymbolId) -> Result<Val, EvalError> {
        if let Some((_, v)) = self.bound.iter().rev().find(|(b, _)| *b == s) {
            return Ok(Val::Int(v.clone()));
        }
        let sort = self.program.symbol_sorts.get(&s).copied();
        match (sort, (self.lookup)(s)) {
            (Some(SymbolSort::Nat), Some(SymbolValue::Nat(v))) => Ok(Val::Int(v.into())),
            (Some(SymbolSort::Int), Some(SymbolValue::Int(v))) => Ok(Val::Int(v.into())),
            (Some(SymbolSort::Scalar(dtype)), Some(value)) => match ScalarVal::from_value(value) {
                Some(scalar) if scalar.matches(dtype) => Ok(Val::Scalar(scalar)),
                _ => Err(EvalError::Unbound(s)),
            },
            _ => Err(EvalError::Unbound(s)),
        }
    }

    fn compute(&mut self, i: u32) -> Result<Val, EvalError> {
        let node = &self.program.nodes[i as usize];
        Ok(match node {
            Node::NatConst(v) => Val::Int((*v).into()),
            Node::IntConst(v) => Val::Int((*v).into()),
            Node::BoolConst(v) => Val::Bool(*v),
            Node::ScalarConst { dtype, bits } => Val::Scalar(ScalarVal::decode(*dtype, *bits)),
            Node::ScalarInteger {
                operation,
                operands,
            } => {
                let operation = *operation;
                let mut inputs = Vec::with_capacity(operands.len());
                for (dtype, value) in operands.iter() {
                    inputs.push(integer_word(*dtype, &self.int((*value).into())?));
                }
                let types = inputs.iter().map(|value| value.dtype()).collect::<Vec<_>>();
                let recipe = reference_math::scalar_recipe(operation, &types);
                let result =
                    reference_math::evaluate(&recipe, &inputs).map_err(EvalError::ScalarFailure)?;
                Val::Int(scalar_integer_value(result).into())
            }
            Node::Symbol(s) => self.symbol(*s)?,
            Node::Unary { op, operand } => match op {
                UnaryOp::Not => {
                    let AnyExpr::Bool(b) = *operand else {
                        return Err(EvalError::Unrepresentable);
                    };
                    Val::Bool(!self.boolean(b)?)
                }
                UnaryOp::NatFromInt => {
                    let v = self.int(*operand)?;
                    if v.is_negative() {
                        return Err(EvalError::NegativeNat);
                    }
                    Val::Int(v)
                }
                UnaryOp::IntFromNat => Val::Int(self.int(*operand)?),
                UnaryOp::ScalarIntegerDefined => Val::Bool(match self.run(ordinal(*operand)) {
                    Ok(Val::Int(_)) => true,
                    Err(EvalError::ScalarFailure(_)) => false,
                    Err(error) => return Err(error),
                    _ => unreachable!("scalar definedness refers to an integer recipe"),
                }),
                UnaryOp::IntFromScalar => match self.run(ordinal(*operand))? {
                    Val::Scalar(ScalarVal::I32(value)) => Val::Int(value.into()),
                    Val::Scalar(ScalarVal::U32(value)) => Val::Int(value.into()),
                    _ => unreachable!("integer injection is constructed only for I32/U32"),
                },
            },
            Node::Binary { op, lhs, rhs } => {
                let (op, lhs, rhs) = (*op, *lhs, *rhs);
                match op {
                    BinaryOp::And | BinaryOp::Or | BinaryOp::Implies => {
                        unreachable!("short-circuit control is evaluated by the explicit stack")
                    }
                    BinaryOp::Iff => {
                        let (AnyExpr::Bool(a), AnyExpr::Bool(b)) = (lhs, rhs) else {
                            return Err(EvalError::Unrepresentable);
                        };
                        Val::Bool(self.boolean(a)? == self.boolean(b)?)
                    }
                    _ => {
                        let nat = matches!(lhs, AnyExpr::Nat(_));
                        let a = self.int(lhs)?;
                        let b = self.int(rhs)?;
                        Val::Int(arith(op, nat, a, b)?)
                    }
                }
            }
            Node::Nary { op, operands } => match op {
                NaryOp::All => {
                    for o in operands.iter() {
                        let AnyExpr::Bool(t) = *o else {
                            return Err(EvalError::Unrepresentable);
                        };
                        if !self.boolean(t)? {
                            return Ok(Val::Bool(false));
                        }
                    }
                    Val::Bool(true)
                }
                NaryOp::Any => {
                    for o in operands.iter() {
                        let AnyExpr::Bool(t) = *o else {
                            return Err(EvalError::Unrepresentable);
                        };
                        if self.boolean(t)? {
                            return Ok(Val::Bool(true));
                        }
                    }
                    Val::Bool(false)
                }
                NaryOp::Product => {
                    let mut acc = BigInt::one();
                    for o in operands.iter() {
                        let f = self.int(*o)?;
                        acc *= f;
                    }
                    Val::Int(acc)
                }
                NaryOp::DurationAdd => {
                    let (mut lower, mut upper, denominator) = self.empty_duration()?;
                    for o in operands.iter() {
                        let AnyExpr::Duration(duration) = *o else {
                            return Err(EvalError::Unrepresentable);
                        };
                        let (term_lower, term_upper, term_denominator) = self.duration(duration)?;
                        if term_denominator != denominator {
                            return Err(EvalError::Unrepresentable);
                        }
                        lower = lower
                            .checked_add(term_lower)
                            .ok_or(EvalError::Unrepresentable)?;
                        upper = upper
                            .checked_add(term_upper)
                            .ok_or(EvalError::Unrepresentable)?;
                    }
                    Val::Duration {
                        lower,
                        upper,
                        denominator,
                    }
                }
            },
            Node::Select {
                cond,
                then,
                otherwise,
            } => {
                let (cond, then, otherwise) = (*cond, *then, *otherwise);
                let taken = if self.boolean(cond)? { then } else { otherwise };
                self.run(ordinal(taken))?
            }
            Node::Cmp { op, lhs, rhs } => {
                let (op, lhs, rhs) = (*op, *lhs, *rhs);
                match lhs {
                    AnyExpr::Scalar(_) => {
                        let (Val::Scalar(a), Val::Scalar(b)) =
                            (self.run(ordinal(lhs))?, self.run(ordinal(rhs))?)
                        else {
                            return Err(EvalError::Unrepresentable);
                        };
                        Val::Bool(a.compare(op, b))
                    }
                    _ => {
                        let a = self.int(lhs)?;
                        let b = self.int(rhs)?;
                        Val::Bool(compare(op, a, b))
                    }
                }
            }
            Node::In { operand, values } => {
                let v = self.int(*operand)?;
                Val::Bool(i64::try_from(v).is_ok_and(|v| values.binary_search(&v).is_ok()))
            }
            Node::Fold {
                op,
                binder,
                start,
                extent,
                body,
            } => {
                let (op, binder, start, extent, body) = (*op, *binder, *start, *extent, *body);
                let start = self.int(AnyExpr::Nat(start))?;
                let n = self.int(AnyExpr::Nat(extent))?;
                let symbols: Vec<SymbolId> = self
                    .program
                    .binder_symbols
                    .get(&binder)
                    .cloned()
                    .unwrap_or_default();
                let mut acc = match op {
                    FoldOp::Sum | FoldOp::Max => BigInt::zero(),
                    FoldOp::Product => BigInt::one(),
                };
                let mut k = BigInt::zero();
                while k < n {
                    let before = self.bound.len();
                    let value = &start + &k;
                    for s in &symbols {
                        self.bound.push((*s, value.clone()));
                    }
                    let term = self.int(AnyExpr::Nat(body));
                    self.bound.truncate(before);
                    let term = term?;
                    acc = match op {
                        FoldOp::Sum => acc + term,
                        FoldOp::Product => acc * term,
                        FoldOp::Max => acc.max(term),
                    };
                    k += 1;
                }
                Val::Int(acc)
            }
            Node::Duration(terms) => {
                let terms = terms.clone();
                let common = self
                    .program
                    .common_denominator
                    .ok_or(EvalError::Unrepresentable)?;
                let (mut lower, mut upper, denominator) = self.empty_duration()?;
                for term in terms.iter() {
                    if term.denominator == 0 {
                        return Err(EvalError::DivisionByZero);
                    }
                    if term.lower_numerator > term.upper_numerator {
                        return Err(EvalError::Unrepresentable);
                    }
                    let demand = self.int(AnyExpr::Nat(term.demand))?;
                    let demand = u128::try_from(demand).map_err(|_| EvalError::Unrepresentable)?;
                    let scale = u128::from(common / term.denominator);
                    lower = lower
                        .checked_add(
                            demand
                                .checked_mul(u128::from(term.lower_numerator))
                                .and_then(|value| value.checked_mul(scale))
                                .ok_or(EvalError::Unrepresentable)?,
                        )
                        .ok_or(EvalError::Unrepresentable)?;
                    upper = upper
                        .checked_add(
                            demand
                                .checked_mul(u128::from(term.upper_numerator))
                                .and_then(|value| value.checked_mul(scale))
                                .ok_or(EvalError::Unrepresentable)?,
                        )
                        .ok_or(EvalError::Unrepresentable)?;
                }
                Val::Duration {
                    lower,
                    upper,
                    denominator,
                }
            }
            Node::DurationScale { duration, by } => {
                let (duration, by) = (*duration, *by);
                let factor = self.int(AnyExpr::Nat(by))?;
                let factor = u128::try_from(factor).map_err(|_| EvalError::Unrepresentable)?;
                let (lower, upper, denominator) = self.duration(duration)?;
                Val::Duration {
                    lower: lower
                        .checked_mul(factor)
                        .ok_or(EvalError::Unrepresentable)?,
                    upper: upper
                        .checked_mul(factor)
                        .ok_or(EvalError::Unrepresentable)?,
                    denominator,
                }
            }
        })
    }

    fn empty_duration(&self) -> Result<(u128, u128, u64), EvalError> {
        let common = self
            .program
            .common_denominator
            .ok_or(EvalError::Unrepresentable)?;
        Ok((0, 0, common))
    }
}

fn integer_word(dtype: DType, value: &BigInt) -> ReferenceScalar {
    let bits = value.rem_euclid(&(BigInt::one() << 32_u32));
    let bits = u32::try_from(bits).expect("word residue is within u32");
    match dtype {
        DType::I32 => ReferenceScalar::I32(bits as i32),
        DType::U32 => ReferenceScalar::U32(bits),
        _ => unreachable!("integer projection has only integer word operands"),
    }
}
fn scalar_integer_value(value: ReferenceScalar) -> i64 {
    match value {
        ReferenceScalar::I32(value) => i64::from(value),
        ReferenceScalar::U32(value) => i64::from(value),
        _ => unreachable!("integer projection has only integer word results"),
    }
}

fn arith(op: BinaryOp, nat: bool, a: BigInt, b: BigInt) -> Result<BigInt, EvalError> {
    Ok(match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => {
            let r = a - b;
            if nat && r.is_negative() {
                return Err(EvalError::NegativeNat);
            }
            r
        }
        BinaryOp::Mul => a * b,
        BinaryOp::Div => {
            if b.is_zero() {
                return Err(EvalError::DivisionByZero);
            }
            a.div_euclid(&b)
        }
        BinaryOp::CeilDiv | BinaryOp::AlignUp => {
            if b.is_zero() {
                return Err(EvalError::DivisionByZero);
            }
            let (q, r) = a.div_rem_euclid(&b);
            let ceiling = if !r.is_zero() && b.is_positive() {
                q + 1
            } else {
                q
            };
            if matches!(op, BinaryOp::AlignUp) {
                ceiling * b
            } else {
                ceiling
            }
        }
        BinaryOp::Rem => {
            if b.is_zero() {
                return Err(EvalError::DivisionByZero);
            }
            a.rem_euclid(&b)
        }
        BinaryOp::Min => a.min(b),
        BinaryOp::Max => a.max(b),
        BinaryOp::And | BinaryOp::Or | BinaryOp::Implies | BinaryOp::Iff => {
            return Err(EvalError::Unrepresentable);
        }
    })
}
