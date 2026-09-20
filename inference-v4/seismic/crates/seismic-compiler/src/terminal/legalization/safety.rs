//! Safety obligation consumption.
//!
//! Logical construction creates obligations undischarged. Each physical
//! alternative consumes every one of them exactly once as `StaticallyProved`
//! (when the logical/runtime-extent math proves it) or `RuntimeChecked` (with
//! a planned predicate, inactive behavior, and a status write). GPU unsafe
//! accesses are predicated and record the first error in the planned status
//! slot; CPU writes the same status. Emitters add and omit no checks: the
//! discharge decided here is the whole truth.

use seismic_lang::{
    logical::{
        GraphRegion, GraphValueId, IdVec, LogicalNode, LogicalNodeKind, PrimitiveOp, RuntimeExtent,
        SafetyObligation, StateTokenId, TaskGraph,
    },
    sir::Literal,
    span::Span,
    types::{ExtentExpr, RuntimeExtentId, ValueType},
};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Discharge vocabulary
// ---------------------------------------------------------------------------

/// The kind of safety violation a status write reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SafetyKind {
    IndexOutOfBounds,
    RangeOutOfBounds,
    DivisionByZero,
    DivisionOverflow,
    ShiftOutOfRange,
    ShapeOverflow,
    EmptyReductionInput,
}

/// How one obligation was consumed.
#[derive(Clone, Debug, PartialEq)]
pub enum ObligationDischarge {
    /// The logical/runtime-extent math proves the obligation holds on every
    /// invocation; no check is planned and none may be emitted.
    StaticallyProved(StaticProof),
    /// A planned predicate guards the unsafe operation; when it is false the
    /// operation is inactive and the first error is recorded in the planned
    /// status slot.
    RuntimeChecked(RuntimeCheck),
    /// The obligation fails on every invocation (for example a static shape
    /// product that cannot fit); the containing alternative is inapplicable
    /// and no check can repair it.
    StaticallyImpossible(String),
}

/// Why an obligation is statically proved.
#[derive(Clone, Debug, PartialEq)]
pub enum StaticProof {
    /// The index value is `Index { bound }` with `bound <= extent`.
    IndexRefinement {
        bound: ExtentExpr,
        extent: ExtentExpr,
    },
    /// A constant operand satisfies the constraint (nonzero divisor, shift
    /// count in range, constant index inside a static extent).
    ConstantValue { value: i64 },
    /// Both range endpoints are proved within the extent.
    RangeEndpoints { extent: ExtentExpr },
    /// All factors are static and the checked product fits `bits`.
    StaticProductFits { product: u64, bits: u8 },
    /// Runtime factors are each bounded by their capacities and the checked
    /// capacity product fits `bits` (a runtime value never exceeds its
    /// capacity).
    CapacityBounded { capacity_product: u64, bits: u8 },
    /// A reduced axis is statically nonempty.
    NonEmpty { length: u64 },
}

impl StaticProof {
    /// The human-readable proof argument, carried by the realization layer's
    /// `ObligationDisposition::StaticallyProved { reason }`.
    pub fn reason(&self) -> String {
        match self {
            StaticProof::IndexRefinement { bound, extent } => {
                format!("the index is refined to {bound}, within {extent}")
            }
            StaticProof::ConstantValue { value } => {
                format!("the operand is the constant {value}, which satisfies the constraint")
            }
            StaticProof::RangeEndpoints { extent } => {
                format!("both range endpoints are proved within {extent}")
            }
            StaticProof::StaticProductFits { product, bits } => {
                format!("the checked static product {product} fits {bits} bits")
            }
            StaticProof::CapacityBounded {
                capacity_product,
                bits,
            } => format!(
                "every runtime factor is bounded by its capacity; the checked capacity product \
                 {capacity_product} fits {bits} bits"
            ),
            StaticProof::NonEmpty { length } => {
                format!("the reduced axis is statically nonempty ({length} element(s))")
            }
        }
    }
}

/// One planned runtime predicate.
#[derive(Clone, Debug, PartialEq)]
pub enum CheckPredicate {
    IndexInBounds {
        index: GraphValueId,
        extent: ExtentExpr,
    },
    RangeInBounds {
        start: GraphValueId,
        end: GraphValueId,
        extent: ExtentExpr,
    },
    DivisorNonZero {
        value: GraphValueId,
    },
    /// Nonzero divisor and no signed overflow (`lhs != i32::MIN || rhs != -1`).
    DivisionSafe {
        lhs: GraphValueId,
        rhs: GraphValueId,
    },
    ShiftInRange {
        value: GraphValueId,
    },
    /// The checked product of the factors fits `bits` at runtime.
    ProductFits {
        factors: Vec<ExtentExpr>,
        bits: u8,
    },
    /// A reduced axis contains at least one element.
    ExtentPositive {
        extent: ExtentExpr,
    },
}

/// What the guarded operation does when its predicate is false: it is not
/// performed. On GPU the operation is predicated; on CPU the same status is
/// written around a branch. Both record the first error only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InactiveBehavior {
    SkipOperation,
}

/// The planned status write of a failing check: kind plus source span,
/// reported by the runtime after synchronous completion.
#[derive(Clone, Debug, PartialEq)]
pub struct StatusWrite {
    pub kind: SafetyKind,
    pub span: Span,
}

/// A planned runtime check: predicate, inactive behavior, status write.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeCheck {
    pub predicate: CheckPredicate,
    pub inactive: InactiveBehavior,
    pub status: StatusWrite,
}

// ---------------------------------------------------------------------------
// Graph facts
// ---------------------------------------------------------------------------

/// The logical facts discharge proofs run over: the type (and index
/// refinement) of every graph value, the constants carried by constant nodes,
/// the state-token/storage relation, and the runtime extents.
#[derive(Clone, Debug, Default)]
pub struct GraphFacts {
    pub types: BTreeMap<GraphValueId, ValueType>,
    pub constants: BTreeMap<GraphValueId, i64>,
    pub states: BTreeMap<StateTokenId, seismic_lang::logical::LogicalStorageId>,
    pub runtime_extents: BTreeMap<RuntimeExtentId, RuntimeExtent>,
}

impl GraphFacts {
    /// Collect the facts of one task graph: every region-parameter and node
    /// origin is visited, nested regions included.
    pub fn collect(
        graph: &TaskGraph,
        runtime_extents: &IdVec<RuntimeExtentId, RuntimeExtent>,
    ) -> GraphFacts {
        let mut facts = GraphFacts {
            runtime_extents: runtime_extents
                .ids()
                .zip(runtime_extents.iter())
                .map(|(id, extent)| (id, extent.clone()))
                .collect(),
            ..GraphFacts::default()
        };
        for parameter in &graph.parameters {
            facts.region_parameter(parameter);
        }
        facts.region(&graph.root);
        facts
    }

    fn region_parameter(&mut self, parameter: &seismic_lang::logical::RegionParameter) {
        use seismic_lang::logical::RegionParameter;
        match parameter {
            RegionParameter::Value { id, ty } => {
                self.types.insert(*id, ty.clone());
            }
            RegionParameter::State { id, storage } => {
                self.states.insert(*id, *storage);
            }
        }
    }

    fn region(&mut self, region: &GraphRegion) {
        for parameter in &region.parameters {
            self.region_parameter(parameter);
        }
        for id in region.nodes.ids() {
            let node = &region.nodes[id];
            self.node(node);
        }
    }

    fn node(&mut self, node: &LogicalNode) {
        for output in &node.outputs {
            self.types.insert(output.id, output.ty.clone());
        }
        for token in &node.state_outputs {
            self.states.insert(token.id, token.storage);
        }
        if let LogicalNodeKind::Primitive(application) = &node.kind {
            if let PrimitiveOp::Constant(Literal::Int(value)) = &application.op {
                if let Some(output) = node.outputs.first() {
                    self.constants.insert(output.id, *value);
                }
            }
        }
        match &node.kind {
            LogicalNodeKind::If(if_node) => {
                self.region(&if_node.then_region);
                self.region(&if_node.else_region);
            }
            LogicalNodeKind::Loop(loop_node) => {
                self.region(&loop_node.body);
            }
            _ => {}
        }
    }

    fn capacity(&self, extent: &ExtentExpr) -> Option<u64> {
        match extent {
            ExtentExpr::Static(n) => Some(*n),
            ExtentExpr::Runtime(id) => self.runtime_extents.get(id).map(|e| e.capacity),
            ExtentExpr::Sym(sym) => sym.as_constant().and_then(|c| u64::try_from(c).ok()),
        }
    }
}

// ---------------------------------------------------------------------------
// Static comparison helpers
// ---------------------------------------------------------------------------

fn static_of(extent: &ExtentExpr) -> Option<u64> {
    extent.as_static()
}

/// `Some(true)` when `a <= b` is provable from static values; `Some(false)`
/// is never returned (an unprovable comparison is `None`).
fn statically_le(a: &ExtentExpr, b: &ExtentExpr) -> Option<bool> {
    let (a, b) = (static_of(a)?, static_of(b)?);
    Some(a <= b)
}

fn checked_product(factors: &[u64]) -> Option<u64> {
    factors
        .iter()
        .try_fold(1u64, |acc, factor| acc.checked_mul(*factor))
}

/// Whether a product fits unsigned `bits` bits (a product that overflows `u64`
/// never fits 64 bits or fewer).
fn fits_bits(product: u64, bits: u8) -> bool {
    if bits >= 64 {
        true
    } else {
        product < (1u64 << bits)
    }
}

// ---------------------------------------------------------------------------
// Discharge
// ---------------------------------------------------------------------------

/// Consume one safety obligation: prove it statically when the logical and
/// runtime-extent math suffices, otherwise plan the runtime check. `span` is
/// the source span of the node carrying the obligation.
pub fn discharge(
    obligation: &SafetyObligation,
    facts: &GraphFacts,
    span: Span,
) -> ObligationDischarge {
    match obligation {
        SafetyObligation::IndexInBounds { index, extent } => {
            // A constant index is proved inside a static extent.
            if let Some(value) = facts.constants.get(index) {
                return match static_of(extent) {
                    Some(n) if (0..n as i64).contains(value) => {
                        ObligationDischarge::StaticallyProved(StaticProof::ConstantValue {
                            value: *value,
                        })
                    }
                    Some(_) => ObligationDischarge::StaticallyImpossible(format!(
                        "constant index {value} is outside its static extent {extent}"
                    )),
                    None => ObligationDischarge::RuntimeChecked(RuntimeCheck {
                        predicate: CheckPredicate::IndexInBounds {
                            index: *index,
                            extent: extent.clone(),
                        },
                        inactive: InactiveBehavior::SkipOperation,
                        status: StatusWrite {
                            kind: SafetyKind::IndexOutOfBounds,
                            span,
                        },
                    }),
                };
            }
            // An index-typed value is refined to its bound.
            if let Some(ValueType::Index { bound }) = facts.types.get(index) {
                if statically_le(bound, extent) == Some(true) {
                    return ObligationDischarge::StaticallyProved(StaticProof::IndexRefinement {
                        bound: bound.clone(),
                        extent: extent.clone(),
                    });
                }
            }
            ObligationDischarge::RuntimeChecked(RuntimeCheck {
                predicate: CheckPredicate::IndexInBounds {
                    index: *index,
                    extent: extent.clone(),
                },
                inactive: InactiveBehavior::SkipOperation,
                status: StatusWrite {
                    kind: SafetyKind::IndexOutOfBounds,
                    span,
                },
            })
        }
        SafetyObligation::RangeInBounds { start, end, extent } => {
            let start_ok = refined_within(facts, start, extent);
            let end_ok = refined_within(facts, end, extent);
            if start_ok && end_ok {
                return ObligationDischarge::StaticallyProved(StaticProof::RangeEndpoints {
                    extent: extent.clone(),
                });
            }
            ObligationDischarge::RuntimeChecked(RuntimeCheck {
                predicate: CheckPredicate::RangeInBounds {
                    start: *start,
                    end: *end,
                    extent: extent.clone(),
                },
                inactive: InactiveBehavior::SkipOperation,
                status: StatusWrite {
                    kind: SafetyKind::RangeOutOfBounds,
                    span,
                },
            })
        }
        SafetyObligation::DivisorNonZero { value } => {
            if let Some(constant) = facts.constants.get(value) {
                return if *constant == 0 {
                    ObligationDischarge::StaticallyImpossible(
                        "division by the constant zero".into(),
                    )
                } else {
                    ObligationDischarge::StaticallyProved(StaticProof::ConstantValue {
                        value: *constant,
                    })
                };
            }
            ObligationDischarge::RuntimeChecked(RuntimeCheck {
                predicate: CheckPredicate::DivisorNonZero { value: *value },
                inactive: InactiveBehavior::SkipOperation,
                status: StatusWrite {
                    kind: SafetyKind::DivisionByZero,
                    span,
                },
            })
        }
        SafetyObligation::SignedDivisionNoOverflow { lhs, rhs } => {
            // A constant divisor other than -1 and 0 never overflows.
            if let Some(divisor) = facts.constants.get(rhs) {
                if *divisor != -1 && *divisor != 0 {
                    return ObligationDischarge::StaticallyProved(StaticProof::ConstantValue {
                        value: *divisor,
                    });
                }
                // Divisor -1 overflows only for i32::MIN; a constant dividend
                // settles it. Divisor 0 is the divisor-nonzero obligation.
                if let Some(dividend) = facts.constants.get(lhs) {
                    if *divisor == -1 && *dividend != i32::MIN as i64 {
                        return ObligationDischarge::StaticallyProved(StaticProof::ConstantValue {
                            value: *dividend,
                        });
                    }
                }
            }
            ObligationDischarge::RuntimeChecked(RuntimeCheck {
                predicate: CheckPredicate::DivisionSafe {
                    lhs: *lhs,
                    rhs: *rhs,
                },
                inactive: InactiveBehavior::SkipOperation,
                status: StatusWrite {
                    kind: SafetyKind::DivisionOverflow,
                    span,
                },
            })
        }
        SafetyObligation::ShiftInRange { value } => {
            if let Some(constant) = facts.constants.get(value) {
                return if (0..32).contains(constant) {
                    ObligationDischarge::StaticallyProved(StaticProof::ConstantValue {
                        value: *constant,
                    })
                } else {
                    ObligationDischarge::StaticallyImpossible(format!(
                        "constant shift count {constant} lies outside 0..32"
                    ))
                };
            }
            ObligationDischarge::RuntimeChecked(RuntimeCheck {
                predicate: CheckPredicate::ShiftInRange { value: *value },
                inactive: InactiveBehavior::SkipOperation,
                status: StatusWrite {
                    kind: SafetyKind::ShiftOutOfRange,
                    span,
                },
            })
        }
        SafetyObligation::ShapeProductFits { factors, bits } => {
            // All-static factors: the checked product decides.
            let statics: Option<Vec<u64>> =
                factors.iter().map(|factor| static_of(factor)).collect();
            if let Some(values) = statics {
                return match checked_product(&values) {
                    Some(product) if fits_bits(product, *bits) => {
                        ObligationDischarge::StaticallyProved(StaticProof::StaticProductFits {
                            product,
                            bits: *bits,
                        })
                    }
                    _ => ObligationDischarge::StaticallyImpossible(format!(
                        "static shape product {factors:?} does not fit in {bits} bits"
                    )),
                };
            }
            // Runtime factors: each value is bounded by its capacity, so a
            // capacity product that fits proves the obligation.
            let capacities: Option<Vec<u64>> = factors
                .iter()
                .map(|factor| facts.capacity(factor))
                .collect();
            if let Some(values) = capacities {
                if let Some(product) = checked_product(&values) {
                    if fits_bits(product, *bits) {
                        return ObligationDischarge::StaticallyProved(
                            StaticProof::CapacityBounded {
                                capacity_product: product,
                                bits: *bits,
                            },
                        );
                    }
                }
                // A capacity product that cannot fit leaves nothing to check:
                // values within capacity still can overflow.
            }
            ObligationDischarge::RuntimeChecked(RuntimeCheck {
                predicate: CheckPredicate::ProductFits {
                    factors: factors.clone(),
                    bits: *bits,
                },
                inactive: InactiveBehavior::SkipOperation,
                status: StatusWrite {
                    kind: SafetyKind::ShapeOverflow,
                    span,
                },
            })
        }
    }
}

/// Whether one value is proved to lie within `extent`: an index refinement
/// with `bound <= extent`, or a constant inside a static extent.
fn refined_within(facts: &GraphFacts, value: &GraphValueId, extent: &ExtentExpr) -> bool {
    if let Some(constant) = facts.constants.get(value) {
        if let Some(n) = static_of(extent) {
            return (0..n as i64).contains(constant);
        }
        return false;
    }
    if let Some(ValueType::Index { bound }) = facts.types.get(value) {
        return statically_le(bound, extent) == Some(true);
    }
    false
}

/// Discharge one reduction precondition (`argmax`/`max`/`min` nonempty input).
pub fn discharge_precondition(
    precondition: &crate::terminal::reduction::ReductionPrecondition,
    facts: &GraphFacts,
    span: Span,
) -> ObligationDischarge {
    use crate::terminal::reduction::ReductionPrecondition;
    match precondition {
        ReductionPrecondition::NonEmpty { length, .. } => {
            if let Some(length) = static_of(length) {
                return if length > 0 {
                    ObligationDischarge::StaticallyProved(StaticProof::NonEmpty { length })
                } else {
                    ObligationDischarge::StaticallyImpossible(
                        "the reduced axis is statically empty".into(),
                    )
                };
            }
            // A runtime length is at most its capacity but may be zero: the
            // nonempty requirement needs a planned predicate. A zero capacity
            // bounds the value to zero, which is statically empty.
            if facts.capacity(length) == Some(0) {
                return ObligationDischarge::StaticallyImpossible(
                    "the reduced axis is bounded by a zero capacity".into(),
                );
            }
            ObligationDischarge::RuntimeChecked(RuntimeCheck {
                predicate: CheckPredicate::ExtentPositive {
                    extent: length.clone(),
                },
                inactive: InactiveBehavior::SkipOperation,
                status: StatusWrite {
                    kind: SafetyKind::EmptyReductionInput,
                    span,
                },
            })
        }
    }
}
