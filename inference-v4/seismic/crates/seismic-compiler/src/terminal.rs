//! Physical primitive formation and common legalization/numerics.
//!
//! This module forms physical
//! primitives from logical `PrimitiveApplication`s against the intrinsic
//! registry, classifies every safety obligation as statically proved or
//! runtime checked, declares the reduction strategy vocabulary and the
//! universal portable strategies, and composes numerical transfers.
//! Reductions are consumed by reduction strategies
//! (`map_reduction`), loops/conditionals/calls by the structured schedule
//! builders (`schedule_loop`/`schedule_if`/`invoke`), and aggregates are
//! leaf-lowered or explicitly stored — they never reach scalar-only emitters.
//!
//! The formation API here is written against the logical layer
//! (`seismic-lang`) and the portable vocabulary this module owns. The
//! consuming family builder (`map_primitive`/`map_reduction`/`fuse`/`split`/
//! `schedule_if`/`schedule_loop`/`invoke`/`discharge`/`complete_result`/
//! `finish_alternative`) and `PhysicalPrimitive` live in the realization
//! layer; the objects produced here are exactly its inputs.

pub mod legalization;
pub mod numerics;
pub mod reduction;

#[cfg(test)]
mod tests;

pub use legalization::{
    discharge, discharge_precondition, CheckPredicate, GraphFacts, InactiveBehavior,
    ObligationDischarge, RuntimeCheck, SafetyKind, StaticProof, StatusWrite,
};
pub use legalization::{
    registry_math_is_versioned, universal_form, universal_numerical, DataAccess, LayoutTransform,
    LegalizationBug, LinearLoopOp, SeismicMathReference, UniversalForm, UniversalLegalization,
    SEISMIC_MATH, SEISMIC_MATH_IDENTITY, SEISMIC_MATH_VERSION,
};
pub use numerics::{
    compose, compose_all, default_tolerance, satisfies_policy, unit_roundoff,
    AssignmentFingerprint, CapabilitySignatureId, CountExpr, EvidenceKey, NumericalEvidence,
    NumericalTransfer, PolicyDecision, ToolchainId, WorkloadFingerprint,
};
pub use reduction::{
    reassociable, reduction_identity, universal as universal_reduction_strategy,
    ReassociationAdmission, ReductionAdmissionError, ReductionIdentity, ReductionPrecondition,
    ReductionResources, ReductionStrategy, ReductionStrategyKind, ReductionTopology, TieRule,
};

use seismic_lang::{
    intrinsics::{atomic_dtype, IntrinsicId},
    logical::{
        AtomicOperation, BoundaryInputKind, BoundaryResultKind, CarriedSlot, ChoiceId,
        GraphValueId, JoinSlot, LogicalNode, LogicalNodeKind, LogicalRange, LogicalStorageId,
        LoopNode, NodeId, PrimitiveOp, ReductionNode, SafetyObligation, StateJoin,
    },
    span::Span,
    types::{canonical_leaves, DType, Elem, ExtentExpr, Leaf, TensorType, ValueType},
};

// ---------------------------------------------------------------------------
// Universal arbitrary-rank iteration
// ---------------------------------------------------------------------------

/// The one common linear/runtime iteration geometry is hosted in
/// `seismic-realization::dispatch`; this module re-exports it. The universal
/// map retains extents, keeps an overflow-checked row-major total (the exact
/// runtime product for runtime domains, with the checked capacity bound),
/// carries the physical linear participant count (`participants`, a planning
/// expression defaulting to one), one-pass or grid-stride traversal,
/// delinearization for every logical axis, and a tail mask. Logical rank is
/// not native grid rank. Ordered axes are ascending serial loops inside each
/// independent point; zero work is a retained launch condition skipped by the
/// runtime, and zero native grids are never submitted.
pub use seismic_realization::dispatch::{
    LaunchCondition, LinearIterationMap, LinearMapError, LinearTotal, Traversal,
};

// ---------------------------------------------------------------------------
// Consequences of universal forms
// ---------------------------------------------------------------------------

/// Live SSA values and emitted statements per universal kernel are bounded so
/// the native contract guarantees at least one resident participant: the
/// universal mapping splits kernels at this threshold instead of emitting an
/// unbounded kernel.
pub const UNIVERSAL_MAX_LIVE_SSA: u32 = 4096;

/// Consequences of one universal primitive mapping: no private/workgroup
/// bytes, `direct_bindings` direct leaf bindings (a descriptor/argument table
/// is the indirect-binding alternative when the target's direct-binding limit is
/// lower), device bytes for cross-launch materialized tensors, bounded code
/// shape, no required capability, and `Exact` numerics. The native contract's
/// admissible domain admits any reflected maximum resident participant count
/// of at least one, so the resolved geometry `min(preferred, native_max)`
/// always guarantees a resident participant.
pub fn universal_consequences(
    direct_bindings: u32,
    device_bytes: u64,
) -> seismic_realization::executable::PhysicalConsequences {
    use seismic_realization::executable::{
        CostEstimate, HardResources, NativeResourceContract, PhysicalConsequences,
    };
    PhysicalConsequences {
        hard: HardResources {
            explicit_private_bytes: 0,
            explicit_workgroup_bytes: 0,
            explicit_device_bytes: device_bytes,
            direct_bindings,
            static_code_units: u64::from(UNIVERSAL_MAX_LIVE_SSA),
            ..HardResources::default()
        },
        native_contract: NativeResourceContract {
            max_resident_participants: (1, u64::MAX),
            native_subgroup_width: None,
        },
        cost: CostEstimate(0),
        numerical: NumericalTransfer::Exact,
        capability: None,
    }
}

/// Consequences of one capability-routed formation: the exact required
/// capability signature (planning kills the alternative against
/// `EffectiveTargetProfile::effective_signatures` using exactly this field),
/// an unqualified capability transfer pending the signature's own bound, and
/// no other hard resources.
pub fn capability_required_consequences(
    intrinsic: IntrinsicId,
    arguments: Vec<ValueType>,
) -> seismic_realization::executable::PhysicalConsequences {
    use seismic_realization::executable::{
        CostEstimate, HardResources, NativeResourceContract, PhysicalConsequences,
    };
    PhysicalConsequences {
        hard: HardResources::default(),
        native_contract: NativeResourceContract {
            max_resident_participants: (1, u64::MAX),
            native_subgroup_width: None,
        },
        cost: CostEstimate(0),
        numerical: NumericalTransfer::Capability {
            signature: CapabilitySignatureId::new(intrinsic.clone(), arguments),
            bound: None,
        },
        capability: Some(intrinsic),
    }
}

// ---------------------------------------------------------------------------
// Physical primitive formation
// ---------------------------------------------------------------------------

/// One canonical leaf of an operand or result, as formed for typed SSA or
/// storage binding. Aggregates are leaf-lowered through the one canonical
/// traversal; a tensor is one semantic leaf (one or more representation
/// planes).
#[derive(Clone, Debug, PartialEq)]
pub enum LeafDType {
    Scalar(DType),
    Index(ExtentExpr),
    Range(ExtentExpr),
    TensorPlane(TensorType),
}

impl LeafDType {
    fn of(ty: &ValueType) -> Result<Vec<(seismic_lang::types::ValuePath, LeafDType)>, String> {
        let leaves = canonical_leaves(ty)?;
        Ok(leaves
            .into_iter()
            .map(|(path, leaf)| {
                let dtype = match leaf {
                    Leaf::Scalar(d) => LeafDType::Scalar(d),
                    Leaf::Index(bound) => LeafDType::Index(bound.clone()),
                    Leaf::Range(bound) => LeafDType::Range(bound.clone()),
                    Leaf::Tensor(s) => LeafDType::TensorPlane(s.clone()),
                };
                (path, dtype)
            })
            .collect())
    }
}

/// One formed operand: its graph value, canonical type, and leaf-lowered
/// slots.
#[derive(Clone, Debug, PartialEq)]
pub struct FormedOperand {
    pub value: GraphValueId,
    pub ty: ValueType,
    pub leaves: Vec<(seismic_lang::types::ValuePath, LeafDType)>,
}

/// One logical primitive application formed as a physical primitive: the
/// exact universal form, operand/result leaf slots, the elementwise iteration
/// domain, discharged obligations, and the `Exact` universal transfer. This
/// is the input of the realization layer's `map_primitive` legalization.
#[derive(Clone, Debug, PartialEq)]
pub struct FormedPrimitive {
    pub node: NodeId,
    pub op: PrimitiveOp,
    pub operands: Vec<FormedOperand>,
    /// The canonical result types, one per node output (real types, never
    /// empty: dialect legalization classifies on them).
    pub results: Vec<ValueType>,
    /// Leaf-lowered result slots through the canonical traversal.
    pub result_leaves: Vec<(seismic_lang::types::ValuePath, LeafDType)>,
    /// The universal physical form (16.1 row).
    pub form: UniversalForm,
    /// The elementwise/aggregate iteration domain, when this primitive
    /// computes over a shape.
    pub iteration: Option<LinearIterationMap>,
    /// Every obligation of the node, consumed exactly once.
    pub obligations: Vec<(SafetyObligation, ObligationDischarge)>,
    /// The universal column is exact relative to the registry reference.
    pub numerical: NumericalTransfer,
    pub span: Span,
}

impl FormedPrimitive {
    /// The exact physical primitive offered to the dialect for legalization:
    /// the registry operation plus the real canonical operand/result types.
    pub fn physical_primitive(&self) -> seismic_realization::executable::PhysicalPrimitive {
        seismic_realization::executable::PhysicalPrimitive {
            op: self.op.clone(),
            inputs: self
                .operands
                .iter()
                .map(|operand| operand.ty.clone())
                .collect(),
            results: self.results.clone(),
        }
    }

    /// The iteration domain of this mapping, or the single-visit serial map
    /// when the primitive computes inside its containing structure.
    pub fn iteration_or_serial(&self) -> LinearIterationMap {
        self.iteration
            .clone()
            .unwrap_or_else(LinearIterationMap::serial)
    }
}

/// Why formation failed. Reductions, loops, conditionals, and calls are not
/// primitives: they are routed to the corresponding family-builder API.
#[derive(Clone, Debug, PartialEq)]
pub enum FormationError {
    /// The node is consumed by a structured builder, not `map_primitive`.
    NotAPrimitive { route: &'static str },
    /// A capability application: the alternative is legal only through the
    /// exact effective capability signature (or the portable reference body).
    RequiresCapability { intrinsic: IntrinsicId },
    /// The elementwise domain's checked total overflows: the alternative is
    /// infeasible.
    InfeasibleGeometry(LinearMapError),
    /// The alternative is inapplicable for this portable reason (for example
    /// a capability value crossing a call boundary).
    Inapplicable(String),
    /// A closed-registry violation that cannot occur for a checked program.
    Bug(String),
}

impl std::fmt::Display for FormationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormationError::NotAPrimitive { route } => {
                write!(f, "this node is consumed by `{route}`, not map_primitive")
            }
            FormationError::RequiresCapability { intrinsic } => write!(
                f,
                "capability `{}` has no portable physical opcode; the exact effective signature \
                 or the portable reference body must supply this alternative",
                intrinsic.path()
            ),
            FormationError::InfeasibleGeometry(error) => write!(f, "infeasible geometry: {error}"),
            FormationError::Inapplicable(reason) => write!(f, "inapplicable: {reason}"),
            FormationError::Bug(reason) => write!(f, "compiler bug in formation: {reason}"),
        }
    }
}
impl std::error::Error for FormationError {}

/// The elementwise iteration domain of one formed primitive: the shape of the
/// first tensor among operands and results. Address expressions, checked
/// accesses, allocations, and atomics compute inside the containing
/// iteration, not their own.
fn iteration_domain(
    form: &UniversalForm,
    operands: &[FormedOperand],
    results: &[ValueType],
    result_leaves: &[(seismic_lang::types::ValuePath, LeafDType)],
) -> Option<Vec<ExtentExpr>> {
    match form {
        UniversalForm::LayoutAddress { .. }
        | UniversalForm::CheckedAccess { .. }
        | UniversalForm::StorageAllocation
        | UniversalForm::SerializedAtomic { .. }
        | UniversalForm::PackedPlaneRead { .. } => None,
        _ => operands
            .iter()
            .find_map(|operand| operand.ty.shaped().map(|s| s.axes.clone()))
            .or_else(|| {
                results
                    .iter()
                    .find_map(|ty| ty.shaped().map(|s| s.axes.clone()))
            })
            .or_else(|| {
                result_leaves.iter().find_map(|(_, leaf)| match leaf {
                    LeafDType::TensorPlane(s) => Some(s.axes.clone()),
                    _ => None,
                })
            }),
    }
}

/// Form one logical primitive node as a physical primitive. The node kind is
/// matched exhaustively: reductions, loops, conditionals, and calls are
/// routed to their builders and never formed as scalar primitives.
pub fn form_primitive(
    node_id: NodeId,
    node: &LogicalNode,
    facts: &GraphFacts,
) -> Result<FormedPrimitive, FormationError> {
    let application = match &node.kind {
        LogicalNodeKind::Primitive(application) => application,
        LogicalNodeKind::Reduction(_) => {
            return Err(FormationError::NotAPrimitive {
                route: "map_reduction",
            });
        }
        LogicalNodeKind::Loop(_) => {
            return Err(FormationError::NotAPrimitive {
                route: "schedule_loop",
            });
        }
        LogicalNodeKind::If(_) => {
            return Err(FormationError::NotAPrimitive {
                route: "schedule_if",
            });
        }
        LogicalNodeKind::Call(_) => return Err(FormationError::NotAPrimitive { route: "invoke" }),
    };
    let operands =
        node.inputs
            .iter()
            .map(|value| {
                let ty =
                    facts.types.get(value).cloned().ok_or_else(|| {
                        FormationError::Bug(format!("operand {value:?} has no type"))
                    })?;
                let leaves = LeafDType::of(&ty).map_err(FormationError::Inapplicable)?;
                Ok(FormedOperand {
                    value: *value,
                    ty,
                    leaves,
                })
            })
            .collect::<Result<Vec<_>, FormationError>>()?;
    let results: Vec<ValueType> = node
        .outputs
        .iter()
        .map(|output| output.ty.clone())
        .collect();
    let result_leaves: Vec<(seismic_lang::types::ValuePath, LeafDType)> = node
        .outputs
        .iter()
        .map(|output| Ok(LeafDType::of(&output.ty).map_err(FormationError::Inapplicable)?))
        .collect::<Result<Vec<_>, FormationError>>()?
        .into_iter()
        .flatten()
        .collect();
    let form = match legalization::universal_form(
        &application.op,
        &operands
            .iter()
            .map(|operand| operand.ty.clone())
            .collect::<Vec<_>>(),
    )
    .map_err(|bug| FormationError::Bug(bug.0))?
    {
        UniversalLegalization::Form(form) => form,
        UniversalLegalization::RequiresCapability { intrinsic } => {
            return Err(FormationError::RequiresCapability { intrinsic });
        }
    };
    let iteration = iteration_domain(&form, &operands, &results, &result_leaves)
        .map(|axes| {
            LinearIterationMap::linear(&axes, &facts.runtime_extents)
                .map_err(FormationError::InfeasibleGeometry)
        })
        .transpose()?;
    let obligations = node
        .safety
        .iter()
        .map(|obligation| (obligation.clone(), discharge(obligation, facts, node.span)))
        .collect();
    Ok(FormedPrimitive {
        node: node_id,
        op: application.op.clone(),
        operands,
        results,
        result_leaves,
        form,
        iteration,
        obligations,
        numerical: NumericalTransfer::Exact,
        span: node.span,
    })
}

// ---------------------------------------------------------------------------
// Universal node constructors (inputs of the family builder)
// ---------------------------------------------------------------------------

/// The universal physical description of one reduction occurrence: its exact
/// strategy plus discharged preconditions. Consumed by `map_reduction`.
#[derive(Clone, Debug)]
pub struct UniversalReduction {
    pub operand: GraphValueId,
    pub axis: usize,
    pub strategy: ReductionStrategy,
    pub preconditions: Vec<(ReductionPrecondition, ObligationDischarge)>,
}

/// The universal physical description of one loop occurrence. Consumed by
/// `schedule_loop`.
#[derive(Clone, Debug)]
pub struct UniversalLoop {
    pub kind: seismic_lang::sir::LoopKind,
    pub range: LogicalRange,
    /// Independent loops: the universal linear map over the iteration domain.
    /// Ordered loops have no participant domain (ascending serial repeat).
    pub iteration: Option<LinearIterationMap>,
    /// An independent loop whose visits update storage atomically is
    /// serialized by the universal atomic-add strategy: exactly one
    /// participant traverses the domain, performing the exact
    /// load/add/round/store.
    pub serialized_for_atomic: bool,
    /// Ordered loops carry every changed captured value.
    pub carries: Vec<CarriedSlot>,
    /// Cross-visit state joins of an independent loop.
    pub joins: Vec<(LogicalStorageId, StateJoin)>,
}

/// The universal physical description of one conditional occurrence.
/// Consumed by `schedule_if`. Aggregate joins are leaf-lowered or
/// materialized; the condition is one retained predicate.
#[derive(Clone, Debug)]
pub struct UniversalIf {
    pub condition: GraphValueId,
    pub joins: Vec<JoinSlot>,
}

/// One canonical leaf of a call boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryLeafSlot {
    pub path: seismic_lang::types::ValuePath,
    pub kind: BoundarySlotKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BoundarySlotKind {
    Value(GraphValueId),
    Storage(LogicalStorageId),
    State(LogicalStorageId),
}

/// The universal physical description of one call occurrence: a synchronous
/// nested plan whose boundary leaves reference caller transports directly.
/// Consumed by `invoke`.
#[derive(Clone, Debug)]
pub struct UniversalCall {
    pub choice: ChoiceId,
    pub synchronous: bool,
    pub inputs: Vec<BoundaryLeafSlot>,
    pub results: Vec<BoundaryLeafSlot>,
}

/// The universal physical description of any logical node: exactly one
/// variant per node kind, exhaustive.
#[derive(Clone, Debug)]
pub enum UniversalNode {
    Primitive(FormedPrimitive),
    Reduction(UniversalReduction),
    Loop(UniversalLoop),
    If(UniversalIf),
    Call(UniversalCall),
}

/// Form the universal physical description of one node of a task graph.
/// Primitive applications become physical primitives; reductions, loops,
/// conditionals, and calls become the descriptions their family-builder APIs
/// consume.
pub fn universal_node(
    node_id: NodeId,
    node: &LogicalNode,
    facts: &GraphFacts,
) -> Result<UniversalNode, FormationError> {
    match &node.kind {
        LogicalNodeKind::Primitive(_) => Ok(UniversalNode::Primitive(form_primitive(
            node_id, node, facts,
        )?)),
        LogicalNodeKind::Reduction(reduction) => {
            universal_reduction(node, reduction, facts).map(UniversalNode::Reduction)
        }
        LogicalNodeKind::Loop(loop_node) => {
            universal_loop(node, loop_node, facts).map(UniversalNode::Loop)
        }
        LogicalNodeKind::If(if_node) => Ok(UniversalNode::If(UniversalIf {
            condition: if_node.condition,
            joins: if_node.joins.clone(),
        })),
        LogicalNodeKind::Call(call) => universal_call(call, facts).map(UniversalNode::Call),
    }
}

fn universal_reduction(
    node: &LogicalNode,
    reduction: &ReductionNode,
    facts: &GraphFacts,
) -> Result<UniversalReduction, FormationError> {
    let operand_type = facts
        .types
        .get(&reduction.operand)
        .cloned()
        .ok_or_else(|| FormationError::Bug("a reduction operand has no type".into()))?;
    let shaped = operand_type
        .shaped()
        .ok_or_else(|| FormationError::Bug("a reduction operand is not a tensor".into()))?
        .clone();
    if reduction.axis >= shaped.rank() {
        return Err(FormationError::Bug(format!(
            "reduction axis {} is outside the operand rank {}",
            reduction.axis,
            shaped.rank()
        )));
    }
    if let Elem::Repr(repr) = &shaped.elem {
        return Err(FormationError::Bug(format!(
            "a packed representation `{repr}` cannot be reduced"
        )));
    }
    let strategy =
        reduction::universal(reduction, &shaped).map_err(|reason| FormationError::Bug(reason))?;
    let preconditions = strategy
        .preconditions
        .iter()
        .cloned()
        .map(|precondition| {
            let discharge = discharge_precondition(&precondition, facts, node.span);
            (precondition, discharge)
        })
        .collect();
    Ok(UniversalReduction {
        operand: reduction.operand,
        axis: reduction.axis,
        strategy,
        preconditions,
    })
}

fn universal_loop(
    node: &LogicalNode,
    loop_node: &LoopNode,
    facts: &GraphFacts,
) -> Result<UniversalLoop, FormationError> {
    let joins: Vec<(LogicalStorageId, StateJoin)> = node
        .state_outputs
        .iter()
        .filter_map(|token| token.join.clone().map(|join| (token.storage, join)))
        .collect();
    // The universal atomic strategy serializes the containing independent
    // domain; every admitted atomic operation must be registry-legal.
    let serialized_for_atomic = joins
        .iter()
        .any(|(_, join)| matches!(join, StateJoin::Atomic { .. }));
    if serialized_for_atomic {
        for (_, join) in &joins {
            if let StateJoin::Atomic { operations } = join {
                for AtomicOperation { op, dtype, .. } in operations {
                    if !atomic_dtype(*dtype) {
                        return Err(FormationError::Bug(format!(
                            "atomic {} is undefined for {}",
                            op.name(),
                            dtype.name()
                        )));
                    }
                }
            }
        }
    }
    let iteration = match loop_node.kind {
        seismic_lang::sir::LoopKind::Ordered => None,
        seismic_lang::sir::LoopKind::Independent => {
            let map = LinearIterationMap::linear(
                &[loop_node.range.bound.clone()],
                &facts.runtime_extents,
            )
            .map_err(FormationError::InfeasibleGeometry)?;
            if serialized_for_atomic {
                Some(LinearIterationMap::serialized(&map))
            } else {
                Some(map)
            }
        }
    };
    Ok(UniversalLoop {
        kind: loop_node.kind,
        range: loop_node.range.clone(),
        iteration,
        serialized_for_atomic,
        carries: loop_node.carried.clone(),
        joins,
    })
}

fn universal_call(
    call: &seismic_lang::logical::CallNode,
    facts: &GraphFacts,
) -> Result<UniversalCall, FormationError> {
    let mut inputs = Vec::new();
    for input in &call.boundary_inputs {
        let kind = match &input.kind {
            BoundaryInputKind::Value(value) => {
                let ty = facts.types.get(value).ok_or_else(|| {
                    FormationError::Bug("a call boundary value has no type".into())
                })?;
                if matches!(ty, ValueType::CapabilityValue(_)) {
                    // A capability value crosses only a same-launch fused
                    // boundary; the universal call alternative is not fused.
                    return Err(FormationError::Inapplicable(format!(
                        "a capability value cannot cross the call boundary at {}",
                        input.path
                    )));
                }
                BoundarySlotKind::Value(*value)
            }
            BoundaryInputKind::Shared { state, .. }
            | BoundaryInputKind::Exclusive { state, .. }
            | BoundaryInputKind::Move { state, .. } => {
                let storage = facts.states.get(state).copied().ok_or_else(|| {
                    FormationError::Bug("a call boundary state token has no storage".into())
                })?;
                BoundarySlotKind::State(storage)
            }
        };
        inputs.push(BoundaryLeafSlot {
            path: input.path.clone(),
            kind,
        });
    }
    let mut results = Vec::new();
    for result in &call.boundary_results {
        let kind = match &result.kind {
            BoundaryResultKind::Value(value) => BoundarySlotKind::Value(*value),
            BoundaryResultKind::Storage { storage, .. } => BoundarySlotKind::Storage(*storage),
            BoundaryResultKind::State(token) => {
                let storage = facts.states.get(token).copied().ok_or_else(|| {
                    FormationError::Bug("a call result state token has no storage".into())
                })?;
                BoundarySlotKind::State(storage)
            }
        };
        results.push(BoundaryLeafSlot {
            path: result.path.clone(),
            kind,
        });
    }
    Ok(UniversalCall {
        choice: call.choice,
        synchronous: true,
        inputs,
        results,
    })
}

// ---------------------------------------------------------------------------
// Formation → builder wiring (composition with the realization layer)
// ---------------------------------------------------------------------------

use seismic_realization::executable::{
    AlternativeBuilder, BuilderError, DispositionReceipt, EffectiveTargetProfile,
    ExecutableDialect, InactiveBehavior as BuilderInactive, Legalized, NodeRef,
    ObligationDisposition, ObligationRef,
};

/// Legalize one formed primitive against a dialect and consume it with the
/// family builder's `map_primitive` transition. The physical primitive
/// carries the real canonical operand/result types from the logical node;
/// the iteration domain is the formed elementwise map, or the single-visit
/// serial map when the primitive computes inside its containing structure.
///
/// A dialect `Inapplicable` of a capability-routed formation is the
/// capability route (the alternative lives only behind the exact effective
/// signature); a universal primitive legalizing as `Inapplicable` is a
/// compiler bug.
pub fn legalize_and_map_primitive<D: ExecutableDialect>(
    builder: &mut AlternativeBuilder<D>,
    target: &EffectiveTargetProfile,
    node: NodeRef,
    formed: &FormedPrimitive,
) -> Result<(), BuilderError> {
    let legalized = D::legalize(&formed.physical_primitive(), target);
    if legalized.ops().is_none() {
        return Err(match &formed.op {
            PrimitiveOp::Capability(intrinsic) => FormationError::RequiresCapability {
                intrinsic: intrinsic.clone(),
            }
            .to_string(),
            _ => FormationError::Bug(
                "a universal primitive mapping legalized as inapplicable".into(),
            )
            .to_string(),
        });
    }
    builder.map_primitive(node, formed.iteration_or_serial(), legalized)
}

/// Consume one classified safety obligation with the family builder's
/// `discharge` transition. A statically proved classification becomes the
/// builder's proved disposition (no check is planned and none may be
/// emitted); a runtime-checked classification legalizes its planned predicate
/// through the dialect; a statically impossible classification makes the
/// alternative inapplicable.
pub fn discharge_with_builder<D: ExecutableDialect>(
    builder: &mut AlternativeBuilder<D>,
    obligation: ObligationRef,
    classified: &ObligationDischarge,
    legalize_predicate: impl FnOnce(&CheckPredicate) -> Legalized<D::Op>,
) -> Result<DispositionReceipt, BuilderError> {
    let disposition = match classified {
        ObligationDischarge::StaticallyProved(proof) => ObligationDisposition::StaticallyProved {
            reason: proof.reason(),
        },
        ObligationDischarge::RuntimeChecked(check) => ObligationDisposition::RuntimeChecked {
            predicate: legalize_predicate(&check.predicate),
            inactive: match check.inactive {
                InactiveBehavior::SkipOperation => BuilderInactive::Skip,
            },
        },
        ObligationDischarge::StaticallyImpossible(reason) => return Err(reason.clone()),
    };
    builder.discharge(obligation, disposition)
}
