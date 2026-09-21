//! Mapping catalogs, mapping proposals, and closed strategy shapes
//! (package S1 owns formation; C0 freezes the catalog contract that backend
//! lanes B1 code against).
//!
//! The core owns graph traversal and structured formation. A backend
//! supplies a declarative catalog of optional mapping rules. A rule may
//! decline before a candidate exists; once it returns a proposal, core
//! formation (`StrategyFormer::form`) either creates a complete legal
//! strategy or reports a compiler defect. The universal rules are core-owned
//! (`formation::strategy::universal_rules`) and cover every applicable
//! portable alternative with a resource-scalable serial/streaming point
//! realization plus a bounded linear-participant peer.
//!
//! The shared structural analyses at the end of this module
//! (`region_nodes`, `owned_nodes`, `node_is_absorbable`,
//! `atomic_join_storages`, `elementwise_domain`, `scan_state`, ...) are what
//! backend rules query instead of walking logical graphs. The core-produced
//! pattern facts (`streaming_segments`, `reduction_shapes`,
//! `scannable_loops`, `tile_candidates`, `capability_uses`,
//! `phase_crossings`) are the declarative structural surface backend rules
//! match against; the universal rules consume the same facts, so universal
//! and backend formation read one truth.

use crate::ids::{
    BlockId, CanonicalLeafId, CanonicalStorageId, CanonicalValueId, ObligationRef, OccurrenceId,
    OwnedGraphKey, OwnedNodeRef, OwnedOccurrence, OwnedRegionRef, OwnedRegionResultRef,
    OwnedStorageRef, OwnedValueRef, OwnedViewRef, PlanParamId,
};
use crate::kernel::{ExecutableDialect, IntrinsicCatalog};
use crate::occurrence::OccurrenceFacts;
use crate::target::{EffectiveTargetProfile, TargetLimits};
use seismic_lang::intrinsics::{
    AtomicOp, CombineLaw, IntrinsicId, MathOp, PrimitiveId, ReduceIdentity, ReduceOp,
};
use seismic_lang::logical::value::{GraphValueKind, TensorSource};
use seismic_lang::logical::{
    AtomicOperation, GraphRegion, GraphValueId, IdVec, LogicalNode, LogicalNodeKind, LoopNode,
    NodeRef, PrimitiveOp, ReductionOrder, RegionInput, RegionParameter, RegionResult, RegionStep,
    SafetyObligation, StateJoin, ViewBase,
};
use seismic_lang::sir::LoopKind;
use seismic_lang::syntax::ast::BinaryOp;
use seismic_lang::types::{DType, ExtentExpr, NonEmpty};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// The name of a mapping rule, for diagnostics and cost attribution only.
pub type RuleName = &'static str;

/// What a backend supplies to `form_plan_space`. Catalog construction
/// validates that every declared intrinsic family has an encoder.
pub trait MappingCatalog<D: ExecutableDialect> {
    /// Optional optimized mapping rules (peers of the universal rules).
    fn rules(&self) -> &[Box<dyn MappingRule>];
    /// Typed intrinsic lowering families authorized on this target.
    fn intrinsics(&self) -> &dyn IntrinsicCatalog<D>;
    /// Backend cost coefficients; ranking only, never legality.
    fn cost_model(&self) -> &dyn CostModel;
    /// Native resource facts declared by the backend which cannot be known
    /// until assembly, with their admissible domains (validated by N1).
    fn native_fact_domains(&self) -> &[NativeFactDomain];
    fn limits(&self) -> &TargetLimits;
}

/// One declarative mapping rule. It matches occurrence facts and proposes
/// ownership/mapping choices; it never allocates storage, emits opcodes,
/// creates schedule nodes, or walks a logical graph recursively.
pub trait MappingRule {
    fn name(&self) -> RuleName;
    fn propose(&self, query: &RuleQuery<'_, '_>) -> Vec<MappingProposal>;
}

/// The read-only facts one rule may query.
pub struct RuleQuery<'f, 'l> {
    pub facts: &'f OccurrenceFacts<'l>,
    pub profile: &'f EffectiveTargetProfile,
    pub occurrence: OccurrenceId,
    pub logical_alternative: u32,
}

/// A backend cost model over sealed operations. Legality never lives here.
pub trait CostModel {
    fn launch_overhead_ns(&self) -> u64;
    fn point_cost_ns(&self, op: &crate::kernel::CostUnit) -> u64;
}

/// A native fact the backend reflects at assembly with its admissible domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeFactDomain {
    pub kind: NativeFactKind,
    pub min: u64,
    pub max: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeFactKind {
    MaxResidentParticipants,
    NativeSubgroupWidth,
}

// ---------------------------------------------------------------------------
// Proposals
// ---------------------------------------------------------------------------

/// Which occurrences one strategy owns completely.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnershipProposal {
    pub root: OwnedOccurrence,
    /// Complete descendant occurrences absorbed by this strategy, with the
    /// logical alternative selected for each.
    pub absorbed: BTreeMap<OccurrenceId, u32>,
}

/// One proposed launch group of a strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaunchGroup(pub u32);

/// Where one owned node executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodePlacement {
    /// Inside the launch group; a control node placed in a launch is absorbed
    /// (its body nodes must be placed in the same launch).
    Launch(LaunchGroup),
    /// Retained as a structured executor step (call, if, repeat); the body
    /// nodes are placed independently.
    Retained,
}

/// A declared tuning parameter of one proposal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuningDeclaration {
    pub name: String,
    pub lower: u64,
    pub upper: u64,
}

/// Reference to a tuning parameter within one proposal, by declaration
/// ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TuningRef(pub u32);

/// How participants cover one launch group's independent domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParticipantPolicy {
    /// One participant traverses everything serially: the universal
    /// feasibility witness with local resources independent of extent.
    Serial,
    /// A linear participant count covers the independent axes by grid
    /// stride; the count is a tuning parameter.
    Linear { participants: TuningRef },
    /// One workgroup of `width` participants cooperates through a typed
    /// intrinsic family (subgroup/matrix).
    Cooperative { width: TuningRef, family: IntrinsicId },
    /// `participants` claim linear coordinates of the independent domain
    /// from a device pull counter (`Traversal::DynamicPull`); offered by the
    /// universal catalog for independent domains with a runtime total. The
    /// former emits `ShapeStep::PullCounterReset` before the launch.
    DynamicPull { participants: TuningRef },
    /// One cooperative grid launch of `participants` with grid-wide barriers
    /// between whole-result dependencies inside the launch. Admissible only
    /// when `profile.limits.cooperative_grid` is `Some`; proposed by backend
    /// catalogs (CUDA), never by the universal rules.
    GridCooperative { participants: TuningRef },
}

/// The algorithm one launch group realizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AlgorithmChoice {
    /// Universal point-wise/serial realization of every owned node.
    Universal,
    /// Reduction realized with the named topology.
    Reduction(crate::numerics::ReductionTopology),
    /// Blocked/tiled realization with the tile extent as a tuning parameter.
    Blocked { tile: TuningRef },
    /// Realization through one typed capability intrinsic.
    Intrinsic(IntrinsicId),
}

/// A kernel-local residence the proposal requires for one launch group.
/// Core creates the residence (D1); the backend never stages storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalResidenceRequirement {
    pub value: CanonicalValueId,
    pub scope: crate::residence::StorageScope,
    pub replication: crate::residence::Replication,
}

/// A numerical freedom the proposal explicitly takes (package M1 derives the
/// transfer). Emitters cannot take one that is not recorded here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumericalChoice {
    Reassociate { node: OwnedNodeRef },
    FastMath { node: OwnedNodeRef, operation: String },
    ReducedPrecision { node: OwnedNodeRef, dtype: seismic_lang::types::DType },
}

/// One launch group of a proposal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchProposal {
    pub participants: ParticipantPolicy,
    pub algorithm: AlgorithmChoice,
    pub local_residences: Vec<LocalResidenceRequirement>,
    pub numerical: Vec<NumericalChoice>,
}

/// A complete ownership and mapping proposal. The former checks that every
/// owned node is assigned exactly once and that the assignment respects
/// structured nesting; a violation is a compiler defect of the rule's owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MappingProposal {
    pub rule: RuleName,
    pub ownership: OwnershipProposal,
    pub placement: BTreeMap<OwnedNodeRef, NodePlacement>,
    pub launches: IdVec<LaunchGroup, LaunchProposal>,
    pub required_intrinsics: BTreeSet<IntrinsicId>,
    pub tuning: Vec<TuningDeclaration>,
}

impl seismic_lang::logical::IdIndex for LaunchGroup {
    fn from_index(index: usize) -> Self {
        LaunchGroup(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

// ---------------------------------------------------------------------------
// Closed strategy shapes (S1 output)
// ---------------------------------------------------------------------------

/// The structured schedule of one strategy. Sibling steps complete in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeSchedule {
    pub steps: Vec<ShapeStep>,
}

/// One structured schedule step. Loops/conditionals consumed wholly by one
/// launch are kernel-local control and do not appear here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapeStep {
    Launch(BlockId),
    Guard {
        obligation: ObligationRef,
        predicate: ExecutorGuardPredicate,
    },
    /// Zero the pull counter of `block` (a `DynamicPull` block) immediately
    /// before its launch. D1 maps it to the counter residence; P1 seals it
    /// as a fill step.
    PullCounterReset {
        block: BlockId,
    },
    Call {
        node: OwnedNodeRef,
        occurrence: OccurrenceId,
    },
    If {
        node: OwnedNodeRef,
        condition: CanonicalValueId,
        then_schedule: ShapeSchedule,
        else_schedule: ShapeSchedule,
        joins: Vec<JoinEdge>,
    },
    Repeat {
        node: OwnedNodeRef,
        kind: LoopKind,
        start: CanonicalValueId,
        end: CanonicalValueId,
        bound: ExtentExpr,
        binder: CanonicalValueId,
        body: ShapeSchedule,
        carries: Vec<CarryEdge>,
    },
}

/// A structural executor guard predicate over retained extents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutorGuardPredicate {
    ProductFits { factors: Vec<ExtentExpr>, bits: u8 },
    ExtentPositive { extent: ExtentExpr },
}

/// One value or state joined across a conditional.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JoinEdge {
    Value {
        then_value: CanonicalValueId,
        else_value: CanonicalValueId,
        joined: CanonicalValueId,
    },
    State {
        storage: crate::ids::CanonicalStorageId,
    },
}

/// One value or state carried across repeat visits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CarryEdge {
    Value {
        initial: CanonicalValueId,
        parameter: CanonicalValueId,
        update: CanonicalValueId,
        result: CanonicalValueId,
    },
    State {
        storage: crate::ids::CanonicalStorageId,
    },
}

/// One absorbed independent-loop binder bound to a kernel axis ordinal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisBinding {
    pub binder: CanonicalValueId,
    pub ordinal: u32,
    pub extent: ExtentExpr,
}

/// One absorbed ordered scalar carry inside a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedScalarCarry {
    pub initial: CanonicalValueId,
    pub parameter: CanonicalValueId,
    pub update: CanonicalValueId,
    pub result: CanonicalValueId,
}

/// The complete participant map of one block, derived by S1 from the
/// logical loop structure and the proposal's participant policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParticipantMap {
    pub policy: ParticipantPolicy,
    pub independent_axes: Vec<AxisBinding>,
    pub serial_binders: BTreeSet<CanonicalValueId>,
    pub ordered_carries: Vec<OrderedScalarCarry>,
}

/// How one obligation is discharged by this strategy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObligationDisposition {
    /// Proved statically; the proof names the extent/interval fact used.
    Static { proof: String },
    /// Checked inside the named block; K1 lowers the check.
    KernelGuard { block: BlockId },
    /// Checked by a structured executor guard step.
    ExecutorGuard,
}

/// One closed kernel block cut, derived (never declared after the fact).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockCut {
    pub nodes: NonEmpty<OwnedNodeRef>,
    pub participants: ParticipantMap,
    pub algorithm: AlgorithmChoice,
    pub local_residences: Vec<LocalResidenceRequirement>,
    pub numerical: Vec<NumericalChoice>,
    /// Non-local leaves consumed by this block.
    pub inputs: BTreeSet<CanonicalLeafId>,
    /// Non-local leaves produced by this block and consumed elsewhere.
    pub outputs: BTreeSet<CanonicalLeafId>,
    /// Storage chains read/written by this block.
    pub state_reads: BTreeSet<crate::ids::CanonicalStorageId>,
    pub state_writes: BTreeSet<crate::ids::CanonicalStorageId>,
    /// Obligations discharged inside this block.
    pub guards: Vec<(ObligationRef, SafetyObligation)>,
}

/// One value or state edge crossing a block/step boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrossEdge {
    pub leaf: CanonicalLeafId,
    pub producer: EdgeEnd,
    pub consumers: NonEmpty<EdgeEnd>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EdgeEnd {
    RootBoundary,
    Block(BlockId),
    Call(OwnedNodeRef),
    Control(OwnedNodeRef),
}

/// A complete, sealed strategy shape. No public constructor; sealed by
/// `StrategyFormer::form` only when the consumed node, result, state, and
/// obligation sets equal their exact expected sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosedStrategyShape {
    root: OwnedOccurrence,
    absorbed: BTreeMap<OccurrenceId, u32>,
    rule: RuleName,
    schedule: ShapeSchedule,
    blocks: IdVec<BlockId, BlockCut>,
    obligations: BTreeMap<ObligationRef, ObligationDisposition>,
    results: BTreeMap<OwnedRegionResultRef, EdgeEnd>,
    edges: Vec<CrossEdge>,
    tuning: IdVec<PlanParamId, TuningDeclaration>,
    required_intrinsics: BTreeSet<IntrinsicId>,
    /// Every region of every owned graph (root regions first, then nested
    /// regions in pre-order), recorded at the seal so lifetime derivation
    /// never re-walks a graph.
    regions: Vec<OwnedRegionRef>,
}

impl ClosedStrategyShape {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn seal(
        root: OwnedOccurrence,
        absorbed: BTreeMap<OccurrenceId, u32>,
        rule: RuleName,
        schedule: ShapeSchedule,
        blocks: IdVec<BlockId, BlockCut>,
        obligations: BTreeMap<ObligationRef, ObligationDisposition>,
        results: BTreeMap<OwnedRegionResultRef, EdgeEnd>,
        edges: Vec<CrossEdge>,
        tuning: IdVec<PlanParamId, TuningDeclaration>,
        required_intrinsics: BTreeSet<IntrinsicId>,
        regions: Vec<OwnedRegionRef>,
    ) -> ClosedStrategyShape {
        ClosedStrategyShape {
            root,
            absorbed,
            rule,
            schedule,
            blocks,
            obligations,
            results,
            edges,
            tuning,
            required_intrinsics,
            regions,
        }
    }

    pub fn root(&self) -> OwnedOccurrence {
        self.root
    }
    pub fn absorbed(&self) -> &BTreeMap<OccurrenceId, u32> {
        &self.absorbed
    }
    pub fn rule(&self) -> RuleName {
        self.rule
    }
    pub fn schedule(&self) -> &ShapeSchedule {
        &self.schedule
    }
    pub fn blocks(&self) -> &IdVec<BlockId, BlockCut> {
        &self.blocks
    }
    pub fn obligations(&self) -> &BTreeMap<ObligationRef, ObligationDisposition> {
        &self.obligations
    }
    pub fn results(&self) -> &BTreeMap<OwnedRegionResultRef, EdgeEnd> {
        &self.results
    }
    pub fn edges(&self) -> &[CrossEdge] {
        &self.edges
    }
    pub fn tuning(&self) -> &IdVec<PlanParamId, TuningDeclaration> {
        &self.tuning
    }
    pub fn required_intrinsics(&self) -> &BTreeSet<IntrinsicId> {
        &self.required_intrinsics
    }
    /// Every region owned by this strategy, for D1 lifetime derivation: the
    /// root region of every owned graph and every nested region, in
    /// pre-order.
    pub fn owned_regions(&self) -> Vec<OwnedRegionRef> {
        self.regions.clone()
    }

    /// The block one owned node executes in, or `None` when the node is a
    /// retained step (call, conditional, repeat).
    pub fn block_of(&self, node: &OwnedNodeRef) -> Option<BlockId> {
        self.blocks
            .entries()
            .find(|(_, block)| block.nodes.iter().any(|owned| owned == node))
            .map(|(id, _)| id)
    }
}

// ---------------------------------------------------------------------------
// Shared structural analyses (pure over occurrence facts)
// ---------------------------------------------------------------------------
//
// Backend mapping rules query these instead of walking logical graphs. Every
// function is total over the sealed program; none allocates or decides.

/// Resolve one keyed occurrence lookup over a reference the same facts
/// minted. Every analysis in this module traverses ids it obtained from
/// `OccurrenceFacts` itself (or a caller-derived root of a minted
/// alternative), so a keyed lookup failure contradicts the expansion
/// invariant and is an S1 defect.
macro_rules! minted {
    ($lookup:expr, $invariant:literal) => {
        match $lookup {
            Ok(value) => value,
            Err(_) => unreachable!($invariant),
        }
    };
}

/// The region one owned node belongs to.
pub fn node_region(node: &OwnedNodeRef) -> OwnedRegionRef {
    OwnedRegionRef {
        graph: node.graph,
        region: node.node.region.clone(),
    }
}

/// The root region of one owned graph.
pub fn root_region(graph: OwnedGraphKey) -> OwnedRegionRef {
    OwnedRegionRef {
        graph,
        region: Vec::new(),
    }
}

/// The direct child nodes of one owned region, in region order.
pub fn region_nodes(facts: &OccurrenceFacts<'_>, region: &OwnedRegionRef) -> Vec<OwnedNodeRef> {
    minted!(facts.region(region), "a traversed region is minted (S1)")
        .nodes
        .ids()
        .map(|id| OwnedNodeRef {
            graph: region.graph,
            node: NodeRef {
                region: region.region.clone(),
                node: id,
            },
        })
        .collect()
}

/// The regions one control node owns (then/else of a conditional, the body
/// of a loop); empty for primitives, reductions, and calls.
pub fn child_regions(facts: &OccurrenceFacts<'_>, node: &OwnedNodeRef) -> Vec<OwnedRegionRef> {
    let child = |step: RegionStep| {
        let mut path = node.node.region.clone();
        path.push(step);
        OwnedRegionRef {
            graph: node.graph,
            region: path,
        }
    };
    match &minted!(facts.node(node), "a traversed node is minted (S1)").kind {
        LogicalNodeKind::If(_) => vec![
            child(RegionStep::IfThen(node.node.node)),
            child(RegionStep::IfElse(node.node.node)),
        ],
        LogicalNodeKind::Loop(_) => vec![child(RegionStep::LoopBody(node.node.node))],
        LogicalNodeKind::Primitive(_) | LogicalNodeKind::Reduction(_) | LogicalNodeKind::Call(_) => {
            Vec::new()
        }
    }
}

/// One node and every node nested under it, in pre-order (the node first).
/// Absorbed callee graphs are not inlined here; see `owned_nodes`.
pub fn subtree_nodes(facts: &OccurrenceFacts<'_>, node: &OwnedNodeRef) -> Vec<OwnedNodeRef> {
    let mut out = Vec::new();
    fn walk(facts: &OccurrenceFacts<'_>, node: &OwnedNodeRef, out: &mut Vec<OwnedNodeRef>) {
        out.push(node.clone());
        for region in child_regions(facts, node) {
            for child in region_nodes(facts, &region) {
                walk(facts, &child, out);
            }
        }
    }
    walk(facts, node, &mut out);
    out
}

/// The values one node reads: its inputs, plus the outer values a
/// conditional captures, plus a loop's range endpoints, invariant captures,
/// and initial carried values. Every entry is an owned reference of the
/// node's graph.
pub fn node_value_edges(facts: &OccurrenceFacts<'_>, node: &OwnedNodeRef) -> Vec<OwnedValueRef> {
    let logical = minted!(facts.node(node), "a traversed node is minted (S1)");
    let ids: Vec<GraphValueId> = match &logical.kind {
        LogicalNodeKind::If(if_node) => logical
            .inputs
            .iter()
            .copied()
            .chain(if_node.captured.iter().map(|capture| capture.outer))
            .collect(),
        LogicalNodeKind::Loop(loop_node) => std::iter::once(loop_node.range.start)
            .chain(std::iter::once(loop_node.range.end))
            .chain(loop_node.invariant_values.iter().copied())
            .chain(loop_node.initial_values.iter().copied())
            .collect(),
        LogicalNodeKind::Primitive(_) | LogicalNodeKind::Reduction(_) | LogicalNodeKind::Call(_) => {
            logical.inputs.clone()
        }
    };
    ids.into_iter()
        .map(|value| OwnedValueRef {
            graph: node.graph,
            value,
        })
        .collect()
}

/// The state tokens one node consumes: a loop consumes its initial states,
/// every other node its `state_inputs`.
pub fn node_state_inputs(logical: &LogicalNode) -> &[seismic_lang::logical::StateTokenId] {
    match &logical.kind {
        LogicalNodeKind::Loop(loop_node) => &loop_node.initial_states,
        LogicalNodeKind::If(_)
        | LogicalNodeKind::Primitive(_)
        | LogicalNodeKind::Reduction(_)
        | LogicalNodeKind::Call(_) => &logical.state_inputs,
    }
}

/// The owned graphs of one ownership proposal that are reachable from its
/// root through absorbed calls: the root key first, then each absorbed
/// callee in call order. An absorbed occurrence that no owned call reaches
/// is not listed (the former reports it as a defect).
pub fn owned_graphs(facts: &OccurrenceFacts<'_>, ownership: &OwnershipProposal) -> Vec<OwnedGraphKey> {
    let root = OwnedGraphKey {
        occurrence: ownership.root.occurrence,
        logical_alternative: ownership.root.logical_alternative,
    };
    let mut out = vec![root];
    let mut queue = VecDeque::from([root]);
    while let Some(key) = queue.pop_front() {
        for (_, occurrence) in &facts.alternative(key).calls {
            if let Some(alternative) = ownership.absorbed.get(occurrence) {
                let child = OwnedGraphKey {
                    occurrence: *occurrence,
                    logical_alternative: *alternative,
                };
                if !out.contains(&child) {
                    out.push(child);
                    queue.push_back(child);
                }
            }
        }
    }
    out
}

/// Every node the proposal owns, in execution order: pre-order over the
/// root graph with each absorbed callee's nodes inlined immediately after
/// the call node that reaches it.
pub fn owned_nodes(facts: &OccurrenceFacts<'_>, ownership: &OwnershipProposal) -> Vec<OwnedNodeRef> {
    fn walk(
        facts: &OccurrenceFacts<'_>,
        ownership: &OwnershipProposal,
        region: &OwnedRegionRef,
        out: &mut Vec<OwnedNodeRef>,
    ) {
        for node in region_nodes(facts, region) {
            out.push(node.clone());
            if let LogicalNodeKind::Call(_) = &minted!(facts.node(&node), "a traversed node is minted (S1)").kind {
                let occurrence = minted!(
                    facts.call_occurrence(&node),
                    "a traversed call node is recorded by its alternative (S1)"
                );
                if let Some(alternative) = ownership.absorbed.get(&occurrence) {
                    let callee = OwnedGraphKey {
                        occurrence,
                        logical_alternative: *alternative,
                    };
                    walk(facts, ownership, &root_region(callee), out);
                }
            }
            for child in child_regions(facts, &node) {
                walk(facts, ownership, &child, out);
            }
        }
    }
    let root = OwnedGraphKey {
        occurrence: ownership.root.occurrence,
        logical_alternative: ownership.root.logical_alternative,
    };
    let mut out = Vec::new();
    walk(facts, ownership, &root_region(root), &mut out);
    out
}

/// Whether one node can live inside a kernel launch: primitives and
/// reductions always; a loop when every value carry is a kernel scalar
/// (scalar, index, or capability value) and its body is absorbable;
/// conditionals and calls never.
pub fn node_is_absorbable(facts: &OccurrenceFacts<'_>, node: &OwnedNodeRef) -> bool {
    match &minted!(facts.node(node), "a traversed node is minted (S1)").kind {
        LogicalNodeKind::Primitive(_) | LogicalNodeKind::Reduction(_) => true,
        LogicalNodeKind::If(_) | LogicalNodeKind::Call(_) => false,
        LogicalNodeKind::Loop(loop_node) => {
            let carries_are_kernel_scalars = loop_node.carried.iter().all(|slot| match slot.initial {
                RegionInput::State(_) => true,
                RegionInput::Value(initial) => {
                    let canonical = minted!(
                        facts.canonical_value(OwnedValueRef {
                            graph: node.graph,
                            value: initial,
                        }),
                        "a carried value of a minted node is canonicalized (S1)"
                    );
                    matches!(
                        facts.value_kind(canonical),
                        GraphValueKind::Scalar(_)
                            | GraphValueKind::Index { .. }
                            | GraphValueKind::Capability(_)
                    )
                }
            });
            carries_are_kernel_scalars
                && child_regions(facts, node)
                    .iter()
                    .all(|body| region_is_absorbable(facts, body))
        }
    }
}

/// Whether every node of a region is absorbable (`node_is_absorbable`).
pub fn region_is_absorbable(facts: &OccurrenceFacts<'_>, region: &OwnedRegionRef) -> bool {
    region_nodes(facts, region)
        .iter()
        .all(|node| node_is_absorbable(facts, node))
}

/// The canonical storages an independent loop joins atomically across its
/// visits (`StateJoin::Atomic`); empty for every other node.
pub fn atomic_join_storages(facts: &OccurrenceFacts<'_>, node: &OwnedNodeRef) -> Vec<CanonicalStorageId> {
    minted!(facts.node(node), "a traversed node is minted (S1)")
        .state_outputs
        .iter()
        .filter(|token| matches!(token.join, Some(StateJoin::Atomic { .. })))
        .map(|token| {
            minted!(
                facts.canonical_storage(OwnedStorageRef {
                    graph: node.graph,
                    storage: token.storage,
                }),
                "a state output of a minted node is canonicalized (S1)"
            )
        })
        .collect()
}

/// Every admitted atomic operation of the `StateJoin::Atomic` joins of one
/// node (empty for every node that joins nothing atomically).
pub fn atomic_operations(facts: &OccurrenceFacts<'_>, node: &OwnedNodeRef) -> Vec<AtomicOperation> {
    minted!(facts.node(node), "a traversed node is minted (S1)")
        .state_outputs
        .iter()
        .filter_map(|token| match &token.join {
            Some(StateJoin::Atomic { operations }) => Some(operations.iter().cloned()),
            Some(StateJoin::DisjointWrite { .. }) | None => None,
        })
        .flatten()
        .collect()
}

/// The top-level nodes of one block cut, in block order: every block node
/// that is not nested under another node of the block (an absorbed callee's
/// root nodes are top-level beside their call). These are the phases a
/// `GridCooperative` launch separates with grid-wide barriers.
pub fn block_top_level_nodes(block: &BlockCut) -> Vec<OwnedNodeRef> {
    let is_nested_under = |node: &OwnedNodeRef, ancestor: &OwnedNodeRef| {
        node.graph == ancestor.graph
            && node.node.region.len() > ancestor.node.region.len()
            && node.node.region[..ancestor.node.region.len()] == ancestor.node.region[..]
            && node.node.region[ancestor.node.region.len()].node() == ancestor.node.node
    };
    block
        .nodes
        .iter()
        .filter(|node| !block.nodes.iter().any(|other| is_nested_under(node, other)))
        .cloned()
        .collect()
}

/// The phase domain of one node for whole-result analysis: the elementwise
/// domain of a primitive or reduction (empty when it has none), the
/// iteration bound of a loop.
pub fn phase_domain(facts: &OccurrenceFacts<'_>, node: &OwnedNodeRef) -> Vec<ExtentExpr> {
    match &minted!(facts.node(node), "a traversed node is minted (S1)").kind {
        LogicalNodeKind::Loop(loop_node) => vec![loop_node.range.bound.clone()],
        LogicalNodeKind::Primitive(_) | LogicalNodeKind::Reduction(_) => {
            elementwise_domain(facts, node).unwrap_or_default()
        }
        LogicalNodeKind::If(_) | LogicalNodeKind::Call(_) => Vec::new(),
    }
}

/// The elementwise iteration domain of one node: the axes its per-element
/// operation ranges over, when the node is a whole-tensor primitive (the
/// result shape of pointwise, cast, fill, bulk-copy, decode, and residual
/// element reads/writes; the source shape of a slice copy) or a reduction
/// (its operand shape). `None` for scalar, structural, view, allocation,
/// extent, atomic, constant, capability, and control nodes.
pub fn elementwise_domain(facts: &OccurrenceFacts<'_>, node: &OwnedNodeRef) -> Option<Vec<ExtentExpr>> {
    let logical = minted!(facts.node(node), "a traversed node is minted (S1)");
    let operand_axes = |index: usize| -> Option<Vec<ExtentExpr>> {
        let value = *logical.inputs.get(index)?;
        let canonical = minted!(
            facts.canonical_value(OwnedValueRef {
                graph: node.graph,
                value,
            }),
            "an input of a minted node is canonicalized (S1)"
        );
        match facts.value_kind(canonical) {
            GraphValueKind::Tensor { ty, .. } => Some(ty.axes.clone()),
            _ => None,
        }
    };
    let result_axes = |index: usize| -> Option<Vec<ExtentExpr>> {
        let output = logical.outputs.get(index)?;
        match output.kind() {
            GraphValueKind::Tensor { ty, .. } => Some(ty.axes.clone()),
            _ => None,
        }
    };
    match &logical.kind {
        LogicalNodeKind::Reduction(_) => operand_axes(0),
        LogicalNodeKind::Primitive(application) => match &application.op {
            PrimitiveOp::Primitive(PrimitiveId::ElementRead { .. }) => result_axes(0),
            PrimitiveOp::Primitive(PrimitiveId::ElementWrite { .. }) => {
                operand_axes(logical.inputs.len().checked_sub(1)?)
            }
            PrimitiveOp::Primitive(PrimitiveId::CopyInto) => operand_axes(1),
            PrimitiveOp::Primitive(
                PrimitiveId::Fill { .. }
                | PrimitiveId::Materialize
                | PrimitiveId::Clone
                | PrimitiveId::Load
                | PrimitiveId::Decode,
            ) => result_axes(0),
            PrimitiveOp::Primitive(
                PrimitiveId::Unary(_)
                | PrimitiveId::Binary(_)
                | PrimitiveId::Cast(_)
                | PrimitiveId::Math(_)
                | PrimitiveId::Select,
            ) => (0..logical.outputs.len())
                .find_map(result_axes)
                .or_else(|| (0..logical.inputs.len()).find_map(operand_axes)),
            PrimitiveOp::Primitive(
                PrimitiveId::TuplePack
                | PrimitiveId::TupleGet(_)
                | PrimitiveId::RangeMake
                | PrimitiveId::RangeStart
                | PrimitiveId::RangeEnd
                | PrimitiveId::TensorAlloc { .. }
                | PrimitiveId::PackedRead(_)
                | PrimitiveId::Transpose
                | PrimitiveId::Reshape
                | PrimitiveId::SliceView { .. }
                | PrimitiveId::Extent { .. }
                | PrimitiveId::ValidExtent { .. }
                | PrimitiveId::Atomic { .. }
                | PrimitiveId::Reduce { .. },
            )
            | PrimitiveOp::Constant(_)
            | PrimitiveOp::RuntimeExtent(_)
            | PrimitiveOp::Capability(_) => None,
        },
        LogicalNodeKind::If(_) | LogicalNodeKind::Loop(_) | LogicalNodeKind::Call(_) => None,
    }
}

// ---------------------------------------------------------------------------
// The structural online-scan rule (shared by streaming mapping rules)
// ---------------------------------------------------------------------------

/// The combining operation of one derived carried accumulator lane, taken
/// from registry semantics only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanCombine {
    /// Registry `max`/`min`: order-insensitive under the registry NaN rule.
    Extremum { op: MathOp },
    /// Registry `add` over window partials: reassociates the ascending fold.
    Additive,
}

impl ScanCombine {
    /// The registry combination law of the lane's operator.
    pub fn law(self) -> CombineLaw {
        match self {
            ScanCombine::Extremum { .. } | ScanCombine::Additive => {
                CombineLaw::AssociativeCommutative
            }
        }
    }
}

/// One derived carried accumulator lane of a scannable ordered loop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CarriedLane {
    /// Ordinal of the carried slot in the loop's carried order.
    pub slot: u32,
    /// The accumulator dtype (the carried value's scalar dtype).
    pub dtype: DType,
    pub combine: ScanCombine,
    /// The registry identity the lane's fold starts from.
    pub identity: ReduceIdentity,
}

/// The derived scan state of an ordered loop whose carried lanes are
/// registry folds: the structural admission of windowed streaming.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanState {
    pub lanes: Vec<CarriedLane>,
    /// An additive lane rescales through an extremum lane's carry (the
    /// online-softmax `m`/`l` shape).
    pub rescale: bool,
    /// The checked capacity bound of the scanned axis (the runtime extent's
    /// capacity for runtime axes, the exact length for static axes).
    pub axis_capacity: u64,
}

/// Apply the structural online-scan rule to one loop node. Pure logical
/// analysis: it recognizes no names, consults no target facts, and never
/// replaces arithmetic. `None` when the node is not an ordered loop whose
/// every carried lane is a registry extremum/additive fold (with the
/// admitted `exp` rescale shape) over a bounded axis.
pub fn scan_state(facts: &OccurrenceFacts<'_>, node: &OwnedNodeRef) -> Option<ScanState> {
    let LogicalNodeKind::Loop(loop_node) =
        &minted!(facts.node(node), "a traversed node is minted (S1)").kind
    else {
        return None;
    };
    if loop_node.kind != LoopKind::Ordered || loop_node.carried.is_empty() {
        return None;
    }
    let axis_capacity = checked_axis_capacity(facts, &loop_node.range.bound)?;
    if axis_capacity == 0 {
        return None;
    }
    let body = &loop_node.body;
    let mut carry_params: Vec<GraphValueId> = Vec::with_capacity(loop_node.carried.len());
    for slot in &loop_node.carried {
        let RegionInput::Value(_) = slot.initial else {
            // A carried storage has no registry scalar combine law.
            return None;
        };
        let Some(RegionParameter::Value { id, .. }) = body.parameters.get(slot.body_parameter.index())
        else {
            return None;
        };
        carry_params.push(*id);
    }
    let all_carry_params: BTreeSet<GraphValueId> = carry_params.iter().copied().collect();
    let mut lanes = Vec::new();
    let mut rescale = false;
    for (ordinal, slot) in loop_node.carried.iter().enumerate() {
        let RegionInput::Value(initial) = slot.initial else {
            return None;
        };
        let canonical = minted!(
            facts.canonical_value(OwnedValueRef {
                graph: node.graph,
                value: initial,
            }),
            "a carried value of a minted node is canonicalized (S1)"
        );
        let GraphValueKind::Scalar(dtype) = facts.value_kind(canonical) else {
            return None;
        };
        let Some(RegionResult::Value { id: result, .. }) = body.results.get(slot.body_result.index())
        else {
            return None;
        };
        match analyze_lane(body, *result, carry_params[ordinal], &all_carry_params)? {
            LaneAnalysis::Extremum { op } => lanes.push(CarriedLane {
                slot: ordinal as u32,
                dtype: *dtype,
                combine: ScanCombine::Extremum { op },
                identity: ReduceIdentity::FirstElement,
            }),
            LaneAnalysis::Additive {
                cross_lane,
                exp_present,
            } => {
                if cross_lane && !exp_present {
                    return None;
                }
                rescale |= cross_lane;
                lanes.push(CarriedLane {
                    slot: ordinal as u32,
                    dtype: *dtype,
                    combine: ScanCombine::Additive,
                    identity: ReduceIdentity::Zero,
                });
            }
        }
    }
    if rescale
        && !lanes
            .iter()
            .any(|lane| matches!(lane.combine, ScanCombine::Extremum { .. }))
    {
        return None;
    }
    Some(ScanState {
        lanes,
        rescale,
        axis_capacity,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LaneAnalysis {
    Extremum { op: MathOp },
    Additive { cross_lane: bool, exp_present: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LaneCombine {
    Extremum(MathOp),
    Additive,
}

/// Walk the body's value graph backwards from one lane's result producer,
/// classifying the fold. The carry path is the chain of nodes from the
/// result producer down to the lane's carry parameter; its first node is
/// the terminal combine, and every node strictly between it and the carry
/// must be a `mul`/`sub` rescale step consuming the carry. Everything off
/// the path is operand arithmetic the mapped body executes exactly.
fn analyze_lane(
    body: &GraphRegion,
    result: GraphValueId,
    own: GraphValueId,
    all_carry_params: &BTreeSet<GraphValueId>,
) -> Option<LaneAnalysis> {
    let mut producer: BTreeMap<GraphValueId, &LogicalNode> = BTreeMap::new();
    for node in body.nodes.iter() {
        for output in &node.outputs {
            producer.insert(output.id(), node);
        }
    }
    let mut prev: BTreeMap<GraphValueId, GraphValueId> = BTreeMap::new();
    let mut visited: BTreeSet<GraphValueId> = BTreeSet::from([result]);
    let mut queue: VecDeque<GraphValueId> = VecDeque::from([result]);
    let mut cross_lane = false;
    let mut exp_present = false;
    while let Some(value) = queue.pop_front() {
        if value == own {
            continue;
        }
        if all_carry_params.contains(&value) {
            cross_lane = true;
            continue;
        }
        let Some(node) = producer.get(&value).copied() else {
            // An invariant region parameter: an admitted leaf.
            continue;
        };
        match &node.kind {
            LogicalNodeKind::Primitive(application) => match &application.op {
                PrimitiveOp::Constant(_) | PrimitiveOp::RuntimeExtent(_) => {}
                PrimitiveOp::Capability(_) => return None,
                PrimitiveOp::Primitive(id) => match id {
                    PrimitiveId::ElementRead { .. } | PrimitiveId::Cast(_) | PrimitiveId::Binary(_) => {}
                    PrimitiveId::Math(op) => {
                        if *op == MathOp::Exp {
                            exp_present = true;
                        }
                    }
                    _ => return None,
                },
            },
            LogicalNodeKind::Reduction(reduction) => match reduction.op {
                ReduceOp::Sum | ReduceOp::Max | ReduceOp::Min => {}
                ReduceOp::Argmax => return None,
            },
            LogicalNodeKind::If(_) | LogicalNodeKind::Loop(_) | LogicalNodeKind::Call(_) => {
                return None
            }
        }
        for input in node.inputs.iter().copied() {
            if visited.insert(input) {
                prev.insert(input, value);
                queue.push_back(input);
            }
        }
    }
    if !visited.contains(&own) {
        // The result does not depend on the carry: a pass-through, not a fold.
        return None;
    }
    let mut chain = vec![own];
    while *chain.last()? != result {
        let current = *chain.last()?;
        chain.push(*prev.get(&current)?);
    }
    chain.reverse();
    if chain.len() < 2 {
        return None;
    }
    let mut path_nodes: Vec<&LogicalNode> = Vec::with_capacity(chain.len() - 1);
    for window in chain.windows(2) {
        path_nodes.push(producer.get(&window[0]).copied()?);
    }
    let combine = match &path_nodes.first()?.kind {
        LogicalNodeKind::Primitive(application) => match &application.op {
            PrimitiveOp::Primitive(PrimitiveId::Math(op @ (MathOp::Max | MathOp::Min))) => {
                LaneCombine::Extremum(*op)
            }
            PrimitiveOp::Primitive(PrimitiveId::Binary(BinaryOp::Add)) => LaneCombine::Additive,
            _ => return None,
        },
        LogicalNodeKind::Reduction(reduction) => match reduction.op {
            ReduceOp::Sum => LaneCombine::Additive,
            ReduceOp::Max => LaneCombine::Extremum(MathOp::Max),
            ReduceOp::Min => LaneCombine::Extremum(MathOp::Min),
            ReduceOp::Argmax => return None,
        },
        LogicalNodeKind::If(_) | LogicalNodeKind::Loop(_) | LogicalNodeKind::Call(_) => return None,
    };
    let mut rescale_step_with_own = false;
    for node in &path_nodes[1..] {
        let admitted = matches!(
            &node.kind,
            LogicalNodeKind::Primitive(application)
                if matches!(
                    &application.op,
                    PrimitiveOp::Primitive(PrimitiveId::Binary(BinaryOp::Mul | BinaryOp::Sub))
                ) && node.inputs.contains(&own)
        );
        if !admitted {
            return None;
        }
        rescale_step_with_own = true;
    }
    match combine {
        LaneCombine::Extremum(op) => {
            if rescale_step_with_own {
                return None;
            }
            Some(LaneAnalysis::Extremum { op })
        }
        LaneCombine::Additive => {
            if rescale_step_with_own && !(cross_lane && exp_present) {
                return None;
            }
            Some(LaneAnalysis::Additive {
                cross_lane,
                exp_present,
            })
        }
    }
}

/// The value carries of one loop as kernel carries, in carried order:
/// `(initial, body parameter, body update, loop result)` canonical values.
/// Only value carries appear; state carries are storage chains. A sealed
/// program pairs every value slot with a value parameter, a value result,
/// and a loop output (`builder::add_loop`); a mismatch is a contradicted
/// L1 invariant returned as an S1 defect.
pub fn loop_value_carries(
    facts: &OccurrenceFacts<'_>,
    node: &OwnedNodeRef,
    loop_node: &LoopNode,
) -> Result<Vec<OrderedScalarCarry>, crate::failure::CompilerDefect> {
    let defect = |invariant: String| {
        crate::failure::CompilerDefect::new(crate::failure::Package::S1, invariant)
    };
    let logical = minted!(facts.node(node), "a traversed node is minted (S1)");
    let canonical = |value: GraphValueId| {
        minted!(
            facts.canonical_value(OwnedValueRef {
                graph: node.graph,
                value,
            }),
            "a value of a minted node is canonicalized (S1)"
        )
    };
    let mut carries = Vec::new();
    for slot in &loop_node.carried {
        let RegionInput::Value(initial) = slot.initial else {
            continue;
        };
        let parameter = match loop_node.body.parameters.get(slot.body_parameter.index()) {
            Some(RegionParameter::Value { id, .. }) => *id,
            Some(RegionParameter::State { .. }) | None => {
                return Err(defect(format!(
                    "value carry slot {} of {node:?} has no value body parameter",
                    slot.body_parameter.0
                )))
            }
        };
        let update = match loop_node.body.results.get(slot.body_result.index()) {
            Some(RegionResult::Value { id, .. }) => *id,
            Some(RegionResult::State { .. }) | None => {
                return Err(defect(format!(
                    "value carry slot {} of {node:?} has no value body result",
                    slot.body_result.0
                )))
            }
        };
        let result = match logical.outputs.get(slot.loop_result.index()) {
            Some(output) => output.id(),
            None => {
                return Err(defect(format!(
                    "value carry slot {} of {node:?} has no loop result",
                    slot.loop_result.0
                )))
            }
        };
        carries.push(OrderedScalarCarry {
            initial: canonical(initial),
            parameter: canonical(parameter),
            update: canonical(update),
            result: canonical(result),
        });
    }
    Ok(carries)
}

// ---------------------------------------------------------------------------
// Core-produced pattern facts (the declarative structural surface backend
// mapping rules match against)
// ---------------------------------------------------------------------------
//
// A backend rule never walks a logical graph or matches a node kind: every
// structural pattern a rule needs is computed here, once, by core traversal,
// as a declarative fact over opaque ids. The universal rules consume these
// same facts, so universal and backend formation read one truth.

/// The checked capacity of one extent bound: the exact length of a static
/// bound, the runtime extent's capacity for a runtime bound, the constant of
/// a constant symbolic bound. `None` when the bound has no derivable
/// capacity.
fn checked_axis_capacity(facts: &OccurrenceFacts<'_>, bound: &ExtentExpr) -> Option<u64> {
    match bound {
        ExtentExpr::Static(n) => Some(*n),
        ExtentExpr::Runtime(id) => Some(facts.runtime_extent(*id).capacity),
        ExtentExpr::Sym(sym) => u64::try_from(sym.as_constant()?).ok(),
    }
}

/// One segment of the streaming segmentation of a region tree: the launch
/// cuts a streaming realization proposes over. Segments of retained bodies
/// precede the retained step's own segment, matching formation order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamingSegment {
    /// One launch group: a single primitive or reduction, an absorbable
    /// loop with its whole subtree, or a merged capability chain.
    Group(StreamingGroup),
    /// One retained structured step (call, conditional, non-absorbable
    /// loop); its body's segments are streamed separately.
    Retained(OwnedNodeRef),
}

/// One launch group of the streaming segmentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamingGroup {
    /// Every node of the group in pre-order (an absorbable loop first, then
    /// its subtree). Never empty.
    pub nodes: Vec<OwnedNodeRef>,
    pub kind: StreamingGroupKind,
}

/// What the group realizes, declaratively.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamingGroupKind {
    /// A primitive, reduction, ordered loop, or merged capability chain:
    /// serial in every peer; no independent domain to cover concurrently.
    Serial,
    /// An absorbable independent loop (the group root). `concurrent` holds
    /// when every atomic join of the subtree is 32-bit, so linear
    /// participants may cover the axis chain; `reassociate` records the
    /// float atomic `add` rounding freedom such coverage takes;
    /// `runtime_total` holds when the axis chain has a runtime extent (the
    /// dynamic-pull admission).
    IndependentLoop {
        loop_node: OwnedNodeRef,
        axis_chain: Vec<ExtentExpr>,
        concurrent: bool,
        reassociate: bool,
        runtime_total: bool,
    },
}

/// The streaming segmentation of one region tree: every primitive and
/// reduction is its own group; every absorbable loop is one group holding
/// the loop and its whole body; calls, conditionals, and non-absorbable
/// loops are retained with their bodies streamed recursively. Capability
/// chains are merged so no capability value crosses a cut.
pub fn streaming_segments(
    facts: &OccurrenceFacts<'_>,
    root: &OwnedRegionRef,
) -> Vec<StreamingSegment> {
    let mut segments = Vec::new();
    stream_region_segments(facts, root, &mut segments);
    segments
}

fn stream_region_segments(
    facts: &OccurrenceFacts<'_>,
    region: &OwnedRegionRef,
    out: &mut Vec<StreamingSegment>,
) {
    let mut segments: Vec<StreamingSegment> = Vec::new();
    // Pure allocation declarations collected while walking; they have no
    // execution and never form a launch of their own.
    let mut pending_allocs: Vec<OwnedNodeRef> = Vec::new();
    for node in region_nodes(facts, region) {
        match &minted!(facts.node(&node), "a traversed node is minted (S1)").kind {
            LogicalNodeKind::Primitive(application) => {
                // Pure declarations — storage allocations and scalar
                // constants — have no tensor execution of their own; they
                // join the following group, where their values are consumed.
                if matches!(
                    application.op,
                    PrimitiveOp::Primitive(PrimitiveId::TensorAlloc { .. })
                        | PrimitiveOp::Constant(_)
                ) {
                    pending_allocs.push(node);
                    continue;
                }
                if pending_allocs.is_empty() {
                    segments.push(StreamingSegment::Group(StreamingGroup {
                        nodes: vec![node],
                        kind: StreamingGroupKind::Serial,
                    }));
                } else {
                    // The declarations precede this primitive in program
                    // order: one serial group owns them together.
                    let mut nodes = std::mem::take(&mut pending_allocs);
                    nodes.push(node);
                    segments.push(StreamingSegment::Group(StreamingGroup {
                        nodes,
                        kind: StreamingGroupKind::Serial,
                    }));
                }
            }
            LogicalNodeKind::Reduction(_) => {
                let mut nodes = std::mem::take(&mut pending_allocs);
                nodes.push(node);
                segments.push(StreamingSegment::Group(StreamingGroup {
                    nodes,
                    kind: StreamingGroupKind::Serial,
                }));
            }
            LogicalNodeKind::Loop(loop_node) if node_is_absorbable(facts, &node) => {
                let mut nodes = std::mem::take(&mut pending_allocs);
                nodes.extend(subtree_nodes(facts, &node));
                let kind = match loop_node.kind {
                    LoopKind::Independent => independent_loop_kind(facts, &node, &nodes),
                    LoopKind::Ordered => StreamingGroupKind::Serial,
                };
                segments.push(StreamingSegment::Group(StreamingGroup { nodes, kind }));
            }
            LogicalNodeKind::Loop(_) | LogicalNodeKind::If(_) | LogicalNodeKind::Call(_) => {
                for child in child_regions(facts, &node) {
                    stream_region_segments(facts, &child, out);
                }
                segments.push(StreamingSegment::Retained(node));
            }
        }
    }
    if !pending_allocs.is_empty() {
        // A region ending in bare declarations: the last group owns them.
        match segments.last_mut() {
            Some(StreamingSegment::Group(last)) => {
                last.nodes.splice(0..0, pending_allocs);
            }
            _ => {
                segments.push(StreamingSegment::Group(StreamingGroup {
                    nodes: pending_allocs,
                    kind: StreamingGroupKind::Serial,
                }));
            }
        }
    }
    out.extend(merge_capability_chains(facts, segments));
}

/// Whether a canonical value is a view over logical storage: its dataflow
/// is the storage's (state edges and storage routes), never a computed
/// spill.
pub fn storage_backed_view(facts: &OccurrenceFacts<'_>, value: CanonicalValueId) -> bool {
    let GraphValueKind::Tensor {
        source: TensorSource::View(view),
        ..
    } = facts.value_kind(value)
    else {
        return false;
    };
    let producer = facts.members(value)[0];
    let view = facts.view(OwnedViewRef {
        graph: producer.graph,
        view: *view,
    });
    matches!(view.base, ViewBase::Storage(_))
}

/// The group kind of one absorbable independent loop with its subtree
/// `nodes` (see `StreamingGroupKind::IndependentLoop`).
fn independent_loop_kind(
    facts: &OccurrenceFacts<'_>,
    node: &OwnedNodeRef,
    nodes: &[OwnedNodeRef],
) -> StreamingGroupKind {
    let operations: Vec<AtomicOperation> = nodes
        .iter()
        .flat_map(|inner| atomic_operations(facts, inner))
        .collect();
    let concurrent = operations
        .iter()
        .all(|operation| matches!(operation.dtype, DType::F32 | DType::I32 | DType::U32));
    let reassociate = concurrent
        && operations
            .iter()
            .any(|operation| operation.op == AtomicOp::Add && operation.dtype.is_float());
    let axis_chain = independent_axis_chain(facts, node);
    let runtime_total = axis_chain
        .iter()
        .any(|bound| matches!(bound, ExtentExpr::Runtime(_)));
    StreamingGroupKind::IndependentLoop {
        loop_node: node.clone(),
        axis_chain,
        concurrent,
        reassociate,
        runtime_total,
    }
}

/// A capability value (a backend-owned opaque value) cannot cross a launch
/// cut. Within one region's segment sequence, every group between the
/// producer of such a value and its consumer is merged into one serialized
/// group. A retained step between them cannot merge; its interval is left
/// unmerged (a rule that then splits the chain declines rather than propose
/// incompletely).
fn merge_capability_chains(
    facts: &OccurrenceFacts<'_>,
    segments: Vec<StreamingSegment>,
) -> Vec<StreamingSegment> {
    let is_capability = |value: CanonicalValueId| {
        matches!(facts.value_kind(value), GraphValueKind::Capability(_))
    };
    let mut producers: BTreeMap<CanonicalValueId, usize> = BTreeMap::new();
    for (index, segment) in segments.iter().enumerate() {
        let StreamingSegment::Group(group) = segment else {
            continue;
        };
        for node in &group.nodes {
            for output in &minted!(facts.node(node), "a traversed node is minted (S1)").outputs {
                let canonical = minted!(
                    facts.canonical_value(OwnedValueRef {
                        graph: node.graph,
                        value: output.id(),
                    }),
                    "an output of a minted node is canonicalized (S1)"
                );
                if is_capability(canonical) {
                    producers.entry(canonical).or_insert(index);
                }
            }
        }
    }
    let mut intervals: Vec<(usize, usize)> = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        let StreamingSegment::Group(group) = segment else {
            continue;
        };
        for node in &group.nodes {
            for value in node_value_edges(facts, node) {
                let canonical = minted!(
                    facts.canonical_value(value),
                    "a value edge of a minted node is canonicalized (S1)"
                );
                if !is_capability(canonical) {
                    continue;
                }
                if let Some(&from) = producers.get(&canonical) {
                    if from < index {
                        intervals.push((from, index));
                    }
                }
            }
        }
    }
    if intervals.is_empty() {
        return segments;
    }
    intervals.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (low, high) in intervals {
        match merged.last_mut() {
            Some((_, end)) if low <= *end => *end = (*end).max(high),
            _ => merged.push((low, high)),
        }
    }
    let mut slots: Vec<Option<StreamingSegment>> = segments.into_iter().map(Some).collect();
    for (low, high) in merged {
        let mergeable = (low..=high).all(|index| matches!(slots[index], Some(StreamingSegment::Group(_))));
        if !mergeable {
            continue;
        }
        let mut nodes = Vec::new();
        for slot in &mut slots[low..=high] {
            if let Some(StreamingSegment::Group(group)) = slot.take() {
                nodes.extend(group.nodes);
            }
        }
        slots[low] = Some(StreamingSegment::Group(StreamingGroup {
            nodes,
            kind: StreamingGroupKind::Serial,
        }));
    }
    slots.into_iter().flatten().collect()
}

/// The extents of the independent-axis chain rooted at one independent
/// loop: the loop's own bound, then each nested independent loop that is
/// the only node of its enclosing body (the same chain formation binds to
/// kernel axes).
pub fn independent_axis_chain(
    facts: &OccurrenceFacts<'_>,
    node: &OwnedNodeRef,
) -> Vec<ExtentExpr> {
    let mut bounds = Vec::new();
    let mut current = node.clone();
    loop {
        let LogicalNodeKind::Loop(loop_node) =
            &minted!(facts.node(&current), "a traversed node is minted (S1)").kind
        else {
            break;
        };
        if loop_node.kind != LoopKind::Independent {
            break;
        }
        bounds.push(loop_node.range.bound.clone());
        let regions = child_regions(facts, &current);
        let body = region_nodes(facts, &regions[0]);
        let [only] = body.as_slice() else {
            break;
        };
        current = only.clone();
    }
    bounds
}

/// One reduction of a region tree with its semantic shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReductionShape {
    pub node: OwnedNodeRef,
    pub op: ReduceOp,
    pub order: ReductionOrder,
    /// The dtype the fold accumulates in.
    pub accumulator: DType,
}

/// Every reduction of a region tree with its semantic shape, in pre-order.
pub fn reduction_shapes(facts: &OccurrenceFacts<'_>, root: &OwnedRegionRef) -> Vec<ReductionShape> {
    let mut shapes = Vec::new();
    collect_reduction_shapes(facts, root, &mut shapes);
    shapes
}

fn collect_reduction_shapes(
    facts: &OccurrenceFacts<'_>,
    region: &OwnedRegionRef,
    out: &mut Vec<ReductionShape>,
) {
    for node in region_nodes(facts, region) {
        if let LogicalNodeKind::Reduction(reduction) =
            &minted!(facts.node(&node), "a traversed node is minted (S1)").kind
        {
            out.push(ReductionShape {
                node: node.clone(),
                op: reduction.op,
                order: reduction.order,
                accumulator: reduction.accumulator,
            });
        }
        for child in child_regions(facts, &node) {
            collect_reduction_shapes(facts, &child, out);
        }
    }
}

/// One ordered loop of a region tree whose carried lanes match the
/// structural scan rule, with its derived scan state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScannableLoop {
    pub node: OwnedNodeRef,
    pub state: ScanState,
}

/// Every scannable ordered loop of a region tree, in pre-order.
pub fn scannable_loops(facts: &OccurrenceFacts<'_>, root: &OwnedRegionRef) -> Vec<ScannableLoop> {
    let mut found = Vec::new();
    collect_scannable_loops(facts, root, &mut found);
    found
}

fn collect_scannable_loops(
    facts: &OccurrenceFacts<'_>,
    region: &OwnedRegionRef,
    out: &mut Vec<ScannableLoop>,
) {
    for node in region_nodes(facts, region) {
        if let LogicalNodeKind::Loop(loop_node) =
            &minted!(facts.node(&node), "a traversed node is minted (S1)").kind
        {
            if loop_node.kind == LoopKind::Ordered {
                if let Some(state) = scan_state(facts, &node) {
                    out.push(ScannableLoop {
                        node: node.clone(),
                        state,
                    });
                }
            }
        }
        for child in child_regions(facts, &node) {
            collect_scannable_loops(facts, &child, out);
        }
    }
}

/// One absorbable independent loop of a region tree that admits a tiled
/// traversal: the tile target, with its independent axis chain and the
/// checked capacity of its own axis (the tile domain's upper bound). A
/// candidate whose own-axis capacity is not derivable, or is zero, is not
/// admitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileCandidate {
    pub node: OwnedNodeRef,
    pub axis_chain: Vec<ExtentExpr>,
    pub axis_capacity: u64,
}

/// Every tile candidate of a region tree, in pre-order.
pub fn tile_candidates(facts: &OccurrenceFacts<'_>, root: &OwnedRegionRef) -> Vec<TileCandidate> {
    let mut found = Vec::new();
    collect_tile_candidates(facts, root, &mut found);
    found
}

fn collect_tile_candidates(
    facts: &OccurrenceFacts<'_>,
    region: &OwnedRegionRef,
    out: &mut Vec<TileCandidate>,
) {
    for node in region_nodes(facts, region) {
        if let LogicalNodeKind::Loop(loop_node) =
            &minted!(facts.node(&node), "a traversed node is minted (S1)").kind
        {
            if loop_node.kind == LoopKind::Independent && node_is_absorbable(facts, &node) {
                if let Some(axis_capacity) = checked_axis_capacity(facts, &loop_node.range.bound) {
                    if axis_capacity > 0 {
                        out.push(TileCandidate {
                            node: node.clone(),
                            axis_chain: independent_axis_chain(facts, &node),
                            axis_capacity,
                        });
                    }
                }
            }
        }
        for child in child_regions(facts, &node) {
            collect_tile_candidates(facts, &child, out);
        }
    }
}

/// One capability intrinsic application of a region tree, with the node that
/// applies it and the typed intrinsic signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityUse {
    pub node: OwnedNodeRef,
    pub intrinsic: IntrinsicId,
}

/// Every capability intrinsic application of a region tree, in pre-order.
pub fn capability_uses(facts: &OccurrenceFacts<'_>, root: &OwnedRegionRef) -> Vec<CapabilityUse> {
    let mut uses = Vec::new();
    collect_capability_uses(facts, root, &mut uses);
    uses
}

fn collect_capability_uses(
    facts: &OccurrenceFacts<'_>,
    region: &OwnedRegionRef,
    out: &mut Vec<CapabilityUse>,
) {
    for node in region_nodes(facts, region) {
        if let LogicalNodeKind::Primitive(application) =
            &minted!(facts.node(&node), "a traversed node is minted (S1)").kind
        {
            if let PrimitiveOp::Capability(intrinsic) = &application.op {
                out.push(CapabilityUse {
                    node: node.clone(),
                    intrinsic: intrinsic.clone(),
                });
            }
        }
        for child in child_regions(facts, &node) {
            collect_capability_uses(facts, &child, out);
        }
    }
}

/// One canonical value produced by one top-level node of a region and
/// consumed by a different top-level node of the same region: a
/// whole-result dependency between the phases of a cooperative launch (the
/// producer phase fully completes before the consumer phase reads it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseCrossing {
    pub value: CanonicalValueId,
}

/// Every whole-result crossing between the top-level nodes of one region,
/// in consumer order, deduplicated.
pub fn phase_crossings(facts: &OccurrenceFacts<'_>, region: &OwnedRegionRef) -> Vec<PhaseCrossing> {
    let nodes = region_nodes(facts, region);
    let mut producers: BTreeMap<GraphValueId, OwnedNodeRef> = BTreeMap::new();
    for node in &nodes {
        for output in &minted!(facts.node(node), "a traversed node is minted (S1)").outputs {
            producers.insert(output.id(), node.clone());
        }
    }
    let mut crossings: Vec<PhaseCrossing> = Vec::new();
    let mut seen: BTreeSet<CanonicalValueId> = BTreeSet::new();
    for node in &nodes {
        for input in &minted!(facts.node(node), "a traversed node is minted (S1)").inputs {
            // Only a value another top-level node produced crosses a phase
            // boundary; values from region parameters enter from outside.
            if let Some(producer) = producers.get(input) {
                if producer != node {
                    let canonical = minted!(
                        facts.canonical_value(OwnedValueRef {
                            graph: node.graph,
                            value: *input,
                        }),
                        "an input of a minted node is canonicalized (S1)"
                    );
                    if seen.insert(canonical) {
                        crossings.push(PhaseCrossing { value: canonical });
                    }
                }
            }
        }
    }
    crossings
}


impl seismic_lang::logical::IdIndex for BlockId {
    fn from_index(index: usize) -> Self {
        BlockId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

impl seismic_lang::logical::IdIndex for PlanParamId {
    fn from_index(index: usize) -> Self {
        PlanParamId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

/// The sole constructor of `ClosedStrategyShape` (package S1).
pub struct StrategyFormer;

impl StrategyFormer {
    /// Form one complete strategy shape from one proposal. A proposal that
    /// cannot be completed is a compiler defect of the proposing rule's
    /// owner; optional rules must decline before proposing.
    pub fn form(
        facts: &OccurrenceFacts<'_>,
        profile: &EffectiveTargetProfile,
        proposal: MappingProposal,
    ) -> Result<ClosedStrategyShape, crate::failure::CompilerDefect> {
        crate::formation::strategy::form(facts, profile, proposal)
    }
}
