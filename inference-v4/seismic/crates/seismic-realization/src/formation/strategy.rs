//! Strategy formation (package S1).
//!
//! Two things live here and nothing else: the core-owned universal rule
//! catalog (`universal_rules`) and the one structured former (`form`) behind
//! `StrategyFormer::form`.
//!
//! The universal rules yield, for every occurrence and logical alternative,
//! at least one complete proposal whose local resource use is independent of
//! total logical extent (the serial/streaming point realization) plus a
//! bounded linear-participant peer. They never absorb occurrences and never
//! choose an algorithm other than `Universal`.
//!
//! The former runs one recursive traversal over the owned graphs (the root
//! alternative plus every absorbed callee, inlined after the call that
//! reaches it) and consumes exactly once: every node, every region result,
//! every state edge, every safety obligation. From the proposal's placement
//! it derives the structured schedule, the participant map of every block,
//! every block cut, and every value edge crossing a cut. It seals only when
//! the consumed sets equal their exact expected sets; any other outcome is a
//! compiler defect of the proposing rule's owner.
//!
//! Forbidden here: backend graph walkers, storage allocation, opcode
//! emission, pending-set repair.

use crate::failure::{CompilerDefect, Package};
use crate::ids::{
    BlockId, CanonicalLeafId, CanonicalStorageId, CanonicalValueId, ObligationRef, OwnedGraphKey,
    OwnedNodeRef, OwnedOccurrence, OwnedRegionRef, OwnedRegionResultRef, OwnedStateRef,
    OwnedStorageRef, OwnedValueRef, OwnedViewRef,
};
use crate::occurrence::{InstantiatedResult, OccurrenceFacts};
use crate::residence::{Replication, StorageScope};
use crate::strategy::{
    capability_uses, child_regions, loop_value_carries, node_is_absorbable, node_state_inputs,
    node_value_edges, owned_graphs, phase_domain, region_nodes, root_region, storage_backed_view,
    streaming_segments, subtree_nodes, AlgorithmChoice, AxisBinding, BlockCut, CarryEdge,
    ClosedStrategyShape, CrossEdge, EdgeEnd, ExecutorGuardPredicate, JoinEdge, LaunchGroup,
    LaunchProposal, LocalResidenceRequirement, MappingProposal, MappingRule, NodePlacement,
    NumericalChoice, ObligationDisposition, OwnershipProposal, ParticipantMap, ParticipantPolicy,
    RuleName, RuleQuery, ShapeSchedule, ShapeStep, StreamingGroupKind, StreamingSegment,
    TuningDeclaration, TuningRef,
};
use crate::target::EffectiveTargetProfile;
use seismic_lang::intrinsics::IntrinsicId;
use seismic_lang::logical::value::{GraphValueKind, TensorSource};
use seismic_lang::logical::{
    GraphValueId, IdVec, JoinSlot, LogicalNodeKind, PrimitiveOp, RegionInput, RegionParameter,
    RegionResult, RegionResultId, SafetyObligation, SliceAxis, StateTokenId, ViewTransform,
};
use seismic_lang::sir::{Literal, LoopKind};
use seismic_lang::intrinsics::PrimitiveId;
use seismic_lang::types::{ExtentExpr, NonEmpty, ValuePath, ValueType};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// The universal rule catalog
// ---------------------------------------------------------------------------

/// The core-owned universal rules. Every backend catalog is extended with
/// these by `form_plan_space`; a backend cannot omit them.
pub(crate) fn universal_rules() -> Vec<Box<dyn MappingRule>> {
    vec![Box::new(UniversalStreaming), Box::new(UniversalPointSerial)]
}

const STREAMING: RuleName = "universal.streaming";
const POINT_SERIAL: RuleName = "universal.point-serial";

/// Every primitive and reduction of a region is its own launch; every
/// absorbable loop is one launch holding the loop and its whole body; calls,
/// conditionals, and non-absorbable loops are retained with their bodies
/// streamed recursively. Peers:
///
/// - the all-serial feasibility witness (always; the serialized atomic
///   strategy for loops joining storage atomically);
/// - the linear-participant peer: every independent loop whose atomic joins
///   (if any) are all 32-bit gets `Linear` participants over its axis chain
///   (the concurrent atomic strategy, recording `Reassociate` for float
///   `add`); a narrow-float atomic join admits no concurrent peer;
/// - the dynamic-pull peer: independent loops whose axis chain has a runtime
///   extent get `DynamicPull` participants instead of `Linear`.
///
/// A peer identical to an earlier one is not emitted.
struct UniversalStreaming;

/// Every maximal run of absorbable nodes of a region is one serial launch;
/// only retained calls, conditionals, and non-absorbable loops split runs.
/// The point witness: one participant, bounded local resources.
struct UniversalPointSerial;

#[derive(Clone, Debug, PartialEq, Eq)]
enum GroupShape {
    /// A single primitive or reduction, a merged run, an ordered loop, or an
    /// independent loop with a narrow-float atomic join: serial in every
    /// peer.
    Serialized,
    /// An independent loop admitting concurrent participants: `Linear` in
    /// the linear peer, `DynamicPull` in the pull peer when `runtime_total`,
    /// serial in the witness. `numerical` is the concurrent peers' recorded
    /// freedom (`Reassociate` on the loop for a float atomic `add`).
    IndependentLoop {
        numerical: Vec<NumericalChoice>,
        runtime_total: bool,
    },
}

struct GroupDraft {
    /// Every node of the group, pre-order (top-level nodes carry their
    /// subtrees).
    nodes: Vec<OwnedNodeRef>,
    shape: GroupShape,
}

#[derive(Default)]
struct Drafts {
    groups: Vec<GroupDraft>,
    retained: Vec<OwnedNodeRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Peer {
    Serial,
    Linear,
    DynamicPull,
}

impl MappingRule for UniversalStreaming {
    fn name(&self) -> RuleName {
        STREAMING
    }

    fn propose(&self, query: &RuleQuery<'_, '_>) -> Vec<MappingProposal> {
        let key = OwnedGraphKey {
            occurrence: query.occurrence,
            logical_alternative: query.logical_alternative,
        };
        let Some(required) = supported_capability_uses(query, key) else {
            return Vec::new();
        };
        let mut drafts = Drafts::default();
        for segment in streaming_segments(query.facts, &root_region(key)) {
            match segment {
                StreamingSegment::Group(group) => {
                    let shape = match &group.kind {
                        StreamingGroupKind::Serial => GroupShape::Serialized,
                        StreamingGroupKind::IndependentLoop {
                            loop_node,
                            concurrent,
                            reassociate,
                            runtime_total,
                            ..
                        } => {
                            if !*concurrent {
                                GroupShape::Serialized
                            } else {
                                let numerical = if *reassociate {
                                    vec![NumericalChoice::Reassociate {
                                        node: loop_node.clone(),
                                    }]
                                } else {
                                    Vec::new()
                                };
                                GroupShape::IndependentLoop {
                                    numerical,
                                    runtime_total: *runtime_total,
                                }
                            }
                        }
                    };
                    drafts.groups.push(GroupDraft {
                        nodes: group.nodes,
                        shape,
                    });
                }
                StreamingSegment::Retained(node) => drafts.retained.push(node),
            }
        }
        let serial = drafts.proposal(STREAMING, key, required.clone(), Peer::Serial, query.profile);
        let linear = drafts.proposal(STREAMING, key, required.clone(), Peer::Linear, query.profile);
        let pull = drafts.proposal(STREAMING, key, required, Peer::DynamicPull, query.profile);
        let mut proposals = vec![serial];
        // A linear peer that declares no participant parameter is the
        // serial witness again; a pull peer with no pulled launch is the
        // linear peer again. Neither is a second proposal.
        if !linear.tuning.is_empty() {
            let pulls = pull
                .launches
                .iter()
                .any(|launch| matches!(launch.participants, ParticipantPolicy::DynamicPull { .. }));
            proposals.push(linear);
            if pulls {
                proposals.push(pull);
            }
        }
        proposals
    }
}

impl MappingRule for UniversalPointSerial {
    fn name(&self) -> RuleName {
        POINT_SERIAL
    }

    fn propose(&self, query: &RuleQuery<'_, '_>) -> Vec<MappingProposal> {
        let key = OwnedGraphKey {
            occurrence: query.occurrence,
            logical_alternative: query.logical_alternative,
        };
        let Some(required) = supported_capability_uses(query, key) else {
            return Vec::new();
        };
        let mut drafts = Drafts::default();
        point_region(query.facts, &root_region(key), &mut drafts);
        vec![drafts.proposal(POINT_SERIAL, key, required, Peer::Serial, query.profile)]
    }
}

/// Every capability intrinsic the alternative's graph applies, when the
/// effective target profile authorizes all of them. `None` declines: logical
/// construction already removed alternatives requiring unsupported
/// capabilities, so a universal rule never proposes for one.
fn supported_capability_uses(
    query: &RuleQuery<'_, '_>,
    key: OwnedGraphKey,
) -> Option<BTreeSet<IntrinsicId>> {
    let mut required = BTreeSet::new();
    for applied in capability_uses(query.facts, &root_region(key)) {
        if !query.profile.effective_signatures.contains(&applied.intrinsic) {
            return None;
        }
        required.insert(applied.intrinsic);
    }
    Some(required)
}

/// The point-serial walk of one region (see `UniversalPointSerial`).
fn point_region(facts: &OccurrenceFacts<'_>, region: &OwnedRegionRef, drafts: &mut Drafts) {
    let mut run: Vec<OwnedNodeRef> = Vec::new();
    for node in region_nodes(facts, region) {
        if node_is_absorbable(facts, &node) {
            run.extend(subtree_nodes(facts, &node));
            continue;
        }
        if !run.is_empty() {
            drafts.groups.push(GroupDraft {
                nodes: std::mem::take(&mut run),
                shape: GroupShape::Serialized,
            });
        }
        for child in child_regions(facts, &node) {
            point_region(facts, &child, drafts);
        }
        drafts.retained.push(node);
    }
    if !run.is_empty() {
        drafts.groups.push(GroupDraft {
            nodes: run,
            shape: GroupShape::Serialized,
        });
    }
}

impl Drafts {
    fn proposal(
        &self,
        rule: RuleName,
        key: OwnedGraphKey,
        required_intrinsics: BTreeSet<IntrinsicId>,
        peer: Peer,
        profile: &EffectiveTargetProfile,
    ) -> MappingProposal {
        let mut placement = BTreeMap::new();
        let mut launches = Vec::new();
        let mut tuning = Vec::new();
        for (index, group) in self.groups.iter().enumerate() {
            let launch = LaunchGroup(index as u32);
            for node in &group.nodes {
                placement.insert(node.clone(), NodePlacement::Launch(launch));
            }
            let declare = |tuning: &mut Vec<TuningDeclaration>| {
                let reference = TuningRef(tuning.len() as u32);
                tuning.push(TuningDeclaration {
                    name: format!("participants.launch{index}"),
                    lower: 1,
                    upper: profile.limits.max_participants,
                });
                reference
            };
            let (participants, numerical) = match (peer, &group.shape) {
                (Peer::Serial, _) | (_, GroupShape::Serialized) => {
                    (ParticipantPolicy::Serial, Vec::new())
                }
                (
                    Peer::Linear,
                    GroupShape::IndependentLoop { numerical, .. },
                )
                | (
                    Peer::DynamicPull,
                    GroupShape::IndependentLoop {
                        numerical,
                        runtime_total: false,
                    },
                ) => (
                    ParticipantPolicy::Linear {
                        participants: declare(&mut tuning),
                    },
                    numerical.clone(),
                ),
                (
                    Peer::DynamicPull,
                    GroupShape::IndependentLoop {
                        numerical,
                        runtime_total: true,
                    },
                ) => (
                    ParticipantPolicy::DynamicPull {
                        participants: declare(&mut tuning),
                    },
                    numerical.clone(),
                ),
            };
            launches.push(LaunchProposal {
                participants,
                algorithm: AlgorithmChoice::Universal,
                local_residences: Vec::new(),
                numerical,
            });
        }
        for node in &self.retained {
            placement.insert(node.clone(), NodePlacement::Retained);
        }
        MappingProposal {
            rule,
            ownership: OwnershipProposal {
                root: OwnedOccurrence {
                    occurrence: key.occurrence,
                    logical_alternative: key.logical_alternative,
                },
                absorbed: BTreeMap::new(),
            },
            placement,
            launches: IdVec::new(launches),
            required_intrinsics,
            tuning,
        }
    }
}

// ---------------------------------------------------------------------------
// The former
// ---------------------------------------------------------------------------

pub(crate) fn form(
    facts: &OccurrenceFacts<'_>,
    profile: &EffectiveTargetProfile,
    proposal: MappingProposal,
) -> Result<ClosedStrategyShape, CompilerDefect> {
    Former::new(facts, profile, &proposal)?.form()
}

/// Who produces one canonical value class within the strategy.
#[derive(Clone, Debug)]
enum Producer {
    /// A root boundary input leaf of the strategy's root occurrence.
    RootBoundary,
    /// An owned node: the class producer is one of its outputs, or one of
    /// its child regions' parameters (binder, carried parameter); or a
    /// retained call whose callee produces the class outside the strategy.
    Node(OwnedNodeRef),
}

/// One state edge of the owned graphs.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum StateEdge {
    NodeInput(OwnedNodeRef, StateTokenId),
    NodeOutput(OwnedNodeRef, StateTokenId),
    RegionResult(OwnedRegionResultRef, StateTokenId),
}

/// Where one state token of an owned graph originates.
#[derive(Clone, Debug)]
enum StateOrigin {
    Node(OwnedNodeRef),
    /// A root region state parameter (an entry state of the graph).
    RootParameter,
    /// A nested region state parameter of the named control node's region.
    RegionParameter(OwnedNodeRef),
}

/// How one obligation was disposed, before the block is known to K1.
enum Disposed {
    Static(String),
    Kernel,
    Executor(ExecutorGuardPredicate),
}

struct BlockDraft {
    id: BlockId,
    group: LaunchGroup,
    /// The nodes placed directly in the block's region sequence (their
    /// subtrees follow them in `nodes`).
    top: Vec<OwnedNodeRef>,
    nodes: Vec<OwnedNodeRef>,
    executor_guards: Vec<(ObligationRef, ExecutorGuardPredicate)>,
    kernel_guards: Vec<(ObligationRef, SafetyObligation)>,
}

struct Former<'f, 'l> {
    facts: &'f OccurrenceFacts<'l>,
    profile: &'f EffectiveTargetProfile,
    proposal: &'f MappingProposal,
    root_key: OwnedGraphKey,
    owned: Vec<OwnedGraphKey>,
    /// Owned call nodes whose callee occurrence is absorbed, with the callee
    /// graph key.
    absorbed_calls: BTreeMap<OwnedNodeRef, OwnedGraphKey>,
    // Exact expected sets.
    expected_nodes: BTreeSet<OwnedNodeRef>,
    expected_results: BTreeSet<OwnedRegionResultRef>,
    expected_obligations: BTreeSet<ObligationRef>,
    expected_state_edges: BTreeSet<StateEdge>,
    regions: Vec<OwnedRegionRef>,
    // Derived tables.
    producers: BTreeMap<CanonicalValueId, Producer>,
    constants: BTreeMap<CanonicalValueId, i64>,
    state_origins: BTreeMap<(OwnedGraphKey, StateTokenId), StateOrigin>,
    // Construction state.
    node_end: BTreeMap<OwnedNodeRef, EdgeEnd>,
    blocks: BTreeMap<BlockId, BlockDraft>,
    next_block: u32,
    group_block: BTreeMap<LaunchGroup, BlockId>,
    obligations: BTreeMap<ObligationRef, ObligationDisposition>,
    /// Region result -> the end that owns (consumes) it.
    result_owner: BTreeMap<OwnedRegionResultRef, EdgeEnd>,
    consumed_nodes: BTreeSet<OwnedNodeRef>,
    consumed_state_edges: BTreeSet<StateEdge>,
}

impl<'f, 'l> Former<'f, 'l> {
    fn new(
        facts: &'f OccurrenceFacts<'l>,
        profile: &'f EffectiveTargetProfile,
        proposal: &'f MappingProposal,
    ) -> Result<Former<'f, 'l>, CompilerDefect> {
        let root = proposal.ownership.root;
        let root_key = OwnedGraphKey {
            occurrence: root.occurrence,
            logical_alternative: root.logical_alternative,
        };
        let mut former = Former {
            facts,
            profile,
            proposal,
            root_key,
            owned: Vec::new(),
            absorbed_calls: BTreeMap::new(),
            expected_nodes: BTreeSet::new(),
            expected_results: BTreeSet::new(),
            expected_obligations: BTreeSet::new(),
            expected_state_edges: BTreeSet::new(),
            regions: Vec::new(),
            producers: BTreeMap::new(),
            constants: BTreeMap::new(),
            state_origins: BTreeMap::new(),
            node_end: BTreeMap::new(),
            blocks: BTreeMap::new(),
            next_block: 0,
            group_block: BTreeMap::new(),
            obligations: BTreeMap::new(),
            result_owner: BTreeMap::new(),
            consumed_nodes: BTreeSet::new(),
            consumed_state_edges: BTreeSet::new(),
        };
        former.validate_ownership()?;
        for key in former.owned.clone() {
            former.collect_region(key, &root_region(key), None)?;
        }
        former.record_retained_call_producers()?;
        former.validate_placement()?;
        former.validate_launches()?;
        Ok(former)
    }

    fn defect(&self, invariant: impl std::fmt::Display) -> CompilerDefect {
        CompilerDefect::new(
            Package::S1,
            format!("rule `{}`: {invariant}", self.proposal.rule),
        )
    }

    // -- ownership --------------------------------------------------------

    fn validate_ownership(&mut self) -> Result<(), CompilerDefect> {
        let root = self.proposal.ownership.root;
        let alternatives = self.facts.occurrence(root.occurrence).alternatives.len();
        if root.logical_alternative as usize >= alternatives {
            return Err(self.defect(format!(
                "root occurrence#{} has {alternatives} alternative(s) but the proposal names alternative {}",
                root.occurrence.0, root.logical_alternative
            )));
        }
        for (occurrence, alternative) in &self.proposal.ownership.absorbed {
            let alternatives = self.facts.occurrence(*occurrence).alternatives.len();
            if *alternative as usize >= alternatives {
                return Err(self.defect(format!(
                    "absorbed occurrence#{} has {alternatives} alternative(s) but the proposal names alternative {alternative}",
                    occurrence.0
                )));
            }
            if *occurrence == root.occurrence {
                return Err(self.defect(format!(
                    "the root occurrence#{} is also listed as absorbed",
                    occurrence.0
                )));
            }
        }
        self.owned = owned_graphs(self.facts, &self.proposal.ownership);
        for occurrence in self.proposal.ownership.absorbed.keys() {
            if !self.owned.iter().any(|key| key.occurrence == *occurrence) {
                return Err(self.defect(format!(
                    "absorbed occurrence#{} is not reached by any owned call",
                    occurrence.0
                )));
            }
        }
        for key in &self.owned {
            for (call, occurrence) in &self.facts.alternative(*key).calls {
                if let Some(alternative) = self.proposal.ownership.absorbed.get(occurrence) {
                    self.absorbed_calls.insert(
                        call.clone(),
                        OwnedGraphKey {
                            occurrence: *occurrence,
                            logical_alternative: *alternative,
                        },
                    );
                }
            }
        }
        Ok(())
    }

    // -- expected sets and tables ----------------------------------------

    fn collect_region(
        &mut self,
        key: OwnedGraphKey,
        region: &OwnedRegionRef,
        owner: Option<&OwnedNodeRef>,
    ) -> Result<(), CompilerDefect> {
        let facts = self.facts;
        let logical_region = facts.region(region)?;
        self.regions.push(region.clone());
        for parameter in &logical_region.parameters {
            match parameter {
                RegionParameter::Value { id, .. } => {
                    let owned = OwnedValueRef {
                        graph: key,
                        value: *id,
                    };
                    let canonical = facts.canonical_value(owned)?;
                    match owner {
                        Some(node) => {
                            if facts.members(canonical)[0] == owned {
                                self.producers
                                    .insert(canonical, Producer::Node(node.clone()));
                            }
                        }
                        None if key == self.root_key => {
                            self.producers.insert(canonical, Producer::RootBoundary);
                        }
                        None => {
                            // An absorbed callee's root parameter is aliased to
                            // the caller's argument, whose producer is owned.
                        }
                    }
                }
                RegionParameter::State { id, .. } => {
                    let origin = match owner {
                        Some(node) => StateOrigin::RegionParameter(node.clone()),
                        None => StateOrigin::RootParameter,
                    };
                    self.state_origins.insert((key, *id), origin);
                }
            }
        }
        for (ordinal, result) in logical_region.results.iter().enumerate() {
            let reference = OwnedRegionResultRef {
                region: region.clone(),
                ordinal: ordinal as u32,
            };
            if let RegionResult::State { id, .. } = result {
                self.expected_state_edges
                    .insert(StateEdge::RegionResult(reference.clone(), *id));
            }
            self.expected_results.insert(reference);
        }
        for node in region_nodes(facts, region) {
            let logical = facts.node(&node)?;
            self.expected_nodes.insert(node.clone());
            for index in 0..logical.safety.len() {
                self.expected_obligations.insert(ObligationRef {
                    node: node.clone(),
                    index: index as u32,
                });
            }
            for token in node_state_inputs(logical) {
                self.expected_state_edges
                    .insert(StateEdge::NodeInput(node.clone(), *token));
            }
            for token in &logical.state_outputs {
                self.expected_state_edges
                    .insert(StateEdge::NodeOutput(node.clone(), token.id));
                self.state_origins
                    .insert((key, token.id), StateOrigin::Node(node.clone()));
            }
            for output in &logical.outputs {
                let owned = OwnedValueRef {
                    graph: key,
                    value: output.id(),
                };
                let canonical = facts.canonical_value(owned)?;
                if facts.members(canonical)[0] == owned {
                    self.producers
                        .insert(canonical, Producer::Node(node.clone()));
                }
            }
            if let LogicalNodeKind::Primitive(application) = &logical.kind {
                if let PrimitiveOp::Constant(Literal::Int(value)) = &application.op {
                    if let Some(output) = logical.outputs.first() {
                        let canonical = facts.canonical_value(OwnedValueRef {
                            graph: key,
                            value: output.id(),
                        })?;
                        self.constants.insert(canonical, *value);
                    }
                }
            }
            for child in child_regions(facts, &node) {
                self.collect_region(key, &child, Some(&node))?;
            }
        }
        Ok(())
    }

    /// A retained call's outputs are produced by its callee outside the
    /// strategy: the call node is their producer end.
    fn record_retained_call_producers(&mut self) -> Result<(), CompilerDefect> {
        let facts = self.facts;
        for node in self.expected_nodes.clone() {
            if self.absorbed_calls.contains_key(&node) {
                continue;
            }
            let logical = facts.node(&node)?;
            if !matches!(logical.kind, LogicalNodeKind::Call(_)) {
                continue;
            }
            for output in &logical.outputs {
                let canonical = facts.canonical_value(OwnedValueRef {
                    graph: node.graph,
                    value: output.id(),
                })?;
                self.producers
                    .entry(canonical)
                    .or_insert_with(|| Producer::Node(node.clone()));
            }
        }
        Ok(())
    }

    // -- proposal validation ---------------------------------------------

    fn validate_placement(&self) -> Result<(), CompilerDefect> {
        for node in &self.expected_nodes {
            if !self.proposal.placement.contains_key(node) {
                return Err(self.defect(format!("owned node {node:?} has no placement")));
            }
        }
        for (node, placement) in &self.proposal.placement {
            if !self.expected_nodes.contains(node) {
                return Err(self.defect(format!(
                    "placement names {node:?}, which the strategy does not own"
                )));
            }
            let logical = self.facts.node(node)?;
            match (&logical.kind, placement) {
                (LogicalNodeKind::Primitive(_) | LogicalNodeKind::Reduction(_), NodePlacement::Retained) => {
                    return Err(self.defect(format!(
                        "{node:?} is a primitive or reduction and cannot be a retained step"
                    )));
                }
                (LogicalNodeKind::Call(_), NodePlacement::Launch(_)) => {
                    if !self.absorbed_calls.contains_key(node) {
                        return Err(self.defect(format!(
                            "call {node:?} is placed in a launch but its occurrence is not absorbed"
                        )));
                    }
                }
                (LogicalNodeKind::Call(_), NodePlacement::Retained) => {
                    if self.absorbed_calls.contains_key(node) {
                        return Err(self.defect(format!(
                            "call {node:?} is retained but its occurrence is absorbed"
                        )));
                    }
                }
                (LogicalNodeKind::Primitive(_) | LogicalNodeKind::Reduction(_), NodePlacement::Launch(_))
                | (LogicalNodeKind::If(_) | LogicalNodeKind::Loop(_), _) => {}
            }
            if let NodePlacement::Launch(group) = placement {
                if self.proposal.launches.get(*group).is_none() {
                    return Err(self.defect(format!(
                        "{node:?} is placed in launch group {} which the proposal does not declare",
                        group.0
                    )));
                }
            }
        }
        Ok(())
    }

    fn tuning_declaration(&self, reference: TuningRef, role: &str) -> Result<(), CompilerDefect> {
        let Some(declaration) = self.proposal.tuning.get(reference.0 as usize) else {
            return Err(self.defect(format!(
                "{role} names tuning parameter {} but the proposal declares {}",
                reference.0,
                self.proposal.tuning.len()
            )));
        };
        if declaration.lower < 1 || declaration.lower > declaration.upper {
            return Err(self.defect(format!(
                "tuning parameter `{}` has the empty or non-positive domain [{}, {}]",
                declaration.name, declaration.lower, declaration.upper
            )));
        }
        Ok(())
    }

    fn validate_launches(&self) -> Result<(), CompilerDefect> {
        for launch in self.proposal.launches.iter() {
            match &launch.participants {
                ParticipantPolicy::Serial => {}
                ParticipantPolicy::Linear { participants } => {
                    self.tuning_declaration(*participants, "a linear participant policy")?
                }
                ParticipantPolicy::Cooperative { width, family } => {
                    self.tuning_declaration(*width, "a cooperative participant policy")?;
                    if !self.profile.effective_signatures.contains(family) {
                        return Err(self.defect(format!(
                            "cooperative family `{family}` is not an effective signature of the target"
                        )));
                    }
                }
                ParticipantPolicy::DynamicPull { participants } => {
                    self.tuning_declaration(*participants, "a dynamic-pull participant policy")?
                }
                ParticipantPolicy::GridCooperative { participants } => {
                    self.tuning_declaration(*participants, "a grid-cooperative participant policy")?;
                    if self.profile.limits.cooperative_grid.is_none() {
                        return Err(self.defect(
                            "a grid-cooperative launch is proposed but the target has no cooperative grid facility",
                        ));
                    }
                }
            }
            match &launch.algorithm {
                AlgorithmChoice::Universal | AlgorithmChoice::Reduction(_) => {}
                AlgorithmChoice::Blocked { tile } => {
                    self.tuning_declaration(*tile, "a blocked algorithm")?
                }
                AlgorithmChoice::Intrinsic(intrinsic) => {
                    if !self.profile.effective_signatures.contains(intrinsic) {
                        return Err(self.defect(format!(
                            "algorithm intrinsic `{intrinsic}` is not an effective signature of the target"
                        )));
                    }
                }
            }
        }
        for intrinsic in self.required_intrinsics()? {
            if !self.profile.effective_signatures.contains(&intrinsic) {
                return Err(self.defect(format!(
                    "required intrinsic `{intrinsic}` is not an effective signature of the target"
                )));
            }
        }
        Ok(())
    }

    /// The proposal's declared intrinsics plus every capability the owned
    /// nodes apply.
    fn required_intrinsics(&self) -> Result<BTreeSet<IntrinsicId>, CompilerDefect> {
        let mut required = self.proposal.required_intrinsics.clone();
        for node in &self.expected_nodes {
            if let LogicalNodeKind::Primitive(application) =
                &self.facts.node(node)?.kind
            {
                if let PrimitiveOp::Capability(intrinsic) = &application.op {
                    required.insert(intrinsic.clone());
                }
            }
        }
        Ok(required)
    }

    // -- structured formation --------------------------------------------

    fn form(mut self) -> Result<ClosedStrategyShape, CompilerDefect> {
        let root = root_region(self.root_key);
        let schedule = self.form_region(&root)?;
        self.seal_sets()?;
        let (edges, inputs, outputs) = self.derive_edges()?;
        let results = self.derive_results()?;
        let blocks = self.finish_blocks(&inputs, &outputs)?;
        let required_intrinsics = self.required_intrinsics()?;
        Ok(ClosedStrategyShape::seal(
            self.proposal.ownership.root,
            self.proposal.ownership.absorbed.clone(),
            self.proposal.rule,
            schedule,
            blocks,
            self.obligations,
            results,
            edges,
            IdVec::new(self.proposal.tuning.clone()),
            required_intrinsics,
            self.regions,
        ))
    }

    fn placement_of(&self, node: &OwnedNodeRef) -> Result<NodePlacement, CompilerDefect> {
        self.proposal
            .placement
            .get(node)
            .copied()
            .ok_or_else(|| self.defect(format!("owned node {node:?} has no placement")))
    }

    /// The region's nodes with every absorbed callee's root region inlined
    /// immediately after the call that reaches it.
    fn linearized(&self, region: &OwnedRegionRef) -> Vec<OwnedNodeRef> {
        let mut out = Vec::new();
        for node in region_nodes(self.facts, region) {
            let callee = self.absorbed_calls.get(&node).copied();
            out.push(node);
            if let Some(callee) = callee {
                out.extend(self.linearized(&root_region(callee)));
            }
        }
        out
    }

    fn form_region(&mut self, region: &OwnedRegionRef) -> Result<ShapeSchedule, CompilerDefect> {
        let mut steps = Vec::new();
        let mut open: Option<BlockDraft> = None;
        for node in self.linearized(region) {
            match self.placement_of(&node)? {
                NodePlacement::Launch(group) => {
                    // The open block is carried by value: either the run
                    // continues the draft already open for this group, or a
                    // fresh draft is constructed for it. Both arms yield a
                    // structurally open block; no positional Option read.
                    let mut block = match open.take() {
                        Some(block) if block.group == group => block,
                        prior => {
                            if let Some(block) = prior {
                                self.close_block(block, &mut steps);
                            }
                            if self.group_block.contains_key(&group) {
                                return Err(self.defect(format!(
                                    "launch group {} is not one contiguous run: {node:?} continues it after a retained step or another group",
                                    group.0
                                )));
                            }
                            let id = BlockId(self.next_block);
                            self.next_block += 1;
                            self.group_block.insert(group, id);
                            BlockDraft {
                                id,
                                group,
                                top: Vec::new(),
                                nodes: Vec::new(),
                                executor_guards: Vec::new(),
                                kernel_guards: Vec::new(),
                            }
                        }
                    };
                    block.top.push(node.clone());
                    self.absorb(&node, group, &mut block)?;
                    open = Some(block);
                }
                NodePlacement::Retained => {
                    if let Some(block) = open.take() {
                        self.close_block(block, &mut steps);
                    }
                    steps.extend(self.retained_guards(&node)?);
                    let step = self.retained_step(&node)?;
                    steps.push(step);
                }
            }
        }
        if let Some(block) = open.take() {
            self.close_block(block, &mut steps);
        }
        Ok(ShapeSchedule { steps })
    }

    fn close_block(&mut self, mut block: BlockDraft, steps: &mut Vec<ShapeStep>) {
        for (obligation, predicate) in block.executor_guards.drain(..) {
            steps.push(ShapeStep::Guard {
                obligation,
                predicate,
            });
        }
        let pulls = matches!(
            self.proposal
                .launches
                .get(block.group)
                .map(|launch| &launch.participants),
            Some(ParticipantPolicy::DynamicPull { .. })
        );
        if pulls {
            steps.push(ShapeStep::PullCounterReset { block: block.id });
        }
        steps.push(ShapeStep::Launch(block.id));
        self.blocks.insert(block.id, block);
    }

    fn consume_node(&mut self, node: &OwnedNodeRef, end: EdgeEnd) -> Result<(), CompilerDefect> {
        if !self.consumed_nodes.insert(node.clone()) {
            return Err(self.defect(format!("{node:?} is consumed twice")));
        }
        let logical = self.facts.node(node)?;
        for token in node_state_inputs(logical) {
            self.consumed_state_edges
                .insert(StateEdge::NodeInput(node.clone(), *token));
        }
        for token in &logical.state_outputs {
            self.consumed_state_edges
                .insert(StateEdge::NodeOutput(node.clone(), token.id));
        }
        self.node_end.insert(node.clone(), end);
        Ok(())
    }

    fn own_region_results(
        &mut self,
        region: &OwnedRegionRef,
        owner: EdgeEnd,
    ) -> Result<(), CompilerDefect> {
        let results = &self.facts.region(region)?.results;
        for (ordinal, result) in results.iter().enumerate() {
            let reference = OwnedRegionResultRef {
                region: region.clone(),
                ordinal: ordinal as u32,
            };
            if let RegionResult::State { id, .. } = result {
                self.consumed_state_edges
                    .insert(StateEdge::RegionResult(reference.clone(), *id));
            }
            if self
                .result_owner
                .insert(reference.clone(), owner.clone())
                .is_some()
            {
                return Err(self.defect(format!("region result {reference:?} is owned twice")));
            }
        }
        Ok(())
    }

    /// Place one node and its whole subtree into the open block.
    fn absorb(
        &mut self,
        node: &OwnedNodeRef,
        group: LaunchGroup,
        block: &mut BlockDraft,
    ) -> Result<(), CompilerDefect> {
        let logical = self.facts.node(node)?;
        self.consume_node(node, EdgeEnd::Block(block.id))?;
        block.nodes.push(node.clone());
        for (index, obligation) in logical.safety.iter().enumerate() {
            let reference = ObligationRef {
                node: node.clone(),
                index: index as u32,
            };
            let disposition = match self.dispose(node, obligation, Some(block.id))? {
                Disposed::Static(proof) => ObligationDisposition::Static { proof },
                Disposed::Kernel => {
                    block
                        .kernel_guards
                        .push((reference.clone(), obligation.clone()));
                    ObligationDisposition::KernelGuard { block: block.id }
                }
                Disposed::Executor(predicate) => {
                    block.executor_guards.push((reference.clone(), predicate));
                    ObligationDisposition::ExecutorGuard
                }
            };
            self.obligations.insert(reference, disposition);
        }
        match &logical.kind {
            LogicalNodeKind::Primitive(_) | LogicalNodeKind::Reduction(_) => {}
            LogicalNodeKind::Call(_) => {
                // An absorbed call (validated): its callee's root nodes follow
                // it in the linearized sequence of the enclosing region.
            }
            LogicalNodeKind::If(_) | LogicalNodeKind::Loop(_) => {
                for child in child_regions(self.facts, node) {
                    self.own_region_results(&child, EdgeEnd::Block(block.id))?;
                    for inner in self.linearized(&child) {
                        match self.placement_of(&inner)? {
                            NodePlacement::Launch(other) if other == group => {
                                self.absorb(&inner, group, block)?;
                            }
                            NodePlacement::Launch(other) => {
                                return Err(self.defect(format!(
                                    "{inner:?} is placed in launch group {} inside {node:?}, which is absorbed by launch group {}",
                                    other.0, group.0
                                )));
                            }
                            NodePlacement::Retained => {
                                return Err(self.defect(format!(
                                    "{inner:?} is retained inside {node:?}, which is absorbed by launch group {}",
                                    group.0
                                )));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Guard steps for the obligations of one retained node (a sealed
    /// program attaches none to control or call nodes; the disposition is
    /// nonetheless total).
    fn retained_guards(&mut self, node: &OwnedNodeRef) -> Result<Vec<ShapeStep>, CompilerDefect> {
        let logical = self.facts.node(node)?;
        let mut steps = Vec::new();
        for (index, obligation) in logical.safety.iter().enumerate() {
            let reference = ObligationRef {
                node: node.clone(),
                index: index as u32,
            };
            let disposition = match self.dispose(node, obligation, None)? {
                Disposed::Static(proof) => ObligationDisposition::Static { proof },
                Disposed::Executor(predicate) => {
                    steps.push(ShapeStep::Guard {
                        obligation: reference.clone(),
                        predicate,
                    });
                    ObligationDisposition::ExecutorGuard
                }
                Disposed::Kernel => {
                    return Err(self.defect(format!(
                        "obligation {index} of retained {node:?} needs kernel values but the node is not in a launch"
                    )));
                }
            };
            self.obligations.insert(reference, disposition);
        }
        Ok(steps)
    }

    fn retained_step(&mut self, node: &OwnedNodeRef) -> Result<ShapeStep, CompilerDefect> {
        let facts = self.facts;
        let logical = facts.node(node)?;
        let canonical = |value: GraphValueId| {
            facts.canonical_value(OwnedValueRef {
                graph: node.graph,
                value,
            })
        };
        match &logical.kind {
            LogicalNodeKind::Primitive(_) | LogicalNodeKind::Reduction(_) => Err(self.defect(
                format!("{node:?} is a primitive or reduction and cannot be a retained step"),
            )),
            LogicalNodeKind::Call(_) => {
                if self.absorbed_calls.contains_key(node) {
                    return Err(self.defect(format!(
                        "call {node:?} is retained but its occurrence is absorbed"
                    )));
                }
                let occurrence = facts.call_occurrence(node)?;
                self.consume_node(node, EdgeEnd::Call(node.clone()))?;
                Ok(ShapeStep::Call {
                    node: node.clone(),
                    occurrence,
                })
            }
            LogicalNodeKind::If(if_node) => {
                self.consume_node(node, EdgeEnd::Control(node.clone()))?;
                let regions = child_regions(facts, node);
                let (then_region, else_region) = (&regions[0], &regions[1]);
                self.own_region_results(then_region, EdgeEnd::Control(node.clone()))?;
                self.own_region_results(else_region, EdgeEnd::Control(node.clone()))?;
                let then_schedule = self.form_region(then_region)?;
                let else_schedule = self.form_region(else_region)?;
                let mut joins = Vec::new();
                for join in &if_node.joins {
                    joins.push(match join {
                        JoinSlot::Value {
                            then_result,
                            else_result,
                            joined,
                            ..
                        } => JoinEdge::Value {
                            then_value: canonical(self.region_result_value(then_region, *then_result)?)?,
                            else_value: canonical(self.region_result_value(else_region, *else_result)?)?,
                            joined: canonical(*joined)?,
                        },
                        JoinSlot::State { storage, .. } => JoinEdge::State {
                            storage: facts.canonical_storage(OwnedStorageRef {
                                graph: node.graph,
                                storage: *storage,
                            })?,
                        },
                    });
                }
                Ok(ShapeStep::If {
                    node: node.clone(),
                    condition: canonical(if_node.condition)?,
                    then_schedule,
                    else_schedule,
                    joins,
                })
            }
            LogicalNodeKind::Loop(loop_node) => {
                self.consume_node(node, EdgeEnd::Control(node.clone()))?;
                let regions = child_regions(facts, node);
                let body_region = &regions[0];
                self.own_region_results(body_region, EdgeEnd::Control(node.clone()))?;
                let body = self.form_region(body_region)?;
                let mut carries = Vec::new();
                let mut value_carries =
                    loop_value_carries(facts, node, loop_node)?.into_iter();
                for slot in &loop_node.carried {
                    carries.push(match slot.initial {
                        RegionInput::Value(_) => {
                            let carry = value_carries.next().ok_or_else(|| {
                                self.defect(format!("{node:?} has fewer value carries than value slots"))
                            })?;
                            CarryEdge::Value {
                                initial: carry.initial,
                                parameter: carry.parameter,
                                update: carry.update,
                                result: carry.result,
                            }
                        }
                        RegionInput::State(token) => CarryEdge::State {
                            storage: facts.canonical_storage(facts.state_storage(OwnedStateRef {
                                graph: node.graph,
                                state: token,
                            }))?,
                        },
                    });
                }
                Ok(ShapeStep::Repeat {
                    node: node.clone(),
                    kind: loop_node.kind,
                    start: canonical(loop_node.range.start)?,
                    end: canonical(loop_node.range.end)?,
                    bound: loop_node.range.bound.clone(),
                    binder: canonical(loop_node.binder)?,
                    body,
                    carries,
                })
            }
        }
    }

    fn region_result_value(
        &self,
        region: &OwnedRegionRef,
        ordinal: RegionResultId,
    ) -> Result<GraphValueId, CompilerDefect> {
        match self.facts.region(region)?.results.get(ordinal.index()) {
            Some(RegionResult::Value { id, .. }) => Ok(*id),
            Some(RegionResult::State { .. }) => Err(self.defect(format!(
                "result {} of {region:?} is a state, not a joined value",
                ordinal.0
            ))),
            None => Err(self.defect(format!(
                "result {} of {region:?} does not exist",
                ordinal.0
            ))),
        }
    }

    // -- obligations -----------------------------------------------------

    fn dispose(
        &self,
        node: &OwnedNodeRef,
        obligation: &SafetyObligation,
        block: Option<BlockId>,
    ) -> Result<Disposed, CompilerDefect> {
        if let Some(proof) = self.static_proof(node.graph, obligation)? {
            return Ok(Disposed::Static(proof));
        }
        match obligation {
            SafetyObligation::ExtentPositive { extent } => {
                Ok(Disposed::Executor(ExecutorGuardPredicate::ExtentPositive {
                    extent: extent.clone(),
                }))
            }
            SafetyObligation::ShapeProductFits { factors, bits } => {
                Ok(Disposed::Executor(ExecutorGuardPredicate::ProductFits {
                    factors: factors.clone(),
                    bits: *bits,
                }))
            }
            SafetyObligation::IndexInBounds { .. }
            | SafetyObligation::RangeInBounds { .. }
            | SafetyObligation::DivisorNonZero { .. }
            | SafetyObligation::SignedDivisionNoOverflow { .. }
            | SafetyObligation::ShiftInRange { .. } => match block {
                Some(_) => Ok(Disposed::Kernel),
                None => Err(self.defect(format!(
                    "obligation {obligation:?} of {node:?} needs kernel values but the node is not in a launch"
                ))),
            },
        }
    }

    fn constant(&self, graph: OwnedGraphKey, value: GraphValueId) -> Result<Option<i64>, CompilerDefect> {
        let canonical = self.facts.canonical_value(OwnedValueRef { graph, value })?;
        Ok(self.constants.get(&canonical).copied())
    }

    fn value_type(&self, graph: OwnedGraphKey, value: GraphValueId) -> Result<ValueType, CompilerDefect> {
        let canonical = self.facts.canonical_value(OwnedValueRef { graph, value })?;
        Ok(self.facts.value_type(canonical))
    }

    fn capacity(&self, extent: &ExtentExpr) -> Option<u64> {
        match extent {
            ExtentExpr::Static(n) => Some(*n),
            ExtentExpr::Runtime(id) => Some(self.facts.runtime_extent(*id).capacity),
            ExtentExpr::Sym(sym) => sym.as_constant().and_then(|c| u64::try_from(c).ok()),
        }
    }

    /// Whether one value is proved to lie within `extent`: a constant inside
    /// a static extent, or an index refinement whose bound is statically at
    /// most the extent.
    fn refined_within(&self, graph: OwnedGraphKey, value: GraphValueId, extent: &ExtentExpr) -> Result<bool, CompilerDefect> {
        if let Some(constant) = self.constant(graph, value)? {
            if let Some(n) = extent.as_static() {
                if u64::try_from(constant).is_ok_and(|c| c < n) {
                    return Ok(true);
                }
            }
        }
        if let ValueType::Index { bound } = self.value_type(graph, value)? {
            if let (Some(bound), Some(extent)) = (bound.as_static(), extent.as_static()) {
                return Ok(bound <= extent);
            }
        }
        Ok(false)
    }

    /// The static proof of one obligation, when the logical and
    /// runtime-extent facts suffice (exhaustive over the seven kinds).
    fn static_proof(&self, graph: OwnedGraphKey, obligation: &SafetyObligation) -> Result<Option<String>, CompilerDefect> {
        match obligation {
            SafetyObligation::ExtentPositive { extent } => {
                let Some(length) = extent.as_static() else {
                    return Ok(None);
                };
                Ok((length > 0).then(|| {
                    format!("the extent is statically nonempty ({length} element(s))")
                }))
            }
            SafetyObligation::IndexInBounds { index, extent } => {
                if let Some(constant) = self.constant(graph, *index)? {
                    if let Some(n) = extent.as_static() {
                        if u64::try_from(constant).is_ok_and(|c| c < n) {
                            return Ok(Some(format!(
                                "the index is the constant {constant}, inside the static extent {n}"
                            )));
                        }
                    }
                }
                if let ValueType::Index { bound } = self.value_type(graph, *index)? {
                    if let (Some(b), Some(n)) = (bound.as_static(), extent.as_static()) {
                        if b <= n {
                            return Ok(Some(format!(
                                "the index is refined to {bound}, within {extent}"
                            )));
                        }
                    }
                }
                Ok(None)
            }
            SafetyObligation::RangeInBounds { start, end, extent } => {
                Ok((self.refined_within(graph, *start, extent)? && self.refined_within(graph, *end, extent)?)
                    .then(|| format!("both range endpoints are proved within {extent}")))
            }
            SafetyObligation::DivisorNonZero { value } => {
                let Some(constant) = self.constant(graph, *value)? else {
                    return Ok(None);
                };
                Ok((constant != 0).then(|| format!("the divisor is the nonzero constant {constant}")))
            }
            SafetyObligation::SignedDivisionNoOverflow { lhs, rhs } => {
                let Some(divisor) = self.constant(graph, *rhs)? else {
                    return Ok(None);
                };
                if divisor != -1 && divisor != 0 {
                    return Ok(Some(format!(
                        "the divisor is the constant {divisor}, which never overflows"
                    )));
                }
                let Some(dividend) = self.constant(graph, *lhs)? else {
                    return Ok(None);
                };
                Ok((divisor == -1 && dividend != i64::from(i32::MIN)).then(|| {
                    format!("the dividend is the constant {dividend}, which does not overflow division by -1")
                }))
            }
            SafetyObligation::ShiftInRange { value } => {
                let Some(constant) = self.constant(graph, *value)? else {
                    return Ok(None);
                };
                Ok((0..32).contains(&constant).then(|| {
                    format!("the shift count is the constant {constant}, within 0..32")
                }))
            }
            SafetyObligation::ShapeProductFits { factors, bits } => {
                let fits = |product: u64| *bits >= 64 || product < (1u64 << *bits);
                let statics: Option<Vec<u64>> = factors.iter().map(ExtentExpr::as_static).collect();
                if let Some(values) = statics {
                    let Some(product) =
                        values.iter().try_fold(1u64, |acc, n| acc.checked_mul(*n))
                    else {
                        return Ok(None);
                    };
                    return Ok(fits(product)
                        .then(|| format!("the checked static product {product} fits {bits} bits")));
                }
                let capacities: Option<Vec<u64>> =
                    factors.iter().map(|factor| self.capacity(factor)).collect();
                let Some(capacity_values) = capacities else {
                    return Ok(None);
                };
                let Some(product) = capacity_values
                    .iter()
                    .try_fold(1u64, |acc, n| acc.checked_mul(*n))
                else {
                    return Ok(None);
                };
                Ok(fits(product).then(|| {
                    format!(
                        "every runtime factor is bounded by its capacity; the checked capacity product {product} fits {bits} bits"
                    )
                }))
            }
        }
    }

    // -- the seal ----------------------------------------------------------

    fn seal_sets(&self) -> Result<(), CompilerDefect> {
        if let Some(node) = self.expected_nodes.difference(&self.consumed_nodes).next() {
            return Err(self.defect(format!("owned node {node:?} was never consumed")));
        }
        if let Some(node) = self.consumed_nodes.difference(&self.expected_nodes).next() {
            return Err(self.defect(format!("{node:?} was consumed but is not owned")));
        }
        let owned_results: BTreeSet<&OwnedRegionResultRef> = self.result_owner.keys().collect();
        let expected_results: BTreeSet<&OwnedRegionResultRef> = self.expected_results.iter().collect();
        if let Some(result) = expected_results.difference(&owned_results).next() {
            return Err(self.defect(format!("region result {result:?} has no owner")));
        }
        if let Some(result) = owned_results.difference(&expected_results).next() {
            return Err(self.defect(format!("region result {result:?} is owned but not expected")));
        }
        if let Some(edge) = self
            .expected_state_edges
            .difference(&self.consumed_state_edges)
            .next()
        {
            return Err(self.defect(format!("state edge {edge:?} was never consumed")));
        }
        if let Some(edge) = self
            .consumed_state_edges
            .difference(&self.expected_state_edges)
            .next()
        {
            return Err(self.defect(format!("state edge {edge:?} was consumed but is not expected")));
        }
        let disposed: BTreeSet<&ObligationRef> = self.obligations.keys().collect();
        let expected_obligations: BTreeSet<&ObligationRef> = self.expected_obligations.iter().collect();
        if let Some(obligation) = expected_obligations.difference(&disposed).next() {
            return Err(self.defect(format!("obligation {obligation:?} was never disposed")));
        }
        if let Some(obligation) = disposed.difference(&expected_obligations).next() {
            return Err(self.defect(format!("obligation {obligation:?} was disposed but is not expected")));
        }
        for group in self.proposal.launches.ids() {
            if !self.group_block.contains_key(&group) {
                return Err(self.defect(format!(
                    "launch group {} is declared but no owned node is placed in it",
                    group.0
                )));
            }
        }
        Ok(())
    }

    // -- cuts and edges ----------------------------------------------------

    fn end_of(&self, node: &OwnedNodeRef) -> Result<EdgeEnd, CompilerDefect> {
        self.node_end
            .get(node)
            .cloned()
            .ok_or_else(|| self.defect(format!("{node:?} has no schedule end")))
    }

    fn producer_end(&self, value: CanonicalValueId) -> Result<EdgeEnd, CompilerDefect> {
        match self.producers.get(&value) {
            Some(Producer::RootBoundary) => Ok(EdgeEnd::RootBoundary),
            Some(Producer::Node(node)) => self.end_of(node),
            None => Err(self.defect(format!(
                "canonical value {} is consumed but nothing owned produces it",
                value.0
            ))),
        }
    }

    /// Record one consumption of a value at an end, including the dynamic
    /// endpoints of a consumed slice view (they address the view).
    fn add_use(
        &self,
        uses: &mut BTreeMap<CanonicalValueId, BTreeSet<EdgeEnd>>,
        value: OwnedValueRef,
        end: &EdgeEnd,
    ) -> Result<(), CompilerDefect> {
        let facts = self.facts;
        let canonical = facts.canonical_value(value)?;
        uses.entry(canonical).or_default().insert(end.clone());
        let producer = facts.members(canonical)[0];
        if !self.owned.contains(&producer.graph) {
            return Ok(());
        }
        let GraphValueKind::Tensor {
            source: TensorSource::View(view),
            ..
        } = facts.value_kind(canonical)
        else {
            return Ok(());
        };
        let view = facts.view(OwnedViewRef {
            graph: producer.graph,
            view: *view,
        });
        let ViewTransform::Slice { axes } = &view.transform else {
            return Ok(());
        };
        for axis in axes {
            let endpoints: Vec<GraphValueId> = match axis {
                SliceAxis::Full => Vec::new(),
                SliceAxis::Point(point) => vec![*point],
                SliceAxis::Range { start, end: stop } => {
                    start.iter().chain(stop.iter()).copied().collect()
                }
            };
            for endpoint in endpoints {
                let canonical = facts.canonical_value(OwnedValueRef {
                    graph: producer.graph,
                    value: endpoint,
                })?;
                uses.entry(canonical).or_default().insert(end.clone());
            }
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn derive_edges(
        &self,
    ) -> Result<
        (
            Vec<CrossEdge>,
            BTreeMap<BlockId, BTreeSet<CanonicalLeafId>>,
            BTreeMap<BlockId, BTreeSet<CanonicalLeafId>>,
        ),
        CompilerDefect,
    > {
        let facts = self.facts;
        let mut uses: BTreeMap<CanonicalValueId, BTreeSet<EdgeEnd>> = BTreeMap::new();
        // Ends that WRITE a value (element-write and copy destinations): a
        // writer is the value's producing side for cut purposes, never a
        // consumer.
        let mut writers: BTreeMap<CanonicalValueId, BTreeSet<EdgeEnd>> = BTreeMap::new();
        for node in &self.expected_nodes {
            let end = self.end_of(node)?;
            let logical = facts.node(node)?;
            match &logical.kind {
                LogicalNodeKind::Primitive(application) => {
                    match &application.op {
                        PrimitiveOp::Primitive(PrimitiveId::ElementWrite { .. })
                        | PrimitiveOp::Primitive(PrimitiveId::CopyInto) => {
                            // The destination is both a use (a value arriving
                            // from outside this block to be written into: an
                            // input) and the block's writer side (output
                            // attribution). In-block-produced destinations
                            // lose the use to their producer, as every use
                            // does.
                            let mut inputs = logical.inputs.iter();
                            if let Some(destination) = inputs.next() {
                                let canonical = facts.canonical_value(OwnedValueRef {
                                    graph: node.graph,
                                    value: *destination,
                                })?;
                                writers.entry(canonical).or_default().insert(end.clone());
                                self.add_use(
                                    &mut uses,
                                    OwnedValueRef {
                                        graph: node.graph,
                                        value: *destination,
                                    },
                                    &end,
                                )?;
                            }
                            for value in inputs {
                                self.add_use(
                                    &mut uses,
                                    OwnedValueRef {
                                        graph: node.graph,
                                        value: *value,
                                    },
                                    &end,
                                )?;
                            }
                        }
                        _ => {
                            for value in node_value_edges(facts, node) {
                                self.add_use(&mut uses, value, &end)?;
                            }
                        }
                    }
                }
                LogicalNodeKind::Call(_) => {
                    // A storage-backed tensor argument is state-carried
                    // (`CallInput::Tensor`): its dataflow is the storage's,
                    // threaded by state edges, never a value cut edge.
                    // Computed tensors and non-tensor values remain edges.
                    for value in &logical.inputs {
                        let owned = OwnedValueRef {
                            graph: node.graph,
                            value: *value,
                        };
                        let canonical = facts.canonical_value(owned.clone())?;
                        if storage_backed_view(facts, canonical) {
                            continue;
                        }
                        self.add_use(&mut uses, owned, &end)?;
                    }
                }
                _ => {
                    for value in node_value_edges(facts, node) {
                        self.add_use(&mut uses, value, &end)?;
                    }
                }
            }
        }
        for (reference, owner) in &self.result_owner {
            if let Some(RegionResult::Value { id, .. }) =
                facts.region(&reference.region)?.results.get(reference.ordinal as usize)
            {
                self.add_use(
                    &mut uses,
                    OwnedValueRef {
                        graph: reference.region.graph,
                        value: *id,
                    },
                    owner,
                )?;
            }
        }
        for result in facts.alternative(self.root_key).boundary.results.values() {
            let callee = match result {
                InstantiatedResult::Root { callee } | InstantiatedResult::Call { callee, .. } => *callee,
            };
            self.add_use(&mut uses, callee, &EdgeEnd::RootBoundary)?;
        }

        let mut edges = Vec::new();
        let mut inputs: BTreeMap<BlockId, BTreeSet<CanonicalLeafId>> = BTreeMap::new();
        let mut outputs: BTreeMap<BlockId, BTreeSet<CanonicalLeafId>> = BTreeMap::new();
        for (value, mut ends) in uses {
            let value_writers = writers.get(&value).cloned().unwrap_or_default();
            if std::env::var_os("SEISMIC_DEBUG_SEAL").is_some() {
                let producer_dbg = match value_writers.iter().max() {
                    Some(last) => format!("{last:?}"),
                    None if storage_backed_view(facts, value) => "none(storage)".into(),
                    None => format!("{:?}", self.producer_end(value)),
                };
                eprintln!(
                    "  EDGE value {} producer={producer_dbg} writers={:?} ends={:?} storage_backed={} kind={:?} members={:?}",
                    value.0,
                    value_writers,
                    ends,
                    storage_backed_view(facts, value),
                    facts.value_kind(value),
                    facts.members(value),
                );
            }
            // The producing side of a written value is its writer, not the
            // node that originated the value id (an allocation declares a
            // storage; it produces no data). A storage-backed view with no
            // writer in this occurrence is carried wholly by storage
            // routing: block readers bind the storage, no block claims an
            // unwritten output.
            let producer = match value_writers.iter().max() {
                Some(last_writer) => Some(last_writer.clone()),
                None if storage_backed_view(facts, value) => None,
                None => Some(self.producer_end(value)?),
            };
            // Consumers are the uses that are neither a writer nor the
            // producer of the value.
            for writer in &value_writers {
                ends.remove(writer);
            }
            if let Some(producer_end) = &producer {
                ends.remove(producer_end);
            }
            let Some(consumers) = NonEmpty::new(ends.into_iter().collect()) else {
                continue;
            };
            let ty = facts.value_type(value);
            for leaf in facts.leaves(value) {
                if leaf_is_capability(&ty, &facts.leaf(*leaf).path) {
                    return Err(self.defect(format!(
                        "capability value {} crosses a cut to {:?}; a capability value lives inside one launch",
                        value.0,
                        consumers.as_slice()
                    )));
                }
                for end in consumers.iter() {
                    if let EdgeEnd::Block(block) = end {
                        inputs.entry(*block).or_default().insert(*leaf);
                    }
                }
                if let Some(EdgeEnd::Block(block)) = &producer {
                    outputs.entry(*block).or_default().insert(*leaf);
                }
                edges.push(CrossEdge {
                    leaf: *leaf,
                    producer: producer.clone().unwrap_or(EdgeEnd::RootBoundary),
                    consumers: consumers.clone(),
                });
            }
        }
        if std::env::var_os("SEISMIC_DEBUG_SEAL").is_some() {
            eprintln!("  FINAL outputs map={:?}", outputs);
        }
        Ok((edges, inputs, outputs))
    }

    fn state_end(&self, graph: OwnedGraphKey, token: StateTokenId) -> Result<EdgeEnd, CompilerDefect> {
        match self.state_origins.get(&(graph, token)) {
            Some(StateOrigin::Node(node)) | Some(StateOrigin::RegionParameter(node)) => self.end_of(node),
            Some(StateOrigin::RootParameter) => {
                if graph == self.root_key {
                    return Ok(EdgeEnd::RootBoundary);
                }
                let call = self
                    .absorbed_calls
                    .iter()
                    .find(|(_, callee)| **callee == graph)
                    .map(|(call, _)| call.clone())
                    .ok_or_else(|| {
                        self.defect(format!("owned graph {graph:?} is reached by no absorbed call"))
                    })?;
                self.end_of(&call)
            }
            None => Err(self.defect(format!(
                "state token {} of {graph:?} has no origin",
                token.0
            ))),
        }
    }

    fn derive_results(&self) -> Result<BTreeMap<OwnedRegionResultRef, EdgeEnd>, CompilerDefect> {
        let mut results = BTreeMap::new();
        for reference in self.result_owner.keys() {
            let region = &reference.region;
            let Some(result) = self.facts.region(region)?.results.get(reference.ordinal as usize) else {
                return Err(self.defect(format!("region result {reference:?} does not exist")));
            };
            let producer = match result {
                RegionResult::Value { id, .. } => {
                    self.producer_end(self.facts.canonical_value(OwnedValueRef {
                        graph: region.graph,
                        value: *id,
                    })?)?
                }
                RegionResult::State { id, .. } => self.state_end(region.graph, *id)?,
            };
            results.insert(reference.clone(), producer);
        }
        Ok(results)
    }

    fn finish_blocks(
        &self,
        inputs: &BTreeMap<BlockId, BTreeSet<CanonicalLeafId>>,
        outputs: &BTreeMap<BlockId, BTreeSet<CanonicalLeafId>>,
    ) -> Result<IdVec<BlockId, BlockCut>, CompilerDefect> {
        let facts = self.facts;
        let mut cuts = Vec::new();
        for (id, draft) in &self.blocks {
            let Some(launch) = self.proposal.launches.get(draft.group) else {
                return Err(self.defect(format!(
                    "launch group {} has no launch proposal",
                    draft.group.0
                )));
            };
            let Some(nodes) = NonEmpty::new(draft.nodes.clone()) else {
                return Err(self.defect(format!("block {} owns no node", id.0)));
            };
            let participants = self.participants(draft, launch.participants.clone())?;
            self.check_whole_result_edges(draft, launch)?;
            let mut state_reads = BTreeSet::new();
            let mut state_writes = BTreeSet::new();
            for node in &draft.nodes {
                let logical = facts.node(node)?;
                for token in node_state_inputs(logical) {
                    state_reads.insert(self.storage_of(node.graph, *token)?);
                }
                for token in &logical.state_outputs {
                    state_writes.insert(self.storage_of(node.graph, token.id)?);
                }
            }
            cuts.push(BlockCut {
                nodes,
                participants,
                algorithm: launch.algorithm.clone(),
                local_residences: launch.local_residences.clone(),
                numerical: launch.numerical.clone(),
                inputs: inputs.get(id).cloned().unwrap_or_default(),
                outputs: outputs.get(id).cloned().unwrap_or_default(),
                state_reads,
                state_writes,
                guards: draft.kernel_guards.clone(),
            });
        }
        Ok(IdVec::new(cuts))
    }

    /// A whole-result edge inside one launch: a value produced by one
    /// top-level phase of the block and consumed by another whose phase
    /// domain differs (the producer must complete before the consumer
    /// starts). Legal under `Serial` (one participant runs the phases in
    /// order) and under `GridCooperative` when the proposal lists a
    /// device-arena `LocalResidenceRequirement` for the intermediate (the
    /// grid barrier separates the phases); a defect of the rule otherwise.
    fn check_whole_result_edges(
        &self,
        draft: &BlockDraft,
        launch: &LaunchProposal,
    ) -> Result<(), CompilerDefect> {
        if matches!(launch.participants, ParticipantPolicy::Serial) {
            return Ok(());
        }
        let facts = self.facts;
        // Declaration nodes (storage allocations, scalar constants) execute
        // nothing and form no phase: each attaches to the phase of the
        // executing node that follows it, and its outputs are phase-local
        // scalars, never whole-result edges.
        let is_declaration = |node: &OwnedNodeRef| {
            matches!(
                facts.node(node).map(|logical| &logical.kind),
                Ok(LogicalNodeKind::Primitive(application))
                    if matches!(
                        application.op,
                        PrimitiveOp::Primitive(PrimitiveId::TensorAlloc { .. })
                            | PrimitiveOp::Constant(_)
                    )
            )
        };
        let mut phase_of: BTreeMap<OwnedNodeRef, OwnedNodeRef> = BTreeMap::new();
        let mut current: Option<OwnedNodeRef> = None;
        for node in &draft.nodes {
            if draft.top.contains(node) && !is_declaration(node) {
                current = Some(node.clone());
            }
            let Some(phase) = &current else {
                if is_declaration(node) {
                    // A declaration ahead of every executing node attaches to
                    // itself; it executes nothing and produces no edges.
                    phase_of.insert(node.clone(), node.clone());
                    continue;
                }
                return Err(self.defect(format!("{node:?} precedes every top-level node of its block")));
            };
            phase_of.insert(node.clone(), phase.clone());
        }
        let mut produced_in: BTreeMap<CanonicalValueId, OwnedNodeRef> = BTreeMap::new();
        for node in &draft.nodes {
            if is_declaration(node) {
                continue;
            }
            for output in &facts.node(node)?.outputs {
                let owned = OwnedValueRef {
                    graph: node.graph,
                    value: output.id(),
                };
                let canonical = facts.canonical_value(owned)?;
                if facts.members(canonical)[0] == owned {
                    produced_in.insert(canonical, phase_of[node].clone());
                }
            }
        }
        for node in &draft.nodes {
            let consumer_phase = &phase_of[node];
            for value in node_value_edges(facts, node) {
                let canonical = facts.canonical_value(value)?;
                let Some(producer_phase) = produced_in.get(&canonical) else {
                    continue;
                };
                if producer_phase == consumer_phase
                    || phase_domain(facts, producer_phase) == phase_domain(facts, consumer_phase)
                {
                    continue;
                }
                match launch.participants {
                    ParticipantPolicy::GridCooperative { .. } => {
                        let required = LocalResidenceRequirement {
                            value: canonical,
                            scope: StorageScope::DeviceArena,
                            replication: Replication::Once,
                        };
                        if !launch.local_residences.contains(&required) {
                            return Err(self.defect(format!(
                                "grid-cooperative launch group {} carries the whole-result value {} from {producer_phase:?} to {consumer_phase:?} without a device-arena local residence requirement for it",
                                draft.group.0, canonical.0
                            )));
                        }
                    }
                    ParticipantPolicy::Serial
                    | ParticipantPolicy::Linear { .. }
                    | ParticipantPolicy::Cooperative { .. }
                    | ParticipantPolicy::DynamicPull { .. } => {
                        return Err(self.defect(format!(
                            "launch group {} carries the whole-result value {} from {producer_phase:?} to {consumer_phase:?}; only a serial or grid-cooperative launch may hold a whole-result edge",
                            draft.group.0, canonical.0
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn storage_of(
        &self,
        graph: OwnedGraphKey,
        token: StateTokenId,
    ) -> Result<CanonicalStorageId, CompilerDefect> {
        Ok(self
            .facts
            .canonical_storage(self.facts.state_storage(OwnedStateRef { graph, state: token }))?)
    }

    /// The participant map of one block: the independent axes are the
    /// chain of independent loops that are, at each level, the block's only
    /// node (outermost first); every other absorbed loop binder is serial;
    /// the value carries of absorbed ordered loops are ordered kernel
    /// carries.
    fn participants(
        &self,
        draft: &BlockDraft,
        policy: ParticipantPolicy,
    ) -> Result<ParticipantMap, CompilerDefect> {
        let facts = self.facts;
        let mut independent_axes = Vec::new();
        let mut chain: BTreeSet<OwnedNodeRef> = BTreeSet::new();
        // Declaration nodes (storage allocations, scalar constants) execute
        // nothing: the block's structural top level is its executing nodes.
        let is_declaration = |node: &OwnedNodeRef| {
            matches!(
                facts.node(node).map(|logical| &logical.kind),
                Ok(LogicalNodeKind::Primitive(application))
                    if matches!(
                        application.op,
                        PrimitiveOp::Primitive(PrimitiveId::TensorAlloc { .. })
                            | PrimitiveOp::Constant(_)
                    )
            )
        };
        let mut level: Vec<OwnedNodeRef> =
            draft.top.iter().filter(|node| !is_declaration(node)).cloned().collect();
        loop {
            let [only] = level.as_slice() else {
                break;
            };
            let LogicalNodeKind::Loop(loop_node) = &facts.node(only)?.kind else {
                break;
            };
            if loop_node.kind != LoopKind::Independent {
                break;
            }
            independent_axes.push(AxisBinding {
                binder: facts.canonical_value(OwnedValueRef {
                    graph: only.graph,
                    value: loop_node.binder,
                })?,
                ordinal: independent_axes.len() as u32,
                extent: loop_node.range.bound.clone(),
            });
            chain.insert(only.clone());
            let regions = child_regions(facts, only);
            level = region_nodes(facts, &regions[0]);
        }
        let mut serial_binders = BTreeSet::new();
        let mut ordered_carries = Vec::new();
        for node in &draft.nodes {
            let LogicalNodeKind::Loop(loop_node) = &facts.node(node)?.kind else {
                continue;
            };
            if chain.contains(node) {
                continue;
            }
            serial_binders.insert(facts.canonical_value(OwnedValueRef {
                graph: node.graph,
                value: loop_node.binder,
            })?);
            for carry in loop_value_carries(facts, node, loop_node)? {
                let kind = facts.value_kind(carry.initial);
                if !matches!(
                    kind,
                    GraphValueKind::Scalar(_)
                        | GraphValueKind::Index { .. }
                        | GraphValueKind::Capability(_)
                ) {
                    return Err(self.defect(format!(
                        "{node:?} is absorbed by launch group {} but carries a {kind:?} value; only kernel scalars can be carried inside a launch",
                        draft.group.0
                    )));
                }
                ordered_carries.push(carry);
            }
        }
        if matches!(
            policy,
            ParticipantPolicy::Linear { .. } | ParticipantPolicy::DynamicPull { .. }
        ) && independent_axes.is_empty()
        {
            return Err(self.defect(format!(
                "launch group {} asks for linear or dynamic-pull participants but its top-level structure is not a single independent loop, so it has no independent axis",
                draft.group.0
            )));
        }
        Ok(ParticipantMap {
            policy,
            independent_axes,
            serial_binders,
            ordered_carries,
        })
    }
}

/// Whether the leaf at `path` of a value type is a capability value.
fn leaf_is_capability(ty: &ValueType, path: &ValuePath) -> bool {
    let mut current = ty;
    for index in &path.0 {
        match current {
            ValueType::Tuple(items) => match items.as_slice().get(*index as usize) {
                Some(item) => current = item,
                None => return false,
            },
            _ => return false,
        }
    }
    matches!(current, ValueType::CapabilityValue(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::OccurrenceId;
    use crate::occurrence::OccurrenceForest;
    use crate::strategy::owned_nodes;
    use crate::target::TargetLimits;
    use seismic_lang::logical::specialization::{ShapeBinding, SpecializationDomain};
    use seismic_lang::logical::{construct, EffectiveTargetIdentity, LogicalProgram};
    use seismic_lang::program::{compile, SourceFile};
    use seismic_lang::sir::Program;

    fn check(source: &str) -> Program {
        let files = vec![SourceFile {
            path: "test.seismic".to_string(),
            text: source.to_string(),
        }];
        compile(&files).unwrap_or_else(|diagnostics| {
            panic!(
                "the source checks: {}",
                diagnostics
                    .iter()
                    .map(|d| d.render())
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        })
    }

    fn supports_all(_: &seismic_lang::sir::IntrinsicUse) -> Result<(), String> {
        Ok(())
    }

    fn logical_on(
        source: &str,
        entry: &str,
        backend: &str,
        shapes: BTreeMap<String, ShapeBinding>,
    ) -> LogicalProgram {
        let program = check(source);
        let domain = SpecializationDomain::new(&program, entry, shapes, BTreeMap::new())
            .expect("the domain binds every entry parameter");
        let target = EffectiveTargetIdentity {
            backend: backend.to_string(),
            capability_fingerprint: "test-fingerprint".to_string(),
        };
        construct(&program, &target, &supports_all, &domain).expect("construction succeeds")
    }

    fn logical(source: &str, entry: &str, shapes: &[(&str, u64)]) -> LogicalProgram {
        let shapes = shapes
            .iter()
            .map(|(name, value)| (name.to_string(), ShapeBinding::Exact(*value)))
            .collect();
        logical_on(source, entry, "cpu", shapes)
    }

    fn profile() -> EffectiveTargetProfile {
        EffectiveTargetProfile {
            backend: "cpu".to_string(),
            capability_fingerprint: "test-fingerprint".to_string(),
            toolchain_fingerprint: "test-toolchain".to_string(),
            effective_signatures: BTreeSet::new(),
            limits: TargetLimits {
                max_participants: 1024,
                max_workgroups_axis: [65535; 3],
                max_workgroup_bytes: 32768,
                max_explicit_private_bytes: 4096,
                max_direct_bindings: 31,
                max_device_bytes: 1 << 30,
                cooperative_grid: None,
            },
        }
    }

    fn proposals(
        facts: &OccurrenceFacts<'_>,
        profile: &EffectiveTargetProfile,
        occurrence: OccurrenceId,
        logical_alternative: u32,
    ) -> Vec<MappingProposal> {
        let query = RuleQuery {
            facts,
            profile,
            occurrence,
            logical_alternative,
        };
        universal_rules()
            .iter()
            .flat_map(|rule| rule.propose(&query))
            .collect()
    }

    fn is_serial_witness(proposal: &MappingProposal) -> bool {
        proposal
            .launches
            .iter()
            .all(|launch| launch.participants == ParticipantPolicy::Serial)
    }

    fn has_linear_launch(proposal: &MappingProposal) -> bool {
        proposal
            .launches
            .iter()
            .any(|launch| matches!(launch.participants, ParticipantPolicy::Linear { .. }))
    }

    /// Every node the schedule executes: block nodes plus retained steps,
    /// in schedule order.
    fn scheduled_nodes(shape: &ClosedStrategyShape) -> Vec<OwnedNodeRef> {
        fn walk(schedule: &ShapeSchedule, shape: &ClosedStrategyShape, out: &mut Vec<OwnedNodeRef>) {
            for step in &schedule.steps {
                match step {
                    ShapeStep::Launch(block) => {
                        out.extend(shape.blocks()[*block].nodes.iter().cloned());
                    }
                    ShapeStep::Guard { .. } | ShapeStep::PullCounterReset { .. } => {}
                    ShapeStep::Call { node, .. } => out.push(node.clone()),
                    ShapeStep::If {
                        node,
                        then_schedule,
                        else_schedule,
                        ..
                    } => {
                        out.push(node.clone());
                        walk(then_schedule, shape, out);
                        walk(else_schedule, shape, out);
                    }
                    ShapeStep::Repeat { node, body, .. } => {
                        out.push(node.clone());
                        walk(body, shape, out);
                    }
                }
            }
        }
        let mut out = Vec::new();
        walk(shape.schedule(), shape, &mut out);
        out
    }

    fn assert_complete(facts: &OccurrenceFacts<'_>, proposal: &MappingProposal, shape: &ClosedStrategyShape) {
        let mut scheduled = scheduled_nodes(shape);
        let count = scheduled.len();
        scheduled.sort();
        scheduled.dedup();
        assert_eq!(count, scheduled.len(), "no node is scheduled twice");
        let mut owned = owned_nodes(facts, &proposal.ownership);
        owned.sort();
        assert_eq!(scheduled, owned, "every owned node is scheduled exactly once");
        // Every obligation of every owned node is disposed exactly once, and
        // every kernel guard names the block that owns its node.
        let mut expected = BTreeSet::new();
        for node in &owned {
            for index in 0..facts.node(node).unwrap().safety.len() {
                expected.insert(ObligationRef {
                    node: node.clone(),
                    index: index as u32,
                });
            }
        }
        let disposed: BTreeSet<ObligationRef> = shape.obligations().keys().cloned().collect();
        assert_eq!(disposed, expected);
        for (obligation, disposition) in shape.obligations() {
            if let ObligationDisposition::KernelGuard { block } = disposition {
                assert_eq!(shape.block_of(&obligation.node), Some(*block));
                assert!(shape.blocks()[*block]
                    .guards
                    .iter()
                    .any(|(guard, _)| guard == obligation));
            }
        }
        for (id, block) in shape.blocks().entries() {
            for (guard, _) in &block.guards {
                assert_eq!(
                    shape.obligations().get(guard),
                    Some(&ObligationDisposition::KernelGuard { block: id })
                );
            }
        }
        // Every launch group formed exactly one block.
        assert_eq!(shape.blocks().len(), proposal.launches.len());
    }

    const KERNEL: &str = "fn add[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut output = result\n    parallel for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + y[row, col]\n    return output\n\nfn linear[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    return add(x, y, result)\n";

    #[test]
    fn universal_rules_form_every_alternative_completely() {
        let logical = logical(KERNEL, "linear", &[("M", 4), ("N", 8)]);
        let forest = OccurrenceForest::expand(&logical).unwrap_or_else(|defect| panic!("{defect}"));
        let facts = forest.facts();
        let profile = profile();
        for (occurrence, record) in facts.occurrences() {
            for alternative in 0..record.alternatives.len() as u32 {
                let proposals = proposals(facts, &profile, occurrence, alternative);
                assert!(!proposals.is_empty());
                assert!(
                    proposals.iter().any(is_serial_witness),
                    "occurrence#{} alternative {alternative} has a serial witness",
                    occurrence.0
                );
                for proposal in proposals {
                    assert!(proposal.ownership.absorbed.is_empty());
                    let shape = form(facts, &profile, proposal.clone())
                        .unwrap_or_else(|defect| panic!("{defect}"));
                    assert_eq!(shape.rule(), proposal.rule);
                    assert_complete(facts, &proposal, &shape);
                    assert!(!shape.owned_regions().is_empty());
                }
            }
        }
    }

    #[test]
    fn streaming_absorbs_the_parallel_loop_with_one_axis_and_a_serial_binder() {
        let logical = logical(KERNEL, "linear", &[("M", 4), ("N", 8)]);
        let forest = OccurrenceForest::expand(&logical).unwrap_or_else(|defect| panic!("{defect}"));
        let facts = forest.facts();
        let profile = profile();
        let add = OccurrenceId(1);
        assert_eq!(facts.occurrence(add).interface.name, "add");
        let proposals = proposals(facts, &profile, add, 0);
        let linear = proposals
            .iter()
            .find(|proposal| proposal.rule == STREAMING && has_linear_launch(proposal))
            .expect("the streaming rule offers a linear peer for the parallel loop");
        assert_eq!(linear.tuning.len(), 1);
        assert_eq!(linear.tuning[0].upper, profile.limits.max_participants);
        let shape = form(facts, &profile, linear.clone()).unwrap_or_else(|defect| panic!("{defect}"));
        assert_complete(facts, linear, &shape);
        // Streaming: every top-level node of the root region is its own
        // launch (the parallel loop with its whole body is one of them).
        let key = OwnedGraphKey {
            occurrence: add,
            logical_alternative: 0,
        };
        let top_level = region_nodes(facts, &root_region(key));
        assert_eq!(shape.blocks().len(), top_level.len());
        let parallel = top_level
            .iter()
            .find(|node| matches!(facts.node(node).unwrap().kind, LogicalNodeKind::Loop(_)))
            .expect("the root region holds the parallel loop");
        let block_id = shape.block_of(parallel).expect("the loop is absorbed by a block");
        let block = &shape.blocks()[block_id];
        assert_eq!(block.nodes.first(), parallel);
        assert_eq!(block.nodes.len(), subtree_nodes(facts, parallel).len());
        assert!(matches!(block.participants.policy, ParticipantPolicy::Linear { .. }));
        assert_eq!(block.participants.independent_axes.len(), 1);
        assert_eq!(block.participants.independent_axes[0].ordinal, 0);
        assert_eq!(block.participants.independent_axes[0].extent, ExtentExpr::Static(4));
        assert_eq!(block.participants.serial_binders.len(), 1);
        // The inner loop carries storage state, never a value.
        assert!(block.participants.ordered_carries.is_empty());
        assert!(!block.state_writes.is_empty(), "the output storage is written");
        assert!(block.state_reads.len() >= 2, "both borrowed inputs are read");
        assert!(!block.inputs.is_empty(), "the borrowed inputs enter the block");
        // Every other block is a serial single-node launch.
        for (id, other) in shape.blocks().entries() {
            if id != block_id {
                assert_eq!(other.participants.policy, ParticipantPolicy::Serial);
                assert!(other.participants.independent_axes.is_empty());
                assert_eq!(other.nodes.len(), 1);
            }
        }
        // The whole graph is launches: no retained steps, one launch step
        // per block.
        assert!(shape
            .schedule()
            .steps
            .iter()
            .all(|step| matches!(step, ShapeStep::Launch(_) | ShapeStep::Guard { .. })));
        assert_eq!(
            shape
                .schedule()
                .steps
                .iter()
                .filter(|step| matches!(step, ShapeStep::Launch(_)))
                .count(),
            shape.blocks().len()
        );
        // Without retained steps, every edge end is the root boundary or a
        // block.
        for edge in shape.edges() {
            for end in std::iter::once(&edge.producer).chain(edge.consumers.iter()) {
                assert!(
                    matches!(end, EdgeEnd::RootBoundary | EdgeEnd::Block(_)),
                    "unexpected edge end {end:?}"
                );
            }
            assert!(!edge.consumers.iter().any(|end| *end == edge.producer));
        }

        // The serial witness keeps the axis and serializes it.
        let serial = proposals
            .iter()
            .find(|proposal| proposal.rule == STREAMING && is_serial_witness(proposal))
            .expect("the streaming rule offers the serial witness");
        let shape = form(facts, &profile, serial.clone()).unwrap_or_else(|defect| panic!("{defect}"));
        let block_id = shape.block_of(parallel).expect("the loop is absorbed by a block");
        let block = &shape.blocks()[block_id];
        assert_eq!(block.participants.policy, ParticipantPolicy::Serial);
        assert_eq!(block.participants.independent_axes.len(), 1);
        assert!(shape.tuning().is_empty());
    }

    #[test]
    fn a_retained_call_is_a_call_step() {
        let logical = logical(KERNEL, "linear", &[("M", 4), ("N", 8)]);
        let forest = OccurrenceForest::expand(&logical).unwrap_or_else(|defect| panic!("{defect}"));
        let facts = forest.facts();
        let profile = profile();
        let entry = facts.entry();
        for proposal in proposals(facts, &profile, entry, 0) {
            assert!(is_serial_witness(&proposal));
            assert!(proposal.launches.is_empty());
            let shape = form(facts, &profile, proposal.clone()).unwrap_or_else(|defect| panic!("{defect}"));
            assert_complete(facts, &proposal, &shape);
            assert!(shape.blocks().is_empty());
            let [ShapeStep::Call { node, occurrence }] = shape.schedule().steps.as_slice() else {
                panic!("the entry schedule is one retained call, got {:?}", shape.schedule());
            };
            assert_eq!(*occurrence, OccurrenceId(1));
            assert_eq!(facts.call_occurrence(node).unwrap(), OccurrenceId(1));
            let call_end = EdgeEnd::Call(node.clone());
            assert!(shape.edges().iter().any(|edge| {
                edge.producer == EdgeEnd::RootBoundary && edge.consumers.iter().any(|end| *end == call_end)
            }));
            assert!(shape.edges().iter().any(|edge| {
                edge.producer == call_end && edge.consumers.iter().any(|end| *end == EdgeEnd::RootBoundary)
            }));
        }
    }

    #[test]
    fn omitting_a_node_is_a_defect_naming_the_rule() {
        let logical = logical(KERNEL, "linear", &[("M", 4), ("N", 8)]);
        let forest = OccurrenceForest::expand(&logical).unwrap_or_else(|defect| panic!("{defect}"));
        let facts = forest.facts();
        let profile = profile();
        let mut proposal = proposals(facts, &profile, OccurrenceId(1), 0)
            .into_iter()
            .find(|proposal| proposal.rule == POINT_SERIAL)
            .expect("the point-serial rule proposes");
        let dropped = proposal
            .placement
            .keys()
            .next()
            .cloned()
            .expect("the proposal places at least one node");
        proposal.placement.remove(&dropped);
        let defect = form(facts, &profile, proposal).expect_err("an unplaced node is a defect");
        assert_eq!(defect.package, Package::S1);
        assert!(defect.invariant.contains(POINT_SERIAL), "{defect}");
        assert!(defect.invariant.contains("has no placement"), "{defect}");

        // A node placed in an undeclared launch group is likewise rejected.
        let mut proposal = proposals(facts, &profile, OccurrenceId(1), 0)
            .into_iter()
            .find(|proposal| proposal.rule == POINT_SERIAL)
            .expect("the point-serial rule proposes");
        let node = proposal.placement.keys().next().cloned().expect("a placed node");
        proposal.placement.insert(node, NodePlacement::Launch(LaunchGroup(99)));
        let defect = form(facts, &profile, proposal).expect_err("an undeclared group is a defect");
        assert_eq!(defect.package, Package::S1);
        assert!(defect.invariant.contains("does not declare"), "{defect}");
    }

    const SAFETY: &str = "fn f[N](x: &tensor[N] f32, i: index[N]) -> f32:\n    return x[i] + 1.0 / x[0]\n";

    #[test]
    fn obligations_are_proved_or_guarded_in_their_block() {
        let logical = logical(SAFETY, "f", &[("N", 4)]);
        let forest = OccurrenceForest::expand(&logical).unwrap_or_else(|defect| panic!("{defect}"));
        let facts = forest.facts();
        let profile = profile();
        let entry = facts.entry();
        let proposals = proposals(facts, &profile, entry, 0);
        assert!(proposals.iter().any(is_serial_witness));
        for proposal in proposals {
            let shape = form(facts, &profile, proposal.clone()).unwrap_or_else(|defect| panic!("{defect}"));
            assert_complete(facts, &proposal, &shape);
            let mut proved_indices = 0;
            let mut guarded_divisors = 0;
            for (obligation, disposition) in shape.obligations() {
                let logical = facts.node(&obligation.node).unwrap();
                match (&logical.safety[obligation.index as usize], disposition) {
                    (SafetyObligation::IndexInBounds { .. }, ObligationDisposition::Static { proof }) => {
                        assert!(!proof.is_empty());
                        proved_indices += 1;
                    }
                    (SafetyObligation::DivisorNonZero { .. }, ObligationDisposition::KernelGuard { .. }) => {
                        guarded_divisors += 1;
                    }
                    (_, ObligationDisposition::ExecutorGuard) => {
                        panic!("no extent-only obligation exists in this kernel")
                    }
                    (_, ObligationDisposition::Static { .. } | ObligationDisposition::KernelGuard { .. }) => {}
                }
            }
            assert!(
                proved_indices >= 2,
                "the refined index and the constant index are proved statically"
            );
            assert_eq!(guarded_divisors, 1, "the runtime divisor is checked in its block");
            // No extent-only obligation exists here, so no executor guard step.
            assert!(shape
                .schedule()
                .steps
                .iter()
                .all(|step| !matches!(step, ShapeStep::Guard { .. })));
        }
    }

    #[test]
    fn a_runtime_domain_gets_a_dynamic_pull_peer_with_a_counter_reset() {
        let shapes = BTreeMap::from([
            (
                "M".to_string(),
                ShapeBinding::Bounded {
                    min: 1,
                    max: 64,
                    expected: 16,
                },
            ),
            ("N".to_string(), ShapeBinding::Exact(8)),
        ]);
        let logical = logical_on(KERNEL, "linear", "cpu", shapes);
        let forest = OccurrenceForest::expand(&logical).unwrap_or_else(|defect| panic!("{defect}"));
        let facts = forest.facts();
        let profile = profile();
        let add = OccurrenceId(1);
        let proposals = proposals(facts, &profile, add, 0);
        assert!(proposals.iter().any(is_serial_witness));
        assert!(proposals.iter().any(has_linear_launch));
        let pull = proposals
            .iter()
            .find(|proposal| {
                proposal.launches.iter().any(|launch| {
                    matches!(launch.participants, ParticipantPolicy::DynamicPull { .. })
                })
            })
            .expect("the runtime parallel domain gets a dynamic-pull peer");
        let shape = form(facts, &profile, pull.clone()).unwrap_or_else(|defect| panic!("{defect}"));
        assert_complete(facts, pull, &shape);
        let (block_id, block) = shape
            .blocks()
            .entries()
            .find(|(_, block)| {
                matches!(block.participants.policy, ParticipantPolicy::DynamicPull { .. })
            })
            .expect("the pulled block exists");
        assert_eq!(block.participants.independent_axes.len(), 1);
        assert!(matches!(
            block.participants.independent_axes[0].extent,
            ExtentExpr::Runtime(_)
        ));
        // The counter reset immediately precedes the launch.
        let steps = &shape.schedule().steps;
        let launch_at = steps
            .iter()
            .position(|step| *step == ShapeStep::Launch(block_id))
            .expect("the pulled block is launched");
        assert!(launch_at >= 1);
        assert_eq!(steps[launch_at - 1], ShapeStep::PullCounterReset { block: block_id });
        assert_eq!(
            steps
                .iter()
                .filter(|step| matches!(step, ShapeStep::PullCounterReset { .. }))
                .count(),
            1
        );
        // A linear peer of the same graph has no reset step.
        let linear = proposals
            .iter()
            .find(|proposal| {
                has_linear_launch(proposal)
                    && !proposal.launches.iter().any(|launch| {
                        matches!(launch.participants, ParticipantPolicy::DynamicPull { .. })
                    })
            })
            .expect("the linear peer exists");
        let shape = form(facts, &profile, linear.clone()).unwrap_or_else(|defect| panic!("{defect}"));
        assert!(shape
            .schedule()
            .steps
            .iter()
            .all(|step| !matches!(step, ShapeStep::PullCounterReset { .. })));
    }

    const ATOMIC: &str = "fn hist[N](x: &tensor[N] f32, out: tensor[64] f32) -> f32 for metal:\n    let mut acc = out\n    parallel for i in 0..N:\n        atomic(add, acc[0], x[i])\n    return reduce(f32(acc), 0, sum)\n";

    #[test]
    fn an_atomic_join_gets_a_serialized_witness_and_a_concurrent_reassociating_peer() {
        let shapes = BTreeMap::from([("N".to_string(), ShapeBinding::Exact(128))]);
        let logical = logical_on(ATOMIC, "hist", "metal", shapes);
        let forest = OccurrenceForest::expand(&logical).unwrap_or_else(|defect| panic!("{defect}"));
        let facts = forest.facts();
        let mut profile = profile();
        profile.backend = "metal".to_string();
        let entry = facts.entry();
        let proposals = proposals(facts, &profile, entry, 0);
        let streaming: Vec<&MappingProposal> = proposals
            .iter()
            .filter(|proposal| proposal.rule == STREAMING)
            .collect();
        let serialized = streaming
            .iter()
            .find(|proposal| is_serial_witness(proposal))
            .expect("the serialized atomic witness is always present");
        let concurrent = streaming
            .iter()
            .find(|proposal| has_linear_launch(proposal))
            .expect("a 32-bit atomic join admits the concurrent peer");
        assert!(serialized
            .launches
            .iter()
            .all(|launch| launch.numerical.is_empty()));
        let key = OwnedGraphKey {
            occurrence: entry,
            logical_alternative: 0,
        };
        let parallel = region_nodes(facts, &root_region(key))
            .into_iter()
            .find(|node| matches!(facts.node(node).unwrap().kind, LogicalNodeKind::Loop(_)))
            .expect("the histogram loop is a root node");
        let concurrent_launch = concurrent
            .launches
            .iter()
            .find(|launch| matches!(launch.participants, ParticipantPolicy::Linear { .. }))
            .expect("the loop's launch is linear in the concurrent peer");
        assert_eq!(
            concurrent_launch.numerical,
            vec![NumericalChoice::Reassociate {
                node: parallel.clone()
            }]
        );
        for proposal in [serialized, concurrent] {
            let shape = form(facts, &profile, (*proposal).clone()).unwrap_or_else(|defect| panic!("{defect}"));
            assert_complete(facts, proposal, &shape);
            let block_id = shape.block_of(&parallel).expect("the loop is absorbed");
            let block = &shape.blocks()[block_id];
            assert_eq!(block.participants.independent_axes.len(), 1);
            let expected = if is_serial_witness(proposal) {
                assert_eq!(block.participants.policy, ParticipantPolicy::Serial);
                Vec::new()
            } else {
                assert!(matches!(block.participants.policy, ParticipantPolicy::Linear { .. }));
                vec![NumericalChoice::Reassociate {
                    node: parallel.clone(),
                }]
            };
            assert_eq!(block.numerical, expected);
        }
    }

    const REDUCE: &str = "fn m[N](x: &tensor[N] f32) -> f32:\n    return reduce(f32(x), 0, max)\n";

    #[test]
    fn a_static_reduction_extent_is_proved_and_each_node_is_its_own_streaming_launch() {
        let logical = logical(REDUCE, "m", &[("N", 16)]);
        let forest = OccurrenceForest::expand(&logical).unwrap_or_else(|defect| panic!("{defect}"));
        let facts = forest.facts();
        let profile = profile();
        let entry = facts.entry();
        let proposals = proposals(facts, &profile, entry, 0);
        let streaming = proposals
            .iter()
            .find(|proposal| proposal.rule == STREAMING)
            .expect("the streaming rule proposes");
        let point = proposals
            .iter()
            .find(|proposal| proposal.rule == POINT_SERIAL)
            .expect("the point-serial rule proposes");
        let owned = owned_nodes(facts, &streaming.ownership);
        assert_eq!(streaming.launches.len(), owned.len(), "one launch per node");
        assert_eq!(point.launches.len(), 1, "one launch for the whole graph");
        for proposal in [streaming, point] {
            let shape = form(facts, &profile, proposal.clone()).unwrap_or_else(|defect| panic!("{defect}"));
            assert_complete(facts, proposal, &shape);
            assert!(shape.obligations().values().all(|disposition| matches!(
                disposition,
                ObligationDisposition::Static { .. }
            )));
        }
    }
}
