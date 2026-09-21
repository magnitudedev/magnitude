//! The Metal mapping catalog: the backend type, its declarative optional
//! mapping rules (streaming, blocked, subgroup, matrix — nothing else), the
//! probe-calibrated cost model, and the pipeline `Backend` implementation.
//!
//! Every rule consumes S1's core-produced pattern facts
//! (`streaming_segments`, `scannable_loops`, `tile_candidates`,
//! `reduction_shapes`, `capability_uses`) plus the effective target
//! profile, and declines before proposing when its predicate fails. No
//! rule touches a logical graph: no `LogicalNode` payloads, no node-kind
//! matches, no recursive walks. Universal coverage comes from the
//! core-owned universal rules; these are optimized peers only.

use crate::estimate::EstimateModel;
use crate::intrinsics::{MetalDialect, MetalIntrinsicCatalog, SUBGROUP_WIDTH};
use crate::target::{Limits, TargetProfile};
use seismic_compiler::pipeline::{AssemblyFailure, Backend, EncodedPlan};
use seismic_lang::{
    intrinsics::{IntrinsicId, ReduceOp},
    logical::ReductionOrder,
    sir::IntrinsicUse,
};
use seismic_realization::ids::OwnedGraphKey;
use seismic_realization::kernel::{CostUnit, IntrinsicCatalog};
use seismic_realization::strategy::{
    capability_uses, reduction_shapes, root_region, scannable_loops, streaming_segments,
    tile_candidates, AlgorithmChoice, CapabilityUse, LaunchGroup, LaunchProposal, MappingCatalog,
    MappingProposal, MappingRule, NativeFactDomain, NativeFactKind, NodePlacement,
    NumericalChoice, OwnershipProposal, ParticipantPolicy, RuleName, RuleQuery, ScanCombine,
    StreamingGroup, StreamingGroupKind, StreamingSegment, TuningDeclaration, TuningRef,
};
use seismic_realization::target::{EffectiveTargetProfile, TargetLimits};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// The catalog
// ---------------------------------------------------------------------------

/// The Metal mapping catalog: optional rules, the cost model, the declared
/// native-fact domains, and the target limits. Construction validates that
/// every declared intrinsic family has an encoder: `lower` matches the
/// registry's `IntrinsicLowering` exhaustively with named arms and no
/// wildcard, so coverage is total by construction over the effective
/// signature set. The estimate model is the calibrated constant model,
/// valid by construction.
pub struct MetalCatalog {
    rules: Vec<Box<dyn MappingRule>>,
    cost: MetalCostModel,
    limits: TargetLimits,
    native_fact_domains: Vec<NativeFactDomain>,
}

impl MetalCatalog {
    pub fn new(profile: &TargetProfile) -> Self {
        let effective = profile.effective_profile();
        MetalCatalog {
            rules: vec![
                Box::new(MetalStreaming),
                Box::new(MetalBlocked),
                Box::new(MetalSubgroup),
                Box::new(MetalMatrix),
            ],
            cost: MetalCostModel {
                estimate: EstimateModel::calibrated(),
            },
            limits: effective.limits.clone(),
            // Every subgroup collective this backend emits operates on the
            // fixed 32-lane topology; the assembler reflects the native
            // width against exactly this domain.
            native_fact_domains: vec![NativeFactDomain {
                kind: NativeFactKind::NativeSubgroupWidth,
                min: u64::from(SUBGROUP_WIDTH),
                max: u64::from(SUBGROUP_WIDTH),
            }],
        }
    }

    pub fn estimate(&self) -> &EstimateModel {
        &self.cost.estimate
    }
}

impl MappingCatalog<MetalDialect> for MetalCatalog {
    fn rules(&self) -> &[Box<dyn MappingRule>] {
        &self.rules
    }

    fn intrinsics(&self) -> &dyn IntrinsicCatalog<MetalDialect> {
        &MetalIntrinsicCatalog
    }

    fn cost_model(&self) -> &dyn seismic_realization::strategy::CostModel {
        &self.cost
    }

    fn native_fact_domains(&self) -> &[NativeFactDomain] {
        &self.native_fact_domains
    }

    fn limits(&self) -> &TargetLimits {
        &self.limits
    }
}

/// The backend cost model over `CostUnit`: ranking only, never legality.
/// Coefficients are the probe-calibrated constants of the estimate model
/// (see `estimate.rs` for provenance).
struct MetalCostModel {
    estimate: EstimateModel,
}

const LANE_OP_COST_NS: u64 = 2;
const ACCESS_COST_NS: u64 = 4;
const MATH_COST_NS: u64 = 8;
const ATOMIC_COST_NS: u64 = 16;
const FOLD_COST_NS: u64 = 8;
const COLLECTIVE_COST_NS: u64 = 25;

impl seismic_realization::strategy::CostModel for MetalCostModel {
    fn launch_overhead_ns(&self) -> u64 {
        self.estimate.launch_ns.ceil() as u64
    }

    fn point_cost_ns(&self, op: &CostUnit) -> u64 {
        match op {
            CostUnit::Scalar => LANE_OP_COST_NS,
            CostUnit::Load { .. } | CostUnit::Store { .. } | CostUnit::PlaneAccess { .. } => {
                ACCESS_COST_NS
            }
            CostUnit::Math { .. } => MATH_COST_NS,
            CostUnit::Atomic { .. } => ATOMIC_COST_NS,
            CostUnit::Fold { .. } => FOLD_COST_NS,
            CostUnit::Barrier => COLLECTIVE_COST_NS,
            CostUnit::Intrinsic(_) => COLLECTIVE_COST_NS,
        }
    }
}

// ---------------------------------------------------------------------------
// The backend type
// ---------------------------------------------------------------------------

/// The Metal backend: the effective target profile and its mapping catalog.
pub struct Metal {
    profile: TargetProfile,
    effective: EffectiveTargetProfile,
    catalog: MetalCatalog,
}

impl Metal {
    /// A planning backend from explicit hard limits (no device observation).
    pub fn synthetic(limits: Limits) -> Self {
        Self::from_profile(TargetProfile::synthetic(limits))
    }

    #[cfg(target_os = "macos")]
    pub fn from_device_info(info: &crate::runtime::DeviceInfo) -> Self {
        Self::from_profile(TargetProfile::from_evidence(
            &info.capability_fingerprint(),
            &info.profile.scalar_dtypes.value,
            &info.profile.matrix_dtypes.value,
            &info.profile.matrix_combinations.value,
            info.max_threads_per_threadgroup,
            info.max_threadgroup_bytes,
            info.max_buffer_bytes,
            info.profile.private_storage_budget_bytes.value,
        ))
    }

    fn from_profile(profile: TargetProfile) -> Self {
        let effective = profile.effective_profile();
        Metal {
            catalog: MetalCatalog::new(&profile),
            effective,
            profile,
        }
    }

    pub fn target_profile(&self) -> &TargetProfile {
        &self.profile
    }

    pub fn effective_profile(&self) -> &EffectiveTargetProfile {
        &self.effective
    }

    pub fn catalog(&self) -> &MetalCatalog {
        &self.catalog
    }
}

/// The compiling Metal backend: the catalog plus the device that owns
/// native assembly. Exists on macOS only (the Metal runtime does).
#[cfg(target_os = "macos")]
pub struct MetalCompiler<'a> {
    metal: Metal,
    device: &'a crate::runtime::Device,
}

#[cfg(target_os = "macos")]
impl<'a> MetalCompiler<'a> {
    pub fn from_device(device: &'a crate::runtime::Device) -> Self {
        Self {
            metal: Metal::from_device_info(&device.info()),
            device,
        }
    }

    pub fn planner(&self) -> &Metal {
        &self.metal
    }
}

#[cfg(target_os = "macos")]
impl Backend for MetalCompiler<'_> {
    type Dialect = MetalDialect;
    type Catalog = MetalCatalog;
    type EncodedLaunch = crate::encode::EncodedLaunch;
    type NativeArtifact = crate::native::NativeArtifact;

    fn profile(&self) -> &EffectiveTargetProfile {
        self.metal.effective_profile()
    }

    fn catalog(&self) -> &Self::Catalog {
        &self.metal.catalog
    }

    fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String> {
        self.metal.profile.supports_intrinsic(intrinsic)
    }

    /// Exhaustive mechanical encoding of one sealed launch. Total.
    fn encode(
        &self,
        launch: &seismic_realization::physical::SealedLaunch<Self::Dialect>,
    ) -> Self::EncodedLaunch {
        crate::encode::encode(launch)
    }

    /// Compile every encoded launch, reflect native facts against their
    /// declared domains, fold into direct handles, and seal one native tree
    /// mirroring the physical schedule exactly once.
    fn assemble(
        &self,
        encoded: EncodedPlan<Self::Dialect, Self::EncodedLaunch>,
    ) -> Result<Self::NativeArtifact, AssemblyFailure> {
        crate::native::assemble(self.device, &encoded)
    }
}

// ---------------------------------------------------------------------------
// Shared declarative helpers over the pattern facts
// ---------------------------------------------------------------------------

/// Whether the exact registry signature of one intrinsic use is effective:
/// the use's id is an effective signature and one registry entry of that
/// id admits the use's concrete argument and result types.
fn exact_signature_effective(profile: &EffectiveTargetProfile, use_id: &IntrinsicId) -> bool {
    if !profile.effective_signatures.contains(use_id) {
        return false;
    }
    seismic_lang::intrinsics::lookup(
        use_id.capability.backend.as_str(),
        use_id.capability.name.as_str(),
        use_id.name.as_str(),
    )
    .into_iter()
    .any(|signature| signature.id == *use_id)
}

/// Every capability intrinsic the alternative's graph applies, when the
/// effective target profile authorizes all of them. `None` declines:
/// logical construction already removed alternatives requiring unsupported
/// capabilities, so a rule never proposes for one.
fn effective_capability_uses(
    query: &RuleQuery<'_, '_>,
    root: &seismic_realization::ids::OwnedRegionRef,
) -> Option<BTreeSet<IntrinsicId>> {
    let mut required = BTreeSet::new();
    for use_ in capability_uses(query.facts, root) {
        if !query.profile.effective_signatures.contains(&use_.intrinsic) {
            return None;
        }
        required.insert(use_.intrinsic);
    }
    Some(required)
}

/// The `metal.matrix` capability uses of the graph, from the shared fact.
fn matrix_uses(query: &RuleQuery<'_, '_>, root: &seismic_realization::ids::OwnedRegionRef) -> Vec<CapabilityUse> {
    capability_uses(query.facts, root)
        .into_iter()
        .filter(|use_| use_.intrinsic.capability.name == "matrix")
        .collect()
}

/// One complete proposal over the core-produced streaming segmentation.
/// `choose` receives each group, its ordinal, and the proposal's tuning
/// accumulator (references are allocated at the position they occupy, the
/// same counting construction the core's universal rules use).
struct ProposalBuilder {
    rule: RuleName,
    key: OwnedGraphKey,
    required: BTreeSet<IntrinsicId>,
}

struct GroupChoice {
    participants: ParticipantPolicy,
    algorithm: AlgorithmChoice,
    numerical: Vec<NumericalChoice>,
}

type GroupChooser<'a> =
    dyn FnMut(&StreamingGroup, usize, &mut Vec<TuningDeclaration>) -> GroupChoice + 'a;

impl ProposalBuilder {
    fn proposal(
        &self,
        segments: &[StreamingSegment],
        choose: &mut GroupChooser<'_>,
    ) -> MappingProposal {
        let mut placement = BTreeMap::new();
        let mut launches = Vec::new();
        let mut tuning = Vec::new();
        for (index, segment) in segments.iter().enumerate() {
            match segment {
                StreamingSegment::Retained(node) => {
                    placement.insert(node.clone(), NodePlacement::Retained);
                }
                StreamingSegment::Group(group) => {
                    let launch = LaunchGroup(index as u32);
                    for node in &group.nodes {
                        placement.insert(node.clone(), NodePlacement::Launch(launch));
                    }
                    let choice = choose(group, index, &mut tuning);
                    launches.push(LaunchProposal {
                        participants: choice.participants,
                        algorithm: choice.algorithm,
                        // Kernel-local residences are created by the core
                        // (D1) from what the algorithm requires; this
                        // backend stages no storage.
                        local_residences: Vec::new(),
                        numerical: choice.numerical,
                    });
                }
            }
        }
        MappingProposal {
            rule: self.rule,
            ownership: OwnershipProposal {
                root: seismic_realization::ids::OwnedOccurrence {
                    occurrence: self.key.occurrence,
                    logical_alternative: self.key.logical_alternative,
                },
                absorbed: BTreeMap::new(),
            },
            placement,
            launches: seismic_lang::logical::IdVec::new(launches),
            required_intrinsics: self.required.clone(),
            tuning,
        }
    }
}

/// Declare one linear participant parameter for group `index` and return
/// its reference.
fn declare_participants(
    index: usize,
    limits: &TargetLimits,
    tuning: &mut Vec<TuningDeclaration>,
) -> TuningRef {
    let reference = TuningRef(tuning.len() as u32);
    tuning.push(TuningDeclaration {
        name: format!("participants.launch{index}"),
        lower: 1,
        upper: limits.max_participants,
    });
    reference
}

/// Declare one further parameter of group `index` and return its reference.
fn declare_tuning(
    index: usize,
    name: &str,
    lower: u64,
    upper: u64,
    tuning: &mut Vec<TuningDeclaration>,
) -> TuningRef {
    let reference = TuningRef(tuning.len() as u32);
    tuning.push(TuningDeclaration {
        name: format!("{name}.launch{index}"),
        lower,
        upper,
    });
    reference
}

fn serial_choice() -> GroupChoice {
    GroupChoice {
        participants: ParticipantPolicy::Serial,
        algorithm: AlgorithmChoice::Universal,
        numerical: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// metal.streaming — the windowed online-scan rule
// ---------------------------------------------------------------------------

const STREAMING: RuleName = "metal.streaming";

/// The windowed online-scan rule: an ordered loop whose carried lanes match
/// the structural scan rule windows the scanned axis over a solver-tunable
/// width. The applicability predicate is the former `streaming_admitted`:
/// some ordered loop of the graph matches the scan rule (the shared
/// `scannable_loops` fact). The window is declared as the launch's blocked
/// tile; the additive lanes' ascending fold reassociates across windows
/// (the recorded `Reassociate` freedom, exactly the former numerical note).
struct MetalStreaming;

impl MappingRule for MetalStreaming {
    fn name(&self) -> RuleName {
        STREAMING
    }

    fn propose(&self, query: &RuleQuery<'_, '_>) -> Vec<MappingProposal> {
        let key = OwnedGraphKey {
            occurrence: query.occurrence,
            logical_alternative: query.logical_alternative,
        };
        let root = root_region(key);
        let Some(required) = effective_capability_uses(query, &root) else {
            return Vec::new();
        };
        let scans = scannable_loops(query.facts, &root);
        if scans.is_empty() {
            return Vec::new();
        }
        let segments = streaming_segments(query.facts, &root);
        let builder = ProposalBuilder {
            rule: STREAMING,
            key,
            required,
        };
        vec![builder.proposal(&segments, &mut |group, index, tuning| {
            let Some(scan) = scans.iter().find(|scan| group.nodes.contains(&scan.node)) else {
                return serial_choice();
            };
            let numerical: Vec<NumericalChoice> = scan
                .state
                .lanes
                .iter()
                .any(|lane| matches!(lane.combine, ScanCombine::Additive))
                .then(|| NumericalChoice::Reassociate {
                    node: scan.node.clone(),
                })
                .into_iter()
                .collect();
            let window = declare_tuning(
                index,
                "window",
                1,
                scan.state.axis_capacity,
                tuning,
            );
            GroupChoice {
                participants: ParticipantPolicy::Serial,
                algorithm: AlgorithmChoice::Blocked { tile: window },
                numerical,
            }
        })]
    }
}

// ---------------------------------------------------------------------------
// metal.blocked — the tiled independent-domain rule
// ---------------------------------------------------------------------------

const BLOCKED: RuleName = "metal.blocked";

/// The tiled traversal rule: an absorbable independent loop's domain is
/// covered by linear participants over a solver-tunable tile. The
/// applicability predicate ports the former `blocked_admitted` where it
/// transfers to the sealed algebra: every reduction of the graph is
/// source-unordered (admitted reassociation) and not argmax, so tiling
/// cannot break ordered-reduction semantics — and at least one tile
/// candidate exists (the shared `tile_candidates` fact carries the checked
/// axis capacity; underivable or zero capacities are not admitted, so no
/// capacity default exists).
///
/// The former blocked *reduction* emission (workgroup partials combined by
/// a lane-ordered tree) is not expressible as a declarative proposal over
/// the frozen vocabulary: the core lowers the reference fold only and a
/// reassociating reduction realization must be a typed intrinsic family
/// (`AlgorithmChoice::Intrinsic` / a cooperative participant policy). No
/// such registry family exists today; the rule therefore never proposes one
/// (see the lane report: a workgroup-tree family is a registry extension,
/// never a backend special case).
struct MetalBlocked;

impl MappingRule for MetalBlocked {
    fn name(&self) -> RuleName {
        BLOCKED
    }

    fn propose(&self, query: &RuleQuery<'_, '_>) -> Vec<MappingProposal> {
        let key = OwnedGraphKey {
            occurrence: query.occurrence,
            logical_alternative: query.logical_alternative,
        };
        let root = root_region(key);
        let Some(required) = effective_capability_uses(query, &root) else {
            return Vec::new();
        };
        // The predicate: every reduction unordered and not argmax (and at
        // least one reduction exists), plus a tile candidate.
        let reductions = reduction_shapes(query.facts, &root);
        if reductions.is_empty() {
            return Vec::new();
        }
        if reductions
            .iter()
            .any(|shape| shape.op == ReduceOp::Argmax || shape.order != ReductionOrder::Unordered)
        {
            return Vec::new();
        }
        let candidates = tile_candidates(query.facts, &root);
        if candidates.is_empty() {
            return Vec::new();
        }
        let segments = streaming_segments(query.facts, &root);
        let builder = ProposalBuilder {
            rule: BLOCKED,
            key,
            required,
        };
        let limits = &query.profile.limits;
        vec![builder.proposal(&segments, &mut |group, index, tuning| {
            let StreamingGroupKind::IndependentLoop { loop_node, .. } = &group.kind else {
                return serial_choice();
            };
            let Some(candidate) = candidates
                .iter()
                .find(|candidate| &candidate.node == loop_node)
            else {
                return serial_choice();
            };
            let participants = declare_participants(index, limits, tuning);
            let tile = declare_tuning(index, "tile", 1, candidate.axis_capacity, tuning);
            GroupChoice {
                participants: ParticipantPolicy::Linear { participants },
                algorithm: AlgorithmChoice::Blocked { tile },
                numerical: Vec::new(),
            }
        })]
    }
}

// ---------------------------------------------------------------------------
// metal.subgroup — the subgroup collective reduction rule
// ---------------------------------------------------------------------------

const SUBGROUP_RULE: RuleName = "metal.subgroup";

/// The subgroup collective rule: a reduction is realized by one subgroup of
/// the `simd_sum` collective. The applicability predicate ports the former
/// `subgroup_admitted` exactly: every reduction of the graph is
/// source-unordered `sum` and the `simd_sum` collective signature is
/// effective on the target (and at least one reduction exists).
struct MetalSubgroup;

impl MappingRule for MetalSubgroup {
    fn name(&self) -> RuleName {
        SUBGROUP_RULE
    }

    fn propose(&self, query: &RuleQuery<'_, '_>) -> Vec<MappingProposal> {
        let key = OwnedGraphKey {
            occurrence: query.occurrence,
            logical_alternative: query.logical_alternative,
        };
        let root = root_region(key);
        let Some(mut required) = effective_capability_uses(query, &root) else {
            return Vec::new();
        };
        let reductions = reduction_shapes(query.facts, &root);
        if reductions.is_empty() {
            return Vec::new();
        }
        if reductions.iter().any(|shape| {
            shape.op != ReduceOp::Sum || shape.order != ReductionOrder::Unordered
        }) {
            return Vec::new();
        }
        let collective = crate::intrinsics::subgroup_intrinsic("simd_sum");
        if !exact_signature_effective(query.profile, &collective) {
            return Vec::new();
        }
        required.insert(collective.clone());
        let segments = streaming_segments(query.facts, &root);
        let builder = ProposalBuilder {
            rule: SUBGROUP_RULE,
            key,
            required,
        };
        vec![builder.proposal(&segments, &mut |group, index, tuning| {
            let has_reduction = group
                .nodes
                .iter()
                .any(|node| reductions.iter().any(|shape| shape.node == *node));
            if !has_reduction {
                return serial_choice();
            }
            // One subgroup per output: the fixed 32-lane collective
            // topology, expressed as a degenerate tuning domain.
            let width = declare_tuning(
                index,
                "subgroup-width",
                u64::from(SUBGROUP_WIDTH),
                u64::from(SUBGROUP_WIDTH),
                tuning,
            );
            GroupChoice {
                participants: ParticipantPolicy::Cooperative {
                    width,
                    family: crate::intrinsics::subgroup_intrinsic("simd_sum"),
                },
                algorithm: AlgorithmChoice::Universal,
                numerical: Vec::new(),
            }
        })]
    }
}

// ---------------------------------------------------------------------------
// metal.matrix — the matrix capability launch rule
// ---------------------------------------------------------------------------

const MATRIX: RuleName = "metal.matrix";

/// The matrix capability rule: a `metal.matrix.{matmul, matmul_add}` use is
/// realized by one launch over the output element domain (linear
/// participants, one ascending-k accumulation chain per participant). The
/// applicability predicate ports the former `map_matrix_node` admission:
/// every matrix capability use of the graph has an effective exact
/// signature, and every launch group's matrix uses share one signature.
struct MetalMatrix;

impl MappingRule for MetalMatrix {
    fn name(&self) -> RuleName {
        MATRIX
    }

    fn propose(&self, query: &RuleQuery<'_, '_>) -> Vec<MappingProposal> {
        let key = OwnedGraphKey {
            occurrence: query.occurrence,
            logical_alternative: query.logical_alternative,
        };
        let root = root_region(key);
        let Some(required) = effective_capability_uses(query, &root) else {
            return Vec::new();
        };
        let uses = matrix_uses(query, &root);
        if uses.is_empty() {
            return Vec::new();
        }
        if uses
            .iter()
            .any(|use_| !exact_signature_effective(query.profile, &use_.intrinsic))
        {
            return Vec::new();
        }
        let segments = streaming_segments(query.facts, &root);
        // A group whose matrix uses name more than one signature cannot be
        // one algorithm choice; the rule declines rather than propose
        // incompletely.
        for segment in &segments {
            let StreamingSegment::Group(group) = segment else {
                continue;
            };
            let mut ids = BTreeSet::new();
            for use_ in &uses {
                if group.nodes.contains(&use_.node) {
                    ids.insert(use_.intrinsic.clone());
                }
            }
            if ids.len() > 1 {
                return Vec::new();
            }
        }
        let builder = ProposalBuilder {
            rule: MATRIX,
            key,
            required,
        };
        let limits = &query.profile.limits;
        vec![builder.proposal(&segments, &mut |group, index, tuning| {
            let ids: BTreeSet<IntrinsicId> = uses
                .iter()
                .filter(|use_| group.nodes.contains(&use_.node))
                .map(|use_| use_.intrinsic.clone())
                .collect();
            let Some(id) = ids.into_iter().next() else {
                return serial_choice();
            };
            let participants = declare_participants(index, limits, tuning);
            GroupChoice {
                participants: ParticipantPolicy::Linear { participants },
                algorithm: AlgorithmChoice::Intrinsic(id),
                numerical: Vec::new(),
            }
        })]
    }
}
