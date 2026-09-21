//! The CUDA mapping catalog: optional mapping rules and the typed intrinsic
//! families authorized on this target, plus the backend cost model and
//! native-fact domains.
//!
//! CUDA declares exactly two optional rule families beside the core-owned
//! universal rules:
//!
//! 1. `cuda.subgroup-cooperative` — one workgroup of a warp-multiple width
//!    cooperating through the `cuda.subgroup` intrinsic family.
//! 2. `cuda.grid-cooperative` — one cooperative grid launch across a
//!    whole-result dependency chain (producer fully complete before the
//!    consumer), with one device-arena kernel-local residence per
//!    intermediate. The rule declines unless the target profile has the
//!    cooperative-grid facility and every capability the region applies is
//!    an effective signature.
//!
//! Both rules are expressed over S1's core-produced pattern facts — the
//! streaming segmentation, capability uses, and phase crossings of the
//! occurrence's region tree — and select mapping choices only: no rule
//! examines a logical node payload or walks a graph. Both decline before
//! proposing whenever their admission test fails; a proposal they do emit
//! is completed by core formation or is a defect of this package.

use crate::intrinsics::{CudaIntrinsicCatalog, Dialect};
use seismic_lang::intrinsics::IntrinsicId;
use seismic_realization::ids::{OwnedGraphKey, OwnedNodeRef};
use seismic_realization::residence::{Replication, StorageScope};
use seismic_realization::strategy::{
    self, LaunchGroup, LaunchProposal, LocalResidenceRequirement, MappingCatalog, MappingProposal,
    MappingRule, NativeFactDomain, NativeFactKind, NodePlacement, OwnershipProposal,
    ParticipantPolicy, RuleQuery, StreamingSegment, TuningDeclaration, TuningRef,
};
use seismic_realization::target::{CooperativeGrid, TargetLimits};
use std::collections::BTreeMap;

/// The CUDA mapping catalog over the CUDA dialect.
pub struct CudaCatalog {
    rules: Vec<Box<dyn MappingRule>>,
    intrinsics: CudaIntrinsicCatalog,
    cost: CostRanking,
    native_fact_domains: Vec<NativeFactDomain>,
    limits: TargetLimits,
}

impl CudaCatalog {
    pub fn new(
        limits: &crate::mapping::Limits,
        cooperative: Option<CooperativeGrid>,
        estimate: &crate::mapping::EstimateModel,
    ) -> Self {
        // The resident-participant domain exists only where the facility
        // does (a cooperative launch is otherwise unproposable); the warp
        // is the subgroup on every supported CUDA target.
        let mut native_fact_domains = vec![NativeFactDomain {
            kind: NativeFactKind::NativeSubgroupWidth,
            min: 32,
            max: 32,
        }];
        if let Some(grid) = cooperative.as_ref() {
            native_fact_domains.push(NativeFactDomain {
                kind: NativeFactKind::MaxResidentParticipants,
                min: 1,
                max: grid.max_resident_participants,
            });
        }
        CudaCatalog {
            rules: vec![
                Box::new(SubgroupCooperative),
                Box::new(GridCooperativeLaunch),
            ],
            intrinsics: CudaIntrinsicCatalog,
            cost: CostRanking {
                launch_ns: estimate.launch_ns.ceil() as u64,
                model: estimate.clone(),
            },
            native_fact_domains,
            limits: crate::target::TargetProfile::synthetic_baseline(
                limits.clone(),
                cooperative.clone(),
            )
            .effective_limits(),
        }
    }
}

impl MappingCatalog<Dialect> for CudaCatalog {
    fn rules(&self) -> &[Box<dyn MappingRule>] {
        &self.rules
    }

    fn intrinsics(&self) -> &dyn seismic_realization::kernel::IntrinsicCatalog<Dialect> {
        &self.intrinsics
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

/// The backend cost model: uncalibrated ranking coefficients under the
/// `cuda-estimate-unqualified-v0` identity. Ranking only, never legality.
struct CostRanking {
    launch_ns: u64,
    model: crate::mapping::EstimateModel,
}

impl seismic_realization::strategy::CostModel for CostRanking {
    fn launch_overhead_ns(&self) -> u64 {
        self.launch_ns
    }

    fn point_cost_ns(&self, op: &seismic_realization::kernel::CostUnit) -> u64 {
        crate::intrinsics::point_cost_ns(op, &self.model)
    }
}

// ---------------------------------------------------------------------------
// Shared pattern-fact admission
// ---------------------------------------------------------------------------

/// The root region of the queried occurrence's alternative.
fn root_of(query: &RuleQuery<'_, '_>) -> seismic_realization::ids::OwnedRegionRef {
    strategy::root_region(OwnedGraphKey {
        occurrence: query.occurrence,
        logical_alternative: query.logical_alternative,
    })
}

/// The nodes of one region streamed wholly into launch groups, in
/// segmentation order: `None` when any segment is retained (a retained
/// step cannot join one launch).
fn streamed_group_nodes(
    query: &RuleQuery<'_, '_>,
    root: &seismic_realization::ids::OwnedRegionRef,
) -> Option<Vec<OwnedNodeRef>> {
    let mut nodes = Vec::new();
    for segment in strategy::streaming_segments(query.facts, root) {
        match segment {
            StreamingSegment::Group(group) => nodes.extend(group.nodes),
            StreamingSegment::Retained(_) => return None,
        }
    }
    Some(nodes)
}

/// The `cuda.subgroup` capability signatures the region applies
/// (deduplicated, in pre-order), when every capability the region applies
/// — of any family — is an effective signature of the target; `None` when
/// some applied capability is inadmissible (it is never lowered).
fn subgroup_uses(
    query: &RuleQuery<'_, '_>,
    root: &seismic_realization::ids::OwnedRegionRef,
) -> Option<Vec<IntrinsicId>> {
    let uses = strategy::capability_uses(query.facts, root);
    if !uses
        .iter()
        .all(|use_| query.profile.effective_signatures.contains(&use_.intrinsic))
    {
        return None;
    }
    let mut applied: Vec<IntrinsicId> = Vec::new();
    for use_ in uses {
        if use_.intrinsic.capability.backend == crate::mapping::TARGET
            && use_.intrinsic.capability.name == "subgroup"
            && !applied.contains(&use_.intrinsic)
        {
            applied.push(use_.intrinsic);
        }
    }
    Some(applied)
}

/// One proposal owning every streamed node of the region in a single
/// launch group under `participants`.
fn whole_region_proposal(
    query: &RuleQuery<'_, '_>,
    rule: strategy::RuleName,
    nodes: Vec<OwnedNodeRef>,
    participants: ParticipantPolicy,
    local_residences: Vec<LocalResidenceRequirement>,
    required_intrinsics: Vec<IntrinsicId>,
    tuning: Vec<TuningDeclaration>,
) -> MappingProposal {
    let ownership = OwnershipProposal {
        root: seismic_realization::ids::OwnedOccurrence {
            occurrence: query.occurrence,
            logical_alternative: query.logical_alternative,
        },
        absorbed: BTreeMap::new(),
    };
    let placement = nodes
        .iter()
        .map(|node| (node.clone(), NodePlacement::Launch(LaunchGroup(0))))
        .collect::<BTreeMap<_, _>>();
    let launches = seismic_lang::logical::IdVec::new(vec![LaunchProposal {
        participants,
        algorithm: seismic_realization::strategy::AlgorithmChoice::Universal,
        local_residences,
        numerical: Vec::new(),
    }]);
    MappingProposal {
        rule,
        ownership,
        placement,
        launches,
        required_intrinsics: required_intrinsics.into_iter().collect(),
        tuning,
    }
}

// ---------------------------------------------------------------------------
// Rule 1: subgroup-cooperative
// ---------------------------------------------------------------------------

/// One workgroup of a warp-multiple participant width cooperating through
/// the `cuda.subgroup` intrinsic family.
struct SubgroupCooperative;

impl MappingRule for SubgroupCooperative {
    fn name(&self) -> strategy::RuleName {
        "cuda.subgroup-cooperative"
    }

    fn propose(&self, query: &RuleQuery<'_, '_>) -> Vec<MappingProposal> {
        let root = root_of(query);
        // Admission: the region applies at least one authorized
        // `cuda.subgroup` capability.
        // Admission: the region applies at least one authorized
        // `cuda.subgroup` capability.
        let applied = match subgroup_uses(query, &root) {
            Some(applied) => applied,
            None => return Vec::new(),
        };
        let family = match applied.as_slice() {
            [first, ..] => first.clone(),
            [] => return Vec::new(),
        };
        // Admission: the region streams wholly into launch groups (a
        // retained step cannot join the one cooperative launch).
        let Some(nodes) = streamed_group_nodes(query, &root) else {
            return Vec::new();
        };
        let max_width = query.profile.limits.max_participants;
        let width = TuningDeclaration {
            name: format!("cuda.subgroup.width.{}", query.occurrence.0),
            lower: u64::from(crate::mapping::WARP),
            upper: max_width.max(u64::from(crate::mapping::WARP)),
        };
        vec![whole_region_proposal(
            query,
            self.name(),
            nodes,
            ParticipantPolicy::Cooperative {
                width: TuningRef(0),
                family,
            },
            Vec::new(),
            applied,
            vec![width],
        )]
    }
}

// ---------------------------------------------------------------------------
// Rule 2: grid-cooperative
// ---------------------------------------------------------------------------

/// One cooperative grid launch across a whole-result dependency chain of
/// the region, with one device-arena kernel-local residence per
/// intermediate and grid-wide barriers between the phases.
struct GridCooperativeLaunch;

impl MappingRule for GridCooperativeLaunch {
    fn name(&self) -> strategy::RuleName {
        "cuda.grid-cooperative"
    }

    fn propose(&self, query: &RuleQuery<'_, '_>) -> Vec<MappingProposal> {
        // Admission: the target has the cooperative-grid facility.
        let Some(cooperative) = query.profile.limits.cooperative_grid.clone() else {
            return Vec::new();
        };
        let root = root_of(query);
        // Admission: the region streams wholly into launch groups.
        let Some(nodes) = streamed_group_nodes(query, &root) else {
            return Vec::new();
        };
        // Admission: at least one whole-result crossing between distinct
        // top-level phases (producer fully complete before the consumer);
        // otherwise the universal rules already cover the region.
        let crossings = strategy::phase_crossings(query.facts, &root);
        if crossings.is_empty() {
            return Vec::new();
        }
        // Admission: every capability the region applies is an effective
        // signature (an inadmissible use is never lowered).
        let applied = match subgroup_uses(query, &root) {
            Some(applied) => applied,
            None => return Vec::new(),
        };
        // One device-arena kernel-local residence per whole-result
        // intermediate: the grid barrier separates the producer's store
        // into it from the consumer's first load.
        let local_residences = crossings
            .iter()
            .map(|crossing| LocalResidenceRequirement {
                value: crossing.value,
                scope: StorageScope::DeviceArena,
                replication: Replication::Once,
            })
            .collect::<Vec<_>>();
        let participants = TuningDeclaration {
            name: format!("cuda.grid.participants.{}", query.occurrence.0),
            lower: 1,
            upper: cooperative
                .max_resident_participants
                .min(query.profile.limits.max_participants)
                .max(1),
        };
        vec![whole_region_proposal(
            query,
            self.name(),
            nodes,
            ParticipantPolicy::GridCooperative {
                participants: TuningRef(0),
            },
            local_residences,
            applied,
            vec![participants],
        )]
    }
}
