//! Hierarchical workgroup/subgroup mapping and storage ownership.
//!
//! One independent *outer* domain maps to workgroups; nested independent
//! *inner* work maps to subgroups within one workgroup. The construction is
//! a `LinearIterationMap` over `[outer axes…, inner axes…]` (outermost first)
//! whose physical linear participant count equals the inner domain: the
//! row-major delinearization then assigns each *outer* coordinate to exactly
//! one workgroup (workgroup id = high-order digits) and each *inner*
//! coordinate to one participant inside that workgroup (participant id =
//! low-order digits). [`HierarchicalMap::coordinate_of`] and
//! [`HierarchicalMap::subgroup_of`] are the exact coordinate maps; both the
//! emitter and the solver consume the same map, so the emitter never changes
//! the cover independently.
//!
//! Ownership: values owned outside and read by inner groups are
//! eligible for one workgroup staging allocation — the strategy reports the
//! exact stage-tile bytes/alignment as a `SizeExpr`
//! ([`HierarchicalMap::stage_tile_bytes`]); the staging `ConditionalStorageTemplate`
//! (scope `Workgroup`, exact replication) is created by the realization
//! surface.
//! Inner-private values are subgroup/participant storage.
//!
//! Group count, workgroup width, and pool pressure are modeled exactly:
//! workgroups and participants are planning expressions over the family's
//! plan parameters and are constrained by the one global model
//! (`1 <= preferred_participants <= target.max_participants`,
//! `explicit_workgroup_bytes <= target.max_workgroup_bytes`).

use crate::strategies::cost::CostModelId;
use seismic_lang::{
    logical::RuntimeExtent,
    sym::Sym,
    types::{ExtentExpr, RuntimeExtentId},
};
use seismic_realization::dispatch::{LinearIterationMap, LinearMapError, Traversal};
use seismic_realization::executable::{
    AlternativeBuilder, BuilderError, ExecutableDialect, FusedStrategyTemplate, Legalized, NodeRef,
};
use std::collections::BTreeMap;

/// Backend-reported subgroup facts (an input from the backend's target
/// profile surface; the strategy library never decides them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubgroupFacts {
    /// Exact hardware subgroup width, when the backend reports one.
    pub width: Option<u32>,
    /// Exact maximum participants per workgroup.
    pub max_participants_per_workgroup: u32,
}

/// Why a hierarchical mapping does not exist for one occurrence. This is
/// ordinary inapplicability of an optional strategy: the family assembly
/// simply does not add the alternative, and the universal strategy remains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HierarchicalError {
    /// The inner domain exceeds the workgroup's exact participant bound.
    InnerDomainExceedsWorkgroup { inner_capacity: u64, bound: u32 },
    /// The checked domain product overflows: the alternative is infeasible.
    Infeasible(LinearMapError),
}

impl std::fmt::Display for HierarchicalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HierarchicalError::InnerDomainExceedsWorkgroup {
                inner_capacity,
                bound,
            } => {
                write!(
                    f,
                    "the inner domain ({inner_capacity} participants) exceeds the workgroup \
                     bound ({bound}); the hierarchical alternative is inapplicable"
                )
            }
            HierarchicalError::Infeasible(error) => write!(f, "infeasible domain: {error}"),
        }
    }
}
impl std::error::Error for HierarchicalError {}

/// One hierarchical iteration mapping: outer domain → workgroups, inner
/// domain → subgroups within one workgroup.
#[derive(Clone, Debug)]
pub struct HierarchicalMap {
    pub map: LinearIterationMap,
    /// The checked capacity bound of the inner domain (participants per
    /// workgroup).
    pub inner_capacity: u64,
    /// The checked capacity bound of the outer domain (workgroups).
    pub outer_capacity: u64,
    /// Backend subgroup width when reported.
    pub subgroup_width: Option<u32>,
    /// Planning expression for the workgroup count of this mapping
    /// (`ceil(total / participants)` over the checked bounds).
    pub workgroups: Sym,
}

impl HierarchicalMap {
    /// The exact coordinate map: the logical (outer…, inner…) coordinates of
    /// one participant `p` inside workgroup `w` (row-major, outermost first).
    /// Together the workgroups partition the domain: every logical
    /// coordinate is visited by exactly one participant of exactly one
    /// workgroup, and the traversal's tail mask skips coordinates beyond the
    /// runtime totals.
    pub fn coordinate_of(&self, workgroup: u64, participant: u64) -> Option<Vec<u64>> {
        let linear = workgroup
            .checked_mul(self.inner_capacity)?
            .checked_add(participant)?;
        self.map.delinearize(linear)
    }

    /// The exact subgroup coordinate map: `(subgroup, lane)` of one
    /// participant inside its workgroup. Requires a reported subgroup width;
    /// the last subgroup of a partial workgroup is smaller (its lanes beyond
    /// the inner domain are tail-masked).
    pub fn subgroup_of(&self, participant: u64) -> Option<(u64, u64)> {
        let width = u64::from(self.subgroup_width?);
        Some((participant / width, participant % width))
    }

    /// Exact bytes of one workgroup staging allocation for a tile of
    /// `tile_elements` elements of `elem_bytes` each. The staging
    /// `ConditionalStorageTemplate` (scope `Workgroup`, replication
    /// per-workgroup, alignment recorded on the template) is created by the
    /// realization surface; this expression is its exact `bytes` input.
    /// Values owned outside and read by inner groups are eligible; inner
    /// private values are not.
    pub fn stage_tile_bytes(&self, tile_elements: &Sym, elem_bytes: i64) -> Sym {
        tile_elements.scale(elem_bytes)
    }

    /// The planning expression bounding pool pressure of this mapping: the
    /// per-workgroup staging bytes times the workgroup count (the solver
    /// constrains workgroup bytes directly against the target bound).
    pub fn pool_pressure(&self, stage_bytes: &Sym) -> Sym {
        self.workgroups.mul(stage_bytes)
    }
}

/// Construct the hierarchical mapping of one nested independent structure.
///
/// `outer`/`inner` are the logical axis extents, outermost first. The
/// physical participant count of the launch is `participants` — a planning
/// expression over a solver-tunable parameter (typically from
/// `AlternativeBuilder::solver_participants(1, inner_capacity)`), bounded by
/// the exact workgroup participant bound of `facts`.
pub fn hierarchical_map(
    outer: &[ExtentExpr],
    inner: &[ExtentExpr],
    runtime_extents: &BTreeMap<RuntimeExtentId, RuntimeExtent>,
    facts: &SubgroupFacts,
    participants: Sym,
) -> Result<HierarchicalMap, HierarchicalError> {
    let inner_map = LinearIterationMap::linear(inner, runtime_extents)
        .map_err(HierarchicalError::Infeasible)?;
    let outer_map = LinearIterationMap::linear(outer, runtime_extents)
        .map_err(HierarchicalError::Infeasible)?;
    let inner_capacity = inner_map.total.bound();
    let outer_capacity = outer_map.total.bound();
    if inner_capacity > u64::from(facts.max_participants_per_workgroup) {
        return Err(HierarchicalError::InnerDomainExceedsWorkgroup {
            inner_capacity,
            bound: facts.max_participants_per_workgroup,
        });
    }
    let mut axes = outer.to_vec();
    axes.extend(inner.iter().cloned());
    let map = LinearIterationMap::linear(&axes, runtime_extents)
        .map_err(HierarchicalError::Infeasible)?
        .with_participants(participants);
    // Grid-stride traversal with tail mask: runtime axes contribute their
    // exact runtime totals at execution and their checked capacities to the
    // planning bound.
    debug_assert_eq!(map.traversal, Traversal::GridStride);
    let total = Sym::constant(i64::try_from(map.total.bound()).unwrap_or(i64::MAX));
    let one = Sym::constant(1);
    let workgroups = total
        .add(&map.participants.clone().sub(&one))
        .quot(&map.participants.clone());
    Ok(HierarchicalMap {
        map,
        inner_capacity,
        outer_capacity,
        subgroup_width: facts.width,
        workgroups,
    })
}

/// Receipt of one hierarchically fused region.
#[derive(Clone, Debug)]
pub struct HierarchicalReceipt {
    /// The mapping (exact coordinate maps, workgroup/subgroup structure).
    pub mapping: HierarchicalMap,
    /// Exact per-workgroup staging bytes, when the region stages a tile.
    pub stage_bytes: Option<Sym>,
    pub cost: Sym,
    pub model: CostModelId,
}

/// Fuse one connected primitive region under the hierarchical mapping: one
/// launch whose participants are the inner domain of `mapping` and whose
/// workgroups are the outer domain. `stage_bytes` records the exact
/// per-workgroup staging allocation the region requires (values owned
/// outside and read by inner groups); it is an exact hard-resource input to
/// the one global model.
pub fn hierarchical_fuse<D: ExecutableDialect>(
    builder: &mut AlternativeBuilder<D>,
    nodes: Vec<NodeRef>,
    mapping: HierarchicalMap,
    stage_bytes: Option<Sym>,
    ops: Legalized<D::Op>,
    model: CostModelId,
) -> Result<HierarchicalReceipt, BuilderError> {
    if nodes.is_empty() {
        return Err("hierarchical_fuse requires a nonempty node set".into());
    }
    let iteration = mapping.map.clone();
    builder.fuse(nodes, FusedStrategyTemplate { iteration, ops })?;
    let cost = mapping
        .workgroups
        .mul(&Sym::constant(
            i64::try_from(mapping.inner_capacity).unwrap_or(1),
        ))
        .scale(1);
    Ok(HierarchicalReceipt {
        mapping,
        stage_bytes,
        cost,
        model,
    })
}
