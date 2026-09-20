//! The optimized strategy library.
//!
//! The strategies in this module are ordinary selectable members of the one
//! `PlanFamily`: reusable constructors over `AlternativeBuilder` that a
//! family assembly (pipeline, backends) calls per occurrence. They
//! are not separate compilers and not new source constructs. Every
//! constructor:
//!
//! - builds exclusively through the eleven public alternative-builder
//!   transitions, so `finish_alternative` accepts the result only when all
//!   logical obligations were consumed;
//! - takes every backend fact (subgroup width, workgroup bounds, legalized
//!   opcodes, measured cost identity) as an *input* — dialect legalization
//!   and cost registration belong to the backend;
//! - carries an exact `CostExpr` plus a `CostModelId` (cost affects ranking
//!   only, never legality);
//! - declares its numerical transfer explicitly (`Exact` for order-preserving
//!   forms, `Reassociate`/`Round` where reordering or rescaling occurs).
//!
//! Strategy families:
//!
//! 1. [`streaming`] — runtime-axis streaming with a solver-tunable window
//!    `PlanParameter` and explicit physical carries, plus the *structural*
//!    online-softmax/scan admission rule (no name recognition, no silent
//!    replacement of arbitrary user arithmetic).
//! 2. [`hierarchical`] — workgroup/subgroup mapping of nested independent
//!    domains with exact coordinate maps and modeled staging pressure.
//! 3. [`fusion`] — shared-operand multi-consumer fusion (one staged
//!    activation only inside a fused alternative that publishes every result
//!    explicitly) and cross-call boundary substitution.
//! 4. [`blocked`] — interleaved and contiguous blocked lane covers for
//!    reductions/normalization, both as distinct alternatives with exact
//!    coordinate maps.
//! 5. [`cost`] — measured-model identity/provenance plumbing.

pub mod blocked;
pub mod cost;
pub mod fusion;
pub mod hierarchical;
pub mod streaming;

#[cfg(test)]
mod tests;

pub use blocked::{blocked_cover, BlockedPlan, BlockedReceipt, LaneCover};
pub use cost::CostModelId;
pub use fusion::{
    complete_fused_results, cross_call_environment, shared_operand_fuse, FusionReceipt,
};
pub use hierarchical::{
    hierarchical_fuse, hierarchical_map, HierarchicalError, HierarchicalMap, SubgroupFacts,
};
pub use streaming::{
    derive_scan_state, stream_dataflow, stream_scan, CarriedLane, CombineLaw, ScanState,
    ScanStateMatch, StreamingReceipt, StreamingWindow,
};

use seismic_lang::logical::{LogicalNode, LogicalNodeKind, NodeId, TaskGraph};

/// Backend-provided physical facts the strategy library consumes as inputs
/// (registered with the backend dialect surface by streams E/F/G; this
/// library never decides them). Missing facts make the corresponding
/// optimized strategy inapplicable — the universal strategy remains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendFacts {
    /// Hardware subgroup width, when the backend reports one exact width.
    pub subgroup_width: Option<u32>,
    /// Maximum participants per workgroup (exact profile bound).
    pub max_workgroup_participants: u32,
}

impl BackendFacts {
    /// Facts without subgroup support: hierarchical strategies degenerate to
    /// whole-workgroup mapping.
    pub fn without_subgroups(max_workgroup_participants: u32) -> Self {
        BackendFacts {
            subgroup_width: None,
            max_workgroup_participants,
        }
    }
}

/// The logical node at one region-qualified reference, or `None` when the
/// path disagrees with the graph structure. Shared by the strategy
/// constructors for occurrence inspection (the builder consumes exact ids).
pub fn node_at(
    graph: &TaskGraph,
    node: &seismic_realization::executable::NodeRef,
) -> Option<LogicalNode> {
    let mut region = graph.root.clone();
    for step in &node.region {
        let next = region.nodes.get(step.node())?;
        match (&next.kind, step) {
            (
                LogicalNodeKind::If(if_node),
                seismic_realization::executable::RegionStep::IfThen(_),
            ) => {
                region = if_node.then_region.clone();
            }
            (
                LogicalNodeKind::If(if_node),
                seismic_realization::executable::RegionStep::IfElse(_),
            ) => {
                region = if_node.else_region.clone();
            }
            (
                LogicalNodeKind::Loop(loop_node),
                seismic_realization::executable::RegionStep::LoopBody(_),
            ) => {
                region = loop_node.body.clone();
            }
            _ => return None,
        }
    }
    region.nodes.get(node.node).cloned()
}

/// The node ids (in id order) of one region reached by a region path.
pub fn region_nodes(
    graph: &TaskGraph,
    region: &[seismic_realization::executable::RegionStep],
) -> Result<Vec<NodeId>, String> {
    let mut current = graph.root.clone();
    for step in region {
        let node = current
            .nodes
            .get(step.node())
            .ok_or_else(|| format!("region path names absent node#{}", step.node().0))?;
        match (&node.kind, step) {
            (
                LogicalNodeKind::If(if_node),
                seismic_realization::executable::RegionStep::IfThen(_),
            ) => {
                current = if_node.then_region.clone();
            }
            (
                LogicalNodeKind::If(if_node),
                seismic_realization::executable::RegionStep::IfElse(_),
            ) => {
                current = if_node.else_region.clone();
            }
            (
                LogicalNodeKind::Loop(loop_node),
                seismic_realization::executable::RegionStep::LoopBody(_),
            ) => {
                current = loop_node.body.clone();
            }
            _ => return Err("region path disagrees with graph structure".into()),
        }
    }
    Ok(current.nodes.ids().collect())
}
