//! Blocked lane coverage.
//!
//! Reductions and normalization offer both an *interleaved* and a
//! *contiguous* blocked cover of the reduced axis, as two distinct
//! alternatives of the same choice. A cover is the exact assignment of
//! reduced-axis coordinates to lanes:
//!
//! - [`LaneCover::Interleaved`]: lane `ℓ` folds coordinates `ℓ, ℓ+P, ℓ+2P, …`
//!   (the grid-stride delinearization of the universal linear map).
//! - [`LaneCover::Contiguous`]: lane `ℓ` folds the contiguous block
//!   `[ℓ·B, (ℓ+1)·B)`, the last block masked by the runtime total.
//!
//! Both covers reassociate the reference ascending fold, so admission goes
//! through the common reduction vocabulary (`terminal::reduction::reassociable`):
//! the source must mark the reduction `unordered=true`, or the caller
//! policy/evidence must permit reassociation; an ordered reduction with no
//! admission is refused, and `argmax` never reassociates (smaller-coordinate
//! ties must survive). The lane-local partial reduction is regenerated for
//! the selected cover — the emitter never changes cover independently: each
//! alternative carries its own exact coordinate map
//! ([`LaneCover::coordinate`]) and its own topology.

use crate::strategies::cost::CostModelId;
use crate::terminal::reduction::{reassociable, ReassociationAdmission, ReductionStrategy};
use seismic_lang::{
    logical::{ReductionNode, RuntimeExtent},
    sym::Sym,
    types::{ExtentExpr, RuntimeExtentId, TensorType},
};
use seismic_realization::dispatch::LinearIterationMap;
use seismic_realization::executable::{
    AlternativeBuilder, BuilderError, ExecutableDialect, Legalized, NodeRef,
    ReductionStrategyTemplate, TransportTemplate,
};
use seismic_realization::numerics::ReductionTopology;
use std::collections::BTreeMap;

/// One exact blocked cover of the reduced axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaneCover {
    /// Lane `ℓ` folds coordinates `ℓ, ℓ+P, ℓ+2P, …` (`lanes` = P).
    Interleaved { lanes: u64 },
    /// Lane `ℓ` folds the contiguous block `[ℓ·B, (ℓ+1)·B)` (`block` = B);
    /// the last block is masked by the runtime total.
    Contiguous { lanes: u64, block: u64 },
}

impl LaneCover {
    /// The exact coordinate map: the reduced-axis coordinate lane `lane`
    /// folds on its `step`-th visit. `None` when the coordinate lies beyond
    /// the cover (a masked tail element).
    pub fn coordinate(&self, lane: u64, step: u64) -> Option<u64> {
        match self {
            LaneCover::Interleaved { lanes } => {
                let lanes = *lanes;
                if lane >= lanes {
                    return None;
                }
                Some(lane + step * lanes)
            }
            LaneCover::Contiguous { lanes, block } => {
                if lane >= *lanes {
                    return None;
                }
                Some(lane * block + step)
            }
        }
    }

    /// The number of elements lane `lane` folds to cover `total` elements.
    pub fn lane_length(&self, total: u64, lane: u64) -> u64 {
        match self {
            LaneCover::Interleaved { lanes } => {
                // Elements ℓ, ℓ+P, … < total: ceil((total − ℓ)/P) for ℓ < total.
                if lane >= *lanes || lane >= total {
                    0
                } else {
                    (total - lane).div_ceil(*lanes)
                }
            }
            LaneCover::Contiguous { lanes, block } => {
                if lane >= *lanes {
                    0
                } else {
                    let start = lane * block;
                    total.saturating_sub(start).min(*block)
                }
            }
        }
    }

    /// Whether the cover partitions `0..total` exactly: every coordinate is
    /// folded by exactly one lane exactly once, and no lane folds beyond
    /// `total`. The contiguous cover's last block is masked by the runtime
    /// total (lane lengths stop at `total`), so any `lanes·B ≥ total`
    /// partitions; `lanes·B < total` does not cover.
    pub fn covers_exactly(&self, total: u64) -> bool {
        let lanes = match self {
            LaneCover::Interleaved { lanes } => *lanes,
            LaneCover::Contiguous { lanes, block } => {
                if lanes
                    .checked_mul(*block)
                    .map(|product| product < total)
                    .unwrap_or(true)
                {
                    return false;
                }
                *lanes
            }
        };
        if lanes == 0 {
            return false;
        }
        let mut seen = std::collections::BTreeSet::new();
        for lane in 0..lanes {
            let length = self.lane_length(total, lane);
            for step in 0..length {
                match self.coordinate(lane, step) {
                    Some(coordinate) if coordinate < total => {
                        if !seen.insert(coordinate) {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        seen.len() == total as usize
    }
}

/// The cover plan of one blocked alternative: the cover shape plus the
/// solver-tunable symbols (lane count for both covers; block width for the
/// contiguous cover).
#[derive(Clone, Debug)]
pub struct BlockedPlan {
    pub cover: LaneCover,
    /// Planning symbol of the lane count (a solver-tunable parameter over
    /// `1..=lanes`).
    pub lanes_symbol: Sym,
    /// Planning symbol of the contiguous block width, when contiguous.
    pub block_symbol: Option<Sym>,
}

/// Receipt of one blocked-cover reduction alternative.
#[derive(Clone, Debug)]
pub struct BlockedReceipt {
    pub plan: BlockedPlan,
    /// The admitted reduction strategy (topology, admission, transfer).
    pub strategy: ReductionStrategy,
    /// The exact iteration map of the alternative (the interleaved cover is
    /// the map's own grid-stride delinearization; the contiguous cover
    /// regenerates the lane-local fold for its blocks).
    pub iteration: LinearIterationMap,
    pub cost: Sym,
    pub model: CostModelId,
}

/// Construct one blocked-cover reduction alternative. Declares the lane
/// count (and, for the contiguous cover, the block width) as solver-tunable
/// plan parameters, admits the reassociating topology through the common
/// reduction vocabulary, and consumes the reduction node through
/// `map_reduction`.
///
/// Refused (ordinary inapplicability — the universal reduction strategy
/// remains) when: the reduction is `argmax` (never reassociates); the source
/// ordered the reduction and no policy/evidence admission was supplied; or
/// the domain is infeasible.
pub fn blocked_cover<D: ExecutableDialect>(
    builder: &mut AlternativeBuilder<D>,
    occurrence: NodeRef,
    reduction: &ReductionNode,
    operand: &TensorType,
    cover: LaneCover,
    admission: ReassociationAdmission,
    runtime_extents: &BTreeMap<RuntimeExtentId, RuntimeExtent>,
    ops: Legalized<D::Op>,
    model: CostModelId,
) -> Result<BlockedReceipt, BuilderError> {
    let axis = reduction.axis;
    if axis >= operand.axes.len() {
        return Err("the reduced axis is outside the operand rank".into());
    }
    let length = &operand.axes[axis];
    let length_bound = extent_bound(length, runtime_extents)?;
    // Solver-tunable lane count over 1..=lanes.
    let lanes_upper = match &cover {
        LaneCover::Interleaved { lanes } | LaneCover::Contiguous { lanes, .. } => *lanes,
    };
    let lanes_symbol = builder.solver_participants(1, lanes_upper as i64)?;
    // The block width parameter of the contiguous cover.
    let block_symbol = match &cover {
        LaneCover::Contiguous { block, .. } => {
            let name = format!("block-node{}", occurrence.node.0);
            let upper = i64::try_from(*block)
                .map_err(|_| "the block width exceeds the plan parameter domain".to_string())?;
            builder.plan_parameter(&name, 1, upper)?;
            Some(Sym::param(&name))
        }
        LaneCover::Interleaved { .. } => None,
    };
    // The reassociating topology of this cover: lanes fold partials
    // (interleaved strides or contiguous blocks) and combine in one tree
    // level; the split form records the contiguous cuts.
    let inner = Box::new(ReductionTopology::SerialAxis {
        axis,
        length: length.clone(),
    });
    let topology = match &cover {
        LaneCover::Interleaved { lanes } => ReductionTopology::Tree {
            fan_in: u32::try_from(*lanes).map_err(|_| "the lane count exceeds u32".to_string())?,
            depth: 1,
            inner,
        },
        LaneCover::Contiguous { lanes, .. } => {
            let count =
                usize::try_from(*lanes).map_err(|_| "the lane count exceeds usize".to_string())?;
            let mut cuts = Vec::with_capacity(count);
            for _ in 0..count.saturating_sub(1) {
                cuts.push(ExtentExpr::Sym(
                    block_symbol
                        .clone()
                        .expect("the contiguous cover has a block symbol"),
                ));
            }
            cuts.push(length.clone());
            ReductionTopology::Split { cuts, inner }
        }
    };
    // Admission through the common vocabulary: argmax never reassociates;
    // an ordered source reduction needs policy/evidence.
    let strategy =
        reassociable(reduction, operand, topology, admission).map_err(|error| match error {
            crate::terminal::reduction::ReductionAdmissionError::ArgmaxNeverReassociates => {
                "argmax never reassociates: smaller-coordinate ties must be preserved".to_string()
            }
            crate::terminal::reduction::ReductionAdmissionError::AscendingOrder => {
                "the source ordered the reduction; reassociation needs `unordered=true`, \
                 caller policy, or evidence"
                    .to_string()
            }
        })?;
    // The exact iteration map: the interleaved cover is the map's own
    // grid-stride traversal over the operand axes; the contiguous cover
    // traverses the same domain with its lanes regenerating block folds.
    let iteration = LinearIterationMap::linear(&operand.axes, runtime_extents)
        .map_err(|error| format!("the blocked domain is infeasible: {error}"))?
        .with_participants(lanes_symbol.clone());
    // The reduced result publishes through its recorded transport.
    let logical = crate::strategies::node_at(builder.graph(), &occurrence)
        .ok_or_else(|| "the region path names no node".to_string())?;
    let result_value = logical
        .outputs
        .first()
        .map(|output| output.id)
        .ok_or_else(|| "a reduction has a result value".to_string())?;
    let result = builder.transport_of(result_value)?;
    if matches!(result, TransportTemplate::Kernel(_)) {
        return Err("a reduction result publishes through retained storage".into());
    }
    builder.map_reduction(
        occurrence,
        ReductionStrategyTemplate {
            topology: strategy.topology.clone(),
            iteration: iteration.clone(),
            ops,
            result,
        },
    )?;
    let cost = Sym::constant(i64::try_from(length_bound).unwrap_or(i64::MAX))
        .add(&lanes_symbol.clone().scale(1));
    Ok(BlockedReceipt {
        plan: BlockedPlan {
            cover,
            lanes_symbol,
            block_symbol,
        },
        strategy,
        iteration,
        cost,
        model,
    })
}

fn extent_bound(
    extent: &ExtentExpr,
    runtime_extents: &BTreeMap<RuntimeExtentId, RuntimeExtent>,
) -> Result<u64, BuilderError> {
    match extent {
        ExtentExpr::Static(n) => Ok(*n),
        ExtentExpr::Runtime(id) => runtime_extents
            .get(id)
            .map(|runtime| runtime.capacity)
            .ok_or_else(|| "an unresolved runtime extent".to_string()),
        ExtentExpr::Sym(sym) => sym
            .as_constant()
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| "an unresolved symbolic extent".to_string()),
    }
}
