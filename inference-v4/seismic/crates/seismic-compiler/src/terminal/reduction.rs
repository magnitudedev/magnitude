//! Reduction strategy vocabulary and universal strategies.
//!
//! Every reduction has an exact strategy. The universal strategy maps parallel
//! outer coordinates, keeps one logical participant per output, folds the
//! reduced axis serially in ascending coordinate order with the registry
//! accumulator/identity/tie semantics, and publishes exactly once. Optimized
//! topologies (tree/subgroup/workgroup/matrix/split/multi-launch) declare
//! exact topology, resources, and numerics, and are admitted for reassociable
//! forms only when the source marked the reduction `unordered` or the caller
//! policy/evidence permits reassociation. `argmax` never reassociates and
//! always preserves smaller-index ties.
//!
//! Backends and the optimized strategy library
//! construct their own declarations from these exact shapes; this
//! module defines the common interfaces and the universal portable forms.

use super::numerics::NumericalTransfer;
use seismic_lang::{
    intrinsics::{accumulator_dtype, ReduceOp},
    logical::{ReductionNode, ReductionOrder},
    types::{DType, Elem, ExtentExpr, TensorType},
};
pub use seismic_realization::numerics::ReductionTopology;

/// The strategy vocabulary. Serial and parallel-outer are the universal forms;
/// the rest are optimized alternatives within the same plan family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReductionStrategyKind {
    /// One participant folds everything, ascending.
    Serial,
    /// Parallel outer coordinates; one logical participant per output folds
    /// the reduced axis serially, ascending.
    ParallelOuter,
    Tree,
    Subgroup,
    Workgroup,
    Matrix,
    Split,
    MultiLaunch,
}

/// How ties are resolved. The registry admits exactly one rule; strategies
/// may not weaken it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TieRule {
    /// The smaller coordinate index wins (`argmax` always).
    SmallerCoordinateIndex,
}

/// The identity element the fold starts from (registry semantics).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReductionIdentity {
    /// `sum`: the additive identity of the accumulator dtype.
    Zero,
    /// `max`/`min`: no identity; the fold starts from the first (ascending)
    /// element, so the reduced axis must be nonempty.
    FirstElement,
    /// `argmax`: no identity, smaller-index ties, nonempty input required.
    FirstElementNonEmpty,
}

/// The registry identity of one reduction operator.
pub fn reduction_identity(op: ReduceOp) -> ReductionIdentity {
    match op {
        ReduceOp::Sum => ReductionIdentity::Zero,
        ReduceOp::Max | ReduceOp::Min => ReductionIdentity::FirstElement,
        ReduceOp::Argmax => ReductionIdentity::FirstElementNonEmpty,
    }
}

/// A semantic precondition a strategy carries besides safety obligations.
#[derive(Clone, Debug, PartialEq)]
pub enum ReductionPrecondition {
    /// The reduced axis must contain at least one element (`argmax`).
    NonEmpty { axis: usize, length: ExtentExpr },
}

/// Whether reassociation (a visiting order different from the reference
/// ascending fold) is admitted at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReassociationAdmission {
    /// The strategy preserves the reference visiting order exactly.
    Ordered,
    /// The source marked the reduction `unordered=true`.
    SourceUnordered,
    /// The caller policy or accepted evidence permits reassociation.
    PolicyOrEvidence,
}

/// Exact hard-resource shape of one reduction alternative. Universal values
/// are all-zero/one; optimized strategies fill exact facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReductionResources {
    /// Logical participants producing one output together.
    pub participants_per_output: u32,
    /// Exact selected workgroup staging bytes.
    pub workgroup_bytes: u64,
    /// Exact explicit private bytes per participant.
    pub private_bytes_per_participant: u64,
    /// Launches this strategy contributes.
    pub launches: u32,
    /// Publications of the final result: exactly one.
    pub publications: u32,
}

impl ReductionResources {
    /// The universal shape: one participant per output, no extra storage, one
    /// launch, one publication.
    pub const UNIVERSAL: ReductionResources = ReductionResources {
        participants_per_output: 1,
        workgroup_bytes: 0,
        private_bytes_per_participant: 0,
        launches: 1,
        publications: 1,
    };
}

/// One reduction strategy declaration: topology, resources, numerics, and the
/// registry semantics it must preserve.
#[derive(Clone, Debug, PartialEq)]
pub struct ReductionStrategy {
    pub kind: ReductionStrategyKind,
    pub topology: ReductionTopology,
    pub reassociation: ReassociationAdmission,
    pub numerical: NumericalTransfer,
    pub tie_rule: TieRule,
    pub identity: ReductionIdentity,
    pub accumulator: DType,
    pub preconditions: Vec<ReductionPrecondition>,
    pub resources: ReductionResources,
}

/// Why a reassociable strategy construction was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReductionAdmissionError {
    /// The source ordered the reduction (`unordered` not set) and no caller
    /// policy/evidence admission was supplied.
    AscendingOrder,
    /// `argmax` never reassociates: ties must keep the smaller coordinate.
    ArgmaxNeverReassociates,
}

/// The input dtype of one reduction operand type.
fn input_dtype(operand: &TensorType) -> Result<DType, String> {
    match &operand.elem {
        Elem::Dtype(d) => Ok(*d),
        Elem::Param(_) => Ok(DType::F32),
        Elem::Repr(repr) => Err(format!(
            "a packed representation `{repr}` cannot be reduced"
        )),
    }
}

/// The universal reduction strategy: parallel outer
/// coordinates, one logical participant per output, serial ascending reduced
/// axis, registry accumulator/identity/tie semantics, one publication. Its
/// numerical transfer is `Exact`.
pub fn universal(
    reduction: &ReductionNode,
    operand: &TensorType,
) -> Result<ReductionStrategy, String> {
    let input = input_dtype(operand)?;
    let mut outer_axes: Vec<ExtentExpr> = operand.axes.clone();
    let length = outer_axes.remove(reduction.axis);
    let inner = ReductionTopology::SerialAxis {
        axis: reduction.axis,
        length,
    };
    let topology = if outer_axes.is_empty() {
        // A full reduction to a scalar: one output, one participant.
        inner
    } else {
        ReductionTopology::ParallelOuter {
            outer_axes,
            inner: Box::new(inner),
        }
    };
    let mut preconditions = Vec::new();
    if reduction.op == ReduceOp::Argmax {
        preconditions.push(ReductionPrecondition::NonEmpty {
            axis: reduction.axis,
            length: operand.axes[reduction.axis].clone(),
        });
    }
    Ok(ReductionStrategy {
        kind: ReductionStrategyKind::ParallelOuter,
        topology,
        reassociation: ReassociationAdmission::Ordered,
        numerical: NumericalTransfer::Exact,
        tie_rule: TieRule::SmallerCoordinateIndex,
        identity: reduction_identity(reduction.op),
        // The logical layer already applied the registry rule; confirm it.
        accumulator: if reduction.accumulator == accumulator_dtype(reduction.op, input) {
            reduction.accumulator
        } else {
            return Err(format!(
                "reduction accumulator {} disagrees with the registry rule for `{}` of {}",
                reduction.accumulator.name(),
                reduction.op.name(),
                input.name()
            ));
        },
        preconditions,
        resources: ReductionResources::UNIVERSAL,
    })
}

/// Construct a reassociating strategy (tree/subgroup/workgroup/matrix/split/
/// multi-launch). Reassociation must be admitted by the source
/// (`unordered=true`) or by caller policy/evidence; `argmax` is always
/// refused, and the declared topology must not itself be the serial fold.
pub fn reassociable(
    reduction: &ReductionNode,
    operand: &TensorType,
    topology: ReductionTopology,
    admission: ReassociationAdmission,
) -> Result<ReductionStrategy, ReductionAdmissionError> {
    if reduction.op == ReduceOp::Argmax {
        return Err(ReductionAdmissionError::ArgmaxNeverReassociates);
    }
    if reduction.order == ReductionOrder::Ascending
        && admission == ReassociationAdmission::SourceUnordered
    {
        return Err(ReductionAdmissionError::AscendingOrder);
    }
    let kind = match &topology {
        ReductionTopology::Tree { .. } => ReductionStrategyKind::Tree,
        ReductionTopology::Subgroup { .. } => ReductionStrategyKind::Subgroup,
        ReductionTopology::Workgroup { .. } => ReductionStrategyKind::Workgroup,
        ReductionTopology::Matrix { .. } => ReductionStrategyKind::Matrix,
        ReductionTopology::Split { .. } => ReductionStrategyKind::Split,
        ReductionTopology::MultiLaunch { .. } => ReductionStrategyKind::MultiLaunch,
        // A pure serial axis or parallel-outer wrapper is the ordered
        // universal form, not a reassociating strategy.
        ReductionTopology::SerialAxis { .. } | ReductionTopology::ParallelOuter { .. } => {
            return Err(ReductionAdmissionError::AscendingOrder);
        }
    };
    let mut preconditions = Vec::new();
    if matches!(
        reduction_identity(reduction.op),
        ReductionIdentity::FirstElement | ReductionIdentity::FirstElementNonEmpty
    ) {
        // max/min over an empty axis has no defined result in any order.
        preconditions.push(ReductionPrecondition::NonEmpty {
            axis: reduction.axis,
            length: operand.axes[reduction.axis].clone(),
        });
    }
    let numerical = NumericalTransfer::Reassociate {
        op: reduction.op,
        topology: topology.clone(),
    };
    Ok(ReductionStrategy {
        kind,
        topology,
        reassociation: admission,
        numerical,
        tie_rule: TieRule::SmallerCoordinateIndex,
        identity: reduction_identity(reduction.op),
        accumulator: reduction.accumulator,
        preconditions,
        // Optimized strategies declare exact facts in their own alternatives.
        resources: ReductionResources {
            participants_per_output: 0,
            workgroup_bytes: 0,
            private_bytes_per_participant: 0,
            launches: 0,
            publications: 1,
        },
    })
}
