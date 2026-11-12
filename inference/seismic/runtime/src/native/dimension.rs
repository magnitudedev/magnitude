//! A recorded graph's views over a trace's facts (`seismic::trace`): the
//! view operations a recorded graph builder made, their numbers kept as
//! expressions (`seismic_lang::size::DimensionExpr`) so a trace can be
//! evaluated at another model's facts. Only Seismic's API builds them, from
//! its symbolic dimensions; the runtime evaluates them where a recorded
//! graph holds them (a view's operations, `WorkflowTensorArgument::ResultView`).

use crate::api::kernel::ViewOperation;
use seismic_lang::size::DimensionExpr;

/// A view operation with its numbers as expressions: what a recorded view
/// is at another model's facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViewExpr {
    LeadingSlice { start: DimensionExpr, end: DimensionExpr },
    Reshape { extents: Vec<DimensionExpr> },
}

impl ViewExpr {
    /// `operation`, whose numbers no fact reaches.
    pub fn constant(operation: &ViewOperation) -> Self {
        match operation {
            ViewOperation::LeadingSlice { start, end } => Self::LeadingSlice {
                start: DimensionExpr::Constant(*start),
                end: DimensionExpr::Constant(*end),
            },
            ViewOperation::Reshape { extents } => Self::Reshape {
                extents: extents.iter().copied().map(DimensionExpr::Constant).collect(),
            },
        }
    }

    /// Whether no fact reaches its numbers.
    pub fn is_constant(&self) -> bool {
        let constant = |expression: &DimensionExpr| matches!(expression, DimensionExpr::Constant(_));
        match self {
            Self::LeadingSlice { start, end } => constant(start) && constant(end),
            Self::Reshape { extents } => extents.iter().all(constant),
        }
    }

    /// The operation at `facts`, `None` where a number does not evaluate.
    pub fn at(&self, facts: &[u64]) -> Option<ViewOperation> {
        Some(match self {
            Self::LeadingSlice { start, end } => ViewOperation::LeadingSlice {
                start: start.evaluate(facts)?,
                end: end.evaluate(facts)?,
            },
            Self::Reshape { extents } => ViewOperation::Reshape {
                extents: extents.iter().map(|extent| extent.evaluate(facts)).collect::<Option<_>>()?,
            },
        })
    }
}
