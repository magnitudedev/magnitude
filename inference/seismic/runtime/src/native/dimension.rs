//! Expressions over a graph trace's facts (`seismic::trace`): the sizes a
//! recorded graph builder computed, kept so a trace can be evaluated at
//! another model's facts. Only Seismic's API builds them, from its symbolic
//! dimensions; the runtime evaluates them where a recorded graph holds them
//! (a view's operations, `WorkflowTensorArgument::ResultView`).

use crate::api::kernel::ViewOperation;
use std::fmt;
use std::sync::Arc;

/// An expression over a recording's facts.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum DimensionExpr {
    Constant(u64),
    /// The recording's fact at this index.
    Fact(usize),
    Apply(Arc<(Operation, DimensionExpr, DimensionExpr)>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    /// Floor division.
    Div,
    CeilDiv,
    Rem,
    Min,
    Max,
}

impl Operation {
    /// The `u64` operation, `None` exactly where it overflows or divides by
    /// zero.
    pub fn apply(self, left: u64, right: u64) -> Option<u64> {
        match self {
            Self::Add => left.checked_add(right),
            Self::Sub => left.checked_sub(right),
            Self::Mul => left.checked_mul(right),
            Self::Div => left.checked_div(right),
            Self::CeilDiv => (right != 0).then(|| left.div_ceil(right)),
            Self::Rem => left.checked_rem(right),
            Self::Min => Some(left.min(right)),
            Self::Max => Some(left.max(right)),
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::CeilDiv => "ceil/",
            Self::Rem => "%",
            Self::Min => "min",
            Self::Max => "max",
        }
    }
}

impl DimensionExpr {
    /// The expression at `facts`, `None` where an operation overflows or
    /// divides by zero.
    ///
    /// # Panics
    ///
    /// Where it reads a fact `facts` does not hold: an expression is
    /// evaluated only at its own recording's facts (a template recorded
    /// outside a trace, at none, holds constants alone).
    pub fn evaluate(&self, facts: &[u64]) -> Option<u64> {
        match self {
            Self::Constant(value) => Some(*value),
            Self::Fact(index) => Some(
                *facts
                    .get(*index)
                    .expect("a dimension is evaluated only at its own recording's facts"),
            ),
            Self::Apply(application) => {
                let (operation, left, right) = &**application;
                operation.apply(left.evaluate(facts)?, right.evaluate(facts)?)
            }
        }
    }
}

impl fmt::Debug for DimensionExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant(value) => write!(f, "{value}"),
            Self::Fact(index) => write!(f, "#{index}"),
            Self::Apply(application) => {
                let (operation, left, right) = &**application;
                write!(f, "({left:?} {} {right:?})", operation.symbol())
            }
        }
    }
}

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
