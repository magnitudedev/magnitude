//! Size expressions: a graph recording (`seismic::trace`) keeps its sizes as
//! expressions over the recording's facts ([`DimensionExpr`]), which a later
//! model's facts evaluate with the same checked `u64` operations a load uses.

use std::fmt;
use std::sync::Arc;

/// A binary operation of a size expression.
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
    /// The operation on `u64`, `None` exactly where it overflows or divides
    /// by zero.
    #[inline]
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

/// An expression over a recording's facts, as a symbolic dimension carries
/// it: a tree whose shared subtrees share their `Arc`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum DimensionExpr {
    Constant(u64),
    /// The recording's fact at this index.
    Fact(usize),
    Apply(Arc<(Operation, DimensionExpr, DimensionExpr)>),
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
            Self::Fact(index) => Some(fact(facts, *index)),
            Self::Apply(application) => {
                let (operation, left, right) = &**application;
                operation.apply(left.evaluate(facts)?, right.evaluate(facts)?)
            }
        }
    }
}

fn fact(facts: &[u64], index: usize) -> u64 {
    *facts
        .get(index)
        .expect("a dimension is evaluated only at its own recording's facts")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operations_answer_none_where_the_u64_operation_does() {
        assert_eq!(Operation::Add.apply(u64::MAX, 1), None);
        assert_eq!(Operation::Sub.apply(1, 2), None);
        assert_eq!(Operation::Mul.apply(u64::MAX, 2), None);
        assert_eq!(Operation::Div.apply(1, 0), None);
        assert_eq!(Operation::CeilDiv.apply(1, 0), None);
        assert_eq!(Operation::CeilDiv.apply(7, 2), Some(4));
        assert_eq!(Operation::Rem.apply(1, 0), None);
        assert_eq!(Operation::Min.apply(3, 5), Some(3));
        assert_eq!(Operation::Max.apply(3, 5), Some(5));
    }
}
