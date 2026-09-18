//! Numerical contracts shared by interpretation, realization and accounting.
use crate::{ir::ReduceOp, types::DType};

pub mod structured;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Identity {
    Zero,
    One,
    NegativeInfinity,
    PositiveInfinity,
    MinI32,
    MaxI32,
    MaxU32,
}
impl Identity {
    pub fn value(self) -> f64 {
        match self {
            Self::Zero => 0.0,
            Self::One => 1.0,
            Self::NegativeInfinity => f64::NEG_INFINITY,
            Self::PositiveInfinity => f64::INFINITY,
            Self::MinI32 => f64::from(i32::MIN),
            Self::MaxI32 => f64::from(i32::MAX),
            Self::MaxU32 => f64::from(u32::MAX),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Combination {
    /// Round each addition to the input dtype in the reference fold.
    FloatingAdd,
    /// Clamp each addition to the input integer range; never wrap.
    SaturatingAdd,
    LogicalOr,
    LogicalAnd,
    Maximum,
    Minimum,
    /// Strict improvement; first index on ties, zero when no value improves.
    FirstMaximum,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Contract {
    pub operation: ReduceOp,
    pub input: DType,
    /// Source-level permission, distinct from an implementation's precision limits.
    pub ordered: bool,
}
impl Contract {
    pub fn new(operation: ReduceOp, input: DType, ordered: bool) -> Self {
        Self {
            operation,
            input,
            ordered,
        }
    }
    pub fn output(self) -> DType {
        if self.operation == ReduceOp::Argmax {
            DType::I32
        } else {
            self.input
        }
    }
    pub fn allows_empty_axis(self) -> bool {
        self.operation != ReduceOp::Argmax
    }
    pub fn combination(self) -> Combination {
        match (self.operation, self.input) {
            (ReduceOp::Argmax, _) => Combination::FirstMaximum,
            (ReduceOp::Sum | ReduceOp::Max, DType::Bool) => Combination::LogicalOr,
            (ReduceOp::Min, DType::Bool) => Combination::LogicalAnd,
            (ReduceOp::Sum, DType::I32 | DType::U32) => Combination::SaturatingAdd,
            (ReduceOp::Sum, _) => Combination::FloatingAdd,
            (ReduceOp::Max, _) => Combination::Maximum,
            (ReduceOp::Min, _) => Combination::Minimum,
        }
    }
    pub fn identity(self) -> Identity {
        match (self.operation, self.input) {
            (ReduceOp::Sum, _) => Identity::Zero,
            (ReduceOp::Min, DType::Bool) => Identity::One,
            (_, DType::Bool) => Identity::Zero,
            (ReduceOp::Min, DType::I32) => Identity::MaxI32,
            (_, DType::I32) => Identity::MinI32,
            (ReduceOp::Min, DType::U32) => Identity::MaxU32,
            (_, DType::U32) => Identity::Zero,
            (ReduceOp::Min, _) => Identity::PositiveInfinity,
            _ => Identity::NegativeInfinity,
        }
    }
}
