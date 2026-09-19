//! Typed backend operations shared by checking, effects, realization and emission.
//! Every admitted operation defines its signature and semantic execution contract
//! exhaustively. These semantics are not native latency or instruction counts.

use crate::exec::types::{Shaped, Ty};
use crate::sym::Sym;
use crate::types::{DType, Elem};

/// The admitted intrinsic vocabulary. Signatures, effects and execution semantics
/// are exhaustive functions of this identity, rather than independent name tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    LaneIndex,
    ShuffleIndex,
    SimdSum,
    SimdMax,
    SimdMin,
    Matrix,
    MatrixLoad,
    MatrixLoadTranspose,
    MatrixStore,
    MatrixMultiplyAccumulate,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Semantics {
    ParticipantIndex,
    Exchange,
    Reduction(crate::exec::ir::ReduceOp),
    Fragment {
        rows: u64,
        columns: u64,
    },
    Load {
        rows: u64,
        columns: u64,
        transpose: bool,
    },
    Store {
        rows: u64,
        columns: u64,
    },
    MultiplyAccumulate {
        rows: u64,
        columns: u64,
        inner: u64,
    },
}
impl Operation {
    pub fn name(self) -> &'static str {
        match self {
            Self::LaneIndex => "lane_index",
            Self::ShuffleIndex => "shuffle_index",
            Self::SimdSum => "simd_sum",
            Self::SimdMax => "simd_max",
            Self::SimdMin => "simd_min",
            Self::Matrix => "simdgroup_matrix",
            Self::MatrixLoad => "simdgroup_load",
            Self::MatrixLoadTranspose => "simdgroup_load_t",
            Self::MatrixStore => "simdgroup_store",
            Self::MatrixMultiplyAccumulate => "simdgroup_multiply_accumulate",
        }
    }
    pub fn semantics(self) -> Semantics {
        match self {
            Self::LaneIndex => Semantics::ParticipantIndex,
            Self::ShuffleIndex => Semantics::Exchange,
            Self::SimdSum => Semantics::Reduction(crate::exec::ir::ReduceOp::Sum),
            Self::SimdMax => Semantics::Reduction(crate::exec::ir::ReduceOp::Max),
            Self::SimdMin => Semantics::Reduction(crate::exec::ir::ReduceOp::Min),
            Self::Matrix => Semantics::Fragment {
                rows: 8,
                columns: 8,
            },
            Self::MatrixLoad => Semantics::Load {
                rows: 8,
                columns: 8,
                transpose: false,
            },
            Self::MatrixLoadTranspose => Semantics::Load {
                rows: 8,
                columns: 8,
                transpose: true,
            },
            Self::MatrixStore => Semantics::Store {
                rows: 8,
                columns: 8,
            },
            Self::MatrixMultiplyAccumulate => Semantics::MultiplyAccumulate {
                rows: 8,
                columns: 8,
                inner: 8,
            },
        }
    }
    /// Declaration creates fragment storage; all executable intrinsics involve
    /// the subgroup. This describes participation, not a barrier insertion policy.
    pub fn collective(self) -> bool {
        !matches!(self, Self::Matrix | Self::LaneIndex)
    }
    pub fn writes_arguments(self) -> &'static [usize] {
        match self {
            Self::MatrixLoad | Self::MatrixLoadTranspose | Self::MatrixMultiplyAccumulate => &[0],
            Self::MatrixStore => &[1],
            Self::LaneIndex | Self::ShuffleIndex | Self::SimdSum | Self::SimdMax | Self::SimdMin | Self::Matrix => &[],
        }
    }
    pub fn writes_tensor_memory(self) -> bool {
        // Tile and fragment operands are value storage, not tensor backing.
        match self {
            Self::LaneIndex
            | Self::ShuffleIndex
            | Self::SimdSum
            | Self::SimdMax
            | Self::SimdMin
            | Self::Matrix
            | Self::MatrixLoad
            | Self::MatrixLoadTranspose
            | Self::MatrixStore
            | Self::MatrixMultiplyAccumulate => false,
        }
    }
    pub fn signature(self) -> Intrinsic {
        use IntrinsicParam::*;
        let (params, result) = match self {
            Self::LaneIndex => (vec![], IntrinsicResult::Integer),
            Self::ShuffleIndex => (vec![FloatScalar, Integer], IntrinsicResult::FloatScalar),
            Self::SimdSum | Self::SimdMax | Self::SimdMin => {
                (vec![FloatScalar], IntrinsicResult::FloatScalar)
            }
            Self::Matrix => (vec![DTypeName], IntrinsicResult::Frag8x8OfNamedDtype),
            Self::MatrixLoad | Self::MatrixLoadTranspose | Self::MatrixStore => {
                (vec![Frag8x8, Tile2, Int, Int], IntrinsicResult::Void)
            }
            Self::MatrixMultiplyAccumulate => (
                vec![Frag8x8, Frag8x8, Frag8x8, Frag8x8],
                IntrinsicResult::Void,
            ),
        };
        Intrinsic {
            operation: self,
            params,
            result,
        }
    }
}
impl std::fmt::Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
#[derive(Clone, Debug)]
pub struct Intrinsic {
    pub operation: Operation,
    pub params: Vec<IntrinsicParam>,
    pub result: IntrinsicResult,
}

#[derive(Clone, Debug)]
pub enum IntrinsicParam {
    /// Any scalar of a float dtype; all `FloatScalar` params of one call share a dtype.
    FloatScalar,
    /// A dtype name, e.g. `f32`.
    DTypeName,
    /// An 8x8 fragment of any dtype.
    Frag8x8,
    /// A tile (or tile view) of rank 2.
    Tile2,
    /// A runtime participant index, checked against the selected group.
    Integer,
    /// A symbolic integer.
    Int,
}

#[derive(Clone, Debug)]
pub enum IntrinsicResult {
    Integer,
    Void,
    /// Same dtype as the `FloatScalar` params.
    FloatScalar,
    /// An 8x8 fragment of the dtype named by the `DTypeName` param.
    Frag8x8OfNamedDtype,
}

pub fn table(backend: &str) -> Option<Vec<Intrinsic>> {
    use Operation::*;
    match backend {
        "metal" => Some(
            [
                LaneIndex,
                ShuffleIndex,
                SimdSum,
                SimdMax,
                SimdMin,
                Matrix,
                MatrixLoad,
                MatrixLoadTranspose,
                MatrixStore,
                MatrixMultiplyAccumulate,
            ]
            .into_iter()
            .map(Operation::signature)
            .collect(),
        ),
        "cuda" => Some([SimdSum, LaneIndex, ShuffleIndex].into_iter().map(Operation::signature).collect()),
        "cpu" | "vulkan" => Some(vec![]),
        _ => None,
    }
}

pub fn frag8x8(dtype: DType) -> Ty {
    Ty::Frag(Shaped::new(
        vec![Sym::constant(8), Sym::constant(8)],
        Elem::Dtype(dtype),
    ))
}
