//! Typed backend operations shared by checking, effects, realization and emission.
//! Every admitted operation defines its signature and semantic execution contract
//! exhaustively. These semantics are not native latency or instruction counts.

/// Revision of the typed intrinsic registry contract. Backend capability fingerprints retain
/// this value so cached availability decisions cannot survive a registry semantic change.
pub const REGISTRY_REVISION: &str = "seismic-intrinsics-v2";

/// Stable source-level identity of one backend capability namespace.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CapabilityId {
    pub backend: String,
    pub name: String,
}

impl CapabilityId {
    pub fn new(backend: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            backend: backend.into(),
            name: name.into(),
        }
    }

    pub fn path(&self) -> String {
        format!("{}.{}", self.backend, self.name)
    }
}

/// Stable source-level identity of an intrinsic within a capability namespace.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IntrinsicId {
    pub capability: CapabilityId,
    pub name: String,
}

impl IntrinsicId {
    pub fn path(&self) -> String {
        format!("{}.{}", self.capability.path(), self.name)
    }
}

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
    /// Logical matrix multiplication over rank-two tensor values.
    MatrixMatmul,
    /// Logical matrix multiplication added to an accumulator tensor.
    MatrixMatmulAdd,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Semantics {
    ParticipantIndex,
    Exchange,
    Reduction(crate::sir::ReduceOp),
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
    MatrixMatmul,
    MatrixMatmulAdd,
}
impl Operation {
    /// The operation produces a fresh owned logical value. Execution lowering must preserve it
    /// as one result-producing operation with a compiler-owned destination; it may not
    /// scalarize the expression into unrelated element operations.
    pub fn produces_owned_result(self) -> bool {
        matches!(self, Self::MatrixMatmul | Self::MatrixMatmulAdd)
    }

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
            Self::MatrixMatmul => "matmul",
            Self::MatrixMatmulAdd => "matmul_add",
        }
    }
    pub fn semantics(self) -> Semantics {
        match self {
            Self::LaneIndex => Semantics::ParticipantIndex,
            Self::ShuffleIndex => Semantics::Exchange,
            Self::SimdSum => Semantics::Reduction(crate::sir::ReduceOp::Sum),
            Self::SimdMax => Semantics::Reduction(crate::sir::ReduceOp::Max),
            Self::SimdMin => Semantics::Reduction(crate::sir::ReduceOp::Min),
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
            Self::MatrixMatmul => Semantics::MatrixMatmul,
            Self::MatrixMatmulAdd => Semantics::MatrixMatmulAdd,
        }
    }
    /// Declaration creates fragment storage; all executable intrinsics involve
    /// the subgroup. This describes participation, not a barrier insertion policy.
    pub fn collective(self) -> bool {
        !matches!(
            self,
            Self::Matrix | Self::LaneIndex | Self::MatrixMatmul | Self::MatrixMatmulAdd
        )
    }
    pub fn writes_arguments(self) -> &'static [usize] {
        match self {
            Self::MatrixLoad | Self::MatrixLoadTranspose | Self::MatrixMultiplyAccumulate => &[0],
            Self::MatrixStore => &[1],
            Self::LaneIndex
            | Self::ShuffleIndex
            | Self::SimdSum
            | Self::SimdMax
            | Self::SimdMin
            | Self::Matrix
            | Self::MatrixMatmul
            | Self::MatrixMatmulAdd => &[],
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
            Self::MatrixMatmul | Self::MatrixMatmulAdd => false,
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
            Self::MatrixMatmul => (
                vec![MatrixOperand, MatrixOperand, DTypeName],
                IntrinsicResult::LogicalMatrix,
            ),
            Self::MatrixMatmulAdd => (
                vec![MatrixOperand, MatrixOperand, MatrixOperand],
                IntrinsicResult::LogicalMatrix,
            ),
        };
        Intrinsic {
            id: IntrinsicId {
                capability: CapabilityId::new("internal", "legacy"),
                name: self.name().into(),
            },
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
    pub id: IntrinsicId,
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
    /// A rank-two logical tensor, view, or owned tensor value.
    MatrixOperand,
}

#[derive(Clone, Debug)]
pub enum IntrinsicResult {
    Integer,
    Void,
    /// Same dtype as the `FloatScalar` params.
    FloatScalar,
    /// An 8x8 fragment of the dtype named by the `DTypeName` param.
    Frag8x8OfNamedDtype,
    /// Rank-two owned result whose exact shape and element type are checked at the call.
    LogicalMatrix,
}

fn registered(backend: &str, capability: &str, name: &str, operation: Operation) -> Intrinsic {
    let mut intrinsic = operation.signature();
    intrinsic.id = IntrinsicId {
        capability: CapabilityId::new(backend, capability),
        name: name.into(),
    };
    intrinsic
}

/// The complete author-visible registry. Availability on an effective target is a
/// later intersection with device, toolchain, and backend-emitter support.
pub fn registry() -> Vec<Intrinsic> {
    use Operation::*;
    let mut entries = Vec::new();
    for backend in ["metal", "cuda"] {
        entries.extend([
            registered(backend, "subgroup", "lane_index", LaneIndex),
            registered(backend, "subgroup", "shuffle", ShuffleIndex),
            registered(backend, "subgroup", "simd_sum", SimdSum),
            registered(backend, "subgroup", "simd_max", SimdMax),
            registered(backend, "subgroup", "simd_min", SimdMin),
            registered(backend, "matrix", "matmul", MatrixMatmul),
            registered(backend, "matrix", "matmul_add", MatrixMatmulAdd),
        ]);
    }
    entries
}

pub fn known_backend(backend: &str) -> bool {
    matches!(backend, "metal" | "cuda" | "cpu" | "vulkan")
}

pub fn capability(backend: &str, name: &str) -> Option<CapabilityId> {
    registry()
        .into_iter()
        .any(|entry| entry.id.capability.backend == backend && entry.id.capability.name == name)
        .then(|| CapabilityId::new(backend, name))
}

pub fn lookup(backend: &str, capability: &str, name: &str) -> Option<Intrinsic> {
    lookup_all(backend, capability, name).into_iter().next()
}

pub fn lookup_all(backend: &str, capability: &str, name: &str) -> Vec<Intrinsic> {
    registry()
        .into_iter()
        .filter(|entry| {
            entry.id.capability.backend == backend
                && entry.id.capability.name == capability
                && entry.id.name == name
        })
        .collect()
}
