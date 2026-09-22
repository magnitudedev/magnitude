//! The closed CUDA intrinsic vocabulary and the sealed capability registry
//! (spec §4.2, §24 R4).
//!
//! `CudaIntrinsic` is the target dialect's intrinsic vocabulary: the only target operations
//! admitted in typed kernel IR for CUDA. `subgroup`/`matrix` enumerate the
//! exact registry rows with native emitter arms; the compiler-owned semantic
//! walker is the only construction path.
//!
//! Registry identities are resolved through `seismic_lang::registry` by the
//! capability's namespace name and the signature's name and operand
//! categories; a signature this backend implements that the registry does
//! not define is an inconsistent static registry (§13.3.1).

pub mod matrix;
pub mod subgroup;

use crate::factory;
use crate::Cuda;
use seismic_compiler::target::{
    CapabilityRegistration, CompilerRegistry, CompilerRegistryParts, IntrinsicImplementation,
};
use seismic_ir::kernel::ops::{AddressableResourceHandle, LogicalTensorMap};
use seismic_ir::target::{DataTypeSupport, TargetLimits};
use seismic_lang::ids::{CapabilityId, IntrinsicId};
use seismic_lang::intrinsics::ReduceOp;
use seismic_lang::registry::{self, BackendName, OperandCategory};
use seismic_lang::types::DType;
use std::collections::BTreeSet;
use std::sync::OnceLock;

/// The closed CUDA intrinsic operation set. Values (`outs`/`args` of
/// `Op::Intrinsic`) are typed kernel SSA scalars; places are carried here
/// because the erased op vocabulary references values only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaIntrinsic {
    /// `cuda.subgroup.lane_index: () -> i32` (`%laneid`).
    LaneIndex,
    /// `cuda.subgroup.shuffle: (T, i32) -> T`, one warp lane exchange.
    Shuffle { dtype: DType },
    /// `cuda.subgroup.simd_{sum,max,min}: (T) -> T`, the 32-lane butterfly.
    SubgroupReduce { op: ReduceOp, dtype: DType },
    /// `cuda.matrix.matmul: (a[rows,inner] E, b[inner,columns] E) -> [rows,columns] f32`
    /// into `destination`, warp-collective.
    MatrixMatmul {
        elem: DType,
        a: LogicalTensorMap,
        b: LogicalTensorMap,
        destination: LogicalTensorMap,
    },
    /// `cuda.matrix.matmul_add: (a, b, c[rows,columns] E) -> [rows,columns] E`
    /// into `destination`, warp-collective.
    MatrixMatmulAdd {
        elem: DType,
        a: LogicalTensorMap,
        b: LogicalTensorMap,
        c: LogicalTensorMap,
        destination: LogicalTensorMap,
    },
    NvFp4Matmul {
        a: LogicalTensorMap,
        b: LogicalTensorMap,
        destination: LogicalTensorMap,
        tensor_memory: AddressableResourceHandle,
    },
    NvFp4MatmulAdd {
        a: LogicalTensorMap,
        b: LogicalTensorMap,
        c: LogicalTensorMap,
        destination: LogicalTensorMap,
        tensor_memory: AddressableResourceHandle,
    },
}

/// The namespace names of the two CUDA capability families of the language
/// contract (`design/inference/seismic-language-and-capabilities.md`).
pub const SUBGROUP: &str = "subgroup";
pub const MATRIX: &str = "matrix";

/// The registry id of a CUDA capability namespace.
pub fn capability_id(name: &str) -> CapabilityId {
    registry::capability(BackendName::Cuda, name)
        .unwrap_or_else(|| panic!("registry inconsistency: the registry defines no `cuda.{name}` capability that seismic-cuda implements"))
}

/// The registry id of one signature of a CUDA capability, selected by name
/// and exact operand categories.
pub fn signature_id(capability: &str, name: &str, arguments: &[OperandCategory]) -> IntrinsicId {
    let id = capability_id(capability);
    registry::intrinsics(id)
        .iter()
        .find(|signature| {
            signature.name == name
                && signature.arguments.len() == arguments.len()
                && signature
                    .arguments
                    .iter()
                    .zip(arguments)
                    .all(|(declared, wanted)| declared.category == *wanted)
        })
        .map(|signature| signature.id)
        .unwrap_or_else(|| {
            panic!(
                "registry inconsistency: the registry defines no `cuda.{capability}.{name}` signature over {arguments:?} that seismic-cuda implements"
            )
        })
}

/// `cuda.subgroup` on this profile: every implemented signature when the
/// device's subgroup is the 32-lane warp the butterfly and shuffle masks
/// assume; none otherwise.
fn subgroup_supported(
    _facts: &crate::profile::CudaFacts,
    limits: &TargetLimits,
    dtypes: &DataTypeSupport,
) -> BTreeSet<IntrinsicId> {
    if limits.subgroup_width != Some(subgroup::WARP_LANES) {
        return BTreeSet::new();
    }
    subgroup::implemented()
        .into_iter()
        .filter(|(dtype, _)| dtype.is_none_or(|dtype| dtypes.scalars.contains(&dtype)))
        .map(|(_, id)| id)
        .collect()
}

/// `cuda.matrix` on this profile: every implemented signature (the `sm_80`
/// floor guarantees `mma.sync.m16n8k16`).
fn matrix_supported(
    facts: &crate::profile::CudaFacts,
    _limits: &TargetLimits,
    dtypes: &DataTypeSupport,
) -> BTreeSet<IntrinsicId> {
    matrix::implemented()
        .into_iter()
        .filter(|id| {
            let signature = registry::intrinsic_signature(*id);
            if matches!(signature.name, "nvfp4_matmul" | "nvfp4_matmul_add")
                && !matches!(
                    facts.tensor_memory,
                    crate::profile::TensorMemory::Tcgen05 { .. }
                )
            {
                return false;
            }
            signature
                .arguments
                .iter()
                .all(|argument| match &argument.category {
                    OperandCategory::Readable { representation, .. } => {
                        dtypes.representations.contains(representation)
                    }
                    _ => true,
                })
        })
        .collect()
}

fn assemble() -> CompilerRegistry<Cuda> {
    let subgroup_ids: Vec<IntrinsicId> = subgroup::implemented()
        .into_iter()
        .map(|(_, id)| id)
        .collect();
    let matrix_ids: Vec<IntrinsicId> = matrix::implemented().into_iter().collect();
    CompilerRegistry::assemble(CompilerRegistryParts {
        capabilities: vec![
            CapabilityRegistration {
                capability: capability_id(SUBGROUP),
                implementations: subgroup_ids
                    .into_iter()
                    .map(|id| IntrinsicImplementation {
                        id,
                        launch_requirements: crate::semantic_intrinsic_requirements,
                        lower: crate::lower_semantic_intrinsic,
                    })
                    .collect(),
                supported: subgroup_supported,
            },
            CapabilityRegistration {
                capability: capability_id(MATRIX),
                implementations: matrix_ids
                    .into_iter()
                    .map(|id| IntrinsicImplementation {
                        id,
                        launch_requirements: crate::semantic_intrinsic_requirements,
                        lower: crate::lower_semantic_intrinsic,
                    })
                    .collect(),
                supported: matrix_supported,
            },
        ],
        structural_factories: factory::structural_factories(),
        independent_launch_mode: crate::CudaLaunchMode::Independent,
        cooperative_launch_mode: crate::cooperative_launch_mode,
        native_launch_constraints: crate::native_launch_constraints,
        addressable_resources: crate::addressable_resources,
        emitted_intrinsics: emitted_intrinsics(),
    })
}

pub(crate) fn emitted_intrinsics() -> BTreeSet<IntrinsicId> {
    subgroup::implemented()
        .into_iter()
        .map(|(_, id)| id)
        .chain(matrix::implemented())
        .collect()
}

static REGISTRY: OnceLock<CompilerRegistry<Cuda>> = OnceLock::new();

/// The sealed CUDA capability registry, assembled once (§4.2). Assembly
/// panics on an inconsistent registration set (§13.3.1).
pub fn registry() -> &'static CompilerRegistry<Cuda> {
    REGISTRY.get_or_init(assemble)
}
