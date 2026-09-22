//! CUDA backend: target discovery, the sealed capability registry, direct
//! typed-kernel PTX compilation, and the CUDA device service/executor.

mod buffer;
pub mod capability;
mod command;
mod compile;
mod driver;
mod executor;
mod factory;
pub mod model;
mod open;
mod profile;
mod ptx;
mod registry;

pub use buffer::Buffer;
pub use capability::CudaIntrinsic;
pub use command::{CompiledKernel, OccupancyRelation};
pub use compile::NativeCandidate;
pub use executor::{Device, Executor};
pub use open::{open, OpenedCuda};
pub use profile::{
    describe, device_count, ComputeCapability, CudaFacts, CudaKernelAbi, DeviceDescriptor,
    DriverApiVersion, PtxFeatureSet, PtxTarget, TensorMemory, BACKEND_REVISION,
};
pub use registry::registry;

use seismic_estimator::ServiceClassId;
use seismic_ir::target::{
    AddressableResourceClass, AddressableResourceRealization, IntrinsicIdentityBuilder,
    ResourceOwnershipScope,
};
use seismic_lang::registry::BackendName;
use seismic_target::{
    ClusterPortability, DeviceDescription, NativeArtifactMetrics, NativeClusterDomain,
    NativeCompilationError, NativeKernelDescription, NativeKernelIdentity, NativeKernelReflection,
    NativeLaunchDomain, NativeNumericalModeIdentity, NativeResourceUsage, NativeResources,
};
use sha2::{Digest, Sha256};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cuda;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CudaLaunchMode {
    Independent,
    CooperativeGrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CudaNumericalMode;

pub(crate) const SERVICE_INSTRUCTION: ServiceClassId =
    ServiceClassId::new("cuda.scalar-instruction");
pub(crate) const SERVICE_GLOBAL_READ: ServiceClassId = ServiceClassId::new("cuda.global-read");
pub(crate) const SERVICE_GLOBAL_WRITE: ServiceClassId = ServiceClassId::new("cuda.global-write");
pub(crate) const SERVICE_SHARED_READ: ServiceClassId = ServiceClassId::new("cuda.shared-read");
pub(crate) const SERVICE_SHARED_WRITE: ServiceClassId = ServiceClassId::new("cuda.shared-write");
pub(crate) const SERVICE_ATOMIC: ServiceClassId = ServiceClassId::new("cuda.atomic");
pub(crate) const SERVICE_BARRIER: ServiceClassId = ServiceClassId::new("cuda.barrier");
pub(crate) const SERVICE_SUBGROUP: ServiceClassId = ServiceClassId::new("cuda.subgroup");
pub(crate) const SERVICE_MATRIX: ServiceClassId = ServiceClassId::new("cuda.matrix-mma");
pub(crate) const SERVICE_REPACK: ServiceClassId = ServiceClassId::new("cuda.representation-repack");

fn occupancy_relation(
    context: &std::sync::Arc<crate::driver::Context>,
    function: crate::driver::Handle,
    max_threads: u32,
    max_dynamic_shared: u64,
) -> Result<crate::command::OccupancyRelation, NativeCompilationError> {
    let query = |threads, bytes| {
        crate::driver::occupancy_max_active_blocks(context, function, threads, bytes)
            .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))
    };
    let mut blocks = Vec::with_capacity(max_threads as usize);
    for threads in 1..=max_threads {
        let mut regimes = Vec::new();
        let mut first = 0u64;
        let mut active = query(threads, first)?;
        loop {
            if first == max_dynamic_shared {
                regimes.push((max_dynamic_shared, active));
                break;
            }
            let final_active = query(threads, max_dynamic_shared)?;
            if final_active == active {
                regimes.push((max_dynamic_shared, active));
                break;
            }
            let mut low = first + 1;
            let mut high = max_dynamic_shared;
            while low < high {
                let middle = low + (high - low) / 2;
                if query(threads, middle)? < active {
                    high = middle;
                } else {
                    low = middle + 1;
                }
            }
            regimes.push((low - 1, active));
            first = low;
            active = query(threads, first)?;
        }
        blocks.push(crate::command::BlockOccupancy { threads, regimes });
    }
    Ok(crate::command::OccupancyRelation { blocks })
}

fn active_blocks_expression(
    arena: &mut seismic_lang::expr::ExprArena,
    relation: &crate::command::OccupancyRelation,
    threads: seismic_lang::expr::NatExpr,
    dynamic_shared: seismic_lang::expr::NatExpr,
) -> seismic_lang::expr::NatExpr {
    let mut selected = arena.nat(0);
    for block in &relation.blocks {
        let mut active = arena.nat(0);
        for &(maximum, blocks) in block.regimes.iter().rev() {
            let maximum = arena.nat(maximum);
            let within = arena.nat_cmp(seismic_lang::expr::CmpOp::Le, dynamic_shared, maximum);
            let blocks = arena.nat(u64::from(blocks));
            active = arena.nat_select(within, blocks, active);
        }
        let block_threads = arena.nat(u64::from(block.threads));
        let matches = arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, threads, block_threads);
        selected = arena.nat_select(matches, active, selected);
    }
    selected
}

impl seismic_ir::target::KernelDialect for Cuda {
    const NAME: BackendName = BackendName::Cuda;
    type Intrinsic = CudaIntrinsic;
    type Facts = CudaFacts;

    fn write_intrinsic_identity(
        intrinsic: &CudaIntrinsic,
        identity: &mut IntrinsicIdentityBuilder,
    ) {
        match intrinsic {
            CudaIntrinsic::LaneIndex => identity.variant("lane-index"),
            CudaIntrinsic::Shuffle { dtype } => {
                identity.variant("shuffle");
                identity.dtype(*dtype);
            }
            CudaIntrinsic::SubgroupReduce { op, dtype } => {
                identity.variant("subgroup-reduce");
                identity.u32(match op {
                    seismic_lang::intrinsics::ReduceOp::Sum => 0,
                    seismic_lang::intrinsics::ReduceOp::Max => 1,
                    seismic_lang::intrinsics::ReduceOp::Min => 2,
                    seismic_lang::intrinsics::ReduceOp::Argmax => 3,
                });
                identity.dtype(*dtype);
            }
            CudaIntrinsic::MatrixMatmul {
                elem,
                a,
                b,
                destination,
            } => {
                identity.variant("matrix-matmul");
                identity.dtype(*elem);
                identity.logical_tensor(a);
                identity.logical_tensor(b);
                identity.logical_tensor(destination);
            }
            CudaIntrinsic::MatrixMatmulAdd {
                elem,
                a,
                b,
                c,
                destination,
            } => {
                identity.variant("matrix-matmul-add");
                identity.dtype(*elem);
                identity.logical_tensor(a);
                identity.logical_tensor(b);
                identity.logical_tensor(c);
                identity.logical_tensor(destination);
            }
            CudaIntrinsic::NvFp4Matmul {
                a,
                b,
                destination,
                tensor_memory,
            } => {
                identity.variant("nvfp4-matmul");
                identity.logical_tensor(a);
                identity.logical_tensor(b);
                identity.logical_tensor(destination);
                identity.u32(tensor_memory.ordinal());
            }
            CudaIntrinsic::NvFp4MatmulAdd {
                a,
                b,
                c,
                destination,
                tensor_memory,
            } => {
                identity.variant("nvfp4-matmul-add");
                identity.logical_tensor(a);
                identity.logical_tensor(b);
                identity.logical_tensor(c);
                identity.logical_tensor(destination);
                identity.u32(tensor_memory.ordinal());
            }
        }
    }

    fn intrinsic_numerics(
        _facts: &Self::Facts,
        signature: &seismic_lang::registry::IntrinsicSignature,
        _intrinsic: &Self::Intrinsic,
    ) -> seismic_ir::target::IntrinsicNumericalSemantics {
        seismic_ir::target::IntrinsicNumericalSemantics {
            arithmetic: signature.numerical.clone(),
            // Every emitted PTX operation uses the non-`.ftz` form. The
            // target advertises no implicit flush relaxation for these
            // concrete subgroup and dense-matrix instructions.
            flush_to_zero: false,
        }
    }

    fn intrinsic_addressable_resources(
        intrinsic: &Self::Intrinsic,
    ) -> Vec<seismic_ir::kernel::ops::AddressableResourceHandle> {
        match intrinsic {
            CudaIntrinsic::NvFp4Matmul { tensor_memory, .. }
            | CudaIntrinsic::NvFp4MatmulAdd { tensor_memory, .. } => vec![*tensor_memory],
            CudaIntrinsic::LaneIndex
            | CudaIntrinsic::Shuffle { .. }
            | CudaIntrinsic::SubgroupReduce { .. }
            | CudaIntrinsic::MatrixMatmul { .. }
            | CudaIntrinsic::MatrixMatmulAdd { .. } => Vec::new(),
        }
    }
}

impl seismic_target::TargetFamily for Cuda {
    type KernelAbi = CudaKernelAbi;
    type NativeLaunchMode = CudaLaunchMode;
    type NativeNumericalMode = CudaNumericalMode;
    type NativeProperties = OccupancyRelation;
}

pub struct CudaNativeCompiler;
static CUDA_NATIVE_COMPILER: CudaNativeCompiler = CudaNativeCompiler;

pub fn native_compiler() -> &'static CudaNativeCompiler {
    &CUDA_NATIVE_COMPILER
}

pub(crate) fn cooperative_launch_mode(facts: &CudaFacts) -> Option<CudaLaunchMode> {
    facts
        .cooperative_launch
        .then_some(CudaLaunchMode::CooperativeGrid)
}

pub(crate) fn native_launch_constraints(
    target: &DeviceDescription<Cuda>,
    arena: &mut seismic_lang::expr::ExprArena,
    launch: &seismic_ir::schedule::Launch,
    locals: &seismic_ir::storage::LaunchLocalLayout,
    _kernel: &seismic_ir::kernel::Kernel<Cuda>,
    native: &seismic_target::NativeKernelDescription<Cuda>,
) -> Vec<seismic_lang::expr::BoolExpr> {
    let facts = target.facts();
    let threads = arena.nat_product(&launch.workgroup);
    let blocks = arena.nat_product(&launch.grid);
    let one = arena.nat(1);
    let resident =
        active_blocks_expression(arena, &native.properties, threads, locals.workgroup_bytes);
    let multiprocessors = arena.nat(u64::from(facts.multiprocessors));
    let capacity = arena.nat_mul(multiprocessors, resident);
    let mut constraints = vec![arena.nat_cmp(seismic_lang::expr::CmpOp::Ge, resident, one)];
    if launch.mode == seismic_ir::schedule::LaunchMode::CooperativeGrid {
        constraints.push(arena.nat_cmp(seismic_lang::expr::CmpOp::Le, blocks, capacity));
    }
    constraints
}

pub(crate) fn addressable_resources(facts: &CudaFacts) -> Vec<AddressableResourceClass> {
    match facts.tensor_memory {
        profile::TensorMemory::Unavailable => Vec::new(),
        profile::TensorMemory::Tcgen05 {
            columns,
            allocation_granularity_columns,
            ..
        } => vec![AddressableResourceClass {
            stable_name: "cuda.tcgen05.tensor-memory",
            unit_name: "column",
            ownership: ResourceOwnershipScope::Workgroup,
            capacity_units: u64::from(columns),
            alignment_units: u64::from(allocation_granularity_columns),
            realization: AddressableResourceRealization::Native,
        }],
    }
}

pub(crate) fn semantic_intrinsic_requirements(
    _target: &DeviceDescription<Cuda>,
    arena: &mut seismic_lang::expr::ExprArena,
    signature: &seismic_lang::registry::IntrinsicSignature,
    _parallel_extent: seismic_lang::expr::NatExpr,
) -> seismic_ir::kernel::ops::SemanticIntrinsicLaunchRequirements {
    if signature.capability != capability::capability_id(capability::SUBGROUP)
        && signature.capability != capability::capability_id(capability::MATRIX)
    {
        panic!(
            "CUDA static capability registry passed a foreign intrinsic to its requirements hook"
        )
    }
    seismic_ir::kernel::ops::SemanticIntrinsicLaunchRequirements {
        required_mode: None,
        required_workgroup: matches!(signature.name, "nvfp4_matmul" | "nvfp4_matmul_add")
            .then(|| [arena.nat(128), arena.nat(1), arena.nat(1)]),
    }
}

pub(crate) fn lower_semantic_intrinsic(
    target: &DeviceDescription<Cuda>,
    _domain: &seismic_ir::kernel::ops::SegmentLaunchDomain,
    call: seismic_ir::kernel::ops::SemanticIntrinsicCall<'_>,
    sink: &mut seismic_ir::kernel::ops::SemanticIntrinsicSink<'_, '_, Cuda>,
) {
    use seismic_ir::kernel::ops::{IntrinsicResources, SemanticIntrinsicOperand};
    use seismic_lang::intrinsics::ReduceOp;

    if !capability::emitted_intrinsics().contains(&call.signature.id) {
        panic!("CUDA semantic lowering received an intrinsic absent from its native emitter set")
    }
    let resources = IntrinsicResources {
        workgroup_bytes: None,
        participant_bytes: None,
        register_bytes: None,
        requires_subgroup: true,
    };
    if call.signature.capability == capability::capability_id(capability::SUBGROUP) {
        let operation = match call.signature.name {
            "lane_index" => CudaIntrinsic::LaneIndex,
            "shuffle" => {
                let dtype = match call.operands.first() {
                    Some(SemanticIntrinsicOperand::Scalar(value)) => value.dtype(),
                    _ => panic!("registered CUDA shuffle has no scalar value operand"),
                };
                CudaIntrinsic::Shuffle { dtype }
            }
            "simd_sum" => {
                let dtype = match call.operands.first() {
                    Some(SemanticIntrinsicOperand::Scalar(value)) => value.dtype(),
                    _ => panic!("registered CUDA subgroup reduction has no scalar operand"),
                };
                CudaIntrinsic::SubgroupReduce {
                    op: ReduceOp::Sum,
                    dtype,
                }
            }
            "simd_max" => {
                let dtype = match call.operands.first() {
                    Some(SemanticIntrinsicOperand::Scalar(value)) => value.dtype(),
                    _ => panic!("registered CUDA subgroup reduction has no scalar operand"),
                };
                CudaIntrinsic::SubgroupReduce {
                    op: ReduceOp::Max,
                    dtype,
                }
            }
            "simd_min" => {
                let dtype = match call.operands.first() {
                    Some(SemanticIntrinsicOperand::Scalar(value)) => value.dtype(),
                    _ => panic!("registered CUDA subgroup reduction has no scalar operand"),
                };
                CudaIntrinsic::SubgroupReduce {
                    op: ReduceOp::Min,
                    dtype,
                }
            }
            _ => panic!("CUDA subgroup registry contains an unimplemented semantic intrinsic"),
        };
        sink.emit(operation, resources, None);
        return;
    }
    if call.signature.capability == capability::capability_id(capability::MATRIX) {
        let destination = call
            .destination
            .unwrap_or_else(|| panic!("registered CUDA matrix intrinsic has no owned destination"));
        let a = sink.readable(call.operands[0].clone());
        let b = sink.readable(call.operands[1].clone());
        let elem = match &call.operands[0] {
            SemanticIntrinsicOperand::Readable(place) => {
                seismic_lang::registry::representation_info(place.representation()).decoded
            }
            _ => panic!("registered CUDA matrix intrinsic has no readable left operand"),
        };
        let operation = match call.signature.name {
            "matmul" => CudaIntrinsic::MatrixMatmul {
                elem,
                a,
                b,
                destination: sink.writable(SemanticIntrinsicOperand::Writable(destination.clone())),
            },
            "matmul_add" => CudaIntrinsic::MatrixMatmulAdd {
                elem,
                a,
                b,
                c: sink.readable(call.operands[2].clone()),
                destination: sink.writable(SemanticIntrinsicOperand::Writable(destination.clone())),
            },
            "nvfp4_matmul" | "nvfp4_matmul_add" => {
                let class = target
                    .addressable_resource_class("cuda.tcgen05.tensor-memory")
                    .unwrap_or_else(|| {
                        panic!("CUDA registry advertised NVFP4 without tensor memory")
                    });
                // One CTA-group::1 m128n8k64 NVFP4 tile requires eight
                // accumulator columns, four block16 SFA columns (one for
                // each 32-row group), and one SFB column for N=8. The
                // exact 13-column live set rounds to the architectural
                // 32-column allocation granularity.
                let tensor_memory_units = sink.nat(32);
                let tensor_memory = sink.addressable_resource(
                    class,
                    tensor_memory_units,
                    32,
                    seismic_ir::target::ResourceLifetime::Operation,
                );
                if call.signature.name == "nvfp4_matmul" {
                    CudaIntrinsic::NvFp4Matmul {
                        a,
                        b,
                        destination: sink
                            .writable(SemanticIntrinsicOperand::Writable(destination.clone())),
                        tensor_memory,
                    }
                } else {
                    CudaIntrinsic::NvFp4MatmulAdd {
                        a,
                        b,
                        c: sink.readable(call.operands[4].clone()),
                        destination: sink
                            .writable(SemanticIntrinsicOperand::Writable(destination.clone())),
                        tensor_memory,
                    }
                }
            }
            _ => panic!("CUDA matrix registry contains an unimplemented semantic intrinsic"),
        };
        sink.emit(operation, resources, Some(destination));
        return;
    }
    panic!("CUDA semantic lowering received an intrinsic from another capability namespace")
}

impl seismic_target::NativeCompiler<Cuda> for CudaNativeCompiler {
    type Context = ();
    type Candidate = NativeCandidate;
    type Handle = CompiledKernel;
    fn form(
        &self,
        _context: &Self::Context,
        target: &DeviceDescription<Cuda>,
        kernel: &seismic_ir::kernel::Kernel<Cuda>,
        layout: &seismic_ir::target::KernelEmissionLayout,
    ) -> Result<Self::Candidate, NativeCompilationError> {
        compile::compile_kernel(target, kernel, layout)
    }
    fn reflect(
        &self,
        target: &DeviceDescription<Cuda>,
        kernel: &seismic_ir::kernel::Kernel<Cuda>,
        _layout: &seismic_ir::target::KernelEmissionLayout,
        candidate: Self::Candidate,
    ) -> Result<NativeKernelReflection<Cuda, Self::Handle>, NativeCompilationError> {
        let reflection_started = Instant::now();
        let context = candidate.module.context.clone();
        let function = crate::driver::module_function(&candidate.module, &candidate.entry)
            .map_err(compile::jit_failure)?;
        let attribute = |key| {
            crate::driver::function_attribute(&context, function, key)
                .map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))
        };
        let nonnegative = |name: &str, value: i32| {
            u64::try_from(value).map_err(|_| {
                NativeCompilationError::MalformedToolchainOutput(format!(
                    "CUDA function reports negative {name}: {value}"
                ))
            })
        };
        let max_threads = nonnegative("maximum threads per block", attribute(0)?)?;
        let static_shared = nonnegative("static shared bytes", attribute(1)?)?;
        let local_bytes = nonnegative("local bytes per thread", attribute(3)?)?;
        let registers = nonnegative("registers per thread", attribute(4)?)?;
        let max_dynamic_shared = nonnegative("maximum dynamic shared bytes", attribute(8)?)?;
        if max_threads == 0 || registers == 0 {
            return Err(NativeCompilationError::MalformedToolchainOutput(
                "CUDA function reflection omitted a required resource fact".into(),
            ));
        }
        let max_threads_u32 = u32::try_from(max_threads).map_err(|_| {
            NativeCompilationError::MalformedToolchainOutput(
                "CUDA function maximum threads exceed u32".into(),
            )
        })?;
        let occupancy =
            occupancy_relation(&context, function, max_threads_u32, max_dynamic_shared)?;
        let mut occupancy_metadata =
            u64::try_from(std::mem::size_of_val(&occupancy)).map_err(|_| {
                NativeCompilationError::MalformedToolchainOutput(
                    "CUDA occupancy metadata size exceeds u64".into(),
                )
            })?;
        for block in &occupancy.blocks {
            let fixed = u64::try_from(std::mem::size_of_val(block)).map_err(|_| {
                NativeCompilationError::MalformedToolchainOutput(
                    "CUDA occupancy metadata size exceeds u64".into(),
                )
            })?;
            let regimes = u64::try_from(block.regimes.len())
                .ok()
                .and_then(|count| count.checked_mul(std::mem::size_of::<(u64, u32)>() as u64))
                .ok_or_else(|| {
                    NativeCompilationError::MalformedToolchainOutput(
                        "CUDA occupancy metadata size exceeds u64".into(),
                    )
                })?;
            occupancy_metadata = occupancy_metadata
                .checked_add(fixed)
                .and_then(|total| total.checked_add(regimes))
                .ok_or_else(|| {
                    NativeCompilationError::MalformedToolchainOutput(
                        "CUDA occupancy metadata size exceeds u64".into(),
                    )
                })?;
        }
        let cluster = if target.facts().compute_capability.major >= 9 {
            let required = attribute(10)?;
            let width = attribute(11)?;
            let height = attribute(12)?;
            let depth = attribute(13)?;
            if required != 0 || [width, height, depth].into_iter().any(|axis| axis != 0) {
                if [width, height, depth].into_iter().any(|axis| axis <= 0) {
                    return Err(NativeCompilationError::MalformedToolchainOutput(
                        "CUDA function has a partial required cluster shape".into(),
                    ));
                }
                let dimension = |axis| {
                    u32::try_from(axis).map_err(|_| {
                        NativeCompilationError::MalformedToolchainOutput(
                            "CUDA function reports an invalid required cluster dimension".into(),
                        )
                    })
                };
                NativeClusterDomain::Required {
                    dimensions: [dimension(width)?, dimension(height)?, dimension(depth)?],
                    portability: if attribute(14)? == 0 {
                        ClusterPortability::Portable
                    } else {
                        ClusterPortability::NonPortableAllowed
                    },
                }
            } else {
                NativeClusterDomain::NotApplicable
            }
        } else {
            NativeClusterDomain::NotApplicable
        };
        let mut modes = vec![CudaLaunchMode::Independent];
        if target.facts().cooperative_launch {
            modes.push(CudaLaunchMode::CooperativeGrid);
        }
        let handle = CompiledKernel {
            module: candidate.module,
            function,
            layout: candidate.layout,
        };
        let description = NativeKernelDescription {
            identity: NativeKernelIdentity {
                compatibility: target.compatibility_identity().clone(),
                artifact_digest: candidate.artifact_digest,
            },
            abi: target.kernel_abi_layout(kernel),
            launch: NativeLaunchDomain {
                modes,
                subgroup_width: Some(capability::subgroup::WARP_LANES),
                cluster,
                max_grid: target.limits().max_grid,
                max_workgroup_size: target.limits().max_workgroup_size,
                max_workgroup_threads: max_threads,
                max_dynamic_local_bytes: max_dynamic_shared,
            },
            resources: NativeResources {
                registers_per_participant: NativeResourceUsage::Exact(registers),
                spill_bytes_per_participant: NativeResourceUsage::Exact(local_bytes),
                static_local_bytes: NativeResourceUsage::Exact(static_shared),
            },
            numerics: CudaNumericalMode,
            numerical_identity: NativeNumericalModeIdentity {
                fingerprint: Sha256::digest(b"seismic-cuda-ptx-rn-no-ftz-v1").into(),
            },
            properties: occupancy,
        };
        let metrics = NativeArtifactMetrics {
            compilation_ns: candidate
                .compilation_ns
                .checked_add(
                    u64::try_from(reflection_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                )
                .unwrap_or(u64::MAX),
            code_bytes: candidate.image_bytes,
            metadata_bytes: candidate
                .metadata_bytes
                .checked_add(occupancy_metadata)
                .ok_or_else(|| {
                    NativeCompilationError::MalformedToolchainOutput(
                        "CUDA native metadata size exceeds u64".into(),
                    )
                })?,
        };
        Ok(NativeKernelReflection::new(handle, description, metrics))
    }
}
