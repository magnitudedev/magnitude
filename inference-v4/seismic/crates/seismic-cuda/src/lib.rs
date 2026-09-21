//! CUDA backend: target discovery, the sealed capability registry, direct
//! typed-kernel PTX compilation, and the CUDA device service/executor.

mod buffer;
pub mod capability;
mod command;
mod compile;
mod driver;
mod executor;
mod factory;
mod profile;
mod ptx;
mod registry;

pub use buffer::Buffer;
pub use capability::CudaIntrinsic;
pub use command::CompiledKernel;
pub use compile::NativeCandidate;
pub use executor::{Device, Executor};
pub use profile::{
    describe, device_count, ComputeCapability, CudaFacts, CudaKernelAbi, DeviceDescriptor,
    DriverApiVersion, PtxFeatureSet, PtxTarget, TensorMemory, BACKEND_REVISION,
};
pub use registry::registry;

use seismic_compiler::errors::NativeCompilationError;
use seismic_compiler::target::{
    AddressableResourceClass, AddressableResourceRealization, Backend, ClusterPortability,
    DemandMode, DemandScope, DeviceContract, ExecutionDemand, IntrinsicIdentityBuilder,
    NativeArtifactMetrics, NativeClusterDomain, NativeKernelIdentity, NativeKernelReflection,
    NativeKernelResources, NativeLaunchDomain, NativeNumericalModeIdentity, NativeResourceUsage,
    ResourceOwnershipScope, ServiceClassId,
};
use seismic_lang::registry::BackendName;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
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

impl Backend for Cuda {
    const NAME: BackendName = BackendName::Cuda;
    type Intrinsic = CudaIntrinsic;
    type Facts = CudaFacts;
    type KernelAbi = CudaKernelAbi;
    type NativeLaunchMode = CudaLaunchMode;
    type RawNativeCandidate = NativeCandidate;
    type NativeKernelHandle = CompiledKernel;
    type NativeNumericalMode = CudaNumericalMode;
    type Executor = Executor;

    fn independent_launch_mode() -> Self::NativeLaunchMode {
        CudaLaunchMode::Independent
    }

    fn cooperative_launch_mode(facts: &Self::Facts) -> Option<Self::NativeLaunchMode> {
        facts
            .cooperative_launch
            .then_some(CudaLaunchMode::CooperativeGrid)
    }

    fn native_launch_constraints(
        target: &DeviceContract<Self>,
        arena: &mut seismic_lang::expr::ExprArena,
        launch: &seismic_compiler::schedule::Launch,
        locals: &seismic_compiler::storage::LaunchLocalLayout,
        _kernel: &seismic_compiler::kernel::Kernel<Self>,
        native: &seismic_compiler::target::NativeKernel<Self>,
    ) -> Vec<seismic_lang::expr::BoolExpr> {
        let facts = target.facts();
        let threads = arena.nat_product(&launch.workgroup);
        let blocks = arena.nat_product(&launch.grid);
        let one = arena.nat(1);
        let resident = active_blocks_expression(
            arena,
            &native.handle().occupancy,
            threads,
            locals.workgroup_bytes,
        );
        let multiprocessors = arena.nat(u64::from(facts.multiprocessors));
        let capacity = arena.nat_mul(multiprocessors, resident);
        let mut constraints = vec![arena.nat_cmp(seismic_lang::expr::CmpOp::Ge, resident, one)];
        if launch.mode == seismic_compiler::schedule::LaunchMode::CooperativeGrid {
            constraints.push(arena.nat_cmp(seismic_lang::expr::CmpOp::Le, blocks, capacity));
        }
        constraints
    }

    fn addressable_resource_classes(facts: &Self::Facts) -> Vec<AddressableResourceClass> {
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

    fn emitted_intrinsics() -> BTreeSet<seismic_lang::ids::IntrinsicId> {
        capability::subgroup::implemented()
            .into_iter()
            .map(|(_, id)| id)
            .chain(capability::matrix::implemented())
            .collect()
    }

    fn required_service_classes(
        _facts: &Self::Facts,
        _supported_intrinsics: &BTreeSet<seismic_lang::ids::IntrinsicId>,
    ) -> BTreeSet<ServiceClassId> {
        [
            SERVICE_INSTRUCTION,
            SERVICE_GLOBAL_READ,
            SERVICE_GLOBAL_WRITE,
            SERVICE_SHARED_READ,
            SERVICE_SHARED_WRITE,
            SERVICE_ATOMIC,
            SERVICE_BARRIER,
            SERVICE_SUBGROUP,
            SERVICE_MATRIX,
            SERVICE_REPACK,
        ]
        .into_iter()
        .collect()
    }

    fn execution_demand(
        _target: &DeviceContract<Self>,
        arena: &mut seismic_lang::expr::ExprArena,
        _kernel: &seismic_compiler::kernel::Kernel<Self>,
        _emission: &seismic_compiler::target::KernelEmissionLayout,
        op: seismic_compiler::kernel::ops::ClosedOpView<'_, Self>,
    ) -> Vec<ExecutionDemand> {
        use seismic_compiler::kernel::ops::{
            ClosedOpView, ClosedPlace, ClosedPlaceKind, ValueType,
        };
        use seismic_compiler::target::LocalRealization;

        fn demand(
            class: ServiceClassId,
            units: seismic_lang::expr::NatExpr,
            mode: DemandMode,
            scope: DemandScope,
        ) -> Vec<ExecutionDemand> {
            vec![ExecutionDemand {
                class,
                units,
                mode,
                scope,
            }]
        }
        fn vector_lanes(value: ValueType) -> u64 {
            match value {
                ValueType::Vector { lanes, .. } => u64::from(lanes),
                _ => panic!("CUDA execution demand received a scalar for a typed vector op"),
            }
        }
        fn read_service<G>(place: &ClosedPlace<G>) -> ServiceClassId {
            match (place.kind, place.realization) {
                (
                    ClosedPlaceKind::Local { .. },
                    Some(LocalRealization::NativeDynamic | LocalRealization::NativeStatic),
                ) => SERVICE_SHARED_READ,
                _ => SERVICE_GLOBAL_READ,
            }
        }
        fn write_service<G>(place: &ClosedPlace<G>) -> ServiceClassId {
            match (place.kind, place.realization) {
                (
                    ClosedPlaceKind::Local { .. },
                    Some(LocalRealization::NativeDynamic | LocalRealization::NativeStatic),
                ) => SERVICE_SHARED_WRITE,
                _ => SERVICE_GLOBAL_WRITE,
            }
        }

        let one = arena.nat(1);
        let participant = DemandScope::PerParticipant;
        match op {
            ClosedOpView::Constant { .. }
            | ClosedOpView::Binary { .. }
            | ClosedOpView::Unary { .. }
            | ClosedOpView::Bit { .. }
            | ClosedOpView::Fma { .. }
            | ClosedOpView::ApproximateMath { .. }
            | ClosedOpView::Cast { .. }
            | ClosedOpView::Bitcast { .. }
            | ClosedOpView::Cmp { .. }
            | ClosedOpView::Select { .. }
            | ClosedOpView::Logic { .. }
            | ClosedOpView::Not { .. }
            | ClosedOpView::Geometry { .. }
            | ClosedOpView::NatArg { .. }
            | ClosedOpView::ScalarArg { .. }
            | ClosedOpView::Extent { .. }
            | ClosedOpView::Branch { .. }
            | ClosedOpView::Repeat { .. } => demand(
                SERVICE_INSTRUCTION,
                one,
                DemandMode::SaturatedCapacity,
                participant,
            ),
            ClosedOpView::VectorSplat { out, .. }
            | ClosedOpView::VectorBinary { out, .. }
            | ClosedOpView::VectorUnary { out, .. }
            | ClosedOpView::VectorBit { out, .. }
            | ClosedOpView::VectorFma { out, .. }
            | ClosedOpView::VectorCast { out, .. } => {
                let units = arena.nat(vector_lanes(out.ty));
                demand(
                    SERVICE_INSTRUCTION,
                    units,
                    DemandMode::SaturatedCapacity,
                    participant,
                )
            }
            ClosedOpView::VectorLane { .. } => demand(
                SERVICE_INSTRUCTION,
                one,
                DemandMode::SaturatedCapacity,
                participant,
            ),
            ClosedOpView::VectorReduceAdd { vector, .. } => {
                let units = arena.nat(vector_lanes(vector.ty).saturating_sub(1));
                demand(
                    SERVICE_INSTRUCTION,
                    units,
                    DemandMode::DependencyLatency,
                    participant,
                )
            }
            ClosedOpView::Read { place, .. } => demand(
                read_service(&place),
                one,
                DemandMode::SaturatedCapacity,
                participant,
            ),
            ClosedOpView::ReadPlane { place, .. } => demand(
                read_service(&place),
                one,
                DemandMode::SaturatedCapacity,
                participant,
            ),
            ClosedOpView::VectorRead { out, place, .. } => {
                let units = arena.nat(vector_lanes(out.ty));
                demand(
                    read_service(&place),
                    units,
                    DemandMode::SaturatedCapacity,
                    participant,
                )
            }
            ClosedOpView::Write { place, .. } => demand(
                write_service(&place),
                one,
                DemandMode::SaturatedCapacity,
                participant,
            ),
            ClosedOpView::VectorWrite { place, value, .. } => {
                let units = arena.nat(vector_lanes(value.ty));
                demand(
                    write_service(&place),
                    units,
                    DemandMode::SaturatedCapacity,
                    participant,
                )
            }
            ClosedOpView::RepresentationConvertPacket { .. } => demand(
                SERVICE_REPACK,
                one,
                DemandMode::SaturatedCapacity,
                participant,
            ),
            ClosedOpView::Atomic { .. } => demand(
                SERVICE_ATOMIC,
                one,
                DemandMode::SaturatedCapacity,
                participant,
            ),
            ClosedOpView::StoreSlot { .. } => demand(
                SERVICE_GLOBAL_WRITE,
                one,
                DemandMode::SaturatedCapacity,
                participant,
            ),
            ClosedOpView::Barrier(_) => demand(
                SERVICE_BARRIER,
                one,
                DemandMode::DependencyLatency,
                participant,
            ),
            ClosedOpView::Intrinsic { op, .. } => match op {
                CudaIntrinsic::LaneIndex
                | CudaIntrinsic::Shuffle { .. }
                | CudaIntrinsic::SubgroupReduce { .. } => demand(
                    SERVICE_SUBGROUP,
                    one,
                    DemandMode::DependencyLatency,
                    participant,
                ),
                CudaIntrinsic::MatrixMatmul {
                    elem,
                    a,
                    destination,
                    ..
                }
                | CudaIntrinsic::MatrixMatmulAdd {
                    elem,
                    a,
                    destination,
                    ..
                } => {
                    let [rows, columns] = destination.logical_extents.as_slice() else {
                        panic!("closed CUDA matrix destination is not rank two")
                    };
                    let [_, inner] = a.logical_extents.as_slice() else {
                        panic!("closed CUDA matrix left operand is not rank two")
                    };
                    if *elem == seismic_lang::types::DType::F32 {
                        let units = arena.nat_product(&[*rows, *columns, *inner]);
                        return demand(
                            SERVICE_REPACK,
                            units,
                            DemandMode::SaturatedCapacity,
                            DemandScope::PerLaunch,
                        );
                    }
                    let m_tile = arena.nat(16);
                    let n_tile = arena.nat(8);
                    let k_tile = arena.nat(16);
                    let m = arena.nat_ceil_div(*rows, m_tile);
                    let n = arena.nat_ceil_div(*columns, n_tile);
                    let k = arena.nat_ceil_div(*inner, k_tile);
                    let units = arena.nat_product(&[m, n, k]);
                    demand(
                        SERVICE_MATRIX,
                        units,
                        DemandMode::SaturatedCapacity,
                        DemandScope::PerLaunch,
                    )
                }
                CudaIntrinsic::NvFp4Matmul { a, destination, .. }
                | CudaIntrinsic::NvFp4MatmulAdd { a, destination, .. } => {
                    let [rows, columns] = destination.logical_extents.as_slice() else {
                        panic!("closed CUDA NVFP4 destination is not rank two")
                    };
                    let [_, inner] = a.logical_extents.as_slice() else {
                        panic!("closed CUDA NVFP4 left operand is not rank two")
                    };
                    let m_tile = arena.nat(128);
                    let n_tile = arena.nat(8);
                    let k_tile = arena.nat(64);
                    let m = arena.nat_ceil_div(*rows, m_tile);
                    let n = arena.nat_ceil_div(*columns, n_tile);
                    let k = arena.nat_ceil_div(*inner, k_tile);
                    let units = arena.nat_product(&[m, n, k]);
                    demand(
                        SERVICE_MATRIX,
                        units,
                        DemandMode::SaturatedCapacity,
                        DemandScope::PerLaunch,
                    )
                }
            },
            ClosedOpView::Yield { .. } => Vec::new(),
        }
    }

    fn refine_execution_demand(
        target: &DeviceContract<Self>,
        arena: &mut seismic_lang::expr::ExprArena,
        launch: &seismic_compiler::schedule::Launch,
        locals: &seismic_compiler::storage::LaunchLocalLayout,
        native: &seismic_compiler::target::NativeKernel<Self>,
        mut demand: ExecutionDemand,
    ) -> Vec<ExecutionDemand> {
        if demand.mode != DemandMode::SaturatedCapacity {
            return vec![demand];
        }
        let facts = target.facts();
        let threads = arena.nat_product(&launch.workgroup);
        let resident_blocks = active_blocks_expression(
            arena,
            &native.handle().occupancy,
            threads,
            locals.workgroup_bytes,
        );
        let (nominal, resident) = if demand.class == SERVICE_MATRIX {
            let warp = arena.nat(u64::from(facts.warp_size));
            let warps = arena.nat_ceil_div(threads, warp);
            let nominal = arena.nat(u64::from(
                (facts.max_threads_per_multiprocessor / facts.warp_size).max(1),
            ));
            (nominal, arena.nat_mul(resident_blocks, warps))
        } else {
            let nominal = arena.nat(u64::from(facts.max_threads_per_multiprocessor));
            (nominal, arena.nat_mul(resident_blocks, threads))
        };
        let one = arena.nat(1);
        let resident = arena.nat_max(resident, one);
        let pressure = arena.nat_mul(demand.units, nominal);
        demand.units = arena.nat_ceil_div(pressure, resident);
        vec![demand]
    }

    fn intrinsic_numerics(
        _facts: &Self::Facts,
        signature: &seismic_lang::registry::IntrinsicSignature,
        _intrinsic: &Self::Intrinsic,
    ) -> seismic_compiler::target::IntrinsicNumericalSemantics {
        seismic_compiler::target::IntrinsicNumericalSemantics {
            arithmetic: signature.numerical.clone(),
            // Every emitted PTX operation uses the non-`.ftz` form. The
            // target advertises no implicit flush relaxation for these
            // concrete subgroup and dense-matrix instructions.
            flush_to_zero: false,
        }
    }

    fn intrinsic_addressable_resources(
        intrinsic: &Self::Intrinsic,
    ) -> Vec<seismic_compiler::kernel::ops::AddressableResourceHandle> {
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

    fn semantic_intrinsic_requirements(
        _target: &DeviceContract<Self>,
        arena: &mut seismic_lang::expr::ExprArena,
        signature: &seismic_lang::registry::IntrinsicSignature,
        _parallel_extent: seismic_lang::expr::NatExpr,
    ) -> seismic_compiler::kernel::ops::SemanticIntrinsicLaunchRequirements {
        if signature.capability != capability::capability_id(capability::SUBGROUP)
            && signature.capability != capability::capability_id(capability::MATRIX)
        {
            panic!(
                "CUDA static capability registry passed a foreign intrinsic to its requirements hook"
            )
        }
        seismic_compiler::kernel::ops::SemanticIntrinsicLaunchRequirements {
            required_mode: None,
            required_workgroup: matches!(signature.name, "nvfp4_matmul" | "nvfp4_matmul_add")
                .then(|| [arena.nat(128), arena.nat(1), arena.nat(1)]),
        }
    }

    fn lower_semantic_intrinsic(
        target: &DeviceContract<Self>,
        _domain: &seismic_compiler::kernel::ops::SegmentLaunchDomain,
        call: seismic_compiler::kernel::ops::SemanticIntrinsicCall<'_>,
        sink: &mut seismic_compiler::kernel::ops::SemanticIntrinsicSink<'_, '_, Self>,
    ) {
        use seismic_compiler::kernel::ops::{IntrinsicResources, SemanticIntrinsicOperand};
        use seismic_lang::intrinsics::ReduceOp;

        if !Self::emitted_intrinsics().contains(&call.signature.id) {
            panic!(
                "CUDA semantic lowering received an intrinsic absent from its native emitter set"
            )
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
                        Some(SemanticIntrinsicOperand::Scalar(value)) => value.dtype,
                        _ => panic!("registered CUDA shuffle has no scalar value operand"),
                    };
                    CudaIntrinsic::Shuffle { dtype }
                }
                "simd_sum" => {
                    let dtype = match call.operands.first() {
                        Some(SemanticIntrinsicOperand::Scalar(value)) => value.dtype,
                        _ => panic!("registered CUDA subgroup reduction has no scalar operand"),
                    };
                    CudaIntrinsic::SubgroupReduce {
                        op: ReduceOp::Sum,
                        dtype,
                    }
                }
                "simd_max" => {
                    let dtype = match call.operands.first() {
                        Some(SemanticIntrinsicOperand::Scalar(value)) => value.dtype,
                        _ => panic!("registered CUDA subgroup reduction has no scalar operand"),
                    };
                    CudaIntrinsic::SubgroupReduce {
                        op: ReduceOp::Max,
                        dtype,
                    }
                }
                "simd_min" => {
                    let dtype = match call.operands.first() {
                        Some(SemanticIntrinsicOperand::Scalar(value)) => value.dtype,
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
            let destination = call.destination.unwrap_or_else(|| {
                panic!("registered CUDA matrix intrinsic has no owned destination")
            });
            let a = sink.readable(call.operands[0].clone());
            let b = sink.readable(call.operands[1].clone());
            let elem = match &call.operands[0] {
                SemanticIntrinsicOperand::Readable(place) => {
                    seismic_lang::registry::representation_info(place.representation).decoded
                }
                _ => panic!("registered CUDA matrix intrinsic has no readable left operand"),
            };
            let operation = match call.signature.name {
                "matmul" => CudaIntrinsic::MatrixMatmul {
                    elem,
                    a,
                    b,
                    destination: sink
                        .writable(SemanticIntrinsicOperand::Writable(destination.clone())),
                },
                "matmul_add" => CudaIntrinsic::MatrixMatmulAdd {
                    elem,
                    a,
                    b,
                    c: sink.readable(call.operands[2].clone()),
                    destination: sink
                        .writable(SemanticIntrinsicOperand::Writable(destination.clone())),
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
                        seismic_compiler::target::ResourceLifetime::Operation,
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

    fn form_native_kernel_candidate(
        target: &DeviceContract<Self>,
        kernel: &seismic_compiler::kernel::Kernel<Self>,
        layout: &seismic_compiler::target::KernelEmissionLayout,
    ) -> Result<Self::RawNativeCandidate, NativeCompilationError> {
        compile::compile_kernel(target, kernel, layout)
    }

    fn reflect_native_kernel(
        target: &DeviceContract<Self>,
        kernel: &seismic_compiler::kernel::Kernel<Self>,
        _layout: &seismic_compiler::target::KernelEmissionLayout,
        candidate: Self::RawNativeCandidate,
    ) -> Result<NativeKernelReflection<Self>, NativeCompilationError> {
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
            occupancy,
        };
        Ok(NativeKernelReflection {
            handle,
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
            resources: NativeKernelResources {
                registers_per_participant: NativeResourceUsage::Exact(registers),
                spill_bytes_per_participant: NativeResourceUsage::Exact(local_bytes),
                static_local_bytes: NativeResourceUsage::Exact(static_shared),
            },
            numerics: CudaNumericalMode,
            numerical_identity: NativeNumericalModeIdentity {
                fingerprint: Sha256::digest(b"seismic-cuda-ptx-rn-no-ftz-v1").into(),
            },
            artifact: NativeArtifactMetrics {
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
            },
        })
    }
}
