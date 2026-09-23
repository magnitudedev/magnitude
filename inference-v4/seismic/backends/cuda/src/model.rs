use crate::*;
use seismic_estimator::*;
use std::collections::BTreeSet;

pub struct CudaAnalyticalModel;

seismic_estimator::analytical_services! {
    pub enum CudaService {
        Instruction => "cuda.scalar-instruction",
        GlobalRead => "cuda.global-read",
        GlobalWrite => "cuda.global-write",
        SharedRead => "cuda.shared-read",
        SharedWrite => "cuda.shared-write",
        Atomic => "cuda.atomic",
        Barrier => "cuda.barrier",
        Subgroup => "cuda.subgroup",
        Matrix => "cuda.matrix-mma",
        Repack => "cuda.representation-repack",
    }
}

impl AnalyticalModelDefinition<Cuda> for CudaAnalyticalModel {
    type Service = CudaService;

    fn model_revision(&self) -> &'static str {
        "seismic-cuda-analytical-model-v2"
    }

    fn operation_cost(
        &self,
        _facts: &<Cuda as seismic_ir::target::PhysicalDialect>::Facts,
        _supported_intrinsics: &BTreeSet<seismic_lang::ids::IntrinsicId>,
        arena: &mut seismic_lang::expr::ExprArena,
        kernel: &seismic_ir::kernel::Kernel<Cuda>,
        _emission: &seismic_ir::target::KernelEmissionLayout,
        _launch: &seismic_ir::schedule::Launch<Cuda>,
        _locals: &seismic_ir::storage::LaunchLocalLayout,
        op: seismic_ir::kernel::ops::ClosedOpView<'_, Cuda>,
    ) -> Result<OperationCost<Self::Service>, ModelLimitation> {
        use seismic_ir::kernel::ops::{ClosedOpView, ClosedPlace, ClosedPlaceKind, ValueType};
        use seismic_ir::target::LocalRealization;

        fn demand(
            class: CudaService,
            units: seismic_lang::expr::NatExpr,
            mode: DemandMode,
            scope: DemandScope,
        ) -> OperationCost<CudaService> {
            OperationCost::one(ExecutionDemand {
                class,
                units,
                mode,
                scope,
            })
        }
        fn vector_lanes(value: ValueType) -> u64 {
            match value {
                ValueType::Vector { lanes, .. } => u64::from(lanes),
                _ => panic!("CUDA execution demand received a scalar for a typed vector op"),
            }
        }
        fn read_service<G>(place: &ClosedPlace<G>) -> CudaService {
            match (place.kind, place.realization) {
                (
                    ClosedPlaceKind::Local { .. },
                    Some(LocalRealization::NativeDynamic | LocalRealization::NativeStatic),
                ) => CudaService::SharedRead,
                _ => CudaService::GlobalRead,
            }
        }
        fn write_service<G>(place: &ClosedPlace<G>) -> CudaService {
            match (place.kind, place.realization) {
                (
                    ClosedPlaceKind::Local { .. },
                    Some(LocalRealization::NativeDynamic | LocalRealization::NativeStatic),
                ) => CudaService::SharedWrite,
                _ => CudaService::GlobalWrite,
            }
        }

        let one = arena.nat(1);
        let participant = DemandScope::PerParticipant;
        use CudaService::{
            Atomic as SERVICE_ATOMIC, Barrier as SERVICE_BARRIER,
            GlobalWrite as SERVICE_GLOBAL_WRITE, Instruction as SERVICE_INSTRUCTION,
            Matrix as SERVICE_MATRIX, Repack as SERVICE_REPACK, Subgroup as SERVICE_SUBGROUP,
        };
        let cost = match op {
            ClosedOpView::Constant { .. }
            | ClosedOpView::Binary { .. }
            | ClosedOpView::Unary { .. }
            | ClosedOpView::Bit { .. }
            | ClosedOpView::Fma { .. }
            | ClosedOpView::ApproximateMath { .. }
            | ClosedOpView::Cast { .. }
            | ClosedOpView::Bitcast { .. }
            | ClosedOpView::ScalarBits { .. }
            | ClosedOpView::ScalarFromBits { .. }
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
            ClosedOpView::VectorFromLanes { out, .. }
            | ClosedOpView::VectorSplat { out, .. }
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
            ClosedOpView::ReadPlaneField { place, .. } | ClosedOpView::ReadPlane { place, .. } => {
                demand(
                    read_service(&place),
                    one,
                    DemandMode::SaturatedCapacity,
                    participant,
                )
            }
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
                    let [rows, columns] = destination.extents.as_slice() else {
                        panic!("closed CUDA matrix destination is not rank two")
                    };
                    let [_, inner] = a.extents.as_slice() else {
                        panic!("closed CUDA matrix left operand is not rank two")
                    };
                    let exact = |value| kernel.exact_nat(value)
                        .ok_or(ModelLimitation::DeviceExtent { value });
                    let (rows, columns, inner) = (exact(*rows)?, exact(*columns)?, exact(*inner)?);
                    if *elem == seismic_lang::types::DType::F32 {
                        let units = arena.nat_product(&[rows, columns, inner]);
                        demand(
                            SERVICE_REPACK,
                            units,
                            DemandMode::SaturatedCapacity,
                            DemandScope::PerLaunch,
                        )
                    } else {
                        let m_tile = arena.nat(16);
                        let n_tile = arena.nat(8);
                        let k_tile = arena.nat(16);
                        let m = arena.nat_ceil_div(rows, m_tile);
                        let n = arena.nat_ceil_div(columns, n_tile);
                        let k = arena.nat_ceil_div(inner, k_tile);
                        let units = arena.nat_product(&[m, n, k]);
                        demand(
                            SERVICE_MATRIX,
                            units,
                            DemandMode::SaturatedCapacity,
                            DemandScope::PerLaunch,
                        )
                    }
                }
                CudaIntrinsic::NvFp4Matmul { a, destination, .. }
                | CudaIntrinsic::NvFp4MatmulAdd { a, destination, .. } => {
                    let [rows, columns] = destination.extents.as_slice() else {
                        panic!("closed CUDA NVFP4 destination is not rank two")
                    };
                    let [_, inner] = a.extents.as_slice() else {
                        panic!("closed CUDA NVFP4 left operand is not rank two")
                    };
                    let exact = |value| kernel.exact_nat(value)
                        .ok_or(ModelLimitation::DeviceExtent { value });
                    let (rows, columns, inner) = (exact(*rows)?, exact(*columns)?, exact(*inner)?);
                    let m_tile = arena.nat(128);
                    let n_tile = arena.nat(8);
                    let k_tile = arena.nat(64);
                    let m = arena.nat_ceil_div(rows, m_tile);
                    let n = arena.nat_ceil_div(columns, n_tile);
                    let k = arena.nat_ceil_div(inner, k_tile);
                    let units = arena.nat_product(&[m, n, k]);
                    demand(
                        SERVICE_MATRIX,
                        units,
                        DemandMode::SaturatedCapacity,
                        DemandScope::PerLaunch,
                    )
                }
            },
            ClosedOpView::Yield { .. } => OperationCost::Elided(ProvenElision::CompileTimeOnly),
        };
        Ok(cost)
    }
}
