//! Pure Metal demand transfer over the authoritative executable IR.
//! Device discovery and evidence acquisition belong to the native adapter.

pub mod characterization;

use seismic_estimator::*;
use seismic_ir::kernel::ops::{ClosedOpView, ClosedPlaceKind, ValueType};
use seismic_ir::metal::*;
use seismic_lang::expr::{ExprArena, NatExpr};
use seismic_lang::types::DType;
use std::collections::BTreeSet;

seismic_estimator::analytical_services! {
    pub enum MetalService {
        Control => "metal.control",
        Integer => "metal.integer",
        F32AddSub => "metal.f32.add-sub",
        F32Multiply => "metal.f32.multiply",
        F32Divide => "metal.f32.divide",
        F32Remainder => "metal.f32.remainder",
        F32MinMax => "metal.f32.min-max",
        F32Fma => "metal.f32.fma",
        F32Compare => "metal.f32.compare",
        F32ToInteger => "metal.f32.to-integer",
        IntegerToF32 => "metal.integer.to-f32",
        F32ToF16 => "metal.f32.to-f16",
        F16ToF32 => "metal.f16.to-f32",
        F32ToBf16 => "metal.f32.to-bf16",
        F16Strict => "metal.f16.strict",
        Bf16Strict => "metal.bf16.strict",
        ApproximateMath => "metal.approximate-math",
        GlobalMemory => "metal.global-memory",
        WorkgroupMemory => "metal.workgroup-memory",
        Representation => "metal.representation",
        Atomic => "metal.atomic",
        Barrier => "metal.barrier",
        Subgroup => "metal.subgroup",
        Matrix => "metal.simdgroup-matrix",
    }
}

fn capability_supported(
    supported_intrinsics: &BTreeSet<seismic_lang::ids::IntrinsicId>,
    capability_name: &str,
) -> bool {
    let supported = |capability_name: &str| {
        let capability = seismic_lang::registry::capability(
            seismic_lang::registry::BackendName::Metal,
            capability_name,
        )
        .expect("static Metal capability is absent");
        seismic_lang::registry::intrinsics(capability)
            .iter()
            .any(|signature| supported_intrinsics.contains(&signature.id))
    };
    supported(capability_name)
}

pub fn service_available(
    bfloat_arithmetic: bool,
    supported_intrinsics: &BTreeSet<seismic_lang::ids::IntrinsicId>,
    service: MetalService,
) -> bool {
    match service {
        MetalService::Bf16Strict | MetalService::F32ToBf16 => bfloat_arithmetic,
        MetalService::Subgroup => capability_supported(supported_intrinsics, "subgroup"),
        MetalService::Matrix => capability_supported(supported_intrinsics, "matrix"),
        _ => true,
    }
}

fn demand(class: MetalService, units: NatExpr, mode: DemandMode) -> ExecutionDemand<MetalService> {
    ExecutionDemand {
        class,
        units,
        mode,
        scope: DemandScope::PerParticipant,
    }
}

fn one(
    arena: &mut ExprArena,
    class: MetalService,
    mode: DemandMode,
) -> ExecutionDemand<MetalService> {
    demand(class, arena.nat(1), mode)
}

fn lanes(arena: &mut ExprArena, ty: ValueType) -> NatExpr {
    arena.nat(match ty {
        ValueType::Vector { lanes, .. } => u64::from(lanes),
        _ => 1,
    })
}

fn arithmetic_class(ty: ValueType) -> MetalService {
    use MetalService::{
        Bf16Strict as BF16_STRICT, Control as CONTROL, F16Strict as F16_STRICT, Integer as INTEGER,
    };
    match ty {
        ValueType::Scalar(DType::F32)
        | ValueType::Vector {
            dtype: DType::F32, ..
        } => panic!("F32 execution demand must use the renderer-selected emission family"),
        ValueType::Scalar(DType::F16)
        | ValueType::Vector {
            dtype: DType::F16, ..
        } => F16_STRICT,
        ValueType::Scalar(DType::BF16)
        | ValueType::Vector {
            dtype: DType::BF16, ..
        } => BF16_STRICT,
        ValueType::Scalar(DType::I32 | DType::U32 | DType::Bool)
        | ValueType::Vector {
            dtype: DType::I32 | DType::U32 | DType::Bool,
            ..
        }
        | ValueType::Index
        | ValueType::Bool => INTEGER,
        ValueType::Opaque { .. } => CONTROL,
    }
}

fn emission_class(family: ScalarEmissionFamily) -> MetalService {
    use MetalService::{
        Control as CONTROL, F16ToF32 as F16_TO_F32, F32AddSub as F32_ADD_SUB,
        F32Compare as F32_COMPARE, F32Divide as F32_DIVIDE, F32Fma as F32_FMA,
        F32MinMax as F32_MIN_MAX, F32Multiply as F32_MULTIPLY, F32Remainder as F32_REMAINDER,
        F32ToBf16 as F32_TO_BF16, F32ToF16 as F32_TO_F16, F32ToInteger as F32_TO_INTEGER,
        Integer as INTEGER, IntegerToF32 as INTEGER_TO_F32,
    };
    match family {
        ScalarEmissionFamily::F32AddSub => F32_ADD_SUB,
        ScalarEmissionFamily::F32Multiply => F32_MULTIPLY,
        ScalarEmissionFamily::F32Divide => F32_DIVIDE,
        ScalarEmissionFamily::F32Remainder => F32_REMAINDER,
        ScalarEmissionFamily::F32MinMax => F32_MIN_MAX,
        ScalarEmissionFamily::F32FusedMultiplyAdd => F32_FMA,
        ScalarEmissionFamily::F32Comparison => F32_COMPARE,
        ScalarEmissionFamily::F32ToF16 => F32_TO_F16,
        ScalarEmissionFamily::F16ToF32 => F16_TO_F32,
        ScalarEmissionFamily::F32ToBF16 => F32_TO_BF16,
        // The renderer's BF16 widen helper is one integer left shift between
        // representation-preserving casts. It is not a separately measurable
        // conversion service.
        ScalarEmissionFamily::BF16ToF32 => INTEGER,
        ScalarEmissionFamily::F32ToInteger => F32_TO_INTEGER,
        ScalarEmissionFamily::IntegerToF32 => INTEGER_TO_F32,
        ScalarEmissionFamily::NativeControl => CONTROL,
        ScalarEmissionFamily::NativeIntegerBit => INTEGER,
    }
}

fn memory_class(kind: ClosedPlaceKind) -> MetalService {
    use MetalService::{GlobalMemory as GLOBAL_MEMORY, WorkgroupMemory as WORKGROUP_MEMORY};
    match kind {
        ClosedPlaceKind::Global { .. } => GLOBAL_MEMORY,
        ClosedPlaceKind::Local { .. } => WORKGROUP_MEMORY,
    }
}

macro_rules! cost {
    ($first:expr $(, $rest:expr)* $(,)?) => {
        OperationCost::demands($first, vec![$($rest),*])
    };
}

/// Exhaustive execution semantics for the same closed operation view the MSL
/// renderer consumes. Unsupported vector operations still have semantics so
/// adding vector support cannot create a late hole; the empty profile vector
/// matrix prevents them from entering a Metal kernel today.
pub fn operation_cost<B: seismic_ir::target::PhysicalDialect<Intrinsic = MetalIntrinsic>>(
    arena: &mut ExprArena,
    kernel: &seismic_ir::kernel::Kernel<B>,
    op: ClosedOpView<'_, B>,
) -> Result<OperationCost<MetalService>, ModelLimitation> {
    use DemandMode::{DependencyLatency as Latency, SaturatedCapacity as Capacity};
    use MetalService::{
        ApproximateMath as APPROXIMATE_MATH, Atomic as ATOMIC, Barrier as BARRIER,
        Control as CONTROL, F16ToF32 as F16_TO_F32, F32AddSub as F32_ADD_SUB, F32Fma as F32_FMA,
        F32ToBf16 as F32_TO_BF16, F32ToF16 as F32_TO_F16, GlobalMemory as GLOBAL_MEMORY,
        Integer as INTEGER, Matrix as MATRIX, Representation as REPRESENTATION,
        Subgroup as SUBGROUP, WorkgroupMemory as WORKGROUP_MEMORY,
    };
    Ok(match op {
        ClosedOpView::Constant { .. } => cost![one(arena, CONTROL, Latency)],
        ClosedOpView::ScalarBits { .. } | ClosedOpView::ScalarFromBits { .. } => {
            cost![one(arena, INTEGER, Latency)]
        }
        ClosedOpView::Unary { out, .. } | ClosedOpView::Bitcast { out, .. } => {
            let class = if matches!(out.ty, ValueType::Scalar(DType::F32)) {
                INTEGER
            } else {
                arithmetic_class(out.ty)
            };
            cost![demand(class, lanes(arena, out.ty), Latency)]
        }
        ClosedOpView::Cast { out, a, to } => {
            let classes = match (a.ty, to) {
                (ValueType::Scalar(DType::F16), ValueType::Scalar(DType::BF16)) => {
                    vec![F16_TO_F32, F32_TO_BF16]
                }
                (ValueType::Scalar(DType::BF16), ValueType::Scalar(DType::F16)) => {
                    vec![INTEGER, F32_TO_F16]
                }
                (from, to) => scalar_conversion_emission_family(&from, &to)
                    .map(emission_class)
                    .into_iter()
                    .collect(),
            };
            match classes.as_slice() {
                [] => cost![demand(
                    arithmetic_class(out.ty),
                    lanes(arena, out.ty),
                    Latency,
                )],
                [first, rest @ ..] => OperationCost::demands(
                    demand(*first, lanes(arena, out.ty), Latency),
                    rest.iter()
                        .map(|class| demand(*class, lanes(arena, out.ty), Latency))
                        .collect(),
                ),
            }
        }
        ClosedOpView::Binary { op, out, .. } => {
            let class = scalar_binary_emission_family(op, &out.ty)
                .map(emission_class)
                .unwrap_or_else(|| arithmetic_class(out.ty));
            cost![demand(class, lanes(arena, out.ty), Latency)]
        }
        ClosedOpView::Bit { out, .. } => {
            let class = if matches!(out.ty, ValueType::Scalar(DType::F32)) {
                INTEGER
            } else {
                arithmetic_class(out.ty)
            };
            cost![demand(class, lanes(arena, out.ty), Latency)]
        }
        ClosedOpView::Fma { out, .. } => {
            let class = scalar_fma_emission_family(&out.ty)
                .map(emission_class)
                .unwrap_or_else(|| arithmetic_class(out.ty));
            cost![demand(class, lanes(arena, out.ty), Latency)]
        }
        ClosedOpView::VectorFromLanes { out, .. }
        | ClosedOpView::VectorSplat { out, .. }
        | ClosedOpView::VectorLane { out, .. } => {
            cost![demand(CONTROL, lanes(arena, out.ty), Latency)]
        }
        ClosedOpView::VectorUnary { out, .. }
        | ClosedOpView::VectorBit { out, .. }
        | ClosedOpView::VectorCast { out, .. } => {
            let class = if matches!(
                out.ty,
                ValueType::Vector {
                    dtype: DType::F32,
                    ..
                }
            ) {
                INTEGER
            } else {
                arithmetic_class(out.ty)
            };
            cost![demand(class, lanes(arena, out.ty), Latency)]
        }
        ClosedOpView::VectorReduceAdd { out, .. } => {
            let class = if matches!(
                out.ty,
                ValueType::Vector {
                    dtype: DType::F32,
                    ..
                }
            ) {
                F32_ADD_SUB
            } else {
                arithmetic_class(out.ty)
            };
            cost![demand(class, lanes(arena, out.ty), Latency)]
        }
        ClosedOpView::VectorBinary { op, out, .. } => {
            let class = if matches!(
                out.ty,
                ValueType::Vector {
                    dtype: DType::F32,
                    ..
                }
            ) {
                scalar_binary_emission_family(op, &ValueType::Scalar(DType::F32))
                    .map(emission_class)
                    .expect("F32 binary emission family is total")
            } else {
                arithmetic_class(out.ty)
            };
            cost![demand(class, lanes(arena, out.ty), Latency)]
        }
        ClosedOpView::VectorFma { out, .. } => {
            let class = if matches!(
                out.ty,
                ValueType::Vector {
                    dtype: DType::F32,
                    ..
                }
            ) {
                F32_FMA
            } else {
                arithmetic_class(out.ty)
            };
            cost![demand(class, lanes(arena, out.ty), Latency)]
        }
        ClosedOpView::ApproximateMath { out, .. } => {
            cost![demand(APPROXIMATE_MATH, lanes(arena, out.ty), Latency)]
        }
        ClosedOpView::Cmp { a, .. } => {
            let class = scalar_comparison_emission_family(&a.ty)
                .map(emission_class)
                .unwrap_or_else(|| arithmetic_class(a.ty));
            cost![demand(class, lanes(arena, a.ty), Latency)]
        }
        ClosedOpView::Select { .. } => cost![one(arena, CONTROL, Latency)],
        ClosedOpView::Logic { .. } | ClosedOpView::Not { .. } => {
            cost![one(arena, INTEGER, Latency)]
        }
        ClosedOpView::Geometry { .. }
        | ClosedOpView::NatArg { .. }
        | ClosedOpView::ScalarArg { .. }
        | ClosedOpView::Extent { .. }
        | ClosedOpView::StoreSlot { .. }
        | ClosedOpView::Branch { .. }
        | ClosedOpView::Repeat { .. }
        | ClosedOpView::Yield { .. } => cost![one(arena, CONTROL, Latency)],
        ClosedOpView::Read { place, .. } => cost![one(arena, memory_class(place.kind), Capacity)],
        ClosedOpView::ReadPlaneField {
            place, plane_info, ..
        } => {
            let mut demands = OperationDemands::one(one(arena, memory_class(place.kind), Capacity));
            if !matches!(
                plane_info.encoding,
                seismic_lang::registry::PlaneEncoding::Dense(_)
            ) {
                demands.push(one(arena, REPRESENTATION, Latency));
            }
            OperationCost::Demands(demands)
        }
        ClosedOpView::ReadPlane { place, .. } => cost![
            one(arena, memory_class(place.kind), Capacity),
            one(arena, REPRESENTATION, Latency),
        ],
        ClosedOpView::VectorRead { out, place, .. } => {
            cost![demand(
                memory_class(place.kind),
                lanes(arena, out.ty),
                Capacity
            )]
        }
        ClosedOpView::Write { place, .. } => {
            cost![one(arena, memory_class(place.kind), Capacity)]
        }
        ClosedOpView::VectorWrite { place, value, .. } => {
            cost![demand(
                memory_class(place.kind),
                lanes(arena, value.ty),
                Capacity,
            )]
        }
        ClosedOpView::RepresentationConvertPacket { destination, .. } => {
            let packet_elems = destination.geometry.layout.group;
            let units = arena.nat(u64::from(packet_elems));
            cost![
                demand(GLOBAL_MEMORY, units, Capacity),
                demand(REPRESENTATION, units, Latency),
            ]
        }
        ClosedOpView::Atomic { .. } => cost![one(arena, ATOMIC, Capacity)],
        ClosedOpView::Barrier(_) => cost![one(arena, BARRIER, Latency)],
        ClosedOpView::Intrinsic { op, .. } => match op {
            MetalIntrinsic::LaneIndex
            | MetalIntrinsic::Shuffle { .. }
            | MetalIntrinsic::SubgroupReduce { .. } => cost![one(arena, SUBGROUP, Latency)],
            MetalIntrinsic::Matrix {
                left,
                right,
                addend,
                ..
            } => {
                // The renderer assigns output tiles across threadgroups, but
                // the total work is invariant under that assignment. State
                // the cooperative instruction and staging traffic once for
                // the complete launch; core must not multiply it by the
                // reflected SIMD width or by the chosen grid.
                let exact = |value| kernel.exact_nat(value)
                    .ok_or(ModelLimitation::DeviceExtent { value });
                let rows = exact(left.extents[0])?;
                let inner = exact(left.extents[1])?;
                let columns = exact(right.extents[1])?;
                let eight = arena.nat(8);
                let row_tiles = arena.nat_ceil_div(rows, eight);
                let column_tiles = arena.nat_ceil_div(columns, eight);
                let tiles = arena.nat_mul(row_tiles, column_tiles);
                let k_blocks = arena.nat_ceil_div(inner, eight);
                let matrix_units = arena.nat_mul(tiles, k_blocks);
                let left_elements = arena.nat_mul(rows, inner);
                let left_reads = arena.nat_mul(column_tiles, left_elements);
                let right_elements = arena.nat_mul(inner, columns);
                let right_reads = arena.nat_mul(row_tiles, right_elements);
                let output_elements = arena.nat_mul(rows, columns);
                let terminal_multiplier = arena.nat(if addend.is_some() { 2 } else { 1 });
                let terminal_global = arena.nat_mul(output_elements, terminal_multiplier);
                let staged_global = arena.nat_add(left_reads, right_reads);
                let global_units = arena.nat_add(staged_global, terminal_global);
                let two_fifty_six = arena.nat(256);
                let staged_workgroup = arena.nat_mul(k_blocks, two_fifty_six);
                let workgroup_per_tile = arena.nat_add(staged_workgroup, two_fifty_six);
                let workgroup_units = arena.nat_mul(tiles, workgroup_per_tile);
                let two = arena.nat(2);
                let barriers_per_tile = arena.nat_add(k_blocks, two);
                let barriers = arena.nat_mul(tiles, barriers_per_tile);
                let launch = DemandScope::PerLaunch;
                let mut result = OperationDemands::with_rest(
                    ExecutionDemand {
                        class: MATRIX,
                        units: matrix_units,
                        mode: Capacity,
                        scope: launch,
                    },
                    vec![
                        ExecutionDemand {
                            class: GLOBAL_MEMORY,
                            units: global_units,
                            mode: Capacity,
                            scope: launch,
                        },
                        ExecutionDemand {
                            class: WORKGROUP_MEMORY,
                            units: workgroup_units,
                            mode: Capacity,
                            scope: launch,
                        },
                        ExecutionDemand {
                            class: BARRIER,
                            units: barriers,
                            mode: Latency,
                            scope: launch,
                        },
                    ],
                );
                if matches!(
                    &seismic_lang::registry::representation_info(right.representation).kind,
                    seismic_lang::registry::RepresentationKind::Packed(_)
                ) {
                    result.push(ExecutionDemand {
                        class: REPRESENTATION,
                        units: right_reads,
                        mode: Latency,
                        scope: launch,
                    });
                }
                OperationCost::Demands(result)
            }
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_ir::{
        construction::Construction,
        kernel::ops::{BinaryOp, ConstantValue},
        target::*,
    };
    #[derive(Debug)]
    struct Dialect;
    impl PhysicalDialect for Dialect {
        type LaunchDescriptor = ();
        fn ordinary_launch() -> Self::LaunchDescriptor {
            ()
        }

        const NAME: seismic_lang::registry::BackendName =
            seismic_lang::registry::BackendName::Metal;
        type Intrinsic = MetalIntrinsic;
        type Facts = ();
        fn write_intrinsic_identity(_: &MetalIntrinsic, _: &mut IntrinsicIdentityBuilder) {
            panic!("fixture has no intrinsics")
        }
        fn intrinsic_numerics(
            _: &(),
            _: &seismic_lang::registry::IntrinsicSignature,
            _: &MetalIntrinsic,
        ) -> IntrinsicNumericalSemantics {
            panic!("fixture has no intrinsics")
        }
        fn intrinsic_addressable_resources(
            _: &MetalIntrinsic,
        ) -> Vec<seismic_ir::kernel::ops::AddressableResourceHandle> {
            vec![]
        }
    }
    #[test]
    fn matrix_demand_uses_actual_value_extents_and_reports_device_extents() {
        use seismic_ir::kernel::ops::{LogicalTensorMap, PlaceRef};
        use seismic_ir::storage::LaunchLocalKind;
        use seismic_lang::{registry, expr::Assignment};
        let capability = registry::capability(registry::BackendName::Metal, "matrix").unwrap();
        let signature = registry::intrinsics(capability).iter().find(|s| s.name == "matmul").unwrap();
        let mut arena = ExprArena::default();
        let capacity = arena.nat(32);
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut builder = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        builder.local_tensor(
            LaunchLocalKind::Workgroup,
            registry::dense(DType::F32),
            vec![capacity, capacity],
        );
        let rows = builder.index_constant(9).raw();
        let inner = builder.index_constant(8).raw();
        let columns = builder.index_constant(16).raw();
        let device_rows = builder.local_id(0).raw();
        builder.close();
        let kernel = &construction.kernels()[0];
        let map = |extents| LogicalTensorMap {
            base: PlaceRef::Local { index: 0 },
            representation: registry::dense(DType::F32), extents, steps: vec![],
        };
        for (actual_rows, unavailable) in [(rows, false), (device_rows, true)] {
            let left = map(vec![actual_rows, inner]);
            let right = map(vec![inner, columns]);
            let into = map(vec![actual_rows, columns]);
            let operation = MetalIntrinsic::Matrix {
                scratch_left: left.clone(), scratch_right: right.clone(),
                scratch_accumulator: into.clone(), left, right, into,
                addend: None, element: DType::F32, accumulator: DType::F32, output: DType::F32,
            };
            let cost = operation_cost(&mut arena, kernel, ClosedOpView::Intrinsic {
                intrinsic: signature.id, signature, op: &operation,
                outputs: vec![], arguments: vec![], mapping_dependencies: vec![],
            });
            if unavailable {
                assert!(matches!(cost, Err(ModelLimitation::DeviceExtent { value }) if value == device_rows));
            } else {
                let OperationCost::Demands(cost) = cost.unwrap() else { panic!("matrix work cannot be elided") };
                let matrix = cost.iter().find(|d| d.class == MetalService::Matrix).unwrap();
                assert_eq!(arena.eval_nat(matrix.units, &Assignment::new()).unwrap(), 4);
            }
        }
    }

    #[test]
    fn bf16_widen_is_integer_service() {
        assert!(emission_class(ScalarEmissionFamily::BF16ToF32) == MetalService::Integer);
    }
    #[test]
    fn scalar_recipes_are_modeled_by_their_constructed_integer_operations() {
        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Rem,
            BinaryOp::Min,
            BinaryOp::Max,
        ] {
            let mut arena = ExprArena::default();
            let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
            let vectors = VectorSupport::default();
            let mut builder = construction.portable_kernel(&mut arena, &(), &[], &vectors);
            let a = builder.constant(ConstantValue::F32(2.0), ValueType::Scalar(DType::F32));
            let b = builder.constant(ConstantValue::F32(3.0), ValueType::Scalar(DType::F32));
            builder.binary(op, a, b);
            builder.close();
            let kernel = &construction.kernels()[0];
            let emission = KernelEmissionLayout {
                words: KernelWordLayout::for_kernel(kernel),
                bindings: vec![],
                locals: vec![],
                addressable_resources: vec![],
                scalar_args: vec![],
                result_types: vec![],
            };
            let mut demands = Vec::new();
            for op in kernel.blocks().iter().flat_map(|block| &block.ops) {
                match operation_cost(&mut arena, kernel, kernel.closed_op(op, &emission)).unwrap() {
                    OperationCost::Demands(operation) => demands.extend(operation.into_iter()),
                    OperationCost::Elided(_) => {}
                }
            }
            assert!(demands.iter().any(|d| d.class == MetalService::Integer));
            assert!(demands
                .iter()
                .all(|d| matches!(d.class, MetalService::Integer | MetalService::Control)));
            assert!(demands
                .iter()
                .all(|d| service_available(false, &BTreeSet::new(), d.class)));
        }
    }
}
