//! The closed Metal intrinsic vocabulary and capability registrations.
//!
//! Authored backend bodies are lowered exclusively by the core semantic
//! walker through `lower_semantic`; there is no second backend-local typed
//! lowering path.

use crate::facts::MetalFacts;
use crate::Metal;
use seismic_compiler::kernel::ops::{IntrinsicResources, LogicalTensorMap};
use seismic_compiler::target::{CapabilityRegistration, DataTypeSupport, TargetLimits};
use seismic_lang::expr::{ExprArena, NatExpr};
use seismic_lang::ids::{CapabilityId, IntrinsicId};
use seismic_lang::intrinsics::ReduceOp;
use seismic_lang::registry::{
    self, BackendName, IntrinsicResultType, IntrinsicSignature, OperandCategory,
};
use seismic_lang::types::DType;
use std::collections::BTreeSet;

pub(crate) fn numerical_semantics(
    signature: &IntrinsicSignature,
    intrinsic: &MetalIntrinsic,
) -> seismic_compiler::target::IntrinsicNumericalSemantics {
    seismic_compiler::target::IntrinsicNumericalSemantics {
        arithmetic: signature.numerical.clone(),
        // Native subgroup matrix arithmetic is allowed to flush subnormal
        // accumulator/intermediate values by Metal. Shuffle/lane queries do
        // no arithmetic, and min/max preserve their selected operand bits.
        // Native subgroup sum can exercise the same target float mode.
        flush_to_zero: matches!(
            intrinsic,
            MetalIntrinsic::Matrix { .. }
                | MetalIntrinsic::SubgroupReduce {
                    op: ReduceOp::Sum,
                    dtype: DType::F32 | DType::F16 | DType::BF16,
                }
        ),
    }
}

pub(crate) fn lower_semantic(
    _target: &seismic_compiler::target::DeviceContract<Metal>,
    _domain: &seismic_compiler::kernel::ops::SegmentLaunchDomain,
    call: seismic_compiler::kernel::ops::SemanticIntrinsicCall<'_>,
    sink: &mut seismic_compiler::kernel::ops::SemanticIntrinsicSink<'_, '_, Metal>,
) {
    use seismic_compiler::kernel::ops::SemanticIntrinsicOperand as Operand;
    match call.signature.name {
        "lane_index" => {
            sink.emit(MetalIntrinsic::LaneIndex, IntrinsicResources {
                requires_subgroup: true, ..IntrinsicResources::default()
            }, call.destination.clone());
        }
        "shuffle" => {
            let dtype = collective_dtype(call.signature).expect("registered shuffle has a scalar dtype");
            sink.emit(MetalIntrinsic::Shuffle { dtype }, subgroup_resources(), call.destination.clone());
        }
        "simd_sum" | "simd_max" | "simd_min" => {
            let dtype = collective_dtype(call.signature).expect("registered collective has a scalar dtype");
            let op = match call.signature.name {
                "simd_sum" => ReduceOp::Sum,
                "simd_max" => ReduceOp::Max,
                "simd_min" => ReduceOp::Min,
                _ => unreachable!(),
            };
            sink.emit(MetalIntrinsic::SubgroupReduce { op, dtype }, subgroup_resources(), call.destination.clone());
        }
        "matmul" | "matmul_add" => {
            let left = sink.readable(call.operands[0].clone());
            let right = sink.readable(call.operands[1].clone());
            let addend = (call.signature.name == "matmul_add")
                .then(|| sink.readable(call.operands[2].clone()));
            let destination = call
                .destination
                .clone()
                .expect("registered matrix signature has an owned destination");
            let into = sink.writable(Operand::Writable(destination.clone()));
            let element = matrix_operand_dtype(&call.signature.arguments[0].category)
                .expect("registered Metal matrix operand is dense");
            let output = matrix_result_dtype(call.signature)
                .expect("registered Metal matrix result is dense");
            let left_representation = registry::dense(element);
            let right_representation = registry::dense(element);
            let output_representation = match call.signature.result {
                IntrinsicResultType::Owned { representation, .. } => representation,
                _ => unreachable!("registered Metal matrix result is owned"),
            };
            let eight = sink.nat(8);
            let scratch_left = sink.workgroup_tensor(left_representation, vec![eight, eight]);
            let scratch_right = sink.workgroup_tensor(right_representation, vec![eight, eight]);
            let scratch_accumulator = sink.workgroup_tensor(output_representation, vec![eight, eight]);
            sink.emit(
                MetalIntrinsic::Matrix {
                    left,
                    right,
                    addend,
                    into,
                    element,
                    accumulator: output,
                    output,
                    scratch_left,
                    scratch_right,
                    scratch_accumulator,
                },
                IntrinsicResources {
                    requires_subgroup: true,
                    ..IntrinsicResources::default()
                },
                Some(destination),
            );
        }
        name => panic!("Metal registry advertised intrinsic `{name}` without an exhaustive semantic dispatcher arm"),
    }
}

pub(crate) fn semantic_requirements(
    _target: &seismic_compiler::target::DeviceContract<Metal>,
    _arena: &mut ExprArena,
    _signature: &IntrinsicSignature,
    _parallel_extent: NatExpr,
) -> seismic_compiler::kernel::ops::SemanticIntrinsicLaunchRequirements {
    seismic_compiler::kernel::ops::SemanticIntrinsicLaunchRequirements {
        required_mode: None,
        required_workgroup: None,
    }
}

pub(crate) fn write_identity(
    intrinsic: &MetalIntrinsic,
    identity: &mut seismic_compiler::target::IntrinsicIdentityBuilder,
) {
    match intrinsic {
        MetalIntrinsic::LaneIndex => identity.variant("lane-index"),
        MetalIntrinsic::Shuffle { dtype } => {
            identity.variant("shuffle");
            identity.dtype(*dtype);
        }
        MetalIntrinsic::SubgroupReduce { op, dtype } => {
            identity.variant("subgroup-reduce");
            identity.u32(match op {
                ReduceOp::Sum => 0,
                ReduceOp::Max => 1,
                ReduceOp::Min => 2,
                ReduceOp::Argmax => 3,
            });
            identity.dtype(*dtype);
        }
        MetalIntrinsic::Matrix {
            left,
            right,
            addend,
            into,
            element,
            accumulator,
            output,
            scratch_left,
            scratch_right,
            scratch_accumulator,
        } => {
            identity.variant("simdgroup-matrix");
            identity.logical_tensor(left);
            identity.logical_tensor(right);
            identity.bool(addend.is_some());
            if let Some(addend) = addend {
                identity.logical_tensor(addend);
            }
            identity.logical_tensor(into);
            identity.dtype(*element);
            identity.dtype(*accumulator);
            identity.dtype(*output);
            identity.logical_tensor(scratch_left);
            identity.logical_tensor(scratch_right);
            identity.logical_tensor(scratch_accumulator);
        }
    }
}

pub(crate) fn emitted_intrinsics() -> BTreeSet<IntrinsicId> {
    registrations()
        .into_iter()
        .flat_map(|registration| registration.implemented)
        .collect()
}

/// The closed Metal intrinsic op set inside `Op::Intrinsic { op, outs, args }`.
/// Scalar operands and results travel as the op's erased `args`/`outs`;
/// matrix operands are places and travel inside the op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalIntrinsic {
    /// `metal.subgroup.lane_index`: `outs[0] := thread_index_in_simdgroup`.
    LaneIndex,
    /// `metal.subgroup.shuffle`: `outs[0] := simd_shuffle(args[0], args[1])`.
    Shuffle { dtype: DType },
    /// `metal.subgroup.simd_{sum,max,min}`: `outs[0] := collective(args[0])`.
    SubgroupReduce { op: ReduceOp, dtype: DType },
    /// A genuine native `simdgroup_multiply_accumulate` implementation.
    Matrix {
        left: LogicalTensorMap,
        right: LogicalTensorMap,
        addend: Option<LogicalTensorMap>,
        into: LogicalTensorMap,
        element: DType,
        accumulator: DType,
        output: DType,
        scratch_left: LogicalTensorMap,
        scratch_right: LogicalTensorMap,
        scratch_accumulator: LogicalTensorMap,
    },
}

// ---------------------------------------------------------------------------
// Registry lookups
// ---------------------------------------------------------------------------

/// The `metal.<name>` capability id. Absence is an inconsistent static
/// registry (§13.3.1).
fn capability(name: &str) -> CapabilityId {
    registry::capability(BackendName::Metal, name)
        .unwrap_or_else(|| panic!("static registry has no `metal.{name}` capability"))
}

fn scalar_argument(signature: &IntrinsicSignature, index: usize, dtype: DType) -> bool {
    signature
        .arguments
        .get(index)
        .map(|argument| argument.category == OperandCategory::Scalar(dtype))
        == Some(true)
}

const SUBGROUP_NAMES: [&str; 5] = ["lane_index", "shuffle", "simd_sum", "simd_max", "simd_min"];
const MATRIX_NAMES: [&str; 2] = ["matmul", "matmul_add"];
const COLLECTIVE_DTYPES: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

/// The scalar dtype of a subgroup signature (`None` for `lane_index`).
fn collective_dtype(signature: &IntrinsicSignature) -> Option<DType> {
    signature
        .arguments
        .first()
        .and_then(|argument| match argument.category {
            OperandCategory::Scalar(dtype) => Some(dtype),
            _ => None,
        })
}

fn matrix_operand_dtype(category: &OperandCategory) -> Option<DType> {
    match category {
        OperandCategory::Readable {
            representation,
            rank: 2,
        } => {
            let info = registry::representation_info(*representation);
            match info.kind {
                registry::RepresentationKind::Dense(dtype) => Some(dtype),
                registry::RepresentationKind::Packed(_) => Some(info.decoded),
                registry::RepresentationKind::External(_) => None,
            }
        }
        _ => None,
    }
}

fn matrix_result_dtype(signature: &IntrinsicSignature) -> Option<DType> {
    match signature.result {
        IntrinsicResultType::Owned {
            representation,
            rank: 2,
        } => match registry::representation_info(representation).kind {
            registry::RepresentationKind::Dense(dtype) => Some(dtype),
            registry::RepresentationKind::Packed(_) | registry::RepresentationKind::External(_) => {
                None
            }
        },
        _ => None,
    }
}

fn matrix_implemented(signature: &IntrinsicSignature) -> bool {
    if !MATRIX_NAMES.contains(&signature.name) {
        return false;
    }
    let Some(left) = signature
        .arguments
        .first()
        .and_then(|a| matrix_operand_dtype(&a.category))
    else {
        return false;
    };
    let Some(right) = signature
        .arguments
        .get(1)
        .and_then(|a| matrix_operand_dtype(&a.category))
    else {
        return false;
    };
    let Some(output) = matrix_result_dtype(signature) else {
        return false;
    };
    left == right
        && match signature.name {
            "matmul" => signature.arguments.len() == 2 && output == DType::F32,
            "matmul_add" => {
                signature.arguments.len() == 3
                    && signature
                        .arguments
                        .get(2)
                        .and_then(|a| matrix_operand_dtype(&a.category))
                        == Some(output)
            }
            _ => false,
        }
}

fn subgroup_implemented(signature: &IntrinsicSignature) -> bool {
    match signature.name {
        "lane_index" => signature.arguments.is_empty(),
        "shuffle" => {
            signature.arguments.len() == 2
                && collective_dtype(signature)
                    .is_some_and(|dtype| COLLECTIVE_DTYPES.contains(&dtype))
                && scalar_argument(signature, 1, DType::I32)
        }
        "simd_sum" | "simd_max" | "simd_min" => {
            signature.arguments.len() == 1
                && collective_dtype(signature)
                    .is_some_and(|dtype| COLLECTIVE_DTYPES.contains(&dtype))
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Registrations
// ---------------------------------------------------------------------------

/// The Metal capability registrations. Matrix rows are admitted only when
/// the device probe compiled the exact element/accumulator combination used
/// by the native simdgroup-matrix renderer.
pub fn registrations() -> Vec<CapabilityRegistration<Metal>> {
    let subgroup = capability("subgroup");
    let matrix = capability("matrix");
    vec![
        CapabilityRegistration {
            capability: subgroup,
            implemented: registry::intrinsics(subgroup)
                .iter()
                .filter(|s| SUBGROUP_NAMES.contains(&s.name) && subgroup_implemented(s))
                .map(|s| s.id)
                .collect(),
            supported: subgroup_supported,
        },
        CapabilityRegistration {
            capability: matrix,
            implemented: registry::intrinsics(matrix)
                .iter()
                .filter(|s| matrix_implemented(s))
                .map(|s| s.id)
                .collect(),
            supported: matrix_supported,
        },
    ]
}

pub(crate) fn matrix_supported(
    facts: &MetalFacts,
    _limits: &TargetLimits,
    dtypes: &DataTypeSupport,
) -> BTreeSet<IntrinsicId> {
    registry::intrinsics(capability("matrix"))
        .iter()
        .filter_map(|signature| {
            let left = signature
                .arguments
                .first()
                .and_then(|argument| matrix_operand_dtype(&argument.category))?;
            let right = signature
                .arguments
                .get(1)
                .and_then(|argument| matrix_operand_dtype(&argument.category))?;
            let accumulator = matrix_result_dtype(signature)?;
            let representations_supported =
                signature
                    .arguments
                    .iter()
                    .all(|argument| match argument.category {
                        OperandCategory::Readable { representation, .. }
                        | OperandCategory::Writable { representation, .. } => {
                            dtypes.representations.contains(&representation)
                        }
                        OperandCategory::Scalar(dtype) | OperandCategory::Constant(dtype) => {
                            dtypes.scalars.contains(&dtype)
                        }
                        OperandCategory::Opaque { .. } => false,
                    })
                    && match signature.result {
                        IntrinsicResultType::Owned { representation, .. } => {
                            dtypes.representations.contains(&representation)
                        }
                        IntrinsicResultType::Scalar(dtype) => dtypes.scalars.contains(&dtype),
                        IntrinsicResultType::Void => true,
                        IntrinsicResultType::Opaque { .. } => false,
                    };
            (matrix_implemented(signature)
                && representations_supported
                && facts
                    .matrix_combinations
                    .contains(&crate::facts::MatrixCombination {
                        accumulator,
                        left,
                        right,
                    }))
            .then_some(signature)
        })
        .map(|signature| signature.id)
        .collect()
}

/// A subgroup signature is supported when the target has SIMD groups and
/// the compiler accepted its dtype in the collective probe.
pub(crate) fn subgroup_supported(
    facts: &MetalFacts,
    _limits: &TargetLimits,
    _dtypes: &DataTypeSupport,
) -> BTreeSet<IntrinsicId> {
    registry::intrinsics(capability("subgroup"))
        .iter()
        .filter(|signature| {
            SUBGROUP_NAMES.contains(&signature.name) && subgroup_implemented(signature)
        })
        .filter(|signature| match collective_dtype(signature) {
            None => true,
            Some(dtype) => facts.scalar_collective_dtypes.contains(&dtype),
        })
        .map(|signature| signature.id)
        .collect()
}

fn subgroup_resources() -> IntrinsicResources {
    IntrinsicResources {
        requires_subgroup: true,
        ..IntrinsicResources::default()
    }
}
