//! Metal executable vocabulary and emission classification shared by native
//! rendering and prediction. This module has no Metal API dependency.

use crate::kernel::ops::{BinaryOp, LogicalTensorMap, ValueType};
use seismic_lang::{intrinsics::ReduceOp, types::DType};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarEmissionFamily {
    F32AddSub,
    F32Multiply,
    F32Divide,
    F32Remainder,
    F32MinMax,
    F32FusedMultiplyAdd,
    F32Comparison,
    F32ToF16,
    F16ToF32,
    F32ToBF16,
    BF16ToF32,
    F32ToInteger,
    IntegerToF32,
    NativeControl,
    NativeIntegerBit,
}

pub fn scalar_binary_emission_family(op: BinaryOp, ty: &ValueType) -> Option<ScalarEmissionFamily> {
    if matches!(ty, ValueType::Scalar(DType::F32)) {
        return Some(match op {
            BinaryOp::Add | BinaryOp::Sub => ScalarEmissionFamily::F32AddSub,
            BinaryOp::Mul => ScalarEmissionFamily::F32Multiply,
            BinaryOp::Div => ScalarEmissionFamily::F32Divide,
            BinaryOp::Rem => ScalarEmissionFamily::F32Remainder,
            BinaryOp::Min | BinaryOp::Max => ScalarEmissionFamily::F32MinMax,
        });
    }
    scalar_native_emission_family(ty)
}

pub fn scalar_fma_emission_family(ty: &ValueType) -> Option<ScalarEmissionFamily> {
    matches!(ty, ValueType::Scalar(DType::F32)).then_some(ScalarEmissionFamily::F32FusedMultiplyAdd)
}

pub fn scalar_comparison_emission_family(ty: &ValueType) -> Option<ScalarEmissionFamily> {
    if matches!(ty, ValueType::Scalar(DType::F32)) {
        Some(ScalarEmissionFamily::F32Comparison)
    } else {
        scalar_native_emission_family(ty)
    }
}

pub fn scalar_conversion_emission_family(
    from: &ValueType,
    to: &ValueType,
) -> Option<ScalarEmissionFamily> {
    let integer = |ty: &ValueType| {
        matches!(
            ty,
            ValueType::Scalar(DType::I32 | DType::U32) | ValueType::Index
        )
    };
    match (from, to) {
        (ValueType::Scalar(DType::F32), ValueType::Scalar(DType::F16)) => {
            Some(ScalarEmissionFamily::F32ToF16)
        }
        (ValueType::Scalar(DType::F16), ValueType::Scalar(DType::F32)) => {
            Some(ScalarEmissionFamily::F16ToF32)
        }
        (ValueType::Scalar(DType::F32), ValueType::Scalar(DType::BF16)) => {
            Some(ScalarEmissionFamily::F32ToBF16)
        }
        (ValueType::Scalar(DType::BF16), ValueType::Scalar(DType::F32)) => {
            Some(ScalarEmissionFamily::BF16ToF32)
        }
        (ValueType::Scalar(DType::F32), destination) if integer(destination) => {
            Some(ScalarEmissionFamily::F32ToInteger)
        }
        (source, ValueType::Scalar(DType::F32)) if integer(source) => {
            Some(ScalarEmissionFamily::IntegerToF32)
        }
        _ if matches!(from, ValueType::Bool | ValueType::Scalar(DType::Bool))
            || matches!(to, ValueType::Bool | ValueType::Scalar(DType::Bool)) =>
        {
            Some(ScalarEmissionFamily::NativeControl)
        }
        _ if integer(from) && integer(to) => Some(ScalarEmissionFamily::NativeIntegerBit),
        _ => None,
    }
}

pub const fn scalar_control_emission_family() -> ScalarEmissionFamily {
    ScalarEmissionFamily::NativeControl
}

pub fn scalar_integer_bit_emission_family(ty: &ValueType) -> Option<ScalarEmissionFamily> {
    matches!(
        ty,
        ValueType::Scalar(DType::I32 | DType::U32) | ValueType::Index
    )
    .then_some(ScalarEmissionFamily::NativeIntegerBit)
}

fn scalar_native_emission_family(ty: &ValueType) -> Option<ScalarEmissionFamily> {
    if matches!(ty, ValueType::Bool | ValueType::Scalar(DType::Bool)) {
        Some(ScalarEmissionFamily::NativeControl)
    } else {
        scalar_integer_bit_emission_family(ty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_emission_descriptors_are_exact() {
        let f32_ty = ValueType::Scalar(DType::F32);
        for op in [BinaryOp::Add, BinaryOp::Sub] {
            assert_eq!(
                scalar_binary_emission_family(op, &f32_ty),
                Some(ScalarEmissionFamily::F32AddSub)
            );
        }
        for (op, family) in [
            (BinaryOp::Mul, ScalarEmissionFamily::F32Multiply),
            (BinaryOp::Div, ScalarEmissionFamily::F32Divide),
            (BinaryOp::Rem, ScalarEmissionFamily::F32Remainder),
            (BinaryOp::Min, ScalarEmissionFamily::F32MinMax),
            (BinaryOp::Max, ScalarEmissionFamily::F32MinMax),
        ] {
            assert_eq!(scalar_binary_emission_family(op, &f32_ty), Some(family));
        }
        assert_eq!(
            scalar_fma_emission_family(&f32_ty),
            Some(ScalarEmissionFamily::F32FusedMultiplyAdd)
        );
        assert_eq!(
            scalar_comparison_emission_family(&f32_ty),
            Some(ScalarEmissionFamily::F32Comparison)
        );

        let f16_ty = ValueType::Scalar(DType::F16);
        let bf16_ty = ValueType::Scalar(DType::BF16);
        let i32_ty = ValueType::Scalar(DType::I32);
        let u32_ty = ValueType::Scalar(DType::U32);
        for (from, to, family) in [
            (&f32_ty, &f16_ty, ScalarEmissionFamily::F32ToF16),
            (&f16_ty, &f32_ty, ScalarEmissionFamily::F16ToF32),
            (&f32_ty, &bf16_ty, ScalarEmissionFamily::F32ToBF16),
            (&bf16_ty, &f32_ty, ScalarEmissionFamily::BF16ToF32),
            (&f32_ty, &i32_ty, ScalarEmissionFamily::F32ToInteger),
            (&u32_ty, &f32_ty, ScalarEmissionFamily::IntegerToF32),
        ] {
            assert_eq!(scalar_conversion_emission_family(from, to), Some(family));
        }
        assert_eq!(
            scalar_binary_emission_family(BinaryOp::Add, &i32_ty),
            Some(ScalarEmissionFamily::NativeIntegerBit)
        );
        assert_eq!(
            scalar_integer_bit_emission_family(&u32_ty),
            Some(ScalarEmissionFamily::NativeIntegerBit)
        );
        assert_eq!(
            scalar_control_emission_family(),
            ScalarEmissionFamily::NativeControl
        );
        assert_eq!(
            scalar_comparison_emission_family(&ValueType::Bool),
            Some(ScalarEmissionFamily::NativeControl)
        );
        assert_eq!(scalar_fma_emission_family(&f16_ty), None);
    }
}
