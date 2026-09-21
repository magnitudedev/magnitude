//! Type-level scalar types and representations for the typed kernel IR
//! (spec §7.1).
//!
//! Every registry representation has one marker type. Kernel handles are
//! parameterized by these markers so a mismatch between a place's
//! representation and an access is a compile error in the factory, not a
//! runtime match. Runtime `RepresentationId` values are dispatched into the
//! typed world exactly once, through [`with_representation`].

use seismic_lang::ids::RepresentationId;
use seismic_lang::registry;
use seismic_lang::types::DType;
use std::fmt;

/// A scalar type admitted in kernel SSA.
pub trait ScalarType:
    'static + Copy + fmt::Debug + Send + Sync + sealed::Sealed + sealed::KernelScalar
{
    const DTYPE: DType;
    const SYMBOL_SORT: seismic_lang::expr::SymbolSort;
    /// Host value used for constants.
    type Value: Copy + fmt::Debug + Send + Sync;
}

/// Scalar types that admit atomic update.
pub trait AtomicType: ScalarType {}

/// Floating scalar types.
pub trait FloatType: ScalarType {}

/// Integer scalar types.
pub trait IntegerType: ScalarType {}
pub trait NumericType: ScalarType {}
pub trait SignedType: NumericType {}
/// Scalar storage elements that may occupy fixed-width kernel vectors.
/// `Idx` is deliberately excluded: it is an address-domain value, not a
/// registered scalar representation.
pub trait VectorElement: ScalarType {}

/// The backend-native index type (unsigned, at least 32 bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Idx {}

pub(crate) mod sealed {
    use crate::kernel::ops::{ConstantValue, ValueType};

    pub trait Sealed {}

    /// Crate-private kernel-side facts of one scalar marker: the erased
    /// value type of its SSA values and the conversion of a host constant.
    /// Unnameable outside the crate, so the public `ScalarType` surface is
    /// unchanged for implementors and callers.
    pub trait KernelScalar {
        const VALUE_TYPE: ValueType;
        fn kernel_constant(value: <Self as super::ScalarType>::Value) -> ConstantValue
        where
            Self: super::ScalarType;
    }
}

macro_rules! scalar {
    ($name:ident, $dtype:expr, $value:ty, $value_type:expr, $constant:expr $(, $extra:ident)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {}
        impl sealed::Sealed for $name {}
        impl sealed::KernelScalar for $name {
            const VALUE_TYPE: crate::kernel::ops::ValueType = $value_type;
            fn kernel_constant(value: $value) -> crate::kernel::ops::ConstantValue {
                $constant(value)
            }
        }
        impl ScalarType for $name {
            const DTYPE: DType = $dtype;
            const SYMBOL_SORT: seismic_lang::expr::SymbolSort = seismic_lang::expr::SymbolSort::Scalar($dtype);
            type Value = $value;
        }
        $(impl $extra for $name {})*
    };
}

use crate::kernel::ops::{ConstantValue, ValueType};

scalar!(
    F32,
    DType::F32,
    f32,
    ValueType::Scalar(DType::F32),
    ConstantValue::F32,
    FloatType,
    AtomicType
);
scalar!(
    F16,
    DType::F16,
    f32,
    ValueType::Scalar(DType::F16),
    ConstantValue::F32,
    FloatType,
    AtomicType
);
scalar!(
    BF16,
    DType::BF16,
    f32,
    ValueType::Scalar(DType::BF16),
    ConstantValue::F32,
    FloatType,
    AtomicType
);
scalar!(
    I32,
    DType::I32,
    i32,
    ValueType::Scalar(DType::I32),
    ConstantValue::I32,
    IntegerType,
    AtomicType
);
scalar!(
    U32,
    DType::U32,
    u32,
    ValueType::Scalar(DType::U32),
    ConstantValue::U32,
    IntegerType,
    AtomicType
);
scalar!(
    Bool,
    DType::Bool,
    bool,
    ValueType::Bool,
    ConstantValue::Bool
);
impl NumericType for F32 {}
impl NumericType for F16 {}
impl NumericType for BF16 {}
impl NumericType for I32 {}
impl NumericType for U32 {}
impl SignedType for F32 {}
impl SignedType for F16 {}
impl SignedType for BF16 {}
impl SignedType for I32 {}
impl VectorElement for F32 {}
impl VectorElement for F16 {}
impl VectorElement for BF16 {}
impl VectorElement for I32 {}
impl VectorElement for U32 {}
impl VectorElement for Bool {}

impl sealed::Sealed for Idx {}
impl sealed::KernelScalar for Idx {
    const VALUE_TYPE: ValueType = ValueType::Index;
    fn kernel_constant(value: u64) -> ConstantValue {
        ConstantValue::Index(value)
    }
}
impl ScalarType for Idx {
    const DTYPE: DType = DType::U32;
    const SYMBOL_SORT: seismic_lang::expr::SymbolSort = seismic_lang::expr::SymbolSort::Nat;
    type Value = u64;
}
impl IntegerType for Idx {}
impl NumericType for Idx {}

/// The erased value type of one scalar marker.
pub(crate) fn value_type_of<T: ScalarType>() -> ValueType {
    <T as sealed::KernelScalar>::VALUE_TYPE
}

/// The kernel constant of one host value.
pub(crate) fn constant_of<T: ScalarType>(value: T::Value) -> ConstantValue {
    <T as sealed::KernelScalar>::kernel_constant(value)
}

pub(crate) fn fill_value_of<T: ScalarType>(value: T::Value) -> crate::schedule::FillValue {
    let (encoded, width) = match constant_of::<T>(value) {
        ConstantValue::F32(value) if T::DTYPE == DType::F16 => (u32::from(f16_bits(value)), 2),
        ConstantValue::F32(value) if T::DTYPE == DType::BF16 => (u32::from(bf16_bits(value)), 2),
        ConstantValue::F32(value) => (value.to_bits(), 4),
        ConstantValue::F16(value) | ConstantValue::BF16(value) => (u32::from(value), 2),
        ConstantValue::I32(value) => (value as u32, 4),
        ConstantValue::U32(value) => (value, 4),
        ConstantValue::Bool(value) => (u32::from(value), 1),
        ConstantValue::Index(_) => panic!("an index scalar is not a storage element type"),
    };
    let bytes = encoded.to_le_bytes();
    match width {
        1 => crate::schedule::FillValue::U8([bytes[0]]),
        2 => crate::schedule::FillValue::U16([bytes[0], bytes[1]]),
        4 => crate::schedule::FillValue::U32(bytes),
        _ => panic!("registered scalar dtype has unsupported fill width"),
    }
}

/// Canonical round-to-nearest-even IEEE-754 binary16 encoding used by the
/// compiler's typed constant and fill construction. This belongs to the
/// representation layer rather than the source-language crate.
pub(crate) fn f16_bits(value: f32) -> u16 {
    let rounded = if value.is_nan() || value.is_infinite() || value == 0.0 {
        value
    } else {
        let magnitude = value.abs();
        if magnitude >= 65_520.0 {
            f32::INFINITY.copysign(value)
        } else {
            let bits = magnitude.to_bits();
            let exponent = ((bits >> 23) & 0xff) as i32 - 127;
            if exponent < -14 {
                let quantum = 2f32.powi(-24);
                ((magnitude / quantum).round_ties_even() * quantum).copysign(value)
            } else {
                let lsb = (bits >> 13) & 1;
                let bits = bits.wrapping_add((1 << 12) - 1 + lsb) & !((1 << 13) - 1);
                f32::from_bits(bits).copysign(value)
            }
        }
    };
    let bits = rounded.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let exponent = exponent - 127 + 15;
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        return sign | (((mantissa | 0x80_0000) >> (1 - exponent + 13)) as u16);
    }
    sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

/// Canonical round-to-nearest-even bfloat16 payload.
pub(crate) fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return (((bits | 0x0040_0000) & 0xffff_0000) >> 16) as u16;
    }
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7fff + lsb) >> 16) as u16
}

pub(crate) fn zero_fill_of<T: ScalarType>() -> crate::schedule::FillValue {
    match T::DTYPE.bytes() {
        1 => crate::schedule::FillValue::U8([0]),
        2 => crate::schedule::FillValue::U16([0; 2]),
        4 => crate::schedule::FillValue::U32([0; 4]),
        _ => panic!("registered scalar dtype has unsupported fill width"),
    }
}

pub(crate) fn fill_constant_for(
    representation: RepresentationId,
    value: seismic_lang::intrinsics::FillConstant,
) -> crate::schedule::FillValue {
    let dtype = match registry::representation_info(representation).kind {
        registry::RepresentationKind::Dense(dtype) => dtype,
        registry::RepresentationKind::Packed(_) => {
            panic!("decode-only packed representation has no fill contract")
        }
        registry::RepresentationKind::External(_) => {
            panic!("external representation has no scalar fill contract")
        }
    };
    let one = matches!(value, seismic_lang::intrinsics::FillConstant::One);
    match dtype {
        DType::F32 => fill_value_of::<F32>(if one { 1.0 } else { 0.0 }),
        DType::F16 => fill_value_of::<F16>(if one { 1.0 } else { 0.0 }),
        DType::BF16 => fill_value_of::<BF16>(if one { 1.0 } else { 0.0 }),
        DType::I32 => fill_value_of::<I32>(i32::from(one)),
        DType::U32 => fill_value_of::<U32>(u32::from(one)),
        DType::Bool => fill_value_of::<Bool>(one),
    }
}

/// A registered element representation, at the type level.
pub trait Representation: 'static + Copy + fmt::Debug + Send + Sync + sealed::Sealed {
    /// Registry name.
    const NAME: &'static str;
    /// The dtype a read of one element produces.
    type Element: VectorElement;
    /// Number of physical planes (1 for dense).
    const PLANES: u32;

    fn id() -> RepresentationId {
        registry::representation(Self::NAME)
            .unwrap_or_else(|| panic!("registry has no representation `{}`", Self::NAME))
    }
}

/// Representations with a canonical element encode/update contract. Packed
/// decode-only formats intentionally do not implement this marker.
pub trait DenseRepresentation: Representation {}
pub trait WritableRepresentation: DenseRepresentation {}

macro_rules! representation {
    ($name:ident, $registry:literal, $elem:ty, $planes:literal) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {}
        impl sealed::Sealed for $name {}
        impl Representation for $name {
            const NAME: &'static str = $registry;
            type Element = $elem;
            const PLANES: u32 = $planes;
        }
    };
}

representation!(DenseF32, "f32", F32, 1);
representation!(DenseF16, "f16", F16, 1);
representation!(DenseBF16, "bf16", BF16, 1);
representation!(DenseI32, "i32", I32, 1);
representation!(DenseU32, "u32", U32, 1);
representation!(DenseBool, "bool", Bool, 1);
impl WritableRepresentation for DenseF32 {}
impl WritableRepresentation for DenseF16 {}
impl WritableRepresentation for DenseBF16 {}
impl WritableRepresentation for DenseI32 {}
impl WritableRepresentation for DenseU32 {}
impl WritableRepresentation for DenseBool {}
impl DenseRepresentation for DenseF32 {}
impl DenseRepresentation for DenseF16 {}
impl DenseRepresentation for DenseBF16 {}
impl DenseRepresentation for DenseI32 {}
impl DenseRepresentation for DenseU32 {}
impl DenseRepresentation for DenseBool {}
representation!(Q4G64, "q4g64", F32, 2);
representation!(Q4G32, "q4g32", F32, 2);
representation!(Q4K, "q4k", F32, 3);
representation!(Q5K, "q5k", F32, 4);
representation!(Q6K, "q6k", F32, 3);
representation!(Q8G32S, "q8g32s", F32, 2);
representation!(IQ4G32, "iq4g32", F32, 2);
representation!(Q8G32, "q8g32", F32, 2);

/// Dispatches a runtime representation id into the typed world.
pub trait RepresentationVisitor {
    type Output;
    fn visit<R: Representation>(self) -> Self::Output;
}

/// The single runtime-to-type dispatch. Every registered representation has
/// an arm; a registry id with no marker is a registry-consistency panic
/// (§13.3.1).
pub fn with_representation<V: RepresentationVisitor>(
    id: RepresentationId,
    visitor: V,
) -> V::Output {
    let name = registry::representation_info(id).name;
    match name {
        "f32" => visitor.visit::<DenseF32>(),
        "f16" => visitor.visit::<DenseF16>(),
        "bf16" => visitor.visit::<DenseBF16>(),
        "i32" => visitor.visit::<DenseI32>(),
        "u32" => visitor.visit::<DenseU32>(),
        "bool" => visitor.visit::<DenseBool>(),
        "q4g64" => visitor.visit::<Q4G64>(),
        "q4g32" => visitor.visit::<Q4G32>(),
        "q4k" => visitor.visit::<Q4K>(),
        "q5k" => visitor.visit::<Q5K>(),
        "q6k" => visitor.visit::<Q6K>(),
        "q8g32s" => visitor.visit::<Q8G32S>(),
        "iq4g32" => visitor.visit::<IQ4G32>(),
        "q8g32" => visitor.visit::<Q8G32>(),
        other => panic!("registry representation `{other}` has no type-level marker"),
    }
}

/// Dispatches a dtype into a scalar marker.
pub trait ScalarVisitor {
    type Output;
    fn visit<T: ScalarType>(self) -> Self::Output;
}

pub fn with_scalar<V: ScalarVisitor>(dtype: DType, visitor: V) -> V::Output {
    match dtype {
        DType::F32 => visitor.visit::<F32>(),
        DType::F16 => visitor.visit::<F16>(),
        DType::BF16 => visitor.visit::<BF16>(),
        DType::I32 => visitor.visit::<I32>(),
        DType::U32 => visitor.visit::<U32>(),
        DType::Bool => visitor.visit::<Bool>(),
    }
}
