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

/// The complete scalar value transported by a kernel/schedule ABI word.
/// Naturals are unsigned 64-bit values, independent of source storage dtypes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    Scalar(DType),
    Nat64,
}

impl ScalarKind {
    pub const fn sort(self) -> seismic_lang::expr::SymbolSort {
        match self {
            Self::Scalar(dtype) => seismic_lang::expr::SymbolSort::Scalar(dtype),
            Self::Nat64 => seismic_lang::expr::SymbolSort::Nat,
        }
    }
    pub const fn value_type(self) -> crate::kernel::ops::ValueType {
        match self {
            Self::Scalar(DType::Bool) => crate::kernel::ops::ValueType::Bool,
            Self::Scalar(dtype) => crate::kernel::ops::ValueType::Scalar(dtype),
            Self::Nat64 => crate::kernel::ops::ValueType::Index,
        }
    }
    pub fn bytes(self) -> u32 {
        match self {
            Self::Scalar(dtype) => dtype.bytes(),
            Self::Nat64 => 8,
        }
    }
    pub fn decode_word(self, word: u64) -> seismic_lang::expr::SymbolValue {
        use seismic_lang::expr::SymbolValue;
        match self {
            Self::Nat64 => SymbolValue::Nat((word).into()),
            Self::Scalar(DType::F32) => SymbolValue::F32(f32::from_bits(word as u32)),
            Self::Scalar(DType::F16) => SymbolValue::F16(word as u16),
            Self::Scalar(DType::BF16) => SymbolValue::BF16(word as u16),
            Self::Scalar(DType::I32) => SymbolValue::I32(word as u32 as i32),
            Self::Scalar(DType::U32) => SymbolValue::U32(word as u32),
            Self::Scalar(DType::Bool) => SymbolValue::Bool(word as u8 != 0),
        }
    }
}

/// A scalar type admitted in kernel SSA.
pub trait ScalarType:
    'static + Copy + fmt::Debug + Send + Sync + sealed::Sealed + sealed::KernelScalar
{
    const KIND: ScalarKind;
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
pub trait VectorElement: ScalarType {
    const DTYPE: DType;
}

/// An unsigned 64-bit natural in kernel SSA and scalar publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Idx {}

pub(crate) mod sealed {
    use crate::kernel::ops::ConstantValue;

    pub trait Sealed {}

    /// Crate-private conversion of a marker's host constant into kernel IR.
    /// The erased SSA type comes from the marker's single `ScalarKind`.
    pub trait KernelScalar {
        fn kernel_constant(value: <Self as super::ScalarType>::Value) -> ConstantValue
        where
            Self: super::ScalarType;
    }
}

macro_rules! scalar {
    ($name:ident, $dtype:expr, $value:ty, $constant:expr $(, $extra:ident)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {}
        impl sealed::Sealed for $name {}
        impl sealed::KernelScalar for $name {
            fn kernel_constant(value: $value) -> crate::kernel::ops::ConstantValue {
                ($constant)(value)
            }
        }
        impl ScalarType for $name {
            const KIND: ScalarKind = ScalarKind::Scalar($dtype);
            type Value = $value;
        }
        impl VectorElement for $name { const DTYPE: DType = $dtype; }
        $(impl $extra for $name {})*
    };
}

use crate::kernel::ops::{ConstantValue, ValueType};

scalar!(
    F32,
    DType::F32,
    f32,
    ConstantValue::F32,
    FloatType,
    AtomicType
);
scalar!(
    F16,
    DType::F16,
    f32,
    |value: f32| ConstantValue::from_scalar(seismic_lang::reference_math::float_literal(
        DType::F16,
        f64::from(value)
    )),
    FloatType,
    AtomicType
);
scalar!(
    BF16,
    DType::BF16,
    f32,
    |value: f32| ConstantValue::from_scalar(seismic_lang::reference_math::float_literal(
        DType::BF16,
        f64::from(value)
    )),
    FloatType,
    AtomicType
);
scalar!(
    I32,
    DType::I32,
    i32,
    ConstantValue::I32,
    IntegerType,
    AtomicType
);
scalar!(
    U32,
    DType::U32,
    u32,
    ConstantValue::U32,
    IntegerType,
    AtomicType
);
scalar!(Bool, DType::Bool, bool, ConstantValue::Bool);
impl NumericType for F32 {}
impl NumericType for F16 {}
impl NumericType for BF16 {}
impl NumericType for I32 {}
impl NumericType for U32 {}
impl SignedType for F32 {}
impl SignedType for F16 {}
impl SignedType for BF16 {}
impl SignedType for I32 {}

impl sealed::Sealed for Idx {}
impl sealed::KernelScalar for Idx {
    fn kernel_constant(value: u64) -> ConstantValue {
        ConstantValue::Index(value)
    }
}
impl ScalarType for Idx {
    const KIND: ScalarKind = ScalarKind::Nat64;
    type Value = u64;
}
impl IntegerType for Idx {}
impl NumericType for Idx {}

/// The erased value type of one scalar marker.
pub(crate) fn value_type_of<T: ScalarType>() -> ValueType {
    T::KIND.value_type()
}

/// The kernel constant of one host value.
pub(crate) fn constant_of<T: ScalarType>(value: T::Value) -> ConstantValue {
    <T as sealed::KernelScalar>::kernel_constant(value)
}

pub(crate) fn fill_value_of<T: VectorElement>(value: T::Value) -> crate::schedule::FillValue {
    let (encoded, width) = match constant_of::<T>(value) {
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

pub(crate) fn zero_fill_of<T: VectorElement>() -> crate::schedule::FillValue {
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

#[cfg(test)]
mod scalar_kind_tests {
    use super::*;
    use seismic_lang::expr::SymbolValue;

    #[test]
    fn abi_words_preserve_naturals_and_ignore_padding_outside_source_width() {
        assert_eq!(
            ScalarKind::Nat64.decode_word(u64::MAX),
            SymbolValue::Nat((u64::MAX).into())
        );
        assert_eq!(ScalarKind::Nat64.value_type(), ValueType::Index);
        assert_eq!(ScalarKind::Nat64.bytes(), 8);
        let padded = 0xffff_ffff_0000_0000;
        assert_eq!(
            ScalarKind::Scalar(DType::U32).decode_word(padded),
            SymbolValue::U32(0)
        );
        assert_eq!(
            ScalarKind::Scalar(DType::Bool).decode_word(padded),
            SymbolValue::Bool(false)
        );
        assert_eq!(
            ScalarKind::Scalar(DType::Bool).decode_word(padded | 1),
            SymbolValue::Bool(true)
        );
        assert_eq!(
            ScalarKind::Scalar(DType::I32).decode_word(padded | 0x8000_0000),
            SymbolValue::I32(i32::MIN)
        );
        assert_eq!(
            ScalarKind::Scalar(DType::F16).decode_word(padded | 0x7c01),
            SymbolValue::F16(0x7c01)
        );
    }
}
