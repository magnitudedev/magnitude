//! Dense element types: the storage of dense tensors, and for `f32`, `bf16`
//! and `f16` their conversions to and from `f32`.
//!
//! Conversions round to nearest, ties to even, and keep NaN payloads quiet,
//! the same contract as the GPU native libraries' compiler conversions. They
//! are written in integer arithmetic, so every tier produces the same bits.

/// A dense element type a CPU kernel can be generic over: its storage, which
/// a kernel may move bit for bit.
pub trait Element: Copy + Send + Sync + 'static {
    /// The stored element.
    type Storage: Copy + Send + Sync + Default + 'static;
    /// The registry name (`f32`, `u32`, ...).
    const NAME: &'static str;
    const BYTES: usize;
}

/// A dense floating element type: an [`Element`] with exact conversions to
/// and from `f32`.
pub trait Dense: Element {
    fn widen(value: Self::Storage) -> f32;
    fn narrow(value: f32) -> Self::Storage;

    /// `value` rounded to this element and widened again: the value a
    /// portable body publishes when it stores to this element.
    #[inline(always)]
    fn round(value: f32) -> f32 {
        Self::widen(Self::narrow(value))
    }
}

macro_rules! elements {
    ($($(#[$doc:meta])* $name:ident: $storage:ty, $registry:literal;)*) => {
        $(
            $(#[$doc])*
            #[derive(Clone, Copy, Debug)]
            pub struct $name;

            impl Element for $name {
                type Storage = $storage;
                const NAME: &'static str = $registry;
                const BYTES: usize = std::mem::size_of::<$storage>();
            }
        )*
    };
}

elements! {
    /// The `f32` element.
    F32: f32, "f32";
    /// The `bf16` element.
    Bf16: u16, "bf16";
    /// The `f16` element.
    F16: u16, "f16";
    /// The `i32` element.
    I32: i32, "i32";
    /// The `u32` element.
    U32: u32, "u32";
}

impl Dense for F32 {
    #[inline(always)]
    fn widen(value: f32) -> f32 {
        value
    }

    #[inline(always)]
    fn narrow(value: f32) -> f32 {
        value
    }
}

impl Dense for Bf16 {
    #[inline(always)]
    fn widen(value: u16) -> f32 {
        bf16_to_f32(value)
    }

    #[inline(always)]
    fn narrow(value: f32) -> u16 {
        f32_to_bf16(value)
    }
}

impl Dense for F16 {
    #[inline(always)]
    fn widen(value: u16) -> f32 {
        f16_to_f32(value)
    }

    #[inline(always)]
    fn narrow(value: f32) -> u16 {
        f32_to_f16(value)
    }
}

#[inline(always)]
pub fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

#[inline(always)]
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

/// Exact and branchless, so it vectorizes: the half's exponent and mantissa
/// placed in an `f32`'s fields denote the half's value times 2^-112 (for
/// subnormal halves too, as `f32` subnormals), so one exact multiplication by
/// 2^112 gives the value; infinities and NaNs take the all-ones exponent.
#[inline(always)]
pub fn f16_to_f32(value: u16) -> f32 {
    let magnitude = u32::from(value & 0x7fff) << 13;
    let scaled = f32::from_bits(magnitude) * f32::from_bits(0x7780_0000);
    let special = u32::from(value & 0x7c00 == 0x7c00);
    let bits = scaled.to_bits() | (special.wrapping_neg() & (0x7f80_0000 | magnitude));
    f32::from_bits(bits | (u32::from(value & 0x8000) << 16))
}

#[inline(always)]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        let payload = if mantissa == 0 { 0 } else { 0x0200 | (mantissa >> 13) as u16 };
        return sign | 0x7c00 | payload;
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        // Normal half: round the 13 dropped mantissa bits to nearest even.
        let half = (((unbiased + 15) as u32) << 10) | (mantissa >> 13);
        let dropped = mantissa & 0x1fff;
        let round_up = dropped > 0x1000 || (dropped == 0x1000 && half & 1 == 1);
        return sign | (half + u32::from(round_up)) as u16;
    }
    if unbiased < -25 {
        return sign;
    }
    // Subnormal half: the implicit bit joins the mantissa.
    let full = mantissa | 0x0080_0000;
    let shift = (-unbiased - 14 + 13) as u32;
    let half = full >> shift;
    let dropped = full & ((1 << shift) - 1);
    let midpoint = 1 << (shift - 1);
    let round_up = dropped > midpoint || (dropped == midpoint && half & 1 == 1);
    sign | (half + u32::from(round_up)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversions_round_trip_every_half() {
        for bits in 0..=u16::MAX {
            let value = f16_to_f32(bits);
            if value.is_nan() {
                assert!(f16_to_f32(f32_to_f16(value)).is_nan());
            } else {
                assert_eq!(f32_to_f16(value), bits, "{bits:#06x}");
            }
        }
    }

    #[test]
    fn half_rounding_is_nearest_even() {
        // 1 + 2^-11 lies halfway between 1 and the next half; ties to even.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(65519.0), 0x7bff);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_f16(2f32.powi(-25) * 1.5), 0x0001);
        assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001);
    }

    #[test]
    fn bfloat_rounding_is_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        for bits in 0..=u16::MAX {
            let value = bf16_to_f32(bits);
            if !value.is_nan() {
                assert_eq!(f32_to_bf16(value), bits);
            }
        }
    }
}
