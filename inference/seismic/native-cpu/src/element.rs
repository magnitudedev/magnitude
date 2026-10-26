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

    /// The eight values at `values` widened, each exactly as
    /// [`Dense::widen`]: its AVX2 vector form.
    ///
    /// # Safety
    /// `values` holds eight values, and the calling tier has AVX2.
    #[cfg(target_arch = "x86_64")]
    unsafe fn widen_x86(values: *const Self::Storage) -> std::arch::x86_64::__m256;

    /// The eight values at `values` widened, each exactly as
    /// [`Dense::widen`]: its NEON vector form, values 0..4 and 4..8.
    ///
    /// # Safety
    /// `values` holds eight values.
    #[cfg(target_arch = "aarch64")]
    unsafe fn widen_neon(values: *const Self::Storage) -> [std::arch::aarch64::float32x4_t; 2];
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
    /// The `u64` element.
    U64: u64, "u64";
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

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn widen_x86(values: *const f32) -> std::arch::x86_64::__m256 {
        unsafe { std::arch::x86_64::_mm256_loadu_ps(values) }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn widen_neon(values: *const f32) -> [std::arch::aarch64::float32x4_t; 2] {
        use std::arch::aarch64::*;
        unsafe { [vld1q_f32(values), vld1q_f32(values.add(4))] }
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

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn widen_x86(values: *const u16) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;
        unsafe {
            let words = _mm256_cvtepu16_epi32(_mm_loadu_si128(values.cast()));
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(words))
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn widen_neon(values: *const u16) -> [std::arch::aarch64::float32x4_t; 2] {
        use std::arch::aarch64::*;
        unsafe {
            let words = vld1q_u16(values);
            [
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(words))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(words)),
            ]
        }
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

    /// [`f16_to_f32`]'s steps lane by lane, NaN payloads included.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn widen_x86(values: *const u16) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;
        unsafe {
            let value = _mm256_cvtepu16_epi32(_mm_loadu_si128(values.cast()));
            let word = |bits: u32| _mm256_set1_epi32(bits as i32);
            let magnitude = _mm256_slli_epi32::<13>(_mm256_and_si256(value, word(0x7fff)));
            let exponent = _mm256_and_si256(magnitude, word(0x0f80_0000));
            let normal = _mm256_add_epi32(magnitude, word(0x3800_0000));
            let special = _mm256_add_epi32(normal, word(0x3800_0000));
            let subnormal = _mm256_castps_si256(_mm256_sub_ps(
                _mm256_castsi256_ps(_mm256_add_epi32(normal, word(0x0080_0000))),
                _mm256_castsi256_ps(word(0x3880_0000)),
            ));
            let bits = _mm256_blendv_epi8(
                _mm256_blendv_epi8(
                    normal,
                    subnormal,
                    _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()),
                ),
                special,
                _mm256_cmpeq_epi32(exponent, word(0x0f80_0000)),
            );
            let sign = _mm256_slli_epi32::<16>(_mm256_and_si256(value, word(0x8000)));
            _mm256_castsi256_ps(_mm256_or_si256(bits, sign))
        }
    }

    /// [`f16_to_f32`]'s steps lane by lane, NaN payloads included.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn widen_neon(values: *const u16) -> [std::arch::aarch64::float32x4_t; 2] {
        use std::arch::aarch64::*;
        unsafe {
            let words = vld1q_u16(values);
            let widen = |value: uint32x4_t| {
                let magnitude = vshlq_n_u32::<13>(vandq_u32(value, vdupq_n_u32(0x7fff)));
                let exponent = vandq_u32(magnitude, vdupq_n_u32(0x0f80_0000));
                let normal = vaddq_u32(magnitude, vdupq_n_u32(0x3800_0000));
                let special = vaddq_u32(normal, vdupq_n_u32(0x3800_0000));
                let subnormal = vreinterpretq_u32_f32(vsubq_f32(
                    vreinterpretq_f32_u32(vaddq_u32(normal, vdupq_n_u32(0x0080_0000))),
                    vreinterpretq_f32_u32(vdupq_n_u32(0x3880_0000)),
                ));
                let bits = vbslq_u32(
                    vceqq_u32(exponent, vdupq_n_u32(0x0f80_0000)),
                    special,
                    vbslq_u32(vceqzq_u32(exponent), subnormal, normal),
                );
                let sign = vshlq_n_u32::<16>(vandq_u32(value, vdupq_n_u32(0x8000)));
                vreinterpretq_f32_u32(vorrq_u32(bits, sign))
            };
            [
                widen(vmovl_u16(vget_low_u16(words))),
                widen(vmovl_high_u16(words)),
            ]
        }
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

/// Exact and branchless, so it vectorizes, and no operand or result of its
/// float arithmetic is subnormal: CPU workers keep denormals, and x86 cores
/// take a microcode assist on every subnormal operand, which quantized
/// scales (often subnormal halves) would hit on every block. Rebiasing the
/// exponent places a normal or special half; a subnormal half's mantissa
/// placed under the exponent of 2^-14 denotes 2^-14 plus the half's value,
/// so one exact subtraction of 2^-14 gives it.
#[inline(always)]
pub fn f16_to_f32(value: u16) -> f32 {
    let magnitude = u32::from(value & 0x7fff) << 13;
    let exponent = magnitude & 0x0f80_0000;
    let normal = magnitude + 0x3800_0000;
    let special = normal + 0x3800_0000;
    let subnormal = (f32::from_bits(normal + 0x0080_0000) - f32::from_bits(0x3880_0000)).to_bits();
    let bits = if exponent == 0x0f80_0000 {
        special
    } else if exponent == 0 {
        subnormal
    } else {
        normal
    };
    f32::from_bits(bits | (u32::from(value & 0x8000) << 16))
}

#[inline(always)]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        let payload = if mantissa == 0 {
            0
        } else {
            0x0200 | (mantissa >> 13) as u16
        };
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

/// A test row's coefficients: the plane `plane` of scale encoding
/// `encoding` filled with finite values, small floats and scale codes, a few
/// moderate codes or, with `every_code`, every finite code (E8M0's
/// subnormal scales included, UE4M3's with and without its ignored sign
/// bit).
#[cfg(test)]
pub(crate) fn finite_scales(plane: &mut [u8], encoding: seismic_lang::registry::ScaleEncoding, every_code: bool) {
    use seismic_lang::registry::ScaleEncoding;
    match encoding {
        ScaleEncoding::F16 | ScaleEncoding::BF16 => {
            for (i, field) in plane.chunks_exact_mut(2).enumerate() {
                let value = 0.002 * (i % 5 + 1) as f32;
                field.copy_from_slice(&if encoding == ScaleEncoding::F16 {
                    f32_to_f16(value).to_le_bytes()
                } else {
                    f32_to_bf16(value).to_le_bytes()
                });
            }
        }
        ScaleEncoding::F32 => {
            for (i, field) in plane.chunks_exact_mut(4).enumerate() {
                field.copy_from_slice(&(0.0004 * (i % 3 + 1) as f32).to_le_bytes());
            }
        }
        // Every exponent but the NaN code 0xff, or 2^(e - 127) for e in
        // 117..=120.
        ScaleEncoding::E8M0 => {
            for (i, field) in plane.iter_mut().enumerate() {
                *field = if every_code { ((i * 37) % 255) as u8 } else { 117 + (i % 4) as u8 };
            }
        }
        // Every code but NaN, or exponent codes 1..=4 (2^-6 ..= 2^-3) with
        // varied mantissas.
        ScaleEncoding::UE4M3 => {
            for (i, field) in plane.iter_mut().enumerate() {
                *field = if every_code {
                    ((i * 37) % 127) as u8 | if i % 2 == 1 { 0x80 } else { 0 }
                } else {
                    8 + (i % 32) as u8
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_widening_is_exact_for_every_half() {
        for bits in 0..=u16::MAX {
            let sign = u32::from(bits & 0x8000) << 16;
            let exponent = i32::from((bits >> 10) & 0x1f);
            let mantissa = u32::from(bits & 0x03ff);
            let expected = match exponent {
                0x1f => 0x7f80_0000 | mantissa << 13 | sign,
                0 => (mantissa as f32 * 2f32.powi(-24)).to_bits() | sign,
                _ => ((1024 + mantissa) as f32 * 2f32.powi(exponent - 25)).to_bits() | sign,
            };
            assert_eq!(f16_to_f32(bits).to_bits(), expected, "{bits:#06x}");
        }
    }

    /// The vector widenings are `widen` bit for bit for every 16-bit value,
    /// NaN payloads included.
    #[test]
    fn vector_widening_is_widen_for_every_value() {
        fn check<E: Dense<Storage = u16>>() {
            let values: Vec<u16> = (0..=u16::MAX).collect();
            for chunk in values.chunks_exact(8) {
                let mut widened = [0.0f32; 8];
                #[cfg(target_arch = "x86_64")]
                {
                    if !std::arch::is_x86_feature_detected!("avx2") {
                        return;
                    }
                    unsafe {
                        std::arch::x86_64::_mm256_storeu_ps(
                            widened.as_mut_ptr(),
                            E::widen_x86(chunk.as_ptr()),
                        )
                    };
                }
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    let [low, high] = E::widen_neon(chunk.as_ptr());
                    std::arch::aarch64::vst1q_f32(widened.as_mut_ptr(), low);
                    std::arch::aarch64::vst1q_f32(widened.as_mut_ptr().add(4), high);
                }
                for (value, widened) in chunk.iter().zip(widened) {
                    assert_eq!(
                        widened.to_bits(),
                        E::widen(*value).to_bits(),
                        "{value:#06x}"
                    );
                }
            }
        }
        check::<F16>();
        check::<Bf16>();
    }

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
