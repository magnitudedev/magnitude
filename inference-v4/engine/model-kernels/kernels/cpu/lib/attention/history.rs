use super::attention::GROUP;
use seismic::cpu::{Dense, F16};

/// The portable `affine_encode` of one vector `x` at `bits` code bits: per
/// group of GROUP values, zero = f16(min), scale = f16((max - min) / L) and
/// code = min(L, u32(fma(x - zero, 1 / scale, 0.5))) (0 when scale is 0),
/// packed 32 / bits codes per word; (scale, zero) F16 bits per group.
pub fn affine_encode(x: &[f32], bits: u32, codes: &mut [u32], coefficients: &mut [u16]) {
    let levels = (1u32 << bits) - 1;
    let per = (32 / bits) as usize;
    codes.fill(0);
    for (group, values) in x.chunks_exact(GROUP).enumerate() {
        let (low, high) = values
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), value| (low.min(*value), high.max(*value)));
        let zero = F16::narrow(low);
        let scale = F16::narrow((high - low) / levels as f32);
        let step = F16::widen(scale);
        let inverse = if step > 0.0 { 1.0 / step } else { 0.0 };
        let base = F16::widen(zero);
        for (offset, value) in values.iter().enumerate() {
            let i = group * GROUP + offset;
            // `as` saturates: a negative or NaN code is 0, as the portable
            // `u32` conversion.
            let code = ((*value - base).mul_add(inverse, 0.5) as u32).min(levels);
            codes[i / per] |= code << ((i % per) as u32 * bits);
        }
        coefficients[2 * group] = scale;
        coefficients[2 * group + 1] = zero;
    }
}

/// The portable `affine_decode` of one vector at `bits` code bits:
/// `fma(code, scale, zero)` with the value's group pair.
#[inline(always)]
pub fn affine_decode(codes: &[u32], coefficients: &[u16], bits: u32, out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if bits == 8 || bits == 4 {
        return neon::affine_decode(codes, coefficients, bits, out);
    }
    let mask = (1u32 << bits) - 1;
    let per = (32 / bits) as usize;
    for (group, values) in out.chunks_exact_mut(GROUP).enumerate() {
        let scale = F16::widen(coefficients[2 * group]);
        let zero = F16::widen(coefficients[2 * group + 1]);
        for (offset, target) in values.iter_mut().enumerate() {
            let i = group * GROUP + offset;
            let code = (codes[i / per] >> ((i % per) as u32 * bits)) & mask;
            *target = (code as f32).mul_add(scale, zero);
        }
    }
}

/// The 8- and 4-bit decode with vector code extraction: a group's 32 codes
/// are 32 bytes (8 bits) or 16 bytes whose low and high nibbles hold the even
/// and odd codes (4 bits). Each value is the same fused `code * scale + zero`
/// as the portable decode.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::GROUP;
    use seismic::cpu::{Dense, F16};
    use std::arch::aarch64::*;

    #[inline(always)]
    pub(super) fn affine_decode(codes: &[u32], coefficients: &[u16], bits: u32, out: &mut [f32]) {
        let group_bytes = GROUP * bits as usize / 8;
        let groups = out.len() / GROUP;
        assert!(codes.len() * 4 >= groups * group_bytes && coefficients.len() >= 2 * groups);
        let bytes = codes.as_ptr().cast::<u8>();
        for (group, values) in out.chunks_exact_mut(GROUP).enumerate() {
            // SAFETY: the assertion bounds every group's code bytes, and a
            // group writes its own 32 outputs.
            unsafe {
                let scale = vdupq_n_f32(F16::widen(coefficients[2 * group]));
                let zero = vdupq_n_f32(F16::widen(coefficients[2 * group + 1]));
                let packed = bytes.add(group * group_bytes);
                let (first, second) = if bits == 8 {
                    (vld1q_u8(packed), vld1q_u8(packed.add(16)))
                } else {
                    let nibbles = vld1q_u8(packed);
                    let (low, high) = (vandq_u8(nibbles, vdupq_n_u8(15)), vshrq_n_u8(nibbles, 4));
                    (vzip1q_u8(low, high), vzip2q_u8(low, high))
                };
                let target = values.as_mut_ptr();
                store(first, scale, zero, target);
                store(second, scale, zero, target.add(16));
            }
        }
    }

    /// Sixteen codes decoded to `out[0..16]`.
    #[inline(always)]
    unsafe fn store(codes: uint8x16_t, scale: float32x4_t, zero: float32x4_t, out: *mut f32) {
        unsafe {
            let (low, high) = (vmovl_u8(vget_low_u8(codes)), vmovl_u8(vget_high_u8(codes)));
            for (index, quarter) in [
                vmovl_u16(vget_low_u16(low)),
                vmovl_u16(vget_high_u16(low)),
                vmovl_u16(vget_low_u16(high)),
                vmovl_u16(vget_high_u16(high)),
            ]
            .into_iter()
            .enumerate()
            {
                vst1q_f32(out.add(4 * index), vfmaq_f32(zero, vcvtq_f32_u32(quarter), scale));
            }
        }
    }
}
