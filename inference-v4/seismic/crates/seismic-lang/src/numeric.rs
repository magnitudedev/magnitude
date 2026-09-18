//! IEEE narrow-float encodings shared by scalar ABI and the reference interpreter.

pub fn bf16_round(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits((x.to_bits() | 0x0040_0000) & 0xffff_0000);
    }
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

pub fn f16_round(x: f32) -> f32 {
    // Round to nearest even at 10 mantissa bits, with the f16 exponent range.
    if x.is_nan() || x.is_infinite() || x == 0.0 {
        return x;
    }
    let a = x.abs();
    if a >= 65520.0 {
        return f32::INFINITY.copysign(x);
    }
    let bits = a.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    if exp < -14 {
        // subnormal in f16: quantum 2^-24
        let q = 2f32.powi(-24);
        return ((a / q).round_ties_even() * q).copysign(x);
    }
    let shift = 13;
    let lsb = (bits >> shift) & 1;
    let rounded = bits.wrapping_add((1 << (shift - 1)) - 1 + lsb) & !((1 << shift) - 1);
    f32::from_bits(rounded).copysign(x)
}

pub fn f16_bits(x: f32) -> u16 {
    let x = f16_round(x);
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (1 - e + 13);
        return sign | m as u16;
    }
    sign | ((e as u16) << 10) | (mant >> 13) as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal
            let mut m = mant;
            let mut e: i32 = 0;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let m = (m & 0x3FF) << 13;
            let e = (e + 1 + 127 - 15) as u32;
            sign | (e << 23) | m
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_subnormals_and_all_finite_roundtrips() {
        assert_eq!(f16_to_f32(1), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x3ff), 1023.0 * 2.0f32.powi(-24));
        for bits in 0..=u16::MAX {
            let x = f16_to_f32(bits);
            if !x.is_nan() {
                assert_eq!(f16_bits(x), bits, "half encoding {bits:04x}");
            }
        }
    }
    #[test]
    fn narrow_nan_never_encodes_as_infinity() {
        for bits in [0x7f800001, 0xff800001, 0x7fc00000, 0x7fffffff] {
            let rounded = bf16_round(f32::from_bits(bits));
            assert!(rounded.is_nan());
            assert_eq!(rounded.to_bits() & 0xffff, 0);
            assert!(f16_to_f32(f16_bits(f32::from_bits(bits))).is_nan());
        }
    }
}

/// Integer arithmetic is modulo 2^32. Signedness determines the interpretation
/// of those bits; integer-to-integer casts preserve them. Float-to-integer casts
/// have their separate saturating conversion semantics.
pub fn integer_value(dtype: crate::types::DType, bits: u32) -> i64 {
    match dtype {
        crate::types::DType::I32 => i64::from(bits as i32),
        crate::types::DType::U32 => i64::from(bits),
        _ => panic!("integer bits require an integer dtype"),
    }
}

/// Whether the available source operands establish defined integer division or
/// remainder. Unknown dividends are sufficient unless signed overflow remains
/// possible. These are the same preconditions for evaluation and omission.
pub fn integer_division_is_defined(
    dtype: crate::types::DType,
    dividend: Option<i64>,
    divisor: Option<i64>,
) -> bool {
    match divisor {
        None | Some(0) => false,
        Some(-1) if dtype == crate::types::DType::I32 => {
            dividend.is_some_and(|a| a != i64::from(i32::MIN))
        }
        Some(_) => true,
    }
}

pub fn integer_shift_is_defined(count: Option<i64>) -> bool {
    count.is_some_and(|n| (0..32).contains(&n))
}
