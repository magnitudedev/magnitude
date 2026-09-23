//! Host access to the scalar conversions and comparisons of the language.
//!
//! Every function evaluates the owning recipe or literal quantizer, so host code
//! (registries, runtime binding, comparison) never carries a second formula for
//! narrow floating formats. Bits are the dtype's payload in the low bits of the word.
use super::*;

/// The payload of `bits` of `from` converted by the source cast to `to`.
pub fn convert_bits(from: DType, to: DType, bits: u32) -> u32 {
    let recipe = scalar_recipe(ScalarOp::Cast(to), &[from]);
    evaluate(&recipe, &[ReferenceScalar::from_bits(from, bits)])
        .expect("casts have no failures")
        .bits()
}

/// Round-to-nearest-even quantization of an exact F64 value to a floating dtype.
pub fn quantize(dtype: DType, value: f64) -> u32 {
    float_literal(dtype, value).bits()
}

/// The exact value of floating payload `bits`. Every NaN payload maps to NaN.
pub fn exact_f64(dtype: DType, bits: u32) -> f64 {
    assert!(dtype.is_float(), "exact value of a non-floating dtype");
    // Widening to F32 is exact, and every F32 value is an F64 value.
    f64::from(f32::from_bits(convert_bits(dtype, DType::F32, bits)))
}

/// The source comparison of two payloads of `dtype`.
pub fn compare_bits(op: CmpOp, dtype: DType, a: u32, b: u32) -> bool {
    let source = match op {
        CmpOp::Eq => ast::BinaryOp::Eq,
        CmpOp::Ne => ast::BinaryOp::Ne,
        CmpOp::Lt => ast::BinaryOp::Lt,
        CmpOp::Le => ast::BinaryOp::Le,
        CmpOp::Gt => ast::BinaryOp::Gt,
        CmpOp::Ge => ast::BinaryOp::Ge,
    };
    let recipe = scalar_recipe(ScalarOp::Binary(source), &[dtype, dtype]);
    let inputs = [
        ReferenceScalar::from_bits(dtype, a),
        ReferenceScalar::from_bits(dtype, b),
    ];
    match evaluate(&recipe, &inputs).expect("comparisons have no failures") {
        ReferenceScalar::Bool(result) => result,
        other => panic!("comparison produced {other:?}"),
    }
}

/// The canonical NaN payload every recipe produces for a floating dtype.
pub fn canonical_nan(dtype: DType) -> u32 {
    quantize(dtype, f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOATS: [DType; 3] = [DType::F32, DType::F16, DType::BF16];
    const OPS: [CmpOp; 6] = [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
    ];

    /// Deterministic SplitMix64 samples; the tests need no external generator.
    struct Samples(u64);
    impl Samples {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    /// Independent test oracle: sign, biased exponent and fraction of a 16-bit
    /// payload, valued with exact F64 scaling.
    fn narrow_value(dtype: DType, bits: u32) -> f64 {
        let (fraction_bits, bias) = match dtype {
            DType::F16 => (10, 15),
            DType::BF16 => (7, 127),
            _ => unreachable!(),
        };
        let exponent_mask = (1u32 << (15 - fraction_bits)) - 1;
        let exponent = ((bits >> fraction_bits) & exponent_mask) as i32;
        let fraction = bits & ((1 << fraction_bits) - 1);
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        if exponent as u32 == exponent_mask {
            return if fraction == 0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            };
        }
        let (significand, scale) = if exponent == 0 {
            (fraction, 1 - bias - fraction_bits)
        } else {
            (fraction | (1 << fraction_bits), exponent - bias - fraction_bits)
        };
        sign * f64::from(significand) * 2f64.powi(scale)
    }
    fn is_nan(dtype: DType, bits: u32) -> bool {
        match dtype {
            DType::F32 => f32::from_bits(bits).is_nan(),
            _ => narrow_value(dtype, bits).is_nan(),
        }
    }
    fn nearest_even_to(dtype: DType, value: f64) -> u32 {
        // Exhaustive search over the finite payloads of one sign; used only for
        // 16-bit formats. Infinity is chosen past the overflow midpoint.
        let (largest, infinity) = match dtype {
            DType::F16 => (0x7bffu32, 0x7c00u32),
            DType::BF16 => (0x7f7f, 0x7f80),
            _ => unreachable!(),
        };
        let sign = if value.is_sign_negative() { 0x8000 } else { 0 };
        let magnitude = value.abs();
        let top = narrow_value(dtype, largest);
        let ulp = top - narrow_value(dtype, largest - 1);
        if magnitude >= top + ulp / 2.0 {
            return sign | infinity;
        }
        // Invariant: value(below) <= magnitude < value(above), with `above` past
        // the largest finite payload standing for the overflow boundary.
        let (mut below, mut above) = (0u32, largest + 1);
        while above - below > 1 {
            let middle = (below + above) / 2;
            if narrow_value(dtype, middle) <= magnitude {
                below = middle;
            } else {
                above = middle;
            }
        }
        if below == largest {
            return sign | below;
        }
        let under = magnitude - narrow_value(dtype, below);
        let over = narrow_value(dtype, below + 1) - magnitude;
        let chosen = if under < over || (under == over && below & 1 == 0) {
            below
        } else {
            below + 1
        };
        sign | chosen
    }

    #[test]
    fn canonical_nans_are_the_recipe_payloads() {
        assert_eq!(canonical_nan(DType::F32), 0x7fc0_0000);
        assert_eq!(canonical_nan(DType::F16), 0x7e00);
        assert_eq!(canonical_nan(DType::BF16), 0x7fc0);
        for dtype in FLOATS {
            let nan = canonical_nan(dtype);
            for other in FLOATS {
                assert_eq!(convert_bits(dtype, other, nan), canonical_nan(other));
            }
        }
    }

    #[test]
    fn narrowing_a_nan_payload_is_canonical() {
        assert_eq!(convert_bits(DType::F32, DType::F16, 0x7fc0_1234), 0x7e00);
        assert_eq!(convert_bits(DType::F32, DType::BF16, 0xffc0_1234), 0x7fc0);
        assert_eq!(convert_bits(DType::F16, DType::F32, 0x7e01), 0x7fc0_0000);
        assert_eq!(convert_bits(DType::BF16, DType::F32, 0xff81), 0x7fc0_0000);
        let mut samples = Samples(1);
        for _ in 0..4096 {
            let payload = samples.next() as u32 & 0x807f_ffff | 0x7f80_0001;
            assert_eq!(convert_bits(DType::F32, DType::F16, payload), 0x7e00);
            assert_eq!(convert_bits(DType::F32, DType::BF16, payload), 0x7fc0);
            assert_eq!(convert_bits(DType::F32, DType::F32, payload), payload);
        }
    }

    #[test]
    fn widening_sixteen_bit_payloads_is_exact_for_every_payload() {
        for dtype in [DType::F16, DType::BF16] {
            for bits in 0..=0xffffu32 {
                let expected = narrow_value(dtype, bits);
                let widened = convert_bits(dtype, DType::F32, bits);
                if expected.is_nan() {
                    assert_eq!(widened, 0x7fc0_0000, "{dtype:?} {bits:#06x}");
                    assert!(exact_f64(dtype, bits).is_nan());
                } else {
                    let value = f64::from(f32::from_bits(widened));
                    assert_eq!(value.to_bits(), expected.to_bits(), "{dtype:?} {bits:#06x}");
                    assert_eq!(exact_f64(dtype, bits).to_bits(), expected.to_bits());
                    assert_eq!(convert_bits(DType::F32, dtype, widened), bits);
                }
            }
        }
    }

    #[test]
    fn sixteen_bit_formats_convert_into_each_other_with_one_rounding() {
        for (from, to) in [(DType::F16, DType::BF16), (DType::BF16, DType::F16)] {
            for bits in 0..=0xffffu32 {
                let value = narrow_value(from, bits);
                let expected = if value.is_nan() {
                    canonical_nan(to)
                } else {
                    nearest_even_to(to, value)
                };
                assert_eq!(convert_bits(from, to, bits), expected, "{from:?} {bits:#06x}");
            }
        }
    }

    #[test]
    fn narrowing_rounds_to_nearest_even_at_every_boundary() {
        // Rounding is monotone, so the decisions at and one F32 step around every
        // midpoint between adjacent 16-bit payloads determine the conversion.
        for dtype in [DType::F16, DType::BF16] {
            let infinity = if dtype == DType::F16 { 0x7c00 } else { 0x7f80 };
            for low in 0..infinity {
                let a = narrow_value(dtype, low);
                // The overflow boundary sits where the next finite payload would be.
                let b = if low + 1 == infinity {
                    2.0 * a - narrow_value(dtype, low - 1)
                } else {
                    narrow_value(dtype, low + 1)
                };
                let middle = (a + b) / 2.0;
                let middle = middle as f32;
                assert_eq!(f64::from(middle), (a + b) / 2.0, "midpoint exact in F32");
                for sign in [0u32, 0x8000_0000] {
                    let bits = middle.to_bits() | sign;
                    let narrow_sign = sign >> 16;
                    let tie = if low & 1 == 0 { low } else { low + 1 };
                    assert_eq!(convert_bits(DType::F32, dtype, bits), tie | narrow_sign);
                    assert_eq!(convert_bits(DType::F32, dtype, bits - 1), low | narrow_sign);
                    assert_eq!(
                        convert_bits(DType::F32, dtype, bits + 1),
                        (low + 1) | narrow_sign
                    );
                }
            }
        }
    }

    #[test]
    fn narrowing_sampled_f32_payloads_picks_the_nearest_even_neighbour() {
        let mut samples = Samples(2);
        for _ in 0..(1 << 18) {
            let bits = samples.next() as u32;
            let value = f64::from(f32::from_bits(bits));
            for dtype in [DType::F16, DType::BF16] {
                let expected = if value.is_nan() {
                    canonical_nan(dtype)
                } else {
                    nearest_even_to(dtype, value)
                };
                assert_eq!(convert_bits(DType::F32, dtype, bits), expected, "{bits:#010x}");
            }
        }
    }

    #[test]
    fn floating_to_integer_truncates_saturates_and_maps_nan_to_zero() {
        let expected = |value: f64, to: DType| -> u32 {
            if value.is_nan() {
                return 0;
            }
            let value = value.trunc();
            match to {
                DType::I32 => value.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32 as u32,
                DType::U32 => value.clamp(0.0, f64::from(u32::MAX)) as u32,
                _ => unreachable!(),
            }
        };
        for from in [DType::F16, DType::BF16] {
            for bits in 0..=0xffffu32 {
                let value = narrow_value(from, bits);
                for to in [DType::I32, DType::U32] {
                    assert_eq!(convert_bits(from, to, bits), expected(value, to), "{from:?}");
                }
            }
        }
        let mut samples = Samples(3);
        let edges = [
            0x4f00_0000, 0xcf00_0000, 0x4eff_ffff, 0xcf00_0001, 0x4f80_0000, 0x4f7f_ffff,
            0x7f80_0000, 0xff80_0000, 0x7fc0_1234, 0x8000_0000, 0x3f7f_ffff, 0xbf7f_ffff,
        ];
        for bits in edges.into_iter().chain((0..(1 << 16)).map(|_| samples.next() as u32)) {
            let value = f64::from(f32::from_bits(bits));
            for to in [DType::I32, DType::U32] {
                assert_eq!(convert_bits(DType::F32, to, bits), expected(value, to), "{bits:#010x}");
            }
        }
    }

    #[test]
    fn integers_and_booleans_convert_through_the_recipe() {
        let mut samples = Samples(4);
        let edges = [0u32, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0x0100_0001, 0x0100_0003];
        for bits in edges.into_iter().chain((0..(1 << 14)).map(|_| samples.next() as u32)) {
            let signed = f64::from(bits as i32);
            let unsigned = f64::from(bits);
            assert_eq!(convert_bits(DType::I32, DType::F32, bits), (bits as i32 as f32).to_bits());
            assert_eq!(convert_bits(DType::U32, DType::F32, bits), (bits as f32).to_bits());
            for dtype in [DType::F16, DType::BF16] {
                assert_eq!(convert_bits(DType::I32, dtype, bits), nearest_even_to(dtype, signed));
                assert_eq!(convert_bits(DType::U32, dtype, bits), nearest_even_to(dtype, unsigned));
            }
            assert_eq!(convert_bits(DType::I32, DType::U32, bits), bits);
            assert_eq!(convert_bits(DType::U32, DType::Bool, bits), u32::from(bits != 0));
        }
        for dtype in FLOATS {
            assert_eq!(convert_bits(DType::Bool, dtype, 1), quantize(dtype, 1.0));
            assert_eq!(convert_bits(DType::Bool, dtype, 0), 0);
            // Negative zero is false; every other payload, including NaN, is true.
            let negative_zero = quantize(dtype, -0.0);
            assert_eq!(convert_bits(dtype, DType::Bool, negative_zero), 0);
            assert_eq!(convert_bits(dtype, DType::Bool, canonical_nan(dtype)), 1);
        }
    }

    #[test]
    fn quantization_rounds_once_from_the_exact_f64() {
        let mut samples = Samples(5);
        for _ in 0..(1 << 16) {
            let value = f64::from_bits(samples.next());
            if value.is_nan() {
                for dtype in FLOATS {
                    assert_eq!(quantize(dtype, value), canonical_nan(dtype));
                }
                continue;
            }
            assert_eq!(quantize(DType::F32, value), (value as f32).to_bits(), "{value:e}");
        }
        for dtype in [DType::F16, DType::BF16] {
            let infinity = if dtype == DType::F16 { 0x7c00 } else { 0x7f80 };
            for bits in 0..=0xffffu32 {
                let value = narrow_value(dtype, bits);
                if value.is_nan() {
                    continue;
                }
                assert_eq!(quantize(dtype, value), bits, "{dtype:?} {bits:#06x}");
                if bits & 0x7fff >= infinity - 1 {
                    continue;
                }
                // Midpoints and their F64 neighbours: the F32 route would round twice.
                let next = narrow_value(dtype, bits + 1);
                let middle = (value + next) / 2.0;
                let tie = if bits & 1 == 0 { bits } else { bits + 1 };
                assert_eq!(quantize(dtype, middle), tie, "{dtype:?} {bits:#06x}");
                let toward = |x: f64, up: bool| {
                    let step = if (x > 0.0) == up { 1i64 } else { -1 };
                    f64::from_bits((x.to_bits() as i64 + step) as u64)
                };
                let outward = bits & 0x8000 == 0;
                assert_eq!(quantize(dtype, toward(middle, !outward)), bits);
                assert_eq!(quantize(dtype, toward(middle, outward)), bits + 1);
            }
        }
        assert_eq!(quantize(DType::F16, 65_519.999), 0x7bff);
        assert_eq!(quantize(DType::F16, 65_520.0), 0x7c00);
        assert_eq!(quantize(DType::F16, -f64::INFINITY), 0xfc00);
        assert_eq!(quantize(DType::BF16, f64::from_bits(1)), 0);
    }

    #[test]
    fn exact_f32_values_are_the_payload_values() {
        let mut samples = Samples(6);
        for _ in 0..(1 << 16) {
            let bits = samples.next() as u32;
            let value = exact_f64(DType::F32, bits);
            if is_nan(DType::F32, bits) {
                assert!(value.is_nan());
            } else {
                assert_eq!(value.to_bits(), f64::from(f32::from_bits(bits)).to_bits());
            }
        }
    }

    #[test]
    fn comparisons_order_every_sixteen_bit_payload_against_special_pivots() {
        let expected = |op: CmpOp, a: f64, b: f64| match op {
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
            CmpOp::Lt => a < b,
            CmpOp::Le => a <= b,
            CmpOp::Gt => a > b,
            CmpOp::Ge => a >= b,
        };
        for dtype in [DType::F16, DType::BF16] {
            let (infinity, fraction) = if dtype == DType::F16 {
                (0x7c00u32, 0x03ffu32)
            } else {
                (0x7f80, 0x007f)
            };
            let mut pivots = vec![0, 1, fraction, fraction + 1, infinity - 1, infinity];
            pivots.extend([infinity + 1, infinity | fraction, 0x3c00, 0x3f80, 0x1234]);
            let negated: Vec<_> = pivots.iter().map(|p| p | 0x8000).collect();
            pivots.extend(negated);
            for &pivot in &pivots {
                let right = narrow_value(dtype, pivot);
                for bits in 0..=0xffffu32 {
                    let left = narrow_value(dtype, bits);
                    for op in OPS {
                        assert_eq!(
                            compare_bits(op, dtype, bits, pivot),
                            expected(op, left, right),
                            "{dtype:?} {op:?} {bits:#06x} {pivot:#06x}"
                        );
                    }
                }
            }
        }
        let mut samples = Samples(7);
        for _ in 0..(1 << 16) {
            let a = samples.next() as u32;
            let b = if samples.next() & 1 == 0 { a ^ 0x8000_0000 } else { samples.next() as u32 };
            for op in OPS {
                let (x, y) = (f32::from_bits(a), f32::from_bits(b));
                assert_eq!(compare_bits(op, DType::F32, a, b), expected(op, x.into(), y.into()));
                assert_eq!(compare_bits(op, DType::I32, a, b), match op {
                    CmpOp::Eq => a == b,
                    CmpOp::Ne => a != b,
                    CmpOp::Lt => (a as i32) < (b as i32),
                    CmpOp::Le => (a as i32) <= (b as i32),
                    CmpOp::Gt => (a as i32) > (b as i32),
                    CmpOp::Ge => (a as i32) >= (b as i32),
                });
                assert_eq!(compare_bits(op, DType::U32, a, b), match op {
                    CmpOp::Eq => a == b,
                    CmpOp::Ne => a != b,
                    CmpOp::Lt => a < b,
                    CmpOp::Le => a <= b,
                    CmpOp::Gt => a > b,
                    CmpOp::Ge => a >= b,
                });
            }
        }
    }
}
