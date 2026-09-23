//! The elementwise policy relation used by controlled observation and validation.
//! Passing a comparison alone does not issue evidence or widen applicability.
use seismic_lang::precision::{PrecisionPolicy, SpecialPolicy, Tolerance};
use seismic_lang::reference_math::conversion;
use seismic_lang::types::DType;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ElementComparison {
    pub accepted: bool,
    pub absolute_error: f64,
    pub relative_error: f64,
    pub ulps: u64,
    pub special_changed: bool,
}

/// The element rule over raw payloads: `reference` and `actual` are the bits of
/// one element of `dtype`, in the low bits of the word.
///
/// Discrete dtypes and `Exact` compare bits, so a NaN payload is observable.
/// `Bounded` applies its `SpecialPolicy` to NaN, infinity, signed zero and
/// subnormal differences and its subject tolerance to finite values; ULPs count
/// steps of the dtype's own bit pattern. `Unconstrained` accepts every float.
pub fn compare_element_bits(
    policy: &PrecisionPolicy,
    subject: &str,
    dtype: DType,
    reference: u32,
    actual: u32,
) -> ElementComparison {
    if !dtype.is_float() {
        return ElementComparison {
            accepted: reference == actual,
            absolute_error: (discrete_value(dtype, reference) - discrete_value(dtype, actual)).abs(),
            ..Default::default()
        };
    }
    let tolerance = policy.tolerance(subject);
    let difference = FloatDifference::measure(
        dtype,
        reference,
        actual,
        tolerance.map_or(0.0, |tolerance| tolerance.relative_floor.get()),
    );
    let accepted = match policy {
        PrecisionPolicy::Exact => reference == actual,
        PrecisionPolicy::Bounded { specials, .. } => difference.within(
            tolerance.expect("a bounded policy has a tolerance for every subject"),
            *specials,
        ),
        PrecisionPolicy::Unconstrained => true,
    };
    ElementComparison {
        accepted,
        ..difference.metrics()
    }
}

/// How two floating payloads of one dtype differ, classified in the order the
/// special policy is applied.
enum FloatDifference {
    /// At least one side is NaN; `changed` when exactly one is.
    Nan { changed: bool },
    /// At least one side is infinite and neither is NaN.
    Infinite { changed: bool },
    Finite {
        reference: f64,
        absolute_error: f64,
        relative_error: f64,
        ulps: u64,
        signed_zero_changed: bool,
        subnormal_changed: bool,
    },
}

impl FloatDifference {
    fn measure(dtype: DType, reference: u32, actual: u32, relative_floor: f64) -> Self {
        let (reference_value, actual_value) = (
            conversion::exact_f64(dtype, reference),
            conversion::exact_f64(dtype, actual),
        );
        if reference_value.is_nan() || actual_value.is_nan() {
            return Self::Nan {
                changed: reference_value.is_nan() != actual_value.is_nan(),
            };
        }
        if reference_value.is_infinite() || actual_value.is_infinite() {
            return Self::Infinite {
                changed: reference != actual,
            };
        }
        let minimum_normal = match dtype {
            DType::F32 | DType::BF16 => 2.0_f64.powi(-126),
            DType::F16 => 2.0_f64.powi(-14),
            DType::I32 | DType::U32 | DType::Bool => unreachable!("{dtype} is not a floating dtype"),
        };
        let subnormal = |value: f64| value != 0.0 && value.abs() < minimum_normal;
        let absolute_error = (reference_value - actual_value).abs();
        Self::Finite {
            reference: reference_value,
            absolute_error,
            relative_error: if absolute_error == 0.0 {
                0.0
            } else {
                absolute_error / reference_value.abs().max(relative_floor)
            },
            ulps: ordered_bits(dtype, reference).abs_diff(ordered_bits(dtype, actual)),
            signed_zero_changed: reference_value == 0.0
                && actual_value == 0.0
                && reference_value.is_sign_negative() != actual_value.is_sign_negative(),
            subnormal_changed: (subnormal(reference_value) || subnormal(actual_value))
                && reference != actual,
        }
    }

    fn within(&self, tolerance: Tolerance, specials: SpecialPolicy) -> bool {
        match *self {
            Self::Nan { changed } => !specials.nan || !changed,
            Self::Infinite { changed } => !specials.infinity || !changed,
            Self::Finite {
                reference,
                absolute_error,
                ulps,
                signed_zero_changed,
                subnormal_changed,
                ..
            } => {
                absolute_error <= tolerance.envelope(reference)
                    && tolerance.ulps.is_none_or(|limit| ulps <= limit)
                    && (!specials.signed_zero || !signed_zero_changed)
                    && (!specials.subnormal || !subnormal_changed)
            }
        }
    }

    fn metrics(&self) -> ElementComparison {
        match *self {
            Self::Nan { changed } | Self::Infinite { changed } => ElementComparison {
                special_changed: changed,
                ..Default::default()
            },
            Self::Finite {
                absolute_error,
                relative_error,
                ulps,
                signed_zero_changed,
                subnormal_changed,
                ..
            } => ElementComparison {
                accepted: false,
                absolute_error,
                relative_error,
                ulps,
                special_changed: signed_zero_changed || subnormal_changed,
            },
        }
    }
}

/// The value of a discrete payload, for the absolute-error diagnostic.
fn discrete_value(dtype: DType, bits: u32) -> f64 {
    match dtype {
        DType::I32 => f64::from(bits as i32),
        DType::U32 | DType::Bool => f64::from(bits),
        DType::F32 | DType::BF16 | DType::F16 => unreachable!("{dtype} is not a discrete dtype"),
    }
}

pub fn compare_element(
    policy: &PrecisionPolicy,
    output: &str,
    dtype: DType,
    reference: f64,
    actual: f64,
) -> ElementComparison {
    // Integer/Boolean semantics are never relaxed by a floating-point policy.
    if matches!(dtype, DType::Bool | DType::I32 | DType::U32) {
        return ElementComparison {
            accepted: reference == actual,
            absolute_error: (reference - actual).abs(),
            ..Default::default()
        };
    }
    let Some(tolerance) = policy.tolerance(output) else {
        return ElementComparison {
            accepted: true,
            ..Default::default()
        };
    };
    let specials = match policy {
        PrecisionPolicy::Bounded { specials, .. } => *specials,
        _ => SpecialPolicy::PRESERVE,
    };
    if reference.is_nan() || actual.is_nan() {
        let changed = reference.is_nan() != actual.is_nan();
        return ElementComparison {
            accepted: !specials.nan || !changed,
            special_changed: changed,
            ..Default::default()
        };
    }
    if reference.is_infinite() || actual.is_infinite() {
        let changed = reference != actual;
        return ElementComparison {
            accepted: !specials.infinity || !changed,
            special_changed: changed,
            ..Default::default()
        };
    }
    let signed_zero_changed = reference == 0.0
        && actual == 0.0
        && reference.is_sign_negative() != actual.is_sign_negative();
    let minimum_normal = match dtype {
        DType::F32 | DType::BF16 => f32::MIN_POSITIVE as f64,
        DType::F16 => 2.0_f64.powi(-14),
        _ => unreachable!(),
    };
    let subnormal = |value: f64| value != 0.0 && value.abs() < minimum_normal;
    let subnormal_changed = (subnormal(reference) || subnormal(actual)) && reference != actual;
    let absolute_error = (reference - actual).abs();
    let scale = reference.abs().max(tolerance.relative_floor.get());
    let relative_error = if absolute_error == 0.0 {
        0.0
    } else {
        absolute_error / scale
    };
    let ulps = ordered_bits(dtype, conversion::quantize(dtype, reference))
        .abs_diff(ordered_bits(dtype, conversion::quantize(dtype, actual)));
    ElementComparison {
        accepted: absolute_error <= tolerance.envelope(reference)
            && tolerance.ulps.is_none_or(|limit| ulps <= limit)
            && (!specials.signed_zero || !signed_zero_changed)
            && (!specials.subnormal || !subnormal_changed),
        absolute_error,
        relative_error,
        ulps,
        special_changed: signed_zero_changed || subnormal_changed,
    }
}

/// The rank of a floating payload in value order, from the dtype's own bit
/// pattern. Adjacent finite values differ by one; both zeros share a rank
/// (the signed-zero policy is separate).
fn ordered_bits(dtype: DType, bits: u32) -> u64 {
    let sign = 1u64 << (dtype.bytes() * 8 - 1);
    let bits = u64::from(bits);
    if bits & sign != 0 {
        sign - (bits & (sign - 1))
    } else {
        sign + bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::precision::Limit;
    use std::collections::BTreeMap;

    const F32_ONE: u32 = 0x3f80_0000;

    fn tolerance(absolute: f64, relative: f64, relative_floor: f64, ulps: Option<u64>) -> Tolerance {
        Tolerance {
            absolute: Limit::new(absolute).unwrap(),
            relative: Limit::new(relative).unwrap(),
            relative_floor: Limit::new(relative_floor).unwrap(),
            ulps,
        }
    }

    fn bounded(tolerance: Tolerance, specials: SpecialPolicy) -> PrecisionPolicy {
        PrecisionPolicy::Bounded {
            default: tolerance,
            outputs: BTreeMap::new(),
            specials,
            inputs: BTreeMap::new(),
        }
    }

    fn accepts(policy: &PrecisionPolicy, dtype: DType, reference: u32, actual: u32) -> bool {
        compare_element_bits(policy, "value", dtype, reference, actual).accepted
    }

    #[test]
    fn nan_payload_is_observed_only_by_exact() {
        let (reference, actual) = (0x7fc0_0000, 0x7fc0_1234);
        assert!(!accepts(&PrecisionPolicy::Exact, DType::F32, reference, actual));
        assert!(accepts(
            &bounded(Tolerance::EXACT, SpecialPolicy::PRESERVE),
            DType::F32,
            reference,
            actual
        ));
        assert!(accepts(&PrecisionPolicy::Unconstrained, DType::F32, reference, actual));
        // The same holds for the narrow formats.
        assert!(!accepts(&PrecisionPolicy::Exact, DType::F16, 0x7e00, 0x7e01));
        assert!(!accepts(&PrecisionPolicy::Exact, DType::BF16, 0x7fc0, 0xffc0));
        assert!(accepts(&PrecisionPolicy::Exact, DType::BF16, 0x7fc0, 0x7fc0));
    }

    #[test]
    fn nan_presence_follows_the_special_policy() {
        let preserve = bounded(tolerance(1e30, 0.0, 0.0, None), SpecialPolicy::PRESERVE);
        let comparison = compare_element_bits(&preserve, "value", DType::F32, 0x7fc0_0000, F32_ONE);
        assert!(!comparison.accepted && comparison.special_changed);
        let relaxed = bounded(
            tolerance(0.0, 0.0, 0.0, Some(0)),
            SpecialPolicy {
                nan: false,
                ..SpecialPolicy::PRESERVE
            },
        );
        assert!(accepts(&relaxed, DType::F32, 0x7fc0_0000, F32_ONE));
    }

    #[test]
    fn exact_is_bit_equality() {
        let exact = PrecisionPolicy::Exact;
        assert!(accepts(&exact, DType::F32, F32_ONE, F32_ONE));
        let next = compare_element_bits(&exact, "value", DType::F32, F32_ONE, F32_ONE + 1);
        assert!(!next.accepted);
        assert_eq!(next.ulps, 1);
        assert_eq!(next.absolute_error, 2.0_f64.powi(-23));
        // Signed zeros share a value but not bits.
        let zeros = compare_element_bits(&exact, "value", DType::F16, 0x8000, 0x0000);
        assert!(!zeros.accepted && zeros.special_changed);
        assert_eq!(zeros.ulps, 0);
        assert!(!accepts(&exact, DType::F16, 0x7c00, 0xfc00));
    }

    #[test]
    fn ulps_are_steps_of_the_dtype_bit_pattern() {
        let measure = |dtype, reference, actual| {
            compare_element_bits(&PrecisionPolicy::Unconstrained, "value", dtype, reference, actual).ulps
        };
        // Adjacent values, including across zero, in each format's own encoding.
        assert_eq!(measure(DType::F16, 0x3c00, 0x3c01), 1);
        assert_eq!(measure(DType::BF16, 0x3f80, 0x3f81), 1);
        assert_eq!(measure(DType::F32, 0x0000_0001, 0x8000_0001), 2);
        assert_eq!(measure(DType::F16, 0x0001, 0x8001), 2);
        assert_eq!(measure(DType::BF16, 0x0000, 0x8000), 0);
        // One F16 ulp at 1.0 is 2^13 F32 ulps; an f32 projection would count those.
        assert_eq!(measure(DType::F16, 0x3c00, 0x3c02), 2);
        assert_eq!(measure(DType::F16, 0xbc00, 0x3c00), 2 * 0x3c00);
    }

    /// Raw-bit rank order agrees with exact value order over every non-NaN
    /// payload of the 16-bit formats, and only the two zeros share a rank.
    #[test]
    fn ordered_bits_rank_every_narrow_value_in_value_order() {
        for dtype in [DType::F16, DType::BF16] {
            let mut ranked: Vec<(u64, f64)> = (0..=u16::MAX as u32)
                .map(|bits| (bits, conversion::exact_f64(dtype, bits)))
                .filter(|(_, value)| !value.is_nan())
                .map(|(bits, value)| (ordered_bits(dtype, bits), value))
                .collect();
            ranked.sort_by_key(|(rank, _)| *rank);
            for pair in ranked.windows(2) {
                let ((low_rank, low), (high_rank, high)) = (pair[0], pair[1]);
                if low_rank == high_rank {
                    assert!(low == 0.0 && high == 0.0, "{dtype}: {low} and {high} share rank {low_rank}");
                } else {
                    assert_eq!(high_rank, low_rank + 1, "{dtype}: ranks are dense");
                    assert!(low < high, "{dtype}: rank {low_rank} ({low}) precedes {high_rank} ({high})");
                }
            }
        }
    }

    #[test]
    fn hybrid_envelope_and_ulp_limit_are_both_required() {
        let ten = 10.0f32.to_bits();
        let envelope = bounded(tolerance(0.1, 0.1, 0.5, None), SpecialPolicy::PRESERVE);
        assert!(accepts(&envelope, DType::F32, ten, 11.0f32.to_bits()));
        assert!(!accepts(&envelope, DType::F32, ten, 11.2f32.to_bits()));
        let limited = bounded(tolerance(0.1, 0.1, 0.5, Some(4)), SpecialPolicy::PRESERVE);
        assert!(accepts(&limited, DType::F32, ten, ten + 4));
        assert!(!accepts(&limited, DType::F32, ten, 11.0f32.to_bits()));
        let relative = compare_element_bits(&envelope, "value", DType::F32, ten, 11.0f32.to_bits());
        assert_eq!(relative.relative_error, 0.1);
    }

    #[test]
    fn subject_tolerance_selects_the_bound() {
        let policy = PrecisionPolicy::Bounded {
            default: Tolerance::EXACT,
            outputs: BTreeMap::from([("r0".to_string(), tolerance(1.0, 0.0, 0.0, Some(1)))]),
            specials: SpecialPolicy::PRESERVE,
            inputs: BTreeMap::new(),
        };
        assert!(compare_element_bits(&policy, "r0", DType::BF16, 0x3f80, 0x3f81).accepted);
        assert!(!compare_element_bits(&policy, "r0", DType::BF16, 0x3f80, 0x3f82).accepted);
        assert!(!compare_element_bits(&policy, "value", DType::BF16, 0x3f80, 0x3f81).accepted);
    }

    #[test]
    fn infinities_signed_zeros_and_subnormals_follow_the_special_policy() {
        let wide = tolerance(1e30, 0.0, 0.0, None);
        let preserve = bounded(wide, SpecialPolicy::PRESERVE);
        let relaxed = bounded(
            wide,
            SpecialPolicy {
                nan: true,
                infinity: false,
                signed_zero: false,
                subnormal: false,
            },
        );
        // BF16 largest finite against +infinity.
        assert!(!accepts(&preserve, DType::BF16, 0x7f7f, 0x7f80));
        assert!(accepts(&relaxed, DType::BF16, 0x7f7f, 0x7f80));
        assert!(!accepts(&preserve, DType::F32, 0x8000_0000, 0x0000_0000));
        assert!(accepts(&relaxed, DType::F32, 0x8000_0000, 0x0000_0000));
        // F16 smallest subnormals.
        let subnormal = compare_element_bits(&preserve, "value", DType::F16, 0x0001, 0x0002);
        assert!(!subnormal.accepted && subnormal.special_changed);
        assert!(accepts(&relaxed, DType::F16, 0x0001, 0x0002));
    }

    #[test]
    fn discrete_payloads_are_never_relaxed() {
        for policy in [
            PrecisionPolicy::Exact,
            bounded(tolerance(1e30, 1.0, 1.0, None), SpecialPolicy::PRESERVE),
            PrecisionPolicy::Unconstrained,
        ] {
            let difference = compare_element_bits(&policy, "value", DType::I32, 1, (-1i32) as u32);
            assert!(!difference.accepted);
            assert_eq!(difference.absolute_error, 2.0);
            assert!(!accepts(&policy, DType::U32, 7, 8));
            assert!(!accepts(&policy, DType::Bool, 0, 1));
            assert!(accepts(&policy, DType::U32, u32::MAX, u32::MAX));
        }
    }

    #[test]
    fn projected_hybrid_envelope_and_ulp_limit_are_both_required() {
        let policy = PrecisionPolicy::bounded(tolerance(0.1, 0.1, 0.5, None));
        assert!(compare_element(&policy, "value", DType::F32, 10.0, 11.0).accepted);
        assert!(!compare_element(&policy, "value", DType::F32, 10.0, 11.2).accepted);
        let next = f32::from_bits(1.0f32.to_bits() + 1) as f64;
        let exact = compare_element(&PrecisionPolicy::Exact, "value", DType::F32, 1.0, next);
        assert!(!exact.accepted);
        assert_eq!(exact.ulps, 1);
    }

    #[test]
    fn projected_specials_and_integer_results_do_not_hide_in_float_tolerance() {
        let exact = PrecisionPolicy::Exact;
        assert!(!compare_element(&exact, "value", DType::F16, -0.0, 0.0).accepted);
        assert!(compare_element(&exact, "value", DType::F16, f64::NAN, f64::NAN).accepted);
        assert!(!compare_element(&exact, "value", DType::F16, f64::INFINITY, f64::NEG_INFINITY).accepted);
        assert!(!compare_element(&PrecisionPolicy::Unconstrained, "value", DType::I32, 1.0, 2.0).accepted);
    }
}
