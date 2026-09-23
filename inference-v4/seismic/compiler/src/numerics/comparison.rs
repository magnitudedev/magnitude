//! The elementwise policy relation used by controlled observation and validation.
//! Passing a comparison alone does not issue evidence or widen applicability.
use seismic_lang::precision::{PrecisionPolicy, SpecialPolicy};
use seismic_lang::{registry, types::DType};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ElementComparison {
    pub accepted: bool,
    pub absolute_error: f64,
    pub relative_error: f64,
    pub ulps: u64,
    pub special_changed: bool,
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
    let ulps = ordered_bits(dtype, reference).abs_diff(ordered_bits(dtype, actual));
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

fn ordered_bits(dtype: DType, value: f64) -> u64 {
    let (bits, sign) = match dtype {
        DType::F32 => ((value as f32).to_bits() as u64, 1u64 << 31),
        DType::F16 => (registry::f16_bits(value as f32) as u64, 1u64 << 15),
        DType::BF16 => (
            ((registry::bf16_round(value as f32).to_bits() >> 16) as u64),
            1u64 << 15,
        ),
        _ => unreachable!(),
    };
    // Both zeros share the magnitude rank; signed-zero policy is separate.
    if bits & sign != 0 {
        sign - (bits & (sign - 1))
    } else {
        sign + bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::precision::{Limit, Tolerance};
    #[test]
    fn hybrid_envelope_and_ulp_limit_are_both_required() {
        let policy = PrecisionPolicy::bounded(Tolerance {
            absolute: Limit::new(0.1).unwrap(),
            relative: Limit::new(0.1).unwrap(),
            relative_floor: Limit::new(0.5).unwrap(),
            ulps: None,
        });
        assert!(compare_element(&policy, "value", DType::F32, 10.0, 11.0).accepted);
        assert!(!compare_element(&policy, "value", DType::F32, 10.0, 11.2).accepted);
        let exact = PrecisionPolicy::Exact;
        assert!(
            !compare_element(
                &exact,
                "value",
                DType::F32,
                1.0,
                f32::from_bits(1.0f32.to_bits() + 1) as f64
            )
            .accepted
        );
        assert_eq!(
            compare_element(
                &exact,
                "value",
                DType::F32,
                1.0,
                f32::from_bits(1.0f32.to_bits() + 1) as f64
            )
            .ulps,
            1
        );
    }
    #[test]
    fn specials_and_integer_results_do_not_hide_in_float_tolerance() {
        assert!(!compare_element(&PrecisionPolicy::Exact, "value", DType::F16, -0.0, 0.0).accepted);
        assert!(
            compare_element(
                &PrecisionPolicy::Exact,
                "value",
                DType::F16,
                f64::NAN,
                f64::NAN
            )
            .accepted
        );
        assert!(
            !compare_element(
                &PrecisionPolicy::Exact,
                "value",
                DType::F16,
                f64::INFINITY,
                f64::NEG_INFINITY
            )
            .accepted
        );
        assert!(
            !compare_element(
                &PrecisionPolicy::Unconstrained,
                "value",
                DType::I32,
                1.0,
                2.0
            )
            .accepted
        );
    }
}
