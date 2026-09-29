//! Explicit serving limits for each floating result and writable state subject.
use seismic::{DType, Limit, PrecisionPolicy, SpecialPolicy, Tolerance, TuneError};
use std::collections::BTreeMap;

pub(super) fn policy(subjects: Vec<(String, DType)>) -> Result<PrecisionPolicy, TuneError> {
    #[cfg(feature = "tuning-precision-experiment")]
    let scale = match std::env::var("MAGNITUDE_TUNING_PRECISION_SCALE") {
        Ok(value) => value
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite() && *value > 0.)
            .ok_or_else(|| {
                TuneError::Declaration(
                    "precision experiment scale must be finite and positive".into(),
                )
            })?,
        Err(std::env::VarError::NotPresent) => 1.,
        Err(error) => return Err(TuneError::Declaration(error.to_string())),
    };
    #[cfg(not(feature = "tuning-precision-experiment"))]
    let scale = 1.;
    let outputs = subjects
        .into_iter()
        .map(|(subject, dtype)| {
            let (absolute, relative) = match dtype {
                DType::F32 => (1e-5, 1e-4),
                DType::F16 => (1e-3, 2e-3),
                DType::BF16 => (1e-2, 1e-2),
                _ => unreachable!("only floating subjects have a bounded tolerance"),
            };
            (
                subject,
                Tolerance {
                    absolute: Limit::new(absolute * scale).unwrap(),
                    relative: Limit::new(relative * scale).unwrap(),
                    relative_floor: Limit::ZERO,
                    ulps: None,
                },
            )
        })
        .collect();
    Ok(PrecisionPolicy::Bounded {
        default: Tolerance::EXACT,
        outputs,
        specials: SpecialPolicy::PRESERVE,
        inputs: BTreeMap::new(),
    })
}
