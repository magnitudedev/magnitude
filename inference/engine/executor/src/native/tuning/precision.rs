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

/// The error classes the engine's kernels declare (`error_class` in their
/// native declarations), each with the envelope a configuration of the class
/// is validated under at tuning: the error the form is measured to have, with
/// room for the tuning inputs, far inside what a defective kernel produces.
/// Whether a model tolerates a class is not decided here: its qualification
/// (top-1 agreement and KL against an F32 forward) admits classes per model,
/// and the host passes the admitted names at load.
const ERROR_CLASSES: &[(&str, seismic::ErrorEnvelope)] = &[
    // Activations quantized to int8 per (row, 32 columns) against exact
    // weights (`dense_expand`, `dense_output` on Metal tensor operations).
    // On the tuning inputs the down projection differs from its default by
    // 1.8e-3 relative RMS (largest element 1.3e-2 reference RMS) and the
    // gate/up product by 3.5e-3; real activations measure 8e-3 to 1e-2.
    ("int8_activations", envelope(2e-2, 0.25)),
];

const fn envelope(relative_rms: f64, peak: f64) -> seismic::ErrorEnvelope {
    seismic::ErrorEnvelope {
        relative_rms: seismic::Limit::from_finite(relative_rms),
        peak: seismic::Limit::from_finite(peak),
    }
}

/// The error classes a load admits, with their envelopes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdmittedErrorClasses(BTreeMap<String, seismic::ErrorEnvelope>);

/// No error class admitted: every tuned configuration agrees with its
/// entry's default within the per-dtype tolerances.
pub static NO_ERROR_CLASSES: AdmittedErrorClasses = AdmittedErrorClasses(BTreeMap::new());

impl AdmittedErrorClasses {
    /// The named classes; a name no kernel declares is refused.
    pub fn of(names: &[String]) -> Result<Self, String> {
        names
            .iter()
            .map(|name| {
                ERROR_CLASSES
                    .iter()
                    .find(|(class, _)| class == name)
                    .map(|(class, envelope)| ((*class).to_owned(), *envelope))
                    .ok_or_else(|| {
                        format!(
                            "unknown error class `{name}`; the kernels declare: {}",
                            ERROR_CLASSES
                                .iter()
                                .map(|(class, _)| *class)
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })
            })
            .collect::<Result<_, _>>()
            .map(Self)
    }

    pub(super) fn envelopes(&self) -> &BTreeMap<String, seismic::ErrorEnvelope> {
        &self.0
    }
}
