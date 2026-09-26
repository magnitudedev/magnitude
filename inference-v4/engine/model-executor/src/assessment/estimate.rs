//! Analytical decode-speed estimation: a model's header-derived decode demand
//! evaluated against the measurement basis at each requested context depth.

use super::basis::{ClassMeasurement, MeasurementBasis, SecondsBand};
use super::demand::DecodeDemand;
use super::AssessmentError;

/// How reliable an estimate is, from the relative width of its measured
/// range `(upper − lower) / estimated`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerformanceConfidence {
    /// Range within 5% of the estimate.
    High,
    /// Range within 15% of the estimate.
    Moderate,
    Low,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerformanceEstimate {
    pub context_tokens: u32,
    pub lower_tokens_per_second: f64,
    pub estimated_tokens_per_second: f64,
    pub upper_tokens_per_second: f64,
    pub confidence: PerformanceConfidence,
}

/// The depths to estimate: requested depths above the context limit are
/// dropped, duplicates removed, the limit itself always included; ascending.
pub fn performance_depths(context_limit: u32, requested: &[u32]) -> Vec<u32> {
    let mut depths = requested
        .iter()
        .copied()
        .filter(|depth| *depth > 0 && *depth < context_limit)
        .collect::<Vec<_>>();
    depths.push(context_limit);
    depths.sort_unstable();
    depths.dedup();
    depths
}

/// Largest relative range `(upper − lower) / estimated` of a `High` estimate.
pub const HIGH_CONFIDENCE_RANGE: f64 = 0.05;
/// Largest relative range of a `Moderate` estimate.
pub const MODERATE_CONFIDENCE_RANGE: f64 = 0.15;

impl PerformanceConfidence {
    pub fn from_relative_range(range: f64) -> Self {
        if range <= HIGH_CONFIDENCE_RANGE {
            Self::High
        } else if range <= MODERATE_CONFIDENCE_RANGE {
            Self::Moderate
        } else {
            Self::Low
        }
    }
}

/// One estimate per depth: the plain decode step's time is the sum of every
/// term's measured class cost at its launches and its bytes at that depth,
/// at the class's median, slowest and fastest measured behavior. Every class
/// the demand needs must be measured in the basis; callers establish that
/// first (an unmeasured class is an incompatibility, decided before
/// estimation).
pub fn estimate_performance(
    demand: &DecodeDemand,
    basis: &MeasurementBasis,
    depths: &[u32],
) -> Result<Vec<PerformanceEstimate>, AssessmentError> {
    let costs = demand
        .terms
        .iter()
        .map(|term| match basis.get(&term.key) {
            Some(ClassMeasurement::Measured { cost, .. }) => Ok((term, cost)),
            _ => Err(AssessmentError::Estimate(format!(
                "{} {:?} is not measured in the basis",
                term.key.class.name(),
                term.key
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    depths
        .iter()
        .map(|&depth| {
            let step = costs
                .iter()
                .try_fold(SecondsBand::ZERO, |step, (term, cost)| {
                    let bytes = term
                        .bytes_per_context_token
                        .checked_mul(u64::from(depth))
                        .and_then(|history| history.checked_add(term.bytes))
                        .ok_or_else(|| {
                            AssessmentError::Estimate(format!(
                                "{} bytes at depth {depth} overflow",
                                term.key.class.name()
                            ))
                        })?;
                    Ok::<_, AssessmentError>(step.plus(cost.seconds(term.launches, bytes, term.launch_bytes)))
                })?;
            if ![step.fast, step.median, step.slow]
                .iter()
                .all(|seconds| seconds.is_finite() && *seconds > 0.0)
            {
                return Err(AssessmentError::Estimate(format!(
                    "decode step at depth {depth} is not a finite positive time: {step:?}"
                )));
            }
            let estimated = 1.0 / step.median;
            let lower = 1.0 / step.slow;
            let upper = 1.0 / step.fast;
            Ok(PerformanceEstimate {
                context_tokens: depth,
                lower_tokens_per_second: lower,
                estimated_tokens_per_second: estimated,
                upper_tokens_per_second: upper,
                confidence: PerformanceConfidence::from_relative_range((upper - lower) / estimated),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::basis::{
        BasisIdentity, ClassCost, CostModel, MeasurementKey, MEASUREMENT_PROTOCOL_VERSION,
    };
    use crate::assessment::demand::DemandTerm;
    use crate::StreamingCost;
    use seismic::Element;

    fn identity() -> BasisIdentity {
        BasisIdentity {
            engine_build: "test".into(),
            backend: "metal".into(),
            device: "device".into(),
            protocol_version: MEASUREMENT_PROTOCOL_VERSION,
        }
    }

    fn measured(model: CostModel, slow_factor: f64, fast_factor: f64) -> ClassMeasurement {
        ClassMeasurement::Measured {
            points: Vec::new(),
            cost: ClassCost {
                model,
                slow_factor,
                fast_factor,
            },
        }
    }

    /// A weight term of 10 launches over 1 GB at 1 µs + 1 ps/byte and a
    /// history term at 1 ns per context byte, 1 KB per token.
    fn synthetic(slow: f64, fast: f64) -> (DecodeDemand, MeasurementBasis) {
        let weights = MeasurementKey::dense_output(Element::bf16(), Element::bf16());
        let history = MeasurementKey::sample_rows();
        let demand = DecodeDemand {
            terms: vec![
                DemandTerm {
                    key: weights.clone(),
                    launches: 10,
                    bytes: 1_000_000_000,
                    bytes_per_context_token: 0,
                    launch_bytes: 100_000_000,
                },
                DemandTerm {
                    key: history.clone(),
                    launches: 1,
                    bytes: 0,
                    bytes_per_context_token: 1_000,
                    launch_bytes: 0,
                },
            ],
        };
        let basis = MeasurementBasis {
            identity: identity(),
            classes: vec![
                (
                    weights,
                    measured(
                        CostModel::Linear(StreamingCost {
                            launch_seconds: 1.0e-6,
                            seconds_per_byte: 1.0e-12,
                        }),
                        slow,
                        fast,
                    ),
                ),
                (
                    history,
                    measured(
                        CostModel::Linear(StreamingCost {
                            launch_seconds: 0.0,
                            seconds_per_byte: 1.0e-12,
                        }),
                        slow,
                        fast,
                    ),
                ),
            ],
        };
        (demand, basis)
    }

    #[test]
    fn step_time_grows_with_depth_and_bands_come_from_sample_spread() {
        let (demand, basis) = synthetic(1.02, 0.99);
        let estimates = estimate_performance(&demand, &basis, &[0, 1_000_000]).unwrap();
        // 10 µs + 1 ms of weights; then another 1 ms of history.
        let near = 1.0 / (10.0e-6 + 1.0e-3);
        let far = 1.0 / (10.0e-6 + 2.0e-3);
        assert!((estimates[0].estimated_tokens_per_second - near).abs() < 1e-6);
        assert!((estimates[1].estimated_tokens_per_second - far).abs() < 1e-6);
        for estimate in &estimates {
            let median = estimate.estimated_tokens_per_second;
            assert!((estimate.lower_tokens_per_second - median / 1.02).abs() < 1e-6);
            assert!((estimate.upper_tokens_per_second - median / 0.99).abs() < 1e-6);
            assert_eq!(estimate.confidence, PerformanceConfidence::High);
        }
        assert_eq!(estimates[1].context_tokens, 1_000_000);
    }

    #[test]
    fn confidence_thresholds_bound_the_relative_range() {
        let range = |slow: f64, fast: f64| {
            let (demand, basis) = synthetic(slow, fast);
            estimate_performance(&demand, &basis, &[1_000]).unwrap()[0].confidence
        };
        // (1/fast − 1/slow) · median: 1/0.98 − 1/1.02 = 0.040.
        assert_eq!(range(1.02, 0.98), PerformanceConfidence::High);
        // 1/0.95 − 1/1.08 = 0.127.
        assert_eq!(range(1.08, 0.95), PerformanceConfidence::Moderate);
        // 1/0.9 − 1/1.1 = 0.202.
        assert_eq!(range(1.1, 0.9), PerformanceConfidence::Low);
        assert_eq!(
            PerformanceConfidence::from_relative_range(HIGH_CONFIDENCE_RANGE),
            PerformanceConfidence::High
        );
        assert_eq!(
            PerformanceConfidence::from_relative_range(MODERATE_CONFIDENCE_RANGE),
            PerformanceConfidence::Moderate
        );
    }

    #[test]
    fn unmeasured_terms_and_nonpositive_steps_are_estimate_errors() {
        let (demand, mut basis) = synthetic(1.0, 1.0);
        basis.classes.pop();
        assert!(matches!(
            estimate_performance(&demand, &basis, &[1]),
            Err(AssessmentError::Estimate(_))
        ));
        let (demand, mut basis) = synthetic(1.0, 1.0);
        basis.classes[1].1 = ClassMeasurement::Unsupported {
            reason: "absent".into(),
        };
        assert!(estimate_performance(&demand, &basis, &[1]).is_err());
        let zero = DecodeDemand {
            terms: vec![DemandTerm {
                key: MeasurementKey::sample_rows(),
                launches: 0,
                bytes: 0,
                bytes_per_context_token: 0,
                launch_bytes: 0,
            }],
        };
        let (_, basis) = synthetic(1.0, 1.0);
        assert!(matches!(
            estimate_performance(&zero, &basis, &[1]),
            Err(AssessmentError::Estimate(_))
        ));
    }

    #[test]
    fn depths_are_filtered_deduplicated_and_end_at_the_limit() {
        assert_eq!(
            performance_depths(262_144, &[25_000, 50_000, 75_000, 50_000]),
            vec![25_000, 50_000, 75_000, 262_144]
        );
        assert_eq!(
            performance_depths(32_768, &[25_000, 50_000, 75_000]),
            vec![25_000, 32_768]
        );
        assert_eq!(
            performance_depths(50_000, &[25_000, 50_000]),
            vec![25_000, 50_000]
        );
    }
}
