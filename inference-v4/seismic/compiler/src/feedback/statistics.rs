//! Finite-stage error spending for adaptive, fixed-case median comparisons.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Interval {
    pub lower: f64,
    pub upper: f64,
}

pub(super) fn alpha(delta: f64, event: u64, stage: u32) -> f64 {
    let event = event as f64;
    delta / (2.0 * event * (event + 1.0) * 2.0_f64.powf(stage as f64 + 1.0))
}

pub(super) fn median(samples: &[f64]) -> f64 {
    let mut values = samples.to_vec();
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n == 0 {
        return f64::INFINITY;
    }
    if n % 2 == 0 {
        values[n / 2 - 1] / 2.0 + values[n / 2] / 2.0
    } else {
        values[n / 2]
    }
}

/// Order-statistic interval, with an uninformative bound when even the extreme
/// order statistics cannot meet the allocated error. Logs avoid underflow.
pub(super) fn median_interval(samples: &[f64], alpha: f64) -> Interval {
    let unknown = Interval {
        lower: 0.0,
        upper: f64::INFINITY,
    };
    if samples.is_empty()
        || !(0.0..1.0).contains(&alpha)
        || samples.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return unknown;
    }
    let n = samples.len();
    let mut log_probability = -(n as f64) * std::f64::consts::LN_2;
    let mut log_tail = f64::NEG_INFINITY;
    let mut index = None;
    for k in 0..=(n - 1) / 2 {
        let max = log_tail.max(log_probability);
        log_tail = max + ((log_tail - max).exp() + (log_probability - max).exp()).ln();
        if std::f64::consts::LN_2 + log_tail <= alpha.ln() {
            index = Some(k);
        } else {
            break;
        }
        log_probability += ((n - k) as f64).ln() - ((k + 1) as f64).ln();
    }
    let Some(index) = index else {
        return unknown;
    };
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Interval {
        lower: sorted[index],
        upper: sorted[n - index - 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn early_stage_can_be_uninformative() {
        assert_eq!(median_interval(&[1.0; 8], 0.05).lower, 1.0);
        assert!(median_interval(&[1.0; 8], alpha(0.05, 2, 0))
            .upper
            .is_infinite());
        assert_eq!(
            median_interval(&[1.0; 16], alpha(0.05, 2, 1)),
            Interval {
                lower: 1.0,
                upper: 1.0
            }
        );
    }
    #[test]
    fn intervals_match_small_exact_binomial_oracle() {
        for n in 1..30 {
            for alpha in [0.001, 0.01, 0.05, 0.2] {
                let values = (1..=n).map(|v| v as f64).collect::<Vec<_>>();
                let mut probability = 2.0_f64.powi(-(n as i32));
                let mut tail = 0.0;
                let mut expected = Interval {
                    lower: 0.0,
                    upper: f64::INFINITY,
                };
                for k in 0..=(n - 1) / 2 {
                    tail += probability;
                    if 2.0 * tail <= alpha {
                        expected = Interval {
                            lower: values[k],
                            upper: values[n - k - 1],
                        };
                    }
                    probability *= (n - k) as f64 / (k + 1) as f64;
                }
                assert_eq!(median_interval(&values, alpha), expected);
            }
        }
    }
    #[test]
    fn error_spending_and_large_samples_remain_bounded() {
        let total: f64 = (1..1000)
            .flat_map(|i| (0..20).map(move |k| 2.0 * alpha(0.05, i, k)))
            .sum();
        assert!(total < 0.05 && total > 0.049);
        assert_eq!(
            median_interval(&vec![7.0; 4096], 1e-12),
            Interval {
                lower: 7.0,
                upper: 7.0
            }
        );
    }
    #[test]
    fn campaign_allowance_is_telescoping_across_events_arms_and_stages() {
        for events in [1u64, 2, 17, 1_000] {
            for stages in [1u32, 2, 10, 32] {
                let allocated: f64 = (1..=events)
                    .flat_map(|event| (0..stages).map(move |stage| 2.0 * alpha(0.05, event, stage)))
                    .sum();
                let exact = 0.05
                    * (1.0 - 1.0 / (events as f64 + 1.0))
                    * (1.0 - 2.0_f64.powi(-(stages as i32)));
                assert!((allocated - exact).abs() < 1e-13);
                assert!(allocated < 0.05);
            }
        }
    }

    #[test]
    fn invalid_samples_and_zero_allowance_cannot_create_confidence() {
        for samples in [
            vec![],
            vec![f64::NAN; 64],
            vec![f64::INFINITY; 64],
            vec![-1.0; 64],
        ] {
            assert_eq!(
                median_interval(&samples, 0.05),
                Interval {
                    lower: 0.0,
                    upper: f64::INFINITY
                }
            );
        }
        assert!(median_interval(&[1.0; 64], 0.0).upper.is_infinite());
        assert!(median_interval(&[1.0; 64], -0.01).upper.is_infinite());
        assert!(median_interval(&[1.0; 64], 1.0).upper.is_infinite());
    }
}
