use std::collections::BTreeMap;

use crate::model::{MetricSummary, ScenarioSample};

pub fn summarize_samples(samples: &[ScenarioSample]) -> BTreeMap<String, MetricSummary> {
    let mut values = BTreeMap::<String, Vec<f64>>::new();
    for sample in samples {
        if !matches!(sample.outcome, crate::model::RunOutcome::Valid) {
            continue;
        }
        for (name, value) in &sample.metrics {
            if value.is_finite() {
                values.entry(name.clone()).or_default().push(*value);
            }
        }
    }
    values
        .into_iter()
        .filter_map(|(name, values)| summary(&values).map(|summary| (name, summary)))
        .collect()
}

pub fn summary(values: &[f64]) -> Option<MetricSummary> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = median_sorted(&sorted);
    let mut deviations = sorted
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    Some(MetricSummary {
        samples: sorted.len(),
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
        median,
        median_absolute_deviation: median_sorted(&deviations),
    })
}

fn median_sorted(values: &[f64]) -> f64 {
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

pub fn paired_ratio_interval(
    candidate: &[f64],
    reference: &[f64],
    iterations: usize,
) -> Option<(f64, f64)> {
    let count = candidate.len().min(reference.len());
    if count < 2 || iterations < 20 {
        return None;
    }
    let pairs = candidate
        .iter()
        .zip(reference)
        .filter_map(|(candidate, reference)| {
            (*reference > 0.0 && candidate.is_finite() && reference.is_finite())
                .then_some(candidate / reference)
        })
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return None;
    }
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut medians = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut resample = Vec::with_capacity(pairs.len());
        for _ in 0..pairs.len() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            resample.push(pairs[(seed as usize) % pairs.len()]);
        }
        resample.sort_by(f64::total_cmp);
        medians.push(median_sorted(&resample));
    }
    medians.sort_by(f64::total_cmp);
    let low = medians[((iterations as f64 * 0.025) as usize).min(iterations - 1)];
    let high = medians[((iterations as f64 * 0.975) as usize).min(iterations - 1)];
    Some((low, high))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizes_robust_center_and_spread() {
        let summary = summary(&[1.0, 2.0, 3.0, 100.0]).unwrap();
        assert_eq!(summary.median, 2.5);
        assert_eq!(summary.median_absolute_deviation, 1.0);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 100.0);
    }

    #[test]
    fn paired_interval_tracks_constant_ratio() {
        let interval = paired_ratio_interval(&[2.0, 4.0, 6.0], &[1.0, 2.0, 3.0], 200).unwrap();
        assert_eq!(interval, (2.0, 2.0));
    }
}
