//! Analytical decode-speed estimation: a model's header-derived decode demand
//! over the device's memory bandwidth at each requested context depth.

use super::bandwidth::DeviceBandwidth;
use super::costs::DECODE_COSTS;
use super::demand::DecodeDemand;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerformanceEstimate {
    pub context_tokens: u32,
    pub tokens_per_second: f64,
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

/// The parts of one decode step's time at one depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepSeconds {
    pub step: f64,
    pub launches: f64,
    pub weights: f64,
    pub history: f64,
}

impl StepSeconds {
    pub fn total(&self) -> f64 {
        self.step + self.launches + self.weights + self.history
    }
}

/// One decode step's time at `depth`: the step and launch costs, the
/// depth-independent bytes over the weight-streaming share of bandwidth, and
/// the history reads over the attention share.
pub fn step_seconds(demand: &DecodeDemand, bandwidth: DeviceBandwidth, depth: u32) -> StepSeconds {
    let bytes_per_second = bandwidth.bytes_per_second as f64;
    StepSeconds {
        step: DECODE_COSTS.step_seconds,
        launches: demand.launches as f64 * DECODE_COSTS.launch_seconds,
        weights: demand.streamed_bytes as f64
            / (bytes_per_second * DECODE_COSTS.weight_efficiency),
        history: demand
            .history
            .iter()
            .map(|read| read.bytes_at(depth) as f64)
            .sum::<f64>()
            / (bytes_per_second * DECODE_COSTS.history_efficiency),
    }
}

/// One estimate per depth.
pub fn estimate_performance(
    demand: &DecodeDemand,
    bandwidth: DeviceBandwidth,
    depths: &[u32],
) -> Vec<PerformanceEstimate> {
    depths
        .iter()
        .map(|&depth| PerformanceEstimate {
            context_tokens: depth,
            tokens_per_second: 1.0 / step_seconds(demand, bandwidth, depth).total(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::bandwidth::BandwidthSource;
    use crate::assessment::demand::HistoryRead;

    fn demand(streamed_bytes: u64) -> DecodeDemand {
        DecodeDemand {
            streamed_bytes,
            history: vec![HistoryRead {
                bytes_per_token: 64 * 1024,
                window: None,
            }],
            launches: 300,
        }
    }

    const BANDWIDTH: DeviceBandwidth = DeviceBandwidth {
        bytes_per_second: 400_000_000_000,
        source: BandwidthSource::Published,
    };

    #[test]
    fn depths_keep_the_limit_and_drop_what_exceeds_it() {
        assert_eq!(
            performance_depths(60_000, &[25_000, 50_000, 75_000, 50_000]),
            vec![25_000, 50_000, 60_000]
        );
        assert_eq!(performance_depths(1_000, &[]), vec![1_000]);
    }

    #[test]
    fn speed_falls_with_depth_and_with_streamed_bytes() {
        let depths = [1_000, 25_000, 100_000];
        let small = estimate_performance(&demand(2_000_000_000), BANDWIDTH, &depths);
        let large = estimate_performance(&demand(4_000_000_000), BANDWIDTH, &depths);
        for pair in small.windows(2) {
            assert!(pair[0].tokens_per_second > pair[1].tokens_per_second);
        }
        for (small, large) in small.iter().zip(&large) {
            assert!(small.tokens_per_second > large.tokens_per_second);
            assert!(small.tokens_per_second.is_finite() && large.tokens_per_second > 0.0);
        }
    }

    #[test]
    fn the_step_is_the_sum_of_its_parts() {
        let parts = step_seconds(&demand(2_000_000_000), BANDWIDTH, 10_000);
        assert_eq!(parts.launches, 300.0 * DECODE_COSTS.launch_seconds);
        assert_eq!(
            parts.history,
            (64.0 * 1024.0 * 10_000.0) / (400e9 * DECODE_COSTS.history_efficiency)
        );
        assert_eq!(
            parts.total(),
            parts.step + parts.launches + parts.weights + parts.history
        );
    }
}
