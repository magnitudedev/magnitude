//! The fixed costs of a decode step: how much of the device's memory
//! bandwidth weight and history reads reach, and the time every entry call
//! and every step add beyond their bytes. One set for every device and
//! backend; nothing is fitted per device or measured at runtime.

/// The decode costs every device is assessed with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeCosts {
    /// Submitting one step and waiting for its selection.
    pub step_seconds: f64,
    /// One entry call beyond the bytes it streams, including its dependency
    /// on the call before it.
    pub launch_seconds: f64,
    /// The share of bandwidth weight streaming reaches: long sequential reads
    /// sustain most of a memory system's peak.
    pub weight_efficiency: f64,
    /// The share of bandwidth decode attention reaches over history.
    pub history_efficiency: f64,
}

/// Taken from one Apple M4 Max (40-core GPU, 546 GB/s published) and used
/// unchanged on every device:
/// - weights: q4_k, q8 and bf16 streaming at 16,384 × 4,096 rows reached
///   473–488 GB/s;
/// - history: decode attention over dense history reached 243 GB/s, and over
///   affine K8/V4 history about 227 GB/s in a real decode;
/// - step: submission and the selection wait, 170 µs;
/// - launch: a real Qwen3.5-4B Q4_K_M decode (8.76 ms per step at 4,096
///   tokens), less its weights, history and step, over its 164 entry calls.
pub const DECODE_COSTS: DecodeCosts = DecodeCosts {
    step_seconds: 170e-6,
    launch_seconds: 15e-6,
    weight_efficiency: 0.87,
    history_efficiency: 0.43,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_step_is_priced_positively() {
        for value in [DECODE_COSTS.step_seconds, DECODE_COSTS.launch_seconds] {
            assert!(value.is_finite() && value > 0.0);
        }
        for efficiency in [
            DECODE_COSTS.weight_efficiency,
            DECODE_COSTS.history_efficiency,
        ] {
            assert!(efficiency > 0.0 && efficiency <= 1.0);
        }
    }
}
