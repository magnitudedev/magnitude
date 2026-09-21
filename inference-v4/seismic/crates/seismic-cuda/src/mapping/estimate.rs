//! `cuda-estimate-unqualified-v0`: an additive execution-time estimate in nanoseconds.
//! No result of this model is an execution upper bound.
//!
//! EVERY COEFFICIENT BELOW IS UNMEASURED. None was calibrated on any NVIDIA device; they
//! are order-of-magnitude placeholders that make the estimate a consistent ranking of
//! work, launches and memory traffic, nothing more. The identity says `unqualified` and
//! must keep saying so until a probe-calibrated revision replaces the values and the name.
//!
//! The backend cost model consumes exactly two facts: one launch overhead and one
//! point cost per cost unit (ranking only, never legality).

/// The identity of the uncalibrated ranking model.
pub const IDENTITY: &str = "cuda-estimate-unqualified-v0";

/// An invalid cost configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EstimateError(pub String);

impl std::fmt::Display for EstimateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EstimateError {}

#[derive(Clone, Debug, PartialEq)]
pub struct EstimateModel {
    /// One `cuLaunchKernel` submission with its synchronization, in
    /// nanoseconds. Placeholder: 20 us.
    pub launch_ns: f64,
    /// Scalar element operations of one CUDA thread executing the PTX
    /// scalar realization, per second. Placeholder: 2e8.
    pub ops_per_second: f64,
    /// Global-memory bandwidth in bytes per second. Placeholder: 273 GB/s,
    /// the published LPDDR5x bandwidth of GB10's unified memory.
    pub memory_bytes_per_second: f64,
}

impl Default for EstimateModel {
    fn default() -> Self {
        EstimateModel {
            launch_ns: 20_000.0,
            ops_per_second: 2.0e8,
            memory_bytes_per_second: 273.0e9,
        }
    }
}

impl EstimateModel {
    pub fn validate(&self) -> Result<(), EstimateError> {
        let positive = [
            self.launch_ns,
            self.ops_per_second,
            self.memory_bytes_per_second,
        ];
        if positive.iter().any(|c| !c.is_finite() || *c <= 0.0) {
            return Err(EstimateError(
                "estimate coefficients must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}
