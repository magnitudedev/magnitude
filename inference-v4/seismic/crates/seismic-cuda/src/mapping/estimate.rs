//! `cuda-estimate-unqualified-v0`: an additive execution-time estimate in nanoseconds
//! (spec 12.4-12.5, plan 4.2). No result of this model is an execution upper bound.
//!
//! EVERY COEFFICIENT BELOW IS UNMEASURED. None was calibrated on any NVIDIA device; they
//! are order-of-magnitude placeholders that make the estimate a consistent ranking of
//! work, launches and memory traffic, nothing more. The identity says `unqualified` and
//! must keep saying so until a probe-calibrated revision replaces the values and the name.
//!
//!   launch_ns               one `cuLaunchKernel` submission with its synchronization: the
//!                           runtime synchronizes after every launch and downloads the status
//!                           words. Placeholder: 20 us.
//!   visit_ns                loop bookkeeping of one ordered/pipeline window. Placeholder.
//!   ops_per_second          scalar element operations of one CUDA thread executing the PTX
//!                           scalar realization (every tile access is a global-memory load or
//!                           store). Placeholder: 2e8.
//!   concurrent_threads      threads the device runs at full per-thread speed. Placeholder:
//!                           6144, the CUDA core count NVIDIA publishes for GB10; the driver
//!                           query exposes no equivalent, so every device inherits it.
//!   memory_bytes_per_second global-memory bandwidth. Placeholder: 273 GB/s, the published
//!                           LPDDR5x bandwidth of GB10's unified memory.
//!
//! Structure of one scope: `launches * launch + (ops / rate + visits * visit) / min(pieces,
//! concurrent threads) + memory bits / bandwidth`. A piece is one thread (one warp of lanes
//! that all execute the same body when a participant intrinsic is present), so there is no
//! intra-piece parallel share.
use super::accounting::Work;
use seismic_compiler::selection::quantity::Quantity;
use seismic_lang::family::SiteId;

pub const IDENTITY: &str = "cuda-estimate-unqualified-v0";

#[derive(Clone, Debug, PartialEq)]
pub struct EstimateModel {
    pub launch_ns: f64,
    pub visit_ns: f64,
    pub ops_per_second: f64,
    pub concurrent_threads: u64,
    pub memory_bytes_per_second: f64,
}

impl Default for EstimateModel {
    fn default() -> Self {
        EstimateModel {
            launch_ns: 20_000.0,
            visit_ns: 20.0,
            ops_per_second: 2.0e8,
            concurrent_threads: 6_144,
            memory_bytes_per_second: 273.0e9,
        }
    }
}

/// Evaluated work totals of one execution scope.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Totals {
    pub ops: u64,
    pub visits: u64,
    pub memory_bits: u64,
}

impl Work {
    pub(crate) fn totals(&self, site: &dyn Fn(SiteId) -> Option<i64>) -> Result<Totals, String> {
        let sum = |terms: &[Quantity]| {
            terms.iter().try_fold(0u64, |acc, q| {
                acc.checked_add(q.eval(site)?)
                    .ok_or_else(|| "work total overflows u64".to_string())
            })
        };
        Ok(Totals {
            ops: sum(&self.ops)?,
            visits: sum(&self.visits)?,
            memory_bits: sum(&self.memory_bits)?,
        })
    }
}

impl EstimateModel {
    pub fn validate(&self) -> Result<(), String> {
        let positive = [
            self.launch_ns,
            self.visit_ns,
            self.ops_per_second,
            self.memory_bytes_per_second,
        ];
        if positive.iter().any(|c| !c.is_finite() || *c <= 0.0) || self.concurrent_threads == 0 {
            return Err("estimate coefficients must be finite and positive, with at least one concurrent thread".into());
        }
        Ok(())
    }

    fn nanoseconds(value: f64) -> Result<u64, String> {
        if !value.is_finite() || value < 0.0 || value >= u64::MAX as f64 {
            return Err(format!(
                "estimate {value} ns is outside the representable range"
            ));
        }
        Ok(value.ceil() as u64)
    }

    /// Estimate of one execution scope: `launches` fixed overheads plus its work spread over
    /// the pieces that run concurrently, plus its memory traffic.
    pub fn scope_ns(&self, launches: u64, totals: &Totals, pieces: u64) -> Result<u64, String> {
        let concurrency = pieces.clamp(1, self.concurrent_threads) as f64;
        let compute = (totals.ops as f64 / self.ops_per_second * 1e9
            + totals.visits as f64 * self.visit_ns)
            / concurrency;
        let memory = totals.memory_bits as f64 / 8.0 / self.memory_bytes_per_second * 1e9;
        Self::nanoseconds(launches as f64 * self.launch_ns + compute + memory)
    }

    /// Estimate of writing and reading back `bits` of scratch storage.
    pub fn materialization_ns(&self, bits: u64) -> Result<u64, String> {
        Self::nanoseconds(2.0 * bits as f64 / 8.0 / self.memory_bytes_per_second * 1e9)
    }
}
