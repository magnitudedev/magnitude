//! `cpu-estimate-unqualified-v0`: an additive execution-time estimate in nanoseconds.
//! No result of this model is an execution upper bound, and no coefficient is measured:
//! they are order-of-magnitude values for scalar Cranelift code on a current desktop core.
//!
//!   span = phases x phase_ns
//!        + pieces x piece_ns / usable
//!        + operations / (operations_per_second x usable)
//!        + bytes / bytes_per_second
//!   usable = min(pieces of the enclosing root `parallel` phase, workers), at least one
use super::accounting::Work;
use seismic_lang::family::SiteId;

pub const IDENTITY: &str = "cpu-estimate-unqualified-v0";

#[derive(Clone, Debug, PartialEq)]
pub struct EstimateModel {
    /// Waking the workers of one phase and joining them.
    pub phase_ns: f64,
    /// Claiming one piece and entering the compiled phase function.
    pub piece_ns: f64,
    /// Scalar element operations per second of one core, address arithmetic included.
    pub operations_per_second: f64,
    /// Worker threads that execute pieces concurrently.
    pub workers: u64,
    /// Memory bandwidth shared by all workers, in bytes per second.
    pub bytes_per_second: f64,
}

impl EstimateModel {
    pub fn unqualified(workers: u64) -> Self {
        EstimateModel { phase_ns: 5_000.0, piece_ns: 200.0, operations_per_second: 2.5e8, workers, bytes_per_second: 2.0e10 }
    }

    pub fn validate(&self) -> Result<(), String> {
        let positive = [self.phase_ns, self.piece_ns, self.operations_per_second, self.bytes_per_second].iter().all(|v| v.is_finite() && *v > 0.0);
        if positive && self.workers > 0 { Ok(()) } else { Err("CPU estimate coefficients must be positive and finite".into()) }
    }

    /// Span of one scope: `phases` phase overheads, `pieces` pieces of the enclosing phase.
    pub fn scope_ns(&self, phases: u64, totals: &Totals, pieces: u64) -> Result<u64, String> {
        let usable = pieces.clamp(1, self.workers) as f64;
        let claimed = if phases > 0 { pieces as f64 * self.piece_ns / usable } else { 0.0 };
        let compute = totals.ops as f64 / (self.operations_per_second * usable) * 1e9;
        let memory = totals.memory_bits as f64 / 8.0 / self.bytes_per_second * 1e9;
        let span = phases as f64 * self.phase_ns + claimed + compute + memory;
        if span.is_finite() && span < u64::MAX as f64 { Ok(span.ceil() as u64) } else { Err("CPU estimate exceeds the representable range".into()) }
    }

    /// Write then read of `bits` of scratch storage.
    pub fn materialization_ns(&self, bits: u64) -> Result<u64, String> {
        self.scope_ns(0, &Totals { ops: 0, memory_bits: bits.checked_mul(2).ok_or("CPU estimate traffic overflows u64")? }, 1)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    pub ops: u64,
    pub memory_bits: u64,
}

impl Work {
    pub(crate) fn totals(&self, site: &dyn Fn(SiteId) -> Option<i64>) -> Result<Totals, String> {
        let sum = |terms: &[seismic_compiler::selection::quantity::Quantity]| terms.iter().try_fold(0u64, |acc, q| acc.checked_add(q.eval(site)?).ok_or_else(|| "derived work overflows u64".to_string()));
        Ok(Totals { ops: sum(&self.ops)?, memory_bits: sum(&self.memory_bits)? })
    }
}
