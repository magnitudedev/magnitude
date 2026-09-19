//! `metal-estimate-probe-calibrated-m4max-20260919-v3`: an additive execution-time estimate in nanoseconds
//! (spec 12.4-12.5, plan 4.2). No result of this model is an execution upper bound.
//!
//! Coefficient provenance. "Probe" means a small standalone Metal kernel timed on an Apple
//! M4 Max (macOS 15) on 2026-09-19, fifty dispatches per command buffer, best of several
//! command buffers. Other devices inherit these values unqualified.
//!
//!   launch_ns               probe: an empty dispatch costs 2-3 us of GPU time inside one
//!                           encoder. Host encoding measured 0.3 us per dispatch (0.2 ms for a
//!                           661-dispatch decode step) and overlaps device execution, because a
//!                           batch is committed in command buffers of 64 dispatches; it is not
//!                           charged. (v2 charged 8 us, attributing the wall-minus-device time
//!                           of a step to encoding; that time is engine-side preparation and
//!                           readback outside the submission.)
//!   lane_ops_per_second,    emitted packed `linear` kernels (2560x9216 q4g64; 1, 32 and 128
//!   visit_ns,               rows; scalar body and the staged 8x8 matrix lowerings), each timed
//!   matrix_multiply_ns,     as 30 dispatches per command buffer at steady GPU clocks. The
//!   matrix_transfer_ns      four rates are one joint fit: the scalar body runs about 6 ns per
//!                           counted element operation on a lane, and a staged matrix
//!                           iteration (two loads with their barriers, one multiply) about
//!                           150 ns. Single cold dispatches run 3-4x slower (GPU clock ramp)
//!                           and are not what a batched forward executes.
//!   concurrent_lanes        probe (compute-bound kernels without memory traffic: a serial FMA
//!                           chain, four independent chains, a loop of subgroup sums; one
//!                           subgroup per threadgroup): 245 subgroups busy at 256 dispatched,
//!                           330-380 at 512, 460-495 at 1024, 500-530 at 4096. The model clamps
//!                           hard, so the coefficient is the 448-subgroup midpoint of that soft
//!                           saturation. (v1 used 256, fitted to packed kernels that were partly
//!                           bandwidth-bound, which priced every launch above 256 pieces twice
//!                           too slow.)
//!   collective (accounting::COLLECTIVE_OPS)
//!                           probe: a loop of `simd_sum` runs 24.6 ns per collective, 16
//!                           counted lane operations
//!   matrix_subgroups        the matrix kernels above stay latency-bound at 640 pieces and run
//!                           as if about 550 subgroups were busy at 1280 pieces
//!   device_bytes_per_second the emitted packet vector kernel over 248320x2560 q4g64 weights
//!                           reads its 357 MB of distinct weight bytes in 0.72 ms of GPU time
//!                           (at least 495 GB/s); a single-stream read of a 1 GiB buffer
//!                           measured 410 GB/s (v1). Only the distinct bytes of a view are
//!                           bus traffic; re-reads by further visits are cache-served and cost
//!                           the lane operations that consume them (an unqualified assumption
//!                           for repeated views larger than the device's caches).
//!   private_pressure_bytes  the emitted 2x2-fragment packed `linear` (128x9216x2560 q4g64), same
//!                           work at three piece shapes whose replicated accumulator tile is
//!                           1, 2 and 8 KiB per thread: 1.77, 2.2 (scaled from 32 rows) and
//!                           3.3-3.8 ms. One coefficient, `1 + bytes / 8 KiB` on the compute
//!                           span, reproduces the ratios within 15%. A standalone probe with a
//!                           randomly addressed private array shows the same collapse of busy
//!                           subgroups from 1 KiB per thread upward.
//!   local_bytes_per_second  probe: threads streaming their threadgroup slice move 600-900 GB/s in
//!                           aggregate; a lower bound (that probe is not purely memory-bound)
use super::accounting::Work;
use seismic_compiler::selection::quantity::Quantity;
use crate::execution::SUBGROUP;
use seismic_lang::family::SiteId;

/// Probe-calibrated on Apple M4 Max, 2026-09-19 (see the module comment for each coefficient).
pub const IDENTITY: &str = "metal-estimate-probe-calibrated-m4max-20260919-v3";

/// Estimate coefficients (see the module comment for their provenance).
#[derive(Clone, Debug, PartialEq)]
pub struct EstimateModel {
    /// Encode + submit + schedule of one launch.
    pub launch_ns: f64,
    /// Loop/window bookkeeping of one ordered or pipeline visit on one lane.
    pub visit_ns: f64,
    /// Scalar element operations per second of one lane.
    pub lane_ops_per_second: f64,
    /// Lanes the device executes concurrently at full per-lane speed.
    pub concurrent_lanes: u64,
    /// One cooperative 8x8 matrix multiply-accumulate of a subgroup.
    pub matrix_multiply_ns: f64,
    /// One matrix load or store against threadgroup memory, with its barrier.
    pub matrix_transfer_ns: f64,
    /// Subgroups that execute matrix atoms concurrently at full speed.
    pub matrix_subgroups: u64,
    /// Device buffer bandwidth in bytes per second.
    pub device_bytes_per_second: f64,
    /// Tile storage (threadgroup / thread-private) bandwidth in bytes per second.
    pub local_bytes_per_second: f64,
    /// Thread-private array bytes per thread at which a kernel's compute span doubles.
    pub private_pressure_bytes: f64,
    /// One matrix load of an operand that is resident where the atom reads it (no staging
    /// write or barrier of its own).
    pub resident_load_ns: f64,
    /// One SIMD group's wait in a threadgroup barrier.
    pub owner_completion_ns: f64,
    /// Threadgroup memory the device keeps resident at once: the threadgroups that run
    /// concurrently are at most this over the bytes one threadgroup declares.
    pub threadgroup_pool_bytes: u64,
}

impl Default for EstimateModel {
    fn default() -> Self {
        EstimateModel {
            launch_ns: 3_000.0,
            visit_ns: 20.0,
            lane_ops_per_second: 6.5e8,
            concurrent_lanes: 14_336,
            matrix_multiply_ns: 12.0,
            matrix_transfer_ns: 70.0,
            matrix_subgroups: 512,
            device_bytes_per_second: 495.0e9,
            local_bytes_per_second: 900.0e9,
            private_pressure_bytes: 8_192.0,
            resident_load_ns: 28.0,
            owner_completion_ns: 50.0,
            threadgroup_pool_bytes: 2 * 1024 * 1024,
        }
    }
}

/// Evaluated work totals of one execution scope.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Totals {
    pub lane_ops: u64,
    pub visits: u64,
    pub matrix_multiplies: u64,
    pub matrix_transfers: u64,
    pub device_bits: u64,
    pub local_bits: u64,
    pub resident_loads: u64,
    pub owner_completions: u64,
}

/// Threadgroup geometry of the kernel a scope executes in, as far as the scope's candidate
/// chain determines it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Group {
    /// SIMD groups of one threadgroup: the inner owners of a launch piece, else one.
    pub subgroups: u64,
    /// Threadgroup memory bytes one threadgroup declares.
    pub bytes: u64,
}

impl Default for Group {
    fn default() -> Self {
        Group { subgroups: 1, bytes: 0 }
    }
}

impl Work {
    pub(crate) fn totals(&self, site: &dyn Fn(SiteId) -> Option<i64>) -> Result<Totals, String> {
        let sum = |terms: &[Quantity]| terms.iter().try_fold(0u64, |acc, q| acc.checked_add(q.eval(site)?).ok_or_else(|| "work total overflows u64".to_string()));
        Ok(Totals {
            lane_ops: sum(&self.lane_ops)?,
            visits: sum(&self.visits)?,
            matrix_multiplies: sum(&self.matrix_multiplies)?,
            matrix_transfers: sum(&self.matrix_transfers)?,
            device_bits: sum(&self.device_bits)?,
            local_bits: sum(&self.local_bits)?,
            resident_loads: sum(&self.resident_loads)?,
            owner_completions: sum(&self.owner_completions)?,
        })
    }
}

impl EstimateModel {
    /// The queried device facts carry no throughput or concurrency measurement, so every
    /// device takes the probe-calibrated coefficients.
    #[cfg(target_os = "macos")]
    pub fn from_device(_device: &crate::runtime::DeviceInfo) -> Self {
        EstimateModel::default()
    }

    pub fn validate(&self) -> Result<(), String> {
        let positive = [self.launch_ns, self.visit_ns, self.lane_ops_per_second, self.matrix_multiply_ns, self.matrix_transfer_ns, self.device_bytes_per_second, self.local_bytes_per_second, self.private_pressure_bytes, self.resident_load_ns, self.owner_completion_ns];
        if positive.iter().any(|c| !c.is_finite() || *c <= 0.0) || self.concurrent_lanes < SUBGROUP as u64 || self.matrix_subgroups == 0 || self.threadgroup_pool_bytes == 0 {
            return Err("estimate coefficients must be finite and positive, with at least one concurrent subgroup".into());
        }
        Ok(())
    }

    fn nanoseconds(value: f64) -> Result<u64, String> {
        if !value.is_finite() || value < 0.0 || value >= u64::MAX as f64 {
            return Err(format!("estimate {value} ns is outside the representable range"));
        }
        Ok(value.ceil() as u64)
    }

    /// Duration of `totals` spread over `pieces` concurrent pieces:
    /// `norm(max(compute, tile traffic), bus traffic)`.
    /// Scalar work and matrix atoms saturate the device at different subgroup counts.
    fn span(&self, totals: &Totals, pieces: u64, private_bytes: u64, group: Group) -> f64 {
        // Threadgroup memory is a device pool: the threadgroups resident at once are the pool
        // over the bytes one threadgroup declares, each running its SIMD groups.
        let resident = match group.bytes {
            0 => u64::MAX,
            bytes => (self.threadgroup_pool_bytes / bytes).max(1).saturating_mul(group.subgroups.max(1)),
        };
        let scalar_concurrency = pieces.clamp(1, (self.concurrent_lanes / SUBGROUP as u64).max(1)).min(resident) as f64;
        let matrix_concurrency = pieces.clamp(1, self.matrix_subgroups).min(resident) as f64;
        let scalar = (totals.lane_ops as f64 / self.lane_ops_per_second * 1e9 + totals.visits as f64 * self.visit_ns) / scalar_concurrency;
        let matrix = (totals.matrix_multiplies as f64 * self.matrix_multiply_ns
            + totals.matrix_transfers as f64 * self.matrix_transfer_ns
            + totals.resident_loads as f64 * self.resident_load_ns
            + totals.owner_completions as f64 * self.owner_completion_ns)
            / matrix_concurrency;
        let bus = totals.device_bits as f64 / 8.0 / self.device_bytes_per_second * 1e9;
        let tiles = totals.local_bits as f64 / 8.0 / self.local_bytes_per_second * 1e9;
        // Tile traffic runs inside the lanes that compute (v1's fit: the larger term). Bus
        // traffic overlaps compute, but not perfectly: their Euclidean norm is the larger term
        // when one dominates and 41% above it when they are equal. Measured on Apple M4 Max: a
        // packet vector kernel whose compute is 0.6 of its bus time runs at the bus time;
        // ordered-chain kernels with compute near their bus time run 1.4 to 1.9 times above
        // the larger term.
        // Thread-private arrays come out of a device pool: the more bytes each thread of a
        // kernel declares, the fewer subgroups run at once (see `private_pressure_bytes`).
        let pressure = 1.0 + private_bytes as f64 / self.private_pressure_bytes;
        ((scalar + matrix).max(tiles) * pressure).hypot(bus)
    }

    /// Estimate of one execution scope: `launches` fixed overheads plus its span.
    pub fn scope_ns(&self, launches: u64, totals: &Totals, pieces: u64, private_bytes: u64, group: Group) -> Result<u64, String> {
        Self::nanoseconds(launches as f64 * self.launch_ns + self.span(totals, pieces, private_bytes, group))
    }

    /// Estimate of writing and reading back `bits` of tile storage.
    pub fn materialization_ns(&self, bits: u64) -> Result<u64, String> {
        Self::nanoseconds(2.0 * bits as f64 / 8.0 / self.local_bytes_per_second * 1e9)
    }
}
