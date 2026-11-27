//! Unified [`WallModel`] and [`TierServicePrices`] — single owner of signed
//! wall-time (`ΔT_wall < 0`) decisions across:
//!
//! 1. **3-tier decode service pricing (`SM_STAGE` vs `CE_DMA`)** via
//!    [`TierServicePrices`], written to each layer's `policy[POLICY_CE_MIN_EXPERTS]`
//!    so the on-device classifier routes small in-RAM bursts (`1..N_ce_min`) to
//!    4-CTA coalesced SM staging (`routed_stage_host`, zero CPU wakeup) and
//!    larger bursts (`>= N_ce_min` or `> DECODE_STAGE_SLOTS`) to Copy-Engine
//!    H2D DMA (`DecodeReactor`).
//! 2. **Background promotion headroom** via [`WallModel::admits_background_promote`],
//!    gating background promotions on live link / `HostDdr` contention.

use crate::expert_lines::ExpertLines;
use crate::launch::LaunchPrices;
use crate::lines::ReadLine;
use crate::regime::PathAdmission;

/// Per-flight VRAM scratch capacity (in expert slots) for coalesced
/// on-device SM staging (`routed_stage_host`, 4 CTAs x 256 threads).
/// Sized to cover `M=1` and `Verify(1..3)` host-RAM miss bursts on-device
/// without forcing a host `DecodeReactor` CE DMA round-trip when SM staging
/// matches or beats CE DMA.
pub const DECODE_STAGE_SLOTS: u32 = 16;

/// Empirical efficiency ratio of 4-CTA coalesced `uint4` `__ldcs` SM staging
/// relative to hardware Copy-Engine H2D DMA when no separate SM-staging probe
/// is supplied (RTX 5090 measured 88.3–94.3%, midpoint ~91%).
const DEFAULT_SM_STAGE_EFFICIENCY: f64 = 0.91;

/// Calibrated per-layer 3-tier decode service prices comparing coalesced SM
/// staging (`routed_stage_host` into bounded per-flight VRAM scratch, zero host
/// wakeup) against Copy-Engine H2D DMA (`DecodeReactor` + `cuMemcpyHtoDAsync`).
///
/// For `n` unique in-RAM host-tier miss experts of `record_bytes` bytes each:
/// - SM staging wall time: `n * record_bytes * inv_b_sm_stage_s_per_byte`
/// - CE DMA wall time:     `t_ce_round_trip_s + n * record_bytes * inv_b_ce_dma_s_per_byte`
///
/// CE DMA strictly beats SM staging once `n >= ce_min_experts()`:
/// `ce_min = ceil(t_ce_round_trip_s / (record_bytes * (inv_b_sm_stage - inv_b_ce_dma)))`.
/// Because `routed_stage_host` checks `unique_host_ram < ce_min`, any crossover
/// strictly greater than `stage_slots` (or `inv_b_sm_stage <= inv_b_ce_dma`)
/// returns `stage_slots + 1` so every burst fitting in `stage_slots`
/// (`1..=stage_slots`) stays purely on-device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TierServicePrices {
    /// Measured per-byte cost of Copy-Engine H2D DMA (s/byte).
    pub inv_b_ce_dma_s_per_byte: f64,
    /// Measured per-byte cost of 4-CTA coalesced SM staging (s/byte).
    pub inv_b_sm_stage_s_per_byte: f64,
    /// Measured fixed host + Copy-Engine round-trip latency (seconds):
    /// `ready_ring` wake + `DecodeReactor` + `cuMemcpyHtoDAsync` + `land_gate`.
    pub t_ce_round_trip_s: f64,
    /// Per-layer expert record size (bytes), `0` if unknown.
    pub record_bytes: u64,
    /// Available per-flight VRAM scratch slots (`DECODE_STAGE_SLOTS`, or `0` if disabled).
    pub stage_slots: u32,
    /// Whether this price model has been calibrated from hardware measurements.
    pub calibrated: bool,
}

impl Default for TierServicePrices {
    /// Pre-calibration model: allow SM staging for every burst fitting in
    /// `DECODE_STAGE_SLOTS` (`ce_min_experts() == DECODE_STAGE_SLOTS + 1`).
    fn default() -> Self {
        Self {
            inv_b_ce_dma_s_per_byte: 0.0,
            inv_b_sm_stage_s_per_byte: 0.0,
            t_ce_round_trip_s: 0.0,
            record_bytes: 0,
            stage_slots: DECODE_STAGE_SLOTS,
            calibrated: false,
        }
    }
}

impl TierServicePrices {
    /// Construct calibrated 3-tier prices from measured per-byte CE DMA rate,
    /// 4-CTA coalesced SM staging rate, fixed CE round-trip latency (seconds),
    /// per-layer expert record size (bytes), and available VRAM scratch slots.
    pub fn from_measured(
        inv_b_ce_dma_s_per_byte: f64,
        inv_b_sm_stage_s_per_byte: f64,
        t_ce_round_trip_s: f64,
        record_bytes: u64,
        stage_slots: u32,
    ) -> Self {
        Self {
            inv_b_ce_dma_s_per_byte,
            inv_b_sm_stage_s_per_byte,
            t_ce_round_trip_s,
            record_bytes,
            stage_slots,
            calibrated: true,
        }
    }

    /// Derive calibrated 3-tier prices from a measured Copy-Engine [`ReadLine`],
    /// optional measured 4-CTA SM-staging rate (s/byte), per-layer [`ExpertLines`],
    /// optional [`LaunchPrices`], and available VRAM `stage_slots`.
    pub fn from_lines(
        transfer: &ReadLine,
        sm_stage_s_per_byte: Option<f64>,
        lines: &ExpertLines,
        launch: Option<&LaunchPrices>,
        stage_slots: u32,
    ) -> Self {
        let record_bytes = lines.host.record_bytes;
        if record_bytes == 0 {
            return Self { stage_slots, ..Self::default() };
        }
        let inv_b_ce = transfer.seconds_per_byte();
        let inv_b_sm = match sm_stage_s_per_byte {
            Some(measured) if measured.is_finite() && measured > 0.0 => measured,
            _ if inv_b_ce.is_finite() && inv_b_ce > 0.0 => inv_b_ce / DEFAULT_SM_STAGE_EFFICIENCY,
            _ => (lines.gpu.link.weights.get() * 1e-9) / (record_bytes as f64),
        };
        let rt_s = (transfer.latency_seconds()
            + launch.map_or(0.0, LaunchPrices::step_round_trip_seconds))
        .max(15.0e-6);
        Self::from_measured(inv_b_ce, inv_b_sm, rt_s, record_bytes, stage_slots)
    }

    /// Minimum number of unique in-RAM host-tier miss experts `N_ce_min` at
    /// which Copy-Engine H2D DMA beats 4-CTA coalesced SM staging.
    ///
    /// Written to `policy[POLICY_CE_MIN_EXPERTS]`:
    /// - `0`: SM staging disabled (e.g. `stage_slots == 0`); every host-RAM miss
    ///   routes to `HOST_WORK` / CE DMA.
    /// - `1..=stage_slots`: bursts with `unique_host_ram < N_ce_min` stay purely
    ///   on-device via `routed_stage_host`; bursts `>= N_ce_min` route to CE DMA.
    /// - `stage_slots + 1`: SM staging is at least as fast per byte as CE DMA, so
    ///   every burst fitting in `stage_slots` (`1..=stage_slots`) stays on-device.
    pub fn ce_min_experts(&self) -> u32 {
        if self.stage_slots == 0 {
            return 0;
        }
        let keep_all_staged = self.stage_slots.saturating_add(1);
        if !self.calibrated || self.record_bytes == 0 {
            return keep_all_staged;
        }
        if !self.inv_b_ce_dma_s_per_byte.is_finite()
            || !self.inv_b_sm_stage_s_per_byte.is_finite()
            || self.inv_b_ce_dma_s_per_byte <= 0.0
            || self.inv_b_sm_stage_s_per_byte <= 0.0
        {
            return keep_all_staged;
        }
        let delta_per_byte = self.inv_b_sm_stage_s_per_byte - self.inv_b_ce_dma_s_per_byte;
        if delta_per_byte <= 0.0 {
            // SM staging is at least as fast per byte as CE DMA: keep every
            // burst that fits in `stage_slots` purely on-device.
            return keep_all_staged;
        }
        let rt_s = if self.t_ce_round_trip_s.is_finite() && self.t_ce_round_trip_s > 0.0 {
            self.t_ce_round_trip_s
        } else {
            0.0
        };
        let delta_per_expert = (self.record_bytes as f64) * delta_per_byte;
        if delta_per_expert <= 0.0 || !delta_per_expert.is_finite() {
            return keep_all_staged;
        }
        let crossover = (rt_s / delta_per_expert).ceil();
        if !crossover.is_finite() || crossover > f64::from(self.stage_slots) {
            // CE DMA does not beat SM staging for any burst `<= stage_slots`.
            keep_all_staged
        } else {
            (crossover as u32).clamp(1, self.stage_slots)
        }
    }
}

/// Unified signed wall-time decision owner (`WallModel`).
pub struct WallModel;

impl WallModel {
    /// Whether background VRAM promotions (`PriorityClass::Background`) have
    /// uncontended headroom on the link / shared `HostDdr` domain.
    ///
    /// `WriteBack` is NEVER gated by this predicate so releases and prompt
    /// leases cannot deadlock.
    pub fn admits_background_promote(
        admission: PathAdmission,
        coupled_blocking_active: bool,
        drive_in_flight: usize,
    ) -> bool {
        let defer = admission.binds()
            && (coupled_blocking_active || (admission.host_memory_binds() && drive_in_flight > 0));
        !defer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_service_prices_5090_and_3060_crossover() {
        const MIB: u64 = 1024 * 1024;
        // RTX 5090 Gen5: CE DMA = 53.0 GB/s, 4-CTA SM staging = 49.0 GB/s (92.5%),
        // CE round-trip = 25 us, 3 MiB expert -> delta = 4.85 us/expert -> ceil(25/4.85) = 6 <= STAGE_SLOTS.
        let p5090 = TierServicePrices::from_measured(
            1.0 / 53.0e9,
            1.0 / 49.0e9,
            25.0e-6,
            3 * MIB,
            DECODE_STAGE_SLOTS,
        );
        assert_eq!(p5090.ce_min_experts(), 6);

        // Exact boundary: delta = 10 us/expert, stage_slots = DECODE_STAGE_SLOTS.
        // crossover == stage_slots -> CE DMA wins at n = stage_slots -> returns stage_slots.
        let rt_eq = f64::from(DECODE_STAGE_SLOTS) * 10.0e-6;
        let p_eq_slots = TierServicePrices::from_measured(
            1.0 / 100.0e9,
            1.0 / 100.0e9 + 10.0e-6 / (MIB as f64),
            rt_eq,
            MIB,
            DECODE_STAGE_SLOTS,
        );
        assert_eq!(p_eq_slots.ce_min_experts(), DECODE_STAGE_SLOTS);

        // crossover == stage_slots + 1 -> SM staging still wins at n = stage_slots -> returns stage_slots + 1.
        let p_gt_slots = TierServicePrices::from_measured(
            1.0 / 100.0e9,
            1.0 / 100.0e9 + 10.0e-6 / (MIB as f64),
            rt_eq + 1.0e-6,
            MIB,
            DECODE_STAGE_SLOTS,
        );
        assert_eq!(p_gt_slots.ce_min_experts(), DECODE_STAGE_SLOTS + 1);

        // Large 8 MiB expert with fast 15 us CE round-trip and slower 40 GB/s SM ->
        // delta = 8 MiB * (1/40e9 - 1/53e9) = 51.4 us/expert -> CE wins at 1 expert.
        let p_large = TierServicePrices::from_measured(
            1.0 / 53.0e9,
            1.0 / 40.0e9,
            15.0e-6,
            8 * MIB,
            DECODE_STAGE_SLOTS,
        );
        assert_eq!(p_large.ce_min_experts(), 1);

        // RTX 3060 Gen4: SM staging >= CE DMA (12.8 GB/s vs 12.78 GB/s) ->
        // all bursts fitting in STAGE_SLOTS stay purely on-device (stage_slots + 1).
        let p3060 = TierServicePrices::from_measured(
            1.0 / 12.78e9,
            1.0 / 12.80e9,
            20.0e-6,
            3 * MIB,
            DECODE_STAGE_SLOTS,
        );
        assert_eq!(p3060.ce_min_experts(), DECODE_STAGE_SLOTS + 1);

        // Pre-calibration default keeps 1..=DECODE_STAGE_SLOTS on SM staging.
        assert_eq!(TierServicePrices::default().ce_min_experts(), DECODE_STAGE_SLOTS + 1);

        // Zero scratch slots disables SM staging completely.
        let disabled = TierServicePrices::from_measured(1.0 / 53.0e9, 1.0 / 49.0e9, 25.0e-6, 3 * MIB, 0);
        assert_eq!(disabled.ce_min_experts(), 0);
    }
}
