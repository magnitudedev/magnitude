//! Unified marginal-value residency allocator for multi-group expert banks.
//!
//! Differentiating per-round miss time with respect to VRAM bytes
//! `m_g = L_g · b_g · v_g` cancels `L_g · b_g` identically, giving the exact
//! marginal miss-time reduction per VRAM byte for advancing group `g` from
//! `v_g` to `v_g + 1` slots per layer:
//!
//! ```text
//! MV_g(v_g) = (Σ_{l ∈ g, v_g < E_l} ΔU_l(v_g) · s_per_byte_l) / |{l ∈ g : v_g < E_l}|
//! ```
//!
//! where `ΔU_l(v)` is the exact discrete fractional hit-rate difference
//! `D_l(r_l) · [h_l((v + 1) / E_l, r_l) − h_l(v / E_l, r_l)]` from
//! [`magnitude_cost_model::MissModel::marginal_hits`], weighted by the layer's
//! passes per decode round and its actual miss path cost `s_per_byte_l`
//! (`1 / B_miss,l`).
//!
//! Because `MissModel`'s per-rank selection probabilities `p_j` are
//! monotonically non-increasing, `MV_g(v)` is monotonically non-increasing in
//! `v` for every group. Ordering single-group slot increments by decreasing
//! `MV_g(v)` produces a single 1D monotonic coordinate `0..=max_slots` whose
//! prefix is near-optimal at every byte budget simultaneously, with O(1)
//! precomputed prefix-count lookup per group.

use super::{BankGroup, BankLayer, Placer};
use magnitude_cost_model::MissModel;
use magnitude_family_contracts::WeightScope;

/// Default relative miss cost per byte for draft vs target on discrete GPUs
/// with an async DMA side queue when measured figures are not yet available:
/// draft layers read over zero-copy SM link (~half the bandwidth of async DMA
/// on PCIe; replaced by exact measured `link_s_per_byte` vs `dma_s_per_byte`
/// once `PlanPrices` are available).
const UNPRICED_DISCRETE_DRAFT_MISS_RATIO: f64 = 2.0;

/// Precomputed 1D monotonic VRAM residency schedule across all bank groups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidencyOrder {
    /// `prefix[group][step]` = VRAM slots per layer committed for `group` at
    /// 1D bank coordinate `step` (`0..=max_slots`). O(1) lookup.
    prefix: Vec<Vec<u64>>,
    /// `Σ_g max_{l ∈ g} E_l`: the exact first coordinate at which every group
    /// holds every expert (`whole_at(max_slots) == true`).
    max_slots: u64,
}

impl ResidencyOrder {
    /// Compute the 1D marginal-value residency schedule for `groups` and
    /// `layers` at `decode_rows` verify rows, `head_steps` chained head passes,
    /// and per-layer miss cost `miss_s_per_byte` (seconds per byte, or any
    /// positive relative scale).
    ///
    /// # Panics
    /// If `groups` is empty or `miss_s_per_byte.len() != layers.len()`.
    pub fn compute(
        groups: &[BankGroup],
        layers: &[BankLayer],
        decode_rows: u64,
        head_steps: u64,
        miss_s_per_byte: &[f64],
    ) -> Self {
        assert!(!groups.is_empty(), "invariant violated: a bank has at least one group");
        assert_eq!(
            miss_s_per_byte.len(),
            layers.len(),
            "invariant violated: per-layer miss cost count must match streamed layers"
        );
        let group_widest: Vec<u64> = groups
            .iter()
            .map(|group| group.layers.max_of(|&idx| layers[idx].experts))
            .collect();
        // Single-group fast path: 1:1 identity `0..=widest`.
        if groups.len() == 1 {
            let widest = group_widest[0];
            let prefix = vec![(0..=widest).collect()];
            return Self { prefix, max_slots: widest };
        }
        let max_possible: u64 = group_widest.iter().sum();
        let cap_steps = usize::try_from(max_possible).expect("total bank slots fit usize");
        let models: Vec<MissModel> = layers
            .iter()
            .map(|layer| MissModel::of(layer.experts, layer.selected))
            .collect();
        let workloads: Vec<(f64, f64)> = layers
            .iter()
            .map(|layer| layer_workload(layer.scope, decode_rows, head_steps))
            .collect();
        let mut current = vec![0u64; groups.len()];
        let mut prefix: Vec<Vec<u64>> = (0..groups.len())
            .map(|_| {
                let mut col = Vec::with_capacity(cap_steps + 1);
                col.push(0);
                col
            })
            .collect();
        let group_mv = |g: usize, v: u64| -> f64 {
            let mut active = 0u64;
            let mut sum = 0.0f64;
            for &layer_idx in groups[g].layers.iter() {
                let layer = &layers[layer_idx];
                if v < layer.experts {
                    active += 1;
                    let (passes, rows) = workloads[layer_idx];
                    let delta_u = passes * models[layer_idx].marginal_hits(v, rows);
                    let cost = miss_s_per_byte[layer_idx];
                    sum += delta_u * cost;
                }
            }
            if active == 0 { 0.0 } else { sum / active as f64 }
        };
        let mut next_mv: Vec<f64> = (0..groups.len())
            .map(|g| if current[g] < group_widest[g] { group_mv(g, 0) } else { -1.0 })
            .collect();
        let mut total_steps = 0usize;
        while current.iter().zip(&group_widest).any(|(&c, &w)| c < w) {
            // Find the maximum marginal value per VRAM byte across all
            // unfinished groups, then advance every group tied at that
            // maximum together (so identical-workload mixed-quant target
            // groups advance 1:1 in lockstep with `max_slots == widest`,
            // while groups with distinct `MV` advance one at a time).
            let mut best_mv = -1.0f64;
            for g in 0..groups.len() {
                if current[g] < group_widest[g] && next_mv[g] > best_mv {
                    best_mv = next_mv[g];
                }
            }
            let tol = 1e-12 * best_mv.abs().max(1e-30);
            for g in 0..groups.len() {
                if current[g] < group_widest[g] && (best_mv - next_mv[g]) <= tol {
                    current[g] += 1;
                    next_mv[g] = if current[g] < group_widest[g] {
                        group_mv(g, current[g])
                    } else {
                        -1.0
                    };
                }
            }
            for (g, col) in prefix.iter_mut().enumerate() {
                col.push(current[g]);
            }
            total_steps += 1;
        }
        let max_slots = u64::try_from(total_steps).expect("total steps fit u64");
        debug_assert!(
            prefix
                .iter()
                .zip(&group_widest)
                .all(|(col, &w)| col.len() == total_steps + 1 && col[total_steps] == w),
            "invariant violated: ResidencyOrder must end at every group's widest expert count"
        );
        Self { prefix, max_slots }
    }

    /// Default unpriced schedule from the bank's topology alone (replaced with
    /// measured per-layer bandwidths once `PlanPrices` are available).
    pub fn default_for(
        groups: &[BankGroup],
        layers: &[BankLayer],
        decode_rows: u64,
        head_steps: u64,
        placer: Placer,
        tier_copy_dma: bool,
    ) -> Self {
        let costs = default_miss_s_per_byte(layers, placer, tier_copy_dma);
        Self::compute(groups, layers, decode_rows, head_steps, &costs)
    }

    /// Maximum 1D coordinate where every group holds all of its experts.
    pub fn max_slots(&self) -> u64 {
        self.max_slots
    }

    /// O(1) lookup of VRAM slots per layer committed for `group` at bank
    /// coordinate `slots`.
    pub fn group_slots(&self, group: usize, slots: u64) -> u64 {
        let index = usize::try_from(slots.min(self.max_slots)).expect("slots fit usize");
        self.prefix[group][index]
    }

    /// O(1) lookup of VRAM slots committed for `layer` at bank coordinate
    /// `slots`.
    pub fn layer_slots(&self, layer: &BankLayer, slots: u64) -> u64 {
        self.group_slots(layer.group, slots).min(layer.experts)
    }
}

/// `(passes_per_round, rows_per_pass)` for a streamed layer's scope:
/// - Target layers run 1 verify pass of `decode_rows` rows per round.
/// - Separate block-diffusion draft layers (`DraftSublayer`, e.g. DSpark /
///   DFlash) run 1 parallel block pass of `(decode_rows - 1).max(1)` rows per
///   round.
/// - Embedded causal MTP head layers (`HeadSublayer`) run `head_steps` chained
///   single-row passes per round.
fn layer_workload(scope: WeightScope, decode_rows: u64, head_steps: u64) -> (f64, f64) {
    match scope {
        WeightScope::DraftSublayer(_) => (1.0, decode_rows.saturating_sub(1).max(1) as f64),
        WeightScope::HeadSublayer(_) => (head_steps.max(1) as f64, 1.0),
        _ => (1.0, decode_rows.max(1) as f64),
    }
}

/// Relative unpriced miss cost per byte per layer.
pub(super) fn default_miss_s_per_byte(
    layers: &[BankLayer],
    placer: Placer,
    tier_copy_dma: bool,
) -> Vec<f64> {
    layers
        .iter()
        .map(|layer| match placer {
            Placer::Bank => 1.0,
            Placer::Tier => {
                if tier_copy_dma && layer.is_draft() {
                    UNPRICED_DISCRETE_DRAFT_MISS_RATIO
                } else {
                    1.0
                }
            }
        })
        .collect()
}
