//! Single-owner `DecodePassPlan (G, K, F)` — unified decode pass planner,
//! pricing scorer, shared batch row-grant coordinator, and single-settlement
//! outcome contract.
//!
//! # Contract
//! - **One scorer, two decision times:**
//!   1. Pre-draft ([`DecodePassScorer::choose_generation_budget`]): chooses a
//!      legal generation budget `G` from population evidence, subject to
//!      scheduler-offered capacity.
//!   2. Post-proposal ([`DecodePassScorer::seal_plan`]): once proposals exist
//!      and a shared row grant is reserved, chooses `(K, F)` under
//!      `0 <= K + F <= actual G` and `1 + K + F <= reserved_capacity`.
//!      `G` may strictly exceed `K + F` (unused proposals still paid draft
//!      time; `G == K + F` is NOT an execution invariant).
//! - **Drafter polymorphism ([`DrafterSpec`]):**
//!   - Embedded [`TwoRowDecode`] supplies an existing chain (`EmbeddedChain`)
//!     without forcing a new drafter run.
//!   - Block (DFlash/DSpark) and autoregressive (MTP) drafters expose their
//!     own feasible widths and marginal costs.
//! - **Shared row grant ([`seal_batch_plans`]):** reserves the shared row
//!   grant across competing requests before sealing, repricing truncated
//!   requests on their smaller offer so batched launches never overbook.
//! - **Pricing accuracy:**
//!   - Per-row compute for `1 + K + F`.
//!   - Immediate compulsory I/O is the distinct absent union of the exact
//!     `1 + K` rows ONLY; tail-only (`F`) misses skip immediate demand I/O.
//!   - Speculative traffic, cache displacement, and DDR contention are charged
//!     separately.
//!   - Survival products require conditional probabilities (never multiplying
//!     cumulative survival twice).
//!   - `F` credit applies ONLY to correctly dated useful bytes that eliminate
//!     exposed future wait beyond existing carryover; first rejection, EOS,
//!     and length limits suppress the bonus.
//! - **Outcome contract ([`PassOutcomeTracker`]):** carries request generation
//!   and pass ID, actual committed prefix (`Discarded | Committed(prefix)`),
//!   and separately judged branch predictions; cancel-before-completion and
//!   verdict-before-feed both settle once.
//! - **Invariance:** any `(K, F)` plan produces the exact same output tokens
//!   and committed state as `(K, 0)`.

use magnitude_cost_model::LaunchPrices;
use magnitude_executor::{Guess, GuessKind, RequestId, Rows, StreamedVerifyPrior, TokenId, TwoRowDecode};
use std::collections::{BTreeMap, BTreeSet};

/// Unique identity of a single decode pass: binds a request, its monotonic
/// request generation (incremented on reset/reuse/eviction), and its pass ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PassId {
    pub request: RequestId,
    pub generation: u64,
    pub pass: u64,
}

/// Polymorphic drafter capability and launch-cost descriptor.
#[derive(Clone, Debug, PartialEq)]
pub enum DrafterSpec {
    /// Embedded [`TwoRowDecode`] supplies an existing chain of `chain_len`
    /// guesses at zero additional drafter launch cost — never forces a new
    /// drafter run.
    EmbeddedChain { chain_len: usize },
    /// Block drafter (e.g. DFlash / DSpark): discrete feasible block widths
    /// with per-width launch cost in seconds.
    Block {
        /// `(width, draft_seconds)` entries for `width >= 1`.
        widths: Vec<(usize, f64)>,
    },
    /// Autoregressive / chained drafter (e.g. MTP): feasible widths
    /// `1..=max_width` with fixed overhead + marginal per-step draft cost.
    Autoregressive {
        max_width: usize,
        fixed_seconds: f64,
        marginal_seconds: f64,
    },
}

impl DrafterSpec {
    /// Construct an [`Self::EmbeddedChain`] spec directly from an existing
    /// [`TwoRowDecode`] chain without forcing a new drafter run.
    pub fn from_two_row(two_row: &TwoRowDecode) -> Self {
        Self::EmbeddedChain {
            chain_len: two_row.chain_len(),
        }
    }

    /// Legal generation widths `G <= max_g` for this drafter.
    pub fn feasible_widths(&self, max_g: usize) -> Vec<usize> {
        match self {
            Self::EmbeddedChain { chain_len } => vec![(*chain_len).min(max_g)],
            Self::Block { widths } => {
                let mut set = BTreeSet::new();
                set.insert(0);
                for &(w, _) in widths {
                    if w > 0 && w <= max_g {
                        set.insert(w);
                    }
                }
                set.into_iter().collect()
            }
            Self::Autoregressive { max_width, .. } => (0..=(*max_width).min(max_g)).collect(),
        }
    }

    /// Whether selecting generation budget `g` requires launching a drafter.
    /// `EmbeddedChain` always returns `false`.
    pub fn needs_draft_launch(&self, g: usize) -> bool {
        match self {
            Self::EmbeddedChain { .. } => false,
            Self::Block { .. } | Self::Autoregressive { .. } => g > 0,
        }
    }

    /// Drafter launch cost `D(g)` in seconds (`0.0` for `EmbeddedChain` or `g == 0`).
    pub fn draft_seconds(&self, g: usize) -> f64 {
        if g == 0 {
            return 0.0;
        }
        match self {
            Self::EmbeddedChain { .. } => 0.0,
            Self::Block { widths } => widths
                .iter()
                .find(|&&(w, _)| w == g)
                .map_or(0.0, |&(_, secs)| secs.max(0.0)),
            Self::Autoregressive {
                fixed_seconds,
                marginal_seconds,
                ..
            } => (fixed_seconds.max(0.0) + g as f64 * marginal_seconds.max(0.0)).max(0.0),
        }
    }
}

/// Typed acceptance profile that makes multiplying cumulative survival twice
/// impossible by construction.
#[derive(Clone, Debug, PartialEq)]
pub enum AcceptanceProfile {
    /// Per-depth conditional probabilities `p[j] = P(depth j stands | depths < j stood)`.
    Conditional(Vec<f64>),
    /// Monotone non-increasing cumulative survival `S[j] = P(depths 0..=j all stand)`.
    Cumulative(Vec<f64>),
}

impl AcceptanceProfile {
    /// Cumulative survival `S[j] = P(depths 0..=j all stand)` for `j in 0..len`.
    /// Computed via product ONCE for `Conditional`, and clamped monotonically
    /// (NEVER multiplied again) for `Cumulative`.
    pub fn cumulative_survival(&self) -> Vec<f64> {
        match self {
            Self::Conditional(cond) => {
                let mut prod = 1.0f64;
                cond.iter()
                    .map(|&p| {
                        let p = if p.is_finite() { p.clamp(0.0, 1.0) } else { 0.0 };
                        prod *= p;
                        prod
                    })
                    .collect()
            }
            Self::Cumulative(cum) => {
                let mut bound = 1.0f64;
                cum.iter()
                    .map(|&s| {
                        let s = if s.is_finite() { s.clamp(0.0, 1.0) } else { 0.0 };
                        bound = bound.min(s);
                        bound
                    })
                    .collect()
            }
        }
    }

    /// Conditional probabilities `p[j] = P(depth j stands | depths < j stood)`.
    pub fn conditional_probs(&self) -> Vec<f64> {
        match self {
            Self::Conditional(cond) => cond
                .iter()
                .map(|&p| if p.is_finite() { p.clamp(0.0, 1.0) } else { 0.0 })
                .collect(),
            Self::Cumulative(_) => {
                let cum = self.cumulative_survival();
                let mut prev = 1.0f64;
                cum.into_iter()
                    .map(|s| {
                        let p = if prev > 0.0 { (s / prev).clamp(0.0, 1.0) } else { 0.0 };
                        prev = s;
                        p
                    })
                    .collect()
            }
        }
    }

    /// Length of the profile (number of proposal depths covered).
    pub fn len(&self) -> usize {
        match self {
            Self::Conditional(v) | Self::Cumulative(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Build a conditional acceptance profile of length `len` from live
    /// measured runtime data:
    /// 1. Watcher per-depth conditional stand rates (`prior_accepts[d - 1]`),
    ///    or `prior_stand` at depth 1 when no verify depth-1 rate is measured yet.
    /// 2. This generation's single-settlement [`PassOutcomeTracker`] per-depth
    ///    empirical conditional rate (`matched / judged`).
    /// 3. For any deeper unmeasured depth, carries forward the last measured
    ///    shallower conditional rate (`1.0` before any depth is measured),
    ///    matching `magnitude_cost_model::Regime::verify_emitted`.
    pub fn from_empirical(
        len: usize,
        prior_accepts: Option<&[Option<f64>]>,
        prior_stand: Option<f64>,
        tracker: &PassOutcomeTracker,
    ) -> Self {
        let mut bound = 1.0f64;
        let mut cond = Vec::with_capacity(len);
        for idx in 0..len {
            let depth = idx + 1;
            let measured = prior_accepts
                .and_then(|accepts| accepts.get(idx).copied().flatten())
                .or(if idx == 0 { prior_stand } else { None })
                .or_else(|| tracker.conditional_rate(depth));
            let p = match measured {
                Some(rate) => {
                    let clamped = if rate.is_finite() {
                        rate.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    bound = clamped;
                    clamped
                }
                None => bound,
            };
            cond.push(p);
        }
        Self::Conditional(cond)
    }
}

/// Request horizon and stop/EOS boundaries that constrain useful `F` credit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HorizonLimits {
    /// Maximum future tokens remaining after the current exact row's emission
    /// (from `max_tokens` and `context_limit`). `0` means the current exact
    /// row is the final token — no future pass exists, so `F` bonus is 0.
    pub remaining_tokens: usize,
    /// Optional 0-based proposal index at which EOS / a stop token is predicted
    /// or reached (`Some(0)` means the first proposal is EOS). Any front row
    /// at or past `eos_at` has zero future pass to benefit, suppressing `F` bonus.
    pub eos_at: Option<usize>,
}

impl HorizonLimits {
    /// Unlimited horizon with no imminent EOS.
    pub const UNBOUNDED: Self = Self {
        remaining_tokens: usize::MAX,
        eos_at: None,
    };

    /// Maximum verify width `K` that can emit tokens: bounded by both
    /// `remaining_tokens` and `eos_at` (a stop token at index `e` is never
    /// accepted as a draft match, so at most `e` proposals before it can commit).
    pub fn effective_verify_horizon(self) -> usize {
        self.remaining_tokens.min(self.eos_at.unwrap_or(usize::MAX))
    }

    /// Maximum proposal depth `d >= 1` (1-based) that still has a future pass
    /// to benefit from foresight bytes.
    pub fn effective_foresight_horizon(self) -> usize {
        self.effective_verify_horizon()
    }
}

/// Per-row compute cost `C(m)` for `m = 1 + K + F` total rows.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputePricing {
    /// Resident compute seconds for `m = 1, 2, ...` rows (`by_rows[m - 1]`).
    /// Past the table end, extrapolates using the last marginal step.
    pub by_rows: Vec<f64>,
}

impl ComputePricing {
    /// Compute seconds for `rows >= 1` total rows (`1 + K + F`).
    pub fn seconds(&self, rows: usize) -> f64 {
        if rows == 0 || self.by_rows.is_empty() {
            return 0.0;
        }
        let idx = rows - 1;
        if let Some(&secs) = self.by_rows.get(idx) {
            return secs.max(0.0);
        }
        let last_idx = self.by_rows.len() - 1;
        let last = self.by_rows[last_idx].max(0.0);
        let slope = if last_idx > 0 {
            (last - self.by_rows[last_idx - 1].max(0.0)).max(0.0)
        } else {
            0.0
        };
        last + (idx - last_idx) as f64 * slope
    }
}

/// Immediate compulsory I/O and speculative tail I/O pricing.
///
/// Immediate compulsory I/O is ALWAYS the distinct absent union of the exact
/// `1 + K` rows ONLY. Tail-only (`F`) misses skip immediate demand I/O and
/// are priced separately as speculative traffic, cache displacement, and DDR
/// contention.
#[derive(Clone, Debug, PartialEq)]
pub enum RowMissPricing {
    /// Exact per-row absent expert sets for rows `0..(1 + K + F)`:
    /// - Compulsory demand I/O = `|⋃_{r=0..=K} per_row_absent[r]| * demand_seconds_per_miss`.
    /// - Tail-only absent set = `(⋃_{r=K+1..=K+F} per_row_absent[r]) \ (⋃_{r=0..=K} per_row_absent[r])`.
    /// - Speculative traffic, displacement, and DDR contention are charged
    ///   strictly on the tail-only absent experts admitted for read-ahead.
    ExactAbsentSets {
        /// Absent experts per row in row order (`0` = exact, `1..=K` = verify, `K+1..=K+F` = front).
        per_row_absent: Vec<BTreeSet<u32>>,
        /// Demand-path seconds per distinct compulsory miss.
        demand_seconds_per_miss: f64,
        /// Speculative path seconds per admitted tail-only miss.
        speculative_seconds_per_miss: f64,
        /// Expected cache displacement penalty in seconds per admitted tail-only miss.
        displacement_seconds_per_miss: f64,
        /// Host DDR / shared memory contention penalty in seconds per admitted tail-only miss.
        ddr_contention_seconds_per_miss: f64,
    },
    /// Population / statistical miss rates when exact per-row routes are not yet known
    /// (used at Decision Time 1 before drafting, or when exact sets are unavailable).
    ExpectedRates {
        /// Expected compulsory misses on the exact row (`row 0`).
        exact_row_misses: f64,
        /// Expected marginal distinct compulsory misses added by verify row `j` (`j = 1..=K`).
        verify_marginal_misses: Vec<f64>,
        /// Expected distinct tail-only misses introduced by front row `m` (`m = 1..=F`).
        front_tail_misses: Vec<f64>,
        /// Demand-path seconds per distinct compulsory miss.
        demand_seconds_per_miss: f64,
        /// Speculative path seconds per admitted tail-only miss.
        speculative_seconds_per_miss: f64,
        /// Expected cache displacement penalty in seconds per admitted tail-only miss.
        displacement_seconds_per_miss: f64,
        /// Host DDR / shared memory contention penalty in seconds per admitted tail-only miss.
        ddr_contention_seconds_per_miss: f64,
    },
}

impl RowMissPricing {
    /// Count of distinct compulsory misses for the exact `1 + K` rows ONLY.
    /// Tail-only (`F`) misses are strictly excluded.
    pub fn compulsory_miss_count(&self, k: usize) -> f64 {
        match self {
            Self::ExactAbsentSets { per_row_absent, .. } => {
                let mut union = BTreeSet::new();
                for set in per_row_absent.iter().take(1 + k) {
                    union.extend(set.iter().copied());
                }
                union.len() as f64
            }
            Self::ExpectedRates {
                exact_row_misses,
                verify_marginal_misses,
                ..
            } => {
                let last = verify_marginal_misses.last().copied().unwrap_or(0.0).max(0.0);
                let extra: f64 = (0..k)
                    .map(|j| verify_marginal_misses.get(j).copied().unwrap_or(last).max(0.0))
                    .sum();
                exact_row_misses.max(0.0) + extra
            }
        }
    }

    /// Count of distinct tail-only misses introduced by the `F` front rows
    /// (`rows K+1..=K+F`) that are NOT already in the compulsory `1 + K` union.
    pub fn tail_only_miss_count(&self, k: usize, f: usize) -> f64 {
        if f == 0 {
            return 0.0;
        }
        match self {
            Self::ExactAbsentSets { per_row_absent, .. } => {
                let mut compulsory = BTreeSet::new();
                for set in per_row_absent.iter().take(1 + k) {
                    compulsory.extend(set.iter().copied());
                }
                let mut tail_only = BTreeSet::new();
                for set in per_row_absent.iter().skip(1 + k).take(f) {
                    for &expert in set {
                        if !compulsory.contains(&expert) {
                            tail_only.insert(expert);
                        }
                    }
                }
                tail_only.len() as f64
            }
            Self::ExpectedRates {
                front_tail_misses, ..
            } => {
                let last = front_tail_misses.last().copied().unwrap_or(0.0).max(0.0);
                (0..f)
                    .map(|m| front_tail_misses.get(m).copied().unwrap_or(last).max(0.0))
                    .sum()
            }
        }
    }

    /// Immediate compulsory demand I/O seconds for the `1 + K` rows ONLY.
    pub fn compulsory_io_seconds(&self, k: usize) -> f64 {
        let rate = match self {
            Self::ExactAbsentSets {
                demand_seconds_per_miss,
                ..
            }
            | Self::ExpectedRates {
                demand_seconds_per_miss,
                ..
            } => demand_seconds_per_miss.max(0.0),
        };
        self.compulsory_miss_count(k) * rate
    }

    /// Separate `(speculative_traffic, displacement, ddr_contention)` seconds
    /// charged for the `F` tail rows.
    pub fn tail_side_costs(&self, k: usize, f: usize) -> (f64, f64, f64) {
        let tail_misses = self.tail_only_miss_count(k, f);
        if tail_misses <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let (spec, disp, ddr) = match self {
            Self::ExactAbsentSets {
                speculative_seconds_per_miss,
                displacement_seconds_per_miss,
                ddr_contention_seconds_per_miss,
                ..
            }
            | Self::ExpectedRates {
                speculative_seconds_per_miss,
                displacement_seconds_per_miss,
                ddr_contention_seconds_per_miss,
                ..
            } => (
                speculative_seconds_per_miss.max(0.0),
                displacement_seconds_per_miss.max(0.0),
                ddr_contention_seconds_per_miss.max(0.0),
            ),
        };
        (tail_misses * spec, tail_misses * disp, tail_misses * ddr)
    }
}

/// Foresight credit model for `F` tail rows: credits `F` ONLY for correctly
/// dated useful bytes that eliminate exposed future wait BEYOND existing
/// carryover.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrontCreditModel {
    /// Exposed future demand wait (in seconds) that front row `m` (0-indexed:
    /// `m = 0` is the first front row at depth `K + 1`) would eliminate if its
    /// branch survives and a future pass occurs.
    pub exposed_future_wait_by_front: Vec<f64>,
    /// Exposed future demand wait (in seconds) ALREADY covered by existing
    /// carryover / in-flight prefetch at front row `m`. Subtracted before
    /// crediting `F`.
    pub carryover_covered_by_front: Vec<f64>,
}

impl FrontCreditModel {
    /// Total exposed future demand wait beyond carryover across all potential
    /// future passes up to the horizon (for a consistent current+future time
    /// horizon: `F` reduces this wait from `total_exposed_wait` down to
    /// `total_exposed_wait - credit_seconds`, never making pass time negative).
    pub fn total_exposed_wait(
        &self,
        cum_survival: &[f64],
        k: usize,
        horizon: HorizonLimits,
    ) -> f64 {
        let max_f = self.exposed_future_wait_by_front.len();
        self.credit_seconds(cum_survival, k, max_f, horizon)
    }

    /// Net useful future wait saved by front rows `1..=F` given:
    /// - cumulative survival `cum_survival` (`cum_survival[d - 1] = S_d`),
    /// - `k` verify rows, `f` front rows,
    /// - `horizon` (`remaining_tokens`, `eos_at`).
    ///
    /// Returns `0.0` whenever `f == 0`, when the branch fails before the front
    /// row (e.g. first rejection `S[0] == 0.0`), when EOS/length limits
    /// suppress future passes, or when existing carryover already covers the
    /// exposed wait.
    pub fn credit_seconds(
        &self,
        cum_survival: &[f64],
        k: usize,
        f: usize,
        horizon: HorizonLimits,
    ) -> f64 {
        if f == 0 {
            return 0.0;
        }
        let max_useful_depth = horizon.effective_foresight_horizon();
        if max_useful_depth == 0 {
            return 0.0;
        }
        let mut total = 0.0f64;
        for m in 0..f {
            // 1-based depth of this front row among proposals: `depth = k + m + 1`.
            let depth = k + m + 1;
            // EOS or length limit suppresses any front row at or past the horizon.
            if depth > max_useful_depth {
                break;
            }
            // Branch survival through this front row: requires every preceding
            // proposal (including depth 1 and all K verify rows) to stand.
            let survival = cum_survival.get(depth - 1).copied().unwrap_or(0.0);
            if survival <= 0.0 {
                break;
            }
            let exposed = self
                .exposed_future_wait_by_front
                .get(m)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let carryover = self
                .carryover_covered_by_front
                .get(m)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let net_useful = (exposed - carryover).max(0.0);
            total += survival * net_useful;
        }
        total
    }
}

/// Complete pricing inputs shared by Decision Time 1 (pre-draft) and Decision
/// Time 2 (post-proposal) in the single scorer.
#[derive(Clone, Debug, PartialEq)]
pub struct PassCostModel {
    pub compute: ComputePricing,
    pub io: RowMissPricing,
    pub front_credit: FrontCreditModel,
    /// Fixed host turn overhead in seconds when `K > 0` (verify pass waits on verdict).
    pub verify_host_overhead_seconds: f64,
    /// Per-draft-step head overhead in seconds inside a verify turn when `K > 0`.
    pub verify_host_overhead_per_step_seconds: f64,
    /// Host turn overhead in seconds when `K == 0` (plain or front-only pass).
    pub non_verify_host_overhead_seconds: f64,
}

impl PassCostModel {
    /// Build a real measured [`PassCostModel`] from a live [`StreamedVerifyPrior`]
    /// snapshot produced by `ExecutorDomain::streamed_verify_prior`:
    /// - `compute.by_rows`: live resident compute `C(1 + m)` (`compute_seconds`,
    ///   falling back to `verify_seconds` if a fixture omitted `compute_seconds`).
    /// - `io`: live compulsory demand I/O `I/O(k)` (`io_seconds`) and trailing
    ///   front waste (`front_waste_per_row`), both in seconds (`*_per_miss = 1.0`).
    /// - `front_credit`: multi-row diminishing stall savings per front row
    ///   (`front_stall_saved_by_row`, or derived via
    ///   `magnitude_cost_model::diminishing_front_stall_saved` from
    ///   `front_stall_saved_per_row` when the per-row vector is absent), less
    ///   `carryover_covered_by_front`. Gated ONLY on actual exposed future
    ///   stall (NOT on `io > compute`), so both path-bound (`D > C`) and
    ///   compute-heavier UMA (`0 < D < C`, e.g. Sparky GB10 paying synchronous
    ///   SSD stall) credit `F` whenever `S * saved > ΔC + waste`, while pure
    ///   resident / zero-stall GPU-bound regimes (`saved == 0`) still get `F = 0`.
    /// - `verify_host_overhead_seconds`: measured host turn after a verify pass.
    pub fn from_streamed_prior(prior: &StreamedVerifyPrior) -> Self {
        let by_rows = if !prior.compute_seconds.is_empty() {
            prior.compute_seconds.clone()
        } else if !prior.verify_seconds.is_empty() {
            prior.verify_seconds.clone()
        } else {
            vec![1.0]
        };
        let exact_row_misses = prior.io_seconds.first().copied().unwrap_or(0.0).max(0.0);
        let verify_marginal_misses: Vec<f64> = prior
            .io_seconds
            .windows(2)
            .map(|w| (w[1] - w[0]).max(0.0))
            .collect();
        let has_io = exact_row_misses > 0.0
            || verify_marginal_misses.iter().any(|&m| m > 0.0);
        let has_tail_waste = prior.front_waste_per_row.iter().any(|&w| w > 0.0);
        let max_front_cap = magnitude_executor::CHAINED_ROWS.saturating_sub(1);
        let exposed_future_wait_by_front: Vec<f64> = if !prior.front_stall_saved_by_row.is_empty() {
            prior
                .front_stall_saved_by_row
                .iter()
                .take(max_front_cap)
                .map(|&s| if s.is_finite() { s.max(0.0) } else { 0.0 })
                .collect()
        } else if prior.front_stall_saved_per_row > 0.0 {
            let max_front = prior
                .front_waste_per_row
                .len()
                .max(by_rows.len().saturating_sub(1))
                .clamp(1, max_front_cap);
            magnitude_cost_model::diminishing_front_stall_saved(
                prior.front_stall_saved_per_row,
                max_front,
            )
        } else {
            Vec::new()
        };
        let has_front_credit = exposed_future_wait_by_front.iter().any(|&s| s > 0.0);
        let front_credit = if has_front_credit {
            let carryover_covered_by_front: Vec<f64> = prior
                .carryover_covered_by_front
                .iter()
                .take(exposed_future_wait_by_front.len())
                .map(|&c| if c.is_finite() { c.max(0.0) } else { 0.0 })
                .collect();
            FrontCreditModel {
                exposed_future_wait_by_front,
                carryover_covered_by_front,
            }
        } else {
            FrontCreditModel::default()
        };
        Self {
            compute: ComputePricing { by_rows },
            io: RowMissPricing::ExpectedRates {
                exact_row_misses,
                verify_marginal_misses,
                front_tail_misses: prior.front_waste_per_row.clone(),
                demand_seconds_per_miss: if has_io { 1.0 } else { 0.0 },
                speculative_seconds_per_miss: if has_tail_waste { 1.0 } else { 0.0 },
                displacement_seconds_per_miss: 0.0,
                ddr_contention_seconds_per_miss: 0.0,
            },
            front_credit,
            verify_host_overhead_seconds: prior.verify_host_overhead_seconds.max(0.0),
            verify_host_overhead_per_step_seconds: prior.verify_host_overhead_per_step_seconds.max(0.0),
            non_verify_host_overhead_seconds: 0.0,
        }
    }

    /// Build a resident/analytical [`PassCostModel`] from a load's [`LaunchPrices`]
    /// snapshot for `1..=max_rows` rows at context `depth`.
    pub fn from_launch_prices(prices: &LaunchPrices, max_rows: usize, depth: u32) -> Option<Self> {
        let cap = max_rows.max(1);
        let banked = prices.is_banked();
        let mut by_rows = Vec::with_capacity(cap);
        for rows in 1..=cap {
            let secs = if rows == 1 {
                prices.plain_seconds(1, depth).ok()
            } else {
                prices.verify_seconds(1, rows as u32, depth).ok()
            }
            .map(|f| if banked { f.streamed } else { f.resident })?;
            if !secs.is_finite() || secs <= 0.0 {
                return None;
            }
            by_rows.push(secs);
        }
        Some(Self {
            compute: ComputePricing { by_rows },
            io: RowMissPricing::ExpectedRates {
                exact_row_misses: 0.0,
                verify_marginal_misses: Vec::new(),
                front_tail_misses: Vec::new(),
                demand_seconds_per_miss: 0.0,
                speculative_seconds_per_miss: 0.0,
                displacement_seconds_per_miss: 0.0,
                ddr_contention_seconds_per_miss: 0.0,
            },
            front_credit: FrontCreditModel::default(),
            verify_host_overhead_seconds: prices.step_round_trip_seconds().max(0.0),
            verify_host_overhead_per_step_seconds: 0.0,
            non_verify_host_overhead_seconds: 0.0,
        })
    }

    /// Unpriced unit-compute model used only when no runtime [`StreamedVerifyPrior`]
    /// or [`LaunchPrices`] has been observed (e.g. isolated unit tests without an
    /// executor domain).
    pub fn unit_compute(max_rows: usize) -> Self {
        Self {
            compute: ComputePricing {
                by_rows: vec![1.0; max_rows.max(1)],
            },
            io: RowMissPricing::ExpectedRates {
                exact_row_misses: 0.0,
                verify_marginal_misses: Vec::new(),
                front_tail_misses: Vec::new(),
                demand_seconds_per_miss: 0.0,
                speculative_seconds_per_miss: 0.0,
                displacement_seconds_per_miss: 0.0,
                ddr_contention_seconds_per_miss: 0.0,
            },
            front_credit: FrontCreditModel::default(),
            verify_host_overhead_seconds: 0.0,
            verify_host_overhead_per_step_seconds: 0.0,
            non_verify_host_overhead_seconds: 0.0,
        }
    }
}

/// Detailed cost breakdown of a scored `(G, K, F)` candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassPricing {
    /// Draft cost `D(G)` paid for the generated budget `G` (even if `K + F < G`).
    pub draft_seconds: f64,
    /// Resident compute `C(1 + K + F)` for all `1 + K + F` rows.
    pub compute_seconds: f64,
    /// Immediate compulsory demand I/O for the distinct absent union of `1 + K` rows ONLY.
    pub compulsory_io_seconds: f64,
    /// Separate speculative traffic cost of `F`-only tail misses.
    pub speculative_traffic_seconds: f64,
    /// Separate cache displacement penalty of `F`-only tail misses.
    pub displacement_seconds: f64,
    /// Separate host DDR contention penalty of `F`-only tail misses.
    pub ddr_contention_seconds: f64,
    /// Host turn overhead.
    pub host_overhead_seconds: f64,
    /// Useful future-wait credit from `F` beyond carryover (0 when suppressed by rejection/EOS).
    pub front_credit_seconds: f64,
    /// Net effective round seconds (`>= 1e-9`).
    pub total_round_seconds: f64,
    /// Expected tokens committed by this pass (`1 + Σ_{j=0..K} S_j`; `F` emits 0 tokens).
    pub expected_tokens: f64,
    /// Expected throughput (`expected_tokens / total_round_seconds`).
    pub rate_tps: f64,
}

/// Decision Time 1 output: legal generation budget `G` chosen from population
/// evidence before drafting, subject to scheduler-offered capacity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftBudgetPlan {
    pub pass_id: PassId,
    pub generated: usize,
    pub needs_draft_launch: bool,
    pub draft_seconds: f64,
    pub projected_verify: usize,
    pub projected_front: usize,
    pub projected_pricing: PassPricing,
}

/// Decision Time 2 output: single-owner sealed decode pass plan `(G, K, F)`.
///
/// Enforced invariants:
/// - `0 <= verify + front <= generated` (`0 <= K + F <= actual G`)
/// - `1 + verify + front + idle_front <= reserved_capacity`
/// - `idle_front` holds physical padding rows (e.g. `FrontGrant::Exactly(4)`
///   with 1 existing guess -> `front = 1, idle_front = 3`), kept strictly
///   distinct from productive guesses `front` so `total_rows()` and
///   `build_rows()` reconstruct the exact queued launch and charge the
///   executed row class without crediting idle rows.
/// - `generated` may strictly exceed `verify + front` (unused proposals still
///   paid `pricing.draft_seconds`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodePassPlan {
    pub pass_id: PassId,
    /// Actual proposals generated (`actual G`).
    pub generated: usize,
    /// Acceptable verify rows (`K`).
    pub verify: usize,
    /// Productive foresight-only front tail rows (`F`).
    pub front: usize,
    /// Physical padding / idle front rows past the existing guess chain (`0`
    /// unless a queued exact-depth launch pads past the chain).
    pub idle_front: usize,
    /// Reserved row capacity granted to this request (`>= 1 + K + F + idle_front`).
    pub reserved_capacity: usize,
    /// Full cost breakdown for this sealed plan.
    pub pricing: PassPricing,
}

impl DecodePassPlan {
    /// Total physical rows executed by this pass (`1 + K + F + idle_front`).
    pub const fn total_rows(&self) -> usize {
        1 + self.verify + self.front + self.idle_front
    }

    /// Productive guessed rows (`K + F`), excluding physical idle padding.
    pub const fn productive_guesses(&self) -> usize {
        self.verify + self.front
    }

    /// Unused proposals (`actual G - (K + F)`). May be strictly positive.
    pub const fn unused_proposals(&self) -> usize {
        self.generated - (self.verify + self.front)
    }

    /// Build the executor [`Rows`] for this sealed plan from `proposals`
    /// (`proposals.len() >= K + F`), tagging `F` productive tail rows with
    /// `front_kind` and appending `idle_front` physical padding rows.
    pub fn build_rows(&self, proposals: &[TokenId], front_kind: GuessKind) -> Result<Rows, String> {
        let used = self.verify + self.front;
        if used > self.generated || self.total_rows() > self.reserved_capacity {
            return Err("sealed DecodePassPlan violates capacity or generated bounds".into());
        }
        if proposals.len() < used {
            return Err(format!(
                "proposals length {} is shorter than sealed K + F = {used}",
                proposals.len()
            ));
        }
        let acceptable = proposals[..self.verify].to_vec();
        let guesses = proposals[self.verify..used]
            .iter()
            .copied()
            .map(|token| Guess {
                token,
                kind: front_kind,
            })
            .collect();
        Ok(Rows::pass(acceptable, guesses, self.idle_front))
    }
}

/// Single-owner scorer for both Decision Time 1 (pre-draft `G`) and Decision
/// Time 2 (post-proposal `(K, F)`).
pub struct DecodePassScorer;

impl DecodePassScorer {
    /// Score a single `(G, K, F)` candidate under the exact pricing contract.
    ///
    /// - `draft_seconds` is the cost for `G` (paid even when `K + F < G`).
    /// - `cum_survival` is the cumulative survival `S[j]` computed ONCE from
    ///   conditional probabilities.
    /// - Immediate compulsory I/O covers the exact `1 + K` rows ONLY.
    /// - Speculative traffic, displacement, and DDR contention are charged separately.
    /// - `F` credit is strictly gated by survival, horizon/EOS, and carryover.
    pub fn score_candidate(
        draft_seconds: f64,
        k: usize,
        f: usize,
        cum_survival: &[f64],
        horizon: HorizonLimits,
        costs: &PassCostModel,
    ) -> PassPricing {
        Self::score_candidate_with_idle(draft_seconds, k, f, 0, cum_survival, horizon, costs)
    }

    /// Score a `(G, K, F, idle_front)` candidate: charges compute for all
    /// executed rows `1 + K + F + idle_front`, immediate compulsory I/O for
    /// `1 + K` rows only, tail side costs and credit strictly for productive
    /// `F` rows, and bounds both `K` and `F` by `horizon.effective_verify_horizon()`.
    pub fn score_candidate_with_idle(
        draft_seconds: f64,
        k: usize,
        f: usize,
        idle_front: usize,
        cum_survival: &[f64],
        horizon: HorizonLimits,
        costs: &PassCostModel,
    ) -> PassPricing {
        let draft_seconds = draft_seconds.max(0.0);
        let compute_seconds = costs.compute.seconds(1 + k + f + idle_front);
        // Immediate compulsory I/O is the distinct absent union of `1 + K` rows ONLY.
        let compulsory_io_seconds = costs.io.compulsory_io_seconds(k);
        // Tail-only (`F`) misses skip demand and are priced separately.
        let (speculative_traffic_seconds, displacement_seconds, ddr_contention_seconds) =
            costs.io.tail_side_costs(k, f);
        let host_overhead_seconds = if k > 0 {
            costs.verify_host_overhead_seconds.max(0.0)
                + k as f64 * costs.verify_host_overhead_per_step_seconds.max(0.0)
        } else {
            costs.non_verify_host_overhead_seconds.max(0.0)
        };
        let front_credit_seconds = costs
            .front_credit
            .credit_seconds(cum_survival, k, f, horizon);
        let total_exposed_wait = costs
            .front_credit
            .total_exposed_wait(cum_survival, k, horizon);
        // Consistent current+future time horizon: current pass body plus
        // unhidden future exposed wait `(total_exposed_wait - front_credit_seconds)`.
        // `effective_pass` is strictly >= `pass_body`, so `front_credit` can
        // never drive round time to 0 (e.g. `1ms - 2ms -> 1B tok/s`).
        let pass_body = compute_seconds.max(compulsory_io_seconds + speculative_traffic_seconds)
            + displacement_seconds
            + ddr_contention_seconds
            + host_overhead_seconds;
        let unhidden_future_wait = (total_exposed_wait - front_credit_seconds).max(0.0);
        let effective_pass = pass_body + unhidden_future_wait;
        let total_round_seconds = (draft_seconds + effective_pass).max(1e-9);
        // Only the exact row + `K` acceptable rows within `effective_verify_horizon()`
        // (`remaining_tokens.min(eos_at)`) can emit tokens; `F` and `idle_front` emit 0.
        let effective_k = k.min(horizon.effective_verify_horizon());
        let expected_tokens = 1.0 + cum_survival.iter().take(effective_k).copied().sum::<f64>();
        let rate_tps = expected_tokens / total_round_seconds;
        PassPricing {
            draft_seconds,
            compute_seconds,
            compulsory_io_seconds,
            speculative_traffic_seconds,
            displacement_seconds,
            ddr_contention_seconds,
            host_overhead_seconds,
            front_credit_seconds,
            total_round_seconds,
            expected_tokens,
            rate_tps,
        }
    }

    /// Choose the best `(K, F)` for a fixed `actual_g`, `draft_seconds`, and
    /// `reserved_capacity` (`>= 1`), enforcing `0 <= K + F <= actual_g`,
    /// `K <= horizon.effective_verify_horizon()`, and `1 + K + F <= reserved_capacity`.
    fn best_kf(
        actual_g: usize,
        draft_seconds: f64,
        reserved_capacity: usize,
        cum_survival: &[f64],
        horizon: HorizonLimits,
        costs: &PassCostModel,
    ) -> (usize, usize, PassPricing) {
        let max_extra = actual_g.min(reserved_capacity.saturating_sub(1));
        let max_k = max_extra.min(horizon.effective_verify_horizon());
        let mut best: Option<(usize, usize, PassPricing)> = None;
        for k in 0..=max_k {
            for f in 0..=(max_extra - k) {
                let pricing =
                    Self::score_candidate(draft_seconds, k, f, cum_survival, horizon, costs);
                let better = match best {
                    None => true,
                    Some((bk, bf, bp)) => {
                        if pricing.rate_tps > bp.rate_tps + 1e-12 {
                            true
                        } else if (pricing.rate_tps - bp.rate_tps).abs() <= 1e-12 {
                            // Tie-break: fewer total rows, then smaller F, then smaller K.
                            (k + f, f, k) < (bk + bf, bf, bk)
                        } else {
                            false
                        }
                    }
                };
                if better {
                    best = Some((k, f, pricing));
                }
            }
        }
        best.expect("at least (0, 0) is always feasible")
    }

    /// Choose the best `(K, F)` restricted to `allowed_k` using the single
    /// scorer. Enforces `K <= actual_g`, `K <= horizon.effective_verify_horizon()`,
    /// and `1 + K + F <= reserved_capacity`.
    #[allow(clippy::too_many_arguments)]
    pub fn seal_plan_on_widths(
        pass_id: PassId,
        actual_generated: usize,
        draft_seconds_paid: f64,
        reserved_capacity: usize,
        allowed_k: impl IntoIterator<Item = usize>,
        proposal_acceptance: &AcceptanceProfile,
        horizon: HorizonLimits,
        costs: &PassCostModel,
    ) -> Result<DecodePassPlan, String> {
        Self::seal_plan_on_widths_inner(
            pass_id,
            actual_generated,
            draft_seconds_paid,
            reserved_capacity,
            allowed_k,
            proposal_acceptance,
            horizon,
            costs,
            None,
        )
    }

    /// Choose the best `(K, F)` restricted to `allowed_k` and `F <= max_front`
    /// for an unchained pass (`chain == None`), enforcing the executor's
    /// [`Operation::validate`] invariant by construction:
    /// - When `K == 0` (`WorkKind::Decode` without `chain`), `F` is strictly `0`
    ///   (front rows on `WorkKind::Decode` require a chained pass).
    /// - When `K >= 1` (`WorkKind::Verify`), `F` is bounded by `max_front` and
    ///   `1 + K + F <= magnitude_executor::CHAINED_ROWS` (when `F == 0`, `K` is
    ///   never artificially capped by `CHAINED_ROWS`).
    /// - When `max_front == 0` (verify-only grant), `front_credit` is suppressed
    ///   so unmeasured future stall never penalizes wide `F = 0` verify passes.
    #[allow(clippy::too_many_arguments)]
    pub fn seal_unchained_plan_on_widths(
        pass_id: PassId,
        actual_generated: usize,
        draft_seconds_paid: f64,
        reserved_capacity: usize,
        allowed_k: impl IntoIterator<Item = usize>,
        max_front: usize,
        proposal_acceptance: &AcceptanceProfile,
        horizon: HorizonLimits,
        costs: &PassCostModel,
    ) -> Result<DecodePassPlan, String> {
        if max_front == 0 && !costs.front_credit.exposed_future_wait_by_front.is_empty() {
            let mut verify_only_costs = costs.clone();
            verify_only_costs.front_credit = FrontCreditModel::default();
            return Self::seal_plan_on_widths_inner(
                pass_id,
                actual_generated,
                draft_seconds_paid,
                reserved_capacity,
                allowed_k,
                proposal_acceptance,
                horizon,
                &verify_only_costs,
                Some(0),
            );
        }
        Self::seal_plan_on_widths_inner(
            pass_id,
            actual_generated,
            draft_seconds_paid,
            reserved_capacity,
            allowed_k,
            proposal_acceptance,
            horizon,
            costs,
            Some(max_front),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn seal_plan_on_widths_inner(
        pass_id: PassId,
        actual_generated: usize,
        draft_seconds_paid: f64,
        reserved_capacity: usize,
        allowed_k: impl IntoIterator<Item = usize>,
        proposal_acceptance: &AcceptanceProfile,
        horizon: HorizonLimits,
        costs: &PassCostModel,
        unchained_max_front: Option<usize>,
    ) -> Result<DecodePassPlan, String> {
        if reserved_capacity == 0 {
            return Err("reserved capacity must be at least 1 (the exact row)".into());
        }
        let max_k = actual_generated
            .min(reserved_capacity.saturating_sub(1))
            .min(horizon.effective_verify_horizon());
        let max_extra = actual_generated.min(reserved_capacity.saturating_sub(1));
        let cum = proposal_acceptance.cumulative_survival();
        let has_front_credit = costs
            .front_credit
            .exposed_future_wait_by_front
            .iter()
            .any(|&s| s > 0.0);
        let mut best: Option<(usize, usize, PassPricing)> = None;
        let mut consider = |k: usize, f: usize| {
            let pricing = Self::score_candidate(draft_seconds_paid, k, f, &cum, horizon, costs);
            let better = match best {
                None => true,
                Some((bk, bf, bp)) => {
                    if pricing.rate_tps > bp.rate_tps + 1e-12 {
                        true
                    } else if (pricing.rate_tps - bp.rate_tps).abs() <= 1e-12 {
                        (k + f, f, k) < (bk + bf, bf, bk)
                    } else {
                        false
                    }
                }
            };
            if better {
                best = Some((k, f, pricing));
            }
        };
        for w in allowed_k {
            if w <= max_k {
                let max_f = match unchained_max_front {
                    Some(front_cap) => {
                        if w == 0 || front_cap == 0 {
                            0
                        } else {
                            front_cap
                                .min(max_extra - w)
                                .min(magnitude_executor::CHAINED_ROWS.saturating_sub(1 + w))
                        }
                    }
                    None => max_extra - w,
                };
                for f in 0..=max_f {
                    consider(w, f);
                }
            }
            // When trailing front rows are enabled and carry positive future-wait
            // credit, also consider `(K, F)` splits of each legal total-row width
            // `w = K + F <= max_extra` (`K >= 1, F >= 1`), keeping the same target
            // launch row class `1 + w` without requiring callers to inject ad-hoc
            // intermediate verify widths (e.g. `widest - 1`).
            if let Some(front_cap) = unchained_max_front {
                if has_front_credit
                    && front_cap > 0
                    && w >= 2
                    && w <= max_extra
                    && 1 + w <= magnitude_executor::CHAINED_ROWS
                {
                    let max_split_f = front_cap.min(w - 1);
                    for f in 1..=max_split_f {
                        let k = w - f;
                        if k <= max_k {
                            consider(k, f);
                        }
                    }
                }
            }
        }
        let (verify, front, pricing) = best.unwrap_or_else(|| {
            (
                0,
                0,
                Self::score_candidate(draft_seconds_paid, 0, 0, &cum, horizon, costs),
            )
        });
        Ok(DecodePassPlan {
            pass_id,
            generated: actual_generated,
            verify,
            front,
            idle_front: 0,
            reserved_capacity,
            pricing,
        })
    }

    /// **Decision Time 1 (Pre-Draft):** choose a legal generation budget `G`
    /// from population evidence, subject to `offered_capacity` (`>= 1`) and
    /// `drafter` feasibility.
    pub fn choose_generation_budget(
        pass_id: PassId,
        drafter: &DrafterSpec,
        offered_capacity: usize,
        population_acceptance: &AcceptanceProfile,
        horizon: HorizonLimits,
        costs: &PassCostModel,
    ) -> Result<DraftBudgetPlan, String> {
        if offered_capacity == 0 {
            return Err("offered capacity must be at least 1 (the exact row)".into());
        }
        // `G` may exceed `K + F` (e.g. a fixed block drafter `G = 8` under a
        // 4-row verify offer `K = 3`), so enumerate all drafter-feasible `G`
        // while `best_kf` enforces `1 + K + F <= offered_capacity`.
        let cum = population_acceptance.cumulative_survival();
        let feasible = drafter.feasible_widths(usize::MAX);
        let mut best: Option<DraftBudgetPlan> = None;
        for g in feasible {
            let draft_seconds = drafter.draft_seconds(g);
            let (k, f, pricing) =
                Self::best_kf(g, draft_seconds, offered_capacity, &cum, horizon, costs);
            let candidate = DraftBudgetPlan {
                pass_id,
                generated: g,
                needs_draft_launch: drafter.needs_draft_launch(g),
                draft_seconds,
                projected_verify: k,
                projected_front: f,
                projected_pricing: pricing,
            };
            let better = match best {
                None => true,
                Some(prev) => {
                    if candidate.projected_pricing.rate_tps
                        > prev.projected_pricing.rate_tps + 1e-12
                    {
                        true
                    } else if (candidate.projected_pricing.rate_tps
                        - prev.projected_pricing.rate_tps)
                        .abs()
                        <= 1e-12
                    {
                        (candidate.generated, candidate.projected_front, candidate.projected_verify)
                            < (prev.generated, prev.projected_front, prev.projected_verify)
                    } else {
                        false
                    }
                }
            };
            if better {
                best = Some(candidate);
            }
        }
        best.ok_or_else(|| "no feasible generation budget".into())
    }

    /// **Decision Time 2 (Post-Proposal, Post-Grant):** once `actual_generated`
    /// proposals exist and `reserved_capacity` (`>= 1`) is reserved, choose
    /// `(K, F)` under `0 <= K + F <= actual_generated` and
    /// `1 + K + F <= reserved_capacity`.
    ///
    /// Note: `draft_seconds_paid` is the cost already spent producing
    /// `actual_generated` proposals, so unused proposals (`actual_generated > K + F`)
    /// still count in the round time.
    #[allow(clippy::too_many_arguments)]
    pub fn seal_plan(
        pass_id: PassId,
        actual_generated: usize,
        draft_seconds_paid: f64,
        reserved_capacity: usize,
        proposal_acceptance: &AcceptanceProfile,
        horizon: HorizonLimits,
        costs: &PassCostModel,
    ) -> Result<DecodePassPlan, String> {
        if reserved_capacity == 0 {
            return Err("reserved capacity must be at least 1 (the exact row)".into());
        }
        let cum = proposal_acceptance.cumulative_survival();
        let (verify, front, pricing) = Self::best_kf(
            actual_generated,
            draft_seconds_paid,
            reserved_capacity,
            &cum,
            horizon,
            costs,
        );
        debug_assert!(verify + front <= actual_generated);
        debug_assert!(1 + verify + front <= reserved_capacity);
        Ok(DecodePassPlan {
            pass_id,
            generated: actual_generated,
            verify,
            front,
            idle_front: 0,
            reserved_capacity,
            pricing,
        })
    }
}

/// One request competing for rows in a batched decode launch.
#[derive(Clone, Debug)]
pub struct BatchPlanCandidate {
    pub pass_id: PassId,
    pub actual_generated: usize,
    pub draft_seconds_paid: f64,
    pub acceptance: AcceptanceProfile,
    pub horizon: HorizonLimits,
    pub costs: PassCostModel,
    /// True if the request was cancelled before the batch grant was sealed.
    pub cancelled: bool,
}

/// Result of sealing a batched decode launch against a shared `total_row_budget`.
#[derive(Clone, Debug, PartialEq)]
pub struct SealedBatch {
    /// Sealed plans for admitted active requests; `sum(p.total_rows()) <= total_row_budget`.
    pub plans: Vec<DecodePassPlan>,
    /// Requests cancelled before sealing (0 rows consumed, settled as `Discarded`).
    pub cancelled: Vec<PassId>,
    /// Active requests that could not fit even 1 exact row in `total_row_budget`
    /// and wait for the next launch.
    pub deferred: Vec<PassId>,
    /// Total rows reserved across all sealed plans (`<= total_row_budget`).
    pub total_reserved_rows: usize,
}

/// Reserve a shared row grant across competing requests BEFORE sealing final
/// plans, repricing any truncated request on its smaller reserved offer so
/// per-request verbatim plans can NEVER overbook a batched launch.
pub fn seal_batch_plans(
    candidates: &[BatchPlanCandidate],
    total_row_budget: usize,
) -> Result<SealedBatch, String> {
    let mut cancelled = Vec::new();
    let mut active = Vec::new();
    for c in candidates {
        if c.cancelled {
            cancelled.push(c.pass_id);
        } else {
            active.push(c);
        }
    }

    let admit_count = active.len().min(total_row_budget);
    let deferred: Vec<PassId> = active[admit_count..]
        .iter()
        .map(|c| c.pass_id)
        .collect();
    let admitted = &active[..admit_count];

    if admitted.is_empty() {
        return Ok(SealedBatch {
            plans: Vec::new(),
            cancelled,
            deferred,
            total_reserved_rows: 0,
        });
    }

    // Step 1: Reserve 1 exact row per admitted request.
    let mut reserved = vec![1usize; admitted.len()];
    let mut remaining = total_row_budget - admitted.len();

    // Step 2: Check each request's unconstrained optimal row demand.
    let mut desired = Vec::with_capacity(admitted.len());
    for c in admitted {
        let unconstrained_cap = 1 + c.actual_generated;
        let plan = DecodePassScorer::seal_plan(
            c.pass_id,
            c.actual_generated,
            c.draft_seconds_paid,
            unconstrained_cap,
            &c.acceptance,
            c.horizon,
            &c.costs,
        )?;
        desired.push(plan.total_rows());
    }

    // Step 3: Allocate remaining shared rows by highest marginal throughput gain
    // without ever exceeding `total_row_budget` or any request's `desired` rows.
    let mut current_rate = Vec::with_capacity(admitted.len());
    for (idx, c) in admitted.iter().enumerate() {
        let p = DecodePassScorer::seal_plan(
            c.pass_id,
            c.actual_generated,
            c.draft_seconds_paid,
            reserved[idx],
            &c.acceptance,
            c.horizon,
            &c.costs,
        )?;
        current_rate.push(p.pricing.rate_tps);
    }

    while remaining > 0 {
        // Evaluate multi-row steps `1..=(desired[idx] - reserved[idx]).min(remaining)`
        // by gain-per-row so a zero-marginal plateau (e.g. verify overhead at K=1
        // paid off at K=2..3) never prematurely halts batch allocation.
        let mut best_step: Option<(usize, usize, f64, f64)> = None;
        for (idx, c) in admitted.iter().enumerate() {
            let max_step = desired[idx].saturating_sub(reserved[idx]).min(remaining);
            if max_step == 0 {
                continue;
            }
            for step in 1..=max_step {
                let next_plan = DecodePassScorer::seal_plan(
                    c.pass_id,
                    c.actual_generated,
                    c.draft_seconds_paid,
                    reserved[idx] + step,
                    &c.acceptance,
                    c.horizon,
                    &c.costs,
                )?;
                let gain = next_plan.pricing.rate_tps - current_rate[idx];
                let gain_per_row = gain / step as f64;
                if gain > 0.0
                    && best_step.is_none_or(|(bidx, bstep, bgpr, _)| {
                        gain_per_row > bgpr + 1e-12
                            || ((gain_per_row - bgpr).abs() <= 1e-12
                                && (step, reserved[idx]) < (bstep, reserved[bidx]))
                    })
                {
                    best_step = Some((idx, step, gain_per_row, next_plan.pricing.rate_tps));
                }
            }
        }
        let Some((winner, step, _, new_rate)) = best_step else {
            break;
        };
        reserved[winner] += step;
        remaining -= step;
        current_rate[winner] = new_rate;
    }

    // Step 4: Seal every admitted request on its reserved offer (repricing
    // truncated requests cleanly on the smaller offer).
    let mut plans = Vec::with_capacity(admitted.len());
    let mut total_reserved_rows = 0usize;
    for (idx, c) in admitted.iter().enumerate() {
        let plan = DecodePassScorer::seal_plan(
            c.pass_id,
            c.actual_generated,
            c.draft_seconds_paid,
            reserved[idx],
            &c.acceptance,
            c.horizon,
            &c.costs,
        )?;
        total_reserved_rows += plan.total_rows();
        plans.push(plan);
    }
    debug_assert!(total_reserved_rows <= total_row_budget);

    Ok(SealedBatch {
        plans,
        cancelled,
        deferred,
        total_reserved_rows,
    })
}

/// Actual committed token history of a decode pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PassCommitment {
    /// Pass was cancelled, evicted, orphaned, or failed before commit.
    Discarded,
    /// Pass committed `prefix` tokens in exact sequence order.
    Committed(Vec<TokenId>),
}

/// Role of a predicted branch token in a decode pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionRole {
    Verify,
    Front,
}

/// Separate judgment of one predicted branch token (carries exact token
/// identity — never collapsed into an ambiguous `front_kept` count alone).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionJudgment {
    /// Awaiting a later pass's committed token.
    Pending { predicted: TokenId },
    /// Matched the actual committed token at this position.
    Matched { predicted: TokenId, actual: TokenId },
    /// Diverged from the actual committed token (or an earlier branch diverged).
    Rejected { predicted: TokenId, actual: TokenId },
    /// Unjudged because the pass or sequence terminated/cancelled first.
    Unjudged { predicted: TokenId },
}

/// Separately judged branch prediction entry in [`PassOutcome`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BranchPrediction {
    /// 1-based depth among proposals (`1..=K` for Verify, `K+1..=K+F` for Front).
    pub depth: usize,
    pub role: PredictionRole,
    pub judgment: PredictionJudgment,
}

/// Complete, self-describing outcome of one decode pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassOutcome {
    pub pass_id: PassId,
    pub commitment: PassCommitment,
    pub branches: Vec<BranchPrediction>,
}

impl PassOutcome {
    /// Build a committed outcome from a sealed plan, its proposals, and the
    /// `1 + K` target samples (same committed prefix as `F = 0`, plus
    /// separately judged verify and front branch predictions).
    pub fn from_samples(
        plan: &DecodePassPlan,
        proposals: &[TokenId],
        samples: &[TokenId],
        stops: &BTreeSet<TokenId>,
    ) -> Result<Self, String> {
        if samples.len() != 1 + plan.verify {
            return Err(format!(
                "expected {} samples for K = {}, got {}",
                1 + plan.verify,
                plan.verify,
                samples.len()
            ));
        }
        let used = plan.verify + plan.front;
        if proposals.len() < used {
            return Err("proposals shorter than plan K + F".into());
        }
        let acceptable = &proposals[..plan.verify];
        let accepted = crate::accept_prefix(acceptable, samples, stops)?;
        let mut prefix = acceptable[..accepted].to_vec();
        prefix.push(samples[accepted]);
        let stopped = stops.contains(prefix.last().expect("committed prefix is non-empty"));

        // Drive EVERY branch judgment from the single committed `prefix` and
        // its stop/mismatch boundary: no token past a committed stop or first
        // mismatch (including a first-front mismatch) is ever marked `Matched`
        // or left `Pending`.
        let mut branches = Vec::with_capacity(used);
        let mut branch_alive = true;
        for (idx, &predicted) in proposals[..used].iter().enumerate() {
            let role = if idx < plan.verify {
                PredictionRole::Verify
            } else {
                PredictionRole::Front
            };
            let judgment = if !branch_alive {
                PredictionJudgment::Unjudged { predicted }
            } else if idx < prefix.len() {
                let actual = prefix[idx];
                if predicted == actual {
                    // If this committed token is a stop/EOS token, generation
                    // ended here: every later branch token is cancelled/unjudged.
                    if idx + 1 == prefix.len() && stopped {
                        branch_alive = false;
                    }
                    PredictionJudgment::Matched { predicted, actual }
                } else {
                    branch_alive = false;
                    PredictionJudgment::Rejected { predicted, actual }
                }
            } else {
                // Past `prefix.len()` while still alive: only reachable for
                // deeper front rows (`idx > plan.verify`) when all K verify
                // rows and front[0] matched without hitting a stop token.
                PredictionJudgment::Pending { predicted }
            };
            branches.push(BranchPrediction {
                depth: idx + 1,
                role,
                judgment,
            });
        }
        Ok(Self {
            pass_id: plan.pass_id,
            commitment: PassCommitment::Committed(prefix),
            branches,
        })
    }

    /// Build a discarded outcome for a cancelled/aborted pass.
    pub fn discarded(pass_id: PassId, plan: Option<&DecodePassPlan>, proposals: &[TokenId]) -> Self {
        let mut branches = Vec::new();
        if let Some(plan) = plan {
            let used = (plan.verify + plan.front).min(proposals.len());
            for (i, &predicted) in proposals[..used].iter().enumerate() {
                let role = if i < plan.verify {
                    PredictionRole::Verify
                } else {
                    PredictionRole::Front
                };
                branches.push(BranchPrediction {
                    depth: i + 1,
                    role,
                    judgment: PredictionJudgment::Unjudged { predicted },
                });
            }
        }
        Self {
            pass_id,
            commitment: PassCommitment::Discarded,
            branches,
        }
    }
}

#[derive(Clone, Debug)]
struct PendingPass {
    plan: DecodePassPlan,
    proposals: Vec<TokenId>,
    completed: bool,
}

/// Maximum settled pass tombstones retained in `PassOutcomeTracker` before
/// retiring the oldest entries (preventing unbounded B-tree growth across
/// long-running serving sessions).
const MAX_SETTLED_RETAINED: usize = 256;

/// Single-settlement tracker for [`PassOutcome`] across asynchronous /
/// out-of-order execution events:
/// - **Cancel-before-completion:** settles ONCE immediately as `Discarded`;
///   later completion or verdict is ignored.
/// - **Verdict-before-feed:** buffers the early verdict until the pass is fed,
///   then settles ONCE; duplicate verdicts or late cancels are ignored.
/// - **Bounded retention:** retires oldest settled tombstones past
///   [`MAX_SETTLED_RETAINED`] and supports explicit [`Self::retire_before`].
#[derive(Clone, Debug, Default)]
pub struct PassOutcomeTracker {
    pending: BTreeMap<PassId, PendingPass>,
    early_verdicts: BTreeMap<PassId, PassOutcome>,
    settled: BTreeSet<PassId>,
    settlement_counts: BTreeMap<PassId, usize>,
    /// Per-request retired generation watermark: any `PassId` with
    /// `generation < watermark` is treated as already retired/settled.
    retired_generation: BTreeMap<RequestId, u64>,
    /// Per-(request, generation) retired pass watermark advanced whenever
    /// oldest settled entries roll off past [`MAX_SETTLED_RETAINED`], so
    /// evicted passes remain permanently recognized as settled.
    retired_pass: BTreeMap<(RequestId, u64), u64>,
    /// Single-settlement empirical conditional acceptance counts per 1-based
    /// proposal depth (`depth -> (matched, judged)`).
    depth_stats: BTreeMap<usize, (u64, u64)>,
}

impl PassOutcomeTracker {
    /// Feed/register a sealed pass into the tracker.
    /// If a verdict arrived before feed (`verdict-before-feed`), settles ONCE
    /// immediately and returns `Some(outcome)`.
    /// If the pass was already cancelled/settled, returns `None`.
    pub fn feed(&mut self, plan: DecodePassPlan, proposals: Vec<TokenId>) -> Option<PassOutcome> {
        let id = plan.pass_id;
        if self.is_settled(id) {
            return None;
        }
        if let Some(outcome) = self.early_verdicts.remove(&id) {
            return Some(self.settle_once(outcome));
        }
        self.pending.insert(
            id,
            PendingPass {
                plan,
                proposals,
                completed: false,
            },
        );
        None
    }

    /// Mark device execution completed for `pass_id`. Does not settle on its
    /// own (verdict commits the prefix); no-op if already settled.
    pub fn mark_completed(&mut self, pass_id: PassId) {
        if self.is_settled(pass_id) {
            return;
        }
        if let Some(entry) = self.pending.get_mut(&pass_id) {
            entry.completed = true;
        }
    }

    /// Deliver a pass verdict.
    /// - If already settled (e.g. cancelled before completion): returns `None`.
    /// - If the pass is already fed: settles ONCE and returns `Some(outcome)`.
    /// - If the pass has not been fed yet (`verdict-before-feed`): buffers the
    ///   verdict (keeping the first if called twice) and returns `None` until
    ///   [`Self::feed`] settles it once.
    pub fn record_verdict(&mut self, outcome: PassOutcome) -> Option<PassOutcome> {
        let id = outcome.pass_id;
        if self.is_settled(id) {
            return None;
        }
        if self.pending.remove(&id).is_some() {
            return Some(self.settle_once(outcome));
        }
        self.early_verdicts.entry(id).or_insert(outcome);
        None
    }

    /// Cancel `pass_id` before completion/settlement.
    /// - If already settled: returns `None`.
    /// - Otherwise: drops any pending/early state and settles ONCE as
    ///   [`PassCommitment::Discarded`].
    pub fn cancel(&mut self, pass_id: PassId) -> Option<PassOutcome> {
        if self.is_settled(pass_id) {
            return None;
        }
        self.early_verdicts.remove(&pass_id);
        let outcome = match self.pending.remove(&pass_id) {
            Some(entry) => PassOutcome::discarded(pass_id, Some(&entry.plan), &entry.proposals),
            None => PassOutcome::discarded(pass_id, None, &[]),
        };
        Some(self.settle_once(outcome))
    }

    /// Number of times `pass_id` has settled (always `0` or `1`).
    pub fn settlement_count(&self, pass_id: PassId) -> usize {
        self.settlement_counts.get(&pass_id).copied().unwrap_or(0)
    }

    /// Whether `pass_id` has already settled (or belongs to a retired
    /// generation or rolled-off pass watermark).
    pub fn is_settled(&self, pass_id: PassId) -> bool {
        self.settled.contains(&pass_id)
            || self
                .retired_generation
                .get(&pass_id.request)
                .is_some_and(|&min_gen| pass_id.generation < min_gen)
            || self
                .retired_pass
                .get(&(pass_id.request, pass_id.generation))
                .is_some_and(|&min_pass| pass_id.pass < min_pass)
    }

    /// Retire all settled/pending/early entries for `request` strictly before
    /// `min_generation` (e.g. after eviction or request reset).
    pub fn retire_before(&mut self, request: RequestId, min_generation: u64) {
        let entry = self.retired_generation.entry(request).or_insert(0);
        *entry = (*entry).max(min_generation);
        self.pending
            .retain(|id, _| id.request != request || id.generation >= min_generation);
        self.early_verdicts
            .retain(|id, _| id.request != request || id.generation >= min_generation);
        self.settled
            .retain(|id| id.request != request || id.generation >= min_generation);
        self.settlement_counts
            .retain(|id, _| id.request != request || id.generation >= min_generation);
        self.retired_pass
            .retain(|&(req, gen), _| req != request || gen >= min_generation);
    }

    /// Number of retained settled tombstone entries currently in the B-tree.
    pub fn retained_settled_len(&self) -> usize {
        self.settled.len()
    }

    /// Empirical conditional acceptance rate `P(depth stands | depths < depth stood)`
    /// at 1-based `depth` from all settled passes (`None` before any judgment at `depth`).
    pub fn conditional_rate(&self, depth: usize) -> Option<f64> {
        let &(matched, judged) = self.depth_stats.get(&depth)?;
        (judged > 0).then(|| matched as f64 / judged as f64)
    }

    fn settle_once(&mut self, outcome: PassOutcome) -> PassOutcome {
        let id = outcome.pass_id;
        let inserted = self.settled.insert(id);
        assert!(inserted, "invariant violated: pass {id:?} settled more than once");
        *self.settlement_counts.entry(id).or_insert(0) += 1;
        for branch in &outcome.branches {
            match branch.judgment {
                PredictionJudgment::Matched { .. } => {
                    let entry = self.depth_stats.entry(branch.depth).or_default();
                    entry.0 += 1;
                    entry.1 += 1;
                }
                PredictionJudgment::Rejected { .. } => {
                    let entry = self.depth_stats.entry(branch.depth).or_default();
                    entry.1 += 1;
                }
                PredictionJudgment::Pending { .. } | PredictionJudgment::Unjudged { .. } => {}
            }
        }
        while self.settled.len() > MAX_SETTLED_RETAINED {
            if let Some(oldest) = self.settled.pop_first() {
                self.settlement_counts.remove(&oldest);
                let wm = self
                    .retired_pass
                    .entry((oldest.request, oldest.generation))
                    .or_insert(0);
                *wm = (*wm).max(oldest.pass.saturating_add(1));
            }
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Demand, RoundState, Sampling, Shaping};
    use magnitude_executor::WorkKind;

    fn pass_id(req: u64, gen: u64, pass: u64) -> PassId {
        PassId {
            request: RequestId(req),
            generation: gen,
            pass,
        }
    }

    fn base_compute() -> ComputePricing {
        // Monotone per-row compute C(1..=8) in seconds.
        ComputePricing {
            by_rows: vec![0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017],
        }
    }

    /// Boundary 1: Two requests competing for last rows — one truncated and
    /// repriced on its smaller offer, one cancelled before sealing (0 rows
    /// consumed, settled as Discarded), never overbooking the shared budget.
    #[test]
    fn two_requests_competing_for_last_rows_one_truncated_one_cancelled() {
        let costs = PassCostModel {
            compute: base_compute(),
            io: RowMissPricing::ExpectedRates {
                exact_row_misses: 2.0,
                verify_marginal_misses: vec![0.5; 4],
                front_tail_misses: vec![0.0; 4],
                demand_seconds_per_miss: 0.001,
                speculative_seconds_per_miss: 0.0,
                displacement_seconds_per_miss: 0.0,
                ddr_contention_seconds_per_miss: 0.0,
            },
            front_credit: FrontCreditModel::default(),
            verify_host_overhead_seconds: 0.001,
            verify_host_overhead_per_step_seconds: 0.0,
            non_verify_host_overhead_seconds: 0.0,
        };

        // Case A: Two active requests each want 4 rows (1 + K=3), competing for 5 rows total.
        // Req 1 has higher acceptance -> gets more rows; Req 2 is truncated and repriced.
        let req1 = BatchPlanCandidate {
            pass_id: pass_id(1, 1, 10),
            actual_generated: 3,
            draft_seconds_paid: 0.002,
            acceptance: AcceptanceProfile::Conditional(vec![0.95, 0.95, 0.95]),
            horizon: HorizonLimits::UNBOUNDED,
            costs: costs.clone(),
            cancelled: false,
        };
        let req2 = BatchPlanCandidate {
            pass_id: pass_id(2, 1, 10),
            actual_generated: 3,
            draft_seconds_paid: 0.002,
            acceptance: AcceptanceProfile::Conditional(vec![0.60, 0.50, 0.40]),
            horizon: HorizonLimits::UNBOUNDED,
            costs: costs.clone(),
            cancelled: false,
        };
        // Also a third competing request cancelled before sealing!
        let req3_cancelled = BatchPlanCandidate {
            pass_id: pass_id(3, 1, 10),
            actual_generated: 3,
            draft_seconds_paid: 0.002,
            acceptance: AcceptanceProfile::Conditional(vec![0.99, 0.99, 0.99]),
            horizon: HorizonLimits::UNBOUNDED,
            costs,
            cancelled: true,
        };

        let budget = 5;
        let sealed = seal_batch_plans(&[req1, req2, req3_cancelled], budget).unwrap();
        assert_eq!(sealed.cancelled, vec![pass_id(3, 1, 10)]);
        assert!(sealed.deferred.is_empty());
        assert_eq!(sealed.plans.len(), 2);
        assert!(
            sealed.total_reserved_rows <= budget,
            "batched launch must NEVER overbook: {} > {budget}",
            sealed.total_reserved_rows
        );
        let p1 = &sealed.plans[0];
        let p2 = &sealed.plans[1];
        assert!(p1.total_rows() <= p1.reserved_capacity);
        assert!(p2.total_rows() <= p2.reserved_capacity);
        assert_eq!(p1.total_rows() + p2.total_rows(), sealed.total_reserved_rows);
        // Req 2 was truncated below its generated=3 (`G > K + F` allowed, unused proposals still priced).
        assert!(
            p2.verify + p2.front < p2.generated,
            "truncated request must reprice below actual G: {p2:?}"
        );
        assert_eq!(
            p2.pricing.draft_seconds, 0.002,
            "unused proposals still pay draft_seconds"
        );
    }

    /// Boundary 2: First rejection (`p_cond[0] == 0.0`) or EOS / length limit
    /// suppresses the `F` foresight bonus completely, forcing `F = 0`.
    #[test]
    fn first_rejection_or_eos_suppresses_front_bonus() {
        let costs = PassCostModel {
            compute: base_compute(),
            io: RowMissPricing::ExpectedRates {
                exact_row_misses: 5.0,
                verify_marginal_misses: vec![5.0, 5.0, 5.0],
                front_tail_misses: vec![1.0, 1.0],
                demand_seconds_per_miss: 0.002,
                speculative_seconds_per_miss: 0.0001,
                displacement_seconds_per_miss: 0.0,
                ddr_contention_seconds_per_miss: 0.0,
            },
            front_credit: FrontCreditModel {
                exposed_future_wait_by_front: vec![0.008, 0.006],
                carryover_covered_by_front: vec![0.001, 0.001],
            },
            verify_host_overhead_seconds: 0.020,
            verify_host_overhead_per_step_seconds: 0.0,
            non_verify_host_overhead_seconds: 0.0,
        };

        // Baseline: high survival and unbounded horizon -> F > 0 is profitable.
        let healthy = DecodePassScorer::seal_plan(
            pass_id(1, 1, 1),
            2,
            0.0,
            4,
            &AcceptanceProfile::Conditional(vec![0.95, 0.90]),
            HorizonLimits::UNBOUNDED,
            &costs,
        )
        .unwrap();
        assert!(healthy.front > 0, "baseline should choose F > 0: {healthy:?}");
        assert!(healthy.pricing.front_credit_seconds > 0.0);

        // First rejection (p_cond[0] == 0.0) -> survival is 0 for all depths -> F bonus = 0, F = 0.
        let rejected = DecodePassScorer::seal_plan(
            pass_id(1, 1, 2),
            2,
            0.0,
            4,
            &AcceptanceProfile::Conditional(vec![0.0, 0.99]),
            HorizonLimits::UNBOUNDED,
            &costs,
        )
        .unwrap();
        assert_eq!(rejected.front, 0, "first rejection must suppress F: {rejected:?}");
        assert_eq!(rejected.pricing.front_credit_seconds, 0.0);

        // EOS at proposal 0 -> effective horizon = 0 -> F bonus = 0, F = 0.
        let eos = DecodePassScorer::seal_plan(
            pass_id(1, 1, 3),
            2,
            0.0,
            4,
            &AcceptanceProfile::Conditional(vec![0.95, 0.90]),
            HorizonLimits {
                remaining_tokens: 10,
                eos_at: Some(0),
            },
            &costs,
        )
        .unwrap();
        assert_eq!(eos.front, 0, "imminent EOS must suppress F: {eos:?}");
        assert_eq!(eos.pricing.front_credit_seconds, 0.0);

        // Length limit reached (`remaining_tokens == 0`) -> F bonus = 0, F = 0.
        let at_limit = DecodePassScorer::seal_plan(
            pass_id(1, 1, 4),
            2,
            0.0,
            4,
            &AcceptanceProfile::Conditional(vec![0.95, 0.90]),
            HorizonLimits {
                remaining_tokens: 0,
                eos_at: None,
            },
            &costs,
        )
        .unwrap();
        assert_eq!(at_limit.front, 0, "length limit must suppress F: {at_limit:?}");
        assert_eq!(at_limit.pricing.front_credit_seconds, 0.0);

        // Carryover already covers exposed wait -> F bonus = 0, F = 0.
        let mut covered_costs = costs;
        covered_costs.front_credit.carryover_covered_by_front = vec![0.010, 0.010];
        let covered = DecodePassScorer::seal_plan(
            pass_id(1, 1, 5),
            2,
            0.0,
            4,
            &AcceptanceProfile::Conditional(vec![0.95, 0.90]),
            HorizonLimits::UNBOUNDED,
            &covered_costs,
        )
        .unwrap();
        assert_eq!(covered.front, 0, "carryover-covered wait must suppress F: {covered:?}");
        assert_eq!(covered.pricing.front_credit_seconds, 0.0);
    }

    /// Boundary 3: `F`-only misses are withheld from immediate compulsory
    /// demand I/O (distinct union of `1 + K` rows ONLY) and charged separately
    /// as speculative traffic, displacement, and DDR contention.
    #[test]
    fn front_only_miss_withheld_from_immediate_demand_io() {
        let io = RowMissPricing::ExactAbsentSets {
            per_row_absent: vec![
                BTreeSet::from([1, 2]),       // row 0 (exact)
                BTreeSet::from([2, 3]),       // row 1 (verify K=1): distinct union with row 0 = {1,2,3} (3 misses)
                BTreeSet::from([3, 99, 100]), // row 2 (front F=1): 99, 100 are tail-only misses!
            ],
            demand_seconds_per_miss: 0.005,
            speculative_seconds_per_miss: 0.001,
            displacement_seconds_per_miss: 0.0002,
            ddr_contention_seconds_per_miss: 0.0003,
        };
        let costs = PassCostModel {
            compute: base_compute(),
            io,
            front_credit: FrontCreditModel::default(),
            verify_host_overhead_seconds: 0.0,
            verify_host_overhead_per_step_seconds: 0.0,
            non_verify_host_overhead_seconds: 0.0,
        };
        let cum = AcceptanceProfile::Conditional(vec![0.9, 0.9]).cumulative_survival();
        let k1_f0 = DecodePassScorer::score_candidate(0.0, 1, 0, &cum, HorizonLimits::UNBOUNDED, &costs);
        let k1_f1 = DecodePassScorer::score_candidate(0.0, 1, 1, &cum, HorizonLimits::UNBOUNDED, &costs);

        // Compulsory demand I/O MUST be identical for (K=1, F=0) and (K=1, F=1): 3 * 0.005 = 0.015s.
        assert!((k1_f0.compulsory_io_seconds - 0.015).abs() < 1e-12);
        assert!(
            (k1_f1.compulsory_io_seconds - k1_f0.compulsory_io_seconds).abs() < 1e-12,
            "F-only misses must NEVER enter compulsory demand I/O: f0={}, f1={}",
            k1_f0.compulsory_io_seconds,
            k1_f1.compulsory_io_seconds
        );
        // Tail-only misses {99, 100} (2 misses) are charged separately!
        assert!((k1_f1.speculative_traffic_seconds - 2.0 * 0.001).abs() < 1e-12);
        assert!((k1_f1.displacement_seconds - 2.0 * 0.0002).abs() < 1e-12);
        assert!((k1_f1.ddr_contention_seconds - 2.0 * 0.0003).abs() < 1e-12);
        assert_eq!(k1_f0.speculative_traffic_seconds, 0.0);
        assert_eq!(k1_f0.displacement_seconds, 0.0);
        assert_eq!(k1_f0.ddr_contention_seconds, 0.0);
        // Compute reflects 1 + K + F rows (3 vs 2).
        assert!(k1_f1.compute_seconds > k1_f0.compute_seconds);
    }

    /// Boundary 4: Invariance — running a sealed plan with `F > 0` produces
    /// the exact same committed prefix, emitted tokens, and state as `F = 0`.
    #[test]
    fn mixed_pass_produces_same_output_and_state_as_f_zero() {
        let stops = BTreeSet::from([TokenId(999)]);
        let pending = TokenId(10);
        let proposals = vec![TokenId(20), TokenId(30), TokenId(40), TokenId(50)];
        let dummy_pricing = PassPricing {
            draft_seconds: 0.0,
            compute_seconds: 0.01,
            compulsory_io_seconds: 0.0,
            speculative_traffic_seconds: 0.0,
            displacement_seconds: 0.0,
            ddr_contention_seconds: 0.0,
            host_overhead_seconds: 0.0,
            front_credit_seconds: 0.0,
            total_round_seconds: 0.01,
            expected_tokens: 2.0,
            rate_tps: 200.0,
        };
        let plan_f0 = DecodePassPlan {
            pass_id: pass_id(1, 1, 1),
            generated: 4,
            verify: 2,
            front: 0,
            idle_front: 0,
            reserved_capacity: 5,
            pricing: dummy_pricing,
        };
        let plan_f2 = DecodePassPlan {
            pass_id: pass_id(1, 1, 1),
            generated: 4,
            verify: 2,
            front: 2,
            idle_front: 0,
            reserved_capacity: 5,
            pricing: dummy_pricing,
        };

        let rows_f0 = plan_f0.build_rows(&proposals, GuessKind::Chained).unwrap();
        let rows_f2 = plan_f2.build_rows(&proposals, GuessKind::Chained).unwrap();
        assert_eq!(rows_f0.row_count(), 3);
        assert_eq!(rows_f2.row_count(), 5);

        let build_round = |rows: Rows| {
            RoundState::pass(
                pending,
                rows,
                5,
                &[TokenId(1), TokenId(2)],
                None,
                Sampling::Greedy,
                Shaping::default(),
                7,
                Demand::NONE,
                None,
            )
            .unwrap()
        };
        let round_f0 = build_round(rows_f0);
        let round_f2 = build_round(rows_f2);
        assert_eq!(round_f0.forward().kind, WorkKind::Verify);
        assert_eq!(round_f2.forward().kind, WorkKind::Verify);
        // Operation::validate must accept mixed (K=2, F=2) verify+front forward!
        round_f2
            .forward()
            .clone()
            .into_operation(RequestId(1), 5, None)
            .unwrap();

        // Test every acceptance case: 0 accepted, 1 accepted, 2 accepted.
        for samples in [
            vec![TokenId(77), TokenId(88), TokenId(99)], // reject at 0
            vec![TokenId(20), TokenId(88), TokenId(99)], // accept 1, reject at 1
            vec![TokenId(20), TokenId(30), TokenId(40)], // accept all 2 + front[0] matches
            vec![TokenId(20), TokenId(30), TokenId(66)], // accept all 2 + front[0] misses
        ] {
            let acc_f0 = round_f0.clone().reconcile(&samples, &stops).unwrap();
            let acc_f2 = round_f2.clone().reconcile(&samples, &stops).unwrap();
            assert_eq!(acc_f0.emitted, acc_f2.emitted);
            assert_eq!(acc_f0.committed_rows, acc_f2.committed_rows);
            assert_eq!(acc_f0.inputs, acc_f2.inputs);
            assert_eq!(acc_f0.proposed, acc_f2.proposed);
            assert_eq!(acc_f0.accepted_proposals, acc_f2.accepted_proposals);

            // Outcomes also carry identical committed prefix and separately judged branches.
            let out_f0 = PassOutcome::from_samples(&plan_f0, &proposals, &samples, &stops).unwrap();
            let out_f2 = PassOutcome::from_samples(&plan_f2, &proposals, &samples, &stops).unwrap();
            assert_eq!(out_f0.commitment, out_f2.commitment);
            assert_eq!(out_f2.branches.len(), 4);
        }
    }

    /// Boundary 5: Cancel-before-completion and verdict-before-feed both
    /// settle ONCE, with zero double-settlement on late/duplicate events.
    #[test]
    fn cancel_before_completion_and_verdict_before_feed_settle_once() {
        let dummy_pricing = PassPricing {
            draft_seconds: 0.0,
            compute_seconds: 0.01,
            compulsory_io_seconds: 0.0,
            speculative_traffic_seconds: 0.0,
            displacement_seconds: 0.0,
            ddr_contention_seconds: 0.0,
            host_overhead_seconds: 0.0,
            front_credit_seconds: 0.0,
            total_round_seconds: 0.01,
            expected_tokens: 1.0,
            rate_tps: 100.0,
        };
        let mut tracker = PassOutcomeTracker::default();

        // 1) Cancel-before-completion: feed -> cancel -> mark_completed -> late verdict -> late cancel.
        let id_cancel = pass_id(1, 2, 100);
        let plan_cancel = DecodePassPlan {
            pass_id: id_cancel,
            generated: 2,
            verify: 1,
            front: 1,
            idle_front: 0,
            reserved_capacity: 3,
            pricing: dummy_pricing,
        };
        let proposals = vec![TokenId(11), TokenId(12)];
        assert!(tracker.feed(plan_cancel, proposals.clone()).is_none());
        let settled_cancel = tracker.cancel(id_cancel).expect("cancel settles once");
        assert_eq!(settled_cancel.pass_id, id_cancel);
        assert_eq!(settled_cancel.commitment, PassCommitment::Discarded);
        assert_eq!(settled_cancel.branches.len(), 2);
        assert_eq!(tracker.settlement_count(id_cancel), 1);

        // Late completion, late verdict, and duplicate cancel MUST be no-ops.
        tracker.mark_completed(id_cancel);
        let late_verdict = PassOutcome {
            pass_id: id_cancel,
            commitment: PassCommitment::Committed(vec![TokenId(11)]),
            branches: Vec::new(),
        };
        assert!(tracker.record_verdict(late_verdict).is_none());
        assert!(tracker.cancel(id_cancel).is_none());
        assert_eq!(tracker.settlement_count(id_cancel), 1);

        // 2) Verdict-before-feed: verdict arrives before feed -> buffered -> feed settles ONCE.
        let id_early = pass_id(2, 3, 200);
        let plan_early = DecodePassPlan {
            pass_id: id_early,
            generated: 2,
            verify: 1,
            front: 1,
            idle_front: 0,
            reserved_capacity: 3,
            pricing: dummy_pricing,
        };
        let stops = BTreeSet::new();
        let early_outcome = PassOutcome::from_samples(
            &plan_early,
            &proposals,
            &[TokenId(11), TokenId(12)],
            &stops,
        )
        .unwrap();
        assert!(
            tracker.record_verdict(early_outcome.clone()).is_none(),
            "verdict before feed buffers without settling"
        );
        assert_eq!(tracker.settlement_count(id_early), 0);

        let settled_early = tracker
            .feed(plan_early, proposals)
            .expect("feed settles buffered early verdict once");
        assert_eq!(settled_early, early_outcome);
        assert_eq!(tracker.settlement_count(id_early), 1);

        // Duplicate verdict, late cancel, or duplicate feed MUST be no-ops.
        assert!(tracker.record_verdict(early_outcome).is_none());
        assert!(tracker.cancel(id_early).is_none());
        assert!(tracker.feed(plan_early, vec![]).is_none());
        assert_eq!(tracker.settlement_count(id_early), 1);
    }

    /// Verify drafter polymorphism (EmbeddedChain never forces a new drafter;
    /// Block and Autoregressive expose their own feasible widths/costs) and
    /// survival product non-duplication (Cumulative is never multiplied twice).
    #[test]
    fn drafter_polymorphism_and_cumulative_survival_not_multiplied_twice() {
        let mut two_row = TwoRowDecode::default();
        two_row.guessed(vec![
            Guess { token: TokenId(5), kind: GuessKind::Chained },
            Guess { token: TokenId(6), kind: GuessKind::Head },
        ]);
        let embedded = DrafterSpec::from_two_row(&two_row);
        assert!(!embedded.needs_draft_launch(2));
        assert_eq!(embedded.draft_seconds(2), 0.0);
        assert_eq!(embedded.feasible_widths(8), vec![2]);

        let block = DrafterSpec::Block {
            widths: vec![(4, 0.003), (8, 0.004)],
        };
        assert_eq!(block.feasible_widths(6), vec![0, 4]);
        assert!(block.needs_draft_launch(4));
        assert!((block.draft_seconds(4) - 0.003).abs() < 1e-12);

        let ar = DrafterSpec::Autoregressive {
            max_width: 3,
            fixed_seconds: 0.001,
            marginal_seconds: 0.0005,
        };
        assert_eq!(ar.feasible_widths(5), vec![0, 1, 2, 3]);
        assert!((ar.draft_seconds(2) - 0.002).abs() < 1e-12);

        // Cumulative survival [0.8, 0.64, 0.512] must NOT be multiplied again!
        let cond = AcceptanceProfile::Conditional(vec![0.8, 0.8, 0.8]);
        let cum = AcceptanceProfile::Cumulative(vec![0.8, 0.64, 0.512]);
        for (a, b) in cond.cumulative_survival().iter().zip(cum.cumulative_survival()) {
            assert!((a - b).abs() < 1e-12, "cumulative survival must match conditional product");
        }
        // Decision Time 1 with EmbeddedChain chooses G = 2 without launching a drafter.
        let costs = PassCostModel {
            compute: base_compute(),
            io: RowMissPricing::ExpectedRates {
                exact_row_misses: 1.0,
                verify_marginal_misses: vec![0.2; 4],
                front_tail_misses: vec![0.0; 4],
                demand_seconds_per_miss: 0.001,
                speculative_seconds_per_miss: 0.0,
                displacement_seconds_per_miss: 0.0,
                ddr_contention_seconds_per_miss: 0.0,
            },
            front_credit: FrontCreditModel::default(),
            verify_host_overhead_seconds: 0.0005,
            verify_host_overhead_per_step_seconds: 0.0,
            non_verify_host_overhead_seconds: 0.0,
        };
        let budget = DecodePassScorer::choose_generation_budget(
            pass_id(7, 1, 1),
            &embedded,
            5,
            &cum,
            HorizonLimits::UNBOUNDED,
            &costs,
        )
        .unwrap();
        assert_eq!(budget.generated, 2);
        assert!(!budget.needs_draft_launch);
        assert_eq!(budget.draft_seconds, 0.0);
    }

    /// Boundary 6 (`astra` m4jrc9 #2): `remaining_tokens == 0` clamps `K = 0`
    /// and `F = 0` even when `G > 0` and acceptance is 1.0.
    #[test]
    fn remaining_tokens_zero_clamps_verify_and_front_to_zero() {
        let costs = PassCostModel {
            compute: base_compute(),
            io: RowMissPricing::ExpectedRates {
                exact_row_misses: 1.0,
                verify_marginal_misses: vec![0.0; 4],
                front_tail_misses: vec![0.0; 4],
                demand_seconds_per_miss: 0.001,
                speculative_seconds_per_miss: 0.0,
                displacement_seconds_per_miss: 0.0,
                ddr_contention_seconds_per_miss: 0.0,
            },
            front_credit: FrontCreditModel {
                exposed_future_wait_by_front: vec![0.005, 0.005],
                carryover_covered_by_front: vec![0.0, 0.0],
            },
            verify_host_overhead_seconds: 0.0,
            verify_host_overhead_per_step_seconds: 0.0,
            non_verify_host_overhead_seconds: 0.0,
        };
        let sealed = DecodePassScorer::seal_plan(
            pass_id(1, 1, 1),
            4,
            0.0,
            5,
            &AcceptanceProfile::Conditional(vec![1.0, 1.0, 1.0, 1.0]),
            HorizonLimits {
                remaining_tokens: 0,
                eos_at: None,
            },
            &costs,
        )
        .unwrap();
        assert_eq!((sealed.verify, sealed.front), (0, 0));
        assert!((sealed.pricing.expected_tokens - 1.0).abs() < 1e-12);

        // Even a direct score_candidate with k=3 at remaining_tokens=0 credits only 1 token.
        let direct = DecodePassScorer::score_candidate(
            0.0,
            3,
            0,
            &[1.0, 1.0, 1.0],
            HorizonLimits {
                remaining_tokens: 0,
                eos_at: None,
            },
            &costs,
        );
        assert!((direct.expected_tokens - 1.0).abs() < 1e-12);
    }

    /// Boundary 7 (`astra` m4jrc9 #3): stop-aware committed prefix drives
    /// EVERY branch judgment — no token past a committed EOS or first-front
    /// mismatch is ever left `Matched` or `Pending`.
    #[test]
    fn stop_aware_commit_and_first_front_mismatch_cancel_all_descendants() {
        let eos = TokenId(999);
        let stops = BTreeSet::from([eos]);
        let dummy_pricing = PassPricing {
            draft_seconds: 0.0,
            compute_seconds: 0.01,
            compulsory_io_seconds: 0.0,
            speculative_traffic_seconds: 0.0,
            displacement_seconds: 0.0,
            ddr_contention_seconds: 0.0,
            host_overhead_seconds: 0.0,
            front_credit_seconds: 0.0,
            total_round_seconds: 0.01,
            expected_tokens: 1.0,
            rate_tps: 100.0,
        };
        let plan = DecodePassPlan {
            pass_id: pass_id(1, 1, 1),
            generated: 4,
            verify: 2,
            front: 2,
            idle_front: 0,
            reserved_capacity: 5,
            pricing: dummy_pricing,
        };

        // Case A: K=2, F=2, proposals=[EOS, x, y, z], samples=[EOS, x, y]
        // Commits ONLY [EOS]; x, y, z MUST all be Unjudged (never Matched or Pending).
        let (x, y, z) = (TokenId(10), TokenId(20), TokenId(30));
        let eos_out =
            PassOutcome::from_samples(&plan, &[eos, x, y, z], &[eos, x, y], &stops).unwrap();
        assert_eq!(eos_out.commitment, PassCommitment::Committed(vec![eos]));
        assert_eq!(
            eos_out.branches[0].judgment,
            PredictionJudgment::Matched { predicted: eos, actual: eos }
        );
        assert_eq!(eos_out.branches[1].judgment, PredictionJudgment::Unjudged { predicted: x });
        assert_eq!(eos_out.branches[2].judgment, PredictionJudgment::Unjudged { predicted: y });
        assert_eq!(eos_out.branches[3].judgment, PredictionJudgment::Unjudged { predicted: z });

        // Case B: First-front mismatch: proposals=[a, b, y, z], samples=[a, b, W]
        // a, b Matched; y Rejected; descendant z MUST be Unjudged (never Pending).
        let (a, b, w) = (TokenId(1), TokenId(2), TokenId(77));
        let mismatch_out =
            PassOutcome::from_samples(&plan, &[a, b, y, z], &[a, b, w], &stops).unwrap();
        assert_eq!(mismatch_out.commitment, PassCommitment::Committed(vec![a, b, w]));
        assert_eq!(
            mismatch_out.branches[0].judgment,
            PredictionJudgment::Matched { predicted: a, actual: a }
        );
        assert_eq!(
            mismatch_out.branches[1].judgment,
            PredictionJudgment::Matched { predicted: b, actual: b }
        );
        assert_eq!(
            mismatch_out.branches[2].judgment,
            PredictionJudgment::Rejected { predicted: y, actual: w }
        );
        assert_eq!(
            mismatch_out.branches[3].judgment,
            PredictionJudgment::Unjudged { predicted: z }
        );
    }

    /// Boundary 8 (`astra` mkf7e4 #4 & #3c): fixed `G=8` block drafter with a
    /// 4-row verify offer (`K=3` legal), consistent current+future horizon
    /// (never `1ms - 2ms -> 1B tok/s`), plateau-proof batch allocation, and
    /// bounded/retirable `PassOutcomeTracker` tombstones.
    #[test]
    fn fixed_block_g8_with_4_row_offer_consistent_horizon_and_tracker_retirement() {
        let costs = PassCostModel {
            compute: ComputePricing { by_rows: vec![0.001, 0.0011, 0.0012, 0.0013] },
            io: RowMissPricing::ExpectedRates {
                exact_row_misses: 0.0,
                verify_marginal_misses: vec![0.0; 8],
                front_tail_misses: vec![0.0; 8],
                demand_seconds_per_miss: 0.0,
                speculative_seconds_per_miss: 0.0,
                displacement_seconds_per_miss: 0.0,
                ddr_contention_seconds_per_miss: 0.0,
            },
            front_credit: FrontCreditModel {
                // Exposed future wait (2ms) > pass compute (1ms): consistent horizon
                // must keep total_round_seconds >= pass_body (never 1ms - 2ms -> 1B tok/s).
                exposed_future_wait_by_front: vec![0.002],
                carryover_covered_by_front: vec![0.0],
            },
            verify_host_overhead_seconds: 0.0001,
            verify_host_overhead_per_step_seconds: 0.0,
            non_verify_host_overhead_seconds: 0.0,
        };
        let scored = DecodePassScorer::score_candidate(
            0.0,
            0,
            1,
            &[1.0],
            HorizonLimits::UNBOUNDED,
            &costs,
        );
        assert!(
            scored.total_round_seconds >= scored.compute_seconds,
            "consistent horizon must keep total_round_seconds >= compute_seconds: {scored:?}"
        );
        assert!(scored.rate_tps < 1e4, "rate must never explode to 1B tok/s: {scored:?}");

        // Fixed G=8 block drafter under a 4-row verify offer chooses G=8, K=3.
        let verify_costs = PassCostModel {
            front_credit: FrontCreditModel::default(),
            ..costs
        };
        let fixed_block = DrafterSpec::Block { widths: vec![(8, 0.0005)] };
        let budget = DecodePassScorer::choose_generation_budget(
            pass_id(1, 1, 1),
            &fixed_block,
            4,
            &AcceptanceProfile::Conditional(vec![0.95; 8]),
            HorizonLimits::UNBOUNDED,
            &verify_costs,
        )
        .unwrap();
        assert_eq!(budget.generated, 8, "fixed G=8 must remain legal under 4-row offer");
        assert_eq!(budget.projected_verify, 3, "K=3 must be chosen within 4-row offer");

        // Batch allocation crosses a zero-marginal plateau (verify overhead makes K=1
        // worse than K=0, while K=2..3 beats K=0).
        let plateau_costs = PassCostModel {
            compute: base_compute(),
            io: RowMissPricing::ExpectedRates {
                exact_row_misses: 0.0,
                verify_marginal_misses: vec![0.0; 4],
                front_tail_misses: vec![0.0; 4],
                demand_seconds_per_miss: 0.0,
                speculative_seconds_per_miss: 0.0,
                displacement_seconds_per_miss: 0.0,
                ddr_contention_seconds_per_miss: 0.0,
            },
            front_credit: FrontCreditModel::default(),
            verify_host_overhead_seconds: 0.012, // K=1: 1.95/(0.011+0.012)=84.8 < K=0: 1/0.010=100; K=3: 3.71/(0.013+0.012)=148.4 > 100!
            verify_host_overhead_per_step_seconds: 0.0,
            non_verify_host_overhead_seconds: 0.0,
        };
        let cand = BatchPlanCandidate {
            pass_id: pass_id(1, 1, 1),
            actual_generated: 3,
            draft_seconds_paid: 0.0,
            acceptance: AcceptanceProfile::Conditional(vec![0.95, 0.95, 0.95]),
            horizon: HorizonLimits::UNBOUNDED,
            costs: plateau_costs,
            cancelled: false,
        };
        let sealed_batch = seal_batch_plans(&[cand], 4).unwrap();
        assert_eq!(
            sealed_batch.plans[0].verify, 3,
            "batch allocator must cross K=1 zero-marginal plateau to reach K=3"
        );

        // PassOutcomeTracker retires old settled tombstones cleanly AND advances
        // watermark so late cancel/verdict/feed after rollover NEVER double-settles.
        let mut tracker = PassOutcomeTracker::default();
        for p in 0..(MAX_SETTLED_RETAINED as u64 + 16) {
            let id = pass_id(1, 1, p);
            assert!(tracker.cancel(id).is_some());
        }
        assert!(tracker.retained_settled_len() <= MAX_SETTLED_RETAINED);
        let rolled_off = pass_id(1, 1, 0);
        assert!(tracker.is_settled(rolled_off), "rolled-off pass 0 must stay settled");
        assert!(tracker.cancel(rolled_off).is_none(), "late cancel(0) after rollover must not double-settle");
        assert!(
            tracker
                .record_verdict(PassOutcome::discarded(rolled_off, None, &[]))
                .is_none(),
            "late verdict(0) after rollover must not double-settle"
        );
        let rolled_plan = DecodePassPlan {
            pass_id: rolled_off,
            generated: 0,
            verify: 0,
            front: 0,
            idle_front: 0,
            reserved_capacity: 1,
            pricing: scored,
        };
        assert!(
            tracker.feed(rolled_plan, Vec::new()).is_none(),
            "late feed(0) after rollover must not double-settle"
        );
        tracker.retire_before(RequestId(1), 2);
        assert_eq!(tracker.retained_settled_len(), 0);
        assert!(tracker.is_settled(rolled_off));
    }

    /// Boundary 9 (`astra` m2m7wv #4 & #8): physical idle padding stays
    /// distinct from productive guesses (`FrontGrant::Exactly(4)` with 1 guess
    /// -> 1 exact + 1 guess + 3 idle = 5 executed rows), and known `eos_at`
    /// bounds `K` as well as `F`.
    #[test]
    fn idle_padding_distinct_from_productive_guesses_and_eos_bounds_k() {
        let costs = PassCostModel {
            compute: base_compute(),
            io: RowMissPricing::ExpectedRates {
                exact_row_misses: 2.0,
                verify_marginal_misses: vec![0.0; 5],
                front_tail_misses: vec![0.0; 5],
                demand_seconds_per_miss: 0.001,
                speculative_seconds_per_miss: 0.0,
                displacement_seconds_per_miss: 0.0,
                ddr_contention_seconds_per_miss: 0.0,
            },
            front_credit: FrontCreditModel {
                exposed_future_wait_by_front: vec![0.004, 0.004],
                carryover_covered_by_front: vec![0.0, 0.0],
            },
            verify_host_overhead_seconds: 0.0,
            verify_host_overhead_per_step_seconds: 0.0,
            non_verify_host_overhead_seconds: 0.0,
        };
        // 1 existing guess + 3 idle padding rows -> 5 physical rows, credit only for 1 guess.
        let cum = AcceptanceProfile::Conditional(vec![0.9]).cumulative_survival();
        let pricing = DecodePassScorer::score_candidate_with_idle(
            0.0,
            0,
            1,
            3,
            &cum,
            HorizonLimits::UNBOUNDED,
            &costs,
        );
        assert!(
            (pricing.compute_seconds - costs.compute.seconds(5)).abs() < 1e-12,
            "must charge executed 5-row class"
        );
        assert!(
            (pricing.front_credit_seconds - 0.9 * 0.004).abs() < 1e-12,
            "idle rows must receive zero front credit"
        );
        let plan = DecodePassPlan {
            pass_id: pass_id(1, 1, 1),
            generated: 1,
            verify: 0,
            front: 1,
            idle_front: 3,
            reserved_capacity: 5,
            pricing,
        };
        assert_eq!(plan.total_rows(), 5);
        assert_eq!(plan.productive_guesses(), 1);
        let rows = plan.build_rows(&[TokenId(42)], GuessKind::Chained).unwrap();
        assert_eq!(rows.row_count(), 5);
        assert_eq!(rows.guesses().len(), 1);
        assert_eq!(rows.front_rows(), 4);
        assert_eq!(
            rows.tokens(TokenId(7)),
            vec![TokenId(7), TokenId(42), TokenId(7), TokenId(7), TokenId(7)]
        );

        // Known EOS at proposal index 1 bounds K <= 1 (and F == 0) even with remaining_tokens = 100.
        let sealed_eos = DecodePassScorer::seal_plan(
            pass_id(1, 1, 2),
            4,
            0.0,
            5,
            &AcceptanceProfile::Conditional(vec![1.0, 1.0, 1.0, 1.0]),
            HorizonLimits {
                remaining_tokens: 100,
                eos_at: Some(1),
            },
            &costs,
        )
        .unwrap();
        assert!(sealed_eos.verify <= 1, "eos_at=Some(1) must bound K <= 1: {sealed_eos:?}");
        assert_eq!(sealed_eos.front, 0, "eos_at=Some(1) with K=1 must suppress F: {sealed_eos:?}");
    }

    /// Boundary 10 (`astra` mh3mxa #3): mixed `K > 0 && F > 0` chosen and
    /// lowered for BOTH DSpark (`DrafterSpec::Block`) and Embedded
    /// `TwoRowDecode` (`DrafterSpec::from_two_row` + `FrontGrant::Mixed`).
    #[test]
    fn mixed_k_and_f_demonstrated_for_dspark_block_and_embedded_two_row() {
        let costs = PassCostModel {
            compute: base_compute(),
            io: RowMissPricing::ExpectedRates {
                exact_row_misses: 4.0,
                verify_marginal_misses: vec![1.0, 1.0, 8.0, 8.0],
                front_tail_misses: vec![1.0, 1.0],
                demand_seconds_per_miss: 0.002,
                speculative_seconds_per_miss: 0.0002,
                displacement_seconds_per_miss: 0.0,
                ddr_contention_seconds_per_miss: 0.0,
            },
            front_credit: FrontCreditModel {
                exposed_future_wait_by_front: vec![0.008],
                carryover_covered_by_front: vec![0.0],
            },
            verify_host_overhead_seconds: 0.001,
            verify_host_overhead_per_step_seconds: 0.0,
            non_verify_host_overhead_seconds: 0.0,
        };
        // 1) DSpark block drafter: Decision Time 1 chooses G = 4; Decision Time 2 seals K > 0 && F > 0!
        let dspark = DrafterSpec::Block { widths: vec![(4, 0.001)] };
        let acc = AcceptanceProfile::Conditional(vec![0.95, 0.95, 0.90, 0.90]);
        let budget = DecodePassScorer::choose_generation_budget(
            pass_id(1, 1, 1),
            &dspark,
            5,
            &acc,
            HorizonLimits::UNBOUNDED,
            &costs,
        )
        .unwrap();
        assert_eq!(budget.generated, 4);
        assert!(budget.needs_draft_launch);
        let sealed = DecodePassScorer::seal_plan(
            pass_id(1, 1, 1),
            4,
            budget.draft_seconds,
            5,
            &acc,
            HorizonLimits::UNBOUNDED,
            &costs,
        )
        .unwrap();
        assert!(sealed.verify > 0 && sealed.front > 0, "DSpark must choose mixed K>0 && F>0: {sealed:?}");
        let proposals = [TokenId(10), TokenId(11), TokenId(12), TokenId(13)];
        let rows = sealed.build_rows(&proposals, GuessKind::Chained).unwrap();
        assert_eq!(rows.acceptable().len(), sealed.verify);
        assert_eq!(rows.front_rows(), sealed.front);

        // 2) Embedded TwoRowDecode: supplies existing chain (no new drafter), seals K > 0 && F > 0,
        //    and lowers identically via `FrontGrant::Mixed`.
        let mut two_row = TwoRowDecode::default();
        two_row.guessed(vec![
            Guess { token: TokenId(10), kind: GuessKind::Chained },
            Guess { token: TokenId(11), kind: GuessKind::Chained },
            Guess { token: TokenId(12), kind: GuessKind::Head },
        ]);
        let embedded = DrafterSpec::from_two_row(&two_row);
        assert!(!embedded.needs_draft_launch(3));
        let emb_budget = DecodePassScorer::choose_generation_budget(
            pass_id(2, 1, 1),
            &embedded,
            4,
            &AcceptanceProfile::Conditional(vec![0.95, 0.95, 0.90]),
            HorizonLimits::UNBOUNDED,
            &costs,
        )
        .unwrap();
        assert_eq!(emb_budget.generated, 3);
        assert!(!emb_budget.needs_draft_launch);
        assert!(
            emb_budget.projected_verify > 0 && emb_budget.projected_front > 0,
            "Embedded TwoRowDecode must choose mixed K>0 && F>0: {emb_budget:?}"
        );
        let grant_rows = two_row.pass(magnitude_executor::FrontGrant::Mixed {
            verify: emb_budget.projected_verify,
            front: emb_budget.projected_front,
        });
        assert_eq!(grant_rows.acceptable().len(), emb_budget.projected_verify);
        assert_eq!(grant_rows.front_rows(), emb_budget.projected_front);
    }

    /// All 4 physical streaming regimes work automatically under the single
    /// `PassCostModel::from_streamed_prior` + `DecodePassScorer` cost function,
    /// with multi-row `F` credit, carryover suppression, and no double-counting:
    /// - Regime 1 (GPU/DRAM-bound): picks `F = 0`.
    /// - Regime 2 (fast link + high-acceptance drafter): picks `K in 3..=5, F in 1..=2`
    ///   (and specifically reaches `F = 2` when link waste is low).
    /// - Regime 3 (heavy SSD-bound + high-acceptance drafter): picks `F = 1` when
    ///   acceptance is high, drops to `F = 0` when acceptance dips.
    /// - Regime 4 (very SSD-bound Mac where verify hurts): picks `K = 0, F in 1..=3`
    ///   (reaching `F = 2` and `F = 3` in chained passes).
    #[test]
    fn four_regimes_multi_front_credit_under_single_cost_function() {
        // Regime 1: Pure resident / GPU-bound with zero exposed stall -> F = 0.
        let r1_prior = StreamedVerifyPrior {
            regime_epoch: 1,
            verify_seconds: vec![0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022],
            compute_seconds: vec![0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022],
            io_seconds: vec![0.0; 8],
            front_stall_saved_per_row: 0.0,
            front_waste_per_row: vec![0.0; 7],
            verify_host_overhead_seconds: 0.001,
            ..Default::default()
        };
        let r1_costs = PassCostModel::from_streamed_prior(&r1_prior);
        assert!(
            r1_costs.front_credit.exposed_future_wait_by_front.is_empty(),
            "resident zero-stall regime must grant zero front credit"
        );
        let r1_plan = DecodePassScorer::seal_plan(
            pass_id(1, 1, 1),
            6,
            0.0,
            7,
            &AcceptanceProfile::Conditional(vec![0.95; 6]),
            HorizonLimits::UNBOUNDED,
            &r1_costs,
        )
        .unwrap();
        assert_eq!(r1_plan.front, 0, "Regime 1 (GPU-bound, zero stall) must pick F = 0: {r1_plan:?}");

        // Regime 1b: GPU-bound where tiny residual stall (0.5 ms) < extra row compute (2 ms) -> F = 0 automatically.
        let r1b_prior = StreamedVerifyPrior {
            regime_epoch: 1,
            verify_seconds: vec![0.015, 0.017, 0.019, 0.021, 0.023, 0.025, 0.027],
            compute_seconds: vec![0.015, 0.017, 0.019, 0.021, 0.023, 0.025, 0.027],
            io_seconds: vec![0.0005, 0.0008, 0.001, 0.0012, 0.0014, 0.0016, 0.0018],
            front_stall_saved_per_row: 0.0005,
            front_waste_per_row: vec![0.0002; 6],
            verify_host_overhead_seconds: 0.001,
            ..Default::default()
        };
        let r1b_plan = DecodePassScorer::seal_plan(
            pass_id(1, 1, 2),
            6,
            0.0,
            7,
            &AcceptanceProfile::Conditional(vec![0.95; 6]),
            HorizonLimits::UNBOUNDED,
            &PassCostModel::from_streamed_prior(&r1b_prior),
        )
        .unwrap();
        assert_eq!(
            r1b_plan.front, 0,
            "GPU-bound regime with exposed stall < ΔC must pick F = 0: {r1b_plan:?}"
        );

        // Exposed-stall UMA regime (`0 < D < C`, e.g. Sparky GB10 `74fbe7`:
        // `C = 127 ms`, `D = 37 ms` synchronous SSD stall serialized as `C + D`).
        // Removing the broken `io > compute` gate lets `F >= 1` overlap that 37 ms stall!
        let sparky_prior = StreamedVerifyPrior {
            regime_epoch: 5,
            verify_seconds: vec![0.127, 0.133, 0.139, 0.145, 0.151, 0.157, 0.163, 0.169],
            compute_seconds: vec![0.127, 0.133, 0.139, 0.145, 0.151, 0.157, 0.163, 0.169],
            io_seconds: vec![0.037, 0.044, 0.051, 0.058, 0.065, 0.072, 0.079, 0.086],
            front_stall_saved_per_row: 0.037,
            front_waste_per_row: vec![0.003; 7],
            verify_host_overhead_seconds: 0.003,
            ..Default::default()
        };
        let sparky_costs = PassCostModel::from_streamed_prior(&sparky_prior);
        assert!(
            !sparky_costs.front_credit.exposed_future_wait_by_front.is_empty(),
            "0 < D < C exposed-stall UMA must retain front credit"
        );
        let sparky_plan = DecodePassScorer::seal_unchained_plan_on_widths(
            pass_id(5, 1, 1),
            7,
            0.0,
            8,
            0..=5,
            4,
            &AcceptanceProfile::Conditional(vec![0.92; 7]),
            HorizonLimits::UNBOUNDED,
            &sparky_costs,
        )
        .unwrap();
        assert!(
            sparky_plan.verify == 5 && sparky_plan.front >= 1,
            "Sparky GB10 (0 < D < C exposed-stall UMA) must pick K = 5, F >= 1 (was starved to F = 0 by path_bound gate): {sparky_plan:?}"
        );

        // Regime 2: Fast link (5090 PCIe 5) + high-acceptance drafter -> K in 3..=5, F in 1..=2.
        let r2_prior = StreamedVerifyPrior {
            regime_epoch: 2,
            verify_seconds: vec![0.024, 0.030, 0.036, 0.042, 0.048, 0.068, 0.092, 0.120],
            compute_seconds: vec![0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015],
            io_seconds: vec![0.024, 0.030, 0.036, 0.042, 0.048, 0.068, 0.092, 0.120],
            front_stall_saved_per_row: 0.016,
            front_waste_per_row: vec![0.003; 7],
            verify_host_overhead_seconds: 0.002,
            ..Default::default()
        };
        let r2_costs = PassCostModel::from_streamed_prior(&r2_prior);
        let r2_acc = AcceptanceProfile::Conditional(vec![0.95; 7]);
        let r2_plan = DecodePassScorer::seal_unchained_plan_on_widths(
            pass_id(2, 1, 1),
            7,
            0.0,
            8,
            0..=7,
            4,
            &r2_acc,
            HorizonLimits::UNBOUNDED,
            &r2_costs,
        )
        .unwrap();
        assert!(
            (3..=5).contains(&r2_plan.verify) && (1..=2).contains(&r2_plan.front),
            "Regime 2 (fast link + high acc) must pick K in 3..=5, F in 1..=2: {r2_plan:?}"
        );
        assert_eq!(
            r2_plan.front, 2,
            "Regime 2 with low link waste must reach F = 2 (was blocked by hardcoded [s, 0.0]): {r2_plan:?}"
        );

        // Regime 3: Heavy SSD-bound + high-acceptance drafter -> F = 1 when high, F = 0 when acc dips.
        let r3_prior = StreamedVerifyPrior {
            regime_epoch: 3,
            verify_seconds: vec![0.080, 0.105, 0.130, 0.165, 0.210, 0.260, 0.315],
            compute_seconds: vec![0.015, 0.017, 0.019, 0.021, 0.023, 0.025, 0.027],
            io_seconds: vec![0.080, 0.105, 0.130, 0.165, 0.210, 0.260, 0.315],
            front_stall_saved_per_row: 0.020,
            front_waste_per_row: vec![0.010; 6],
            verify_host_overhead_seconds: 0.003,
            ..Default::default()
        };
        let r3_costs = PassCostModel::from_streamed_prior(&r3_prior);
        let r3_high = DecodePassScorer::seal_plan(
            pass_id(3, 1, 1),
            6,
            0.0,
            7,
            &AcceptanceProfile::Conditional(vec![0.90; 6]),
            HorizonLimits::UNBOUNDED,
            &r3_costs,
        )
        .unwrap();
        assert_eq!(
            r3_high.front, 1,
            "Regime 3 (heavy SSD + high acc) must pick F = 1: {r3_high:?}"
        );
        let r3_dip = DecodePassScorer::seal_plan(
            pass_id(3, 1, 2),
            6,
            0.0,
            7,
            &AcceptanceProfile::Conditional(vec![0.40; 6]),
            HorizonLimits::UNBOUNDED,
            &r3_costs,
        )
        .unwrap();
        assert_eq!(
            r3_dip.front, 0,
            "Regime 3 (heavy SSD + acceptance dip) must drop to F = 0: {r3_dip:?}"
        );

        // Regime 4: Very SSD-bound Mac where verify hurts (each verify row adds
        // a full demand miss burst + verify host turn overhead) -> K = 0, F in 1..=3.
        let r4_prior = StreamedVerifyPrior {
            regime_epoch: 4,
            verify_seconds: vec![0.140, 0.280, 0.420, 0.560, 0.700, 0.840, 0.980],
            compute_seconds: vec![0.032, 0.034, 0.036, 0.038, 0.040, 0.042, 0.044],
            io_seconds: vec![0.140, 0.280, 0.420, 0.560, 0.700, 0.840, 0.980],
            front_stall_saved_per_row: 0.040,
            front_waste_per_row: vec![0.004; 6],
            verify_host_overhead_seconds: 0.025,
            ..Default::default()
        };
        let r4_costs = PassCostModel::from_streamed_prior(&r4_prior);
        let r4_plan = DecodePassScorer::seal_plan(
            pass_id(4, 1, 1),
            6,
            0.0,
            7,
            &AcceptanceProfile::Conditional(vec![0.88; 6]),
            HorizonLimits::UNBOUNDED,
            &r4_costs,
        )
        .unwrap();
        assert_eq!(
            r4_plan.verify, 0,
            "Regime 4 (very SSD-bound Mac where verify hurts) must pick K = 0: {r4_plan:?}"
        );
        assert!(
            (1..=3).contains(&r4_plan.front),
            "Regime 4 must pick F in 1..=3: {r4_plan:?}"
        );
        assert_eq!(
            r4_plan.front, 3,
            "Regime 4 with low standing waste must reach F = 3: {r4_plan:?}"
        );

        // Carryover suppression & non-double-counting invariant:
        let mut r4_carry = r4_prior.clone();
        r4_carry.carryover_covered_by_front = vec![0.040, 0.020, 0.010, 0.005, 0.0025, 0.00125];
        let r4_carry_costs = PassCostModel::from_streamed_prior(&r4_carry);
        let r4_carry_plan = DecodePassScorer::seal_plan(
            pass_id(4, 1, 2),
            6,
            0.0,
            7,
            &AcceptanceProfile::Conditional(vec![0.88; 6]),
            HorizonLimits::UNBOUNDED,
            &r4_carry_costs,
        )
        .unwrap();
        assert_eq!(
            r4_carry_plan.front, 0,
            "carryover covering all exposed wait must suppress F to 0: {r4_carry_plan:?}"
        );

        // Credit past exposed_future_wait_by_front.len() must never exceed total_exposed_wait.
        let cum = AcceptanceProfile::Conditional(vec![0.95; 10]).cumulative_survival();
        let total = r4_costs
            .front_credit
            .total_exposed_wait(&cum, 0, HorizonLimits::UNBOUNDED);
        let credit_over = r4_costs
            .front_credit
            .credit_seconds(&cum, 0, 10, HorizonLimits::UNBOUNDED);
        assert!(
            (credit_over - total).abs() < 1e-12 && credit_over <= total + 1e-12,
            "credit_seconds past table end must never exceed total_exposed_wait: {credit_over} vs {total}"
        );
    }
}
