//! True-route shadows at every depth of foresight and their real-vs-shadow gaps
//! (`designs/streamed-decode-ranking.md` §3.7, §4, §8 A9).
//!
//! Each depth `j` in `0..DEPTHS` runs its own pools, turnover (`horizon::Time`)
//! and decode-miss window (`horizon::Window`) `BEHIND = DEPTHS - 1` steps behind
//! the log on **true routes** (every ahead token's actual route, stand rate 1):
//! a live lower bound `U_shadow(j)` that tracks the workload whatever depth the
//! run executes. When depth `j` executes, the gap `Δ_j = U_real − U_shadow(j)`
//! is measured into an `horizon::Aged` window on the bank-wide epoch; while `j`
//! does not run, `Δ_j` holds its last value with its age in tokens and
//! `U(j) = U_shadow(j) + Δ_j` follows the live shadow. Before `j` first runs,
//! `Δ_j` is `Delta::Unmeasured` (0), so `U(j)` is the lower bound and the regime
//! tries `j` at most once.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use super::super::directory::RoutedRead;
use super::super::horizon::{self, Advance, Horizon, Reader, RegimeAged, RegimeEpoch, RegimeWindow, Tier, Tokens};
use super::super::watcher::Step;
use super::super::DEPTHS;
use super::calibration::{Calibration, ShadowScratch};
use super::history::History;
use super::{Evidence, Pools, Rows};

/// How many complete decode steps behind the log the shadows read: the deepest
/// foresight measured, in tokens. Every decode step emits at least one token
/// (its exact row's), so every depth's true routes are logged when the step is
/// read.
pub(super) const BEHIND: usize = DEPTHS - 1;

/// Depth `j`'s term of `U` for the regime (design §3.7): `U(j) = U_shadow(j) + Δ_j`,
/// printed as `(U_shadow, Δ, age)` with every regime decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gap {
    /// `U_shadow(j)`: the true-route shadow's decode misses a token, live at
    /// every depth whatever executes.
    pub shadow: f64,
    pub delta: Delta,
}

/// `Δ_j = U_real − U_shadow(j)`, a regime-scoped `RegimeAged` window on the
/// bank-wide epochs counted only on steps that execute depth `j` within the
/// same physical [`RegimeEpoch`] (design §3.7, A9).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Delta {
    /// `j` has never executed in the current physical regime: `Δ_j = 0`,
    /// `U(j)` is the lower bound.
    Unmeasured,
    /// Measured in `regime`; `age` is the tokens since a step last executed `j`
    /// (0 while it runs), the frozen gap's age while it does not. Inapplicable
    /// by construction once the bank's [`RegimeEpoch`] advances past `regime`.
    Measured { regime: RegimeEpoch, delta: f64, age: Tokens },
}

impl Gap {
    /// `U(j) = max(0, U_shadow(j) + Δ_j)` (`Δ_j = 0` while unmeasured).
    /// Misses per token are physically non-negative; clamping prevents a
    /// cold-start negative `Δ_j` from turning `U(j)` negative once the shadow
    /// warms up. Panics where the result is not finite.
    pub(crate) fn needed(self) -> f64 {
        let delta = match self.delta {
            Delta::Unmeasured => 0.0,
            Delta::Measured { delta, .. } => delta,
        };
        let raw = self.shadow + delta;
        assert!(raw.is_finite(), "non-finite gap: {self:?}");
        raw.max(0.0)
    }
}

impl fmt::Display for Gap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.delta {
            Delta::Unmeasured => write!(f, "(U_shadow={:.3}, Δ=unmeasured)", self.shadow),
            Delta::Measured { delta, age, .. } => {
                write!(f, "(U_shadow={:.3}, Δ={:+.3}, age={:.1})", self.shadow, delta, age.get())
            }
        }
    }
}

/// One re-entry comparison for assumption A9 (design §4, §8): the first window
/// of `Δ_j` measured after re-entering a previously frozen depth, against its
/// frozen value and age.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Reentry {
    pub depth: usize,
    pub frozen: f64,
    pub age: Tokens,
    pub observed: f64,
    pub stderr: f64,
    pub samples: u64,
}

impl fmt::Display for Reentry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "A9 depth {} re-entry: frozen={:+.3} (age={:.1} tok) vs observed={:+.3} ± {:.3} (n={})",
            self.depth,
            self.frozen,
            self.age.get(),
            self.observed,
            self.stderr,
            self.samples,
        )
    }
}

/// What the ranking logged for its shadows (`Reuse::shadows`), in the order
/// logged: each read with its rows, its evidence kind and its step's position
/// on the token clock (`horizon::Clock`), and each resize of a group's pools.
pub(crate) enum ShadowFeed {
    Read {
        read: RoutedRead,
        rows: Option<Arc<Rows>>,
        evidence: Evidence,
        clock: Tokens,
    },
    Resize {
        group: usize,
        records: usize,
    },
}

/// One token's true route at a layer, `1..DEPTHS` tokens ahead of the step a
/// shadow is reading: either a keyed row's depth in `Rows`, or the unkeyed
/// experts the step read.
pub(super) enum TrueRoute<'a> {
    Row { rows: &'a Rows, depth: u8 },
    Unkeyed { experts: &'a [usize] },
}

/// A logged read the shadows have yet to read.
struct Logged {
    read: RoutedRead,
    rows: Option<Arc<Rows>>,
    evidence: Evidence,
}

impl Logged {
    /// Its true route at row depth `row_depth` (0 the exact row, `d > 0` an
    /// accepted verify row of depth `d`); none where no such row was logged.
    fn true_route(&self, row_depth: u8) -> Option<TrueRoute<'_>> {
        match (row_depth, self.rows.as_deref()) {
            (0, Some(rows)) if rows.has_depth(0) => Some(TrueRoute::Row { rows, depth: 0 }),
            (0, _) => Some(TrueRoute::Unkeyed { experts: &self.read.experts }),
            (depth, Some(rows)) if rows.has_depth(depth) => Some(TrueRoute::Row { rows, depth }),
            _ => None,
        }
    }
}

/// A step of the log the shadows have yet to read: its sequence, its clock
/// position (`Tokens::ZERO` for prompt-only steps), and its reads in the order
/// logged.
struct Behind {
    sequence: Step,
    clock: Tokens,
    reads: Vec<Logged>,
}

impl Behind {
    fn is_decode(&self) -> bool {
        self.reads.iter().any(|logged| logged.evidence == Evidence::Decode)
    }
}

/// Decode misses and emitted tokens counted in one epoch of a shadow's
/// bank-wide turnover.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Missed {
    misses: u64,
    tokens: f64,
}

impl horizon::Tally for Missed {
    fn is_empty(&self) -> bool {
        self.tokens == 0.0
    }
}

/// Observations of `Δ_j = U_real − U_shadow(j)` in one bank-wide epoch, with
/// squared sum for A9's standard error.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct DeltaTally {
    sum: f64,
    sum_sq: f64,
    count: u64,
}

impl horizon::Tally for DeltaTally {
    fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl DeltaTally {
    fn add(&mut self, value: f64) {
        self.sum += value;
        self.sum_sq += value * value;
        self.count += 1;
    }

    fn mean(previous: &Self, current: &Self) -> f64 {
        let count = previous.count + current.count;
        assert!(count > 0, "mean of an empty delta tally");
        (previous.sum + current.sum) / count as f64
    }

    fn single_mean(&self) -> f64 {
        assert!(self.count > 0, "mean of an empty delta tally");
        self.sum / self.count as f64
    }

    /// Standard error of the mean (0 when fewer than 2 samples).
    fn stderr(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        let mean = self.sum / n;
        let variance = ((self.sum_sq - n * mean * mean) / (n - 1.0)).max(0.0);
        (variance / n).sqrt()
    }
}

/// One depth's shadow state: its pools, its own per-group and bank-wide
/// turnovers (`horizon::Time`), and its decode-miss window (`U_shadow`).
struct DepthShadow {
    pools: Pools,
    time: horizon::Time,
    missed: horizon::Window<Missed>,
}

impl DepthShadow {
    fn new(experts: &[usize], records: &[usize]) -> Self {
        Self::with_pools(Pools::new(experts, records))
    }

    fn with_pools(pools: Pools) -> Self {
        let tiers: Vec<Vec<u64>> = pools.records.iter().map(|&count| vec![count as u64]).collect();
        Self {
            pools,
            time: horizon::Time::new(&tiers, 2),
            missed: horizon::Window::new(0),
        }
    }
}

/// An in-progress A9 re-entry measurement for a depth that resumed after being
/// frozen within the same physical [`RegimeEpoch`].
#[derive(Clone, Copy, Debug)]
struct PendingReentry {
    regime: RegimeEpoch,
    frozen: f64,
    age: Tokens,
    entered_epoch: u64,
    fresh: DeltaTally,
}

/// The shadows of the bank at every depth of foresight (`0..DEPTHS`), [`BEHIND`]
/// decode steps behind the log on true routes, plus each depth's real-vs-shadow
/// gap `Δ_j`. They read apart from the live ranking, under no lock of its, from
/// what it logged for them ([`Self::read_log`]).
pub(crate) struct Shadows {
    /// True-route calibration per layer (`Calibration::new_shadow`: route and
    /// sample ring only; no `fell`, `history`, or `far` tables).
    layers: Vec<Calibration>,
    /// Measured exponential-decay half-life history selector for true routes.
    history: History,
    groups: Vec<usize>,
    /// The steps logged the shadows have yet to read, the latest still logging.
    log: VecDeque<Behind>,
    /// Per-depth shadow pools, turnovers, and `U_shadow` windows.
    depths: [DepthShadow; DEPTHS],
    /// Per-depth front-pass `Δ_j = U_real − U_shadow(j)` keyed to the physical
    /// [`RegimeEpoch`] it was measured in and rolling on the bank-wide epoch.
    /// Any physical regime transition makes prior-regime deltas inapplicable
    /// by construction.
    deltas: [RegimeAged<DeltaTally>; DEPTHS],
    /// Single shared live verify-pass per-row delta
    /// `Δ_verify = (missed_total / (1 + k)) − U_shadow(k)` across all verify
    /// depths in the current physical [`RegimeEpoch`]. Aggregating across every
    /// verify depth eliminates per-depth `n = 1` winner's-curse freezing while
    /// `U_shadow(k)` + `shadow_verify_misses` + live `Stands` capture the
    /// exact depth-dependent foresight, sharing, and rejection costs.
    verify_delta: RegimeWindow<DeltaTally>,
    /// The `(regime, depth)` executed on the previous front [`Self::executed`]
    /// call, for within-regime A9 re-entry detection.
    last_executed: Option<(RegimeEpoch, usize)>,
    /// Pending and latest completed A9 re-entry comparison per front depth.
    pending_reentry: [Option<PendingReentry>; DEPTHS],
    reentries: [Option<Reentry>; DEPTHS],
    /// Fixed-size scratch buffers reused across layers/steps in `prepare_shadow`.
    scratch: ShadowScratch,
}

impl Shadows {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn new(experts: &[usize], groups: &[usize], records: &[usize]) -> Self {
        Self::new_with_pools(experts, groups, &Pools::new(experts, records))
    }

    /// Construct [`Shadows`] seeded with the live ranking's current resident
    /// `pools` so `U_shadow` starts warm beside the live bank instead of
    /// suffering cold-start empty-pool misses.
    pub(super) fn new_with_pools(experts: &[usize], groups: &[usize], pools: &Pools) -> Self {
        assert_eq!(experts.len(), groups.len(), "each layer has a group");
        assert!(groups.iter().all(|&group| group < pools.records.len()), "group out of range");
        let max_experts = experts.iter().copied().max().unwrap_or(0);
        Self {
            layers: experts.iter().map(|&count| Calibration::new_shadow(count, 1, 0)).collect(),
            history: History::new(experts, groups, &pools.records),
            groups: groups.to_vec(),
            log: VecDeque::new(),
            depths: std::array::from_fn(|_| DepthShadow::with_pools(pools.clone())),
            deltas: std::array::from_fn(|_| RegimeAged::new(RegimeEpoch::ZERO, 0)),
            verify_delta: RegimeWindow::new(RegimeEpoch::ZERO, 0),
            last_executed: None,
            pending_reentry: [None; DEPTHS],
            reentries: [None; DEPTHS],
            scratch: ShadowScratch::new(max_experts),
        }
    }

    /// Fixed memory charge of the shadows' calibration tables and history
    /// selector for `experts` layers (constant in run length, design §1 bar 2, §4).
    pub(super) fn calibration_bytes(experts: &[usize], groups: &[usize], records: &[usize]) -> u64 {
        let cal: u64 = experts.iter().map(|&count| Calibration::new_shadow(count, 1, 0).bytes()).sum();
        cal + History::new(experts, groups, records).bytes()
    }

    /// Read what the ranking logged since ([`ShadowFeed`]), in order.
    pub(crate) fn read_log(&mut self, fed: Vec<ShadowFeed>) {
        for item in fed {
            match item {
                ShadowFeed::Read { read, rows, evidence, clock } => {
                    self.log(Logged { read, rows, evidence }, clock);
                }
                ShadowFeed::Resize { group, records } => self.resize(group, records),
            }
        }
    }

    /// A step that executed `(verifies, depth)` measured `real` misses a token
    /// (`U_real` for front passes, or per-row `missed_total / (1 + depth)` for
    /// verify passes): counts `real − U_shadow(depth)` into its regime-scoped
    /// window on `(regime, epoch)` at clock `now` (design §3.7, A9).
    pub(crate) fn executed(&mut self, verifies: bool, depth: usize, real: f64, regime: RegimeEpoch, epoch: u64, now: Tokens) {
        assert!(depth < DEPTHS, "depth {depth} past {DEPTHS}");
        assert!(real.is_finite() && real >= 0.0, "invalid real misses a token: {real}");
        let Some(shadow) = self.shadow_misses(depth) else {
            return;
        };
        let diff = real - shadow;
        assert!(diff.is_finite(), "non-finite delta: {real} - {shadow}");

        if verifies {
            self.verify_delta.count(regime, epoch).add(diff);
            return;
        }

        // Detect within-regime front depth switch for A9 re-entry tracking.
        // Cross-regime switches discard stale pending re-entries rather than
        // comparing deltas across different tier/residency regimes.
        if let Some((prev_regime, prev_depth)) = self.last_executed {
            if prev_regime != regime {
                self.pending_reentry = [None; DEPTHS];
            } else if prev_depth != depth {
                self.finalize_reentry(prev_depth);
                if let Some(((prev_t, cur_t), age)) = self.deltas[depth].read(regime, epoch, now) {
                    self.pending_reentry[depth] = Some(PendingReentry {
                        regime,
                        frozen: DeltaTally::mean(prev_t, cur_t),
                        age,
                        entered_epoch: epoch,
                        fresh: DeltaTally::default(),
                    });
                }
            }
        }
        self.last_executed = Some((regime, depth));

        // Discard any pending re-entry from an older regime, and finalize one
        // whose first epoch in the same regime has completed.
        if self.pending_reentry[depth].is_some_and(|pending| pending.regime != regime) {
            self.pending_reentry[depth] = None;
        } else if self.pending_reentry[depth].is_some_and(|pending| epoch > pending.entered_epoch) {
            self.finalize_reentry(depth);
        }

        if let Some(pending) = &mut self.pending_reentry[depth] {
            pending.fresh.add(diff);
        }
        self.deltas[depth].count(regime, epoch, now).add(diff);
    }

    /// Every `(form, depth)`'s `(U_shadow, Δ, age)` (`0 = front`, `1 = verify`)
    /// at physical regime `regime`, bank-wide epoch `epoch`, and clock `now`;
    /// none at a depth whose shadow has measured no token yet (design §3.7, A9).
    /// Deltas measured in an earlier physical regime are returned as
    /// `Delta::Unmeasured` by construction. Verify depths share one live
    /// regime-scoped per-row delta (`verify_delta`) so no single depth can
    /// freeze out others via single-step noise.
    pub(crate) fn gaps(&mut self, regime: RegimeEpoch, epoch: u64, now: Tokens) -> [[Option<Gap>; DEPTHS]; 2] {
        let shadows: [Option<f64>; DEPTHS] = std::array::from_fn(|depth| self.shadow_misses(depth));
        let front = std::array::from_fn(|depth| {
            let shadow = shadows[depth]?;
            let delta = match self.deltas[depth].read(regime, epoch, now) {
                None => Delta::Unmeasured,
                Some(((previous, current), age)) => Delta::Measured {
                    regime,
                    delta: DeltaTally::mean(previous, current),
                    age,
                },
            };
            Some(Gap { shadow, delta })
        });
        let shared_verify = self.verify_delta.read(regime, epoch).map(|(previous, current)| Delta::Measured {
            regime,
            delta: DeltaTally::mean(previous, current),
            age: Tokens::ZERO,
        });
        let verify = std::array::from_fn(|depth| {
            let shadow = shadows[depth]?;
            Some(Gap { shadow, delta: shared_verify.unwrap_or(Delta::Unmeasured) })
        });
        [front, verify]
    }

    /// `U_shadow(k)` at each depth `k` of foresight: the true-route shadow's
    /// decode misses a token over its bank-wide turnover window, [`BEHIND`]
    /// steps behind the log; none before a decode step has been read.
    pub(crate) fn misses(&mut self) -> [Option<f64>; DEPTHS] {
        std::array::from_fn(|depth| self.shadow_misses(depth))
    }

    /// Completed A9 re-entry comparisons since the last take (design §8 A9).
    pub(crate) fn take_reentries(&mut self) -> Vec<Reentry> {
        self.reentries.iter_mut().filter_map(Option::take).collect()
    }

    /// Memory charge of the shadows' calibration tables, history selector,
    /// sample rings, and pools, constant in run length (design §1 bar 2, §3.4, §4).
    pub(super) fn bytes(&self) -> u64 {
        let calibration: u64 = self.layers.iter().map(Calibration::bytes).sum();
        let pools: u64 = self.depths.iter().map(|shadow| shadow.pools.bytes()).sum();
        std::mem::size_of::<Self>() as u64 + calibration + self.history.bytes() + pools + self.scratch.bytes()
    }

    fn finalize_reentry(&mut self, depth: usize) {
        if let Some(pending) = self.pending_reentry[depth].take() {
            if pending.fresh.count > 0 {
                self.reentries[depth] = Some(Reentry {
                    depth,
                    frozen: pending.frozen,
                    age: pending.age,
                    observed: pending.fresh.single_mean(),
                    stderr: pending.fresh.stderr(),
                    samples: pending.fresh.count,
                });
            }
        }
    }

    fn shadow_misses(&mut self, depth: usize) -> Option<f64> {
        let shadow = &mut self.depths[depth];
        let epoch = shadow.time.lock().bank_epoch();
        let (previous, current) = shadow.missed.read(epoch)?;
        let tokens = previous.tokens + current.tokens;
        assert!(tokens > 0.0, "non-empty missed window has {tokens} tokens");
        let rate = (previous.misses + current.misses) as f64 / tokens;
        assert!(rate.is_finite(), "non-finite shadow misses: {rate}");
        Some(rate)
    }

    fn resize(&mut self, group: usize, records: usize) {
        self.history.resize(group, records);
        for shadow in &mut self.depths {
            shadow.pools.fit(group, records);
            shadow.time.lock().resized(group, Tier::COMPUTE, records as u64);
        }
        // Additive deltas `Δ = U_real − U_shadow` measured at the previous slot
        // count are invalid after a resize; clearing them returns unexecuted
        // depths to `Needed::Shadow(U_shadow(k))` priced at the new bank size.
        self.deltas = std::array::from_fn(|_| RegimeAged::new(RegimeEpoch::ZERO, 0));
        self.verify_delta = RegimeWindow::new(RegimeEpoch::ZERO, 0);
        self.pending_reentry = [None; DEPTHS];
        self.last_executed = None;
    }

    /// Log `logged` as the ranking logged it, its step at `clock`: a read of a
    /// new step completes the one before, and every ready step at the front of
    /// the buffer is stepped.
    fn log(&mut self, logged: Logged, clock: Tokens) {
        let sequence = logged.read.sequence;
        if self.log.back().is_none_or(|step| step.sequence != sequence) {
            assert!(
                self.log.back().is_none_or(|step| step.sequence < sequence),
                "the ranking logs steps in order: {sequence} after {:?}",
                self.log.back().map(|step| step.sequence),
            );
            self.log.push_back(Behind { sequence, clock, reads: Vec::new() });
            while self.ready() {
                self.step();
            }
        }
        self.log.back_mut().expect("the step logging").reads.push(logged);
    }

    /// Whether the front step of `self.log` is complete and has [`BEHIND`]
    /// complete decode steps ahead of it (or is a prompt-only step, which needs
    /// none). The back entry is still logging.
    fn ready(&self) -> bool {
        if self.log.len() < 2 {
            return false;
        }
        let front = self.log.front().expect("at least two steps");
        if !front.is_decode() {
            return true;
        }
        let complete_decode_ahead = self
            .log
            .iter()
            .skip(1)
            .take(self.log.len() - 2)
            .filter(|step| step.is_decode())
            .count();
        complete_decode_ahead >= BEHIND
    }

    /// Step the front of `self.log` across all depths on true routes.
    fn step(&mut self) {
        let step = self.log.pop_front().expect("a ready step");
        if !step.is_decode() {
            self.step_prompt(&step);
            return;
        }

        // Collect (first_token, span, Option<log_index>) for this decode step
        // and the decode steps ahead of it in `self.log`.
        let decode_indices: Vec<usize> = self
            .log
            .iter()
            .enumerate()
            .filter(|(_, behind)| behind.is_decode())
            .map(|(index, _)| index)
            .collect();
        let next_clock = self.log[decode_indices[0]].clock;
        let emitted = next_clock.since(step.clock);
        assert!(
            emitted.get() >= 1.0,
            "decode step {} emitted {emitted:?}, below one token",
            step.sequence,
        );

        let mut clocks: Vec<(Tokens, Option<usize>)> = Vec::with_capacity(1 + decode_indices.len());
        clocks.push((step.clock, None));
        for idx in decode_indices {
            clocks.push((self.log[idx].clock, Some(idx)));
        }
        let spans: Vec<(u64, u64, Option<usize>)> = (0..clocks.len())
            .map(|i| {
                let first = clocks[i].0.get().floor() as u64;
                let span = clocks
                    .get(i + 1)
                    .map_or(1, |&(next, _)| (next.get().floor() as u64).saturating_sub(first).max(1));
                (first, span, clocks[i].1)
            })
            .collect();
        let at = spans[0].0;

        // Register the decode step on every depth shadow's clock & turnovers,
        // and snapshot each depth's (τ, ā) per group once for the step (§3.2).
        let group_count = self.depths[0].pools.records.len();
        let mut step_params: Vec<Vec<(Horizon, f64)>> = Vec::with_capacity(DEPTHS);
        for shadow in &mut self.depths {
            let mut times = shadow.time.lock();
            times.register(step.sequence, Advance::Known(emitted));
            let params = (0..group_count)
                .map(|group| {
                    if shadow.pools.records[group] == 0 {
                        (Horizon::Unbounded, 1.0)
                    } else {
                        let horizon = times
                            .horizon(group, Tier::COMPUTE, Reader::Move)
                            .expect("decode step registered");
                        let a_bar = times
                            .tokens_per_step(group)
                            .expect("decode step registered");
                        (horizon, a_bar)
                    }
                })
                .collect();
            step_params.push(params);
        }

        let mut step_misses = [0u64; DEPTHS];
        for logged in &step.reads {
            let layer = logged.read.layer;
            let group = self.groups[layer];
            let cal_epoch = self.depths[0].time.lock().group_epoch(group);
            if logged.evidence == Evidence::Decode {
                self.history.read_decode(layer, logged.read.sequence, &logged.read.experts);
            }
            self.layers[layer].read_shadow(
                logged.read.sequence,
                &logged.read.experts,
                logged.rows.as_deref(),
                logged.evidence,
                cal_epoch,
                &self.history,
                layer,
            );

            // True routes for tokens `at + 1 .. at + DEPTHS` at this layer.
            let truths: Vec<TrueRoute<'_>> = (1..DEPTHS)
                .map_while(|ahead| {
                    let token = at + ahead as u64;
                    let &(first, _, log_idx) = spans
                        .iter()
                        .find(|&&(first, span, _)| token >= first && token < first + span)?;
                    let row_depth = u8::try_from(token - first).expect("row depth fits u8");
                    let target = match log_idx {
                        None => logged,
                        Some(idx) => self.log[idx].reads.iter().find(|r| r.read.layer == layer)?,
                    };
                    target.true_route(row_depth)
                })
                .collect();

            let prepared = self.layers[layer].prepare_shadow(&mut self.scratch, &truths, cal_epoch, &self.history, layer);
            for (depth, shadow) in self.depths.iter_mut().enumerate() {
                let (horizon, a_bar) = step_params[depth][group];
                let lacked = if shadow.pools.records[group] == 0 {
                    logged.read.experts.len() as u64
                } else {
                    shadow.pools.read(layer, group, &logged.read.experts, |expert| {
                        prepared.rank(expert, depth, horizon, a_bar)
                    })
                };
                if logged.evidence == Evidence::Decode {
                    if shadow.pools.records[group] > 0 && lacked > 0 {
                        let mut times = shadow.time.lock();
                        for _ in 0..lacked {
                            times.moved(group, Tier::COMPUTE);
                        }
                    }
                    step_misses[depth] += lacked;
                }
            }
        }

        for (depth, shadow) in self.depths.iter_mut().enumerate() {
            let epoch = shadow.time.lock().bank_epoch();
            let tally = shadow.missed.count(epoch);
            tally.misses += step_misses[depth];
            tally.tokens += emitted.get();
        }
    }

    /// Feed a prompt-only step's reads to calibration and warm pools without
    /// advancing decode clocks, turnovers, or miss windows (§3.2, §4).
    fn step_prompt(&mut self, step: &Behind) {
        for logged in &step.reads {
            let layer = logged.read.layer;
            let group = self.groups[layer];
            let cal_epoch = self.depths[0].time.lock().group_epoch(group);
            self.layers[layer].read_shadow(
                logged.read.sequence,
                &logged.read.experts,
                logged.rows.as_deref(),
                logged.evidence,
                cal_epoch,
                &self.history,
                layer,
            );
            let prepared = self.layers[layer].prepare_shadow(&mut self.scratch, &[], cal_epoch, &self.history, layer);
            for (depth, shadow) in self.depths.iter_mut().enumerate() {
                if shadow.pools.records[group] == 0 {
                    continue;
                }
                let (horizon, a_bar) = {
                    let times = shadow.time.lock();
                    match times.horizon(group, Tier::COMPUTE, Reader::Move) {
                        Some(h) => (h, times.tokens_per_step(group).expect("ā exists wherever τ does")),
                        None => (Horizon::Unbounded, 1.0),
                    }
                };
                let _ = shadow.pools.read(layer, group, &logged.read.experts, |expert| {
                    prepared.rank(expert, depth, horizon, a_bar)
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(experts: usize, chosen: &[usize]) -> Vec<f32> {
        let mut scores = vec![0.0f32; experts];
        for &expert in chosen {
            scores[expert] = 1.0;
        }
        scores
    }

    fn feed_cycle(
        shadows: &mut Shadows,
        steps: std::ops::RangeInclusive<Step>,
        period: usize,
        experts: usize,
    ) {
        let fed = steps
            .map(|step| {
                let expert = ((step - 1) as usize) % period;
                ShadowFeed::Read {
                    read: RoutedRead { sequence: step, layer: 0, experts: vec![expert], choices: 1 },
                    rows: Some(Arc::new(Rows::of(1, &[0], &keys(experts, &[expert])))),
                    evidence: Evidence::Decode,
                    clock: Tokens::new((step - 1) as f64),
                }
            })
            .collect();
        shadows.read_log(fed);
    }

    /// Design §6: shadows run on true routes at every depth whatever runs.
    /// Δ_j is 0 before j runs, is measured while it runs, then holds with its
    /// age while it does not, and U(j) = U_shadow(j) + Δ_j follows the shadow.
    /// Nothing returns to unmeasured. Also checks A9 re-entry and constant
    /// memory charge.
    #[test]
    fn a_depth_not_running_keeps_its_gap_with_its_age() {
        let mut shadows = Shadows::new(&[8], &[0], &[2]);
        let initial_bytes = shadows.bytes();
        let r0 = RegimeEpoch::ZERO;
        assert_eq!(shadows.gaps(r0, 0, Tokens::ZERO), [[None; DEPTHS]; 2], "no decode step read yet");

        // Feed a 3-expert cyclic trace on depth 0 alone; shadows run on true
        // routes at every depth 0..DEPTHS.
        feed_cycle(&mut shadows, 1..=20, 3, 8);
        let initial = shadows.gaps(r0, 0, Tokens::new(10.0));
        assert!(initial[0].iter().all(Option::is_some), "every depth measured: {initial:?}");
        for (depth, gap) in initial[0].iter().enumerate() {
            let gap = gap.expect("measured");
            assert_eq!(gap.delta, Delta::Unmeasured, "depth {depth} never ran");
            assert_eq!(gap.needed(), gap.shadow, "unmeasured Δ is 0");
        }
        let shadow_2_before = initial[0][2].expect("depth 2").shadow;

        // Front depth 2 executes for 3 steps at bank epoch 0 with real = shadow + 0.25.
        for t in [10.0, 11.0, 12.0] {
            let u_shadow = shadows.misses()[2].expect("depth 2 shadow");
            shadows.executed(false, 2, u_shadow + 0.25, r0, 0, Tokens::new(t));
        }
        // Verify depth 2 executing with a higher delta does NOT overwrite front depth 2,
        // and updates the shared live verify delta across all verify depths.
        let u_shadow = shadows.misses()[2].expect("depth 2 shadow");
        shadows.executed(true, 2, u_shadow + 5.0, r0, 0, Tokens::new(12.0));
        let running = shadows.gaps(r0, 0, Tokens::new(12.0));
        let gap_2 = running[0][2].expect("front depth 2");
        match gap_2.delta {
            Delta::Measured { regime, delta, age } => {
                assert_eq!(regime, r0);
                assert!((delta - 0.25).abs() < 1e-9, "measured front delta {delta}");
                assert_eq!(age, Tokens::ZERO, "age is 0 while depth 2 runs");
            }
            Delta::Unmeasured => panic!("front depth 2 was measured"),
        }
        assert!((gap_2.needed() - (gap_2.shadow + 0.25)).abs() < 1e-9);
        let verify_2 = running[1][2].expect("verify depth 2");
        assert!((verify_2.needed() - (verify_2.shadow + 5.0)).abs() < 1e-9);
        assert_eq!(running[0][3].expect("depth 3").delta, Delta::Unmeasured, "front depth 3 still never ran");

        // Depth 2 stops running; depth 0 runs while the bank advances several
        // epochs and the workload shifts to a 6-expert cycle (higher U_shadow).
        feed_cycle(&mut shadows, 21..=60, 6, 8);
        for epoch in 1..=5 {
            let now = Tokens::new(12.0 + (epoch as f64) * 7.6);
            let u0 = shadows.misses()[0].expect("depth 0 shadow");
            shadows.executed(false, 0, u0, r0, epoch, now);
        }

        let after = shadows.gaps(r0, 5, Tokens::new(50.0));
        let frozen_2 = after[0][2].expect("depth 2");
        match frozen_2.delta {
            Delta::Measured { regime, delta, age } => {
                assert_eq!(regime, r0);
                assert!((delta - 0.25).abs() < 1e-9, "frozen delta held: {delta}");
                assert_eq!(age, Tokens::new(38.0), "age is 50 - 12 tokens");
            }
            Delta::Unmeasured => panic!("depth 2 must hold within the same regime"),
        }
        assert!(
            (frozen_2.shadow - shadow_2_before).abs() > 1e-6,
            "live shadow tracked the workload shift: {} vs {}",
            frozen_2.shadow,
            shadow_2_before,
        );
        assert!(
            (frozen_2.needed() - (frozen_2.shadow + 0.25)).abs() < 1e-9,
            "U(2) = U_shadow(2) + frozen Δ_2",
        );

        // Re-entering front depth 2 records an A9 comparison once its first window closes.
        let u2_new = shadows.misses()[2].expect("depth 2 shadow");
        shadows.executed(false, 2, u2_new + 0.40, r0, 5, Tokens::new(51.0));
        shadows.executed(false, 2, u2_new + 0.40, r0, 6, Tokens::new(52.0));
        let reentries = shadows.take_reentries();
        assert_eq!(reentries.len(), 1, "one A9 re-entry recorded: {reentries:?}");
        assert_eq!(reentries[0].depth, 2);
        assert!((reentries[0].frozen - 0.25).abs() < 1e-9);
        assert_eq!(reentries[0].age, Tokens::new(39.0));
        assert!((reentries[0].observed - 0.40).abs() < 1e-9);
        assert_eq!(reentries[0].samples, 1);

        // Memory charge is constant in run length.
        assert_eq!(shadows.bytes(), initial_bytes);
    }

    #[test]
    fn gap_needed_clamps_negative_cold_start_delta_to_zero() {
        let gap = Gap {
            shadow: 21.5,
            delta: Delta::Measured {
                regime: RegimeEpoch::ZERO,
                delta: -117.0,
                age: Tokens::new(10.0),
            },
        };
        assert_eq!(gap.needed(), 0.0);
    }

    /// Resizing the bank invalidates additive `Δ = U_real − U_shadow` measured
    /// at the previous slot count, so `Shadows::resize` clears `deltas` and
    /// `pending_reentry`, returning unexecuted depths to `Delta::Unmeasured`
    /// (`Needed::Shadow(U_shadow(k))` at the new bank size).
    #[test]
    fn resize_clears_stale_deltas_and_pending_reentry() {
        let mut shadows = Shadows::new(&[8], &[0], &[2]);
        let r0 = RegimeEpoch::ZERO;
        feed_cycle(&mut shadows, 1..=20, 4, 8);
        let u1 = shadows.misses()[1].expect("depth 1 shadow");
        let u0 = shadows.misses()[0].expect("depth 0 shadow");
        shadows.executed(false, 1, u1 + 4.0, r0, 0, Tokens::new(10.0));
        shadows.executed(false, 0, u0, r0, 0, Tokens::new(11.0));
        // Re-enter front depth 1 to arm `pending_reentry`.
        shadows.executed(false, 1, u1 + 4.0, r0, 0, Tokens::new(12.0));
        shadows.executed(true, 1, u1 + 4.0, r0, 0, Tokens::new(12.0));
        assert!(matches!(shadows.gaps(r0, 0, Tokens::new(12.0))[0][1].unwrap().delta, Delta::Measured { .. }));
        assert!(matches!(shadows.gaps(r0, 0, Tokens::new(12.0))[1][1].unwrap().delta, Delta::Measured { .. }));
        assert!(shadows.pending_reentry[1].is_some());

        // Resize group 0 from 2 to 4 slots.
        shadows.read_log(vec![ShadowFeed::Resize { group: 0, records: 4 }]);
        assert!(shadows.pending_reentry.iter().all(Option::is_none));
        assert!(shadows.last_executed.is_none());

        // After new steps at the larger bank size, previously measured deltas
        // are unmeasured (`Delta::Unmeasured`) instead of carrying the old +4.0.
        feed_cycle(&mut shadows, 21..=35, 4, 8);
        let gaps = shadows.gaps(r0, 1, Tokens::new(30.0));
        assert_eq!(gaps[1][1].expect("verify depth 1").delta, Delta::Unmeasured);
        assert_eq!(gaps[0][0].expect("front depth 0").delta, Delta::Unmeasured);
    }

    /// Verify passes share one live regime-scoped per-row delta across all
    /// depths so a single lucky narrow step (e.g. `n=1` at depth 1) can never
    /// freeze out deeper verify depths, and advancing `RegimeEpoch` makes
    /// prior-regime deltas inapplicable by construction without any reset call.
    #[test]
    fn verify_delta_is_shared_across_depths_and_invalidated_by_regime_epoch() {
        let mut shadows = Shadows::new(&[8], &[0], &[2]);
        let r0 = RegimeEpoch::ZERO;
        let r1 = r0.next();
        feed_cycle(&mut shadows, 1..=20, 4, 8);
        let u1 = shadows.misses()[1].expect("depth 1 shadow");
        let u4 = shadows.misses()[4].expect("depth 4 shadow");

        // A single lucky verify step at depth 1 records a negative delta.
        shadows.executed(true, 1, (u1 - 0.5).max(0.0), r0, 0, Tokens::new(10.0));
        let gaps_r0 = shadows.gaps(r0, 0, Tokens::new(10.0));
        // Every verify depth (1..DEPTHS) shares the exact same live delta —
        // depth 1 gets NO artificial discount over depth 4.
        assert_eq!(gaps_r0[1][1].unwrap().delta, gaps_r0[1][4].unwrap().delta);
        assert!(matches!(gaps_r0[1][4].unwrap().delta, Delta::Measured { regime, .. } if regime == r0));

        // Front depth 2 also recorded a delta in r0.
        shadows.executed(false, 2, u4 + 1.0, r0, 0, Tokens::new(10.0));
        assert!(matches!(shadows.gaps(r0, 0, Tokens::new(10.0))[0][2].unwrap().delta, Delta::Measured { .. }));

        // Advancing RegimeEpoch (e.g. tier_complete / policy / link_staging)
        // makes ALL r0 deltas Delta::Unmeasured by construction — no reset needed.
        let gaps_r1 = shadows.gaps(r1, 0, Tokens::new(10.0));
        assert_eq!(gaps_r1[1][1].unwrap().delta, Delta::Unmeasured);
        assert_eq!(gaps_r1[1][4].unwrap().delta, Delta::Unmeasured);
        assert_eq!(gaps_r1[0][2].unwrap().delta, Delta::Unmeasured);
    }

    /// Seeding `Shadows` with the live bank's warm `Pools` (`new_with_pools`)
    /// eliminates the empty-pool cold-start miss cliff that previously inflated
    /// `U_shadow` far above `U_real`.
    #[test]
    fn seeded_pools_eliminate_cold_start_shadow_miss_cliff() {
        let mut warm = Pools::new(&[8], &[4]);
        // Warm the live pool with the 4-expert working set {0, 1, 2, 3}.
        warm.read(0, 0, &[0, 1, 2, 3], |e| super::super::Rank::new(super::super::Expected::new(1.0, 0.0), Some(e as u64)));

        let mut cold_shadows = Shadows::new(&[8], &[0], &[4]);
        let mut warm_shadows = Shadows::new_with_pools(&[8], &[0], &warm);

        // Feed the exact same 4-expert working set to both shadows.
        feed_cycle(&mut cold_shadows, 1..=12, 4, 8);
        feed_cycle(&mut warm_shadows, 1..=12, 4, 8);

        let cold_u = cold_shadows.misses()[0].expect("cold shadow measured");
        let warm_u = warm_shadows.misses()[0].expect("warm shadow measured");
        assert!(
            warm_u < cold_u,
            "warm-seeded shadow ({warm_u:.3}) must avoid cold-start miss inflation ({cold_u:.3})"
        );
        assert!(warm_u.abs() < 1e-9, "all 4 experts were already warm in seeded pool: {warm_u}");
    }
}
