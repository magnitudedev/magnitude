//! Measured per-layer exponential-decay half-life selection (`HALF_LIVES` ×
//! per-layer Mattson stacks): the single owner of what decode history predicts
//! (`p_∞` and the history tie-break order).
//!
//! Each half-life `h` in [`HALF_LIVES`] tracks every expert's decayed read count
//! via its equivalent step [`Score`] (`h · log2 Σ 2^(t/h)`, or last read at
//! `h = 0`), together with the layer's equivalent step across all decode steps
//! `steps_score`. The normalized decayed rate
//! `r_e(h) = 2^((score_e − steps_score) / h)` lies strictly in `[0, 1]`, sums
//! to the active choices `k` across all experts, and is strictly monotone in
//! `Score` with zero rescaling and zero tuned constants.
//!
//! Each half-life also maintains a per-layer Mattson priority stack counting
//! exponentially decayed decode misses at each layer's nominal share of its
//! group's live capacity. Each layer independently selects the half-life with
//! the fewest decayed misses (breaking ties toward longer half-lives so
//! frequency skew is preserved during cold start and warmup), letting skewed
//! layers expose high `p_∞` while flat or bursty layers select short decay.

use super::super::watcher::Step;
use std::mem::size_of;

/// The half-lives, in decode steps, the history selector measures reads by.
/// `f64::INFINITY` tracks exact un-decayed frequency (`count / steps`), preserving
/// stationary Zipf tails over long sessions without `exp2`.
pub(super) const HALF_LIVES: [f64; 13] = [0., 1., 2., 4., 8., 16., 32., 64., 128., 256., 512., 1024., f64::INFINITY];

/// Half-life, in decode steps, of the exponentially decayed miss counts used
/// to select the live candidate. Decaying miss counts prevents earlier
/// sessions from locking in a stale half-life across workload shifts.
const MISS_HALF_LIFE_STEPS: f64 = 128.0;

/// An expert's reuse as its equivalent step: greater is more reused. An expert
/// never read scores below every other.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Score(f64);

impl Score {
    pub(super) const NEVER: Self = Self(f64::NEG_INFINITY);

    /// The score of one read at step `step`, at any half-life.
    #[inline]
    pub(super) fn at(step: Step) -> Self {
        Self(step as f64)
    }

    /// The score of the reads of `self` and of `other` together at finite
    /// half-life `half_life`: `h · log2(2^(a/h) + 2^(b/h))`, their later at `h = 0`.
    #[inline]
    pub(super) fn sum(self, other: Self, half_life: f64) -> Self {
        debug_assert!(half_life.is_finite(), "Score::sum requires a finite half-life");
        let (low, high) = if self <= other { (self.0, other.0) } else { (other.0, self.0) };
        if half_life == 0. || low == f64::NEG_INFINITY {
            return Self(high);
        }
        let diff = (low - high) / half_life;
        if diff < -60. {
            return Self(high);
        }
        Self(high + half_life * (1. + diff.exp2()).log2())
    }
}

impl Eq for Score {}

impl PartialOrd for Score {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Score {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// One expert's logged decode reads at a single half-life.
#[derive(Clone, Copy, Debug)]
struct Logged {
    score: Score,
    last: Option<Step>,
    order: u64,
}

impl Logged {
    const NONE: Self = Self { score: Score::NEVER, last: None, order: 0 };
}

/// Total history order over read experts at one half-life, least reused first:
/// equivalent step [`Score`], then last decode step, then logging order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct HistoryRank {
    score: Score,
    last: Option<Step>,
    order: u64,
}

/// Sentinel index in `Measure::pos` when an expert is not currently in the
/// layer's live min-heap cache.
const NOT_IN_HEAP: u32 = u32::MAX;

#[cfg(test)]
thread_local! {
    static HISTORY_HEAP_STEPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Decode history measured at one half-life: per-layer scores, equivalent step
/// across all decode steps, and per-layer indexed min-heaps of the live
/// `slots`-sized cache with decayed misses (`O(K log slots)` per step,
/// independent of total expert count `E`, zero per-step allocations).
struct Measure {
    half_life: f64,
    miss_decay: f64,
    layers: Vec<Vec<Logged>>,
    steps_score: Vec<Score>,
    logged: u64,
    /// Per-layer min-heap of resident expert IDs ordered by ascending
    /// [`HistoryRank`] (root = coldest victim), pre-allocated to `n` experts.
    heaps: Vec<Vec<usize>>,
    /// Per-layer inverse index `pos[layer][expert] -> heap index` (or
    /// [`NOT_IN_HEAP`] when absent), giving `O(1)` hit/miss and key-update
    /// lookup.
    pos: Vec<Vec<u32>>,
    misses: Vec<f64>,
    /// Pre-allocated scratch buffers (fixed capacity `max_experts`, zero per-call allocs).
    scratch_entering: Vec<usize>,
    scratch_reinsert: Vec<usize>,
}

impl Measure {
    fn new(half_life: f64, experts: &[usize]) -> Self {
        let max_experts = experts.iter().copied().max().unwrap_or(0);
        let miss_decay = 0.5_f64.powf(1.0 / MISS_HALF_LIFE_STEPS);
        Self {
            half_life,
            miss_decay,
            layers: experts.iter().map(|&n| vec![Logged::NONE; n]).collect(),
            steps_score: vec![Score::NEVER; experts.len()],
            logged: 0,
            heaps: experts.iter().map(|&n| Vec::with_capacity(n)).collect(),
            pos: experts.iter().map(|&n| vec![NOT_IN_HEAP; n]).collect(),
            misses: vec![0.0; experts.len()],
            scratch_entering: Vec::with_capacity(max_experts),
            scratch_reinsert: Vec::with_capacity(max_experts),
        }
    }

    #[inline]
    fn rank(&self, layer: usize, expert: usize) -> HistoryRank {
        let Logged { score, last, order } = self.layers[layer][expert];
        HistoryRank { score, last, order }
    }

    #[inline]
    fn sift_up(&mut self, layer: usize, mut idx: usize) {
        let expert = self.heaps[layer][idx];
        let rank = self.rank(layer, expert);
        while idx > 0 {
            #[cfg(test)]
            HISTORY_HEAP_STEPS.with(|c| c.set(c.get() + 1));
            let parent = (idx - 1) / 2;
            let p_expert = self.heaps[layer][parent];
            if rank < self.rank(layer, p_expert) {
                self.heaps[layer][idx] = p_expert;
                self.pos[layer][p_expert] = idx as u32;
                idx = parent;
            } else {
                break;
            }
        }
        self.heaps[layer][idx] = expert;
        self.pos[layer][expert] = idx as u32;
    }

    #[inline]
    fn sift_down(&mut self, layer: usize, mut idx: usize) {
        let len = self.heaps[layer].len();
        let expert = self.heaps[layer][idx];
        let rank = self.rank(layer, expert);
        loop {
            let left = 2 * idx + 1;
            if left >= len {
                break;
            }
            #[cfg(test)]
            HISTORY_HEAP_STEPS.with(|c| c.set(c.get() + 1));
            let mut smallest = left;
            let mut smallest_expert = self.heaps[layer][left];
            let mut smallest_rank = self.rank(layer, smallest_expert);
            let right = left + 1;
            if right < len {
                let r_expert = self.heaps[layer][right];
                let r_rank = self.rank(layer, r_expert);
                if r_rank < smallest_rank {
                    smallest = right;
                    smallest_expert = r_expert;
                    smallest_rank = r_rank;
                }
            }
            if smallest_rank < rank {
                self.heaps[layer][idx] = smallest_expert;
                self.pos[layer][smallest_expert] = idx as u32;
                idx = smallest;
            } else {
                break;
            }
        }
        self.heaps[layer][idx] = expert;
        self.pos[layer][expert] = idx as u32;
    }

    #[inline]
    fn heap_push(&mut self, layer: usize, expert: usize) {
        let idx = self.heaps[layer].len();
        self.heaps[layer].push(expert);
        self.pos[layer][expert] = idx as u32;
        self.sift_up(layer, idx);
    }

    #[inline]
    fn heap_pop_min(&mut self, layer: usize) {
        let popped = self.heaps[layer].swap_remove(0);
        self.pos[layer][popped] = NOT_IN_HEAP;
        if !self.heaps[layer].is_empty() {
            let moved = self.heaps[layer][0];
            self.pos[layer][moved] = 0;
            self.sift_down(layer, 0);
        }
    }

    #[inline]
    fn heap_remove(&mut self, layer: usize, expert: usize) {
        let idx = self.pos[layer][expert] as usize;
        let removed = self.heaps[layer].swap_remove(idx);
        debug_assert_eq!(removed, expert);
        self.pos[layer][expert] = NOT_IN_HEAP;
        if idx < self.heaps[layer].len() {
            let moved = self.heaps[layer][idx];
            self.pos[layer][moved] = idx as u32;
            if idx > 0 && self.rank(layer, moved) < self.rank(layer, self.heaps[layer][(idx - 1) / 2]) {
                self.sift_up(layer, idx);
            } else {
                self.sift_down(layer, idx);
            }
        }
    }

    fn fill_entering(&mut self, layer: usize, experts: &[usize]) {
        self.scratch_entering.clear();
        self.scratch_entering.extend_from_slice(experts);
        let layer_logged = &self.layers[layer];
        self.scratch_entering.sort_by_key(|&expert| {
            let Logged { score, last, order } = layer_logged[expert];
            HistoryRank { score, last, order }
        });
    }

    fn read_decode(&mut self, layer: usize, token_step: u64, step: Step, experts: &[usize], slots: usize) {
        // Trim heap if live slot share shrank since the previous step.
        while self.heaps[layer].len() > slots {
            self.heap_pop_min(layer);
        }

        let mut misses = 0u64;
        for &expert in experts {
            if self.pos[layer][expert] == NOT_IN_HEAP {
                misses += 1;
            }
        }
        self.misses[layer] = self.misses[layer] * self.miss_decay + misses as f64;

        // Remove already-resident experts before mutating their ranks so every
        // heap operation sees a valid min-heap invariant.
        self.scratch_reinsert.clear();
        for &expert in experts {
            if self.pos[layer][expert] != NOT_IN_HEAP {
                self.heap_remove(layer, expert);
                self.scratch_reinsert.push(expert);
            }
        }

        self.fill_entering(layer, experts);
        for i in 0..self.scratch_entering.len() {
            let expert = self.scratch_entering[i];
            self.logged += 1;
            self.layers[layer][expert].order = self.logged;
        }
        if self.half_life.is_infinite() {
            self.steps_score[layer] = if self.steps_score[layer] == Score::NEVER {
                Score(1.0)
            } else {
                Score(self.steps_score[layer].0 + 1.0)
            };
            for &expert in experts {
                let logged = &mut self.layers[layer][expert];
                logged.score = if logged.score == Score::NEVER {
                    Score(1.0)
                } else {
                    Score(logged.score.0 + 1.0)
                };
                logged.last = logged.last.max(Some(step));
            }
        } else {
            let now = Score::at(token_step);
            self.steps_score[layer] = self.steps_score[layer].sum(now, self.half_life);
            for &expert in experts {
                let logged = &mut self.layers[layer][expert];
                logged.score = logged.score.sum(now, self.half_life);
                logged.last = logged.last.max(Some(step));
            }
        }

        if slots > 0 {
            for i in 0..self.scratch_reinsert.len() {
                let expert = self.scratch_reinsert[i];
                self.heap_push(layer, expert);
            }
            self.fill_entering(layer, experts);
            for i in 0..self.scratch_entering.len() {
                let expert = self.scratch_entering[i];
                if self.pos[layer][expert] == NOT_IN_HEAP {
                    if self.heaps[layer].len() >= slots {
                        self.heap_pop_min(layer);
                    }
                    self.heap_push(layer, expert);
                }
            }
        }
    }

    /// Normalized decayed decode rate for `(layer, expert)` in `[0, 1]`, or
    /// `None` before the first decode step at `layer`.
    #[inline]
    fn rate(&self, layer: usize, expert: usize, harmonic: f64) -> Option<f64> {
        let steps = self.steps_score[layer];
        if steps == Score::NEVER {
            return None;
        }
        let score = self.layers[layer][expert].score;
        if score == Score::NEVER {
            return Some(0.0);
        }
        let rate = if self.half_life == 0. {
            // Pure recency: harmonic-normalized inverse age `(1 / (1 + age)) / H(steps)`,
            // the step-space analog of `S_eff(h)` that never underflows against route noise.
            let age = (steps.0 - score.0).max(0.0);
            1.0 / ((1.0 + age) * harmonic)
        } else if self.half_life.is_infinite() {
            // Unbounded stationary frequency: exact `count / steps`, no `exp2`.
            score.0 / steps.0
        } else {
            ((score.0 - steps.0) / self.half_life).exp2()
        };
        Some(rate.clamp(0.0, 1.0))
    }

    /// Fixed memory charge (constant in run length).
    fn bytes(&self) -> u64 {
        let layers: usize = self.layers.iter().map(|l| l.capacity() * size_of::<Logged>()).sum();
        let steps = self.steps_score.capacity() * size_of::<Score>();
        let heaps: usize = self.heaps.iter().map(|h| h.capacity() * size_of::<usize>()).sum();
        let pos: usize = self.pos.iter().map(|p| p.capacity() * size_of::<u32>()).sum();
        let misses = self.misses.capacity() * size_of::<f64>();
        let scratch = (self.scratch_entering.capacity() + self.scratch_reinsert.capacity()) * size_of::<usize>();
        (size_of::<Self>() + layers + steps + heaps + pos + misses + scratch) as u64
    }
}

/// Per-layer live half-life selector across [`HALF_LIVES`]: the single owner of
/// what decode history predicts.
pub(super) struct History {
    measures: Vec<Measure>,
    /// Per-layer decode token-step count and harmonic normalizer `H(n) = Σ_{d=1..n} 1/d`
    /// for `h = 0` recency rates.
    decode_steps: Vec<u64>,
    harmonic: Vec<f64>,
    /// Per-layer winning half-life index in `0..HALF_LIVES.len()`.
    live: Vec<usize>,
}

impl History {
    pub(super) fn new(experts: &[usize]) -> Self {
        let measures = HALF_LIVES.iter().map(|&half_life| Measure::new(half_life, experts)).collect();
        Self {
            measures,
            decode_steps: vec![0; experts.len()],
            harmonic: vec![0.0; experts.len()],
            live: vec![HALF_LIVES.len() - 1; experts.len()],
        }
    }

    #[inline]
    fn live_measure(&self, layer: usize) -> &Measure {
        &self.measures[self.live[layer]]
    }

    /// Log one decode token's routed `experts` at `layer` and `step` against
    /// `layer`'s live pooled slot share `slots`: updates all half-life stacks
    /// for `layer` and selects `layer`'s half-life with fewest decayed misses
    /// (breaking ties toward longer half-lives so frequency skew is preserved
    /// during cold start and warmup).
    pub(super) fn read_decode(&mut self, layer: usize, step: Step, experts: &[usize], slots: usize) {
        self.decode_steps[layer] += 1;
        let token_step = self.decode_steps[layer];
        self.harmonic[layer] += 1.0 / token_step as f64;
        for measure in &mut self.measures {
            measure.read_decode(layer, token_step, step, experts, slots);
        }
        self.live[layer] = (0..HALF_LIVES.len())
            .min_by(|&a, &b| {
                self.measures[a].misses[layer]
                    .total_cmp(&self.measures[b].misses[layer])
                    .then_with(|| b.cmp(&a))
            })
            .expect("HALF_LIVES is non-empty");
    }

    /// Normalized decayed decode rate `p_∞ ∈ [0, 1]` for `(layer, expert)`
    /// under `layer`'s live half-life, or `None` before the first decode step
    /// at `layer`.
    #[inline]
    pub(super) fn rate(&self, layer: usize, expert: usize) -> Option<f64> {
        self.live_measure(layer).rate(layer, expert, self.harmonic[layer])
    }

    /// History tie-break rank for `(layer, expert)` under `layer`'s live
    /// half-life.
    #[inline]
    pub(super) fn rank(&self, layer: usize, expert: usize) -> HistoryRank {
        self.live_measure(layer).rank(layer, expert)
    }

    /// `layer`'s live half-life, in steps.
    #[cfg(test)]
    pub(super) fn half_life(&self, layer: usize) -> f64 {
        self.live_measure(layer).half_life
    }

    /// Number of accepted decode token steps logged at `layer`.
    #[cfg(test)]
    pub(super) fn decode_steps(&self, layer: usize) -> u64 {
        self.decode_steps[layer]
    }

    /// Fixed memory charge in bytes (constant in run length).
    pub(super) fn bytes(&self) -> u64 {
        let measures: u64 = self.measures.iter().map(Measure::bytes).sum();
        let meta = size_of::<Self>()
            + self.live.capacity() * size_of::<usize>()
            + self.decode_steps.capacity() * size_of::<u64>()
            + self.harmonic.capacity() * size_of::<f64>();
        measures + meta as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_outranks_one_later_read_over_a_half_life() {
        let mut measure = Measure::new(4., &[2]);
        for step in [1, 2, 3, 4] {
            measure.read_decode(0, step, step, &[0], 2);
        }
        measure.read_decode(0, 5, 5, &[1], 2);
        assert!(measure.rank(0, 0) > measure.rank(0, 1));
        assert!(measure.rate(0, 0, 1.0).unwrap() > measure.rate(0, 1, 1.0).unwrap());
        assert!((Score::at(0).sum(Score::at(0), 4.).0 - 4.).abs() < 1e-12);
    }

    #[test]
    fn decayed_rate_sums_to_active_and_is_monotone_in_score() {
        let mut history = History::new(&[16]);
        assert_eq!(history.rate(0, 0), None, "cold start before any decode step");
        for step in 1..=40u64 {
            let experts = [0, ((step as usize) % 15) + 1];
            history.read_decode(0, step, &experts, 4);
        }
        // Check on every non-zero half-life (including f64::INFINITY): sum of rates equals k = 2.0 exactly.
        let h = history.harmonic[0];
        for measure in &history.measures[1..] {
            let sum: f64 = (0..16).map(|e| measure.rate(0, e, h).unwrap()).sum();
            assert!((sum - 2.0).abs() < 1e-9, "half-life {} rate sum {sum} != 2.0", measure.half_life);
            for a in 0..16 {
                for b in 0..16 {
                    if measure.layers[0][a].score > measure.layers[0][b].score {
                        assert!(measure.rate(0, a, h).unwrap() >= measure.rate(0, b, h).unwrap());
                    }
                }
            }
        }
    }

    #[test]
    fn recency_harmonic_rate_is_strictly_monotone_and_in_unit_interval() {
        let mut history = History::new(&[16]);
        for step in 1..=10u64 {
            history.read_decode(0, step, &[step as usize], 4);
        }
        // All 10 reads are distinct -> all candidates miss identically -> ties
        // prefer the longest half-life (f64::INFINITY) so frequency skew is never erased on cold start.
        assert_eq!(history.half_life(0), f64::INFINITY);
        let expected_h: f64 = (1..=10).map(|d| 1.0 / d as f64).sum();
        assert!((history.harmonic[0] - expected_h).abs() < 1e-12);
        // Verify the h = 0 recency measure directly.
        let recency = &history.measures[0];
        assert_eq!(recency.rate(0, 0, expected_h), Some(0.0), "unread expert has rate 0");
        let mut prev = 0.0;
        for step in 1..=10usize {
            let r = recency.rate(0, step, expected_h).unwrap();
            assert!(r > 0.0 && r <= 1.0, "rate {r} for step {step} out of (0, 1]");
            assert!(r > prev, "more recent step {step} ({r}) must strictly outrank older ({prev})");
            let want = 1.0 / ((1 + (10 - step)) as f64 * expected_h);
            assert!((r - want).abs() < 1e-12, "got {r}, want {want}");
            prev = r;
        }
    }

    #[test]
    fn the_half_life_selected_is_the_one_that_missed_least() {
        let mut history = History::new(&[128]);
        for step in 1..=60u64 {
            let experts = if step % 3 == 0 { vec![0, step as usize] } else { vec![step as usize] };
            history.read_decode(0, step, &experts, 2);
        }
        assert!(history.half_life(0) > 0., "a half-life kept the hot expert");
        let recency = &history.measures[0];
        let live = history.live_measure(0);
        assert!(live.misses[0] < recency.misses[0], "{} of {}", live.misses[0], recency.misses[0]);
        assert!(history.rate(0, 0).unwrap() > history.rate(0, 60).unwrap());
    }

    #[test]
    fn history_bytes_is_constant_in_run_length() {
        let mut history = History::new(&[16]);
        let initial = history.bytes();
        for step in 1..=500u64 {
            let e0 = (step as usize) % 16;
            let e1 = (step as usize * 3 + 1) % 16;
            let mut chosen = vec![e0, e1];
            chosen.sort_unstable();
            chosen.dedup();
            history.read_decode(0, step, &chosen, 4);
        }
        assert_eq!(history.bytes(), initial, "history memory footprint must be constant");
    }

    /// Decayed miss selection adapts cleanly across sessions: a long session
    /// favoring long half-lives cannot lock in a stale half-life when a later
    /// session shifts to a pure-recency burst workload.
    #[test]
    fn decayed_misses_adapt_across_session_regime_shifts() {
        let mut history = History::new(&[128]);
        // Session 1 (5,000 steps): expert 0 is hot every 3rd step beside one-off
        // cold experts — long half-life wins by a wide margin.
        let mut step = 1u64;
        for _ in 0..5_000 {
            let cold = 1 + ((step as usize) % 127);
            if step % 3 == 0 {
                history.read_decode(0, step, &[0, cold], 2);
            } else {
                history.read_decode(0, step, &[cold], 2);
            }
            step += 1;
        }
        assert!(history.half_life(0) > 0., "session 1 selects a positive half-life");

        // Session 2 (600 steps): shifting bursts of 2 experts for 6 steps each.
        // Pure recency (h = 0) adapts immediately on each burst change, whereas
        // long half-lives cling to stale high-frequency experts from previous bursts.
        for burst in 0..100usize {
            let a = (burst * 2) % 126 + 1;
            let b = a + 1;
            for _ in 0..6 {
                history.read_decode(0, step, &[a, b], 2);
                step += 1;
            }
        }
        assert!(
            history.half_life(0) <= 4.0,
            "session 2 should adapt to a short half-life, got {}",
            history.half_life(0)
        );
    }

    /// Per-layer half-life selection lets a stationary skewed layer pick a long
    /// half-life while a bursty/shifting layer in the same group picks a short
    /// half-life, exposing high `p_∞` on the skewed layer without cross-layer interference.
    #[test]
    fn per_layer_half_life_selects_independently_for_skewed_and_bursty_layers() {
        let mut history = History::new(&[128, 128]);
        for step in 1..=300u64 {
            // Layer 0: stationary skewed — expert 0 every 3rd step beside cold experts.
            let cold = 1 + ((step as usize) % 127);
            if step % 3 == 0 {
                history.read_decode(0, step, &[0, cold], 2);
            } else {
                history.read_decode(0, step, &[cold], 2);
            }
            // Layer 1: shifting bursts of 2 experts for 6 steps each (short half-life wins).
            let burst = ((step - 1) / 6) as usize;
            let a = (burst * 2) % 126 + 1;
            let b = a + 1;
            history.read_decode(1, step, &[a, b], 2);
        }
        assert!(
            history.half_life(0) >= 16.0,
            "skewed layer 0 should select a long half-life, got {}",
            history.half_life(0)
        );
        assert!(
            history.half_life(1) <= 4.0,
            "bursty layer 1 should select a short half-life, got {}",
            history.half_life(1)
        );
    }

    /// Unbounded half-life (`f64::INFINITY`) computes exact un-decayed frequency
    /// `count / steps` without decay over 4,000+ token sessions, preserving
    /// stationary Zipf tails.
    #[test]
    fn infinite_half_life_tracks_exact_undecayed_frequency() {
        let mut history = History::new(&[64]);
        // 4,000 steps: expert 0 chosen every 2nd step (50%), expert 1 every 10th step (10%),
        // expert 2 every 40th step (2.5%), plus rotating cold experts.
        for step in 1..=4_000u64 {
            let mut chosen = Vec::new();
            if step % 2 == 0 {
                chosen.push(0);
            }
            if step % 10 == 0 {
                chosen.push(1);
            }
            if step % 40 == 0 {
                chosen.push(2);
            }
            let cold = 3 + ((step as usize) % 61);
            chosen.push(cold);
            chosen.sort_unstable();
            history.read_decode(0, step, &chosen, 4);
        }
        let inf = history.measures.last().expect("f64::INFINITY is last");
        assert!(inf.half_life.is_infinite());
        assert!((inf.rate(0, 0, 1.0).unwrap() - 0.5).abs() < 1e-12);
        assert!((inf.rate(0, 1, 1.0).unwrap() - 0.1).abs() < 1e-12);
        assert!((inf.rate(0, 2, 1.0).unwrap() - 0.025).abs() < 1e-12);
        assert_eq!(history.half_life(0), f64::INFINITY);
    }

    #[test]
    fn work_bound_history_read_decode_independent_of_total_experts_1x_vs_16x() {
        let measure = |num_experts: usize| -> usize {
            let mut history = History::new(&[num_experts]);
            let slots = 16usize;
            // Warm up the 16-slot cache.
            for step in 1..=32u64 {
                let base = ((step as usize) * 7) % num_experts;
                let mut experts = Vec::with_capacity(8);
                for k in 0..8 {
                    experts.push((base + k) % num_experts);
                }
                history.read_decode(0, step, &experts, slots);
            }
            HISTORY_HEAP_STEPS.with(|c| c.set(0));
            for step in 33..=64u64 {
                let base = ((step as usize) * 13) % num_experts;
                let mut experts = Vec::with_capacity(8);
                for k in 0..8 {
                    experts.push((base + k) % num_experts);
                }
                history.read_decode(0, step, &experts, slots);
            }
            HISTORY_HEAP_STEPS.with(|c| c.get())
        };

        let steps_1x = measure(64);
        let steps_16x = measure(1024);
        // With an indexed min-heap of size `slots = 16`, heap work per step is
        // `O(13 * K * log2(slots))` — completely independent of total experts `E`.
        assert!(
            steps_16x <= steps_1x.saturating_mul(2),
            "expected O(K log slots) independent of E: 1x(E=64)={steps_1x}, 16x(E=1024)={steps_16x}"
        );
    }
}
