//! Measured exponential-decay half-life selection (`HALF_LIVES` × per-layer
//! Mattson stacks and per-group pools): the single owner of what decode history
//! predicts (`p_∞` and the history tie-break order).
//!
//! Each half-life `h` in [`HALF_LIVES`] tracks every expert's decayed read count
//! via its equivalent step [`Score`] (`h · log2 Σ 2^(t/h)`, or last read at
//! `h = 0`), together with the layer's equivalent step across all decode steps
//! `steps_score`. The normalized decayed rate
//! `r_e(h) = 2^((score_e − steps_score) / h)` lies strictly in `[0, 1]`, sums
//! to the active choices `k` across all experts, and is strictly monotone in
//! `Score` with zero rescaling and zero tuned constants.
//!
//! Each half-life also maintains a per-layer Mattson priority stack and a
//! per-group pool, counting decode misses at the live capacities. The candidate
//! `(scope, half_life)` with the fewest decode misses so far is live, and its
//! half-life supplies `p_∞` and the history order.

use super::super::watcher::Step;
use std::collections::BTreeSet;
use std::mem::size_of;

/// The half-lives, in decode steps, the history selector measures reads by.
pub(super) const HALF_LIVES: [f64; 10] = [0., 1., 2., 4., 8., 16., 32., 64., 128., 256.];

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

    /// The score of the reads of `self` and of `other` together at half-life
    /// `half_life`: `h · log2(2^(a/h) + 2^(b/h))`, their later at `h = 0`.
    #[inline]
    pub(super) fn sum(self, other: Self, half_life: f64) -> Self {
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

type Pooled = (HistoryRank, usize, usize);

struct Pool {
    records: usize,
    held: BTreeSet<Pooled>,
}

/// Decode history measured at one half-life: per-layer scores, equivalent step
/// across all decode steps, per-layer Mattson stacks, and per-group pools.
struct Measure {
    half_life: f64,
    layers: Vec<Vec<Logged>>,
    steps_score: Vec<Score>,
    logged: u64,
    stacks: Vec<Vec<usize>>,
    misses: u64,
    groups: Vec<usize>,
    pools: Vec<Pool>,
    pooled: Vec<Vec<bool>>,
    pooled_misses: u64,
}

impl Measure {
    fn new(half_life: f64, experts: &[usize], groups: &[usize], records: &[usize]) -> Self {
        Self {
            half_life,
            layers: experts.iter().map(|&n| vec![Logged::NONE; n]).collect(),
            steps_score: vec![Score::NEVER; experts.len()],
            logged: 0,
            stacks: experts.iter().map(|&n| Vec::with_capacity(n)).collect(),
            misses: 0,
            groups: groups.to_vec(),
            pools: records.iter().map(|&cap| Pool { records: cap, held: BTreeSet::new() }).collect(),
            pooled: experts.iter().map(|&n| vec![false; n]).collect(),
            pooled_misses: 0,
        }
    }

    #[inline]
    fn rank(&self, layer: usize, expert: usize) -> HistoryRank {
        let Logged { score, last, order } = self.layers[layer][expert];
        HistoryRank { score, last, order }
    }

    #[inline]
    fn pooled_rank(&self, layer: usize, expert: usize) -> Pooled {
        (self.rank(layer, expert), layer, expert)
    }

    fn fit(&mut self, group: usize, records: usize) {
        let pool = &mut self.pools[group];
        pool.records = records;
        while pool.held.len() > records {
            let (.., layer, expert) = pool.held.pop_first().expect("a pool past its records holds one");
            self.pooled[layer][expert] = false;
        }
    }

    fn depth(&self, layer: usize, expert: usize) -> usize {
        let stack = &self.stacks[layer];
        stack.iter().position(|&above| above == expert).unwrap_or(self.layers[layer].len())
    }

    fn entering(&self, layer: usize, experts: &[usize]) -> Vec<usize> {
        let mut entering = experts.to_vec();
        entering.sort_by_key(|&expert| self.rank(layer, expert));
        entering
    }

    fn read_decode(&mut self, layer: usize, step: Step, experts: &[usize], slots: usize) {
        let group = self.groups[layer];
        let depths: Vec<usize> = experts.iter().map(|&expert| self.depth(layer, expert)).collect();
        let (held, lacked): (Vec<usize>, Vec<usize>) =
            experts.iter().partition(|&&expert| self.pooled[layer][expert]);
        self.misses += depths.iter().filter(|&&depth| depth >= slots).count() as u64;
        self.pooled_misses += lacked.len() as u64;

        for &expert in &held {
            let place = self.pooled_rank(layer, expert);
            self.pools[group].held.remove(&place);
        }
        for expert in self.entering(layer, experts) {
            self.logged += 1;
            self.layers[layer][expert].order = self.logged;
        }
        let now = Score::at(step);
        self.steps_score[layer] = self.steps_score[layer].sum(now, self.half_life);
        for &expert in experts {
            let logged = &mut self.layers[layer][expert];
            logged.score = logged.score.sum(now, self.half_life);
            logged.last = logged.last.max(Some(step));
        }
        for &expert in &held {
            let place = self.pooled_rank(layer, expert);
            self.pools[group].held.insert(place);
        }
        let entering = self.entering(layer, experts);
        for &expert in &entering {
            self.promote(layer, expert);
        }
        for expert in entering.into_iter().filter(|expert| lacked.contains(expert)) {
            let pool = &mut self.pools[group];
            if pool.held.len() >= pool.records {
                let Some((.., of, victim)) = pool.held.pop_first() else {
                    continue;
                };
                self.pooled[of][victim] = false;
            }
            if pool.held.len() < pool.records {
                let place = self.pooled_rank(layer, expert);
                self.pools[group].held.insert(place);
                self.pooled[layer][expert] = true;
            }
        }
    }

    /// Mattson's priority stack update: move `expert` to the top of `layer`'s
    /// stack, settling each passed expert below the more reused above it.
    fn promote(&mut self, layer: usize, expert: usize) {
        let at = self.stacks[layer].iter().position(|&above| above == expert);
        let to = at.unwrap_or(self.stacks[layer].len());
        if to == 0 {
            if at.is_none() {
                self.stacks[layer].push(expert);
            }
            return;
        }
        let mut carried = std::mem::replace(&mut self.stacks[layer][0], expert);
        for depth in 1..to {
            let held = self.stacks[layer][depth];
            if self.rank(layer, carried) > self.rank(layer, held) {
                self.stacks[layer][depth] = carried;
                carried = held;
            }
        }
        match at {
            Some(at) => self.stacks[layer][at] = carried,
            None => self.stacks[layer].push(carried),
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
        } else {
            ((score.0 - steps.0) / self.half_life).exp2()
        };
        Some(rate.clamp(0.0, 1.0))
    }

    /// Fixed memory charge (constant in run length).
    fn bytes(&self) -> u64 {
        let layers: usize = self.layers.iter().map(|l| l.capacity() * size_of::<Logged>()).sum();
        let steps = self.steps_score.capacity() * size_of::<Score>();
        let stacks: usize = self.stacks.iter().map(|s| s.capacity() * size_of::<usize>()).sum();
        let groups = self.groups.capacity() * size_of::<usize>();
        let pools: usize = self.pools.iter().map(|p| p.records * size_of::<Pooled>()).sum();
        let pooled: usize = self.pooled.iter().map(|p| p.capacity() * size_of::<bool>()).sum();
        (size_of::<Self>() + layers + steps + stacks + groups + pools + pooled) as u64
    }
}

/// Live half-life selector across [`HALF_LIVES`] × `{per-layer, group-pool}`:
/// the single owner of what decode history predicts.
pub(super) struct History {
    measures: Vec<Measure>,
    groups: Vec<usize>,
    members: Vec<usize>,
    slots: Vec<usize>,
    /// Per-layer decode step count and harmonic normalizer `H(n) = Σ_{d=1..n} 1/d`
    /// for `h = 0` recency rates.
    decode_steps: Vec<u64>,
    harmonic: Vec<f64>,
    /// Winning candidate in `0..2 * HALF_LIVES.len()`: `live % HALF_LIVES.len()`
    /// is the winning half-life index (`< HALF_LIVES.len()` per-layer, `>=` pooled).
    live: usize,
}

impl History {
    pub(super) fn new(experts: &[usize], groups: &[usize], records: &[usize]) -> Self {
        assert_eq!(experts.len(), groups.len(), "each layer has its group");
        let mut members = vec![0usize; records.len()];
        for &group in groups {
            members[group] += 1;
        }
        let slots: Vec<usize> = groups
            .iter()
            .map(|&group| if members[group] > 0 { records[group] / members[group] } else { 0 })
            .collect();
        let measures = HALF_LIVES
            .iter()
            .map(|&half_life| Measure::new(half_life, experts, groups, records))
            .collect();
        Self {
            measures,
            groups: groups.to_vec(),
            members,
            slots,
            decode_steps: vec![0; experts.len()],
            harmonic: vec![0.0; experts.len()],
            live: 0,
        }
    }

    #[inline]
    fn live_measure(&self) -> &Measure {
        &self.measures[self.live % HALF_LIVES.len()]
    }

    #[inline]
    fn missed(&self, candidate: usize) -> u64 {
        let measure = &self.measures[candidate % HALF_LIVES.len()];
        if candidate >= HALF_LIVES.len() {
            measure.pooled_misses
        } else {
            measure.misses
        }
    }

    /// Log a decode step's routed `experts` at `layer` and `step`: updates all
    /// half-life stacks and pools and selects the candidate with fewest misses.
    pub(super) fn read_decode(&mut self, layer: usize, step: Step, experts: &[usize]) {
        self.decode_steps[layer] += 1;
        self.harmonic[layer] += 1.0 / self.decode_steps[layer] as f64;
        let slots = self.slots[layer];
        for measure in &mut self.measures {
            measure.read_decode(layer, step, experts, slots);
        }
        self.live = (0..2 * HALF_LIVES.len())
            .min_by_key(|&candidate| self.missed(candidate))
            .expect("HALF_LIVES is non-empty");
    }

    /// Group `group` holds `records` records from now: updates each member
    /// layer's nominal share and fits every half-life's group pool.
    pub(super) fn resize(&mut self, group: usize, records: usize) {
        let share = if self.members[group] > 0 { records / self.members[group] } else { 0 };
        for (layer, &g) in self.groups.iter().enumerate() {
            if g == group {
                self.slots[layer] = share;
            }
        }
        for measure in &mut self.measures {
            measure.fit(group, records);
        }
    }

    /// Normalized decayed decode rate `p_∞ ∈ [0, 1]` for `(layer, expert)`
    /// under the live half-life, or `None` before the first decode step at `layer`.
    #[inline]
    pub(super) fn rate(&self, layer: usize, expert: usize) -> Option<f64> {
        self.live_measure().rate(layer, expert, self.harmonic[layer])
    }

    /// History tie-break rank for `(layer, expert)` under the live half-life.
    #[inline]
    pub(super) fn rank(&self, layer: usize, expert: usize) -> HistoryRank {
        self.live_measure().rank(layer, expert)
    }

    /// The live half-life, in steps.
    #[cfg(test)]
    pub(super) fn half_life(&self) -> f64 {
        self.live_measure().half_life
    }

    /// Fixed memory charge in bytes (constant in run length).
    pub(super) fn bytes(&self) -> u64 {
        let measures: u64 = self.measures.iter().map(Measure::bytes).sum();
        let meta = size_of::<Self>()
            + (self.groups.capacity() + self.members.capacity() + self.slots.capacity()) * size_of::<usize>()
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
        let mut measure = Measure::new(4., &[2], &[0], &[2]);
        for step in [1, 2, 3, 4] {
            measure.read_decode(0, step, &[0], 2);
        }
        measure.read_decode(0, 5, &[1], 2);
        assert!(measure.rank(0, 0) > measure.rank(0, 1));
        assert!(measure.rate(0, 0, 1.0).unwrap() > measure.rate(0, 1, 1.0).unwrap());
        assert!((Score::at(0).sum(Score::at(0), 4.).0 - 4.).abs() < 1e-12);
    }

    #[test]
    fn decayed_rate_sums_to_active_and_is_monotone_in_score() {
        let mut history = History::new(&[16], &[0], &[4]);
        assert_eq!(history.rate(0, 0), None, "cold start before any decode step");
        for step in 1..=40u64 {
            let experts = [0, ((step as usize) % 15) + 1];
            history.read_decode(0, step, &experts);
        }
        // Check on every non-zero half-life: sum of rates equals k = 2.0 exactly.
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
        let mut history = History::new(&[16], &[0], &[4]);
        for step in 1..=10u64 {
            history.read_decode(0, step, &[step as usize]);
        }
        // All 10 reads are distinct -> all candidates miss identically -> live stays on h = 0.
        assert_eq!(history.half_life(), 0.0);
        let expected_h: f64 = (1..=10).map(|d| 1.0 / d as f64).sum();
        assert!((history.harmonic[0] - expected_h).abs() < 1e-12);
        assert_eq!(history.rate(0, 0), Some(0.0), "unread expert has rate 0");
        let mut prev = 0.0;
        for step in 1..=10usize {
            let r = history.rate(0, step).unwrap();
            assert!(r > 0.0 && r <= 1.0, "rate {r} for step {step} out of (0, 1]");
            assert!(r > prev, "more recent step {step} ({r}) must strictly outrank older ({prev})");
            let want = 1.0 / ((1 + (10 - step)) as f64 * expected_h);
            assert!((r - want).abs() < 1e-12, "got {r}, want {want}");
            prev = r;
        }
    }

    #[test]
    fn the_half_life_selected_is_the_one_that_missed_least() {
        let mut history = History::new(&[128], &[0], &[2]);
        for step in 1..=60u64 {
            let experts = if step % 3 == 0 { vec![0, step as usize] } else { vec![step as usize] };
            history.read_decode(0, step, &experts);
        }
        assert!(history.half_life() > 0., "a half-life kept the hot expert");
        let recency = &history.measures[0];
        let live = history.live_measure();
        assert!(live.misses < recency.misses, "{} of {}", live.misses, recency.misses);
        assert!(history.rate(0, 0).unwrap() > history.rate(0, 60).unwrap());
    }

    #[test]
    fn history_bytes_is_constant_in_run_length() {
        let mut history = History::new(&[16], &[0], &[4]);
        let initial = history.bytes();
        for step in 1..=500u64 {
            let e0 = (step as usize) % 16;
            let e1 = (step as usize * 3 + 1) % 16;
            let mut chosen = vec![e0, e1];
            chosen.sort_unstable();
            chosen.dedup();
            history.read_decode(0, step, &chosen);
        }
        assert_eq!(history.bytes(), initial, "history memory footprint must be constant");
    }
}
