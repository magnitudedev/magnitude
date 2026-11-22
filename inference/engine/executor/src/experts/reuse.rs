//! Placement: the one ranking of what the bank holds, and the one owner of
//! how a group's records are split across its layers (`designs/streamed-decode-ranking.md`
//! §3.4, §3.7). It decides every residency choice (the store's demand victims,
//! its reserve's and a shrink's, a prompt group's seeds; beside a host tier
//! the bank's promotions and their victims), and prices every bank size
//! (`Stack`'s miss curve). A tier that covers every expert keeps them all, so
//! it reads them in the order the steps need them, by layer and index
//! (`Directory::cover_order`), with no choice for the ranking to make.
//!
//! Each tier's turnover `τ` and tokens per decode step `ā` come from their
//! single owner (`horizon::Time`), and front-row stand rates `r_f` from their
//! single owner (`interest::Stands`). Each layer's routing calibration
//! (`calibration::Calibration`) maintains O(1)-memory epoch-windowed tables
//! (`route`, `fell`, `history`, `far`) ordered by route/history position with
//! tie-class spreading.
//!
//! An expert's expected uses within `τ` tokens (`Expected { n, s }`, both
//! finite by construction) are computed via the exact first-fall decomposition
//! over the latest step's front rows `1..=d` for leads `g ≤ L = DEPTHS` and
//! the calibrated history rate `p_∞` past `L` (A1), with `S = Σ_{h ≤ τ/ā} N_h`
//! breaking ties in favour of sooner uses. At `Horizon::Unbounded` (an
//! observer before the first move, or a prompt seed before any decode step),
//! `Expected` carries the exact `τ → ∞` limit order `(p_∞, Σ_{g ≤ L} (p_g − p_∞))`.
//!
//! Shadows of the bank at every depth of foresight (`shadows::Shadows`) run on
//! true routes `BEHIND` steps behind the log, tracking `U_shadow(j)` and the
//! frozen gap `Δ_j = U_real − U_shadow(j)` with its age (A9).

mod calibration;
mod history;
mod shadows;

use calibration::Calibration;
use history::History;
pub(crate) use shadows::{Delta, Gap, ShadowFeed, Shadows};

use super::directory::RoutedRead;
use super::horizon::{self, Reader, Tier};
use super::interest::Stands;
use super::watcher::Step;
use super::yielding::Yielding;
use super::DEPTHS;
use std::collections::BTreeSet;
use std::sync::{Arc, MutexGuard};

/// What a read logged into the ranking is evidence of: a decode step's
/// reads, whose outcomes calibrate decode reuse, or a prompt group's union,
/// which seeds the `prompt` table and is never a decode outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    Decode,
    Prompt,
}

/// A step's rows at one layer as placement reads them: each row's depth (0
/// an exact row's, `j` a guessed row's routing the token `j` past its
/// exact row's), whether its guessed rows are verify rows (`verifies`, a
/// pass's rows being of one form), how many each chooses, and each
/// expert's prior `π` and choice indicator per row, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Rows {
    pub(super) experts: usize,
    pub(super) choices: usize,
    pub(super) depths: Vec<u8>,
    pub(super) verifies: bool,
    pub(super) priors: Vec<f32>,
    pub(super) chosen: Vec<bool>,
}

impl Rows {
    /// The rows of `keys`, `[rows, E]` row-major, each row's selection key
    /// of every expert as its route ranked them (a NaN as −∞, ties to the
    /// higher expert), each choosing `choices`, their depths `depths`. A
    /// row's prior of an expert is its key's share of the row's keys,
    /// scaled to the row's `K` choices (the fused form's softmax
    /// probability `K·p`, the general form's biased score as the route
    /// selects by it), at most 1.
    pub fn of(choices: usize, depths: &[u8], keys: &[f32]) -> Self {
        assert!(!depths.is_empty() && keys.len().is_multiple_of(depths.len()), "{} keys of {} rows", keys.len(), depths.len());
        let experts = keys.len() / depths.len();
        assert!(choices > 0 && choices <= experts, "a row chooses {choices} of {experts}");
        let mut priors = vec![0.0f32; keys.len()];
        let mut chosen = vec![false; keys.len()];
        let mut order = (0..experts).collect::<Vec<_>>();
        for (row, keys) in keys.chunks(experts).enumerate() {
            let key = |expert: usize| if keys[expert].is_nan() { f32::NEG_INFINITY } else { keys[expert] };
            order.sort_unstable_by(|&left, &right| key(right).total_cmp(&key(left)).then(right.cmp(&left)));
            let row_chosen = &mut chosen[row * experts..][..experts];
            for &expert in &order[..choices] {
                row_chosen[expert] = true;
            }
            let mass = keys.iter().map(|&key| f64::from(key.max(0.0))).sum::<f64>();
            let row_priors = &mut priors[row * experts..][..experts];
            if mass > 0.0 {
                for (prior, &key) in row_priors.iter_mut().zip(keys) {
                    *prior = (choices as f64 * f64::from(key.max(0.0)) / mass).min(1.0) as f32;
                }
            } else {
                // No key above zero: the choice is all the row says.
                for &expert in &order[..choices] {
                    row_priors[expert] = 1.0;
                }
            }
        }
        Self { experts, choices, depths: depths.to_vec(), verifies: false, priors, chosen }
    }

    /// A prompt group's union of `experts` experts, whose keys are unknown:
    /// each routed expert's choices `routed` over the group's rows, its
    /// prior its share of them scaled to a row's `choices`, the prompt's
    /// own rate of choosing it.
    pub fn union(experts: usize, choices: usize, routed: &[(usize, u32)]) -> Self {
        let total = routed.iter().map(|&(_, count)| f64::from(count)).sum::<f64>();
        let mut priors = vec![0.0f32; experts];
        let mut chosen = vec![false; experts];
        for &(expert, count) in routed {
            chosen[expert] = true;
            if total > 0.0 {
                priors[expert] = (choices as f64 * f64::from(count) / total).min(1.0) as f32;
            }
        }
        Self { experts, choices, depths: vec![0], verifies: false, priors, chosen }
    }

    pub(super) fn experts(&self) -> usize {
        self.experts
    }

    pub(super) fn verifies(&self) -> bool {
        self.verifies
    }

    pub(super) fn deepest(&self) -> u8 {
        self.depths.iter().copied().max().unwrap_or(0)
    }

    pub(super) fn has_depth(&self, depth: u8) -> bool {
        self.depths.contains(&depth)
    }

    /// Number of rows in this layer's routing keys.
    pub(in crate::experts) fn row_count(&self) -> usize {
        self.depths.len()
    }

    /// Chosen expert indices of row `row` (`0..self.row_count()`), ascending.
    pub(in crate::experts) fn row_experts(&self, row: usize) -> Vec<usize> {
        let start = row * self.experts;
        self.chosen[start..start + self.experts]
            .iter()
            .enumerate()
            .filter_map(|(expert, &picked)| picked.then_some(expert))
            .collect()
    }

    /// The rows of depth `depth` merged: each expert's combined prior
    /// `1 − Π_r (1 − π_{r,e})` and whether any row of that depth chose it;
    /// none where no row has that depth.
    #[cfg(test)]
    pub(super) fn at(&self, depth: u8) -> Option<(Vec<f32>, Vec<bool>)> {
        let mut rows = self.depths.iter().enumerate().filter_map(|(row, &of)| (of == depth).then_some(row));
        let first = rows.next()?;
        let mut priors = self.priors[first * self.experts..][..self.experts].to_vec();
        let mut chosen = self.chosen[first * self.experts..][..self.experts].to_vec();
        for other in rows {
            let other_priors = &self.priors[other * self.experts..][..self.experts];
            let other_chosen = &self.chosen[other * self.experts..][..self.experts];
            for (merged, &prior) in priors.iter_mut().zip(other_priors) {
                *merged = 1.0 - (1.0 - *merged) * (1.0 - prior);
            }
            for (merged, &picked) in chosen.iter_mut().zip(other_chosen) {
                *merged |= picked;
            }
        }
        Some((priors, chosen))
    }
}

/// A decode step's selection keys as its routes wrote them
/// (`streaming_words.seismic`'s keys): `K` choices a row, each row's depth
/// (0 an exact row's, `j` a guessed row's of a request whose exact row is
/// in the step `j` positions before it), whether its guessed rows are
/// verify rows, each request's and its first exact row's position, and
/// per streamed layer every row's key of every expert, `[rows, E]`
/// row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct StepKeys {
    pub choices: usize,
    pub depths: Vec<u8>,
    pub verifies: bool,
    pub positions: Arc<[(u64, u64)]>,
    pub layers: Vec<Vec<f32>>,
}

impl StepKeys {
    /// `layer`'s rows as placement reads them ([`Rows::of`]).
    pub fn rows(&self, layer: usize) -> Rows {
        Rows { verifies: self.verifies, ..Rows::of(self.choices, &self.depths, &self.layers[layer]) }
    }
}

/// An expert's expected uses (design §3.4), the rank's key before its last
/// read, both finite by construction. At a finite `τ`: `n = N`, its expected
/// uses within `τ`, and `s = S = Σ_{h ≤ τ/ā} N_h`, its uses within every
/// shorter horizon summed (the sooner use kept). At
/// [`horizon::Horizon::Unbounded`]: `N`'s exact `τ → ∞` limit order,
/// `n = p_∞` and `s = Σ_{g ≤ L} (p_g − p_∞)`. Expectations taken at different
/// horizons are never compared.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Expected {
    n: f64,
    s: f64,
}

impl Expected {
    /// Zero expected uses.
    pub(super) const ZERO: Self = Self { n: 0.0, s: 0.0 };

    /// Panics where `n` or `s` is not finite: an invariant violation
    /// (design §1 bar 3).
    pub fn new(n: f64, s: f64) -> Self {
        assert!(n.is_finite() && s.is_finite(), "non-finite Expected({n}, {s})");
        let n = if n == 0.0 { 0.0 } else { n };
        let s = if s == 0.0 { 0.0 } else { s };
        Self { n, s }
    }

    /// `N` (or `p_∞` at the limit order): what a promotion's earning reads at
    /// `τ_0` (`Directory::earning`).
    pub fn uses(self) -> f64 {
        self.n
    }

    /// `S` (or `Σ_{g ≤ L} (p_g − p_∞)` at the limit order): the sooner-use
    /// tie-break.
    pub fn sooner(self) -> f64 {
        self.s
    }
}

impl Eq for Expected {}

impl PartialOrd for Expected {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Lexicographic `(n, s)`, least first.
impl Ord for Expected {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.n.total_cmp(&other.n).then(self.s.total_cmp(&other.s))
    }
}

/// An expert's rank, least first: its expected uses `(N, S)`, then its last
/// read, so experts nothing tells apart rank by what was read later. The one
/// order every choice of what to keep compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rank {
    expected: Expected,
    last: Option<Step>,
}

impl Rank {
    pub(super) fn new(expected: Expected, last: Option<Step>) -> Self {
        Self { expected, last }
    }

    /// The step that last read the expert, if one has.
    pub fn last(self) -> Option<Step> {
        self.last
    }

    pub(super) fn expected(self) -> Expected {
        self.expected
    }
}

/// A read the ranking has not logged yet: its step (a decode landing's, or
/// a running prompt group's latest). Later reads order after earlier ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Unlogged {
    pub step: Step,
}

impl Unlogged {
    pub fn at(step: Step) -> Self {
        Self { step }
    }
}

/// The share of whole lead `g ≥ 1` within a horizon of `horizon_steps = τ / ā`
/// decode steps: whole to its last whole lead, the fraction of the next.
pub(super) fn share(horizon_steps: f64, g: usize) -> f64 {
    (horizon_steps - (g - 1) as f64).clamp(0.0, 1.0)
}

/// Conditional front-row stand rates `r_f` for `f = 1..DEPTHS` (0-indexed:
/// `rates[f - 1] = r_f`).
pub(super) type StandRates = [f64; DEPTHS - 1];

/// Read conditional front-row stand rates `r_1..r_{DEPTHS-1}` from `stands`.
/// Unjudged depths read their bound 1.0 (design §3.3).
///
/// Must NOT be called while holding `horizon::Time::lock()`, as
/// `Stands::front_rates` acquires that lock internally.
pub(super) fn front_rates(stands: &Stands) -> StandRates {
    stands.front_rates()
}

/// First-fall decomposition of `p_g` for `g = 1..=DEPTHS` (design §3.4):
///
/// `p_g = Σ_{f=1}^{min(g,d)} σ_{f-1} (1 − r_f) · Fell_{g-f+1}(c_{f-1}) + σ_{min(g,d)} · E_g`
///
/// where `E_g = [row g chose e]` for `g ≤ d`, else `Route_{g-d}(c_d)`.
pub(super) fn lead_probs(
    calibration: &Calibration,
    history: &History,
    layer: usize,
    expert: usize,
    rates: &StandRates,
    epoch: u64,
) -> [f64; DEPTHS] {
    let form = calibration.latest_form().unwrap_or(calibration::Form::Front);
    let d = calibration.latest_front_depth();
    assert!(d < DEPTHS, "front depth {d} must be below DEPTHS ({DEPTHS})");
    if d == 0 {
        return std::array::from_fn(|index| calibration.route_prob(form, index + 1, 0, expert, epoch, history, layer));
    }
    let mut sigma = [1.0f64; DEPTHS];
    let mut fall_weight = [0.0f64; DEPTHS - 1];
    for f in 1..=d {
        let r_f = rates[f - 1];
        fall_weight[f - 1] = sigma[f - 1] * (1.0 - r_f);
        sigma[f] = sigma[f - 1] * r_f;
    }
    std::array::from_fn(|index| {
        let g = index + 1;
        let m = g.min(d);
        let mut p = 0.0;
        for f in 1..=m {
            let w = fall_weight[f - 1];
            if w > 0.0 {
                p += w * calibration.fell_prob(g - f + 1, f - 1, expert, epoch, history, layer);
            }
        }
        if sigma[m] > 0.0 {
            let e_g = if g <= d {
                f64::from(u8::from(calibration.row_chose(g, expert)))
            } else {
                calibration.route_prob(form, g - d, d, expert, epoch, history, layer)
            };
            p += sigma[m] * e_g;
        }
        p.clamp(0.0, 1.0)
    })
}

/// Aggregate per-lead probabilities `probs` (`g = 1..=DEPTHS`) and history rate
/// `p_inf` (`g > DEPTHS`, A1) into [`Expected`] at `horizon` and `a_bar`, with
/// an optional single-lead replacement `(g_0, p)` (used for [`Unlogged`] with
/// `p = 1.0` and [`Reuse::expected_with`] with the ledger's `p`).
///
/// Closed-form O(L) evaluation: no loop over `τ`.
pub(super) fn expected_from(
    probs: [f64; DEPTHS],
    p_inf: f64,
    horizon: horizon::Horizon,
    a_bar: f64,
    override_lead: Option<(usize, f64)>,
) -> Expected {
    assert!(a_bar.is_finite() && a_bar > 0.0, "invalid tokens per step {a_bar}");
    assert!(p_inf.is_finite() && (0.0..=1.0).contains(&p_inf), "invalid history rate {p_inf}");
    let delta = override_lead.map(|(lead, p)| {
        assert!(lead >= 1, "lead must be at least 1");
        assert!(p.is_finite() && (0.0..=1.0).contains(&p), "invalid override probability {p}");
        let old = if lead <= DEPTHS { probs[lead - 1] } else { p_inf };
        (lead, p - old)
    });
    match horizon {
        horizon::Horizon::Tokens(tau) => {
            let h_steps = tau.get() / a_bar;
            let m = h_steps.floor();
            let mut n = (h_steps - DEPTHS as f64).max(0.0) * p_inf;
            let tail = (m - DEPTHS as f64).max(0.0);
            let mut s = tail * (tail + 1.0) * 0.5 * p_inf;
            for (index, &p_g) in probs.iter().enumerate() {
                let g = index + 1;
                n += share(h_steps, g) * p_g;
                s += (m - index as f64).max(0.0) * p_g;
            }
            if let Some((lead, diff)) = delta {
                n += share(h_steps, lead) * diff;
                s += (m - (lead - 1) as f64).max(0.0) * diff;
            }
            Expected::new(n, s)
        }
        horizon::Horizon::Unbounded => {
            // Exact τ → ∞ limit of N: (τ/ā) * p_∞ + Σ_{g ≤ L} (p_g − p_∞) + override diff.
            let mut excess = probs.iter().map(|&p_g| p_g - p_inf).sum::<f64>();
            if let Some((_, diff)) = delta {
                excess += diff;
            }
            Expected::new(p_inf, excess)
        }
    }
}

/// Convenience: compute [`Expected`] directly from `calibration`, `history`,
/// `layer`, `expert`, `rates`, `horizon`, `a_bar`, and optional single-lead replacement.
#[allow(clippy::too_many_arguments)]
pub(super) fn expected_for(
    calibration: &Calibration,
    history: &History,
    layer: usize,
    expert: usize,
    rates: &StandRates,
    horizon: horizon::Horizon,
    a_bar: f64,
    epoch: u64,
    override_lead: Option<(usize, f64)>,
) -> Expected {
    let probs = lead_probs(calibration, history, layer, expert, rates, epoch);
    let p_inf = calibration.history_prob(expert, history, layer);
    expected_from(probs, p_inf, horizon, a_bar, override_lead)
}

/// An expert's place in its group's order, least first.
type Placed = (Rank, usize, usize);

/// Sentinel index in [`Pools::pos`] for an expert not currently held in its group's pool.
const NOT_HELD: usize = usize::MAX;

/// Each group's records as a pool of what is read: the experts each holds,
/// by the rank each was placed at, and how many it holds at most.
///
/// Backed by an indexed binary min-heap per group (ordered by strict total
/// order `Placed = (Rank, layer, expert)`) so rank updates sift in place in
/// contiguous memory instead of removing and re-inserting into a `BTreeSet`.
#[derive(Clone, Debug)]
pub(super) struct Pools {
    pub(super) records: Vec<usize>,
    held: Vec<Vec<Placed>>,
    pos: Vec<Vec<usize>>,
}

impl Pools {
    pub(super) fn new(experts: &[usize], records: &[usize]) -> Self {
        Self {
            records: records.to_vec(),
            held: records.iter().map(|&cap| Vec::with_capacity(cap)).collect(),
            pos: experts.iter().map(|&experts| vec![NOT_HELD; experts]).collect(),
        }
    }

    #[inline]
    fn sift_up(&mut self, group: usize, mut idx: usize) {
        let heap = &mut self.held[group];
        let entry = heap[idx];
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if entry < heap[parent] {
                let moved = heap[parent];
                heap[idx] = moved;
                self.pos[moved.1][moved.2] = idx;
                idx = parent;
            } else {
                break;
            }
        }
        heap[idx] = entry;
        self.pos[entry.1][entry.2] = idx;
    }

    #[inline]
    fn sift_down(&mut self, group: usize, mut idx: usize) {
        let heap = &mut self.held[group];
        let len = heap.len();
        let entry = heap[idx];
        loop {
            let left = 2 * idx + 1;
            if left >= len {
                break;
            }
            let right = left + 1;
            let smallest = if right < len && heap[right] < heap[left] { right } else { left };
            if heap[smallest] < entry {
                let moved = heap[smallest];
                heap[idx] = moved;
                self.pos[moved.1][moved.2] = idx;
                idx = smallest;
            } else {
                break;
            }
        }
        heap[idx] = entry;
        self.pos[entry.1][entry.2] = idx;
    }

    #[inline]
    fn pop_min(&mut self, group: usize) -> Option<Placed> {
        let heap = &mut self.held[group];
        if heap.is_empty() {
            return None;
        }
        let min = heap.swap_remove(0);
        self.pos[min.1][min.2] = NOT_HELD;
        if !heap.is_empty() {
            let top = heap[0];
            self.pos[top.1][top.2] = 0;
            self.sift_down(group, 0);
        }
        Some(min)
    }

    #[inline]
    fn push(&mut self, group: usize, entry: Placed) {
        let idx = self.held[group].len();
        self.held[group].push(entry);
        self.pos[entry.1][entry.2] = idx;
        self.sift_up(group, idx);
    }

    /// `layer`'s read of `experts` in group `group`, `rank` each of the
    /// layer's experts' rank now: its held experts move to their ranks, and
    /// each it lacks enters, least ranked first, into the record of the
    /// pool's least ranked expert where it is full. How many it lacked.
    pub(super) fn read(&mut self, layer: usize, group: usize, experts: &[usize], rank: impl Fn(usize) -> Rank) -> u64 {
        let num_experts = self.pos[layer].len();
        for expert in 0..num_experts {
            let idx = self.pos[layer][expert];
            if idx != NOT_HELD {
                let was = self.held[group][idx].0;
                let now = rank(expert);
                if now != was {
                    self.held[group][idx].0 = now;
                    if now < was {
                        self.sift_up(group, idx);
                    } else {
                        self.sift_down(group, idx);
                    }
                }
            }
        }
        let mut lacked = experts
            .iter()
            .copied()
            .filter(|&expert| self.pos[layer][expert] == NOT_HELD)
            .map(|expert| (rank(expert), expert))
            .collect::<Vec<_>>();
        lacked.sort_unstable();
        for &(rank, expert) in &lacked {
            if self.held[group].len() >= self.records[group] {
                if self.pop_min(group).is_none() {
                    continue;
                }
            }
            if self.held[group].len() < self.records[group] {
                self.push(group, (rank, layer, expert));
            }
        }
        lacked.len() as u64
    }

    /// Group `group` holds `records` from now: its least ranked leave until
    /// it fits.
    pub(super) fn fit(&mut self, group: usize, records: usize) {
        self.records[group] = records;
        while self.held[group].len() > records {
            self.pop_min(group).expect("a pool past its records holds one");
        }
    }

    /// Fixed memory charge of the pool structures (constant in run length).
    pub(super) fn bytes(&self) -> u64 {
        let pos: usize = self.pos.iter().map(|layer| layer.len() * std::mem::size_of::<usize>()).sum();
        let records = self.records.len() * std::mem::size_of::<usize>();
        let cap: usize = self.records.iter().map(|&records| records * std::mem::size_of::<Placed>()).sum();
        (pos + records + cap) as u64
    }
}

/// Every streamed layer's experts ranked by their expected uses within
/// their group's turnover, as the routes calibrate them.
pub struct Reuse {
    layers: Vec<Calibration>,
    /// Measured exponential-decay half-life history selector (single owner of `p_∞`).
    history: History,
    /// Each layer's group, and each group's member layers.
    groups: Vec<usize>,
    members: Vec<usize>,
    /// The pools of what is read.
    pools: Pools,
    /// Shared time owner (clock, per-(group, tier) turnovers, epochs).
    time: horizon::Time,
    /// Cached tier-0 Move horizon per group for the latest logged step:
    /// a step reads τ once at its first log so every layer ranks at one value.
    step_horizon: Vec<Option<(Step, horizon::Horizon, f64)>>,
    /// The one owner of stand rates, read for front-row foresight.
    stands: Stands,
    /// Each group's experts by their ranks, least first, each layer's
    /// experts at their ranks as their layer was last read: what prices
    /// every pool size ([`Self::depth`]) and what the bank is asked to hold
    /// ([`Self::in_top`]).
    order: Vec<Vec<Placed>>,
    ranks: Vec<Vec<Rank>>,
    /// Layers whose `ranks[layer]` changed since their group's `order` and
    /// `top` were last rebuilt ([`Self::read_layer`], [`Self::finish_batch`]).
    dirty: Vec<bool>,
    top: Vec<Vec<bool>>,
    /// The (layer, expert)s whose membership of what the bank is asked to
    /// hold changed since they were last taken ([`Self::take_changes`]).
    changed: BTreeSet<(usize, usize)>,
    /// The reads logged ([`Self::reads`]).
    reads: u64,
    /// What the shadows have yet to read of the log ([`Self::shadows`]),
    /// none before they are made.
    shadowed: Option<Vec<ShadowFeed>>,
}

/// The one ranking, shared by its feed (`Stack`, which reads it in the
/// device's order, one hold a read) and everything that ranks by it (the
/// directory, whose demand claims a parked request waits on): the feed is
/// its background, so a claim waits at most one read's hold, never the
/// whole feed (a read of each of a decode step's streamed layers).
#[derive(Clone)]
pub struct Ranking(Arc<Yielding<Reuse>>);

/// A lock-free per-group snapshot of `(horizon, a_bar, epoch, front_rates)`
/// for ranking experts of `group` at `(tier, reader)` without re-acquiring
/// `horizon::Time` or `interest::Stands` on every record.
#[derive(Clone, Copy)]
pub struct Ranker<'a> {
    reuse: &'a Reuse,
    group: usize,
    horizon: horizon::Horizon,
    a_bar: f64,
    epoch: u64,
    rates: StandRates,
}

impl<'a> Ranker<'a> {
    /// [`Reuse::expected`] for `layer` (a member of this ranker's group)
    /// using the snapshotted `(horizon, a_bar, epoch, rates)`.
    pub fn expected(&self, layer: usize, expert: usize, read: Option<Unlogged>) -> Expected {
        debug_assert_eq!(
            self.reuse.groups[layer], self.group,
            "ranker for group {} called on layer {layer} of group {}",
            self.group, self.reuse.groups[layer],
        );
        let override_lead = read.and_then(|read| self.reuse.lead_of(layer, read.step)).map(|lead| (lead, 1.0));
        expected_for(
            &self.reuse.layers[layer],
            &self.reuse.history,
            layer,
            expert,
            &self.rates,
            self.horizon,
            self.a_bar,
            self.epoch,
            override_lead,
        )
    }

    /// [`Reuse::rank`] for `layer` (a member of this ranker's group)
    /// using the snapshotted `(horizon, a_bar, epoch, rates)`.
    pub fn rank(&self, layer: usize, expert: usize, read: Option<Unlogged>) -> Rank {
        let expected = self.expected(layer, expert, read);
        let last = self.reuse.layers[layer].last_read(expert).max(read.map(|read| read.step));
        Rank::new(expected, last)
    }
}

impl Reuse {
    /// No read logged of layers of `experts` experts each, in bank groups
    /// `groups`, whose pools hold `records` records each, timed by `time`.
    pub fn new(experts: &[usize], groups: &[usize], records: &[usize], time: horizon::Time) -> Self {
        assert_eq!(experts.len(), groups.len(), "each layer has its group");
        let mut members = vec![0; records.len()];
        groups.iter().for_each(|&group| members[group] += 1);
        let none = Rank::new(Expected::ZERO, None);
        let mut order = vec![Vec::new(); records.len()];
        for (layer, (&experts, &group)) in experts.iter().zip(groups).enumerate() {
            order[group].extend((0..experts).map(|expert| (none, layer, expert)));
        }
        order.iter_mut().for_each(|order| order.sort_unstable());
        let epochs: Vec<u64> = {
            let times = time.lock();
            (0..records.len()).map(|group| times.group_epoch(group)).collect()
        };
        let layers = experts
            .iter()
            .zip(groups)
            .map(|(&experts, &group)| Calibration::new(experts, 1, epochs[group]))
            .collect();
        let stands = Stands::new(time.clone());
        let history = History::new(experts, groups, records);
        Self {
            layers,
            history,
            groups: groups.to_vec(),
            members,
            pools: Pools::new(experts, records),
            time,
            step_horizon: vec![None; records.len()],
            stands,
            order,
            ranks: experts.iter().map(|&experts| vec![none; experts]).collect(),
            dirty: vec![false; experts.len()],
            top: experts.iter().map(|&experts| vec![false; experts]).collect(),
            changed: BTreeSet::new(),
            reads: 0,
            shadowed: None,
        }
    }

    /// [`Self::new`], shared.
    pub fn ranking(experts: &[usize], groups: &[usize], records: &[usize], time: horizon::Time) -> Ranking {
        Ranking(Arc::new(Yielding::new(Self::new(experts, groups, records, time))))
    }

    /// The ranking `ranking` holds now, held ahead of its feed.
    pub fn of(ranking: &Ranking) -> MutexGuard<'_, Self> {
        ranking.0.hold("the reuse ranking")
    }

    /// The ranking `ranking` holds now, for its feed's next read: once no
    /// other holder waits for it ([`Ranking`]).
    pub fn feeding(ranking: &Ranking) -> MutexGuard<'_, Self> {
        ranking.0.background("the reuse ranking")
    }

    /// The load's time handle and stand-rates owner, shared with the ledger,
    /// gate, and telemetry.
    pub(in crate::experts) fn rates(&self) -> (horizon::Time, Stands) {
        (self.time.clone(), self.stands.clone())
    }

    /// Read `(horizon, a_bar)` for `(group, tier, reader)` from [`horizon::Time`].
    /// Before the first decode step (when `Times::horizon` returns `None`),
    /// falls back to `Horizon::Unbounded` and `ā = 1.0`. Wherever `τ` exists,
    /// `ā` must also exist (same window invariant).
    fn horizon_of(&self, group: usize, tier: Tier, reader: Reader) -> (horizon::Horizon, f64, u64) {
        let times = self.time.lock();
        let epoch = times.group_epoch(group);
        let (horizon, a_bar) = match times.horizon(group, tier, reader) {
            Some(horizon) => (horizon, times.tokens_per_step(group).expect("ā exists wherever τ does")),
            None => (horizon::Horizon::Unbounded, 1.0),
        };
        (horizon, a_bar, epoch)
    }

    /// The step-cached tier-0 Move `(horizon, a_bar)` for `group` at `step`
    /// (design §3.2: a step reads τ once at its first log, so all its layers
    /// rank at one value), plus the group's epoch and, when shadows are
    /// attached, the step's position on the token clock (`Tokens::ZERO` for
    /// prompt reads).
    fn step_horizon_at(&mut self, group: usize, step: Step, evidence: Evidence, clock: Option<horizon::Tokens>) -> (horizon::Horizon, f64, u64, horizon::Tokens) {
        let times = self.time.lock();
        let epoch = times.group_epoch(group);
        let clock = match (self.shadowed.is_some(), evidence, clock) {
            (true, Evidence::Decode, Some(at)) => at,
            (true, Evidence::Decode, None) => times.clock.at(step),
            _ => horizon::Tokens::ZERO,
        };
        let (horizon, a_bar) = match self.step_horizon[group] {
            Some((at, horizon, a_bar)) if at >= step => (horizon, a_bar),
            _ => {
                let (horizon, a_bar) = match times.horizon(group, Tier::COMPUTE, Reader::Move) {
                    Some(horizon) => (horizon, times.tokens_per_step(group).expect("ā exists wherever τ does")),
                    None => (horizon::Horizon::Unbounded, 1.0),
                };
                self.step_horizon[group] = Some((step, horizon, a_bar));
                (horizon, a_bar)
            }
        };
        (horizon, a_bar, epoch, clock)
    }

    /// Lead `g ≥ 1` of target step `step` past `layer`'s latest logged step.
    /// Before the layer's first log, taken at lead 1 (design §3.4).
    /// Returns `None` if the log has already reached or passed `step`.
    pub(crate) fn lead_of(&self, layer: usize, step: Step) -> Option<usize> {
        match self.layers[layer].latest_step() {
            None => Some(1),
            Some(latest) => step.checked_sub(latest).filter(|&lead| lead > 0).map(|lead| usize::try_from(lead).expect("lead fits usize")),
        }
    }

    /// Log the route's `read` (each expert once, ascending), its rows'
    /// keys `rows` where known, and whether it is decode or prompt evidence:
    /// returns each expert's depth in slots before the read ([`Self::depth`]).
    pub fn read(&mut self, read: &RoutedRead, rows: Option<Arc<Rows>>, evidence: Evidence) -> Vec<usize> {
        self.read_at(read, rows, evidence, None)
    }

    /// [`Self::read`] with a pre-captured token clock position `clock` for
    /// `read.sequence` (captured while the step was still live on `Clock`'s
    /// ring, so asynchronous stack feeding never races ring eviction).
    pub(crate) fn read_at(&mut self, read: &RoutedRead, rows: Option<Arc<Rows>>, evidence: Evidence, clock: Option<horizon::Tokens>) -> Vec<usize> {
        let depths = self.read_layer(read, rows, evidence, clock);
        self.rerank_group(self.groups[read.layer]);
        depths
    }

    /// Log one layer's read and update its `history`, calibration, `pools`,
    /// and `ranks[layer]`, deferring the group-wide `order` rebuild and
    /// `retop` until [`Self::finish_batch`]. If the same layer is already
    /// dirty from an earlier read in the same batch, its group is rebuilt
    /// first so `depths` reflects the prior read.
    pub(crate) fn read_layer(&mut self, read: &RoutedRead, rows: Option<Arc<Rows>>, evidence: Evidence, clock: Option<horizon::Tokens>) -> Vec<usize> {
        assert!(read.experts.windows(2).all(|pair| pair[0] < pair[1]), "a read lists each expert once, ascending");
        self.reads += 1;
        let (layer, group) = (read.layer, self.groups[read.layer]);
        if self.dirty[layer] {
            self.rerank_group(group);
        }
        let depths = read.experts.iter().map(|&expert| self.depth(layer, expert)).collect();
        let (horizon, a_bar, epoch, clock) = self.step_horizon_at(group, read.sequence, evidence, clock);
        let far_lead = match horizon {
            horizon::Horizon::Tokens(tau) => Some((tau.get() / a_bar).ceil().max(1.0) as usize),
            horizon::Horizon::Unbounded => None,
        };
        let choices = usize::try_from(read.choices).expect("choices fits usize");
        match evidence {
            Evidence::Decode => {
                if choices > 0 {
                    self.layers[layer].set_active(choices);
                }
                let form = if rows.as_deref().is_some_and(Rows::verifies) {
                    calibration::Form::Verify
                } else {
                    calibration::Form::Front
                };
                self.history.read_decode(layer, read.sequence, &read.experts);
                self.layers[layer].log_decode(
                    read.sequence,
                    form,
                    rows.as_deref(),
                    &read.experts,
                    epoch,
                    far_lead,
                    &self.history,
                    layer,
                );
            }
            Evidence::Prompt => {
                if let Some(rows) = rows.as_deref() {
                    self.layers[layer].log_prompt(read.sequence, rows, &self.history, layer);
                } else {
                    let counts: Vec<(usize, u32)> = read.experts.iter().map(|&e| (e, 1)).collect();
                    let union = Rows::union(self.layers[layer].experts(), choices.max(1), &counts);
                    self.layers[layer].log_prompt(read.sequence, &union, &self.history, layer);
                }
            }
        }
        // Release time lock already happened above; safe to read stands now.
        let rates = front_rates(&self.stands);
        let calibration = &self.layers[layer];
        let ranks: Vec<Rank> = (0..calibration.experts())
            .map(|expert| {
                let expected = expected_for(calibration, &self.history, layer, expert, &rates, horizon, a_bar, epoch, None);
                Rank::new(expected, calibration.last_read(expert))
            })
            .collect();
        self.pools.read(layer, group, &read.experts, |expert| ranks[expert]);
        self.ranks[layer] = ranks;
        self.dirty[layer] = true;
        if let Some(shadowed) = &mut self.shadowed {
            shadowed.push(ShadowFeed::Read { read: read.clone(), rows, evidence, clock });
        }
        depths
    }

    /// Rebuild `order` and `top` once for every group that had at least one
    /// layer updated by [`Self::read_layer`].
    pub(crate) fn finish_batch(&mut self) {
        for group in 0..self.order.len() {
            self.rerank_group(group);
        }
    }

    /// Record that `step`'s front row `first_fall` (1-based, `1..DEPTHS`) was
    /// the first guessed row to fall at `layer`: counts its `fell` table
    /// against the outcomes following `step` (design §3.4).
    pub fn fell(&mut self, layer: usize, step: Step, first_fall: usize) {
        let group = self.groups[layer];
        let epoch = self.time.lock().group_epoch(group);
        self.layers[layer].record_fell(step, first_fall, epoch);
    }

    /// Rebuild `self.order[group]` and `self.top` once for every dirty layer
    /// in `group`. Returns true if any layer in `group` was dirty.
    fn rerank_group(&mut self, group: usize) -> bool {
        let dirty_count = self
            .groups
            .iter()
            .enumerate()
            .filter(|&(layer, &g)| g == group && self.dirty[layer])
            .count();
        if dirty_count == 0 {
            return false;
        }
        if dirty_count == self.members[group] {
            let order = &mut self.order[group];
            order.clear();
            for (layer, &g) in self.groups.iter().enumerate() {
                if g == group {
                    self.dirty[layer] = false;
                    order.extend(self.ranks[layer].iter().enumerate().map(|(expert, &rank)| (rank, layer, expert)));
                }
            }
            order.sort_unstable();
        } else {
            let mut entering = Vec::new();
            for (layer, &g) in self.groups.iter().enumerate() {
                if g == group && self.dirty[layer] {
                    entering.extend(self.ranks[layer].iter().enumerate().map(|(expert, &rank)| (rank, layer, expert)));
                }
            }
            entering.sort_unstable();
            let order = std::mem::take(&mut self.order[group]);
            let mut merged = Vec::with_capacity(order.len());
            let mut entering = entering.into_iter().peekable();
            for placed in order.into_iter().filter(|&(_, of, _)| !self.dirty[of]) {
                while let Some(next) = entering.next_if(|next| *next < placed) {
                    merged.push(next);
                }
                merged.push(placed);
            }
            merged.extend(entering);
            self.order[group] = merged;
            for (layer, &g) in self.groups.iter().enumerate() {
                if g == group {
                    self.dirty[layer] = false;
                }
            }
        }
        self.retop(group);
        true
    }

    /// What the bank is asked to hold of group `group`: its most ranked as
    /// many as its pool's records, of those any use is expected of. Each
    /// change is noted.
    fn retop(&mut self, group: usize) {
        let order = &self.order[group];
        let cut = order.len().saturating_sub(self.pools.records[group]);
        for (at, &(rank, layer, expert)) in order.iter().enumerate() {
            let top = at >= cut && rank.expected > Expected::ZERO;
            if self.top[layer][expert] != top {
                self.top[layer][expert] = top;
                self.changed.insert((layer, expert));
            }
        }
    }

    /// Group `group`'s pools hold `records` records from now: its least
    /// ranked leave the pool of what is read, and what the bank is asked to
    /// hold is taken again; the shadows' as they read it.
    pub fn resize(&mut self, group: usize, records: usize) {
        self.pools.fit(group, records);
        self.history.resize(group, records);
        self.step_horizon[group] = None;
        if !self.rerank_group(group) {
            self.retop(group);
        }
        if let Some(shadowed) = &mut self.shadowed {
            shadowed.push(ShadowFeed::Resize { group, records });
        }
    }

    /// The records group `group`'s pools hold.
    pub fn records(&self, group: usize) -> usize {
        self.pools.records[group]
    }

    /// Whether the ranking has logged a read of `layer`.
    pub fn has_read(&self, layer: usize) -> bool {
        self.layers[layer].latest_step().is_some()
    }

    /// Group `group`'s ranked `(rank, layer, expert)` entries in ascending rank
    /// order (highest-ranked at the end).
    pub fn top_order(&self, group: usize) -> &[(Rank, usize, usize)] {
        &self.order[group]
    }

    /// Whether the bank is asked to hold `layer`'s `expert`: its group's
    /// most ranked, as many as its records.
    pub fn in_top(&self, layer: usize, expert: usize) -> bool {
        self.top[layer][expert]
    }

    /// Up to `limit` highest-expected-use experts across all groups matching
    /// `accept(layer, expert)` (with positive expected uses), ordered
    /// descending by expected uses (`Rank`).
    pub fn top_candidates(&self, limit: usize, mut accept: impl FnMut(usize, usize) -> bool) -> Vec<(Rank, usize, usize)> {
        if limit == 0 {
            return Vec::new();
        }
        let mut best = Vec::new();
        for group_order in &self.order {
            let mut taken = 0;
            for &(rank, layer, expert) in group_order.iter().rev() {
                if rank.expected <= Expected::ZERO {
                    break;
                }
                if accept(layer, expert) {
                    best.push((rank, layer, expert));
                    taken += 1;
                    if taken >= limit {
                        break;
                    }
                }
            }
        }
        if self.order.len() > 1 {
            best.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            best.truncate(limit);
        }
        best
    }

    /// The reads logged so far: a want withheld at one count may earn its
    /// copy once it moves (`Directory::promotions`).
    pub fn reads(&self) -> u64 {
        self.reads
    }

    /// Take the (layer, expert)s whose membership of what the bank is asked
    /// to hold ([`Self::in_top`]) changed since last taken.
    pub fn take_changes(&mut self) -> BTreeSet<(usize, usize)> {
        std::mem::take(&mut self.changed)
    }

    /// `layer`'s `expert`'s depth in slots now: the experts of its group
    /// that outrank it, over the group's layers; its layer's experts where
    /// never read (deeper than any pool).
    pub fn depth(&self, layer: usize, expert: usize) -> usize {
        let experts = self.ranks[layer].len();
        if self.layers[layer].last_read(expert).is_none() {
            return experts;
        }
        let order = &self.order[self.groups[layer]];
        let place = (self.ranks[layer][expert], layer, expert);
        let above = order.len() - order.partition_point(|&placed| placed <= place);
        (above / self.members[self.groups[layer]]).min(experts)
    }

    /// The step that last read `layer`'s `expert`, if one has.
    pub fn last(&self, layer: usize, expert: usize) -> Option<Step> {
        self.layers[layer].last_read(expert)
    }

    /// Snapshot `(horizon, a_bar, epoch, front_rates)` once for `group` at
    /// `(tier, reader)` so per-record scans rank in O(1) without locking.
    pub fn ranker(&self, group: usize, tier: Tier, reader: Reader) -> Ranker<'_> {
        let (horizon, a_bar, epoch) = self.horizon_of(group, tier, reader);
        let rates = front_rates(&self.stands);
        Ranker { reuse: self, group, horizon, a_bar, epoch, rates }
    }

    /// `N` and `S` of `layer`'s `expert` at `tier`'s `τ` for `reader`
    /// ([`horizon::Times::horizon`]; `Unbounded` gives the limit order),
    /// certainly used at a read the log has yet to bring (`read`: a
    /// landing's step, or a running prompt group's), taken at lead 1 before
    /// the layer's first log.
    pub fn expected(&self, layer: usize, expert: usize, tier: Tier, reader: Reader, read: Option<Unlogged>) -> Expected {
        let group = self.groups[layer];
        let (horizon, a_bar, epoch) = self.horizon_of(group, tier, reader);
        let rates = front_rates(&self.stands);
        let override_lead = read.and_then(|read| self.lead_of(layer, read.step)).map(|lead| (lead, 1.0));
        expected_for(&self.layers[layer], &self.history, layer, expert, &rates, horizon, a_bar, epoch, override_lead)
    }

    /// [`Self::expected`] at `tier`'s move `τ` (`Reader::Move`: admission is
    /// a move) with step `lead`'s term `p_{g_s}` **replaced** by `p`, the
    /// ledger's probability of use at that step given every live name: the
    /// read ahead's candidate rank (design §3.7, finding 1). Never mixes `p`
    /// with `p_{g_s}`.
    pub fn expected_with(&self, layer: usize, expert: usize, tier: Tier, lead: Step, p: f64) -> Expected {
        let group = self.groups[layer];
        let (horizon, a_bar, epoch) = self.horizon_of(group, tier, Reader::Move);
        let rates = front_rates(&self.stands);
        let override_lead = self.lead_of(layer, lead).map(|g| (g, p));
        expected_for(&self.layers[layer], &self.history, layer, expert, &rates, horizon, a_bar, epoch, override_lead)
    }

    /// `layer`'s `expert`'s rank ([`Rank`]) at `tier`'s `τ` for `reader`,
    /// certainly used at a read the log has yet to bring (`read`: a
    /// landing's step, or a running prompt group's), and its last read.
    pub fn rank(&self, layer: usize, expert: usize, tier: Tier, reader: Reader, read: Option<Unlogged>) -> Rank {
        let expected = self.expected(layer, expert, tier, reader, read);
        let last = self.layers[layer].last_read(expert).max(read.map(|read| read.step));
        Rank::new(expected, last)
    }

    /// A candidate's rank from its expectation ([`Self::expected_with`]),
    /// beside its last read: what a victim test compares with the victim's
    /// [`Self::rank`].
    pub fn rank_of(&self, layer: usize, expert: usize, expected: Expected) -> Rank {
        Rank::new(expected, self.layers[layer].last_read(expert))
    }

    /// The ranks of a prompt group's seeds of `layer` at tier 0 for
    /// `Reader::Move` (a seed displaces), each `routed` entry `(expert, its
    /// count in the union, the step that last routed it)` in order: ranked by
    /// the `prompt` table with prior = its count share (`Rows::union`), at
    /// the limit order before any decode step (design §3.7 Seeds, §4).
    pub fn rank_union(&self, layer: usize, routed: &[(usize, u32, Step)]) -> Vec<Rank> {
        if routed.is_empty() {
            return Vec::new();
        }
        let group = self.groups[layer];
        let calibration = &self.layers[layer];
        // Once a decode step has been logged, prompt evidence has left the
        // ranking (§4): score candidates with the same decode `Ranker` that
        // `store_rank` uses for kept records, never mixing `Form::Prompt`
        // candidates against `Form::Decode` victims.
        if calibration.has_decode() {
            let ranker = self.ranker(group, Tier::COMPUTE, Reader::Move);
            return routed
                .iter()
                .map(|&(expert, _, last)| {
                    let expected = ranker.expected(layer, expert, None);
                    let last = calibration.last_read(expert).max(Some(last));
                    Rank::new(expected, last)
                })
                .collect();
        }
        let (horizon, a_bar, epoch) = self.horizon_of(group, Tier::COMPUTE, Reader::Move);
        let probs_all = calibration.prompt_probs(routed, epoch, &self.history, layer);
        routed
            .iter()
            .zip(probs_all)
            .map(|(&(expert, _, last), probs)| {
                let p_inf = calibration.history_prob(expert, &self.history, layer);
                let expected = expected_from(probs, p_inf, horizon, a_bar, None);
                let last = calibration.last_read(expert).max(Some(last));
                Rank::new(expected, last)
            })
            .collect()
    }

    /// Memory charge of the ranking: its per-layer calibration tables, history
    /// selector, sample rings, pools, order arrays, and the shadows' calibration
    /// (design §4 "Memory charge", bar 2). Constant in run length and `τ`.
    pub fn bytes(&self) -> u64 {
        let calibration: u64 = self.layers.iter().map(Calibration::bytes).sum();
        let experts: Vec<usize> = self.layers.iter().map(Calibration::experts).collect();
        let shadows = Shadows::calibration_bytes(&experts, &self.groups, &self.pools.records);
        let pools = self.pools.bytes();
        let order: usize = self.order.iter().map(|group| group.len() * std::mem::size_of::<Placed>()).sum();
        let ranks: usize = self.ranks.iter().map(|layer| layer.len() * std::mem::size_of::<Rank>()).sum();
        let top: usize = self.top.iter().map(|layer| layer.len() * std::mem::size_of::<bool>()).sum();
        calibration + self.history.bytes() + shadows + pools + (order + ranks + top) as u64
    }

    /// Begin logging reads for shadows (`Self::take_shadowed`) without
    /// allocating them yet: idempotent.
    pub(crate) fn enable_shadows(&mut self) {
        self.shadowed.get_or_insert_with(Vec::new);
    }

    /// Construct [`Shadows`] matching the pools' current shapes and resident
    /// contents so `U_shadow` starts warm beside the live bank instead of
    /// suffering a cold-start empty-pool miss cliff.
    pub(crate) fn build_shadows(&self) -> Shadows {
        let experts = self.layers.iter().map(Calibration::experts).collect::<Vec<_>>();
        Shadows::new_with_pools(&experts, &self.groups, &self.pools)
    }

    /// The shadows of its pools at every depth of foresight, as they stand
    /// ([`Shadows`]), which read what it logs from now
    /// ([`Self::take_shadowed`]).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn shadows(&mut self) -> Shadows {
        assert!(self.shadowed.is_none(), "one owner of a ranking's shadows");
        self.shadowed = Some(Vec::new());
        self.build_shadows()
    }

    /// What it logged for its shadows since this was last taken
    /// ([`Shadows::read_log`]); nothing before they are made.
    pub(crate) fn take_shadowed(&mut self) -> Vec<ShadowFeed> {
        self.shadowed.as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// Per-decile `(history (rate, se), far (rate, se))` aggregated across all
    /// streamed layers at each group's current epoch (§8 A1 diagnostic).
    pub(crate) fn far_deciles(&self) -> [(Option<(f64, f64)>, Option<(f64, f64)>); 10] {
        let epochs: Vec<u64> = {
            let times = self.time.lock();
            (0..self.members.len()).map(|group| times.group_epoch(group)).collect()
        };
        let mut total = [calibration::FarDecile::default(); 10];
        for (layer, cal) in self.layers.iter().enumerate() {
            if let Some(deciles) = cal.far_deciles(epochs[self.groups[layer]]) {
                for (acc, decile) in total.iter_mut().zip(deciles) {
                    acc.history_seen += decile.history_seen;
                    acc.history_used += decile.history_used;
                    acc.far_seen += decile.far_seen;
                    acc.far_used += decile.far_used;
                }
            }
        }
        total.map(|d| (d.history_rate().zip(d.history_se()), d.far_rate().zip(d.far_se())))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::experts::horizon::{Advance, Horizon as H, Time, Tokens};
    use crate::experts::interest::Form;

    fn time_for(records: &[usize]) -> Time {
        let tiers: Vec<Vec<u64>> = records.iter().map(|&r| vec![r as u64]).collect();
        Time::new(&tiers, 32)
    }

    fn read(sequence: Step, experts: &[usize]) -> RoutedRead {
        RoutedRead { sequence, layer: 0, experts: experts.to_vec(), choices: experts.len() as u64 }
    }

    fn register_front(time: &Time, step: Step) {
        time.lock().register(step, Advance::Known(Tokens::new(1.0)));
    }

    /// Log `read` into `reuse` after registering a front step on `time`, its keys unknown.
    fn log(reuse: &mut Reuse, time: &Time, r: RoutedRead) -> Vec<usize> {
        register_front(time, r.sequence);
        reuse.read(&r, None, Evidence::Decode)
    }

    /// Step `step`'s experts of a pattern that rewards frequency: expert 0
    /// every third step, beside a fresh one, `step`, at every step.
    pub(in crate::experts) fn every_third(step: Step) -> Vec<usize> {
        match step % 3 {
            0 => vec![0, step as usize],
            _ => vec![step as usize],
        }
    }

    /// One exact row's keys over `experts` experts ranking `order` first to
    /// last, the rest below them by index, higher first.
    fn keys(experts: usize, order: &[usize]) -> Vec<f32> {
        let mut keys = (0..experts).map(|expert| -(experts as f32) + expert as f32 * 1e-3).collect::<Vec<_>>();
        for (at, &expert) in order.iter().enumerate() {
            keys[expert] = (order.len() - at) as f32;
        }
        keys
    }

    /// `Rows::of` computes normalized priors (scaled to `choices`, clamped to 1)
    /// and merges rows of one depth by `1 − Π(1 − π)`.
    #[test]
    fn rows_prior_and_merge() {
        let mut row = keys(16, &[7, 3]);
        row[3] = row[7];
        row[0] = f32::NAN;
        let rows = Rows::of(2, &[0, 1, 1], &[row, keys(16, &[5, 9]), keys(16, &[9, 2])].concat());
        let (exact_priors, exact_chosen) = rows.at(0).expect("an exact row");
        assert!(exact_chosen[7] && exact_chosen[3] && !exact_chosen[1]);
        assert_eq!((exact_priors[7], exact_priors[3], exact_priors[0], exact_priors[1]), (1.0, 1.0, 0.0, 0.0));
        let (front_priors, front_chosen) = rows.at(1).expect("two front rows");
        assert!(front_chosen[9] && front_chosen[5] && front_chosen[2]);
        assert_eq!((front_priors[9], front_priors[5]), (1.0, 1.0));
        assert!((f64::from(front_priors[2]) - 2.0 / 3.0).abs() < 1e-6, "{}", front_priors[2]);
        assert_eq!((rows.deepest(), rows.at(2)), (1, None));
        let (probs, _) = Rows::of(2, &[0], &[0.5, 0.25, 0.125, 0.125]).at(0).expect("a row");
        assert_eq!(probs, [1.0, 0.5, 0.25, 0.25]);
    }

    /// Design §6: with M = 0, an observer (`Reader::Observe`) sees
    /// `Horizon::Unbounded` and ranks by `(p_∞, Σ_{g ≤ L}(p_g − p_∞), last)`.
    /// That order equals `N`'s order at `τ = 10^6` on the same state.
    #[test]
    fn an_unbounded_horizon_ranks_by_the_limit_order() {
        let time = time_for(&[4]);
        let mut reuse = Reuse::new(&[8], &[0], &[4], time.clone());
        // Build distinct histories (p_∞) and distinct route keys (p_g) with M = 0.
        for step in 1..=20 {
            register_front(&time, step);
            let chosen = match step % 4 {
                0 => vec![0, 1],
                1 => vec![0, 2],
                2 => vec![0, 3],
                _ => vec![1, 4],
            };
            let row = keys(8, &[chosen[0], chosen[1], 5, 6]);
            let r = RoutedRead { sequence: step, layer: 0, experts: chosen, choices: 2 };
            reuse.read(&r, Some(Arc::new(Rows::of(2, &[0], &row))), Evidence::Decode);
        }
        assert_eq!(time.lock().horizon(0, Tier::COMPUTE, Reader::Observe), Some(H::Unbounded));
        let rates = front_rates(&reuse.stands);
        let mut unbounded: Vec<(Rank, usize)> = (0..8)
            .map(|e| (reuse.rank(0, e, Tier::COMPUTE, Reader::Observe, None), e))
            .collect();
        unbounded.sort();

        let epoch = time.lock().group_epoch(0);
        let huge_tau = H::Tokens(Tokens::new(1e6));
        let mut at_huge_tau: Vec<(Rank, usize)> = (0..8)
            .map(|e| {
                let exp = expected_for(&reuse.layers[0], &reuse.history, 0, e, &rates, huge_tau, 1.0, epoch, None);
                (Rank::new(exp, reuse.last(0, e)), e)
            })
            .collect();
        at_huge_tau.sort();

        let order_unbounded: Vec<usize> = unbounded.iter().map(|&(_, e)| e).collect();
        let order_huge: Vec<usize> = at_huge_tau.iter().map(|&(_, e)| e).collect();
        assert_eq!(order_unbounded, order_huge, "unbounded limit order must match N's order at τ = 10^6");

        // Also check the exact (p_∞, Σ(p_g - p_∞)) formula directly.
        for e in 0..8 {
            let probs = lead_probs(&reuse.layers[0], &reuse.history, 0, e, &rates, epoch);
            let p_inf = reuse.layers[0].history_prob(e, &reuse.history, 0);
            let expected_limit = Expected::new(p_inf, probs.iter().map(|&p| p - p_inf).sum());
            assert_eq!(reuse.expected(0, e, Tier::COMPUTE, Reader::Observe, None), expected_limit);
        }
    }

    /// Design §6: equal `N` within `τ`, larger `S` (sooner use) ranks higher.
    #[test]
    fn the_sooner_use_breaks_a_tie_of_uses_within_tau() {
        let mut probs_soon = [0.0; DEPTHS];
        probs_soon[0] = 1.0; // used at lead 1
        let mut probs_later = [0.0; DEPTHS];
        probs_later[1] = 1.0; // used at lead 2

        let tau = H::Tokens(Tokens::new(2.0));
        let soon = expected_from(probs_soon, 0.0, tau, 1.0, None);
        let later = expected_from(probs_later, 0.0, tau, 1.0, None);
        assert_eq!(soon.uses(), 1.0);
        assert_eq!(later.uses(), 1.0);
        assert!(soon.sooner() > later.sooner(), "S(lead 1) = {} > S(lead 2) = {}", soon.sooner(), later.sooner());
        assert!(soon > later, "sooner use must outrank later use when N ties");
    }

    /// Design §6: an unlogged read is a certain use even before its layer is
    /// logged, giving a finite rank strictly above an unread expert's.
    #[test]
    fn an_unlogged_read_is_a_certain_use_before_its_layer_is_logged() {
        let time = time_for(&[4]);
        let reuse = Reuse::new(&[8], &[0], &[4], time.clone());
        // Before any decode step (Unbounded limit order):
        let unread_cold = reuse.rank(0, 1, Tier::COMPUTE, Reader::Move, None);
        let unlogged_cold = reuse.rank(0, 0, Tier::COMPUTE, Reader::Move, Some(Unlogged::at(1)));
        assert!(unlogged_cold > unread_cold, "unlogged before any decode step must outrank unread");

        // After first decode step is registered (finite Move τ), still before layer 0 logs:
        register_front(&time, 1);
        let unread = reuse.rank(0, 1, Tier::COMPUTE, Reader::Move, None);
        let unlogged = reuse.rank(0, 0, Tier::COMPUTE, Reader::Move, Some(Unlogged::at(1)));
        assert!(unlogged.expected().uses() > unread.expected().uses());
        assert!(unlogged > unread);
        assert_eq!(unlogged.last(), Some(1));
    }

    /// Design §6: no NaN or ∞ over cold start, resize to 1 record, and a
    /// prompt union; `Expected::new` panics on any non-finite value.
    #[test]
    fn every_rank_is_finite_from_cold_start() {
        assert!(std::panic::catch_unwind(|| Expected::new(f64::INFINITY, 0.0)).is_err());
        assert!(std::panic::catch_unwind(|| Expected::new(f64::NAN, 0.0)).is_err());
        assert!(std::panic::catch_unwind(|| Expected::new(0.0, f64::NEG_INFINITY)).is_err());
        assert!(std::panic::catch_unwind(|| Expected::new(0.0, f64::NAN)).is_err());

        let time = time_for(&[4]);
        let mut reuse = Reuse::new(&[8], &[0], &[4], time.clone());
        let check_all = |reuse: &Reuse| {
            for e in 0..8 {
                for reader in [Reader::Move, Reader::Observe] {
                    let exp = reuse.expected(0, e, Tier::COMPUTE, reader, None);
                    assert!(exp.uses().is_finite() && exp.sooner().is_finite());
                    let exp_u = reuse.expected(0, e, Tier::COMPUTE, reader, Some(Unlogged::at(10)));
                    assert!(exp_u.uses().is_finite() && exp_u.sooner().is_finite());
                }
            }
        };
        // Cold start (before any decode step).
        check_all(&reuse);
        // Prompt union before any decode step.
        let seeded = reuse.rank_union(0, &[(0, 3, 1), (1, 1, 1)]);
        assert_eq!(seeded.len(), 2);
        assert!(seeded[0] > seeded[1]);
        reuse.read(&read(1, &[0, 1]), Some(Arc::new(Rows::union(8, 2, &[(0, 3), (1, 1)]))), Evidence::Prompt);
        check_all(&reuse);
        // First decode step.
        log(&mut reuse, &time, read(2, &[0, 2]));
        check_all(&reuse);
        // Resize to 1 record.
        time.lock().resized(0, Tier::COMPUTE, 1);
        reuse.resize(0, 1);
        check_all(&reuse);
    }

    /// Design §6: memory size of calibration & ranking is constant in run
    /// length (equal after 10 steps and after 10,000 steps).
    #[test]
    fn the_calibrations_size_is_set_by_experts_and_depths() {
        let time = time_for(&[4]);
        let mut reuse = Reuse::new(&[16], &[0], &[4], time.clone());
        let mut bytes_at_10 = 0;
        for step in 1..=10_000 {
            register_front(&time, step);
            let e0 = (step as usize) % 16;
            let e1 = (step as usize * 3 + 1) % 16;
            let mut chosen = vec![e0, e1];
            chosen.sort_unstable();
            chosen.dedup();
            let r = RoutedRead { sequence: step, layer: 0, experts: chosen, choices: 2 };
            reuse.read(&r, None, Evidence::Decode);
            if step % 5 == 0 {
                time.lock().moved(0, Tier::COMPUTE);
            }
            if step == 10 {
                bytes_at_10 = reuse.bytes();
            }
        }
        let bytes_at_10k = reuse.bytes();
        assert!(bytes_at_10 > 0);
        assert_eq!(bytes_at_10, bytes_at_10k, "size must be constant over 10,000 steps");
    }

    /// Design §6: `p_g` decomposes over the first fallen row `f` using the
    /// `fell` table (counted only on steps whose row fell), and the branch
    /// weights sum to 1.
    #[test]
    fn a_fallen_row_reads_the_fell_table() {
        let time = time_for(&[4]);
        let mut reuse = Reuse::new(&[8], &[0], &[4], time.clone());
        let (_, stands) = reuse.rates();
        // Set r_1 = 0.5 (1 stood, 1 fell) and r_2 = 0.5 (1 stood, 1 fell).
        stands.count(1, Form::Front, true);
        stands.count(1, Form::Front, false);
        stands.count(2, Form::Front, true);
        stands.count(2, Form::Front, false);

        // Log steps with front rows depth 1 and 2; mark step 1's row 1 as fallen.
        for step in 1..=10 {
            register_front(&time, step);
            let exact = keys(8, &[0]);
            let front1 = keys(8, &[1]);
            let front2 = keys(8, &[2]);
            let rows = Rows::of(1, &[0, 1, 2], &[exact, front1, front2].concat());
            reuse.read(&read(step, &[0]), Some(Arc::new(rows)), Evidence::Decode);
            if step == 1 {
                reuse.fell(0, 1, 1);
            }
        }

        let rates = front_rates(&reuse.stands);
        assert!((rates[0] - 0.5).abs() < 1e-12);
        assert!((rates[1] - 0.5).abs() < 1e-12);
        // Check weights sum to 1: w_1 = 0.5, w_2 = 0.25, sigma_2 = 0.25.
        let w1 = 1.0 - rates[0];
        let w2 = rates[0] * (1.0 - rates[1]);
        let s2 = rates[0] * rates[1];
        assert!((w1 + w2 + s2 - 1.0).abs() < 1e-12);

        // Verify lead_probs matches the exact first-fall decomposition formula for g = 1, 2, 3.
        let epoch = time.lock().group_epoch(0);
        let cal = &reuse.layers[0];
        let form = cal.latest_form().unwrap_or(calibration::Form::Front);
        for e in 0..8 {
            let probs = lead_probs(cal, &reuse.history, 0, e, &rates, epoch);
            let p1 = w1 * cal.fell_prob(1, 0, e, epoch, &reuse.history, 0) + rates[0] * f64::from(u8::from(cal.row_chose(1, e)));
            let p2 = w1 * cal.fell_prob(2, 0, e, epoch, &reuse.history, 0) + w2 * cal.fell_prob(1, 1, e, epoch, &reuse.history, 0) + s2 * f64::from(u8::from(cal.row_chose(2, e)));
            let p3 = w1 * cal.fell_prob(3, 0, e, epoch, &reuse.history, 0) + w2 * cal.fell_prob(2, 1, e, epoch, &reuse.history, 0) + s2 * cal.route_prob(form, 1, 2, e, epoch, &reuse.history, 0);
            assert!((probs[0] - p1.clamp(0.0, 1.0)).abs() < 1e-12);
            assert!((probs[1] - p2.clamp(0.0, 1.0)).abs() < 1e-12);
            assert!((probs[2] - p3.clamp(0.0, 1.0)).abs() < 1e-12);
        }
    }

    /// Design §6: history prior is `k/E` while `S_W = 0` and `ā = 1` at the
    /// first decode step; no 0/0.
    #[test]
    fn cold_start_priors_are_finite() {
        let time = time_for(&[4]);
        let mut reuse = Reuse::new(&[8], &[0], &[4], time.clone());
        // Prompt read sets k = 2 while decode step count S_W = 0.
        reuse.read(&RoutedRead { sequence: 1, layer: 0, experts: vec![0, 1], choices: 2 }, None, Evidence::Prompt);
        for e in 0..8 {
            let p_inf = reuse.layers[0].history_prob(e, &reuse.history, 0);
            assert!(p_inf.is_finite(), "cold-start history_prob must be finite");
            assert!((p_inf - 2.0 / 8.0).abs() < 1e-12, "expected k/E = 0.25, got {p_inf}");
        }
        // First decode step: ā = 1.0.
        register_front(&time, 2);
        assert_eq!(time.lock().tokens_per_step(0), Some(1.0));
        for e in 0..8 {
            let exp = reuse.expected(0, e, Tier::COMPUTE, Reader::Move, None);
            assert!(exp.uses().is_finite() && exp.sooner().is_finite());
        }
    }

    /// Before any count, a just-read expert never ranks below one never
    /// read; a read the log has passed adds nothing, one past it is a use at
    /// its step.
    #[test]
    fn a_just_read_expert_never_ranks_below_one_never_read() {
        let time = time_for(&[2]);
        let mut reuse = Reuse::new(&[3], &[0], &[2], time.clone());
        assert_eq!(log(&mut reuse, &time, read(4, &[0])), [3], "a first read lies past every expert");
        assert!((1..3).all(|expert| reuse.rank(0, 0, Tier::COMPUTE, Reader::Move, None) > reuse.rank(0, expert, Tier::COMPUTE, Reader::Move, None)));
        assert_eq!(reuse.rank(0, 0, Tier::COMPUTE, Reader::Move, Some(Unlogged::at(3))), reuse.rank(0, 0, Tier::COMPUTE, Reader::Move, None));
        assert!(reuse.rank(0, 2, Tier::COMPUTE, Reader::Move, Some(Unlogged::at(7))) > reuse.rank(0, 2, Tier::COMPUTE, Reader::Move, None));
        assert_eq!(reuse.rank(0, 2, Tier::COMPUTE, Reader::Move, Some(Unlogged::at(7))).last(), Some(7));
        assert_eq!(reuse.last(0, 0), Some(4));

        let time_keyed = time_for(&[2]);
        let mut keyed = Reuse::new(&[4], &[0], &[2], time_keyed.clone());
        register_front(&time_keyed, 1);
        let rows = Rows::of(1, &[0], &[0.26, 0.25, 0.25, 0.24]);
        keyed.read(&read(1, &[0]), Some(Arc::new(rows)), Evidence::Decode);
        assert!((1..4).all(|expert| keyed.rank(0, 0, Tier::COMPUTE, Reader::Move, None) > keyed.rank(0, expert, Tier::COMPUTE, Reader::Move, None)));
    }

    /// A group's pool holds what its layers' reads are expected to use most
    /// wherever their layer: a layer whose four experts recur every step
    /// keeps them while another layer's one-off reads take the rest.
    #[test]
    fn a_pool_allocates_across_its_layers_by_expected_uses() {
        let time = time_for(&[6]);
        let mut reuse = Reuse::new(&[64, 64], &[0, 0], &[6], time.clone());
        for step in 1..=40 {
            register_front(&time, step);
            let hot = RoutedRead { sequence: step, layer: 0, experts: vec![0, 1, 2, 3], choices: 4 };
            let cold = RoutedRead { sequence: step, layer: 1, experts: vec![step as usize], choices: 1 };
            reuse.read(&hot, None, Evidence::Decode);
            reuse.read(&cold, None, Evidence::Decode);
            time.lock().moved(0, Tier::COMPUTE);
        }
        let held = |layer: usize| (0..64).filter(|&expert| reuse.pools.pos[layer][expert] != NOT_HELD).count();
        assert_eq!((held(0), held(1)), (4, 2), "the recurring layer holds its four");
        assert!((0..4).all(|expert| reuse.in_top(0, expert)), "and the bank is asked to hold them");
        assert!((0..4).all(|expert| reuse.depth(0, expert) <= 1), "three experts at most outrank a hot one: a slot of two layers");
        assert_eq!(reuse.depth(1, 63), 64, "an expert never read lies deeper than any pool");
    }

    /// The bank is asked to hold what a front row foresees before any step
    /// reads it: the group's most expected experts, read or not.
    #[test]
    fn the_bank_is_asked_to_hold_what_a_front_row_foresees() {
        let time = time_for(&[2]);
        let mut reuse = Reuse::new(&[16], &[0], &[2], time.clone());
        let (_, stands) = reuse.rates();
        stands.count(1, Form::Front, true);
        stands.count(1, Form::Front, false);
        let rows = |front: usize| Arc::new(Rows::of(1, &[0, 1], &[keys(16, &[0]), keys(16, &[front])].concat()));
        for step in 1..=20 {
            register_front(&time, step);
            reuse.read(&read(step, &[0]), Some(rows(1)), Evidence::Decode);
            time.lock().moved(0, Tier::COMPUTE);
        }
        assert!(reuse.in_top(0, 0) && reuse.in_top(0, 1), "the chosen and the foreseen");
        reuse.take_changes();
        register_front(&time, 21);
        reuse.read(&read(21, &[0]), Some(rows(9)), Evidence::Decode);
        assert!(reuse.in_top(0, 9) && !reuse.in_top(0, 1), "the foresight moved");
        assert_eq!(reuse.take_changes(), BTreeSet::from([(0, 1), (0, 9)]));
        assert_eq!(reuse.pools.pos[0][9], NOT_HELD, "the pool of what is read admits only reads");
    }

    /// A pool resized smaller sheds its least ranked from what is read and
    /// from what the bank is asked to hold, noted as changes.
    #[test]
    fn a_resize_notes_its_changes() {
        let time = time_for(&[4]);
        let mut reuse = Reuse::new(&[8], &[0], &[4], time.clone());
        for step in 1..=4 {
            log(&mut reuse, &time, read(step, &[step as usize]));
        }
        assert_eq!((0..8).filter(|&expert| reuse.in_top(0, expert)).count(), 4);
        reuse.take_changes();
        time.lock().resized(0, Tier::COMPUTE, 2);
        reuse.resize(0, 2);
        let top = (0..8).filter(|&expert| reuse.in_top(0, expert)).collect::<BTreeSet<_>>();
        assert_eq!(top.len(), 2);
        assert_eq!(reuse.take_changes().len(), 2, "two left");
        assert_eq!((0..8).filter(|&expert| reuse.pools.pos[0][expert] != NOT_HELD).count(), 2);
    }

    /// B-2 / B-3: `Reuse::top_candidates` returns the highest-expected-use
    /// experts matching `accept`, in descending rank order, skipping zero-use
    /// experts.
    #[test]
    fn top_candidates_orders_accepted_lacked_experts_by_expected_uses() {
        let time = time_for(&[4]);
        let mut reuse = Reuse::new(&[8], &[0], &[4], time.clone());
        for step in 1..=4 {
            log(&mut reuse, &time, read(step, &[1, 3]));
        }
        log(&mut reuse, &time, read(5, &[3, 5]));
        // Expert 3 is already resident (rejected by `accept`): top 2 lacked must
        // be the carried-over experts 5 and 1 (strictly ahead of never-seen
        // experts' prior), in descending rank order, never returning 3.
        let top = reuse.top_candidates(2, |_layer, expert| expert != 3);
        let experts: Vec<usize> = top.iter().map(|&(_, _, expert)| expert).collect();
        assert_eq!(experts, vec![5, 1], "must return top lacked experts in descending rank order: {top:?}");
        assert!(top[0].0 >= top[1].0, "must be sorted descending by Rank: {top:?}");
        assert!(top.iter().all(|&(rank, _, _)| rank.expected() > Expected::ZERO));
    }
}
