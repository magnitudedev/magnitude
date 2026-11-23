//! Adaptive draft width: each round's generated and verified proposal counts.
//!
//! A round drafts `g` proposals and verifies the first `v` of them. The
//! controller takes the pair that commits the most tokens per second by two
//! measured tables: how often a block is right through each depth, and what
//! each launch costs.
//!
//! A drafted token is scored against the tokens the request goes on to
//! commit, whether or not it was verified, so every depth up to `g` is
//! observed by ordinary rounds. Deeper ones are observed in two ways. A width
//! the text stops supporting is still drafted for a while, so its return is
//! seen at once. And a look drafts wider than the fastest pair does: an
//! ordinary round that is slower by a known amount, taken as often as a
//! fixed share of the decode time pays for that amount. A depth no round
//! scores, as while no round drafts, is expected to be right as often as the
//! load's blocks usually are once what it last scored is old
//! ([`DepthMeans`]).
//!
//! The plain decode is the reference every drafting width is compared
//! against, and while a load drafts no ordinary round measures it. Where it
//! is measured, it is taken again while its price is unproven or stale.
//! Where it is not, a plain look takes it once no price it costs at least
//! lasts any longer.
//!
//! A round verifies its drafted block by the drafter's confidence in each
//! proposal where the drafter gives one: how often a proposal at a depth is
//! right after the ones before it divides by that confidence
//! ([`Calibration`]), so the round's rows go to the proposals worth them.
//!
//! Requests verified in one launch share its rows ([`share_rows`]).

use magnitude_cost_model::LaunchPrices;
use magnitude_executor::{draft_ladder, LaunchDuration, StreamedVerifyPrior, TokenId};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

/// The weight of a new block in a depth's mean over the current text once
/// the depth has had its share of blocks. A depth scored every round follows
/// the text within a few rounds; one scored only by looks moves as slowly as
/// they come.
const MEMORY: f64 = 0.25;
/// The weight of a new block in a depth's usual acceptance, the mean over
/// the load's text, once the depth has had its share of blocks.
const USUAL: f64 = 1.0 / 32.0;
/// The latest durations a launch's cost is the median of. A launch another
/// process held up once does not move the median, so it reprices nothing:
/// a launch priced by such a duration and then no longer taken would keep
/// that price, and the launches measured after it would look cheaper.
const COST_SAMPLES: usize = 3;
/// Maximum `(context_class, sequences)` cost buckets retained in `DraftLoad`
/// across engine lifetime; least-recently-measured buckets are evicted past
/// this bound.
const MAX_COST_BUCKETS: usize = 64;
/// The rounds of a load after which a launch's durations are stale: a
/// launch measured again after them is priced by its new durations alone,
/// and a plain decode not taken for that long is taken again, so a price
/// left by a busier or freer machine lasts no longer.
const STALE_ROUNDS: u64 = 100;
/// Minimum rounds before an older streamed verify width that is >1.5x slower
/// than what a more recent, warmed-up narrower width predicts expires on read,
/// preventing cold-start residency timings from locking out `K = 7` (`M = 8`).
const WARMUP_DRIFT_ROUNDS: u64 = 16;
/// The share of decode time spent on looks: wider and plain, each scaled
/// down where little can be gained.
const LOOK_SHARE: f64 = 0.01;
/// The equal shares of [0, 1] a depth's calibration divides the drafter's
/// confidence into.
const BINS: usize = 8;
/// The weight of a new proposal in its depth's calibration: each bin holds
/// about as many proposals as a depth's usual mean holds blocks.
const CALIBRATION: f64 = USUAL / BINS as f64;

/// Which launch of a round a duration belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimedLaunch {
    /// The drafter's launch that proposes.
    Draft,
    /// A drafter launch that only enters pairs: the catch-up of a round
    /// drafting after rounds that did not, or the entry of held pairs past
    /// their bound. It is part of the round it serves.
    Entry,
    /// The target's launch.
    Target,
}

/// A round that changes between drafting and not: one that drafts after a
/// round that did not discards the decode queued behind that round and
/// first enters the pairs held since, and one that does not draft after a
/// round that did starts the decode pipeline again. Where a round drafting
/// nothing publishes no feature (a drafter whose feature costs the step),
/// a round cannot draft after one that did not: the round choosing to
/// draft drafts nothing and has its queued decode publish its feature; that
/// decode is the next round, which prepares, discarding nothing, and the
/// round after drafts from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Switch {
    Enter,
    Leave,
}

/// A launch's latest durations, oldest first, stored inline (zero heap allocs).
#[derive(Clone, Copy, Debug)]
struct Durations {
    samples: [Measured; COST_SAMPLES],
    len: u8,
}

impl Default for Durations {
    fn default() -> Self {
        Self {
            samples: [Measured { round: 0, seconds: 0.0, covering: false }; COST_SAMPLES],
            len: 0,
        }
    }
}

impl PartialEq for Durations {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

/// One duration of a launch: its seconds, the load's round it was measured
/// in, and whether the expert bank's cover read beside it
/// (`LaunchDuration::covering`), which prices the launch only until the
/// cover reads nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Measured {
    round: u64,
    seconds: f64,
    covering: bool,
}

impl Durations {
    fn as_slice(&self) -> &[Measured] {
        &self.samples[..usize::from(self.len)]
    }

    fn retain(&mut self, mut keep: impl FnMut(&Measured) -> bool) {
        let mut write = 0usize;
        for read in 0..usize::from(self.len) {
            if keep(&self.samples[read]) {
                self.samples[write] = self.samples[read];
                write += 1;
            }
        }
        self.len = write as u8;
    }

    fn record(&mut self, round: u64, seconds: f64, covering: bool) {
        self.retain(|measured| round.saturating_sub(measured.round) < STALE_ROUNDS);
        let len = usize::from(self.len);
        if len == COST_SAMPLES {
            self.samples.copy_within(1..COST_SAMPLES, 0);
            self.samples[COST_SAMPLES - 1] = Measured { round, seconds, covering };
        } else {
            self.samples[len] = Measured { round, seconds, covering };
            self.len = (len + 1) as u8;
        }
    }

    /// The launch's cost: the median of its durations, the lower of two.
    fn cost(&self) -> f64 {
        let n = usize::from(self.len);
        let mut sorted = [0.0f64; COST_SAMPLES];
        for (dst, src) in sorted[..n].iter_mut().zip(self.as_slice()) {
            *dst = src.seconds;
        }
        sorted[..n].sort_by(f64::total_cmp);
        sorted[(n - 1) / 2]
    }

    /// `(round, seconds)` durations measured with no cover read beside them.
    #[cfg(test)]
    fn settled(durations: &[(u64, f64)]) -> Self {
        assert!(durations.len() <= COST_SAMPLES, "at most COST_SAMPLES durations");
        let mut out = Self::default();
        for &(round, seconds) in durations {
            out.samples[usize::from(out.len)] = Measured { round, seconds, covering: false };
            out.len += 1;
        }
        out
    }

    /// Drop the durations measured beside a cover read, which has ended.
    fn settle(&mut self) {
        self.retain(|measured| !measured.covering);
    }

    fn measured(&self) -> bool {
        self.len > 0
    }

    /// Whether the latest measurement is still within [`STALE_ROUNDS`] of `round`.
    fn is_fresh(&self, round: u64) -> bool {
        self.len > 0 && round.saturating_sub(self.latest()) < STALE_ROUNDS
    }

    /// Whether the cost rests on its full share of durations.
    fn proven(&self) -> bool {
        usize::from(self.len) == COST_SAMPLES
    }

    fn latest(&self) -> u64 {
        self.as_slice().last().map_or(0, |measured| measured.round)
    }

    fn scale(&mut self, factor: f64) {
        for measured in &mut self.samples[..usize::from(self.len)] {
            measured.seconds *= factor;
        }
    }
}

/// How often a block is right through one depth, as two means of its scored
/// blocks: over the current text, which follows a change within a few
/// blocks, and over the load's text, which the current text departs from and
/// returns to. The current mean also moves with the noise of its few blocks,
/// so how far it departs from the usual one counts only as far as the
/// departures measured over the load exceed that noise: on text that does not
/// change, a run of wrong blocks moves no estimate, and the estimate stands
/// on the many blocks of the usual mean.
#[derive(Clone, Debug, PartialEq)]
struct DepthMeans {
    current: f64,
    usual: f64,
    spread: f64,
    blocks: u32,
}

impl DepthMeans {
    /// A depth no block has scored: every proposal taken as right.
    const UNSCORED: Self = Self {
        current: 1.0,
        usual: 1.0,
        spread: 0.0,
        blocks: 0,
    };

    /// Both means are plain until the depth has had their share of blocks,
    /// so a load's first block replaces what it assumed.
    fn score(&mut self, right: bool) {
        self.blocks = self.blocks.saturating_add(1);
        let weight = |memory: f64| (1.0 / f64::from(self.blocks)).max(memory);
        let right = f64::from(u8::from(right));
        self.current += weight(MEMORY) * (right - self.current);
        self.usual += weight(USUAL) * (right - self.usual);
        let departure = self.current - self.usual;
        self.spread += weight(USUAL) * (departure * departure - self.spread);
    }

    /// A round that scored no block of the depth. The current text's mean
    /// returns to the usual one as the usual mean forgets a block: a text
    /// the depth has not seen for as many rounds as the usual mean holds
    /// blocks is no better described by what it last scored than by the
    /// usual. A load that stopped drafting on a stretch of wrong blocks thus
    /// expects the load's acceptance again once the stretch is that old,
    /// whether or not a look has come since.
    fn unscored(&mut self) {
        self.current += USUAL * (self.usual - self.current);
    }

    /// The share of the current mean's departure from the usual one that is
    /// the text's: the departures' measured mean square beyond what the
    /// binomial noise of the current mean's blocks alone gives.
    fn text(&self) -> f64 {
        let noise = self.usual * (1.0 - self.usual) * MEMORY / (2.0 - MEMORY);
        if self.spread > noise {
            1.0 - noise / self.spread
        } else {
            0.0
        }
    }

    fn estimate(&self) -> f64 {
        self.usual + self.text() * (self.current - self.usual)
    }

    /// The weight a new block has in the estimate.
    fn weight(&self) -> f64 {
        let text = self.text();
        text * MEMORY + (1.0 - text) * USUAL
    }
}

/// How often a proposal that reached one depth (every proposal before it
/// right) is right, by the drafter's confidence in it, over the load's text.
/// It divides the depth's acceptance, which its [`DepthMeans`] give for the
/// current text, among confidences: a proposal of a confidence is right that
/// many times as often as the depth's proposals are.
#[derive(Clone, Debug, PartialEq)]
struct Calibration {
    /// Per bin, the weight of its proposals that were right and of all of
    /// them.
    bins: [(f64, f64); BINS],
}

impl Calibration {
    const EMPTY: Self = Self {
        bins: [(0.0, 0.0); BINS],
    };

    fn bin(confidence: f32) -> usize {
        // A NaN confidence falls in the lowest bin.
        ((confidence.clamp(0.0, 1.0) * BINS as f32) as usize).min(BINS - 1)
    }

    fn score(&mut self, confidence: f32, right: bool) {
        for (right, reached) in &mut self.bins {
            *right *= 1.0 - CALIBRATION;
            *reached *= 1.0 - CALIBRATION;
        }
        let (rights, reached) = &mut self.bins[Self::bin(confidence)];
        *rights += f64::from(u8::from(right));
        *reached += 1.0;
    }

    /// How many times as often as the depth's proposals one of `confidence`
    /// is right. Each mean counts one proposal more at the mean around it
    /// (an even chance for the depth's), so a bin that has seen nothing is
    /// right as often as the depth's proposals are.
    fn lift(&self, confidence: f32) -> f64 {
        let (right, reached) = self.bins.iter().fold((0.0, 0.0), |(right, reached), bin| {
            (right + bin.0, reached + bin.1)
        });
        let depth = (right + 0.5) / (reached + 1.0);
        let (right, reached) = self.bins[Self::bin(confidence)];
        (right + depth) / (reached + 1.0) / depth
    }
}

/// Measured launch costs at one context class and sequence count.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LaunchCosts {
    /// By generated proposals.
    draft: BTreeMap<usize, Durations>,
    /// By target rows.
    verify: BTreeMap<usize, Durations>,
    enter: Durations,
    leave: Durations,
}

impl LaunchCosts {
    fn record(costs: &mut BTreeMap<usize, Durations>, round: u64, size: usize, seconds: f64, covering: bool) {
        costs.entry(size).or_default().record(round, seconds, covering);
    }

    /// The most the measurements prove a launch of `size` costs at least: a
    /// larger launch costs no less than a smaller one. A larger launch
    /// proves nothing about a smaller one (a 16-row launch costs more than
    /// twice an 8-row one where the rows cross a class).
    fn at_least(costs: &BTreeMap<usize, Durations>, size: usize) -> f64 {
        costs
            .range(..=size)
            .map(|(_, durations)| durations.cost())
            .fold(0.0, f64::max)
    }
}

/// Measured target launch costs at one `(regime_epoch, context_class, sequences)`:
/// keyed to the target expert bank's physical `regime_epoch` so target durations
/// measured under a prior tier/residency regime are inapplicable by construction.
#[derive(Clone, Debug, Default, PartialEq)]
struct TargetCosts {
    /// By target rows.
    verify: BTreeMap<usize, Durations>,
    enter: Durations,
    leave: Durations,
}

impl TargetCosts {
    fn settle(&mut self) {
        self.verify.retain(|_, durations| {
            durations.settle();
            durations.measured()
        });
        self.enter.settle();
        self.leave.settle();
    }

    fn record_plain(&mut self, round: u64, seconds: f64, covering: bool) -> Option<f64> {
        let before = self.verify.get(&1).map(Durations::cost);
        LaunchCosts::record(&mut self.verify, round, 1, seconds, covering);
        let before = before.filter(|before| *before > 0.0)?;
        let factor = self.verify[&1].cost() / before;
        for (_, durations) in self.verify.iter_mut().filter(|(rows, _)| **rows != 1) {
            durations.scale(factor);
        }
        self.enter.scale(factor);
        self.leave.scale(factor);
        Some(factor)
    }

    fn switch(&self, switch: Switch) -> &Durations {
        match switch {
            Switch::Enter => &self.enter,
            Switch::Leave => &self.leave,
        }
    }

    fn switch_mut(&mut self, switch: Switch) -> &mut Durations {
        match switch {
            Switch::Enter => &mut self.enter,
            Switch::Leave => &mut self.leave,
        }
    }

    fn is_empty(&self) -> bool {
        self.verify.is_empty() && !self.enter.measured() && !self.leave.measured()
    }

    fn latest(&self) -> u64 {
        self.verify
            .values()
            .chain([&self.enter, &self.leave])
            .map(Durations::latest)
            .max()
            .unwrap_or(0)
    }
}

/// Evict the least-recently-updated `(context_class, sequences)` draft buckets
/// whenever `costs` exceeds [`MAX_COST_BUCKETS`], never evicting `keep`.
fn prune_draft_costs(
    costs: &mut BTreeMap<(usize, usize), BTreeMap<usize, Durations>>,
    keep: (usize, usize),
) {
    while costs.len() > MAX_COST_BUCKETS {
        let Some(victim) = costs
            .iter()
            .filter(|&(&key, _)| key != keep)
            .min_by_key(|(_, bucket)| bucket.values().map(Durations::latest).max().unwrap_or(0))
            .map(|(&key, _)| key)
        else {
            break;
        };
        costs.remove(&victim);
    }
}

/// Evict the least-recently-updated target buckets whenever `costs` exceeds
/// [`MAX_COST_BUCKETS`], never evicting `keep`.
fn prune_target_costs(
    costs: &mut BTreeMap<(Option<u64>, usize, usize), TargetCosts>,
    keep: (Option<u64>, usize, usize),
) {
    while costs.len() > MAX_COST_BUCKETS {
        let Some(victim) = costs
            .iter()
            .filter(|&(&key, _)| key != keep)
            .min_by_key(|(_, bucket)| bucket.latest())
            .map(|(&key, _)| key)
        else {
            break;
        };
        costs.remove(&victim);
    }
}

/// The launch costs of one physical regime, context class, and number of
/// requests sharing a launch. Borrows `DraftLoadState`'s draft and target cost
/// tables under the load's lock with zero cloning.
struct RoundCosts<'a> {
    draft_costs: &'a BTreeMap<(usize, usize), BTreeMap<usize, Durations>>,
    target_costs: &'a BTreeMap<(Option<u64>, usize, usize), TargetCosts>,
    verify_prior: Option<&'a StreamedVerifyPrior>,
    launch_prices: Option<&'a LaunchPrices>,
    regime_epoch: Option<u64>,
    round: u64,
    context_class: usize,
    /// Requests sharing each launch.
    sequences: usize,
    after_drafting: Option<bool>,
    switch_weight: f64,
}

impl<'a> RoundCosts<'a> {
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        draft_costs: &'a BTreeMap<(usize, usize), BTreeMap<usize, Durations>>,
        target_costs: &'a BTreeMap<(Option<u64>, usize, usize), TargetCosts>,
        verify_prior: Option<&'a StreamedVerifyPrior>,
        launch_prices: Option<&'a LaunchPrices>,
        round: u64,
        context_class: usize,
        sequences: usize,
        after_drafting: Option<bool>,
    ) -> Self {
        let regime_epoch = verify_prior.map(|p| p.regime_epoch);
        Self {
            draft_costs,
            target_costs,
            verify_prior,
            launch_prices,
            regime_epoch,
            round,
            context_class,
            sequences,
            after_drafting,
            switch_weight: 0.0,
        }
    }

    fn of(
        state: &'a DraftLoadState,
        context_class: usize,
        sequences: usize,
        after_drafting: Option<bool>,
    ) -> Self {
        Self::from_parts(
            &state.draft_costs,
            &state.target_costs,
            state.verify_prior.as_ref(),
            state.launch_prices.as_ref(),
            state.rounds,
            context_class,
            sequences,
            after_drafting,
        )
    }

    fn own_draft(&self) -> Option<&BTreeMap<usize, Durations>> {
        self.draft_costs.get(&(self.context_class, self.sequences))
    }

    fn own_target(&self) -> Option<&TargetCosts> {
        self.target_costs.get(&(self.regime_epoch, self.context_class, self.sequences))
    }

    /// Draft measurements at no longer a context and of no more requests.
    fn lesser_draft(&self) -> impl Iterator<Item = &BTreeMap<usize, Durations>> + '_ {
        let (context_class, sequences) = (self.context_class, self.sequences);
        self.draft_costs
            .range(..=(context_class, sequences))
            .filter(move |&(&(context, shared), _)| {
                context <= context_class
                    && shared <= sequences
                    && (context, shared) != (context_class, sequences)
            })
            .map(|(_, costs)| costs)
    }

    /// Target measurements in the current regime at no longer a context and of
    /// no more requests.
    fn lesser_target(&self) -> impl Iterator<Item = &TargetCosts> + '_ {
        let (regime, context_class, sequences) = (self.regime_epoch, self.context_class, self.sequences);
        self.target_costs
            .range((regime, 0, 0)..=(regime, context_class, sequences))
            .filter(move |&(&(r, context, shared), _)| {
                r == regime
                    && context <= context_class
                    && shared <= sequences
                    && (context, shared) != (context_class, sequences)
            })
            .map(|(_, costs)| costs)
    }

    /// Draft measurements of as many requests at a shorter context, with how
    /// many times longer this context class is.
    fn shorter_draft(&self) -> impl Iterator<Item = (f64, &BTreeMap<usize, Durations>)> + '_ {
        let (context_class, sequences) = (self.context_class, self.sequences);
        self.draft_costs
            .range(..(context_class, 0))
            .filter(move |&(&(context, shared), _)| context < context_class && shared == sequences)
            .map(move |(&(context, _), costs)| (context_class as f64 / context as f64, costs))
    }

    /// Target measurements in the current regime of as many requests at a
    /// shorter context, with how many times longer this context class is.
    fn shorter_target(&self) -> impl Iterator<Item = (f64, &TargetCosts)> + '_ {
        let (regime, context_class, sequences) = (self.regime_epoch, self.context_class, self.sequences);
        self.target_costs
            .range((regime, 0, 0)..(regime, context_class, 0))
            .filter(move |&(&(r, context, shared), _)| r == regime && context < context_class && shared == sequences)
            .map(move |(&(_, context, _), costs)| (context_class as f64 / context as f64, costs))
    }

    /// These costs with a round that would switch paying `weight` of the
    /// switch (see [`Self::round`]).
    fn sharing_switches(self, weight: f64) -> Self {
        Self {
            switch_weight: weight,
            ..self
        }
    }

    fn draft_seconds_from_table(&self, size: usize) -> f64 {
        let own = self.own_draft();
        let unproven = match own.and_then(|own| own.get(&size)) {
            Some(durations) if durations.proven() => return durations.cost(),
            Some(durations) => durations.cost(),
            None => own
                .into_iter()
                .map(|own| LaunchCosts::at_least(own, size))
                .chain(self.lesser_draft().map(|measured| LaunchCosts::at_least(measured, size)))
                .fold(0.0, f64::max),
        };
        let larger = own
            .into_iter()
            .flat_map(|own| own.range(size + 1..))
            .filter(|(_, larger)| larger.proven())
            .map(|(_, larger)| larger.cost());
        let shorter = self.shorter_draft().filter_map(|(ratio, measured)| {
            measured
                .get(&size)
                .filter(|durations| durations.proven())
                .map(|durations| ratio * durations.cost())
        });
        larger.chain(shorter).fold(unproven, f64::min)
    }

    fn verify_seconds_from_table(&self, size: usize) -> f64 {
        let own = self.own_target().map(|t| &t.verify);
        let is_non_plain = size > self.sequences.max(1);
        let fresh_or_plain = |durations: &Durations| !is_non_plain || durations.is_fresh(self.round);
        let unproven = match own.and_then(|own| own.get(&size)).filter(|d| fresh_or_plain(d)) {
            Some(durations) if durations.proven() => return durations.cost(),
            Some(durations) => durations.cost(),
            None => own
                .into_iter()
                .map(|own| LaunchCosts::at_least(own, size))
                .chain(self.lesser_target().map(|measured| LaunchCosts::at_least(&measured.verify, size)))
                .fold(0.0, f64::max),
        };
        let larger = own
            .into_iter()
            .flat_map(|own| own.range(size + 1..))
            .filter(|(_, larger)| larger.proven() && larger.is_fresh(self.round))
            .map(|(_, larger)| larger.cost());
        let shorter = self.shorter_target().filter_map(|(ratio, measured)| {
            measured
                .verify
                .get(&size)
                .filter(|durations| durations.proven() && fresh_or_plain(durations))
                .map(|durations| ratio * durations.cost())
        });
        larger.chain(shorter).fold(unproven, f64::min)
    }

    fn draft(&self, generated: usize) -> f64 {
        if generated == 0 {
            return 0.0;
        }
        if self.own_draft().is_none_or(|own| !own.contains_key(&generated)) {
            if let Some(prior) = self.launch_prices.and_then(|prices| {
                prices
                    .draft_seconds(
                        self.sequences.max(1) as u32,
                        generated.min(usize::from(u8::MAX)) as u8,
                        self.context_class as u32,
                    )
                    .ok()
                    .flatten()
            }) {
                return prior;
            }
        }
        self.draft_seconds_from_table(generated)
    }

    fn prior_verify_seconds(prior: &StreamedVerifyPrior, sequences: usize, rows: usize) -> Option<f64> {
        if prior.verify_seconds.is_empty() {
            return None;
        }
        let k = (rows / sequences.max(1)).saturating_sub(1);
        if let Some(&seconds) = prior.verify_seconds.get(k) {
            return Some(seconds);
        }
        let last_idx = prior.verify_seconds.len() - 1;
        let last = prior.verify_seconds[last_idx];
        let slope = if last_idx > 0 {
            (last - prior.verify_seconds[last_idx - 1]).max(0.0)
        } else {
            0.0
        };
        Some(last + (k - last_idx) as f64 * slope)
    }

    /// Derived prior duration for plain decode (`rows == self.sequences`) from
    /// the current regime's `StreamedVerifyPrior` (anchored to the nearest
    /// proven, fresh measured verify width if any) or non-banked `LaunchPrices`,
    /// without falling back to stale `lesser_target()` measurements.
    fn derived_plain_prior(&self) -> Option<f64> {
        let rows = self.sequences.max(1);
        if let Some(prior) = self.verify_prior {
            if let Some(prior_rows) = Self::prior_verify_seconds(prior, self.sequences, rows) {
                if let Some((&ref_rows, ref_dur)) = self.own_target().and_then(|own| {
                    own.verify
                        .iter()
                        .filter(|(_, d)| d.proven() && d.is_fresh(self.round))
                        .min_by_key(|(&r, _)| r.abs_diff(rows))
                }) {
                    if let Some(prior_ref) =
                        Self::prior_verify_seconds(prior, self.sequences, ref_rows).filter(|&p| p > 0.0)
                    {
                        let anchored = ref_dur.cost() * (prior_rows / prior_ref);
                        if anchored > 0.0 {
                            return Some(anchored);
                        }
                    }
                }
                if prior_rows > 0.0 {
                    return Some(prior_rows);
                }
            }
        }
        if self.own_target().is_none_or(|own| own.verify.is_empty()) {
            if let Some(prior) = self.launch_prices.filter(|p| !p.is_banked()).and_then(|prices| {
                prices
                    .plain_seconds(rows as u32, self.context_class as u32)
                    .ok()
                    .map(|f| f.resident)
                    .filter(|&s| s > 0.0)
            }) {
                return Some(prior);
            }
        }
        None
    }

    fn verify(&self, rows: usize) -> f64 {
        let is_non_plain = rows > self.sequences.max(1);
        // On streamed MoE (`verify_prior.is_some()`), single-round SSD/miss
        // bursts vary 2-4x from token to token: require a proven median
        // (`COST_SAMPLES = 3`) before overriding the anchored `StreamedVerifyPrior`
        // so one noisy verify spike or one lucky low-miss plain round cannot
        // collapse draft width to M = 1.
        let streamed = self.verify_prior.is_some();
        let has_live = self.own_target().is_some_and(|own| {
            own.verify.get(&rows).is_some_and(|d| {
                let valid = (!streamed || d.proven()) && (!is_non_plain || d.is_fresh(self.round));
                if !valid {
                    return false;
                }
                // On streamed MoE, if residency warmed up after `d` was measured
                // (a narrower width measured more recently implies >1.5x lower
                // cost via `StreamedVerifyPrior` and `d` is at least
                // `WARMUP_DRIFT_ROUNDS` old), expire `d` so wide bursts (`M=8`)
                // are re-measured instead of staying locked out for 100 rounds.
                if streamed && is_non_plain && self.round.saturating_sub(d.latest()) >= WARMUP_DRIFT_ROUNDS {
                    if let Some(prior) = self.verify_prior {
                        if let Some(prior_rows) = Self::prior_verify_seconds(prior, self.sequences, rows) {
                            let warmed_up = own.verify.iter().any(|(&ref_rows, ref_dur)| {
                                ref_rows < rows
                                    && ref_dur.proven()
                                    && ref_dur.is_fresh(self.round)
                                    && ref_dur.latest() > d.latest()
                                    && Self::prior_verify_seconds(prior, self.sequences, ref_rows)
                                        .filter(|&p| p > 0.0)
                                        .is_some_and(|prior_ref| {
                                            ref_dur.cost() * (prior_rows / prior_ref) < d.cost() * 0.65
                                        })
                            });
                            if warmed_up {
                                return false;
                            }
                        }
                    }
                }
                true
            })
        });
        if !has_live {
            if let Some(prior) = self.verify_prior {
                if let Some(prior_rows) = Self::prior_verify_seconds(prior, self.sequences, rows) {
                    // Anchor the streamed verify prior to the nearest proven,
                    // fresh measured verify width in the current regime via
                    // ratio scaling (`measured(ref) * prior(rows) / prior(ref)`),
                    // preventing scale mismatches between live measured widths
                    // and unmeasured wider widths from locking out exploration.
                    if let Some((&ref_rows, ref_dur)) = self.own_target().and_then(|own| {
                        own.verify
                            .iter()
                            .filter(|&(&r, d)| r != rows && d.proven() && d.is_fresh(self.round))
                            .min_by_key(|(&r, _)| r.abs_diff(rows))
                    }) {
                        if let Some(prior_ref) =
                            Self::prior_verify_seconds(prior, self.sequences, ref_rows).filter(|&p| p > 0.0)
                        {
                            return ref_dur.cost() * (prior_rows / prior_ref);
                        }
                    }
                    return prior_rows;
                }
            }
            // Only use GPU-only `LaunchPrices` resident priors on cold start (before any
            // verify launch is measured) and only for non-banked resident targets; banked
            // MoE targets include streaming miss service that `.resident` omits, and
            // mixing a cold analytical prior for small `rows` against live measured
            // durations for large `rows` collapses draft width.
            if self.own_target().is_none_or(|own| own.verify.is_empty()) {
                if let Some(prior) = self.launch_prices.filter(|p| !p.is_banked()).and_then(|prices| {
                    let seq = self.sequences.max(1) as u32;
                    let depth = self.context_class as u32;
                    if rows == self.sequences.max(1) {
                        prices.plain_seconds(seq, depth).ok().map(|f| f.resident)
                    } else {
                        prices.verify_seconds(seq, rows as u32, depth).ok().map(|f| f.resident)
                    }
                }) {
                    return prior;
                }
            }
        }
        self.verify_seconds_from_table(rows)
    }

    /// What changing between drafting and not costs a round beyond its
    /// launches: measured here in the current regime, or else the most it
    /// measured at no longer a context (it enters pairs and restarts a
    /// pipeline, not attention).
    fn switch(&self, switch: Switch) -> Option<f64> {
        if let Some(own) = self.own_target().map(|own| own.switch(switch)).filter(|durations| durations.measured()) {
            return Some(own.cost());
        }
        self.lesser_target()
            .map(|costs| costs.switch(switch))
            .filter(|durations| durations.measured())
            .map(Durations::cost)
            .reduce(f64::max)
    }

    /// The switch a round drafting (or preparing to) or not makes from the
    /// request's previous round.
    fn switching(&self, drafting: bool) -> Option<Switch> {
        match (self.after_drafting, drafting) {
            (Some(false), true) => Some(Switch::Enter),
            (Some(true), false) => Some(Switch::Leave),
            _ => None,
        }
    }

    /// Seconds of a round generating `generated` and verifying `verified`,
    /// in launches whose other requests take the same rows, with its share
    /// of what changing between drafting and not costs.
    fn round(&self, generated: usize, verified: usize) -> f64 {
        let switch = self
            .switching(generated > 0)
            .and_then(|switch| self.switch(switch))
            .map_or(0.0, |switch| self.switch_weight * switch);
        (self.draft(generated) + self.verify(self.sequences * (verified + 1)) + switch).max(1e-9)
    }

    /// What a round generating `generated` that is a look pays for its
    /// switches beyond its share: a look that switches (one drafting after
    /// a round that did not, or a plain one after a round that drafted)
    /// does so for its round alone, so it switches and switches back whole.
    fn excursion(&self, generated: usize) -> f64 {
        let Some(there) = self.switching(generated > 0) else {
            return 0.0;
        };
        let back = match there {
            Switch::Enter => Switch::Leave,
            Switch::Leave => Switch::Enter,
        };
        let cost = |switch| self.switch(switch).unwrap_or(0.0);
        (1.0 - self.switch_weight) * cost(there) + cost(back)
    }

    /// The price of a round's launches when both are measured here in the
    /// current regime.
    fn steady(&self, generated: usize, rows: usize) -> Option<f64> {
        let drafted = generated == 0 || self.own_draft().is_some_and(|own| own.contains_key(&generated));
        let verified = self.own_target().is_some_and(|own| own.verify.contains_key(&rows));
        (drafted && verified).then(|| self.draft(generated) + self.verify(rows))
    }
}

/// A request's drafted block awaiting verification, for sharing one
/// launch's rows among the requests in it.
#[derive(Clone, Debug)]
pub struct DraftOutlook {
    load: Arc<DraftLoad>,
    context_class: usize,
    /// How often its blocks are right through each depth of this block.
    right: Vec<f64>,
    /// Seconds its drafter launch took.
    draft_seconds: f64,
}

/// The target rows each of several requests takes in one verification
/// launch of at most `rows`: its anchor's and those of the proposals it
/// verifies. A launch runs as a row class, so its rows go to the proposals
/// most often right, whichever requests drafted them, and the class is the
/// one committing the most tokens per second. `None` when fewer than two
/// requests share the launch: one request decides for itself.
pub fn share_rows(outlooks: &[DraftOutlook], rows: usize) -> Option<Vec<usize>> {
    let requests = outlooks.len();
    if requests < 2 || rows < requests {
        return None;
    }
    let context_class = outlooks.iter().map(|outlook| outlook.context_class).max()?;
    let state = outlooks[0].load.lock();
    let costs = RoundCosts::of(&state, context_class, requests, None);
    let drafted: f64 = outlooks.iter().map(|outlook| outlook.draft_seconds).sum();
    // A request's proposals are no more often right the deeper they are, so
    // taking proposals in this order takes each request's in depth order.
    let mut proposals = outlooks
        .iter()
        .enumerate()
        .flat_map(|(request, outlook)| {
            outlook
                .right
                .iter()
                .enumerate()
                .map(move |(depth, right)| (*right, depth, request))
        })
        .collect::<Vec<_>>();
    proposals.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut best = (0, f64::NEG_INFINITY);
    let mut class = requests.next_power_of_two();
    loop {
        let verified = class
            .min(rows)
            .saturating_sub(requests)
            .min(proposals.len());
        let tokens = requests as f64
            + proposals[..verified]
                .iter()
                .map(|(right, ..)| right)
                .sum::<f64>();
        let rate = tokens / (drafted + costs.verify(requests + verified)).max(1e-9);
        if rate > best.1 {
            best = (verified, rate);
        }
        if class >= rows || verified == proposals.len() {
            break;
        }
        class *= 2;
    }
    let mut shares = vec![1; requests];
    for (_, _, request) in &proposals[..best.0] {
        shares[*request] += 1;
    }
    Some(shares)
}

/// Shared mutable state of a [`DraftLoad`], guarded by a single mutex so
/// coupled fields (draft costs, regime-keyed target costs, cover state,
/// rounds, depth means + calibration, and look budgets) update atomically
/// with no nested locks.
#[derive(Clone, Debug)]
struct DraftLoadState {
    /// Drafter launch costs by `(context_class, sequences)`: independent of
    /// target expert-bank residency regimes.
    draft_costs: BTreeMap<(usize, usize), BTreeMap<usize, Durations>>,
    /// Target verify/switch costs by `(regime_epoch, context_class, sequences)`:
    /// keyed to the target bank's physical `regime_epoch` so prior-regime
    /// measurements are inapplicable by construction.
    target_costs: BTreeMap<(Option<u64>, usize, usize), TargetCosts>,
    rounds: u64,
    depths: Vec<(DepthMeans, Calibration)>,
    /// The run of accepted rounds that earned its latest request a look.
    look_run: u32,
    look_seconds: f64,
    /// The decode time set aside for the next plain look.
    plain_seconds: f64,
    /// Whether `draft_costs` / `target_costs` hold a duration measured beside
    /// the expert bank's cover read ([`Self::settle`]).
    beside_cover: bool,
    /// Latest streamed verify prior from the cost model's regime. Its
    /// `regime_epoch` keys `target_costs` directly.
    verify_prior: Option<StreamedVerifyPrior>,
    /// Analytical launch-price snapshot for unmeasured `D(g)` and `V(v+1)`
    /// launches (replaced on first live observation via `Durations`) and
    /// structural `sigma_prior` for unscored depths (replaced via `DepthMeans`).
    launch_prices: Option<LaunchPrices>,
}

impl Default for DraftLoadState {
    fn default() -> Self {
        Self {
            draft_costs: BTreeMap::new(),
            target_costs: BTreeMap::new(),
            rounds: 0,
            depths: Vec::new(),
            look_run: 1,
            look_seconds: 0.0,
            plain_seconds: 0.0,
            beside_cover: false,
            verify_prior: None,
            launch_prices: None,
        }
    }
}

impl DraftLoadState {
    /// Drop every duration measured beside the bank's cover read, which a
    /// launch measured with none beside it shows ended ([`LaunchCosts::settle`]).
    fn settle(&mut self) {
        if std::mem::take(&mut self.beside_cover) {
            self.draft_costs.retain(|_, drafts| {
                drafts.retain(|_, durations| {
                    durations.settle();
                    durations.measured()
                });
                !drafts.is_empty()
            });
            self.target_costs.retain(|_, costs| {
                costs.settle();
                !costs.is_empty()
            });
        }
    }

    fn regime_epoch(&self) -> Option<u64> {
        self.verify_prior.as_ref().map(|p| p.regime_epoch)
    }
}

/// What a load's requests share: launch costs by context class and sequence
/// count, the acceptance means its latest request reached, and the decode
/// time set aside for the next look. That time carries from request to
/// request, so a load of requests each too short to earn a look by itself
/// still takes them at their share of the decode time, and drafting that
/// stopped paying is tried again.
#[derive(Debug, Default)]
pub struct DraftLoad {
    state: Mutex<DraftLoadState>,
}

impl DraftLoad {
    fn lock(&self) -> std::sync::MutexGuard<'_, DraftLoadState> {
        self.state.lock().expect("draft load state")
    }

    /// Observe the streamed target verify prior for the current round.
    /// Target verify/enter/leave durations are keyed directly by
    /// `regime_epoch`, so any regime change makes prior-regime target costs
    /// inapplicable by construction; older-regime entries are also pruned to
    /// bound memory while preserving `draft_costs`.
    pub fn observe_streamed(&self, prior: Option<StreamedVerifyPrior>) {
        let mut state = self.lock();
        let next_regime = prior.as_ref().map(|p| p.regime_epoch);
        if state.regime_epoch() != next_regime {
            state.target_costs.retain(|&(regime, ..), _| regime == next_regime);
        }
        state.verify_prior = prior;
    }

    /// Supply the load's analytical [`LaunchPrices`] snapshot so unmeasured
    /// `D(g)` and `V(v+1)` launches and unscored proposal depths start from
    /// the cost model's prior rather than zero-cost `UNSCORED` probing.
    pub fn observe_launch_prices(&self, prices: Option<LaunchPrices>) {
        self.lock().launch_prices = prices;
    }
}

/// A drafted block being scored against the tokens committed after it.
#[derive(Clone, Debug, PartialEq)]
struct Block {
    tokens: Vec<TokenId>,
    /// The drafter's confidence in each token, when it gives one.
    confidence: Option<Vec<f32>>,
    matched: usize,
    /// Drafted by a look taken on a run of accepted rounds.
    run_look: bool,
}

/// One round's choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RoundChoice {
    pub generated: usize,
    pub verified: usize,
    /// Whether the round is a look: wider than its kept width, to score the
    /// deeper proposals, or plain while drafting, to measure the plain
    /// decode again.
    pub look: bool,
}

impl RoundChoice {
    /// Whether the round drafts, or prepares to (see [`Switch`]).
    fn drafting(self, preparing: bool) -> bool {
        self.generated > 0 || preparing
    }
}

/// Per-depth acceptance estimate, confidence calibration, and whether a block
/// scored this depth in the current round.
#[derive(Clone, Debug)]
struct DepthSlot {
    means: DepthMeans,
    calibration: Calibration,
    scored: bool,
}

impl DepthSlot {
    const UNSCORED: Self = Self {
        means: DepthMeans::UNSCORED,
        calibration: Calibration::EMPTY,
        scored: false,
    };
}

/// The launch this round is currently expecting a [`TimedLaunch`] duration for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ExpectedLaunch {
    /// No launch is currently awaiting timing.
    #[default]
    None,
    /// Expecting the drafter's proposing launch for this many generated proposals.
    Draft(usize),
    /// Expecting the target's verification or plain-decode launch.
    Target,
}

/// Duration of drafter entry launches accumulated for the upcoming timed launch.
#[derive(Clone, Copy, Debug, PartialEq)]
enum EntryTiming {
    /// All entries so far replayed their graphs, totalling these seconds.
    Replayed(f64),
    /// At least one entry recorded its graph, so the launch it serves has no
    /// replay duration.
    Recorded,
}

impl Default for EntryTiming {
    fn default() -> Self {
        Self::Replayed(0.0)
    }
}

impl EntryTiming {
    fn record(&mut self, seconds: Option<f64>) {
        *self = match (*self, seconds) {
            (Self::Replayed(total), Some(seconds)) => Self::Replayed(total + seconds),
            _ => Self::Recorded,
        };
    }

    fn take_with(&mut self, seconds: Option<f64>) -> Option<f64> {
        match (std::mem::take(self), seconds) {
            (Self::Replayed(entered), Some(seconds)) => Some(entered + seconds),
            _ => None,
        }
    }
}

/// This round's drafter launch outcome, awaited by the target launch that
/// closes the round.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum DrafterTiming {
    /// This round did not run a timed drafter launch.
    #[default]
    None,
    /// The drafter launch (or an entry folded into it) recorded its graph,
    /// so it ran as part of this round but has no replay duration.
    Recorded,
    /// The drafter launch (and any entries folded into it) replayed in these
    /// seconds.
    Replayed(f64),
}

impl DrafterTiming {
    fn ran(self) -> bool {
        !matches!(self, Self::None)
    }

    fn seconds(self) -> Option<f64> {
        match self {
            Self::None => Some(0.0),
            Self::Recorded => None,
            Self::Replayed(seconds) => Some(seconds),
        }
    }
}

/// Timing state for the round currently in flight.
#[derive(Clone, Copy, Debug, Default)]
struct RoundTiming {
    expected: ExpectedLaunch,
    entered: EntryTiming,
    drafter: DrafterTiming,
}

/// State of a deferred switch from plain decoding to drafting (see [`Switch`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SwitchStage {
    /// No switch is in flight.
    Ready,
    /// Chosen by this round, which drafts nothing and whose queued decode
    /// publishes its feature: the next round prepares, as that decode.
    QueuedBehind(RoundChoice),
    /// This round is preparing to draft: its target step publishes its feature
    /// for the next round's draft.
    Preparing(RoundChoice),
    /// The preparing round ended; the next round drafts this choice if anchored
    /// and within its limit.
    Prepared(RoundChoice),
}

/// One request's draft width controller.
#[derive(Clone, Debug)]
pub struct DraftController {
    /// Candidate widths, increasing, without 0.
    ladder: Vec<usize>,
    load: Arc<DraftLoad>,
    depths: Vec<DepthSlot>,
    blocks: Vec<Block>,
    /// Rounds in a row whose verified proposals were all accepted, and the
    /// run of them that earns a look. A look that finds its block right
    /// halves that run and one that finds it wrong doubles it, so looks
    /// follow one another while they keep finding more and thin out as
    /// they find nothing.
    accepted_run: u32,
    look_run: u32,
    run_look: bool,
    /// The width last generated by the fastest pair or kept past it, and
    /// the rounds of being kept it has earned.
    kept: Option<(usize, f64)>,
    /// Launch timing state for the round in flight.
    timing: RoundTiming,
    after_drafting: Option<bool>,
    chosen: Option<bool>,
    /// How often this round's drafted block is expected to be right through
    /// each of its depths, until it is verified.
    drafted: Option<Vec<f64>>,
    context_class: usize,
    sequences: usize,
    choice: RoundChoice,
    switch: SwitchStage,
}

impl DraftController {
    /// A controller choosing among widths up to `widest`, from the load's
    /// latest acceptance means. A load that has scored nothing takes every
    /// proposal as right.
    pub fn new(widest: usize, load: Arc<DraftLoad>) -> Self {
        let (depths, look_run) = {
            let state = load.lock();
            let depths = if state.depths.len() == widest {
                state
                    .depths
                    .iter()
                    .map(|(means, calibration)| DepthSlot {
                        means: means.clone(),
                        calibration: calibration.clone(),
                        scored: false,
                    })
                    .collect()
            } else {
                vec![DepthSlot::UNSCORED; widest]
            };
            (depths, state.look_run)
        };
        Self {
            ladder: draft_ladder(widest),
            load,
            depths,
            blocks: Vec::new(),
            accepted_run: 0,
            look_run,
            run_look: false,
            kept: None,
            timing: RoundTiming::default(),
            after_drafting: None,
            chosen: None,
            drafted: None,
            context_class: 1,
            sequences: 1,
            choice: RoundChoice::default(),
            switch: SwitchStage::Ready,
        }
    }

    pub fn choice(&self) -> RoundChoice {
        self.choice
    }

    /// Whether this round prepares to draft: its target step publishes its
    /// feature for the next round's draft (see [`Switch`]).
    pub fn preparing(&self) -> bool {
        matches!(self.switch, SwitchStage::Preparing(_))
    }

    /// Whether the decode queued behind this round, which drafts nothing,
    /// publishes its feature: the next round prepares (see [`Switch`]).
    pub fn publishes_behind(&self) -> bool {
        matches!(self.switch, SwitchStage::QueuedBehind(_))
    }

    /// How often a block is expected to be right through each depth, from
    /// depth 1 (see [`DepthMeans`]). A block right through a depth is right
    /// through every shallower one, so no depth is rated above a shallower
    /// one. Unscored depths keep optimistic [`UNSCORED`] (`1.0`), carrying
    /// the deepest measured depth's live acceptance forward so high-acceptance
    /// workloads explore deeper widths instead of getting trapped by a
    /// pessimistic offline prior.
    pub fn right(&self) -> Vec<f64> {
        let mut bound = 1.0f64;
        self.depths
            .iter()
            .map(|slot| {
                bound = bound.min(slot.means.estimate());
                bound
            })
            .collect()
    }

    /// The measured launch seconds at this round's context class and current
    /// physical regime.
    pub fn measured(&self) -> LaunchCosts {
        let state = self.load.lock();
        let draft = state
            .draft_costs
            .get(&(self.context_class, self.sequences))
            .cloned()
            .unwrap_or_default();
        let target = state
            .target_costs
            .get(&(state.regime_epoch(), self.context_class, self.sequences))
            .cloned()
            .unwrap_or_default();
        LaunchCosts {
            draft,
            verify: target.verify,
            enter: target.enter,
            leave: target.leave,
        }
    }

    /// The costs this round chooses by. A plain decode taken to measure it
    /// changes no choice: a load drafting goes on drafting after it, so the
    /// round coming back from it is priced as one continuing to draft. A
    /// switch is shared by the weight a block has in the first depth's
    /// estimate, which every drafting round depends on.
    fn costs<'a>(&self, state: &'a DraftLoadState) -> RoundCosts<'a> {
        RoundCosts::of(state, self.context_class, self.sequences, self.chosen)
            .sharing_switches(self.depths.first().map_or(MEMORY, |slot| slot.means.weight()))
    }

    /// This round's drafted block, while it awaits verification.
    pub fn outlook(&self) -> Option<DraftOutlook> {
        let right = self.drafted.clone()?;
        let state = self.load.lock();
        let draft_seconds = self.costs(&state).draft(self.choice.generated);
        Some(DraftOutlook {
            load: self.load.clone(),
            context_class: self.context_class,
            right,
            draft_seconds,
        })
    }

    /// The candidate widths of at most `limit`, no drafting first.
    fn widths(&self, limit: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::once(0).chain(
            self.ladder
                .iter()
                .copied()
                .filter(move |width| *width <= limit),
        )
    }

    /// Tokens a round verifying `verified` proposals is expected to commit.
    fn tokens(right: &[f64], verified: usize) -> f64 {
        1.0 + right[..verified].iter().sum::<f64>()
    }

    /// The width a round generating `generated` verifies for the most
    /// tokens per second, of at most `limit`, with that rate. Of equal
    /// rates the narrowest.
    fn verified(
        &self,
        right: &[f64],
        costs: &RoundCosts,
        generated: usize,
        limit: usize,
    ) -> (usize, f64) {
        let mut best = (0, f64::NEG_INFINITY);
        for verified in self.widths(limit.min(generated)) {
            let rate = Self::tokens(right, verified) / costs.round(generated, verified);
            if rate > best.1 {
                best = (verified, rate);
            }
        }
        best
    }

    /// The proposals this round generates, of at most `limit`, for a round
    /// anchored at target position `position`. A drafter without its
    /// anchor's feature (`anchored` false: the round before published none)
    /// cannot draft: a round choosing to drafts nothing, the decode queued
    /// behind it publishes its feature as the next round, which prepares,
    /// and the round after drafts the choice (see [`Switch`]).
    pub fn generate(&mut self, limit: usize, position: usize, anchored: bool) -> usize {
        self.context_class = position.max(1).next_power_of_two();
        // Entries before a round chooses are no round's: the prompt's, or
        // pairs held past their bound by rounds that drafted nothing.
        self.timing.entered = EntryTiming::default();
        match std::mem::replace(&mut self.switch, SwitchStage::Ready) {
            // The round after the choice prepares: it is the decode queued
            // publishing its feature, and drafts nothing.
            SwitchStage::QueuedBehind(choice) if !anchored => {
                self.switch = SwitchStage::Preparing(choice);
                self.choice = RoundChoice { generated: 0, verified: 0, look: choice.look };
                self.timing.expected = ExpectedLaunch::Target;
                return 0;
            }
            // The round after the one that prepared drafts the choice, where
            // the request still has room for it.
            SwitchStage::Preparing(choice) | SwitchStage::Prepared(choice)
                if anchored && choice.generated <= limit =>
            {
                self.chosen = Some(true);
                self.choice = choice;
                self.timing.expected = ExpectedLaunch::Draft(choice.generated);
                return choice.generated;
            }
            _ => {}
        }
        let mut load = self.load.lock();
        let state = &mut *load;
        let costs = RoundCosts::from_parts(
            &state.draft_costs,
            &state.target_costs,
            state.verify_prior.as_ref(),
            state.launch_prices.as_ref(),
            state.rounds,
            self.context_class,
            self.sequences,
            self.chosen,
        )
        .sharing_switches(self.depths.first().map_or(MEMORY, |slot| slot.means.weight()));
        let right = self.right();
        // The fastest pair. Of equal rates the narrowest.
        let mut fastest = (0, 0, f64::NEG_INFINITY);
        for generated in self.widths(limit) {
            let (verified, rate) = self.verified(&right, &costs, generated, limit);
            if rate > fastest.2 {
                fastest = (generated, verified, rate);
            }
        }
        let (fastest, verified, rate) = fastest;
        // The plain decode is the one launch no wider look takes, so while
        // it is measured here but its cost is unproven or stale it is taken
        // again in place of drafting: a duration another process held up,
        // or a machine since freed, would otherwise price it for good.
        let round = state.rounds;
        let plain = costs.own_target().and_then(|own| own.verify.get(&costs.sequences));
        // On streamed MoE (`verify_prior.is_some()`), `derived_plain_prior()`
        // already prices `V(1)` in the current regime: do not hijack active
        // drafting rounds into back-to-back M = 1 plain decode bursts.
        let plain_unsettled = fastest > 0
            && costs.verify_prior.is_none()
            && plain.is_some_and(|plain| {
                !plain.proven() || round.saturating_sub(plain.latest()) >= STALE_ROUNDS
            });
        // How much faster than the fastest pair a width could be at best:
        // its rate if every proposal it drafts were right, over the fastest
        // pair's, less one.
        // And what a round generating it loses to the fastest pair: its
        // time beyond what its tokens take at the fastest pair's rate, with
        // the whole of the switches it makes as a look.
        let all_right = vec![1.0; right.len()];
        let wider = self
            .widths(limit)
            .filter(|generated| *generated > fastest)
            .map(|generated| {
                let (_, best) = self.verified(&all_right, &costs, generated, limit);
                let (verified, _) = self.verified(&right, &costs, generated, limit);
                let lost = costs.round(generated, verified) + costs.excursion(generated)
                    - Self::tokens(&right, verified) / rate;
                (generated, (best / rate - 1.0).max(0.0), lost.max(0.0))
            })
            .collect::<Vec<_>>();
        let gain = |generated: usize| {
            wider
                .iter()
                .find(|(width, ..)| *width == generated)
                .map_or(0.0, |(_, gain, _)| *gain)
        };
        // A width the text stopped supporting is kept for a while: its
        // rounds go on scoring its depths, so its return is seen at once.
        // Narrowing gives that up, and coming back then takes looks. Each
        // round a width is the fastest earns it a round of being kept, up
        // to what the looks would take to bring it back (one in
        // `LOOK_SHARE` rounds), and a kept round spends more of that the
        // less of the rate is at stake.
        let kept = match self.kept {
            Some((kept, earned)) if kept > limit => {
                self.kept = Some((kept, earned));
                fastest
            }
            Some((kept, earned)) if kept > fastest && earned > 0.0 => {
                let stake = gain(kept) / (1.0 + gain(kept));
                self.kept = Some((kept, earned - 1.0 / stake.max(LOOK_SHARE)));
                kept
            }
            Some((kept, earned)) if kept == fastest => {
                self.kept = Some((kept, (earned + 1.0).min(1.0 / LOOK_SHARE)));
                kept
            }
            _ => {
                self.kept = Some((fastest, 1.0));
                fastest
            }
        };
        // The look: the wider width with the most to gain for the time it
        // loses. It is taken as often as the looks' share of the decode
        // time pays for it, a share scaled down where little can be gained.
        // It is also taken on a run of rounds that accepted everything they
        // verified: text that became predictable is the one case a narrow
        // width cannot see, and such a run is what it looks like from
        // there.
        let look = wider
            .iter()
            .copied()
            .filter(|(generated, gain, _)| *generated > kept && *gain > 0.0)
            .max_by(|a, b| {
                let worth = |(_, gain, lost): &(usize, f64, f64)| gain / lost.max(1e-12);
                worth(a).total_cmp(&worth(b)).then(a.0.cmp(&b.0))
            });
        let generated = match look {
            Some((generated, gain, lost)) => {
                self.run_look = false;
                state.look_seconds += LOOK_SHARE * gain.min(1.0) * costs.round(fastest, verified);
                if state.look_seconds >= lost {
                    state.look_seconds = 0.0;
                    generated
                } else if self.accepted_run >= self.look_run {
                    self.accepted_run = 0;
                    self.run_look = true;
                    generated
                } else {
                    kept
                }
            }
            // Nothing wider to look at this round (the fastest pair is the
            // widest, or the request's last tokens leave no room): the time
            // set aside so far is kept for the next round that has one.
            None => {
                self.run_look = false;
                kept
            }
        };
        // A plain decode not measured here costs at least what it measured
        // at no longer a context and among no more requests, while the
        // machine and the expert bank are as they were then: for as long as
        // any price lasts. Past that nothing measured bounds it, and it could
        // be faster than the fastest pair by any amount. It is then a look,
        // as a wider width is: taken as often as the share of the decode time
        // for what it could gain pays for what it loses (its round's time
        // beyond what its token takes at the fastest pair's rate, with the
        // whole of the switches it makes), on a round of the fastest pair,
        // so that a width kept past it goes on measuring itself.
        let plain_look = fastest > 0 && plain.is_none() && {
            let measured_floor = costs
                .lesser_target()
                .flat_map(|measured| measured.verify.range(..=costs.sequences))
                .filter(|(_, durations)| round.saturating_sub(durations.latest()) < STALE_ROUNDS)
                .map(|(_, durations)| durations.cost())
                .reduce(f64::max);
            let floor = match (measured_floor, costs.derived_plain_prior()) {
                (Some(m), Some(p)) => Some(m.max(p)),
                (m, p) => m.or(p),
            };
            let gain = floor.map_or(1.0, |floor| (1.0 / floor / rate - 1.0).min(1.0));
            gain > 0.0 && {
                state.plain_seconds += LOOK_SHARE * gain * costs.round(fastest, verified);
                let lost = costs.round(0, 0) + costs.excursion(0) - 1.0 / rate;
                let due = generated == kept && kept == fastest && state.plain_seconds >= lost;
                if due {
                    state.plain_seconds = 0.0;
                }
                due
            }
        };
        drop(load);
        // A plain decode taken to measure it changes no choice (see
        // [`Self::costs`]).
        let measuring = plain_unsettled || plain_look;
        let generated = if measuring { 0 } else { generated };
        if !measuring {
            self.chosen = Some(generated > 0);
        }
        let choice = RoundChoice {
            generated,
            verified: 0,
            look: generated != kept,
        };
        // Without its anchor's feature the round drafts nothing; the decode
        // queued behind it prepares, and the round after drafts this choice.
        let generated = if anchored { generated } else { 0 };
        if choice.generated > generated {
            self.switch = SwitchStage::QueuedBehind(choice);
        }
        self.choice = RoundChoice { generated, ..choice };
        self.timing.expected = if generated > 0 {
            ExpectedLaunch::Draft(generated)
        } else {
            ExpectedLaunch::Target
        };
        generated
    }

    /// The block the drafter returned for a round that generated
    /// `generated`, with its confidence in each token when it gives one; a
    /// block cut short is wrong from where it ends.
    pub fn drafted(&mut self, generated: usize, block: &[TokenId], confidence: Option<&[f32]>) {
        assert!(
            confidence.is_none_or(|confidence| confidence.len() == block.len()),
            "a drafter's confidences are its block's"
        );
        for depth in block.len() + 1..=generated {
            self.score(depth, false);
        }
        if !block.is_empty() {
            self.blocks.push(Block {
                tokens: block.to_vec(),
                confidence: confidence.map(<[f32]>::to_vec),
                matched: 0,
                run_look: self.run_look,
            });
        }
        self.drafted = Some(self.survival(block.len(), confidence));
    }

    /// How often a drafted block of `drafted` is expected to be right
    /// through each of its depths: each depth's acceptance after the one
    /// before it, as its estimate gives it ([`Self::right`]), times how much
    /// more or less often proposals of the drafter's confidence in this one
    /// are right there ([`Calibration`]).
    fn survival(&self, drafted: usize, confidence: Option<&[f32]>) -> Vec<f64> {
        let right = self.right();
        let Some(confidence) = confidence else {
            return right[..drafted].to_vec();
        };
        let (mut before, mut survival) = (1.0, 1.0);
        right[..drafted]
            .iter()
            .zip(confidence)
            .zip(&self.depths)
            .map(|((right, confidence), slot)| {
                let after = if before > 0.0 { right / before } else { 0.0 };
                before = *right;
                survival *= (after * slot.calibration.lift(*confidence)).min(1.0);
                survival
            })
            .collect()
    }

    /// The proposals of this round's drafted block it verifies, of at most
    /// `limit`. The drafter's launch is paid whatever is verified.
    pub fn verify(&mut self, limit: usize) -> usize {
        let right = self
            .drafted
            .take()
            .expect("a round verifies the block it drafted");
        let verified = {
            let state = self.load.lock();
            self.verified(
                &right,
                &self.costs(&state),
                self.choice.generated,
                limit.min(right.len()),
            )
            .0
        };
        self.choice.verified = verified;
        self.timing.expected = ExpectedLaunch::Target;
        verified
    }

    fn score(&mut self, depth: usize, right: bool) {
        let slot = &mut self.depths[depth - 1];
        slot.means.score(right);
        slot.scored = true;
    }

    /// Score every open block against the tokens just committed, which end
    /// the round: a depth no block scored in it goes unscored for a round.
    pub fn committed(&mut self, tokens: &[TokenId]) {
        // The round ended: a round that does not choose prepares nothing.
        if let SwitchStage::Preparing(choice) = self.switch {
            self.switch = SwitchStage::Prepared(choice);
        }
        // A round that verified proposals and committed them all extends
        // the run; any other ends it.
        if self.choice.verified > 0 && tokens.len() == self.choice.verified + 1 {
            self.accepted_run = self.accepted_run.saturating_add(1);
        } else {
            self.accepted_run = 0;
        }
        let mut blocks = std::mem::take(&mut self.blocks);
        for token in tokens {
            blocks.retain_mut(|block| {
                let right = block.tokens[block.matched] == *token;
                if let Some(confidence) = &block.confidence {
                    self.depths[block.matched]
                        .calibration
                        .score(confidence[block.matched], right);
                }
                if right {
                    block.matched += 1;
                    self.score(block.matched, true);
                    if block.matched == block.tokens.len() && block.run_look {
                        self.look_run = (self.look_run / 2).max(1);
                    }
                    block.matched < block.tokens.len()
                } else {
                    for depth in block.matched + 1..=block.tokens.len() {
                        self.score(depth, false);
                    }
                    if block.run_look {
                        self.look_run = self.look_run.saturating_mul(2);
                    }
                    false
                }
            });
        }
        self.blocks = blocks;
        for slot in &mut self.depths {
            if !std::mem::take(&mut slot.scored) {
                slot.means.unscored();
            }
        }
        let mut state = self.load.lock();
        state.depths = self
            .depths
            .iter()
            .map(|slot| (slot.means.clone(), slot.calibration.clone()))
            .collect();
        state.look_run = self.look_run;
    }

    /// A launch of this request finished in `duration`: `rows` rows of
    /// `sequences` requests. A drafter launch is measured by the proposals
    /// it generated, a target launch by its rows. The drafter's entries for
    /// a round are part of its proposing launch, or of the target's when the
    /// round proposed nothing. A switch between drafting and not (see
    /// [`Switch`]) is paid in the launch that follows it: the discarded
    /// decode and the catch-up before the drafter proposes, or in a
    /// preparing round's target launch the taps it publishes beyond a plain
    /// decode; the decode started again in the target's launch. That
    /// launch is measured without the switch's price, and the switch by what
    /// its round took beyond the price of its launches; a preparing round's
    /// launch is its switch alone. A launch that recorded its graph
    /// (`duration` `None`) measured that recording, not itself: it carries
    /// its round on and prices nothing. A launch the expert bank's cover read
    /// beside prices its round until a launch measured with no cover read
    /// beside it shows the cover ended: then every duration measured beside
    /// it is dropped ([`LaunchCosts::settle`]), and the rounds measure their
    /// launches anew.
    pub fn timed(
        &mut self,
        launch: TimedLaunch,
        duration: Option<LaunchDuration>,
        sequences: usize,
        rows: usize,
    ) {
        let covering = duration.is_some_and(|duration| duration.covering);
        let seconds = duration.map(|duration| duration.duration.as_secs_f64());
        let mut load = self.load.lock();
        match duration {
            Some(LaunchDuration { covering: true, .. }) => load.beside_cover = true,
            Some(_) => load.settle(),
            None => {}
        }
        let size = match (launch, self.timing.expected) {
            (TimedLaunch::Entry, _) => {
                self.timing.entered.record(seconds);
                return;
            }
            (TimedLaunch::Draft, ExpectedLaunch::Draft(generated)) => {
                self.timing.expected = ExpectedLaunch::None;
                generated
            }
            (TimedLaunch::Target, ExpectedLaunch::Target) => {
                self.timing.expected = ExpectedLaunch::None;
                rows
            }
            _ => return,
        };
        self.sequences = sequences.max(1);
        let drafting = self.choice.drafting(self.preparing());
        let generated_for_steady = if self.timing.drafter.ran() {
            self.choice.generated
        } else {
            0
        };
        let (switch, switch_cost, steady) = {
            let costs = RoundCosts::of(
                &load,
                self.context_class,
                self.sequences,
                self.after_drafting,
            );
            let switch = costs.switching(drafting);
            let switch_cost = switch.and_then(|s| costs.switch(s));
            let steady = if launch == TimedLaunch::Target {
                costs.steady(generated_for_steady, size)
            } else {
                None
            };
            (switch, switch_cost, steady)
        };
        // The switch's price, when this launch paid one. While the switch is
        // unmeasured, a drafter launch that paid it is not measured: its
        // duration would carry the discarded decode, a switch measured
        // against it would come out as nothing, and drafting measured only
        // as it enters (as looks from plain decoding are) would keep that
        // price for good. A target launch that paid it is measured if it
        // has no other measurement: it carries only the restarted pipeline,
        // and the plain decode is taken again until the median of its
        // durations leaves that one out.
        let paid = |paying: Switch, target: Option<&TargetCosts>, size: usize| {
            if switch != Some(paying) {
                return Some(0.0);
            }
            switch_cost.or_else(|| {
                (paying == Switch::Leave && target.is_none_or(|t| !t.verify.contains_key(&size))).then_some(0.0)
            })
        };
        let took = self.timing.entered.take_with(seconds);
        if launch == TimedLaunch::Target {
            load.rounds += 1;
        }
        let round = load.rounds;
        let draft_key = (self.context_class, self.sequences);
        let regime = load.regime_epoch();
        let target_key = (regime, self.context_class, self.sequences);
        if launch == TimedLaunch::Draft {
            self.timing.drafter = match took {
                Some(seconds) => DrafterTiming::Replayed(seconds),
                None => DrafterTiming::Recorded,
            };
            if let (Some(took), Some(paid)) = (took, paid(Switch::Enter, load.target_costs.get(&target_key), size)) {
                let drafts = load.draft_costs.entry(draft_key).or_default();
                LaunchCosts::record(drafts, round, size, (took - paid).max(0.0), covering);
            }
            prune_draft_costs(&mut load.draft_costs, draft_key);
            return;
        }
        let drafter = std::mem::take(&mut self.timing.drafter);
        self.after_drafting = Some(drafting);
        // A launch that recorded its graph measured that, not itself.
        let Some(took) = took else {
            return;
        };
        let streamed = load.verify_prior.is_some();
        let target = load.target_costs.entry(target_key).or_default();
        if let (Some(switch), Some(steady), Some(drafter_took)) =
            (switch, steady, drafter.seconds())
        {
            target
                .switch_mut(switch)
                .record(round, (took + drafter_took - steady).max(0.0), covering);
        }
        if self.preparing() {
            prune_target_costs(&mut load.target_costs, target_key);
            return;
        }
        let Some(paid) = paid(Switch::Leave, Some(target), size) else {
            prune_target_costs(&mut load.target_costs, target_key);
            return;
        };
        let seconds = (took - paid).max(0.0);
        let scaled_factor = match (size, self.choice.generated) {
            // On streamed MoE, plain decode latency varies 2-4x from token to
            // token with miss-count noise rather than GPU clock scaling: record
            // the plain duration directly without rescaling `verify(rows > 1)`
            // or `draft_costs` by `plain_new / plain_old`.
            (1, 0) if streamed => {
                LaunchCosts::record(&mut target.verify, round, 1, seconds, covering);
                None
            }
            (1, 0) => target.record_plain(round, seconds, covering),
            // A drafting round that verified nothing decoded one row behind
            // its drafter's launch, not in a decode pipeline.
            (1, _) => None,
            (rows, _) => {
                LaunchCosts::record(&mut target.verify, round, rows, seconds, covering);
                None
            }
        };
        if let Some(factor) = scaled_factor {
            if let Some(drafts) = load.draft_costs.get_mut(&draft_key) {
                for durations in drafts.values_mut() {
                    durations.scale(factor);
                }
            }
        }
        prune_target_costs(&mut load.target_costs, target_key);
    }
}

#[cfg(test)]
mod tests;
