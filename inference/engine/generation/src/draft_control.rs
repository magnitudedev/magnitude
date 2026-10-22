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
//! Requests verified in one launch share its rows ([`share_rows`]).

use magnitude_executor::{draft_ladder, TokenId};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
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
/// The rounds of a load after which a launch's durations are stale: a
/// launch measured again after them is priced by its new durations alone,
/// and a plain decode not taken for that long is taken again, so a price
/// left by a busier or freer machine lasts no longer.
const STALE_ROUNDS: u64 = 100;
/// The share of decode time spent on looks.
const LOOK_SHARE: f64 = 0.01;

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
/// round that did starts the decode pipeline again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Switch {
    Enter,
    Leave,
}

/// A launch's latest durations in seconds, oldest first, each with the
/// load's round it was measured in.
#[derive(Clone, Debug, Default, PartialEq)]
struct Durations(Vec<(u64, f64)>);

impl Durations {
    fn record(&mut self, round: u64, seconds: f64) {
        self.0
            .retain(|(measured, _)| round.saturating_sub(*measured) < STALE_ROUNDS);
        if self.0.len() == COST_SAMPLES {
            self.0.remove(0);
        }
        self.0.push((round, seconds));
    }

    /// The launch's cost: the median of its durations, the lower of two.
    fn cost(&self) -> f64 {
        let mut sorted = self
            .0
            .iter()
            .map(|(_, seconds)| *seconds)
            .collect::<Vec<_>>();
        sorted.sort_by(f64::total_cmp);
        sorted[(sorted.len() - 1) / 2]
    }

    fn measured(&self) -> bool {
        !self.0.is_empty()
    }

    /// Whether the cost rests on its full share of durations.
    fn proven(&self) -> bool {
        self.0.len() == COST_SAMPLES
    }

    fn latest(&self) -> u64 {
        self.0.last().map_or(0, |(round, _)| *round)
    }

    fn scale(&mut self, factor: f64) {
        for (_, seconds) in &mut self.0 {
            *seconds *= factor;
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
    fn record(costs: &mut BTreeMap<usize, Durations>, round: u64, size: usize, seconds: f64) {
        costs.entry(size).or_default().record(round, seconds);
    }

    /// A plain launch (one target row) took `seconds`. The machine's speed
    /// moves every launch alike (another process taking the CPU, or giving
    /// it back), so each other measurement moves with the plain one's cost:
    /// a launch no longer taken keeps its cost relative to a plain decode
    /// rather than the seconds it took on a machine that was busier or
    /// freer, until it is measured again.
    fn record_plain(&mut self, round: u64, seconds: f64) {
        let before = self.verify.get(&1).map(Durations::cost);
        Self::record(&mut self.verify, round, 1, seconds);
        let Some(before) = before.filter(|before| *before > 0.0) else {
            return;
        };
        let factor = self.verify[&1].cost() / before;
        for (_, durations) in self.verify.iter_mut().filter(|(rows, _)| **rows != 1) {
            durations.scale(factor);
        }
        for durations in self.draft.values_mut() {
            durations.scale(factor);
        }
        self.enter.scale(factor);
        self.leave.scale(factor);
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

/// The launch costs of one context class and number of requests sharing a
/// launch. A launch not measured there costs at least what its other
/// measurements prove, and at least what the same launch measured over no
/// longer a history and among no more requests: neither makes it cheaper.
///
/// A launch not proven there (measured fewer times than its cost needs, or
/// not at all) costs no more than a larger launch proven there, nor than the
/// same launch proven among as many requests over a shorter history times
/// the ratio of the context classes (only its attention reads a longer
/// history, and reads it at most that many times as long). One held-up
/// duration would otherwise price it out of being taken, and so of ever
/// being measured again.
struct RoundCosts {
    own: LaunchCosts,
    /// Measurements at no longer a context and of no more requests.
    lesser: Vec<LaunchCosts>,
    /// Measurements of as many requests at a shorter context, with how many
    /// times longer this context class is.
    shorter: Vec<(f64, LaunchCosts)>,
    /// Requests sharing each launch.
    sequences: usize,
    after_drafting: Option<bool>,
    switch_weight: f64,
}

impl RoundCosts {
    fn of(
        load: &DraftLoad,
        context_class: usize,
        sequences: usize,
        after_drafting: Option<bool>,
    ) -> Self {
        let costs = load.costs.lock().expect("draft load costs");
        Self {
            own: costs
                .get(&(context_class, sequences))
                .cloned()
                .unwrap_or_default(),
            lesser: costs
                .iter()
                .filter(|((context, shared), _)| {
                    *context <= context_class
                        && *shared <= sequences
                        && (*context, *shared) != (context_class, sequences)
                })
                .map(|(_, costs)| costs.clone())
                .collect(),
            shorter: costs
                .iter()
                .filter(|((context, shared), _)| *context < context_class && *shared == sequences)
                .map(|((context, _), costs)| (context_class as f64 / *context as f64, costs.clone()))
                .collect(),
            sequences,
            after_drafting,
            switch_weight: 0.0,
        }
    }

    /// These costs with a round that would switch paying `weight` of the
    /// switch (see [`Self::round`]).
    fn sharing_switches(self, weight: f64) -> Self {
        Self {
            switch_weight: weight,
            ..self
        }
    }

    fn seconds(&self, costs: fn(&LaunchCosts) -> &BTreeMap<usize, Durations>, size: usize) -> f64 {
        let own = costs(&self.own);
        let unproven = match own.get(&size) {
            Some(durations) if durations.proven() => return durations.cost(),
            Some(durations) => durations.cost(),
            None => std::iter::once(&self.own)
                .chain(&self.lesser)
                .map(|measured| LaunchCosts::at_least(costs(measured), size))
                .fold(0.0, f64::max),
        };
        let larger = own
            .range(size + 1..)
            .filter(|(_, larger)| larger.proven())
            .map(|(_, larger)| larger.cost());
        let shorter = self.shorter.iter().filter_map(|(ratio, measured)| {
            costs(measured)
                .get(&size)
                .filter(|durations| durations.proven())
                .map(|durations| ratio * durations.cost())
        });
        larger.chain(shorter).fold(unproven, f64::min)
    }

    fn draft(&self, generated: usize) -> f64 {
        if generated == 0 {
            0.0
        } else {
            self.seconds(|costs| &costs.draft, generated)
        }
    }

    fn verify(&self, rows: usize) -> f64 {
        self.seconds(|costs| &costs.verify, rows)
    }

    /// What changing between drafting and not costs a round beyond its
    /// launches: measured here, or else the most it measured at no longer a
    /// context (it enters pairs and restarts a pipeline, not attention).
    fn switch(&self, switch: Switch) -> Option<f64> {
        let own = self.own.switch(switch);
        if own.measured() {
            return Some(own.cost());
        }
        self.lesser
            .iter()
            .map(|costs| costs.switch(switch))
            .filter(|durations| durations.measured())
            .map(Durations::cost)
            .reduce(f64::max)
    }

    /// The switch a round generating `generated` makes from the request's
    /// previous round.
    fn switching(&self, generated: usize) -> Option<Switch> {
        match (self.after_drafting, generated > 0) {
            (Some(false), true) => Some(Switch::Enter),
            (Some(true), false) => Some(Switch::Leave),
            _ => None,
        }
    }

    /// Seconds of a round generating `generated` and verifying `verified`,
    /// in launches whose other requests take the same rows, with its share
    /// of what changing between drafting and not costs. A switch serves the
    /// rounds after it too: the choice it follows lasts at least as many
    /// rounds as the estimates take to move, so a round that would switch
    /// pays the weight a block has in them of it, and nothing until it is
    /// measured. Launches nothing has measured cost the same, so a load's
    /// first round is its widest.
    fn round(&self, generated: usize, verified: usize) -> f64 {
        let switch = self
            .switching(generated)
            .and_then(|switch| self.switch(switch))
            .map_or(0.0, |switch| self.switch_weight * switch);
        (self.draft(generated) + self.verify(self.sequences * (verified + 1)) + switch).max(1e-9)
    }

    /// What a round generating `generated` that is a look pays for its
    /// switches beyond its share: a look after a round that drafted nothing
    /// drafts for that round alone, so it enters and leaves again whole.
    fn excursion(&self, generated: usize) -> f64 {
        if self.switching(generated) != Some(Switch::Enter) {
            return 0.0;
        }
        let enter = self.switch(Switch::Enter).unwrap_or(0.0);
        let leave = self.switch(Switch::Leave).unwrap_or(0.0);
        (1.0 - self.switch_weight) * enter + leave
    }

    /// The price of a round's launches when both are measured here.
    fn steady(&self, generated: usize, rows: usize) -> Option<f64> {
        let drafted = generated == 0 || self.own.draft.contains_key(&generated);
        (drafted && self.own.verify.contains_key(&rows))
            .then(|| self.draft(generated) + self.verify(rows))
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
    let costs = RoundCosts::of(&outlooks[0].load, context_class, requests, None);
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

/// What a load's requests share: launch costs by context class and sequence
/// count, the acceptance means its latest request reached, and the decode
/// time set aside for the next look. That time carries from request to
/// request, so a load of requests each too short to earn a look by itself
/// still takes them at their share of the decode time, and drafting that
/// stopped paying is tried again.
#[derive(Debug, Default)]
pub struct DraftLoad {
    costs: Mutex<BTreeMap<(usize, usize), LaunchCosts>>,
    rounds: Mutex<u64>,
    depths: Mutex<Option<Vec<DepthMeans>>>,
    /// The run of accepted rounds that earned its latest request a look.
    look_run: Mutex<Option<u32>>,
    look_seconds: Mutex<f64>,
}

/// A drafted block being scored against the tokens committed after it.
#[derive(Clone, Debug, PartialEq)]
struct Block {
    tokens: Vec<TokenId>,
    matched: usize,
    /// Drafted by a look taken on a run of accepted rounds.
    run_look: bool,
}

/// One round's choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RoundChoice {
    pub generated: usize,
    pub verified: usize,
    /// Whether the round drafted wider than its fastest pair, to score the
    /// deeper proposals.
    pub look: bool,
}

/// One request's draft width controller.
#[derive(Clone, Debug)]
pub struct DraftController {
    /// Candidate widths, increasing, without 0.
    ladder: Vec<usize>,
    load: Arc<DraftLoad>,
    depths: Vec<DepthMeans>,
    scored: Vec<bool>,
    blocks: Vec<Block>,
    /// Seconds of decode time set aside for the next look (see
    /// [`DraftLoad`]).
    look_seconds: f64,
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
    /// The launches this round's durations belong to.
    timing_draft: Option<usize>,
    timing_target: bool,
    /// What this round's entries took; `None` when one recorded its graph.
    entered: Option<f64>,
    /// This round's drafter launch, with what it took when it replayed its
    /// graph.
    drafter: Option<Option<f64>>,
    after_drafting: Option<bool>,
    chosen: Option<bool>,
    /// The length of this round's drafted block, until it is verified.
    drafted: Option<usize>,
    context_class: usize,
    sequences: usize,
    choice: RoundChoice,
}

impl DraftController {
    /// A controller choosing among widths up to `widest`, from the load's
    /// latest acceptance means. A load that has scored nothing takes every
    /// proposal as right.
    pub fn new(widest: usize, load: Arc<DraftLoad>) -> Self {
        let seed = load.depths.lock().expect("draft load acceptance").clone();
        let look_run = load.look_run.lock().expect("draft load looks").unwrap_or(1);
        let look_seconds = *load.look_seconds.lock().expect("draft load look time");
        let depths = seed
            .filter(|depths| depths.len() == widest)
            .unwrap_or_else(|| vec![DepthMeans::UNSCORED; widest]);
        Self {
            ladder: draft_ladder(widest),
            load,
            depths,
            scored: vec![false; widest],
            blocks: Vec::new(),
            look_seconds,
            accepted_run: 0,
            look_run,
            run_look: false,
            kept: None,
            timing_draft: None,
            timing_target: false,
            entered: Some(0.0),
            drafter: None,
            after_drafting: None,
            chosen: None,
            drafted: None,
            context_class: 1,
            sequences: 1,
            choice: RoundChoice::default(),
        }
    }

    pub fn choice(&self) -> RoundChoice {
        self.choice
    }

    /// How often a block is expected to be right through each depth, from
    /// depth 1 (see [`DepthMeans`]). A block right through a depth is right
    /// through every shallower one, so no depth is rated above a shallower
    /// one.
    pub fn right(&self) -> Vec<f64> {
        let mut bound = 1.0f64;
        self.depths
            .iter()
            .map(|depth| {
                bound = bound.min(depth.estimate());
                bound
            })
            .collect()
    }

    /// The measured launch seconds at this round's context class.
    pub fn measured(&self) -> LaunchCosts {
        self.costs().own
    }

    /// The costs this round chooses by. A plain decode taken to measure it
    /// changes no choice: a load drafting goes on drafting after it, so the
    /// round coming back from it is priced as one continuing to draft. A
    /// switch is shared by the weight a block has in the first depth's
    /// estimate, which every drafting round depends on.
    fn costs(&self) -> RoundCosts {
        RoundCosts::of(&self.load, self.context_class, self.sequences, self.chosen)
            .sharing_switches(self.depths.first().map_or(MEMORY, DepthMeans::weight))
    }

    /// This round's drafted block, while it awaits verification.
    pub fn outlook(&self) -> Option<DraftOutlook> {
        let drafted = self.drafted?;
        Some(DraftOutlook {
            load: self.load.clone(),
            context_class: self.context_class,
            right: self.right()[..drafted].to_vec(),
            draft_seconds: self.costs().draft(self.choice.generated),
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
    /// anchored at target position `position`.
    pub fn generate(&mut self, limit: usize, position: usize) -> usize {
        self.context_class = position.max(1).next_power_of_two();
        // Entries before a round chooses are no round's: the prompt's, or
        // pairs held past their bound by rounds that drafted nothing.
        self.entered = Some(0.0);
        let costs = self.costs();
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
        // The plain decode is the one launch no look takes, so while it is
        // measured here but its cost is unproven or stale it is taken again
        // in place of drafting: a duration another process held up, or a
        // machine since freed, would otherwise price it for good.
        let plain_unsettled = fastest > 0 && {
            let round = *self.load.rounds.lock().expect("draft load rounds");
            costs.own.verify.get(&1).is_some_and(|plain| {
                !plain.proven() || round.saturating_sub(plain.latest()) >= STALE_ROUNDS
            })
        };
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
                self.look_seconds += LOOK_SHARE * gain.min(1.0) * costs.round(fastest, verified);
                self.run_look = false;
                if self.look_seconds >= lost {
                    self.look_seconds = 0.0;
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
        let generated = if plain_unsettled { 0 } else { generated };
        if !plain_unsettled {
            self.chosen = Some(generated > 0);
        }
        *self.load.look_seconds.lock().expect("draft load look time") = self.look_seconds;
        self.choice = RoundChoice {
            generated,
            verified: 0,
            look: generated != kept,
        };
        self.timing_draft = (generated > 0).then_some(generated);
        self.timing_target = generated == 0;
        generated
    }

    /// The block the drafter returned for a round that generated
    /// `generated`; a block cut short is wrong from where it ends.
    pub fn drafted(&mut self, generated: usize, block: &[TokenId]) {
        for depth in block.len() + 1..=generated {
            self.score(depth, false);
        }
        if !block.is_empty() {
            self.blocks.push(Block {
                tokens: block.to_vec(),
                matched: 0,
                run_look: self.run_look,
            });
        }
        self.drafted = Some(block.len());
    }

    /// The proposals of a drafted block of `drafted` this round verifies, of
    /// at most `limit`. The drafter's launch is paid whatever is verified.
    pub fn verify(&mut self, drafted: usize, limit: usize) -> usize {
        let (verified, _) = self.verified(
            &self.right(),
            &self.costs(),
            self.choice.generated,
            limit.min(drafted),
        );
        self.choice.verified = verified;
        self.timing_target = true;
        self.drafted = None;
        verified
    }

    fn score(&mut self, depth: usize, right: bool) {
        self.depths[depth - 1].score(right);
        self.scored[depth - 1] = true;
    }

    /// Score every open block against the tokens just committed, which end
    /// the round: a depth no block scored in it goes unscored for a round.
    pub fn committed(&mut self, tokens: &[TokenId]) {
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
                if block.tokens[block.matched] == *token {
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
        for (depth, scored) in self.depths.iter_mut().zip(&mut self.scored) {
            if !std::mem::take(scored) {
                depth.unscored();
            }
        }
        *self.load.depths.lock().expect("draft load acceptance") = Some(self.depths.clone());
        *self.load.look_run.lock().expect("draft load looks") = Some(self.look_run);
    }

    /// A launch of this request finished in `duration`: `rows` rows of
    /// `sequences` requests. A drafter launch is measured by the proposals
    /// it generated, a target launch by its rows. The drafter's entries for
    /// a round are part of its proposing launch, or of the target's when the
    /// round proposed nothing. A switch between drafting and not (see
    /// [`Switch`]) is paid in the launch that follows it: the discarded
    /// decode and the catch-up before the drafter proposes, the decode
    /// started again in the target's launch. That launch is measured without
    /// the switch's price, and the switch by what its round took beyond the
    /// price of its launches. A launch that recorded its graph (`duration`
    /// `None`) measured that recording, not itself: it carries its round on
    /// and prices nothing.
    pub fn timed(
        &mut self,
        launch: TimedLaunch,
        duration: Option<Duration>,
        sequences: usize,
        rows: usize,
    ) {
        let seconds = duration.map(|duration| duration.as_secs_f64());
        let size = match launch {
            TimedLaunch::Entry => {
                self.entered = self.entered.zip(seconds).map(|(entered, seconds)| entered + seconds);
                return;
            }
            TimedLaunch::Draft => self.timing_draft.take(),
            TimedLaunch::Target => std::mem::take(&mut self.timing_target).then_some(rows),
        };
        let Some(size) = size else {
            return;
        };
        self.sequences = sequences.max(1);
        let costs = RoundCosts::of(
            &self.load,
            self.context_class,
            self.sequences,
            self.after_drafting,
        );
        let switch = costs.switching(self.choice.generated);
        // The switch's price, when this launch paid one. While the switch is
        // unmeasured, a drafter launch that paid it is not measured: its
        // duration would carry the discarded decode, a switch measured
        // against it would come out as nothing, and drafting measured only
        // as it enters (as looks from plain decoding are) would keep that
        // price for good. A target launch that paid it is measured if it
        // has no other measurement: it carries only the restarted pipeline,
        // and the plain decode is taken again until the median of its
        // durations leaves that one out.
        let paid = |paying: Switch, launches: &LaunchCosts, size: usize| {
            if switch != Some(paying) {
                return Some(0.0);
            }
            costs.switch(paying).or_else(|| {
                (paying == Switch::Leave && !launches.verify.contains_key(&size)).then_some(0.0)
            })
        };
        let took = seconds
            .zip(self.entered.replace(0.0))
            .map(|(seconds, entered)| seconds + entered);
        let round = {
            let mut rounds = self.load.rounds.lock().expect("draft load rounds");
            if launch == TimedLaunch::Target {
                *rounds += 1;
            }
            *rounds
        };
        let mut measured = self.load.costs.lock().expect("draft load costs");
        let launches = measured
            .entry((self.context_class, self.sequences))
            .or_default();
        if launch == TimedLaunch::Draft {
            self.drafter = Some(took);
            if let (Some(took), Some(paid)) = (took, paid(Switch::Enter, launches, size)) {
                LaunchCosts::record(&mut launches.draft, round, size, (took - paid).max(0.0));
            }
            return;
        }
        let drafter = self.drafter.take();
        self.after_drafting = Some(self.choice.generated > 0);
        let generated = if drafter.is_some() {
            self.choice.generated
        } else {
            0
        };
        // A launch that recorded its graph measured that, not itself.
        let Some(took) = took else {
            return;
        };
        let drafter_took = drafter.map_or(Some(0.0), |drafter| drafter);
        if let (Some(switch), Some(steady), Some(drafter_took)) =
            (switch, costs.steady(generated, size), drafter_took)
        {
            launches
                .switch_mut(switch)
                .record(round, (took + drafter_took - steady).max(0.0));
        }
        let Some(paid) = paid(Switch::Leave, launches, size) else {
            return;
        };
        let seconds = (took - paid).max(0.0);
        match (size, self.choice.generated) {
            (1, 0) => launches.record_plain(round, seconds),
            // A drafting round that verified nothing decoded one row behind
            // its drafter's launch, not in a decode pipeline.
            (1, _) => {}
            (rows, _) => LaunchCosts::record(&mut launches.verify, round, rows, seconds),
        }
    }
}

#[cfg(test)]
mod tests;
