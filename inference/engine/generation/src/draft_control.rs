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
//! fixed share of the decode time pays for that amount.
//!
//! Requests verified in one launch share its rows ([`share_rows`]).

use magnitude_executor::{TokenId, draft_ladder};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
};

/// The weight of a new block in a depth's estimate once the depth has had
/// its share of blocks. A depth scored every round follows the text within a
/// few rounds; one scored only by looks moves as slowly as they come.
const MEMORY: f64 = 0.25;
/// The weight of a new launch duration once a launch has been measured.
const COST_MEMORY: f64 = 0.25;
/// The share of decode time spent on looks.
const LOOK_SHARE: f64 = 0.01;

/// Which launch of a round a duration belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimedLaunch {
    /// The drafter's launch.
    Draft,
    /// The target's launch.
    Target,
}

/// Measured launch seconds at one context class and sequence count.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LaunchCosts {
    /// By generated proposals.
    draft: BTreeMap<usize, f64>,
    /// By target rows.
    verify: BTreeMap<usize, f64>,
}

impl LaunchCosts {
    fn record(costs: &mut BTreeMap<usize, f64>, size: usize, seconds: f64) {
        costs
            .entry(size)
            .and_modify(|mean| *mean += COST_MEMORY * (seconds - *mean))
            .or_insert(seconds);
    }

    /// The most the measurements prove a launch of `size` costs at least: a
    /// larger launch costs no less than a smaller one. A larger launch
    /// proves nothing about a smaller one (a 16-row launch costs more than
    /// twice an 8-row one where the rows cross a class).
    fn at_least(costs: &BTreeMap<usize, f64>, size: usize) -> f64 {
        costs
            .range(..=size)
            .map(|(_, seconds)| *seconds)
            .fold(0.0, f64::max)
    }
}

/// The launch costs of one context class and number of requests sharing a
/// launch. A launch not measured there costs at least what its other
/// measurements prove, and at least what the same launch measured over no
/// longer a history and among no more requests: neither makes it cheaper.
struct RoundCosts {
    own: LaunchCosts,
    /// Measurements at no longer a context and of no more requests.
    lesser: Vec<LaunchCosts>,
    /// Requests sharing each launch.
    sequences: usize,
}

impl RoundCosts {
    fn of(load: &DraftLoad, context_class: usize, sequences: usize) -> Self {
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
            sequences,
        }
    }

    fn seconds(&self, costs: fn(&LaunchCosts) -> &BTreeMap<usize, f64>, size: usize) -> f64 {
        match costs(&self.own).get(&size) {
            Some(seconds) => *seconds,
            None => std::iter::once(&self.own)
                .chain(&self.lesser)
                .map(|measured| LaunchCosts::at_least(costs(measured), size))
                .fold(0.0, f64::max),
        }
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

    /// Seconds of a round generating `generated` and verifying `verified`,
    /// in launches whose other requests take the same rows. Launches nothing
    /// has measured cost the same, so a load's first round is its widest.
    fn round(&self, generated: usize, verified: usize) -> f64 {
        (self.draft(generated) + self.verify(self.sequences * (verified + 1))).max(1e-9)
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
    let costs = RoundCosts::of(&outlooks[0].load, context_class, requests);
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
/// count, the acceptance its latest request reached, and the decode time set
/// aside for the next look. That time carries from request to request, so a
/// load of requests each too short to earn a look by itself still takes them
/// at their share of the decode time, and drafting that stopped paying is
/// tried again.
#[derive(Debug, Default)]
pub struct DraftLoad {
    costs: Mutex<BTreeMap<(usize, usize), LaunchCosts>>,
    right: Mutex<Option<Vec<f64>>>,
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
    /// `right[k - 1]`: how often a block is right through depth `k`.
    right: Vec<f64>,
    /// Blocks behind each depth's estimate, counted up to its share.
    blocks_scored: Vec<u32>,
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
    /// The length of this round's drafted block, until it is verified.
    drafted: Option<usize>,
    context_class: usize,
    sequences: usize,
    choice: RoundChoice,
}

impl DraftController {
    /// A controller choosing among widths up to `widest`. A load that has
    /// scored nothing takes every proposal as right.
    pub fn new(widest: usize, load: Arc<DraftLoad>) -> Self {
        let seed = load.right.lock().expect("draft load acceptance").clone();
        let look_run = load.look_run.lock().expect("draft load looks").unwrap_or(1);
        let look_seconds = *load.look_seconds.lock().expect("draft load look time");
        let (right, scored) = match seed {
            Some(right) if right.len() == widest => (right, u32::MAX),
            _ => (vec![1.0; widest], 0),
        };
        Self {
            ladder: draft_ladder(widest),
            load,
            right,
            blocks_scored: vec![scored; widest],
            blocks: Vec::new(),
            look_seconds,
            accepted_run: 0,
            look_run,
            run_look: false,
            kept: None,
            timing_draft: None,
            timing_target: false,
            drafted: None,
            context_class: 1,
            sequences: 1,
            choice: RoundChoice::default(),
        }
    }

    pub fn choice(&self) -> RoundChoice {
        self.choice
    }

    /// How often a block is right through each depth, from depth 1. A block
    /// right through a depth is right through every shallower one, so no
    /// depth is rated above a shallower one.
    pub fn right(&self) -> Vec<f64> {
        let mut bound = 1.0f64;
        self.right
            .iter()
            .map(|right| {
                bound = bound.min(*right);
                bound
            })
            .collect()
    }

    /// The measured launch seconds at this round's context class.
    pub fn measured(&self) -> LaunchCosts {
        self.costs().own
    }

    fn costs(&self) -> RoundCosts {
        RoundCosts::of(&self.load, self.context_class, self.sequences)
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
        // How much faster than the fastest pair a width could be at best:
        // its rate if every proposal it drafts were right, over the fastest
        // pair's, less one.
        // And what a round generating it loses to the fastest pair: its
        // time beyond what its tokens take at the fastest pair's rate.
        let all_right = vec![1.0; right.len()];
        let wider = self
            .widths(limit)
            .filter(|generated| *generated > fastest)
            .map(|generated| {
                let (_, best) = self.verified(&all_right, &costs, generated, limit);
                let (verified, _) = self.verified(&right, &costs, generated, limit);
                let lost = costs.round(generated, verified) - Self::tokens(&right, verified) / rate;
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
        // The mean is plain until a depth has its share of blocks, so a
        // load's first block replaces what it assumed.
        let scored = &mut self.blocks_scored[depth - 1];
        *scored = scored.saturating_add(1);
        let weight = (1.0 / f64::from(*scored)).max(MEMORY);
        let estimate = &mut self.right[depth - 1];
        *estimate += weight * (f64::from(u8::from(right)) - *estimate);
    }

    /// Score every open block against the tokens just committed.
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
        *self.load.right.lock().expect("draft load acceptance") = Some(self.right.clone());
        *self.load.look_run.lock().expect("draft load looks") = Some(self.look_run);
    }

    /// A launch of this request finished in `duration`: `rows` rows of
    /// `sequences` requests. A drafter launch is measured by the proposals
    /// it generated, a target launch by its rows.
    pub fn timed(
        &mut self,
        launch: TimedLaunch,
        duration: Duration,
        sequences: usize,
        rows: usize,
    ) {
        let size = match launch {
            TimedLaunch::Draft => self.timing_draft.take(),
            TimedLaunch::Target => std::mem::take(&mut self.timing_target).then_some(rows),
        };
        let Some(size) = size else {
            return;
        };
        self.sequences = sequences.max(1);
        let mut costs = self.load.costs.lock().expect("draft load costs");
        let costs = costs
            .entry((self.context_class, self.sequences))
            .or_default();
        match launch {
            TimedLaunch::Draft => {
                LaunchCosts::record(&mut costs.draft, size, duration.as_secs_f64())
            }
            TimedLaunch::Target => {
                LaunchCosts::record(&mut costs.verify, size, duration.as_secs_f64())
            }
        }
    }
}

#[cfg(test)]
mod tests;
