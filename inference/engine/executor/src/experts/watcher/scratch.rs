//! Scratch requests (`routed_scratch_request`): a prompt group's experts
//! served through records it holds. Each request is taken as its step's
//! front publishes it, in step order: of the experts no earlier step of the
//! group routed at its layer, the hits, those the layer's map holds for the
//! step as the device's pass resolves them (`StampedMap::resolve`), are
//! read where the bank holds them, and the rest (the cold experts) read
//! through the conveyor into the conveyor's records, each once per group,
//! at once or as soon as a record frees (`Scratch::convey`), and converted
//! by the rounds into the scratch, slot `e` of the layer's view holding
//! expert `e`, which the pass reads it from (`routed_experts`' `expert_*`).
//! Each request is served once its first round arrives: once the group holds
//! every record the scratch lies in, then converted round by round, each
//! round once the experts it lists are held whole in the conveyor. A group runs as a wavefront: a step's front at a layer follows
//! its back at the layer before, so a layer's requests are taken while the
//! layer before is served, and every one before the layer's first round.
//!
//! No expert memory is a prompt group's alone: the conveyor and the
//! scratch lie in the bottom records of their lender (`planning::ScratchPlan`,
//! `planning::ConveyorCarrier`): on unified memory both in the one store's,
//! beside a discrete device the conveyor in the host tier's and the scratch
//! in the lender's bank records, the pass's. The group claims them all as
//! its first request is taken (`Directory::claim_lent`, `Directory::lend`)
//! and returns them empty, so it takes no other layer's records: every
//! layer keeps its decode experts through it. Beside a host tier its last
//! step seeds the bank, copying its experts' slots into records the bank's
//! ranking gives them (`Directory::seed`); on the one store nothing seeds.
//! A claim takes no record a step that may still resolve it names: the
//! group runs after every step begun before it, a claimed record's expert
//! is unmapped from the group's first step on, and the group's steps read
//! the map only at a pass (`routed_pass_map`), so what they may read is
//! what the host counted as hits. The lent records are claimed before any
//! hit is taken, so none lies in one; a seed that takes a hit of the next
//! layer, whose rounds have not arrived, makes it a read
//! ([`Scratch::demote`]). An expert placed meanwhile (a promotion's, an earlier group's seed) is
//! mapped from the first step not yet begun, after every step of the
//! group, which all begin together: the group's steps resolve it absent,
//! so their requests read it.
//!
//! A cold expert the host tier keeps whole is not read and takes no
//! conveyor record: its request's round 0 answer lists it, (tier record,
//! scratch slot), and the fill's tier copy after round 0 reads the tier
//! record, in the resident form, in place into its scratch slot. The tier
//! record is held from that answer until the device is past the group. The
//! answer is host work into host memory, as every answer of a device wait
//! is: the device does the moving, in work already in the graph behind the
//! wait. The rounds convert only the experts read from the file, and their
//! lists name only those.

use super::core::{At, Core, Listed, Taken};
use super::prefetch::Prefetch;
use super::telemetry::{Anticipation, GroupLayer, Mark, RequestCost};
use super::{Shared, Step};
use crate::experts::{ListBounds, Words};
use crate::experts::compute::Urgency;
use crate::experts::conveyor::{Conveyor, ConveyorLanding, Serving, State, readiness};
use crate::experts::directory::{BankRecord, HostRecord, Point};
use crate::experts::landing::Converting;
use crate::experts::staging::Staged;
use seismic_io::Want;
use magnitude_kernels::constants;
use std::time::Instant;

/// A step's request at the tier's layer, taken as it was published: the
/// experts it routed, and of those no earlier step of the group routed
/// there, the hits its map holds for the step (each with that record,
/// which its pass reads) and the rest, which its rounds convert.
pub(super) struct Request {
    pub(super) sequence: Step,
    routed: Vec<usize>,
    hits: Vec<(usize, usize)>,
    cold: Vec<usize>,
}

/// A layer's scratch tier contents: every expert of layer `layer` that
/// group `group`'s steps routed to (`routed[e]`, their choices of expert
/// `e`), whose cold ones their requests fill. The device's tier holds one
/// layer at a time, filled by the layer's backs; the host holds the layer served and
/// the next, whose requests are taken meanwhile, each until the device is
/// past the layer (`Scratch::advance`).
struct Tier {
    group: Step,
    layer: usize,
    routed: Vec<u32>,
    /// Per expert, the last step of the group that routed it here.
    last: Vec<Step>,
    /// The requests taken, one per chain place in step order, until served.
    taken: usize,
    requests: Vec<Option<Request>>,
    /// Per expert, whether it holds its scratch slot (slot `e` for expert
    /// `e`, once the group holds the scratch), which its rounds convert it
    /// into and the pass reads it from.
    scratch: Vec<bool>,
    /// The cold experts not yet read (none carries them): no record was
    /// free for them in the device's order (`Scratch::convey`).
    deferred: Vec<usize>,
    /// Per expert, the step whose request reads it (cold), if any.
    reader: Vec<Option<Step>>,
    /// Per expert, the round of its reader's request that lists it, once
    /// that request is served.
    round: Vec<Option<u32>>,
    /// The cold experts whose read failed, with why: the step
    /// that reads each fails at its round, at once.
    failed: Vec<(usize, String)>,
}

/// The record step `step`'s pass reads `layer`'s `expert` in, as the device
/// resolves its map: a hit, none where it reads it from its scratch slot.
/// Never a record lent to the step's group (`lending`, the conveyor's): a
/// group takes its lent records before it takes a hit, unmapping what they
/// held from its first step, and nothing maps into one while it holds it;
/// beside a host tier the bank's records assert it
/// (`BankRecords::resolve`), on the one store its lent host records.
fn hit(core: &Core, lending: &crate::experts::conveyor::Lending, layer: usize, expert: usize, step: Step) -> Option<usize> {
    match core.extents.placer {
        crate::planning::Placer::Tier => core.directory.resolve(layer, expert, step).map(BankRecord::index),
        crate::planning::Placer::Bank => {
            let record = core.layers[layer].map.resolve(expert, step)?;
            assert!(
                core.layers[layer].group != lending.group || record >= lending.records,
                "step {step} resolves expert {expert} of layer {layer} in lent record {record}"
            );
            Some(record)
        }
    }
}

/// Whether a prompt group's cold expert can come from the host tier: only
/// where the device places from one, whose tier copy reads it. On the one
/// store no tier copy binds (`BankBinding::tier` is none), and a landing
/// into the store is mapped past every step already begun, so its rounds
/// read each cold expert from the file, even one the store keeps by now.
fn copies_from_tier(core: &Core) -> bool {
    core.extents.placer == crate::planning::Placer::Tier
}

impl Tier {
    /// Step `reader`'s request reads expert `expert` from now, its rounds
    /// not yet served: the one rule for every way an expert becomes a
    /// step's to read (its request taken, a hit demoted, a failed step's
    /// expert taken over). It waits for a record unless the conveyor
    /// carries it already (`carried`: a failed step's read, landed or on
    /// its way), which its rounds then list: an expert is read at most once
    /// while carried.
    fn assign(&mut self, expert: usize, reader: Step, carried: bool) {
        self.reader[expert] = Some(reader);
        self.round[expert] = None;
        if !carried && !self.deferred.contains(&expert) {
            self.deferred.push(expert);
        }
    }

    /// No step reads expert `expert` from now: it waits for no record, and
    /// whatever the conveyor carries of it is any earlier need's to take.
    fn unassign(&mut self, expert: usize) {
        self.reader[expert] = None;
        self.deferred.retain(|&deferred| deferred != expert);
    }

    /// Where expert `expert` lies in the device's order of the tier's
    /// experts: by the step that reads it, then by the round that lists it
    /// (a request's experts all unlisted until it is served); one no step
    /// reads lies after every other.
    fn order(&self, expert: usize) -> (bool, Option<Step>, Option<u32>) {
        (self.reader[expert].is_none(), self.reader[expert], self.round[expert])
    }
}

/// A request's cold experts, `cold` in the order they are ready, composed
/// into rounds of `round_records` in that order, but that an expert whose
/// slot carries another of them (`borrowers`, by `cold`'s place:
/// `Scratch::lendable_slot`) goes in a round after that borrower's, which
/// converts it out of the slot first. Only where a round cannot be filled
/// otherwise does such an expert join its borrower's round, whose answer
/// then has the borrower yield the slot (`Scratch::reclaim_slots`).
fn compose_rounds(cold: &[usize], borrowers: &[Option<usize>], round_records: usize) -> Vec<usize> {
    let mut placed = vec![false; cold.iter().max().map_or(0, |&expert| expert + 1)];
    let mut composed = Vec::with_capacity(cold.len());
    let mut rest = cold.iter().copied().zip(borrowers.iter().copied()).collect::<Vec<_>>();
    while !rest.is_empty() {
        let size = round_records.min(rest.len());
        let (mut round, mut blocked) = (Vec::with_capacity(size), Vec::new());
        for &(expert, borrower) in &rest {
            match borrower.is_none_or(|other| placed[other]) {
                true if round.len() < size => round.push(expert),
                true => {}
                false => blocked.push(expert),
            }
        }
        round.extend(blocked.iter().take(size - round.len()));
        for &expert in &round {
            placed[expert] = true;
        }
        rest.retain(|&(expert, _)| !placed[expert]);
        composed.extend(round);
    }
    composed
}

/// The load's scratch words: each chain place's flags, each scratch
/// round's convert list, the tier list and the seed list.
pub(super) struct ScratchWords {
    pub(super) flags: Vec<Words>,
    pub(super) lists: Vec<Words>,
    pub(super) tiered: Words,
    pub(super) seeds: Words,
}

/// Each layer's conveyor form, of layers whose routed tensors read the
/// pools `layers` names (of `pools`): the class of layers sharing a pool
/// with it, directly or through others. Record `e` of a pool holds one
/// layer's expert `e` at a time, so layers that share any pool share its
/// records: a Q4_K_M layer's gate and up pools are those of layers whose
/// down pools differ.
pub(super) fn conveyor_forms(pools: usize, layers: &[Vec<usize>]) -> Vec<usize> {
    let mut root = (0..pools).collect::<Vec<_>>();
    fn find(root: &mut [usize], pool: usize) -> usize {
        let mut at = pool;
        while root[at] != at {
            root[at] = root[root[at]];
            at = root[at];
        }
        at
    }
    for layer in layers {
        for &pool in &layer[1..] {
            let (a, b) = (find(&mut root, layer[0]), find(&mut root, pool));
            root[a] = b;
        }
    }
    let mut forms: Vec<usize> = Vec::new();
    layers
        .iter()
        .map(|layer| {
            let class = find(&mut root, layer[0]);
            forms.iter().position(|&form| form == class).unwrap_or_else(|| {
                forms.push(class);
                forms.len() - 1
            })
        })
        .collect()
}

/// Fail step `sequence`'s scratch request of `layer`: the failure is
/// recorded, then published in every lane of its slot not yet answered,
/// the request's and each round's (the last round's among them, which its
/// seed commit reads).
pub(super) fn fail(core: &Core, shared: &Shared, layer: usize, sequence: Step, reason: String) {
    shared.steps.fail(sequence, reason);
    let scratch = &core.layers[layer].scratch;
    for lane in 0..scratch.table().lanes() {
        scratch.answer(sequence, lane, true);
    }
    shared.parks.settle();
}

/// The lane of round `round` of a step's slot of a layer's scratch table.
pub(super) fn round_lane(round: u64) -> u64 {
    constants::SCRATCH_ROUND + round
}

/// The scratch requests taken and served, the tiers and the conveyor.
pub(super) struct Scratch {
    /// Per layer, the last request taken.
    published: Vec<Step>,
    words: ScratchWords,
    /// The tiers of the layers a group's requests were taken at and the
    /// device is not yet past: the layer served and the next, at most.
    tiers: Vec<Tier>,
    /// The host tier records the group's tier copies read, (layer, record),
    /// held until the device is past the group.
    held: Vec<(usize, HostRecord)>,
    pub(super) conveyor: Conveyor,
}

/// A request taken, from its taking until its last round is answered or it
/// fails, as the request service carries it, never waiting: arriving until
/// its round 0 does, then served.
pub(super) enum Pending {
    /// Taken at its point from its chain place, its round 0 not yet arrived.
    Arriving { at: At, place: usize },
    Served(Serve),
}

impl Pending {
    /// Where it is, for the exchange's watchdog: its step, place and
    /// stage, and of its rounds the one served now, what it awaits and
    /// since when.
    pub(super) fn describe(&self) -> String {
        match self {
            Self::Arriving { at, place } => format!("step {} place {place} arriving (round 0 not yet requested)", at.point.sequence),
            Self::Served(Serve { at, place, arrived, stage }) => {
                let stage = match stage {
                    RequestStage::Arrived { request } => format!("arrived (cold {:?})", request.cold),
                    RequestStage::Awaiting(opened) => format!("awaiting landings and the scratch for {:?}", opened.cold),
                    RequestStage::Rounds(rounds) => format!(
                        "round {} of {}, cold {:?}, tiered {:?}, waiting {:?}, failure {:?}",
                        rounds.round,
                        rounds.rounds,
                        rounds.cold,
                        rounds.tiered,
                        rounds.waited.elapsed(),
                        rounds.failure
                    ),
                };
                format!("step {} place {place} arrived {:?} ago, {stage}", at.point.sequence, arrived.elapsed())
            }
        }
    }
}

/// A request whose round 0 arrived, being served ([`Scratch::carry`]).
pub(super) struct Serve {
    at: At,
    place: usize,
    /// When its round 0 arrived.
    arrived: Instant,
    stage: RequestStage,
}

/// Where a served request is, each stage waiting on its own events.
enum RequestStage {
    /// Its round 0 arrived.
    Arrived { request: Request },
    /// Opened (and seeded, by a group's last step): its cold experts a read
    /// ahead lands into the host tier are awaited, then its rounds fixed.
    Awaiting(Opened),
    /// Its rounds, one at a time, each answered once its experts are whole.
    Rounds(Rounds),
}

/// A request opened ([`RequestStage::Awaiting`]).
struct Opened {
    routed: Vec<usize>,
    hits: Vec<(usize, usize)>,
    cold: Vec<usize>,
}

/// A request's rounds ([`RequestStage::Rounds`]): its cold experts read from the
/// file, a round's records at a time; those the host tier keeps whole,
/// which the tier copy after round 0 reads; the round served now, since
/// `waited`; and what it counts.
struct Rounds {
    hits: Vec<(usize, usize)>,
    cold: Vec<usize>,
    tiered: Vec<usize>,
    rounds: u64,
    round: u64,
    waited: Instant,
    served: RequestCost,
    took: GroupLayer,
    seeded: u64,
    failure: Option<String>,
}

impl Scratch {
    pub(super) fn new(layers: usize, words: ScratchWords, conveyor: Conveyor) -> Self {
        Self {
            published: vec![0; layers],
            words,
            tiers: Vec::new(),
            held: Vec::new(),
            conveyor,
        }
    }

    /// The bank holds `flags`, each place's, from now: between flights,
    /// every step retired, so no tier holds a request.
    pub(super) fn place(&mut self, flags: Vec<Words>) {
        self.words.flags = flags;
    }

    /// Layer `layer`'s earliest request published and not yet taken, at its
    /// chain place ([`Self::publish`] takes it): none while every one is, is
    /// of a step ended (read now: steps end beside the request service; one
    /// a device error ended, which no launch waits for), or a step before it
    /// in its group is not yet taken there. A group's chain places are its
    /// steps in order, so the next is the tier's next place at the layer, or
    /// the first of a later group.
    pub(super) fn published(&self, core: &Core, shared: &Shared, layer: usize, layers: usize) -> Option<(usize, At)> {
        let tier = self.tiers.iter().find(|tier| tier.layer == layer);
        let scratch = &core.layers[layer].scratch;
        let live = shared.steps.live();
        let unpublished = live.start.max(self.published[layer] + 1)..live.end;
        let sequence = scratch.earliest_requested(unpublished, constants::SCRATCH_ASK, |step| !shared.steps.ended(step))?;
        let at = shared.steps.at(sequence, layer, layers);
        let place = usize::try_from(sequence - at.point.group).expect("a group's places");
        let taken = match tier {
            Some(tier) if tier.group == at.point.group => tier.taken,
            _ => 0,
        };
        (place == taken).then_some((place, at))
    }

    /// The conveyor's group and each scratch tier, for the exchange's
    /// watchdog: its group and layer, the requests taken, whether open, the
    /// experts deferred and failed, and where each of its routed experts
    /// is on its way into the conveyor.
    pub(super) fn describe(&self) -> String {
        let tiers = self
            .tiers
            .iter()
            .map(|tier| {
                let mut ways = [0usize; 4];
                for expert in (0..tier.routed.len()).filter(|&expert| tier.routed[expert] > 0) {
                    ways[match self.conveyor.state(tier.layer, expert) {
                        None => 0,
                        Some(State::Reading { .. }) => 1,
                        Some(State::Held { .. }) => 2,
                        Some(State::Listed { .. }) => 3,
                    }] += 1;
                }
                format!(
                    "group {} layer {}: taken {}, deferred {:?}, failed {:?}, routed (none, reading, held, listed) {ways:?}",
                    tier.group,
                    tier.layer,
                    tier.taken,
                    tier.deferred,
                    tier.failed.iter().map(|(expert, _)| expert).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        format!("conveyor serving {:?}; scratch tiers [{}]", self.conveyor.serving, tiers.join("; "))
    }

    /// The highest layer a request of group `group` (its first step) was
    /// taken at, whose hits are decided; none before its first. No later
    /// group may have begun: its requests would count as the group's.
    pub(super) fn decided(&self, group: Step) -> Option<usize> {
        self.published.iter().rposition(|&sequence| sequence >= group)
    }

    /// Whether a request of group `group` taken at `layer` routed `expert`:
    /// its later steps' passes then resolve it as the first one's did,
    /// reading a hit where that one's was, whose record must hold it for
    /// every step of the group. Only while the device is not past the layer
    /// (its tier kept, `Self::advance`): past it, none of the group's passes
    /// reads the layer again.
    pub(super) fn routed(&self, group: Step, layer: usize, expert: usize) -> bool {
        self.tiers.iter().any(|tier| (tier.group, tier.layer) == (group, layer) && tier.routed[expert] > 0)
    }

    /// Whether the conveyor still clears the group the device is past: no
    /// later group's request is taken before ([`Self::clear_past`]).
    pub(super) fn clearing(&self) -> bool {
        self.conveyor.serving.is_some_and(|serving| serving.past)
    }

    /// Layer `layer`'s next request taken and not yet served, in step
    /// order: its chain place and step, left in its tier until taken
    /// ([`Self::take`]); none while there is none, those of steps ended
    /// (read now) dropped.
    pub(super) fn unserved(&mut self, shared: &Shared, layer: usize) -> Option<(usize, Step)> {
        let tier = self.tiers.iter_mut().find(|tier| tier.layer == layer)?;
        loop {
            let place = tier.requests.iter().position(Option::is_some)?;
            let sequence = tier.requests[place].as_ref().expect("a taken request").sequence;
            if !shared.steps.ended(sequence) {
                return Some((place, sequence));
            }
            tier.requests[place] = None;
        }
    }

    /// Take group `group`'s request at `layer` from chain place `place` to
    /// serve it: its tier's index and the request. None once the tier went
    /// with its layer (`advance`): the device is past the layer, which only
    /// a step that failed before it passes without its rounds answered, so
    /// nothing waits for the request.
    pub(super) fn take(&mut self, group: Step, layer: usize, place: usize) -> Option<(usize, Request)> {
        let index = self.tiers.iter().position(|tier| (tier.group, tier.layer) == (group, layer))?;
        let request = self.tiers[index].requests[place].take()?;
        Some((index, request))
    }

    /// The device is at `progress`: forget the tiers of the layers the
    /// served group is past (their backs read no record any more), free
    /// the conveyor's records of those layers, and once it is past the
    /// group, clear the conveyor as soon as nothing of the group runs
    /// ([`Self::clear_past`]).
    pub(super) fn advance(&mut self, progress: Point, core: &mut Core) {
        let Some(serving) = self.conveyor.serving.as_mut() else {
            return;
        };
        if progress >= Point::after(serving.group) {
            serving.past = true;
        }
        let serving = *serving;
        // A tier the device is past holds nothing: its slots are the
        // scratch's, which the group returns as it clears.
        self.tiers.retain(|tier| {
            progress
                <= Point {
                    group: tier.group,
                    layer: tier.layer,
                    sequence: serving.last,
                }
        });
        if !serving.past {
            // A layer every step of the group is past lists none of the
            // records carrying its experts.
            self.conveyor.release_past(|layer| {
                progress
                    > Point {
                        group: serving.group,
                        layer,
                        sequence: serving.last,
                    }
            });
            return;
        }
        self.clear_past(core);
    }

    /// The group served is past: once no read or landing of it runs (a
    /// failed step's reads, which no round listed, the conveyor's landings
    /// into the host tier, each an event as it ends), empty the conveyor,
    /// release the tier records its rounds
    /// held and return the lender's records empty. Nothing takes what
    /// those ran for, so a failed one's outcome is no step's. Until then
    /// nothing is done, and the next group's requests wait
    /// ([`Self::clearing`]). Whether it cleared.
    pub(super) fn clear_past(&mut self, core: &mut Core) -> bool {
        if !self.conveyor.serving.is_some_and(|serving| serving.past) {
            return false;
        }
        // Its landings from the conveyor end before a record takes another
        // read, each learned before a decode request asks the tier for its
        // expert.
        self.settle_landings(core);
        if !self.conveyor.quiet() {
            return false;
        }
        self.conveyor.clear();
        // The device is past every tier copy of the group.
        for (layer, record) in self.held.drain(..) {
            core.directory.unhold_tier(layer, record);
        }
        core.directory.end_prompt_group(self.conveyor.lending().group);
        self.conveyor.serving = None;
        true
    }

    /// `mark` came at `at` at `layer` of the group the conveyor serves
    /// (`Telemetry::mark`).
    fn mark_at(&self, shared: &Shared, layer: usize, mark: Mark, at: Instant) {
        let serving = self.conveyor.serving.expect("a group's event comes while the conveyor serves it");
        shared.telemetry.mark(serving.group, serving.last, layer, mark, at);
    }

    /// `mark` came now at `layer` of the group the conveyor serves.
    fn mark(&self, shared: &Shared, layer: usize, mark: Mark) {
        self.mark_at(shared, layer, mark, Instant::now());
    }

    /// Serve group `group`'s scratch requests from the conveyor, from its
    /// first: beside a host tier the bank's lent records are the group's
    /// from now (`Directory::lend`), before any hit is taken, and its
    /// conveyor claims the lender's host records as it takes them
    /// ([`Self::claim_lent`]); every one returns once the device is past
    /// the group.
    fn serve_group(&mut self, group: Step, core: &mut Core, shared: &Shared) {
        match self.conveyor.serving {
            Some(serving) if serving.group == group => return,
            Some(serving) => panic!("the conveyor serves group {} past it", serving.group),
            None => {}
        }
        core.directory.lend(group, &shared.lent.lock());
        let (_, steps) = shared.steps.group_of(group);
        self.conveyor.serving = Some(Serving {
            group,
            last: group + steps - 1,
            past: false,
            claimed: false,
        });
    }

    /// Take the request `at` its point from chain place `place`, as its
    /// step's front published it, the device past its own step's back at
    /// the layer before: of the experts it is the group's first to route
    /// there, pin the bank's hits, read in place, and read the rest into their
    /// conveyor records, as demand reads behind those taken before it
    /// (`Scratch::convey`).
    pub(super) fn publish(&mut self, at: At, place: usize, core: &mut Core, shared: &Shared) {
        let Point { group, layer, sequence } = at.point;
        let experts = core.layers[layer].experts;
        self.published[layer] = sequence;
        self.serve_group(group, core, shared);
        self.mark(shared, layer, Mark::Requested(sequence));
        // The conveyor's records are the group's before any hit is taken,
        // so no tier pins one.
        self.claim_lent(core);
        let index = match self.tiers.iter().position(|tier| (tier.group, tier.layer) == (group, layer)) {
            Some(index) => index,
            None => {
                self.tiers.push(Tier {
                    group,
                    layer,
                    routed: vec![0; experts],
                    last: vec![0; experts],
                    taken: 0,
                    requests: (0..self.words.flags.len()).map(|_| None).collect(),
                    scratch: vec![false; experts],
                    deferred: Vec::new(),
                    reader: vec![None; experts],
                    round: vec![None; experts],
                    failed: Vec::new(),
                });
                self.tiers.len() - 1
            }
        };
        let tier = &mut self.tiers[index];
        let flags = (0..experts)
            .map(|expert| u32::try_from(self.words.flags[place].load(expert)).expect("a routed count"))
            .collect::<Vec<_>>();
        let routed = (0..experts).filter(|&expert| flags[expert] > 0).collect::<Vec<_>>();
        let first = routed
            .iter()
            .copied()
            .filter(|&expert| tier.routed[expert] == 0)
            .collect::<Vec<_>>();
        for ((count, last), flag) in tier.routed.iter_mut().zip(&mut tier.last).zip(&flags) {
            *count += flag;
            if *flag > 0 {
                *last = sequence;
            }
        }
        let tier = &mut self.tiers[index];
        let (mut hits, mut cold) = (Vec::new(), Vec::new());
        for expert in first {
            match hit(core, self.conveyor.lending(), layer, expert, sequence) {
                Some(record) => hits.push((expert, record)),
                None => cold.push(expert),
            }
        }
        // The group reads its hits at this step, which the ranking logs once
        // the group ends: until then every taking ranks each beside it, so
        // none while the group runs prefers it to what no prompt read.
        for &(expert, _) in &hits {
            core.directory.read_pending(layer, expert, sequence);
        }
        for &expert in &cold {
            tier.assign(expert, sequence, self.conveyor.state(layer, expert).is_some());
        }
        tier.requests[place] = Some(Request {
            sequence,
            routed,
            hits,
            cold,
        });
        tier.taken += 1;
        self.convey(core, shared);
    }

    /// Carry every tier's cold experts as far as records allow, in the
    /// device's order (the `Conveyor`'s rule): by tier, then by the step
    /// that reads each expert, then by the round that lists it
    /// (`Tier::order`). Free the records of the rounds the device has
    /// converted (`Core::take_converted`); settle the reads that ended,
    /// landing each into the host tier where there is one
    /// (`Self::land_conveyed`); and read each
    /// deferred expert into a free conveyor record of its form whose lent
    /// records it may claim. An expert takes a record only while those free exceed
    /// what the experts before it in that order still lack, so no record
    /// goes to an expert the device needs after one still waiting for it;
    /// and the earliest expert waiting, none free, takes the record of the
    /// latest expert after it that holds one unlisted (held whole), which is
    /// read again. So the expert the device waits on always progresses. A
    /// read that fails, or an engine that takes no job, fails the step that
    /// reads the expert,
    /// with why, and stays its tier's (`Tier::failed`) until a later step
    /// takes it over. Deferred experts the host tier keeps whole take no
    /// record: the fill copies them from the tier in place. Whether a read
    /// or landing it settled had ended, or a round's records freed as the
    /// device converted them: completed progress.
    pub(super) fn convey(&mut self, core: &mut Core, shared: &Shared) -> bool {
        // A group past carries nothing more: what of it runs ends, and the
        // conveyor clears (`Self::clear_past`).
        if !self.conveyor.serving.is_some_and(|serving| !serving.past) {
            return false;
        }
        // The records of the rounds the device has converted free first.
        let converted = core.take_converted();
        let freed = !converted.is_empty();
        for listed in converted {
            listed.release(&mut self.conveyor);
        }
        let settled = self.conveyor.settle(|ticket| ticket.wait_until(Instant::now()));
        for &(layer, ..) in &settled.landed {
            self.mark(shared, layer, Mark::Landed);
        }
        let landed = self.settle_landings(core);
        let ended = freed || settled.ended + landed > 0;
        self.land_conveyed(core, shared, settled.landed);
        let mut failed = settled.failed;
        self.claim_lent(core);
        let knee = core.table.source().knee();
        for index in 0..self.tiers.len() {
            let layer = self.tiers[index].layer;
            // What the earlier tiers of the layer's form still lack: their
            // experts not yet read, a conveyor record each.
            let lack_read = self.tiers[..index]
                .iter()
                .filter(|tier| self.conveyor.shares_form(tier.layer, layer))
                .map(|tier| tier.deferred.iter().filter(|&&expert| !self.waits(core, tier, expert)).count())
                .sum::<usize>();
            // The conveyor records free of the layer's form whose lent
            // records the group holds claimed.
            let admits = |conveyor: &Conveyor, core: &Core, record: u32| {
                let lending = conveyor.lending();
                core.layers[layer].stored.iter().flat_map(|stored| lending.records(stored.pool, record)).all(|lent| {
                    core.directory.lent_claimed(lending.group, HostRecord(u32::try_from(lent).expect("lent records fit u32")))
                })
            };
            let free_conveyor =
                |conveyor: &Conveyor, core: &Core| conveyor.free_conveyor(layer, |record| admits(conveyor, core, record));
            let mut deferred = std::mem::take(&mut self.tiers[index].deferred);
            let tier = &self.tiers[index];
            deferred.sort_by_key(|&expert| tier.order(expert));
            // Those the host tier keeps whole, or lands, take no record.
            let (waiting, deferred): (Vec<_>, Vec<_>) =
                deferred.into_iter().partition(|&expert| self.waits(core, &self.tiers[index], expert));
            let mut kept = waiting;
            let mut blocked = false;
            for expert in deferred {
                if blocked {
                    kept.push(expert);
                    continue;
                }
                // A scratch slot no step uses before the expert's round
                // converts it carries it first, at no conveyor record, while
                // the drive lacks work, its reads in flight short of its
                // link's knee: past that a read only queues, ahead of
                // whatever the device turns out to need sooner.
                let short_of_work = self.conveyor.reads_in_flight() as u64 * core.layers[layer].expert_bytes() < knee;
                let lent = if short_of_work { self.lendable_slot(core, index, expert) } else { None };
                let record = match lent {
                    Some(slot) => self.conveyor.slot_record(slot),
                    None => {
                        let mut records = free_conveyor(&self.conveyor, core);
                        if records.is_empty() && lack_read == 0 {
                            let holds = |state: &State| {
                                matches!(*state, State::Held { record } if self.conveyor.slot_of(record).is_none() && admits(&self.conveyor, core, record))
                            };
                            if let Some((at, other)) = self.latest_after(index, expert, holds) {
                                self.yield_record(at, other);
                                records = free_conveyor(&self.conveyor, core);
                            }
                        }
                        let Some(&record) = records.get(lack_read) else {
                            blocked = true;
                            kept.push(expert);
                            continue;
                        };
                        record
                    }
                };
                let reads = core.reads_to(layer, expert, &self.conveyor, record);
                match core.table.source().request(reads, Want::now()) {
                    Ok(ticket) => {
                        // Its end is an event: a round may await it.
                        let events = shared.events().clone();
                        ticket.on_end(move |_, _| events.notify());
                        self.conveyor.take(layer, expert, record);
                        self.conveyor.reading(layer, expert, ticket, record);
                        self.mark(shared, layer, Mark::Submitted);
                    }
                    Err(reason) => failed.push((layer, expert, reason)),
                }
            }
            // Behind what this tier's later experts yielded meanwhile.
            self.tiers[index].deferred.extend(kept);
        }
        // Each failed expert stays its tier's, failing the step that reads
        // it: at once, if its round is being served, else at its round 0.
        for (layer, expert, reason) in failed {
            if let Some(tier) = self.tiers.iter_mut().find(|tier| tier.layer == layer) {
                if let Some(reader) = tier.reader[expert] {
                    shared.steps.fail(reader, reason.clone());
                }
                tier.failed.push((expert, reason));
            }
        }
        ended
    }

    /// The scratch slot that may carry deferred expert `expert` of tier
    /// `index` until its round converts it out, if one may: where the host
    /// addresses the scratch, once the group holds it, a slot no expert
    /// carries that nothing writes or reads before that round.
    /// - The layers before this one still served must lay their slots in
    ///   its bytes (one scratch form) and have every step's request taken,
    ///   so their use is known: a slot of one of their cold experts is
    ///   converted into and read by their passes until the device is past
    ///   their layer.
    /// - This layer converts into a slot only in its own expert's round,
    ///   where that expert is cold, which must follow the borrower's: a
    ///   later step's does; of the borrower's own step, a later round once
    ///   the rounds are fixed, or, before, the round [`compose_rounds`]
    ///   gives an expert no record carries yet, after its slot's borrower.
    ///   That last is taken only where no slot is free outright, for a step
    ///   of more than one round, and by a borrower whose own slot no expert
    ///   of its step borrows, so composing binds one way only.
    /// - A later layer converts only after this layer's pass.
    ///
    /// [`Self::reclaim_slots`] holds a round whose slot's borrower could not
    /// be ordered first.
    fn lendable_slot(&self, core: &Core, index: usize, expert: usize) -> Option<usize> {
        let tier = &self.tiers[index];
        let reader = tier.reader[expert]?;
        let serving = self.conveyor.serving?;
        if core.layers[tier.layer].slots.is_none() || !self.scratch_claimed(core) {
            return None;
        }
        let steps = (serving.last - serving.group + 1) as usize;
        let form = &core.layers[tier.layer].scratch_form;
        let before = || self.tiers.iter().filter(|other| (other.group, other.layer) < (tier.group, tier.layer));
        if before().any(|other| other.taken < steps || core.layers[other.layer].scratch_form != *form) {
            return None;
        }
        let read_before = |slot: usize| {
            before().any(|other| other.scratch.get(slot).is_some_and(|&held| held) || other.reader.get(slot).is_some_and(Option::is_some))
        };
        // Whether its rounds may still be composed to put its slot's
        // expert after it: they are not yet fixed, they are more than one,
        // and no expert of them borrows its own slot (so none is both a
        // borrower and a slot's expert, and composing binds only one way).
        let composable = tier.round[expert].is_none()
            && tier.reader.iter().filter(|&&other| other == Some(reader)).count() > core.extents.round_records
            && !self.conveyor.borrower(expert).is_some_and(|(layer, other)| layer == tier.layer && tier.reader[other] == Some(reader));
        // Whether this layer converts into `slot` only after the expert's
        // round: outright, or once its rounds are composed (the slot taken
        // only where none is free outright).
        let after = |slot: usize| match (tier.reader[slot], tier.round[slot]) {
            _ if slot == expert => None,
            (None, _) => (!tier.scratch[slot]).then_some(false),
            (Some(step), _) if step != reader => (step > reader).then_some(false),
            (Some(_), Some(round)) => tier.round[expert].is_some_and(|own| own < round).then_some(false),
            (Some(_), None) => (composable && self.conveyor.state(tier.layer, slot).is_none()).then_some(true),
        };
        (0..core.layers[tier.layer].experts.min(self.conveyor.slots()))
            .filter(|&slot| self.conveyor.borrower(slot).is_none() && !read_before(slot))
            .filter_map(|slot| Some((after(slot)?, slot)))
            .min()
            .map(|(_, slot)| slot)
    }

    /// Whether round experts `chunk` of `layer` may be converted into their
    /// slots now: none's slot carries stored bytes a convert after it would
    /// read. A slot an expert listed already carries is read by a round the
    /// device runs before (a layer's rounds are answered in device order);
    /// one an expert holds unlisted yields it (it is read again elsewhere);
    /// one being read holds the round until its read ends. A slot is lent
    /// and its rounds composed to need none of this
    /// ([`Self::lendable_slot`], [`compose_rounds`]), but for a round
    /// that could not be filled otherwise, or a failed step's experts moved
    /// to later steps.
    fn reclaim_slots(&mut self, chunk: &[usize]) -> bool {
        let mut free = true;
        for &slot in chunk {
            let Some((layer, expert)) = self.conveyor.borrower(slot) else {
                continue;
            };
            match self.conveyor.state(layer, expert) {
                Some(State::Listed { .. }) => {}
                Some(State::Held { .. }) => {
                    let index = self.tiers.iter().position(|tier| tier.layer == layer).expect("a carried expert's tier");
                    self.yield_record(index, expert);
                }
                Some(State::Reading { .. }) => free = false,
                None => unreachable!("scratch slot {slot} carries expert {expert} of layer {layer}, which the conveyor does not"),
            }
        }
        free
    }

    /// Whether deferred expert `expert` of `tier` comes from the host tier,
    /// taking no record: beside a host tier (where the device places), the
    /// tier keeps it whole, or a landing into it is in flight (a read
    /// ahead's, `Scratch::awaited`), which the tier copy reads once landed.
    /// No expert is read twice.
    fn waits(&self, core: &Core, tier: &Tier, expert: usize) -> bool {
        copies_from_tier(core) && (core.directory.in_tier(tier.layer, expert) || core.directory.lands(tier.layer, expert))
    }

    /// Whether the conveyor carries `staged`: a read of it, or its record,
    /// is the group's, which lands it into the host tier where it is read
    /// ([`Self::land_conveyed`]); the tier's fill reads none of these.
    pub(super) fn carries(&self, (layer, expert): Staged) -> bool {
        self.conveyor.state(layer, expert).is_some()
    }

    /// The deferred experts a step waits for while a landing into the host
    /// tier lands them ([`Self::waits`]): a read ahead's, which the watcher
    /// promotes to demand (`Prefetch::hasten`).
    pub(super) fn awaited<'a>(&'a self, core: &'a Core) -> impl Iterator<Item = Staged> + 'a {
        self.tiers.iter().flat_map(move |tier| {
            tier.deferred
                .iter()
                .filter(move |&&expert| copies_from_tier(core) && core.directory.lands(tier.layer, expert))
                .map(move |&expert| (tier.layer, expert))
        })
    }

    /// Beside a host tier (where the device places), land each expert just
    /// read whole into the conveyor, `landed` (layer, expert, conveyor
    /// record), into its group's host tier from its conveyor record, where
    /// the tier lacks it and nothing lands it: a prompt group's read is the
    /// tier's landing too (`experts::landing`, as every landing), so neither
    /// the tier's fill nor a later miss reads it again. It takes a ready
    /// record as a demand landing does (`Directory::claim_landing` leaving no
    /// reserve, which the tier refills by reuse), so the prompt's experts
    /// displace the tier's least reused; with none ready it lands nothing.
    /// The conveyor record takes no read until its landing ends
    /// (`Conveyor::free_conveyor`).
    fn land_conveyed(&mut self, core: &mut Core, shared: &Shared, landed: Vec<(usize, usize, u32)>) {
        if !copies_from_tier(core) {
            return;
        }
        for (layer, expert, record) in landed {
            let crate::experts::host_records::Claim::Taken(grant) = core.directory.claim_landing(layer, expert, 0) else {
                continue;
            };
            let tier = grant.record();
            let pools = self.conveyor.pools();
            let landing = core.landing_from(grant, |pool| (&pools[pool], u64::from(record)));
            let ended = crate::experts::landing::Ended::new(shared.events());
            // SAFETY: the read landed whole in conveyor record `record`,
            // which no read lands in until this landing ends; the tier
            // record was ready (no step reads it) when taken.
            let converting = unsafe {
                Converting::new(landing, Urgency::waited(), shared.landings.clone(), ended.clone(), || {})
            };
            shared.compute.submit(std::sync::Arc::new(converting));
            let read = self.tiers.iter().find(|tier| tier.layer == layer).and_then(|tier| tier.reader[expert]);
            self.conveyor.land(ConveyorLanding { layer, expert, record, tier, read, ended });
        }
    }

    /// Learn the conveyor landings that ended: each tier record keeps its
    /// expert. How many ended.
    fn settle_landings(&mut self, core: &mut Core) -> usize {
        let landed = self.conveyor.landed();
        for landing in &landed {
            core.directory.landed(landing.layer, HostRecord(landing.tier), landing.expert, landing.read);
        }
        landed.len()
    }

    /// Of the experts of tier `index`'s form, from that tier on, that the
    /// device needs after its expert `expert` and that hold a record
    /// `holds` accepts, the one it needs last: its tier's index and expert.
    fn latest_after(&self, index: usize, expert: usize, holds: impl Fn(&State) -> bool) -> Option<(usize, usize)> {
        let layer = self.tiers[index].layer;
        let after = (index, self.tiers[index].order(expert));
        self.tiers
            .iter()
            .enumerate()
            .skip(index)
            .filter(|(_, tier)| self.conveyor.shares_form(tier.layer, layer))
            .flat_map(|(at, tier)| (0..tier.reader.len()).map(move |other| ((at, tier.order(other)), at, other)))
            .filter(|&(order, at, other)| {
                order > after && self.conveyor.state(self.tiers[at].layer, other).is_some_and(&holds)
            })
            .max_by_key(|&(order, ..)| order)
            .map(|(_, at, other)| (at, other))
    }

    /// Expert `expert` of tier `index` yields the record it holds unlisted
    /// (`Conveyor::yield_record`): it waits for one again, where a step
    /// reads it.
    fn yield_record(&mut self, index: usize, expert: usize) {
        let tier = &mut self.tiers[index];
        self.conveyor.yield_record(tier.layer, expert);
        if tier.reader[expert].is_some() {
            tier.deferred.push(expert);
        }
    }

    /// Claim, for the group served, each host record the conveyor's
    /// lending lends it that it may claim now (`Directory::claim_lent`):
    /// the conveyor takes a record only once every lent record it lies
    /// across is the group's. Its first request's claim comes before any
    /// hit is taken, and what lands after is mapped past the group, so no
    /// hit lies in a lent record ([`hit`]). A record still held (a copy
    /// reading it or a landing into it) is claimed on a later pass. Each
    /// expert a claim unmaps is unmapped for every step of the group on,
    /// which the device runs after every step that read its record, before
    /// any read lands in it. The bank's lent records beside a host tier
    /// are the group's from its service's start (`Directory::lend`).
    fn claim_lent(&mut self, core: &mut Core) {
        let serving = self.conveyor.serving.expect("the conveyor claims for the group it serves");
        if serving.claimed {
            return;
        }
        let lending = self.conveyor.lending();
        let lent = u32::try_from(lending.records).expect("lent records fit u32");
        // Every record claimed now, each a claim of its own.
        let claims = (0..lent).map(|record| core.directory.claim_lent(lending.group, HostRecord(record), serving.group)).collect::<Vec<_>>();
        if claims.iter().all(|&claimed| claimed) {
            self.conveyor.serving = Some(Serving { claimed: true, ..serving });
        }
    }

    /// Whether the group served holds every record the scratch lies in: its
    /// slots take no convert before. Beside a host tier they are the bank's
    /// lent records, the group's from its service's start
    /// ([`Self::serve_group`]); on the one store the lent host records,
    /// once every one is claimed.
    fn scratch_claimed(&self, core: &Core) -> bool {
        match core.extents.placer {
            crate::planning::Placer::Tier => self.conveyor.serving.is_some(),
            crate::planning::Placer::Bank => self.conveyor.serving.is_some_and(|serving| serving.claimed),
        }
    }

    /// Give each of `experts` of tier `index` its scratch slot, slot `e`
    /// for expert `e` of the scratch the lender lends the group, once the
    /// group holds every record it lies in: whether every one has its slot.
    fn claim_scratch(&mut self, index: usize, experts: &[usize], core: &mut Core) -> bool {
        if !self.scratch_claimed(core) {
            self.claim_lent(core);
            if !self.scratch_claimed(core) {
                return false;
            }
        }
        for &expert in experts {
            self.tiers[index].scratch[expert] = true;
        }
        true
    }

    /// Take the request `at` its point from chain place `place`, its round
    /// 0 arrived at `arrived`, the device past every point before it: the
    /// request served from now ([`Self::carry`]). None once its tier went
    /// with its layer ([`Self::take`]): nothing waits for it.
    pub(super) fn arrived(&mut self, at: At, place: usize, arrived: Instant) -> Option<Serve> {
        let (_, request) = self.take(at.point.group, at.point.layer, place)?;
        Some(Serve { at, place, arrived, stage: RequestStage::Arrived { request } })
    }

    /// The step of the request `at` its point from chain place `place`
    /// ended before its round 0 arrived (a device error): its experts are
    /// the next routing step's, and its failure is recorded and published.
    pub(super) fn unarrived(&mut self, at: At, place: usize, core: &Core, shared: &Shared) {
        let Point { group, layer, sequence } = at.point;
        if let Some((index, request)) = self.take(group, layer, place) {
            self.forfeit(index, place, request.experts(), core);
        }
        fail(core, shared, layer, sequence, format!("step {sequence} ended before its scratch round 0"));
    }

    /// Carry `serve` as far as it goes without waiting, each stage on its
    /// own events: its seeding, by a group's last step; the landings into
    /// the host tier of its cold experts (their ends); then each round,
    /// answered once the experts it lists are whole in their conveyor
    /// records (reads' ends, the queue's completions), its records free
    /// once the device arrives at the round after. The request, none once
    /// served whole or failed, and whether it moved.
    pub(super) fn carry(&mut self, serve: Serve, core: &mut Core, prefetch: &mut Prefetch, shared: &Shared) -> (Option<Serve>, bool) {
        let Serve { at, place, arrived, mut stage } = serve;
        let Point { group, layer, .. } = at.point;
        let mut moved = false;
        loop {
            // The tier went with its layer (`advance`): the device is past
            // it, which only a step that failed passes without its rounds
            // answered, so nothing waits.
            let Some(index) = self.tiers.iter().position(|tier| (tier.group, tier.layer) == (group, layer)) else {
                return (None, true);
            };
            stage = match stage {
                RequestStage::Arrived { request } => RequestStage::Awaiting(self.opened(at, arrived, request, core, shared)),
                RequestStage::Awaiting(opened) => {
                    match self.await_landings(layer, &opened.cold, core, prefetch) && self.claim_scratch(index, &opened.cold, core) {
                        true => RequestStage::Rounds(self.fix_rounds(at, opened, core, shared)),
                        false => return (Some(Serve { at, place, arrived, stage: RequestStage::Awaiting(opened) }), moved),
                    }
                }
                RequestStage::Rounds(mut rounds) => {
                    let before = rounds.round;
                    if self.round(at, &mut rounds, core, prefetch, shared) {
                        self.finish(at, place, arrived, index, rounds, core, shared);
                        return (None, true);
                    }
                    let answered = rounds.round != before;
                    return (Some(Serve { at, place, arrived, stage: RequestStage::Rounds(rounds) }), moved | answered);
                }
            };
            moved = true;
        }
    }

    /// The request `at`, arrived: what the records freed since let
    /// the conveyor carry, its arrival counted, and, of a group's last
    /// step, its seeding.
    fn opened(&mut self, at: At, arrived: Instant, request: Request, core: &mut Core, shared: &Shared) -> Opened {
        let Point { group, layer, sequence } = at.point;
        let Request { routed, hits, cold, .. } = request;
        // What the records freed since let the conveyor carry, this
        // request's deferred experts among them.
        self.convey(core, shared);
        // The device's time since the group's request before this one was
        // served whole is that request's compute (`Telemetry::arrived`).
        // Each round's list is free to write: this first round arrived past
        // every reader of the lists, staging and conveyor before it.
        shared.telemetry.arrived(group, arrived);
        self.mark_at(shared, layer, Mark::Arrived(sequence), arrived);
        Opened { routed, hits, cold }
    }

    /// Whether no cold expert of the request a read ahead lands into the
    /// host tier (the fill's, promoted to demand) is still landing: the fill
    /// copies each from the tier once it lands, never read again, and the
    /// rounds, fixed next, wait for it first. Each still landing is
    /// hastened; its landing's end is an event. On the one store no landing
    /// serves a request ([`copies_from_tier`]): nothing waits for one.
    fn await_landings(&mut self, layer: usize, cold: &[usize], core: &mut Core, prefetch: &mut Prefetch) -> bool {
        let landing = |scratch: &Self, core: &Core| {
            cold.iter()
                .filter(|&&expert| {
                    copies_from_tier(core) && scratch.conveyor.state(layer, expert).is_none() && core.directory.lands(layer, expert)
                })
                .map(|&expert| (layer, expert))
                .collect::<Vec<_>>()
        };
        let awaited = landing(self, core);
        if awaited.is_empty() {
            return true;
        }
        for &staged in &awaited {
            prefetch.hasten(staged);
        }
        prefetch.settle(&mut core.directory);
        self.settle_landings(core);
        landing(self, core).is_empty()
    }

    /// Fix the rounds of the request `at`, opened: the cold experts read
    /// from the file, a round's records at a time, those the conveyor holds
    /// first (whole, then being read, not yet read), so the device converts them while the rest arrive; the plan's
    /// rounds deliver every expert (`ExpertBankPlan::rounds`). Their count,
    /// written before round 0 is answered, is what the device's grids give
    /// the rounds (`routed_scratch_grids`): the rounds past the last with
    /// records neither park nor convert, and are not served; round 0 parks
    /// always. The cold experts are final here: a hit the layer before's
    /// seeding demoted was before this request's round 0 arrived. Those the
    /// host tier keeps whole the tier copy reads, which round 0's answer
    /// lists (the tier list, written before it).
    fn fix_rounds(&mut self, at: At, opened: Opened, core: &mut Core, shared: &Shared) -> Rounds {
        let Point { group, layer, sequence } = at.point;
        let Opened { routed, hits, cold } = opened;
        self.convey(core, shared);
        let round_records = core.extents.round_records;
        let index = self.tiers.iter().position(|tier| (tier.group, tier.layer) == (group, layer)).expect("a served request's tier");
        // Every cold expert's scratch slot, given while it awaited them,
        // which the pass reads once its rounds are past.
        let scratch = |tier: &Tier, expert: usize| {
            assert!(tier.scratch[expert], "a fixed request's cold experts hold scratch slots");
            expert
        };
        // Every expert the request routes is read in place: a hit in the
        // record the map holds for its pass, or a cold one in its scratch
        // slot. One with neither would read a slot no round filled: an
        // invariant violation, which ends serving.
        let tier = &self.tiers[index];
        for &expert in &routed {
            assert!(
                tier.scratch[expert] || hit(core, self.conveyor.lending(), layer, expert, sequence).is_some(),
                "step {sequence} routes expert {expert} at layer {layer} with neither a held record nor a scratch slot"
            );
        }
        // Those the host tier keeps whole, each its tier record held until
        // the device is past the group, which the tier copy after round 0
        // reads into its scratch slot; the rest the rounds convert, every one
        // on the one store ([`copies_from_tier`]). One whose slot carries a
        // borrower's stored bytes is converted by a round, which
        // `compose_rounds` puts after the borrower's (or `Self::reclaim_slots`
        // holds), as the copy would overwrite them.
        let mut tiered = Vec::new();
        let mut listed = Vec::new();
        let mut cold = cold
            .into_iter()
            .filter(|&expert| {
                let copied = copies_from_tier(core)
                    && self.conveyor.state(layer, expert).is_none()
                    && self.conveyor.borrower(expert).is_none();
                let record = copied.then(|| core.directory.hold_tier(layer, expert)).flatten();
                if let Some(record) = record {
                    tiered.push(expert);
                    listed.push((record.0 as i32, scratch(&self.tiers[index], expert) as i32));
                    self.held.push((layer, record));
                }
                record.is_none()
            })
            .collect::<Vec<_>>();
        cold.sort_by_cached_key(|&expert| readiness(self.conveyor.state(layer, expert)));
        let borrowers = cold
            .iter()
            .map(|&expert| match self.conveyor.borrower(expert) {
                Some((of, other)) if of == layer && cold.contains(&other) => Some(other),
                _ => None,
            })
            .collect::<Vec<_>>();
        let cold = compose_rounds(&cold, &borrowers, round_records);
        let rounds = cold.len().div_ceil(round_records) as u64;
        // Beside a host tier the group's last step seeds its bank group with
        // what the group routed here, the most reused of it and of what the
        // bank holds (`Directory::seed`), each routed expert ranked beside
        // the last step that routed it: only an expert whose scratch slot
        // the group filled, as a seed copies its slot. Its seeds take no
        // record of an expert its merge set routes, which the set's pass
        // reads in place after the seeding. Each expert a seed evicts is
        // unmapped for every step of the group on. After the last round its
        // seed copy and commit read the list, and the host maps each seed
        // committed once the device is past it (`Core::retire`). On the one
        // store nothing seeds: the group took no layer's records, and the
        // fill lands each layer's ranked share once the group returns its
        // lent records. An earlier step's list names another step: they seed
        // nothing.
        let last = self.conveyor.serving.expect("a scratch request's group is served").last;
        let mut seeded = 0;
        if sequence == last && core.extents.placer == crate::planning::Placer::Tier {
            let tier = &self.tiers[index];
            let failed = |expert: usize| tier.failed.iter().any(|&(of, _)| of == expert);
            let tier_routed = (0..tier.routed.len())
                .filter(|&expert| tier.routed[expert] > 0)
                .map(|expert| (expert, tier.last[expert]))
                .collect::<Vec<_>>();
            // Every expert the group routed here is read in place by some
            // pass of it, which may run after the seeding: no seed takes one.
            let read = tier_routed.iter().map(|&(expert, _)| expert).collect::<Vec<_>>();
            let filled = |expert: usize| tier.scratch[expert] && !failed(expert);
            let (seeds, evicted) = core.directory.seed(at.point, layer, &tier_routed, &read, filled);
            for (owner, expert) in evicted {
                self.demote(owner, expert);
            }
            // Named from the first record a seed takes: past the lent
            // records the group holds claimed in the scratch's lender
            // (`ExpertBankPlan::seeded_from`), which its seed copy binds
            // from.
            let first = core.directory.seeded_from(core.layers[layer].group);
            let pairs = seeds
                .iter()
                .map(|&(expert, record)| {
                    let named = record.index().checked_sub(first).expect("a seed takes no lent record the group holds");
                    (expert as i32, i32::try_from(named).expect("bank records fit i32"))
                })
                .collect::<Vec<_>>();
            let seeds = ListBounds { from: core.layers[layer].experts as u64, to: core.layers[layer].lists.bank - first as u64, lent: 0 };
            self.words.seeds.publish_list(sequence, &pairs, seeds);
            seeded = pairs.len() as u64;
        }
        // The rounds fix the device's order of the request's experts: a
        // record goes to a later round's only while every earlier one's
        // experts have theirs (`convey`).
        for (round, chunk) in (0..).zip(cold.chunks(round_records)) {
            for &expert in chunk {
                self.tiers[index].round[expert] = Some(round);
            }
        }
        // The tier list, which the tier copy reads after round 0's answer.
        self.tiers[index].deferred.retain(|expert| !tiered.contains(expert));
        let tier = ListBounds { from: core.layers[layer].lists.tier, to: core.layers[layer].experts as u64, lent: 0 };
        self.words.tiered.publish_list(sequence, &listed, tier);
        // The request's answer: the count its rounds convert, final now,
        // before round 0's.
        let table = &core.layers[layer].scratch;
        table.store(sequence, constants::SCRATCH_ASK, constants::SCRATCH_STAGED, cold.len() as i32);
        table.answer(sequence, constants::SCRATCH_ASK, false);
        Rounds {
            hits,
            cold,
            tiered,
            rounds,
            round: 0,
            waited: Instant::now(),
            served: RequestCost::default(),
            took: GroupLayer::default(),
            seeded,
            failure: None,
        }
    }

    /// Serve `rounds` as far as their experts allow: each round, once every
    /// expert it lists is held whole in its conveyor record, is listed and,
    /// but for the last, answered (the last is answered once the request is
    /// counted: its answer releases the device to complete the step). A
    /// round's records free once the device arrives at the round after it,
    /// past its convert. Whether the rounds
    /// are over: every one served, or the request failed (an expert's read
    /// failed, or its step ended without the device arriving, a device
    /// error).
    fn round(&mut self, at: At, rounds: &mut Rounds, core: &mut Core, prefetch: &mut Prefetch, shared: &Shared) -> bool {
        let Point { group, layer, sequence } = at.point;
        let round_records = core.extents.round_records;
        let table = core.layers[layer].scratch.clone();
        // A layer's cold experts fill at most its rounds (`rounds`: the
        // widest layer's, in records of `round_records`; a lane past them is
        // refused, `SlotTable::lane`).
        let last_round = rounds.rounds.max(1) - 1;
        // The bytes one expert of the layer reads.
        let expert_bytes = core.layers[layer].expert_bytes();
        while rounds.failure.is_none() && rounds.round < rounds.rounds {
            let round = rounds.round;
            let chunk = rounds.cold.chunks(round_records).nth(round as usize).unwrap_or_default().to_vec();
            // Reads into staging that ended free their records, as the
            // rounds the device has converted free theirs (`convey`).
            prefetch.settle(&mut core.directory);
            self.convey(core, shared);
            let tier = self.tiers.iter().find(|tier| (tier.group, tier.layer) == (group, layer)).expect("a served request's tier");
            let failed = tier.failed.iter().find(|(expert, _)| chunk.contains(expert)).map(|(_, reason)| reason.clone());
            if let Some(reason) = failed {
                rounds.failure = Some(reason);
            } else if !self.reclaim_slots(&chunk) || !chunk.iter().all(|&expert| self.conveyor.whole(layer, expert)) {
                // A step that ended without the device arriving (a device
                // error) leaves nothing to arrive.
                if !shared.steps.ended(sequence) {
                    return false;
                }
                rounds.failure = Some(format!(
                    "step {sequence} ended before scratch round {round} of layer {layer}: the step ended, the round waiting for experts {chunk:?}"
                ));
            }
            rounds.served.read += rounds.waited.elapsed();
            if rounds.failure.is_some() {
                break;
            }
            let tier = self.tiers.iter().find(|tier| (tier.group, tier.layer) == (group, layer)).expect("a served request's tier");
            assert!(chunk.iter().all(|&expert| tier.scratch[expert]), "a listed expert holds its scratch slot");
            let pairs = chunk
                .iter()
                .map(|&expert| {
                    let record = self.conveyor.list(layer, expert);
                    (self.conveyor.listed_from(record), expert as i32)
                })
                .collect::<Vec<_>>();
            // Its records free as the device arrives at the round after
            // (parked there by `routed_scratch_grids`), whose lane its
            // slot has unless this is the slot's last.
            core.taken.push(Taken {
                records: chunk.iter().map(|&expert| Listed { layer, expert }).collect(),
                until: at.point,
                after: (round + 1 < core.extents.words.rounds).then(|| round_lane(round + 1)),
            });
            let experts = core.layers[layer].experts as u64;
            let lent = if core.layers[layer].slots.is_some() { experts } else { 0 };
            let conveyed = ListBounds { from: core.layers[layer].lists.conveyor, to: experts, lent };
            self.words.lists[round as usize].publish_list(sequence, &pairs, conveyed);
            rounds.served.misses += chunk.len() as u64;
            // A prompt group's experts are its own reads, none a hint's.
            rounds.served.anticipated[Anticipation::Demand as usize] += chunk.len() as u64;
            rounds.served.bytes += chunk.len() as u64 * expert_bytes;
            rounds.took.conveyed += chunk.len() as u64;
            if round != last_round {
                table.answer(sequence, round_lane(round), false);
                shared.parks.settle();
            }
            rounds.round += 1;
            rounds.waited = Instant::now();
        }
        true
    }

    /// The rounds of the request `at` its point from chain place `place`,
    /// arrived at `arrived`, of tier `index`, are over: count it and answer
    /// its last round, or fail it.
    #[allow(clippy::too_many_arguments)]
    fn finish(&mut self, at: At, place: usize, arrived: Instant, index: usize, rounds: Rounds, core: &mut Core, shared: &Shared) {
        let point = at.point;
        let Point { layer, sequence, .. } = point;
        let Rounds { hits, cold, tiered, rounds, mut served, mut took, seeded, failure, .. } = rounds;
        let last_round = rounds.max(1) - 1;
        let expert_bytes = core.layers[layer].expert_bytes();
        // The misses the fill copies from the host tier, served as a decode
        // request's tier misses are counted: none is read.
        if failure.is_none() {
            served.misses += tiered.len() as u64;
            served.hosted += tiered.len() as u64;
            served.bytes += tiered.len() as u64 * expert_bytes;
            took.pulled += tiered.len() as u64;
        }
        served.parked = arrived.elapsed();
        took.parked = served.parked;
        shared.telemetry.serve(layer, sequence, served);
        let last = self.conveyor.serving.expect("a scratch request's group is served").last;
        let tier = &self.tiers[index];
        took.union = tier.routed.iter().filter(|&&count| count > 0).count() as u64;
        took.hits = hits.len() as u64;
        took.seeded = seeded;
        shared.telemetry.group(point, last, took);
        match failure {
            // The device's time after a failed step is no layer's compute.
            Some(reason) => {
                self.forfeit(index, place, hits.iter().map(|&(expert, _)| expert).chain(cold).chain(tiered), core);
                shared.telemetry.served_whole(None);
                fail(core, shared, layer, sequence, reason);
            }
            None => {
                shared.telemetry.served_whole(Some((point, Instant::now())));
                self.mark(shared, layer, Mark::Served(sequence));
                // The rounds past the last with records are answered before
                // it (its answer releases the device to complete the step):
                // where kernels wait in place they do not wait (their park
                // grid is empty, `routed_scratch_grids`, but for the round
                // after the last, which an unfollowed fill parks at, its
                // arrival freeing the last round's records) but after a round
                // 0 whose wait expired, whose step's every round parks and
                // so finds its answer, and where a park
                // is a submission boundary (`Parks`) each still waits for
                // its answer; the last round's answer is the seed commit's.
                let table = &core.layers[layer].scratch;
                for round in (last_round..core.extents.words.rounds).rev() {
                    table.answer(sequence, round_lane(round), false);
                    shared.parks.settle();
                }
            }
        }
    }

    /// A scratch claim unmapped expert `expert` of `layer`: a request taken
    /// there that counted it a hit, whose rounds have not arrived, reads it
    /// instead.
    fn demote(&mut self, layer: usize, expert: usize) {
        let Some(tier) = self.tiers.iter_mut().find(|tier| tier.layer == layer) else {
            return;
        };
        let mut reader = None;
        for request in tier.requests.iter_mut().flatten() {
            if let Some(at) = request.hits.iter().position(|&(hit, _)| hit == expert) {
                request.hits.swap_remove(at);
                request.cold.push(expert);
                reader = Some(request.sequence);
            }
        }
        if let Some(reader) = reader {
            tier.assign(expert, reader, self.conveyor.state(layer, expert).is_some());
        }
    }

    /// The step at chain place `place` failed without converting
    /// `experts`, its request's at tier `index`'s layer, which no step of the group
    /// before it routed there: each is the next later step's that routes
    /// it, a hit or read in its rounds (`Tier::assign`), and one no later
    /// step took is the tier's no more, so neither seeded, counted nor
    /// read: a step that routes it later, whose request comes after (a
    /// failed step runs on unparked, ahead of the steps behind it), takes
    /// it as the group's first to route it there.
    pub(super) fn forfeit(&mut self, index: usize, place: usize, experts: impl IntoIterator<Item = usize>, core: &Core) {
        let tier = &mut self.tiers[index];
        let layer = tier.layer;
        for expert in experts {
            // A later step that takes the expert over reads it again.
            tier.failed.retain(|&(of, _)| of != expert);
            let later = tier.requests[place + 1..]
                .iter_mut()
                .flatten()
                .find(|later| later.routed.contains(&expert));
            let lending = self.conveyor.lending();
            let held = |later: &Request| hit(core, lending, layer, expert, later.sequence);
            let reader = match later.map(|later| (held(later), later)) {
                Some((Some(record), later)) => {
                    later.hits.push((expert, record));
                    None
                }
                Some((None, later)) => {
                    later.cold.push(expert);
                    Some(later.sequence)
                }
                None => {
                    tier.routed[expert] = 0;
                    None
                }
            };
            match reader {
                Some(reader) => tier.assign(expert, reader, self.conveyor.state(layer, expert).is_some()),
                None => tier.unassign(expert),
            }
        }
    }
}

impl Request {
    /// The experts it is the group's first to route at its layer.
    pub(super) fn experts(&self) -> Vec<usize> {
        self.hits.iter().map(|&(expert, _)| expert).chain(self.cold.iter().copied()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::compose_rounds;

    /// A slot's expert converts in a round after its borrower's, which
    /// converts the borrower out of it, the rest in the order they are
    /// ready; where no other expert fills a round, it joins its
    /// borrower's.
    #[test]
    fn a_slots_expert_converts_after_its_borrower() {
        // Expert 3 borrows 1's slot, 5 borrows 0's: 1 and 0 wait a round.
        let cold = [1, 3, 0, 5, 2, 4];
        let borrowers = [Some(3), None, Some(5), None, None, None];
        assert_eq!(compose_rounds(&cold, &borrowers, 2), [3, 5, 1, 0, 2, 4]);
        // One round: the slot's expert joins its borrower's.
        assert_eq!(compose_rounds(&[1, 3], &[Some(3), None], 2), [3, 1]);
        // Another expert fills round 0: 1 waits for round 1.
        assert_eq!(compose_rounds(&[1, 3, 0], &[Some(3), None, None], 2), [3, 0, 1]);
        // Its borrower ready last, in the last full round, which nothing
        // else fills: 1 joins it.
        assert_eq!(compose_rounds(&[2, 4, 3, 1], &[None, None, None, Some(3)], 2), [2, 4, 3, 1]);
    }
}
