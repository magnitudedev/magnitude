//! What serving costs: each request's and step's misses and reads, the
//! prompt groups' layers, the facts the group size rule reads, and the
//! routing the ranking's stack is fed (the performance model's inputs, and
//! the one ranking's feed: `Reuse`).

use crate::operation::LaunchWork;
use super::core::{Extents, Layer, RoutingRecord};
use super::steps::{CompletedRoute, CompletedStep, Decoded, EndedReads, Recording, Step, Steps};
use crate::assessment::{
    ExpertLines, ExpertPass, FileLine, GroupLayerTerms, GroupUnion, LayerStep, Notice, ReadLine,
    ReadLineError, RequestReads, Stages,
};
use crate::experts::Words;
use crate::experts::directory::{CountedLaunch, Point, RoutedRead, Stack};
use crate::experts::reuse::Ranking;
use crate::planning::RegionWords;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The tracing target of each completed step's reads (`complete_step`),
/// at debug: what a step that ran slow read.
pub const STEP_TRACE: &str = "magnitude_executor::expert_steps";

/// What steps cost a streamed layer, summed over the requests served: the
/// performance model's per-layer inputs (`assessment::streaming`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayerCost {
    /// Experts the bank lacked (`n_l`), and their stored bytes (`β_l`).
    pub misses: u64,
    pub missed_bytes: u64,
    /// The misses the device served from the host tier, which read nothing; the rest
    /// by what prediction had done when they were requested
    /// (`Anticipation`): already reading (a hint's read, promoted), or not
    /// (read on demand). They sum to `misses`.
    pub hosted: u64,
    /// Of `hosted`, those the CPU site served: the device read the rest
    /// over the link.
    pub sited: u64,
    pub reading: u64,
    pub unanticipated: u64,
    /// Each request's lead summed over the requests it is measured on
    /// (`led`): the layers between the submission of its last-submitted
    /// read and the request (0 when it read on demand), the `h` it hid. A
    /// request every miss of which had landed already saw none of its
    /// reads, so has none.
    pub lead: u64,
    pub led: u64,
    /// From the host's notice of a request (its publication, within one
    /// idle rest) to its acknowledgement, on the host's clock.
    pub parked: Duration,
    /// From the reads' submission to their end.
    pub read: Duration,
}

/// What one step cost every streamed layer together, and each layer's
/// misses and reads (the performance model's timeline inputs, in decoder
/// order).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepCost {
    /// The experts the step read on demand (`m`), and their stored bytes.
    pub misses: u64,
    pub missed_bytes: u64,
    pub parked: Duration,
    pub read: Duration,
    pub layers: Vec<LayerStep>,
    /// Each streamed layer's experts: what the bank and tier lacked, and
    /// what the step routed to there.
    pub experts: Vec<LayerExperts>,
    /// Its time from the later of its begin and the end of the step before
    /// ([`step_seconds`]); none of a prompt step.
    pub seconds: f64,
    /// What each volume the model's files lie on did since the step before
    /// ([`seismic_io::Totals`]): its bytes, requests, busy time, depth and
    /// demand waits by class, the same on every backend.
    pub storage: Vec<(seismic_io::Volume, seismic_io::Totals)>,
}

/// One streamed layer's experts in one step, by index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LayerExperts {
    /// The experts its request named absent, in request order.
    pub missed: Vec<usize>,
    /// Of its tier misses, those the CPU site served, and its request's
    /// path there (taken to answered).
    pub sited: u64,
    pub site_path: Duration,
    /// Every expert the step routed to here, ascending, and its choices,
    /// from its routing record; none of a grouped step
    /// ([`Telemetry::routed`]).
    pub routed: Option<RoutedRead>,
}

/// One hint candidate of a decode step's ring as admitted: the expert, its
/// rank (the rank of the choice it was hinted as, HINT_RANK: 0 a row's
/// best, whatever the step's rows), and whether its hint was read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Admitted {
    pub(super) expert: usize,
    pub(super) rank: usize,
    pub(super) read: bool,
}

impl StepCost {
    fn of(layers: usize) -> Self {
        Self {
            layers: vec![LayerStep::default(); layers],
            experts: vec![LayerExperts::default(); layers],
            ..Self::default()
        }
    }
}

/// What steps have cost the load: each streamed layer's sums, in decoder
/// order, its decode requests' demand reads and what its volume delivered
/// (the served file's read line), and the last step completed (its sequence and
/// cost).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExpertCosts {
    pub layers: Vec<LayerCost>,
    pub reads: RequestReads,
    /// Each streamed layer's landing line at the one site its bank lands
    /// at: the host's landings observed beside a host tier, none until
    /// observed; the device's convert (its landing launch's) where the bank
    /// places, once measured.
    pub landings: Vec<Option<crate::assessment::LandingLine>>,
    pub hints: HintCost,
    pub completed: Option<(u64, StepCost)>,
    /// The latest prompt group served.
    pub group: Option<GroupCost>,
    /// The promotions asked since the load.
    pub promoted: u64,
    /// The bank's shape: the slots each layer commits, each group's records
    /// its experts are held in (`Core::bank_records`), each streamed
    /// layer's group, and a layer's experts.
    pub slots: usize,
    pub records: Vec<usize>,
    pub layer_groups: Vec<usize>,
    pub experts: usize,
    /// What the CPU site's answered requests took since the load: the
    /// misses it served (the rest of the tier's misses the device read over
    /// the link) and each term of their path.
    pub site: crate::experts::site::SiteServed,
}

/// What a prompt group's steps took at each streamed layer (in decoder
/// order): the bound `Σ_l max(compute_l, read_l)` is computed from these
/// and a read line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GroupCost {
    /// Its first sequence and its steps.
    pub first: u64,
    pub steps: u64,
    pub layers: Vec<GroupLayer>,
}

/// A prompt group's cost as serving counts it.
struct Counted {
    cost: GroupCost,
    /// Its last request served whole, whose compute interval runs from its
    /// acknowledgement. A newer group's count drops it with this group.
    acknowledged: Option<Acknowledged>,
    /// When its first request was taken: what its layers' timelines
    /// ([`LayerTimeline`]) count from.
    opened: Option<Instant>,
}

/// One instant of a prompt group's serving at a layer ([`LayerTimeline`]):
/// a read's, or the step `Step` of the group's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mark {
    /// The step's request there was taken, as its front published it.
    Requested(Step),
    /// A cold expert's read into the conveyor was submitted.
    Submitted,
    /// A cold expert's read into the conveyor ended.
    Landed,
    /// The step's round 0 there arrived: the device reached the step's
    /// fill, its pass past the layer before.
    Arrived(Step),
    /// The step's request there was served whole: the device released to
    /// its pass and back.
    Served(Step),
}

/// When a group's events at a layer came, from the group's first request
/// taken ([`Counted::opened`]): its cold experts' reads submitted and
/// ended, first and last, and each step's own ([`StepTimeline`]). The
/// layer's pass runs from its first step's round 0 to after its last
/// step's request served whole; its reads overlap the layer before where
/// they end before its first round 0.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayerTimeline {
    pub submitted: Option<Span>,
    pub landed: Option<Span>,
    /// Per step of the group, in its order.
    pub steps: Vec<StepTimeline>,
}

/// A step's events at a layer, from its group's first request taken: its
/// request taken, its round 0 arrived, and its request served whole. Its
/// rounds run from its arrival to its service;
/// the device's time from its service to the next step's arrival is
/// compute.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepTimeline {
    pub requested: Option<Duration>,
    pub arrived: Option<Duration>,
    pub served: Option<Duration>,
}

/// The first and last of an event, from a group's first request taken.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub first: Duration,
    pub last: Duration,
}

impl LayerTimeline {
    /// `mark` came at `at` of the group whose first step is `group`.
    fn mark(&mut self, group: Step, mark: Mark, at: Duration) {
        let span = |span: &mut Option<Span>| {
            *span = Some(span.map_or(Span { first: at, last: at }, |span| Span { first: span.first.min(at), last: span.last.max(at) }));
        };
        // A step's request is taken, arrives and is served once at a layer.
        let once = |slot: &mut Option<Duration>, what: &str| {
            assert!(slot.replace(at).is_none(), "a step's request {what} twice at a layer");
        };
        match mark {
            Mark::Submitted => span(&mut self.submitted),
            Mark::Landed => span(&mut self.landed),
            Mark::Requested(sequence) => once(&mut self.step(group, sequence).requested, "taken"),
            Mark::Arrived(sequence) => once(&mut self.step(group, sequence).arrived, "arrived"),
            Mark::Served(sequence) => once(&mut self.step(group, sequence).served, "served"),
        }
    }

    /// Step `sequence`'s events, of the group whose first step is `group`.
    fn step(&mut self, group: Step, sequence: Step) -> &mut StepTimeline {
        let index = usize::try_from(sequence - group).expect("a group's steps fit usize");
        if self.steps.len() <= index {
            self.steps.resize_with(index + 1, StepTimeline::default);
        }
        &mut self.steps[index]
    }
}

/// A group's request served whole: its point, and when its last round was
/// acknowledged.
#[derive(Clone, Copy)]
struct Acknowledged {
    point: Point,
    at: Instant,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GroupLayer {
    /// The experts the group's steps routed to (the union), of them the
    /// bank's hits, and the rest: conveyed (read into the conveyor, or its
    /// ring, as their requests were published), and their stored bytes;
    /// and pulled from the host tier into the scratch tier, and their
    /// resident bytes. Each counted once.
    pub union: u64,
    pub hits: u64,
    pub conveyed: u64,
    pub cold_bytes: u64,
    pub pulled: u64,
    pub tier_bytes: u64,
    /// The experts the group seeded into the bank: its last step's seed
    /// copies beside a host tier, its reads landed in place of less ranked
    /// experts on the one store.
    pub seeded: u64,
    /// What the group's steps waited here for their experts: the watcher's
    /// serving of each step's request, from its round 0 to its last.
    pub parked: Duration,
    /// The compute of the group's steps here, measured: each from its last
    /// round's acknowledgement to the device's next round 0, the last
    /// layer's last step's to its completion, so the `computed` steps
    /// measured. A group runs as a wavefront over merge sets, so that is
    /// the step's back and the next layer's front it carries (the host sees
    /// no boundary between), its back alone at the last layer; of a merge
    /// set, its fills' tails and, at its last step, its pass and backs.
    /// Their sum is the layer's compute whatever its sets.
    pub compute: Duration,
    pub computed: u32,
    /// When the group's events here came ([`LayerTimeline`]).
    pub timeline: LayerTimeline,
}

impl GroupLayer {
    /// The compute of a group of `steps` steps here: the sum measured once
    /// every step is, each step at the mean of those measured before:
    /// none while none is.
    pub fn compute_of(&self, steps: u32) -> Option<Duration> {
        (self.computed > 0).then(|| self.compute / self.computed * steps)
    }
}

/// The reads hints made: hint reads submitted and their stored bytes. And
/// how the prediction did over completed decode steps, whether or not a
/// hint was read:
/// - `candidates`: hinted experts nothing held, kept, read or landed
///   when their ring was read;
/// - `named`: those the step's route at the hinted layer chose as a miss
///   their hint saves ([`saves`]), of which `early` the route found landed
///   already (a hit the hint made, so never a miss);
/// - `hintable`: the file misses (experts the route found absent) of layers
///   a ring hints, the misses a hint could have saved beside `early`;
/// - `unhintable`: those of layers no ring hints (the first streamed
///   layer's), which no prediction reaches by construction, so neither
///   [`Self::recall`] nor [`Self::precision`] counts them;
/// - the layers of completed decode steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HintCost {
    pub reads: u64,
    pub bytes: u64,
    pub candidates: u64,
    pub named: u64,
    pub early: u64,
    pub hintable: u64,
    pub unhintable: u64,
    pub layer_steps: u64,
}

impl HintCost {
    /// The one recall of the hints: of the misses a hint could save at
    /// layers a ring hints, those a candidate named, its hint landed early
    /// (a hit it made, `early`, which no miss count holds) or not:
    /// `named / (hintable + early)`. None before any.
    pub fn recall(&self) -> Option<f64> {
        let wanted = self.hintable + self.early;
        (wanted > 0).then(|| self.named as f64 / wanted as f64)
    }

    /// The one precision of the hints: the share of candidates a route
    /// named, early landings among them (`named / candidates`), whether or
    /// not each was read, so it judges the prediction and not the reads.
    /// None before any.
    pub fn precision(&self) -> Option<f64> {
        (self.candidates > 0).then(|| self.named as f64 / self.candidates as f64)
    }

    /// What was counted after `start`, a count of the same load taken
    /// earlier.
    pub fn since(&self, start: &Self) -> Self {
        Self {
            reads: self.reads - start.reads,
            bytes: self.bytes - start.bytes,
            candidates: self.candidates - start.candidates,
            named: self.named - start.named,
            early: self.early - start.early,
            hintable: self.hintable - start.hintable,
            unhintable: self.unhintable - start.unhintable,
            layer_steps: self.layer_steps - start.layer_steps,
        }
    }
}

/// Whether a hinted candidate the hinted route chose, by the source the
/// route resolved it to (the routing record's, before any promotion), was a
/// miss a hint of it saves: one the route found absent (the file's) or in
/// the records misses land into, whoever landed it there, its hint read or
/// not, so a rank's measured precision never depends on whether its hints
/// were read. One the route found in a bank record no landing writes (a
/// discrete bank's, promoted from the tier) needed no read; so one promoted
/// between its ring and its route counts wrong, though a read of it would
/// have served: rare, and on the cautious side.
fn saves(used: Use) -> bool {
    match used {
        Use::File | Use::Tier => true,
        Use::Bank => false,
    }
}

/// One definition of a step's time, from the later of its begin and the
/// last step's end (`previous`) to its end, `now`: a gap with no step in
/// flight (a client's, a tool call's) is no step's time.
fn step_seconds(previous: Option<Instant>, begun: Instant, now: Instant) -> f64 {
    let start = previous.map_or(begun, |previous| previous.max(begun));
    now.saturating_duration_since(start).as_secs_f64()
}

/// The decode time a hint's price is measured over, the latest
/// ([`Recent`]): a second holds 10–70 decode steps at the cells' step
/// times, enough steps to average a step's reads and to count each ring
/// rank's candidates by the hundred, and is shorter than the M4 Pro
/// volume's burst bucket takes to drain (≈ 2.8 GB at 0.66 GB/s past its
/// sustained rate, about 4 s), so the busy share follows the bucket's state.
/// A rank's precision falls as decode's post-prefill transient drains (the
/// M4 Pro's rank 7 from 0.35 to 0.26 over a run's window): a mean since the
/// load would price the transient's precision long after it ended.
const WINDOW: f64 = 1.0;

/// The latest decode steps, within [`WINDOW`] of decode time: their times
/// and the time the volume was busy meanwhile with reads a step needs
/// ([`needed`], from its account, `seismic_io::accounting`), what share of
/// the time it is busy so;
/// and their hint candidates by choice rank, each rank's precision.
/// Every step it holds is a generation step's since the last prompt step, which
/// drops its reads and starts the window again (`Telemetry::complete_step`).
#[derive(Default)]
struct Recent {
    steps: std::collections::VecDeque<RecentStep>,
    seconds: f64,
}

/// A decode step of [`Recent`]: its seconds, its rings' candidates by rank,
/// and the seconds its volume was busy with reads a step needs.
struct RecentStep {
    seconds: f64,
    ranks: Ranks,
    busy: f64,
}

impl Recent {
    /// A decode step of `seconds`, its rings' candidates resolved into
    /// `ranks`, its volume busy `busy` seconds meanwhile: the oldest steps
    /// leave once the rest cover the window.
    fn step(&mut self, seconds: f64, ranks: Ranks, busy: f64) {
        self.steps.push_back(RecentStep { seconds, ranks, busy });
        self.seconds += seconds;
        while let Some(&RecentStep { seconds: oldest, .. }) = self.steps.front() {
            if self.seconds - oldest < WINDOW {
                break;
            }
            self.seconds -= oldest;
            self.steps.pop_front();
        }
    }

    /// The window's steps' candidates together ([`Ranks::precision`]).
    fn ranks(&self) -> Ranks {
        let mut window = Ranks::default();
        for step in &self.steps {
            window.add(&step.ranks);
        }
        window
    }

    /// The window's mean decode step, in seconds; none before one is timed.
    fn mean_step(&self) -> Option<f64> {
        (!self.steps.is_empty()).then(|| self.seconds / self.steps.len() as f64)
    }

    /// The share of the window's time the volume was busy with reads a
    /// step needs ([`needed`]), at most all of it: over the decode steps
    /// since the last prompt step until they make a whole window. Before
    /// any is timed nothing says the volume is busy, so a hint is priced on
    /// an idle one and its rank's prior read, as its precision is before
    /// its rank is measured ([`saving`]); the first step timed prices the
    /// rest.
    fn busy(&self) -> f64 {
        if self.seconds == 0. {
            return 0.;
        }
        (self.steps.iter().map(|step| step.busy).sum::<f64>() / self.seconds).min(1.)
    }

}

/// Each lead's choice ranks' candidates of completed decode steps and those
/// a hint saves ([`saves`]): each lead's rank's precision, read or not.
#[derive(Default)]
struct Ranks([Vec<[u64; 2]>; crate::planning::HINT_LEADS as usize]);

impl Ranks {
    fn count(&mut self, lead: usize, rank: usize, named: bool) {
        let ranks = &mut self.0[lead];
        if ranks.len() <= rank {
            ranks.resize(rank + 1, [0; 2]);
        }
        ranks[rank][0] += 1;
        ranks[rank][1] += u64::from(named);
    }

    /// Count `other`'s candidates too.
    fn add(&mut self, other: &Ranks) {
        for (mine, theirs) in self.0.iter_mut().zip(&other.0) {
            if mine.len() < theirs.len() {
                mine.resize(theirs.len(), [0; 2]);
            }
            for (mine, theirs) in mine.iter_mut().zip(theirs) {
                mine[0] += theirs[0];
                mine[1] += theirs[1];
            }
        }
    }

    /// Rank `rank`'s precision at lead `lead` as the mean of its uniform
    /// prior updated by its candidates, `(named + 1)/(candidates + 2)`: a
    /// half before any is seen.
    fn precision(&self, lead: usize, rank: usize) -> f64 {
        let [candidates, named] = self.0[lead].get(rank).copied().unwrap_or([0; 2]);
        (named + 1) as f64 / (candidates + 2) as f64
    }
}

/// What a hint's read is priced by at one admission ([`Telemetry::price`]):
/// each rank's precision (every rank a ring may name, [`Ranks::precision`]),
/// the volume's share busy with reads a step needs ([`Recent::busy`]), a
/// streamed layer's time in the latest decode
/// steps (their mean over their streamed layers; none before one is timed),
/// and the volume's read line (none before it is measured).
pub(super) struct HintPrice {
    precision: Vec<f64>,
    busy: f64,
    layer: Option<f64>,
    line: Option<ReadLine>,
}

impl HintPrice {
    /// Whether a hint at rank `rank` of `bytes` pays for its read
    /// ([`pays`]): a hint's rank is a row's choice's, each of which is
    /// priced.
    pub(super) fn pays(&self, rank: usize, bytes: u64) -> bool {
        pays(self.precision[rank], self.busy, self.latency(bytes))
    }

    /// A read of `bytes`' latency over its transfer, `λ/(b/D)`: none before
    /// the volume's line is measured.
    fn latency(&self, bytes: u64) -> f64 {
        assert!(bytes > 0, "a hint reads an expert's bytes");
        self.line.map_or(0., |line| line.latency_seconds() / line.transfer_seconds(bytes as f64))
    }

    /// What a hint at rank `rank` of `bytes` is wanted as on its read's link
    /// (`seismic_io::Want`), its layer `ahead` streamed layers past the
    /// device's at `now`: expected from that layer's estimated start,
    /// `ahead` layer times after `now` (read ahead until a step waits on it,
    /// `Prefetch::hasten`), worth the step time it is expected to save net
    /// of what it costs ([`saving`] of its bytes' transfer, `b/D`; nothing
    /// before the volume's line is measured); undated before a decode step
    /// is timed.
    pub(super) fn want(&self, rank: usize, ahead: usize, bytes: u64, now: std::time::Instant) -> seismic_io::Want {
        let saving = saving(self.precision[rank], self.busy, self.latency(bytes)).max(0.);
        let value = self.line.map_or(0., |line| saving * line.transfer_seconds(bytes as f64));
        match self.layer {
            Some(layer) => seismic_io::Want::expected(now + std::time::Duration::from_secs_f64(layer * ahead as f64), value),
            None => seismic_io::Want::undated(value),
        }
    }
}

/// The step time a hint whose prediction is right with probability
/// `precision` is expected to save, net of what it costs, per second of its
/// bytes' transfer `b/D`: the volume busy a share `busy` of the time with
/// reads a step needs ([`needed`]), its read's latency `latency` times its
/// transfer (`λ/(b/D)`, none before the volume's line is measured). A right
/// hint's bytes are read for its request anyway: reading them ahead saves
/// the request's park on them, `λ + b/D`, only as far as the volume would
/// otherwise idle, `1 − busy`, since on a volume busy with what steps need
/// they only take the place of bytes a step waits on. A wrong one adds its
/// bytes' time, `b/D`, to the volume, which delays the reads steps wait on
/// in the share `busy` of the time they hold it. So
/// `p·(1 − busy)·(1 + λ/(b/D)) − (1 − p)·busy`: on a saturated volume no
/// prediction short of certain saves anything; on an idle one, every one.
/// No term is a job's time from its submission, which holds the pieces
/// ahead of it (a hint's own bytes, which would price more hints the more
/// were read), and `busy` leaves out the wrong hints' reads, so whether
/// hints are read never moves their price.
fn saving(precision: f64, busy: f64, latency: f64) -> f64 {
    precision * (1. - busy) * (1. + latency) - (1. - precision) * busy
}

/// Whether a hint pays for its read: its expected [`saving`] is none lost.
fn pays(precision: f64, busy: f64, latency: f64) -> bool {
    saving(precision, busy, latency) >= 0.
}

/// The seconds of a span the volume was busy with reads a step needs:
/// its busy seconds `busy` but the share of what it delivered (`spans`,
/// each volume's) that hints read for experts no route chose, the bytes
/// read ahead beyond `named`, those of hints a route chose. What else it
/// read ahead is the cover's (`Prefetch::cover`), which orders behind every
/// hint and delays no hint's read, so no hint's price counts it. A hint a
/// step hastened reads its last pieces due, so `named` may exceed what it
/// read ahead: the share is then at most the volume's, never below what
/// steps needed.
fn needed(busy: f64, spans: &[(seismic_io::Volume, seismic_io::Totals)], named: u64) -> f64 {
    let (bytes, ahead) = spans.iter().fold((0, 0), |(bytes, ahead), (_, span)| (bytes + span.bytes(), ahead + span.ahead.bytes));
    if bytes == 0 {
        return busy;
    }
    busy * (1. - ahead.saturating_sub(named) as f64 / bytes as f64)
}

/// Where a route served a choice from, by its routing record's source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Use {
    /// A bank record no landing writes: a source `>= 0` where the device
    /// places beside a host tier.
    Bank,
    /// A record misses land into: a host tier's, read in place (a source
    /// `-2 - t`), or on unified memory the bank's own, the one store (a
    /// source `>= 0`).
    Tier,
    /// Neither (a source `-1`): read from the file and landed into the tier.
    File,
}

impl Use {
    /// Where a source word served a choice from, by who places into the
    /// bank (`placer`): beside a host tier a record `>= 0` is the device's
    /// bank, on unified memory the one store misses land into.
    fn of(source: i32, placer: crate::planning::Placer) -> Self {
        match source {
            0.. => match placer {
                crate::planning::Placer::Tier => Self::Bank,
                crate::planning::Placer::Bank => Self::Tier,
            },
            -1 => Self::File,
            _ => Self::Tier,
        }
    }
}

/// The costs as serving records them: the steps still running keep theirs
/// until they complete. `line` is the read line of `reads`, none until the
/// load's first demand read landed and its volume delivered, which each read
/// and step refits.
/// `hinted` is each ring read whose step has not ended, by (step, hinted
/// layer, lead): its candidates, each with whether its hint was read and its
/// rank.
struct Costs {
    layers: Vec<LayerCost>,
    reads: RequestReads,
    line: Option<ReadLine>,
    /// Each bank group's landing line, observed from its requests' landings
    /// ([`Telemetry::landings_ended`]).
    landings: Vec<seismic::measure::Line<2>>,
    /// Each streamed layer's CPU site requests, observed as they took
    /// (`Telemetry::complete_step`).
    sites: Vec<SiteRequests>,
    /// Each streamed layer's block line, observed from sampled decode
    /// steps (`Telemetry::complete_step`): its block's device time over its
    /// hits, link misses and sited misses, each an expert of one row, the
    /// time its rows past each one's first take at the step's lines taken
    /// from it, as a site's requests are (`SiteRequests`).
    blocks: Vec<seismic::measure::Line<4>>,
    /// Each bank group's device convert (its landing launches'), once the job measured
    /// it ([`Telemetry::device_convert`]).
    convert: Vec<Option<crate::assessment::RecordLine>>,
    hints: HintCost,
    hinted: BTreeMap<(Step, usize, usize), Vec<Admitted>>,
    /// The latest decode steps' file reads and choice ranks' candidates
    /// ([`Recent`]).
    recent: Recent,
    /// Each volume's counts (`Routing::volumes`) as the last step's end
    /// saw it, from none: the first step counts what each read since the
    /// process began.
    looked: Vec<seismic_io::Totals>,
    running: BTreeMap<Step, StepCost>,
    completed: Option<(Step, StepCost)>,
    group: Option<Counted>,
}

/// What prediction had done for a miss when its request came, as the
/// performance model takes it (`assessment::streaming`): a hint's read
/// still reading it, which the request
/// waits out (a hint is always the layer before's route's), or neither:
/// read on demand. The one classification a layer's cost
/// ([`LayerCost`]) and its step's anticipation read.
#[derive(Clone, Copy)]
pub(super) enum Anticipation {
    Reading,
    Demand,
}

/// One request's cost.
#[derive(Default)]
pub(super) struct RequestCost {
    pub(super) misses: u64,
    pub(super) bytes: u64,
    /// The misses the device served from the host tier, and the rest by
    /// [`Anticipation`].
    pub(super) hosted: u64,
    pub(super) anticipated: [u64; 2],
    /// Its lead (`LayerCost::lead`), where it saw a read of its misses.
    pub(super) lead: Option<u64>,
    /// The absent experts it read (not kept already).
    pub(super) missed: Vec<usize>,
    pub(super) parked: Duration,
    /// What the request waited for its reads (a scratch request's; a
    /// decode request's reads land after it is served, [`Telemetry::landed`]).
    pub(super) read: Duration,
}

/// What the prompt-group size rule reads ([`Telemetry::group_size`]), per
/// streamed layer: the compute of one scratch-served step there, measured
/// on this run (the mean of the latest group's steps measured there:
/// [`GroupLayer::compute_of`]), the experts the layer holds, the experts
/// the last scratch-served group to end there read from the file and
/// pulled from the host tier there (its [`GroupLayer::conveyed`] and
/// [`GroupLayer::pulled`] once its last step's request there is taken:
/// [`Telemetry::group`]), the union the last prompt group the stack was fed
/// read there (`unions`: its experts at their depths and the choices routed
/// to each, [`Telemetry::flush`]), and one expert's stored bytes; and each
/// layer's experts.
pub(super) struct GroupFacts {
    pub(super) compute: Vec<Option<Duration>>,
    pub(super) held: Vec<usize>,
    /// Per layer, the experts its host records hold alone (a host tier's,
    /// exclusive of the bank), as the watcher last published them.
    pub(super) alone: Vec<usize>,
    pub(super) read: Vec<Option<u64>>,
    pub(super) pulled: Vec<Option<u64>>,
    pub(super) unions: Vec<Option<GroupUnion>>,
    pub(super) bytes: Vec<u64>,
    /// Per layer, the resident bytes one pull of its experts copies, where
    /// prompt groups pull from a host tier (`TierPulls::bytes`).
    pub(super) resident: Option<Vec<u64>>,
    pub(super) experts: Vec<usize>,
}

impl GroupFacts {
    /// Each layer's terms of its group's stages ([`Stages`]): its step's
    /// compute, and the experts the next group will read from the file and
    /// pull from the host tier there. That is what the last group to end
    /// there read and pulled: a request's prompt groups have one size and
    /// route over a bank that only grows as its use fills it, so the reads of
    /// the group before predict a group's. Before any group has ended there,
    /// every expert the layer lacks now (a prompt's first group reads cold),
    /// those the host tier holds alone pulled where prompt groups pull, the
    /// rest read. A group that reads more than this (a new prompt's first,
    /// routing elsewhere) exposes read time: its steps wait for every expert
    /// they route whatever their merge sets, so nothing computes wrongly.
    /// None until the run has measured every layer's compute.
    pub(super) fn terms(&self) -> Option<Vec<GroupLayerTerms>> {
        (0..self.compute.len())
            .map(|layer| {
                let (filed, tiered) = self.cold(layer);
                Some(GroupLayerTerms { compute_seconds: self.compute[layer]?.as_secs_f64(), filed, tiered })
            })
            .collect()
    }

    /// The experts the next group will read from the file and pull from
    /// the host tier at `layer` ([`Self::terms`]), which need no measured
    /// compute.
    fn cold(&self, layer: usize) -> (u64, u64) {
        let lacked = self.experts[layer]
            .checked_sub(self.held[layer])
            .expect("a layer's bank holds no more experts than the layer has");
        let alone = self.resident.as_ref().map_or(0, |_| self.alone[layer]);
        let filed = lacked
            .checked_sub(alone)
            .expect("the host tier holds alone no more experts than the bank lacks");
        (self.read[layer].unwrap_or(filed as u64), self.pulled[layer].unwrap_or(alone as u64))
    }

    /// [`Telemetry::group_size`] along each layer's `lines` and the read
    /// line `line`: a group runs to capacity until the run has measured
    /// every layer and priced its lines.
    pub(super) fn group_size(&self, lines: Option<&[ExpertLines]>, line: Option<ReadLine>, notice: Notice, capacity: usize) -> usize {
        match (self.terms(), lines, line) {
            (Some(layers), Some(lines), Some(line)) => Stages { layers: layers.as_slice(), lines, line, notice }.size(capacity),
            _ => capacity,
        }
    }

    /// [`Telemetry::merge_sets`] along each layer's `lines`, the read line
    /// `line` and the passes' price `pass`. Until the run has measured
    /// every layer, counted a group's union at each and priced its lines
    /// and passes, nothing prices a set: a layer with nothing cold after it
    /// merges whole, as the rule would, and any other runs its first back
    /// alone, the least compute that starts the next layer's reads before
    /// the pass ends.
    pub(super) fn merge_sets(&self, lines: Option<&[ExpertLines]>, line: Option<ReadLine>, pass: Option<&ExpertPass>, notice: Notice, steps: usize) -> Vec<usize> {
        let unions = self.unions.iter().map(Option::as_ref).collect::<Option<Vec<_>>>();
        match (self.terms(), unions, lines, line, pass) {
            (Some(layers), Some(unions), Some(lines), Some(line), Some(pass)) => {
                Stages { layers: layers.as_slice(), lines, line, notice }.merge_sets(steps, &unions, pass)
            }
            _ => (0..self.compute.len())
                .map(|layer| match layer + 1 < self.compute.len() && self.cold(layer + 1) != (0, 0) {
                    true => steps.saturating_sub(1).max(1),
                    false => steps,
                })
                .collect(),
        }
    }
}

/// What the load's routing is read from: each streamed layer's group and
/// map, its routing records (`routed_route_streamed`, shaped by `words`),
/// and its group reads of the places a group holds, the layer's `experts`
/// words each.
struct Routing {
    maps: Vec<(usize, crate::experts::map::MapView)>,
    records: Vec<Words>,
    words: RegionWords,
    reads: Mutex<GroupReadWords>,
    experts: Vec<usize>,
    /// Who places into the bank, which tells what a source `>= 0` is
    /// ([`Use`]).
    placer: crate::planning::Placer,
    /// `K`: a row's choices, the ranks a hint is hinted at.
    selected: usize,
    /// Per streamed layer, whether a ring row hints it ([`Layer::hints`]).
    /// The first streamed layer is hinted by none.
    hinted: Vec<bool>,
    /// The volumes its files lie on (`ExpertSource::volumes`).
    volumes: Vec<seismic_io::Volume>,
}

/// Each streamed layer's group reads (`BankPart::GroupReads`) of `places`
/// chain places, as the bank holds them now (`ExpertBank::hold_places`).
pub(super) struct GroupReadWords {
    pub(super) layers: Vec<super::PlacedLayerWords>,
    pub(super) places: usize,
}

/// What the ranking's stack is fed, in the device's order: a prompt group's reads
/// as it ends (its passes' records, `Telemetry::group_read`) and a decode
/// step's as it completes (its routing records), fed once every step of the
/// group and of those before it has ended, by group, layer, then sequence.
struct Feed {
    /// The reads not yet fed, each with its group and what it is.
    pending: Vec<(Step, RoutedRead, Fed)>,
    /// The last step's end.
    ended: Option<Instant>,
}

/// What a read fed to the stack is: a decode step's, counted while its load
/// records it, or a prompt group's union, counted into the group-read
/// curve (`Stack::read_group`), with the group's steps, the steps its pass
/// merged at the layer, and the choices its rows routed to each expert read,
/// in the read's order.
enum Fed {
    /// A decode read, counted with its step's launch where the step is
    /// (`counted`).
    Decode { counted: Option<CountedLaunch> },
    Group { steps: usize, merged: usize, choices: Vec<u32> },
}

/// What the stack's readers read of it, as its last feed left it
/// ([`Telemetry::feed_stack`]): none of them feeds it or waits on its lock,
/// so a reader never waits behind the milliseconds a feed takes.
struct Ranked {
    /// The load's counted decode ([`Stack::live`]).
    live: Option<crate::assessment::LiveDecode>,
}

/// Everything telemetry records, under its one lock.
struct Recorded {
    costs: Costs,
    facts: GroupFacts,
    feed: Feed,
    ranked: Ranked,
    /// Each group's committed records, which each resize carries.
    records: Vec<usize>,
}

/// What serving and steps record of the load, and the routing they read.
pub(super) struct Telemetry {
    routing: Routing,
    recorded: Mutex<Recorded>,
    /// The slots each layer commits, as the bank commits them
    /// (`Commitment::slots`, its one owner): every resize of every
    /// placer's, as it happens.
    slots: Arc<AtomicU64>,
    /// The ranking's stack, under a lock of its own: feeding it takes
    /// milliseconds, which no step's completion waits behind, and its
    /// readers read what each feed publishes (`Recorded::ranked`), not the
    /// stack. Taken before `recorded`, never while holding it.
    stack: Mutex<Stack>,
    /// Per layer, whether it is whole: its route writes no routing record,
    /// so it has no reads of a decode step
    /// ([`Residency::publish_policy`](super::residency::Residency::publish_policy),
    /// while no step runs).
    whole: Vec<AtomicBool>,
    /// The first step whose group has not ended: the reads of every group
    /// before it are ready to feed ([`Self::ended`], [`Self::feed`]). A
    /// step's end only raises it, so the thread ending steps never waits
    /// on the stack.
    feedable: AtomicU64,
    /// The share busy with what steps need that hints are priced at in
    /// place of the one measured ([`Self::price_at`]).
    #[cfg(test)]
    priced_busy: Mutex<Option<f64>>,
    /// Whether a block's span is the device's cost of its work: its backend
    /// times spans of a graph run (`seismic::BackendName::times_spans`) and
    /// its parks wait inside kernels (`parks_in_kernels`). Where parks are
    /// submission boundaries (Vulkan), a block's span holds its parks' host
    /// answers and its submissions' starts, which its line would price as
    /// the device's cost of a hit or a link miss. Where either fails, no
    /// step is sampled and the job's lines stand.
    spans: bool,
    /// The block runs a sampled step's streamed layers took, by step and
    /// layer, until the step completes or ends ([`Self::block_sampled`]);
    /// under a lock of its own, never held with another.
    sampled: Mutex<BTreeMap<(Step, usize), Sampled>>,
}

/// A sampled block run: its spans and the marks at its ends.
struct Sampled {
    spans: seismic::NativeSpans,
    marks: (usize, usize),
}

/// A completed sampled decode step's blocks: the lines it was sampled by,
/// whose rows past each expert's first its blocks' times are taken less
/// (`block_point`), the device seconds of each streamed layer's block that
/// was sampled (a whole block's, `PreparedTargetBlockGraph::span`), and
/// what each layer read, none of a whole layer ([`Telemetry::read_rows`]).
struct SampledBlocks {
    lines: Arc<[ExpertLines]>,
    seconds: BTreeMap<usize, f64>,
    rows: Vec<Option<ReadRows>>,
}

/// A streamed layer's experts one step read, each as its rows (the choices
/// that named it), by where its route served them (`Use::of`): its hits
/// (`Use::Bank`), then its misses, those a record misses land into served
/// (`Use::Tier`) and those the file's (`Use::File`, as `Telemetry::absent`).
struct ReadRows {
    hits: Vec<u64>,
    misses: Vec<u64>,
}

impl Costs {
    /// Group `group`'s landing line now, none until landings of two counts
    /// make one. Its points are counts and clock seconds, finite and not
    /// negative, so every fit of them is a line.
    fn landing(&self, group: usize) -> Option<FileLine> {
        self.landings[group].state().expect("landings' counts and clock seconds make a line").line().map(FileLine::of)
    }

    /// Streamed layer `layer`'s CPU site line as its requests took
    /// ([`SiteRequests::line`]).
    fn site(&self, layer: usize) -> Option<[f64; 2]> {
        self.sites[layer].line()
    }

    /// Streamed layer `layer`'s block line as its sampled steps took,
    /// `[fixed, per hit, per link miss, per sited miss]` seconds, each an
    /// expert of one row; none before a step was observed. Its experts'
    /// costs are seeded lightly by the job's `lines`, which place only what
    /// the observations leave unplaced: a decode that routes as many
    /// experts each step pays the fixed term in step with their sum, which
    /// its observations alone never tell apart.
    fn block(&self, layer: usize, lines: &ExpertLines) -> Option<[f64; 4]> {
        let mut line = self.blocks[layer];
        let cpu = lines.host.cpu.map_or(0., |cpu| cpu.expert.one.get() * 1e-9);
        for (term, seconds) in [(1, lines.gpu.resident.nanoseconds(1.) * 1e-9), (2, lines.gpu.link.nanoseconds(1.) * 1e-9), (3, cpu)] {
            line.seed(std::array::from_fn(|at| f64::from(u8::from(at == term))), seconds, BLOCK_SEED_WEIGHT);
        }
        match line.state().expect("blocks' counts and device seconds make a line") {
            seismic::measure::LineState::Pending => None,
            seismic::measure::LineState::Observed(terms) => Some(terms.costs()),
            seismic::measure::LineState::Seeded(_) => unreachable!("the seeds leave a block's fixed term unplaced"),
        }
    }
}

/// The weight of each seed of a block line (`Costs::block`): one
/// observation's, light beside the steps a load samples.
const BLOCK_SEED_WEIGHT: f64 = 1.;

/// Why a completed step's costs could not be recorded: the served file's
/// read line is lost, or a sampled block's stamps did not read (the
/// device's error, the step having passed them). Either ends serving
/// (`Shared::failed_completion`).
#[derive(Debug)]
pub(super) enum CompletionError {
    ReadLine(ReadLineError),
    Stamps { layer: usize, error: String },
}

impl From<ReadLineError> for CompletionError {
    fn from(error: ReadLineError) -> Self {
        Self::ReadLine(error)
    }
}

/// A streamed layer's CPU site requests as they took, each observed at its
/// experts against its path's time less its rows past each expert's first
/// at the job's row cost, as a block's are (`block_point`): the site's line
/// of a row an expert (`ExpertLines::sited`).
#[derive(Clone, Copy, Debug, Default)]
struct SiteRequests(seismic::measure::Line<2>);

impl SiteRequests {
    /// A request that served `experts` experts of `rows` rows in `seconds`,
    /// a row past an expert's first at `row` seconds.
    fn observe(&mut self, experts: u64, rows: u64, seconds: f64, row: f64) {
        let extra = rows.checked_sub(experts).expect("every expert a request serves names a row");
        self.0.observe([1., experts as f64], seconds - row * extra as f64);
    }

    /// The `[fixed, per expert]` seconds of a request whose experts each
    /// take one row; none before the requests tell the two apart.
    fn line(&self) -> Option<[f64; 2]> {
        match self.0.state().expect("site requests' experts and clock seconds make a line") {
            seismic::measure::LineState::Pending => None,
            seismic::measure::LineState::Observed(terms) => Some(terms.costs()),
            seismic::measure::LineState::Seeded(_) => unreachable!("a site's requests seed nothing"),
        }
    }
}

impl Telemetry {
    /// The telemetry of `layers` of `extents`, their group reads `reads`,
    /// committing the bank's `slots` slots and each group `records` records,
    /// each layer's pull of `resident`
    /// bytes where prompt groups pull from a host tier, nothing recorded.
    pub(super) fn new(
        layers: &[Layer],
        reads: GroupReadWords,
        extents: Extents,
        slots: Arc<AtomicU64>,
        records: Vec<usize>,
        resident: Option<Vec<u64>>,
        ranking: Ranking,
        volumes: Vec<seismic_io::Volume>,
        spans: bool,
    ) -> Self {
        let groups = layers.iter().map(|layer| layer.group + 1).max().unwrap_or(0);
        let looked = vec![seismic_io::Totals::default(); volumes.len()];
        let stack = Stack::new(ranking, layers.len(), extents.widest);
        Self {
            routing: Routing {
                maps: layers.iter().map(|layer| (layer.group, layer.map.clone())).collect(),
                records: layers.iter().map(|layer| layer.routing.clone()).collect(),
                words: extents.words,
                selected: extents.selected,
                reads: Mutex::new(reads),
                experts: layers.iter().map(|layer| layer.experts).collect(),
                placer: extents.placer,
                hinted: (0..layers.len())
                    .map(|target| layers.iter().any(|layer| layer.hints.contains(&target)))
                    .collect(),
                volumes,
            },
            recorded: Mutex::new(Recorded {
                costs: Costs {
                    layers: vec![LayerCost::default(); layers.len()],
                    reads: RequestReads::default(),
                    line: None,
                    landings: vec![seismic::measure::Line::new(); groups],
                    sites: vec![SiteRequests::default(); layers.len()],
                    blocks: vec![seismic::measure::Line::new(); layers.len()],
                    convert: vec![None; groups],
                    hints: HintCost::default(),
                    hinted: BTreeMap::new(),
                    recent: Recent::default(),
                    looked,
                    running: BTreeMap::new(),
                    completed: None,
                    group: None,
                },
                facts: GroupFacts {
                    compute: vec![None; layers.len()],
                    held: vec![0; layers.len()],
                    alone: vec![0; layers.len()],
                    read: vec![None; layers.len()],
                    pulled: vec![None; layers.len()],
                    unions: vec![None; layers.len()],
                    bytes: layers
                        .iter()
                        .map(Layer::expert_bytes)
                        .collect(),
                    resident,
                    experts: layers.iter().map(|layer| layer.experts).collect(),
                },
                feed: Feed { pending: Vec::new(), ended: None },
                ranked: Ranked { live: stack.live() },
                records,
            }),
            slots,
            stack: Mutex::new(stack),
            whole: layers.iter().map(|_| AtomicBool::new(false)).collect(),
            feedable: AtomicU64::new(0),
            #[cfg(test)]
            priced_busy: Mutex::new(None),
            spans,
            sampled: Mutex::new(BTreeMap::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Recorded> {
        self.recorded.lock().expect("watcher telemetry")
    }

    /// The run's read line now; none until its reads make one.
    pub(super) fn line(&self) -> Option<ReadLine> {
        self.lock().costs.line
    }

    /// `count` landings of `layer`'s request ended `seconds` after their
    /// read (`FileLine`): a point of its group's landing line.
    pub(super) fn landings_ended(&self, layer: usize, count: usize, seconds: f64) {
        let group = self.routing.maps[layer].0;
        self.lock().costs.landings[group].observe([1., count as f64], seconds);
    }

    /// Each bank group's device convert takes its `lines` from now on.
    pub(super) fn device_convert(&self, lines: &[crate::assessment::RecordLine]) {
        self.lock().costs.convert = lines.iter().copied().map(Some).collect();
    }

    /// Each streamed layer's landing line now, none until landings of two
    /// counts make one.
    pub(super) fn landing_lines(&self) -> Vec<Option<FileLine>> {
        let recorded = self.lock();
        self.routing.maps.iter().map(|&(group, _)| recorded.costs.landing(group)).collect()
    }

    /// Whether a decode step `sequence` is sampled for its blocks' device
    /// time, where the backend times spans: one step of each flight, the
    /// interval the steps in flight already are, so a block line follows
    /// the load at a stamp a block in each. The step decides it once, as
    /// it begins (`ExpertStep::samples`).
    pub(super) fn samples(&self, sequence: Step) -> bool {
        self.spans && sequence % self.routing.words.flight == 0
    }

    /// Step `sequence`'s run of streamed layer `layer`'s block was sampled
    /// (`spans`, `marks` at its ends): its device time is read once the
    /// step completes ([`Self::complete_step`]), or dropped as it ends
    /// ([`Self::end_step`]).
    pub(super) fn block_sampled(&self, sequence: Step, layer: usize, spans: seismic::NativeSpans, marks: (usize, usize)) {
        self.sampled.lock().expect("the sampled blocks").insert((sequence, layer), Sampled { spans, marks });
    }

    /// Completed step `sequence`'s sampled blocks, sampled by `lines`: each
    /// one's device time, and what each layer read.
    fn sampled_blocks(&self, sequence: Step, lines: Arc<[ExpertLines]>, records: &[RoutingRecord]) -> Result<SampledBlocks, CompletionError> {
        let runs = {
            let mut sampled = self.sampled.lock().expect("the sampled blocks");
            let mut runs = sampled.split_off(&(sequence, 0));
            sampled.append(&mut runs.split_off(&(sequence + 1, 0)));
            runs
        };
        // The step completed, so its work passed both marks: a stamp that
        // does not read is the device's error.
        let seconds = runs
            .into_iter()
            .map(|((_, layer), Sampled { spans, marks: (from, to) })| {
                spans.seconds(from, to).map(|seconds| (layer, seconds)).map_err(|error| CompletionError::Stamps { layer, error: error.to_string() })
            })
            .collect::<Result<_, _>>()?;
        Ok(SampledBlocks { lines, seconds, rows: self.read_rows(records) })
    }

    /// What each streamed layer's routing record of `records` read
    /// ([`ReadRows`]); none of a whole layer, whose route writes no record
    /// of the experts its block ran.
    fn read_rows(&self, records: &[RoutingRecord]) -> Vec<Option<ReadRows>> {
        let placer = self.routing.placer;
        records
            .iter()
            .map(|record| {
                record.written().then(|| {
                    let [mut hits, mut tiered, mut absent] = [(); 3].map(|_| BTreeMap::new());
                    for (expert, source) in record.choices() {
                        let rows = match Use::of(source, placer) {
                            Use::Bank => &mut hits,
                            Use::Tier => &mut tiered,
                            Use::File => &mut absent,
                        };
                        *rows.entry(expert).or_insert(0_u64) += 1;
                    }
                    ReadRows { hits: hits.into_values().collect(), misses: tiered.into_values().chain(absent.into_values()).collect() }
                })
            })
            .collect()
    }

    /// `lines` refined by the load's real work now ([`Self::with_observed`]):
    /// what the route's policy words are written from.
    pub(super) fn refined(&self, lines: &[ExpertLines]) -> Arc<[ExpertLines]> {
        self.with_observed(&self.lock().costs, lines)
    }

    /// `lines` with each layer's landing line, CPU site line and GPU lines
    /// now, which the load observes beside the lines it was priced on: the
    /// landing line of the one site its bank lands at ([`Self::landing`]).
    fn with_observed(&self, costs: &Costs, lines: &[ExpertLines]) -> Arc<[ExpertLines]> {
        lines
            .iter()
            .zip(&self.routing.maps)
            .enumerate()
            .map(|(layer, (lines, &(group, _)))| {
                let site = costs.site(layer);
                lines.landed_by(self.landing(costs, group)).sited(site).passed(costs.block(layer, lines))
            })
            .collect()
    }

    /// Group `group`'s landing line at the one site its bank lands at: the
    /// host's landings observed beside a host tier, the device's convert as
    /// the job measured it where the bank places.
    fn landing(&self, costs: &Costs, group: usize) -> Option<crate::assessment::LandingLine> {
        use crate::assessment::LandingLine;
        match self.routing.placer {
            crate::planning::Placer::Tier => costs.landing(group).map(LandingLine::Host),
            crate::planning::Placer::Bank => costs.convert.get(group).copied().flatten().map(LandingLine::Device),
        }
    }

    /// The slots the bank commits.
    fn slots(&self) -> usize {
        usize::try_from(self.slots.load(Ordering::Acquire)).expect("bank slots fit the host")
    }

    /// A resize left each group committing `records` records (the records
    /// its experts are held in).
    pub(super) fn resized(&self, records: Vec<usize>) {
        self.lock().records = records;
    }

    /// The steps a prompt group runs while decode waits, at most `capacity`:
    /// the fewest whose compute covers each layer's read, `g · compute_l ≥
    /// λ + β_l / D`, with `compute_l` measured on this run, `β_l` the stored
    /// bytes the layer's next group reads ([`GroupFacts::terms`]), and `λ`,
    /// `D` the run's own read line. A group runs to capacity until the run
    /// has measured every layer's compute.
    pub(super) fn group_size(&self, lines: Option<&[ExpertLines]>, notice: Notice, capacity: usize) -> usize {
        let recorded = self.lock();
        let lines = lines.map(|lines| self.with_observed(&recorded.costs, lines));
        recorded.facts.group_size(lines.as_deref(), recorded.costs.line, notice, capacity)
    }

    /// Each streamed layer's merge set in a scratch-served group of `steps`
    /// steps (`Stages::merge_sets`), from this run's measured compute, what
    /// each layer's next group moves ([`GroupFacts::terms`]), each layer's
    /// last group union, each layer's `lines`, the run's own read line and
    /// the passes' price `pass`.
    pub(super) fn merge_sets(&self, lines: Option<&[ExpertLines]>, pass: Option<&ExpertPass>, notice: Notice, steps: usize) -> Vec<usize> {
        let recorded = self.lock();
        let lines = lines.map(|lines| self.with_observed(&recorded.costs, lines));
        recorded.facts.merge_sets(lines.as_deref(), recorded.costs.line, pass, notice, steps)
    }

    /// The costs so far, beside what the CPU site's requests took, `site`.
    pub(super) fn costs(&self, site: crate::experts::site::SiteServed, promoted: u64) -> ExpertCosts {
        let recorded = self.lock();
        let costs = &recorded.costs;
        ExpertCosts {
            layers: costs.layers.clone(),
            reads: costs.reads,
            landings: self.routing.maps.iter().map(|&(group, _)| self.landing(costs, group)).collect(),
            hints: costs.hints,
            completed: costs.completed.clone(),
            group: costs.group.as_ref().map(|counted| counted.cost.clone()),
            promoted,
            slots: self.slots(),
            records: recorded.records.clone(),
            layer_groups: self.routing.maps.iter().map(|(group, _)| *group).collect(),
            experts: self.routing.experts.iter().copied().max().unwrap_or(0),
            site,
        }
    }

    /// Every group before `earliest` has ended: their reads are ready to
    /// feed. Fed by residency's thread before each of its turns, and by any
    /// reader of the stack before it reads ([`Self::fed`]).
    pub(super) fn ended(&self, earliest: Step) {
        self.feedable.fetch_max(earliest, Ordering::AcqRel);
    }

    /// Feed the stack every read ready to feed ([`Self::fed`]).
    pub(super) fn feed(&self) {
        drop(self.fed());
    }

    /// Whether the stack has been fed every read recorded so far: a test's
    /// wait for what a completed step read to rank.
    #[cfg(test)]
    pub(super) fn fed_all(&self) -> bool {
        self.feed();
        self.lock().feed.pending.is_empty()
    }

    /// `layer` is now `whole`, or not.
    pub(super) fn whole(&self, layer: usize, whole: bool) {
        self.whole[layer].store(whole, Ordering::Release);
    }

    /// The stack, fed ([`Self::feed_stack`]).
    fn fed(&self) -> std::sync::MutexGuard<'_, Stack> {
        let mut stack = self.stack.lock().expect("watcher telemetry stack");
        self.feed_stack(&mut stack);
        stack
    }

    /// Feed `stack` the reads of every group that has ended
    /// ([`Self::ended`]), each of whose steps has ended, in the device's
    /// order: a prompt group's union at a layer, ranked at its depths, is
    /// the layer's last group union ([`GroupFacts::unions`]). Only the
    /// taking of the reads and each union's note hold `recorded`.
    fn feed_stack(&self, stack: &mut Stack) {
        let ended = self.feedable.load(Ordering::Acquire);
        let mut reads = {
            let feed = &mut self.lock().feed;
            let (reads, pending) = std::mem::take(&mut feed.pending)
                .into_iter()
                .partition::<Vec<_>, _>(|(group, ..)| *group < ended);
            feed.pending = pending;
            reads
        };
        reads.retain(|(_, read, _)| !read.experts.is_empty());
        reads.sort_by_key(|(group, read, _)| (*group, read.layer, read.sequence));
        let (mut decoded, mut unions) = (false, Vec::new());
        for (_, read, fed) in reads {
            match fed {
                Fed::Decode { counted } => {
                    stack.read(&read, counted);
                    decoded = true;
                }
                Fed::Group { steps, merged, choices } => {
                    let depths = stack.read_group(&read, &choices);
                    unions.push((read.layer, GroupUnion::new(steps, merged, depths.into_iter().zip(choices).collect())));
                }
            }
        }
        // What the feed counted, in one take.
        let live = decoded.then(|| stack.live());
        let recorded = &mut *self.lock();
        for (layer, union) in unions {
            recorded.facts.unions[layer] = Some(union);
        }
        if let Some(live) = live {
            recorded.ranked.live = live;
        }
    }

    /// The depth curve of every decode step the ranking's stack read since
    /// the load ([`Stack::decoded`]), where it read one since `seen` of
    /// them, which it then counts.
    pub(super) fn decoded_since(&self, seen: &mut u64) -> Option<crate::experts::directory::MissCurve> {
        let stack = self.stack.lock().expect("watcher telemetry stack");
        let decoded = stack.decoded();
        (std::mem::replace(seen, decoded.steps) != decoded.steps).then(|| decoded.clone())
    }

    /// Whether layer `layer` holds a group union, every ended step's group
    /// fed: a prompt group's read there ([`GroupFacts::unions`]).
    #[cfg(test)]
    pub(super) fn group_union(&self, layer: usize) -> bool {
        drop(self.fed());
        self.lock().facts.unions[layer].is_some()
    }

    /// The step the stack last fed a read of a layer's expert into the
    /// ranking: the reads of every ended step's group; none for an expert
    /// no step has read.
    #[cfg(test)]
    pub(super) fn last_read(&self) -> impl Fn(usize, usize) -> Option<Step> {
        let ranking = self.fed().ranking();
        move |layer, expert| crate::experts::reuse::Reuse::of(&ranking).last(layer, expert)
    }

    /// Whether the ranking's share of a layer holds its expert
    /// (`Reuse::in_top`), every ended step's group fed.
    #[cfg(test)]
    pub(super) fn in_top(&self) -> impl Fn(usize, usize) -> bool {
        let ranking = self.fed().ranking();
        move |layer, expert| crate::experts::reuse::Reuse::of(&ranking).in_top(layer, expert)
    }

    /// Step `sequence` ended, completed or not: what it left uncompleted,
    /// its cost, its rings and its sampled blocks, is dropped.
    pub(super) fn end_step(&self, sequence: Step) {
        {
            let costs = &mut self.lock().costs;
            costs.running.remove(&sequence);
            costs.hinted.retain(|&(hinted, ..), _| hinted != sequence);
        }
        if self.samples(sequence) {
            self.sampled.lock().expect("the sampled blocks").retain(|&(step, _), _| step != sequence);
        }
    }

    /// Step `sequence` completed (`completion`): its cost is the last
    /// completed step's, and a decode step's routing is fed to the stack,
    /// counted while its load records it. Fails when the served file's
    /// reads make no read line.
    pub(super) fn complete_step(
        &self,
        steps: &Steps,
        sequence: Step,
        completion: CompletedStep,
        sited: &dyn Fn(usize) -> (u64, u64, Duration),
        priced: Option<&[ExpertLines]>,
    ) -> Result<(), CompletionError> {
        let CompletedStep { work, begun, route } = completion;
        // A decode step's routing at each streamed layer, read from its
        // records before anything is recorded.
        let decoded = match route {
            CompletedRoute::Grouped => None,
            CompletedRoute::Decode(mut decoded) => {
                // The step completes, so it holds its records until it ends.
                let records = self.records(steps, sequence);
                if seismic_io::timeline::recording() {
                    for (layer, record) in records.iter().enumerate() {
                        for (expert, source) in record.choices() {
                            seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("route"), sequence, (layer as i64, expert as i64, source as i64));
                        }
                    }
                }
                let blocks = decoded.sampled.take().map(|lines| self.sampled_blocks(sequence, lines, &records)).transpose()?;
                Some((decoded, self.routed(sequence, &records), Self::tiered(&records), self.sourced(&records), Self::absent(&records), blocks))
            }
        };
        let mut recorded = self.lock();
        let Recorded { costs, facts, feed, .. } = &mut *recorded;
        // The last request served whole is its group's last step's at the
        // last streamed layer: its compute runs to the step's completion.
        if let Some(group) = &mut costs.group {
            if let Some(Acknowledged { point, at }) = group.acknowledged.take_if(|served| served.point.sequence == sequence) {
                compute(&mut group.cost, facts, point, at.elapsed());
            }
        }
        let layers = costs.layers.len();
        let mut cost = costs
            .running
            .remove(&sequence)
            .unwrap_or_else(|| StepCost::of(layers));
        // The step's hint candidates by choice rank, into the window that
        // prices hints ([`Recent`]).
        let mut ranks = Ranks::default();
        // The bytes its hints read of experts its routes chose ([`needed`]).
        let mut named_bytes = 0;
        if let Some((_, routed, tiered, sourced, absent, blocks)) = &decoded {
            for (experts, read) in cost.experts.iter_mut().zip(routed) {
                experts.routed = Some(read.clone());
            }
            // What the step routed at each layer: the experts and their
            // choices, whose hits and misses the cost function prices
            // (`LayerStep::exposed`).
            for (step, read) in cost.layers.iter_mut().zip(routed) {
                step.routed = read.experts.len() as u64;
                step.choices = read.choices;
            }
            // The misses the device served from the host tier: the step's
            // tier misses (`LayerStep::hosted`), and reads the tier ranks
            // by.
            for (layer, served) in tiered.iter().enumerate() {
                let hosted = served.len() as u64;
                let total = &mut costs.layers[layer];
                total.misses += hosted;
                total.hosted += hosted;
                total.missed_bytes += hosted * facts.bytes[layer];
                cost.misses += hosted;
                cost.missed_bytes += hosted * facts.bytes[layer];
                cost.layers[layer].hosted += hosted;
                // Of them, those the CPU site served; the device read the
                // rest over the link.
                // Of the step's misses, from its routing record (those
                // resolved to a tier record, and those resolved absent and
                // landed from the file during the step), those the CPU
                // site served; the device read the rest over the link.
                // Its rows past each expert's first at the priced row cost: a
                // site serves only a priced layer, and a request whose lines
                // were unpriced since has no row cost to place it by.
                let (cpu, rows, path) = sited(layer);
                let row = priced.and_then(|lines| lines[layer].host.cpu).map(|cpu| cpu.expert.row.bound().get() * 1e-9);
                if let Some(row) = row.filter(|_| cpu > 0) {
                    costs.sites[layer].observe(cpu, rows, path.as_secs_f64(), row);
                }
                served_within(cpu, hosted, absent[layer], layer, sequence);
                // A sampled step's block at this layer, over what it read,
                // where its route recorded it.
                if let Some((blocks, read, &seconds)) =
                    blocks.as_ref().and_then(|blocks| Some((blocks, blocks.rows[layer].as_ref()?, blocks.seconds.get(&layer)?)))
                {
                    let (x, seconds) = block_point(&blocks.lines[layer], read, (cpu, rows), seconds);
                    costs.blocks[layer].observe(x, seconds);
                }
                total.sited += cpu;
                (cost.experts[layer].sited, cost.experts[layer].site_path) = (cpu, path);
                cost.experts[layer].missed.extend(served.iter().copied());
            }
            // The step's hints resolved against what its routes chose at
            // the hinted layers: the prediction's precision ([`saves`]).
            let hints = &mut costs.hints;
            // The file misses a hint could have saved, at layers a ring
            // hints, and those no ring reaches.
            for (layer, &absent) in absent.iter().enumerate() {
                match self.routing.hinted[layer] {
                    true => hints.hintable += absent,
                    false => hints.unhintable += absent,
                }
            }
            // An expert two leads named counts once in the step's recall and
            // precision (one a longer lead's hint did not read is a shorter
            // lead's candidate again); each lead's rank counts it apart.
            let mut counted = std::collections::BTreeSet::new();
            for (&(_, layer, lead), hinted) in costs.hinted.range((sequence, 0, 0)..=(sequence, usize::MAX, usize::MAX)) {
                for &candidate in hinted {
                    let used = sourced
                        .iter()
                        .find(|&&(at, chosen, used)| at == layer && chosen == candidate.expert && saves(used))
                        .map(|&(.., used)| used);
                    let named = used.is_some();
                    ranks.count(lead, candidate.rank, named);
                    // The reads its hints made for the layer, which the
                    // step's record counts: row `lead` of a ring hints
                    // `lead + 1` layers ahead.
                    if candidate.read {
                        named_bytes += u64::from(named) * facts.bytes[layer];
                        let reads = &mut cost.layers[layer];
                        reads.hint_reads += 1;
                        reads.used += u64::from(named);
                        reads.lead += lead as u64 + 1;
                    }
                    if counted.insert((layer, candidate.expert)) {
                        hints.candidates += 1;
                        hints.named += u64::from(named);
                        // Named and not absent: its hint landed before the route.
                        hints.early += u64::from(used.is_some_and(|used| used != Use::File));
                    }
                }
            }
            hints.layer_steps += layers as u64;
        }
        // Steps complete in order: every ring of this step or an earlier
        // one is resolved or its step ended.
        costs.hinted.retain(|&(hinted, ..), _| hinted > sequence);
        // Every step's end, from which the next decode step's time is taken.
        let now = Instant::now();
        let ended = (feed.ended.replace(now), now);
        if let Some(begun_at) = seismic_io::timeline::stamp(begun) {
            let previous = ended.0.and_then(seismic_io::timeline::stamp).map_or(-1, |previous| previous as i64);
            seismic_io::timeline::note_at(now, seismic_io::timeline::Kind::Noted("step"), sequence, (begun_at as i64, previous, (work == LaunchWork::Generation) as i64));
        }
        // What the volumes read since the step before: a generation step's
        // busy time, over its time, whatever its form (a verify wider than
        // the decode form ranks no hints). A prompt step's reads drain a
        // burst no generation step measured: the window starts again (as busy
        // as can be until it is whole).
        match work {
            LaunchWork::Generation => {
                cost.seconds = step_seconds(ended.0, begun, ended.1);
                let busy;
                (busy, cost.storage) = self.delivered(costs);
                costs.recent.step(cost.seconds, ranks, needed(busy, &cost.storage, named_bytes));
            }
            LaunchWork::Prompt => {
                (_, cost.storage) = self.delivered(costs);
                costs.recent = Recent::default();
            }
        }
        costs.line = costs.reads.measured()?;
        if let Some((decoded, routed, ..)) = decoded {
            feed_decode(feed, routed, decoded);
        }
        // What the step cost in reads, for whoever traces steps (a step slow
        // of its own is read by its misses and reads).
        tracing::debug!(
            target: STEP_TRACE,
            sequence,
            misses = cost.misses,
            missed_bytes = cost.missed_bytes,
            read_micros = cost.read.as_micros() as u64,
            parked_micros = cost.parked.as_micros() as u64,
            storage = ?cost.storage,
            "expert step completed"
        );
        // Steps run in submission order on one queue: the device is past
        // every earlier one too.
        costs.running.retain(|running, _| *running > sequence);
        costs.completed = Some((sequence, cost));
        Ok(())
    }

    /// Visit the words of group reads record `record` at chain place `place`
    /// of each streamed layer (`BankPart::GroupReads`), as the bank holds
    /// them now: the layer, its words, where the place's begin, and its
    /// experts.
    fn reads_at(&self, record: usize, place: usize, mut visit: impl FnMut(usize, &Words, usize, usize)) {
        let reads = self.routing.reads.lock().expect("watcher group reads");
        for (layer, (placed, &experts)) in reads.layers.iter().zip(&self.routing.experts).enumerate() {
            visit(layer, &placed.group_reads, (record * reads.places + place) * placed.stride, experts);
        }
    }

    /// The bank holds `reads` from now: the group reads of the places a
    /// group holds. Between flights, when no group holds a record.
    pub(super) fn place(&self, reads: GroupReadWords) {
        *self.routing.reads.lock().expect("watcher group reads") = reads;
    }

    /// Zero group reads record `record` at the first `places` chain places,
    /// as a group of that many steps takes it, before any of them is queued.
    pub(super) fn clear_reads(&self, record: usize, places: usize) {
        for place in 0..places {
            self.reads_at(record, place, |_, words, base, experts| {
                (base..base + experts).for_each(|word| words.store(word, 0));
            });
        }
    }

    /// A group of grouped steps ended (`reads`): each streamed layer's
    /// experts its passes grouped a choice to, at any of its places, are
    /// read at its last step, in the device's order. A prompt group's are
    /// its group read, with the choices routed to each over its places and
    /// the steps the layer's pass merged; a generation group's (a verify
    /// wider than the decode form) are a decode's reads, each expert read
    /// once, which rank the bank as any decode's do and price no prompt.
    /// Neither is counted into the decode record, which counts the decode
    /// form's routing records alone.
    pub(super) fn group_read(&self, reads: EndedReads) {
        let EndedReads { first, steps, record, merges, prompt } = reads;
        let mut choices = self.routing.experts.iter().map(|&experts| vec![0u32; experts]).collect::<Vec<_>>();
        for place in 0..steps as usize {
            self.reads_at(record, place, |layer, words, base, _| {
                for (expert, count) in choices[layer].iter_mut().enumerate() {
                    *count += u32::try_from(words.load(base + expert)).expect("a group read counts choices");
                }
            });
        }
        let mut recorded = self.lock();
        for ((layer, choices), merged) in choices.into_iter().enumerate().zip(merges) {
            let (experts, choices): (Vec<_>, Vec<_>) =
                choices.into_iter().enumerate().filter(|&(_, count)| count > 0).unzip();
            let routed = choices.iter().map(|&count| u64::from(count)).sum();
            let read = RoutedRead { sequence: first + steps - 1, layer, experts, choices: routed };
            let fed = match prompt {
                true => Fed::Group { steps: steps as usize, merged, choices },
                false => Fed::Decode { counted: None },
            };
            recorded.feed.pending.push((first, read, fed));
        }
    }

    /// What the load's counted decode read and ran so far, as the stack's
    /// last feed left it ([`Ranked`]); none before it counted a step.
    pub(super) fn live(&self) -> Option<crate::assessment::LiveDecode> {
        self.lock().ranked.live.clone()
    }

    /// Step `sequence`'s routing record at each streamed layer, by layer:
    /// each copied while the step holds them ([`RoutingRecord::copy`]),
    /// none of a whole layer, then read.
    fn records(&self, steps: &Steps, sequence: Step) -> Vec<RoutingRecord> {
        let Routing { records, words, .. } = &self.routing;
        let copied = steps
            .while_owned(sequence, |owned| {
                records
                    .iter()
                    .enumerate()
                    .map(|(layer, record)| (!self.whole[layer].load(Ordering::Acquire)).then(|| RoutingRecord::copy(owned, record, *words)))
                    .collect::<Vec<_>>()
            })
            .expect("a completing step owns its routing records");
        copied
            .into_iter()
            .enumerate()
            .map(|(layer, copied)| copied.map_or_else(RoutingRecord::unwritten, |copied| RoutingRecord::read(copied, *words, layer)))
            .collect()
    }

    /// What completed decode step `sequence` routed to at each streamed
    /// layer, by its `records`, ascending: a choice of no expert is the
    /// device's broken invariant.
    fn routed(&self, sequence: Step, records: &[RoutingRecord]) -> Vec<RoutedRead> {
        records
            .iter()
            .enumerate()
            .map(|(layer, record)| {
                let mut routed = record
                    .choices()
                    .map(|(expert, _)| {
                        usize::try_from(expert)
                            .ok()
                            .filter(|expert| *expert < self.routing.experts[layer])
                            .unwrap_or_else(|| panic!("step {sequence}'s routing record at layer {layer} names expert {expert}"))
                    })
                    .collect::<Vec<_>>();
                routed.sort_unstable();
                routed.dedup();
                RoutedRead { sequence, layer, experts: routed, choices: record.count() as u64 }
            })
            .collect()
    }

    /// The experts each streamed layer's routing record of `records`
    /// resolved absent (`routed.seismic`'s `resolve`: the file's), each
    /// once: misses landed into the tier during the step.
    fn absent(records: &[RoutingRecord]) -> Vec<u64> {
        records
            .iter()
            .map(|record| {
                let mut absent = record.choices().filter(|&(_, source)| source == -1).map(|(expert, _)| expert).collect::<Vec<_>>();
                absent.sort_unstable();
                absent.dedup();
                absent.len() as u64
            })
            .collect()
    }

    /// The experts each streamed layer's routing record of `records`
    /// names served from the host tier (a source the route resolved to a
    /// tier record, `routed.seismic`'s `resolve`), each once.
    fn tiered(records: &[RoutingRecord]) -> Vec<Vec<usize>> {
        records
            .iter()
            .map(|record| {
                let mut served = record.choices().filter(|&(_, source)| source <= -2).map(|(expert, _)| expert as usize).collect::<Vec<_>>();
                served.sort_unstable();
                served.dedup();
                served
            })
            .collect()
    }

    /// Each choice of each streamed layer's routing record of `records`,
    /// (layer, expert, use), by the source the route resolved
    /// it to (`routed.seismic`'s `resolve`): a bank record no landing writes
    /// served it, a record misses land into, or none (absent: the file's).
    fn sourced(&self, records: &[RoutingRecord]) -> Vec<(usize, usize, Use)> {
        let placer = self.routing.placer;
        records
            .iter()
            .enumerate()
            .flat_map(|(layer, record)| record.choices().map(|(expert, source)| (layer, expert as usize, Use::of(source, placer))).collect::<Vec<_>>())
            .collect()
    }

    /// The experts each streamed layer's slot map names held, ascending.
    pub(super) fn held(&self) -> Vec<Vec<usize>> {
        self.routing
            .maps
            .iter()
            .map(|(_, map)| (0..map.experts()).filter(|&expert| map.held(expert).is_some()).collect())
            .collect()
    }

    /// Count a request of step `sequence` at `layer` that read experts.
    pub(super) fn serve(&self, layer: usize, sequence: Step, served: RequestCost) {
        if served.misses == 0 {
            return;
        }
        let costs = &mut self.lock().costs;
        let [reading, demand] = served.anticipated;
        let cost = &mut costs.layers[layer];
        cost.misses += served.misses;
        cost.missed_bytes += served.bytes;
        cost.hosted += served.hosted;
        cost.reading += reading;
        cost.unanticipated += demand;
        if let Some(lead) = served.lead {
            cost.lead += lead;
            cost.led += 1;
        }
        cost.parked += served.parked;
        cost.read += served.read;
        let layers = costs.layers.len();
        let step = costs.running.entry(sequence).or_insert_with(|| StepCost::of(layers));
        step.misses += served.misses;
        step.missed_bytes += served.bytes;
        step.parked += served.parked;
        step.read += served.read;
        let timeline = &mut step.layers[layer];
        timeline.misses += served.misses - served.hosted;
        timeline.hosted += served.hosted;
        timeline.hinted += reading;
        let experts = &mut step.experts[layer];
        experts.missed.extend(served.missed);
    }

    /// A read step `sequence`'s decode request at `layer` named landed,
    /// `read` after the request named it: where it was the request's one
    /// demand job of all its missed bytes (`timed`: those bytes, and the
    /// job's own seconds from its submit, of which `queued` is what it
    /// waited for a worker), a point of the served file's read line
    /// (`ExpertCosts::reads`), which it refits. Why the refit line is
    /// none of a broken clock's.
    pub(super) fn landed(&self, layer: usize, sequence: Step, read: Duration, timed: Option<(u64, Duration)>, queued: Duration) -> Result<(), ReadLineError> {
        let mut recorded = self.lock();
        let costs = &mut recorded.costs;
        let cost = &mut costs.layers[layer];
        cost.read += read;
        if let Some(step) = costs.running.get_mut(&sequence) {
            step.read += read;
        }
        if let Some((bytes, job)) = timed {
            costs.reads.add(bytes, job.saturating_sub(queued).as_secs_f64());
            costs.line = costs.reads.measured()?;
        }
        Ok(())
    }

    /// Step `sequence`'s ring row of lead `lead` hinting layer `layer` was
    /// read before that layer's request, hinting `candidates`, each with
    /// whether its hint was read and its rank.
    pub(super) fn hinted(&self, sequence: Step, layer: usize, lead: usize, candidates: Vec<Admitted>) {
        self.lock().costs.hinted.insert((sequence, layer, lead), candidates);
    }

    /// Count what each of the source's volumes read since the step before
    /// into `costs.reads`, from their counts (`seismic_io::accounting`):
    /// the seconds the busiest was busy, and each volume's span, the step's
    /// attribution.
    fn delivered(&self, costs: &mut Costs) -> (f64, Vec<(seismic_io::Volume, seismic_io::Totals)>) {
        let mut busiest = 0f64;
        let spans = self
            .routing
            .volumes
            .iter()
            .zip(&mut costs.looked)
            .map(|(&volume, looked)| {
                let now = seismic_io::accounting::totals(volume);
                let span = now.since(looked);
                *looked = now;
                costs.reads.deliver(span.bytes(), span.busy);
                busiest = busiest.max(span.busy);
                (volume, span)
            })
            .collect();
        (busiest, spans)
    }

    /// What a hint at lead `lead` is priced by now ([`HintPrice`]): each of
    /// the lead's ranks' precision, the volume's busy share, a streamed
    /// layer's time over the latest decode time and the volume's read line
    /// ([`Recent::ranks`], [`Recent::busy`], [`Recent::mean_step`]), read
    /// once for an admission.
    pub(super) fn price(&self, lead: usize) -> HintPrice {
        let recorded = self.lock();
        let costs = &recorded.costs;
        let window = costs.recent.ranks();
        let ranks = (0..self.routing.selected).map(|rank| window.precision(lead, rank)).collect();
        let layer = costs.recent.mean_step().map(|step| step / self.routing.maps.len() as f64);
        let busy = costs.recent.busy();
        #[cfg(test)]
        let busy = self.priced_busy.lock().expect("the priced busy share").unwrap_or(busy);
        HintPrice { precision: ranks, busy, layer, line: costs.line }
    }

    /// Price hints on a volume busy `busy` of the time with what steps
    /// need: a harness's steps compute nothing, so every step it times is
    /// read-bound and, measured, would price every hint out; its hints'
    /// mechanics are tested on a compute-bound load's volume.
    #[cfg(test)]
    pub(super) fn price_at(&self, busy: f64) {
        *self.priced_busy.lock().expect("the priced busy share") = Some(busy);
    }

    /// Count a hint read of `bytes`.
    pub(super) fn hint(&self, bytes: u64) {
        let mut recorded = self.lock();
        let costs = &mut recorded.costs;
        costs.hints.reads += 1;
        costs.hints.bytes += bytes;
    }

    /// Count what a scratch request at `point` of the group whose last step
    /// is `last` took into its group's cost: the union of the requests taken
    /// so far, its own hits, reads, conveyed, pulled and seeded experts and
    /// wait.
    /// The group's last step's request there ends its reads there: what
    /// they read is the size rule's ([`GroupFacts::terms`]). A layer's
    /// requests are taken in their steps' order, every step's front of a
    /// layer publishing before any back runs.
    pub(super) fn group(&self, point: Point, last: Step, took: GroupLayer) {
        let mut recorded = self.lock();
        let Recorded { costs, facts, .. } = &mut *recorded;
        let bytes = facts.bytes[point.layer];
        let group = counted(costs, point.group, last);
        let layer = &mut group.cost.layers[point.layer];
        layer.union = took.union;
        layer.hits += took.hits;
        layer.conveyed += took.conveyed;
        layer.cold_bytes += took.conveyed * bytes;
        layer.pulled += took.pulled;
        layer.tier_bytes += took.pulled * facts.resident.as_ref().map_or(0, |resident| resident[point.layer]);
        layer.seeded += took.seeded;
        layer.parked += took.parked;
        if point.sequence == last {
            facts.read[point.layer] = Some(layer.conveyed);
            facts.pulled[point.layer] = Some(layer.pulled);
        }
    }

    /// `mark` came at `at` at layer `layer` of the group `group`, whose
    /// last step is `last` ([`LayerTimeline`]).
    pub(super) fn mark(&self, group: Step, last: Step, layer: usize, mark: Mark, at: Instant) {
        let mut recorded = self.lock();
        let counted = counted(&mut recorded.costs, group, last);
        let opened = *counted.opened.get_or_insert(at);
        counted.cost.layers[layer].timeline.mark(group, mark, at.saturating_duration_since(opened));
    }

    /// The scratch request at `served`'s point was served whole at its
    /// instant (none after a failed one: the device's time after it is no
    /// layer's compute).
    pub(super) fn served_whole(&self, served: Option<(Point, Instant)>) {
        if let Some(group) = &mut self.lock().costs.group {
            group.acknowledged = served
                .filter(|(served, _)| served.group == group.cost.first)
                .map(|(point, at)| Acknowledged { point, at });
        }
    }

    /// A request of group `group` arrived at its round 0 at `started`: the
    /// device waited for nothing since the group's request before it in
    /// device order was served whole (its own reads come after), so that is
    /// the request's compute: its back, and the fronts at the next layer it
    /// carries, or its share of its merge set's (a set's pass and backs
    /// follow its last fill, the rest its fills' tails).
    pub(super) fn arrived(&self, group: Step, started: Instant) {
        let mut recorded = self.lock();
        let Recorded { costs, facts, .. } = &mut *recorded;
        if let Some(counted) = costs.group.as_mut().filter(|counted| counted.cost.first == group) {
            if let Some(Acknowledged { point, at }) = counted.acknowledged {
                compute(&mut counted.cost, facts, point, started.saturating_duration_since(at));
            }
        }
    }

    /// Publish the experts each layer holds, `held`, to the size rule, and
    /// those its host records hold alone.
    pub(super) fn publish_held(&self, held: &[usize], alone: &[usize]) {
        let mut recorded = self.lock();
        recorded.facts.held.copy_from_slice(held);
        recorded.facts.alone.copy_from_slice(alone);
    }

    /// Per layer, the experts its bank holds and those its host records hold
    /// alone, as last published together.
    pub(super) fn holdings(&self) -> (Vec<usize>, Vec<usize>) {
        let recorded = self.lock();
        (recorded.facts.held.clone(), recorded.facts.alone.clone())
    }
}

/// The count of group `first`, whose last step is `last`: a newer group's
/// count replaces an older one's.
fn counted(costs: &mut Costs, first: Step, last: Step) -> &mut Counted {
    if costs.group.as_ref().is_none_or(|group| group.cost.first != first) {
        costs.group = Some(Counted {
            cost: GroupCost {
                first,
                steps: last - first + 1,
                layers: vec![GroupLayer::default(); costs.layers.len()],
            },
            acknowledged: None,
            opened: None,
        });
    }
    costs.group.as_mut().expect("the group's count, just made where it was another's")
}

/// Count `interval`, the device's time from the request at `served`,
/// `group`'s, being served whole to the next round 0, or the group's last
/// step's to its completion, into `served`'s compute at its layer
/// ([`GroupLayer::compute`]), and publish the layer's step there to the
/// size and merge-set rules (`facts`).
fn compute(group: &mut GroupCost, facts: &mut GroupFacts, served: Point, interval: Duration) {
    let steps = u32::try_from(group.steps).expect("a group's steps");
    let layer = &mut group.layers[served.layer];
    layer.compute += interval;
    layer.computed += 1;
    let step = layer.compute_of(steps).map(|compute| compute / steps);
    facts.compute[served.layer] = step;
}

/// A sampled block's point of its layer's block line: its hits, link
/// misses and sited misses (`sited`: the CPU site's experts and their rows)
/// as experts of one row, against its device `seconds` less what their rows
/// past each one's first take at the step's `lines` (the resident line a
/// hit's, the link line a link miss's, the CPU's row cost a sited one's,
/// as `SiteRequests` takes it). Which misses the site served is known by
/// their count and rows alone, so those are taken off the misses' rows
/// along the link line at their mean rows.
fn block_point(lines: &ExpertLines, read: &ReadRows, sited: (u64, u64), seconds: f64) -> ([f64; 4], f64) {
    let (cpu, cpu_rows) = sited;
    let past = |line: &crate::assessment::PassLine, rows: &[u64]| rows.iter().map(|&rows| ExpertLines::rows_past_first(line, rows as f64)).sum::<f64>();
    let links = (read.misses.len() as u64).checked_sub(cpu).expect("the CPU site serves only misses");
    let sited_link = match cpu {
        0 => 0.,
        _ => cpu as f64 * ExpertLines::rows_past_first(&lines.gpu.link, cpu_rows as f64 / cpu as f64),
    };
    let sited_rows = cpu_rows.checked_sub(cpu).expect("every expert a request serves names a row");
    let sited_cpu = lines.host.cpu.map_or(0., |site| site.expert.row.bound().get() * 1e-9) * sited_rows as f64;
    let rows = past(&lines.gpu.resident, &read.hits) + past(&lines.gpu.link, &read.misses) - sited_link + sited_cpu;
    ([1., read.hits.len() as f64, links as f64, cpu as f64], seconds - rows)
}

/// Layer `layer`'s CPU site served `cpu` of step `sequence`'s misses, of
/// which its routing record resolved `tiered` to tier records and `absent`
/// to the file (landed into the tier during the step): the site serves
/// only misses, so more is an invariant violation.
fn served_within(cpu: u64, tiered: u64, absent: u64, layer: usize, sequence: Step) {
    assert!(
        cpu <= tiered + absent,
        "layer {layer}'s site served {cpu} of step {sequence}'s {tiered} tier and {absent} file misses"
    );
}

/// A completed decode step's routing at every layer, `routed`, is fed;
/// while its load records it, counted with its launch.
fn feed_decode(feed: &mut Feed, routed: Vec<RoutedRead>, decoded: Decoded) {
    let Decoded { recording, launch, group, .. } = decoded;
    // A recorded decode step runs its launch (`ExpertStep::decoding`).
    let counted = recording.map(|Recording| {
        let launch = launch.expect("a recorded decode step has its launch");
        CountedLaunch { rows: launch.rows as u64, tokens: launch.tokens }
    });
    feed.pending.extend(routed.into_iter().map(|read| (group, read, Fed::Decode { counted })));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A candidate the route chose is a miss a hint of it saves wherever a
    /// landing put it, its hint read or not; one a bank record no landing
    /// writes served is not. On unified memory every record the route names
    /// (a source `>= 0`) is one misses land into, so a hint that landed
    /// before its route counts.
    #[test]
    fn a_chosen_candidate_counts_wherever_a_landing_put_it() {
        use crate::planning::Placer;
        assert!(saves(Use::of(-1, Placer::Bank)) && saves(Use::of(-1, Placer::Tier)), "absent: the file's");
        assert!(saves(Use::of(7, Placer::Bank)), "unified: the one store, landed");
        assert!(saves(Use::of(-2 - 7, Placer::Tier)), "a host tier record, landed");
        assert!(!saves(Use::of(7, Placer::Tier)), "a discrete bank record no landing writes");
    }

    /// A site's line is its requests' time less their rows past each
    /// expert's first at the row cost: requests of 5 µs and 40 µs an expert,
    /// some experts of several rows at 2 µs a row, fit 5 and 40 alone; none
    /// before two expert counts.
    #[test]
    fn a_site_line_leaves_out_the_rows_its_row_cost_prices() {
        let mut requests = SiteRequests::default();
        requests.observe(2, 2, 85e-6, 2e-6);
        assert_eq!(requests.line(), None);
        requests.observe(4, 8, 165e-6 + 4. * 2e-6, 2e-6);
        requests.observe(3, 5, 125e-6 + 2. * 2e-6, 2e-6);
        let [fixed, expert] = requests.line().unwrap();
        assert!((fixed - 5e-6).abs() < 1e-12 && (expert - 40e-6).abs() < 1e-12, "{fixed} {expert}");
    }

    /// Recall counts a hint that landed before its route (a hit it made,
    /// which no miss count holds) as saved and wanted, and leaves out the
    /// misses of layers no ring hints; precision is named candidates over
    /// candidates, read or not.
    #[test]
    fn hint_recall_counts_early_landings_and_leaves_out_unhinted_layers() {
        let hints = HintCost { candidates: 10, named: 6, early: 2, hintable: 8, unhintable: 5, ..HintCost::default() };
        assert_eq!(hints.recall(), Some(6. / 10.));
        assert_eq!(hints.precision(), Some(6. / 10.));
        assert_eq!(HintCost::default().recall(), None);
        assert_eq!(hints.since(&hints), HintCost::default());
    }

    /// Each rank's precision is its candidates' share named, from a half
    /// before any is seen, whatever else its neighbours count.
    #[test]
    fn each_rank_counts_its_own_precision() {
        let mut ranks = Ranks::default();
        assert_eq!(ranks.precision(0, 3), 0.5);
        for named in [true, true, false] {
            ranks.count(0, 0, named);
        }
        ranks.count(0, 2, false);
        ranks.count(1, 0, false);
        assert_eq!(ranks.precision(0, 0), 3. / 5.);
        assert_eq!(ranks.precision(0, 1), 0.5, "no candidate at rank 1");
        assert_eq!(ranks.precision(0, 2), 1. / 3.);
        assert_eq!(ranks.precision(1, 0), 1. / 3., "each lead's rank apart");
    }

    /// A rank's precision is its candidates' over the latest window of
    /// decode time: a transient's precise steps leave it as they age out.
    #[test]
    fn a_ranks_precision_follows_the_latest_window() {
        let ranks = |named: u64| {
            let mut ranks = Ranks::default();
            for index in 0..10 {
                ranks.count(0, 7, index < named);
            }
            ranks
        };
        // Steps of 1/16 s, exact in binary: a window holds 16.
        let mut recent = Recent::default();
        for _ in 0..16 {
            recent.step(0.0625, ranks(8), 0.);
        }
        assert!((recent.ranks().precision(0, 7) - 129. / 162.).abs() < 1e-12);
        for _ in 0..16 {
            recent.step(0.0625, ranks(2), 0.);
        }
        assert!((recent.ranks().precision(0, 7) - 33. / 162.).abs() < 1e-12, "the transient's steps left: {}", recent.ranks().precision(0, 7));
    }

    /// The volume's busy share is its busy time over the latest decode
    /// steps' time, at most 1, and none before a step is timed: a gap
    /// between steps (a tool call's) is no step's time, and steps past the
    /// window leave it.
    #[test]
    fn the_busy_share_is_the_volumes_busy_time_over_the_latest_steps() {
        let mut traffic = Recent::default();
        assert_eq!(traffic.busy(), 0., "unmeasured: nothing says it is busy");
        // 50 ms steps, the volume busy 25 ms of each: half busy from the
        // first, and over a whole window.
        traffic.step(0.05, Ranks::default(), 0.025);
        assert!((traffic.busy() - 0.5).abs() < 1e-9, "the first step prices the rest");
        for _ in 0..19 {
            traffic.step(0.05, Ranks::default(), 0.025);
        }
        assert!((traffic.busy() - 0.5).abs() < 1e-9);
        // A step begun after a 30 s idle gap counts from its begin.
        let begun = Instant::now();
        let previous = begun - std::time::Duration::from_secs(30);
        let seconds = step_seconds(Some(previous), begun, begun + std::time::Duration::from_millis(50));
        assert!((seconds - 0.05).abs() < 1e-6, "{seconds}");
        traffic.step(seconds, Ranks::default(), 0.025);
        assert!((traffic.busy() - 0.5).abs() < 1e-6, "the gap did not idle the volume");
        // Steps of an idle volume for a whole window leave the share at 0.
        for _ in 0..40 {
            traffic.step(0.05, Ranks::default(), 0.);
        }
        assert_eq!(traffic.busy(), 0.);
    }

    /// A hint is read where its precision pays for its bytes at the
    /// volume's own rate, which cancels: before the read's latency is
    /// measured, where it is right as often as the volume is busy with what
    /// steps need; a latency it spares too lowers the bar. On an idle volume
    /// every hint pays; on one busy all the time none short of certain does,
    /// the prior's half included.
    #[test]
    fn a_hint_is_read_where_its_rank_pays_for_its_bytes() {
        assert!(pays(0.5, 0.5, 0.) && !pays(0.49, 0.5, 0.));
        assert!(pays(0.3, 0.3, 0.) && !pays(0.29, 0.3, 0.));
        // Flash Next's ~1.9 MB expert at λ = 0.26 ms and 3 GB/s.
        assert!(pays(0.24, 0.3, 0.41) && !pays(0.23, 0.3, 0.41));
        assert!(pays(0.01, 0., 0.));
        assert!(!pays(0.5, 1., 0.), "the prior on a saturated volume");
        assert!(pays(0.5, Recent::default().busy(), 0.), "the prior before a step is timed");
        assert!(pays(1., 1., 0.));
    }

    /// The volume's busy time counts the reads steps need: demand reads and
    /// hints a route chose. Wrong hints' bytes, and what a hastened hint
    /// read due beside them, never count against a hint.
    #[test]
    fn a_volume_is_busy_with_what_steps_need() {
        let span = |due: u64, ahead: u64| seismic_io::Totals {
            due: seismic_io::Class { bytes: due, ..Default::default() },
            ahead: seismic_io::Class { bytes: ahead, ..Default::default() },
            ..Default::default()
        };
        let volume = seismic_io::Volume(0);
        let busy = |due, ahead, named| needed(0.8, &[(volume, span(due, ahead))], named);
        assert!((busy(60, 40, 0) - 0.48).abs() < 1e-12, "every hint wrong");
        assert!((busy(60, 40, 10) - 0.56).abs() < 1e-12);
        assert_eq!(busy(60, 40, 50), 0.8, "a hastened hint read due");
        assert_eq!(busy(100, 0, 0), 0.8, "no hint read");
        assert_eq!(needed(0., &[], 0), 0., "nothing delivered");
    }

    /// On a volume busy with what steps need, a right hint's bytes are read
    /// anyway and save nothing; a wrong one's delay them. DeepSeek V4.1 on an
    /// M4 Pro at 16 slots: busy 0.81 of the step without hints, its best
    /// ranks right about half the time.
    #[test]
    fn a_hint_on_a_saturated_volume_saves_nothing() {
        assert!(!pays(0.45, 0.81, 0.));
        assert!(!pays(0.5, 0.97, 0.));
    }
}

#[cfg(test)]
mod served_tests {
    /// A resident load's first decode step: its routing record resolves one
    /// expert to the tier and seven absent, which land into the tier during
    /// the step; the site serves all eight. More than the step missed is an
    /// invariant violation.
    #[test]
    fn the_site_serves_a_steps_tier_and_landed_misses() {
        super::served_within(8, 1, 7, 0, 1);
    }

    #[test]
    #[should_panic(expected = "site served 9 of step 1's 1 tier and 7 file misses")]
    fn the_site_serving_more_than_the_step_missed_panics() {
        super::served_within(9, 1, 7, 0, 1);
    }
}
