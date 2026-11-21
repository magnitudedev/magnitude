//! What serving costs: each request's and step's misses and reads, the
//! prompt groups' layers, the facts the group size rule reads, and the
//! routing the ranking's stack is fed (the performance model's inputs, and
//! the one ranking's feed: `Reuse`).

use crate::operation::LaunchWork;
use super::core::{Extents, Layer, RoutingRecord};
use super::gate::HintGate;
use super::passes::{DEPTHS, Measured, Pass, Passes, Routed, LAGS};
use super::steps::{CompletedRoute, CompletedStep, DecodeRouting, Decoded, EndedReads, Recording, Step, Steps};
use crate::assessment::{GroupLayerTerms, GroupUnion, LayerStep, Notice, RequestReads, Stages};
use magnitude_cost_model::{ExpertLines, ExpertPass, FileLine, Hits, Miss, ReadLine};
use crate::experts::Words;
use crate::experts::directory::{CountedLaunch, Point, RoutedRead, Stack};
use crate::experts::horizon::Time;
use crate::experts::interest::{Accepted, Basis, Chosen, Cohort, Judged, Resolved, Source, Spent, Stands};
use crate::experts::reuse::{Gap, Ranking, Shadows, StepKeys};
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
    /// The decode steps by whether its hint launches ran (off, on): what
    /// the hint gate chose (`Residency::publish_policy`).
    pub hints: [u64; 2],
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
    /// What the reads ahead's bytes were spent on since the step before:
    /// those issued, and of those the drive read, those a later exact row
    /// used (the rest wasted).
    pub ahead: AheadUse,
    /// Each streamed layer's exact rows' hints for the step, by lead
    /// ([`LeadHints`]).
    pub hints: Vec<[LeadHints; crate::planning::HINT_LEADS as usize]>,
}

/// What one lead of a streamed layer's hint launch did for one decode
/// step, its exact rows' hints alone: the candidates it named for the step
/// (experts living only in the file) and of them those whose probability
/// reached the read-ahead break-even as they were published
/// (`ReadPrice::admits`), the reads ahead its names led and of them those
/// the step's route used, and whether the layer's hint launch ran (its
/// gate in effect as the step completed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeadHints {
    pub published: u64,
    pub admitted: u64,
    pub read: u64,
    pub used: u64,
    pub on: bool,
}

impl LeadHints {
    /// The reads its names led that the route did not use.
    pub fn wasted(&self) -> u64 {
        self.read - self.used
    }
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

impl StepCost {
    fn of(layers: usize) -> Self {
        Self {
            layers: vec![LayerStep::default(); layers],
            experts: vec![LayerExperts::default(); layers],
            hints: vec![[LeadHints::default(); crate::planning::HINT_LEADS as usize]; layers],
            ..Self::default()
        }
    }

    /// The seconds of its time the drive held no piece in flight: its time
    /// less its busiest volume's busy time.
    pub fn drive_idle(&self) -> f64 {
        (self.seconds - self.busiest(|totals| totals.busy)).max(0.)
    }

    /// The seconds its busiest volume held due pieces beside pieces read
    /// ahead (`seismic_io::Totals::shared`): the time reads ahead shared the
    /// drive with its parked reads, within the knee ("Order and depth"), the
    /// most they delayed them.
    pub fn ahead_stall(&self) -> f64 {
        self.busiest(|totals| totals.shared)
    }

    /// The most of `count` of any of its volumes.
    fn busiest(&self, count: impl Fn(&seismic_io::Totals) -> f64) -> f64 {
        self.storage.iter().map(|(_, totals)| count(totals)).fold(0., f64::max)
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
    /// The drive's line now (`seismic_io::Totals::line`, of the volumes
    /// the model's files lie on): none before a demand read ended.
    pub line: Option<ReadLine>,
    /// Each streamed layer's landing line at the one site its bank lands
    /// at: the host's landings observed beside a host tier, none until
    /// observed; the device's convert (its landing launch's) where the bank
    /// places, once measured.
    pub landings: Vec<Option<magnitude_cost_model::LandingLine>>,
    pub hints: HintCost,
    /// What the front rows skipped since the load.
    pub front: FrontSkips,
    /// What the reads ahead's bytes were spent on since the load.
    pub ahead: AheadUse,
    pub completed: Option<(u64, StepCost)>,
    /// The latest prompt group served.
    pub group: Option<GroupCost>,
    /// The promotions asked since the load.
    pub promoted: u64,
    /// The bank's shape: the slots each layer commits, each group's decode
    /// capacity (`Directory::capacities`), each streamed
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

/// A group's window fill served whole: its point, when its last round was
/// acknowledged, and whether it was the step's last scratch window there.
#[derive(Clone, Copy)]
struct Acknowledged {
    point: Point,
    at: Instant,
    last_window: bool,
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

/// The front rows' record since the load: the decode passes with front
/// rows completed, and the reads ahead their names made (their front rings'
/// and their lead hints'), counted here and never as hints' (their bytes
/// are [`AheadUse`]'s); and their guesses the verdicts judged, and of them
/// those that stood, by the guess's kind (`interest::Judged`,
/// `crate::GuessKind::index`) and by depth (index 0 a pass's first front
/// row): a deeper row is judged only where every shallower row of its pass
/// stood, so `stood[d] / judged[d]` is the depth's stand rate given its
/// basis stood, since the load: a report's (the rate a name is priced by is
/// the one owner's over the latest turnover, `interest::Stands`). Beside them,
/// the verify passes' acceptable rows its own verdicts judged
/// (`acceptable_at`) and of them those that stood (`accepted_at`), by
/// depth alike (`interest::Accepted`). An acceptable row deeper than the decode form's
/// (a verification wider than it) is of no pass the regime prices, and
/// counts into none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrontSkips {
    pub passes: u64,
    pub reads: u64,
    pub judged: [u64; crate::GuessKind::ALL.len()],
    pub stood: [u64; crate::GuessKind::ALL.len()],
    pub judged_at: [u64; FRONT_DEPTHS],
    pub stood_at: [u64; FRONT_DEPTHS],
    pub acceptable_at: [u64; FRONT_DEPTHS],
    pub accepted_at: [u64; FRONT_DEPTHS],
}

/// The most front rows a pass runs: the decode form's rows past its exact
/// row (`VERIFY_ROWS`).
pub const FRONT_DEPTHS: usize = magnitude_kernels::constants::VERIFY_ROWS as usize - 1;

impl FrontSkips {
    /// What was counted after `start`, a count of the same load taken earlier.
    pub fn since(&self, start: &Self) -> Self {
        fn less<const N: usize>(now: [u64; N], then: [u64; N]) -> [u64; N] {
            std::array::from_fn(|at| now[at] - then[at])
        }
        Self {
            passes: self.passes - start.passes,
            reads: self.reads - start.reads,
            judged: less(self.judged, start.judged),
            stood: less(self.stood, start.stood),
            judged_at: less(self.judged_at, start.judged_at),
            stood_at: less(self.stood_at, start.stood_at),
            acceptable_at: less(self.acceptable_at, start.acceptable_at),
            accepted_at: less(self.accepted_at, start.accepted_at),
        }
    }
}

/// The bytes of the reads ahead since the load, by the kind of name each
/// was issued for (`interest::Basis`) and whether that name's guess fell
/// (`interest::Spent`): issued, fallen, and of each class cancelled before
/// any piece began and used by a later exact row before it was evicted. The
/// interest ledger is their one producer: the issuing executor counts a
/// read as it issues it, and the ledger's resolution, verdict and expiry
/// spend it. And the reads themselves by the depth of the row whose name
/// led them (`interest::Source::depth`, 0 an exact row's hint), apart by
/// whether that row was a front row or an exact row (index 0) or a verify
/// row (index 1, `interest::Source::verifies`): issued, cancelled and
/// used, of which the rest are moves nothing used. A verify row's route is
/// not among them ([`moves`]): its pass reads every expert its rows route
/// to, fallen or not, each among the pass's reads its misses count (`U`),
/// so the moves nothing used it counts apart (`W'`) are only those no row
/// of the pass reads, its rows' lookahead hints'.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AheadUse {
    issued: [u64; 3],
    fell: [u64; 3],
    cancelled: [[u64; 2]; 3],
    used: [[u64; 2]; 3],
    /// Reads ahead settled unused, by `(verify, depth)` (never counting
    /// pending reads awaiting a future step).
    unused: [[u64; DEPTHS]; 2],
}

/// One class of reads ahead's bytes over a window ([`AheadUse::classes`]):
/// those the drive read (issued, less cancelled before any piece began),
/// and of them those a later exact row used. Signed: a read issued before a
/// window and fallen or cancelled within it moves its bytes across the
/// window's edge, so a short window may show a class below zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadUse {
    pub read: i64,
    pub used: i64,
}

impl ReadUse {
    /// The bytes read that no exact row used before they were evicted.
    pub fn unused(&self) -> i64 {
        self.read - self.used
    }
}

/// Whether a read ahead issued for `basis`, led by a verify row where
/// `verify`, counts among the moves by depth ([`AheadUse::unused`]): every
/// one but a verify row's route, which its own pass reads ([`AheadUse`]).
fn moves(basis: Basis, verify: bool) -> bool {
    !(verify && basis == Basis::Front)
}

impl AheadUse {
    fn issue(&mut self, source: Source, bytes: u64) {
        let basis = Basis::of(source);
        self.issued[basis.index()] += bytes;
    }

    /// Each depth's reads ahead settled unused (never counting pending reads
    /// awaiting a future step), those front and exact rows led at index 0 and
    /// those verify rows led at 1.
    pub fn unused(&self) -> [[f64; DEPTHS]; 2] {
        self.unused.map(|form| form.map(|unused| unused as f64))
    }

    /// The bytes of the reads issued for the front rows' names: their
    /// routes' and their lookahead hints' ([`FrontSkips::reads`]).
    pub fn front_issued(&self) -> u64 {
        self.issued[Basis::Front.index()] + self.issued[Basis::AheadHint.index()]
    }

    /// The bytes of the reads issued for the exact rows' lookahead hints
    /// ([`HintCost::reads`]).
    pub fn hints_issued(&self) -> u64 {
        self.issued[Basis::Hint.index()]
    }

    fn spend(&mut self, spent: Spent) {
        match spent {
            Spent::Fell { basis: Basis::Hint, .. } => panic!("an exact row's hint guesses nothing, so no guess of its falls"),
            Spent::Fell { basis, bytes } => self.fell[basis.index()] += bytes,
            Spent::Cancelled { basis, fell, bytes, .. } => {
                self.cancelled[basis.index()][usize::from(fell)] += bytes;
            }
            Spent::Used { basis, fell, bytes, .. } => {
                self.used[basis.index()][usize::from(fell)] += bytes;
            }
            Spent::Unused { basis, depth, verify, .. } => {
                if moves(basis, verify) {
                    self.unused[usize::from(verify)][usize::from(depth)] += 1;
                }
            }
        }
    }

    /// Every class's bytes together: those issued, and those read and used
    /// ([`Self::classes`]).
    pub fn total(&self) -> (u64, ReadUse) {
        let issued = self.issued.iter().sum();
        let read = self.classes().iter().fold(ReadUse { read: 0, used: 0 }, |total, (_, class)| ReadUse { read: total.read + class.read, used: total.used + class.used });
        (issued, read)
    }

    /// What was counted after `start`, a count of the same load taken earlier.
    pub fn since(&self, start: &Self) -> Self {
        let mut since = *self;
        for at in 0..Basis::ALL.len() {
            since.issued[at] -= start.issued[at];
            since.fell[at] -= start.fell[at];
            for fell in 0..2 {
                since.cancelled[at][fell] -= start.cancelled[at][fell];
                since.used[at][fell] -= start.used[at][fell];
            }
        }
        for (since, start) in since.unused.iter_mut().flatten().zip(start.unused.iter().flatten()) {
            *since -= *start;
        }
        since
    }

    /// Each class's bytes read and used, named: a front row's route
    /// (`front`), an ahead row's lookahead hint (`front_hints`), each of a
    /// guess that stood or fell (`_stood`, `_wrong`: a wrong guess, or a
    /// pass or request that never came), and an exact row's lookahead hint
    /// (`hints`), which guesses nothing.
    pub fn classes(&self) -> [(&'static str, ReadUse); 5] {
        let signed = |bytes: u64| i64::try_from(bytes).expect("a load's bytes fit i64");
        let class = |basis: Basis, fell: bool| {
            let (at, of) = (basis.index(), usize::from(fell));
            let named = match fell {
                true => signed(self.fell[at]),
                false => signed(self.issued[at]) - signed(self.fell[at]),
            };
            ReadUse { read: named - signed(self.cancelled[at][of]), used: signed(self.used[at][of]) }
        };
        [
            ("front_stood", class(Basis::Front, false)),
            ("front_wrong", class(Basis::Front, true)),
            ("front_hints_stood", class(Basis::AheadHint, false)),
            ("front_hints_wrong", class(Basis::AheadHint, true)),
            ("hints", class(Basis::Hint, false)),
        ]
    }
}

/// The reads hints made: hint reads submitted (their bytes are
/// [`AheadUse`]'s). And
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
///   layer's), which no prediction reaches by construction;
/// - the layers of completed decode steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HintCost {
    pub reads: u64,
    pub candidates: u64,
    pub named: u64,
    pub early: u64,
    pub hintable: u64,
    pub unhintable: u64,
    pub layer_steps: u64,
}

impl HintCost {
    /// What was counted after `start`, a count of the same load taken
    /// earlier.
    pub fn since(&self, start: &Self) -> Self {
        Self {
            reads: self.reads - start.reads,
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
        Use::Bank(_) => false,
    }
}

/// One definition of a step's time, from the later of its begin and the
/// last step's end (`previous`) to its end, `now`: a gap with no step in
/// flight (a client's, a tool call's) is no step's time.
fn step_seconds(previous: Option<Instant>, begun: Instant, now: Instant) -> f64 {
    let start = previous.map_or(begun, |previous| previous.max(begun));
    now.saturating_duration_since(start).as_secs_f64()
}

/// Exact wall-clock length of the union of `[start, end]` park intervals
/// clipped to `[step_start, step_end]`. Overlapping concurrent layer reads
/// merge rather than double-count, so the result is guaranteed `<= step_end - step_start`
/// by construction.
fn parked_union(intervals: Vec<(Instant, Instant)>, step_start: Instant, step_end: Instant) -> Duration {
    let mut clipped: Vec<(Instant, Instant)> = intervals
        .into_iter()
        .filter_map(|(start, end)| {
            let s = start.max(step_start);
            let e = end.min(step_end);
            (e > s).then_some((s, e))
        })
        .collect();
    if clipped.is_empty() {
        return Duration::ZERO;
    }
    clipped.sort_unstable_by_key(|&(start, _)| start);
    let mut total = Duration::ZERO;
    let (mut cur_start, mut cur_end) = clipped[0];
    for (start, end) in clipped.into_iter().skip(1) {
        if start <= cur_end {
            cur_end = cur_end.max(end);
        } else {
            total += cur_end.saturating_duration_since(cur_start);
            cur_start = start;
            cur_end = end;
        }
    }
    total + cur_end.saturating_duration_since(cur_start)
}

/// What a read ahead is priced by ([`Telemetry::price`]), whatever named it
/// (its probability is the interest ledger's, `experts::interest`): what a
/// miss costs the decode ([`MissCost`], over the measured passes' window),
/// by which every read ahead is admitted at its break-even
/// ([`Self::admits`]); and the drive's line (none before a demand read
/// ended).
pub(super) struct ReadPrice {
    miss: super::passes::MissCost,
    line: Option<ReadLine>,
    /// The break-even a test admits at in place of the measured one
    /// ([`Telemetry::admit_at`]).
    #[cfg(test)]
    even: Option<f64>,
}

impl ReadPrice {
    /// A price before anything is measured.
    #[cfg(test)]
    pub(super) fn unpriced() -> Self {
        Self { miss: super::passes::MissCost::default(), line: None, even: None }
    }

    /// A price with an explicit miss cost and drive line.
    #[cfg(test)]
    pub(super) fn priced(miss: super::passes::MissCost, line: ReadLine) -> Self {
        Self { miss, line: Some(line), even: None }
    }

    /// Whether a read ahead of `bytes` of an expert used with probability
    /// `p` is admitted with no idle compute overlap (`idle = 0`).
    pub(super) fn admits(&self, p: f64, bytes: u64) -> bool {
        self.admits_with_idle(p, bytes, 0.)
    }

    /// Whether a read ahead of `bytes` of an expert used with probability
    /// `p` is admitted given `idle` seconds of remaining per-layer idle
    /// compute (`MissCost::break_even`, `designs/streamed-decode.md` §5).
    /// Before the line is measured a read's time is unknown, and every read
    /// is admitted.
    pub(super) fn admits_with_idle(&self, p: f64, bytes: u64, idle: f64) -> bool {
        assert!(bytes > 0, "a read ahead reads an expert's bytes");
        #[cfg(test)]
        if let Some(even) = self.even {
            return p >= even;
        }
        self.line.is_none_or(|line| p >= self.miss.break_even(line.seconds(bytes), idle))
    }

    /// Mean resident compute seconds per decode pass (`MissCost::compute`).
    pub(super) fn compute(&self) -> f64 {
        self.miss.compute
    }

    /// Read seconds of an expert of `bytes` on the measured drive line;
    /// `0.0` before the line is measured.
    pub(super) fn read_seconds(&self, bytes: u64) -> f64 {
        self.line.map_or(0., |line| line.seconds(bytes))
    }
}

/// Where a route served a choice from, by its routing record's source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Use {
    /// A bank record no landing writes: a source `>= 0` where the device
    /// places beside a host tier, the record.
    Bank(u32),
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
                crate::planning::Placer::Tier => Self::Bank(source.unsigned_abs()),
                crate::planning::Placer::Bank => Self::Tier,
            },
            -1 => Self::File,
            _ => Self::Tier,
        }
    }
}

/// What a decode pass's misses beside a host tier held each resource, of
/// its streamed layers' refined lines (`Telemetry::with_observed`): the
/// link's seconds of its reads in place (each layer's misses the CPU site
/// did not serve, at the link's line) and its moves; the CPU site's
/// requests' path time; host memory's reads (every miss served, read by
/// the CPU or over the link, every file miss landed, every record moved)
/// at the memory's line; and what its misses served on the CPU or over the
/// link held the pass past its hits, each layer's by the cost function
/// (`ExpertLines::serve`). None of a layer without a host tier: on unified
/// memory the GPU reads its misses in place from the memory `C` reads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PassService {
    link: f64,
    cpu: f64,
    memory: f64,
    served: f64,
}

impl PassService {
    /// The pass's service, each layer's `routed` experts, of them `tiered`
    /// served from the host tier and `absent` from the file (landed into
    /// it), `sites` the CPU site's served misses and path time, beside
    /// `moved` (the moves' link seconds and records) of `lines`.
    fn of(lines: &[ExpertLines], routed: &[RoutedRead], tiered: &[Vec<usize>], absent: &[u64], sites: &[(u64, Duration)], moved: (f64, f64)) -> Self {
        let mut service = Self { link: moved.0, ..Self::default() };
        let (mut memory, layers) = (0., lines.len() as f64);
        for (layer, lines) in lines.iter().enumerate() {
            let Some(line) = lines.host.memory else {
                continue;
            };
            let tiered_count = tiered[layer].len() as u64;
            let misses = tiered_count + absent[layer];
            let (cpu, path) = sites[layer];
            let experts = routed[layer].experts.len();
            let rows = match experts {
                0 => 1.,
                experts => routed[layer].choices as f64 / experts as f64,
            };
            // Only host-tier misses (`tiered`) are served without parking on
            // the file; file misses (`absent`) are already accounted in
            // `Pass::parked`, so excluding them here avoids double-subtracting
            // them in `Terms::compute`.
            let hits = Hits { experts: experts.saturating_sub(tiered_count as usize) as f64, rows };
            let served = lines.serve(magnitude_cost_model::Pass::Decode, &[hits], &vec![Miss { rows, ready_seconds: 0. }; tiered_count as usize]);
            service.served += served.exposed.total();
            service.link += misses.saturating_sub(cpu) as f64 * lines.gpu.link.nanoseconds(rows) * 1e-9;
            service.cpu += path.as_secs_f64();
            service.memory += (misses + absent[layer]) as f64 * line.get() * 1e-9;
            memory += line.get() * 1e-9 / layers;
        }
        service.memory += moved.1 * memory;
        service
    }
}

/// What one sequence's exact row's routes show (`routed`, each streamed
/// layer's experts, ascending; `absent`, each layer's file misses among
/// them), lagged against its tokens before (`recent`, most recent first),
/// which it then joins: at each lag `j`, the experts it chose that none of
/// the `j` tokens before chose, at every layer they were seen; its picks,
/// and those the bank held.
fn lagged(recent: &mut std::collections::VecDeque<Vec<Vec<usize>>>, routed: &[RoutedRead], absent: &[u64]) -> Routed {
    let mut lags = Routed::default();
    for (layer, read) in routed.iter().enumerate() {
        lags.picks += read.choices as f64;
        lags.held += read.choices.saturating_sub(absent[layer]) as f64;
        let mut seen = Vec::<usize>::new();
        for (lag, before) in recent.iter().enumerate() {
            seen.extend(&before[layer]);
            lags.fresh[lag] += read.experts.iter().filter(|expert| !seen.contains(*expert)).count() as f64;
            lags.layers[lag] += 1.;
        }
    }
    recent.push_front(routed.iter().map(|read| read.experts.clone()).collect());
    recent.truncate(LAGS);
    lags
}

/// The costs as serving records them: the steps still running keep theirs
/// until they complete.
struct Costs {
    layers: Vec<LayerCost>,
    reads: RequestReads,
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
    convert: Vec<Option<magnitude_cost_model::RecordLine>>,
    hints: HintCost,
    /// Each running decode step's exact rows' hints' reads ahead, as they
    /// were issued ([`Telemetry::read_ahead`]): the layer, the expert, its
    /// lead and the layer whose hint named it. Its completion counts them
    /// into its layers' costs.
    ahead: BTreeMap<Step, Vec<(usize, usize, u8, usize)>>,
    front: FrontSkips,
    /// What the reads ahead's bytes were spent on ([`Telemetry::spent`]),
    /// and as the last step's end saw it.
    spent: AheadUse,
    spent_seen: AheadUse,
    /// The reads ahead an exact row used since the last decode pass was
    /// timed, and their bytes: the drive time of those its exact rows
    /// needed that were read ahead, which the next pass counts.
    used_ahead: (u64, u64),
    /// The promotions' copies completed since the last decode pass was
    /// timed: the seconds they held the link and the records they read
    /// ([`Telemetry::moved`]), which the next pass counts.
    moved: (f64, f64),
    /// The decode passes as measured ([`Passes`]).
    passes: Passes,
    /// The experts each streamed layer's exact row chose at its sequence's
    /// latest passes, most recent first, at most [`LAGS`] of them: what a
    /// pass's routes are lagged against ([`lagged`]). A pass of several
    /// sequences' rows, or a prompt, begins them again.
    recent: std::collections::VecDeque<Vec<Vec<usize>>>,
    /// Whether each streamed layer's hint launch runs ([`HintGate`]).
    hints_gate: HintGate,
    /// Each volume's counts (`Routing::volumes`) as the last step's end
    /// saw it, from none: the first step counts what each read since the
    /// process began.
    looked: Vec<seismic_io::Totals>,
    running: BTreeMap<Step, StepCost>,
    /// Wall-clock `[start, end]` intervals during which a running step was
    /// parked waiting on request dispatch (`Self::serve`) or demand reads
    /// landing (`Self::landed`). Taking their union clipped to the step window
    /// avoids double-counting overlapping concurrent layer reads.
    layer_wait: BTreeMap<Step, Vec<(Instant, Instant)>>,
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
    pub(super) windows: u64,
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
        if self.windows > 1 {
            return vec![steps; self.compute.len()];
        }
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
    /// `K`: a row's choices.
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
    /// (`counted`), its step's selection keys where its launch wrote them
    /// (`keys`), and its token clock position captured while the step was
    /// still live on `Clock`'s ring (`clock`).
    Decode { counted: Option<CountedLaunch>, keys: Option<std::sync::Arc<StepKeys>>, clock: Option<crate::experts::horizon::Tokens> },
    Group { steps: usize, merged: usize, choices: Vec<u32> },
}

/// What the stack's readers read of it, as its last feed left it
/// ([`Telemetry::feed_stack`]): none of them feeds it or waits on its lock,
/// so a reader never waits behind the milliseconds a feed takes.
struct Ranked {
    /// The load's counted decode ([`Stack::live`]).
    live: Option<magnitude_cost_model::LiveDecode>,
}

/// Everything telemetry records, under its one lock.
struct Recorded {
    costs: Costs,
    facts: GroupFacts,
    feed: Feed,
    ranked: Ranked,
    /// Each group's decode capacity, as residency last published it
    /// (`Directory::capacities`).
    records: Vec<usize>,
}

/// One completed decode step's measured misses per token queued for
/// `Shadows::executed` on `expert-residenc` without blocking `magnitude-main`.
/// Epoch and clock are read under `self.shadows` in [`Telemetry::shadow`], so
/// `Shadows::deltas` never receives a detached stale epoch or clock.
struct ExecutedStep {
    verifies: bool,
    depth: usize,
    real: f64,
}

/// Short-locked bridge between `magnitude-main` and `Shadows` on
/// `expert-residenc`: never held during `Shadows::step()`, so `complete_step`,
/// `verify_executed`, and `measured` never block behind shadow stepping.
struct ShadowState {
    pending_verify: Option<(Step, usize, f64)>,
    pending_executed: Vec<ExecutedStep>,
    gaps: [[Option<Gap>; DEPTHS]; 2],
}

impl Default for ShadowState {
    fn default() -> Self {
        Self {
            pending_verify: None,
            pending_executed: Vec::new(),
            gaps: [[None; DEPTHS]; 2],
        }
    }
}

/// What serving and steps record of the load, and the routing they read.
pub(super) struct Telemetry {
    routing: Routing,
    /// Every streamed layer's key records ([`KeyRecords`](super::routes::KeyRecords)),
    /// and the trace of a load recording its routes.
    keys: super::routes::KeyRecords,
    routes: Option<super::routes::RouteRecorder>,
    /// The load's token clock and turnovers (`horizon::Time`, shared via
    /// `Reuse::rates`), and the one owner of the guesses' stand rates
    /// (`interest::Stands`), which the regime reads.
    horizon: Time,
    stands: Stands,
    /// The one placement ranking, shared with `Residency` and `Stack`: used
    /// outside `recorded`'s lock to record fallen front rows (`Reuse::fell`),
    /// feed the shadows (`Self::shadow`), and read the A1 diagnostic
    /// (`Reuse::far_deciles`).
    ranking: Ranking,
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
    /// The ranking's shadows at every depth of foresight, lazily allocated on
    /// `expert-residenc` once a caller asks for regime pricing (`measured`),
    /// under a lock only `Telemetry::shadow` ever acquires.
    shadows: Mutex<Option<Shadows>>,
    /// Whether any caller has asked for regime pricing (`measured`): while
    /// false (e.g. plain decode with no fronting), shadows neither allocate
    /// nor step.
    shadows_needed: AtomicBool,
    /// Short-locked queue of executed-step observations and latest published
    /// shadow gaps, decoupling `magnitude-main` from `self.shadows`.
    shadow_state: Mutex<ShadowState>,
    /// Per layer, whether it is whole: every expert held at its home, so
    /// it has no reads of a decode step (its routing record, which its
    /// route writes beside its keys, is a trace's alone)
    /// ([`Residency::publish_policy`](super::residency::Residency::publish_policy),
    /// while no step runs).
    whole: Vec<AtomicBool>,
    /// The first step whose group has not ended: the reads of every group
    /// before it are ready to feed ([`Self::ended`], [`Self::feed`]). A
    /// step's end only raises it, so the thread ending steps never waits
    /// on the stack.
    feedable: AtomicU64,
    /// The break-even reads ahead are admitted at in place of the measured
    /// one ([`Self::admit_at`]).
    #[cfg(test)]
    even: Mutex<Option<f64>>,
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
    /// costs are seeded by the job's `lines`, which settle only what the
    /// observations leave undetermined: a decode that routes as many
    /// experts each step pays the fixed term in step with their sum, which
    /// its observations alone never tell apart.
    fn block(&self, layer: usize, lines: &ExpertLines) -> Option<[f64; 4]> {
        let mut line = self.blocks[layer];
        let cpu = lines.host.cpu.map_or(0., |cpu| cpu.expert.one.get() * 1e-9);
        for (term, seconds) in [(1, lines.gpu.resident.nanoseconds(1.) * 1e-9), (2, lines.gpu.link.nanoseconds(1.) * 1e-9), (3, cpu)] {
            line.seed(std::array::from_fn(|at| f64::from(u8::from(at == term))), seconds);
        }
        match line.state().expect("blocks' counts and device seconds make a line") {
            seismic::measure::LineState::Pending => None,
            seismic::measure::LineState::Observed(terms) => Some(terms.costs()),
            seismic::measure::LineState::Seeded(_) => unreachable!("the seeds leave a block's fixed term unplaced"),
        }
    }
}

/// Why a completed step's costs could not be recorded: a sampled block's
/// stamps at `layer` did not read (the device's error, the step having
/// passed them). It ends serving (`Shared::failed_completion`).
#[derive(Debug)]
pub(super) struct CompletionError {
    pub layer: usize,
    pub error: String,
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
        let (horizon, stands) = crate::experts::reuse::Reuse::of(&ranking).rates();
        let shadows = Mutex::new(None);
        let shadows_needed = AtomicBool::new(false);
        let shadow_state = Mutex::new(ShadowState::default());
        let stack = Stack::new(ranking.clone(), layers.len(), extents.widest);
        Self {
            keys: super::routes::KeyRecords::of(layers, extents.words, extents.selected),
            routes: extents.recording.then(|| super::routes::RouteRecorder::of(layers, extents.words, extents.selected)),
            stands,
            ranking,
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
                    landings: vec![seismic::measure::Line::new(); groups],
                    sites: vec![SiteRequests::default(); layers.len()],
                    blocks: vec![seismic::measure::Line::new(); layers.len()],
                    convert: vec![None; groups],
                    hints: HintCost::default(),
                    ahead: BTreeMap::new(),
                    front: FrontSkips::default(),
                    spent: AheadUse::default(),
                    spent_seen: AheadUse::default(),
                    used_ahead: (0, 0),
                    moved: (0., 0.),
                    passes: Passes::new(horizon.clone()),
                    recent: std::collections::VecDeque::new(),
                    hints_gate: HintGate::new(layers.iter().map(|layer| layer.hints.clone()).collect(), horizon.clone()),
                    looked,
                    running: BTreeMap::new(),
                    layer_wait: BTreeMap::new(),
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
                    windows: extents.windows,
                },
                feed: Feed { pending: Vec::new(), ended: None },
                ranked: Ranked { live: stack.live() },
                records,
            }),
            slots,
            stack: Mutex::new(stack),
            shadows,
            shadows_needed,
            shadow_state,
            whole: layers.iter().map(|_| AtomicBool::new(false)).collect(),
            feedable: AtomicU64::new(0),
            #[cfg(test)]
            even: Mutex::new(None),
            spans,
            sampled: Mutex::new(BTreeMap::new()),
            horizon,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Recorded> {
        self.recorded.lock().expect("watcher telemetry")
    }

    /// The drive's line now: the one line of the volumes the model's files
    /// lie on (`seismic_io::Totals::line`), together; none before a demand
    /// read ended.
    ///
    /// # Panics
    /// Where the volumes' line makes no read line: its latency is never
    /// below none and its rate is a busy volume's, so that is an invariant
    /// violation.
    pub(super) fn line(&self) -> Option<ReadLine> {
        let line = self.routing.volumes.iter().map(|&volume| seismic_io::accounting::totals(volume)).sum::<seismic_io::Totals>().line()?;
        Some(ReadLine::new(line.latency, line.byte_seconds).unwrap_or_else(|error| panic!("invariant violated: the drive's line {line:?} makes no read line: {error:?}")))
    }

    /// `count` landings of `layer`'s request ended `seconds` after their
    /// read (`FileLine`): a point of its group's landing line.
    pub(super) fn landings_ended(&self, layer: usize, count: usize, seconds: f64) {
        let group = self.routing.maps[layer].0;
        self.lock().costs.landings[group].observe([1., count as f64], seconds);
    }

    /// Each bank group's device convert takes its `lines` from now on.
    pub(super) fn device_convert(&self, lines: &[magnitude_cost_model::RecordLine]) {
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
                spans.seconds(from, to).map(|seconds| (layer, seconds)).map_err(|error| CompletionError { layer, error: error.to_string() })
            })
            .collect::<Result<_, _>>()?;
        Ok(SampledBlocks { lines, seconds, rows: self.read_rows(records) })
    }

    /// What each streamed layer's routing record of `records` read
    /// ([`ReadRows`]); none of a whole layer, whose record is not read
    /// (`Self::records`): its block ran every expert from its home.
    fn read_rows(&self, records: &[RoutingRecord]) -> Vec<Option<ReadRows>> {
        let placer = self.routing.placer;
        records
            .iter()
            .map(|record| {
                record.written().then(|| {
                    let [mut hits, mut tiered, mut absent] = [(); 3].map(|_| BTreeMap::new());
                    for (expert, source) in record.choices() {
                        let rows = match Use::of(source, placer) {
                            Use::Bank(_) => &mut hits,
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
        self.with_observed(&mut self.lock().costs, lines)
    }

    /// `lines` with each layer's landing line, CPU site line, GPU lines and
    /// moves now, which the load observes beside the lines it was priced
    /// on: the landing line of the one site its bank lands at
    /// ([`Self::landing`]), and an even share of the decode passes' moves
    /// ([`Passes::moves`]; the copy engine moves whatever layer a step is
    /// at).
    fn with_observed(&self, costs: &mut Costs, lines: &[ExpertLines]) -> Arc<[ExpertLines]> {
        let share = 1. / lines.len().max(1) as f64;
        let moves = costs.passes.moves().map(|(link, records)| (link * share, records * share));
        lines
            .iter()
            .zip(&self.routing.maps)
            .enumerate()
            .map(|(layer, (lines, &(group, _)))| {
                let site = costs.site(layer);
                lines.landed_by(self.landing(costs, group)).sited(site).passed(costs.block(layer, lines)).moving(moves)
            })
            .collect()
    }

    /// Group `group`'s landing line at the one site its bank lands at: the
    /// host's landings observed beside a host tier, the device's convert as
    /// the job measured it where the bank places.
    fn landing(&self, costs: &Costs, group: usize) -> Option<magnitude_cost_model::LandingLine> {
        use magnitude_cost_model::LandingLine;
        match self.routing.placer {
            crate::planning::Placer::Tier => costs.landing(group).map(LandingLine::Host),
            crate::planning::Placer::Bank => costs.convert.get(group).copied().flatten().map(LandingLine::Device),
        }
    }

    /// The slots the bank commits.
    fn slots(&self) -> usize {
        usize::try_from(self.slots.load(Ordering::Acquire)).expect("bank slots fit the host")
    }

    /// Each group's decode capacity is `records` from now
    /// (`Directory::capacities`), as a resize or a lease changed it.
    pub(super) fn capacity(&self, records: Vec<usize>) {
        self.lock().records = records;
    }

    /// The steps a prompt group runs while decode waits, at most `capacity`:
    /// the fewest whose compute covers each layer's read, `g · compute_l ≥
    /// λ + β_l / D`, with `compute_l` measured on this run, `β_l` the stored
    /// bytes the layer's next group reads ([`GroupFacts::terms`]), and `λ`,
    /// `D` the run's own read line. A group runs to capacity until the run
    /// has measured every layer's compute.
    pub(super) fn group_size(&self, lines: Option<&[ExpertLines]>, notice: Notice, capacity: usize) -> usize {
        let mut recorded = self.lock();
        let lines = lines.map(|lines| self.with_observed(&mut recorded.costs, lines));
        recorded.facts.group_size(lines.as_deref(), self.line(), notice, capacity)
    }

    /// Each streamed layer's merge set in a scratch-served group of `steps`
    /// steps (`Stages::merge_sets`), from this run's measured compute, what
    /// each layer's next group moves ([`GroupFacts::terms`]), each layer's
    /// last group union, each layer's `lines`, the run's own read line and
    /// the passes' price `pass`.
    pub(super) fn merge_sets(&self, lines: Option<&[ExpertLines]>, pass: Option<&ExpertPass>, notice: Notice, steps: usize) -> Vec<usize> {
        let mut recorded = self.lock();
        let lines = lines.map(|lines| self.with_observed(&mut recorded.costs, lines));
        recorded.facts.merge_sets(lines.as_deref(), self.line(), pass, notice, steps)
    }

    /// The costs so far, beside what the CPU site's requests took, `site`.
    pub(super) fn costs(&self, site: crate::experts::site::SiteServed, promoted: u64) -> ExpertCosts {
        let recorded = self.lock();
        let costs = &recorded.costs;
        ExpertCosts {
            layers: costs.layers.clone(),
            reads: costs.reads,
            line: self.line(),
            landings: self.routing.maps.iter().map(|&(group, _)| self.landing(costs, group)).collect(),
            hints: costs.hints,
            front: costs.front,
            ahead: costs.spent,
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

    /// Whether `layer` is whole, as its policy word was last written.
    pub(super) fn is_whole(&self, layer: usize) -> bool {
        self.whole[layer].load(Ordering::Acquire)
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
                Fed::Decode { counted, keys, clock } => {
                    stack.read_at_batched(&read, counted, keys.as_deref(), clock);
                    decoded = true;
                }
                Fed::Group { steps, merged, choices } => {
                    let routed = read.experts.iter().copied().zip(choices.iter().copied()).collect::<Vec<_>>();
                    let rows = crate::experts::reuse::Rows::union(self.routing.experts[read.layer], self.routing.selected, &routed);
                    let depths = stack.read_group_batched(&read, rows);
                    unions.push((read.layer, GroupUnion::new(steps, merged, depths.into_iter().zip(choices).collect())));
                }
            }
        }
        stack.finish_batch();
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

    #[cfg(test)]
    pub(super) fn ahead_len(&self) -> usize {
        self.lock().costs.ahead.len()
    }

    /// Step `sequence` ended, completed or not: what it left uncompleted,
    /// its cost and its sampled blocks, is dropped.
    pub(super) fn end_step(&self, sequence: Step) {
        {
            let costs = &mut self.lock().costs;
            costs.running.remove(&sequence);
            costs.layer_wait.remove(&sequence);
            costs.ahead.remove(&sequence);
        }
        if self.samples(sequence) {
            self.sampled.lock().expect("the sampled blocks").retain(|&(step, _), _| step != sequence);
        }
    }

    /// Step `sequence` completed (`completion`): its cost is the last
    /// completed step's, and a decode step's routing is fed to the stack,
    /// counted while its load records it, and returned: at each streamed
    /// layer, the experts its exact rows chose, by which the interest
    /// ledger resolves the step's interests (`interest::Told::Completed`).
    /// A decode pass is timed into the measured passes ([`Passes`]). Fails
    /// where a sampled block's stamps do not read.
    pub(super) fn complete_step(
        &self,
        steps: &Steps,
        sequence: Step,
        completion: CompletedStep,
        sited: &dyn Fn(usize) -> (u64, u64, Duration),
        priced: Option<&[ExpertLines]>,
    ) -> Result<Option<DecodeRouting>, CompletionError> {
        let CompletedStep { work, begun, route } = completion;
        // A decode step's routing at each streamed layer, read from its
        // records before anything is recorded.
        let decoded = match route {
            CompletedRoute::Grouped => None,
            CompletedRoute::Decode(mut decoded) => {
                // The step completes, so it holds its records until it ends.
                // Its exact rows' choices: a front row's are the next
                // step's, which its exact row routes again.
                let front = decoded.launch.map_or(0, |launch| launch.front);
                // Its rows' keys, which placement ranks by and a recording
                // load traces.
                let keys = decoded.launch.map(|launch| std::sync::Arc::new(self.keys.read(steps, sequence, launch)));
                if let Some(routes) = &self.routes {
                    let launch = decoded.launch.expect("a recording load's decode step has its launch");
                    routes.record(steps, sequence, launch, keys.as_deref().expect("a launch's keys"));
                }
                let selected = self.routing.selected;
                let records = self.records(steps, sequence);
                // Its front rows' guess kinds, as every record it wrote
                // carries them.
                let guesses = records.iter().filter_map(RoutingRecord::guesses).reduce(|first, other| {
                    assert_eq!(first, other, "step {sequence}'s routing records disagree on its guess kinds");
                    first
                });
                let records = records.into_iter().map(|record| record.exact(front, selected)).collect::<Vec<_>>();
                if seismic_io::timeline::recording() {
                    for (layer, record) in records.iter().enumerate() {
                        for (expert, source) in record.choices() {
                            seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("route"), sequence, (layer as i64, expert as i64, source as i64));
                        }
                    }
                }
                let blocks = decoded.sampled.take().map(|lines| self.sampled_blocks(sequence, lines, &records)).transpose()?;
                let sourced = self.sourced(&records);
                // Each layer's experts its exact rows chose, ascending, each
                // where a read ahead saves it ([`saves`]) and where the tier
                // served it: what resolves the step's interests.
                let mut chose = vec![Vec::<Chosen>::new(); records.len()];
                for &(layer, expert, used) in &sourced {
                    chose[layer].push(Chosen { expert, saves: saves(used), tier: used == Use::Tier });
                }
                for chose in &mut chose {
                    chose.sort_unstable_by_key(|chosen| chosen.expert);
                    // A record's choices name an expert once per row: the
                    // first standing choice's source serves every one.
                    chose.dedup_by_key(|chosen| chosen.expert);
                }
                Some(((decoded, keys), self.routed(sequence, &records), Self::tiered(&records), chose, Self::absent(&records), blocks, guesses))
            }
        };
        // The drive's line the pass's reads are timed along, read before
        // the costs are taken.
        let line = self.line();
        let mut recorded = self.lock();
        let Recorded { costs, facts, feed, .. } = &mut *recorded;
        // The last request served whole is its group's last step's at the
        // last streamed layer: its compute runs to the step's completion.
        if let Some(group) = &mut costs.group {
            if let Some(Acknowledged { point, at, .. }) = group
                .acknowledged
                .take_if(|served| served.point.sequence == sequence && served.last_window)
            {
                compute(&mut group.cost, facts, point, at.elapsed(), true);
            }
        }
        let layers = costs.layers.len();
        let mut cost = costs
            .running
            .remove(&sequence)
            .unwrap_or_else(|| StepCost::of(layers));
        let layer_wait = costs.layer_wait.remove(&sequence);
        let ahead = costs.ahead.remove(&sequence).unwrap_or_default();
        // Each streamed layer's misses the CPU site served and its request's
        // path time, which the pass's terms count.
        let mut sites = vec![(0u64, Duration::ZERO); layers];
        if let Some((_, routed, tiered, chose, absent, blocks, _)) = &decoded {
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
                sites[layer] = (cpu, path);
                cost.experts[layer].missed.extend(served.iter().copied());
            }
            // The reads its exact rows' hints made for each layer, which
            // the step's record counts at the layer read and at the lead
            // that named it: those of an expert its exact rows chose saved
            // a read.
            for (layer, expert, lead, hinter) in ahead {
                let used = chose[layer].binary_search_by_key(&expert, |chosen| chosen.expert).is_ok_and(|at| chose[layer][at].saves);
                let reads = &mut cost.layers[layer];
                reads.hint_reads += 1;
                reads.used += u64::from(used);
                reads.lead += u64::from(lead);
                let named = &mut cost.hints[hinter][usize::from(lead) - 1];
                named.read += 1;
                named.used += u64::from(used);
            }
            // The misses a hint could save, for the hints' recall (their
            // candidates are counted as the ledger resolves them,
            // [`Self::resolved`]).
            let hints = &mut costs.hints;
            // The file misses a hint could have saved, at layers a ring
            // hints, and those no ring reaches.
            for (layer, &absent) in absent.iter().enumerate() {
                match self.routing.hinted[layer] {
                    true => hints.hintable += absent,
                    false => hints.unhintable += absent,
                }
            }
            hints.layer_steps += layers as u64;
            // Whether its hints ran.
            for (layer, (total, leads)) in costs.layers.iter_mut().zip(&mut cost.hints).enumerate() {
                let on = costs.hints_gate.published(layer);
                total.hints[usize::from(on)] += 1;
                leads.iter_mut().for_each(|lead| lead.on = on);
            }
        }
        // Every step's end, from which the next decode step's time is taken.
        let now = Instant::now();
        let ended = (feed.ended.replace(now), now);
        if let Some(begun_at) = seismic_io::timeline::stamp(begun) {
            let previous = ended.0.and_then(seismic_io::timeline::stamp).map_or(-1, |previous| previous as i64);
            seismic_io::timeline::note_at(now, seismic_io::timeline::Kind::Noted("step"), sequence, (begun_at as i64, previous, (work == LaunchWork::Generation) as i64));
        }
        // What the reads ahead's bytes were spent on since the step before.
        cost.ahead = costs.spent.since(&costs.spent_seen);
        costs.spent_seen = costs.spent;
        // A generation step's time, whatever its form (a verify wider than
        // the decode form ranks no hints), what it parked, its file misses
        // and the drive time its exact rows needed, read on demand or ahead
        // along the drive's line, are a measured pass ([`Passes`]), its
        // front rows' among them. A prompt step's reads drain a burst no
        // pass measured: it leaves no pass, and the reads ahead used
        // meanwhile no pass's drive.
        match work {
            LaunchWork::Generation => {
                let step_start = ended.0.map_or(begun, |previous| previous.max(begun));
                cost.seconds = step_seconds(ended.0, begun, ended.1);
                if let Some(intervals) = layer_wait {
                    cost.parked = parked_union(intervals, step_start, ended.1);
                }
                cost.storage = self.delivered(costs);
                let (used, used_bytes) = std::mem::take(&mut costs.used_ahead);
                if let Some(launch) = decoded.as_ref().and_then(|decoded| decoded.0.0.launch) {
                    let decoded = decoded.as_ref().expect("a launch's step is decoded");
                    let absent = &decoded.4;
                    // One sequence's exact row is its next token: its routes
                    // lag against its tokens before.
                    let routed = match launch.rows - launch.front.count_ones() as usize {
                        1 => lagged(&mut costs.recent, &decoded.1, absent),
                        _ => {
                            costs.recent.clear();
                            Routed::default()
                        }
                    };
                    // Before the drive's line is measured no read is timed:
                    // the pass's drive counts none. Use transfer time (bytes / D)
                    // rather than `line.latency_seconds()` (saturated queue wait
                    // from `seismic_io::Totals::line`), which otherwise inflates
                    // `miss_ms` by 3x–10x and swamps compute and turn overhead
                    // in `Regime`.
                    let drive = line.map_or(0., |line| {
                        let missed = absent
                            .iter()
                            .zip(&facts.bytes)
                            .map(|(&misses, &bytes)| line.transfer_seconds((misses * bytes) as f64))
                            .sum::<f64>();
                        missed + line.transfer_seconds(used_bytes as f64)
                    });
                    let moved = std::mem::take(&mut costs.moved);
                    let tiered = &decoded.2;
                    let served = priced.map(|lines| self.with_observed(costs, lines)).map(|lines| PassService::of(&lines, &decoded.1, tiered, absent, &sites, moved));
                    let PassService { link, cpu, memory, served } = served.unwrap_or_default();
                    // Its exact rows' misses of the memory the GPU computes
                    // from: its file and host tier misses, and where reads
                    // ahead land there (unified memory, the bank the one
                    // store) the reads ahead they used.
                    let landed = match self.routing.placer {
                        crate::planning::Placer::Bank => used,
                        crate::planning::Placer::Tier => 0,
                    };
                    let tier = tiered.iter().map(Vec::len).sum::<usize>() as u64;
                    let missed_total = (absent.iter().sum::<u64>() + tier + landed) as f64;
                    let depth = launch.depth();
                    let pass_rows = u32::try_from(launch.rows).expect("a launch's rows fit u32");
                    costs.passes.timed(Pass {
                        rows: pass_rows,
                        depth,
                        verifies: launch.verifies(),
                        seconds: cost.seconds,
                        parked: cost.parked.as_secs_f64(),
                        misses: absent.iter().sum::<u64>() as f64,
                        missed: missed_total,
                        reads: (absent.iter().sum::<u64>() + used) as f64,
                        unused: cost.ahead.unused(),
                        drive,
                        link,
                        moved: moved.0,
                        moved_records: moved.1,
                        cpu,
                        memory,
                        served,
                        routed,
                    });
                    if depth < DEPTHS && pass_rows > 0 && self.shadows_needed.load(Ordering::Acquire) {
                        if launch.verifies() {
                            self.shadow_state.lock().expect("watcher telemetry shadow state").pending_verify = Some((sequence, depth, missed_total));
                        } else {
                            // Plain and front passes emit 1.0 token per step on
                            // the clock (`Advance::Known(1.0)`), matching
                            // `Shadows::step`'s denominator.
                            self.shadow_state.lock().expect("watcher telemetry shadow state").pending_executed.push(ExecutedStep {
                                verifies: false,
                                depth,
                                real: missed_total,
                            });
                        }
                    }
                    if launch.front != 0 {
                        costs.front.passes += 1;
                    }
                }
            }
            LaunchWork::Prompt => {
                cost.storage = self.delivered(costs);
                costs.used_ahead = (0, 0);
                costs.moved = (0., 0.);
                costs.recent.clear();
            }
        }
        let chose = decoded.map(|((decoded, keys), routed, _, chose, _, _, guesses)| {
            let clock = decoded.launch.map(|_| self.horizon.lock().clock.at(sequence));
            feed_decode(feed, routed, decoded, keys, clock);
            DecodeRouting { chose, guesses }
        });
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
        costs.layer_wait.retain(|running, _| *running > sequence);
        costs.ahead.retain(|running, _| *running > sequence);
        costs.completed = Some((sequence, cost));
        Ok(chose)
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
        let clock = (!prompt).then(|| self.horizon.lock().clock.at(first + steps - 1));
        let mut recorded = self.lock();
        for ((layer, choices), merged) in choices.into_iter().enumerate().zip(merges) {
            let (experts, choices): (Vec<_>, Vec<_>) =
                choices.into_iter().enumerate().filter(|&(_, count)| count > 0).unzip();
            let routed = choices.iter().map(|&count| u64::from(count)).sum();
            let read = RoutedRead { sequence: first + steps - 1, layer, experts, choices: routed };
            let fed = match prompt {
                true => Fed::Group { steps: steps as usize, merged, choices },
                false => Fed::Decode { counted: None, keys: None, clock },
            };
            recorded.feed.pending.push((first, read, fed));
        }
    }

    /// What a load recording its routes recorded since last taken; none of
    /// a load that records none.
    pub(super) fn take_routes(&self) -> Option<super::routes::RecordedRoutes> {
        self.routes.as_ref().map(super::routes::RouteRecorder::take)
    }

    /// What the load's counted decode read and ran so far, as the stack's
    /// last feed left it ([`Ranked`]); none before it counted a step.
    pub(super) fn live(&self) -> Option<magnitude_cost_model::LiveDecode> {
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
        let end = Instant::now();
        if let Some(start) = end.checked_sub(served.parked) {
            costs.layer_wait.entry(sequence).or_default().push((start, end));
        }
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
    /// waited for a worker), counted into the demand reads
    /// (`ExpertCosts::reads`).
    pub(super) fn landed(&self, layer: usize, sequence: Step, read: Duration, timed: Option<(u64, Duration)>, queued: Duration) {
        let mut recorded = self.lock();
        let costs = &mut recorded.costs;
        let cost = &mut costs.layers[layer];
        cost.read += read;
        if let Some(step) = costs.running.get_mut(&sequence) {
            step.read += read;
        }
        let end = Instant::now();
        if let Some(start) = end.checked_sub(read) {
            costs.layer_wait.entry(sequence).or_default().push((start, end));
        }
        if let Some((bytes, job)) = timed {
            costs.reads.add(bytes, job.saturating_sub(queued).as_secs_f64());
        }
    }

    /// What each of the source's volumes read since the step before, from
    /// their counts (`seismic_io::accounting`): each volume's span, the
    /// step's attribution.
    fn delivered(&self, costs: &mut Costs) -> Vec<(seismic_io::Volume, seismic_io::Totals)> {
        self.routing
            .volumes
            .iter()
            .zip(&mut costs.looked)
            .map(|(&volume, looked)| {
                let now = seismic_io::accounting::totals(volume);
                let span = now.since(looked);
                *looked = now;
                (volume, span)
            })
            .collect()
    }

    /// What a read ahead is priced by now ([`ReadPrice`]): what a miss
    /// costs over the measured passes' window and the drive's line, read
    /// once for a pass over the interests.
    pub(super) fn price(&self) -> ReadPrice {
        let line = self.line();
        let mut recorded = self.lock();
        let costs = &mut recorded.costs;
        ReadPrice {
            miss: costs.passes.miss_cost(),
            line,
            #[cfg(test)]
            even: *self.even.lock().expect("the admitted break-even"),
        }
    }

    /// Admit reads ahead at break-even `even` in place of the measured one:
    /// a harness's steps compute nothing and park on every miss, so the one
    /// measured prices the reads of a read-bound load; their mechanics are
    /// tested at a break-even of their own.
    #[cfg(test)]
    pub(super) fn admit_at(&self, even: f64) {
        *self.even.lock().expect("the admitted break-even") = Some(even);
    }

    /// Count a read ahead of `bytes` of `expert` of `layer`, issued for
    /// the interest of `source` that led its expert, wanting it at `step`,
    /// named at layer `from`'s route: an exact row's hint's (a hint's,
    /// `Source::Lead` of no row) counts among the hints' reads and with its
    /// step's layer and its naming layer's lead; an ahead row's, its
    /// route's or its hints', among the front rows' ([`FrontSkips`]); and
    /// its bytes as issued for its source's class ([`AheadUse`]).
    pub(super) fn read_ahead(&self, source: Source, step: Step, (layer, expert): (usize, usize), from: usize, bytes: u64) {
        let mut recorded = self.lock();
        let costs = &mut recorded.costs;
        costs.spent.issue(source, bytes);
        match source {
            Source::Lead { distance, row: None } => {
                costs.hints.reads += 1;
                if costs.completed.as_ref().is_none_or(|(completed, _)| step > *completed) {
                    costs.ahead.entry(step).or_default().push((layer, expert, distance, from));
                }
            }
            Source::Ahead(_) | Source::Lead { row: Some(_), .. } => {
                costs.front.reads += 1;
            }
            Source::Exact => unreachable!("no read ahead is an exact request's"),
        }
    }

    /// Exact rows' hints published `named` candidates to the interest
    /// ledger, each its step, the layer whose hint named it, its lead, and
    /// whether its probability reached the read-ahead break-even
    /// ([`LeadHints`]).
    pub(super) fn hints_named(&self, named: &[(Step, usize, u8, bool)]) {
        let costs = &mut self.lock().costs;
        let layers = costs.layers.len();
        let completed = costs.completed.as_ref().map(|(step, _)| *step);
        for &(step, layer, distance, admitted) in named {
            if completed.is_some_and(|done| step <= done) {
                continue;
            }
            let lead = &mut costs.running.entry(step).or_insert_with(|| StepCost::of(layers)).hints[layer][usize::from(distance) - 1];
            lead.published += 1;
            lead.admitted += u64::from(admitted);
        }
    }

    /// What the interest ledger learned the reads ahead's bytes were spent
    /// on (`interest::Spent`): counted by their class, and each an exact
    /// row used into the next pass's drive time ([`Pass::drive`]).
    pub(super) fn spent(&self, spent: &[Spent]) {
        let mut recorded = self.lock();
        let costs = &mut recorded.costs;
        for &spent in spent {
            costs.spent.spend(spent);
            if let Spent::Used { bytes, .. } = spent {
                costs.used_ahead.0 += 1;
                costs.used_ahead.1 += bytes;
            }
        }
    }

    /// The verdicts judged front rows' guesses (`interest::Judged`): into
    /// the front rows' record by kind and by depth; and verify passes'
    /// acceptable rows' (`interest::Accepted`), into the record by depth.
    /// The stand rates decisions read are their one owner's
    /// (`interest::Stands`), which the ledger counts as it judges.
    pub(super) fn judged(&self, judged: &[Judged], accepted: &[Accepted]) {
        if let Some(routes) = &self.routes {
            routes.judged(judged);
        }
        let mut fallen = BTreeMap::<Step, usize>::new();
        {
            let front = &mut self.lock().costs.front;
            for judged in judged {
                if let Some(stood) = judged.stood {
                    let depth = usize::from(judged.depth);
                    front.judged[judged.kind.index()] += 1;
                    front.stood[judged.kind.index()] += u64::from(stood);
                    front.judged_at[depth - 1] += 1;
                    front.stood_at[depth - 1] += u64::from(stood);
                    if !stood && (1..DEPTHS).contains(&depth) {
                        fallen
                            .entry(judged.step)
                            .and_modify(|first| *first = (*first).min(depth))
                            .or_insert(depth);
                    }
                }
            }
            for accepted in accepted.iter().filter(|accepted| usize::from(accepted.depth) <= FRONT_DEPTHS) {
                let depth = usize::from(accepted.depth) - 1;
                front.acceptable_at[depth] += 1;
                front.accepted_at[depth] += u64::from(accepted.stood);
            }
        }
        if !fallen.is_empty() {
            let mut reuse = crate::experts::reuse::Reuse::of(&self.ranking);
            for (step, first_fall) in fallen {
                for layer in 0..self.routing.experts.len() {
                    reuse.fell(layer, step, first_fall);
                }
            }
        }
    }

    /// Whether each streamed layer's hint launch runs now
    /// (`gate::HintGate::decide`), priced by `prices` (none before the
    /// load's prices are had), a miss's cost and a break's as the measured
    /// passes say ([`Passes`]). `resolved` says which layers will be
    /// resolved as the gate is written.
    pub(super) fn hint_gates(&self, resolved: &[bool], prices: Option<&magnitude_cost_model::LaunchPrices>) -> Vec<bool> {
        let costs = &mut self.lock().costs;
        let (miss, latest) = (costs.passes.miss_cost(), costs.passes.latest());
        costs.hints_gate.decide(resolved, prices, miss, latest)
    }

    /// Residency's write of `hints` took effect: in effect from now.
    pub(super) fn publish_hints(&self, hints: &[bool]) {
        self.lock().costs.hints_gate.publish(hints);
    }

    /// The promotions' copies completed: `records` records whose copies
    /// held the link `link` seconds, which the next decode pass counts.
    pub(super) fn moved(&self, link: f64, records: usize) {
        let moved = &mut self.lock().costs.moved;
        moved.0 += link;
        moved.1 += records as f64;
    }

    /// The shares of the decode's time a promotion's copy runs beside, as
    /// the measured passes say ([`Passes::beside`]).
    pub(super) fn beside(&self) -> crate::assessment::Beside {
        self.lock().costs.passes.beside()
    }

    /// The passes' mean terms and the drive's seconds a read, as the
    /// measured passes say ([`Passes::terms`]).
    pub(super) fn terms(&self) -> Option<(magnitude_cost_model::Terms, f64)> {
        self.lock().costs.passes.terms()
    }

    /// What the regime is decided on, as the measured passes say
    /// ([`Passes::measured`]), beside the ranking's misses at each depth of
    /// foresight as its shadows last read them ([`Self::shadow`]), and the
    /// guesses' stand rates as their one owner holds them; none before a
    /// decode pass is timed.
    pub(super) fn measured(&self) -> Option<Measured> {
        if !self.shadows_needed.swap(true, Ordering::AcqRel) {
            crate::experts::reuse::Reuse::of(&self.ranking).enable_shadows();
        }
        let gaps = self.shadow_state.lock().expect("watcher telemetry shadow state").gaps;
        let needed = gaps.map(|form| {
            form.map(|gap| {
                gap.map(|gap| match gap.delta {
                    crate::experts::reuse::Delta::Unmeasured => magnitude_cost_model::Needed::Shadow(gap.needed()),
                    crate::experts::reuse::Delta::Measured { .. } => magnitude_cost_model::Needed::Executed(gap.needed()),
                })
            })
        });
        let measured = self.lock().costs.passes.measured(needed, &self.stands);
        if measured.is_some() && gaps[0].iter().any(Option::is_some) {
            tracing::debug!(
                "expert shadow gaps: {}",
                gaps[0]
                    .iter()
                    .enumerate()
                    .filter_map(|(depth, gap)| gap.map(|gap| format!("d{depth}={gap}")))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }
        measured
    }

    /// A verify step's verdicts are all in (`emitted`: mean tokens emitted per
    /// request, `1 + accepted / requests`): queue its `Δ_{verify, depth}` with
    /// the same per-token denominator as `Shadows::step`, without blocking on
    /// `self.shadows`.
    pub(super) fn verify_executed(&self, step: Step, emitted: f64) {
        if !self.shadows_needed.load(Ordering::Acquire) {
            return;
        }
        let pending = self
            .shadow_state
            .lock()
            .expect("watcher telemetry shadow state")
            .pending_verify
            .take_if(|(pending_step, ..)| *pending_step == step);
        if let Some((_, depth, missed_total)) = pending {
            self.shadow_state.lock().expect("watcher telemetry shadow state").pending_executed.push(ExecutedStep {
                verifies: true,
                depth,
                real: missed_total / emitted,
            });
        }
    }

    /// Formatted §8 A1 `history` vs `far` per-decile summary across streamed
    /// layers; none where no decile has observations.
    pub(super) fn far_deciles_line(&self) -> Option<String> {
        let deciles = crate::experts::reuse::Reuse::of(&self.ranking).far_deciles();
        if deciles.iter().all(|&(h, f)| h.is_none() && f.is_none()) {
            return None;
        }
        let fmt = |stat: Option<(f64, f64)>| match stat {
            Some((rate, se)) => format!("{rate:.4}±{se:.4}"),
            None => "-".to_owned(),
        };
        let parts: Vec<String> = deciles
            .iter()
            .enumerate()
            .map(|(d, &(h, f))| format!("d{d}=({} vs {})", fmt(h), fmt(f)))
            .collect();
        Some(format!("A1 history vs far: {}", parts.join(" ")))
    }

    /// The ranking's shadows read what it logged for them
    /// (`Shadows::read_log`) and drain any executed-step observations queued
    /// by `magnitude-main`, publishing the updated shadow gaps snapshot into
    /// `shadow_state` so `magnitude-main` never blocks on `self.shadows`.
    /// No-op until a caller asks for regime pricing (`measured`).
    pub(super) fn shadow(&self) {
        if !self.shadows_needed.load(Ordering::Acquire) {
            return;
        }
        let reentries = {
            let mut guard = self.shadows.lock().expect("watcher telemetry shadows");
            let fed = {
                let mut reuse = crate::experts::reuse::Reuse::of(&self.ranking);
                if guard.is_none() {
                    *guard = Some(reuse.build_shadows());
                }
                reuse.take_shadowed()
            };
            let executed = std::mem::take(&mut self.shadow_state.lock().expect("watcher telemetry shadow state").pending_executed);
            if fed.is_empty() && executed.is_empty() {
                return;
            }
            let shadows = guard.as_mut().expect("shadows initialized");
            shadows.read_log(fed);
            let (epoch, now) = {
                let times = self.horizon.lock();
                (times.bank_epoch(), times.clock.through_latest())
            };
            for ev in executed {
                shadows.executed(ev.verifies, ev.depth, ev.real, epoch, now);
            }
            let reentries = shadows.take_reentries();
            let gaps = shadows.gaps(epoch, now);
            self.shadow_state.lock().expect("watcher telemetry shadow state").gaps = gaps;
            reentries
        };
        for reentry in reentries {
            tracing::info!("{reentry}");
        }
    }

    /// The predicted interests a completed decode step resolved
    /// (`interest::Ledger::tell`), its exact rows having chosen `file`
    /// experts from the file at each streamed layer: its exact rows' hints'
    /// candidates count into the hints' record, an expert two leads named
    /// once (one a longer lead's hint did not read is a shorter lead's
    /// candidate again). An ahead row's names and its hints are the front
    /// rows', which the hints never count. The hint gate counts the misses
    /// a hint could have saved at each layer (its file misses and the
    /// candidates a hint landed before the route) and the uses the ledger
    /// expected of each lead's candidates (`gate::HintGate::resolved`), at
    /// the step's `sequence`.
    pub(super) fn resolved(&self, resolved: &[Resolved], cohorts: &[Cohort]) {
        let mut recorded = self.lock();
        let hints = &mut recorded.costs.hints;
        let mut counted = std::collections::BTreeSet::new();
        for resolved in resolved.iter().filter(|resolved| resolved.candidate) {
            let Source::Lead { row: None, .. } = resolved.source else {
                continue;
            };
            if counted.insert((resolved.layer, resolved.expert)) {
                hints.candidates += 1;
                hints.named += u64::from(resolved.used);
                // Used and in a record misses land into: its hint landed
                // before the route, a miss it saved.
                hints.early += u64::from(resolved.early);
            }
        }
        recorded.costs.hints_gate.resolved(cohorts);
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
    pub(super) fn served_whole(&self, served: Option<(Point, Instant, bool)>) {
        if let Some(group) = &mut self.lock().costs.group {
            group.acknowledged = served
                .filter(|(served, _, _)| served.group == group.cost.first)
                .map(|(point, at, last_window)| Acknowledged { point, at, last_window });
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
            if let Some(Acknowledged { point, at, last_window }) = counted.acknowledged.take() {
                compute(&mut counted.cost, facts, point, started.saturating_duration_since(at), last_window);
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
fn compute(group: &mut GroupCost, facts: &mut GroupFacts, served: Point, interval: Duration, last_window: bool) {
    let steps = u32::try_from(group.steps).expect("a group's steps");
    let layer = &mut group.layers[served.layer];
    layer.compute += interval;
    if last_window {
        layer.computed += 1;
    }
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
    let past = |line: &magnitude_cost_model::PassLine, rows: &[u64]| rows.iter().map(|&rows| ExpertLines::rows_past_first(line, rows as f64)).sum::<f64>();
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

/// A completed decode step's routing at every layer, `routed`, is fed with
/// its rows' keys `keys` where its launch wrote them; while its load
/// records it, counted with its launch.
fn feed_decode(
    feed: &mut Feed,
    routed: Vec<RoutedRead>,
    decoded: Decoded,
    keys: Option<std::sync::Arc<StepKeys>>,
    clock: Option<crate::experts::horizon::Tokens>,
) {
    let Decoded { recording, launch, group, .. } = decoded;
    // A recorded decode step runs its launch (`ExpertStep::decoding`). Its
    // reads are its exact rows' (a front row's are the next step's), so it
    // counts those rows: every decode launch has one.
    let counted = recording.map(|Recording| {
        let launch = launch.expect("a recorded decode step has its launch");
        let rows = launch.rows as u64 - u64::from(launch.front.count_ones());
        assert!(rows > 0, "a decode launch has an exact row");
        CountedLaunch { rows, tokens: launch.tokens }
    });
    feed.pending.extend(routed.into_iter().map(|read| (group, read, Fed::Decode { counted, keys: keys.clone(), clock })));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pass's misses beside a host tier hold each resource as the lines
    /// price them: of a layer's three misses (two the tier's, one the
    /// file's, landed), the CPU site served one in 50 µs and the link read
    /// two in place at 86 µs, beside 0.5 ms of moves of four records; host
    /// memory read the three, landed the one and read the four moved, at
    /// 38 µs a record; and the cost function prices only the two host-tier
    /// misses (the file miss is already in `Pass::parked`), holding the layer
    /// 44 µs past its 40 µs of hits (the CPU's two end at 84 µs). A layer that
    /// missed nothing, and one without a host tier (unified memory), hold
    /// nothing.
    #[test]
    fn a_pass_holds_each_resource_its_misses_cross() {
        let tiered = ExpertLines::reference(ExpertLines::REFERENCE_RECORD);
        let unified = ExpertLines { host: magnitude_cost_model::HostLines { cpu: None, memory: None, ..tiered.host }, ..tiered };
        let routed = |layer: usize, experts: usize| RoutedRead { sequence: 0, layer, experts: (0..experts).collect(), choices: experts as u64 };
        let service = PassService::of(
            &[tiered, tiered, unified],
            &[routed(0, 10), routed(1, 8), routed(2, 8)],
            &[vec![3, 4], Vec::new(), vec![1]],
            &[1, 0, 2],
            &[(1, Duration::from_micros(50)), (0, Duration::ZERO), (0, Duration::ZERO)],
            (0.5e-3, 4.),
        );
        let close = |actual: f64, expected: f64| assert!((actual - expected).abs() < 1e-12, "{actual} ≠ {expected}: {service:?}");
        close(service.link, 0.5e-3 + 2. * 86e-6);
        close(service.cpu, 50e-6);
        close(service.memory, (3. + 1.) * 38e-6 + 4. * 38e-6 * 2. / 3.);
        close(service.served, 84e-6 - 40e-6);
    }

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

    /// A pass's time runs from the later of its begin and the step before's
    /// end: a gap between passes (a tool call's) is no pass's time.
    #[test]
    fn a_pass_begun_after_an_idle_gap_counts_from_its_begin() {
        let begun = Instant::now();
        let previous = begun - std::time::Duration::from_secs(30);
        let seconds = step_seconds(Some(previous), begun, begun + std::time::Duration::from_millis(50));
        assert!((seconds - 0.05).abs() < 1e-6, "{seconds}");
    }

    /// A read ahead is admitted where its probability of use reaches the
    /// break-even of a read of its bytes on the drive's line
    /// (`passes::MissCost::break_even`), whatever named it, an exact row's
    /// hint or a front row's name: where the drive binds (68 ms of reads a
    /// 100 ms pass computing 35 ms, a miss stalling 0.96 ms), a 1 MB expert
    /// read in 1 ms is admitted from about 0.511, so a front hint at 0.15
    /// is not; beside a resident-bound pass the drive's slack admits it.
    /// Before the line is measured every read is admitted.
    #[test]
    fn a_read_ahead_is_admitted_at_its_measured_break_even() {
        let line = ReadLine::new(0., 1e-9).expect("a line");
        let price = |pass: Pass| {
            let mut passes = Passes::new(Time::new(&[vec![1]], 8));
            passes.timed(pass);
            ReadPrice { miss: passes.miss_cost(), line: Some(line), ..ReadPrice::unpriced() }
        };
        let bound = price(Pass { rows: 1, depth: 0, seconds: 0.1, parked: 0.085, misses: 68., drive: 0.068, ..Pass::default() });
        // stall = 1.25 ms, read = 1.0 ms -> overhead = 0.25 ms -> break_even(1.0ms, 0.0) = 0.80.
        assert!(bound.admits(0.81, 1_000_000) && !bound.admits(0.79, 1_000_000) && !bound.admits(0.15, 1_000_000));
        assert!(bound.admits(1., 1_000_000));
        let resident = price(Pass { rows: 1, depth: 0, seconds: 0.05, parked: 0.018, misses: 21., drive: 0.021, ..Pass::default() });
        assert!(resident.admits(0.15, 1_000_000), "an idle drive's read saves a stall at no cost");
        let unmeasured = ReadPrice { miss: bound.miss, ..ReadPrice::unpriced() };
        assert!(unmeasured.admits(0.01, 1_000_000), "no line: every read admitted");
    }

    /// A window's classes of reads ahead: bytes issued for a front row's
    /// route stand until their guess falls, cancelled bytes were never
    /// read, and the rest is used or unused.
    #[test]
    fn reads_ahead_are_classed_by_their_basis_and_guess() {
        let mut spent = AheadUse::default();
        let front = Source::Ahead(crate::experts::interest::Row { index: 0, front: Some(crate::GuessKind::Chained) });
        let hint = Source::Lead { distance: 1, row: None };
        spent.issue(front, 100);
        spent.issue(front, 40);
        spent.issue(hint, 30);
        let start = spent;
        spent.issue(front, 50);
        spent.spend(Spent::Fell { basis: Basis::Front, bytes: 50 });
        spent.spend(Spent::Cancelled { basis: Basis::Front, depth: 1, verify: false, fell: true, bytes: 50 });
        spent.spend(Spent::Fell { basis: Basis::Front, bytes: 40 });
        spent.spend(Spent::Unused { basis: Basis::Front, depth: 1, verify: false, fell: true, bytes: 40 });
        spent.spend(Spent::Used { basis: Basis::Front, depth: 1, verify: false, fell: false, bytes: 100 });
        spent.spend(Spent::Used { basis: Basis::Hint, depth: 0, verify: false, fell: false, bytes: 30 });
        let class = |use_: &AheadUse, name: &str| use_.classes().into_iter().find(|(named, _)| *named == name).map(|(_, class)| class).unwrap();
        assert_eq!(class(&spent, "front_stood"), ReadUse { read: 100, used: 100 });
        assert_eq!(class(&spent, "front_wrong"), ReadUse { read: 40, used: 0 });
        assert_eq!(class(&spent, "front_wrong").unused(), 40);
        // By depth: only the settled-unused 40-byte read counts as unused, at depth 1
        // (never negative across a window boundary, and 0 while pending).
        assert_eq!(spent.unused()[0][..2], [0., 1.]);
        assert_eq!(spent.since(&start).unused()[0][..2], [0., 1.]);
        assert_eq!(class(&spent, "hints"), ReadUse { read: 30, used: 30 });
        // The window since `start`: 40 bytes issued before it fell within it.
        let window = spent.since(&start);
        assert_eq!(class(&window, "front_stood"), ReadUse { read: -40, used: 100 });
        assert_eq!(class(&window, "front_wrong"), ReadUse { read: 40, used: 0 });
    }

    /// A front read ahead issued on step `s` for `s + 1` is pending when step `s`
    /// completes: it reports 0 waste at step `s`, and only reports waste once
    /// step `s + 1` settles without using it.
    #[test]
    fn pending_front_read_ahead_reports_zero_waste_until_target_step_settles() {
        let mut spent = AheadUse::default();
        let front = Source::Ahead(crate::experts::interest::Row { index: 0, front: Some(crate::GuessKind::Chained) });
        // Step s: front row 1 issues two reads ahead for step s + 1.
        spent.issue(front, 100);
        spent.issue(front, 100);
        let at_s = spent;
        assert_eq!(at_s.unused()[0][1], 0., "pending reads for step s + 1 must report 0 waste at step s");
        // Step s + 1: one read is used, the other settles unused.
        spent.spend(Spent::Used { basis: Basis::Front, depth: 1, verify: false, fell: false, bytes: 100 });
        spent.spend(Spent::Unused { basis: Basis::Front, depth: 1, verify: false, fell: false, bytes: 100 });
        assert_eq!(spent.since(&at_s).unused()[0][1], 1., "only the read settled unused at step s + 1 counts as waste");
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
