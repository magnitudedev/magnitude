//! The steps the watcher serves, from their beginning in a group until
//! each ends, completed or dropped, and why one failed.
//!
//! Steps end in any order, by any exit: completed, refused, dropped, or
//! never queued (an admission that unwinds). Whatever the exit, the device
//! runs nothing of a step once it has ended (submission owners drain issued
//! work before releasing a step). The one completion fact every reuse of a
//! record rests on is the frontier ([`Steps::frontier`]): the least step not
//! yet ended, below which every step has ended. Every exit advances it.
//!
//! The steps are the one owner of the device's order (`Point`): the pieces
//! each group runs as ([`Piece`]), a group's own from its begin, and one
//! more each time a group is cut in before one of its layers
//! ([`Shared::begin_cut_in`]). Every point a step's resolution has yet to
//! reach is read from them when it is compared ([`Steps::after`],
//! [`Steps::point`], [`Steps::park`], [`Steps::served`]), since a cut moves
//! it later; only points the device has reached are kept.

use super::Shared;
use super::core::At;
use crate::assessment::ExpertLines;
use crate::experts::directory::{Park, Piece, Point};
use crate::experts::events::Events;
use crate::FlightDepth;
use crate::operation::LaunchWork;
use crate::programs::Chain;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// A step's sequence, the host's one step type: the u64 the host and the
/// device share (`streaming_words.seismic`'s steps), consecutive from one.
pub(crate) type Step = u64;

/// Step `step`'s low word: what the words that tell steps apart only within
/// a flight carry (rings, lists and routing records,
/// `streaming_words.seismic`), taken here alone.
pub(crate) fn low_word(step: Step) -> i32 {
    step as u32 as i32
}

/// Why a submitted step failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StepFailure {
    /// It could not obtain its experts: a read failed (the step's rows that
    /// named them fail). The load goes on.
    Read { rows: FailedRows, reason: String },
    /// The CPU site failed its share of the step's tier-served choices (a
    /// layer's request answered `-1 - q`): the rows its entries name fail.
    /// The load goes on.
    Site { rows: FailedRows, reason: String },
    /// The watcher stopped serving, for this reason (it panicked, or a
    /// demand read left the served file with no read line): it serves no
    /// request again, so the load completes no step.
    Watcher(String),
    /// A wait of the step ended on its still pulse before its answer
    /// (`exchange::Pulse`, which its kernel recorded: `Parks::expired`), so
    /// its outputs are not to be used and its seeds are never mapped, but
    /// its inputs are whole: it is to be re-run from its accepted state.
    /// The load goes on.
    Interrupted(String),
}

impl std::fmt::Display for StepFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read { reason, .. } | Self::Site { reason, .. } => f.write_str(reason),
            Self::Watcher(reason) => write!(f, "the expert watcher stopped serving: {reason}"),
            Self::Interrupted(reason) => f.write_str(reason),
        }
    }
}

/// The rows of a step a failure fails: every one (a failure no expert's,
/// such as the watcher's), or
/// those whose choices named an expert that could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FailedRows {
    Every,
    Only(std::collections::BTreeSet<usize>),
}

impl FailedRows {
    /// Whether any of the step's rows `rows` fails.
    pub(crate) fn fails(&self, rows: std::ops::Range<usize>) -> bool {
        match self {
            Self::Every => true,
            Self::Only(failed) => failed.range(rows).next().is_some(),
        }
    }

    /// What fails once `other` fails too.
    pub(crate) fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Only(mut rows), Self::Only(more)) => {
                rows.extend(more);
                Self::Only(rows)
            }
            _ => Self::Every,
        }
    }
}

/// One of a group's pieces ([`Piece`]): its layers from `from` up to the
/// next piece's, and where it begins in the device's order of layer runs.
#[derive(Clone, Copy, Debug)]
struct Opened {
    from: usize,
    piece: Piece,
    base: Base,
}

/// Where a piece begins among the device's layer runs, `steps` steps' runs
/// (the bank's streamed layers each) and `runs` more, which a step's point
/// counts from (`At`): a group begun after every other is past the runs of
/// every step before it, and a group cut in before a layer of another is
/// past that group's runs before the layer and its fronts at it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Base {
    steps: u64,
    runs: u64,
}

impl Base {
    fn at(self, layers: usize) -> u64 {
        self.steps * layers as u64 + self.runs
    }
}

/// A group of steps begun: its steps, those ended, when it began, and
/// whether the watcher has retired it (the device is past it).
struct Begun {
    steps: u64,
    /// The pieces it runs as, by their first layer: one, its own, until a
    /// group is cut in before one of its layers.
    pieces: Vec<Opened>,
    ended: u64,
    begun: Instant,
    retired: bool,
    /// Whether its steps are scratch-served: a prompt group's requests.
    scratch: bool,
    /// Each streamed layer's merge set, the steps the group ends with
    /// whose backs run in one pass (`Telemetry::merge_sets`).
    merges: Vec<usize>,
    /// The group reads record a group of grouped steps holds
    /// (`BankPart::GroupReads`), until every step of it has ended.
    reads: Option<GroupRecord>,
    /// Whether a launch of it is a prompt's (`LaunchWork::Prompt`): its
    /// reads are a prompt's, and it runs as one.
    prompt: bool,
}

/// A group's hold of one of the group reads records, one for each group in
/// flight ([`FlightDepth::groups`])
/// (`BankPart::GroupReads`): taken from the pool as the group begins and
/// returned once every step of it has ended.
struct GroupRecord(usize);

/// What a step's launch routes by, its form's protocol: a grouped launch's
/// rows (more than the decode form's) run its group's passes, which record
/// the group's reads (`BankPart::GroupReads`); a decode launch's route
/// writes its routing record. Which work the rows are, a prompt's or
/// generation's, is the launch's own (`LaunchWork`), apart from its form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepRoute {
    Grouped,
    Decode,
}

/// A group of grouped steps whose every step has ended, with the group
/// reads record it held, each streamed layer's merge set, and whether it
/// was a prompt's: only a prompt's reads price a prompt's group reads.
pub(super) struct EndedReads {
    pub first: Step,
    pub steps: u64,
    pub record: usize,
    pub merges: Vec<usize>,
    pub prompt: bool,
}

/// The steps' state, under one lock.
struct State {
    /// The last sequence begun, and the steps begun and not yet ended.
    last: Step,
    running: usize,
    /// The frontier, the least step not yet ended, and the steps above it
    /// that have ended: every step below the frontier has.
    frontier: Step,
    ended: BTreeSet<Step>,
    /// The step owning each routing record (sequence modulo the flight,
    /// `RegionWords::flight`), none where no step does (sequences begin at
    /// one): no step begins on a record a running step still owns, whatever
    /// order steps end in.
    slots: Box<[Option<NonZeroU64>]>,
    /// The group reads records no group holds.
    records: Vec<GroupRecord>,
    /// The groups begun, by first sequence (its name, [`Point::group`]),
    /// each kept until it is both retired and ended.
    groups: BTreeMap<Step, Begun>,
    /// Steps a request of which failed, the rows that fails and why (the
    /// first failure's). Recorded before the request's failure is
    /// published, so before its step can complete.
    failed: BTreeMap<Step, (FailedRows, String)>,
    /// The runs every cut added before the groups begun since
    /// ([`State::open`]): a group begun after every other begins past
    /// them.
    cut_runs: u64,
    /// Steps whose runs serve their streamed layers from `layer` on over a
    /// window of the step's rows starting at batch row `first` (a stateless
    /// tail's, `Shared::window`): (layer, first). A request of such a layer
    /// names its rows within the window.
    windows: BTreeMap<Step, (usize, usize)>,
    /// Steps a wait of which ended on its still pulse before its answer
    /// (`Parks::expired`), so their seeds are never mapped
    /// (`Core::advance`). Kept until residency is past their group.
    tainted: BTreeSet<Step>,
}

impl Begun {
    /// The piece its steps resolve `layer` in: its pass there (a back's
    /// read of the layer's experts).
    fn pass(&self, layer: usize) -> &Opened {
        self.pieces.iter().rev().find(|opened| opened.from <= layer).expect("a group's first piece is from its first layer")
    }

    /// The piece its steps request `layer` in: a front's, which at a layer a
    /// group was cut in before runs before the group cut in (only the pass
    /// after it is cut), and so in the piece before.
    fn front(&self, layer: usize) -> &Opened {
        self.pieces.iter().rev().find(|opened| opened.from < layer.max(1)).expect("a group's first piece is from its first layer")
    }

    /// The point past every layer of every one of its steps.
    fn after(&self, group: Step) -> Point {
        let piece = self.pieces.last().expect("a group runs a piece").piece;
        Point { piece, layer: usize::MAX, sequence: Step::MAX, group }
    }
}

impl State {
    /// The point of layer `layer` of step `sequence` of group `group` that
    /// its pass there reaches ([`Begun::pass`]): the start of the device's
    /// order for a group forgotten, every step of which ended.
    fn pass_point(&self, group: Step, layer: usize, sequence: Step) -> Point {
        self.groups.get(&group).map_or(Point::START, |begun| Point { piece: begun.pass(layer).piece, layer, sequence, group })
    }

    /// The pieces of the group of `count` steps from `first` beginning now,
    /// beside the cut it makes where it is cut in `before` (group, layer):
    /// that group's layers from `layer` on become a piece of their own,
    /// behind the new group's, which begins past that group's runs before
    /// the layer and its fronts there. A cut takes the group whose piece
    /// is the latest submitted (nothing but groups cut in before it began
    /// since), at a layer past its first and none before its last piece's:
    /// none of its runs from the layer on can have been submitted. At its
    /// last piece's own layer (another group cut in at the same cut) that
    /// piece moves behind the new group. Any other cut is an invariant
    /// violation.
    fn open(&mut self, first: Step, count: u64, before: Option<(Step, usize)>) -> Vec<Opened> {
        let Some((group, layer)) = before else {
            return vec![Opened { from: 0, piece: Piece::of(first), base: Base { steps: first, runs: self.cut_runs } }];
        };
        let latest = self.groups.values().map(|begun| begun.pieces.last().expect("a group runs a piece").piece).max();
        let cut = self.groups.get_mut(&group).unwrap_or_else(|| panic!("a group cut in before group {group}, which is not running"));
        let last = *cut.pieces.last().expect("a group runs a piece");
        assert_eq!(Some(last.piece), latest, "a group cut in before group {group}, whose piece {:?} is not the latest submitted", last.piece);
        assert!(cut.ended < cut.steps, "a group cut in before group {group}, every step of which ended");
        assert!(layer > 0 && layer >= last.from, "a group cut in before layer {layer} of group {group}, whose last piece runs from layer {}", last.from);
        let base = match layer == last.from {
            true => last.base,
            false => {
                let runs = (layer - last.from + 1) as u64 * cut.steps;
                self.cut_runs += cut.steps;
                Base { runs: last.base.runs + runs, ..last.base }
            }
        };
        let behind = Opened { from: layer, piece: Piece { opened: first, behind: true }, base: Base { steps: base.steps + count, ..base } };
        if layer == last.from {
            cut.pieces.pop();
        }
        cut.pieces.push(behind);
        vec![Opened { from: 0, piece: Piece::of(first), base }]
    }

    /// The first step of the group of `point` or a later one whose pass at
    /// `layer` lies after `point` (`Park::unresolved`): the first such
    /// group's first step, or the parked step's next at its own layer; the
    /// first step not yet begun where none.
    fn first_unresolved(&self, point: Point, layer: usize) -> Step {
        use std::cmp::Ordering;
        for (&first, begun) in self.groups.range(point.group..) {
            let unresolved = match begun.pass(layer).piece.cmp(&point.piece) {
                Ordering::Greater => Some(first),
                Ordering::Less => None,
                Ordering::Equal => match layer.cmp(&point.layer) {
                    Ordering::Greater => Some(first),
                    Ordering::Equal => Some(point.sequence + 1).filter(|&next| next < first + begun.steps),
                    Ordering::Less => None,
                },
            };
            if let Some(unresolved) = unresolved {
                return unresolved;
            }
        }
        self.last + 1
    }

    /// The steps of groups begun before `point`'s whose pass at `layer`
    /// lies after it, (first, last) (`Park::earlier`): a group's steps
    /// resolve a layer in one piece, all after a point of another's or none.
    fn unresolved_before(&self, point: Point, layer: usize) -> Option<(Step, Step)> {
        self.groups
            .range(..point.group)
            .filter(|(_, begun)| begun.pass(layer).piece > point.piece)
            .map(|(&first, begun)| (first, first + begun.steps - 1))
            .reduce(|(first, _), (_, last)| (first, last))
    }

    /// The group step `sequence` began in, by first sequence.
    fn group(&mut self, sequence: Step) -> (Step, &mut Begun) {
        self.groups
            .range_mut(..=sequence)
            .next_back()
            .filter(|(first, group)| sequence - **first < group.steps)
            .map(|(&first, group)| (first, group))
            .unwrap_or_else(|| panic!("step {sequence} began in no group"))
    }

    /// Forget the groups both retired and ended.
    fn forget(&mut self) {
        self.groups.retain(|_, group| !(group.retired && group.ended == group.steps));
    }
}

/// The first step not yet begun, while the lock every step begins under is
/// held ([`Steps::unbegun`], its one maker): what a map entry is stored
/// from or until so that it holds for every step that begins after it.
pub(in crate::experts) struct Unbegun(Step);

impl Unbegun {
    pub(in crate::experts) fn next(&self) -> Step {
        self.0
    }

    /// The first step not yet begun at `next`, where a test plays the steps.
    #[cfg(test)]
    pub(in crate::experts) fn at(next: Step) -> Self {
        Self(next)
    }
}

/// The steps begun and how far they are: the one owner of the groups they
/// begin in. Serving holds its lock across no wait. Every begin, end and
/// resize asked is one of the load's events ([`Self::events`]), whose
/// count it owns.
pub(in crate::experts) struct Steps {
    state: Mutex<State>,
    events: Arc<Events>,
    /// The frontier as its readers read it ([`Self::frontier`]), stored
    /// under the lock as each end advances it.
    frontier: AtomicU64,
    /// The steps in flight each layer keeps a routing record for
    /// (`RegionWords::flight`): no step begins beyond them.
    flight: usize,
    /// The load's pulse, which every table of its exchange beats with
    /// (`exchange::Pulse`).
    pulse: crate::experts::exchange::Pulse,
    /// The steps the load interrupted (`StepFailure::Interrupted`), each re-run:
    /// reported with the load's end, so a host that interrupts every step
    /// shows as a count, not a silent slowdown.
    interrupted: AtomicU64,
}

impl Steps {
    /// Steps of a load of `places` chain places flying at `flights`: a
    /// routing record for each step in flight ([`FlightDepth::steps`]) and a
    /// group reads record for each group, whose tables beat with `pulse`.
    pub(in crate::experts) fn new(flights: FlightDepth, places: u64, pulse: crate::experts::exchange::Pulse) -> Self {
        let flight = usize::try_from(flights.steps(places)).expect("the steps in flight fit the host");
        Self {
            flight,
            state: Mutex::new(State {
                last: 0,
                running: 0,
                frontier: 1,
                ended: BTreeSet::new(),
                slots: vec![None; flight].into_boxed_slice(),
                records: (0..flights.groups()).rev().map(GroupRecord).collect(),
                groups: BTreeMap::new(),
                failed: BTreeMap::new(),
                tainted: BTreeSet::new(),
                windows: BTreeMap::new(),
                cut_runs: 0,
            }),
            events: Arc::default(),
            frontier: AtomicU64::new(1),
            pulse,
            interrupted: AtomicU64::new(0),
        }
    }

    /// How many of the load's steps were interrupted, and how many beats of
    /// its pulse came late: the load's report of its starved host.
    pub(in crate::experts) fn interruptions(&self) -> String {
        format!(
            "steps interrupted by a wait ending on a still pulse and re-run: {} ({} late beats)",
            self.interrupted.load(Ordering::Acquire),
            self.pulse.late_beats()
        )
    }

    /// The load's pulse, which every table of its exchange joins as it is
    /// made (`ExchangeTable::new`).
    pub(in crate::experts) fn pulse(&self) -> &crate::experts::exchange::Pulse {
        &self.pulse
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("watcher steps")
    }

    /// The load's events, which every host job of streaming waits on.
    pub(in crate::experts) fn events(&self) -> &Arc<Events> {
        &self.events
    }

    /// The routing record step `sequence` owns while it runs.
    fn slot(&self, sequence: Step) -> usize {
        (sequence % self.flight as u64) as usize
    }

    /// The group step `sequence` began in: its first sequence and its
    /// steps.
    pub(super) fn group_of(&self, sequence: Step) -> (Step, u64) {
        let mut state = self.lock();
        let (first, group) = state.group(sequence);
        (first, group.steps)
    }

    /// Step `sequence`'s group and when it began.
    fn begun(&self, sequence: Step) -> (Step, Instant) {
        let mut state = self.lock();
        let (first, group) = state.group(sequence);
        (first, group.begun)
    }

    /// Step `sequence` ended, by whichever exit, its failure forgotten, and
    /// the frontier advanced past it where it was the least step running:
    /// `ended` takes its group with its group reads record, where it was the
    /// group's last step to end and the group holds one, and the first
    /// sequence of the earliest group not every step of which has ended
    /// (every group before it has), before anything between flights can
    /// observe the step's end. The record is free once `ended` has read it.
    fn end(&self, sequence: Step, ended: impl FnOnce(Option<EndedReads>, Step)) {
        let mut state = self.lock();
        let owner = state.slots[self.slot(sequence)].take();
        assert_eq!(owner.map(NonZeroU64::get), Some(sequence), "an ending step owns its routing record");
        state.failed.remove(&sequence);
        state.windows.remove(&sequence);
        let State { ended: above, frontier, .. } = &mut *state;
        above.insert(sequence);
        let before = *frontier;
        while above.remove(frontier) {
            *frontier += 1;
        }
        if *frontier != before {
            self.frontier.store(*frontier, Ordering::Release);
        }
        let (first, group) = state.group(sequence);
        group.ended += 1;
        let steps = group.steps;
        let prompt = group.prompt;
        let held = match group.ended == steps {
            true => group.reads.take().map(|record| (record, group.merges.clone())),
            false => None,
        };
        state.running -= 1;
        state.forget();
        let earliest = state
            .groups
            .iter()
            .find(|(_, group)| group.ended < group.steps)
            .map_or(state.last + 1, |(&first, _)| first);
        let (record, reads) = held
            .map(|(GroupRecord(record), merges)| (GroupRecord(record), EndedReads { first, steps, record, merges, prompt }))
            .unzip();
        ended(reads, earliest);
        state.records.extend(record);
        drop(state);
        self.events.notify();
    }

    /// The request at layer `layer` of step `sequence` in the device's
    /// order, of a bank of `layers` layers: its front's point
    /// ([`Begun::front`]), and the layer runs before it.
    pub(in crate::experts) fn at(&self, sequence: Step, layer: usize, layers: usize) -> At {
        let mut state = self.lock();
        let (group, begun) = state.group(sequence);
        let opened = *begun.front(layer);
        let position = opened.base.at(layers) + (layer - opened.from) as u64 * begun.steps + (sequence - group);
        At::new(Point { piece: opened.piece, layer, sequence, group }, position)
    }

    /// The point of layer `layer` of step `sequence` of group `group`
    /// (which may name the step after the group's last) that its pass
    /// there reaches, read now: the start of the device's order for a
    /// group forgotten, every step of which ended.
    pub(super) fn point(&self, group: Step, layer: usize, sequence: Step) -> Point {
        self.lock().pass_point(group, layer, sequence)
    }

    /// The point past every layer of every step of group `group`, read
    /// now: the start of the device's order for a group forgotten, every
    /// step of which ended.
    pub(in crate::experts) fn after(&self, group: Step) -> Point {
        self.lock().groups.get(&group).map_or(Point::START, |begun| begun.after(group))
    }

    /// The device parked at `point` (a decode request at its layer), of a
    /// bank of `layers` layers ([`Park`]).
    pub(in crate::experts) fn park(&self, point: Point, layers: usize) -> Park {
        let state = self.lock();
        Park {
            point,
            unresolved: (0..layers).map(|layer| state.first_unresolved(point, layer)).collect(),
            earlier: (0..layers).map(|layer| state.unresolved_before(point, layer)).collect(),
        }
    }

    /// Whether every step of `[from, until)` has passed `layer` (its pass
    /// there lies before `progress`) or ended, read now: an entry of the
    /// layer mapped for those steps and unmapped is then read by none, so
    /// its record is reused. Steps not yet begun read what was stored
    /// before they begin. A group's steps pass a layer in sequence order, so each group
    /// running in the range is past where its last step there not ended is.
    pub(in crate::experts) fn served(&self, from: Step, until: Step, layer: usize, progress: Point) -> bool {
        let state = self.lock();
        let (from, until) = (from.max(state.frontier), until.min(state.last + 1));
        if from >= until {
            return true;
        }
        let first = state.groups.range(..=from).next_back().map_or(from, |(&first, _)| first);
        state.groups.range(first..until).all(|(&group, begun)| {
            let mut running = (group.max(from)..(group + begun.steps).min(until)).rev().filter(|step| !state.ended.contains(step));
            running.next().is_none_or(|last| Point { piece: begun.pass(layer).piece, layer, sequence: last, group } < progress)
        })
    }

    /// The first group the device is not past at `progress`: the earlier
    /// of `progress`'s own and the first whose last piece it is not past
    /// (one it runs inside, cut in before that one's later layers).
    pub(super) fn first_unpassed(&self, progress: Point) -> Step {
        let state = self.lock();
        let unpassed = state.groups.iter().find(|&(&first, begun)| begun.after(first) > progress).map(|(&first, _)| first);
        unpassed.map_or(progress.group, |first| first.min(progress.group))
    }

    /// Retire the groups whose every step is below `frontier`: the device
    /// is past them. The point past the latest of them in the device's
    /// order, read before any is forgotten, and how many scratch-served
    /// groups retired now (each held the lent records' reservation,
    /// `lent`).
    pub(super) fn retire_groups(&self, frontier: Step) -> (Option<Point>, usize) {
        let mut state = self.lock();
        let (mut past, mut reserving) = (None::<Point>, 0);
        for (&first, group) in &mut state.groups {
            if first + group.steps <= frontier {
                reserving += usize::from(!group.retired && group.scratch);
                group.retired = true;
                past = past.max(Some(group.after(first)));
            }
        }
        state.forget();
        (past, reserving)
    }

    /// The step begun in flight whose low word is `low` ([`low_word`]): the
    /// step a word that carries only its low word names. The steps from the
    /// frontier to the last begun are fewer than the routing records (a step
    /// begins only once the step a flight before it ended), far fewer than
    /// 2^32, so at most one has it; none where that step has ended, or none
    /// such has begun.
    pub(in crate::experts) fn of_low_word(&self, low: i32) -> Option<Step> {
        let state = self.lock();
        let span = state.frontier..=state.last;
        let step = (state.last & !u64::from(u32::MAX)) | u64::from(low as u32);
        [step, step.wrapping_sub(1 << 32)]
            .into_iter()
            .find(|step| span.contains(step) && !state.ended.contains(step))
    }

    /// The frontier: the least step not yet ended, every step below it
    /// ended by whichever exit, so the device runs none of them. A record
    /// unmapped from a step on is reusable once the frontier reaches it.
    pub(in crate::experts) fn frontier(&self) -> Step {
        self.frontier.load(Ordering::Acquire)
    }

    /// Whether step `step` has ended, by whichever exit: below the
    /// frontier, or above it and ended out of order.
    pub(in crate::experts) fn ended(&self, step: Step) -> bool {
        let state = self.lock();
        step < state.frontier || state.ended.contains(&step)
    }

    /// `write` given the first step not yet begun, under the lock every step
    /// begins under: no step begins until it returns, so every step from
    /// that one on is queued after what it stores (a map entry's FROM or
    /// UNTIL, `experts::map`).
    pub(in crate::experts) fn unbegun<T>(&self, write: impl FnOnce(&Unbegun) -> T) -> T {
        let state = self.lock();
        // Admission keeps the step after the last one begun representable.
        write(&Unbegun(state.last + 1))
    }

    /// `write` given the frontier and the first step not yet begun, read
    /// together under the lock every step begins and ends under: every step
    /// in between is in flight, so they are at most the flight apart.
    pub(in crate::experts) fn unbegun_beside_frontier<T>(&self, write: impl FnOnce(Step, &Unbegun) -> T) -> T {
        let state = self.lock();
        write(state.frontier, &Unbegun(state.last + 1))
    }

    /// Whether step `last` is the last begun: no group began after its own.
    pub(super) fn latest(&self, last: Step) -> bool {
        self.lock().last == last
    }

    /// Whether a running group is a prompt group's, however it is served
    /// (scratch or resident): its steps run many at once.
    pub(super) fn prompt_running(&self) -> bool {
        self.lock().groups.values().any(|group| group.prompt && group.ended < group.steps)
    }

    /// Steps through `last` begun, each a group of one run whole, where a
    /// test plays them without launches: what an entry stored from the
    /// first step not yet begun is stored from.
    #[cfg(test)]
    pub(in crate::experts) fn begin_through(&self, last: Step) {
        while self.lock().last < last {
            self.begin_group(1, None);
        }
    }

    /// A group of `count` steps begun, where a test plays it without
    /// launches, cut in `before` a running group's layer where given
    /// ([`Shared::begin_cut_in`]): its first step.
    #[cfg(test)]
    pub(in crate::experts) fn begin_group(&self, count: u64, before: Option<(Step, usize)>) -> Step {
        let mut state = self.lock();
        let first = state.last + 1;
        let pieces = state.open(first, count, before);
        let begun = Begun {
            steps: count,
            pieces,
            ended: 0,
            begun: Instant::now(),
            retired: false,
            scratch: false,
            merges: Vec::new(),
            reads: None,
            prompt: false,
        };
        state.groups.insert(first, begun);
        state.last = first + count - 1;
        state.running += count as usize;
        first
    }

    /// Step `sequence` of a group a test began ([`Self::begin_group`])
    /// ended: the frontier advanced past it where it was the least
    /// running.
    #[cfg(test)]
    pub(in crate::experts) fn end_played(&self, sequence: Step) {
        let mut state = self.lock();
        let State { ended, frontier, .. } = &mut *state;
        ended.insert(sequence);
        while ended.remove(frontier) {
            *frontier += 1;
        }
        let frontier = *frontier;
        self.frontier.store(frontier, Ordering::Release);
        state.group(sequence).1.ended += 1;
        state.running -= 1;
    }

    /// The steps that may hold a request now: from the frontier through
    /// the last begun.
    pub(super) fn live(&self) -> std::ops::Range<Step> {
        let state = self.lock();
        state.frontier..state.last + 1
    }

    pub(super) fn in_flight(&self) -> usize {
        self.lock().running
    }

    /// Run `answer`, a write of step `sequence`'s slot of a layer's answers,
    /// while the step owns its routing record, under the lock its begin and
    /// end take: the slot is the step's alone until it ends, and the next
    /// step of the slot begins only after, so no other step's answer is
    /// written while it reads its own. Nothing runs once the step ended.
    pub(in crate::experts) fn while_owned(&self, sequence: Step, answer: impl FnOnce()) {
        let state = self.lock();
        if state.slots[self.slot(sequence)].map(NonZeroU64::get) == Some(sequence) {
            answer();
        }
    }

    pub(super) fn fail(&self, sequence: Step, reason: String) {
        self.fail_rows(sequence, FailedRows::Every, reason);
    }

    /// How the work at `point` ended, read under one lock: its step and
    /// every earlier step of its group, whose scratch rounds converted
    /// experts it may read (a seed copies them), running still, or all
    /// ended whole, or one tainted. A step is tainted before it ends
    /// (`ExpertStep::drop`), so one read ended carries its taint.
    pub(super) fn ending(&self, point: Point) -> crate::experts::directory::Ending {
        use crate::experts::directory::Ending;
        let state = self.lock();
        let steps = point.group..=point.sequence;
        let ended = steps.clone().all(|sequence| sequence < state.frontier || state.ended.contains(&sequence));
        match (ended, state.tainted.range(steps).next().is_some()) {
            (false, _) => Ending::Running,
            (true, false) => Ending::Ended,
            (true, true) => Ending::Tainted,
        }
    }

    /// Residency is past every group before `group`: their taints are done
    /// with.
    pub(super) fn untaint_before(&self, group: Step) {
        self.lock().tainted.retain(|&sequence| sequence >= group);
    }

    /// Step `sequence`'s runs serve its streamed layers from `layer` on over
    /// a window of its rows starting at batch row `first`.
    pub(super) fn window(&self, sequence: Step, layer: usize, first: usize) {
        self.lock().windows.insert(sequence, (layer, first));
    }

    /// The batch rows of step `sequence` that rows `rows` of its request at
    /// `layer` are: past the window the layer is served over
    /// ([`Self::window`]). Every failure a request's rows name reaches the
    /// step through this one translation.
    fn batch_rows(&self, sequence: Step, layer: usize, rows: &BTreeSet<usize>) -> BTreeSet<usize> {
        let first = match self.lock().windows.get(&sequence) {
            Some(&(from, first)) if layer >= from => first,
            _ => 0,
        };
        rows.iter().map(|row| first + row).collect()
    }

    /// Step `sequence`'s request at `layer` failed for `reason` at its rows
    /// `rows`: the step's batch rows they are ([`Self::batch_rows`]) fail.
    pub(super) fn fail_request_rows(&self, sequence: Step, layer: usize, rows: &BTreeSet<usize>, reason: String) {
        self.fail_rows(sequence, FailedRows::Only(self.batch_rows(sequence, layer, rows)), reason);
    }

    /// Step `sequence`'s `rows` fail for `reason` (beside any that failed
    /// before, whose reason is kept).
    fn fail_rows(&self, sequence: Step, rows: FailedRows, reason: String) {
        let mut state = self.lock();
        match state.failed.remove(&sequence) {
            Some((failed, first)) => state.failed.insert(sequence, (failed.and(rows), first)),
            None => state.failed.insert(sequence, (rows, reason)),
        };
    }

    /// Between flights (no step runs), `then`, an event; why not while
    /// steps run, which the owner rules out by asking only in a gap
    /// (`Shared::in_gap`).
    pub(super) fn between_flights(&self, then: impl FnOnce()) -> Result<(), String> {
        let state = self.lock();
        if state.running != 0 {
            return Err(format!("a bank resize while {} steps run", state.running));
        }
        then();
        drop(state);
        self.events.notify();
        Ok(())
    }

    /// `write` while no step runs, under the lock every step begins under:
    /// no step begins until it returns, so every run encoded after it reads
    /// what it wrote (the policy words, `Residency::publish_policy`). None
    /// while steps run.
    pub(super) fn while_idle<T>(&self, write: impl FnOnce() -> T) -> Option<T> {
        let state = self.lock();
        (state.running == 0).then(write)
    }

    /// Whether no step runs, as of now.
    pub(super) fn idle(&self) -> bool {
        self.lock().running == 0
    }

    /// Whether a running group is scratch-served: its requests may come
    /// whatever its layers hold.
    pub(super) fn scratch_running(&self) -> bool {
        self.lock().groups.values().any(|group| group.scratch && group.ended < group.steps)
    }

    /// The steps' state, for the exchange's watchdog: the frontier, the
    /// last begun, those running and ended above the frontier, each group
    /// begun, and the failures recorded.
    pub(in crate::experts) fn describe(&self) -> String {
        let state = self.lock();
        let groups = state
            .groups
            .iter()
            .map(|(first, group)| {
                format!(
                    "{first}: {} steps, {} ended, began {:?} ago, retired {}, scratch {}",
                    group.steps,
                    group.ended,
                    group.begun.elapsed(),
                    group.retired,
                    group.scratch
                )
            })
            .collect::<Vec<_>>();
        format!(
            "frontier {}, last begun {}, running {}, ended above the frontier {:?}; groups [{}]; failed {:?}",
            state.frontier,
            state.last,
            state.running,
            state.ended,
            groups.join("; "),
            state.failed.iter().map(|(step, (_, reason))| (step, reason)).collect::<Vec<_>>(),
        )
    }
}

impl Shared {
    /// Begin the next group of steps of `launches`' rows, in sequence
    /// order: a chain's launches, which the device runs layer-major, or one
    /// step; `scratch` says whether they are scratch-served, read where no
    /// publication of the whole words runs (they are published while no
    /// step runs, under this lock), so what it reads of them holds until
    /// the group ends. A scratch-served group holds the lent records
    /// reserved from before its steps begin (`lent`): where a copy into
    /// them runs, it waits for every one to complete outside this lock,
    /// then reads `scratch` again; or for serving's end, why it cannot
    /// begin. Each streamed layer's merge set comes with them: the
    /// priced one (`Telemetry::merge_sets`) for a scratch-served group, the
    /// whole group otherwise. A group with a grouped step takes a group reads
    /// record from the pool, one for each group the queue can have in
    /// flight (`FlightDepth::groups`), zeroed for its steps' places before any
    /// is queued. No group begins beyond the steps in flight (`Steps::flight`):
    /// each step's routing record, its sequence's modulo them, then holds
    /// until the step ends. Each launch comes with its step; why the group
    /// cannot begin comes with the launches. The group runs after every
    /// group begun before it: one piece, its own.
    pub fn begin<L>(
        self: &Arc<Self>,
        launches: Chain<L>,
        route: impl Fn(&L) -> StepRoute,
        work: impl Fn(&L) -> LaunchWork,
        scratch: impl FnMut() -> Result<bool, String>,
    ) -> Result<(Chain<(L, ExpertStep)>, Vec<usize>), (String, Chain<L>)> {
        self.open(launches, route, work, scratch, None)
    }

    /// [`Self::begin`] a group cut in before layer `layer` of running group
    /// `group` (a decode step run between two layers of a prompt chain):
    /// it runs after that group's runs before the layer and its fronts and
    /// fills at the layer, and before its passes from the layer on, which
    /// become a piece of their own behind it (`Piece`), in the one lock its
    /// begin takes, so no point of it is read before the cut. Its runs are
    /// submitted before any of that group's from the layer on, and that
    /// group's piece is the latest submitted (no group but those cut in
    /// before it began since); any other cut is an invariant violation.
    pub fn begin_cut_in<L>(
        self: &Arc<Self>,
        launches: Chain<L>,
        route: impl Fn(&L) -> StepRoute,
        work: impl Fn(&L) -> LaunchWork,
        scratch: impl FnMut() -> Result<bool, String>,
        (group, layer): (Step, usize),
    ) -> Result<(Chain<(L, ExpertStep)>, Vec<usize>), (String, Chain<L>)> {
        self.open(launches, route, work, scratch, Some((group, layer)))
    }

    /// Begin the next group of `launches` ([`Self::begin`]), cut in
    /// `before` a running group's layer where given
    /// ([`Self::begin_cut_in`]).
    fn open<L>(
        self: &Arc<Self>,
        launches: Chain<L>,
        route: impl Fn(&L) -> StepRoute,
        work: impl Fn(&L) -> LaunchWork,
        mut scratch: impl FnMut() -> Result<bool, String>,
        before: Option<(Step, usize)>,
    ) -> Result<(Chain<(L, ExpertStep)>, Vec<usize>), (String, Chain<L>)> {
        let grouped = launches.iter().any(|launch| route(launch) == StepRoute::Grouped);
        let prompt = launches.iter().any(|launch| work(launch) == LaunchWork::Prompt);
        // Whether the group holds the lent records reserved (`lent`): a
        // scratch-served group does before any of its steps begins, until
        // it retires. Where a copy into them runs it waits outside the lock
        // and looks again, holding them: at most twice.
        let mut reserved = false;
        let begun = loop {
            let attempt = (|| -> Result<Option<_>, String> {
                let count = launches.len() as u64;
                let mut state = self.steps.lock();
                if launches.len() > self.steps.flight {
                    return Err(format!("a group of {} steps exceeds the {} routing records", launches.len(), self.steps.flight));
                }
                // Retirement names the sequence after the last one begun, so
                // that one must be representable too.
                let first = state
                    .last
                    .checked_add(1)
                    .filter(|first| first.checked_add(count).is_some())
                    .ok_or("the load's step sequences are exhausted")?;
                for sequence in first..first + count {
                    if let Some(held) = state.slots[self.steps.slot(sequence)] {
                        return Err(format!("step {sequence}'s routing record is still owned by step {held}"));
                    }
                }
                let scratch = scratch()?;
                if scratch && !reserved {
                    if !self.lent.reserve() {
                        return Ok(None);
                    }
                    reserved = true;
                }
                if !scratch && reserved {
                    self.lent.release();
                    reserved = false;
                }
                let reads = match grouped {
                    false => None,
                    true => {
                        let record = state
                            .records
                            .pop()
                            .ok_or("a group of grouped steps begun beyond the groups the queue holds in flight")?;
                        self.telemetry.clear_reads(record.0, launches.len());
                        Some(record)
                    }
                };
                let index = reads.as_ref().map(|&GroupRecord(index)| index);
                let merges = self
                    .telemetry
                    .merge_sets(self.priced().as_deref(), self.passes().as_deref(), self.notice(), launches.len())
                    .into_iter()
                    .map(|size| if scratch { size } else { launches.len() })
                    .collect::<Vec<_>>();
                // Known to the watcher before any of its steps' requests can be.
                let pieces = state.open(first, count, before);
                state.groups.insert(
                    first,
                    Begun {
                        steps: count,
                        pieces,
                        ended: 0,
                        begun: Instant::now(),
                        retired: false,
                        scratch,
                        merges: merges.clone(),
                        reads,
                        prompt,
                    },
                );
                for sequence in first..first + count {
                    state.slots[self.steps.slot(sequence)] = Some(NonZeroU64::new(sequence).expect("sequences begin at one"));
                }
                state.last = first + count - 1;
                state.running += launches.len();
                Ok(Some((first, index, merges)))
            })();
            match attempt {
                Ok(None) => match self.lent.await_copies(&self.life) {
                    Ok(()) => reserved = true,
                    Err(error) => break Err(error),
                },
                Ok(Some(begun)) => break Ok(begun),
                Err(error) => break Err(error),
            }
        };
        let (first, index, merges) = match begun {
            Ok(begun) => begun,
            Err(error) => {
                if reserved {
                    self.lent.release();
                }
                return Err((error, launches));
            }
        };
        // Known to serving before any of its steps' requests can be.
        self.steps.events.notify();
        // Each launch's step, in chain order from `first`. The admission
        // owns every sequence no step owns yet: should mapping a launch
        // unwind, it ends them, so none keeps its routing record.
        let mut admission = Admission { shared: self, next: first, remaining: launches.len() };
        let steps = launches.map(|launch| {
            let route = route(&launch);
            let work = work(&launch);
            let step = ExpertStep {
                // A decode step issues a site flight, through which one of
                // the executor's workers polls the site's requests: none
                // where no layer's split leaves the CPU features, nor while
                // every sited layer is whole, which asks the CPU nothing.
                _flight: (route == StepRoute::Decode && self.sites_served.load(std::sync::atomic::Ordering::Acquire))
                    .then(|| self.compute.flight())
                    .flatten(),
                // A decode step on its flight's stamp times its blocks by
                // the lines it begins under, once they are priced.
                sampled: (route == StepRoute::Decode && self.telemetry.samples(admission.next)).then(|| self.priced()).flatten(),
                route,
                work,
                reads: index,
                sequence: admission.next,
                shared: self.clone(),
                launch: None,
                recorded: None,
            };
            admission.remaining -= 1;
            admission.next += 1;
            (launch, step)
        });
        Ok((steps, merges))
    }
}

/// A step the watcher serves, from before its first launch is queued until
/// it ends: `complete`d once the device is past it, or dropped when nothing
/// will take its outputs. Either ends it, so the watcher stops serving when
/// no step runs.
#[must_use = "a step's completion is how its read failures reach its sequences"]
pub(crate) struct ExpertStep {
    sequence: Step,
    route: StepRoute,
    /// Its launch's work, a prompt's or generation's.
    work: LaunchWork,
    /// Its group's group reads record, of a group with a grouped step.
    reads: Option<usize>,
    shared: Arc<Shared>,
    /// Its decode launch's rows and context; none of a grouped step.
    launch: Option<DecodeLaunch>,
    recorded: Option<Recording>,
    /// The site flight a decode step issues while it runs, where the load
    /// has site work.
    _flight: Option<crate::experts::compute::Flight>,
    /// The lines a sampled decode step's blocks are timed by
    /// (`Telemetry::samples`), decided as it begins; none of a step not
    /// sampled.
    sampled: Option<Arc<[ExpertLines]>>,
}

/// A step whose load counts its decode (`Stack::live`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Recording;

/// A step completed: its launch's work, when its group began, and how its
/// route feeds it.
pub(super) struct CompletedStep {
    pub(super) work: LaunchWork,
    pub(super) begun: Instant,
    pub(super) route: CompletedRoute,
}

/// How a completed step is fed: a grouped step, whose reads its group's
/// passes recorded (fed when the group ends), or a decode step, fed from its
/// routing record.
pub(super) enum CompletedRoute {
    Grouped,
    Decode(Decoded),
}

/// A decode launch: its rows, and the mean tokens its rows' histories
/// read.
#[derive(Clone, Copy, Debug)]
pub(super) struct DecodeLaunch {
    pub(super) rows: usize,
    pub(super) tokens: u64,
}

/// A decode step completed: how its load records it, its launch, its
/// group, and the lines its blocks were
/// sampled by, where they were.
pub(super) struct Decoded {
    pub(super) recording: Option<Recording>,
    pub(super) launch: Option<DecodeLaunch>,
    pub(super) group: Step,
    pub(super) sampled: Option<Arc<[ExpertLines]>>,
}

impl ExpertStep {
    /// The sequence the step's runs set the step word to (`routed_step`).
    pub fn sequence(&self) -> Step {
        self.sequence
    }

    /// Whether the step's runs time their blocks, each whose ends are
    /// marked (`Telemetry::samples`).
    pub fn samples(&self) -> bool {
        self.sampled.is_some()
    }

    /// The group reads record its group's passes write their reads to
    /// (`BankPart::GroupReads`), of a group with a grouped step.
    pub fn group_reads(&self) -> Option<usize> {
        self.reads
    }

    /// The step's decode launch: its rows, at `positions`, counted at its
    /// rows' mean context (`Stack::live`), so rows of several sequences at
    /// their own depths take theirs, not the first row's.
    pub fn decoding(mut self, positions: impl ExactSizeIterator<Item = u64>) -> Self {
        let rows = positions.len();
        assert!(rows > 0, "a decode launch runs a row");
        self.launch = Some(DecodeLaunch { rows, tokens: positions.map(|position| position + 1).sum::<u64>() / rows as u64 });
        self
    }

    /// The step's load counts its decode.
    pub fn recorded(mut self, recording: Recording) -> Self {
        self.recorded = Some(recording);
        self
    }

    /// The device has completed the step: the watcher retires its fills,
    /// and why its requests failed, if one did, is returned. A step
    /// not [`Self::recorded`] (a load's
    /// warm-up's) is fed to the stack but not counted. Once the watcher has
    /// stopped serving every step completes with why.
    pub fn complete(mut self) -> Result<(), StepFailure> {
        let shared = &self.shared;
        let (group, begun) = shared.steps.begun(self.sequence);
        let route = match self.route {
            StepRoute::Grouped => CompletedRoute::Grouped,
            StepRoute::Decode => CompletedRoute::Decode(Decoded {
                recording: self.recorded.take(),
                launch: self.launch,
                group,
                sampled: self.sampled.take(),
            }),
        };
        let completion = CompletedStep { work: self.work, begun, route };
        let sequence = self.sequence;
        if let Err(error) = shared.telemetry.complete_step(
            sequence,
            completion,
            &|layer| shared.compute.sited(layer, sequence),
            shared.priced().as_deref(),
        ) {
            shared.failed_completion(error);
        }
        if let Some(reason) = shared.life.ended() {
            return Err(StepFailure::Watcher(reason));
        }
        let read = shared.steps.lock().failed.remove(&self.sequence);
        let both = |first: (FailedRows, String), second: (FailedRows, String)| {
            (first.0.and(second.0), format!("{}; {}", first.1, second.1))
        };
        let site = shared.compute.take_site_failure(self.sequence).map(|failure| {
            let layers = failure.rows.keys().map(usize::to_string).collect::<Vec<_>>().join(", ");
            let rows = failure
                .rows
                .iter()
                .flat_map(|(&layer, rows)| shared.steps.batch_rows(self.sequence, layer, rows))
                .collect();
            (
                FailedRows::Only(rows),
                format!(
                    "the CPU site failed step {}'s tier-served choices at streamed layers {layers} ({})",
                    self.sequence,
                    failure.reasons.join("; ")
                ),
            )
        });
        // A step a wait of which ended on its still pulse ran on without
        // that answer: it is interrupted, to be re-run.
        if self.expired() {
            shared.steps.interrupted.fetch_add(1, Ordering::AcqRel);
            return Err(StepFailure::Interrupted(format!(
                "a wait of step {} ended on its still pulse before its answer (the host's pulse beat late)",
                self.sequence
            )));
        }
        match (read, site) {
            (None, None) => Ok(()),
            (Some((rows, reason)), None) => Err(StepFailure::Read { rows, reason }),
            (None, Some((rows, reason))) => Err(StepFailure::Site { rows, reason }),
            (Some(read), Some(site)) => {
                let (rows, reason) = both(read, site);
                Err(StepFailure::Read { rows, reason })
            }
        }
    }

    /// The device is past the step, whose runs its chain's submission
    /// refused: nothing of it is fed (its records hold what no run of it
    /// wrote) and it ends uncompleted. Why the watcher stopped serving, if
    /// it did.
    pub fn refuse(self) -> Result<(), StepFailure> {
        match self.shared.life.ended() {
            Some(reason) => Err(StepFailure::Watcher(reason)),
            None => Ok(()),
        }
    }
}

impl Drop for ExpertStep {
    /// The step has ended: it is no longer in flight, and a failure of it
    /// no consumer took is forgotten. One a wait of which ended on its still
    /// pulse is tainted first, so its seeds are never mapped, whether or not
    /// anything took its outputs: the device is past it, as every exit
    /// drains its issued work first ([`Self::expired`]).
    fn drop(&mut self) {
        if self.expired() {
            self.shared.steps.lock().tainted.insert(self.sequence);
        }
        self.shared.end_step(self.sequence);
    }
}

impl ExpertStep {
    /// Whether a wait of the step ended on its still pulse before its
    /// answer (`Parks::expired`), read once the device is past the step: as
    /// it completes, or as it drops, by whichever exit; the device runs
    /// nothing of a step once it has ended, since its submission's owner
    /// drains the work it issued before releasing the step (this module's
    /// rule), and a step never queued requested nothing, so none of its
    /// waits stored an expiry.
    fn expired(&self) -> bool {
        self.shared.parks.expired(self.sequence)
    }
}

/// The sequences a group admitted that no [`ExpertStep`] owns yet, ended
/// should their mapping unwind.
struct Admission<'a> {
    shared: &'a Arc<Shared>,
    next: Step,
    remaining: usize,
}

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        for sequence in (self.next..).take(self.remaining) {
            self.shared.end_step(sequence);
        }
    }
}

impl Shared {
    /// Step `sequence` has ended. Submission owners drain issued work before
    /// releasing a step: its parks retire before its routing record is free
    /// for another sequence.
    fn end_step(&self, sequence: Step) {
        self.parks.retire(sequence);
        self.telemetry.end_step(sequence);
        // A group is fed only once its every step has ended, its passes'
        // reads with it: residency's thread feeds it before its next turn,
        // which is where a shrink between flights ranks its reads
        // (`resize::plan`), and before which this end's event wakes it.
        self.steps.end(sequence, |reads, ended| {
            if let Some(reads) = reads {
                self.telemetry.group_read(reads);
            }
            self.telemetry.ended(ended);
        });
    }
}

#[cfg(test)]
impl ExpertStep {
    /// A step whose request failed for `reason`, of steps shared with no
    /// watcher and no layer: what a failed expert read leaves a step with,
    /// for the domain's failure tests.
    pub fn failed(reason: &str) -> Self {
        let shared = Self::unbanked(FlightDepth::PIPELINED, 1);
        let begun = shared.begin(Chain::one(()), |_| StepRoute::Decode, |_| LaunchWork::Generation, || Ok(false));
        let ((_, step), _) = begun.map_err(|(error, _)| error).unwrap().0.split_first();
        shared.steps.fail(step.sequence, reason.to_owned());
        step
    }

    /// A watcher's shared state of no layer, of `places` chain places flying
    /// at `flights`, on the CPU, whose kernels park.
    fn unbanked(flights: FlightDepth, places: u64) -> Arc<Shared> {
        use super::{Extents, Life, ResizeHandoff, Telemetry};
        use crate::planning::RegionWords;
        let words = RegionWords {
            choices: 0,
            rounds: 1,
            flight: flights.steps(places),
            sites: flights.steps(1),
        };
        let extents = Extents {
            widest: 0,
            round_records: 1,
            words,
            selected: 1,
            demand: 1,
            reserve: 0,
            placer: crate::planning::Placer::Tier,
        };
        let steps = Arc::new(Steps::new(flights, places, crate::experts::exchange::Pulse::new()));
        let events = steps.events().clone();
        Arc::new(Shared {
            steps,
            telemetry: Telemetry::new(
                &[],
                super::telemetry::GroupReadWords { layers: Vec::new(), places: 1 },
                extents,
                Arc::new(std::sync::atomic::AtomicU64::new(0)),
                Vec::new(),
                None,
                crate::experts::reuse::Reuse::ranking(&[], &[], Vec::new()),
                Vec::new(),
                false,
            ),
            lent: super::lent::LentRecords::new(),
            fills: std::sync::atomic::AtomicU64::new(0),
            resizing: ResizeHandoff::new(),
            traffic_held: Default::default(),
            traffic_quiet: Default::default(),
            backing: Default::default(),
            joins: super::Joins::new(),
            life: Life::new(),
            // An executor of no thread and no site layer.
            sites_served: std::sync::atomic::AtomicBool::new(true),
            compute: crate::experts::HostCompute::start(Vec::new(), &crate::experts::ComputeThreads::unplaced(Vec::new()))
                .unwrap()
                .handle(),
            // Parks of no layer, on the CPU's timeline: kernels wait in
            // place, so nothing registers.
            parks: {
                let cpu = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu)
                    .unwrap();
                Arc::new(super::Parks::new(seismic::Timeline::new(&cpu).unwrap(), Vec::new(), events))
            },
            landings: Default::default(),
            priced: std::sync::Mutex::new(None),
            passes: std::sync::Mutex::new(None),
            settled: Default::default(),
            promoted: Default::default(),
            copy: std::sync::Mutex::new(None),
        })
    }
}

#[cfg(test)]
mod slot_tests {
    use super::*;

    fn begin(shared: &Arc<Shared>, labels: &[usize]) -> Vec<ExpertStep> {
        let mut chain = Chain::one(labels[0]);
        for &label in &labels[1..] { chain.push(label); }
        shared.begin(chain, |_| StepRoute::Decode, |_| LaunchWork::Generation, || Ok(false))
            .map_err(|(error, _)| error).unwrap().0.into_iter().map(|(_, step)| step).collect()
    }

    /// The load's traffic is held while any hold stands: each holder
    /// releases only its own, and one that panics releases its own as it
    /// unwinds.
    #[test]
    fn each_traffic_hold_releases_only_its_own() {
        let shared = ExpertStep::unbanked(FlightDepth::SERIAL, 2);
        let (claim, tool) = (shared.hold_traffic(), shared.hold_traffic());
        drop(claim);
        assert!(shared.traffic_held(), "a second holder's hold stands");
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _panicking = shared.hold_traffic();
            panic!("a holder panics");
        }));
        assert!(unwound.is_err());
        assert!(shared.traffic_held(), "the panicking holder released only its own");
        drop(tool);
        assert!(!shared.traffic_held(), "every hold released");
    }

    /// A request that failed names its rows within the window its layer is
    /// served over: from the windowed layer on, its step's failed batch rows
    /// are the window's from its first row; before it, the rows themselves.
    #[test]
    fn a_failed_requests_rows_are_its_steps_batch_rows() {
        let shared = ExpertStep::unbanked(FlightDepth::SERIAL, 2);
        let step = begin(&shared, &[0]).pop().unwrap();
        let sequence = step.sequence();
        shared.window(sequence, 2, 4);
        let rows = |rows: &[usize]| rows.iter().copied().collect::<BTreeSet<_>>();
        shared.steps.fail_request_rows(sequence, 1, &rows(&[0]), "before the window".into());
        shared.steps.fail_request_rows(sequence, 3, &rows(&[0, 1]), "in the window".into());
        let failed = shared.steps.lock().failed.get(&sequence).cloned();
        assert_eq!(failed, Some((FailedRows::Only(rows(&[0, 4, 5])), "before the window".to_owned())));
    }

    /// A CPU site failure names its request's rows, which its step's window
    /// at that layer places in the batch as a file read's are: from the
    /// windowed layer on, the window's from its first row; before it, the
    /// rows themselves.
    #[test]
    fn a_site_failures_rows_are_its_steps_batch_rows() {
        let shared = ExpertStep::unbanked(FlightDepth::SERIAL, 2);
        let step = begin(&shared, &[0]).pop().unwrap();
        let sequence = step.sequence();
        shared.window(sequence, 2, 4);
        let rows = |rows: &[usize]| rows.iter().copied().collect::<BTreeSet<_>>();
        shared.compute.fail_site(sequence, 1, rows(&[0]));
        shared.compute.fail_site(sequence, 3, rows(&[0, 1]));
        match step.complete() {
            Err(StepFailure::Site { rows: failed, .. }) => assert_eq!(failed, FailedRows::Only(rows(&[0, 4, 5]))),
            outcome => panic!("a step its site failed completed {:?}", outcome.map(|_| ())),
        }
    }

    /// A step still running keeps its routing record however many later
    /// steps end first: a sequence landing on it is refused, untouched.
    #[test]
    fn routing_records_stay_owned_without_fifo_completion() {
        for flight in [1, 2, 5] {
            let shared = ExpertStep::unbanked(FlightDepth::SERIAL, flight as u64);
            let mut initial = begin(&shared, &(0..flight).collect::<Vec<_>>());
            let live = initial.pop().unwrap();
            drop(initial);
            for label in 0..flight - 1 {
                drop(begin(&shared, &[label]));
            }
            let (last, running, slots) = {
                let state = shared.steps.lock();
                (state.last, state.running, state.slots.clone())
            };
            for _ in 0..3 {
                let refused = shared.begin(Chain::one(91usize), |_| StepRoute::Decode, |_| LaunchWork::Generation,
                    || panic!("a refused admission prepares nothing"));
                let (error, labels) = match refused { Err(refused) => refused, Ok(_) => panic!("a live record overwritten") };
                assert!(error.contains("routing record"), "{error}");
                assert_eq!(labels.into_iter().collect::<Vec<_>>(), [91]);
                let state = shared.steps.lock();
                assert_eq!((state.last, state.running, &state.slots), (last, running, &slots));
            }
            drop(live);
            assert_eq!(shared.steps.in_flight(), 0);
            drop(begin(&shared, &[92]));
            assert_eq!(shared.steps.lock().last, last + 1);
        }
    }

    /// The frontier is the least step not yet ended, whatever the exit: a
    /// later step ending first moves it nowhere, and the earlier one's drop
    /// moves it past both.
    #[test]
    fn the_frontier_is_the_least_step_not_yet_ended() {
        let shared = ExpertStep::unbanked(FlightDepth::SERIAL, 4);
        let mut steps = begin(&shared, &[0, 1, 2]).into_iter();
        let (first, second, third) = (steps.next().unwrap(), steps.next().unwrap(), steps.next().unwrap());
        let start = first.sequence();
        assert_eq!(shared.steps.frontier(), start);
        third.complete().unwrap();
        second.refuse().unwrap();
        assert_eq!(shared.steps.frontier(), start, "later steps ended out of order");
        assert!(!shared.steps.ended(start) && shared.steps.ended(start + 2));
        drop(first);
        assert_eq!(shared.steps.frontier(), start + 3, "a dropped step ends too");
    }

    /// A word that carries a step's low word names the step in flight with
    /// it, across a 2^32 boundary, and none once that step has ended.
    #[test]
    fn a_low_word_names_its_step_in_flight() {
        let shared = ExpertStep::unbanked(FlightDepth::SERIAL, 4);
        {
            let mut state = shared.steps.lock();
            state.last = (1 << 32) - 2;
            state.frontier = (1 << 32) - 1;
        }
        let steps = begin(&shared, &[0, 1]);
        let (below, above) = (steps[0].sequence(), steps[1].sequence());
        assert_eq!((below, above), ((1 << 32) - 1, 1 << 32));
        assert_eq!(shared.steps.of_low_word(low_word(below)), Some(below));
        assert_eq!(shared.steps.of_low_word(low_word(above)), Some(above));
        assert_eq!(shared.steps.of_low_word(low_word(above + 1)), None, "not begun");
        drop(steps);
        assert_eq!(shared.steps.of_low_word(low_word(below)), None, "ended");
    }

    /// Mapping a launch that unwinds ends every admitted sequence.
    #[test]
    fn an_unwinding_admission_frees_its_routing_records() {
        let shared = ExpertStep::unbanked(FlightDepth::SERIAL, 3);
        let calls = std::cell::Cell::new(0);
        let mut launches = Chain::one(0usize);
        launches.push(1);
        launches.push(2);
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = shared.begin(launches, |_| {
                calls.set(calls.get() + 1);
                // The grouped scan routes all three; then the second mapping panics.
                if calls.get() == 5 { panic!("second step mapping"); }
                StepRoute::Decode
            }, |_| LaunchWork::Generation, || Ok(false));
        }));
        assert!(unwind.is_err());
        assert_eq!(shared.steps.in_flight(), 0);
        assert!(shared.steps.lock().slots.iter().all(Option::is_none));
        drop(begin(&shared, &[99]));
    }

    /// The sequence after the last one begun, which retirement names, stays
    /// representable.
    #[test]
    fn admission_keeps_the_retirement_watermark_representable() {
        let shared = ExpertStep::unbanked(FlightDepth::SERIAL, 1);
        shared.steps.lock().last = Step::MAX - 2;
        drop(begin(&shared, &[1]));
        assert_eq!(shared.steps.lock().last, Step::MAX - 1);
        let (reason, launches) = match shared.begin(Chain::one(2usize), |_| StepRoute::Decode, |_| LaunchWork::Generation,
            || panic!("sequence refusal precedes preparation")) {
            Err(error) => error, Ok(_) => panic!("an unrepresentable watermark admitted"),
        };
        assert!(reason.contains("exhausted"));
        assert_eq!(launches.into_iter().collect::<Vec<_>>(), [2]);
    }

    /// Begin a prompt group of one step, `scratch`-served, and end it: the
    /// steps it began, or why it could not begin.
    fn prompt(shared: &Arc<Shared>, scratch: bool) -> Result<usize, String> {
        shared
            .begin(Chain::one(0usize), |_| StepRoute::Grouped, |_| LaunchWork::Prompt, || Ok(scratch))
            .map(|(steps, _)| steps.len())
            .map_err(|(error, _)| error)
    }

    /// Spin until `ready`.
    fn spin(what: &str, ready: impl Fn() -> bool) {
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while !ready() {
            assert!(Instant::now() < deadline, "{what}");
            std::thread::yield_now();
        }
    }

    /// A scratch-served group that finds a copy into the lent records
    /// running waits to begin; serving's end, after which no copy
    /// completes, ends the wait, and the group cannot begin, for its reason.
    #[test]
    fn a_begin_waiting_for_a_copy_returns_once_serving_ends() {
        let shared = ExpertStep::unbanked(FlightDepth::SERIAL, 2);
        shared.lent.lock().copying();
        let waiting = {
            let shared = shared.clone();
            std::thread::spawn(move || prompt(&shared, true))
        };
        spin("the group never waited", || shared.lent.lock().reserved());
        shared.end_serving("the request service panicked".into());
        assert_eq!(waiting.join().unwrap(), Err("the request service panicked".to_owned()));
        assert!(!shared.lent.lock().reserved(), "an ended wait holds nothing");
        assert_eq!(shared.steps.in_flight(), 0, "no step began");
    }

    /// While a scratch-served group waits for a copy into the lent records,
    /// they count as reserved, so no promotion queues another copy into
    /// them; once the copy running completes, the group begins holding
    /// them. A group that serves no scratch reserves nothing and waits for
    /// no copy.
    #[test]
    fn only_a_scratch_served_group_reserves_and_waits_for_the_copy_running() {
        let shared = ExpertStep::unbanked(FlightDepth::SERIAL, 4);
        shared.lent.lock().copying();
        assert_eq!(prompt(&shared, false), Ok(1), "a group serving no scratch begins at once");
        assert!(!shared.lent.lock().reserved(), "and reserves nothing");
        let waiting = {
            let shared = shared.clone();
            std::thread::spawn(move || prompt(&shared, true))
        };
        spin("the group never waited", || shared.lent.lock().reserved());
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!waiting.is_finished(), "a group began beside a copy into its lent records");
        shared.lent.copied();
        assert_eq!(waiting.join().unwrap(), Ok(1));
        assert!(shared.lent.lock().reserved(), "the group holds the lent records until it retires");
    }
}

#[cfg(test)]
mod order_tests {
    use super::*;

    /// The streamed layers of a test's bank.
    const LAYERS: usize = 4;

    fn steps() -> Steps {
        Steps::new(FlightDepth::SERIAL, 4, crate::experts::exchange::Pulse::new())
    }

    /// A chain of steps 1 and 2, and a decode step 3 cut in before its
    /// layer 2: the chain's layers 0 and 1, then its fronts at layer 2,
    /// then step 3's every layer, then the chain's passes from layer 2 on.
    fn cut() -> Steps {
        let steps = steps();
        assert_eq!(steps.begin_group(2, None), 1);
        assert_eq!(steps.begin_group(1, Some((1, 2))), 3);
        steps
    }

    /// A group cut in before a layer of another runs between that group's
    /// passes before the layer and those from it on, its fronts there
    /// before it; its runs are counted between them in the device's order,
    /// and a group begun after both follows both.
    #[test]
    fn a_group_cut_in_runs_between_the_layers_it_is_cut_before() {
        let steps = cut();
        let decode = |layer| steps.point(3, layer, 3);
        assert!(steps.point(1, 1, 2) < decode(0), "the chain's layer 1 runs before");
        assert!(decode(LAYERS - 1) < steps.point(1, 2, 1), "its layer 2's passes after");
        assert!(steps.at(2, 2, LAYERS).point < decode(0), "its fronts at layer 2 before");
        assert!(steps.at(1, 3, LAYERS).point > decode(LAYERS - 1), "its fronts past layer 2 after");
        assert!(steps.after(3) < steps.after(1), "the chain ends past the step cut in");
        // Every request's position follows the device's order, the
        // chain's passes at layer 2 (which request nothing) between the
        // step cut in and its fronts at layer 3.
        let order = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3), (1, 3), (2, 3)];
        let positions = order.map(|(sequence, layer)| steps.at(sequence, layer, LAYERS));
        let runs = positions.windows(2).map(|pair| pair[0].runs_until(pair[1])).collect::<Vec<_>>();
        assert_eq!(runs, [1, 1, 1, 1, 1, 1, 1, 3, 1]);
        // A step begun after both follows the chain's last layer.
        let next = steps.begin_group(1, None);
        assert!(steps.point(next, 0, next) > steps.after(1));
        assert!(steps.at(2, 3, LAYERS).runs_until(steps.at(next, 0, LAYERS)) >= 1);
    }

    /// A second group cut in at the same cut runs after the first, the
    /// cut layers behind both.
    #[test]
    fn a_second_group_cut_in_at_one_cut_runs_after_the_first() {
        let steps = cut();
        assert_eq!(steps.begin_group(1, Some((1, 2))), 4);
        assert!(steps.after(3) < steps.point(4, 0, 4));
        assert!(steps.after(4) < steps.point(1, 2, 1));
        assert!(steps.point(1, 1, 2) < steps.point(3, 0, 3));
        assert_eq!(steps.at(3, 3, LAYERS).runs_until(steps.at(4, 0, LAYERS)), 1);
        assert_eq!(steps.at(4, 3, LAYERS).runs_until(steps.at(1, 3, LAYERS)), 3, "past the chain's passes at layer 2");
    }

    /// A park reads every step's resolution from the order: the step cut
    /// in, parked, finds the chain it runs inside unresolved at the layers
    /// past the cut, and every later resolution its own group's or later;
    /// the chain parked before the cut finds the step cut in after its
    /// every layer.
    #[test]
    fn a_park_reads_the_order_of_every_group() {
        let steps = cut();
        let park = steps.park(steps.at(3, 3, LAYERS).point, LAYERS);
        assert!(park.cut_in());
        assert_eq!(park.earlier, [None, None, Some((1, 2)), Some((1, 2))]);
        assert_eq!(park.unresolved, [4; LAYERS], "the step cut in resolved every layer to its own");
        assert!(!park.frees(2, 0, Step::MAX) && park.frees(2, 3, Step::MAX) && park.frees(1, 0, Step::MAX));
        let chain = steps.park(steps.at(1, 1, LAYERS).point, LAYERS);
        assert!(!chain.cut_in());
        assert_eq!(chain.unresolved, [3, 2, 1, 1], "the step cut in, the chain's next, its first, its first");
        let last = steps.park(steps.at(2, 1, LAYERS).point, LAYERS);
        assert_eq!(last.unresolved, [3, 3, 1, 1]);
    }

    /// A record is served once each step it could serve is past its layer
    /// or ended: the step cut in needs no wait for the chain at a layer
    /// before the cut, nor for the chain's steps below its FROM; the chain's
    /// steps past the cut are past it only once their passes are, or once
    /// they end.
    #[test]
    fn a_record_is_served_once_its_steps_pass_its_layer_or_end() {
        let steps = cut();
        let decoded = steps.after(3);
        assert!(steps.served(1, 4, 1, decoded), "the chain passed layer 1 before the step cut in");
        assert!(!steps.served(1, 4, 2, decoded), "its layer 2 runs after");
        assert!(steps.served(3, 4, 2, decoded), "no chain step is mapped from 3");
        assert!(steps.served(1, 4, 2, steps.point(1, 3, 1)));
        steps.end_played(1);
        steps.end_played(2);
        assert!(steps.served(1, 4, 2, decoded), "ended, the chain reads it no more");
        assert_eq!(steps.first_unpassed(decoded), 1, "while running, the chain was not past");
    }

    /// The first group the device is not past is the chain it runs inside,
    /// before the step cut in; with nothing cut in, its own.
    #[test]
    fn the_first_unpassed_group_is_the_one_run_inside() {
        let steps = cut();
        assert_eq!(steps.first_unpassed(steps.point(3, 2, 3)), 1);
        let whole = self::steps();
        whole.begin_through(4);
        assert_eq!(whole.first_unpassed(Point::whole(3, 1, 3)), 3);
        assert_eq!(whole.first_unpassed(whole.after(3)), 3);
    }

    /// Retiring both, the point past them is the chain's, past the step
    /// cut in.
    #[test]
    fn retiring_past_a_cut_names_the_latest_in_the_order() {
        let steps = cut();
        for step in 1..=3 {
            steps.end_played(step);
        }
        let chain = steps.after(1);
        assert_eq!(steps.retire_groups(4), (Some(chain), 0));
        assert_eq!(steps.after(1), Point::START, "a group forgotten is past");
    }

    /// Only the latest piece submitted is cut: a group begun since runs
    /// after it whole, so no group cuts in before it.
    #[test]
    #[should_panic(expected = "is not the latest submitted")]
    fn a_cut_behind_a_later_group_panics() {
        let steps = steps();
        steps.begin_group(2, None);
        steps.begin_group(1, None);
        steps.begin_group(1, Some((1, 2)));
    }

    /// No cut lies before a layer already cut past.
    #[test]
    #[should_panic(expected = "whose last piece runs from layer 2")]
    fn a_cut_before_an_earlier_cut_panics() {
        let steps = cut();
        steps.begin_group(1, Some((1, 1)));
    }
}
