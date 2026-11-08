//! The steps the watcher serves, from their beginning in a group until
//! each ends, completed or dropped, and why one failed.
//!
//! Steps end in any order, by any exit: completed, refused, dropped, or
//! never queued (an admission that unwinds). Whatever the exit, the device
//! runs nothing of a step once it has ended (submission owners drain issued
//! work before releasing a step). The one completion fact every reuse of a
//! record rests on is the frontier ([`Steps::frontier`]): the least step not
//! yet ended, below which every step has ended. Every exit advances it.

use super::Shared;
use super::core::At;
use crate::assessment::ExpertLines;
use crate::experts::directory::Point;
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

/// A group of steps begun: its steps, those ended, when it began, and
/// whether the watcher has retired it (the device is past it).
struct Begun {
    steps: u64,
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

impl State {
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

    /// Layer `layer` of step `sequence` in the device's order, of a bank of
    /// `layers` layers.
    pub(super) fn at(&self, sequence: Step, layer: usize, layers: usize) -> At {
        let (group, steps) = self.group_of(sequence);
        At::new(Point { group, layer, sequence }, steps, layers)
    }

    /// Retire the groups whose every step is below `frontier`: the device
    /// is past them. The latest of them still known, by its first sequence,
    /// and how many scratch-served groups retired now (each held the lent
    /// records' reservation, `lent`).
    pub(super) fn retire_groups(&self, frontier: Step) -> (Option<Step>, usize) {
        let mut state = self.lock();
        let (mut latest, mut reserving) = (None, 0);
        for (&first, group) in &mut state.groups {
            if first + group.steps <= frontier {
                reserving += usize::from(!group.retired && group.scratch);
                group.retired = true;
                latest = Some(first);
            }
        }
        state.forget();
        (latest, reserving)
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

    /// `write` given the last step begun and the group it began in,
    /// `(group, sequence)` (a group forgotten, retired and ended, names
    /// itself by the step, a point no earlier than its own), and the first
    /// step not yet begun, under the lock every step begins under: none
    /// begins until it returns.
    pub(super) fn while_none_begins<T>(&self, write: impl FnOnce((Step, Step), &Unbegun) -> T) -> T {
        let state = self.lock();
        let last = state.last;
        let first = state
            .groups
            .range(..=last)
            .next_back()
            .filter(|(first, group)| last - **first < group.steps)
            .map_or(last, |(&first, _)| first);
        write((first, last), &Unbegun(last + 1))
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

    /// Steps through `last` begun, where a test plays them without
    /// launches: what an entry stored from the first step not yet begun is
    /// stored from.
    #[cfg(test)]
    pub(in crate::experts) fn begin_through(&self, last: Step) {
        self.lock().last = last;
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
    /// cannot begin comes with the launches.
    pub fn begin<L>(
        self: &Arc<Self>,
        launches: Chain<L>,
        route: impl Fn(&L) -> StepRoute,
        work: impl Fn(&L) -> LaunchWork,
        mut scratch: impl FnMut() -> Result<bool, String>,
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
                state.groups.insert(
                    first,
                    Begun {
                        steps: count,
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
