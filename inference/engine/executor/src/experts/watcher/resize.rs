//! The bank's resizes. A shrink runs between flights: the owner asks, and
//! residency answers once nothing moves a record (its turns wait on the
//! load's events, never on a time), or at once for a shrink of the records
//! the host lands into, which keeps every record in flight where it is. A
//! bank's shrink is planned with its evictions published at once
//! (always safe: the bank only loses experts), and answered with a plan the
//! owner commits once its moves are queued on the device ahead of every
//! later step, or drops. The one store is also cut beside a running chain
//! ([`Shared::cut_running`]), moving nothing, when memory runs short while
//! a long chain is encoded. A growth asks nothing: each of its slots joins
//! as soon as every record tensor backs it ([`Shared::join`]), beside
//! running steps, which reach a record only once the host maps it. Each
//! carries the bank's slots, which the watcher's telemetry reads.

use super::{Shared, Step};
use super::core::{Core, PlacedWords};
use super::residency::Residency;
use super::telemetry::GroupReadWords;
use crate::experts::directory::Shrink;
use crate::experts::reuse::Reuse;
use crate::experts::source::Records;
use std::sync::{Condvar, Mutex};
use crate::domain::Trip;
use std::time::Instant;

/// A shrink the watcher planned between flights, its evictions published
/// ([`Shared::plan_shrink`]): what the owner commits once its moves are
/// queued on the device ahead of every later step
/// ([`Shared::commit_shrink`]), or drops, which leaves every kept expert
/// where its bytes are. It holds the records it moves into or retires
/// (`BankRecords::shrinking`) while it lives: dropped uncommitted, it
/// returns them (`BankRecords::unshrink`), so no path leaves them held.
#[must_use]
pub(crate) struct ShrinkPlan {
    slots: usize,
    groups: Vec<Shrink>,
    trip: Trip,
    /// The watcher it returns its records to, until it commits.
    holds: Option<std::sync::Arc<Shared>>,
}

impl Drop for ShrinkPlan {
    fn drop(&mut self) {
        if let Some(shared) = self.holds.take() {
            shared.resizing.abandoned.store(true, std::sync::atomic::Ordering::Release);
            shared.events().notify();
        }
    }
}

impl ShrinkPlan {
    /// The slots the plan commits the bank to.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// The plan's trip through the watcher.
    pub fn trip(&self) -> Trip {
        self.trip
    }

    /// Each group's moved records, (from, to).
    pub fn moves(&self) -> impl Iterator<Item = Vec<(usize, usize)>> + '_ {
        self.groups
            .iter()
            .map(|plan| plan.moved.iter().map(|&(.., from, to)| (from, to)).collect())
    }
}

/// A shrink's candidate commitments: from `shallowest` slots down to
/// `deepest` (the shrink's target), each group's records at a candidate's
/// slots formed by `records` only for the candidates the watcher weighs.
pub(crate) struct Candidates {
    pub shallowest: usize,
    pub deepest: usize,
    pub records: Box<dyn Fn(usize) -> Vec<usize> + Send>,
}

impl Candidates {
    /// One candidate: `records` at `slots`, the shrink's target.
    #[cfg(test)]
    pub(crate) fn one(slots: usize, records: Vec<usize>) -> Self {
        Self {
            shallowest: slots,
            deepest: slots,
            records: Box::new(move |_| records.clone()),
        }
    }

    fn count(&self) -> usize {
        self.shallowest - self.deepest + 1
    }

    /// The `index`th candidate, shallowest first: its slots.
    fn slots(&self, index: usize) -> usize {
        self.shallowest - index
    }
}

/// What the owner asks of the watcher between flights to resize the bank.
pub(super) enum Ask {
    /// Plan each group's commitment at `records` (the bank at `slots`), and
    /// evict what it evicts.
    Shrink { slots: usize, records: Vec<usize> },
    /// Take a planned shrink's moves, which the device carries.
    Commit(ShrinkPlan),
    /// Shrink each group's records the host lands into at the deepest of
    /// `candidates` that keeps every record in flight, its kept experts
    /// moved below the cut by `keeping`; its tier tensors then address the
    /// records below the cut alone.
    ShrinkHost { candidates: Candidates, keeping: Keeping },
    /// Serve the places the bank holds from now by their words.
    Place(PlacedWords),
}

/// A growth's change, which the watcher's directory applies at its next
/// pass in the order posted ([`Shared::join`]): one slot, backed by every
/// record tensor of its holding; or, of the records the host lands into,
/// a growth begun or ended.
pub(crate) enum Join {
    /// Each group of the bank holds `records` records, beside a host tier:
    /// the directory takes the new ones, free, and promotes into them.
    Bank { records: Vec<usize> },
    /// Each group's records the host lands into (its host tier's, or the
    /// one store's where the host places) are `records`, addressed by
    /// `views` (each group's record tensors' records committed at them):
    /// the directory takes the new ones, ready.
    Landed { records: Vec<u32>, views: Vec<Vec<Records>> },
    /// A growth of the records the host lands into begins: posted as it is
    /// decided, before any of its slots, so the tier is filling from then
    /// (`Directory::fill_pending`).
    Growing,
    /// That growth ended, after its last slot joined: no more join.
    Grown,
}

/// The slots a running growth has joined, in order, until the watcher
/// takes them, and how many it took and has not applied yet.
pub(super) struct Joins(Mutex<JoinQueue>);

#[derive(Default)]
struct JoinQueue {
    posted: Vec<Join>,
    applying: usize,
}

impl Joins {
    pub(super) fn new() -> Self {
        Self(Mutex::new(JoinQueue::default()))
    }

    fn queue(&self) -> std::sync::MutexGuard<'_, JoinQueue> {
        self.0.lock().expect("watcher joins")
    }

    fn post(&self, join: Join) {
        self.queue().posted.push(join);
    }

    /// The joins posted since the last take, in order, applying until
    /// [`Self::applied`].
    pub(super) fn take(&self) -> Vec<Join> {
        let mut queue = self.queue();
        let taken = std::mem::take(&mut queue.posted);
        queue.applying += taken.len();
        taken
    }

    /// `count` joins taken are applied: the directory and telemetry hold
    /// what they joined.
    fn applied(&self, count: usize) {
        self.queue().applying -= count;
    }

    /// Wait until serving applied every join posted: any request published
    /// from then on is served beside what they joined (the request
    /// service's pass applies the joins before it reads a request).
    #[cfg(test)]
    pub(super) fn wait_taken(&self) {
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while {
            let queue = self.queue();
            !queue.posted.is_empty() || queue.applying > 0
        } {
            assert!(Instant::now() < deadline, "the watcher applies a join within its rest");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

impl Ask {
    /// Whether it is served only once nothing moves a record: every resize
    /// but a shrink of the records the host lands into, which keeps each
    /// record in flight where it is (`HostRecords::shrink`), so it is served
    /// as it is asked.
    fn waits(&self) -> bool {
        !matches!(self, Self::ShrinkHost { .. })
    }
}

enum Answer {
    Planned(ShrinkPlan),
    Done(Trip),
    /// A shrink of the records the host lands into: cut, none where even
    /// its shallowest candidate cuts below a record in flight, or refused.
    Cut(Result<Option<Cut>, String>),
}

/// Where a shrink of the records the host lands into cut them: the slots it
/// commits them to, and its trip through the watcher.
pub(crate) struct HostCut {
    pub slots: usize,
    pub trip: Trip,
}

/// A shrink of the records the host lands into, cut, by its [`Keeping`].
enum Cut {
    /// Its kept records' bytes copied on the host.
    Copied(HostCut),
    /// Each group's moves of its kept records, (from, to), for the device
    /// to carry.
    Kept(HostCut, KeptMoves),
}

/// Who moves the kept experts of a shrink of the records the host lands
/// into below its cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Keeping {
    /// The host copies them before it answers: a host tier, in host RAM,
    /// which no device work reaches.
    Host,
    /// The device moves them, in queue order ahead of every later step,
    /// as the answer names them: the one store, where the host places, so
    /// the owner's thread only queues them. Every host write of a record
    /// the moves write follows the device past them: the records then
    /// ready are none of theirs, a record unmapped later frees only as the
    /// device's frontier passes a step queued after them or it parks in
    /// one, and a landing a step asks for follows the step.
    Device,
}

/// Each group's moves of a shrink's kept records, (from, to), for the device
/// to carry ([`Keeping::Device`]).
pub(crate) type KeptMoves = Vec<Vec<(u32, u32)>>;

enum Resize {
    None,
    /// Asked, at that instant.
    Asked(Ask, Instant),
    Answered(Answer),
}

/// A cut of the one store beside a running chain ([`Shared::cut_running`]),
/// which the owner never waits for: asked, then taken by residency, then
/// answered, each read by the owner as it encodes the chain or at a later
/// boundary.
enum Running {
    /// None asked, or residency serving one.
    None,
    /// Asked at that instant beside the chain whose steps run from `first`
    /// through `last`, at the deepest of `candidates` it may cut.
    Asked { first: Step, last: Step, candidates: Candidates, at: Instant },
    Answered(Result<Option<RunningCut>, String>),
}

/// A cut of the one store beside a running chain, answered
/// ([`Shared::cut_running`]): the slots it commits the store to, the
/// highest streamed layer whose hits the chain's steps had decided at the
/// cut (none where they had decided none), and its trip through the
/// watcher. No run the chain submits after every pass of that layer
/// resolves a record at or past the cut.
pub(crate) struct RunningCut {
    pub slots: usize,
    pub decided: Option<usize>,
    pub trip: Trip,
}

/// The resize the owner and the watcher exchange.
pub(super) struct ResizeHandoff {
    resize: Mutex<Resize>,
    resized: Condvar,
    /// The cut beside a running chain, apart from the resizes asked
    /// between flights, which wait on it never.
    running: Mutex<Running>,
    /// Whether a plan was dropped uncommitted since residency last looked
    /// ([`ShrinkPlan`]'s drop): its records return.
    abandoned: std::sync::atomic::AtomicBool,
}

impl ResizeHandoff {
    pub(super) fn new() -> Self {
        Self {
            resize: Mutex::new(Resize::None),
            resized: Condvar::new(),
            running: Mutex::new(Running::None),
            abandoned: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn running(&self) -> std::sync::MutexGuard<'_, Running> {
        self.running.lock().expect("watcher running cut")
    }

    /// The cut beside a running chain asked, taken.
    fn running_asked(&self) -> Option<(Step, Step, Candidates, Instant)> {
        let mut running = self.running();
        match std::mem::replace(&mut *running, Running::None) {
            Running::Asked { first, last, candidates, at } => Some((first, last, candidates, at)),
            other => {
                *running = other;
                None
            }
        }
    }

    /// What the owner asks of a resize, and when it asked, taken.
    pub(super) fn asked(&self) -> Option<(Ask, Instant)> {
        let mut resize = self.resize.lock().expect("watcher resize");
        match std::mem::replace(&mut *resize, Resize::None) {
            Resize::Asked(asked, at) => Some((asked, at)),
            other => {
                *resize = other;
                None
            }
        }
    }

    /// Answer the owner, once what the resize changed is published
    /// ([`Residency::publish_resized`]) and what each layer holds ([`Owed`]).
    fn answer(&self, answer: Answer, _: Published) {
        let mut resize = self.resize.lock().expect("watcher resize");
        *resize = Resize::Answered(answer);
        self.resized.notify_all();
    }

    /// Wake an owner waiting for an answer: serving ended. Taken under the
    /// resize's lock, so an owner that found serving going on is already
    /// waiting.
    pub(super) fn wake(&self) {
        let _resize = self.resize.lock().expect("watcher resize");
        self.resized.notify_all();
    }
}

impl Shared {
    /// Plan the bank's commitment at `slots`, each group's `records` there,
    /// between flights: every begun step ended and the device past it. The watcher retires every fill (the conveyor returns
    /// the lender's records), plans each shrinking group to keep its most
    /// recently read experts below its new commitment
    /// (`Directory::plan_shrink`), and evicts the rest: their slot map
    /// entries and directory records are cleared now.
    pub fn plan_shrink(&self, slots: usize, records: Vec<usize>) -> Result<ShrinkPlan, String> {
        match self.ask(Ask::Shrink { slots, records })? {
            Answer::Planned(plan) => Ok(plan),
            Answer::Done(_) | Answer::Cut(_) => unreachable!("a shrink is answered with its plan"),
        }
    }

    /// Commit `plan` once its moves are queued on the device ahead of every
    /// later step, still between flights: each moved expert's map entry
    /// follows it, and each group's records end at its new commitment.
    pub fn commit_shrink(&self, plan: ShrinkPlan) -> Result<Trip, String> {
        match self.ask(Ask::Commit(plan))? {
            Answer::Done(trip) => Ok(trip),
            Answer::Planned(_) | Answer::Cut(_) => unreachable!("a commit is answered done"),
        }
    }

    /// A growth's next slot is backed by every record tensor of its
    /// holding, beside running steps or not: residency takes it at its next
    /// turn, or the request service at its next pass, whichever is first
    /// (`Residency::join`), before either promotes, maps, lands or hints,
    /// and before any shrink asked after it is served. Nothing reaches its
    /// records until then: the device reaches a record only once the host
    /// maps it.
    pub(in crate::experts) fn join(&self, join: Join) {
        self.joins.post(join);
        self.events().notify();
    }

    /// Shrink each group's host tier, between flights, at the deepest of
    /// `candidates` whose cut keeps every record in flight below it
    /// ([`Self::cut_host`]), its kept records' bytes copied below the cut
    /// before it answers. None where every candidate cuts below a record in
    /// flight.
    pub(in crate::experts) fn shrink_tier(&self, candidates: Candidates) -> Result<Option<HostCut>, String> {
        match self.cut_host(candidates, Keeping::Host)? {
            Some(Cut::Copied(cut)) => Ok(Some(cut)),
            None => Ok(None),
            Some(Cut::Kept(..)) => unreachable!("a host tier's shrink is copied on the host"),
        }
    }

    /// Shrink the one store (where the host places), between flights, at
    /// the deepest of `candidates` whose cut keeps every record in flight
    /// below it ([`Self::cut_host`]): answered with each group's moves of
    /// its kept records, which the device carries (`Keeping::Device`). None
    /// where every candidate cuts below a record in flight.
    pub(in crate::experts) fn shrink_store(&self, candidates: Candidates) -> Result<Option<(HostCut, KeptMoves)>, String> {
        match self.cut_host(candidates, Keeping::Device)? {
            Some(Cut::Kept(cut, moves)) => Ok(Some((cut, moves))),
            None => Ok(None),
            Some(Cut::Copied(_)) => unreachable!("the one store's shrink is moved on the device"),
        }
    }

    /// Each group's records, a host tier's or the one store's, cut between
    /// flights at the deepest of `candidates` whose cut lies at or above its
    /// floor (`Directory::host_floor`): every step retired, it is served as
    /// asked, with no wait for the reads, landings, copies or prompt groups
    /// in flight, whose records the shrink keeps where they are
    /// (`HostRecords::shrink`), each group's most valuable other records
    /// kept below the cut. Its tier tensors then address the records below
    /// the cut alone, and each landing in flight holds only its own record
    /// (`TierTensor::landed`), so no view the watcher made holds a released
    /// record once it answers. None where even the shallowest candidate
    /// cuts below a record in flight: nothing changes.
    fn cut_host(&self, candidates: Candidates, keeping: Keeping) -> Result<Option<Cut>, String> {
        match self.ask(Ask::ShrinkHost { candidates, keeping })? {
            Answer::Cut(cut) => cut,
            Answer::Planned(_) | Answer::Done(_) => unreachable!("a shrink of the records the host lands into is answered by its cut"),
        }
    }

    /// The bank holds `words` for the places a group holds from now,
    /// between flights (`ExpertBank::hold_places`): every step retired, the
    /// watcher serves each place's scratch requests and the telemetry reads
    /// each record's group reads from them.
    pub(in crate::experts) fn place(&self, words: PlacedWords) -> Result<(), String> {
        self.ask(Ask::Place(words)).map(drop)
    }

    /// Ask residency to cut the one store (where the host places) beside
    /// the running chain whose steps run from `first` through `last`, at its
    /// next turn, at the deepest of `candidates` whose cut keeps every
    /// record in flight below it ([`Residency::serve_running`]); the owner
    /// reads the answer as it comes ([`Self::running_cut`]) and never waits
    /// for it. One at a time.
    pub(in crate::experts) fn cut_running(&self, (first, last): (Step, Step), candidates: Candidates) {
        let mut running = self.resizing.running();
        assert!(matches!(*running, Running::None), "a cut beside running steps asked beside another");
        *running = Running::Asked { first, last, candidates, at: Instant::now() };
        drop(running);
        self.events().notify();
    }

    /// The cut asked beside a running chain ([`Self::cut_running`]), once
    /// residency answered it: cut, none where it cuts nothing (every
    /// candidate below a record in flight, a whole layer, or a group begun
    /// after the chain), or why serving ended. None while it is not
    /// answered yet.
    pub(in crate::experts) fn running_cut(&self) -> Option<Result<Option<RunningCut>, String>> {
        let mut running = self.resizing.running();
        match std::mem::replace(&mut *running, Running::None) {
            Running::Answered(answer) => Some(answer),
            asked => {
                *running = asked;
                let reason = self.life.ended()?;
                *running = Running::None;
                Some(Err(format!("the expert watcher stopped serving: {reason}")))
            }
        }
    }

    fn ask(&self, ask: Ask) -> Result<Answer, String> {
        let resizing = &self.resizing;
        self.steps.between_flights(|| {
            *resizing.resize.lock().expect("watcher resize") = Resize::Asked(ask, Instant::now());
        })?;
        // The watcher answers, or ends serving (`Shared::end_serving`), and
        // either notifies under this lock.
        let mut resize = resizing.resize.lock().expect("watcher resize");
        loop {
            if let Some(reason) = self.life.ended() {
                *resize = Resize::None;
                return Err(format!("the expert watcher stopped serving: {reason}"));
            }
            match std::mem::replace(&mut *resize, Resize::None) {
                Resize::Answered(answer) => return Ok(answer),
                asked => *resize = asked,
            }
            resize = resizing.resized.wait(resize).expect("watcher resize");
        }
    }
}

/// A resize the owner asked at `at`, taken by residency at `taken`, served
/// once nothing moves a record ([`Residency::resize`]).
pub(super) struct Asked {
    ask: Ask,
    at: Instant,
    taken: Instant,
}

impl Asked {
    /// `asked`, taken now.
    pub(super) fn taken((ask, at): (Ask, Instant)) -> Self {
        Self { ask, at, taken: Instant::now() }
    }
}

/// That a resize's changes are published ([`Residency::publish_resized`]),
/// which answering its owner takes: none answers with a map change the
/// device has not been given.
struct Published(());

/// What waits on what each layer holds being published, answered once
/// residency has published it as it is released
/// (`super::residency::Held`): a resize's answer, whose owner reads the
/// holdings it left, and the joins applied, whose holdings follow them.
#[derive(Default)]
pub(super) struct Owed {
    answer: Option<(Answer, Published)>,
    running: Option<Result<Option<RunningCut>, String>>,
    joined: usize,
}

impl Owed {
    /// Answer what waits, what each layer holds published.
    pub(super) fn settle(&mut self, shared: &Shared) {
        if let Some((answer, published)) = self.answer.take() {
            shared.resizing.answer(answer, published);
        }
        if let Some(cut) = self.running.take() {
            *shared.resizing.running() = Running::Answered(cut);
        }
        if self.joined > 0 {
            shared.joins.applied(std::mem::take(&mut self.joined));
        }
    }
}

impl Residency {
    /// Return the records of every plan dropped uncommitted
    /// ([`ShrinkPlan`]'s drop), as residency's lock is taken
    /// ([`super::residency::lock`]): no holder meets them, a resize served
    /// or a prompt group lent its records alike.
    pub(super) fn unshrink(&mut self) {
        if self.shared.resizing.abandoned.swap(false, std::sync::atomic::Ordering::AcqRel) {
            self.core.directory.unshrink();
        }
    }

    /// Take each slot a growth joined since ([`Shared::join`]), in order:
    /// the directory takes its new records (free bank records it then
    /// promotes into, or ready landed records, which file misses and the
    /// hints' reads land into), and the landings and the CPU site address
    /// landed records by their views. Whether it took any.
    pub(super) fn join(&mut self) -> bool {
        let joins = self.shared.joins.take();
        let count = joins.len();
        for join in joins {
            match join {
                Join::Bank { records } => {
                    for (group, records) in records.into_iter().enumerate() {
                        self.core.directory.grow(group, records);
                    }
                }
                Join::Landed { records, views } => {
                    for (group, records) in records.into_iter().enumerate() {
                        self.core.directory.grow_host(group, records);
                    }
                    self.address_tier(views);
                }
                Join::Growing => self.core.directory.host_growth(true),
                Join::Grown => self.core.directory.host_growth(false),
            }
        }
        if count > 0 {
            self.shared.telemetry.resized(self.core.bank_records());
            self.owed.joined += count;
        }
        count > 0
    }

    /// Serve `asked`: between flights every step has ended, so the frontier
    /// is past the last flight, and its steps retire. A resize that waits
    /// ([`Ask::waits`]) is answered only once nothing moves a record: the
    /// promotions in flight end (each copy's completion an event), then the
    /// reads in flight land (each landing's end an event), so no copy and no
    /// landing writes a record while a resize moves them, and a prompt
    /// group the device is past clears, returning the records it claimed;
    /// meanwhile nothing new is promoted or read ahead (`Residency::tend`).
    /// The ask, while it waits; none once answered, or once serving ended
    /// (whose end wakes the owner).
    pub(super) fn resize(&mut self, asked: Asked, copier: Option<&super::promotions::Copier>) -> Option<Asked> {
        // Every slot a growth joined before the ask is the directory's.
        self.join();
        self.retire();
        if let Some(copier) = copier {
            super::promotions::carry(&mut self.core, &self.shared, copier, self.prefetch.reached());
        }
        self.prefetch.settle(&mut self.core.directory);
        if self.shared.life.ended().is_some() {
            return None;
        }
        if asked.ask.waits() {
            if self.core.directory.moving() || self.prefetch.outstanding() {
                return Some(asked);
            }
            // A prompt group the device is past holds its records claimed
            // until its conveyor clears (its landings from the conveyor end,
            // each an event): a resize moves no record a group still holds.
            if self.scratch.clearing() && !self.scratch.clear_past(&mut self.core) {
                return Some(asked);
            }
        }
        let Asked { ask, at, taken } = asked;
        let trip = || Trip {
            picked: taken - at,
            served: taken.elapsed(),
        };
        let answer = self.serve_resize(ask, trip);
        let published = self.publish_resized();
        assert!(self.owed.answer.replace((answer, published)).is_none(), "a resize is answered before residency is released");
        None
    }

    /// Serve the cut asked beside a running chain, if one is
    /// ([`Self::serve_running`]), answered as residency is released.
    /// Whether one was.
    pub(super) fn cut_running(&mut self) -> bool {
        let Some((first, last, candidates, at)) = self.shared.resizing.running_asked() else {
            return false;
        };
        let taken = Instant::now();
        let trip = || Trip { picked: taken - at, served: taken.elapsed() };
        let cut = self.serve_running((first, last), &candidates, trip);
        assert!(self.owed.running.replace(cut).is_none(), "a running cut is answered before residency is released");
        true
    }

    /// Still between flights, before the owner begins a step, publish what a
    /// resize changed: the maps of every record it moved or released (a
    /// shrink's moves within the host's records name new records, its drops
    /// none), so no step resolves an expert at a record that holds another
    /// or is released; then the policy words, the whole words learning which
    /// groups it left whole. The one way to the proof an answer takes.
    fn publish_resized(&mut self) -> Published {
        self.refill_tier();
        let steps = self.shared.steps.clone();
        steps.while_idle(|| self.publish_policy());
        self.publish_lines();
        Published(())
    }

    /// Serve `asked`, every step retired, and the host's records' movements
    /// drained where it waits for them ([`Ask::waits`]).
    fn serve_resize(&mut self, asked: Ask, trip: impl Fn() -> Trip) -> Answer {
        match asked {
            Ask::Shrink { slots, records } => {
                // Ranked by every step that ended before the gap it is asked
                // in, whichever turn of this thread their ends fell in.
                self.shared.telemetry.feed();
                let ranking = self.core.directory.ranking();
                let last_read = |layer, expert| Reuse::of(&ranking).last(layer, expert);
                let groups = plan(&mut self.core, &records, last_read);
                Answer::Planned(ShrinkPlan { slots, groups, trip: trip(), holds: Some(self.shared.clone()) })
            }
            Ask::Commit(mut plan) => {
                commit(&mut self.core, &plan);
                plan.holds = None;
                self.shared.telemetry.resized(self.core.bank_records());
                Answer::Done(trip())
            }
            Ask::ShrinkHost { candidates, keeping } => Answer::Cut(self.serve_cut(&candidates, keeping, trip)),
            Ask::Place(PlacedWords { flags, layers }) => {
                let places = flags.len();
                self.scratch.place(flags);
                self.shared.telemetry.place(GroupReadWords { layers, places });
                Answer::Done(trip())
            }
        }
    }
}

impl Residency {
    /// Cut each group's records the host lands into at the deepest of
    /// `candidates` whose records lie at or above its floor
    /// (`Directory::host_floor`), so every record in flight stays where it
    /// is; none where none does. The views of the records below the cut
    /// are formed first, so a refusal changes nothing.
    fn serve_cut(&mut self, candidates: &Candidates, keeping: Keeping, trip: impl Fn() -> Trip) -> Result<Option<Cut>, String> {
        let floors = (0..self.core.groups.len()).map(|group| self.core.directory.host_floor(group)).collect::<Vec<_>>();
        let Some((slots, records)) = deepest(candidates, &floors) else {
            return Ok(None);
        };
        let views = self.core.tier_views(&records)?;
        let moves = records
            .iter()
            .enumerate()
            .map(|(group, &records)| self.core.directory.shrink_host(group, records))
            .collect::<Vec<_>>();
        if keeping == Keeping::Host {
            for (group, moves) in moves.iter().enumerate() {
                let tensors = self.core.groups[group].tier.as_ref().expect("a host tier shrinks only beside its tensors");
                for &(from, to) in moves {
                    for tensor in tensors.iter().map(|tensor| &tensor.records) {
                        // SAFETY: both are committed records of the tensor
                        // that no landing, copy or step reaches meanwhile:
                        // neither is in flight (`HostRecords::shrink`), and
                        // every step retired, between flights. Distinct
                        // records never overlap.
                        unsafe { tensor.copy_record(u64::from(to.0), tensor, u64::from(from.0), tensor.bytes()) };
                    }
                }
            }
        }
        self.address_tier(views);
        self.shared.telemetry.resized(self.core.bank_records());
        let cut = HostCut { slots, trip: trip() };
        Ok(Some(match keeping {
            Keeping::Host => Cut::Copied(cut),
            Keeping::Device => Cut::Kept(cut, moves.into_iter().map(|moves| moves.into_iter().map(|(from, to)| (from.0, to.0)).collect()).collect()),
        }))
    }
}

impl Residency {
    /// Cut the one store's records beside the running chain whose steps run
    /// from `first` through `last`, at the deepest of `candidates` whose
    /// records lie at or above each group's running floor
    /// (`Directory::running_floor`), moving nothing. Each expert kept at or
    /// past the cut is unmapped where its entry still names that record
    /// (`Directory::cut_running`): from the chain's first step where none
    /// of its requests taken so far routed it at its layer, so every step of
    /// the chain still to decide the layer's hits resolves it a miss and its
    /// rounds read it, as any cold expert; else from the first step not yet
    /// begun, so the chain's later passes there, which resolve it as the
    /// pass whose hit it was, still read it, the decided layers' passes
    /// being submitted before the cut's storage installs (`RunningCut`).
    /// Each later chain begins after the stores and resolves none of them.
    /// The views of the records below the cut are formed first, so a
    /// refusal changes nothing.
    ///
    /// None where a layer is whole, whose steps read their experts at their
    /// homes without the map (`POLICY_WHOLE`), or once a group began after
    /// the chain, whose steps decided their hits by the map before the cut;
    /// nor where even the shallowest candidate cuts below a record in
    /// flight.
    fn serve_running(&mut self, (first, last): (Step, Step), candidates: &Candidates, trip: impl Fn() -> Trip) -> Result<Option<RunningCut>, String> {
        if self.whole.iter().any(|&whole| whole) || !self.shared.steps.latest(last) {
            return Ok(None);
        }
        let floors = (0..self.core.groups.len()).map(|group| self.core.directory.running_floor(group)).collect::<Vec<_>>();
        let Some((slots, records)) = deepest(candidates, &floors) else {
            return Ok(None);
        };
        let views = self.core.tier_views(&records)?;
        let decided = self.scratch.decided(first);
        let (scratch, directory) = (&self.scratch, &mut self.core.directory);
        for (group, &records) in records.iter().enumerate() {
            directory.cut_running(group, records, |(layer, expert)| match scratch.routed(first, layer, expert) {
                true => Step::MAX,
                false => first,
            });
        }
        self.address_tier(views);
        self.shared.telemetry.resized(self.core.bank_records());
        Ok(Some(RunningCut { slots, decided, trip: trip() }))
    }
}

/// The deepest of `candidates` whose records (each group's, fitting u32)
/// lie at or above each group's floor in `floors`: its slots and records.
fn deepest(candidates: &Candidates, floors: &[u32]) -> Option<(usize, Vec<u32>)> {
    (0..candidates.count()).rev().map(|index| candidates.slots(index)).find_map(|slots| {
        let records = (candidates.records)(slots)
            .into_iter()
            .map(|records| u32::try_from(records).expect("the records the host lands into fit u32"))
            .collect::<Vec<_>>();
        let keeps = records.iter().zip(floors).all(|(records, floor)| records >= floor);
        keeps.then_some((slots, records))
    })
}

/// Each group's shrink to `records`, every placement ended, its evictions
/// taken: the directory unmaps them from the first step not yet begun,
/// every step before it completed, and holds every record a move writes or
/// the shrink retires until it commits (`BankRecords::shrinking`), so
/// nothing maps to a record a move writes. An expert's last read is the
/// ranking's (`Reuse::last`), its one owner, which the stack has fed every
/// ended step's reads, a whole group's decode reads too (its routing
/// records).
fn plan(
    core: &mut Core,
    records: &[usize],
    last_read: impl Fn(usize, usize) -> Option<super::Step>,
) -> Vec<Shrink> {
    let (directory, steps) = (&mut core.directory, &core.steps);
    steps.unbegun(|at| {
        records
            .iter()
            .enumerate()
            .map(|(group, &records)| {
                let plan = directory.plan_shrink(group, records, &last_read);
                directory.evict(group, &plan, &last_read, at);
                plan
            })
            .collect()
    })
}

/// `plan`'s moves, which the device carries ahead of every later step: the
/// directory follows each, and each moved expert is mapped to its new
/// record from the first step not yet begun.
fn commit(core: &mut Core, plan: &ShrinkPlan) {
    let (directory, steps) = (&mut core.directory, &core.steps);
    steps.unbegun(|at| {
        for (group, shrink) in plan.groups.iter().enumerate() {
            directory.commit_shrink(group, shrink, at);
        }
    });
}
