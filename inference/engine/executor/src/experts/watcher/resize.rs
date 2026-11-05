//! The bank's resizes. A shrink runs between flights: the owner asks,
//! residency answers once nothing moves a record (its turns wait on the
//! load's events, never on a time). It is planned with its evictions published at once
//! (always safe: the bank only loses experts), and answered with a plan the
//! owner commits once its moves are queued on the device ahead of every
//! later step, or drops. A growth asks nothing: each of its slots joins as
//! soon as every record tensor backs it ([`Shared::join`]), beside running
//! steps, which reach a record only once the host maps it. Each
//! carries the bank's slots, which the watcher's telemetry reads.

use super::Shared;
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

/// What paces a shrink: the device seconds one resize's moves may take (a
/// decode step's), and each group's move line, which prices its launches
/// at the records it moves.
pub(crate) struct MoveBudget {
    pub seconds: f64,
    pub lines: crate::assessment::MoveLines,
}

impl MoveBudget {
    /// Whether moving `moved(g)` records of each group `g` fits.
    pub fn fits(&self, moved: &dyn Fn(usize) -> u64) -> bool {
        self.lines.seconds(moved) <= self.seconds
    }
}

/// A shrink's candidate commitments: from `shallowest` slots (one fewer
/// than held) down to `deepest` (the shrink's target), each group's records
/// at a candidate's slots formed by `records` only for the candidates the
/// pacing prices.
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
    /// Plan each group's commitment at the candidate `budget` paces, and
    /// evict what it evicts.
    Shrink { candidates: Candidates, budget: Option<MoveBudget> },
    /// Take a planned shrink's moves, which the device carries.
    Commit(ShrinkPlan),
    /// Shrink each group's host tier to `records`, its kept experts moved
    /// below them first within the records its tier tensors address, which
    /// then address `views` (each group's record tensors' records
    /// committed after it).
    ShrinkHost { records: Vec<u32>, views: Vec<Vec<Records>> },
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

enum Answer {
    Planned(ShrinkPlan),
    Done(Trip),
}

enum Resize {
    None,
    /// Asked, at that instant.
    Asked(Ask, Instant),
    Answered(Answer),
}

/// The resize the owner and the watcher exchange.
pub(super) struct ResizeHandoff {
    resize: Mutex<Resize>,
    resized: Condvar,
    /// Whether a plan was dropped uncommitted since residency last looked
    /// ([`ShrinkPlan`]'s drop): its records return.
    abandoned: std::sync::atomic::AtomicBool,
}

impl ResizeHandoff {
    pub(super) fn new() -> Self {
        Self {
            resize: Mutex::new(Resize::None),
            resized: Condvar::new(),
            abandoned: std::sync::atomic::AtomicBool::new(false),
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
    /// ([`Residency::publish_resized`]).
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
    /// Plan the bank's commitment at the deepest of `candidates` (its slots,
    /// each group's records), shallowest first, whose moves fit `budget`,
    /// or the shallowest where none does (`Directory::paced`); at the deepest
    /// without a budget: between flights, every begun step ended and the
    /// device past it. The watcher retires every fill (the conveyor returns
    /// the lender's records), plans each shrinking group to keep its most
    /// recently read experts below its new commitment
    /// (`Directory::plan_shrink`), and evicts the rest: their slot map
    /// entries and directory records are cleared now.
    pub fn plan_shrink(&self, candidates: Candidates, budget: Option<MoveBudget>) -> Result<ShrinkPlan, String> {
        match self.ask(Ask::Shrink { candidates, budget })? {
            Answer::Planned(plan) => Ok(plan),
            Answer::Done(_) => unreachable!("a shrink is answered with its plan"),
        }
    }

    /// Commit `plan` once its moves are queued on the device ahead of every
    /// later step, still between flights: each moved expert's map entry
    /// follows it, and each group's records end at its new commitment.
    pub fn commit_shrink(&self, plan: ShrinkPlan) -> Result<Trip, String> {
        match self.ask(Ask::Commit(plan))? {
            Answer::Done(trip) => Ok(trip),
            Answer::Planned(_) => unreachable!("a commit is answered done"),
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

    /// Each group's host tier holds `records` records from now, at most as
    /// many as before, between flights; the rest are released once this
    /// returns. It retires every step and waits for the reads in flight to
    /// land, so every record is free (no step reads one and no landing
    /// writes one), and
    /// keeps each group's most valuable records (`HostRecords::shrink`),
    /// their bytes copied below its new records within those its tier
    /// tensors address, which address `views` from then on: the watcher
    /// holds no view of a released record once it answers.
    pub(in crate::experts) fn shrink_host(&self, records: Vec<u32>, views: Vec<Vec<Records>>) -> Result<Trip, String> {
        self.host_done(Ask::ShrinkHost { records, views })
    }

    fn host_done(&self, ask: Ask) -> Result<Trip, String> {
        match self.ask(ask)? {
            Answer::Done(trip) => Ok(trip),
            Answer::Planned(_) => unreachable!("a host tier's resize is answered done"),
        }
    }

    /// The bank holds `words` for the places a group holds from now,
    /// between flights (`ExpertBank::hold_places`): every step retired, the
    /// watcher serves each place's scratch requests and the telemetry reads
    /// each record's group reads from them.
    pub(in crate::experts) fn place(&self, words: PlacedWords) -> Result<(), String> {
        self.ask(Ask::Place(words)).map(drop)
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
            self.publish_held();
            self.shared.joins.applied(count);
        }
        count > 0
    }

    /// Serve `asked` and answer once nothing moves a record: between
    /// flights every step has ended, so the frontier is past the last
    /// flight; its steps retire, the promotions in flight end (each copy's
    /// completion an event), then the reads in flight land (each landing's
    /// end an event): no copy and no landing writes a record while a resize
    /// moves them, and a prompt group the device is past clears, returning
    /// the records it claimed. Meanwhile nothing new is promoted or read ahead
    /// (`Residency::tend`). The ask, while it waits; none once answered, or
    /// once serving ended (whose end wakes the owner).
    pub(super) fn resize(&mut self, asked: Asked, copier: Option<&super::promotions::Copier>) -> Option<Asked> {
        // Every slot a growth joined before the ask is the directory's.
        self.join();
        self.retire();
        if let Some(copier) = copier {
            super::promotions::carry(&mut self.core, &self.shared, copier);
        }
        self.prefetch.settle(&mut self.core.directory);
        if self.shared.life.ended().is_some() {
            return None;
        }
        if self.core.directory.moving() || self.prefetch.outstanding() {
            return Some(asked);
        }
        // A prompt group the device is past holds its records claimed until
        // its conveyor clears (its landings from the conveyor end, each an
        // event): a resize moves no record a group still holds.
        if self.scratch.clearing() && !self.scratch.clear_past(&mut self.core, &mut self.prefetch) {
            return Some(asked);
        }
        let Asked { ask, at, taken } = asked;
        let trip = || Trip {
            picked: taken - at,
            served: taken.elapsed(),
        };
        let answer = self.serve_resize(ask, trip);
        let published = self.publish_resized();
        self.shared.resizing.answer(answer, published);
        None
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
        Published(())
    }

    /// Serve `asked`, every step retired and the host tier's movements
    /// drained.
    fn serve_resize(&mut self, asked: Ask, trip: impl Fn() -> Trip) -> Answer {
        match asked {
            Ask::Shrink { candidates, budget } => {
                // Ranked by every step that ended before the gap it is asked
                // in, whichever turn of this thread their ends fell in.
                self.shared.telemetry.feed();
                let ranking = self.core.directory.ranking();
                let last_read = |layer, expert| Reuse::of(&ranking).last(layer, expert);
                // The deepest candidate is the shrink's target.
                let target = (candidates.records)(candidates.deepest);
                let chosen = match budget {
                    Some(budget) => self.core.directory.paced(
                        candidates.count(),
                        |index| (candidates.records)(candidates.slots(index)),
                        |moved| budget.fits(moved),
                        &last_read,
                    ),
                    None => candidates.count() - 1,
                };
                let slots = candidates.slots(chosen);
                let records = (candidates.records)(slots);
                let groups = plan(&mut self.core, &records, &target, last_read);
                self.publish_held();
                Answer::Planned(ShrinkPlan { slots, groups, trip: trip(), holds: Some(self.shared.clone()) })
            }
            Ask::Commit(mut plan) => {
                commit(&mut self.core, &plan);
                plan.holds = None;
                self.shared.telemetry.resized(self.core.bank_records());
                self.publish_held();
                Answer::Done(trip())
            }
            Ask::ShrinkHost { records, views } => {
                for (group, records) in records.into_iter().enumerate() {
                    let tensors = self.core.groups[group].tier.as_ref().expect("a host tier shrinks only beside its tensors");
                    for (from, to) in self.core.directory.shrink_host(group, records) {
                        for tensor in tensors.iter().map(|tensor| &tensor.records) {
                            // SAFETY: both are committed records of the
                            // tensor, nothing reads or writes them meanwhile
                            // (every step retired, between flights), and
                            // distinct records never overlap.
                            unsafe { tensor.copy_record(u64::from(to.0), tensor, u64::from(from.0), tensor.bytes()) };
                        }
                    }
                }
                self.address_tier(views);
                self.shared.telemetry.resized(self.core.bank_records());
                self.publish_held();
                Answer::Done(trip())
            }
            Ask::Place(PlacedWords { flags, layers }) => {
                let places = flags.len();
                self.scratch.place(flags);
                self.shared.telemetry.place(GroupReadWords { layers, places });
                Answer::Done(trip())
            }
        }
    }
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
    target: &[usize],
    last_read: impl Fn(usize, usize) -> Option<super::Step>,
) -> Vec<Shrink> {
    let (directory, steps) = (&mut core.directory, &core.steps);
    steps.unbegun(|at| {
        records
            .iter()
            .zip(target)
            .enumerate()
            .map(|(group, (&records, &target))| {
                let plan = directory.plan_shrink(group, records, target, &last_read);
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
