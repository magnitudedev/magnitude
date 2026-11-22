//! Residency: the one owner of where experts live and what moves them. It
//! holds the directory (every map entry's write, through the steps' lock,
//! `experts::map`), the host tier and its mirror in each layer's tier map,
//! staging and the landings into the tier, the conveyor and its scratch
//! tiers, the promotions into the bank and the policy words, behind one
//! lock whose every critical section is short: no method waits for
//! anything.
//!
//! Two threads act through it. The request service (`requests`) takes the
//! device's requests and answers them from it. Residency's own thread
//! ([`Residency::keep`]) does what no request asks: it feeds the ended
//! steps' reads to the ranking (outside the lock), retires the steps the
//! frontier passed, learns the landings that ended, keeps the tier's
//! map and reserve, carries the promotions, clears a prompt group's
//! conveyor once nothing of it runs, hastens the reads ahead a prompt
//! group's step waits for, joins a growth's slots, serves a resize (a cut
//! of the one store beside a running chain among them) and writes the
//! policy words in a gap. Each turn of it runs on the load's events (`events`): a step's
//! end (the frontier), a landing's or a read's end, a copy's completion, a
//! growth joined, a resize asked, the request service's progress. It never
//! waits on a time, and no job it does waits for another. It holds the lock
//! as its background ([`lock`]): it takes it only once no request waits,
//! and gives it up between its jobs to any request that came meanwhile,
//! so a request waits out at most one job of a turn, never the turn.

use super::core::Core;
use super::decode;
use super::link::{Copier, LinkMover};
use super::prefetch::Prefetch;
use super::resize::{Asked, Owed};
use super::scratch::Scratch;
use super::{Shared, Step};
use magnitude_cost_model::{DomainBudget, ExpertLines, SiteLines};
use crate::experts::directory::{MissCurve, Point};
use crate::experts::interest::{Ledger, Told};
use crate::experts::yielding::Yielding;
use magnitude_kernels::constants;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, MutexGuard};
use std::time::Instant;

/// The policy words a run's encoding reads, as residency stages them
/// ([`Residency::publish_policy`]): each streamed layer's, and whether a
/// CPU site serves any layer.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::experts) struct PolicyWords {
    layers: Vec<LayerPolicy>,
    served: bool,
}

/// A streamed layer's policy words: whole, resolved, its home, and whether
/// its hint launches run.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LayerPolicy {
    whole: bool,
    resolved: bool,
    home: i32,
    hints: bool,
}

/// Where experts live and what moves them, behind one lock
/// ([`lock`]).
pub(super) struct Residency {
    /// Per layer, its home as its policy names it (`policy[POLICY_HOME]`),
    /// the directory's ([`Directory::home`](crate::experts::directory::Directory::home)).
    pub(super) homes: Vec<i32>,
    pub(super) shared: Arc<Shared>,
    pub(super) core: Core,
    /// Shared `NameLedger` owned by residency: fed by ring reads & `Told`
    /// events, and consumed by both Path 1 (`Prefetch`) and Path 0 (`LinkMover`).
    pub(super) ledger: Ledger,
    pub(super) inbox: Receiver<Told>,
    pub(super) prefetch: Prefetch,
    pub(super) scratch: Scratch,
    /// Unified Path 0 (`RAM -> VRAM`) link mover: write-backs, `NameLedger`-driven
    /// opportunistic DMA staging, and `Reuse`-ranked permanent promotions.
    pub(super) link: LinkMover,
    /// A resize asked, waiting until nothing moves a record.
    pub(super) asked: Option<Asked>,
    /// When residency opened, and whether the host tier keeps every expert
    /// (`Directory::tier_complete`), as last published.
    pub(super) opened: Instant,
    pub(super) complete: bool,
    /// Each layer's line words as last published ([`Self::publish_lines`]);
    /// none before the first.
    pub(super) lines: Option<Vec<Vec<(usize, i32)>>>,
    /// What each layer holds, as last published ([`Self::publish_held`]).
    pub(super) published: Published,
    /// What waits on that publication, answered as residency is released
    /// ([`Held`]).
    pub(super) owed: Owed,
}

/// What each layer holds as residency last published it: the directory's
/// count of holding changes then, and each layer's experts held and kept
/// alone, the buffers every publication reuses.
pub(super) struct Published {
    holdings: Option<u64>,
    held: Vec<usize>,
    alone: Vec<usize>,
    lend: Vec<super::GroupLend>,
    capacities: Vec<usize>,
}

impl Published {
    /// None published yet, of `layers` layers, telemetry told each group's
    /// `capacities` at the load (`Directory::capacities`).
    pub(super) fn new(layers: usize, capacities: Vec<usize>) -> Self {
        Self { holdings: None, held: vec![0; layers], alone: vec![0; layers], lend: Vec::new(), capacities }
    }
}

/// Residency's lock: its keeper is its background ([`Yielding`]), so a
/// request never waits out a turn of the keeper's, only the one job of it
/// running ([`Residency::keep`]). Whoever releases it publishes what each
/// layer holds, then answers whoever waits on that publication ([`Held`]),
/// so every holder, and every owner a resize answers, finds it published.
/// A panic while it is held is caught inside it, and serving ends before
/// it is released (`Residency::keep`, `Requests::serve`): no thread serves
/// from the state it left, which only the refusals read
/// (`liveness::refuse`), so a poisoned lock is never observed.
pub(super) fn lock(residency: &Yielding<Residency>) -> Held<'_> {
    Held::new(residency.hold(POISONED))
}

/// [`lock`], `counted` run once the holder is counted waiting
/// ([`Yielding::hold_counted`]): from then the keeper gives way to it at
/// its next job's end, so what `counted` does (the request service's look,
/// `decode::Look`) runs beside a job the keeper holds, never before one it
/// starts.
pub(super) fn lock_counted(residency: &Yielding<Residency>, counted: impl FnOnce()) -> Held<'_> {
    Held::new(residency.hold_counted(POISONED, counted))
}

/// Why residency's lock is never poisoned ([`lock`]).
const POISONED: &str = "residency is never left by a panic";

/// Residency held ([`lock`]): as it is released, what each layer holds is
/// published under the lock, then the lock and `NoBlockingAllocationGuard` are
/// released, any evicted/replaced `Tensor`/`Records` views are dropped outside
/// the lock, and finally what waits on that publication (`Owed`) is answered.
pub(super) struct Held<'a> {
    guard: Option<MutexGuard<'a, Residency>>,
    no_alloc: Option<seismic::memory::NoBlockingAllocationGuard>,
}

impl<'a> Held<'a> {
    fn new(held: MutexGuard<'a, Residency>) -> Self {
        Self {
            guard: Some(held),
            no_alloc: Some(seismic::memory::enter_no_blocking_allocation()),
        }
    }

    /// Given first to every holder waiting for it (the keeper's, between
    /// its jobs: [`Yielding::give_way`]), released as every holder is.
    fn give_way(self, residency: &'a Yielding<Residency>) -> Self {
        residency.give_way(self, POISONED, Self::new)
    }
}

impl std::ops::Deref for Held<'_> {
    type Target = Residency;
    fn deref(&self) -> &Residency {
        self.guard.as_deref().expect("Held guard is present while live")
    }
}

impl std::ops::DerefMut for Held<'_> {
    fn deref_mut(&mut self) -> &mut Residency {
        self.guard.as_deref_mut().expect("Held guard is present while live")
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        let mut guard = self.guard.take().expect("Held guard dropped once");
        let shared = guard.shared.clone();
        // A panic is caught inside the lock, never unwinding through it,
        // and ends serving before the lock is released ([`lock`]): nothing
        // is published or answered from the state it left.
        let settle = !std::thread::panicking() && guard.shared.life.ended().is_none();
        if settle {
            guard.publish_held();
        }
        let mut owed = std::mem::take(&mut guard.owed);
        let deferred_records = std::mem::take(&mut owed.deferred_records);
        // Release residency and `NoBlockingAllocationGuard` before dropping
        // replaced views (whose final `SlabHold`/`Allocation` drop may touch
        // pool or driver locks), and before `Owed::settle` wakes the resize
        // thread so `Stores::free` sees no lingering view references.
        drop(guard);
        drop(self.no_alloc.take());
        drop(deferred_records);
        if settle {
            owed.settle(&shared);
        }
    }
}

/// What one turn of the keeper did so far, carried from job to job
/// ([`Residency::TURN`]): only what the turn announces at its end. Nothing
/// a job reads of the load is carried, since a request may hold residency
/// between two jobs and change it.
#[derive(Default)]
pub(super) struct Turn {
    /// What a request is served by changed (a step retired, a landing
    /// learned, a promotion asked, a growth joined, a prompt group
    /// cleared): the request service looks again.
    changed: bool,
}

/// One job of a turn, given what the turn did so far: whether the turn
/// goes on.
pub(super) type Job<'a> = &'a dyn Fn(&mut Residency, &mut Turn, Option<&Copier>, Option<&MissCurve>) -> bool;

impl Residency {
    /// Residency's thread: a turn ([`Self::tend`]) on every event of the
    /// load, until it stops. Once serving ended (the device lost, a read
    /// line lost or a promotion's copy failed), it does nothing more: the request service refuses
    /// what the device waits on, and the owner waiting for a resize was
    /// woken with why.
    /// Promotions copy by `copier` (none where the bank has no host tier),
    /// which this thread alone holds.
    pub(super) fn keep(residency: Arc<Yielding<Self>>, shared: Arc<Shared>, copier: Option<Copier>) {
        let events = shared.events().clone();
        // The decode steps the reserve was last sized at.
        let mut decoded_steps = 0;
        loop {
            let seen = events.seen();
            if shared.life.stopping() {
                return;
            }
            if shared.life.ended().is_none() {
                // The ended groups' reads, fed to the ranking before the turn
                // ranks by it, and the decode steps' depth curve where a step
                // was read since (which sizes the promotions' reserve),
                // outside residency's lock so no request waits on the stack.
                shared.telemetry.feed();
                let decoded = copier.as_ref().and_then(|_| shared.telemetry.decoded_since(&mut decoded_steps));
                Self::tend(&residency, copier.as_ref(), decoded.as_ref());
                // Then the ranking's shadows read what it logged, after the
                // turn: no request, read ahead or promotion waits on them.
                shared.telemetry.shadow();
            }
            events.wait(seen);
        }
    }

    /// One turn, none of whose jobs waits: a growth's slots joined; the
    /// steps the frontier passed retired; the landings that ended learned
    /// and the tier map published; then a resize asked, served once the
    /// promotions and landings in flight end (until then nothing new is
    /// promoted or read ahead); else the promotions asked and carried, a
    /// prompt group past cleared, what each layer holds published, the
    /// reads ahead a prompt group's step waits for hastened, the line words
    /// written where they changed, and in a gap the policy words a run's
    /// encoding reads. A turn that changed what a request is served by (a
    /// retirement, a landing learned, a promotion, a group cleared, a growth
    /// joined) is an event: the request service looks again.
    ///
    /// The turn takes residency's lock once no request waits for it, and
    /// between its jobs gives it to any request that came to wait since
    /// ([`Yielding::give_way`]): a request waits out at most the job it
    /// found running. Each job leaves residency as a turn's end does (what it
    /// moved published as it gives way, [`Held`], and nothing it decided
    /// left for a later job), so a pass
    /// between two serves from it as from one between turns.
    fn tend(residency: &Yielding<Self>, copier: Option<&Copier>, decoded: Option<&MissCurve>) {
        Self::turn(residency, &Self::TURN, copier, decoded);
    }

    /// A turn of `jobs` ([`Self::tend`]'s, [`Self::TURN`]).
    pub(super) fn turn(residency: &Yielding<Self>, jobs: &[Job<'_>], copier: Option<&Copier>, decoded: Option<&MissCurve>) {
        let mut held = Held::new(residency.background(POISONED));
        let mut turn = Turn::default();
        for job in jobs {
            if !job(&mut held, &mut turn, copier, decoded) {
                return;
            }
            held = held.give_way(residency);
        }
    }

    /// A turn's jobs, in order ([`Self::tend`]).
    pub(super) const TURN: [Job<'static>; 5] = [&Self::learn, &Self::promote, &Self::clear, &Self::hasten, &Self::publish_turn];

    /// A growth's slots joined, the steps the frontier passed retired, the
    /// landings that ended learned and the tier's reserve kept; then a
    /// resize asked served, which ends the turn. A fault of the executor's
    /// work, or the device lost under the parks, ends serving, as
    /// residency's own panic does, and the turn.
    /// Drain all pending `Told` events into the shared `NameLedger`, updating
    /// telemetry and notifying `Prefetch` on step begin / draft window changes.
    pub(super) fn learn_told(&mut self) -> bool {
        let mut told = false;
        let link_only = self.core.directory.tier_complete() && !self.core.directory.all_groups_whole();
        while let Ok(mut event) = self.inbox.try_recv() {
            told = true;
            let completed = matches!(&event, Told::Completed { .. });
            let began = matches!(&event, Told::Began { .. });
            let draft = matches!(&event, Told::DraftWindow { .. });
            // On 100%-RAM discrete GPUs (`tier_complete()` with partial VRAM),
            // `Use::Tier` is a Path 0 (`Link`) miss rather than a Drive hit:
            // reclassify un-staged Host-RAM misses as `tier: false` so
            // `HintGate` measures the true live Link miss cohort instead of 0.
            if link_only {
                if let Told::Completed { step, ref mut chose, .. } = event {
                    for (layer, choices) in chose.iter_mut().enumerate() {
                        for chosen in choices.iter_mut() {
                            if chosen.saves && chosen.tier && !self.ledger.has_link_landed(step, layer, chosen.expert) {
                                chosen.tier = false;
                            }
                        }
                    }
                }
            }
            let learned = self.ledger.tell(event);
            if began {
                self.prefetch.on_began();
            } else if draft {
                self.prefetch.on_draft_window();
            }
            if completed {
                self.shared.telemetry.resolved(&learned.resolved, &learned.cohorts);
            }
        }
        let (judged, accepted) = self.ledger.take_judged();
        if !judged.is_empty() || !accepted.is_empty() {
            self.shared.telemetry.judged(&judged, &accepted);
        }
        Prefetch::spend(&mut self.ledger, &self.shared);
        told
    }

    fn learn(&mut self, turn: &mut Turn, copier: Option<&Copier>, _: Option<&MissCurve>) -> bool {
        if let Some(reason) = self.shared.fault() {
            self.shared.end_serving(reason);
            return false;
        }
        // Drain `Told` events (`DraftWindow`, `Began`, `Completed`) before
        // `promote` (`LinkMover::advance`) so Path 0 immediately sees fresh
        // draft windows and completed steps on the same turn.
        let told = self.learn_told();
        let joined = self.join();
        let retired = self.retire();
        let (landed, newly_landed) = self.prefetch.settle(&mut self.core.directory, &mut self.ledger);
        let refilled = self.refill_tier();
        // Pipeline chain-through: if Path 1 just landed an expert into Tier 1
        // that is live in `NameLedger` for an upcoming step, mark changed so
        // `promote` (`LinkMover::advance`) immediately stages it into Tier 0.
        let chain_through = newly_landed.into_iter().any(|(layer, expert)| self.ledger.is_wanted(layer, expert));
        turn.changed = told || joined || retired || landed || refilled || chain_through;
        if let Some(asked) = self.asked.take().or_else(|| self.shared.resizing.asked().map(Asked::taken)) {
            self.asked = self.resize(asked, copier);
            return false;
        }
        true
    }

    /// Unified Path 0 (`LinkMover`): write-backs, `NameLedger`-driven
    /// opportunistic DMA staging, and `Reuse`-ranked permanent promotions.
    fn promote(&mut self, turn: &mut Turn, copier: Option<&Copier>, decoded: Option<&MissCurve>) -> bool {
        let held = self.traffic_held();
        let progressed = copier.is_some_and(|copier| {
            self.link.advance(&mut self.core, &mut self.ledger, &self.shared, copier, decoded, held, self.prefetch.reached())
        });
        turn.changed |= progressed;
        true
    }

    /// A prompt group past cleared from the conveyor.
    fn clear(&mut self, turn: &mut Turn, _: Option<&Copier>, _: Option<&MissCurve>) -> bool {
        turn.changed |= self.scratch.clear_past(&mut self.core);
        true
    }

    /// Drive Path 1 (`Prefetch`), cover reads, and prompt-group hastening.
    fn hasten(&mut self, turn: &mut Turn, copier: Option<&Copier>, _: Option<&MissCurve>) -> bool {
        let held = self.traffic_held();
        if held {
            self.prefetch.cancel_cover();
        } else {
            let told = self.learn_told();
            let reads_finished = self.prefetch.take_reads_finished();
            turn.changed |= told || reads_finished;
            if turn.changed {
                if !self.core.directory.tier_complete() {
                    self.prefetch.drive(&mut self.core, &mut self.ledger, &self.shared);
                }
            }
            let scratch = &self.scratch;
            self.prefetch.cover(&mut self.core, &self.shared, |staged| scratch.carries(staged));
            for staged in self.scratch.awaited(&self.core) {
                self.prefetch.hasten(staged);
            }
        }
        self.shared.publish_cover(self.prefetch.covering(&self.core));
        let copying = copier.is_some_and(|copier| !copier.copies.idle());
        let quiet = held && !copying && !self.prefetch.outstanding();
        if self.shared.traffic_quiet.swap(quiet, std::sync::atomic::Ordering::AcqRel) != quiet {
            self.shared.events().notify();
        }
        true
    }

    /// The words a run's encoding reads, only while no step runs; the
    /// lines at every turn; and the turn's change announced.
    fn publish_turn(&mut self, turn: &mut Turn, copier: Option<&Copier>, _: Option<&MissCurve>) -> bool {
        self.publish_policy();
        self.shared.write_staged();
        self.publish_lines(copier);
        if turn.changed {
            self.shared.events().notify();
        }
        true
    }

    /// Whether traffic is held for a measurement beside the idle load: as
    /// of now, a step running ending it ([`Self::promote`]).
    fn traffic_held(&self) -> bool {
        self.shared.traffic_held() && self.shared.steps.idle()
    }

    /// Retire through the frontier (`Steps::frontier`): the device is past
    /// every group whose every step is below it. Whether the frontier moved.
    pub(super) fn retire(&mut self) -> bool {
        let frontier = self.shared.steps.frontier();
        if frontier <= self.core.retired {
            return false;
        }
        self.core.retired = frontier;
        self.core.directory.drain_staging(self.prefetch.reached(), &self.core.steps);
        if let Some(past) = self.shared.steps.retire_groups(frontier) {
            self.advance(past);
        }
        true
    }

    /// Read live hint and front rings while any group is not whole in Tier 0
    /// (`!core.directory.all_groups_whole()`), unblocking foresight on
    /// 100%-RAM discrete GPUs as well as 2-tier and 3-tier workloads.
    pub(super) fn rings(&mut self, shared: &Arc<Shared>, served: &[Step], mut freed: bool) -> bool {
        if self.core.directory.all_groups_whole() {
            return self.learn_told();
        }
        let raw = self.prefetch.read_rings_raw(&self.core);
        let told = self.learn_told();
        let read = self.prefetch.resolve_rings(&self.core, &mut self.ledger, shared, raw, told);
        if let Some(latest) = read.latest {
            if latest > self.prefetch.reached() {
                self.advance_retired(latest);
                freed = true;
            }
        }
        if read.any {
            self.prefetch.pass(&mut self.core, &mut self.ledger, shared, None);
        }
        let (settled, newly_landed) = self.prefetch.settle(&mut self.core.directory, &mut self.ledger);
        let chain_through = newly_landed.into_iter().any(|(layer, expert)| self.ledger.is_wanted(layer, expert));
        let refilled = self.refill_tier();
        let reads_finished = self.prefetch.take_reads_finished();
        freed |= settled || refilled || chain_through || reads_finished;
        self.prefetch.finish_rings(&mut self.core, &mut self.ledger, shared, read, freed, served)
    }

    pub(super) fn advance(&mut self, progress: Point) {
        self.advance_retired(progress);
        self.refill_tier();
    }

    /// Advance device progress to `progress` and retire completed landings,
    /// seeds, and tier graces without refilling the reserve: shared by
    /// [`Self::advance`] and [`Self::rings`], where `refill_tier` runs once
    /// after settling.
    fn advance_retired(&mut self, progress: Point) {
        self.prefetch.past(progress);
        let _ = self.prefetch.settle(&mut self.core.directory, &mut self.ledger);
        for listed in self.core.take_past(progress) {
            listed.release(&mut self.scratch.conveyor);
        }
        // The tiers it is past place their seeds, which it is past too.
        self.scratch.advance(progress, &mut self.core);
        self.core.directory.drain_staging(progress, &self.core.steps);
        self.core.retire(progress);
    }

    /// A later scratch window's round 0 arrived: settle finished landings and
    /// free the conveyor records whose converts have completed, without moving
    /// device progress backwards across steps in the set.
    pub(super) fn advance_window(&mut self) {
        let _ = self.prefetch.settle(&mut self.core.directory, &mut self.ledger);
        for listed in self.core.take_converted() {
            listed.release(&mut self.scratch.conveyor);
        }
        self.refill_tier();
    }

    /// Keep each group's reserve of tier records holding no expert mapped
    /// (`HostRecords::reserved`): the victims chosen from the ranking with
    /// no lock of the steps' held
    /// (`Directory::refill_victims`), then, where no step begins, each
    /// unmapped, unchanged since (`Directory::refill`), graced for the
    /// steps begun; a lent record only while no scratch-served group's
    /// lease is out (from its begin until it retires, `Steps::lending`,
    /// read again under the steps' lock), so none is graced for a group's
    /// own step. Under the steps' lock: a check, a map store and a state
    /// change a victim.
    pub(super) fn refill_tier(&mut self) -> bool {
        let steps = &self.shared.steps;
        let directory = &mut self.core.directory;
        if !directory.refill_short() {
            return false;
        }
        let victims = directory.refill_victims(steps.lending());
        if victims.iter().all(Vec::is_empty) {
            return false;
        }
        let mut unmapped = false;
        steps.unbegun(|at| {
            unmapped = directory.refill(victims, at);
        });
        // A grace for steps the device is already past frees now: no
        // frontier's move would come for it until the next step.
        unmapped | directory.retire_graces(self.prefetch.reached(), steps)
    }

    /// Take `layer`'s scratch requests published since, in step order,
    /// reading what each needs as its records free; none while the conveyor
    /// still clears a group past (`Scratch::clearing`), whose clearing is an
    /// event. True when it took any.
    pub(super) fn publish(&mut self, layer: usize) -> bool {
        // The device's requests are in place only while it is parked, where
        // parks are submission boundaries (`Parks::parked`).
        if !self.shared.parks.parked() {
            return false;
        }
        let layers = self.core.layers.len();
        let mut took = false;
        while let Some((place, at)) = self.scratch.published(&self.core, &self.shared, layer, layers) {
            // A request is its step's front's, which follows its step's back
            // at the layer before (the group's steps before it there too):
            // the device is past them, or past the groups before at the
            // first layer. The back's point is its pass's piece's, read now:
            // at a layer a group was cut in before, the front runs before the
            // group cut in, but past the back before.
            let Point { group, sequence, .. } = at.point;
            self.advance(match layer {
                0 => Point { sequence: group, ..at.point },
                _ => self.shared.steps.point(group, layer - 1, sequence + 1),
            });
            if self.scratch.clearing() {
                break;
            }
            self.scratch.publish(at, place, &mut self.core, &self.shared);
            took = true;
        }
        took
    }

    /// Publish the experts each layer holds to the size rule, and those its
    /// group's host tier holds alone, where they changed since last
    /// published: otherwise it counts nothing and takes no lock.
    pub(super) fn publish_held(&mut self) {
        let holdings = self.core.directory.holdings();
        if self.published.holdings != Some(holdings) {
            let Published { held, alone, .. } = &mut self.published;
            for layer in 0..self.core.layers.len() {
                held[layer] = self.core.directory.held_count(layer);
                alone[layer] = self.core.directory.alone(layer);
            }
            self.shared.telemetry.publish_held(held, alone);
            self.published.holdings = Some(holdings);
        }
        // Each group's lend and capacity, published only as a resize, a
        // claim or a return changes them.
        let lend = self.core.directory.hosts().iter().map(|host| super::GroupLend { claimed: host.claimed(), capacity: host.capacity() });
        if !lend.clone().eq(self.published.lend.iter().copied()) {
            self.published.lend = lend.collect();
            self.shared.publish_lend(&self.published.lend);
        }
        let directory = &self.core.directory;
        let capacities = (0..self.published.capacities.len()).map(|group| directory.capacity(group));
        if !capacities.clone().eq(self.published.capacities.iter().copied()) {
            self.published.capacities = capacities.collect();
            self.shared.telemetry.capacity(self.published.capacities.clone());
        }
        // The moment the records misses land into first keep every expert:
        // from then no decode miss reads the file. They are a host tier's
        // where the device places, the bank's own (the one store) where the
        // host does.
        let complete = self.core.directory.tier_complete();
        if complete && !self.complete {
            tracing::info!(
                "{} keeps every expert, {:.1} s after the bank opened",
                self.store(),
                self.opened.elapsed().as_secs_f64()
            );
        }
        self.complete = complete;
    }

    /// The store misses land into: a host tier's where the device places,
    /// the bank's own (the one store) where the host does.
    fn store(&self) -> &'static str {
        match self.core.extents.placer {
            crate::planning::Placer::Tier => "the host tier",
            crate::planning::Placer::Bank => "the bank",
        }
    }

    /// Stage the write of each layer's policy words a run's encoding reads
    /// where they changed, written between flights (`Shared::stage_policy`)
    /// (`BankPart::Policy`, this their one writer; its line words are
    /// [`Self::publish_lines`]'): its home (`Directory::home`, the same every
    /// flight, which a graph shared by blocks reads through each block's
    /// own policy port), its whole word, set
    /// while its group is whole and at home, no expert of it able to miss
    /// (`Directory::group_whole`: on the one store, its records the bank's,
    /// while it keeps every expert, as a bank imported whole there does) and
    /// each at its home record (`Directory::group_at_home`), its records
    /// every one its reservation holds (`Core::group_resident`): its route then
    /// writes no routing record, so the ranking is fed nothing of it and
    /// starts blank after a shrink,
    /// so a prompt group of whole layers is resident and lends nothing; and
    /// the group that lends the conveyor its records only while every group
    /// is (a partial group's prompt demand claims the lender's records
    /// mid-flight,
    /// `Directory::lease`). Nothing misses in a whole group, so no
    /// placement, seed or claim takes its records before the next flight
    /// ends, and every run encoded and launched until then reads the word
    /// alike. The write runs only while no step runs, under the lock every
    /// step begins under (`Shared::write_staged`), and a step begins before
    /// its runs are encoded, so no encoding overlaps a publication; what the
    /// load reads of the words (which layers are whole, the gates in effect)
    /// changes with the write. A placement
    /// counts once it holds its record: a group's word turns on a flight
    /// late, never early. Its resolved word is set while it is whole or its
    /// group's host tier keeps every expert outside the conveyor's lent
    /// records (`HostRecords::settled`): no route of its can find an
    /// expert absent, so its landing launch is not encoded. The tier gives up a
    /// kept expert mid-flight only to land one it lacks, which a complete
    /// tier never does, or to a prompt group's claim of a lent record,
    /// which a settled tier keeps none in; otherwise only in a shrink
    /// between flights, after which this publishes.
    ///
    /// Then each layer's hints word, whether its hint launches run
    /// (`policy[POLICY_HINTS]`, read where a run is queued), priced on the
    /// load's evidence now (`gate::HintGate::decide`). Each change of any
    /// word breaks the pipeline of queued steps once, so the hint gate moves
    /// only where that pays, and words equal to those written stage nothing,
    /// dropping any write staged before (`Shared::stage_policy`): a change
    /// undone before it is written breaks nothing.
    pub(super) fn publish_policy(&mut self) {
        let layer_count = self.core.layers.len();
        let directory = &self.core.directory;
        // Per-layer residency at home: every expert `0..E` of `layer` is
        // currently kept at its home record `home + e` (so every slab in
        // `[home .. home + E)` is backed and holds its expert weight).
        let at_home = (0..layer_count).map(|layer| directory.layer_at_home(layer)).collect::<Vec<_>>();
        let every = at_home.iter().all(|&home| home);
        let lender = self.scratch.conveyor.lending().group;
        let lent = match self.core.extents.placer {
            crate::planning::Placer::Bank => directory.host_lent(lender) as usize,
            crate::planning::Placer::Tier => directory.lender().filter(|&(group, _)| group == lender).map_or(0, |(_, count)| count),
        };
        // When any layer is not resident (`!every`), a prompt group may lease
        // `[0 .. lent)` of `lender` mid-flight. Only layers whose home range
        // overlaps `[0 .. lent)` (i.e. `home(layer) < lent` in `lender`) are
        // exposed to that lease; all other layers in `lender` are disjoint.
        let mut whole = (0..layer_count)
            .map(|layer| at_home[layer] && (every || self.core.layers[layer].group != lender || directory.home(layer) >= lent))
            .collect::<Vec<_>>();
        // Whole layers are protected from mid-flight eviction while
        // `POLICY_WHOLE == 1`. When a group is partially whole, ensure the
        // non-whole records outside `[0 .. lent)` still leave enough evictable
        // headroom so mid-flight demand/refill/promotions/seeds never starve:
        // - `Placer::Bank`: `reserve + 2 * K` (tier reserve + cut-in pool + one
        //   decode step's choices) beyond `lent`.
        // - `Placer::Tier`: at least `K` bank records beyond `lent`.
        let need = self.core.extents.selected;
        for group in 0..self.core.groups.len() {
            let members: Vec<usize> = (0..layer_count)
                .filter(|&layer| self.core.layers[layer].group == group)
                .collect();
            if members.iter().all(|&layer| whole[layer]) {
                continue;
            }
            let (total_records, lent_in_group, min_evictable) = match self.core.extents.placer {
                crate::planning::Placer::Bank => (
                    directory.host(group).records() as usize,
                    directory.host_lent(group) as usize,
                    directory.host(group).reserve() + 2 * need,
                ),
                crate::planning::Placer::Tier => (
                    directory.records()[group],
                    directory.lender().filter(|&(g, _)| g == group).map_or(0, |(_, count)| count),
                    need.max(1),
                ),
            };
            let usable = total_records.saturating_sub(lent_in_group);
            let mut whole_records: usize = members
                .iter()
                .filter(|&&layer| whole[layer])
                .map(|&layer| self.core.layers[layer].experts)
                .sum();
            for &layer in members.iter().rev() {
                if usable.saturating_sub(whole_records) >= min_evictable {
                    break;
                }
                if whole[layer] {
                    whole[layer] = false;
                    whole_records -= self.core.layers[layer].experts;
                }
            }
        }
        let served = self.core.layers.iter().enumerate().any(|(index, layer)| layer.site.is_some() && !whole[index]);
        // `POLICY_RESOLVED` may only be set when the layer is `whole` (protected
        // in Tier 0 at home) or its group's entire host tier is settled outside
        // `0..lent` (so no layer in the group can miss the tier and trigger
        // mid-flight tier eviction). A single unprotected layer in a partial
        // pooled tier can still lose experts mid-flight and must keep
        // `POLICY_RESOLVED == 0` so `routed_land` and `routed_hint_route` remain
        // active.
        let resolved = self
            .core
            .layers
            .iter()
            .zip(&whole)
            .map(|(layer, &whole)| whole || directory.group_tier_settled(layer.group))
            .collect::<Vec<_>>();
        // Keep `POLICY_RESOLVED` strictly as the `routed_land` / file-exchange
        // omission bit, while gating `HintGate` on `&whole` (`!all_groups_whole`)
        // so 100%-RAM discrete GPUs keep foresight hints active for Link staging.
        let link_line = self
            .core
            .directory
            .tier_complete()
            .then(|| self.shared.copy.lock().expect("a copy's price").as_ref().map(|c| c.transfer))
            .flatten();
        let core = &self.core;
        let ledger = &mut self.ledger;
        let layer_count = core.layers.len();
        let hints = self.shared.telemetry.hint_gates(&whole, link_line, self.shared.launch_prices().as_ref(), |target, lead, miss| {
            let bytes = core.layers[target].expert_bytes();
            let idle = miss.layer_idle(layer_count);
            ledger.hint_admitted(target, lead, miss.break_even(bytes, idle))
        });
        let layers = (0..self.core.layers.len())
            .map(|index| LayerPolicy {
                whole: whole[index],
                resolved: resolved[index],
                home: self.homes[index],
                hints: hints[index],
            })
            .collect();
        let words = PolicyWords { layers, served };
        let (shared, layers) = (Arc::downgrade(&self.shared), &self.core.layers);
        self.shared.stage_policy(words, || {
            let policies = layers.iter().map(|layer| layer.policy.clone()).collect::<Vec<_>>();
            Box::new(move |words: &PolicyWords| {
                let shared = shared.upgrade().expect("a load's staged policy is written while it serves");
                shared.sites_served.store(words.served, std::sync::atomic::Ordering::Release);
                for (index, (policy, layer)) in policies.iter().zip(&words.layers).enumerate() {
                    shared.telemetry.whole(index, layer.whole);
                    policy.store(constants::POLICY_WHOLE as usize, i32::from(layer.whole));
                    policy.store(constants::POLICY_RESOLVED as usize, i32::from(layer.resolved));
                    policy.store(constants::POLICY_HOME as usize, layer.home);
                    policy.store(constants::POLICY_HINTS as usize, i32::from(layer.hints));
                }
                shared.telemetry.publish_hints(&words.layers.iter().map(|layer| layer.hints).collect::<Vec<_>>());
            })
        });
    }

    /// Write each layer's line words (`ExpertLines::policy`, this the one
    /// writer) from its lines as the load's own work refines them (its
    /// landings, what its CPU site's requests take, what its sampled steps'
    /// blocks take on the device and the moves beside its reads,
    /// `Telemetry::refined`), and price the promotions' copies on the same
    /// lines beside the decode's CPU site and reads in place as its passes
    /// measure them now (`Telemetry::beside`; `Directory::price_copies`
    /// asks them again only where a price moved), and the releases'
    /// write-backs on the passes' terms (`Directory::price_write_backs`). The words are stored
    /// only where they differ from those last published, so a turn that
    /// moved none of them takes no steps' lock. Unlike the words a run's encoding reads (whole, resolved, home:
    /// [`Self::publish_policy`]), the lines decide
    /// only the route's choice of a miss's site, and any value of each word
    /// is a valid one for any step: so they are stored under the lock every
    /// step begins under (`Steps::unbegun`), as a map entry is
    /// (`experts::map`), at any turn, the lock held for the stores alone.
    /// Every step from the first not yet begun reads them, and one begun
    /// that has not routed the layer yet may, as a mapping may take effect
    /// from such a step: word by word, some of a layer's words old and
    /// some new. A decode that never idles is repriced so.
    pub(super) fn publish_lines(&mut self, copier: Option<&Copier>) {
        let priced = self.shared.priced().map(|lines| self.shared.telemetry.refined(&lines));
        let copy = self.shared.copy.lock().expect("a copy's price").clone();
        if let (Some(lines), Some(copy)) = (&priced, &copy) {
            let beside = self.shared.telemetry.beside();
            let terms = self.shared.telemetry.terms();
            for (layer, lines) in lines.iter().enumerate() {
                self.core.directory.price_copies(layer, crate::assessment::copy_hits(lines, copy, beside));
                // A write-back holds the link its record's transfer out of
                // the device, on the transfer queue's line that way.
                let link = copy.back.transfer_seconds(lines.host.record_bytes as f64);
                let price = terms.map(|(terms, drive)| crate::experts::directory::WriteBack { terms, drive, link });
                self.core.directory.price_write_backs(layer, price);
            }
        }
        // Continuous shared-domain bandwidth share (`DomainBudget`) across
        // live PCIe DMA + live NVMe landing bytes vs Host DDR5 (`HostLines::memory`),
        // replacing the binary `dma_active: bool` hack.
        let dma_active = copier.is_some_and(|copier| !copier.copies.idle()) || self.core.directory.staging_in_flight();
        let landing_active = self.core.extents.placer == crate::planning::Placer::Tier && self.prefetch.landing_active();
        let b_dma = if dma_active {
            copy.as_ref().map_or(0.0, |c| {
                let spb = c.transfer.seconds_per_byte();
                if spb > 0.0 { 1.0 / spb } else { 0.0 }
            })
        } else {
            0.0
        };
        let b_landing = if landing_active {
            self.shared.telemetry.drive_line().map_or(0.0, |d| {
                let spb = d.seconds_per_byte();
                if spb > 0.0 { 1.0 / spb } else { 0.0 }
            })
        } else {
            0.0
        };
        let words = self
            .core
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let cpu = priced.as_ref().and_then(|lines| {
                    let layer_lines = &lines[index];
                    layer_lines.host.cpu.map(|cpu| {
                        let b_mem = layer_lines
                            .host
                            .memory
                            .filter(|m| m.get() > 0.0)
                            .map_or(0.0, |m| layer_lines.host.record_bytes as f64 / (m.get() * 1e-9));
                        cpu.with_domain_budget(DomainBudget::new(b_dma, b_landing, b_mem))
                    })
                });
                let site = match (&layer.site, &cpu) {
                    (None, _) => SiteLines::Unsited,
                    (Some(_), None) => SiteLines::Pending,
                    (Some(_), Some(cpu)) => SiteLines::Measured(cpu),
                };
                ExpertLines::policy(&crate::assessment::POLICY_LAYOUT, site, priced.as_ref().map(|lines| &lines[index]))
            })
            .collect::<Vec<_>>();
        if self.lines.as_ref() == Some(&words) {
            return;
        }
        let layers = &self.core.layers;
        self.shared.steps.unbegun(|_| {
            for (layer, words) in layers.iter().zip(&words) {
                for &(word, value) in words {
                    layer.policy.store(word, value);
                }
            }
        });
        self.lines = Some(words);
    }

    /// What residency holds and moves, for the exchange's watchdog:
    /// staging and the reads and landings in flight, the promotions, the
    /// scratch tiers and conveyor, and per layer the experts its bank holds
    /// and those its group's host tier holds alone.
    pub(super) fn describe(&self) -> String {
        let layers = (0..self.core.layers.len())
            .map(|layer| (self.core.directory.held_count(layer), self.core.directory.alone(layer)))
            .collect::<Vec<_>>();
        format!(
            "{}; promotions {}; {}; tier keeps every expert {}; retired through {}; records taken {}; resize asked {}; \
             per layer (bank holds, tier holds alone) {layers:?}",
            self.prefetch.describe(),
            self.core.directory.describe_promotions(),
            self.scratch.describe(),
            self.complete,
            self.core.retired,
            self.core.taken.len(),
            self.asked.is_some(),
        )
    }

    /// Whether every group is whole in Tier 0 (`Directory::all_groups_whole`):
    /// false on 100%-RAM discrete GPUs so `Requests::polls` keeps reading
    /// foresight rings for Link staging, true once VRAM holds every expert.
    pub(super) fn all_whole(&self) -> bool {
        self.core.directory.all_groups_whole()
    }

    /// Each group's host tier records `views` address from now (a shrink's,
    /// between flights, or a growth's joined slot): the landings' tier
    /// tensors and the CPU site's, which read the same records.
    pub(super) fn address_tier(&mut self, views: Vec<Vec<crate::experts::source::Records>>) {
        self.owed.deferred_records.extend(self.shared.compute.address(&views));
        self.core.address_tier(views, &mut self.owed.deferred_records);
    }

    /// What decode serving reads and writes of it (`decode::Serving`).
    pub(super) fn serving(&mut self) -> decode::Serving<'_> {
        decode::Serving {
            core: &mut self.core,
            prefetch: &mut self.prefetch,
            ledger: &mut self.ledger,
            shared: &self.shared,
        }
    }
}
