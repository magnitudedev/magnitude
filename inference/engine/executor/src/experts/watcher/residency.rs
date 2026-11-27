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
use magnitude_cost_model::{ExpertLines, SiteLines};
use crate::experts::directory::{MissCurve, Point};
use crate::experts::interest::{Launched, Ledger, Told};
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

impl PolicyWords {
    /// Fingerprint of per-layer `(POLICY_WHOLE, POLICY_RESOLVED)` flags for
    /// `horizon::ResidencyState`.
    pub(super) fn residency_fingerprint(&self) -> u64 {
        crate::experts::horizon::policy_fingerprint(
            self.layers.iter().map(|layer| (layer.whole, layer.resolved)),
        )
    }
}

/// A streamed layer's policy words: whole, resolved, and its home.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LayerPolicy {
    whole: bool,
    resolved: bool,
    home: i32,
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
    /// Whether the one-shot initial priced `publish_lines` summary has been logged.
    pub(super) logged_initial_lines: bool,
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
    Held::new(residency.hold(POISONED), true)
}

/// [`lock`] without publishing layer-held telemetry on drop: used by hot
/// per-layer decode demand operations so `publish_held` runs once per step
/// (at step completion) rather than 40x per step.
pub(super) fn lock_deferred(residency: &Yielding<Residency>) -> Held<'_> {
    Held::new(residency.hold(POISONED), false)
}

/// [`lock`], `counted` run once the holder is counted waiting
/// ([`Yielding::hold_counted`]): from then the keeper gives way to it at
/// its next job's end, so what `counted` does (the request service's look,
/// `decode::Look`) runs beside a job the keeper holds, never before one it
/// starts.
pub(super) fn lock_counted(residency: &Yielding<Residency>, counted: impl FnOnce()) -> Held<'_> {
    Held::new(residency.hold_counted(POISONED, counted), true)
}

/// Why residency's lock is never poisoned ([`lock`]).
const POISONED: &str = "residency is never left by a panic";

/// Residency held ([`lock`]): as it is released, what each layer holds is
/// published under the lock (when `publish_on_drop` is set), then the lock and
/// `NoBlockingAllocationGuard` are released, any evicted/replaced
/// `Tensor`/`Records` views are dropped outside the lock, and finally what
/// waits on that publication (`Owed`) is answered.
pub(super) struct Held<'a> {
    guard: Option<MutexGuard<'a, Residency>>,
    no_alloc: Option<seismic::memory::NoBlockingAllocationGuard>,
    publish_on_drop: bool,
}

impl<'a> Held<'a> {
    fn new(held: MutexGuard<'a, Residency>, publish_on_drop: bool) -> Self {
        Self {
            guard: Some(held),
            no_alloc: Some(seismic::memory::enter_no_blocking_allocation()),
            publish_on_drop,
        }
    }

    /// Given first to every holder waiting for it (the keeper's, between
    /// its jobs: [`Yielding::give_way`]), released as every holder is.
    pub(super) fn give_way(self, residency: &'a Yielding<Residency>) -> Self {
        let publish = self.publish_on_drop;
        residency.give_way(self, POISONED, |held| Self::new(held, publish))
    }

    /// Yield a foreground holder (`Requests::serve`) once when another
    /// foreground holder (e.g. `DecodeReactor`) is waiting, re-acquiring as a
    /// foreground holder so `Residency::keep` can never starve it.
    pub(super) fn yield_to_waiters(self, residency: &'a Yielding<Residency>) -> Self {
        let publish = self.publish_on_drop;
        residency.yield_to_waiters(self, POISONED, |held| Self::new(held, publish))
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
        if settle && self.publish_on_drop {
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
    /// A resize ask was answered on this turn (so `Residency::keep` should run
    /// one immediate follow-up turn without parking to drain any `Told` or
    /// traffic work deferred behind the resize).
    answered_resize: bool,
}

/// Outcome of a single [`Residency::turn`] pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TurnOutcome {
    pub(super) answered_resize: bool,
}

/// RAII turn guard for [`Residency::turn`]: guarantees
/// [`Residency::publish_liveness`] runs unconditionally on every exit path
/// (normal completion, resize early return, or `resizing.is_pending()` early
/// return) before `Held` releases residency's lock and settles `Owed`.
struct TurnGuard<'a, 'c> {
    held: Option<Held<'a>>,
    copier: Option<&'c Copier>,
}

impl<'a, 'c> TurnGuard<'a, 'c> {
    fn new(held: Held<'a>, copier: Option<&'c Copier>) -> Self {
        Self { held: Some(held), copier }
    }

    fn give_way(mut self, residency: &'a Yielding<Residency>) -> Self {
        let held = self.held.take().expect("TurnGuard live").give_way(residency);
        Self { held: Some(held), copier: self.copier }
    }
}

impl std::ops::Deref for TurnGuard<'_, '_> {
    type Target = Residency;
    fn deref(&self) -> &Residency {
        self.held.as_deref().expect("TurnGuard live")
    }
}

impl std::ops::DerefMut for TurnGuard<'_, '_> {
    fn deref_mut(&mut self) -> &mut Residency {
        self.held.as_deref_mut().expect("TurnGuard live")
    }
}

impl Drop for TurnGuard<'_, '_> {
    fn drop(&mut self) {
        if let Some(held) = self.held.take() {
            if !std::thread::panicking() && held.shared.life.ended().is_none() {
                held.publish_liveness(self.copier);
            }
            drop(held);
        }
    }
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
            let mut answered_resize = false;
            if shared.life.ended().is_none() {
                // Prioritize pending `ResizeHandoff` requests over background
                // telemetry feeding so bank shrinks never stall behind a
                // `feed_stack` batch.
                let decoded = if shared.resizing.is_pending() {
                    None
                } else {
                    shared.telemetry.feed_while(|| {
                        !shared.resizing.is_pending()
                            && !shared.life.stopping()
                            && shared.life.ended().is_none()
                    });
                    if shared.resizing.is_pending() {
                        None
                    } else {
                        copier.as_ref().and_then(|_| shared.telemetry.decoded_since(&mut decoded_steps))
                    }
                };
                answered_resize = Self::tend(&residency, copier.as_ref(), decoded.as_ref()).answered_resize;
            }
            if !shared.resizing.is_pending() && !answered_resize {
                events.wait(seen);
            }
        }
    }

    /// Dedicated background thread for the ranking's true-route shadows
    /// ([`super::telemetry::Telemetry::shadow_while`]), keeping shadow
    /// simulation completely off `expert-residency` so `ResizeHandoff`,
    /// `Join`s, `Retire`, `LinkMover`, and `Prefetch` never block behind a
    /// shadow backlog.
    pub(super) fn keep_shadows(shared: Arc<Shared>) {
        let events = shared.telemetry.shadow_events().clone();
        loop {
            let seen = events.seen();
            if shared.life.stopping() {
                return;
            }
            if shared.life.ended().is_none() {
                shared.telemetry.shadow_while(|| !shared.life.stopping() && shared.life.ended().is_none());
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
    fn tend(residency: &Yielding<Self>, copier: Option<&Copier>, decoded: Option<&MissCurve>) -> TurnOutcome {
        Self::turn(residency, &Self::TURN, copier, decoded)
    }

    /// A turn of `jobs` ([`Self::tend`]'s, [`Self::TURN`]).
    /// Uses [`TurnGuard`] so [`Self::publish_liveness`] runs unconditionally
    /// on every exit path, including early returns from resize or pending
    /// resize preemption.
    pub(super) fn turn(residency: &Yielding<Self>, jobs: &[Job<'_>], copier: Option<&Copier>, decoded: Option<&MissCurve>) -> TurnOutcome {
        let mut guard = TurnGuard::new(Held::new(residency.background(POISONED), true), copier);
        let mut turn = Turn::default();
        for job in jobs {
            if !job(&mut guard, &mut turn, copier, decoded) {
                break;
            }
            if guard.shared.resizing.is_pending() {
                break;
            }
            guard = guard.give_way(residency);
            if guard.shared.resizing.is_pending() {
                break;
            }
        }
        TurnOutcome { answered_resize: turn.answered_resize }
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
        while !self.shared.resizing.is_pending() {
            let Ok(mut event) = self.inbox.try_recv() else {
                break;
            };
            told = true;
            let completed = matches!(&event, Told::Completed { .. });
            let began_step = match &event {
                Told::Began { step, .. } => Some(*step),
                _ => None,
            };
            let decode_began = match &event {
                Told::Began { step, launch: Launched::Decode { rows, .. } } => Some((*step, rows.len().max(1))),
                _ => None,
            };
            let ended_step = match &event {
                Told::Ended { step, .. } => Some(*step),
                _ => None,
            };
            let draft = matches!(&event, Told::DraftWindow { .. });
            let draft_tokens = matches!(&event, Told::DraftTokens { .. });
            let closed_request = match &event {
                Told::Closed(request) => Some(*request),
                _ => None,
            };
            // Account zero-park landed prefetch hits and on-time LinkMover
            // `MAP_STAGE` hits against the actual routed experts of every
            // completed step, and on 100%-RAM discrete GPUs (`link_only`)
            // classify each choice for `Ledger::resolve`.
            if let Told::Completed { step, ref mut chose, physical_rows, ref layer_misses, .. } = event {
                let engagement = self.shared.telemetry.engagement();
                for (layer, choices) in chose.iter_mut().enumerate() {
                    let layer_bytes = self.core.layers[layer].expert_bytes();
                    let (gpu_host_misses, _) =
                        layer_misses.get(layer).copied().unwrap_or((0, 0));
                    let stack = self.core.stack(layer);
                    let group = self.core.directory.layer_groups()[stack];
                    self.core.directory.observe_demand_misses(
                        group,
                        gpu_host_misses as usize,
                        physical_rows as usize,
                    );
                    for chosen in choices.iter_mut() {
                        if let Some(lp) = self.prefetch.take_landed_prefetch((stack, chosen.expert)) {
                            engagement.record_landed_hit(lp.bytes, lp.carryover);
                        }
                        let link_landed = self.ledger.has_link_landed(step, layer, chosen.expert);
                        if classify_link_stage_hit(chosen, link_landed, link_only) {
                            engagement.record_staged_hit(layer_bytes, false);
                        }
                    }
                }
                self.prefetch.prune_evicted_prefetches(&mut self.core, &self.shared);
            }
            let _ = self.ledger.tell(event);
            if began_step.is_some() || draft_tokens {
                let core_layers = &self.core.layers;
                self.core.directory.set_token_reads(
                    self.ledger
                        .token_protected()
                        .iter()
                        .map(|&((layer, expert), step, p)| ((core_layers[layer].stack, expert), step, p)),
                );
            }
            if let Some(step) = began_step {
                if let Some((step, m_active)) = decode_began {
                    self.core.directory.set_decode_rows(m_active);
                    self.prefetch.decode_step_began(step, &self.shared);
                }
                self.prefetch.on_began(step);
            } else if draft {
                self.shared.telemetry.engagement().on_draft_window(true, Instant::now());
                self.prefetch.on_draft_window();
            }
            if let Some(step) = ended_step {
                self.prefetch.step_ended(step, &self.shared);
            }
            let _ = completed;
            if let Some(request) = closed_request {
                self.core.directory.clear_token_reads();
                self.shared.telemetry.engagement().on_draft_window(false, Instant::now());
                self.prefetch.prune_evicted_prefetches(&mut self.core, &self.shared);
                self.prefetch.on_request_closed();
                let site_served = self.shared.compute.site_served();
                let mut residency = self.shared.telemetry.residency();
                residency.vram_held = (0..self.core.directory.layers())
                    .map(|stack| self.core.directory.tier0_held_count(stack) as u64)
                    .sum();
                residency.vram_capacity = self
                    .core
                    .directory
                    .capacities()
                    .into_iter()
                    .map(|cap| cap as u64)
                    .sum();
                residency.vram_reserve = (0..self.core.groups.len())
                    .map(|group| self.core.directory.reserved(group) as u64)
                    .sum();
                if let Some(line) = self.shared.telemetry.engagement().request_summary_line(
                    site_served.requests,
                    site_served.experts,
                    residency,
                ) {
                    tracing::info!("expert engagement (request {}): {line}", request.0);
                }
                if let Some(line) = self.shared.telemetry.take_route_stats_summary(residency) {
                    tracing::info!("route stats (request {}): {line}", request.0);
                }
            }
        }
        if !self.shared.resizing.is_pending() {
            let (judged, accepted) = self.ledger.take_judged();
            if !judged.is_empty() || !accepted.is_empty() {
                self.shared.telemetry.judged(&judged, &accepted);
            }
            Prefetch::spend(&mut self.ledger, &self.shared);
        }
        told
    }

    fn serve_resize(&mut self, asked: Asked, turn: &mut Turn, copier: Option<&Copier>) -> bool {
        self.asked = self.resize(asked, copier);
        turn.answered_resize = self.asked.is_none() && self.shared.life.ended().is_none();
        false
    }

    fn learn(&mut self, turn: &mut Turn, copier: Option<&Copier>, _: Option<&MissCurve>) -> bool {
        if let Some(reason) = self.shared.fault() {
            self.shared.end_serving(reason);
            return false;
        }
        // Serve any pending resize immediately before draining `Told` events
        // or refilling tier reserves (`Self::resize` already runs `join`,
        // `retire`, `link.settle`, and `prefetch.settle`, and `publish_resized`
        // runs `refill_tier`), so `plan_picked_micros` never stalls behind a
        // `learn_told` or `telemetry.judged` backlog.
        if let Some(asked) = self.asked.take().or_else(|| self.shared.resizing.asked().map(Asked::taken)) {
            return self.serve_resize(asked, turn, copier);
        }
        // Drain `Told` events (`DraftWindow`, `Began`, `Completed`) before
        // `promote` (`LinkMover::advance`) so Path 0 immediately sees fresh
        // draft windows and completed steps on the same turn.
        let told = self.learn_told();
        if let Some(asked) = self.shared.resizing.asked().map(Asked::taken) {
            return self.serve_resize(asked, turn, copier);
        }
        let joined = self.join();
        let retired = self.retire();
        let (landed, newly_landed) = self.prefetch.settle(&mut self.core.directory, &mut self.ledger, &self.shared);
        let refilled = self.refill_tier();
        // Pipeline chain-through: if Path 1 just landed an expert into Tier 1
        // that is live in `NameLedger` for an upcoming step, mark changed so
        // `promote` (`LinkMover::advance`) immediately stages it into Tier 0.
        let chain_through = newly_landed.into_iter().any(|(layer, expert)| self.ledger.is_wanted(layer, expert));
        turn.changed = told || joined || retired || landed || refilled || chain_through;
        if let Some(asked) = self.shared.resizing.asked().map(Asked::taken) {
            return self.serve_resize(asked, turn, copier);
        }
        true
    }

    /// Unified Path 0 (`LinkMover`): write-backs, `NameLedger`-driven
    /// opportunistic DMA staging, and `Reuse`-ranked permanent promotions.
    fn promote(&mut self, turn: &mut Turn, copier: Option<&Copier>, decoded: Option<&MissCurve>) -> bool {
        let held = self.traffic_held();
        let drive_in_flight = self.prefetch.in_flight();
        let progressed = copier.is_some_and(|copier| {
            self.link.advance(
                &mut self.core,
                &mut self.ledger,
                &self.shared,
                copier,
                decoded,
                held,
                self.prefetch.reached(),
                drive_in_flight,
            )
        });
        turn.changed |= progressed;
        true
    }

    /// A prompt group past cleared from the conveyor.
    fn clear(&mut self, turn: &mut Turn, _: Option<&Copier>, _: Option<&MissCurve>) -> bool {
        turn.changed |= self.scratch.clear_past(&mut self.core, &self.shared);
        true
    }

    /// Drive Path 1 (`Prefetch`), cover reads, and prompt-group hastening.
    fn hasten(&mut self, turn: &mut Turn, _: Option<&Copier>, _: Option<&MissCurve>) -> bool {
        let held = self.traffic_held();
        let paused = held || self.asked.is_some() || self.shared.resizing.is_pending();
        if paused {
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
            let stack_visits = self.core.stack_visits.clone();
            self.prefetch.cover(&mut self.core, &self.shared, |staged| {
                stack_visits[staged.0].iter().any(|&visit| scratch.carries((visit, staged.1)))
            });
            for staged in self.scratch.awaited(&self.core) {
                self.prefetch.hasten((self.core.stack(staged.0), staged.1));
            }
        }
        true
    }

    /// Single authority for publishing residency's liveness state (host-tier
    /// cover reading state and idle traffic-quiet state), notifying `events`
    /// on any state transition. Invoked unconditionally by [`TurnGuard::drop`]
    /// on every exit of [`Self::turn`].
    pub(super) fn publish_liveness(&self, copier: Option<&Copier>) {
        self.shared.publish_cover(self.prefetch.covering(&self.core));
        let held = self.traffic_held();
        let copying = copier.is_some_and(|copier| !copier.copies.idle());
        let quiet = held && !copying && !self.prefetch.outstanding();
        if self.shared.traffic_quiet.swap(quiet, std::sync::atomic::Ordering::AcqRel) != quiet {
            self.shared.events().notify();
        }
    }

    /// The words a run's encoding reads, only while no step runs; the
    /// lines at every turn; and the turn's change announced.
    fn publish_turn(&mut self, turn: &mut Turn, _: Option<&Copier>, _: Option<&MissCurve>) -> bool {
        self.publish_policy();
        self.shared.write_staged();
        self.publish_lines();
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
        self.core.directory.drain_pending(frontier);
        self.core.directory.drain_staging_with(self.prefetch.reached(), &self.core.steps, |group, slot| {
            self.shared
                .transport
                .try_release_hint_slot(crate::experts::transport::DestSlot { group, slot })
        });
        if let Some(past) = self.shared.steps.retire_groups(frontier) {
            self.advance(past);
        }
        true
    }

    /// Read live hint and front rings while any group is not whole in Tier 0
    /// (`!core.directory.all_groups_whole()`), unblocking foresight on
    /// 100%-RAM discrete GPUs as well as 2-tier and 3-tier workloads.
    /// Immediately drives Path 0 `LinkMover::advance` when `copier` is present
    /// so `HINT_LEADS=2` hints dispatch PCIe DMA within microseconds instead
    /// of waiting for `Residency::keep`'s background `TURN`.
    pub(super) fn rings(&mut self, shared: &Arc<Shared>, served: &[Step], mut freed: bool, copier: Option<&Copier>) -> bool {
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
        let paused = self.traffic_held() || self.asked.is_some() || shared.resizing.is_pending();
        if paused {
            self.prefetch.cancel_cover();
        }
        let (settled, newly_landed) = self.prefetch.settle(&mut self.core.directory, &mut self.ledger, shared);
        let chain_through = !paused && newly_landed.into_iter().any(|(layer, expert)| self.ledger.is_wanted(layer, expert));
        let refilled = self.refill_tier();
        let reads_finished = self.prefetch.take_reads_finished();
        freed |= settled || refilled || chain_through || reads_finished;
        // Admit newly read rings into `self.ledger` (and drive Path 1 when not
        // paused for a traffic hold or pending/waiting resize) BEFORE
        // `LinkMover::advance` so Path 0 sees fresh `HINT_LEADS` hints on the
        // same tick rather than one tick late.
        let finished = if paused {
            self.prefetch.finish_rings_paused(&mut self.core, &mut self.ledger, shared, read, served);
            settled || refilled || reads_finished
        } else {
            self.prefetch.finish_rings(&mut self.core, &mut self.ledger, shared, read, freed, served)
        };
        let drive_in_flight = self.prefetch.in_flight();
        let link_progressed = copier.is_some_and(|copier| {
            self.link.advance(
                &mut self.core,
                &mut self.ledger,
                shared,
                copier,
                None,
                paused,
                self.prefetch.reached(),
                drive_in_flight,
            )
        });
        finished || link_progressed
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
        let _ = self.prefetch.settle(&mut self.core.directory, &mut self.ledger, &self.shared);
        for listed in self.core.take_past(progress) {
            listed.release(&mut self.scratch.conveyor);
        }
        // The tiers it is past place their seeds, which it is past too.
        self.scratch.advance(progress, &mut self.core, &self.shared);
        self.core.directory.drain_staging_with(progress, &self.core.steps, |group, slot| {
            self.shared
                .transport
                .try_release_hint_slot(crate::experts::transport::DestSlot { group, slot })
        });
        self.core.retire(progress);
    }

    /// A later scratch window's round 0 arrived: settle finished landings and
    /// free the conveyor records whose converts have completed, without moving
    /// device progress backwards across steps in the set.
    pub(super) fn advance_window(&mut self) {
        let _ = self.prefetch.settle(&mut self.core.directory, &mut self.ledger, &self.shared);
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
        let victims = directory.refill_victims_cached(steps.lending());
        if victims.iter().all(Vec::is_empty) {
            return false;
        }
        let evicted_records: Vec<(usize, usize)> = victims
            .iter()
            .enumerate()
            .flat_map(|(group, records)| records.iter().map(move |&(record, _)| (group, record as usize)))
            .collect();
        let tier_copy_dma = self.core.extents.tier_copy_dma;
        let mut unmapped = false;
        steps.unbegun(|at| {
            let immediate = tier_copy_dma && at.all_scratch();
            unmapped = directory.refill_with(victims, at, immediate);
        });
        if unmapped {
            for (group, tier_record) in evicted_records {
                self.shared.transport.bump_source_generation(group, tier_record);
            }
        }
        // A grace for steps the device is already past frees now: no
        // frontier's move would come for it until the next step.
        unmapped | directory.retire_graces(self.prefetch.reached(), steps)
    }

    /// Take `layer`'s scratch requests published since, in step order,
    /// reading what each needs as its records free; none while the conveyor
    /// still clears a group past (`Scratch::clearing`), whose clearing is an
    /// event. Host visibility is enforced per layer by
    /// `Parks::scratch_visible_end` inside `Scratch::published`. True when it
    /// took any.
    pub(super) fn publish(&mut self, layer: usize) -> bool {
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
        let first_publish = self.published.holdings.is_none();
        let holdings = self.core.directory.holdings();
        if self.published.holdings != Some(holdings) {
            let Published { held, alone, .. } = &mut self.published;
            for layer in 0..self.core.layers.len() {
                let stack = self.core.stack(layer);
                held[layer] = self.core.directory.held_count(stack);
                alone[layer] = self.core.directory.alone(stack);
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
        if complete != self.complete || first_publish {
            self.complete = complete;
            self.shared.time.lock().update_residency(|state| {
                state.tier_complete = complete;
            });
        }
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
        let at_home = (0..layer_count)
            .map(|layer| directory.layer_at_home(self.core.stack(layer)))
            .collect::<Vec<_>>();
        let every = at_home.iter().all(|&home| home);
        let lender = self.scratch.conveyor.lending().group;
        let lent = match self.core.extents.placer {
            crate::planning::Placer::Bank => directory.host_lent(lender) as usize,
            crate::planning::Placer::Tier => directory.lender().filter(|&(group, _)| group == lender).map_or(0, |(_, count)| count),
        };
        // When any layer is not resident (`!every`), a prompt group may lease
        // `[0 .. lent)` of `lender` mid-flight. Only layers whose home range
        // overlaps `[0 .. lent)` (i.e. `home(stack) < lent` in `lender`) are
        // exposed to that lease; all other layers in `lender` are disjoint.
        let mut whole = (0..layer_count)
            .map(|layer| at_home[layer] && (every || self.core.layers[layer].group != lender || directory.home(self.core.stack(layer)) >= lent))
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
            let group_stacks: Vec<usize> = (0..self.core.stack_visits.len())
                .filter(|&stack| self.core.layers[self.core.stack_rep(stack)].group == group)
                .collect();
            if group_stacks.iter().all(|&stack| whole[self.core.stack_rep(stack)]) {
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
            let mut whole_records: usize = group_stacks
                .iter()
                .filter(|&&stack| whole[self.core.stack_rep(stack)])
                .map(|&stack| self.core.layers[self.core.stack_rep(stack)].experts)
                .sum();
            for &stack in group_stacks.iter().rev() {
                if usable.saturating_sub(whole_records) >= min_evictable {
                    break;
                }
                let rep = self.core.stack_rep(stack);
                if whole[rep] {
                    for &visit in &self.core.stack_visits[stack] {
                        whole[visit] = false;
                    }
                    whole_records -= self.core.layers[rep].experts;
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
        let layers = (0..self.core.layers.len())
            .map(|index| LayerPolicy {
                whole: whole[index],
                resolved: resolved[index],
                home: self.homes[index],
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
                }
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
    pub(super) fn publish_lines(&mut self) {
        let priced = self.shared.priced().map(|lines| self.shared.telemetry.refined(&lines));
        let copy = self.shared.copy.lock().expect("a copy's price").clone();
        if let (Some(lines), Some(copy)) = (&priced, &copy) {
            let beside = self.shared.telemetry.beside();
            let terms = self.shared.telemetry.terms();
            for (layer, lines) in lines.iter().enumerate() {
                let stack = self.core.stack(layer);
                if layer != self.core.stack_rep(stack) {
                    continue;
                }
                self.core.directory.price_copies(stack, crate::assessment::copy_hits(lines, copy, beside));
                // A write-back holds the link its record's transfer out of
                // the device, on the transfer queue's line that way.
                let link = copy.back.transfer_seconds(lines.host.record_bytes as f64);
                let price = terms.map(|(terms, drive)| crate::experts::directory::WriteBack { terms, drive, link });
                self.core.directory.price_write_backs(stack, price);
            }
        }
        // A layer whose misses the transport lands by DMA beside the step
        // (`DecodeFront -> DMA -> DecodeBack`) reads them from the bank, so
        // its link line is the copy's ([`decode_dma_lines`]). The CPU's line
        // is the site's own: as its requests took on this load once it
        // served one (`Telemetry::refined`), the job's before; the
        // assignment weighs it against the GPU's at those prices alone.
        let all_groups_whole = self.core.directory.all_groups_whole();
        let has_decode_dma = !all_groups_whole && self.shared.transport.has_decode_dma();
        for group in 0..self.core.groups.len() {
            let active = has_decode_dma
                && self
                    .core
                    .layers
                    .iter()
                    .any(|layer| layer.group == group && !layer.draft);
            self.core.directory.set_group_splitting(group, active);
        }
        let mut initial_log = None;
        let words = self
            .core
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let layer_decode_dma = has_decode_dma && !layer.draft;
                let adjusted = priced.as_ref().map(|lines| decode_dma_lines(lines[index], copy.as_ref(), layer_decode_dma));
                let site = match (&layer.site, adjusted.as_ref().and_then(|lines| lines.host.cpu.as_ref())) {
                    (None, _) => SiteLines::Unsited,
                    (Some(_), None) => SiteLines::Pending,
                    (Some(_), Some(cpu)) => SiteLines::Measured(cpu),
                };
                let mut policy_words = ExpertLines::policy(&crate::assessment::POLICY_LAYOUT, site, adjusted.as_ref());
                let ce_min = self.shared.transport.tier_ce_min_experts(index) as i32;
                policy_words.push((magnitude_kernels::constants::POLICY_CE_MIN_EXPERTS as usize, ce_min));
                if index == 0 && !self.logged_initial_lines && adjusted.is_some() {
                    let word = |of: usize| policy_words.iter().find_map(|&(word, value)| (word == of).then_some(value));
                    let layout = &crate::assessment::POLICY_LAYOUT;
                    initial_log = Some((layer_decode_dma, ce_min, word(layout.link), word(layout.link_contended), word(layout.cpu)));
                }
                policy_words
            })
            .collect::<Vec<_>>();
        if let Some((unified_dma, ce_min, link_ns, link_contended_ns, cpu_ns)) = initial_log {
            self.logged_initial_lines = true;
            tracing::info!("expert pricing (layer 0): unified_dma={unified_dma} POLICY_CE_MIN_EXPERTS={ce_min} POLICY_LINK_NS={link_ns:?} POLICY_LINK_CONTENDED_NS={link_contended_ns:?} POLICY_CPU_NS={cpu_ns:?}");
        }
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
            .map(|layer| {
                let stack = self.core.stack(layer);
                (self.core.directory.held_count(stack), self.core.directory.alone(stack))
            })
            .collect::<Vec<_>>();
        format!(
            "{}; promotions {}; {}; {}; tier keeps every expert {}; retired through {}; records taken {}; resize asked {}; \
             per layer (bank holds, tier holds alone) {layers:?}",
            self.prefetch.describe(),
            self.core.directory.describe_promotions(),
            self.scratch.describe(),
            self.scratch.describe_tables(&self.core),
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

    /// Whether every layer is resolved (`POLICY_RESOLVED == 1`: either whole in
    /// VRAM or 100% resident in `HostTier`, so decode never emits NVMe
    /// `FILE_REQUEST`s).
    pub(super) fn all_resolved(&self) -> bool {
        self.core.directory.all_groups_whole() || self.core.directory.tier_complete()
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

/// Return whether `chosen` actually hit an on-time `LinkMover` `MAP_STAGE`
/// slot (`!chosen.saves && link_landed`) and, on 100%-RAM discrete GPUs
/// (`link_only`), classify `chosen` for `Ledger::resolve` so on-time `MAP_STAGE`
/// hits credit `HintGate` and calibration while late or un-staged Host-RAM
/// misses (already DMA'd by `DecodeFront` / `land_decode_misses`) are neither
/// double-counted as `staged_hits` nor miscounted as saved hints.
fn classify_link_stage_hit(chosen: &mut crate::experts::interest::Chosen, link_landed: bool, link_only: bool) -> bool {
    let staged_hit = !chosen.saves && link_landed;
    if link_only {
        if staged_hit {
            chosen.saves = true;
            chosen.tier = true;
        } else if chosen.saves && chosen.tier {
            chosen.tier = false;
        }
    }
    staged_hit
}

/// Layer `lines` as the assignment prices them beside `copy`:
/// - When `copy` measured concurrent `DMA ∥ CPU` host-RAM throughput
///   (`copy.concurrent_host_byte_seconds`), record the per-record joint bus
///   time (`lines.host.joint_memory`, clamped at `lines.host.memory`) so
///   [`ExpertLines::policy`] and [`ExpertLines::serve`] account for shared
///   `HostRam` contention.
/// - Where the transport lands the layer's misses by DMA beside the step
///   (`layer_decode_dma`), a miss is read from the bank once its copy lands,
///   so its link line is the copy's transfer plus a resident read where that
///   is cheaper than reading in place.
/// - On non-splitting discrete-tier layers (`!layer_decode_dma`, e.g. draft
///   layers), shared `HostRam` bus contention is still governed by the
///   hardware PCIe DMA link: when DMA-priced contention suppresses CPU-site
///   (`effective_cpu().is_none()`), clear `lines.host.cpu` so CPU-site is
///   suppressed on all discrete-tier layers rather than falling back to
///   contended CPU-site against slower SM zero-copy.
fn decode_dma_lines(mut lines: ExpertLines, copy: Option<&crate::assessment::TierCopyCosts>, layer_decode_dma: bool) -> ExpertLines {
    let Some(copy) = copy else {
        return lines;
    };
    if let (Some(mem), Some(joint_bs)) = (lines.host.memory, copy.concurrent_host_byte_seconds) {
        let joint_seconds = lines.host.record_bytes as f64 * joint_bs;
        if joint_seconds.is_finite() && joint_seconds > 0.0 {
            let joint_ns = magnitude_cost_model::Nanoseconds::saturating(joint_seconds);
            lines.host.joint_memory = Some(joint_ns.max(mem));
        }
    }
    let dma_ns = copy.transfer.transfer_seconds(lines.host.record_bytes as f64) * 1e9;
    let contended_dma_ns = copy
        .concurrent_dma_byte_seconds
        .map(|bs| lines.host.record_bytes as f64 * bs * 1e9)
        .filter(|ns| ns.is_finite() && *ns > 0.0)
        .map(|ns| ns.max(dma_ns));
    let mut dma_priced = lines;
    let zc_ns = copy
        .routed_zerocopy_byte_seconds
        .filter(|bs| bs.is_finite() && *bs > 0.0)
        .map_or(lines.gpu.link.weights.get(), |bs| {
            (lines.host.record_bytes as f64 * bs * 1e9).max(lines.gpu.link.weights.get())
        });
    if zc_ns > lines.gpu.link.weights.get() {
        if let Ok(ns) = magnitude_cost_model::Nanoseconds::try_from(zc_ns.round() as u32) {
            lines.gpu.link.weights = ns;
            dma_priced.gpu.link.weights = ns;
        }
    }
    let landed_ns = dma_ns + lines.gpu.resident.weights.get();
    let uses_dma_link = landed_ns.is_finite() && landed_ns > 0.0 && landed_ns < zc_ns;
    if uses_dma_link {
        if let Ok(ns) = magnitude_cost_model::Nanoseconds::try_from(landed_ns.round() as u32) {
            dma_priced.gpu.link.weights = ns;
            if let Ok(dma) = magnitude_cost_model::Nanoseconds::try_from(dma_ns.round() as u32) {
                dma_priced.gpu.baseline_dma = Some(match dma_priced.gpu.baseline_dma {
                    Some(prev) => prev.min(dma),
                    None => dma,
                });
            }
        }
    }
    if let Some(contended_dma_ns) = contended_dma_ns {
        let contended_ns = if uses_dma_link {
            contended_dma_ns + dma_priced.gpu.resident.weights.get()
        } else {
            dma_priced.gpu.link.weights.get() + (contended_dma_ns - dma_ns).max(0.0)
        };
        let contended_ns = contended_ns.max(dma_priced.gpu.link.weights.get());
        dma_priced.gpu.contended_link = Some(magnitude_cost_model::Nanoseconds::saturating(contended_ns * 1e-9));
        let in_place_contended = (lines.gpu.link.weights.get() + (contended_dma_ns - dma_ns).max(0.0))
            .max(lines.gpu.link.weights.get());
        lines.gpu.contended_link = Some(magnitude_cost_model::Nanoseconds::saturating(in_place_contended * 1e-9));
    }
    if layer_decode_dma {
        return dma_priced;
    }
    if dma_priced.effective_cpu().is_none() {
        lines.host.cpu = None;
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::{classify_link_stage_hit, decode_dma_lines};
    use crate::experts::horizon::Time;
    use crate::experts::interest::{Chosen, Ledger, Stands};
    use std::time::Instant;

    #[test]
    fn late_map_stage_does_not_double_count_staged_hits() {
        // Case 1: `MAP_STAGE` won the race against `resolve` (`source >= 0` ->
        // `Use::Bank`, so `saves == false, tier == false`). Must credit
        // `staged_hits` and, on `link_only`, mark `(saves=true, tier=true)`
        // so `Ledger::resolve` credits `HintGate` / calibration.
        let mut on_time = Chosen { expert: 7, saves: false, tier: false };
        assert!(classify_link_stage_hit(&mut on_time, true, true));
        assert_eq!(on_time, Chosen { expert: 7, saves: true, tier: true });

        // Case 2: `MAP_STAGE` lost the race against `resolve` (`source = -2 - t`
        // -> `Use::Tier`, so `saves == true, tier == true`), and `DecodeFront` /
        // `land_decode_misses` already DMA'd the expert (`req_dma_experts += 1`)
        // before the late `LinkMover` copy settled (`link_landed == true`).
        // Must NOT credit `staged_hits`, and on `link_only` must reclassify to
        // `tier == false` so `Ledger::resolve` counts it as a live Link miss.
        let mut late_stage = Chosen { expert: 8, saves: true, tier: true };
        assert!(!classify_link_stage_hit(&mut late_stage, true, true));
        assert_eq!(late_stage, Chosen { expert: 8, saves: true, tier: false });

        // Case 3: Un-staged Host-RAM miss on `link_only`.
        let mut unstaged = Chosen { expert: 9, saves: true, tier: true };
        assert!(!classify_link_stage_hit(&mut unstaged, false, true));
        assert_eq!(unstaged, Chosen { expert: 9, saves: true, tier: false });

        // Case 4: Permanent VRAM hit (`link_landed == false`).
        let mut resident = Chosen { expert: 10, saves: false, tier: false };
        assert!(!classify_link_stage_hit(&mut resident, false, true));
        assert_eq!(resident, Chosen { expert: 10, saves: false, tier: false });

        // Case 5: 3-tier (`link_only == false`) late stage must not credit
        // `staged_hits` and must leave Drive `tier == true` untouched.
        let mut three_tier_late = Chosen { expert: 11, saves: true, tier: true };
        assert!(!classify_link_stage_hit(&mut three_tier_late, true, false));
        assert_eq!(three_tier_late, Chosen { expert: 11, saves: true, tier: true });
    }

    #[test]
    fn has_link_landed_only_reports_link_mover_stages_not_drive_hints() {
        let time = Time::new(&[vec![4, 8]], 16);
        let stands = Stands::new(time.clone());
        let mut ledger = Ledger::new(vec![0, 0], 4, time, stands);
        let now = Instant::now();

        // Only `link_landed` sets `has_link_landed`.
        assert!(!ledger.has_link_landed(5, 0, 3));
        ledger.link_landed(5, 0, 3, now, now);
        assert!(ledger.has_link_landed(5, 0, 3));
        assert!(!ledger.has_link_landed(5, 0, 4));
    }

    /// A layer whose misses land by DMA reads them from the bank: its link
    /// line is the copy's transfer plus a resident read (a 3 MB expert at
    /// 48 GB/s, 62.5 µs, beside 15 µs resident), where that undercuts the
    /// read in place (270 µs); a layer that reads in place, or whose read
    /// in place is the cheaper, keeps its lines, and the CPU's line is
    /// never touched.
    #[test]
    fn a_dma_landed_layer_is_priced_at_its_copy() {
        use magnitude_cost_model::{CpuLines, ExpertLines, GpuLines, HostLines, Moves, Nanoseconds, PassLine, ReadLine, RowCost, RowLine};
        let ns = |v: u32| Nanoseconds::try_from(v).unwrap();
        let lines = ExpertLines {
            gpu: GpuLines {
                link: PassLine { weights: ns(270_000), row: ns(1_000) },
                resident: PassLine { weights: ns(15_000), row: ns(1_000) },
                moves: Moves::NONE,
                baseline_dma: None,
                contended_link: None,
            },
            host: HostLines {
                cpu: Some(CpuLines { expert: RowLine { one: ns(75_000), row: RowCost::Measured(ns(1_000)) }, handshake: ns(4_000) }),
                memory: Some(ns(40_000)),
                joint_memory: None,
                landing: None,
                record_bytes: 3_000_000,
            },
        };
        let line = ReadLine::new(0.0, 1.0 / 48e9).unwrap();
        let copy = crate::assessment::TierCopyCosts { blocks: 0, whole_seconds: 0.0, stall_share: 0.0, transfer: line, back: line, concurrent_host_byte_seconds: None, concurrent_dma_byte_seconds: None, routed_zerocopy_byte_seconds: None };
        let landed = decode_dma_lines(lines, Some(&copy), true);
        assert_eq!(landed.gpu.link.weights.get() as u64, 77_500);
        assert_eq!(landed.host, lines.host);
        assert_eq!(decode_dma_lines(lines, Some(&copy), false), lines);
        assert_eq!(decode_dma_lines(lines, None, true), lines);
        let near = ExpertLines { gpu: GpuLines { link: PassLine { weights: ns(70_000), ..lines.gpu.link }, ..lines.gpu }, ..lines };
        assert_eq!(decode_dma_lines(near, Some(&copy), true), near);

        // Measured concurrent `DMA ∥ CPU` contention populates `host.joint_memory`
        // for both split-DMA and in-place layers and suppresses CPU when `B_joint <= B_link`.
        let gen5_line = ReadLine::new(0.0, 1.0 / 43.6e9).unwrap();
        let gen5_copy = crate::assessment::TierCopyCosts {
            blocks: 0,
            whole_seconds: 0.0,
            stall_share: 0.0,
            transfer: gen5_line,
            back: gen5_line,
            concurrent_host_byte_seconds: Some(1.0 / 42.0e9),
            concurrent_dma_byte_seconds: None,
            routed_zerocopy_byte_seconds: None,
        };
        let gen5_lines = ExpertLines {
            gpu: GpuLines {
                link: PassLine { weights: ns(270_000), row: ns(1_000) },
                resident: PassLine { weights: ns(2_000), row: ns(1_000) },
                moves: Moves::NONE,
                baseline_dma: None,
                contended_link: None,
            },
            host: HostLines {
                cpu: Some(CpuLines { expert: RowLine { one: ns(75_000), row: RowCost::Measured(ns(1_000)) }, handshake: ns(4_000) }),
                memory: Some(Nanoseconds::of(3_000_000.0 / 47.8e9).unwrap()),
                joint_memory: None,
                landing: None,
                record_bytes: 3_000_000,
            },
        };
        let adjusted = decode_dma_lines(gen5_lines, Some(&gen5_copy), true);
        assert_eq!(adjusted.host.joint_memory, Some(Nanoseconds::of(3_000_000.0 / 42.0e9).unwrap()));
        assert_eq!(adjusted.effective_cpu(), None, "Gen5 + 2-ch DDR5 (B_joint=42 GB/s < B_link=43.6 GB/s) must suppress CPU-site");
        let in_place_adjusted = decode_dma_lines(gen5_lines, Some(&gen5_copy), false);
        assert_eq!(in_place_adjusted.host.joint_memory, adjusted.host.joint_memory);
        assert_eq!(
            in_place_adjusted.effective_cpu(),
            None,
            "Gen5 bus contention must also suppress CPU-site on non-splitting (draft) layers"
        );
    }

    #[test]
    fn gen5_effective_cpu_none_suppresses_cpu_site_on_both_split_target_and_non_split_draft_layers() {
        use magnitude_cost_model::{
            CpuLines, ExpertLines, GpuLines, Hits, HostLines, Miss, Moves, Nanoseconds, Pass, PassLine, ReadLine, RowCost, RowLine, Site, SiteLines,
        };
        let ns = |v: u32| Nanoseconds::try_from(v).unwrap();
        // Gen5 2-ch DDR5: B_dma = 43.6 GB/s, B_joint = 42.0 GB/s, B_mem = 47.8 GB/s.
        let gen5_transfer = ReadLine::new(0.0, 1.0 / 43.6e9).unwrap();
        let gen5_copy = crate::assessment::TierCopyCosts {
            blocks: 0,
            whole_seconds: 0.0,
            stall_share: 0.0,
            transfer: gen5_transfer,
            back: gen5_transfer,
            concurrent_host_byte_seconds: Some(1.0 / 42.0e9),
            concurrent_dma_byte_seconds: None,
            routed_zerocopy_byte_seconds: None,
        };
        // DSpark draft layer: 18.84 MB/expert MXFP4, SM zero-copy link ~1.7 ms, resident ~40 µs.
        let dspark_bytes = 18_840_000u64;
        let draft_lines = ExpertLines {
            gpu: GpuLines {
                link: PassLine { weights: ns(1_700_000), row: ns(1_000) },
                resident: PassLine { weights: ns(40_000), row: ns(1_000) },
                moves: Moves::NONE,
                baseline_dma: None,
                contended_link: None,
            },
            host: HostLines {
                cpu: Some(CpuLines {
                    expert: RowLine { one: ns(450_000), row: RowCost::Measured(ns(2_000)) },
                    handshake: ns(10_000),
                }),
                memory: Some(Nanoseconds::of(dspark_bytes as f64 / 47.8e9).unwrap()),
                joint_memory: None,
                landing: None,
                record_bytes: dspark_bytes,
            },
        };
        let misses = vec![Miss { rows: 1., ready_seconds: 0. }; 8];
        let hits = [Hits { experts: 2., rows: 1. }];

        for layer_decode_dma in [true, false] {
            let adjusted = decode_dma_lines(draft_lines, Some(&gen5_copy), layer_decode_dma);
            assert_eq!(
                adjusted.effective_cpu(),
                None,
                "layer_decode_dma={layer_decode_dma}: Gen5 must suppress CPU-site"
            );
            let served = adjusted.serve(Pass::Decode, &hits, &misses);
            let cpu_assigned = served.sites.iter().filter(|&&s| s == Site::Cpu).count();
            assert_eq!(
                cpu_assigned,
                0,
                "layer_decode_dma={layer_decode_dma}: cpu_site must be 0"
            );
            let site = match adjusted.host.cpu.as_ref() {
                Some(cpu) => SiteLines::Measured(cpu),
                None => SiteLines::Pending,
            };
            let policy: std::collections::BTreeMap<_, _> =
                ExpertLines::policy(&crate::assessment::POLICY_LAYOUT, site, Some(&adjusted)).into_iter().collect();
            assert_eq!(
                policy[&crate::assessment::POLICY_LAYOUT.priced], 0,
                "layer_decode_dma={layer_decode_dma}: POLICY_PRICED must be 0 when CPU-site is suppressed"
            );
        }

        // Gen4 3090 (B_dma = 19.3 GB/s, B_joint = 36.9 GB/s): ample HostRam headroom keeps CPU-site enabled.
        let gen4_transfer = ReadLine::new(0.0, 1.0 / 19.3e9).unwrap();
        let gen4_copy = crate::assessment::TierCopyCosts {
            blocks: 0,
            whole_seconds: 0.0,
            stall_share: 0.0,
            transfer: gen4_transfer,
            back: gen4_transfer,
            concurrent_host_byte_seconds: Some(1.0 / 36.9e9),
            concurrent_dma_byte_seconds: None,
            routed_zerocopy_byte_seconds: None,
        };
        for layer_decode_dma in [true, false] {
            let adjusted = decode_dma_lines(draft_lines, Some(&gen4_copy), layer_decode_dma);
            assert!(
                adjusted.effective_cpu().is_some(),
                "layer_decode_dma={layer_decode_dma}: Gen4 3090 must keep CPU-site enabled"
            );
        }

        // Real DDR4 contention on Gen4 (3060-32g): uncontended DMA 20 GB/s (150 µs),
        // contended DMA 11 GB/s (272.7 µs), aggregate B_joint = 26 GB/s (> 20 GB/s),
        // CPU 320 µs -> decode_dma_lines wires concurrent_dma_byte_seconds into
        // contended_link and serve() keeps all 4 misses on GPU (C = 0) because every
        // contended split is slower than 4 * 155 µs uncontended.
        let ddr4_transfer = ReadLine::new(0.0, 1.0 / 20.0e9).unwrap();
        let ddr4_copy = crate::assessment::TierCopyCosts {
            blocks: 0,
            whole_seconds: 0.0,
            stall_share: 0.0,
            transfer: ddr4_transfer,
            back: ddr4_transfer,
            concurrent_host_byte_seconds: Some(1.0 / 26.0e9),
            concurrent_dma_byte_seconds: Some(1.0 / 11.0e9),
            routed_zerocopy_byte_seconds: None,
        };
        let ddr4_lines = ExpertLines {
            gpu: GpuLines {
                link: PassLine { weights: ns(350_000), row: ns(100) },
                resident: PassLine { weights: ns(5_000), row: ns(100) },
                moves: Moves::NONE,
                baseline_dma: None,
                contended_link: None,
            },
            host: HostLines {
                cpu: Some(CpuLines {
                    expert: RowLine { one: ns(320_000), row: RowCost::Measured(ns(1_000)) },
                    handshake: ns(10_000),
                }),
                memory: Some(Nanoseconds::of(3_000_000.0 / 30.0e9).unwrap()),
                joint_memory: None,
                landing: None,
                record_bytes: 3_000_000,
            },
        };
        let adjusted = decode_dma_lines(ddr4_lines, Some(&ddr4_copy), true);
        assert_eq!(adjusted.gpu.link.weights.get() as u64, 155_000);
        let contended_expected = (3_000_000.0_f64 / 11.0e9 * 1e9 + 5_000.0).round() as u64;
        assert_eq!(adjusted.gpu.contended_link.unwrap().get() as u64, contended_expected);
        let served = adjusted.serve(Pass::Decode, &[], &vec![Miss { rows: 1., ready_seconds: 0. }; 4]);
        assert_eq!(served.sites.iter().filter(|&&s| s == Site::Cpu).count(), 0, "{served:?}");
        assert_eq!(served.sites.iter().filter(|&&s| s == Site::Link).count(), 4, "{served:?}");
    }
}
