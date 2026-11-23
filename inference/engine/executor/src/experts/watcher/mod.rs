//! The host's serving of a streamed load's running steps. Every streamed entry of every step runs
//! in submission order on one queue, so a decode request of step `q` at
//! layer `l`, or a scratch request's round 0, proves the device past every
//! earlier layer of `q` and every earlier step (a scratch request itself,
//! in its step's front, only past its own and the earlier steps' backs at
//! the layer before): the directory retires the fills placed before it,
//! and the staging they were converted from is free.
//!
//! - A decode route (`routed_route_streamed`) serves every miss its group's
//!   host tier holds itself, from the tier. The tier map it resolves
//!   through is residency's mirror of what the tier records keep whole
//!   (`HostRecords`, its one writer); the map it resolves its hits through is
//!   the host's alone, step-stamped (`experts::map`). In the background
//!   residency promotes the tier's experts the ranking keeps into the bank
//!   on the copy engine (`promotions`).
//! - A decode request names the choices neither the bank nor the tier
//!   holds. The request service lands each absent expert into a ready tier
//!   record of the layer's group (`experts::landing`): revived from
//!   staging, by its hint's read, or read now into staging, the landing
//!   converting it as its read ends, mapping it, and answering each choice
//!   that named it, all without waiting for a read. The landing launch
//!   waits for each choice's answer (`parks`) and sources it from the tier
//!   (`routed_land`). The tier keeps a reserve of ready records,
//!   unmapping its least valuable kept ones ahead of need, each reused only
//!   once every step that could have resolved it has retired.
//! - A scratch request (`routed_scratch_request`) flags the experts a
//!   larger launch routes to, from its chain place's region. A prompt group
//!   runs as a wavefront: a step's front at a layer follows its back at the
//!   layer before, and every request of a layer comes before any of its
//!   rounds there. Each is taken as it is published, in step order: of the
//!   experts no earlier step of the group routed there, the ones the bank
//!   lacks are read into the conveyor as their records free (at once, or
//!   once the layer before converted what they held), and the bank's are
//!   read where it holds them. Each is served in step order once its first
//!   round arrives, the device then past the steps before it: each round
//!   lists conveyed (or staged) records for the convert, half the staging
//!   at a time.
//!
//! Serving is split by owner, each waiting on its own events (`events`),
//! none on another's job:
//!
//! - **Residency** (`residency`): the one owner of the directory and its
//!   map writes, the host tier, staging and landings, the conveyor, the
//!   promotions and the policy words, behind one lock no holder waits
//!   under. Its own thread retires, settles landings, promotes, clears a
//!   prompt group's conveyor, hastens the reads ahead a prompt group's step
//!   waits for, joins growths, serves resizes
//!   and writes the policy words (those a run's encoding reads in a gap,
//!   the line words wherever they changed), on the load's events alone.
//! - **Request service** (`requests`): the one owner of noticing the
//!   device's requests and answering them, each request waiting on its own
//!   slot of the exchange. It keeps the one poll of serving, where kernels
//!   wait in place and nothing announces the device's words.
//! - **The transfer queue's issuer and watch** (`Copies::start`) and, where
//!   parks are submission boundaries, **the parks' keeper** (`parks`): the
//!   threads that call the device's queues for serving and turn its
//!   completions into events. Serving's owners post batches and never call
//!   the device's queue themselves: an issue can wait (on host access to
//!   its tensors, or in the driver behind a queue the parked step fills),
//!   and none of them may wait under residency's lock.
//!
//! Every request the device waits on is answered, by its step or by its
//! failure, while the host lives (streaming_words.seismic: a spinning wait
//! ends on its own only once the host is gone). A
//! read that fails with an I/O error fails the request that named its
//! record: every waiting launch of the request returns, nothing it read is
//! committed, and its step fails (only that step's sequences); a hint's
//! read no request named fails nothing. A panic while serving, in either
//! thread or a read worker reading for it, fails every request from then on
//! and every step's completion, so the load fails instead of hanging. Steps
//! begin and complete through `Shared`, whose facts serving touches only
//! briefly, never across a wait.
//!
//! Each request that reads experts is counted against its layer and its
//! step ([`ExpertCosts`]): what the performance model takes from a run.
//!
//! Each module owns one responsibility's state:
//!
//! - `core`: the source, directory, layers and groups, and the records taken
//!   until the device is past them;
//! - `residency`: the owner of `core` and of everything below that moves a
//!   record, and its thread;
//! - `requests`: the request service, its thread and its one poll;
//! - `prefetch`: staging and the reads into it, each landing into the tier:
//!   requests' and the hints' ahead of them;
//! - `decode` and `scratch`: the two kinds of request, the scratch tier and
//!   the conveyor, each serving over the core and the prefetch;
//! - `promotions`: the host tier's experts copied into the bank;
//! - `resize`: the bank's resize between flights;
//! - `liveness`: the end of serving, and the exchange's watchdog over the
//!   load, its bound and what each owner reports;
//! - `parks`: the device's waits where they are submission boundaries
//!   (Vulkan), which the host answers;
//! - `telemetry`: what serving costs, under one lock;
//! - `steps`: the steps begun, their groups and their failures.

mod core;
mod decode;
mod gate;
mod link;
mod liveness;
mod parks;
mod passes;
mod prefetch;
mod requests;
mod residency;
mod resize;
mod routes;
mod scratch;
mod steps;
mod telemetry;
#[cfg(test)]
mod tests;

pub(in crate::experts) use self::core::{
    At, Extents, Group, Layer, ListRecords, LoadWords, PlacedLayerWords, PlacedWords, Pool, StoredPart, StoredTensor,
};
pub(crate) use parks::{ParkKind, ParkSite, Parks};
pub(crate) use passes::{DEPTHS, Measured};
pub(in crate::experts) use link::Copier;
pub(crate) use resize::{Candidates, Evicted, Evicting, Join};
pub(crate) use steps::{AheadOf, DecodeRow, ExpertStep, ScratchGroup, Recording, Step, StepFailure, StepRoute, low_word};
pub use routes::{RecordedRoutes, RoutedStep};
pub use telemetry::{ExpertCosts, GroupCost, GroupLayer, LayerTimeline, Span, StepTimeline, LayerCost, LayerExperts, StepCost, STEP_TRACE};

/// A group's host records as a prompt group's lend leaves them: the lent
/// records it holds claimed, and the records decode keeps experts in
/// (`HostRecords::capacity`: those outside the reserve, less the claimed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct GroupLend {
    pub claimed: usize,
    pub capacity: usize,
}

use self::core::{Core, ReadTable};
use super::conveyor::{Carrier, Conveyor};
use super::directory::{Directory, HostTier};
use super::source::ExpertSource;
use magnitude_cost_model::ExpertLines;
use liveness::Life;
use link::LinkMover;
use prefetch::Prefetch;
use requests::Requests;
use residency::Residency;
use resize::{Joins, ResizeHandoff};
use scratch::Scratch;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;
pub(in crate::experts) use steps::{MapFrom, MapStep, ParkFrom, Steps, Unbegun, UnmapStep};
use telemetry::Telemetry;

/// The threads serving keeps busy while a flight runs, by their measured
/// live use (an RTX 3090, a 16.5 ms decode step): the request service's,
/// which polls the device's requests where kernels wait in place (9.7 ms a
/// step), and residency's (3.9 ms). The transfer queue's issuer and watch
/// and the watchdog wait on the load's events or the device, each turn a
/// short burst.
pub(in crate::experts) const THREADS: usize = 2;

/// What steps and serving share, each part its module's.
pub(super) struct Shared {
    /// Shared with every answer, which writes under its step's ownership of
    /// its slot (`Steps::while_owned`).
    steps: Arc<Steps>,
    /// The load's token clock and turnovers (`Directory::time`).
    time: crate::experts::horizon::Time,
    /// The verify decode step whose verdicts are still pending on the clock,
    /// with `(step, total_requests, remaining_requests, accepted_rows)`: only
    /// its verdicts (or discards, as 0 accepted) call `Times::decided`, and
    /// the last verdict reports the step's actual emitted tokens to telemetry.
    pending_verify: Mutex<Option<(Step, u32, u32, u64)>>,
    telemetry: Telemetry,
    /// What other threads tell the interest ledger, which `Prefetch` owns
    /// on the request service's thread and drains before it reads the
    /// rings (`experts::interest::Told`): no thread waits on the ledger.
    told: std::sync::mpsc::Sender<crate::experts::interest::Told>,
    resizing: ResizeHandoff,
    /// The holds of the load's own traffic standing (`Shared::hold_traffic`,
    /// each a [`TrafficHold`]): while any stands no promotion or read ahead
    /// begins.
    traffic_held: std::sync::atomic::AtomicUsize,
    /// Whether, as residency last saw, no promotion's copy nor read ahead
    /// of its runs ([`Shared::traffic_quiet`]).
    traffic_quiet: std::sync::atomic::AtomicBool,
    /// Whether a growth's slot is being backed ([`Shared::backing_slot`]):
    /// traffic of the load's own, as its copies and reads are.
    backing: std::sync::atomic::AtomicBool,
    /// A running growth's slots, each joined as it is backed.
    joins: Joins,
    life: Life,
    /// The landings' account over the load, logged as the watcher stops.
    landings: Arc<crate::experts::landing::LandingAccount>,
    /// The host compute executor's handle: landings are submitted to it,
    /// and decode steps issue its site flights.
    pub(super) compute: crate::experts::Compute,
    /// Whether a decode step's route can assign the CPU anything: some
    /// layer with a site is not whole, as residency last published
    /// (`Residency::publish_policy`, while no step runs). Only then does a
    /// decode step issue a site flight, whose workers spin.
    pub(super) sites_served: std::sync::atomic::AtomicBool,
    /// The device's waits where they are submission boundaries, which each
    /// run's parks register with and every answer settles.
    pub(crate) parks: Arc<Parks>,
    /// Each streamed layer's lines once the executor's costs priced the
    /// GPU's (`Shared::price`), which the route's line words are written
    /// from (`Residency::publish_lines`); none before.
    priced: Mutex<Option<Arc<[ExpertLines]>>>,
    /// The promotions asked since the load, the directory's count
    /// (`Directory::asked`) as residency last published it.
    promoted: std::sync::atomic::AtomicU64,
    /// What a copy over the link costs, once measured
    /// (`assessment::TierCopyCosts`): what a promotion's copy is priced at.
    copy: Mutex<Option<crate::assessment::TierCopyCosts>>,
    /// Each group's lend and decode capacity as residency last published
    /// them ([`Self::lend`]).
    lend: Mutex<Vec<GroupLend>>,
    /// The cover's changes as residency published them
    /// (`Residency::hasten`): each start and each end of its reads counts
    /// one, from reading as the bank opens ([`Self::cover`]).
    covers: std::sync::atomic::AtomicU64,
    /// What a routed pass over each streamed layer's experts costs, once
    /// priced (`Self::price`): what a scratch-served group's merge sets are
    /// chosen by (`Shared::begin`).
    passes: Mutex<Option<Arc<magnitude_cost_model::ExpertPass>>>,
    /// The load's launch prices once priced (`Self::price`): what the front
    /// gates price their values by (`Residency::publish_policy`).
    prices: Mutex<Option<magnitude_cost_model::LaunchPrices>>,
    /// Whether the load's account was given as it settled.
    settled: crate::settle::Once,
    /// Unified copy-engine-first `ExpertTransport` shared across `LinkMover`,
    /// `Scratch`, `Copies`, and `ExpertBank`.
    pub(crate) transport: Arc<crate::experts::transport::ExpertTransport>,
    /// The policy words a run's encoding reads, as last written, and the
    /// write residency staged since ([`Self::stage_policy`]), written only
    /// while no step runs.
    policy: Mutex<Policy>,
}

/// The policy words a run's encoding reads ([`Shared::stage_policy`]): the
/// words last written, and a write staged since of words that differ from
/// them, none once written.
#[derive(Default)]
struct Policy {
    written: Option<residency::PolicyWords>,
    staged: Option<(residency::PolicyWords, PolicyWrite)>,
}

/// The write of policy words a run's encoding reads ([`Shared::stage_policy`]).
pub(super) type PolicyWrite = Box<dyn FnOnce(&residency::PolicyWords) + Send>;

/// A growth's slot being backed ([`Shared::backing_slot`]), until it drops:
/// then the load is told, as a hold waits on it.
pub(in crate::experts) struct BackingSlot<'s>(&'s Shared);

impl Drop for BackingSlot<'_> {
    fn drop(&mut self) {
        self.0.backing.store(false, std::sync::atomic::Ordering::Release);
        self.0.events().notify();
    }
}

impl Shared {
    /// What a starting watcher's steps and serving share (`Watcher::start`),
    /// built in a frame of its own: it is the largest of the start's parts.
    fn new(
        steps: Arc<Steps>,
        time: crate::experts::horizon::Time,
        telemetry: Telemetry,
        told: std::sync::mpsc::Sender<crate::experts::interest::Told>,
        compute: crate::experts::Compute,
        parks: Arc<Parks>,
        transport: Arc<crate::experts::transport::ExpertTransport>,
    ) -> Arc<Self> {
        Arc::new(Self {
            steps,
            time,
            pending_verify: Mutex::new(None),
            telemetry,
            told,
            resizing: ResizeHandoff::new(),
            traffic_held: Default::default(),
            traffic_quiet: Default::default(),
            backing: Default::default(),
            joins: Joins::new(),
            life: Life::new(),
            compute,
            sites_served: std::sync::atomic::AtomicBool::new(true),
            parks,
            landings: Default::default(),
            priced: Mutex::new(None),
            promoted: Default::default(),
            copy: Mutex::new(None),
            lend: Mutex::default(),
            covers: Default::default(),
            passes: Mutex::new(None),
            prices: Mutex::new(None),
            settled: Default::default(),
            transport,
            policy: Mutex::default(),
        })
    }

    /// Tell the interest ledger `told`, and wake the service that drains
    /// it. Once residency is gone nothing drains it, and what is told is
    /// dropped.
    pub(super) fn tell(&self, told: crate::experts::interest::Told) {
        let _ = self.told.send(told);
        self.events().notify();
    }

    /// Hold the load's own traffic until the returned hold drops: while any
    /// hold stands, and while no step runs, no promotion is asked nor any
    /// read ahead begun, so a measurement beside the idle load reads none
    /// of its copies, reads or landings once those running end
    /// ([`Self::traffic_quiet`]). A step beginning ends the hold's effect
    /// at once: serving never waits on a measurement. Holds are counted, so
    /// each holder (the measurement job's claim, a tool beside the load)
    /// releases only its own, by whichever exit. Each change is an event:
    /// residency looks again.
    pub fn hold_traffic(self: &Arc<Self>) -> crate::domain::TrafficHold {
        // Quiet is read again under this hold: no turn before it reads it so.
        self.traffic_quiet.store(false, std::sync::atomic::Ordering::Release);
        self.traffic_held.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        self.events().notify();
        let shared = self.clone();
        crate::domain::TrafficHold::new(move || {
            shared.traffic_held.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            shared.events().notify();
        })
    }

    /// Whether the load's traffic is held, no step runs and none of its
    /// traffic runs: the copies and reads begun before the hold have ended,
    /// as residency last saw, and no growth's slot is being backed.
    pub fn traffic_quiet(&self) -> bool {
        let ordering = std::sync::atomic::Ordering::Acquire;
        self.traffic_held()
            && self.traffic_quiet.load(ordering)
            && !self.backing.load(ordering)
            && self.steps.idle()
    }

    /// Wait until the load's traffic, held, is quiet
    /// ([`Self::traffic_quiet`]): each of its turns quiet, and each slot
    /// backed, is an event; why serving ended, if it ends first, or that
    /// the load stops.
    pub fn wait_traffic_quiet(&self) -> Result<(), String> {
        let events = self.events();
        loop {
            let seen = events.seen();
            if self.traffic_quiet() {
                return Ok(());
            }
            if let Some(reason) = self.life.ended() {
                return Err(format!("the load's traffic drain: serving ended: {reason}"));
            }
            if self.life.stopping() {
                return Err("the load's traffic drain: the load stops".into());
            }
            events.wait(seen);
        }
    }

    /// Begin backing a growth's slot, once the hold allows it: while the
    /// load's traffic is held and no step runs, the growth waits between
    /// its slots, until the hold ends, a step begins (serving never waits
    /// on a measurement) or `stop` ends the growth (a shrink's, or the
    /// measurement's claim, `ExpertBank::stop_growth`); none where it
    /// stopped. The slot is marked backing first, so no hold reads it
    /// quiet while it begins.
    pub(in crate::experts) fn backing_slot(&self, stop: &std::sync::atomic::AtomicBool) -> Option<BackingSlot<'_>> {
        let holds = || self.traffic_held() && self.steps.idle();
        loop {
            self.backing.store(true, std::sync::atomic::Ordering::Release);
            let slot = BackingSlot(self);
            if stop.load(std::sync::atomic::Ordering::Acquire) {
                return None;
            }
            if !holds() {
                return Some(slot);
            }
            // Unmarked (an event), then waited past what is seen once
            // unmarked, the hold and `stop` looked at again first: no change
            // after is missed, and a stop landing before is never waited past.
            drop(slot);
            let seen = self.events().seen();
            if holds() && !stop.load(std::sync::atomic::Ordering::Acquire) {
                self.events().wait(seen);
            }
        }
    }

    /// Step `sequence`'s run of layer `layer`'s block was sampled.
    pub(crate) fn block_sampled(&self, sequence: Step, layer: usize, spans: seismic::NativeSpans, marks: (usize, usize)) {
        self.telemetry.block_sampled(sequence, layer, spans, marks);
    }

    /// Whether any hold of the load's traffic stands ([`Self::hold_traffic`]).
    pub(super) fn traffic_held(&self) -> bool {
        self.traffic_held.load(std::sync::atomic::Ordering::Acquire) > 0
    }

    /// Each streamed layer's lines, its pass's price and the load's launch
    /// prices from now on, the GPU's priced, or none where they are not: the
    /// route's line words and the gates follow at the next turn
    /// (`Residency::publish_lines`, `Residency::publish_policy`), this an
    /// event.
    pub fn price(&self, lines: Option<Arc<[ExpertLines]>>, passes: Option<magnitude_cost_model::ExpertPass>, prices: Option<magnitude_cost_model::LaunchPrices>) {
        *self.priced.lock().expect("the priced lines") = lines;
        *self.passes.lock().expect("the priced passes") = passes.map(Arc::new);
        *self.prices.lock().expect("the launch prices") = prices;
        self.events().notify();
    }

    /// The load's launch prices; none before they are priced.
    pub(super) fn launch_prices(&self) -> Option<magnitude_cost_model::LaunchPrices> {
        self.prices.lock().expect("the launch prices").clone()
    }

    /// Request `request`'s pass of step `step` was decided: its guessed
    /// rows fared as `verdict` says (it judged none and learned no truth,
    /// where empty). Each guess it checked is noted on the read timeline
    /// (`guess`: its depth, token and the selection it was checked against,
    /// then `stood`), the key a pass's record joins its `ahead_row`s by. A
    /// guess stood where the selection it was checked against is its
    /// token, a guessed stop among them, whether or not the round committed
    /// it. The ledger counts its acceptable rows' guesses into their stands
    /// and drops the names of its rows past a wrong one, and judges by the
    /// truths it learned every front row of the request's passes whose
    /// token they decide, its own and its earlier passes' deeper ones
    /// (`interest::Verdict`); a fallen front row's pass drops its names
    /// from that row on, whose unbegun reads its next derivation cancels.
    pub fn decided(&self, step: Step, request: crate::operation::RequestId, verdict: &crate::AheadVerdict) {
        let checked = verdict.checked();
        if seismic_io::timeline::recording() {
            for (depth, &(guess, truth)) in checked.iter().enumerate() {
                seismic_io::timeline::note(
                    seismic_io::timeline::Kind::Noted("guess"),
                    request.0,
                    (depth as i64 + 1, i64::from(guess.0), i64::from(truth.0)),
                );
            }
            seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("stood"), request.0, (verdict.stood() as i64, 0, 0));
        }
        let acceptable: Vec<bool> = checked[..verdict.acceptable().min(checked.len())].iter().map(|(guess, truth)| guess == truth).collect();
        let accepted = acceptable.iter().take_while(|&&stood| stood).count() as u32;
        self.decide_verify(step, accepted);
        let verdict = crate::experts::interest::Verdict { acceptable, front: verdict.front().to_vec(), truths: verdict.truths().to_vec() };
        self.tell(crate::experts::interest::Told::Decided { step, request, verdict });
    }

    /// Request `request`'s pass of step `step` was discarded: it completed,
    /// but its outputs are no sequence's, so no verdict will judge it. The
    /// names its rows published drop, and their unbegun reads cancel.
    pub fn discarded(&self, step: Step, request: crate::operation::RequestId) {
        self.decide_verify(step, 0);
        self.tell(crate::experts::interest::Told::Discarded { step, request });
    }

    /// The load's token clock and turnovers (`experts::horizon::Time`).
    pub(crate) fn time(&self) -> &crate::experts::horizon::Time {
        &self.time
    }

    /// Register decode step `step` on the load's clock before its launch is
    /// queued (design §3.1): `Advance::Pending` remembers its verify
    /// request count so only its verdicts/discards call `Times::decided`.
    pub(super) fn register_step(&self, step: Step, advance: crate::experts::horizon::Advance) {
        if let crate::experts::horizon::Advance::Pending { requests } = advance {
            let mut pending = self.pending_verify.lock().expect("pending verify step");
            assert!(pending.is_none(), "verify step {step} registered while {:?} is pending", *pending);
            *pending = Some((step, requests, requests, 0));
        }
        self.time.lock().register(step, advance);
    }

    /// Count one request's verdict (or discard, with 0 accepted) where
    /// `step` is the pending verify step on the clock. Panics where a
    /// verdict for another step arrives while a verify step is pending, or
    /// where no request remains.
    fn decide_verify(&self, step: Step, accepted: u32) {
        let mut pending = self.pending_verify.lock().expect("pending verify step");
        let Some((pending_step, total, remaining, rows)) = *pending else {
            return;
        };
        assert_eq!(
            pending_step, step,
            "verdict for step {step} arrived while verify step {pending_step} is pending"
        );
        assert!(remaining > 0, "verify step {step} decided with no requests remaining");
        self.time.lock().decided(step, accepted);
        let rows = rows + u64::from(accepted);
        let left = remaining - 1;
        if left > 0 {
            *pending = Some((pending_step, total, left, rows));
        } else {
            *pending = None;
        }
    }

    /// Request `request` closed: none of its passes will come, so the
    /// names its rows published drop, and their unbegun reads cancel.
    pub fn closed(&self, request: crate::operation::RequestId) {
        self.tell(crate::experts::interest::Told::Closed(request));
    }

    /// An MTP head draft launched after a verify step completed: the drive is
    /// idle for `estimated_seconds` until the next verify step begins.
    pub fn draft_window(&self, estimated_seconds: f64) {
        self.tell(crate::experts::interest::Told::DraftWindow { estimated_seconds });
    }

    /// Drafted tokens for `request`'s upcoming burst (anchor `y_0` on head
    /// launch, then drafted proposals `[y_1..y_k]` on head completion):
    /// publishes token-conditioned `Source::Token` interests into `NameLedger`.
    pub fn draft_tokens(&self, request: crate::operation::RequestId, tokens: Vec<crate::operation::TokenId>) {
        if !tokens.is_empty() {
            self.tell(crate::experts::interest::Told::DraftTokens { request, tokens });
        }
    }

    /// When the host reads a step's request: at a park where parks are
    /// submission boundaries, as it is written where kernels wait in place.
    pub(super) fn notice(&self) -> crate::assessment::Notice {
        match self.parks.kept {
            true => crate::assessment::Notice::Parked,
            false => crate::assessment::Notice::Written,
        }
    }

    /// What a routed pass costs once priced ([`Self::price`]); none before.
    pub(super) fn passes(&self) -> Option<Arc<magnitude_cost_model::ExpertPass>> {
        self.passes.lock().expect("the priced passes").clone()
    }

    /// Each streamed layer's lines once priced ([`Self::price`]); none
    /// before.
    pub(super) fn priced(&self) -> Option<Arc<[ExpertLines]>> {
        self.priced.lock().expect("the priced lines").clone()
    }

    /// A copy over the link costs `costs` from now on.
    pub fn price_copy(&self, costs: &crate::assessment::TierCopyCosts) {
        tracing::info!(
            "a promotion's copy costs its link time at {:.2} GB/s, and {:.0}% of that in device time beside it",
            1e-9 / costs.transfer.seconds_per_byte(),
            costs.stall_share * 100.
        );
        *self.copy.lock().expect("a copy's price") = Some(costs.clone());
    }

    /// The covering tier's reads ([`crate::ExpertCover`]) as residency last
    /// published them: reading from the bank's open until its first turn
    /// finds none left to read (`Prefetch::covering`).
    pub fn cover(&self) -> crate::ExpertCover {
        crate::ExpertCover::after(self.covers.load(std::sync::atomic::Ordering::Acquire))
    }

    /// Publish whether the cover still reads, `reading`; a change is an
    /// event, which wakes [`Self::wait_covered`] and advances `regime_epoch`.
    pub(super) fn publish_cover(&self, reading: bool) {
        self.time.lock().update_residency(|state| {
            state.covering = reading;
        });
        let read = u64::from(!reading);
        let changed = self
            .covers
            .fetch_update(std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Acquire, |changes| {
                (changes % 2 != read).then_some(changes + 1)
            })
            .is_ok();
        if changed {
            self.events().notify();
        }
    }

    /// Wait until the cover reads nothing, as residency publishes it
    /// ([`Self::cover`]), woken by each of the load's events; why serving
    /// ended, if it ends first, or that the load stops.
    pub fn wait_covered(&self) -> Result<(), String> {
        let events = self.events();
        loop {
            let seen = events.seen();
            if !self.cover().reading {
                return Ok(());
            }
            if let Some(reason) = self.life.ended() {
                return Err(format!("the host tier's cover: serving ended: {reason}"));
            }
            if self.life.stopping() {
                return Err("the host tier's cover: the load stops".into());
            }
            events.wait(seen);
        }
    }

    /// The load's events, which every serving thread waits on
    /// (`Steps::events`).
    pub(super) fn events(&self) -> &Arc<super::events::Events> {
        self.steps.events()
    }

    /// The steps a prompt group runs while decode waits, at most `capacity`
    /// (`Telemetry::group_size`).
    pub fn group_size(&self, capacity: usize) -> usize {
        self.telemetry.group_size(self.priced().as_deref(), self.notice(), capacity)
    }

    /// Each bank group's device convert takes its `lines` from now on.
    pub(in crate::experts) fn device_convert(&self, lines: &[magnitude_cost_model::RecordLine]) {
        self.telemetry.device_convert(lines);
    }

    /// The landings the device has converted.
    #[cfg(test)]
    pub(in crate::experts) fn device_landed(&self) -> u64 {
        self.landings.device.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// What streamed layer `layer`'s CPU site answered of step `step`
    /// (`Compute::answered`).
    #[cfg(test)]
    pub(crate) fn site_answered(&self, layer: usize, step: u64) -> Option<crate::experts::Answered> {
        self.compute.answered(layer, step)
    }

    /// What steps have cost the load, the CPU site's serving among it.
    pub fn costs(&self) -> ExpertCosts {
        self.telemetry.costs(self.compute.site_served(), self.promoted.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Whether no step runs: a gap between flights, where residency may be
    /// asked to resize. Steps begin and end only on the owner's thread, so a
    /// gap it sees holds until it begins one.
    pub fn in_gap(&self) -> bool {
        self.steps.in_flight() == 0
    }

    /// Stage residency's `words`, the policy words a run's encoding reads,
    /// written by the write `writer` makes, in place of any staged before:
    /// written at the first point no step runs ([`Self::write_staged`]),
    /// whether residency's next turn or the next step's begin. While a
    /// write is staged no step is queued behind another
    /// (`ElasticBank::wants_gap`), so a pipeline of queued steps breaks for
    /// it at the next step boundary. Only words that differ from those
    /// written are staged: words equal to them drop any write staged
    /// instead, so a change undone before it is written breaks nothing, and
    /// a break stays rare.
    pub(super) fn stage_policy(&self, words: residency::PolicyWords, writer: impl FnOnce() -> PolicyWrite) {
        let mut policy = self.policy.lock().expect("staged policy");
        if policy.written.as_ref() == Some(&words) {
            policy.staged = None;
        } else if policy.staged.as_ref().is_none_or(|(staged, _)| *staged != words) {
            policy.staged = Some((words, writer()));
        }
    }

    /// Whether policy words are staged and not yet written.
    pub fn policy_staged(&self) -> bool {
        self.policy.lock().expect("staged policy").staged.is_some()
    }

    /// Write the staged policy words where no step runs, under the lock
    /// every step begins under ([`Steps::while_idle`]): every run encoded
    /// after it reads them, and they are the words written. Nothing while
    /// steps run, or none is staged. The policy's lock is held across the
    /// write, so a stage compares against the words written.
    pub(super) fn write_staged(&self) {
        self.steps.while_idle(|| {
            let mut policy = self.policy.lock().expect("staged policy");
            if let Some((words, write)) = policy.staged.take() {
                write(&words);
                let policy_hash = words.residency_fingerprint();
                self.time.lock().update_residency(|state| {
                    state.policy_hash = policy_hash;
                });
                policy.written = Some(words);
            }
        });
    }

    /// Step `sequence`'s runs serve its streamed layers from `layer` on over
    /// a window of its rows starting at batch row `first` (a stateless
    /// tail's): their requests name rows within it.
    pub fn window(&self, sequence: Step, layer: usize, first: usize) {
        self.steps.window(sequence, layer, first);
    }

    /// What the load's counted decode read and ran so far
    /// (`Stack::live`), which prices its experts' decisions; none before it
    /// counted a step.
    pub fn live(&self) -> Option<magnitude_cost_model::LiveDecode> {
        self.telemetry.live()
    }

    /// What the decode's passes measured, which the regime bit is decided
    /// on (`passes::Measured`); none before a pass is timed.
    pub fn measured(&self) -> Option<Measured> {
        self.telemetry.measured()
    }

    /// The bank-wide turnover's current epoch (`horizon::Times::bank_epoch`):
    /// what a live rate measured outside the watcher rolls on.
    pub fn bank_epoch(&self) -> u64 {
        self.time.lock().bank_epoch()
    }

    /// The physical tier/residency regime epoch (`horizon::Times::regime_epoch`):
    /// advances on any tier resize, `tier_complete`, residency policy, link
    /// staging, or cover-read transition.
    pub fn regime_epoch(&self) -> u64 {
        self.time.lock().regime_epoch().get()
    }

    /// The experts each streamed layer's slot map names held, ascending.
    pub fn held(&self) -> Vec<Vec<usize>> {
        self.telemetry.held()
    }

    /// What a load recording its routes recorded since last taken
    /// (`RouteRecorder::take`); none of a load that records none.
    pub fn take_routes(&self) -> Option<RecordedRoutes> {
        self.telemetry.take_routes()
    }

    /// Each group's records a prompt group holds claimed and the records
    /// decode keeps experts in, as residency last published them
    /// (`HostRecords::capacity`): a lend that returned leaves none claimed
    /// and the whole capacity.
    pub fn lend(&self) -> Vec<GroupLend> {
        self.lend.lock().expect("the lend's publication").clone()
    }

    /// Publish each group's lend ([`Self::lend`]).
    pub(super) fn publish_lend(&self, lend: &[GroupLend]) {
        lend.clone_into(&mut self.lend.lock().expect("the lend's publication"));
    }

    /// Per streamed layer, the experts its bank holds and those its group's
    /// host tier holds alone, as residency last published them, together:
    /// an expert is in at most one of them.
    pub fn holdings(&self) -> (Vec<usize>, Vec<usize>) {
        self.telemetry.holdings()
    }

    pub(in crate::experts) fn record_dma_call(
        &self,
        skipped_zero_miss: bool,
        experts_landed: u64,
        bytes_landed: u64,
        overflow_experts: u64,
    ) {
        self.telemetry.engagement().record_dma_call(
            skipped_zero_miss,
            experts_landed,
            bytes_landed,
            overflow_experts,
        );
    }

    pub(in crate::experts) fn record_dma_host_wait(&self, elapsed: std::time::Duration) {
        self.telemetry.engagement().record_dma_host_wait(elapsed);
    }

    /// Wake the CPU site's noticing worker immediately when `DecodeFront`
    /// publishes at least one CPU-sited choice (`SOURCE_SITE >= 0`).
    pub(in crate::experts) fn wake_site_noticer(&self) {
        self.compute.wake_noticer();
    }

    pub(in crate::experts) fn is_ended(&self) -> bool {
        self.life.ended().is_some()
    }

    pub(in crate::experts) fn record_growth_refusal(&self) {
        self.telemetry.engagement().record_growth_refusal();
    }
}

/// The running watcher: serving's threads and what they share.
pub(super) struct Watcher {
    shared: Arc<Shared>,
    /// The host compute executor, stopped once serving has ended.
    compute: Option<crate::experts::HostCompute>,
    /// The request service's thread, residency's, and the transfer queue's
    /// issuer and watch where the load has the queue.
    requests: Option<JoinHandle<()>>,
    residency: Option<JoinHandle<()>>,
    copies: Vec<JoinHandle<()>>,
    /// The parks' keeper, where parks are submission boundaries.
    keeper: Option<JoinHandle<()>>,
    /// The exchange's watchdog over the load (`liveness::watch`).
    watchdog: Option<crate::experts::exchange::Watchdog>,
    /// Residency's state, shared with `ExpertBank` for zero-recopy unified
    /// demand-landing slot allocation and O(1) completion promotion.
    state: Arc<crate::experts::yielding::Yielding<Residency>>,
}

/// What a streamed load hands its watcher to serve.
pub(super) struct WatcherParts {
    pub layers: Vec<Layer>,
    pub group_reads: Vec<PlacedLayerWords>,
    pub groups: Vec<Group>,
    pub load: LoadWords,
    pub pools: Vec<Pool>,
    pub extents: Extents,
    pub slots: Arc<std::sync::atomic::AtomicU64>,
    pub host_slots: Arc<std::sync::atomic::AtomicU64>,
    pub whole: bool,
    pub staging: u32,
    pub host: HostTier,
    pub carrier: Carrier,
    pub source: ExpertSource,
    pub compute: crate::experts::HostCompute,
    pub copier: Option<Copier>,
    pub steps: Arc<Steps>,
    /// Whether the device's backend times spans of a graph run, where
    /// sampled decode steps time their blocks (`Telemetry::samples`).
    pub spans: bool,
    /// What the exchange's watchdog does with its report of a wait past its
    /// bound or a kernel's violation ([`crate::experts::exchange::Trip`]).
    pub trip: crate::experts::exchange::Trip,
    /// The model store's read line the load's plan was priced on
    /// (`BankPaths::line`): what bounds the exchange's waits until the
    /// load's own reads measure the drive; none for a plan priced before the
    /// machine's figures were measured.
    pub line: Option<magnitude_cost_model::ReadLine>,
    /// Unified copy-engine-first `ExpertTransport` shared with `ExpertBank`
    /// and `Copies`.
    pub transport: Arc<crate::experts::transport::ExpertTransport>,
}

impl Watcher {
    pub fn start(parts: WatcherParts) -> Result<Self, String> {
        Self::start_inner(parts, None)
    }

    #[cfg(test)]
    pub(super) fn start_with_bounds(
        parts: WatcherParts,
        bounds: Box<dyn Fn() -> crate::experts::exchange::Bounds + Send>,
    ) -> Result<Self, String> {
        Self::start_inner(parts, Some(bounds))
    }

    fn start_inner(
        parts: WatcherParts,
        bounds: Option<Box<dyn Fn() -> crate::experts::exchange::Bounds + Send>>,
    ) -> Result<Self, String> {
        let WatcherParts { layers, group_reads, groups, load, pools, extents, slots, host_slots, whole, staging, host, carrier, source, compute, copier, steps, spans, trip, line, transport } = parts;
        // The geometry every read lands by, checked once: each pool's
        // staging holds the staging records, and the conveyor lies in each
        // pool, at most a record of every expert of the widest layer.
        let conveyors = &carrier.pools;
        if carrier.records == 0 || carrier.records > extents.widest {
            return Err(format!("a conveyor of {} records beside layers of {} experts", carrier.records, extents.widest));
        }
        if carrier.lending.pools.len() != pools.len() {
            return Err(format!("a lent conveyor of another count of pools than {}", pools.len()));
        }
        if conveyors.len() != pools.len() {
            return Err(format!("{} staging pools and conveyors of another count", pools.len()));
        }
        for (pool, conveyor) in pools.iter().zip(conveyors) {
            if pool.staging.count() != u64::from(staging) || conveyor.count() != carrier.records as u64 {
                return Err(format!(
                    "a staging pool of {} records and a conveyor of {}, not {staging} and {}",
                    pool.staging.count(),
                    conveyor.count(),
                    carrier.records
                ));
            }
        }
        for stored in layers.iter().flat_map(|layer| layer.stored.iter()) {
            let pool = pools
                .get(stored.pool)
                .ok_or_else(|| format!("a routed tensor reads staging pool {} of {}", stored.pool, pools.len()))?;
            let carried = conveyors[stored.pool].bytes();
            if stored.bytes() > pool.staging.bytes() || stored.bytes() > carried {
                return Err(format!(
                    "a stored expert of {} bytes exceeds its pool's records of {} or the conveyor's of {carried}",
                    stored.bytes(),
                    pool.staging.bytes(),
                ));
            }
        }
        // What bounds every answer of the exchange (`liveness::Answers`):
        // the records reads land into, and the largest expert.
        let in_flight = u64::from(staging) + carrier.records as u64;
        let largest = layers.iter().map(Layer::expert_bytes).max().unwrap_or(0);
        let layer_pools = layers
            .iter()
            .map(|layer| layer.stored.iter().map(|stored| stored.pool).collect())
            .collect::<Vec<_>>();
        let experts = layers.iter().map(|layer| layer.experts).collect::<Vec<_>>();
        let records = groups.iter().map(|group| group.records).collect::<Vec<_>>();
        let mut directory = Directory::new(
            &records,
            &layers.iter().map(|layer| layer.group).collect::<Vec<_>>(),
            &layers.iter().map(|layer| layer.experts).collect::<Vec<_>>(),
            host,
        );
        // A bank imported whole: every expert at its home, as its maps name
        // them, held beside a host tier and kept as the landed records
        // where the bank is the one store.
        if whole {
            match extents.placer {
                crate::planning::Placer::Tier => directory.hold_homes(),
                crate::planning::Placer::Bank => directory.keep_homes(),
            }
        }
        // Each layer's home as the device reads it (`policy[POLICY_HOME]`):
        // the directory's, by which the host places and names map words.
        let homes = (0..layers.len())
            .map(|layer| i32::try_from(directory.home(layer)).map_err(|_| "a layer's map words exceed i32".to_owned()))
            .collect::<Result<Vec<_>, _>>()?;
        let LoadWords { flags } = load;
        let count = layers.len();
        let mut core = Core {
            table: Arc::new(ReadTable::new(source, pools, &layers)),
            directory,
            layers,
            groups,
            extents,
            retired: steps.frontier(),
            steps: steps.clone(),
            taken: Vec::new(),
        };
        let reads = telemetry::GroupReadWords { layers: group_reads, places: flags.len() };
        // Beside a host tier, the resident bytes the fill's tier copy reads
        // of one expert of each layer.
        let resident = match core.extents.placer {
            crate::planning::Placer::Tier => core
                .layers
                .iter()
                .map(|layer| {
                    core.groups[layer.group].tier.as_ref().map(|tensors| tensors.iter().map(|tensor| tensor.records.bytes()).sum::<u64>())
                })
                .collect::<Option<Vec<_>>>(),
            crate::planning::Placer::Bank => None,
        };
        // Each group's decode capacity, told telemetry now and by residency
        // as it changes (`Residency::publish_held`).
        let capacities = core.directory.capacities();
        let telemetry = Telemetry::new(
            &core.layers,
            reads,
            extents,
            slots,
            host_slots,
            capacities.clone(),
            resident,
            core.directory.ranking(),
            core.table.source().volumes(),
            spans,
        );
        core.directory.bind_whole_layers(telemetry.whole_flags());
        // The load's parks, on one timeline of its device, which each answer
        // of a slot of the exchange settles.
        let parks = {
            let timeline = seismic::Timeline::new(&core.layers[0].file.device())
                .map_err(|error| format!("the expert parks' timeline: {error}"))?;
            let layers = core
                .layers
                .iter()
                .map(|layer| parks::ParkLayer { file: layer.file.clone(), scratch: layer.scratch.clone(), site: layer.site.clone() })
                .collect();
            Arc::new(Parks::new(timeline, layers, steps.events().clone()))
        };
        compute.handle().gated(parks.clone());
        let answers = liveness::Answers {
            records: in_flight,
            bytes: largest,
            sites: compute
                .handle()
                .layers()
                .iter()
                .map(|site| {
                    let crate::experts::SiteShape { rows, choices, experts, .. } = site.shape;
                    (site.layer(), rows as f64, (rows * choices).min(experts) as f64)
                })
                .collect(),
            planned: line,
        };
        // The ledger's queue: told from any thread, drained by `Prefetch`.
        let (told, inbox) = std::sync::mpsc::channel();
        let time = core.directory.time().clone();
        let shared = Shared::new(steps, time.clone(), telemetry, told, compute.handle(), parks, transport);
        crate::settle::owe_shared(Arc::downgrade(&shared));
        // The transfer queue the promotions queue on, whose completions its
        // watch announces.
        let copies = copier.as_ref().map(|copier| copier.copies.clone());
        // The ledger's rates are the ranking's (`Reuse::rates`).
        let ranking = core.directory.ranking();
        let groups = core.directory.layer_groups();
        let ranks = core.extents.selected;
        let (_, stands) = crate::experts::reuse::Reuse::of(&ranking).rates();
        let ledger = crate::experts::interest::Ledger::new(groups, ranks, time, stands);
        let mut residency = Residency {
            homes,
            shared: shared.clone(),
            core,
            ledger,
            inbox,
            prefetch: Prefetch::new(staging, count),
            scratch: Scratch::new(
                count,
                flags,
                Conveyor::new(layer_pools, &experts, carrier),
                copier.clone(),
            ),
            link: LinkMover::new(),
            asked: None,
            opened: Instant::now(),
            complete: false,
            lines: None,
            logged_initial_lines: false,
            published: residency::Published::new(count, capacities),
            owed: resize::Owed::default(),
        };
        residency.publish_held();
        residency.publish_policy();
        // No step runs yet: the open's words are written now, as every
        // later gap writes what it staged.
        shared.write_staged();
        residency.publish_lines();
        let residency = Arc::new(crate::experts::yielding::Yielding::new(residency));
        // From here a thread that cannot start drops the watcher, which
        // stops and joins those that did.
        let mut watcher = Self {
            shared: shared.clone(),
            compute: Some(compute),
            requests: None,
            residency: None,
            copies: Vec::new(),
            keeper: None,
            watchdog: None,
            state: residency.clone(),
        };
        let requests = Requests::new(residency.clone(), shared.clone(), copier.clone());
        watcher.watchdog = Some({
            let held = residency::lock(&residency);
            liveness::watch(&held.core, &shared, residency.clone(), requests.service(), copies.clone(), answers, bounds, trip)?
        });
        watcher.requests = Some(
            seismic::threads::spawn("expert-requests", move || {
                seismic_io::waited_on();
                requests.serve();
            })
            .map_err(|error| format!("expert request service: {error}"))?,
        );
        let kept = shared.clone();
        watcher.residency = Some(
            seismic::threads::spawn("expert-residency", move || Residency::keep(residency, kept, copier))
            .map_err(|error| format!("expert residency: {error}"))?,
        );
        if let Some(copies) = copies {
            let watched = shared.clone();
            watcher.copies = copies.start(move || watched.life.stopping())?;
        }
        if shared.parks.kept {
            watcher.keeper = Some(Parks::keep(shared.parks.clone())?);
        }
        Ok(watcher)
    }

    /// Starve the load's host for `duration` from now
    /// (`ExpertBank::starve`).
    #[cfg(test)]
    pub(in crate::experts) fn starve(&self, duration: std::time::Duration) {
        self.shared.steps.pulse().starve(duration);
    }

    pub fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }

    /// Whether bank record `record` of group `group` is lent to a prompt
    /// group.
    #[cfg(test)]
    pub fn bank_lent(&self, group: usize, record: u32) -> bool {
        residency::lock(&self.state).core.directory.lent(group, record as usize)
    }

    /// Claim a unified VRAM slot in `group` for an on-demand decode DMA
    /// landing of `tier_record` (non-blocking).
    pub(in crate::experts) fn claim_demand_slot(
        &self,
        group: usize,
        tier_record: u32,
    ) -> crate::experts::directory::DemandSlotClaim {
        let mut held = residency::lock(&self.state);
        let lending = held.core.steps.lending();
        let transport = self.shared.transport.clone();
        transport.with_slot_reuse(|can_reuse| {
            held.core.directory.claim_demand_landing(group, tier_record, lending, |slot| {
                can_reuse(crate::experts::transport::DestSlot { group, slot })
            })
        })
    }

    /// Release aborted/faulted `Record::DemandLanding` slots back to `Free`.
    pub(in crate::experts) fn abort_demand_slots(&self, slots: &[(usize, usize)]) {
        if slots.is_empty() {
            return;
        }
        let mut held = residency::lock(&self.state);
        for &(group, slot) in slots {
            held.core.directory.release_demand_landing(group, slot);
        }
    }

    /// Asynchronously release lowest-ranked unpinned `Record::Held` victims
    /// in `group` into `Record::Releasing` when `Free` headroom is scarce.
    pub(in crate::experts) fn replenish_demand_free(&self, group: usize, want_free: usize) {
        let mut held = residency::lock(&self.state);
        let transport = self.shared.transport.clone();
        let core = &mut held.core;
        let (directory, steps) = (&mut core.directory, &core.steps);
        let _released = steps.unbegun(|at| {
            transport.with_slot_reuse(|can_reuse| {
                directory.replenish_demand_free(group, want_free, at, |slot| {
                    can_reuse(crate::experts::transport::DestSlot { group, slot })
                })
            })
        });
    }

    /// Zero-recopy O(1) completion of freshly landed `DemandSlot`s: promotes
    /// qualifying slots `DemandLanding -> Held` (with `copy_hits = 0.0`) and
    /// publishes via `StampedMap::map_covering`, or returns unpromoted slots
    /// to `Free` (deferring if a joined consumer still pins the slot).
    pub(in crate::experts) fn finish_demand_landings(
        &self,
        layer: usize,
        sequence: u64,
        slots: &[crate::experts::transport::DemandSlot],
    ) {
        if slots.is_empty() {
            return;
        }
        let mut held = residency::lock(&self.state);
        let progress = held.core.steps.point(sequence, layer, sequence);
        let transport = self.shared.transport.clone();
        let mut promoted_any = false;
        let mut landed_at_home = false;
        let core = &mut held.core;
        let (directory, steps) = (&mut core.directory, &core.steps);
        transport.with_slot_reuse(|can_reuse| {
            for &slot in slots {
                let reusable = can_reuse(crate::experts::transport::DestSlot {
                    group: slot.group,
                    slot: slot.slot,
                });
                let promoted = directory.finish_demand_landing(
                    slot.group,
                    slot.slot,
                    slot.staged,
                    progress,
                    steps,
                    reusable,
                );
                if promoted {
                    promoted_any = true;
                    if let Some((staged_layer, staged_expert)) = slot.staged {
                        if slot.slot == directory.home(staged_layer) + staged_expert {
                            landed_at_home = true;
                        }
                    }
                }
            }
            if promoted_any {
                let group = slots[0].group;
                steps.unbegun(|at| {
                    directory.replenish_demand_free(group, 0, at, |slot| {
                        can_reuse(crate::experts::transport::DestSlot { group, slot })
                    })
                });
            }
        });
        if promoted_any {
            held.publish_held();
            if landed_at_home {
                held.publish_policy();
            }
        }
    }
}

/// The load's account of its landings and of its starved steps, once
/// serving stopped: as the load settles, or else as the watcher drops.
impl crate::settle::Owed for Shared {
    fn settle(&self) {
        if self.settled.first() {
            tracing::info!("expert {}", self.landings.line());
            tracing::info!("expert {}", self.steps.interruptions());
            if let Some(a1) = self.telemetry.far_deciles_line() {
                tracing::info!("expert {a1}");
            }
            let site_served = self.compute.site_served();
            let residency = self.telemetry.residency();
            if let Some(engagement) = self
                .telemetry
                .engagement()
                .summary_line_with(site_served.requests, site_served.experts, residency)
            {
                tracing::info!("expert engagement: {engagement}");
            }
        }
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        // Nothing is answered once serving stops, and while the load's
        // tables live its pulse beats, so a running step's wait on the
        // device never ends: every step's owner drains it before the bank
        // drops (the domain its queued ones, as it drops).
        let running = self.shared.steps.in_flight();
        assert!(
            running == 0,
            "the expert watcher stops with {running} steps running, whose waits it would never answer: {}",
            self.shared.steps.describe()
        );
        // The watchdog stops first.
        drop(self.watchdog.take());
        self.shared.life.stop();
        self.shared.events().notify();
        for thread in [self.requests.take(), self.residency.take()].into_iter().flatten().chain(self.copies.drain(..)) {
            thread.join().expect("a serving thread's panic ends the process");
        }
        self.shared.parks.stop();
        if let Some(keeper) = self.keeper.take() {
            keeper.join().expect("the expert parks' keeper never panics");
        }
        // Nothing submits to the executor any more: it stops, and the jobs
        // it never ran drop.
        drop(self.compute.take());
        crate::settle::Owed::settle(&*self.shared);
    }
}


