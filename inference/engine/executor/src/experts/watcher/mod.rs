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
//!   that named it, all without waiting for a read. The tier wait waits for
//!   each choice's answer (`parks`) and sources it from the tier
//!   (`routed_tier_wait`). The tier keeps a reserve of ready records,
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
//!   prompt group's conveyor, fills the tier, joins growths, serves resizes
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
mod lent;
mod liveness;
mod parks;
mod prefetch;
mod promotions;
mod requests;
mod residency;
mod resize;
mod scratch;
mod steps;
mod telemetry;
#[cfg(test)]
mod tests;

pub(in crate::experts) use self::core::{
    Extents, Group, Layer, ListRecords, LoadWords, PlacedLayerWords, PlacedWords, Pool, StoredTensor,
};
pub(crate) use parks::{ParkKind, ParkSite, Parks};
pub(in crate::experts) use promotions::{Copier, CopyTensor};
pub(crate) use resize::{Candidates, Join};
pub(crate) use steps::{ExpertStep, Recording, Step, StepFailure, StepRoute, low_word};
pub use telemetry::{ExpertCosts, GroupCost, GroupLayer, LayerTimeline, Span, StepTimeline, LayerCost, LayerExperts, StepCost, STEP_TRACE};

use self::core::{Core, ReadTable};
use super::conveyor::{Carrier, Conveyor};
use super::directory::{Directory, HostTier};
use super::source::ExpertSource;
use crate::assessment::ExpertLines;
use liveness::Life;
pub(crate) use liveness::panic_message;
use prefetch::Prefetch;
use promotions::Promotions;
use requests::Requests;
use residency::Residency;
use resize::{Joins, ResizeHandoff};
use scratch::{Scratch, ScratchWords, conveyor_forms};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;
pub(in crate::experts) use steps::{Steps, Unbegun};
pub(in crate::experts) use lent::{Lending, LentRecords};
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
    telemetry: Telemetry,
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
    /// The lender's lent bank records' reservation, beside a host tier
    /// (`lent`).
    lent: lent::LentRecords,
    /// The fill's changes as residency published them
    /// (`Residency::read_ahead`): each start and each finish counts one, from
    /// filling as the bank opens, so the fill has finished when the count
    /// is odd ([`Self::fill_epoch`]).
    fills: std::sync::atomic::AtomicU64,
    /// What a routed pass over each streamed layer's experts costs, once
    /// priced (`Self::price`): what a scratch-served group's merge sets are
    /// chosen by (`Shared::begin`).
    passes: Mutex<Option<Arc<crate::assessment::ExpertPass>>>,
    /// Whether the load's account was given as it settled.
    settled: crate::settle::Once,
}

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

    /// Each streamed layer's lines and its pass's price from now on, the
    /// GPU's priced, or none where they are not: the route's line words
    /// follow at the next turn (`Residency::publish_lines`), this an event.
    pub fn price(&self, lines: Option<Arc<[ExpertLines]>>, passes: Option<crate::assessment::ExpertPass>) {
        *self.priced.lock().expect("the priced lines") = lines;
        *self.passes.lock().expect("the priced passes") = passes.map(Arc::new);
        self.events().notify();
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
    pub(super) fn passes(&self) -> Option<Arc<crate::assessment::ExpertPass>> {
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
            1e-9 / costs.link_byte_seconds(),
            costs.stall_share * 100.
        );
        *self.copy.lock().expect("a copy's price") = Some(costs.clone());
    }

    /// The fill of the store misses land into (the host tier where the
    /// device places, the one store where the host does) as residency last
    /// published it: filling from the bank's open until its first turn
    /// finds it filled (`Prefetch::filling`).
    pub fn tier_fill(&self) -> crate::TierFill {
        match self.fill_epoch() % 2 {
            1 => crate::TierFill::Filled,
            _ => crate::TierFill::Filling,
        }
    }

    /// How many times the fill started or finished so far: odd once it
    /// finished, even while it lands experts. Two equal readings with the
    /// fill finished at both saw no fill between them.
    pub fn fill_epoch(&self) -> u64 {
        self.fills.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Publish the host tier's fill, `fill`; a change is an event, which
    /// wakes [`Self::wait_filled`]. Whether it changed.
    pub(super) fn publish_fill(&self, fill: crate::TierFill) -> bool {
        let filled = u64::from(fill == crate::TierFill::Filled);
        let changed = self
            .fills
            .fetch_update(std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Acquire, |fills| {
                (fills % 2 != filled).then_some(fills + 1)
            })
            .is_ok();
        if changed {
            self.events().notify();
        }
        changed
    }

    /// Wait until the store misses land into is filled, as residency
    /// publishes it ([`Self::tier_fill`]), woken
    /// by each of the load's events; why serving ended, if it ends first,
    /// or that the load stops.
    pub fn wait_filled(&self) -> Result<(), String> {
        let events = self.events();
        loop {
            let seen = events.seen();
            if self.tier_fill() == crate::TierFill::Filled {
                return Ok(());
            }
            if let Some(reason) = self.life.ended() {
                return Err(format!("the host tier's fill: serving ended: {reason}"));
            }
            if self.life.stopping() {
                return Err("the host tier's fill: the load stops".into());
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

    /// Each bank group's convert in a decode barrier takes its `lines` from
    /// now on.
    pub(in crate::experts) fn device_convert(&self, lines: &[crate::assessment::RecordLine]) {
        self.telemetry.device_convert(lines);
    }

    /// Measure the host's landings at a line of two counts each, each read
    /// held back 2 ms: beside a device convert line far under it
    /// (`ExpertMeasures::convert`), every miss is priced to land on the
    /// device.
    #[cfg(test)]
    pub(in crate::experts) fn slow_host_landing(&self) {
        for layer in 0..self.telemetry.landing_lines().len() {
            self.telemetry.landings_ended(layer, 1, 1.);
            self.telemetry.landings_ended(layer, 2, 2.);
        }
        self.landings.held_ms.store(2, std::sync::atomic::Ordering::Relaxed);
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

    /// Step `sequence`'s runs serve its streamed layers from `layer` on over
    /// a window of its rows starting at batch row `first` (a stateless
    /// tail's): their requests name rows within it.
    pub fn window(&self, sequence: Step, layer: usize, first: usize) {
        self.steps.window(sequence, layer, first);
    }

    /// What the load's counted decode read and ran so far
    /// (`Stack::live`), which prices its experts' decisions; none before it
    /// counted a step.
    pub fn live(&self) -> Option<crate::assessment::LiveDecode> {
        self.telemetry.live()
    }

    /// The experts each streamed layer's slot map names held, ascending.
    pub fn held(&self) -> Vec<Vec<usize>> {
        self.telemetry.held()
    }

    /// Per streamed layer, the experts its bank holds and those its group's
    /// host tier holds alone, as residency last published them, together:
    /// an expert is in at most one of them.
    pub fn holdings(&self) -> (Vec<usize>, Vec<usize>) {
        self.telemetry.holdings()
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
    /// Residency's state, which a test reads to set up what serving finds.
    #[cfg(test)]
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
}

impl Watcher {
    pub fn start(parts: WatcherParts) -> Result<Self, String> {
        let WatcherParts { layers, group_reads, groups, load, pools, extents, slots, whole, staging, host, carrier, source, compute, copier, steps, spans } = parts;
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
            if stored.bytes > pool.staging.bytes() || stored.bytes > carried {
                return Err(format!(
                    "a stored expert of {} bytes exceeds its pool's records of {} or the conveyor's of {carried}",
                    stored.bytes,
                    pool.staging.bytes(),
                ));
            }
        }
        // What bounds every answer of the exchange (`liveness::Answers`):
        // the records reads land into, and the largest expert.
        let in_flight = u64::from(staging) + carrier.records as u64;
        let largest = layers.iter().map(Layer::expert_bytes).max().unwrap_or(0);
        let form_of = conveyor_forms(
            pools.len(),
            &layers
                .iter()
                .map(|layer| layer.stored.iter().map(|stored| stored.pool).collect())
                .collect::<Vec<_>>(),
        );
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
        let LoadWords {
            flags,
            lists,
            tiered,
            seeds,
        } = load;
        let count = layers.len();
        let core = Core {
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
        let telemetry = Telemetry::new(
            &core.layers,
            reads,
            extents,
            slots,
            core.bank_records(),
            resident,
            core.directory.ranking(),
            core.table.source().volumes(),
            spans,
        );
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
        };
        let shared = Arc::new(Shared {
            steps,
            telemetry,
            resizing: ResizeHandoff::new(),
            traffic_held: Default::default(),
            traffic_quiet: Default::default(),
            backing: Default::default(),
            joins: Joins::new(),
            life: Life::new(),
            compute: compute.handle(),
            sites_served: std::sync::atomic::AtomicBool::new(true),
            parks,
            landings: Default::default(),
            priced: Mutex::new(None),
            promoted: Default::default(),
            copy: Mutex::new(None),
            lent: lent::LentRecords::new(),
            fills: std::sync::atomic::AtomicU64::new(0),
            passes: Mutex::new(None),
            settled: Default::default(),
        });
        crate::settle::owe_shared(Arc::downgrade(&shared));
        // The transfer queue the promotions queue on, whose completions its
        // watch announces.
        let copies = copier.as_ref().map(|copier| copier.copies.clone());
        let mut residency = Residency {
            whole: vec![false; count],
            homes,
            shared: shared.clone(),
            core,
            prefetch: Prefetch::new(staging, count),
            scratch: Scratch::new(
                count,
                ScratchWords { flags, lists, tiered, seeds },
                Conveyor::new(form_of, &experts, carrier),
            ),
            promotions: Promotions::new(),
            asked: None,
            opened: Instant::now(),
            complete: false,
            site_opened: shared.compute.site_served(),
            site_share: Default::default(),
            lines: None,
            published: residency::Published::new(count),
            owed: resize::Owed::default(),
        };
        residency.publish_held();
        residency.publish_policy();
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
            #[cfg(test)]
            state: residency.clone(),
        };
        let requests = Requests::new(residency.clone(), shared.clone());
        watcher.watchdog = Some({
            let held = residency::lock(&residency);
            liveness::watch(&held.core, &shared, residency.clone(), requests.service(), copies.clone(), answers, crate::experts::exchange::abort())?
        });
        watcher.requests = Some(
            std::thread::Builder::new()
                .name("expert-requests".into())
                .spawn(move || {
                    seismic_io::waited_on();
                    requests.serve();
                })
                .map_err(|error| format!("expert request service: {error}"))?,
        );
        let kept = shared.clone();
        watcher.residency = Some(
            std::thread::Builder::new()
                .name("expert-residency".into())
                .spawn(move || Residency::keep(residency, kept, copier))
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
}

/// The load's account of its landings and of its starved steps, once
/// serving stopped: as the load settles, or else as the watcher drops.
impl crate::settle::Owed for Shared {
    fn settle(&self) {
        if self.settled.first() {
            tracing::info!("expert {}", self.landings.line());
            tracing::info!("expert {}", self.steps.interruptions());
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
        self.shared.lent.wake();
        self.shared.events().notify();
        for thread in [self.requests.take(), self.residency.take()].into_iter().flatten().chain(self.copies.drain(..)) {
            thread.join().expect("serving's threads catch their panics");
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


