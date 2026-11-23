//! The request service: the one owner of noticing what the device asks and
//! answering it. It takes each layer's decode requests (file misses) and
//! scratch requests and rounds, and the hint rings, and serves them through
//! residency's methods (`residency`), never waiting inside one: each
//! request waits on its own slot of the exchange (`experts::exchange`),
//! carried a step further at each pass its events allow, so a scratch round
//! awaiting its arrival or its experts never holds back a decode request,
//! nor another layer's round.
//!
//! The device's request is a word it writes, which the host sees only by
//! reading it: where kernels wait in place (CUDA, Metal, the CPU) nothing
//! announces it, so while a step that may ask anything runs, the service
//! looks again within its rest ([`Rest`]) of its last look: the one poll of
//! serving, which the hardware forces. Each turn looks at the decode requests first,
//! before residency's lock: the look posts each request's reads with no
//! decision of residency's before them (`decode::Look`), and the pass under
//! the lock then serves what it took. Everything else it waits for is an
//! event of the load (a landing's or a read's end, a copy's completion, a
//! step's begin or end), which ends the rest at once. Where parks are submission
//! boundaries (Vulkan) the device's words are the host's only at a park,
//! and the parks' keeper announces each change it sees: there the service
//! never polls.

use super::decode::{Began, Decode, Look, Posted};
use super::liveness;
use super::residency::{Residency, lock, lock_counted};
use super::scratch::Pending;
use super::Shared;
use crate::experts::ExchangeTable;
use crate::experts::yielding::Yielding;
use magnitude_kernels::constants;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest the request service rests between two looks at the
/// device's words while a step that may ask anything runs (module
/// documentation). A request waits at worst one rest and one pass to be
/// noticed. Measured (an M4 Pro, 64k, 2026-10-25): this rest sleeps
/// 33.6 µs and a miss's notice averages ~18 µs, ~1.6% of a ~1.1 ms miss,
/// while the service spins on no memory the device uses. Its thread's
/// rests are precise (`seismic_io::waited_on`): no timer slack lengthens them.
pub(super) const IDLE: Duration = Duration::from_micros(20);

/// A poller's rest between two looks at what the device writes and no
/// event announces: the request service's at the device's words where
/// kernels wait in place (module documentation), the parks' keeper's at
/// the device's timeline (`parks`). Each is priced as the run measures it,
/// so no rest is a constant.
///
/// Polling at rest `r` costs, per second polled, its looks, `l / r` (`l`,
/// what a look costs: its measured time, [`Looked::Empty`]), and its
/// notices' wait: each thing the device writes and then waits on is
/// noticed on average `r / 2` after it is written, at `λ` of them a
/// second, `λ · r / 2` of stall. Their sum is least at
/// `r = √(2 · l / λ)`.
///
/// - **`l`**: a look's line (`seismic::measure::Line`, one fixed term)
///   through every look that found and changed nothing, from the poller's
///   first, before any step: until one is measured the rest is none.
/// - **`λ`**: the notices of the phase polled ([`Phase`]) counted over its
///   seconds polled, the notice that may come now among them:
///   `(n + 1) / T`, never below the measured rate `n / T`, so the rest
///   never exceeds the optimum at it, and bounded before the phase's first
///   notice (`√(2 · l · T)`, growing only as nothing is noticed). Before
///   any time of the phase is polled the rest is none (`T = 0`): the
///   poller looks again at once, its stall bounded by a look. A prompt and
///   a decode ask at their own rates, so each phase counts apart, from its
///   own start ([`Rest::begin`], a boundary each poller derives from its
///   steps), never from a window, whose length would be a tuned constant.
///   A notice is what the poller waits for: the request service's each
///   decode request and scratch round it found, the keeper's each look
///   that found the device arrived.
///
/// So `r = √(2 · l · T / (n + 1))`, of the phase polled. Its thread's rests
/// are precise (`seismic_io::waited_on`): no timer slack lengthens them.
pub(super) struct Rest {
    looks: seismic::measure::Line<1>,
    /// Each phase's count since its start, by [`Phase::index`].
    counts: [Count; 2],
    /// When the last look ended and the phase it polls, where the poller
    /// looks next by time.
    polling: Option<(Instant, Phase)>,
    rested: Rested,
}

/// What a poller polls for: the requests of a prompt (its groups' scratch
/// rounds and requests, many steps at once) or a decode's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    Prompt,
    Decode,
}

impl Phase {
    fn index(self) -> usize {
        match self {
            Self::Prompt => 0,
            Self::Decode => 1,
        }
    }
}

/// A phase's notices and seconds polled since its start.
#[derive(Clone, Copy, Default)]
struct Count {
    noticed: u64,
    polled: f64,
}

/// A poller's rest now ([`Rest`]), as others read it: the exchange's
/// watchdog bounds an answer by it (`liveness::Answers::longest`).
#[derive(Clone, Default)]
pub(super) struct Rested(Arc<AtomicU64>);

impl Rested {
    /// A fixed rest, for pollers with a constant bound ([`IDLE`]).
    fn fixed(duration: Duration) -> Self {
        Self(Arc::new(AtomicU64::new(u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX))))
    }

    /// The rest the poller last set.
    pub(super) fn get(&self) -> Duration {
        Duration::from_nanos(self.0.load(Ordering::Relaxed))
    }
}

/// What a look did ([`Rest::looked`]).
pub(super) enum Looked {
    /// It found and changed nothing, in its time: a look's price.
    Empty(Duration),
    /// It changed something, noticing `noticed` of what the device wrote.
    Changed { noticed: u64 },
}

impl Rest {
    /// A poller's rest, none measured, published to `rested`.
    pub(super) fn new(rested: Rested) -> Self {
        Self { looks: seismic::measure::Line::new(), counts: [Count::default(); 2], polling: None, rested }
    }

    /// `phase` starts again: its count begins anew, what it counted before
    /// another phase ran not its own. A look's price is every phase's.
    pub(super) fn begin(&mut self, phase: Phase) {
        self.counts[phase.index()] = Count::default();
    }

    /// A look ended: what it did, `looked`, and the phase the poller looks
    /// again by time for (`polls`), none where it waits on events. The
    /// seconds since the last look count as polled, and this look's
    /// notices among them, of the phase that look polled, where it was
    /// followed by time. The rest until the next look, where it polls.
    ///
    /// # Panics
    /// Where the looks' times make no line: each is a duration, so that is
    /// an invariant violation.
    pub(super) fn looked(&mut self, looked: Looked, polls: Option<Phase>) -> Option<Duration> {
        let now = Instant::now();
        let noticed = match looked {
            Looked::Empty(look) => {
                self.looks.observe([1.], look.as_secs_f64());
                0
            }
            Looked::Changed { noticed } => noticed,
        };
        if let Some((since, phase)) = self.polling.take() {
            let count = &mut self.counts[phase.index()];
            count.polled += now.saturating_duration_since(since).as_secs_f64();
            count.noticed += noticed;
        }
        let phase = polls?;
        let look = self
            .looks
            .state()
            .unwrap_or_else(|error| panic!("invariant violated: a look's durations make no line: {error:?}"))
            .line()
            .map_or(0., |terms| terms.costs()[0]);
        let Count { noticed, polled } = self.counts[phase.index()];
        let rest = Duration::from_secs_f64((2. * look * polled / (noticed + 1) as f64).sqrt());
        self.rested.0.store(u64::try_from(rest.as_nanos()).unwrap_or(u64::MAX), Ordering::Relaxed);
        self.polling = Some((now, phase));
        Some(rest)
    }
}

/// The request service: what it serves through, and its own state.
pub(super) struct Requests {
    residency: Arc<Yielding<Residency>>,
    shared: Arc<Shared>,
    /// Path 0 link copier handle (`Copier` is `Clone` and posts lock-free to
    /// `Copies`'s priority queue without ever blocking on CUDA), so
    /// `Residency::rings()` dispatches `HINT_LEADS=2` DMA staging immediately
    /// after resolving live hints instead of waiting for `Residency::keep`'s
    /// background `TURN`.
    copier: Option<super::Copier>,
    /// Held by its thread through each pass, beside residency's lock: the
    /// exchange's watchdog reads it ([`Service::report`]).
    service: Arc<Mutex<Service>>,
}

/// The request service's own state: its look at the decode requests, the
/// misses waiting, immutable per-layer scratch tables (so scratch arrival
/// observation runs outside `residency`'s lock), and per layer the scratch
/// request being served.
pub(super) struct Service {
    look: Look,
    decode: Decode,
    scratch_tables: Vec<ExchangeTable>,
    window_rounds: u64,
    /// Per layer, the scratch request taken and not yet served whole: a
    /// layer's requests are served in step order, one at a time, as the
    /// device runs their rounds.
    pending: Vec<Option<Pending>>,
    /// The service's rest ([`IDLE`] while a step runs; waits on events when
    /// idle), for the exchange's watchdog.
    rested: Rested,
}

impl Service {
    /// Its rest now, for the exchange's watchdog.
    pub(super) fn rested(&self) -> Rested {
        self.rested.clone()
    }

    /// What the service waits on, for the exchange's watchdog: each decode
    /// miss waiting, and each layer's scratch request in serving, by stage.
    pub(super) fn report(&self) -> String {
        let pending = self
            .pending
            .iter()
            .enumerate()
            .filter_map(|(layer, pending)| pending.as_ref().map(|pending| format!("layer {layer}: {}", pending.describe())))
            .collect::<Vec<_>>();
        format!("decode {}; scratch [{}]", self.decode.describe(), pending.join("; "))
    }
}

impl Requests {
    pub(super) fn new(residency: Arc<Yielding<Residency>>, shared: Arc<Shared>, copier: Option<super::Copier>) -> Self {
        let service = {
            let held = lock(&residency);
            let layers = held.core.layers.len();
            Service {
                look: Look::new(&held.core, held.prefetch.staging.clone()),
                decode: Decode::new(),
                scratch_tables: held.core.layers.iter().map(|layer| layer.scratch.clone()).collect(),
                window_rounds: held.core.extents.window_rounds,
                pending: (0..layers).map(|_| None).collect(),
                rested: Rested::fixed(IDLE),
            }
        };
        Self { residency, shared, copier, service: Arc::new(Mutex::new(service)) }
    }

    /// Its state, as the exchange's watchdog reads it.
    pub(super) fn service(&self) -> Arc<Mutex<Service>> {
        self.service.clone()
    }

    /// The service's thread: a look then a pass on every event, and while
    /// a step that may ask anything runs where kernels wait in place, within
    /// its rest of the last ([`Rest`]: a turn that changed nothing is a
    /// look's price, and each request and round it took a notice); a pass
    /// that progressed is an event itself
    /// (residency reads what it changed). Once serving ended, each pass
    /// refuses what the device waits on instead, and nothing is looked at.
    /// Every device and every completing step waits on this thread, so a
    /// panic while serving must not end it: it is recorded before
    /// residency's lock is released, and the state it left is never served
    /// from again.
    pub(super) fn serve(self) {
        let events = self.shared.events().clone();
        let residency = self.residency.clone();
        let mut ended = Instant::now();
        loop {
            let seen = events.seen();
            if self.shared.life.stopping() {
                return;
            }
            // Poll completed prefill `Pass` signals outside `residency`'s lock
            // and `NoBlockingAllocationGuard` so CUDA event teardown never runs
            // under a mutex.
            let (completed_passes, pass_error) = self.shared.transport.poll_completed_passes();
            if let Some(error) = pass_error {
                self.shared.end_serving(format!("a prefill pass failed: {error}"));
            }
            // The look runs once the service is counted waiting for
            // residency, so the keeper starts no job before its pass.
            let began = Began { ended, looked: Instant::now() };
            let mut posted = Vec::new();
            let mut scratch_arrived = Vec::new();
            let (progressed, polls) = {
                let mut held = lock_counted(&residency, || {
                    let (p, s) = self.look(began);
                    posted = p;
                    scratch_arrived = s;
                });
                // When the pass holds residency, for the read timeline: the
                // look's requests wait for it, their reads already running.
                for request in &posted {
                    seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("held"), request.sequence, (request.layer as i64, 0, 0));
                }
                let mut service = self.service.lock().expect("the request service is never left by a panic");
                let progressed = match self.shared.life.ended() {
                    // Serving ended: each request the look posted is given
                    // up, its reads' staging freed as they end, and refused.
                    Some(_) => {
                        let held = &mut *held;
                        let gave_up = posted.into_iter().fold(false, |gave_up, request| request.give_up(&mut held.core, &held.prefetch, &self.shared) | gave_up);
                        liveness::refuse(&held.core, &self.shared) | gave_up
                    }
                    None => self.pass(&mut service, &mut held, posted, &mut scratch_arrived, &completed_passes),
                };
                (progressed, self.polls(&held))
            };
            ended = Instant::now();
            if progressed {
                events.notify();
                continue;
            }
            match polls {
                true => events.wait_at_most(seen, IDLE),
                false => events.wait(seen),
            }
        }
    }

    /// The decode requests the device posted since the last look, each
    /// absent expert's read submitted (`Look::look`), and any pending scratch
    /// arrivals observed on `Service::scratch_tables` outside `residency`'s
    /// lock: none once serving ended.
    fn look(&self, began: Began) -> (Vec<Posted>, Vec<Option<Instant>>) {
        let _no_alloc = seismic::memory::enter_no_blocking_allocation();
        if self.shared.life.ended().is_some() {
            return (Vec::new(), Vec::new());
        }
        let mut service = self.service.lock().expect("the request service is never left by a panic");
        let posted = service.look.look(&self.shared, began);
        let arrived = service
            .pending
            .iter()
            .enumerate()
            .map(|(layer, pending)| match pending {
                Some(Pending::Arriving { at, window, .. }) => {
                    let lane = constants::ROUND_LANE(*window * service.window_rounds);
                    service.scratch_tables[layer].requested(at.point.sequence, lane).then(Instant::now)
                }
                _ => None,
            })
            .collect();
        (posted, arrived)
    }

    /// Whether the device's words are polled while a step runs: where kernels
    /// wait in place, while a step runs that may ask anything (or publish
    /// foresight hints/front rings while any group is not whole in Tier 0,
    /// including 100%-RAM discrete GPUs where `POLICY_RESOLVED == 1` but
    /// `POLICY_WHOLE == 0`, or while a prompt group runs).
    fn polls(&self, residency: &Residency) -> bool {
        let steps = &self.shared.steps;
        !self.shared.parks.kept && steps.in_flight() > 0 && (!residency.all_whole() || steps.scratch_running())
    }

    /// One pass: the misses waiting resumed, the conveyor carried, then
    /// each layer's requests served in step order (`posted`, the look's)
    /// and the hints read. A route publishes its request and the next
    /// layer's hints at once, and the look submitted the request's reads
    /// before this pass, so they reach the readers before any read of those
    /// hints, and a ring whose hinted layer's request is noticed is
    /// decided, not read (`Prefetch::rings`). True when it changed anything;
    /// each scratch round that arrived is counted `noticed`.
    fn pass(
        &self,
        service: &mut Service,
        residency: &mut Residency,
        posted: Vec<Posted>,
        scratch_arrived: &mut [Option<Instant>],
        completed_passes: &[crate::experts::transport::PassKey],
    ) -> bool {
        // A panic of the executor's work, or the device lost under the
        // parks, ends serving, as the service's own panic does.
        if let Some(reason) = self.shared.fault() {
            self.shared.end_serving(reason);
        }
        // A growth's slots joined since are landed into and hinted into
        // from this pass on.
        let joined = residency.join();
        // Release `ScratchRing` slots for prefill passes that just finished on
        // the GPU (before `Combine(L)` and `Front(L+1)`), and release per-layer
        // `HostTier` holds whose `PrefillFill` DMA completed on `Copies`.
        let released = residency.scratch.release_passes(completed_passes)
            | residency.scratch.settle_prefill_copies(&mut residency.core, &self.shared);
        let refilled = if residency.core.extents.tier_copy_dma && residency.core.directory.refill_short() {
            residency.refill_tier()
        } else {
            false
        };
        // The misses waiting for a tier or staging record, or a read
        // ahead's landing.
        let resumed = service.decode.resume(residency.serving());
        // A served group's reads that landed move on, ahead of its rounds.
        let conveyed = residency.scratch.convey(&mut residency.core, &self.shared);
        let served = self.serve_requests(
            service,
            residency,
            posted,
            scratch_arrived,
            joined | released | refilled | resumed | conveyed,
        );
        released | refilled | resumed | conveyed | served
    }

    /// Each layer's decode and scratch requests, served in step order, then
    /// the hints. The device's words (its requests and hints) are read only
    /// once they are in place: where a park is a submission boundary, while
    /// the device is parked (`Parks::parked`, the look's alike), its hints,
    /// which only ever advise, once they may be (`Parks::settled`). What the
    /// host alone holds (a scratch request taken, its rounds, the decode
    /// requests the look took, `posted`) is carried every pass. Whether it
    /// served any.
    fn serve_requests(
        &self,
        service: &mut Service,
        residency: &mut Residency,
        posted: Vec<Posted>,
        scratch_arrived: &mut [Option<Instant>],
        freed: bool,
    ) -> bool {
        let parked = self.shared.parks.parked();
        let layers = residency.core.layers.len();
        let mut progressed = false;
        let mut posted = posted.into_iter().peekable();
        for layer in 0..layers {
            // The look's requests, by layer: their reads already run, and
            // what decides their landings follows (the device's progress to
            // the request, `Residency::advance` — including reserve refill so
            // newly settled landings unmapped into the reserve are restamped
            // and readied by this park before the request is answered — and a
            // tier record each).
            while let Some(request) = posted.next_if(|request| request.layer == layer) {
                let at = self.shared.steps.at(request.sequence, layer, layers);
                residency.advance(at.point);
                service.decode.serve(request, at, residency.serving());
                progressed = true;
            }
            if parked {
                progressed |= residency.publish(layer);
            }
            let arrived = scratch_arrived.get_mut(layer).and_then(Option::take);
            progressed |= self.scratch(service, residency, layer, arrived);
        }
        assert!(posted.next().is_none(), "the look takes requests of the load's layers, in order");
        if self.shared.parks.settled() {
            progressed |= residency.rings(&self.shared, service.look.served(), freed | progressed, self.copier.as_ref());
        }
        progressed
    }

    /// Carry layer `layer`'s scratch requests, in step order, as far as
    /// each goes without waiting: one arriving is served once its round 0
    /// arrives (requests its lane of the step's scratch slot), which parks
    /// behind the backs of the group's steps before it there, so the device
    /// is then past them; then served round by round (`Scratch::carry`).
    /// Once one arrives every front of the layer has run, so every request
    /// of the layer is taken then: the layer is served whole before the
    /// next, whose requests wait on its backs. A step that ended before its
    /// round 0 arrived (a device error) fails its request. Whether any
    /// moved.
    fn scratch(&self, service: &mut Service, residency: &mut Residency, layer: usize, mut pre_arrived: Option<Instant>) -> bool {
        let layers = residency.core.layers.len();
        let mut moved = false;
        loop {
            let pending = match service.pending[layer].take() {
                Some(pending) => pending,
                None => match residency.scratch.unserved(&self.shared, layer) {
                    Some((place, window, sequence)) => Pending::Arriving {
                        at: self.shared.steps.at(sequence, layer, layers),
                        place,
                        window,
                    },
                    None => return moved,
                },
            };
            let serve = match pending {
                Pending::Served(serve) => serve,
                Pending::Arriving { at, place, window } => {
                    let sequence = at.point.sequence;
                    if residency.core.extents.tier_copy_dma {
                        if self.shared.steps.ended(sequence) {
                            residency.scratch.unarrived(at, place, window, &mut residency.core, &self.shared);
                            moved = true;
                            continue;
                        }
                        if !residency.scratch.can_arrive_async(layer, at.point.group, place) {
                            service.pending[layer] = Some(Pending::Arriving { at, place, window });
                            return moved;
                        }
                        residency.publish(layer);
                        if window == 0 {
                            residency.advance(at.point);
                        } else {
                            residency.advance_window();
                        }
                        moved = true;
                        let Some(serve) = residency.scratch.arrived(at, place, window, Instant::now(), &mut residency.core, &self.shared) else {
                            continue;
                        };
                        serve
                    } else {
                        let lane = constants::ROUND_LANE(window * service.window_rounds);
                        let arrived = match pre_arrived.take() {
                            Some(arrived) => arrived,
                            None => {
                                if !service.scratch_tables[layer].requested(sequence, lane) {
                                    if !self.shared.steps.ended(sequence) {
                                        service.pending[layer] = Some(Pending::Arriving { at, place, window });
                                        return moved;
                                    }
                                    residency.scratch.unarrived(at, place, window, &mut residency.core, &self.shared);
                                    moved = true;
                                    continue;
                                }
                                Instant::now()
                            }
                        };
                        residency.publish(layer);
                        if window == 0 {
                            residency.advance(at.point);
                        } else {
                            residency.advance_window();
                        }
                        moved = true;
                        // While the device arrived, the layers the group published
                        // since may have taken the request's tier with its layer.
                        let Some(serve) = residency.scratch.arrived(at, place, window, arrived, &mut residency.core, &self.shared) else {
                            continue;
                        };
                        serve
                    }
                }
            };
            let Residency { scratch, core, prefetch, ledger, .. } = residency;
            let (serve, carried) = scratch.carry(serve, core, prefetch, ledger, &self.shared);
            moved |= carried;
            if let Some(serve) = serve {
                service.pending[layer] = Some(Pending::Served(serve));
                return moved;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rest is `√(2 · l · T / (n + 1))` of the phase polled (`Rest`):
    /// none before any time of it is polled, at least that of the time
    /// polled with nothing noticed, shorter once many notices are counted;
    /// each phase counts apart, and from its start again once it begins.
    #[test]
    fn a_rest_is_priced_on_its_looks_and_its_phases_notices() {
        let decode = Some(Phase::Decode);
        let look = Duration::from_micros(4);
        let rested = Rested::default();
        let mut rest = Rest::new(rested.clone());
        assert_eq!(rest.looked(Looked::Empty(look), decode), Some(Duration::ZERO), "nothing polled: the rest is none");
        let polled = Duration::from_millis(1);
        std::thread::sleep(polled);
        let quiet = rest.looked(Looked::Empty(look), decode).unwrap();
        assert!(quiet >= Duration::from_secs_f64((2. * look.as_secs_f64() * polled.as_secs_f64()).sqrt()), "{quiet:?} after {polled:?} polled");
        assert_eq!(rested.get(), quiet, "the rest is published as set");
        let busy = rest.looked(Looked::Changed { noticed: 1000 }, decode).unwrap();
        assert!(busy < quiet, "notices shorten the rest: {busy:?} against {quiet:?}");
        assert_eq!(rest.looked(Looked::Empty(look), Some(Phase::Prompt)), Some(Duration::ZERO), "the prompt's count is its own: nothing of it polled");
        assert_eq!(rest.looked(Looked::Empty(look), None), None, "no rest where the poller waits on events");
        rest.begin(Phase::Decode);
        assert_eq!(rest.looked(Looked::Empty(look), decode), Some(Duration::ZERO), "a decode begun again counts from its start");
    }
}
