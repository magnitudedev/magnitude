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
//! looks again within [`IDLE`] of its last look: the one poll of serving,
//! which the hardware forces. Each turn looks at the decode requests first,
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
use super::scratch::{self, Pending};
use super::{Shared, panic_message};
use crate::experts::yielding::Yielding;
use std::panic::AssertUnwindSafe;
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

/// The request service: what it serves through, and its own state.
pub(super) struct Requests {
    residency: Arc<Yielding<Residency>>,
    shared: Arc<Shared>,
    /// Held by its thread through each pass, beside residency's lock: the
    /// exchange's watchdog reads it ([`Service::report`]).
    service: Arc<Mutex<Service>>,
}

/// The request service's own state: its look at the decode requests, the
/// misses waiting, and per layer the scratch request being served.
pub(super) struct Service {
    look: Look,
    decode: Decode,
    /// Per layer, the scratch request taken and not yet served whole: a
    /// layer's requests are served in step order, one at a time, as the
    /// device runs their rounds.
    pending: Vec<Option<Pending>>,
}

impl Service {
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
    pub(super) fn new(residency: Arc<Yielding<Residency>>, shared: Arc<Shared>) -> Self {
        let service = {
            let held = lock(&residency);
            let layers = held.core.layers.len();
            Service { look: Look::new(&held.core, held.prefetch.staging.clone()), decode: Decode::new(), pending: (0..layers).map(|_| None).collect() }
        };
        Self { residency, shared, service: Arc::new(Mutex::new(service)) }
    }

    /// Its state, as the exchange's watchdog reads it.
    pub(super) fn service(&self) -> Arc<Mutex<Service>> {
        self.service.clone()
    }

    /// The service's thread: a look then a pass on every event, and while
    /// a step that may ask anything runs where kernels wait in place, within
    /// [`IDLE`] of the last; a pass that progressed is an event itself
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
            // The look runs once the service is counted waiting for
            // residency, so the keeper starts no job before its pass.
            let began = Began { ended, looked: Instant::now() };
            let mut posted = Vec::new();
            let (progressed, polls) = {
                let mut held = lock_counted(&residency, || posted = self.look(began));
                // When the pass holds residency, for the read timeline: the
                // look's requests wait for it, their reads already running.
                for request in &posted {
                    seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("held"), request.sequence, (request.layer as i64, 0, 0));
                }
                let mut service = self.service.lock().expect("the request service is never left by a panic");
                let progressed = match self.shared.life.ended() {
                    Some(_) => liveness::refuse(&held.core, &self.shared),
                    None => match std::panic::catch_unwind(AssertUnwindSafe(|| self.pass(&mut service, &mut held, posted))) {
                        Ok(progressed) => progressed,
                        Err(payload) => {
                            self.shared.end_serving(format!("it panicked: {}", panic_message(payload.as_ref())));
                            true
                        }
                    },
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
    /// absent expert's read submitted (`Look::look`): none once serving
    /// ended. A panic ends serving, as the pass's does.
    fn look(&self, began: Began) -> Vec<Posted> {
        if self.shared.life.ended().is_some() {
            return Vec::new();
        }
        let mut service = self.service.lock().expect("the request service is never left by a panic");
        match std::panic::catch_unwind(AssertUnwindSafe(|| service.look.look(&self.shared, began))) {
            Ok(posted) => posted,
            Err(payload) => {
                self.shared.end_serving(format!("its look panicked: {}", panic_message(payload.as_ref())));
                Vec::new()
            }
        }
    }

    /// Whether the device's words are looked at by time: where kernels wait
    /// in place, while a step runs that may ask anything (with every layer
    /// whole a decode step asks nothing, no miss and no hint, but a prompt
    /// group's requests may come).
    fn polls(&self, residency: &Residency) -> bool {
        let steps = &self.shared.steps;
        !self.shared.parks.kept && steps.in_flight() > 0 && (!residency.whole() || steps.scratch_running())
    }

    /// One pass: the misses waiting resumed, the conveyor carried, then
    /// each layer's requests served in step order (`posted`, the look's)
    /// and the hints read. A route publishes its request and the next
    /// layer's hints at once, and the look submitted the request's reads
    /// before this pass, so they reach the readers before any read of those
    /// hints, and a ring whose hinted layer's request is noticed is
    /// decided, not read (`Prefetch::hint`). True when it changed anything.
    fn pass(&self, service: &mut Service, residency: &mut Residency, posted: Vec<Posted>) -> bool {
        // A panic of the executor's work, or the device lost under the
        // parks, ends serving, as the service's own panic does.
        if let Some(reason) = self.shared.fault() {
            self.shared.end_serving(reason);
        }
        // A growth's slots joined since are landed into and hinted into
        // from this pass on.
        residency.join();
        // The misses waiting for a tier or staging record, or a read
        // ahead's landing.
        let resumed = service.decode.resume(residency.serving());
        // A served group's reads that landed move on, ahead of its rounds.
        let conveyed = residency.scratch.convey(&mut residency.core, &self.shared);
        let served = self.serve_requests(service, residency, posted);
        resumed | conveyed | served
    }

    /// Each layer's decode and scratch requests, served in step order, then
    /// the hints. The device's words (its requests and hints) are read only
    /// once they are in place: where a park is a submission boundary, while
    /// the device is parked (`Parks::parked`, the look's alike), its hints,
    /// which only ever advise, once they may be (`Parks::settled`). What the
    /// host alone holds (a scratch request taken, its rounds, the decode
    /// requests the look took, `posted`) is carried every pass. Whether it
    /// served any.
    fn serve_requests(&self, service: &mut Service, residency: &mut Residency, posted: Vec<Posted>) -> bool {
        let parked = self.shared.parks.parked();
        let layers = residency.core.layers.len();
        let mut progressed = false;
        let mut posted = posted.into_iter().peekable();
        for layer in 0..layers {
            // The look's requests, by layer: their reads already run, and
            // what decides their landings follows (the device's progress to
            // the request, `Residency::advance`, and a tier record each).
            while let Some(request) = posted.next_if(|request| request.layer == layer) {
                let at = self.shared.steps.at(request.sequence, layer, layers);
                residency.advance(at.point);
                service.decode.serve(request, at, residency.serving());
                progressed = true;
            }
            if parked {
                progressed |= residency.publish(layer);
            }
            progressed |= self.scratch(service, residency, layer);
        }
        assert!(posted.next().is_none(), "the look takes requests of the load's layers, in order");
        if self.shared.parks.settled() {
            progressed |= residency.prefetch.hint(&mut residency.core, &self.shared, service.look.served());
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
    fn scratch(&self, service: &mut Service, residency: &mut Residency, layer: usize) -> bool {
        let layers = residency.core.layers.len();
        let mut moved = false;
        loop {
            let pending = match service.pending[layer].take() {
                Some(pending) => pending,
                None => match residency.scratch.unserved(&self.shared, layer) {
                    Some((place, sequence)) => Pending::Arriving { at: self.shared.steps.at(sequence, layer, layers), place },
                    None => return moved,
                },
            };
            let serve = match pending {
                Pending::Served(serve) => serve,
                Pending::Arriving { at, place } => {
                    let sequence = at.point.sequence;
                    if !residency.core.layers[layer].scratch.requested(sequence, scratch::round_lane(0)) {
                        if !self.shared.steps.ended(sequence) {
                            service.pending[layer] = Some(Pending::Arriving { at, place });
                            return moved;
                        }
                        residency.scratch.unarrived(at, place, &residency.core, &self.shared);
                        moved = true;
                        continue;
                    }
                    let arrived = Instant::now();
                    residency.publish(layer);
                    residency.advance(at.point);
                    moved = true;
                    // While the device arrived, the layers the group published
                    // since may have taken the request's tier with its layer.
                    let Some(serve) = residency.scratch.arrived(at, place, arrived) else {
                        continue;
                    };
                    serve
                }
            };
            let Residency { scratch, core, prefetch, .. } = residency;
            let (serve, carried) = scratch.carry(serve, core, prefetch, &self.shared);
            moved |= carried;
            if let Some(serve) = serve {
                service.pending[layer] = Some(Pending::Served(serve));
                return moved;
            }
        }
    }
}
