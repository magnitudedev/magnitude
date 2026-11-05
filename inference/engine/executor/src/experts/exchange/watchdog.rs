//! The exchange's watchdog: the one owner of the exchange's liveness.
//!
//! Every wait of the device on the host is a lane of a slot of the
//! exchange, and every one the device waits on is answered (`watcher`,
//! `site`). A lane the device waits on whose request stays newer than its
//! answer past the load's bound ([`Bound`]: a generous multiple of the
//! longest any answer can take while the host progresses), counted in the
//! time the host was able to answer and did not progress ([`HostClock`]),
//! means the host cannot answer it, and the device waiting on it never will
//! be released.
//! The watchdog then ends the process with everything needed to find why:
//! each lane unanswered, whether any host owner raised an event since, and
//! each owner's state as it reports it ([`Witness`]).
//!
//! A lane the device waits on ([`Kind::waited`]) is a file slot's (a
//! decode layer's tier wait), a site slot's (a sited combine's sum) and a
//! scratch slot's round lanes (each round's wait, requested as the device
//! arrives there). A scratch request's own lane is answered only once its
//! round 0 arrives, after every earlier place's back at its layer, and the
//! device waits on nothing there: it is reported, never timed. A step that
//! ended (a device error) leaves nothing waiting. Where parks are
//! submission boundaries (Vulkan), the device also waits at a park it has
//! reached until an answer releases it, whether or not its lane's request
//! was ever written: such a park is watched as a lane is ([`Parked`]).

use super::{FAILED, ExchangeTable};
use crate::experts::events::Events;
use magnitude_kernels::constants;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use crate::watched::{HostClock, PieceBound, Reads, Serving, UNPRICED_PROGRESS};
pub(in crate::experts) use crate::watched::{Bound, Trip, abort};
use std::time::{Duration, Instant};

/// Which kind of the exchange a table is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) enum Kind {
    File,
    Site,
    Scratch,
}

impl Kind {
    /// The lanes of a slot of this kind the device waits on.
    fn waited(self, lanes: u64) -> std::ops::Range<u64> {
        match self {
            Self::File | Self::Site => 0..1,
            Self::Scratch => constants::SCRATCH_ROUND..lanes,
        }
    }
}

/// A table of the exchange the watchdog watches: its kind, its streamed
/// layer and its slots.
pub(in crate::experts) struct Watched {
    pub kind: Kind,
    pub layer: usize,
    pub slots: ExchangeTable,
}

/// An owner's state as the watchdog reports it: its name and its report,
/// which never waits on the owner (a stuck owner reports that it is held).
/// Each is the process's report on demand too while the watchdog runs
/// (`crate::diagnostics`).
pub(in crate::experts) struct Witness {
    pub name: &'static str,
    pub report: Arc<dyn Fn() -> String + Send + Sync>,
}


/// A park the device has reached and no answer has released, where parks
/// are submission boundaries (`watcher::parks`): the device waits there on
/// the host whoever writes its lane's request, so it is watched as a lane
/// is. Its value, its step, and where it is.
pub(in crate::experts) struct Parked {
    pub value: u64,
    pub step: u64,
    pub site: String,
}

/// The load's park the device waits at, if any ([`Parked`]).
pub(in crate::experts) type ParkedAt = Box<dyn Fn() -> Option<Parked> + Send>;

/// Whether a step has ended, by whichever exit (`Steps::ended`).
pub(in crate::experts) type Ended = Box<dyn Fn(u64) -> bool + Send>;

/// A waited lane seen unanswered: where, what it held, and when it was
/// first seen so, with the load's events then.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Lane {
    table: usize,
    slot: u64,
    lane: u64,
    request: u64,
    answer: u64,
}

struct Open {
    lane: Lane,
    /// The host's time able to answer without progress ([`HostClock`])
    /// when first seen unanswered.
    since: Duration,
    events: u64,
}

/// What bounds the load's waits at a look, as its lines are measured: each
/// lane and park, and each piece of its reads in flight.
pub(in crate::experts) struct Bounds {
    pub lane: Bound,
    pub pieces: PieceBound,
}

impl Bounds {
    /// The bounds of a load none of whose reads made a read line yet: its
    /// reads' progress alone, which takes no measured figure. A piece in
    /// flight past [`UNPRICED_PROGRESS`] is no progress, and a lane or park
    /// then waits as long again, with the host's time able to answer
    /// counted as ever.
    pub fn unpriced() -> Self {
        let derivation = "no read line yet: the load's reads' progress alone".to_owned();
        Self { lane: Bound { limit: UNPRICED_PROGRESS, derivation }, pieces: PieceBound(None) }
    }
}

/// The running watchdog: its thread, until it drops.
pub(in crate::experts) struct Watchdog {
    stop: Arc<(Mutex<bool>, Condvar)>,
    thread: Option<JoinHandle<()>>,
    /// Its witnesses' registrations in the process's report on demand.
    _reports: Vec<crate::diagnostics::Registered>,
}

impl Watchdog {
    /// Watch `tables` of the load and the park it waits at (`parked`),
    /// whose steps end as `ended` says, whose events are `events` and whose
    /// reads of the read engine are as `reads` says ([`Serving`]), each
    /// lane, park and piece in flight bounded by `bounds` (read at every
    /// look, as the load's lines are measured), reporting `witnesses` and
    /// the engine's pieces in flight when one trips.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::experts) fn start(
        tables: Vec<Watched>,
        parked: ParkedAt,
        ended: Ended,
        events: Arc<Events>,
        reads: Box<dyn Fn() -> Reads + Send>,
        bounds: Box<dyn Fn() -> Bounds + Send>,
        witnesses: Vec<Witness>,
        trip: Trip,
    ) -> Result<Self, String> {
        let reports = witnesses
            .iter()
            .map(|witness| {
                let report = witness.report.clone();
                crate::diagnostics::register(witness.name, move || report())
            })
            .collect();
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let stopped = stop.clone();
        let thread = std::thread::Builder::new()
            .name("expert-exchange-watchdog".into())
            .spawn(move || {
                let watch = Watch::new(tables, parked, ended, events, reads, witnesses, Instant::now());
                run(watch, &*bounds, trip, &stopped)
            })
            .map_err(|error| format!("the exchange's watchdog: {error}"))?;
        Ok(Self { stop, thread: Some(thread), _reports: reports })
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        let (stopped, changed) = &*self.stop;
        *stopped.lock().expect("the watchdog's stop") = true;
        changed.notify_all();
        if let Some(thread) = self.thread.take() {
            thread.join().expect("the exchange's watchdog never panics");
        }
    }
}

/// Every waited lane of `tables` whose request is newer than its answer,
/// of a step not ended.
fn unanswered(tables: &[Watched], ended: &dyn Fn(u64) -> bool) -> Vec<Lane> {
    let mut open = Vec::new();
    for (table, watched) in tables.iter().enumerate() {
        let shape = watched.slots.table();
        for slot in 0..shape.slots() {
            for lane in watched.kind.waited(shape.lanes()) {
                let (request, answer) = watched.slots.held(slot, lane);
                if request != 0 && answer & !FAILED < request && !ended(request) {
                    open.push(Lane { table, slot, lane, request, answer });
                }
            }
        }
    }
    open
}

/// The watchdog's thread: a look every tenth of the bound, until stopped
/// or a lane or the park is overdue.
fn run(mut watch: Watch, bounds: &dyn Fn() -> Bounds, trip: Trip, stop: &(Mutex<bool>, Condvar)) {
    loop {
        let bounds = bounds();
        {
            let (stopped, changed) = stop;
            let stopped = stopped.lock().expect("the watchdog's stop");
            let period = bounds.lane.limit / 10;
            let (stopped, _) = changed.wait_timeout_while(stopped, period, |stopped| !*stopped).expect("the watchdog's stop");
            if *stopped {
                return;
            }
        }
        if let Some(report) = watch.look(Instant::now(), &bounds) {
            trip(report);
            return;
        }
    }
}

/// What the watchdog watches and has seen: the load's tables, park, steps,
/// events and reads, the lanes open, the park the device waits at, and the
/// host's time able to answer without progress. Its looks are the
/// watchdog's every decision, at the instants its thread looks.
struct Watch {
    tables: Vec<Watched>,
    parked: ParkedAt,
    ended: Ended,
    events: Arc<Events>,
    reads: Box<dyn Fn() -> Reads + Send>,
    witnesses: Vec<Witness>,
    open: Vec<Open>,
    /// The park the device waits at: its value, the host's time able to
    /// answer without progress when first seen, and the load's events then.
    park: Option<(u64, Duration, u64)>,
    clock: HostClock,
    serving: Serving,
}

impl Watch {
    /// The watch of a load ([`Watchdog::start`]) begun at `at`.
    fn new(
        tables: Vec<Watched>,
        parked: ParkedAt,
        ended: Ended,
        events: Arc<Events>,
        reads: Box<dyn Fn() -> Reads + Send>,
        witnesses: Vec<Witness>,
        at: Instant,
    ) -> Self {
        let serving = Serving::new(&reads().engine);
        Self { tables, parked, ended, events, reads, witnesses, open: Vec::new(), park: None, clock: HostClock::new(at), serving }
    }

    /// A look at `at`, a tenth of `bounds`' lane bound after the last: the
    /// report of the lane or park overdue, if one is.
    fn look(&mut self, at: Instant, bounds: &Bounds) -> Option<String> {
        let Bounds { lane: Bound { limit, derivation }, pieces } = bounds;
        let (limit, pieces) = (*limit, *pieces);
        let (tables, ended, witnesses) = (&self.tables, &*self.ended, &self.witnesses);
        let looked = (self.reads)();
        let served = self.serving.look(&looked, pieces, at);
        let (now, seen) = (self.clock.look(at, limit / 10, served), self.events.seen());
        let engine = || format!("{}, the load's reads outstanding {}", pieces.describe(&looked.engine, at), looked.outstanding);
        let current = unanswered(tables, ended);
        self.open.retain(|open| current.contains(&open.lane));
        for &lane in &current {
            if !self.open.iter().any(|open| open.lane == lane) {
                self.open.push(Open { lane, since: now, events: seen });
            }
        }
        if let Some(oldest) = self.open.iter().filter(|open| now - open.since >= limit).min_by_key(|open| open.since) {
            let header = format!(
                "magnitude-engine: the exchange left a request the device waits on unanswered past its bound of {limit:?} ({derivation}); \
                 the host cannot answer it."
            );
            return Some(report(header, tables, &self.open, oldest.events, now, seen, ended, &engine(), witnesses));
        }
        match (self.parked)().filter(|at| !ended(at.step)) {
            Some(at) => {
                if self.park.is_none_or(|(value, ..)| value != at.value) {
                    self.park = Some((at.value, now, seen));
                }
                let (_, since, events) = self.park.expect("the park is recorded above");
                if now - since >= limit {
                    let header = format!(
                        "magnitude-engine: the device is parked at {} (value {}, step {}) and no answer released it past its bound of {limit:?} ({derivation}); \
                         the host cannot answer it.",
                        at.site, at.value, at.step
                    );
                    return Some(report(header, tables, &self.open, events, now, seen, ended, &engine(), witnesses));
                }
            }
            None => self.park = None,
        }
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn report(
    header: String,
    tables: &[Watched],
    open: &[Open],
    since_events: u64,
    now: Duration,
    seen: u64,
    ended: &dyn Fn(u64) -> bool,
    engine: &str,
    witnesses: &[Witness],
) -> String {
    use std::fmt::Write;
    let mut report = header + "\n  unanswered (kind layer slot lane: request step, answer word, for, in the host's time able to answer without progress):";
    let mut lanes = open.iter().collect::<Vec<_>>();
    lanes.sort_by_key(|open| open.since);
    for open in lanes {
        let Lane { table, slot, lane, request, answer } = open.lane;
        let watched = &tables[table];
        let _ = write!(
            report,
            "\n    {:?} {} {slot} {lane}: {request}, {answer:#x}, {:?}",
            watched.kind,
            watched.layer,
            now - open.since
        );
    }
    // A scratch request's own lane is never timed (the module
    // documentation), but where one is taken but unanswered it says how
    // far serving got.
    let asks = tables
        .iter()
        .filter(|watched| watched.kind == Kind::Scratch)
        .flat_map(|watched| {
            (0..watched.slots.table().slots()).filter_map(move |slot| {
                let (request, answer) = watched.slots.held(slot, constants::SCRATCH_ASK);
                (request != 0 && answer & !FAILED < request && !ended(request)).then_some((watched.layer, slot, request, answer))
            })
        })
        .collect::<Vec<_>>();
    let _ = write!(report, "\n  scratch requests taken by the device and unanswered (layer slot: request step, answer word):");
    for (layer, slot, request, answer) in asks {
        let _ = write!(report, " {layer} {slot}: {request}, {answer:#x};");
    }
    let raised = seen - since_events;
    let _ = write!(
        report,
        "\n  the load's events: {seen}, {raised} raised since the oldest was first seen{}",
        if raised == 0 { ": no host owner progressed" } else { "" }
    );
    let _ = write!(report, "\n  the read engine: {engine}");
    for witness in witnesses {
        let _ = write!(report, "\n  {}: {}", witness.name, (witness.report)());
    }
    report
}

#[cfg(test)]
mod tests;
