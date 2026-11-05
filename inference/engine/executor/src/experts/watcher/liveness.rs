//! Whether the watcher serves: the end of serving, the requests a watcher
//! that stopped refuses, and the exchange's watchdog over the load
//! ([`watch`]): its tables, its bound and what each owner reports.

use super::core::Core;
use super::requests::Service;
use super::residency::Residency;
use super::{Shared, decode, scratch};
use crate::assessment::{FileLine, ReadLineError};
use crate::experts::ExchangeTable;
use crate::experts::exchange::{Bound, Bounds, Kind, Trip, Watchdog, Watched, Witness};
use magnitude_kernels::constants;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::watched::{MULTIPLE, PieceBound};

/// What bounds every answer of the load's exchange, fixed as the watcher
/// starts beside the read line, which the run measures as it goes.
pub(super) struct Answers {
    /// Every record a read can land into at once: the staging records and
    /// the conveyor's. No answer waits on more reads than these: a read
    /// past them waits for a record the device frees, so for another lane.
    pub(super) records: u64,
    /// The stored bytes of the largest expert of any streamed layer.
    pub(super) bytes: u64,
    /// Each sited layer, its rows and most distinct experts of a request.
    pub(super) sites: Vec<(usize, f64, f64)>,
}

impl Answers {
    /// The longest any answer can take while the host progresses, at the
    /// served file's read line `line`, and how it was derived: every record
    /// in flight read one after another, then landed (a layer whose
    /// landings are unmeasured lands no slower than it reads the same
    /// records: a landing reads and writes each byte once in host memory),
    /// beside the CPU's longest request and the longest rest of a serving
    /// thread before it looks again. `landings` are each layer's landing
    /// lines now, and `priced` its lines as the load refines them
    /// (`Telemetry::refined`): the CPU's longest request is, per
    /// sited layer with lines, its distinct experts at most each cold over
    /// every row, and the handshake.
    pub(super) fn longest(&self, line: crate::assessment::ReadLine, landings: &[Option<FileLine>], priced: Option<&[crate::assessment::ExpertLines]>) -> Bound {
        // Every term is a measured line's at its records, and no line's
        // term is negative (`assessment::Terms`), so none shortens the bound.
        let read = self.records as f64 * line.seconds(self.bytes);
        let unmeasured = landings.iter().filter(|landing| landing.is_none()).count();
        let landing = landings.iter().map(|landing| landing.map_or(read, |landing| landing.seconds(self.records))).fold(0.0, f64::max);
        let notice = super::requests::IDLE;
        let cpu = self
            .sites
            .iter()
            .filter_map(|&(layer, rows, distinct)| {
                let lines = priced?[layer].host.cpu?;
                Some((distinct * lines.expert.nanoseconds(rows) + lines.handshake.get()) * 1e-9)
            })
            .fold(0.0, f64::max);
        let seconds = read + landing + cpu + notice.as_secs_f64();
        Bound {
            limit: Duration::from_secs_f64(seconds) * MULTIPLE,
            derivation: format!(
                "{MULTIPLE} × ({} records of {} bytes read {read:.3} s at {:.3} ms + {:.3} ms/GB + landed {landing:.3} s ({unmeasured} layers' landings unmeasured, at their reads) \
                 + the CPU's longest request {:.3} s + a rest {notice:?})",
                self.records,
                self.bytes,
                line.latency_seconds() * 1e3,
                line.seconds_per_byte() * 1e12,
                cpu
            ),
        }
    }
}

/// The exchange's watchdog over the load: every table of every streamed
/// layer, bounded by [`Answers::longest`] at the run's read line, reporting
/// the steps, the request service, residency, the transfer queue, the
/// parks, the CPU site and serving's life.
pub(super) fn watch(
    core: &Core,
    shared: &Arc<Shared>,
    residency: Arc<Mutex<Residency>>,
    service: Arc<Mutex<Service>>,
    copies: Option<crate::experts::copies::Copies>,
    answers: Answers,
    trip: Trip,
) -> Result<Watchdog, String> {
    let mut tables = Vec::new();
    for (layer, served) in core.layers.iter().enumerate() {
        tables.push(Watched { kind: Kind::File, layer, slots: served.file.clone() });
        tables.push(Watched { kind: Kind::Scratch, layer, slots: served.scratch.clone() });
        if let Some(site) = &served.site {
            tables.push(Watched { kind: Kind::Site, layer, slots: site.clone() });
        }
    }
    let bounds = {
        let shared = shared.clone();
        Box::new(move || {
            match shared.telemetry.line() {
                Some(line) => {
                    let priced = shared.priced().map(|lines| shared.telemetry.refined(&lines));
                    let lane = answers.longest(line, &shared.telemetry.landing_lines(), priced.as_deref());
                    Bounds { lane, pieces: PieceBound(Some(line)) }
                }
                None => Bounds::unpriced(),
            }
        })
    };
    let reads = Box::new(core.source.reads());
    let witnesses = vec![
        Witness { name: "steps", report: Arc::new({ let shared = shared.clone(); move || shared.steps.describe() }) },
        Witness { name: "request service", report: Arc::new(move || held(&service, Service::report)) },
        Witness { name: "residency", report: Arc::new(move || held(&residency, Residency::describe)) },
        Witness {
            name: "transfer queue",
            report: Arc::new(move || copies.as_ref().map_or_else(|| "none".into(), crate::experts::copies::Copies::describe)),
        },
        Witness {
            name: "parks",
            report: Arc::new({
                let shared = shared.clone();
                move || shared.parks.describe().unwrap_or_else(|| "held by another thread".into())
            }),
        },
        Witness { name: "CPU site", report: Arc::new({ let shared = shared.clone(); move || shared.compute.report() }) },
        Witness {
            name: "serving",
            report: Arc::new({
                let shared = shared.clone();
                move || format!("ended {:?}, stopping {}", shared.life.ended(), shared.life.stopping())
            }),
        },
    ];
    let ended = {
        let steps = shared.steps.clone();
        Box::new(move |step| steps.ended(step))
    };
    let parked = {
        let shared = shared.clone();
        Box::new(move || shared.parks.reached())
    };
    Watchdog::start(tables, parked, ended, shared.events().clone(), reads, bounds, witnesses, trip)
}

/// `describe` of what `mutex` guards, once it is free within a short
/// while: a thread that holds it longer is stuck inside it, which is
/// itself the report.
fn held<T>(mutex: &Mutex<T>, describe: fn(&T) -> String) -> String {
    let deadline = std::time::Instant::now() + Duration::from_millis(100);
    loop {
        match mutex.try_lock() {
            Ok(guard) => return describe(&guard),
            Err(std::sync::TryLockError::Poisoned(_)) => return "left by a panic".into(),
            Err(std::sync::TryLockError::WouldBlock) if std::time::Instant::now() < deadline => std::thread::yield_now(),
            Err(std::sync::TryLockError::WouldBlock) => return "held by its thread past 100 ms: stuck inside a pass".into(),
        }
    }
}

pub(in crate::experts) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    match (payload.downcast_ref::<&str>(), payload.downcast_ref::<String>()) {
        (Some(message), _) => (*message).to_owned(),
        (_, Some(message)) => message.clone(),
        _ => "a panic without a message".to_owned(),
    }
}

/// Whether serving goes on: why it ended, and whether the load stops.
pub(super) struct Life {
    /// Why the watcher's serving ended: it panicked, or a demand read left
    /// the served file with no read line. Recorded before any request is
    /// refused for it.
    ended: Mutex<Option<String>>,
    stopping: AtomicBool,
}

impl Life {
    pub(super) fn new() -> Self {
        Self {
            ended: Mutex::new(None),
            stopping: AtomicBool::new(false),
        }
    }

    pub(super) fn ended(&self) -> Option<String> {
        self.ended.lock().expect("watcher ended").clone()
    }

    /// End serving for `reason`, unless it ended already
    /// ([`Shared::end_serving`]).
    fn end(&self, reason: String) {
        self.ended.lock().expect("watcher ended").get_or_insert(reason);
    }

    /// The load stops: every serving thread returns as it next looks (the
    /// stop's event wakes it, `Watcher::drop`).
    pub(super) fn stop(&self) {
        self.stopping.store(true, Ordering::Release);
    }

    pub(super) fn stopping(&self) -> bool {
        self.stopping.load(Ordering::Acquire)
    }
}

/// The request service's pass once serving ended: fail every request the
/// device may wait on, so its waiting launches return: a file request not
/// answered, and a scratch request whose last round is not (every round of
/// it fails with it). One of a step that ended has nothing waiting on it.
pub(super) fn refuse(core: &Core, shared: &Shared) -> bool {
    let reason = || "the expert watcher stopped serving".to_owned();
    // The steps `slots` holds requests of in `lane`, begun and not ended,
    // whose lane `answered` is not answered.
    let open = |slots: &ExchangeTable, lane: u64, answered: u64| {
        (0..slots.table().slots())
            .map(|slot| slots.held(slot, lane).0)
            .filter(|&step| step != 0 && !shared.steps.ended(step) && slots.answer_of(step, answered).is_none())
            .collect::<Vec<_>>()
    };
    let mut refused = false;
    for (layer, served) in core.layers.iter().enumerate() {
        for sequence in open(&served.file, 0, 0) {
            decode::fail(core, shared, layer, sequence, reason());
            refused = true;
        }
        let last = served.scratch.table().lanes() - 1;
        for sequence in open(&served.scratch, constants::SCRATCH_ASK, last) {
            scratch::fail(core, shared, layer, sequence, reason());
            refused = true;
        }
    }
    refused
}

impl Shared {
    /// End the watcher's serving for `reason`, unless it ended already,
    /// and wake an owner waiting for a resize's answer, or a prompt group's
    /// begin waiting for a copy, neither of which comes; an event, so the
    /// request service turns to refusing at once. Never called under the
    /// lent records' lock.
    pub(super) fn end_serving(&self, reason: String) {
        self.life.end(reason);
        self.resizing.wake();
        self.lent.wake();
        self.events().notify();
    }

    /// Count a read step `sequence`'s decode request at `layer` named that
    /// landed (`Telemetry::landed`): one that leaves the served file with
    /// no read line ends serving: every rule that weighs reads against
    /// compute reads that line.
    pub(super) fn landed(
        &self,
        layer: usize,
        sequence: super::Step,
        read: std::time::Duration,
        timed: Option<(u64, std::time::Duration)>,
        queued: std::time::Duration,
    ) {
        if let Err(error) = self.telemetry.landed(layer, sequence, read, timed, queued) {
            self.lost_read_line(error);
        }
    }

    /// Why serving ends that no serving thread met itself: a panic of the
    /// executor's work, or the device lost under the parks.
    pub(super) fn fault(&self) -> Option<String> {
        self.compute.panicked().or_else(|| self.parks.lost())
    }

    /// The served file's reads make no read line (`error`): serving ends,
    /// as every rule that weighs reads against compute reads that line.
    pub(super) fn lost_read_line(&self, error: ReadLineError) {
        self.end_serving(format!("the served file's read line: {error}"));
    }
}
