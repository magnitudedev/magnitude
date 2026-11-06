//! Watched waits: a blocking wait
//! the load or serving makes is owed within a bound derived from the work
//! it waits on, measured in the host's time able to answer without progress
//! ([`HostClock`]); while it blocks it is in the process's report on demand
//! (`crate::diagnostics`), and past its bound it prints that report, naming
//! its owner, and ends the process, where a raw wait would hang silently.
//! A wait for work that may never be owed (a worker between jobs) is
//! [`Wait::idle`]: visible, never tripped.
//!
//! The exchange's watchdog measures its lanes on the same clock and trips
//! the same way.

use crate::assessment::ReadLine;
use seismic_io::{EngineWork, Flight};
use std::sync::{Condvar, MutexGuard};
use std::time::{Duration, Instant};

/// How many times the longest its awaited work can take while the host
/// progresses a wait may last: that work's bound already takes every term
/// serially, each at its measured line, so a slow but progressing host
/// stays well inside it.
pub(crate) const MULTIPLE: u32 = 10;

/// How long a wait may last, in the host's time able to answer without
/// progress ([`HostClock`]), and how that was derived from the load's
/// lines.
pub struct Bound {
    pub limit: Duration,
    pub derivation: String,
}

/// What a waiter's owner waits on of the read engine at one look: whether
/// it has reads submitted and not yet ended, and the engine's work then.
#[derive(Clone, Debug)]
pub(crate) struct Reads {
    pub(crate) outstanding: bool,
    pub(crate) engine: EngineWork,
}

/// How long a load's reads may go without progress before any of them made
/// a read line: no line prices a piece yet, so this is what the bounds are
/// until one does. Not a cost: nothing is priced by it, and only a read the
/// storage never returns reaches it.
pub(crate) const UNPRICED_PROGRESS: Duration = Duration::from_secs(60);

/// How long one piece of a read may be in flight: [`MULTIPLE`] times the
/// read line's time for it and for every other piece the engine has in
/// flight on its device, each read on its own, all as one snapshot of the
/// engine's work shows them: the longest the device takes over them,
/// whether it serves them in turn or together. A slow piece on a volume saturated by the
/// engine's own reads stays inside it; one the storage never returns does
/// not. Before the load has a read line (none), [`UNPRICED_PROGRESS`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct PieceBound(pub(crate) Option<ReadLine>);

impl PieceBound {
    /// The bound of `engine`'s piece in flight `flight` (one of its
    /// `reading`).
    pub(crate) fn of(self, flight: &Flight, engine: &EngineWork) -> Duration {
        let Some(line) = self.0 else {
            return UNPRICED_PROGRESS;
        };
        let sharing = engine
            .reading
            .iter()
            .filter(|other| other.volume == flight.volume)
            .map(|other| line.seconds(other.bytes))
            .sum::<f64>();
        Duration::from_secs_f64(sharing) * MULTIPLE
    }

    /// Whether `engine`'s piece in flight `flight` is past its bound at `now`.
    fn overdue(self, flight: &Flight, engine: &EngineWork, now: Instant) -> bool {
        now.saturating_duration_since(flight.since) >= self.of(flight, engine)
    }

    /// Each piece `engine` has in flight at `now`: its file, bytes, time in
    /// flight and bound, and whether it is past it.
    pub(crate) fn describe(self, engine: &EngineWork, now: Instant) -> String {
        let flights = engine
            .reading
            .iter()
            .map(|flight| {
                format!(
                    "{} at {} ({} bytes, volume {}) for {:?} of {:?}{}",
                    flight.path.display(),
                    flight.offset,
                    flight.bytes,
                    flight.volume,
                    now.saturating_duration_since(flight.since),
                    self.of(flight, engine),
                    if self.overdue(flight, engine, now) { ", past its bound" } else { "" }
                )
            })
            .collect::<Vec<_>>();
        format!("{} pieces ended, in flight [{}]", engine.ended, flights.join("; "))
    }
}

/// Whether an owner's reads were being served between two looks: they were
/// outstanding and the read engine progressed, ending a piece meanwhile or
/// reading one now, with no piece it reads past its own bound
/// ([`PieceBound`]). While they are, the engine is serving them or the reads
/// queued ahead of them, however slowly the storage returns each piece; an
/// engine holding a piece the storage never returns, or reading nothing
/// and ending nothing while reads are owed, is not serving them.
pub(crate) struct Serving {
    ended: u64,
}

impl Serving {
    pub(crate) fn new(engine: &EngineWork) -> Self {
        Self { ended: engine.ended }
    }

    /// Whether `reads`, seen at `now` with each piece bounded by `pieces`,
    /// were served since the last look.
    pub(crate) fn look(&mut self, reads: &Reads, pieces: PieceBound, now: Instant) -> bool {
        let engine = &reads.engine;
        let progressed = (engine.ended != self.ended || !engine.reading.is_empty())
            && !engine.reading.iter().any(|flight| pieces.overdue(flight, engine, now));
        self.ended = engine.ended;
        reads.outstanding && progressed
    }
}

/// How long the host has been able to answer and has not progressed, since
/// the clock began: what a wait's time is measured in, so neither a host
/// the system does not schedule (a starved process) nor one whose reads the
/// read engine is serving ([`Serving`]) is taken for one that cannot answer.
/// The waiter is the clock's heartbeat: it looks every tenth of the bound,
/// and each look credits the time since the one before up to twice that
/// interval, unless the owner's reads were served meanwhile. A look later
/// than that was a time the process was not run, and neither were the
/// threads that answer. A host whose answerers are runnable but wait for
/// what never comes is credited in full and trips within its bound,
/// whatever else the process's engine reads, and so is one whose reads the
/// engine does not serve.
pub(crate) struct HostClock {
    looked: Instant,
    pub(crate) ran: Duration,
}

impl HostClock {
    pub(crate) fn new(now: Instant) -> Self {
        Self { looked: now, ran: Duration::ZERO }
    }

    /// The look at `now` of a waiter that looks every `interval`, its
    /// owner's reads `served` since the last: the host's time able to answer
    /// without progress since the clock began.
    pub(crate) fn look(&mut self, now: Instant, interval: Duration, served: bool) -> Duration {
        if !served {
            self.ran += now.duration_since(self.looked).min(interval * 2);
        }
        self.looked = now;
        self.ran
    }
}

/// What a wait past its bound does with its report: print it and end the
/// process ([`abort`]), or, in a test, hand it over.
pub(crate) type Trip = Box<dyn Fn(String) + Send + Sync>;

/// Print `report` and end the process: on the process's own stderr, as a
/// test's captured output ends with the process unprinted.
pub(crate) fn abort() -> Trip {
    Box::new(|report| {
        use std::io::Write;
        // The process ends either way.
        drop(writeln!(std::io::stderr(), "{report}"));
        std::process::abort();
    })
}

/// One blocking wait: its owner, what it waits for, and its bound (none for
/// an idle wait).
pub struct Wait {
    owner: &'static str,
    what: String,
    bound: Option<Bound>,
    trip: Trip,
}

impl Wait {
    /// A wait by `owner` for `what`, owed within `bound`.
    pub fn owed(owner: &'static str, what: impl Into<String>, bound: Bound) -> Self {
        Self { owner, what: what.into(), bound: Some(bound), trip: abort() }
    }

    /// A wait by `owner` for `what`, which may never be owed: visible in the
    /// report on demand, never tripped.
    pub fn idle(owner: &'static str, what: impl Into<String>) -> Self {
        Self { owner, what: what.into(), bound: None, trip: abort() }
    }

    /// This wait handing its report to `trip` past its bound instead of
    /// ending the process.
    #[cfg(test)]
    pub(crate) fn tripping(self, trip: Trip) -> Self {
        Self { trip, ..self }
    }

    /// Block on `condvar` (paired with `guard`'s mutex) until `ready`, owed
    /// as this wait is: registered in the report on demand while it blocks,
    /// and past its bound reported, its owner named, and the process ended.
    pub fn on<'g, T>(&self, mut guard: MutexGuard<'g, T>, condvar: &Condvar, mut ready: impl FnMut(&mut T) -> bool) -> MutexGuard<'g, T> {
        let since = Instant::now();
        let described = self.describe();
        let _registered = crate::diagnostics::register(self.owner, move || format!("{described} for {:?}", since.elapsed()));
        let interval = self.bound.as_ref().map_or(Duration::from_secs(1), |bound| (bound.limit / 10).max(Duration::from_millis(1)));
        let mut clock = HostClock::new(since);
        loop {
            if ready(&mut guard) {
                return guard;
            }
            guard = condvar.wait_timeout(guard, interval).expect("a watched wait's lock").0;
            let ran = clock.look(Instant::now(), interval, false);
            if let Some(bound) = self.bound.as_ref().filter(|bound| ran >= bound.limit) {
                (self.trip)(format!(
                    "magnitude-engine: {} waited for {} past its bound of {:?} ({}); nothing will answer it.\n{}",
                    self.owner,
                    self.what,
                    bound.limit,
                    bound.derivation,
                    crate::diagnostics::report()
                ));
                return guard;
            }
        }
    }

    fn describe(&self) -> String {
        match &self.bound {
            Some(bound) => format!("waiting for {} (owed within {:?}: {})", self.what, bound.limit, bound.derivation),
            None => format!("waiting for {} (idle)", self.what),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// A piece in flight's bytes: any read size a volume reads in.
    const PIECE: u64 = 128 << 10;
    use std::path::Path;
    use std::sync::{Arc, Mutex, mpsc};

    /// The host's time unanswered advances with each look on time, in full;
    /// by at most two looks' intervals across a look the system delayed (a
    /// starved process); and not at all across a look while the owner's
    /// reads were served.
    #[test]
    fn only_time_able_to_answer_without_progress_counts() {
        let interval = Duration::from_millis(10);
        let start = Instant::now();
        let mut clock = HostClock::new(start);
        let mut at = start;
        for _ in 0..5 {
            at += interval;
            clock.look(at, interval, false);
        }
        assert_eq!(clock.ran, interval * 5, "looks on time are credited in full");
        at += Duration::from_secs(1);
        assert_eq!(clock.look(at, interval, false), interval * 7, "a delayed look credits at most two intervals");
        at += interval;
        assert_eq!(clock.look(at, interval, true), interval * 7, "a look across reads served credits nothing");
        at += interval;
        assert_eq!(clock.look(at, interval, false), interval * 8, "reads not served credit again");
    }

    /// An owner's reads are served while outstanding and the engine ends a
    /// piece (the reads queued ahead of them) or reads one within its bound,
    /// however slowly (its own read on a saturated volume); not while it
    /// reads a piece past its bound (one the storage never returns), nor
    /// while it neither ends nor reads one, nor while the owner reads
    /// nothing whatever the engine serves others.
    #[test]
    fn reads_are_served_while_the_engine_progresses_within_each_pieces_bound() {
        let pieces = PieceBound(Some(ReadLine::new(1e-3, 0.0).unwrap()));
        let now = Instant::now();
        let flight = |age| Flight { path: Path::new("/experts").into(), volume: seismic_io::Volume(1), offset: 0, bytes: PIECE, since: now - age };
        let reads = |outstanding, ended, reading: Vec<Flight>| Reads { outstanding, engine: EngineWork { ended, reading } };
        let mut serving = Serving::new(&EngineWork { ended: 0, reading: Vec::new() });
        assert!(!serving.look(&reads(true, 0, Vec::new()), pieces, now), "an engine that neither ends nor reads a piece serves nothing");
        assert!(serving.look(&reads(true, 3, Vec::new()), pieces, now), "pieces ended serve the reads queued behind them");
        assert!(serving.look(&reads(true, 3, vec![flight(Duration::from_millis(9))]), pieces, now), "a slow piece within its bound is served");
        assert!(!serving.look(&reads(true, 3, vec![flight(Duration::from_millis(10))]), pieces, now), "a piece past its bound is not");
        assert!(
            !serving.look(&reads(true, 4, vec![flight(Duration::from_millis(1)), flight(Duration::from_secs(5))]), pieces, now),
            "nor is any read while a piece the storage never returns is held, whatever else ends"
        );
        assert!(!serving.look(&reads(false, 9, vec![flight(Duration::ZERO)]), pieces, now), "nor the reads of an owner that reads nothing");
        let report = pieces.describe(&reads(true, 9, vec![flight(Duration::from_secs(5))]).engine, now);
        assert!(report.contains("/experts at 0 (131072 bytes, volume 1) for 5s of 10ms, past its bound"), "{report}");
    }

    /// A piece's bound is MULTIPLE times the line's time for it and for
    /// every other piece in flight on its device, each on its own, none on
    /// another.
    #[test]
    fn a_pieces_bound_covers_the_pieces_sharing_its_device() {
        let pieces = PieceBound(Some(ReadLine::new(1e-6, 9e-9).unwrap()));
        let now = Instant::now();
        let flight = |volume| Flight { path: Path::new("/experts").into(), volume: seismic_io::Volume(volume), offset: 0, bytes: 1000, since: now };
        let alone = EngineWork { ended: 0, reading: vec![flight(1), flight(2), flight(2)] };
        assert_eq!(pieces.of(&alone.reading[0], &alone), Duration::from_secs_f64(1e-5) * MULTIPLE, "alone on its device");
        let sharing = EngineWork { ended: 0, reading: vec![flight(1), flight(1), flight(1), flight(2)] };
        assert_eq!(pieces.of(&sharing.reading[0], &sharing), Duration::from_secs_f64(3e-5) * MULTIPLE, "behind two others on its device");
    }

    /// A wait whose awaited value never comes trips past its bound, its
    /// report naming its owner and what it waited for; one that comes in
    /// time returns it and trips nothing.
    #[test]
    fn a_wait_past_its_bound_reports_its_owner() {
        let state = Arc::new((Mutex::new(false), Condvar::new()));
        let (tripped, report) = mpsc::channel();
        let bound = || Bound { limit: Duration::from_millis(50), derivation: "the test's".into() };
        let wait = Wait::owed("owner-under-test", "a value never set", bound()).tripping(Box::new(move |text| {
            tripped.send(text).unwrap();
        }));
        let began = Instant::now();
        drop(wait.on(state.0.lock().unwrap(), &state.1, |set| *set));
        let report = report.recv_timeout(Duration::from_secs(5)).expect("the wait tripped");
        assert!(began.elapsed() >= Duration::from_millis(50), "tripped before its bound");
        assert!(report.contains("owner-under-test waited for a value never set past its bound"), "{report}");
        assert!(report.contains("report on demand"), "the trip carries every owner's report: {report}");

        let setter = state.clone();
        let set = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            *setter.0.lock().unwrap() = true;
            setter.1.notify_all();
        });
        let wait = Wait::owed("owner-under-test", "a value set in time", Bound { limit: Duration::from_secs(5), derivation: "the test's".into() })
            .tripping(Box::new(|text| panic!("tripped: {text}")));
        assert!(*wait.on(state.0.lock().unwrap(), &state.1, |set| *set));
        set.join().unwrap();
    }
}
