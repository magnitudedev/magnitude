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
use seismic_io::{EngineWork, Flight, Volume};
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

/// How long a read may go without progress where no line prices it: a
/// volume's silence and each lane before any of a load's reads made a read
/// line, and at any time one piece while its volume ends others. Not a cost:
/// nothing is priced by it, and only a read the storage never returns
/// reaches it.
pub(crate) const UNPRICED_PROGRESS: Duration = Duration::from_secs(60);

/// How long a volume may go ending none of the read engine's pieces in
/// flight on it: [`MULTIPLE`] times the read line's time for its window of
/// pieces, each read on its own: the longest the volume takes over all the
/// engine may hold there, whether it serves them in turn or together, and
/// however few of them are left in flight now, so the last of a burst, held
/// behind I/O the engine does not see, is not bounded by a lone read's time.
/// Its silence runs from the later of its last end and its oldest piece in
/// flight's start. Before the load has a read line (none),
/// [`UNPRICED_PROGRESS`].
///
/// A volume that keeps ending pieces may hold one of them long past what
/// the line prices for it (behind I/O the engine does not see, or behind
/// pieces started after it, thousands of which may pass it meanwhile), so
/// its age is held to no line, and neither is how many pass it. Once the
/// lane waiting on it stops the requests that read, its volume drains and
/// falls silent; while other work keeps it busy, the piece is stuck only at
/// [`UNPRICED_PROGRESS`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct VolumeBound(pub(crate) Option<ReadLine>);

/// What one volume does of the engine's work at one look: its pieces in
/// flight and since when it has ended none of them.
struct VolumeWork<'a> {
    volume: Volume,
    reading: Vec<&'a Flight>,
    quiet_since: Instant,
}

impl VolumeBound {
    /// Each volume `engine` has a piece in flight on.
    fn volumes(engine: &EngineWork) -> Vec<VolumeWork<'_>> {
        let mut volumes = Vec::<VolumeWork<'_>>::new();
        for flight in &engine.reading {
            match volumes.iter_mut().find(|work| work.volume == flight.volume) {
                Some(work) => {
                    work.reading.push(flight);
                    work.quiet_since = work.quiet_since.min(flight.since);
                }
                None => volumes.push(VolumeWork { volume: flight.volume, reading: vec![flight], quiet_since: flight.since }),
            }
        }
        for work in &mut volumes {
            if let Some(&last) = engine.last_ended.get(&work.volume) {
                work.quiet_since = work.quiet_since.max(last);
            }
        }
        volumes
    }

    /// The bound of a volume with `reading` in flight: its widest window of
    /// such pieces (a piece wider than its window is read alone).
    fn of(self, reading: &[&Flight]) -> Duration {
        let Some(line) = self.0 else {
            return UNPRICED_PROGRESS;
        };
        let longest = reading.iter().map(|flight| (flight.window / flight.bytes).max(1) as f64 * line.seconds(flight.bytes)).fold(0.0, f64::max);
        Duration::from_secs_f64(longest) * MULTIPLE
    }

    /// Whether `flight` is stuck at `now`, however busy its volume.
    fn stuck(flight: &Flight, now: Instant) -> bool {
        now.saturating_duration_since(flight.since) >= UNPRICED_PROGRESS
    }

    /// Whether some volume of `engine` has gone past its bound at `now`
    /// ending none of its pieces in flight, or holds one stuck.
    fn stalled(self, engine: &EngineWork, now: Instant) -> bool {
        Self::volumes(engine).iter().any(|work| now.saturating_duration_since(work.quiet_since) >= self.of(&work.reading))
            || engine.reading.iter().any(|flight| Self::stuck(flight, now))
    }

    /// Each volume `engine` has pieces in flight on at `now`: how long it
    /// has ended none of them, of its bound, and whether it is past it; and
    /// each of its pieces: its file, bytes and time in flight, and whether
    /// it is stuck.
    pub(crate) fn describe(self, engine: &EngineWork, now: Instant) -> String {
        let volumes = Self::volumes(engine)
            .iter()
            .map(|work| {
                let quiet = now.saturating_duration_since(work.quiet_since);
                let bound = self.of(&work.reading);
                let flights = work
                    .reading
                    .iter()
                    .map(|flight| {
                        format!(
                            "{} at {} ({} bytes) for {:?}{}",
                            flight.path.display(),
                            flight.offset,
                            flight.bytes,
                            now.saturating_duration_since(flight.since),
                            if Self::stuck(flight, now) { ", stuck" } else { "" }
                        )
                    })
                    .collect::<Vec<_>>();
                format!(
                    "volume {} ended none for {quiet:?} of {bound:?}{}, in flight [{}]",
                    work.volume,
                    if quiet >= bound { ", past its bound" } else { "" },
                    flights.join("; ")
                )
            })
            .collect::<Vec<_>>();
        format!("{} pieces ended, volumes [{}]", engine.ended, volumes.join("; "))
    }
}

/// Whether an owner's reads were being served between two looks: they were
/// outstanding and the read engine progressed, ending a piece meanwhile or
/// reading one now, with no volume past its bound ending none of its pieces
/// and no piece stuck ([`VolumeBound`]). While they are, the engine is
/// serving them or the reads queued ahead of them, however slowly the
/// storage returns each piece; an engine whose volume stopped returning
/// reads or holds a piece stuck, or reading nothing and ending nothing while
/// reads are owed, is not serving them.
pub(crate) struct Serving {
    ended: u64,
}

impl Serving {
    pub(crate) fn new(engine: &EngineWork) -> Self {
        Self { ended: engine.ended }
    }

    /// Whether `reads`, seen at `now` with each volume bounded by
    /// `volumes`, were served since the last look.
    pub(crate) fn look(&mut self, reads: &Reads, volumes: VolumeBound, now: Instant) -> bool {
        let engine = &reads.engine;
        let progressed = (engine.ended != self.ended || !engine.reading.is_empty()) && !volumes.stalled(engine, now);
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
    use std::collections::HashMap;
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

    /// A piece of `PIECE` bytes in flight since `age` before `now`, on
    /// `volume`, of a window of `window` such pieces.
    fn flown(volume: u64, window: u64, now: Instant, age: Duration) -> Flight {
        Flight { path: Path::new("/experts").into(), volume: Volume(volume), offset: 0, bytes: PIECE, window: window * PIECE, since: now - age }
    }

    /// An owner's reads are served while outstanding and the engine ends a
    /// piece (the reads queued ahead of them) or reads one, its volume
    /// within its bound; not once its volume has ended none of its pieces
    /// past it (one that stopped returning reads), nor while the engine
    /// neither ends nor reads one, nor while the owner reads nothing
    /// whatever the engine serves others.
    #[test]
    fn reads_are_served_while_each_volume_ends_pieces_within_its_bound() {
        let volumes = VolumeBound(Some(ReadLine::new(1e-3, 0.0).unwrap()));
        let now = Instant::now();
        let reads = |outstanding, ended, reading: Vec<Flight>| Reads { outstanding, engine: EngineWork { ended, reading, last_ended: HashMap::new() } };
        let mut serving = Serving::new(&EngineWork { ended: 0, reading: Vec::new(), last_ended: HashMap::new() });
        assert!(!serving.look(&reads(true, 0, Vec::new()), volumes, now), "an engine that neither ends nor reads a piece serves nothing");
        assert!(serving.look(&reads(true, 3, Vec::new()), volumes, now), "pieces ended serve the reads queued behind them");
        assert!(serving.look(&reads(true, 3, vec![flown(1, 1, now, Duration::from_millis(9))]), volumes, now), "a volume within its bound is served");
        assert!(!serving.look(&reads(true, 3, vec![flown(1, 1, now, Duration::from_millis(10))]), volumes, now), "one past it is not");
        assert!(!serving.look(&reads(false, 9, vec![flown(1, 1, now, Duration::ZERO)]), volumes, now), "nor the reads of an owner that reads nothing");
        let report = volumes.describe(&reads(true, 9, vec![flown(1, 1, now, Duration::from_secs(5))]).engine, now);
        assert!(report.contains("volume 1 ended none for 5s of 10ms, past its bound, in flight [/experts at 0 (131072 bytes) for 5s]"), "{report}");
    }

    /// The failure that made this rule: pieces of one window on a volume
    /// that keeps ending the others, two of them held past what the line
    /// prices for the whole window, thousands passing them. They are served
    /// throughout, and only at [`UNPRICED_PROGRESS`] is one reported stuck,
    /// however busy its volume.
    #[test]
    fn a_piece_held_while_its_volume_ends_others_is_served_until_it_is_stuck() {
        // 2.51 s for a window of 256 pieces, as observed.
        let volumes = VolumeBound(Some(ReadLine::new(2.51 / 2560.0, 0.0).unwrap()));
        let now = Instant::now();
        let held = |age| {
            let mut reading = vec![flown(1, 256, now, age), flown(1, 256, now, age)];
            reading.extend((0..254).map(|_| flown(1, 256, now, Duration::from_micros(500))));
            Reads { outstanding: true, engine: EngineWork { ended: 5000, reading, last_ended: HashMap::from([(Volume(1), now - Duration::from_micros(100))]) } }
        };
        let mut serving = Serving::new(&EngineWork { ended: 0, reading: Vec::new(), last_ended: HashMap::new() });
        assert!(serving.look(&held(Duration::from_millis(2960)), volumes, now), "held past the window's bound as the others end");
        assert!(serving.look(&held(UNPRICED_PROGRESS - Duration::from_millis(1)), volumes, now), "held just short of stuck");
        assert!(!serving.look(&held(UNPRICED_PROGRESS), volumes, now), "stuck, however busy its volume");
        let report = volumes.describe(&held(UNPRICED_PROGRESS).engine, now);
        assert!(report.contains("for 60s, stuck"), "{report}");
    }

    /// A volume's bound is MULTIPLE times the line's time for its window of
    /// pieces, however few are in flight now, so the last of a burst is not
    /// bounded by a lone read's time; a piece wider than its window is
    /// read alone. One volume's progress does not hide another's silence.
    #[test]
    fn a_volumes_bound_covers_its_window_whatever_is_left_in_flight() {
        let volumes = VolumeBound(Some(ReadLine::new(1e-3, 0.0).unwrap()));
        let now = Instant::now();
        let bound = |flight: &Flight| volumes.of(&[flight]);
        assert_eq!(bound(&flown(1, 16, now, Duration::ZERO)), Duration::from_millis(160), "the last of a window of 16");
        let wide = Flight { window: 1, ..flown(1, 16, now, Duration::ZERO) };
        assert_eq!(bound(&wide), Duration::from_millis(10), "wider than its window, read alone");

        let engine = |quiet| EngineWork {
            ended: 1,
            reading: vec![flown(1, 16, now, Duration::from_secs(1)), flown(2, 16, now, Duration::from_secs(1))],
            last_ended: HashMap::from([(Volume(1), now), (Volume(2), now - quiet)]),
        };
        assert!(!volumes.stalled(&engine(Duration::from_millis(159)), now), "both within their bounds");
        assert!(volumes.stalled(&engine(Duration::from_millis(160)), now), "the first's ends do not cover the second's silence");
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
