//! A recorded timeline of the read engine's work and its callers' events,
//! for reconstructing a step's critical path: every piece's submission,
//! start and end, every promotion and withdrawal, and the events callers
//! note beside them (a request noticed, a landing, an answer).
//!
//! Off unless [`record`] was called: then each event is one push under a
//! lock, stamped against the instant recording began. A measurement that
//! wants it turns it on, drains what it does not measure with [`take`], and
//! takes the measured span's events at its end.

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// What happened. The reader's own kinds name the job by its submission id;
/// a caller's kinds carry whatever its fields say ([`note`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A job queued: `a` its priority (0 demand, 1 prefetch, 2 load), `b`
    /// its pieces, `c` its bytes.
    Submit,
    /// A piece taken: `a` its file offset, `b` its bytes, `c` the volume's
    /// bytes in flight before it.
    Take,
    /// A piece's read returned, its bytes leaving the volume's flight: `a`
    /// its offset, `b` its bytes.
    Read,
    /// A job promoted: `a` its new priority.
    Promote,
    /// A job withdrawn: `a` its pieces not taken.
    Withdraw,
    /// A caller's event, named by the caller.
    Noted(&'static str),
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Submit => "submit",
            Kind::Take => "take",
            Kind::Read => "read",
            Kind::Promote => "promote",
            Kind::Withdraw => "withdraw",
            Kind::Noted(name) => name,
        }
    }
}

/// One event: its time in nanoseconds since recording began, its kind, a
/// job (0 for none) and three fields its kind defines.
#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub at: u64,
    pub kind: Kind,
    pub job: u64,
    pub a: i64,
    pub b: i64,
    pub c: i64,
}

struct Timeline {
    epoch: Instant,
    events: Mutex<Vec<Event>>,
}

static TIMELINE: OnceLock<Timeline> = OnceLock::new();

/// Record from now on, for the rest of the process.
pub fn record() {
    TIMELINE.get_or_init(|| Timeline { epoch: Instant::now(), events: Mutex::new(Vec::with_capacity(1 << 20)) });
}

/// Whether events are recorded.
pub fn recording() -> bool {
    TIMELINE.get().is_some()
}

/// Note an event at `at`, when recording.
pub fn note_at(at: Instant, kind: Kind, job: u64, (a, b, c): (i64, i64, i64)) {
    if let Some(timeline) = TIMELINE.get() {
        let at = at.saturating_duration_since(timeline.epoch).as_nanos() as u64;
        timeline.events.lock().expect("read timeline").push(Event { at, kind, job, a, b, c });
    }
}

/// Note an event now, when recording.
pub fn note(kind: Kind, job: u64, fields: (i64, i64, i64)) {
    if TIMELINE.get().is_some() {
        note_at(Instant::now(), kind, job, fields);
    }
}

/// The nanoseconds since recording began at `at` (none when not recording).
pub fn stamp(at: Instant) -> Option<u64> {
    TIMELINE.get().map(|timeline| at.saturating_duration_since(timeline.epoch).as_nanos() as u64)
}

/// The events recorded since the last take, in time order.
pub fn take() -> Vec<Event> {
    let Some(timeline) = TIMELINE.get() else {
        return Vec::new();
    };
    let mut events = std::mem::take(&mut *timeline.events.lock().expect("read timeline"));
    events.sort_by_key(|event| event.at);
    events
}
