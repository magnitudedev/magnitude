//! The read engine: the requests callers make, cut into pieces, each
//! volume's link that alone decides which of its pieces are in flight
//! ("Order and depth"), and the process's pool of workers that reads them.

use crate::disk::{SourceFile, VolumeReads};
use crate::{Destination, Error};
use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Instant;

mod engine;
mod handle;
mod queue;

pub use engine::{engine_work, read_buffers, EngineWork, Flight, Outstanding};
pub use handle::{Ending, Handle};

pub(crate) use engine::{deepen, reading, totals, Buffers, Reading};
pub(crate) use handle::RequestState;
pub(crate) use queue::{pieces, Piece, Queued, Span};
#[cfg(test)]
pub(crate) use {engine::Engine, queue::{Part, Queue}};

use engine::engine;

/// Ranges whose aligned blocks are at most this far apart are read in one.
/// A gap costs at most its own bytes, `COALESCE_GAP / D` = 43 µs at the
/// pool volume's measured D = 3.03 GB/s, and saves a separate read's fixed
/// cost, λ = 0.26 ms there (2026-10-22, an M4 Pro): joining is cheaper
/// while `λ · D` (≈ 790 KB) exceeds the gap read.
const COALESCE_GAP: u64 = 128 * 1024;

/// How a request's consumer waits on it: a step parked on it since an
/// instant; a stream consuming it at the link's rate since an instant (a
/// prompt group's conveyor, which keeps the link at its due depth and parks
/// on no single read); none yet, a step expected to park on it from an
/// instant (a hint, whose instant is a prediction: passing it parks
/// nothing, and a step that comes to wait on it reprices it parked); or
/// none at any instant (a load, a fill). The first two are due and order
/// before every request read ahead, of one instant a parked one first; of
/// those read ahead, an expected one orders by its instant, before every
/// undated one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Deadline {
    Parked(Instant),
    Streamed(Instant),
    Expected(Instant),
    None,
}

impl Deadline {
    /// What it orders by: due before an expected instant before none, then
    /// its instant, then a parked wait before a stream's.
    fn key(&self) -> (u8, Option<Instant>, bool) {
        match *self {
            Self::Parked(at) => (0, Some(at), false),
            Self::Streamed(at) => (0, Some(at), true),
            Self::Expected(at) => (1, Some(at), false),
            Self::None => (2, None, false),
        }
    }
}

impl Ord for Deadline {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key().cmp(&other.key())
    }
}

impl PartialOrd for Deadline {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// How a request stands with its consumer ("Order and depth"): a step
/// waits on it parked, a stream consumes it at the link's rate, or none
/// waits yet (read ahead).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Standing {
    Parked,
    Streamed,
    Ahead,
}

impl Standing {
    /// Whether a consumer waits on it: due.
    pub(crate) fn due(self) -> bool {
        self != Self::Ahead
    }
}

/// What a request is for, as its link orders it ("Order and depth"): its
/// deadline, and its value, the step time it is expected to save, in
/// seconds (a read ahead right with probability `p` saves its bytes' time
/// at the volume's rate, `p · b/R`), which only orders requests of one
/// deadline. A request is **due** while a consumer waits on it, a step
/// parked or a stream at the link's rate; an expected deadline's passing
/// makes none due, since only a step's own wait says one waits. More
/// urgent orders first: due before read ahead, the earlier deadline, then
/// the greater value.
#[derive(Clone, Copy, Debug)]
pub struct Want {
    deadline: Deadline,
    value: f64,
}

impl Want {
    /// Due now, a step parked on it.
    pub fn now() -> Self {
        Self::new(Deadline::Parked(Instant::now()), 0.)
    }

    /// Read ahead, a step expected to park on it from `deadline`; worth
    /// `value`.
    pub fn expected(deadline: Instant, value: f64) -> Self {
        Self::new(Deadline::Expected(deadline), value)
    }

    /// Due now, consumed by a stream at the link's rate: a prompt group's
    /// read, which holds back nothing read ahead ("Order and depth").
    pub fn streamed() -> Self {
        Self::new(Deadline::Streamed(Instant::now()), 0.)
    }

    /// Wanted at no instant (a load, a fill), worth `value`.
    pub fn undated(value: f64) -> Self {
        Self::new(Deadline::None, value)
    }

    fn new(deadline: Deadline, value: f64) -> Self {
        assert!(value.is_finite() && value >= 0., "a want's value is seconds saved, {value}");
        Self { deadline, value }
    }

    pub fn deadline(&self) -> Deadline {
        self.deadline
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    /// Whether a consumer waits on it.
    pub fn due(&self) -> bool {
        self.standing().due()
    }

    /// How it stands.
    pub(crate) fn standing(&self) -> Standing {
        match self.deadline {
            Deadline::Parked(_) => Standing::Parked,
            Deadline::Streamed(_) => Standing::Streamed,
            Deadline::Expected(_) | Deadline::None => Standing::Ahead,
        }
    }
}

impl Ord for Want {
    fn cmp(&self, other: &Self) -> Ordering {
        self.deadline.cmp(&other.deadline).then_with(|| other.value.total_cmp(&self.value))
    }
}

impl PartialOrd for Want {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Want {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Want {}

/// One range of a request: `length` bytes at `offset` of the reader's
/// file, landed at `at` bytes into `into`, which the request holds until
/// its last piece has ended ([`crate::Reader::request`]).
#[derive(Clone)]
pub struct Range {
    pub offset: u64,
    pub length: usize,
    pub into: Arc<dyn Destination>,
    pub at: usize,
}

impl std::fmt::Debug for Range {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Range")
            .field("offset", &self.offset)
            .field("length", &self.length)
            .field("into", &self.into.region().0)
            .field("at", &self.at)
            .finish()
    }
}

/// How a request ended.
#[derive(Clone, Debug)]
pub enum Ended {
    /// Every piece was read, or failed: the ranges that did not land, each
    /// (its index in the request) with the error that kept it from landing.
    Read { failed: Vec<(usize, Arc<Error>)> },
    /// A worker panicked reading one of its pieces: a defect, which its
    /// observer resumes ([`Panic::resume`]).
    Panicked(Panic),
}

/// A read worker's panic, carried to its request's observers. Its payload
/// unwinds on the first to resume it.
#[derive(Clone)]
pub struct Panic(Arc<std::sync::Mutex<Option<Box<dyn std::any::Any + Send>>>>);

impl Panic {
    pub(crate) fn new(payload: Box<dyn std::any::Any + Send>) -> Self {
        Self(Arc::new(std::sync::Mutex::new(Some(payload))))
    }

    /// Unwind with the worker's panic: its payload, or, where another
    /// observer of the request resumed that, a panic naming it.
    pub fn resume(&self) -> ! {
        let payload = self.0.lock().expect("a read worker's panic").take();
        match payload {
            Some(payload) => std::panic::resume_unwind(payload),
            None => panic!("a read worker panicked; another observer of its request resumed the panic"),
        }
    }
}

impl std::fmt::Debug for Panic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let payload = self.0.lock().expect("a read worker's panic");
        let message = payload.as_deref().and_then(|payload| {
            payload.downcast_ref::<&str>().copied().or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        });
        formatter.debug_tuple("Panic").field(&message).finish()
    }
}

/// Queue `ranges` of `file` on its volume's link for `want`, cut as
/// `volume` says, counted in `outstanding` until the request ends: a
/// reader's request ([`crate::Reader::request`]), or a probe's lone one.
///
/// # Safety
/// As [`crate::Reader::request`].
pub(crate) unsafe fn request(file: &Arc<SourceFile>, volume: &VolumeReads, outstanding: &Arc<std::sync::atomic::AtomicUsize>, ranges: Vec<Range>, want: Want) -> Result<Handle, Error> {
    let engine = engine();
    let outside = Arc::new(Error::Invalid("read outside the file".into()));
    let mut failed = Vec::new();
    // Each range in the file by its index and span, in offset order.
    let mut order = Vec::with_capacity(ranges.len());
    for (index, range) in ranges.iter().enumerate() {
        let (base, length) = range.into.region();
        assert!(
            range.at.checked_add(range.length).is_some_and(|end| end <= length),
            "a range of {} bytes at {} of a destination of {length}",
            range.length,
            range.at,
        );
        if range.offset.checked_add(range.length as u64).is_none_or(|end| end > file.size) {
            failed.push((index, outside.clone()));
        } else {
            // SAFETY: within the destination's region, checked above.
            order.push((index, Span { offset: range.offset, length: range.length, sink: unsafe { base.add(range.at) } }));
        }
    }
    order.sort_by_key(|(_, span)| span.offset);
    let read = volume.shape.read(want.due());
    let pieces = pieces(&order, file.size, read);
    outstanding.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    let state = RequestState::new(pieces.len(), failed, outstanding.clone(), ranges);
    // One worker per piece the link may start now: a worker woken with no
    // piece it may take would only contend for the queue with the
    // submitter's next request.
    let (id, woken) = engine.queue().push(Queued { state: state.clone(), file: file.clone(), volume: volume.clone(), pieces: pieces.into() }, want);
    engine.wake(woken);
    Ok(Handle { id, ending: handle::Ending(state), engine })
}
