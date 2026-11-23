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

pub use engine::{engine_work, read_buffers, read_workers, EngineWork, Flight, Outstanding};
pub use handle::{Ending, Handle};

pub(crate) use engine::{need, reading, totals, Buffers, Need, Reading};
pub(crate) use handle::RequestState;
pub(crate) use queue::{pieces, Part, Piece, Queued, Span};
#[cfg(test)]
pub(crate) use {engine::Engine, queue::Queue};

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
/// on no single read); none yet, the step expected to park on it, by the
/// caller's count of steps (a read ahead: a step that comes to wait on it
/// reprices it parked); or none at all (a load, a fill). The first two are
/// due and order before every request read ahead, of one instant a parked
/// one first; of those read ahead, an expected one orders before every
/// undated one, an expected one by its step, the soonest first, and within
/// a step, as an undated one, by its [`Want`]'s value. A read ahead carries
/// no instant: its step is the caller's own count, so no estimate of when a
/// step reaches it enters the order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Deadline {
    Parked(Instant),
    Streamed(Instant),
    Expected(u64),
    None,
}

impl Deadline {
    /// Its class: due, expected, or undated.
    fn class(&self) -> u8 {
        self.key().0
    }

    /// The step an expected one is wanted by; none other has one.
    fn step(&self) -> u64 {
        self.key().2
    }

    /// What it orders by: due before expected before none, then a due
    /// one's instant, then an expected one's step, then a parked wait
    /// before a stream's.
    fn key(&self) -> (u8, Option<Instant>, u64, bool) {
        match *self {
            Self::Parked(at) => (0, Some(at), 0, false),
            Self::Streamed(at) => (0, Some(at), 0, true),
            Self::Expected(step) => (1, None, step, false),
            Self::None => (2, None, 0, false),
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
/// deadline, and its value, the probability its consumer uses it (one
/// wanted at no instant, a load's or a fill's, is worth what its caller
/// says). A request is **due** while a consumer waits on it, a step parked
/// or a stream at the link's rate; an expected deadline's passing makes
/// none due, since only a step's own wait says one waits. More urgent
/// orders first: due before read ahead, by its instant; of those read
/// ahead, expected before undated, an expected one wanted by a sooner step
/// first, then the greater value. A read ahead takes drive time no queued
/// parked read waits for ("Order and depth"), where a right one saves its
/// consumer's wait only if it lands before that step parks on it and a
/// wrong one spends its bytes' time: so the soonest wanted take it first,
/// and of one step the likeliest.
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

    /// Read ahead, step `step` (the caller's count) expected to park on
    /// it, used with probability `value`.
    pub fn expected(step: u64, value: f64) -> Self {
        Self::new(Deadline::Expected(step), value)
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
        assert!(value.is_finite() && value >= 0., "a want's value is a probability of use, {value}");
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
        // A due request is worth nothing (`Self::now`, `Self::streamed`):
        // its instant orders it. An expected one's step leads its value.
        self.deadline
            .class()
            .cmp(&other.deadline.class())
            .then_with(|| self.deadline.step().cmp(&other.deadline.step()))
            .then_with(|| other.value.total_cmp(&self.value))
            .then_with(|| self.deadline.cmp(&other.deadline))
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

/// A [`Range`] widened to [`crate::ALIGNMENT`] boundaries for zero-bounce
/// direct I/O, together with the `lead` offset where the requested payload
/// begins inside the landed region.
///
/// The destination at `at` must be [`crate::ALIGNMENT`]-aligned and have at
/// least `length + ALIGNED_HEADROOM` bytes of capacity (or at least the
/// aligned span covering `[offset, offset + length)`). Once the request
/// ends, the requested bytes lie at `at + lead .. at + lead + payload`.
#[derive(Clone, Debug)]
pub struct AlignedRange {
    pub range: Range,
    /// Byte offset from `range.at` where `offset`'s first byte lands.
    pub lead: usize,
    /// Requested payload length in bytes.
    pub payload: usize,
}

impl AlignedRange {
    /// Align `[offset, offset + length)` of a file of `file_size` bytes to
    /// [`crate::ALIGNMENT`] boundaries into `into` at `at`.
    ///
    /// An empty or out-of-bounds range is passed through unchanged so
    /// [`crate::Reader::request`] reports its normal outcome.
    pub fn new(offset: u64, length: usize, file_size: u64, into: Arc<dyn Destination>, at: usize) -> Self {
        if length == 0 || offset.checked_add(length as u64).is_none_or(|end| end > file_size) {
            return Self {
                range: Range { offset, length, into, at },
                lead: 0,
                payload: length,
            };
        }
        let align = crate::ALIGNMENT;
        let lead = (offset % align) as usize;
        let aligned_start = offset - lead as u64;
        let aligned_end = (offset + length as u64).div_ceil(align) * align;
        let read_end = aligned_end.min(file_size);
        let read_len = (read_end - aligned_start) as usize;
        Self {
            range: Range { offset: aligned_start, length: read_len, into, at },
            lead,
            payload: length,
        }
    }

    /// The requested payload slice inside the destination's byte slice once
    /// the request has ended.
    pub fn slice<'a>(&self, bytes: &'a [u8]) -> &'a [u8] {
        let start = self.range.at + self.lead;
        &bytes[start..start + self.payload]
    }
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

/// A read worker's panic, carried to its request's observers: the first to
/// resume it unwinds with it ([`Panic::resume`]).
pub use seismic_threads::Panic;

/// Queue `ranges` of `file` on its volume's link for `want`, cut as
/// `volume` says, under `need`'s claim on the pool's workers, counted in
/// `outstanding` until the request ends: a reader's request
/// ([`crate::Reader::request`]), or a probe's.
///
/// # Safety
/// As [`crate::Reader::request`].
pub(crate) unsafe fn request(
    file: &Arc<SourceFile>,
    volume: &VolumeReads,
    need: &Arc<Need>,
    outstanding: &Arc<std::sync::atomic::AtomicUsize>,
    ranges: Vec<Range>,
    want: Want,
) -> Result<Handle, Error> {
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
    let read = volume.shape.read(want.standing());
    let pieces = pieces(&order, file.size, read);
    // The pool holds a worker for each piece its link may admit, up to its
    // device's queue depth, before any is queued.
    need.submitted(&pieces)?;
    outstanding.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    let state = RequestState::new(pieces.len(), failed, outstanding.clone(), ranges);
    // One worker per piece the link may start now: a worker woken with no
    // piece it may take would only contend for the queue with the
    // submitter's next request.
    let queued = Queued { state: state.clone(), file: file.clone(), volume: volume.clone(), need: need.clone(), pieces: pieces.into() };
    let (id, woken) = engine.queue().push(queued, want);
    engine.wake(woken);
    Ok(Handle { id, ending: handle::Ending(state), engine })
}
