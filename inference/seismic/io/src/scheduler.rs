//! The read engine: the jobs callers submit, cut into pieces, and the
//! process's pool of workers that reads them most urgent first ("The read
//! engine", "How reads share the engine").

use crate::disk::{SourceFile, VolumeReads};
use crate::{Destination, Error};
use std::sync::Arc;

mod engine;
mod job;
mod queue;

pub use engine::{engine_work, EngineWork, Flight, Outstanding};
pub use job::Job;

pub(crate) use engine::{deepen, totals, Buffers};
pub(crate) use job::JobState;
pub(crate) use queue::{pieces, Piece, Queued, Span};
#[cfg(test)]
pub(crate) use {engine::Engine, queue::{Part, Queue, VolumeFlight}};

use engine::engine;

/// Ranges whose aligned blocks are at most this far apart are read in one.
/// A gap costs at most its own bytes, `COALESCE_GAP / D` = 43 µs at the
/// pool volume's measured D = 3.03 GB/s, and saves a separate read's fixed
/// cost, λ = 0.26 ms there (2026-10-22, an M4 Pro): joining is cheaper
/// while `λ · D` (≈ 790 KB) exceeds the gap read.
const COALESCE_GAP: u64 = 128 * 1024;

/// How urgently a job's pieces are read: every piece of a more urgent job
/// before any of a less urgent one. A volume's accounting is kept by it
/// ([`crate::accounting::Class`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// A step waits for the bytes.
    Demand,
    /// A step is expected to need them.
    Prefetch,
    /// A load waits for them, no step: read behind every step's reads, so
    /// a model loading never delays a running one's, whose prefetch is
    /// bounded by its staging while a load reads gigabytes.
    Load,
}

impl Priority {
    pub const ALL: [Self; 3] = [Self::Demand, Self::Prefetch, Self::Load];

    /// Whether no step waits on its bytes: what a demand read waits behind
    /// as speculative ([`crate::Waits`]).
    pub(crate) fn speculative(self) -> bool {
        self != Self::Demand
    }
}

/// One range of a job: `length` bytes at `offset` of the reader's file,
/// landed at `at` bytes into `into`, which the job holds until its last
/// piece has ended ([`crate::Reader::submit`]).
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

/// How a job ended.
#[derive(Clone, Debug)]
pub enum Ended {
    /// Every piece was read, or failed: the ranges that did not land, each
    /// (its index in the job) with the error that kept it from landing.
    Read { failed: Vec<(usize, Arc<Error>)> },
    /// A worker panicked reading one of its pieces: a defect, which its
    /// observer resumes ([`Panic::resume`]).
    Panicked(Panic),
}

/// A read worker's panic, carried to its job's observers. Its payload
/// unwinds on the first to resume it.
#[derive(Clone)]
pub struct Panic(Arc<std::sync::Mutex<Option<Box<dyn std::any::Any + Send>>>>);

impl Panic {
    pub(crate) fn new(payload: Box<dyn std::any::Any + Send>) -> Self {
        Self(Arc::new(std::sync::Mutex::new(Some(payload))))
    }

    /// Unwind with the worker's panic: its payload, or, where another
    /// observer of the job resumed that, a panic naming it.
    pub fn resume(&self) -> ! {
        let payload = self.0.lock().expect("a read worker's panic").take();
        match payload {
            Some(payload) => std::panic::resume_unwind(payload),
            None => panic!("a read worker panicked; another observer of its job resumed the panic"),
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

/// Queue `ranges` of `file` for the pool at `priority`, read as `volume`
/// says, counted in `outstanding` until the job ends: a reader's job
/// ([`crate::Reader::submit`]), or a probe's lone request.
///
/// # Safety
/// As [`crate::Reader::submit`].
pub(crate) unsafe fn queue(file: &Arc<SourceFile>, volume: &VolumeReads, outstanding: &Arc<std::sync::atomic::AtomicUsize>, ranges: Vec<Range>, priority: Priority) -> Result<Job, Error> {
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
    let read = match priority {
        Priority::Demand => volume.choice.demand_read,
        _ => volume.choice.read,
    };
    let pieces = pieces(&order, file.size, read);
    outstanding.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    let state = JobState::new(pieces.len(), failed, outstanding.clone(), ranges);
    // One worker per piece the volume's window has room for: a worker
    // woken with no piece it may take would only contend for the queue
    // with the submitter's next job.
    let (id, woken) = engine.queue.lock().expect("read queue").push(
        Queued { state: state.clone(), file: file.clone(), volume: volume.clone(), pieces: pieces.into() },
        priority,
    );
    for _ in 0..woken {
        engine.work.notify_one();
    }
    Ok(Job { id, state, engine })
}
