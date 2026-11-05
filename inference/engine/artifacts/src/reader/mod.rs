//! Positional reads of an artifact file, as steps name them: the rows of a
//! streamed table and the experts of a streamed load. A reader reads by its
//! [`CachePolicy`].
//!
//! **Uncached** reads do not fill the operating system's file cache, so
//! they never grow memory the plan did not charge:
//!
//! - Linux: `O_DIRECT`, whose offsets, lengths and buffers are aligned to
//!   [`ALIGNMENT`].
//! - macOS: `F_NOCACHE`.
//! - Windows: `FILE_FLAG_NO_BUFFERING`, aligned as `O_DIRECT`.
//!
//! A filesystem that refuses direct I/O, when the file is opened or at its
//! first direct read (`EINVAL` or `EOPNOTSUPP` on Linux,
//! `ERROR_INVALID_PARAMETER` on Windows), is read through the file cache
//! from then on, as is a file macOS cannot open again (it opens by identity
//! or name, and nothing names a file replaced under its source); the pages
//! each read filled are dropped (Linux `POSIX_FADV_DONTNEED`, macOS
//! `MS_INVALIDATE`).
//!
//! **Cached** reads go through the file cache and keep what they fill. The
//! file is advised
//! random access on Linux (`POSIX_FADV_RANDOM`: no read-ahead past the
//! ranges named), and every piece lands through its worker's buffer, then
//! an exact copy into each sink: reading the cache straight into mapped
//! staging measured slower on Linux.
//!
//! # The read engine
//!
//! Every read of the process runs on one pool of persistent workers. Each
//! volume is read in reads of at most its read size, holding at most its
//! window of bytes in flight ("A volume's window"), and the pool holds as
//! many workers as the widest window a reader has opened needs. A job is
//! the ranges one caller names, each landing in memory the caller gives
//! (its sink): they are read in ascending order, ranges whose aligned
//! blocks touch or lie within [`COALESCE_GAP`] of each other in one aligned
//! read, and each read split into pieces of the volume's read size, which
//! the workers take most urgent first: by the
//! job's [`Priority`], then by submission, then in the job's order. Storage
//! serves parallel reads at nearly the latency of one. An uncached piece
//! that lies within one range is read straight into its sink where the sink
//! meets the reader's buffer alignment; any other piece is read into the
//! worker's own aligned buffer and copied into the sinks of the ranges it
//! serves. A job
//! is promoted (never demoted), and withdrawn ([`Job::withdraw`], or dropped):
//! the pieces no worker has taken are read by none, the ranges they serve
//! failed [`Error::Withdrawn`], and it ends with the pieces in flight. A
//! failed piece fails only the ranges it serves, of its job.
//!
//! # How reads share the engine
//!
//! A read of `b` bytes the engine serves alone ends `λ + b/D` after it is
//! submitted. Reads queued together overlap their latencies: each read
//! queued behind others holds the engine only for its bytes, `b/D`, so `k`
//! reads submitted at once end `λ + k·b/D` after, and only reads each
//! waited for before the next pay `k·(λ + b/D)`. A read more urgent than
//! those queued goes ahead of them, but waits for the pieces the workers
//! already hold: at most its volume's window ([`ArtifactReader::window`]).
//! Here `λ` and `D` are the volume's read line, from a load's own reads:
//! `D` the volume's bytes over its busy time ([`Delivery`]), `λ` what the
//! demand reads took beyond that: what the performance model prices every
//! read by. Measured on an RTX 3090 host (EPYC 7313), uncached records at random
//! offsets of Qwen3.6 35B's file (`tests/read_law.rs`):
//! a 1.9 MB record alone 0.53–0.58 ms, sixteen at once 6.69 ms (each more
//! 0.41 ms, 4.7 GB/s), sixteen each waited for 10.97 ms; 256 KiB records
//! 0.29 ms alone and 0.135 ms each more at once (1.9 GB/s), 8 MiB 1.71 ms
//! and 1.61 ms each more (5.3 GB/s). So nothing in the engine serves reads
//! one after another, and a latency a 2 MiB window (sixteen 1.9 MB records,
//! fifteen windows, 8.3 ms) is not what storage does; the queued rate falls
//! with a record's size (a random access per record), which a line through
//! records of the size served carries in its `D`.

use crate::Error;
use std::io;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

mod engine;
mod file;
mod job;
mod platform;
mod probe;
mod queue;
mod volume;

pub use engine::{engine_work, EngineWork, Flight, Outstanding};
pub use job::Job;
pub use platform::cached_pages;
pub(crate) use platform::open_cached;
pub use probe::Urgent;
pub use volume::{adopt_volume, probe_volume, read_cores, Delivery, VolumeChoice};

use engine::engine;
use file::{SourceFile, Through};
use job::JobState;
use platform::{evict, uncached};
use queue::{pieces, Queued};
use volume::{Registered, Volume};

#[cfg(doc)]
use volume::Bound;

/// The alignment of a direct read's offset, length and buffer: the largest
/// logical block size of the storage served.
pub const ALIGNMENT: u64 = 4096;

/// Ranges whose aligned blocks are at most this far apart are read in one.
/// A gap costs at most its own bytes, `COALESCE_GAP / D` = 43 µs at the
/// pool volume's measured D = 3.03 GB/s, and saves a separate read's fixed
/// cost, λ = 0.26 ms there (2026-10-22, an M4 Pro): joining is cheaper
/// while `λ · D` (≈ 790 KB) exceeds the gap read.
const COALESCE_GAP: u64 = 128 * 1024;

/// How urgently a job's pieces are read: every piece of a more urgent job
/// before any of a less urgent one.
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

/// Whether a reader's reads fill the file cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CachePolicy {
    /// Past the file cache, filling none of it.
    Uncached,
    /// Through the file cache, keeping the pages read.
    Cached,
}

/// One range of a job: `length` bytes at `offset` of the reader's file,
/// landed at `sink`.
#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub offset: u64,
    pub length: usize,
    /// Where the range's bytes land: `length` bytes the caller owns until
    /// the job ends ([`ArtifactReader::submit`]).
    pub sink: NonNull<u8>,
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
    pub(super) fn new(payload: Box<dyn std::any::Any + Send>) -> Self {
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

/// An artifact file opened for reads by one [`CachePolicy`], read as its
/// urgent request needs ([`Urgent`]).
#[derive(Debug, Clone)]
pub struct ArtifactReader {
    file: Arc<SourceFile>,
    volume: Volume,
    /// An urgent reader's window in its volume's bound while any clone of
    /// it lives ([`Bound`]).
    _registered: Option<Arc<Registered>>,
    /// Its jobs submitted and not yet ended ([`Self::outstanding`]).
    outstanding: Arc<std::sync::atomic::AtomicUsize>,
}

impl ArtifactReader {
    /// The artifact file at `path`, opened as a [`crate::FileSource`], read
    /// by a reader no step waits on.
    pub fn open(path: impl AsRef<Path>, policy: CachePolicy) -> Result<Self, Error> {
        Self::of(&crate::FileSource::open(path)?, policy)
    }

    /// The artifact file at `path`, read for steps that wait on `urgent`.
    pub fn open_urgent(path: impl AsRef<Path>, policy: CachePolicy, urgent: &Urgent) -> Result<Self, Error> {
        Self::of_urgent(&crate::FileSource::open(path)?, policy, urgent)
    }

    /// The file `source` has open, read for steps that wait on `urgent`.
    pub fn of_urgent(source: &crate::FileSource, policy: CachePolicy, urgent: &Urgent) -> Result<Self, Error> {
        Self::reading(source, policy, Some(urgent))
    }

    /// The file `source` has open, read by `policy` by a reader no step
    /// waits on.
    pub fn of(source: &crate::FileSource, policy: CachePolicy) -> Result<Self, Error> {
        Self::reading(source, policy, None)
    }

    /// The file `source` has open, read by `policy`: its descriptor shared
    /// where cached, the same file opened again past the cache where not;
    /// read as its volume's choice for `urgent` says, whatever its policy;
    /// starting the pool's workers it needs ("A volume's window").
    fn reading(source: &crate::FileSource, policy: CachePolicy, urgent: Option<&Urgent>) -> Result<Self, Error> {
        let file = source.file();
        let through = match policy {
            CachePolicy::Uncached => uncached(file)?,
            CachePolicy::Cached => Through::Cache(file.try_clone()?),
        };
        let path: Arc<Path> = source.path().into();
        let volume = Volume::of(file, urgent)?;
        engine().deepen(volume.workers())?;
        let registered = urgent.map(|_| Arc::new(Registered::new(volume.background.clone(), volume.choice.window)));
        Ok(Self {
            file: Arc::new(SourceFile {
                path,
                size: source.size(),
                through,
            }),
            volume,
            _registered: registered,
            outstanding: Arc::default(),
        })
    }

    /// Its jobs submitted and not yet ended, as a handle that reads them
    /// while the reader lives on elsewhere: while any is, a waiter on one
    /// waits on the engine, whose work [`engine_work`] reports.
    pub fn outstanding(&self) -> Outstanding {
        Outstanding(self.outstanding.clone())
    }

    pub fn size(&self) -> u64 {
        self.file.size
    }

    /// The most bytes in flight a less urgent piece joins: what a demand read
    /// waits behind of them ("A volume's window", "How reads share the
    /// engine").
    pub fn window(&self) -> u64 {
        self.volume.choice.window
    }

    /// What its volume reads and how long it is busy ([`Delivery`]), shared
    /// by every reader of the volume.
    pub fn delivery(&self) -> Arc<Delivery> {
        self.volume.delivery.clone()
    }

    /// Drop every page of the file the cache holds, so the reads after it
    /// reach storage: what a read must do whose time is the storage's (a
    /// probe of its volume). Uncached reads alone would not: macOS's
    /// `F_NOCACHE` serves pages already cached, and a filesystem that
    /// refused direct I/O reads through the cache. On Linux the pages are
    /// dropped (`POSIX_FADV_DONTNEED`), on macOS invalidated through a
    /// mapping (`msync` with `MS_INVALIDATE`); Windows reads uncached with
    /// no cache to drop.
    pub fn evict_cached(&self) -> Result<(), Error> {
        let file = match &self.file.through {
            Through::Direct { file, .. } | Through::Cache(file) => file,
            Through::Dropped(file) => file,
        };
        evict(file, self.file.size)
    }

    /// `length` bytes from `offset`, read in order as a stream: each read
    /// of the stream is one [`Priority::Load`] job into the caller's
    /// buffer, so the stream holds none of the bytes itself.
    pub fn range(&self, offset: u64, length: u64) -> RangeReader<'_> {
        RangeReader {
            reader: self,
            offset,
            remaining: length,
        }
    }

    /// Queue `ranges` for the pool at `priority`, behind every piece of a
    /// job at least as urgent submitted before it. A range outside the file
    /// fails at once.
    ///
    /// # Safety
    /// Each range's sink is `length` writable bytes, disjoint from every
    /// other range's of the job and from every byte the caller touches,
    /// until the job has ended: its [`Job::wait`] returned, or it was
    /// dropped (which withdraws it and waits).
    pub unsafe fn submit(&self, ranges: Vec<Range>, priority: Priority) -> Result<Job, Error> {
        let engine = engine();
        let outside = Arc::new(Error::Invalid("read outside artifact".into()));
        let mut failed = Vec::new();
        let mut order = Vec::with_capacity(ranges.len());
        for (index, range) in ranges.iter().enumerate() {
            if range
                .offset
                .checked_add(range.length as u64)
                .is_none_or(|end| end > self.file.size)
            {
                failed.push((index, outside.clone()));
            } else {
                order.push(index);
            }
        }
        order.sort_by_key(|index| ranges[*index].offset);
        let read = match priority {
            Priority::Demand => self.volume.choice.demand_read,
            _ => self.volume.choice.read,
        };
        let pieces = pieces(&ranges, &order, self.file.size, read);
        self.outstanding.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let state = JobState::new(pieces.len(), failed, self.outstanding.clone());
        // One worker per piece the volume's window has room for: a worker
        // woken with no piece it may take would only contend for the queue
        // with the submitter's next job.
        let (id, woken) = engine.queue.lock().expect("read queue").push(Queued {
            state: state.clone(),
            file: self.file.clone(),
            volume: self.volume.clone(),
            pieces: pieces.into(),
        }, priority);
        for _ in 0..woken {
            engine.work.notify_one();
        }
        Ok(Job { id, state, engine })
    }
}

/// A range of an artifact file read in order ([`ArtifactReader::range`]).
pub struct RangeReader<'a> {
    reader: &'a ArtifactReader,
    offset: u64,
    remaining: u64,
}

impl io::Read for RangeReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let length = usize::try_from(self.remaining).map_or(buffer.len(), |left| left.min(buffer.len()));
        let Some(sink) = NonNull::new(buffer.as_mut_ptr()).filter(|_| length != 0) else {
            return Ok(0);
        };
        let range = Range {
            offset: self.offset,
            length,
            sink,
        };
        // SAFETY: `buffer` is the caller's, `length` writable bytes of it,
        // touched by nothing else until the job ends (`wait`).
        // An error keeps its kind and itself as the source.
        let job = unsafe { self.reader.submit(vec![range], Priority::Load) }.map_err(|error| match error {
            Error::Io(error) => error,
            error => io::Error::other(error),
        })?;
        match job.wait() {
            Ended::Read { failed } => match failed.into_iter().next() {
                None => {}
                Some((_, error)) => {
                    let kind = match &*error {
                        Error::Io(error) => error.kind(),
                        _ => io::ErrorKind::Other,
                    };
                    return Err(io::Error::new(kind, error));
                }
            },
            Ended::Panicked(panic) => panic.resume(),
        }
        self.offset += length as u64;
        self.remaining -= length as u64;
        Ok(length)
    }
}

#[cfg(test)]
mod tests;
