//! Seismic's I/O layer: positional reads of files on storage volumes, the
//! scheduler every reader's jobs run on, and its accounting per volume.
//!
//! - [`destination`]: what a read lands in ([`Destination`]).
//! - [`disk`]: storage volumes ([`Volume`]), files opened for reads
//!   ([`OpenFile`], [`Reader`]), the platform's direct and cached reads, and
//!   each volume's probe.
//! - [`scheduler`]: the read engine: jobs, their pieces and the queue.
//! - [`accounting`]: what each volume did, by class, never consumed.
//!
//! A reader reads by its [`CachePolicy`].
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
//! file is opened with no read-ahead past the ranges named (Linux
//! `POSIX_FADV_RANDOM`, macOS `F_RDAHEAD` off, Windows
//! `FILE_FLAG_RANDOM_ACCESS`), and every piece lands through its worker's
//! buffer, then an exact copy into each destination: reading the cache
//! straight into mapped staging measured slower on Linux.
//!
//! # The read engine
//!
//! Every read a [`Reader`] submits runs on one pool of persistent workers. Each
//! volume is read in reads of at most its read size, holding at most its
//! window of bytes in flight ("A volume's window"), and the pool holds as
//! many workers as the widest window a reader has opened needs. A job is
//! the ranges one caller names, each landing in a [`Destination`] the job
//! holds: they are read in ascending order, ranges whose aligned blocks
//! touch or lie within the coalescing gap (128 KiB) of each other in one aligned
//! read, and each read split into pieces of the volume's read size, which
//! the workers take most urgent first: by the
//! job's [`Priority`], then by submission, then in the job's order. Storage
//! serves parallel reads at nearly the latency of one. An uncached piece
//! that lies within one range is read straight into its destination where
//! that meets the reader's buffer alignment; any other piece is read into
//! the worker's own aligned buffer and copied into the destinations of the
//! ranges it serves. A job
//! is promoted (never demoted), and withdrawn ([`Job::withdraw`], or dropped):
//! the pieces no worker has taken are read by none, the ranges they serve
//! failed [`Error::Withdrawn`], and it ends with the pieces in flight. A
//! failed piece fails only the ranges whose own bytes fail to read: a piece
//! serving several is read again range by range.
//!
//! # How reads share the engine
//!
//! A read of `b` bytes the engine serves alone ends `λ + b/D` after it is
//! submitted. Reads queued together overlap their latencies: each read
//! queued behind others holds the engine only for its bytes, `b/D`, so `k`
//! reads submitted at once end `λ + k·b/D` after, and only reads each
//! waited for before the next pay `k·(λ + b/D)`. A read more urgent than
//! those queued goes ahead of them, but waits for the pieces the workers
//! already hold: at most its volume's window ([`Reader::window`]).
//! Here `λ` and `D` are the volume's read line, from a load's own reads:
//! `D` the volume's bytes over its busy time (its [`accounting`]), `λ` what the
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

pub mod accounting;
pub mod destination;
pub mod disk;
mod error;
pub mod scheduler;

pub use accounting::{Class, Totals, Waits};
pub use destination::{Buffer, Destination};
pub use disk::{adopt_volume, cached_pages, probe_volume, read_cores, CachePolicy, OpenError, OpenFile, RangeReader, Reader, Urgent, Volume, VolumeChoice, ALIGNMENT};
pub use error::Error;
pub use scheduler::{engine_work, Ended, EngineWork, Flight, Job, Outstanding, Panic, Priority, Range};

#[cfg(test)]
mod tests;
