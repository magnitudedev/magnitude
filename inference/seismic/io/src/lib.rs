//! Seismic's I/O layer: positional reads of files on storage volumes, the
//! scheduler every reader's jobs run on, and its accounting per volume.
//!
//! - [`destination`]: what a read lands in ([`Destination`]).
//! - [`disk`]: storage volumes ([`Volume`]), files opened for reads
//!   ([`OpenFile`], [`Reader`]), the platform's direct and cached reads, and
//!   each volume's probe.
//! - [`scheduler`]: the read engine: requests, their pieces, each volume's
//!   link and the pool of workers.
//! - [`accounting`]: what each volume did, due and ahead, never consumed.
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
//! Every read a [`Reader`] requests runs on one pool of persistent workers,
//! and every volume has one **link**, the only authority on what of it is
//! in flight. A request names ranges of a file, each landing in a
//! [`Destination`] the request holds, and what it is for, its [`Want`]: a
//! deadline (the instant its consumer would wait from, or none) and a
//! value. It is read in ascending order, ranges whose aligned blocks touch
//! or lie within the coalescing gap (128 KiB) of each other in one aligned
//! read, each read split into pieces of the link's read size (a due
//! request's own, "A volume's curve"). An uncached piece that lies within
//! one range is read straight into its destination where that meets the
//! reader's buffer alignment; any other piece is read into the worker's own
//! aligned buffer and copied into the destinations of the ranges it serves.
//! A [`Handle`] owns the request: [`Handle::reprice`] reorders its pieces
//! not yet taken, and dropping it cancels them (they read nothing, the
//! ranges they serve fail [`Error::Cancelled`], and it ends with the pieces
//! in flight). Its [`Ending`] observes the end, and may outlive it. A failed piece fails only the ranges whose own bytes fail to
//! read: a piece serving several is read again range by range.
//!
//! # Order and depth
//!
//! A link orders its requests by want: the earlier deadline, then the
//! greater value, then submission. A request is due once its deadline has
//! come, so the due ones lead. A due request is waited on in one of two
//! ways: a step **parks** on it (a decode step's miss, a hint it comes to
//! wait on), or a **stream** consumes it at the link's rate (a prompt
//! group's conveyor, which keeps the link at its due depth and parks on no
//! single read). Its depths come from the volume's curve ("A volume's
//! curve"), in bytes in flight:
//!
//! 1. A due piece, parked or streamed, starts whenever the due bytes in
//!    flight leave it room under the due depth.
//! 2. A piece not due starts only while no parked piece is queued, and only
//!    while the parked bytes in flight and those read ahead leave it room
//!    under the knee: below the knee depth still adds rate, so the read
//!    ahead runs on throughput the parked reads leave unused; past it every
//!    byte more in flight would delay a parked byte, so the read ahead
//!    waits. A stream's bytes are not counted against the knee: it takes
//!    the rate it takes, and a read ahead beside it (a bank's fill during a
//!    prompt) shares the device as the stream's own reads do.
//! 3. A request held back at one depth holds back only the requests counted
//!    against it: a due one the due pieces behind it, a parked one
//!    everything read ahead too, a streamed one nothing read ahead.
//!
//! A piece past its depth starts where nothing it is counted against is in
//! flight. So what a wrong read ahead costs a parked read is bounded. One
//! not started when its handle drops costs nothing. One that started and
//! ended while no parked read was outstanding used link time no step was
//! parked on. One in flight when a parked read arrives shares the device
//! with it: the parked read's pieces start at once (rule 1), and its extra
//! time is at most the bytes read ahead in flight over the rate, at most
//! the knee's, which nothing later adds to (rules 1 and 2). Where no stream
//! runs (decode), these are the rules of a link with one kind of due read.
//!
//! # How reads share the engine
//!
//! A read of `b` bytes the engine serves alone ends `λ + b/D` after it is
//! submitted. Reads queued together overlap their latencies: each read
//! queued behind others holds the engine only for its bytes, `b/D`, so `k`
//! reads submitted at once end `λ + k·b/D` after, and only reads each
//! waited for before the next pay `k·(λ + b/D)`. Here `λ` and `D` are the
//! volume's read line, from a load's own reads: `D` the volume's bytes over
//! its busy time (its [`accounting`]), `λ` what the due reads took beyond
//! that: what the performance model prices every read by. Measured on an
//! RTX 3090 host (EPYC 7313), uncached records at random offsets of Qwen3.6
//! 35B's file (`tests/read_law.rs`): a 1.9 MB record alone 0.53–0.58 ms,
//! sixteen at once 6.69 ms (each more 0.41 ms, 4.7 GB/s), sixteen each
//! waited for 10.97 ms; 256 KiB records 0.29 ms alone and 0.135 ms each more
//! at once (1.9 GB/s), 8 MiB 1.71 ms and 1.61 ms each more (5.3 GB/s). So
//! nothing in the engine serves reads one after another; the queued rate
//! falls with a record's size (a random access per record), which a line
//! through records of the size served carries in its `D`.

pub mod accounting;
pub mod destination;
pub mod disk;
mod error;
pub mod scheduler;
pub mod timeline;
mod waited;

pub use accounting::{Class, Totals, Waits};
pub use destination::{Buffer, Destination};
pub use disk::{adopt_earlier, adopt_volume, cached_pages, link_shape, probe_volume, read_cores, CachePolicy, Curve, EarlierChoice, Lone, OpenError, OpenFile, Point, Reader, Shape, Urgent, Volume, ALIGNMENT};
pub use error::Error;
pub use scheduler::{engine_work, read_buffers, Deadline, Ended, Ending, EngineWork, Flight, Handle, Outstanding, Panic, Range, Want};
pub use waited::waited_on;

#[cfg(test)]
mod tests;
