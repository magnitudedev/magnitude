//! Files on storage volumes, opened for reads: how each is read (direct
//! past the file cache, or through it), each volume's identity and probe
//! ("A volume's curve", [`probe_volume`]), and the reader a caller makes
//! requests through.

use crate::scheduler::{self, Ended, Handle, Outstanding, Range, Want};
use crate::{Buffer, Destination, Error};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

mod file;
mod platform;
mod probe;
mod volume;

pub use platform::cached_pages;
pub use probe::{Curve, Lone, Point, Urgent};
pub use volume::{adopt_earlier, adopt_volume, link_shape, probe_volume, read_cores, EarlierChoice, Shape};

pub(crate) use file::{SourceFile, Through};
pub(crate) use probe::LEAST_READ;
pub(crate) use volume::VolumeReads;

#[cfg(all(test, unix))]
pub(crate) use platform::drop_pages;
#[cfg(test)]
pub(crate) use probe::{full_rate, grid, Request, VolumeLines, WIDEST_WINDOW};

use platform::{evict, open_cached, uncached};

/// The alignment of a direct read's offset, length and buffer: the largest
/// logical block size of the storage served.
pub const ALIGNMENT: u64 = 4096;

/// A regular file opened for reads, by the name it was opened by, made
/// absolute but not resolved through links (a content-addressed store
/// names its files by links): what a [`Reader`] of it and its volume's
/// probe read. Reads remain attached to this open file if its pathname is
/// later replaced. Opened with no read-ahead past the ranges read (Linux
/// `POSIX_FADV_RANDOM`, macOS `F_RDAHEAD` off, Windows
/// `FILE_FLAG_RANDOM_ACCESS`).
#[derive(Debug)]
pub struct OpenFile {
    file: File,
    path: PathBuf,
    size: u64,
}

/// Why a file could not be opened for reads.
#[derive(Debug)]
pub enum OpenError {
    Io(io::Error),
    /// It names no regular file.
    NotAFile,
}

impl From<io::Error> for OpenError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl OpenFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, OpenError> {
        let path = std::path::absolute(path.as_ref())?;
        let file = open_cached(&path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(OpenError::NotAFile);
        }
        Ok(Self { file, path, size: metadata.len() })
    }

    pub fn file(&self) -> &File {
        &self.file
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

impl AsRef<OpenFile> for OpenFile {
    fn as_ref(&self) -> &OpenFile {
        self
    }
}

/// A storage volume the scheduler reads, by its identity within
/// the process (`st_dev` on unix, the volume's serial number on Windows).
/// Its accounting ([`crate::accounting`]) and the pieces in flight on it
/// ([`crate::Flight`]) are named by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Volume(pub u64);

impl std::fmt::Display for Volume {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Whether a reader's reads fill the file cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CachePolicy {
    /// Past the file cache, filling none of it.
    Uncached,
    /// Through the file cache, keeping the pages read.
    Cached,
}

/// A file opened for reads by one [`CachePolicy`], its due pieces cut as
/// its urgent request needs ([`Urgent`]).
#[derive(Debug, Clone)]
pub struct Reader {
    pub(crate) file: Arc<SourceFile>,
    pub(crate) volume: VolumeReads,
    /// Its requests made and not yet ended ([`Self::outstanding`]).
    outstanding: Arc<std::sync::atomic::AtomicUsize>,
}

impl Reader {
    /// The file at `path`, read by a reader no step waits on.
    pub fn open(path: impl AsRef<Path>, policy: CachePolicy) -> Result<Self, Error> {
        Self::of(&OpenFile::open(path)?, policy)
    }

    /// The file at `path`, read for steps that wait on `urgent`.
    pub fn open_urgent(path: impl AsRef<Path>, policy: CachePolicy, urgent: &Urgent) -> Result<Self, Error> {
        Self::of_urgent(&OpenFile::open(path)?, policy, urgent)
    }

    /// The file `source` has open, read for steps that wait on `urgent`.
    pub fn of_urgent(source: &impl AsRef<OpenFile>, policy: CachePolicy, urgent: &Urgent) -> Result<Self, Error> {
        Self::reading(source.as_ref(), policy, Some(urgent))
    }

    /// The file `source` has open, read by `policy` by a reader no step
    /// waits on.
    pub fn of(source: &impl AsRef<OpenFile>, policy: CachePolicy) -> Result<Self, Error> {
        Self::reading(source.as_ref(), policy, None)
    }

    /// The file `source` has open, read by `policy`: its descriptor shared
    /// where cached, the same file opened again past the cache where not;
    /// read as its volume's shape for `urgent` says, whatever its policy;
    /// starting the pool's workers it needs ("A volume's curve").
    fn reading(source: &OpenFile, policy: CachePolicy, urgent: Option<&Urgent>) -> Result<Self, Error> {
        let file = source.file();
        let through = match policy {
            CachePolicy::Uncached => uncached(file)?,
            CachePolicy::Cached => Through::Cache(file.try_clone()?),
        };
        let path: Arc<Path> = source.path().into();
        let volume = VolumeReads::of(file, urgent)?;
        scheduler::deepen(volume.shape.workers())?;
        Ok(Self { file: Arc::new(SourceFile { path, size: source.size(), through }), volume, outstanding: Arc::default() })
    }

    /// Its requests made and not yet ended, as a handle that reads them
    /// while the reader lives on elsewhere: while any is, a waiter on one
    /// waits on the engine, whose work [`crate::engine_work`] reports.
    pub fn outstanding(&self) -> Outstanding {
        Outstanding(self.outstanding.clone())
    }

    pub fn size(&self) -> u64 {
        self.file.size
    }

    /// The bytes in flight at which its link reads at its full rate (its
    /// knee, "A volume's curve"): below them the link lacks work.
    pub fn knee(&self) -> u64 {
        self.volume.shape.knee
    }

    /// The volume its file lies on, whose counts every reader of it shares
    /// ([`crate::accounting::totals`]).
    pub fn volume(&self) -> Volume {
        self.volume.id
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

    /// Read each of `reads`' buffers from its offset, as one request for
    /// `want`, and wait for it: a caller that needs the bytes before it goes
    /// on (a header, a table's rows wanted now). The first read's error
    /// fails it, its kind kept.
    pub fn read_at<'a>(&self, reads: impl IntoIterator<Item = (u64, &'a mut [u8])>, want: Want) -> io::Result<()> {
        let ranges = reads
            .into_iter()
            .filter_map(|(offset, into)| {
                let length = into.len();
                NonNull::new(into.as_mut_ptr()).filter(|_| length != 0).map(|base| Range { offset, length, into: Arc::new(Borrowed(base, length)), at: 0 })
            })
            .collect::<Vec<_>>();
        if ranges.is_empty() {
            return Ok(());
        }
        // SAFETY: each buffer is the caller's, borrowed alone (disjoint `&mut`
        // borrows) until the request ends (`wait`), which it does before this
        // returns.
        let request = unsafe { self.request(ranges, want) }.map_err(io_error)?;
        ended(request.wait())
    }

    /// Read the `length` bytes from `offset` in order, as requests for
    /// `want` of half the link's knee each (at least one read), handing each
    /// chunk to `sink` with its offset in the range as it lands: three
    /// requested at once, so the link holds its knee while `sink` takes one
    /// and the stream holds a knee and a half of the file. The first
    /// error, the read's or `sink`'s, ends it; the requests not yet taken
    /// are cancelled.
    pub fn stream<E: From<io::Error>>(
        &self,
        offset: u64,
        length: u64,
        want: Want,
        mut sink: impl FnMut(u64, &[u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        let chunk = (self.knee() / 2).max(LEAST_READ) as usize;
        let depth = 3;
        let mut flight = std::collections::VecDeque::<(Handle, Arc<Buffer>, u64, usize)>::new();
        let mut spare = Vec::<Arc<Buffer>>::new();
        let mut next = 0u64;
        loop {
            while flight.len() < depth && next < length {
                let bytes = (length - next).min(chunk as u64) as usize;
                let buffer = spare.pop().unwrap_or_else(|| Arc::new(Buffer::new(chunk)));
                let range = Range { offset: offset + next, length: bytes, into: buffer.clone(), at: 0 };
                // SAFETY: the buffer is the stream's alone, read by it only
                // once its request has ended (`wait`).
                let request = unsafe { self.request(vec![range], want) }.map_err(io_error)?;
                flight.push_back((request, buffer, next, bytes));
                next += bytes as u64;
            }
            let Some((request, buffer, at, bytes)) = flight.pop_front() else {
                return Ok(());
            };
            ended(request.wait())?;
            // SAFETY: its one request has ended.
            sink(at, &unsafe { buffer.bytes() }[..bytes])?;
            spare.push(buffer);
        }
    }

    /// Request `ranges` of its volume's link for `want`. A range outside
    /// the file fails at once; one past its destination's region is a
    /// defect, and panics. The request holds every range's destination until
    /// its last piece has ended, whatever becomes of its handle.
    ///
    /// # Safety
    /// Each range's bytes of its destination are disjoint from every other
    /// range's of the request and of any request in flight, and from every
    /// byte the caller touches, until the request has ended: its
    /// [`Handle::wait`] returned, or its hook ran ([`Handle::on_end`]).
    /// Dropping the handle cancels the pieces not yet taken and does not
    /// wait for those in flight.
    pub unsafe fn request(&self, ranges: Vec<Range>, want: Want) -> Result<Handle, Error> {
        // SAFETY: the caller's.
        unsafe { scheduler::request(&self.file, &self.volume, &self.outstanding, ranges, want) }
    }
}

/// A caller's buffer as a read's destination, for a read the caller waits
/// out before the borrow ends ([`Reader::read_at`]): no request outlives
/// the read that writes it.
struct Borrowed(NonNull<u8>, usize);

// SAFETY: the buffer is the borrower's alone while its one request runs.
unsafe impl Send for Borrowed {}
unsafe impl Sync for Borrowed {}

// SAFETY: the borrow it is made from outlasts every write to it: its
// request ends before `Reader::read_at` returns.
unsafe impl Destination for Borrowed {
    fn region(&self) -> (NonNull<u8>, usize) {
        (self.0, self.1)
    }
}

/// An error as an I/O error, its kind kept where it is one.
fn io_error(error: Error) -> io::Error {
    match error {
        Error::Io(error) => error,
        error => io::Error::other(error),
    }
}

/// A request's end as the first of its reads' errors, kind kept.
fn ended(end: Ended) -> io::Result<()> {
    match end {
        Ended::Read { failed } => match failed.into_iter().next() {
            Some((_, error)) => {
                let kind = match &*error {
                    Error::Io(error) => error.kind(),
                    _ => io::ErrorKind::Other,
                };
                Err(io::Error::new(kind, error))
            }
            None => Ok(()),
        },
        Ended::Panicked(panic) => panic.resume(),
    }
}
