//! Files on storage volumes, opened for reads: how each is read (direct
//! past the file cache, or through it), each volume's identity and probe
//! ("A volume's curve", [`probe_volume`]), and the reader a caller makes
//! requests through.

use crate::scheduler::{self, Ended, Handle, Outstanding, Range, Want};
use crate::{Destination, Error};
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
pub(crate) use probe::{full_rate, grid, overlap, WIDEST_WINDOW};

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

    /// `out.len()` bytes from `offset`, read through the file cache on the
    /// caller's thread. A range past the file's end, or a file that ends
    /// short of it, fails as `UnexpectedEof`.
    pub fn read_exact_at(&self, offset: u64, out: &mut [u8]) -> io::Result<()> {
        let wanted = out.len();
        match file::read_positioned(&self.file, offset, out, wanted) {
            Ok(done) if done == wanted => Ok(()),
            Ok(_) => Err(file::truncated()),
            Err((error, _)) => Err(error),
        }
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

    /// `length` bytes from `offset`, read in order as a stream: each read
    /// of the stream is one undated request into the caller's buffer, so
    /// the stream holds none of the bytes itself.
    pub fn range(&self, offset: u64, length: u64) -> RangeReader<'_> {
        RangeReader { reader: self, offset, remaining: length }
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

/// A range of a file read in order ([`Reader::range`]).
pub struct RangeReader<'a> {
    reader: &'a Reader,
    offset: u64,
    remaining: u64,
}

/// A caller's buffer as a read's destination, for a read the caller waits
/// out before the borrow ends ([`RangeReader`]): no request outlives the
/// read that writes it.
struct Borrowed(NonNull<u8>, usize);

// SAFETY: the buffer is the borrower's alone while its one request runs.
unsafe impl Send for Borrowed {}
unsafe impl Sync for Borrowed {}

// SAFETY: the borrow it is made from outlasts every write to it: its
// request ends before `RangeReader::read` returns.
unsafe impl Destination for Borrowed {
    fn region(&self) -> (NonNull<u8>, usize) {
        (self.0, self.1)
    }
}

impl io::Read for RangeReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let length = usize::try_from(self.remaining).map_or(buffer.len(), |left| left.min(buffer.len()));
        let Some(sink) = NonNull::new(buffer.as_mut_ptr()).filter(|_| length != 0) else {
            return Ok(0);
        };
        let range = Range { offset: self.offset, length, into: Arc::new(Borrowed(sink, length)), at: 0 };
        // SAFETY: `buffer` is the caller's, `length` writable bytes of it,
        // touched by nothing else until the request ends (`wait`).
        // An error keeps its kind and itself as the source.
        let request = unsafe { self.reader.request(vec![range], Want::undated(0.)) }.map_err(|error| match error {
            Error::Io(error) => error,
            error => io::Error::other(error),
        })?;
        match request.wait() {
            Ended::Read { failed } => {
                if let Some((_, error)) = failed.into_iter().next() {
                    let kind = match &*error {
                        Error::Io(error) => error.kind(),
                        _ => io::ErrorKind::Other,
                    };
                    return Err(io::Error::new(kind, error));
                }
            }
            Ended::Panicked(panic) => panic.resume(),
        }
        self.offset += length as u64;
        self.remaining -= length as u64;
        Ok(length)
    }
}
