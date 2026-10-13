//! Positional reads of an artifact file that do not fill the operating
//! system's file cache: the rows of a streamed table, read as steps name
//! them, never grow memory the plan did not charge.
//!
//! - Linux: `O_DIRECT`, whose offsets, lengths and buffers are aligned to
//!   [`ALIGNMENT`].
//! - macOS: `F_NOCACHE`.
//! - Windows: `FILE_FLAG_NO_BUFFERING`, aligned as `O_DIRECT`.
//!
//! A filesystem that refuses direct I/O, when the file is opened or at its
//! first direct read (`EINVAL` or `EOPNOTSUPP` on Linux,
//! `ERROR_INVALID_PARAMETER` on Windows), is read through the file cache
//! from then on, and on Linux the pages each read filled are dropped
//! (`POSIX_FADV_DONTNEED`).
//!
//! Requested ranges are read in ascending order, ranges whose aligned
//! blocks touch or lie within [`COALESCE_GAP`] of each other in one read,
//! and up to [`READERS`] reads at once: storage serves parallel reads at
//! nearly the latency of one.

use crate::Error;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The alignment of a direct read's offset, length and buffer: the largest
/// logical block size of the storage served.
pub const ALIGNMENT: u64 = 4096;

/// Ranges whose aligned blocks are at most this far apart are read in one.
pub const COALESCE_GAP: u64 = 64 * 1024;

/// Reads in flight at once.
pub const READERS: usize = 16;

/// An artifact file opened for reads that bypass the file cache.
#[derive(Debug)]
pub struct UncachedSource {
    path: PathBuf,
    /// The file opened for direct reads.
    file: File,
    size: u64,
    /// The file opened for reads through the cache, once its filesystem
    /// refused direct I/O: every later read uses it.
    cached: OnceLock<File>,
}

impl UncachedSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let (file, direct) = open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(Error::Invalid("artifact source is not a regular file".into()));
        }
        let source = Self {
            path: path.to_owned(),
            size: metadata.len(),
            file,
            cached: OnceLock::new(),
        };
        if !direct {
            source.fall_back()?;
        }
        Ok(source)
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Whether reads go through the file cache: the filesystem refused
    /// direct I/O.
    pub fn reads_cached(&self) -> bool {
        self.cached.get().is_some()
    }

    /// Read every range of `ranges`, `(offset, length)` in the file, and
    /// hand `take` each one's index in `ranges` and its bytes, or the error
    /// that kept them from being read: a failed read fails only the ranges
    /// it serves.
    pub fn read_ranges(
        &self,
        ranges: &[(u64, usize)],
        mut take: impl FnMut(usize, Result<&[u8], &Error>),
    ) {
        let outside = Error::Invalid("read outside artifact".into());
        let mut order = Vec::with_capacity(ranges.len());
        for (index, (offset, length)) in ranges.iter().enumerate() {
            if offset
                .checked_add(*length as u64)
                .is_none_or(|end| end > self.size)
            {
                take(index, Err(&outside));
            } else {
                order.push(index);
            }
        }
        order.sort_by_key(|index| ranges[*index].0);
        let reads = coalesce(ranges, &order, self.size);
        // Each read lands in a buffer of its own: at most READERS at once.
        for batch in reads.chunks(READERS) {
            let results = std::thread::scope(|scope| {
                let workers = batch
                    .iter()
                    .map(|read| scope.spawn(move || self.read_block(read.start, read.end)))
                    .collect::<Vec<_>>();
                workers
                    .into_iter()
                    .map(|worker| worker.join().expect("an artifact read never panics"))
                    .collect::<Vec<_>>()
            });
            for (read, result) in batch.iter().zip(results) {
                for index in &order[read.ranges.clone()] {
                    let (offset, length) = ranges[*index];
                    let at = (offset - read.start) as usize;
                    take(
                        *index,
                        result
                            .as_ref()
                            .map(|buffer| &buffer.bytes()[at..at + length]),
                    );
                }
            }
        }
    }

    /// Read through the file cache from now on.
    fn fall_back(&self) -> Result<&File, Error> {
        if let Some(file) = self.cached.get() {
            return Ok(file);
        }
        let file = File::open(&self.path)?;
        // A concurrent read may have fallen back first: either file reads
        // the same bytes.
        Ok(self.cached.get_or_init(|| file))
    }

    /// The bytes `[start, end)`, aligned, read whole (the tail of the file
    /// short of `end`). A direct read the filesystem refuses is read again
    /// through the cache, as every later one is.
    fn read_block(&self, start: u64, end: u64) -> Result<Aligned, Error> {
        if let Some(cached) = self.cached.get() {
            return self.read_block_from(cached, start, end);
        }
        match self.read_block_from(&self.file, start, end) {
            Err(Error::Io(error)) if refuses_direct(&error) => {
                let cached = self.fall_back()?;
                self.read_block_from(cached, start, end)
            }
            result => result,
        }
    }

    fn read_block_from(&self, file: &File, start: u64, end: u64) -> Result<Aligned, Error> {
        let mut buffer = Aligned::new((end - start) as usize);
        let wanted = (end.min(self.size) - start) as usize;
        let mut done = 0usize;
        while done < wanted {
            let out = &mut buffer.bytes_mut()[done..];
            #[cfg(unix)]
            let result = {
                use std::os::unix::fs::FileExt;
                file.read_at(out, start + done as u64)
            };
            #[cfg(windows)]
            let result = {
                use std::os::windows::fs::FileExt;
                file.seek_read(out, start + done as u64)
            };
            match result {
                // A direct read of the file's last block ends at its end.
                Ok(0) => break,
                Ok(count) => done += count,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
        if done < wanted {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "open artifact changed or was truncated",
            )
            .into());
        }
        if self.reads_cached() {
            drop_pages(file, start, end);
        }
        Ok(buffer)
    }
}

/// One read: aligned `[start, end)` and the requested ranges it serves, as
/// a range of the sorted order.
struct Read {
    start: u64,
    end: u64,
    ranges: std::ops::Range<usize>,
}

/// The aligned reads serving `ranges` in ascending `order`: a range joins
/// the read before it when its aligned blocks touch it or lie within
/// [`COALESCE_GAP`]. A read's end may pass the file's end by less than a
/// block.
fn coalesce(ranges: &[(u64, usize)], order: &[usize], size: u64) -> Vec<Read> {
    let aligned = |offset: u64, length: usize| {
        let start = offset - offset % ALIGNMENT;
        let end = (offset + length as u64).div_ceil(ALIGNMENT) * ALIGNMENT;
        (start, end.max(start + ALIGNMENT).min(size.div_ceil(ALIGNMENT) * ALIGNMENT))
    };
    let mut reads: Vec<Read> = Vec::new();
    for (position, index) in order.iter().enumerate() {
        let (offset, length) = ranges[*index];
        let (start, end) = aligned(offset, length);
        match reads.last_mut() {
            Some(read) if start <= read.end + COALESCE_GAP => {
                read.end = read.end.max(end);
                read.ranges.end = position + 1;
            }
            _ => reads.push(Read {
                start,
                end,
                ranges: position..position + 1,
            }),
        }
    }
    reads
}

/// A heap buffer whose first byte is [`ALIGNMENT`]-aligned.
struct Aligned {
    storage: Vec<u8>,
    first: usize,
    length: usize,
}

impl Aligned {
    fn new(length: usize) -> Self {
        let storage = vec![0u8; length + ALIGNMENT as usize];
        let first = storage.as_ptr().align_offset(ALIGNMENT as usize);
        Self {
            storage,
            first,
            length,
        }
    }

    fn bytes(&self) -> &[u8] {
        &self.storage[self.first..self.first + self.length]
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.storage[self.first..self.first + self.length]
    }
}

/// The file opened for reads without the cache, and whether it was: a
/// filesystem that refuses direct I/O opens it for reads through the cache.
#[cfg(target_os = "linux")]
fn open(path: &Path) -> Result<(File, bool), Error> {
    use std::os::unix::fs::OpenOptionsExt;
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
    {
        Ok(file) => Ok((file, true)),
        // A filesystem without direct I/O (tmpfs, some overlays).
        Err(error) if refuses_direct(&error) => Ok((File::open(path)?, false)),
        Err(error) => Err(error.into()),
    }
}

#[cfg(target_os = "macos")]
fn open(path: &Path) -> Result<(File, bool), Error> {
    use std::os::fd::AsRawFd;
    let file = File::open(path)?;
    // SAFETY: an fcntl on a descriptor this function owns.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
        return Err(io::Error::last_os_error().into());
    }
    Ok((file, true))
}

#[cfg(windows)]
fn open(path: &Path) -> Result<(File, bool), Error> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING)
        .open(path)
    {
        Ok(file) => Ok((file, true)),
        Err(error) if refuses_direct(&error) => Ok((File::open(path)?, false)),
        Err(error) => Err(error.into()),
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn open(path: &Path) -> Result<(File, bool), Error> {
    Ok((File::open(path)?, false))
}

/// Whether `error` is a filesystem's refusal of direct I/O, rather than a
/// failure of the storage.
#[cfg(target_os = "linux")]
fn refuses_direct(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(libc::EINVAL | libc::EOPNOTSUPP))
}

#[cfg(windows)]
fn refuses_direct(error: &io::Error) -> bool {
    const ERROR_INVALID_PARAMETER: i32 = 87;
    error.raw_os_error() == Some(ERROR_INVALID_PARAMETER)
}

/// `F_NOCACHE` has no alignment to refuse.
#[cfg(not(any(target_os = "linux", windows)))]
fn refuses_direct(_error: &io::Error) -> bool {
    false
}

#[cfg(target_os = "linux")]
fn drop_pages(file: &File, start: u64, end: u64) {
    use std::os::fd::AsRawFd;
    // SAFETY: advice on a range of a descriptor this source owns.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            start as libc::off_t,
            (end - start) as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        )
    };
}

#[cfg(not(target_os = "linux"))]
fn drop_pages(_file: &File, _start: u64, _end: u64) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_read_as_stored_whatever_their_alignment_and_order() {
        let path = std::env::temp_dir().join(format!(
            "magnitude-uncached-{}.bin",
            std::process::id()
        ));
        // A file whose size is no multiple of the block.
        let bytes = (0..3 * ALIGNMENT as usize + 777)
            .map(|index| (index * 31 % 251) as u8)
            .collect::<Vec<_>>();
        std::fs::write(&path, &bytes).unwrap();
        let source = UncachedSource::open(&path).unwrap();
        assert_eq!(source.size(), bytes.len() as u64);
        let ranges = [
            (5000u64, 90usize),
            (3, 90),
            (4090, 90),
            (bytes.len() as u64 - 90, 90),
            (5000, 90),
            (0, bytes.len()),
            (2 * ALIGNMENT, 0),
        ];
        let mut read = vec![None; ranges.len()];
        source.read_ranges(&ranges, |index, data| {
            read[index] = Some(data.unwrap().to_vec())
        });
        for (index, (offset, length)) in ranges.iter().enumerate() {
            let offset = *offset as usize;
            assert_eq!(
                read[index].as_deref(),
                Some(&bytes[offset..offset + length]),
                "range {index}"
            );
        }
        std::fs::remove_file(path).unwrap();
    }

    /// A range that cannot be read fails alone: the ranges of other reads
    /// are read, whether the range lies outside the file as opened or the
    /// file was cut short under the open source.
    #[test]
    fn a_failed_read_fails_only_the_ranges_it_serves() {
        let path = std::env::temp_dir().join(format!(
            "magnitude-uncached-failed-{}.bin",
            std::process::id()
        ));
        let bytes = (0..64 * ALIGNMENT as usize)
            .map(|index| (index * 7 % 253) as u8)
            .collect::<Vec<_>>();
        std::fs::write(&path, &bytes).unwrap();
        let source = UncachedSource::open(&path).unwrap();
        // Cut the file to its first half: reads of the second half end
        // short.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(32 * ALIGNMENT)
            .unwrap();
        let ranges = [
            (100u64, 90usize),
            (48 * ALIGNMENT, 90),
            (bytes.len() as u64 - 1, 2),
            (16 * ALIGNMENT, 90),
        ];
        let mut read = vec![None; ranges.len()];
        source.read_ranges(&ranges, |index, data| {
            read[index] = Some(data.map(<[u8]>::to_vec).map_err(|error| error.to_string()))
        });
        assert_eq!(read[0], Some(Ok(bytes[100..190].to_vec())));
        assert!(matches!(read[1], Some(Err(_))));
        assert!(matches!(read[2], Some(Err(_))));
        let at = 16 * ALIGNMENT as usize;
        assert_eq!(read[3], Some(Ok(bytes[at..at + 90].to_vec())));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn near_ranges_share_a_read_and_far_ones_do_not() {
        let ranges = [(0u64, 90usize), (8000, 90), (2_000_000, 90), (1_000_000, 90)];
        let order = [0, 1, 3, 2];
        let reads = coalesce(&ranges, &order, 10_000_000);
        assert_eq!(reads.len(), 3);
        assert_eq!((reads[0].start, reads[0].end), (0, 2 * ALIGNMENT));
        assert_eq!(reads[0].ranges, 0..2);
        assert_eq!(reads[1].start, 1_000_000 - 1_000_000 % ALIGNMENT);
        assert_eq!(reads[2].ranges, 3..4);
    }
}
