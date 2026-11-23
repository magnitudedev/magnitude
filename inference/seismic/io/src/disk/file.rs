#[cfg(doc)]
use crate::scheduler::Flight;
use super::platform::{drop_pages, refuses_direct, reopen};
use super::ALIGNMENT;
use crate::scheduler::{Buffers, Piece};
use crate::{Buffer, Error};
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::{Arc, OnceLock};

/// A piece's bytes as its read from the volume left them
/// ([`SourceFile::fetch`]): landed in its one sink, or held in a buffer of
/// the pool's until they land in its parts' sinks ([`Self::land`]).
pub(crate) enum Fetched {
    Landed,
    Held(Buffer),
}

impl Fetched {
    /// Land `piece`'s parts: each one's bytes copied out of the buffer that
    /// holds them, which returns to the worker's `scratch` for the next piece.
    pub(crate) fn land(self, piece: &Piece, scratch: &mut Option<Buffer>) {
        let Self::Held(mut buffer) = self else {
            return;
        };
        let bytes = buffer.bytes_mut();
        for part in &piece.parts {
            // SAFETY: `part.length` bytes of the part's sink, which the pool
            // writes alone until the job ends (`Reader::submit`).
            unsafe { std::ptr::copy_nonoverlapping(bytes[part.from..].as_ptr(), part.sink.as_ptr(), part.length) };
        }
        *scratch = Some(buffer);
    }
}

#[derive(Debug)]
pub(crate) struct SourceFile {
    /// The name its source gave it, which a piece in flight is reported by
    /// ([`Flight`]).
    pub(crate) path: Arc<Path>,
    pub(crate) size: u64,
    pub(crate) through: Through,
}

/// How a reader's file is read. Every descriptor is of the one file the
/// reader's [`crate::OpenFile`] has open, never of what its path names
/// later, so a replaced path changes no byte a reader reads.
#[derive(Debug)]
pub(crate) enum Through {
    /// Uncached: direct reads of `file`, and once its filesystem refused
    /// direct I/O at a read, reads of `fallback` (`file` reopened through
    /// the cache) through the cache, each read's pages dropped: every later
    /// read uses it.
    Direct { file: File, fallback: OnceLock<File> },
    /// Uncached where the file could not be opened again past the cache:
    /// its filesystem refused direct I/O when the file was opened, or (on
    /// macOS, which opens a file again only by name) nothing names it any
    /// more. Reads through the cache, each read's pages dropped.
    Dropped(File),
    /// Cached: reads of the file through the cache, keeping their pages.
    Cache(File),
}

impl Through {
    /// Direct reads of `file`, until its filesystem refuses one.
    pub(crate) fn direct(file: File) -> Self {
        Self::Direct {
            file,
            fallback: OnceLock::new(),
        }
    }

    /// The file an uncached reader reads through the cache, dropping each
    /// read's pages, once its filesystem refused direct I/O.
    fn dropping(&self) -> Option<&File> {
        match self {
            Self::Direct { fallback, .. } => fallback.get(),
            Self::Dropped(file) => Some(file),
            Self::Cache(_) => None,
        }
    }
}

impl SourceFile {
    /// Read `piece` and land its parts: straight into the one sink of an
    /// uncached piece that lies within one range where the sink meets the
    /// reader's alignment, otherwise through the worker's `scratch` buffer
    /// (falling back to `buffers` only when `scratch` is absent or too small).
    pub(crate) fn read_piece(&self, piece: &Piece, scratch: &mut Option<Buffer>, buffers: &Buffers) -> Result<(), Error> {
        let fetched = self.fetch(piece, scratch, buffers)?;
        fetched.land(piece, scratch);
        Ok(())
    }

    /// Read `piece` from the volume: straight into its sink(s) where it
    /// lands straight ([`Self::lands_straight`]), otherwise into the worker's
    /// `scratch` buffer (borrowing from `buffers` only when `scratch` is empty
    /// or smaller than `span`), its parts still to land ([`Fetched::land`]).
    pub(crate) fn fetch(&self, piece: &Piece, scratch: &mut Option<Buffer>, buffers: &Buffers) -> Result<Fetched, Error> {
        let span = (piece.end - piece.start) as usize;
        if self.lands_straight(piece) {
            if let [part] = piece.parts.as_slice() {
                // SAFETY: the piece's one part's sink is `span` bytes the pool
                // writes alone until the job ends (`Reader::request`).
                let sink = unsafe { std::slice::from_raw_parts_mut(part.sink.as_ptr(), span) };
                return self.read_block(piece.start, sink).map(|()| Fetched::Landed);
            }
            #[cfg(unix)]
            return self.read_block_vectored(piece.start, span, &piece.parts).map(|()| Fetched::Landed);
        }
        let mut buffer = match scratch.take() {
            Some(buffer) if buffer.len() >= span => buffer,
            smaller => {
                if let Some(smaller) = smaller {
                    buffers.give(smaller);
                }
                buffers.take(span)
            }
        };
        match self.read_block(piece.start, &mut buffer.bytes_mut()[..span]) {
            Ok(()) => Ok(Fetched::Held(buffer)),
            Err(error) => {
                *scratch = Some(buffer);
                Err(error)
            }
        }
    }

    /// Whether `piece` is read straight into its sink(s): an uncached piece
    /// whose parts tile `[0, piece.end - piece.start)` contiguously with no
    /// gaps or overlaps and meet the alignment its read needs ([`ALIGNMENT`]
    /// where the file is read direct, otherwise none). A single part uses
    /// `pread`; on Unix, up to `IOV_MAX_DIRECT` contiguous aligned parts use
    /// `preadv` scatter-gather with zero bounce copy.
    /// Linux and Windows refuse a misaligned direct read; macOS serves one
    /// slowly: an expert record (1.92 MB in 16 pieces at an unaligned file
    /// offset) read straight into its staging took 1.18 ms, through the
    /// worker's buffer 0.93 ms (M4 Pro; in a streamed 4k step, demand
    /// reads 1.55 → 1.98 GB/s). A cached piece never is.
    pub(crate) fn lands_straight(&self, piece: &Piece) -> bool {
        let alignment = match (&self.through, self.through.dropping()) {
            (Through::Cache(_), _) => return false,
            (_, Some(_)) => 1,
            (_, None) => ALIGNMENT as usize,
        };
        let span = (piece.end - piece.start) as usize;
        match piece.parts.as_slice() {
            [part] => part.from == 0 && part.length == span && (part.sink.as_ptr() as usize).is_multiple_of(alignment),
            #[cfg(unix)]
            parts if !parts.is_empty() && parts.len() <= IOV_MAX_DIRECT => {
                let mut cursor = 0usize;
                for part in parts {
                    if part.from != cursor
                        || part.length == 0
                        || !part.length.is_multiple_of(alignment)
                        || !(part.sink.as_ptr() as usize).is_multiple_of(alignment)
                    {
                        return false;
                    }
                    cursor += part.length;
                }
                cursor == span
            }
            _ => false,
        }
    }

    /// Dispatch an aligned read of `span` bytes at `start` through
    /// `self.through`: cached reads run directly, direct reads fall back once
    /// on `refuses_direct`, and dropped/fallback reads evict their pages.
    fn read_aligned(
        &self,
        start: u64,
        span: usize,
        mut read: impl FnMut(&File, bool) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let cached = match &self.through {
            Through::Cache(file) => return read(file, false),
            Through::Dropped(file) => file,
            Through::Direct { file, fallback } => match fallback.get() {
                Some(cached) => cached,
                None => match read(file, true) {
                    Err(Error::Io(error)) if refuses_direct(&error) => {
                        let file = reopen(file, false)?;
                        // A concurrent read may have fallen back first:
                        // either file reads the same bytes.
                        let mut initialized = false;
                        let cached = fallback.get_or_init(|| {
                            initialized = true;
                            file
                        });
                        if initialized {
                            tracing::warn!(
                                path = %self.path.display(),
                                error = %error,
                                "direct I/O refused on read; falling back to cached reads with page eviction"
                            );
                        }
                        cached
                    }
                    result => return result,
                },
            },
        };
        read(cached, false)?;
        drop_pages(cached, start, (start + span as u64).min(self.size))?;
        Ok(())
    }

    /// The bytes from `start` (aligned) into `into` (an aligned length),
    /// whole (the tail of the file short of its end). An uncached read the
    /// filesystem refuses direct is read again through the cache, as every
    /// later one is, and the pages it filled are dropped.
    pub(crate) fn read_block(&self, start: u64, into: &mut [u8]) -> Result<(), Error> {
        self.read_aligned(start, into.len(), |file, direct| self.read_block_from(file, start, into, direct))
    }

    /// Scatter-gather direct read of contiguous aligned `parts` tiling
    /// `[start, start + span)` directly into each part's sink with `preadv`.
    #[cfg(unix)]
    fn read_block_vectored(&self, start: u64, span: usize, parts: &[crate::scheduler::Part]) -> Result<(), Error> {
        self.read_aligned(start, span, |file, direct| self.read_block_vectored_from(file, start, span, parts, direct))
    }

    #[cfg(unix)]
    fn read_block_vectored_from(
        &self,
        file: &File,
        start: u64,
        span: usize,
        parts: &[crate::scheduler::Part],
        direct: bool,
    ) -> Result<(), Error> {
        let end = start + span as u64;
        let wanted = (end.min(self.size) - start) as usize;
        let mut iovecs = [libc::iovec { iov_base: std::ptr::null_mut(), iov_len: 0 }; IOV_MAX_DIRECT];
        let mut count = 0usize;
        for part in parts {
            let base = part.sink.as_ptr() as *mut libc::c_void;
            if count > 0 {
                let prev = &mut iovecs[count - 1];
                // SAFETY: comparing end of previous part's sink with next part's sink.
                if unsafe { (prev.iov_base as *const u8).add(prev.iov_len) } == base as *const u8 {
                    prev.iov_len += part.length;
                    continue;
                }
            }
            iovecs[count] = libc::iovec { iov_base: base, iov_len: part.length };
            count += 1;
        }
        match read_positioned_vectored(file, start, &mut iovecs[..count], wanted) {
            Ok(done) if done < wanted => Err(truncated().into()),
            Ok(_) => Ok(()),
            Err((error, done)) if direct && !(done as u64).is_multiple_of(ALIGNMENT) => Err(io::Error::new(
                error.kind(),
                format!("a direct read at {start}, misaligned {done} bytes in by a short read: {error}"),
            )
            .into()),
            Err((error, _)) => Err(error.into()),
        }
    }

    /// The bytes from `start` into `into`, read from `file` until the
    /// file's end. Only an aligned `direct` request's failure can be the
    /// filesystem's refusal of direct I/O (`refuses_direct`): one a short
    /// read left misaligned fails as an I/O error.
    fn read_block_from(&self, file: &File, start: u64, into: &mut [u8], direct: bool) -> Result<(), Error> {
        let end = start + into.len() as u64;
        let wanted = (end.min(self.size) - start) as usize;
        match read_positioned(file, start, into, wanted) {
            Ok(done) if done < wanted => Err(truncated().into()),
            Ok(_) => Ok(()),
            Err((error, done)) if direct && !(done as u64).is_multiple_of(ALIGNMENT) => Err(io::Error::new(
                error.kind(),
                format!("a direct read at {start}, misaligned {done} bytes in by a short read: {error}"),
            )
            .into()),
            Err((error, _)) => Err(error.into()),
        }
    }
}

/// Read `file` from `start` into `into` until `wanted` bytes have landed or
/// the file ends (a direct read of its last block ends at its end), each
/// read asking for the rest of `into`: the bytes read, or the error that
/// stopped it with the bytes read before. Every positional read of the
/// process loops here.
pub(crate) fn read_positioned(file: &File, start: u64, into: &mut [u8], wanted: usize) -> Result<usize, (io::Error, usize)> {
    let mut done = 0usize;
    while done < wanted {
        let out = &mut into[done..];
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
            Ok(0) => break,
            Ok(count) => done += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err((error, done)),
        }
    }
    Ok(done)
}

/// Maximum parts in a single direct `preadv` scatter-gather read (stack-sized,
/// well below `IOV_MAX` = 1024 on Linux and macOS).
#[cfg(unix)]
const IOV_MAX_DIRECT: usize = 64;

/// Scatter-gather positional read from `start` into `iovecs` until `wanted`
/// bytes land or EOF, handling `EINTR` and partial reads.
#[cfg(unix)]
fn read_positioned_vectored(
    file: &File,
    start: u64,
    iovecs: &mut [libc::iovec],
    wanted: usize,
) -> Result<usize, (io::Error, usize)> {
    use std::os::fd::AsRawFd;
    let mut done = 0usize;
    let mut idx = 0usize;
    while done < wanted {
        while idx < iovecs.len() && iovecs[idx].iov_len == 0 {
            idx += 1;
        }
        if idx >= iovecs.len() {
            break;
        }
        let Ok(offset) = libc::off_t::try_from(start + done as u64) else {
            return Err((io::Error::new(io::ErrorKind::InvalidInput, "read offset exceeds host file domain"), done));
        };
        let remaining = &iovecs[idx..];
        // SAFETY: every iovec in `remaining` points into a destination slice
        // owned exclusively by this piece until its request ends.
        let n = unsafe { libc::preadv(file.as_raw_fd(), remaining.as_ptr(), remaining.len() as libc::c_int, offset) };
        if n < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err((error, done));
        }
        if n == 0 {
            break;
        }
        let mut advance = n as usize;
        done += advance;
        while advance > 0 && idx < iovecs.len() {
            if advance >= iovecs[idx].iov_len {
                advance -= iovecs[idx].iov_len;
                iovecs[idx].iov_len = 0;
                idx += 1;
            } else {
                // SAFETY: `advance < iov_len`, stays within the part's slice.
                iovecs[idx].iov_base = unsafe { (iovecs[idx].iov_base as *mut u8).add(advance) as *mut libc::c_void };
                iovecs[idx].iov_len -= advance;
                advance = 0;
            }
        }
    }
    Ok(done)
}

/// A file that ended short of the bytes it held when it was opened.
pub(crate) fn truncated() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "open file changed or was truncated")
}
