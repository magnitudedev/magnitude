#[cfg(doc)]
use crate::scheduler::Flight;
use super::platform::{drop_pages, refuses_direct, reopen};
use super::ALIGNMENT;
use crate::scheduler::{Buffers, Piece};
use crate::Error;
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::{Arc, OnceLock};

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
    /// reader's alignment, otherwise through a buffer of `buffers`.
    pub(crate) fn read_piece(&self, piece: &Piece, buffers: &Buffers) -> Result<(), Error> {
        let span = (piece.end - piece.start) as usize;
        if self.lands_straight(piece) {
            // SAFETY: the piece's one part's sink is `span` bytes the pool
            // writes alone until the job ends (`Reader::submit`).
            let sink =
                unsafe { std::slice::from_raw_parts_mut(piece.parts[0].sink.as_ptr(), span) };
            return self.read_block(piece.start, sink);
        }
        let mut buffer = buffers.take(span);
        let bytes = &mut buffer.bytes_mut()[..span];
        let read = self.read_block(piece.start, bytes);
        if read.is_ok() {
            for part in &piece.parts {
                // SAFETY: as above, `part.length` bytes of the sink.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes[part.from..].as_ptr(),
                        part.sink.as_ptr(),
                        part.length,
                    )
                };
            }
        }
        buffers.give(buffer);
        read
    }

    /// Whether `piece` is read straight into its sink: an uncached piece
    /// that lies within one range whose sink meets the alignment its read
    /// needs ([`ALIGNMENT`] where the file is read direct, otherwise none).
    /// Linux and Windows refuse a misaligned direct read; macOS serves one
    /// slowly: an expert record (1.92 MB in 16 pieces at an unaligned file
    /// offset) read straight into its staging took 1.18 ms, through the
    /// worker's buffer 0.93 ms (M4 Pro; in a streamed 4k step, demand
    /// reads 1.55 → 1.98 GB/s). A cached
    /// piece never is.
    pub(crate) fn lands_straight(&self, piece: &Piece) -> bool {
        let alignment = match (&self.through, self.through.dropping()) {
            (Through::Cache(_), _) => return false,
            (_, Some(_)) => 1,
            (_, None) => ALIGNMENT as usize,
        };
        match piece.parts.as_slice() {
            [part] => {
                part.from == 0
                    && part.length as u64 == piece.end - piece.start
                    && part.sink.as_ptr() as usize % alignment == 0
            }
            _ => false,
        }
    }

    /// The bytes from `start` (aligned) into `into` (an aligned length),
    /// whole (the tail of the file short of its end). An uncached read the
    /// filesystem refuses direct is read again through the cache, as every
    /// later one is, and the pages it filled are dropped.
    pub(crate) fn read_block(&self, start: u64, into: &mut [u8]) -> Result<(), Error> {
        let cached = match &self.through {
            Through::Cache(file) => return self.read_block_from(file, start, into, false),
            Through::Dropped(file) => file,
            Through::Direct { file, fallback } => match fallback.get() {
                Some(cached) => cached,
                None => match self.read_block_from(file, start, into, true) {
                    Err(Error::Io(error)) if refuses_direct(&error) => {
                        let file = reopen(file, false)?;
                        // A concurrent read may have fallen back first:
                        // either file reads the same bytes.
                        fallback.get_or_init(|| file)
                    }
                    result => return result,
                },
            },
        };
        self.read_block_from(cached, start, into, false)?;
        drop_pages(cached, start, (start + into.len() as u64).min(self.size))?;
        Ok(())
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
            Err((error, done)) if direct && done as u64 % ALIGNMENT != 0 => Err(io::Error::new(
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

/// A file that ended short of the bytes it held when it was opened.
pub(crate) fn truncated() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "open file changed or was truncated")
}
