//! A volume's read line as the measurement job takes it: `λ + b/D`, a read
//! of `b` bytes, from two kinds of read of a probe file the job keeps on the
//! volume ([`StorageProbe`]), past the host's file cache:
//!
//! - **lone reads** ([`StorageProbe::lone`]): one [`LONE_READ`] at a time,
//!   at random aligned offsets: `λ + LONE_READ/D`, a demand request's;
//! - **full reads** ([`StorageProbe::full`]): [`FULL_DEPTH`] readers each
//!   reading [`FULL_READ`] at random aligned offsets, the volume's widest
//!   probed window in flight ([`WIDEST_WINDOW`]): `D`, its full rate.
//!
//! Both are of the volume, not of any model: the file is the job's own, so
//! no model's file is read or evicted to time them, and a volume that holds
//! no model yet is measured as one that does.

use super::file::SourceFile;
use super::platform::{evict, uncached};
use super::probe::{LEAST_READ, WIDEST_WINDOW};
use super::ALIGNMENT;
use crate::{Buffer, Error};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The bytes of the probe file: 256 MiB, far past what a lone or a full
/// read touches, so random offsets rarely meet, and written once.
pub const STORAGE_PROBE_BYTES: u64 = 256 << 20;

/// A lone read's bytes: the least read size a volume is read in.
pub const LONE_READ: u64 = LEAST_READ;

/// A full read's bytes and its readers: the volume's widest probed window
/// in flight, where every measured device reaches its rate.
pub const FULL_READ: u64 = 1 << 20;
pub const FULL_DEPTH: u64 = WIDEST_WINDOW / FULL_READ;

/// The probe file the measurement job reads a volume through.
pub struct StorageProbe {
    file: SourceFile,
    state: Mutex<u64>,
}

impl StorageProbe {
    /// The probe file in `directory`, written there first where it is not
    /// whole, its pages then dropped from the host's file cache.
    pub fn open(directory: &Path) -> Result<Self, Error> {
        let path = Self::path(directory);
        if std::fs::metadata(&path).map_or(true, |metadata| metadata.len() != STORAGE_PROBE_BYTES) {
            Self::write(directory, &path)?;
        }
        let file = std::fs::File::open(&path)?;
        evict(&file, STORAGE_PROBE_BYTES)?;
        let file = SourceFile { path: path.as_path().into(), size: STORAGE_PROBE_BYTES, through: uncached(&file)? };
        Ok(Self { file, state: Mutex::new(0x9e37_79b9_7f4a_7c15) })
    }

    /// Where the probe file of `directory` lies.
    pub fn path(directory: &Path) -> PathBuf {
        directory.join(".storage-probe")
    }

    /// Write the probe file: bytes no volume stores as a hole or compresses
    /// (a xorshift stream), into a file beside it renamed into place, so a
    /// job killed while writing leaves no short probe file.
    fn write(directory: &Path, path: &Path) -> Result<(), Error> {
        std::fs::create_dir_all(directory)?;
        let partial = path.with_extension("partial");
        let mut file = std::fs::File::create(&partial)?;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut chunk = vec![0u8; 8 << 20];
        for _ in 0..STORAGE_PROBE_BYTES / chunk.len() as u64 {
            for word in chunk.chunks_exact_mut(8) {
                state = next(state);
                word.copy_from_slice(&state.to_le_bytes());
            }
            file.write_all(&chunk)?;
        }
        file.sync_all()?;
        std::fs::rename(&partial, path)?;
        Ok(())
    }

    /// `passes` lone reads, one after another.
    pub fn lone(&self, passes: usize) -> Result<(), Error> {
        let mut buffer = Buffer::new(LONE_READ as usize);
        for _ in 0..passes {
            let at = self.offset(LONE_READ);
            self.file.read_block(at, buffer.bytes_mut())?;
        }
        Ok(())
    }

    /// `passes` full reads: [`FULL_DEPTH`] readers, each reading `passes`
    /// reads of [`FULL_READ`], one after another.
    pub fn full(&self, passes: usize) -> Result<(), Error> {
        let offsets = (0..FULL_DEPTH).map(|_| (0..passes).map(|_| self.offset(FULL_READ)).collect::<Vec<_>>()).collect::<Vec<_>>();
        std::thread::scope(|scope| {
            let readers = offsets
                .iter()
                .map(|offsets| {
                    std::thread::Builder::new().spawn_scoped(scope, move || {
                        let mut buffer = Buffer::new(FULL_READ as usize);
                        offsets.iter().try_for_each(|&at| self.file.read_block(at, buffer.bytes_mut()))
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(Error::ThreadStart)?;
            readers.into_iter().try_for_each(|reader| reader.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
        })
    }

    /// A random aligned offset a read of `length` bytes lies within the
    /// file from.
    fn offset(&self, length: u64) -> u64 {
        let mut state = self.state.lock().expect("the probe's offsets");
        *state = next(*state);
        *state % ((STORAGE_PROBE_BYTES - length) / ALIGNMENT + 1) * ALIGNMENT
    }
}

/// The xorshift step after `state`.
fn next(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}
