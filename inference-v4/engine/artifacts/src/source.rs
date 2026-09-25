use crate::Error;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(unix)]
use std::os::fd::AsRawFd;

/// A page-aligned, read-only window of one open artifact. The page-rounded
/// mapping is exposed separately from the requested byte range so a device
/// can wrap the mapping and address a tensor within it.
#[cfg(unix)]
pub struct MappedWindow {
    _source: Arc<FileSource>,
    pointer: std::ptr::NonNull<u8>,
    mapped_len: usize,
    data_offset: usize,
    data_len: usize,
}

#[cfg(unix)]
unsafe impl Send for MappedWindow {}
#[cfg(unix)]
unsafe impl Sync for MappedWindow {}

#[cfg(unix)]
impl MappedWindow {
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }
    pub fn data_len(&self) -> usize {
        self.data_len
    }
    pub fn mapped_len(&self) -> usize {
        self.mapped_len
    }
    pub fn data(&self) -> &[u8] {
        &self.as_ref()[self.data_offset..self.data_offset + self.data_len]
    }
}

#[cfg(unix)]
impl AsRef<[u8]> for MappedWindow {
    fn as_ref(&self) -> &[u8] {
        // The mapping is read-only, stays live through this owner, and its
        // page-rounded length was checked against the host address domain.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.mapped_len) }
    }
}

#[cfg(unix)]
impl Drop for MappedWindow {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.mapped_len);
        }
    }
}

/// An open view of a regular artifact file.
///
/// Reads remain attached to this open file if its pathname is later replaced.
/// Callers must keep the file's contents unchanged while this source is in use.
#[derive(Debug)]
pub struct FileSource {
    file: File,
    path: PathBuf,
    size: u64,
}

impl FileSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref().canonicalize()?;
        let file = File::open(&path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(Error::Invalid(
                "artifact source is not a regular file".into(),
            ));
        }
        Ok(Self {
            file,
            path,
            size: metadata.len(),
        })
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn metadata(&self) -> Result<std::fs::Metadata, Error> {
        Ok(self.file.metadata()?)
    }

    /// Map a nonempty source range read-only. The mapping owns this open file
    /// and may outlive path replacement or the package that created it.
    #[cfg(unix)]
    pub fn map_window(
        self: &Arc<Self>,
        offset: u64,
        length: u64,
    ) -> Result<Arc<MappedWindow>, Error> {
        let end = offset
            .checked_add(length)
            .filter(|end| length != 0 && *end <= self.size)
            .ok_or_else(|| {
                Error::Invalid("mapped artifact range is empty or outside the file".into())
            })?;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = u64::try_from(page)
            .map_err(|_| Error::Invalid("host page size is unavailable".into()))?;
        let mapped_start = offset / page * page;
        let mapped_end = end
            .checked_add(page - 1)
            .map(|end| end / page * page)
            .ok_or_else(|| Error::Invalid("mapped artifact range overflows".into()))?;
        let mapped_len = usize::try_from(mapped_end - mapped_start).map_err(|_| {
            Error::Invalid("mapped artifact range exceeds host address space".into())
        })?;
        let file_offset = libc::off_t::try_from(mapped_start).map_err(|_| {
            Error::Invalid("mapped artifact offset exceeds host file domain".into())
        })?;
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                self.file.as_raw_fd(),
                file_offset,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(io::Error::last_os_error().into());
        }
        let Some(pointer) = std::ptr::NonNull::new(pointer.cast()) else {
            unsafe { libc::munmap(std::ptr::null_mut(), mapped_len) };
            return Err(Error::Invalid(
                "host mapping returned a null address".into(),
            ));
        };
        Ok(Arc::new(MappedWindow {
            _source: self.clone(),
            pointer,
            mapped_len,
            data_offset: usize::try_from(offset - mapped_start)
                .expect("offset lies within mapped range"),
            data_len: usize::try_from(length).expect("mapped length fits host address space"),
        }))
    }

    pub fn read_into(&self, offset: u64, out: &mut [u8]) -> Result<(), Error> {
        let end = offset
            .checked_add(out.len() as u64)
            .ok_or_else(|| Error::Invalid("artifact read overflows".into()))?;
        if end > self.size {
            return Err(Error::Invalid("read outside artifact".into()));
        }
        let mut done = 0;
        while done < out.len() {
            #[cfg(unix)]
            let result = self.file.read_at(&mut out[done..], offset + done as u64);
            #[cfg(windows)]
            let result = self.file.seek_read(&mut out[done..], offset + done as u64);
            match result {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "open artifact changed or was truncated",
                    )
                    .into())
                }
                Ok(n) => done += n,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    pub fn read(&self, offset: u64, length: usize) -> Result<Vec<u8>, Error> {
        if offset
            .checked_add(length as u64)
            .is_none_or(|end| end > self.size)
        {
            return Err(Error::Invalid("read outside artifact".into()));
        }
        let mut data = Vec::new();
        data.try_reserve_exact(length)
            .map_err(|_| Error::Invalid("cannot allocate artifact read".into()))?;
        data.resize(length, 0);
        self.read_into(offset, &mut data)?;
        Ok(data)
    }

    pub fn reader(&self) -> SourceReader<'_> {
        SourceReader {
            source: self,
            offset: 0,
        }
    }
}

pub struct SourceReader<'a> {
    source: &'a FileSource,
    offset: u64,
}

impl Read for SourceReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let count = (out.len() as u64).min(self.source.size.saturating_sub(self.offset)) as usize;
        if count == 0 {
            return Ok(0);
        }
        self.source
            .read_into(self.offset, &mut out[..count])
            .map_err(io::Error::other)?;
        self.offset += count as u64;
        Ok(count)
    }
}

impl Seek for SourceReader<'_> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let next = match from {
            SeekFrom::Start(offset) => i128::from(offset),
            SeekFrom::Current(offset) => i128::from(self.offset) + i128::from(offset),
            SeekFrom::End(offset) => i128::from(self.source.size) + i128::from(offset),
        };
        self.offset = u64::try_from(next)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid artifact seek"))?;
        Ok(self.offset)
    }
}
