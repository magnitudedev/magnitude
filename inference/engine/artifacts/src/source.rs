use crate::Error;
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::Path,
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
    opened: seismic_io::OpenFile,
}

impl FileSource {
    /// The source keeps the name `path` gives it, made absolute but not
    /// resolved through links: a content-addressed store names its files by
    /// links, and a split GGUF finds its shards by that name.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        // Its reads fill the cache with the ranges they name, no more.
        Ok(Self { opened: seismic_io::OpenFile::open(path)? })
    }

    pub fn size(&self) -> u64 {
        self.opened.size()
    }

    pub fn path(&self) -> &Path {
        self.opened.path()
    }

    fn file(&self) -> &File {
        self.opened.file()
    }

    pub fn metadata(&self) -> Result<std::fs::Metadata, Error> {
        Ok(self.file().metadata()?)
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
            .filter(|end| length != 0 && *end <= self.size())
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
                self.file().as_raw_fd(),
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
        if end > self.size() {
            return Err(Error::Invalid("read outside artifact".into()));
        }
        Ok(self.opened.read_exact_at(offset, out)?)
    }

    pub fn read(&self, offset: u64, length: usize) -> Result<Vec<u8>, Error> {
        if offset
            .checked_add(length as u64)
            .is_none_or(|end| end > self.size())
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

    /// A reader of the `length` bytes from `offset`, whose pages the cache
    /// reads ahead of the reads as one range (on Linux, `POSIX_FADV_WILLNEED`
    /// over exactly it): the file reads ahead of no range on its own
    /// ([`FileSource::open`]), so a range read whole asks for all of it. An
    /// empty range asks for nothing.
    pub fn range(&self, offset: u64, length: u64) -> Result<io::Take<SourceReader<'_>>, Error> {
        offset
            .checked_add(length)
            .filter(|end| *end <= self.size())
            .ok_or_else(|| Error::Invalid("read outside artifact".into()))?;
        if let Some(length) = std::num::NonZeroU64::new(length) {
            self.will_need(offset, length)?;
        }
        Ok(SourceReader {
            source: self,
            offset,
        }
        .take(length))
    }

    /// Ask the cache to read the `length` bytes from `offset` ahead. Never
    /// empty: advice of length zero means "to the end of the file".
    #[cfg(target_os = "linux")]
    fn will_need(&self, offset: u64, length: std::num::NonZeroU64) -> Result<(), Error> {
        let (start, count) = (libc::off_t::try_from(offset), libc::off_t::try_from(length.get()));
        let (Ok(start), Ok(count)) = (start, count) else {
            return Err(Error::Invalid("artifact range exceeds the host file domain".into()));
        };
        // SAFETY: advice on a range of a descriptor this source owns.
        let advised = unsafe {
            libc::posix_fadvise(self.file().as_raw_fd(), start, count, libc::POSIX_FADV_WILLNEED)
        };
        if advised != 0 {
            return Err(io::Error::from_raw_os_error(advised).into());
        }
        Ok(())
    }

    /// The platform reads ahead as reads go.
    #[cfg(not(target_os = "linux"))]
    fn will_need(&self, _offset: u64, _length: std::num::NonZeroU64) -> Result<(), Error> {
        Ok(())
    }
}

/// The file a source has open, which its readers read
/// (`seismic_io::Reader::of`) and its volume's probe times.
impl AsRef<seismic_io::OpenFile> for FileSource {
    fn as_ref(&self) -> &seismic_io::OpenFile {
        &self.opened
    }
}

pub struct SourceReader<'a> {
    source: &'a FileSource,
    offset: u64,
}

impl Read for SourceReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let count = (out.len() as u64).min(self.source.size().saturating_sub(self.offset)) as usize;
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
            SeekFrom::End(offset) => i128::from(self.source.size()) + i128::from(offset),
        };
        self.offset = u64::try_from(next)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid artifact seek"))?;
        Ok(self.offset)
    }
}
