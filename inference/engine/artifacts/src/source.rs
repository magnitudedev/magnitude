use crate::Error;
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::Path,
};

/// An open view of a regular artifact file.
///
/// Reads remain attached to this open file if its pathname is later replaced.
/// Callers must keep the file's contents unchanged while this source is in use.
///
/// Every read of it is a request of its volume's link
/// (`seismic_io::Reader`), around the file cache: a model's bytes, read
/// once into what holds them, would otherwise stay in the cache as memory
/// taken from free pages, which macOS reclaims only beside other programs'
/// anonymous pages.
pub struct FileSource {
    opened: seismic_io::OpenFile,
    reader: seismic_io::Reader,
}

impl std::fmt::Debug for FileSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("FileSource").field("path", &self.path()).finish()
    }
}

impl FileSource {
    /// The source keeps the name `path` gives it, made absolute but not
    /// resolved through links: a content-addressed store names its files by
    /// links, and a split GGUF finds its shards by that name.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let opened = seismic_io::OpenFile::open(path)?;
        let reader = seismic_io::Reader::of(&opened, seismic_io::CachePolicy::Uncached)?;
        Ok(Self { opened, reader })
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

    pub fn read_into(&self, offset: u64, out: &mut [u8]) -> Result<(), Error> {
        let end = offset
            .checked_add(out.len() as u64)
            .ok_or_else(|| Error::Invalid("artifact read overflows".into()))?;
        if end > self.size() {
            return Err(Error::Invalid("read outside artifact".into()));
        }
        // A caller waits on the bytes: due now.
        Ok(self.reader.read_at([(offset, out)], seismic_io::Want::now())?)
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

    /// Read the `length` bytes from `offset` in order for `want`, handing
    /// each chunk to `sink` with its offset in the
    /// range as it lands (`seismic_io::Reader::stream`): the link reads the
    /// range at its depth, and no more of the file.
    pub fn stream<E: From<io::Error>>(
        &self,
        offset: u64,
        length: u64,
        want: seismic_io::Want,
        sink: impl FnMut(u64, &[u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        if offset.checked_add(length).is_none_or(|end| end > self.size()) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "read outside artifact").into());
        }
        self.reader.stream(offset, length, want, sink)
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
