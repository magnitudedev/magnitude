use crate::Error;
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

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

    pub fn digest(&self) -> Result<crate::ArtifactIdentity, Error> {
        let mut digest = Sha256::new();
        let mut reader = self.reader();
        let mut buffer = vec![0; 8 * 1024 * 1024];
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        Ok(crate::ArtifactIdentity(digest.finalize().into()))
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
