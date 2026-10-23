//! A bounded, atomic JSON file store a host can keep records in
//! ([`JsonFileStore`]): one file per key, named by the digest of the key's
//! JSON.

use crate::measure::{Record, RecordStore, StoreError};
use serde::de::DeserializeOwned;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// One JSON file per key in a directory, named by the digest of the key's
/// JSON. Writes are atomic; a read is bounded by the most bytes a value of
/// its key can be written as, which its caller states; every failure but
/// an absent file is an error.
pub struct JsonFileStore {
    directory: PathBuf,
}

fn error(error: impl std::fmt::Display) -> StoreError {
    StoreError(error.to_string())
}

impl JsonFileStore {
    /// The values kept in `directory`, created on the first store.
    pub fn new(directory: &Path) -> Self {
        Self {
            directory: directory.to_owned(),
        }
    }

    fn path(&self, key: &impl Serialize) -> Result<PathBuf, StoreError> {
        let key = serde_json::to_vec(key).map_err(error)?;
        Ok(self
            .directory
            .join(format!("{:x}.json", Sha256::digest(key))))
    }

    /// The bytes stored under `key`, none when none is, read to at most
    /// `largest` + 1 bytes; every failure but an absent file is an error.
    fn bytes(&self, key: &impl Serialize, largest: u64) -> Result<Option<Vec<u8>>, StoreError> {
        let path = self.path(key)?;
        let failed = |cause: std::io::Error| error(format!("{}: {cause}", path.display()));
        let file = match std::fs::File::open(&path) {
            Ok(file) => file,
            Err(cause) if cause.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(cause) => return Err(failed(cause)),
        };
        let mut bytes = Vec::new();
        file.take(largest + 1)
            .read_to_end(&mut bytes)
            .map_err(failed)?;
        Ok(Some(bytes))
    }

    /// The value stored under `key`, none when none is; a file larger than
    /// `largest` bytes, the most a value of `key` is written as, is an
    /// error like any other unreadable one.
    pub fn load<V: DeserializeOwned>(
        &self,
        key: &impl Serialize,
        largest: u64,
    ) -> Result<Option<V>, StoreError> {
        let Some(bytes) = self.bytes(key, largest)? else {
            return Ok(None);
        };
        let path = self.path(key)?;
        if bytes.len() as u64 > largest {
            return Err(error(format!(
                "{}: larger than the {largest} bytes a value of its key is written as",
                path.display()
            )));
        }
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|cause| error(format!("{}: {cause}", path.display())))
    }

    /// Store `value` under `key`, atomically replacing what was stored
    /// there.
    pub fn store(&self, key: &impl Serialize, value: &impl Serialize) -> Result<(), StoreError> {
        std::fs::create_dir_all(&self.directory).map_err(error)?;
        let path = self.path(key)?;
        let bytes = serde_json::to_vec(value).map_err(error)?;
        let mut temporary = tempfile::NamedTempFile::new_in(&self.directory).map_err(error)?;
        temporary.write_all(&bytes).map_err(error)?;
        temporary.as_file().sync_all().map_err(error)?;
        temporary.persist(&path).map_err(error)?;
        Ok(())
    }
}

/// Records under serializable keys, each read bounded by
/// [`Record::largest`]. A record is a measurement's cache entry, which
/// losing costs only measuring its point again: a file that is no record
/// (larger than any, or not decoding as one, as a record of an earlier
/// form does not) holds none, so its point is measured again and the file
/// replaced. A file that cannot be read is an error.
impl<K: Serialize> RecordStore<K> for JsonFileStore {
    fn load(&self, key: &K) -> Result<Option<Record>, StoreError> {
        let largest = Record::largest();
        Ok(self
            .bytes(key, largest)?
            .filter(|bytes| bytes.len() as u64 <= largest)
            .and_then(|bytes| serde_json::from_slice(&bytes).ok()))
    }

    fn store(&self, key: &K, record: &Record) -> Result<(), StoreError> {
        JsonFileStore::store(self, key, record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::Reason;
    use crate::statistics::Timing;

    /// A stored record loads back; a file larger than any record, or one
    /// that does not decode as one, holds none; the widest records fit the
    /// bound.
    #[test]
    fn records_load_back_bounded_by_the_largest_record() {
        let directory = tempfile::tempdir().unwrap();
        let store = JsonFileStore::new(directory.path());
        let key = ("point", 7u32);
        assert_eq!(RecordStore::load(&store, &key).unwrap(), None);
        let widest = -f64::MIN_POSITIVE;
        let measured = Record::Measured(Timing {
            median: widest,
            interval: Some((widest, widest)),
            samples: u32::MAX,
        });
        RecordStore::store(&store, &key, &measured).unwrap();
        assert_eq!(RecordStore::load(&store, &key).unwrap(), Some(measured));
        let skipped = Record::Unobtainable {
            reason: Reason::new("\u{1}".repeat(Reason::BYTES + 10)),
        };
        RecordStore::store(&store, &key, &skipped).unwrap();
        assert_eq!(RecordStore::load(&store, &key).unwrap(), Some(skipped));
        let path = store.path(&key).unwrap();
        std::fs::write(&path, vec![b' '; Record::largest() as usize + 1]).unwrap();
        assert_eq!(RecordStore::load(&store, &key).unwrap(), None);
        std::fs::write(&path, b"{").unwrap();
        assert_eq!(RecordStore::load(&store, &key).unwrap(), None);
    }

    /// A reason is cut to its bound at a character boundary.
    #[test]
    fn a_reason_is_cut_at_a_character_boundary() {
        let reason = Reason::new("é".repeat(Reason::BYTES));
        assert!(reason.as_str().len() <= Reason::BYTES);
        assert!(reason.as_str().len() > Reason::BYTES - 2);
    }
}
