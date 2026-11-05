//! A bounded, atomic JSON file store a host can keep records in
//! ([`JsonFileStore`]): one file per key, named by the digest of the key's
//! JSON, in the directory of its value's schema ([`Stored`]).

use crate::measure::{Record, RecordStore, StoreError};
use serde::de::DeserializeOwned;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// A value a [`JsonFileStore`] keeps. Its schema names the directory its
/// values lie in, so a value written in another form, by another build, is
/// never read as one: a file there that does not decode, or is larger than
/// any value, is a broken store. A change to how the type serializes takes a
/// new schema.
pub trait Stored: Serialize + DeserializeOwned {
    /// The schema's name, which names a directory: letters, digits, `-`,
    /// `.` and `+` only.
    const SCHEMA: &'static str;
}

/// One JSON file per key, named by the digest of the key's JSON, in the
/// directory of its value's schema. Writes are atomic; a read is bounded by
/// the most bytes a value of its key can be written as, which its caller
/// states. A file that cannot be read, or holds no value of its schema, is
/// an error.
pub struct JsonFileStore {
    directory: PathBuf,
}

fn error(error: impl std::fmt::Display) -> StoreError {
    StoreError(error.to_string())
}

/// The value `bytes` read from `path` hold, read bounded by `largest`.
fn value<V: DeserializeOwned>(path: &Path, bytes: &[u8], largest: u64) -> Result<V, StoreError> {
    if bytes.len() as u64 > largest {
        return Err(error(format!("{}: larger than the {largest} bytes any value of it is", path.display())));
    }
    serde_json::from_slice(bytes).map_err(|cause| error(format!("{}: {cause}", path.display())))
}

impl JsonFileStore {
    /// The values kept in `directory`, created on the first store.
    pub fn new(directory: &Path) -> Self {
        Self {
            directory: directory.to_owned(),
        }
    }

    /// The directory values of `V` lie in.
    fn schema<V: Stored>(&self) -> PathBuf {
        assert!(
            V::SCHEMA.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'+'))
                && !V::SCHEMA.starts_with('.'),
            "a stored schema names a directory: {:?}",
            V::SCHEMA
        );
        self.directory.join(V::SCHEMA)
    }

    fn path<V: Stored>(&self, key: &impl Serialize) -> Result<PathBuf, StoreError> {
        let key = serde_json::to_vec(key).map_err(error)?;
        Ok(self.schema::<V>().join(format!("{:x}.json", Sha256::digest(key))))
    }

    /// The bytes at `path`, none when there are none, read to at most
    /// `largest` + 1 bytes.
    fn bytes(path: &Path, largest: u64) -> Result<Option<Vec<u8>>, StoreError> {
        let failed = |cause: std::io::Error| error(format!("{}: {cause}", path.display()));
        let file = match std::fs::File::open(path) {
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

    /// The value stored under `key`, none when none is.
    pub fn load<V: Stored>(
        &self,
        key: &impl Serialize,
        largest: u64,
    ) -> Result<Option<V>, StoreError> {
        let path = self.path::<V>(key)?;
        Self::bytes(&path, largest)?.map(|bytes| value(&path, &bytes, largest)).transpose()
    }

    /// Store `value` under `key`, atomically replacing what was stored
    /// there; a value serializing larger than `largest`, the most
    /// [`Self::load`] of `key` reads, is an error and stores nothing.
    pub fn store<V: Stored>(&self, key: &impl Serialize, value: &V, largest: u64) -> Result<(), StoreError> {
        std::fs::create_dir_all(self.schema::<V>()).map_err(error)?;
        self.write(&self.path::<V>(key)?, value, largest)
    }

    /// Write `value` to `path` atomically, refusing, with nothing written,
    /// a value larger than the `largest` bytes a read of `path` accepts.
    fn write<V: Stored>(&self, path: &Path, value: &V, largest: u64) -> Result<(), StoreError> {
        let bytes = serde_json::to_vec(value).map_err(error)?;
        if bytes.len() as u64 > largest {
            return Err(error(format!(
                "{}: a value of {} bytes is larger than the {largest} bytes its key is read to",
                path.display(),
                bytes.len()
            )));
        }
        let mut temporary = tempfile::NamedTempFile::new_in(self.schema::<V>()).map_err(error)?;
        temporary.write_all(&bytes).map_err(error)?;
        temporary.as_file().sync_all().map_err(error)?;
        temporary.persist(path).map_err(error)?;
        Ok(())
    }

    /// Every value stored, each read bounded by `largest` as [`Self::load`]
    /// reads it, in no particular order; none where nothing was stored. A
    /// file that cannot be read, or holds no value, is an error.
    pub fn values<V: Stored>(&self, largest: u64) -> Result<Vec<V>, StoreError> {
        let directory = self.schema::<V>();
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(cause) if cause.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(cause) => return Err(error(format!("{}: {cause}", directory.display()))),
        };
        let mut values = Vec::new();
        for entry in entries {
            let path = entry.map_err(error)?.path();
            if path.extension().is_none_or(|extension| extension != "json") {
                continue;
            }
            // None where it was removed since it was listed.
            if let Some(bytes) = Self::bytes(&path, largest)? {
                values.push(value(&path, &bytes, largest)?);
            }
        }
        Ok(values)
    }

    /// Remove the value of `V` under `key`; nothing where none is stored.
    pub fn remove<V: Stored>(&self, key: &impl Serialize) -> Result<(), StoreError> {
        let path = self.path::<V>(key)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(cause) if cause.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(cause) => Err(error(format!("{}: {cause}", path.display()))),
        }
    }

    /// Replace the value under `key` by `update` of it (none when none is
    /// stored, read bounded by `largest` as [`Self::load`] reads), every
    /// other update of `key` from this or another process waiting: an
    /// exclusive lock on the key's lock file is held from the read until
    /// the new value is renamed into place. Why `update` refused, which
    /// stores nothing; a new value larger than `largest` is an error and
    /// stores nothing, so what is stored always loads again.
    pub fn update<V: Stored, E>(
        &self,
        key: &impl Serialize,
        largest: u64,
        update: impl FnOnce(Option<V>) -> Result<V, E>,
    ) -> Result<Result<(), E>, StoreError> {
        std::fs::create_dir_all(self.schema::<V>()).map_err(error)?;
        let path = self.path::<V>(key)?;
        let locked = path.with_extension("lock");
        let failed = |cause: std::io::Error| error(format!("{}: {cause}", locked.display()));
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&locked)
            .map_err(failed)?;
        lock.lock().map_err(failed)?;
        let value = match update(self.load(key, largest)?) {
            Ok(value) => value,
            Err(refused) => return Ok(Err(refused)),
        };
        self.write(&path, &value, largest)?;
        // The lock is released as `lock` closes.
        Ok(Ok(()))
    }
}

impl Stored for Record {
    const SCHEMA: &'static str = "record.1";
}

/// Records under serializable keys, each read bounded by
/// [`Record::largest`].
impl<K: Serialize> RecordStore<K> for JsonFileStore {
    fn load(&self, key: &K) -> Result<Option<Record>, StoreError> {
        JsonFileStore::load(self, key, Record::largest())
    }

    fn store(&self, key: &K, record: &Record) -> Result<(), StoreError> {
        JsonFileStore::store(self, key, record, Record::largest())
    }
}

/// Records held for as long as the store lives, for a host that keeps none
/// beyond it: each under its key's JSON.
#[derive(Default)]
pub struct HeldStore(std::cell::RefCell<std::collections::HashMap<Vec<u8>, Record>>);

impl<K: Serialize> RecordStore<K> for HeldStore {
    fn load(&self, key: &K) -> Result<Option<Record>, StoreError> {
        Ok(self.0.borrow().get(&serde_json::to_vec(key).map_err(error)?).cloned())
    }

    fn store(&self, key: &K, record: &Record) -> Result<(), StoreError> {
        self.0.borrow_mut().insert(serde_json::to_vec(key).map_err(error)?, record.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::Reason;
    use crate::statistics::Timing;

    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Count(u64);
    impl Stored for Count {
        const SCHEMA: &'static str = "count.1";
    }

    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Text(String);
    impl Stored for Text {
        const SCHEMA: &'static str = "text.1";
    }

    /// A value of another schema lies apart, never read as one: a load
    /// finds none and a listing leaves it out.
    #[test]
    fn a_value_of_another_schema_is_never_read() {
        let directory = tempfile::tempdir().unwrap();
        let store = JsonFileStore::new(directory.path());
        store.store(&"kept", &Count(1), 64).unwrap();
        store.store(&"other", &Text("another form".into()), 64).unwrap();
        assert_eq!(store.load::<Count>(&"other", 64).unwrap(), None);
        assert_eq!(store.values::<Count>(64).unwrap(), [Count(1)]);
        store.update(&"other", 64, |held: Option<Count>| Ok::<_, ()>(Count(held.map_or(2, |held| held.0 + 1)))).unwrap().unwrap();
        assert_eq!(store.load::<Count>(&"other", 64).unwrap(), Some(Count(2)));
    }

    /// A stored record loads back; a file of its schema larger than any
    /// record, or that does not decode as one, is a broken store; the
    /// widest records fit the bound.
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
        let path = store.path::<Record>(&key).unwrap();
        std::fs::write(&path, vec![b' '; Record::largest() as usize + 1]).unwrap();
        assert!(RecordStore::load(&store, &key).is_err());
        std::fs::write(&path, b"{").unwrap();
        assert!(RecordStore::load(&store, &key).is_err());
        assert!(store.values::<Record>(Record::largest()).is_err());
    }

    /// Writers updating one key at once each read what the others wrote:
    /// every one of their additions is kept; the lock file is no value; a
    /// refused update stores nothing.
    #[test]
    fn concurrent_updates_of_a_key_each_keep_their_addition() {
        let directory = tempfile::tempdir().unwrap();
        let (writers, additions) = (2, 200u64);
        std::thread::scope(|scope| {
            for _ in 0..writers {
                scope.spawn(|| {
                    // A store of its own, as another process holds.
                    let store = JsonFileStore::new(directory.path());
                    for _ in 0..additions {
                        let added = store.update(&"sum", 32, |sum: Option<Count>| Ok::<_, ()>(Count(sum.map_or(0, |sum| sum.0) + 1)));
                        assert_eq!(added.unwrap(), Ok(()));
                    }
                });
            }
        });
        let store = JsonFileStore::new(directory.path());
        assert_eq!(store.load::<Count>(&"sum", 32).unwrap(), Some(Count(writers * additions)));
        assert_eq!(store.update(&"sum", 32, |_: Option<Count>| Err("refused")).unwrap(), Err("refused"));
        assert_eq!(store.load::<Count>(&"sum", 32).unwrap(), Some(Count(writers * additions)));
    }

    /// A value larger than its key is read to is refused by a store and by
    /// an update, which write nothing: what was stored still loads.
    #[test]
    fn a_value_larger_than_its_key_is_read_to_is_not_written() {
        let directory = tempfile::tempdir().unwrap();
        let store = JsonFileStore::new(directory.path());
        store.store(&"key", &Text("fits".into()), 6).unwrap();
        assert!(store.store(&"key", &Text("too long".into()), 6).is_err());
        assert_eq!(store.load::<Text>(&"key", 6).unwrap(), Some(Text("fits".into())));
        assert!(store.update(&"key", 6, |_: Option<Text>| Ok::<_, ()>(Text("too long".into()))).is_err());
        assert_eq!(store.load::<Text>(&"key", 6).unwrap(), Some(Text("fits".into())));
    }

    /// A reason is cut to its bound at a character boundary.
    #[test]
    fn a_reason_is_cut_at_a_character_boundary() {
        let reason = Reason::new("é".repeat(Reason::BYTES));
        assert!(reason.as_str().len() <= Reason::BYTES);
        assert!(reason.as_str().len() > Reason::BYTES - 2);
    }
}
