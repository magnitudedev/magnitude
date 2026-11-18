//! A bounded, atomic JSON file store a host can keep records in
//! ([`JsonFileStore`]): one file per key, named by the digest of the key's
//! JSON, in the directory of its value's schema ([`Stored`]).

use crate::measure::{Record, RecordStore, StoreError};
use serde::de::DeserializeOwned;
use serde::Serialize;
use sha2::{Digest, Sha256};
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
/// directory of its value's schema, kept in a [`seismic_io::Store`] at the
/// store's directory. A load reads without a lock (every value is published
/// whole); every store, removal and update of a key holds the key's record
/// lock ([`seismic_io::Store::lock`]), in this process or another, so no
/// write of a key falls inside another's update. A read is bounded by the
/// most bytes a value of its key can be written as, which its caller states.
/// A file that cannot be read, or holds no value of its schema, is an error.
pub struct JsonFileStore {
    store: seismic_io::Store,
}

fn error(error: impl std::fmt::Display) -> StoreError {
    StoreError(error.to_string())
}

/// The value `bytes` read from `path` hold.
fn value<V: DeserializeOwned>(path: &Path, bytes: &[u8]) -> Result<V, StoreError> {
    serde_json::from_slice(bytes).map_err(|cause| error(format!("{}: {cause}", path.display())))
}

impl JsonFileStore {
    /// The values kept in `directory`, created on the first store.
    pub fn new(directory: &Path) -> Self {
        Self { store: seismic_io::Store::new(directory.to_owned(), seismic_io::Privacy::Shared) }
    }

    /// The directory values of `V` lie in, under the store's.
    fn schema<V: Stored>() -> PathBuf {
        assert!(
            V::SCHEMA.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'+'))
                && !V::SCHEMA.starts_with('.'),
            "a stored schema names a directory: {:?}",
            V::SCHEMA
        );
        PathBuf::from(V::SCHEMA)
    }

    /// Where `key`'s value of `V` lies, under the store's directory.
    fn path<V: Stored>(&self, key: &impl Serialize) -> Result<PathBuf, StoreError> {
        let key = serde_json::to_vec(key).map_err(error)?;
        Ok(Self::schema::<V>().join(format!("{:x}.json", Sha256::digest(key))))
    }

    /// What failed at `path` under the store's directory.
    fn failed(&self, path: &Path) -> impl Fn(std::io::Error) -> StoreError + '_ {
        let path = self.store.root().join(path);
        move |cause| error(format!("{}: {cause}", path.display()))
    }

    /// The value stored under `key`, none when none is.
    pub fn load<V: Stored>(&self, key: &impl Serialize, largest: u64) -> Result<Option<V>, StoreError> {
        let path = self.path::<V>(key)?;
        let bytes = self.store.read(&path, largest).map_err(self.failed(&path))?;
        bytes.map(|bytes| value(&self.store.root().join(&path), &bytes)).transpose()
    }

    /// Store `value` under `key`, atomically replacing what was stored
    /// there, the key's lock held; a value serializing larger than
    /// `largest`, the most [`Self::load`] of `key` reads, is an error and
    /// stores nothing.
    pub fn store<V: Stored>(&self, key: &impl Serialize, value: &V, largest: u64) -> Result<(), StoreError> {
        let path = self.path::<V>(key)?;
        let bytes = Self::encoded(&self.store.root().join(&path), value, largest)?;
        let record = self.store.lock(&path).map_err(self.failed(&path))?;
        record.write(&bytes, seismic_io::Durability::Durable).map_err(self.failed(&path))
    }

    /// `value` encoded, refusing a value larger than the `largest` bytes a
    /// read of `path` accepts.
    fn encoded<V: Stored>(path: &Path, value: &V, largest: u64) -> Result<Vec<u8>, StoreError> {
        let bytes = serde_json::to_vec(value).map_err(error)?;
        if bytes.len() as u64 > largest {
            return Err(error(format!(
                "{}: a value of {} bytes is larger than the {largest} bytes its key is read to",
                path.display(),
                bytes.len()
            )));
        }
        Ok(bytes)
    }

    /// Every value stored, each read bounded by `largest` as [`Self::load`]
    /// reads it, in no particular order; none where nothing was stored. A
    /// file that cannot be read, or holds no value, is an error.
    pub fn values<V: Stored>(&self, largest: u64) -> Result<Vec<V>, StoreError> {
        let directory = Self::schema::<V>();
        let mut values = Vec::new();
        for name in self.store.list(&directory).map_err(self.failed(&directory))? {
            if !name.ends_with(".json") {
                continue;
            }
            let path = directory.join(name);
            // None where it was removed since it was listed.
            if let Some(bytes) = self.store.read(&path, largest).map_err(self.failed(&path))? {
                values.push(value(&self.store.root().join(&path), &bytes)?);
            }
        }
        Ok(values)
    }

    /// Remove the value of `V` under `key`, the key's lock held; nothing
    /// where none is stored.
    pub fn remove<V: Stored>(&self, key: &impl Serialize) -> Result<(), StoreError> {
        let path = self.path::<V>(key)?;
        let record = self.store.lock(&path).map_err(self.failed(&path))?;
        record.remove().map_err(self.failed(&path))
    }

    /// Replace the value under `key` by `update` of it (none when none is
    /// stored, read bounded by `largest` as [`Self::load`] reads), the
    /// key's lock held from the read until the new value is in place, so
    /// every other store, removal or update of `key`, from this or another
    /// process, waits. `update` gives the new value, or none to remove the
    /// value, or why it refused, which leaves the value as it is; a new
    /// value larger than `largest` is an error and stores nothing, so what
    /// is stored always loads again. `update` must not reach this store's
    /// `key`: it would wait for itself.
    pub fn update<V: Stored, E>(
        &self,
        key: &impl Serialize,
        largest: u64,
        update: impl FnOnce(Option<V>) -> Result<Option<V>, E>,
    ) -> Result<Result<(), E>, StoreError> {
        let path = self.path::<V>(key)?;
        let shown = self.store.root().join(&path);
        let record = self.store.lock(&path).map_err(self.failed(&path))?;
        let held = record.read(largest).map_err(self.failed(&path))?.map(|bytes| value(&shown, &bytes)).transpose()?;
        match update(held) {
            Err(refused) => Ok(Err(refused)),
            Ok(None) => record.remove().map(Ok).map_err(self.failed(&path)),
            Ok(Some(value)) => {
                let bytes = Self::encoded(&shown, &value, largest)?;
                record.write(&bytes, seismic_io::Durability::Durable).map(Ok).map_err(self.failed(&path))
            }
        }
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
        store.update(&"other", 64, |held: Option<Count>| Ok::<_, ()>(Some(Count(held.map_or(2, |held| held.0 + 1))))).unwrap().unwrap();
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
        let path = directory.path().join(store.path::<Record>(&key).unwrap());
        std::fs::write(&path, vec![b' '; Record::largest() as usize + 1]).unwrap();
        assert!(RecordStore::load(&store, &key).is_err());
        std::fs::write(&path, b"{").unwrap();
        assert!(RecordStore::load(&store, &key).is_err());
        assert!(store.values::<Record>(Record::largest()).is_err());
    }

    /// Writers updating one key at once each read what the others wrote:
    /// every one of their additions is kept; the key's lock is no value; a
    /// refused update stores nothing; an update to none removes the value,
    /// and its lock with it.
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
                        let added = store.update(&"sum", 32, |sum: Option<Count>| Ok::<_, ()>(Some(Count(sum.map_or(0, |sum| sum.0) + 1))));
                        assert_eq!(added.unwrap(), Ok(()));
                    }
                });
            }
        });
        let store = JsonFileStore::new(directory.path());
        assert_eq!(store.load::<Count>(&"sum", 32).unwrap(), Some(Count(writers * additions)));
        assert_eq!(store.values::<Count>(32).unwrap(), [Count(writers * additions)]);
        assert_eq!(store.update(&"sum", 32, |_: Option<Count>| Err("refused")).unwrap(), Err("refused"));
        assert_eq!(store.load::<Count>(&"sum", 32).unwrap(), Some(Count(writers * additions)));
        assert_eq!(store.update(&"sum", 32, |_: Option<Count>| Ok::<_, ()>(None)).unwrap(), Ok(()));
        assert_eq!(store.load::<Count>(&"sum", 32).unwrap(), None);
        assert!(!directory.path().join(".locks").join(store.path::<Count>(&"sum").unwrap()).exists(), "a removed value's lock goes with it");
    }

    /// A store and a removal of a key wait for an update of it under way:
    /// neither falls between the update's read and its write, so each
    /// lands after it, whichever store (as another process holds) they go
    /// through.
    #[test]
    fn a_store_or_removal_waits_for_an_update_under_way() {
        let directory = tempfile::tempdir().unwrap();
        for removing in [false, true] {
            let (inside, entered) = std::sync::mpsc::channel();
            let (release, released) = std::sync::mpsc::channel::<()>();
            let landed = std::sync::atomic::AtomicBool::new(false);
            let (root, landed) = (directory.path(), &landed);
            std::thread::scope(|scope| {
                // Moved in: a receiver is no thread's to share.
                let updating = scope.spawn(move || {
                    JsonFileStore::new(root)
                        .update(&"key", 32, |held: Option<Count>| {
                            inside.send(()).unwrap();
                            released.recv().unwrap();
                            Ok::<_, ()>(Some(Count(held.map_or(1, |held| held.0 + 1))))
                        })
                        .unwrap()
                        .unwrap();
                });
                entered.recv().unwrap();
                let writing = scope.spawn(move || {
                    let store = JsonFileStore::new(root);
                    match removing {
                        true => store.remove::<Count>(&"key").unwrap(),
                        false => store.store(&"key", &Count(100), 32).unwrap(),
                    }
                    landed.store(true, std::sync::atomic::Ordering::SeqCst);
                });
                std::thread::sleep(std::time::Duration::from_millis(200));
                assert!(!landed.load(std::sync::atomic::Ordering::SeqCst), "nothing lands inside an update");
                release.send(()).unwrap();
                updating.join().unwrap();
                writing.join().unwrap();
            });
            let found = JsonFileStore::new(directory.path()).load::<Count>(&"key", 32).unwrap();
            assert_eq!(found, (!removing).then_some(Count(100)), "the write that waited lands last");
        }
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
        assert!(store.update(&"key", 6, |_: Option<Text>| Ok::<_, ()>(Some(Text("too long".into())))).is_err());
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
