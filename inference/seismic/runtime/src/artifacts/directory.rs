//! A store kept in files under one directory, which every process that
//! opens it on the same directory shares: the user's program store
//! ([`super::ProgramStore::user`], laid out by [`programs`]), and the
//! engine's kernel cache of tuning records, laid out by its own.
//!
//! Its files are a [`seismic_io::Store`], which owns how they are
//! published, locked, read and evicted; a directory store is what Seismic
//! keeps there: the programs, as an [`ArtifactStore`], and when the store
//! evicts. Layout under the root:
//!
//! ```text
//! artifacts/<namespace>/<key>   what a device's program former stores
//!                               ([`ArtifactStore`]): each compile's artifact
//!                               and each entry's pointer, one namespace per
//!                               toolchain; `artifacts/probes` holds the
//!                               devices' probe answers
//! <directory>/<name>            an embedder's own records ([`DirectoryStore::writing`])
//! ```
//!
//! beside the files the store itself keeps ([`seismic_io::store`]).
//!
//! An embedder's records are reached only through the store's lock: held
//! shared to read them ([`DirectoryStore::reading`]), exclusively to write
//! or remove them ([`DirectoryStore::writing`]), in this process or another.
//! The artifacts take no lock to be read or written: each is a content
//! address, never written again, and every reader checks the one it reads
//! whole. One a platform opens by path is held ([`ArtifactStore::held`]):
//! eviction leaves a held file at its path while the hold lives, so the
//! platform finds it there for as long as it reads it.
//!
//! Opening the store, and every quarter of its capacity written after,
//! begins an eviction ([`seismic_io::Store::evict`]) of the least recently
//! used entries beyond its capacity, on a thread of its own, off the path
//! that opened or wrote it. Each artifact is an entry of its own; what the
//! embedder keeps beside them is placed by its layout ([`Eviction`]).
//!
//! Every artifact key is a content address, so an artifact is never
//! invalidated; any artifact that cannot be read is a miss.

use super::{ArtifactKey, ArtifactStore};
use seismic_io::{Durability, Eviction, Examining, Exclusive, Privacy, Shared, Store};
use seismic_native_target::HeldFile;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Bytes a store keeps unless its opener states otherwise.
pub const DEFAULT_CAPACITY: u64 = 1 << 30;

/// The directory under the root that holds the artifacts.
const ARTIFACTS: &str = "artifacts";

/// The layout of a store of programs: each artifact and pointer an entry of
/// its own, and the probe answers (`artifacts/probes`) kept, as forming one
/// again runs its probes.
pub fn programs(path: &Path, _: &Examining<'_>) -> Eviction {
    if path == Path::new(ARTIFACTS).join(crate::probes::NAMESPACE) {
        Eviction::Kept
    } else {
        Eviction::Alone
    }
}

/// A store in files under one directory.
#[derive(Debug)]
pub struct DirectoryStore {
    store: Arc<Store>,
    capacity: u64,
    layout: fn(&Path, &Examining<'_>) -> Eviction,
    /// The eviction under way, if one is.
    eviction: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// The embedder's records of a store, its lock held exclusively
/// ([`seismic_io::Exclusive`], which it derefs to): no reader, other writer
/// or eviction is among them until it is dropped. Dropped, the lock
/// released, it begins an eviction if what the store has written since the
/// last one is a quarter of its capacity.
#[must_use = "the records are held only while this lives"]
#[derive(Debug)]
pub struct Writing<'a> {
    directory: &'a DirectoryStore,
    /// The lock, held until this is dropped.
    held: Option<Exclusive<'a>>,
}

impl<'a> std::ops::Deref for Writing<'a> {
    type Target = Exclusive<'a>;

    fn deref(&self) -> &Exclusive<'a> {
        self.held.as_ref().expect("the lock is held until the records are dropped")
    }
}

impl Drop for Writing<'_> {
    fn drop(&mut self) {
        // Released first: the eviction it begins takes the lock.
        drop(self.held.take());
        self.directory.evict_if_outgrown();
    }
}

impl DirectoryStore {
    /// The store at `root`, created if absent, keeping `capacity` bytes of
    /// entries, the embedder's files placed by `layout`; it begins removing
    /// the least recently used beyond them.
    pub fn open(root: PathBuf, capacity: u64, layout: fn(&Path, &Examining<'_>) -> Eviction) -> std::io::Result<Self> {
        let opened = Store::open(root, Privacy::Shared).map_err(std::io::Error::other)?;
        let store = Self {
            store: Arc::new(opened),
            capacity,
            layout,
            eviction: std::sync::Mutex::new(None),
        };
        store.evict();
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        self.store.root()
    }

    /// The records, read under the store's lock held shared, waiting until
    /// no writer or eviction holds it.
    pub fn reading(&self) -> std::io::Result<Shared<'_>> {
        self.store.shared().map_err(std::io::Error::other)
    }

    /// The records, read, written and removed under the store's lock held
    /// exclusively, waiting until no other holder holds it. A thread holding
    /// the lock takes it again only to wait for itself.
    pub fn writing(&self) -> std::io::Result<Writing<'_>> {
        Ok(Writing { directory: self, held: Some(self.store.exclusive().map_err(std::io::Error::other)?) })
    }

    /// Write `bytes` under `key` of `namespace`, synced, as [`ArtifactStore::put`]
    /// does; whether they were written, which a put only reports.
    pub(crate) fn put_synced(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]) -> bool {
        self.put_reported(namespace, key, bytes, Durability::Durable)
    }

    /// Publish an artifact, reporting a failure as a warning event, which the
    /// embedder's subscriber carries to its telemetry: a missing artifact
    /// costs its compile again.
    fn put_reported(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8], durability: Durability) -> bool {
        let path = Self::artifact(namespace, key);
        let published = self.store.shared().and_then(|shared| shared.publish(&path, durability, |file| file.write_all(bytes)));
        self.evict_if_outgrown();
        match published {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!("artifact store: cannot write {}: {error}", self.root().join(path).display());
                false
            }
        }
    }

    fn artifact(namespace: &str, key: &ArtifactKey) -> PathBuf {
        Path::new(ARTIFACTS).join(namespace).join(key.as_str())
    }

    /// Begin an eviction if what the store wrote since the last one is a
    /// quarter of its capacity.
    fn evict_if_outgrown(&self) {
        if self.store.written() >= self.capacity / 4 {
            self.evict();
        }
    }

    /// Begin removing the least recently used entries beyond the capacity,
    /// on a thread of its own, unless an eviction is under way. Eviction has
    /// no caller to answer: one that finds the store held passes, and the
    /// store stays beyond its capacity until the next, a quarter of its
    /// capacity written later; a failure is reported as an error event.
    fn evict(&self) {
        let mut eviction = self.eviction.lock().expect("no eviction panics starting");
        if eviction.as_ref().is_some_and(|under_way| !under_way.is_finished()) {
            return;
        }
        let (store, capacity, layout) = (self.store.clone(), self.capacity, self.layout);
        let evicting = move || {
            if let Err(error) = store.evict(capacity, Duration::ZERO, &layout) {
                tracing::error!("artifact store: the eviction of {} failed: {error}", store.root().display());
            }
        };
        match seismic_threads::spawn("seismic-evict", evicting) {
            Ok(under_way) => *eviction = Some(under_way),
            Err(error) => tracing::warn!("artifact store: the eviction of {} does not start: {error}", self.root().display()),
        }
    }

    /// Wait for the eviction under way to end.
    #[cfg(test)]
    fn evicted(&self) {
        if let Some(under_way) = self.eviction.lock().unwrap().take() {
            under_way.join().unwrap();
        }
    }
}

impl ArtifactStore for DirectoryStore {
    /// Read without a lock, marked used: an artifact is a content address
    /// and a pointer is checked whole by its reader.
    fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>> {
        self.store.fetch(&Self::artifact(namespace, key)).ok().flatten()
    }

    fn contains(&self, namespace: &str, key: &ArtifactKey) -> bool {
        self.root().join(Self::artifact(namespace, key)).is_file()
    }

    fn held(&self, namespace: &str, key: &ArtifactKey) -> Option<(Vec<u8>, HeldFile)> {
        let path = Self::artifact(namespace, key);
        let held = match self.store.hold(&path) {
            Ok(held) => held?,
            Err(error) => {
                tracing::warn!("artifact store: cannot hold {}: {error}", self.root().join(path).display());
                return None;
            }
        };
        let bytes = match held.bytes() {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::warn!("artifact store: cannot read {}: {error}", held.path().display());
                return None;
            }
        };
        Some((bytes, HeldFile::new(held.path().to_owned(), Box::new(held))))
    }

    /// A program's entries are unsynced: every reader checks them whole (an
    /// artifact against its pointer's digest, a pointer by decoding it) and
    /// misses one a crash left torn, which costs one compile. A probe answer
    /// is synced: it is kept from eviction, rare to write, and costly to
    /// form again.
    fn put(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]) {
        let durability = if namespace == crate::probes::NAMESPACE { Durability::Durable } else { Durability::Volatile };
        self.put_reported(namespace, key, bytes, durability);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    /// The store's lock, as [`seismic_io::store`] names it.
    const LOCK: &str = ".lock";

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("seismic-directory-store-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn alone(_: &Path, _: &Examining<'_>) -> Eviction {
        Eviction::Alone
    }

    fn aged(path: &Path, seconds: u64) {
        let file = fs::File::options().write(true).open(path).unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(seconds)).unwrap();
    }

    /// Write one record under the store's lock, released again at once.
    fn write(store: &DirectoryStore, path: &Path, bytes: &[u8]) {
        store.writing().unwrap().write(path, bytes, Durability::Durable).unwrap();
    }

    /// What one process stores, another opening the same directory reads.
    #[test]
    fn an_artifact_outlives_its_store() {
        let root = scratch("outlives");
        let key = ArtifactKey::of(&[b"program"]);
        DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, alone).unwrap().put("metal", &key, b"compiled");
        let store = DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, alone).unwrap();
        assert!(ArtifactStore::contains(&store, "metal", &key));
        assert_eq!(store.get("metal", &key).unwrap(), b"compiled");
        let (bytes, file) = store.held("metal", &key).expect("a directory store keeps each artifact in a file");
        assert_eq!((bytes.as_slice(), fs::read(file.path()).unwrap().as_slice()), (&b"compiled"[..], &b"compiled"[..]));
        drop(file);
        assert!(store.get("cuda", &key).is_none());
        let held = fs::read_dir(root.join(ARTIFACTS).join("metal")).unwrap().flatten().count();
        assert_eq!(held, 1, "no temporary file remains");
        fs::remove_dir_all(root).unwrap();
    }

    /// An artifact held through the store stays at its path while the hold
    /// lives, whatever an eviction over the same directory, as another
    /// process's, finds no room for; released, the next eviction removes it.
    #[test]
    fn a_held_artifact_stays_until_it_is_released() {
        let root = scratch("held");
        let store = DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, alone).unwrap();
        store.evicted();
        let key = ArtifactKey::of(&[b"archive"]);
        store.put("metal", &key, b"archive");
        let (bytes, held) = store.held("metal", &key).expect("the artifact is kept in a file");
        assert_eq!(bytes, b"archive");
        // Room for nothing: its eviction runs, and passes over the hold.
        DirectoryStore::open(root.clone(), 0, alone).unwrap().evicted();
        assert_eq!(fs::read(held.path()).unwrap(), b"archive", "eviction leaves a held file at its path");
        let path = held.path().to_owned();
        drop(held);
        DirectoryStore::open(root.clone(), 0, alone).unwrap().evicted();
        assert!(!path.exists(), "released, the artifact is evicted");
        assert!(store.get("metal", &key).is_none());
        fs::remove_dir_all(root).unwrap();
    }

    /// A layout of records, each a choice and the evidence it names, one
    /// entry by their stem, the choice first; and of measurements kept.
    fn records(path: &Path, examining: &Examining<'_>) -> Eviction {
        let name = path.file_name().unwrap().to_string_lossy();
        let stem = name.split('.').next().unwrap().to_owned();
        match path.iter().next().unwrap().to_str().unwrap() {
            "measured" => Eviction::Kept,
            "choice" => Eviction::Part { entry: stem, order: 0 },
            "evidence" => Eviction::Part { entry: stem, order: 1 },
            _ => programs(path, examining),
        }
    }

    /// [`records`], with evidence its key's choice does not name, by the
    /// evidence's file name, unreachable.
    fn named(path: &Path, examining: &Examining<'_>) -> Eviction {
        let name = path.file_name().unwrap().to_string_lossy();
        match (path.iter().next().unwrap().to_str().unwrap(), name.split_once('.')) {
            ("evidence", Some((stem, _))) => {
                let choice = examining.examine(&Path::new("choice").join(format!("{stem}.json")), u64::MAX).unwrap();
                if choice.as_deref() == Some(name.as_bytes()) {
                    records(path, examining)
                } else {
                    Eviction::Unreachable
                }
            }
            _ => records(path, examining),
        }
    }

    /// Eviction's walk removes every file the layout reads as unreachable,
    /// evidence a live record's choice does not name and evidence no choice
    /// names, however much room there is, and keeps the reachable record.
    #[test]
    fn eviction_removes_what_the_layout_reads_unreachable() {
        let root = scratch("unreachable");
        let store = DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, named).unwrap();
        store.evicted();
        let (choice, evidence) = (Path::new("choice/live.json"), Path::new("evidence/live.1.json"));
        let replaced = Path::new("evidence/live.0.json");
        let dead = Path::new("evidence/dead.0.json");
        write(&store, choice, b"live.1.json");
        for path in [evidence, replaced, dead] {
            write(&store, path, b"evidence");
        }
        store.evicted();
        DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, named).unwrap().evicted();
        assert!(root.join(choice).exists() && root.join(evidence).exists(), "a reachable record stays");
        assert!(!root.join(replaced).exists(), "evidence its live key's choice does not name is removed");
        assert!(!root.join(dead).exists(), "evidence no choice names is removed");
        fs::remove_dir_all(root).unwrap();
    }

    /// Opening evicts the least recently used entries beyond capacity,
    /// each as its layout groups it, whole; it leaves what the layout
    /// keeps, the probe results however old, the store's lock, and files
    /// that share a stem in directories the layout does not group.
    #[test]
    fn opening_evicts_least_recently_used_entries_whole() {
        let root = scratch("evict");
        let store = DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, records).unwrap();
        let old = DirectoryStore::artifact("metal", &ArtifactKey::of(&[b"old"]));
        let probe = DirectoryStore::artifact(crate::probes::NAMESPACE, &ArtifactKey::of(&[b"probe"]));
        let (head, tail) = (Path::new("choice/used.json"), Path::new("evidence/used.0123.json"));
        let (idle_head, idle_tail) = (Path::new("choice/idle.json"), Path::new("evidence/idle.4567.json"));
        let kept = Path::new("measured/point.json");
        let apart = Path::new("prepared/used.json");
        for path in [&old, &probe, head, tail, idle_head, idle_tail, kept, apart] {
            write(&store, path, &[0; 400]);
        }
        store.evicted();
        // The record in use: its head read now, its tail older than the
        // artifact, which by its own age would stay. The file apart shares
        // the record's stem and is older than everything else held.
        aged(&root.join(tail), 3600);
        aged(&root.join(&old), 1800);
        aged(&root.join(apart), 5400);
        for path in [idle_head, idle_tail, kept, &probe, Path::new(LOCK)] {
            aged(&root.join(path), 7200);
        }
        let store = DirectoryStore::open(root.clone(), 1000, records).unwrap();
        store.evicted();
        assert!(root.join(head).exists() && root.join(tail).exists());
        assert!(!root.join(&old).exists() && !root.join(apart).exists());
        assert!(!root.join(idle_head).exists() && !root.join(idle_tail).exists());
        assert!(root.join(kept).exists() && root.join(&probe).exists());
        assert!(root.join(LOCK).exists(), "eviction never removes the lock it holds");
        fs::remove_dir_all(root).unwrap();
    }

    /// Eviction lists and removes under the store's lock, as one step:
    /// while a writer holds the lock an eviction passes, neither listing nor
    /// removing nor waiting, and the next lists what was written meanwhile
    /// and keeps the newest that fits.
    #[test]
    fn eviction_lists_and_removes_under_the_stores_lock() {
        let root = scratch("locked");
        let store = DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, records).unwrap();
        let idle = [Path::new("choice/idle.json"), Path::new("evidence/idle.0.json")];
        for path in idle {
            write(&store, path, &[0; 400]);
            aged(&root.join(path), 3600);
        }
        store.evicted();
        let writing = store.writing().unwrap();
        // Room for one record: the idle one fits alone.
        DirectoryStore::open(root.clone(), 900, records).unwrap().evicted();
        let fresh = [Path::new("choice/fresh.json"), Path::new("evidence/fresh.0.json")];
        for path in fresh {
            writing.write(path, &[1; 400], Durability::Durable).unwrap();
        }
        assert!(idle.iter().all(|path| root.join(path).exists()), "eviction removes nothing while a writer holds the lock");
        drop(writing);
        let tight = DirectoryStore::open(root.clone(), 900, records).unwrap();
        tight.evicted();
        assert!(fresh.iter().all(|path| root.join(path).exists()), "the record written meanwhile is listed, and kept as the newest");
        assert!(idle.iter().all(|path| !root.join(path).exists()), "the older record is evicted whole");
        fs::remove_dir_all(root).unwrap();
    }

    /// A record is read and listed through the lock; an absent one is
    /// none, and a lock that cannot be taken is an error, not an absent
    /// record.
    #[test]
    fn records_are_reached_through_the_lock_or_not_at_all() {
        let root = scratch("unlockable");
        let store = DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, records).unwrap();
        store.evicted();
        let record = Path::new("choice/kept.json");
        write(&store, record, b"record");
        assert_eq!(store.reading().unwrap().read(record, u64::MAX).unwrap().unwrap(), b"record");
        assert_eq!(store.reading().unwrap().read(Path::new("choice/absent.json"), u64::MAX).unwrap(), None);
        assert_eq!(store.writing().unwrap().list(Path::new("choice")).unwrap(), ["kept.json"]);
        store.writing().unwrap().remove(record).unwrap();
        assert_eq!(store.reading().unwrap().read(record, u64::MAX).unwrap(), None);
        // A directory where the lock is: no file to lock.
        fs::remove_file(root.join(LOCK)).unwrap();
        fs::create_dir(root.join(LOCK)).unwrap();
        assert!(store.reading().is_err() && store.writing().is_err());
        fs::remove_dir_all(root).unwrap();
    }

    /// Writing a quarter of the capacity evicts again, without reopening.
    #[test]
    fn writing_evicts_beyond_capacity() {
        let root = scratch("written");
        let store = DirectoryStore::open(root.clone(), 1000, alone).unwrap();
        store.evicted();
        let (old, new) = (Path::new("old"), Path::new("new"));
        write(&store, old, &[0; 600]);
        store.evicted();
        aged(&root.join(old), 3600);
        write(&store, new, &[0; 600]);
        store.evicted();
        assert!(!root.join(old).exists() && root.join(new).exists());
        fs::remove_dir_all(root).unwrap();
    }

    /// A temporary file a dead writer left is removed once abandoned, in a
    /// directory the layout keeps too (the probe answers'), whose kept files
    /// stay however old.
    #[test]
    fn an_abandoned_write_is_removed() {
        let root = scratch("abandoned");
        DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, programs).unwrap().evicted();
        let probes = root.join(ARTIFACTS).join(crate::probes::NAMESPACE);
        fs::create_dir_all(&probes).unwrap();
        let (fresh, abandoned) = (root.join(".a.tmp"), root.join(".b.tmp"));
        let (answers, abandoned_answers) = (probes.join("answers"), probes.join(".answers.tmp"));
        for file in [&fresh, &abandoned, &answers, &abandoned_answers] {
            fs::write(file, b"").unwrap();
        }
        for file in [&abandoned, &answers, &abandoned_answers] {
            aged(file, 2 * seismic_io::ABANDONED.as_secs());
        }
        DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, programs).unwrap().evicted();
        assert!(fresh.exists() && !abandoned.exists());
        assert!(answers.exists() && !abandoned_answers.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
