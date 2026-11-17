//! A store kept in files under one directory, which every process that
//! opens it on the same directory shares: the user's program store
//! ([`super::ProgramStore::user`], laid out by [`programs`]), and the
//! engine's kernel cache of tuning records, laid out by its own.
//!
//! Layout under the root:
//!
//! ```text
//! artifacts/<namespace>/<key>   what a device's program former stores
//!                               ([`ArtifactStore`]): each compile's artifact
//!                               and each entry's pointer, one namespace per
//!                               toolchain; `artifacts/probes` holds the
//!                               devices' probe answers
//! <directory>/<name>            an embedder's own records ([`Writing::write`])
//! lock                          the store's lock ([`DirectoryStore::reading`],
//!                               [`DirectoryStore::writing`])
//! ```
//!
//! Every write is published ([`seismic_io::publish`]), so no reader, of this
//! store, another over the same directory, or another process, ever sees a
//! partial file; a temporary file older than [`ABANDONED_WRITE`] belongs to
//! a writer that died and is removed. Reading a file refreshes its
//! modification time, its use time.
//!
//! An embedder's records are reached only through the store's lock: held
//! shared to read them ([`Reading`]), exclusively to write or remove them
//! ([`Writing`]), in this process or another. Eviction holds it exclusively
//! from its listing to its last removal, so a reader's records stay as they
//! were while it holds them, and eviction never removes what a writer wrote
//! since it listed. A lock that cannot be taken is an error of the access
//! that wanted it, never an absent record. The artifacts take no lock: each
//! is a content address, never written again, and every reader checks the
//! one it reads whole.
//!
//! Opening the store, and every quarter of its capacity written after,
//! removes the least recently used entries beyond its capacity, on a thread
//! of its own, off the path that opened or wrote it. Each artifact is an
//! entry of its own; what the embedder keeps beside them is read by its
//! layout ([`Layout`]): a file may be part of an entry with others, which
//! is as large as its files together, used when the latest of them was, and
//! removed whole in the order its layout gives, so a reader finds it whole
//! or not at all. A file the layout reads as unreachable, judged by what
//! it examines of the records as eviction walks them ([`Examining`]), is
//! removed by that walk whether or not its entry fits. What the layout
//! keeps is no entry and is never removed:
//! eviction walks it only for the abandoned temporary files of writes into
//! it, its own and any other writer's that writes the same way.
//!
//! Every artifact key is a content address, so an artifact is never
//! invalidated; any artifact that cannot be read is a miss.

use super::{ArtifactKey, ArtifactStore};
use seismic_io::Durability;
use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// Bytes a store keeps unless its opener states otherwise.
pub const DEFAULT_CAPACITY: u64 = 1 << 30;

/// Temporary files older than this belong to a writer that died.
const ABANDONED_WRITE: Duration = Duration::from_secs(60 * 60);

/// The directory under the root that holds the artifacts.
const ARTIFACTS: &str = "artifacts";

/// The store's lock, under the root: no entry, never evicted.
const LOCK: &str = "lock";

/// How eviction reads a path the embedder keeps beside the artifacts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Eviction {
    /// Never removed; a directory so read holds no entry, and only its
    /// abandoned temporary files are.
    Kept,
    /// An entry of its own.
    Alone,
    /// Part of the entry named `entry`, with every other file so named,
    /// removed in `order`, lowest first: the file that names the others
    /// goes first.
    Part { entry: String, order: u8 },
    /// A file nothing the embedder keeps reaches: the walk that reads it so
    /// removes it, whatever the capacity. Of a directory, the walk reads
    /// only whether it is kept.
    Unreachable,
}

/// The embedder's layout: how eviction reads each path under the root, a
/// directory before it walks it and each file in one it walks, examining
/// what it needs of the records to judge whether a file is reachable.
pub type Layout = fn(&Path, &Examining<'_>) -> Eviction;

/// The records as eviction's walk lets a layout read them: examined, their
/// use time as it was, under the store's lock the walk holds exclusively,
/// so no writer is between files that name one another.
pub struct Examining<'a> {
    root: &'a Path,
}

impl Examining<'_> {
    /// The record at `path` under the root, none if no file is there.
    pub fn examine(&self, path: &Path) -> std::io::Result<Option<Vec<u8>>> {
        read(&self.root.join(path))
    }
}

/// The bytes of the file at `path`, none if there is none.
fn read(path: &Path) -> std::io::Result<Option<Vec<u8>>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

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
    root: PathBuf,
    capacity: u64,
    layout: Layout,
    /// Bytes written since eviction last began.
    written: AtomicU64,
    /// The eviction under way, if one is.
    eviction: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// The embedder's records of a store, its lock held shared: every reader's
/// records are as one writer left them until it is dropped.
#[must_use = "the records are held only while this lives"]
#[derive(Debug)]
pub struct Reading<'a> {
    store: &'a DirectoryStore,
    _lock: fs::File,
}

/// The embedder's records of a store, its lock held exclusively: no
/// reader, other writer or eviction is among them until it is dropped.
#[must_use = "the records are held only while this lives"]
#[derive(Debug)]
pub struct Writing<'a> {
    store: &'a DirectoryStore,
    _lock: fs::File,
}

impl DirectoryStore {
    /// The store at `root`, created if absent, keeping `capacity` bytes of
    /// entries, the embedder's files read by `layout`; it begins removing
    /// the least recently used beyond them.
    pub fn open(root: PathBuf, capacity: u64, layout: Layout) -> std::io::Result<Self> {
        fs::create_dir_all(&root)?;
        let store = Self {
            root,
            capacity,
            layout,
            written: AtomicU64::new(0),
            eviction: std::sync::Mutex::new(None),
        };
        store.evict();
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The records, read under the store's lock held shared, waiting until
    /// no writer or eviction holds it.
    pub fn reading(&self) -> std::io::Result<Reading<'_>> {
        Ok(Reading { store: self, _lock: lock(&self.root, Locking::Shared)? })
    }

    /// The records, read, written and removed under the store's lock held
    /// exclusively, waiting until no other holder holds it. A thread holding
    /// the lock takes it again only to wait for itself.
    pub fn writing(&self) -> std::io::Result<Writing<'_>> {
        Ok(Writing { store: self, _lock: lock(&self.root, Locking::Exclusive)? })
    }

    /// The bytes at `path` under the root, none if no file is there, read
    /// as `reading` says.
    fn read(&self, path: &Path, reading: Use) -> std::io::Result<Option<Vec<u8>>> {
        let path = self.root.join(path);
        let bytes = read(&path)?;
        if bytes.is_some() && reading == Use::Used {
            used(&path);
        }
        Ok(bytes)
    }

    /// Publish `bytes` at `path` under the root, to `durability`; a failure
    /// leaves no partial file.
    fn written(&self, path: &Path, bytes: &[u8], durability: Durability) -> std::io::Result<()> {
        let path = self.root.join(path);
        fs::create_dir_all(path.parent().expect("a stored file lives in a directory"))?;
        seismic_io::publish(&path, durability, |file| file.write_all(bytes))?;
        let length = bytes.len() as u64;
        if self.written.fetch_add(length, Ordering::Relaxed) + length >= self.capacity / 4 {
            self.evict();
        }
        Ok(())
    }

    /// Write `bytes` under `key` of `namespace`, synced, as [`ArtifactStore::put`]
    /// does; whether they were written, which a put only reports.
    pub(crate) fn put_synced(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]) -> bool {
        self.put_reported(namespace, key, bytes, Durability::Durable)
    }

    /// Publish an artifact, reporting a failure as a warning event, which
    /// the embedder's subscriber carries to its telemetry: a missing
    /// artifact costs its compile again.
    fn put_reported(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8], durability: Durability) -> bool {
        let path = Self::artifact(namespace, key);
        match self.written(&path, bytes, durability) {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!("artifact store: cannot write {}: {error}", self.root.join(path).display());
                false
            }
        }
    }

    fn artifact(namespace: &str, key: &ArtifactKey) -> PathBuf {
        Path::new(ARTIFACTS).join(namespace).join(key.as_str())
    }

    /// Begin removing the least recently used entries beyond the capacity,
    /// on a thread of its own, unless an eviction is under way.
    fn evict(&self) {
        let mut eviction = self.eviction.lock().expect("no eviction panics starting");
        if eviction.as_ref().is_some_and(|under_way| !under_way.is_finished()) {
            return;
        }
        self.written.store(0, Ordering::Relaxed);
        let (root, capacity, layout) = (self.root.clone(), self.capacity, self.layout);
        match seismic_threads::spawn("seismic-evict", move || evict(&root, capacity, layout))
        {
            Ok(under_way) => *eviction = Some(under_way),
            Err(error) => tracing::warn!("artifact store: the eviction of {} does not start: {error}", self.root.display()),
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

/// Whether a read is a use of what it reads, which keeps it from eviction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Use {
    Used,
    /// Read to maintain the records, which leaves eviction's order as it was.
    Examined,
}

impl Reading<'_> {
    /// The record at `path` under the root, none if no file is there,
    /// refreshing its use time.
    pub fn read(&self, path: &Path) -> std::io::Result<Option<Vec<u8>>> {
        self.store.read(path, Use::Used)
    }
}

impl Writing<'_> {
    /// The record at `path` under the root, none if no file is there,
    /// refreshing its use time.
    pub fn read(&self, path: &Path) -> std::io::Result<Option<Vec<u8>>> {
        self.store.read(path, Use::Used)
    }

    /// [`Writing::read`], leaving the record's use time as it was: what
    /// maintains the records reads them without using them.
    pub fn examine(&self, path: &Path) -> std::io::Result<Option<Vec<u8>>> {
        self.store.read(path, Use::Examined)
    }

    /// Publish `bytes` at `path` under the root, durably. A failure leaves
    /// no partial file.
    pub fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        self.store.written(path, bytes, Durability::Durable)
    }

    /// Remove the record at `path` under the root, if one is there.
    pub fn remove(&self, path: &Path) -> std::io::Result<()> {
        match fs::remove_file(self.store.root.join(path)) {
            Err(error) if error.kind() != ErrorKind::NotFound => Err(error),
            _ => Ok(()),
        }
    }

    /// The names of the records in `directory` under the root, none if it
    /// is absent; a write's temporary files are none of them.
    pub fn records(&self, directory: &Path) -> std::io::Result<Vec<String>> {
        let listing = match fs::read_dir(self.store.root.join(directory)) {
            Ok(listing) => listing,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let mut names = Vec::new();
        for item in listing {
            let name = item?.file_name();
            if !seismic_io::unpublished(&name) {
                names.push(name.to_string_lossy().into_owned());
            }
        }
        Ok(names)
    }
}

/// Refresh the use time of the file at `path`: a stale one only makes its
/// entry an earlier eviction candidate.
fn used(path: &Path) {
    if let Ok(file) = fs::File::options().write(true).open(path) {
        let _ = file.set_modified(SystemTime::now());
    }
}

/// How the store's lock is held.
#[derive(Clone, Copy)]
enum Locking {
    Shared,
    Exclusive,
}

/// Hold the lock of the store at `root` as `locking` says, waiting until it
/// can: shared with every other shared holder, exclusive of every other
/// holder, in this process (each through its own file) or another.
fn lock(root: &Path, locking: Locking) -> std::io::Result<fs::File> {
    let file = fs::File::options().create(true).truncate(false).write(true).open(root.join(LOCK))?;
    match locking {
        Locking::Shared => file.lock_shared()?,
        Locking::Exclusive => file.lock()?,
    }
    Ok(file)
}

/// What eviction listed of one entry.
struct Listed {
    /// When the latest of its files was used.
    used: SystemTime,
    bytes: u64,
    /// Its files, each with its order of removal.
    paths: Vec<(u8, PathBuf)>,
}

/// An entry eviction removes whole.
#[derive(PartialEq, Eq, Hash)]
enum Entry {
    /// A file of its own, by its path.
    Alone(PathBuf),
    /// The files the layout names one entry.
    Named(String),
}

/// Remove the abandoned temporary files under `root`, in what the layout
/// keeps too, and what the layout reads as unreachable, then the least
/// recently used entries until the rest fit
/// `capacity`, the store's lock held exclusively throughout. Eviction has
/// no caller to answer: a lock it cannot take is reported as an error
/// event, and the store stays beyond its capacity until an eviction can.
fn evict(root: &Path, capacity: u64, layout: Layout) {
    let _locked = match lock(root, Locking::Exclusive) {
        Ok(locked) => locked,
        Err(error) => {
            tracing::error!("artifact store: the eviction of {} cannot take the store's lock: {error}; nothing is evicted until an eviction can", root.display());
            return;
        }
    };
    let now = SystemTime::now();
    let lock = root.join(LOCK);
    // How eviction reads `path`, under the root.
    let examining = Examining { root };
    let read = |path: &Path| layout(path.strip_prefix(root).expect("eviction walks under its root"), &examining);
    let mut entries = std::collections::HashMap::<Entry, Listed>::new();
    // Each directory to walk, and whether the layout keeps it: a kept
    // directory holds no entry, and is walked only for abandoned writes.
    let mut directories = vec![(root.to_path_buf(), false)];
    while let Some((directory, kept)) = directories.pop() {
        let Ok(listing) = fs::read_dir(&directory) else {
            continue;
        };
        for item in listing.flatten() {
            let Ok(metadata) = item.metadata() else {
                continue;
            };
            let path = item.path();
            if metadata.is_dir() {
                let kept = kept || read(&path) == Eviction::Kept;
                directories.push((path, kept));
                continue;
            }
            let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            if seismic_io::unpublished(&item.file_name()) {
                if now.duration_since(modified).unwrap_or_default() > ABANDONED_WRITE {
                    let _ = fs::remove_file(&path);
                }
                continue;
            }
            if kept || path == lock {
                continue;
            }
            let (entry, order) = match read(&path) {
                Eviction::Kept => continue,
                Eviction::Unreachable => {
                    if let Err(error) = fs::remove_file(&path) {
                        tracing::warn!("artifact store: cannot remove {}, which nothing reaches: {error}", path.display());
                    }
                    continue;
                }
                Eviction::Alone => (Entry::Alone(path.clone()), 0),
                Eviction::Part { entry, order } => (Entry::Named(entry), order),
            };
            let entry = entries.entry(entry).or_insert(Listed { used: SystemTime::UNIX_EPOCH, bytes: 0, paths: Vec::new() });
            entry.used = entry.used.max(modified);
            entry.bytes += metadata.len();
            entry.paths.push((order, path));
        }
    }
    let mut entries = entries.into_values().collect::<Vec<_>>();
    entries.sort_by(|left, right| right.used.cmp(&left.used));
    let mut held = 0u64;
    for mut entry in entries {
        held += entry.bytes;
        if held > capacity {
            entry.paths.sort();
            for (_, path) in entry.paths {
                let _ = fs::remove_file(path);
            }
        }
    }
}

impl ArtifactStore for DirectoryStore {
    fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>> {
        self.read(&Self::artifact(namespace, key), Use::Used).ok().flatten()
    }

    fn contains(&self, namespace: &str, key: &ArtifactKey) -> bool {
        self.root.join(Self::artifact(namespace, key)).is_file()
    }

    fn file(&self, namespace: &str, key: &ArtifactKey) -> Option<PathBuf> {
        let path = self.root.join(Self::artifact(namespace, key));
        path.is_file().then_some(path)
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
        store.writing().unwrap().write(path, bytes).unwrap();
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
        let file = store.file("metal", &key).expect("a directory store keeps each artifact in a file");
        assert_eq!(fs::read(file).unwrap(), b"compiled");
        assert!(store.get("cuda", &key).is_none());
        let leftovers = fs::read_dir(root.join(ARTIFACTS).join("metal"))
            .unwrap()
            .flatten()
            .filter(|item| seismic_io::unpublished(&item.file_name()))
            .count();
        assert_eq!(leftovers, 0, "no temporary file remains");
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
                let choice = examining.examine(&Path::new("choice").join(format!("{stem}.json"))).unwrap();
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
    /// while a writer holds the lock it neither lists nor removes, and once
    /// it may, it lists what was written meanwhile and keeps the newest that
    /// fits.
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
        let tight = DirectoryStore::open(root.clone(), 900, records).unwrap();
        std::thread::sleep(Duration::from_millis(500));
        // Written while eviction waits for the lock, before it lists.
        let fresh = [Path::new("choice/fresh.json"), Path::new("evidence/fresh.0.json")];
        for path in fresh {
            writing.write(path, &[1; 400]).unwrap();
        }
        assert!(idle.iter().all(|path| root.join(path).exists()), "eviction removes nothing while a writer holds the lock");
        drop(writing);
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
        assert_eq!(store.reading().unwrap().read(record).unwrap().unwrap(), b"record");
        assert_eq!(store.reading().unwrap().read(Path::new("choice/absent.json")).unwrap(), None);
        assert_eq!(store.writing().unwrap().records(Path::new("choice")).unwrap(), ["kept.json"]);
        store.writing().unwrap().remove(record).unwrap();
        assert_eq!(store.reading().unwrap().read(record).unwrap(), None);
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
            aged(file, 2 * ABANDONED_WRITE.as_secs());
        }
        DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, programs).unwrap().evicted();
        assert!(fresh.exists() && !abandoned.exists());
        assert!(answers.exists() && !abandoned_answers.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
