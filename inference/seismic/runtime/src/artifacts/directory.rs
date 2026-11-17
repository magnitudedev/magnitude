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
//! <directory>/<name>            an embedder's own entries ([`DirectoryStore::write`])
//! ```
//!
//! Writes go to a temporary file in the same directory and are renamed into
//! place, so concurrent processes never see a partial file; a temporary file
//! older than [`ABANDONED_WRITE`] belongs to a writer that died and is
//! removed. Reading a file refreshes its modification time, its use time.
//!
//! Opening the store, and every quarter of its capacity written after,
//! removes the least recently used entries beyond its capacity, on a thread
//! of its own, off the path that opened or wrote it. Each artifact is an
//! entry of its own; what the embedder keeps beside them is read by its
//! layout ([`Layout`]): a file may be part of an entry with others, which
//! is as large as its files together, used when the latest of them was, and
//! removed whole in the order its layout gives, so a reader finds it whole
//! or not at all. What the layout keeps is no entry and is never removed:
//! eviction walks it only for the abandoned temporary files of writes into
//! it, its own and any other writer's that writes the same way.
//!
//! Every artifact key is a content address, so an artifact is never
//! invalidated; any file that cannot be read is a miss.

use super::{ArtifactKey, ArtifactStore};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// Bytes a store keeps unless its opener states otherwise.
pub const DEFAULT_CAPACITY: u64 = 1 << 30;

/// Temporary files older than this belong to a writer that died.
const ABANDONED_WRITE: Duration = Duration::from_secs(60 * 60);

/// The directory under the root that holds the artifacts.
const ARTIFACTS: &str = "artifacts";

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
}

/// The embedder's layout: how eviction reads each path under the root, a
/// directory before it walks it and each file in one it walks.
pub type Layout = fn(&Path) -> Eviction;

/// The layout of a store of programs: each artifact and pointer an entry of
/// its own, and the probe answers (`artifacts/probes`) kept, as forming one
/// again runs its probes.
pub fn programs(path: &Path) -> Eviction {
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
    /// Distinguishes this process's concurrent temporary files.
    writes: AtomicU64,
    /// Bytes written since eviction last began.
    written: AtomicU64,
    /// The eviction under way, if one is.
    eviction: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
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
            writes: AtomicU64::new(0),
            written: AtomicU64::new(0),
            eviction: std::sync::Mutex::new(None),
        };
        store.evict();
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The bytes at `path` under the root, refreshing its use time; none on
    /// any failure.
    pub fn read(&self, path: &Path) -> Option<Vec<u8>> {
        let path = self.root.join(path);
        let bytes = fs::read(&path).ok()?;
        used(&path);
        Some(bytes)
    }

    /// Whether a file is at `path` under the root, read from its metadata.
    pub fn contains(&self, path: &Path) -> bool {
        self.root.join(path).is_file()
    }

    /// Write `bytes` to `path` under the root through a temporary file in
    /// the same directory, synced before it is renamed into place. A
    /// failure is reported as a warning event, which the embedder's
    /// subscriber carries to its telemetry, and leaves no partial file.
    /// Returns whether the file was written.
    pub fn write(&self, path: &Path, bytes: &[u8]) -> bool {
        self.written(path, bytes, true)
    }

    /// [`DirectoryStore::write`], the file synced first where `synced`.
    fn written(&self, path: &Path, bytes: &[u8], synced: bool) -> bool {
        let path = self.root.join(path);
        let directory = path.parent().expect("a stored file lives in a directory");
        if let Err(error) = fs::create_dir_all(directory) {
            tracing::warn!("artifact store: cannot create {}: {error}", directory.display());
            return false;
        }
        let temporary = directory.join(format!(
            ".{}.{}.{}.tmp",
            path.file_name().expect("a stored file has a name").to_string_lossy(),
            std::process::id(),
            self.writes.fetch_add(1, Ordering::Relaxed)
        ));
        let written = fs::File::create(&temporary)
            .and_then(|mut file| file.write_all(bytes).and_then(|()| if synced { file.sync_all() } else { Ok(()) }))
            .and_then(|()| fs::rename(&temporary, &path));
        if let Err(error) = &written {
            let _ = fs::remove_file(&temporary);
            tracing::warn!("artifact store: cannot write {}: {error}", path.display());
            return false;
        }
        let length = bytes.len() as u64;
        if self.written.fetch_add(length, Ordering::Relaxed) + length >= self.capacity / 4 {
            self.evict();
        }
        true
    }

    /// Remove the file at `path` under the root, if there is one.
    pub fn remove(&self, path: &Path) {
        let _ = fs::remove_file(self.root.join(path));
    }

    /// Write `bytes` under `key` of `namespace`, synced, as [`ArtifactStore::put`]
    /// does; whether they were written, which a put only reports.
    pub(crate) fn put_synced(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]) -> bool {
        self.written(&Self::artifact(namespace, key), bytes, true)
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

/// Refresh the use time of the file at `path`: a stale one only makes its
/// entry an earlier eviction candidate.
fn used(path: &Path) {
    if let Ok(file) = fs::File::options().write(true).open(path) {
        let _ = file.set_modified(SystemTime::now());
    }
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
/// keeps too, then the least recently used entries until the rest fit
/// `capacity`.
fn evict(root: &Path, capacity: u64, layout: Layout) {
    let now = SystemTime::now();
    // How eviction reads `path`, under the root.
    let read = |path: &Path| layout(path.strip_prefix(root).expect("eviction walks under its root"));
    let mut entries = std::collections::HashMap::<Entry, (SystemTime, u64, Vec<(u8, PathBuf)>)>::new();
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
            if item.file_name().to_string_lossy().starts_with('.') {
                if now.duration_since(modified).unwrap_or_default() > ABANDONED_WRITE {
                    let _ = fs::remove_file(&path);
                }
                continue;
            }
            if kept {
                continue;
            }
            let (entry, order) = match read(&path) {
                Eviction::Kept => continue,
                Eviction::Alone => (Entry::Alone(path.clone()), 0),
                Eviction::Part { entry, order } => (Entry::Named(entry), order),
            };
            let entry = entries.entry(entry).or_insert((SystemTime::UNIX_EPOCH, 0, Vec::new()));
            entry.0 = entry.0.max(modified);
            entry.1 += metadata.len();
            entry.2.push((order, path));
        }
    }
    let mut entries = entries.into_values().collect::<Vec<_>>();
    entries.sort_by(|left, right| right.0.cmp(&left.0));
    let mut held = 0u64;
    for (_, bytes, mut paths) in entries {
        held += bytes;
        if held > capacity {
            paths.sort();
            for (_, path) in paths {
                let _ = fs::remove_file(path);
            }
        }
    }
}

impl ArtifactStore for DirectoryStore {
    fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>> {
        self.read(&Self::artifact(namespace, key))
    }

    fn contains(&self, namespace: &str, key: &ArtifactKey) -> bool {
        DirectoryStore::contains(self, &Self::artifact(namespace, key))
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
        self.written(&Self::artifact(namespace, key), bytes, namespace == crate::probes::NAMESPACE);
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

    fn alone(_: &Path) -> Eviction {
        Eviction::Alone
    }

    fn aged(path: &Path, seconds: u64) {
        let file = fs::File::options().write(true).open(path).unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(seconds)).unwrap();
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
            .filter(|item| item.file_name().to_string_lossy().starts_with('.'))
            .count();
        assert_eq!(leftovers, 0, "no temporary file remains");
        fs::remove_dir_all(root).unwrap();
    }

    /// A layout of records, each a choice and the evidence it names, one
    /// entry by their stem, the choice first; and of measurements kept.
    fn records(path: &Path) -> Eviction {
        let name = path.file_name().unwrap().to_string_lossy();
        let stem = name.split('.').next().unwrap().to_owned();
        match path.iter().next().unwrap().to_str().unwrap() {
            "measured" => Eviction::Kept,
            "choice" => Eviction::Part { entry: stem, order: 0 },
            "evidence" => Eviction::Part { entry: stem, order: 1 },
            _ => programs(path),
        }
    }

    /// Opening evicts the least recently used entries beyond capacity,
    /// each as its layout groups it, whole; it leaves what the layout
    /// keeps and the probe results however old, and files that share a
    /// stem in directories the layout does not group.
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
            assert!(store.write(path, &[0; 400]));
        }
        store.evicted();
        // The record in use: its head read now, its tail older than the
        // artifact, which by its own age would stay. The file apart shares
        // the record's stem and is older than everything else held.
        aged(&root.join(tail), 3600);
        aged(&root.join(&old), 1800);
        aged(&root.join(apart), 5400);
        for path in [idle_head, idle_tail, kept, &probe] {
            aged(&root.join(path), 7200);
        }
        let store = DirectoryStore::open(root.clone(), 1000, records).unwrap();
        store.evicted();
        assert!(root.join(head).exists() && root.join(tail).exists());
        assert!(!root.join(&old).exists() && !root.join(apart).exists());
        assert!(!root.join(idle_head).exists() && !root.join(idle_tail).exists());
        assert!(root.join(kept).exists() && root.join(&probe).exists());
        fs::remove_dir_all(root).unwrap();
    }

    /// Writing a quarter of the capacity evicts again, without reopening.
    #[test]
    fn writing_evicts_beyond_capacity() {
        let root = scratch("written");
        let store = DirectoryStore::open(root.clone(), 1000, alone).unwrap();
        store.evicted();
        let (old, new) = (Path::new("old"), Path::new("new"));
        assert!(store.write(old, &[0; 600]));
        store.evicted();
        aged(&root.join(old), 3600);
        assert!(store.write(new, &[0; 600]));
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
