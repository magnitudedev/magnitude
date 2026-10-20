//! The engine's persistent cache of formed kernels and tuning results.
//!
//! The host names the directory (ACN passes `--cache-dir
//! <dataDir>/cache/kernels`; `magnitude-harness` takes the same flag); without
//! one, nothing is cached. Layout:
//!
//! ```text
//! <root>/artifacts/<namespace>/<key>   what each toolchain keeps of the programs
//!                                      Seismic formed (compiled bytes on CUDA and
//!                                      Vulkan, a record of what was compiled on
//!                                      Metal), one namespace per toolchain (its
//!                                      ArtifactStore)
//! <root>/tuning/<key>.json             one unit's choice per tuning key: what a load
//!                                      serves and what it holds that choice to,
//!                                      replaced whole when the unit's search changes
//!                                      its served choice and once more when the
//!                                      search's optional work stops
//! <root>/evidence/<key>.<digest>.json  the Seismic TuningResult that choice was made
//!                                      from: every candidate and measurement of the
//!                                      unit's search, named by the key and a digest
//!                                      of its own bytes, which the choice states
//! <root>/readout/<key>.json            completed composed-recipe choice
//! <root>/prepared/<key>                a Prepare of the load this key names ran to
//!                                      its end: a load asked to report unprepared
//!                                      units finds none before it opens a device
//! <root>/records/<digest>.json         Seismic's measurement records, one outcome
//!                                      per point key (its timing, or why it is
//!                                      unobtainable)
//!                                      ([`KernelCache::measurement_records`])
//! <root>/measurement-attempts/<digest>.json
//!                                      how many measurement jobs ended while
//!                                      measuring one point
//!                                      ([`KernelCache::measurement_attempts`])
//! ```
//!
//! The measurement records and their attempts are not the cache's entries.
//! A record is valid while its key (the executor's hardware and driver, the
//! kernel's source, its configuration and point) matches, and is taken
//! again when it does not; losing one costs a measurement job. An attempt is
//! the measurement job's record of a point that ended jobs, and losing one
//! would have the job retry a point that hangs the driver. So eviction never
//! removes them and their use time is never kept; everything below about
//! entries is about the other directories.
//!
//! A unit's record is its choice and its evidence. They are separate files
//! because their readers are: every load decodes the choice, and the
//! evidence, whose size follows what the search measured (megabytes for one
//! unit), is decoded only by what states it or chooses from it again (a
//! tuning trace, a load under another numerical requirement). They are one
//! record all the same, and a choice is never without the evidence it was
//! made from:
//!
//! - The choice names its evidence by content. A record is written evidence
//!   first, then the choice, each renamed into place; the evidence a
//!   replaced choice named is removed after. A writer that dies between
//!   leaves evidence no choice names, never a choice without its evidence
//!   or with another's.
//! - Eviction takes a record whole: the files of one tuning key are one
//!   entry, of their summed size and their latest use.
//!
//! So evidence that a choice names and that cannot be read is a broken
//! store, reported as that, not a miss.
//!
//! Every key is a content address, so nothing is ever invalidated: a changed
//! input gives a new key. Writes go to a temporary file in the same
//! directory and are renamed into place, so concurrent processes never see a
//! partial entry. A file that cannot be read or parsed is a miss (it is
//! rewritten), never an error. Reading an entry refreshes its modification
//! time, and opening the cache evicts the least recently used entries above
//! its capacity.

use seismic::measure::JsonFileStore;
#[cfg(test)]
use seismic::TuningTime;
use seismic::{ArtifactKey, ArtifactStore, TuningResult};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// Bytes the cache keeps by default.
pub const DEFAULT_KERNEL_CACHE_BYTES: u64 = 1 << 30;

/// Temporary files older than this belong to a writer that died; eviction
/// removes them.
const ABANDONED_WRITE: Duration = Duration::from_secs(60 * 60);

const ARTIFACTS: &str = "artifacts";
const TUNING: &str = "tuning";
const EVIDENCE: &str = "evidence";
const READOUT: &str = "readout";
const PREPARED: &str = "prepared";
const RECORDS: &str = "records";
const MEASUREMENT_ATTEMPTS: &str = "measurement-attempts";
/// The directories under the root that hold no entry: eviction never walks
/// them.
const KEPT: [&str; 2] = [RECORDS, MEASUREMENT_ATTEMPTS];

/// The bytes at `path`, refreshing its use time; `None` on any failure.
fn read_entry(path: &Path) -> Option<Vec<u8>> {
    let bytes = fs::read(path).ok()?;
    if let Ok(file) = fs::File::options().write(true).open(path) {
        // Best effort: a stale use time only makes the entry an earlier
        // eviction candidate.
        let _ = file.set_modified(SystemTime::now());
    }
    Some(bytes)
}

/// Where a cache keeps the evidence of one unit's stored choice: the
/// Seismic result the choice was made from. Naming it reads nothing, and it
/// is read wherever its reader is, on any thread.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredEvidence(PathBuf);

impl StoredEvidence {
    /// The stored result. A choice names it and the store keeps the two
    /// together, so an error here is a store that broke that, not a miss.
    pub fn read(&self) -> Result<TuningResult, String> {
        let broken = |detail: String| {
            format!(
                "kernel cache invariant broken: tuning evidence {} that a stored choice names {detail}",
                self.0.display()
            )
        };
        let bytes = read_entry(&self.0).ok_or_else(|| broken("cannot be read".into()))?;
        serde_json::from_slice(&bytes).map_err(|error| broken(format!("does not parse: {error}")))
    }
}

/// A unit's choice as its file states it: the choice, and the digest that
/// names the evidence it was made from.
#[derive(serde::Serialize, serde::Deserialize)]
struct ChoiceEntry<C> {
    evidence: String,
    choice: C,
}

#[derive(Debug)]
pub enum KernelCacheError {
    /// The cache directory could not be created.
    Create {
        path: PathBuf,
        error: std::io::Error,
    },
}

impl std::fmt::Display for KernelCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Create { path, error } => {
                write!(f, "cannot create kernel cache {}: {error}", path.display())
            }
        }
    }
}

impl std::error::Error for KernelCacheError {}

/// The content address of one stored tuning result: the SHA-256 of its key
/// material.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TuningCacheKey(String);

impl TuningCacheKey {
    /// The key of `material`, a canonical rendering of everything the result
    /// depends on.
    pub fn of(material: &str) -> Self {
        let digest = Sha256::digest(material.as_bytes());
        Self(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A kernel cache directory.
#[derive(Debug)]
pub struct KernelCache {
    root: PathBuf,
    /// Distinguishes this process's concurrent temporary files.
    writes: AtomicU64,
}

impl KernelCache {
    /// Open the cache at `root`, creating it, and evict the least recently
    /// used entries beyond `capacity` bytes.
    pub fn open(root: PathBuf, capacity: u64) -> Result<Self, KernelCacheError> {
        for directory in [ARTIFACTS, TUNING, EVIDENCE, READOUT, PREPARED] {
            let path = root.join(directory);
            fs::create_dir_all(&path).map_err(|error| KernelCacheError::Create { path, error })?;
        }
        let cache = Self {
            root,
            writes: AtomicU64::new(0),
        };
        cache.evict(capacity);
        Ok(cache)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The measurement job's attempt records kept in the kernel cache at
    /// `root`, beside the measurement records they are of. Eviction leaves
    /// them alone.
    pub fn measurement_attempts(root: &Path) -> JsonFileStore {
        JsonFileStore::new(&root.join(MEASUREMENT_ATTEMPTS))
    }

    /// Seismic's measurement records kept in the kernel cache at `root`, one
    /// file per point key: the store a measurement reads and writes them
    /// through. Eviction leaves them alone: a record is valid while its key
    /// matches.
    pub fn measurement_records(root: &Path) -> JsonFileStore {
        JsonFileStore::new(&root.join(RECORDS))
    }

    fn entry(&self, directory: &str, key: &str, extension: &str) -> PathBuf {
        self.root.join(directory).join(format!("{key}.{extension}"))
    }

    fn artifact(&self, namespace: &str, key: &ArtifactKey) -> PathBuf {
        self.root.join(ARTIFACTS).join(namespace).join(key.as_str())
    }

    /// The bytes at `path`, refreshing its use time; `None` on any failure.
    fn read(&self, path: &Path) -> Option<Vec<u8>> {
        read_entry(path)
    }

    /// Write `bytes` to `path` through a temporary file in the same
    /// directory. A failure is reported and leaves no partial entry.
    /// Returns whether the entry was written.
    fn write(&self, path: &Path, bytes: &[u8]) -> bool {
        let directory = path.parent().expect("cache entries live in a directory");
        if let Err(error) = fs::create_dir_all(directory) {
            eprintln!(
                "magnitude-engine: kernel cache: cannot create {}: {error}",
                directory.display()
            );
            return false;
        }
        let temporary = directory.join(format!(
            ".{}.{}.{}.tmp",
            path.file_name()
                .expect("cache entries have names")
                .to_string_lossy(),
            std::process::id(),
            self.writes.fetch_add(1, Ordering::Relaxed)
        ));
        let written = fs::File::create(&temporary)
            .and_then(|mut file| file.write_all(bytes).and_then(|()| file.sync_all()))
            .and_then(|()| fs::rename(&temporary, path));
        if let Err(error) = &written {
            let _ = fs::remove_file(&temporary);
            eprintln!(
                "magnitude-engine: kernel cache: cannot write {}: {error}",
                path.display()
            );
        }
        written.is_ok()
    }

    /// The choice of the unit stored under `key`, if any parses, and where
    /// its evidence is: the choice is all a load decodes of the record.
    pub fn tuning<C: serde::de::DeserializeOwned>(
        &self,
        key: &TuningCacheKey,
    ) -> Option<(C, StoredEvidence)> {
        let bytes = self.read(&self.entry(TUNING, key.as_str(), "json"))?;
        let entry = serde_json::from_slice::<ChoiceEntry<C>>(&bytes).ok()?;
        Some((entry.choice, self.evidence(key, &entry.evidence)))
    }

    fn evidence(&self, key: &TuningCacheKey, digest: &str) -> StoredEvidence {
        StoredEvidence(self.entry(EVIDENCE, &format!("{}.{digest}", key.as_str()), "json"))
    }

    /// Store a unit's record under `key`: the `evidence` its search made,
    /// then the `choice` made from it, which names that evidence; then the
    /// evidence of the choice this one replaced is removed. Without its
    /// evidence a choice is not stored, and a writer that dies part way
    /// leaves the record that was there, readable, or none.
    pub fn store_tuning(
        &self,
        key: &TuningCacheKey,
        choice: &impl serde::Serialize,
        evidence: &TuningResult,
    ) {
        let choice_path = self.entry(TUNING, key.as_str(), "json");
        let replaced = fs::read(&choice_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ChoiceEntry<serde_json::Value>>(&bytes).ok())
            .map(|entry| entry.evidence);
        let bytes = serde_json::to_vec(evidence).expect("a tuning result serializes");
        let digest = Sha256::digest(&bytes)
            .iter()
            .take(8)
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if !self.write(&self.evidence(key, &digest).0, &bytes) {
            return;
        }
        let entry = ChoiceEntry {
            evidence: digest.clone(),
            choice,
        };
        let bytes = serde_json::to_vec(&entry).expect("a tuning choice serializes");
        if !self.write(&choice_path, &bytes) {
            return;
        }
        if let Some(replaced) = replaced.filter(|replaced| *replaced != digest) {
            let _ = fs::remove_file(self.evidence(key, &replaced).0);
        }
    }

    /// Whether a Prepare of the load `key` names ran to its end. A host that
    /// loads in its own process asks this before it opens a device: a
    /// process that has formed any program does not see the programs
    /// another process forms afterwards, so a load that opened a device
    /// only to find its units unprepared would form everything again after
    /// its Prepare. Absent, the load is unprepared without looking further.
    /// Present, the units' records say the rest: a record that is missing
    /// all the same (another device, an evicted entry) is found when the
    /// load prepares its programs, as it would be without this entry.
    pub fn prepared(&self, key: &TuningCacheKey) -> bool {
        self.read(&self.entry(PREPARED, key.as_str(), "json")).is_some()
    }

    /// Record that a Prepare of the load `key` names ran to its end.
    pub fn store_prepared(&self, key: &TuningCacheKey) {
        self.write(&self.entry(PREPARED, key.as_str(), "json"), b"{}");
    }

    /// Completed composed-recipe choice; false is a permanent canonical
    /// fallback for this first-load identity, including unmeasured scope.
    pub(crate) fn readout_choice(&self, key: &TuningCacheKey) -> Option<bool> {
        let bytes = self.read(&self.entry(READOUT, key.as_str(), "json"))?;
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()?
            .get("selected")?
            .as_bool()
    }

    /// The exact finalized canonical specialization against which a selected
    /// optional recipe completed its paired comparison.
    pub(crate) fn readout_canonical_choice(&self, key: &TuningCacheKey) -> Option<String> {
        let bytes = self.read(&self.entry(READOUT, key.as_str(), "json"))?;
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()?
            .get("canonical_specification")?
            .as_str()
            .map(str::to_owned)
    }

    pub(crate) fn store_readout_choice(
        &self,
        key: &TuningCacheKey,
        receipt: &impl serde::Serialize,
    ) {
        let bytes = serde_json::to_vec(receipt).expect("a completed choice serializes");
        self.write(&self.entry(READOUT, key.as_str(), "json"), &bytes);
    }

    /// What `path` is evicted with: the files of one tuning key (its choice
    /// and the evidence the choice names) are one entry; any other file is
    /// its own.
    fn eviction_entry(&self, path: &Path) -> PathBuf {
        let record = [TUNING, EVIDENCE]
            .iter()
            .any(|directory| path.parent() == Some(self.root.join(directory).as_path()));
        match path.file_name().and_then(|name| name.to_str()) {
            Some(name) if record => {
                let key = name.split('.').next().expect("a split has a first part");
                self.root.join(TUNING).join(key)
            }
            _ => path.to_path_buf(),
        }
    }

    /// Remove abandoned temporary files, then the least recently used
    /// entries until the rest fit `capacity`. Every file under the root but
    /// the measurement records and their attempts belongs to an entry
    /// ([`Self::eviction_entry`]); an entry is as large as its files
    /// together, was used when the latest of them was, and is removed whole.
    /// The directories of [`KEPT`] are never walked.
    fn evict(&self, capacity: u64) {
        let now = SystemTime::now();
        let mut files = Vec::new();
        let mut directories = vec![self.root.clone()];
        while let Some(directory) = directories.pop() {
            let Ok(listing) = fs::read_dir(&directory) else {
                continue;
            };
            for item in listing.flatten() {
                let Ok(metadata) = item.metadata() else {
                    continue;
                };
                let path = item.path();
                if metadata.is_dir() {
                    // Measurement records and the job's attempts are valid by
                    // their key, not their use time.
                    if !KEPT.iter().any(|kept| path == self.root.join(kept)) {
                        directories.push(path);
                    }
                    continue;
                }
                let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                if item.file_name().to_string_lossy().starts_with('.') {
                    if now.duration_since(modified).unwrap_or_default() > ABANDONED_WRITE {
                        let _ = fs::remove_file(&path);
                    }
                    continue;
                }
                files.push((modified, metadata.len(), path));
            }
        }
        let mut entries = std::collections::HashMap::<PathBuf, (SystemTime, u64, Vec<PathBuf>)>::new();
        for (modified, bytes, path) in files {
            let entry = entries
                .entry(self.eviction_entry(&path))
                .or_insert((SystemTime::UNIX_EPOCH, 0, Vec::new()));
            entry.0 = entry.0.max(modified);
            entry.1 += bytes;
            entry.2.push(path);
        }
        let mut entries = entries.into_values().collect::<Vec<_>>();
        entries.sort_by(|left, right| right.0.cmp(&left.0));
        let mut kept = 0u64;
        for (_, bytes, paths) in entries {
            kept += bytes;
            if kept > capacity {
                // The choice first: a reader then finds no record, never a
                // choice whose evidence is going.
                for path in paths.iter().filter(|path| path.parent() == Some(self.root.join(TUNING).as_path())) {
                    let _ = fs::remove_file(path);
                }
                for path in &paths {
                    let _ = fs::remove_file(path);
                }
            }
        }
    }
}

impl ArtifactStore for KernelCache {
    fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>> {
        self.read(&self.artifact(namespace, key))
    }

    fn put(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]) {
        self.write(&self.artifact(namespace, key), bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "magnitude-kernel-cache-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn a_corrupt_entry_is_a_miss_and_is_rewritten() {
        let root = scratch("corrupt");
        let cache = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let key = TuningCacheKey::of("material");
        fs::write(
            root.join(TUNING).join(format!("{}.json", key.as_str())),
            b"{ not json",
        )
        .unwrap();
        assert!(cache.tuning::<serde_json::Value>(&key).is_none());
        let path = cache.entry(TUNING, key.as_str(), "json");
        cache.write(&path, b"rewritten");
        assert_eq!(fs::read(&path).unwrap(), b"rewritten");
        let leftovers = fs::read_dir(root.join(TUNING))
            .unwrap()
            .flatten()
            .filter(|item| item.file_name().to_string_lossy().starts_with('.'))
            .count();
        assert_eq!(leftovers, 0, "no temporary file remains");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn opening_evicts_the_least_recently_used_beyond_capacity() {
        let root = scratch("evict");
        let cache = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let old = cache.root.join(ARTIFACTS).join("metal").join("old");
        let recent = cache.entry(TUNING, "recent", "json");
        cache.write(&old, &[0; 600]);
        cache.write(&recent, &[0; 600]);
        let file = fs::File::options().write(true).open(&old).unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(3600))
            .unwrap();
        drop(KernelCache::open(root.clone(), 1000).unwrap());
        assert!(!old.exists());
        assert!(recent.exists());
        fs::remove_dir_all(root).unwrap();
    }

    /// Measurement records and the job's attempts are no entry of the
    /// cache: eviction leaves them however old they are and whatever the
    /// capacity.
    #[test]
    fn eviction_leaves_the_measurements() {
        let root = scratch("measurements");
        let kept = KEPT.map(|directory| {
            fs::create_dir_all(root.join(directory)).unwrap();
            let path = root.join(directory).join("digest.json");
            fs::write(&path, [0; 600]).unwrap();
            let file = fs::File::options().write(true).open(&path).unwrap();
            file.set_modified(SystemTime::now() - Duration::from_secs(3600))
                .unwrap();
            path
        });
        drop(KernelCache::open(root.clone(), 0).unwrap());
        assert!(kept.iter().all(|path| path.exists()));
        fs::remove_dir_all(root).unwrap();
    }

    /// A unit's choice and the evidence it names are one entry: as large
    /// as both, used when either last was, kept or evicted together.
    #[test]
    fn a_record_is_kept_or_evicted_whole() {
        let root = scratch("record");
        let cache = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let aged = |path: &Path, seconds: u64| {
            let file = fs::File::options().write(true).open(path).unwrap();
            file.set_modified(SystemTime::now() - Duration::from_secs(seconds))
                .unwrap();
        };
        let record = |name: &str| {
            let choice = cache.entry(TUNING, name, "json");
            let evidence = cache.entry(EVIDENCE, &format!("{name}.0123456789abcdef"), "json");
            cache.write(&evidence, &[0; 600]);
            cache.write(&choice, &[0; 600]);
            (choice, evidence)
        };
        // A load read this record's choice just now and nothing of its
        // evidence, which is older than the artifact beside it.
        let (used_choice, used_evidence) = record("used");
        aged(&used_evidence, 3600);
        let artifact = cache.root.join(ARTIFACTS).join("metal").join("between");
        cache.write(&artifact, &[0; 600]);
        aged(&artifact, 1800);
        let (idle_choice, idle_evidence) = record("idle");
        aged(&idle_choice, 7200);
        aged(&idle_evidence, 7200);
        // Room for the record in use and no more: by its files' own ages
        // its evidence would go and the artifact stay.
        drop(KernelCache::open(root.clone(), 1500).unwrap());
        assert!(used_choice.exists() && used_evidence.exists());
        assert!(!artifact.exists());
        assert!(!idle_choice.exists() && !idle_evidence.exists());
        fs::remove_dir_all(root).unwrap();
    }

    /// A stored choice names the evidence it was made from; replacing the
    /// record leaves one choice and its own evidence.
    #[test]
    fn a_stored_choice_names_its_evidence_and_a_replacement_removes_the_old() {
        let root = scratch("pair");
        let cache = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let key = TuningCacheKey::of("material");
        let evidence_files = || fs::read_dir(root.join(EVIDENCE)).unwrap().count();
        let mut result = TuningResult {
            disposition: seismic::TuningDisposition::QualifiedCandidate,
            tuning_identity: "device".into(),
            entry: "entry".into(),
            backend: "metal".into(),
            points: Vec::new(),
            requirement: seismic::NumericalRequirement::RoundingSteps {
                alpha: 4.0,
                activation: None,
            },
            measurement_identity: String::new(),
            observations: Vec::new(),
            served: Vec::new(),
            reselection: None,
            implementation_identity: String::new(),
            parameters: Vec::new(),
            configurations: Vec::new(),
            overall: seismic::Configuration {
                statics: Default::default(),
                params: Default::default(),
                launches: Vec::new(),
            },
            method: seismic::TuningMethod::Census,
            time: TuningTime::default(),
            coverage: None,
        };
        cache.store_tuning(&key, &1u32, &result);
        let (choice, evidence) = cache.tuning::<u32>(&key).unwrap();
        assert_eq!((choice, evidence.read().unwrap()), (1, result.clone()));
        result.entry = "replaced".into();
        cache.store_tuning(&key, &2u32, &result);
        let (choice, evidence) = cache.tuning::<u32>(&key).unwrap();
        assert_eq!((choice, evidence.read().unwrap()), (2, result));
        assert_eq!(evidence_files(), 1);
        fs::remove_dir_all(root).unwrap();
    }
}
