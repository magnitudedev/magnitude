//! The engine's persistent cache of formed kernels and tuning results.
//!
//! The host names the directory (ACN passes `--cache-dir
//! <dataDir>/cache/kernels`; `magnitude-harness` takes the same flag); without
//! one, no tuning result is kept. The programs a load forms and the
//! measurement job's probe results are not the cache's: every device forms
//! through Seismic's one store of the process (`seismic::ProgramStore::user`),
//! whichever cache a load names. The directory is a Seismic [`DirectoryStore`],
//! which owns its files, their writes and their eviction: the cache is the
//! engine's entries in it. Layout:
//!
//! ```text
//! <root>/tuning/<key>.json             one unit's choice per tuning key: what a load
//!                                      serves and what it holds that choice to,
//!                                      replaced whole when the unit's search changes
//!                                      its served choice and once more when the
//!                                      search's optional work stops
//! <root>/evidence/<key>.<digest>.json  the Seismic TuningResult that choice was made
//!                                      from: every candidate and measurement of the
//!                                      unit's search, named by the key and a digest
//!                                      of its own bytes, which the choice states
//! <root>/prepared/<key>.json           a Prepare of the load this key names ran to
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
//! A cache an earlier build kept also held `<root>/artifacts/`: the programs
//! Seismic formed, by toolchain namespace, and the measurement job's probe
//! answers (`artifacts/probes/<key>`). Opening such a cache carries each
//! probe answer the user's store lacks into it, under the same key, which
//! is still the answer's (the executor's identity, the probe revision and
//! layout), so no job probes again for it; once every answer is on disk
//! there, it removes the directory, once, off the opening path
//! ([`KernelCache::open`]), and keeps it otherwise for a later opening. Until it is gone the
//! directory is no entry of the cache: it counts against no capacity.
//!
//! The measurement records, their attempts, the point requests and the
//! machine figures are not the cache's entries, and eviction never removes
//! them. A record is valid while its key (the
//! executor's hardware and driver, the kernel's source, its configuration and
//! point) matches, and is taken again when it does not; losing one costs a
//! measurement job. An attempt is the measurement job's record of a point
//! that ended jobs, and losing one would have the job retry a point that
//! hangs the driver.
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
//!   first, then the choice; the evidence a replaced choice named is removed
//!   after. A writer that dies between leaves evidence no choice names, never
//!   a choice without its evidence or with another's.
//! - The cache's layout makes both files one entry of the store, by the key
//!   ([`layout`]), evicted whole, choice first: a reader then finds no
//!   record, never a choice whose evidence is going.
//!
//! So evidence that a choice names and that cannot be read is a broken
//! store, reported as that, not a miss. Any other file that cannot be read or
//! parsed is a miss (it is rewritten), never an error.

use seismic::measure::JsonFileStore;
#[cfg(test)]
use seismic::TuningTime;
use seismic::{DirectoryStore, Eviction, TuningResult};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Bytes the cache keeps by default.
pub const DEFAULT_KERNEL_CACHE_BYTES: u64 = seismic::DEFAULT_STORE_CAPACITY;

const TUNING: &str = "tuning";
const EVIDENCE: &str = "evidence";
const PREPARED: &str = "prepared";
const RECORDS: &str = "records";
const MEASUREMENT_ATTEMPTS: &str = "measurement-attempts";
const MEASUREMENT_REQUESTS: &str = "measurement-requests";
const MACHINE_FIGURES: &str = "machine-figures";
/// The directories under the root that hold no entry: eviction removes
/// nothing of them but abandoned temporary files.
const KEPT: [&str; 4] = [RECORDS, MEASUREMENT_ATTEMPTS, MEASUREMENT_REQUESTS, MACHINE_FIGURES];
/// Where a cache of an earlier build kept the programs Seismic formed and
/// the probe answers, which opening carries and removes ([`KernelCache::open`]).
const FORMER_ARTIFACTS: &str = "artifacts";

/// How the store's eviction reads the cache's files (`DirectoryStore::open`): a
/// unit's choice and the evidence it names one entry, by their key, the
/// choice first; the measurements kept ([`KEPT`]), and what an earlier
/// build kept of probe answers, until opening carries and removes them
/// ([`FORMER_ARTIFACTS`]); any other file, the programs an earlier build
/// kept among them, an entry of its own.
fn layout(path: &Path) -> Eviction {
    let directory = path.iter().next().and_then(|directory| directory.to_str()).unwrap_or_default();
    let key = || {
        let name = path.file_name().map(|name| name.to_string_lossy()).unwrap_or_default();
        name.split('.').next().unwrap_or_default().to_owned()
    };
    match directory {
        _ if KEPT.contains(&directory) => Eviction::Kept,
        FORMER_ARTIFACTS if path.iter().nth(1).and_then(|namespace| namespace.to_str()) == Some(seismic::PROBE_NAMESPACE) => {
            Eviction::Kept
        }
        TUNING => Eviction::Part { entry: key(), order: 0 },
        EVIDENCE => Eviction::Part { entry: key(), order: 1 },
        _ => Eviction::Alone,
    }
}

/// Where a cache keeps the evidence of one unit's stored choice: the
/// Seismic result the choice was made from. Naming it reads nothing, and it
/// is read wherever its reader is, on any thread.
#[derive(Clone, Debug)]
pub struct StoredEvidence {
    store: Arc<DirectoryStore>,
    path: PathBuf,
}

impl PartialEq for StoredEvidence {
    fn eq(&self, other: &Self) -> bool {
        self.store.root() == other.store.root() && self.path == other.path
    }
}

impl Eq for StoredEvidence {}

impl StoredEvidence {
    /// The stored result. A choice names it and the store keeps the two
    /// together, so an error here is a store that broke that, not a miss.
    pub fn read(&self) -> Result<TuningResult, String> {
        let broken = |detail: String| {
            format!(
                "kernel cache invariant broken: tuning evidence {} that a stored choice names {detail}",
                self.store.root().join(&self.path).display()
            )
        };
        let bytes = self.store.read(&self.path).ok_or_else(|| broken("cannot be read".into()))?;
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

/// Remove `former`, the programs a cache of an earlier build kept, on a
/// thread of its own, off the path that opened the cache. A process that
/// ends first leaves the rest for its next opening; a process removing it
/// at once with another finds what the other removed gone.
fn remove_former_artifacts(former: PathBuf) {
    let removing = move || {
        match std::fs::remove_dir_all(&former) {
            Ok(()) => tracing::info!(
                "kernel cache: removed {}, the programs an earlier build kept there; every program forms through the user's program store",
                former.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing::warn!("kernel cache: cannot remove {}: {error}; its next opening removes it", former.display()),
        }
    };
    if let Err(error) = seismic::threads::spawn("magnitude-cache-migrate", removing) {
        tracing::warn!("kernel cache: the removal of what an earlier build kept does not start: {error}; its next opening removes it");
    }
}

/// A kernel cache directory.
#[derive(Debug)]
pub struct KernelCache {
    store: Arc<DirectoryStore>,
}

impl KernelCache {
    /// Open the cache at `root`, creating it, keeping `capacity` bytes of
    /// entries: the store begins evicting the least recently used beyond
    /// them. What a cache of an earlier build kept of programs is removed,
    /// its probe answers carried first into the user's store, before any
    /// device opens with them.
    pub fn open(root: PathBuf, capacity: u64) -> Result<Self, KernelCacheError> {
        let former = root.join(FORMER_ARTIFACTS);
        // Its probe answers are carried, synced, before any device opens
        // with them; it is removed only once every one is on disk in the
        // user's store, which the key the job stored each under still reads
        // (the executor's identity, the probe revision and layout).
        let removable = former.is_dir()
            && match seismic::carry_probe_results(&former.join(seismic::PROBE_NAMESPACE)) {
                Ok(carried) => {
                    if carried > 0 {
                        tracing::info!("kernel cache: carried {carried} probe answers an earlier build kept in {} into the user's program store", former.display());
                    }
                    true
                }
                Err(reason) => {
                    tracing::warn!("kernel cache: the probe answers an earlier build kept in {} are not carried ({reason}); it is kept for a later opening", former.display());
                    false
                }
            };
        let store = DirectoryStore::open(root.clone(), capacity, layout)
            .map_err(|error| KernelCacheError::Create { path: root, error })?;
        if removable {
            remove_former_artifacts(former);
        }
        Ok(Self { store: Arc::new(store) })
    }

    pub fn root(&self) -> &Path {
        self.store.root()
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

    /// The point requests kept in the kernel cache at `root`
    /// (`assessment::PointRequests`): what the measurement job measures
    /// next, beside the records it stores. Eviction leaves them alone.
    pub fn measurement_requests(root: &Path) -> JsonFileStore {
        JsonFileStore::new(&root.join(MEASUREMENT_REQUESTS))
    }

    /// The machine's own generic figures kept in the kernel cache at `root`,
    /// measured once per machine and keyed by its identity (the host's
    /// page-release rate, `platform::ReleaseRate::of_machine`; a volume's
    /// write rate, `platform::VolumeReserve::of_volume`). Eviction leaves
    /// them alone.
    pub fn machine_figures(root: &Path) -> JsonFileStore {
        JsonFileStore::new(&root.join(MACHINE_FIGURES))
    }

    fn choice(key: &TuningCacheKey) -> PathBuf {
        Path::new(TUNING).join(format!("{}.json", key.as_str()))
    }

    fn evidence(&self, key: &TuningCacheKey, digest: &str) -> StoredEvidence {
        StoredEvidence {
            store: self.store.clone(),
            path: Self::evidence_path(key, digest),
        }
    }

    fn evidence_path(key: &TuningCacheKey, digest: &str) -> PathBuf {
        Path::new(EVIDENCE).join(format!("{}.{digest}.json", key.as_str()))
    }

    fn prepared_path(key: &TuningCacheKey) -> PathBuf {
        Path::new(PREPARED).join(format!("{}.json", key.as_str()))
    }

    /// The choice of the unit stored under `key`, if any parses, and where
    /// its evidence is: the choice is all a load decodes of the record.
    pub fn tuning<C: serde::de::DeserializeOwned>(
        &self,
        key: &TuningCacheKey,
    ) -> Option<(C, StoredEvidence)> {
        let bytes = self.store.read(&Self::choice(key))?;
        let entry = serde_json::from_slice::<ChoiceEntry<C>>(&bytes).ok()?;
        Some((entry.choice, self.evidence(key, &entry.evidence)))
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
        let choice_path = Self::choice(key);
        let replaced = self
            .store
            .read(&choice_path)
            .and_then(|bytes| serde_json::from_slice::<ChoiceEntry<serde_json::Value>>(&bytes).ok())
            .map(|entry| entry.evidence);
        let bytes = serde_json::to_vec(evidence).expect("a tuning result serializes");
        let digest = Sha256::digest(&bytes)
            .iter()
            .take(8)
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if !self.store.write(&self.evidence(key, &digest).path, &bytes) {
            return;
        }
        let entry = ChoiceEntry {
            evidence: digest.clone(),
            choice,
        };
        let bytes = serde_json::to_vec(&entry).expect("a tuning choice serializes");
        if !self.store.write(&choice_path, &bytes) {
            return;
        }
        if let Some(replaced) = replaced.filter(|replaced| *replaced != digest) {
            self.store.remove(&self.evidence(key, &replaced).path);
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
        self.store.read(&Self::prepared_path(key)).is_some()
    }

    /// Record that a Prepare of the load `key` names ran to its end.
    pub fn store_prepared(&self, key: &TuningCacheKey) {
        self.store.write(&Self::prepared_path(key), b"{}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "magnitude-kernel-cache-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn a_corrupt_choice_is_a_miss_and_is_rewritten() {
        let root = scratch("corrupt");
        let cache = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let key = TuningCacheKey::of("material");
        fs::create_dir_all(root.join(TUNING)).unwrap();
        fs::write(root.join(KernelCache::choice(&key)), b"{ not json").unwrap();
        assert!(cache.tuning::<serde_json::Value>(&key).is_none());
        cache.store.write(&KernelCache::choice(&key), b"rewritten");
        assert_eq!(fs::read(root.join(KernelCache::choice(&key))).unwrap(), b"rewritten");
        fs::remove_dir_all(root).unwrap();
    }

    /// A unit's choice and the evidence it names are one entry of the
    /// store, the choice removed first; the load's prepared record of the
    /// same key is another; the measurements and their attempts are no
    /// entry, and eviction leaves them however old they are.
    #[test]
    fn the_layout_keeps_a_record_whole_and_the_measurements() {
        let key = TuningCacheKey::of("material");
        let entry = |order| Eviction::Part { entry: key.as_str().to_owned(), order };
        assert_eq!(layout(&KernelCache::choice(&key)), entry(0));
        assert_eq!(layout(&KernelCache::evidence_path(&key, "0123456789abcdef")), entry(1));
        assert_eq!(layout(&KernelCache::prepared_path(&key)), Eviction::Alone);
        let former_probes = Path::new(FORMER_ARTIFACTS).join(seismic::PROBE_NAMESPACE);
        for directory in KEPT.iter().map(Path::new).chain([former_probes.as_path()]) {
            assert_eq!(layout(directory), Eviction::Kept);
            assert_eq!(layout(&directory.join("digest.json")), Eviction::Kept);
        }
        // An earlier build's programs are entries, evicted whether or not
        // its probe answers have been carried.
        assert_eq!(layout(Path::new(FORMER_ARTIFACTS)), Eviction::Alone);
        assert_eq!(layout(&Path::new(FORMER_ARTIFACTS).join("metal").join("digest")), Eviction::Alone);
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
            default_launches: Vec::new(),
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
