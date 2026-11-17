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
//! <root>/lock                          the store's lock, which orders the records'
//!                                      readers, writers and eviction
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
//!   first, then the choice, and only then is the evidence the replaced
//!   choice named removed: a writer that dies between leaves evidence no
//!   choice names, never a choice without its evidence or with another's.
//! - The store's lock orders every reader, writer and remover of the
//!   records: the store reaches them through it alone, and a lock it cannot
//!   take is an error, never a miss. A writer holds it exclusively from its
//!   evidence to its choice and the removal of what that choice replaced. A
//!   reader holds it shared while it reads a choice and the whole of the
//!   evidence named, and keeps both in memory: nothing reads the evidence
//!   file again, so no load needs any file once it has taken a record, and
//!   a writer may remove what it replaced at once. Evidence is removed only
//!   under the lock held exclusively: by the writer that replaced its
//!   choice, and by eviction, of whole records and of evidence its key's
//!   choice does not name ([`layout`]), which it lists and removes in one
//!   step under the lock. So no remover sees a record half written,
//!   and none removes what a choice names but with that choice.
//! - The cache's layout makes both files one entry of the store, by the key
//!   ([`placement`]), evicted whole, choice first: a reader then finds no
//!   record, never a choice whose evidence is going. What a dead writer
//!   left that no choice names, eviction's next pass removes, as every
//!   opening of the store begins one.
//!
//! So evidence that a choice names and that cannot be read is a broken
//! store, reported as that, not a miss. A file that cannot be parsed is a
//! miss (it is rewritten); one that cannot be read, or a lock that cannot
//! be taken, is an error of the access, never a miss.

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

/// How the store's eviction reads the cache's files: as [`placement`]
/// places them, but evidence its key's choice does not name, examined as
/// the walk reaches it, unreachable, so the walk removes it. Eviction holds
/// the store's lock exclusively, so no writer is between its evidence and
/// its choice: unnamed evidence is what a writer that died left, never
/// what one is writing. Evidence whose choice cannot be read is kept,
/// reported, for a later pass.
fn layout(path: &Path, records: &seismic::Examining<'_>) -> Eviction {
    let placed = placement(path);
    let Eviction::Part { entry: key, order: 1 } = &placed else {
        return placed;
    };
    let name = path.file_name().and_then(|name| name.to_str());
    let Some((_, digest)) = name.and_then(|name| name.strip_suffix(".json")).and_then(|stem| stem.split_once('.')) else {
        return placed;
    };
    let choice = KernelCache::choice(&TuningCacheKey(key.clone()));
    match records.examine(&choice) {
        Ok(choice) if choice.as_deref().and_then(named_evidence).as_deref() == Some(digest) => placed,
        Ok(_) => Eviction::Unreachable,
        Err(error) => {
            tracing::warn!("kernel cache: eviction cannot read {}: {error}; the evidence {} stays for a later pass", choice.display(), path.display());
            placed
        }
    }
}

/// The digest of the evidence the choice in `bytes` names, none if it does
/// not parse.
fn named_evidence(bytes: &[u8]) -> Option<String> {
    serde_json::from_slice::<ChoiceEntry<serde::de::IgnoredAny>>(bytes).ok().map(|entry| entry.evidence)
}

/// Where each of the cache's files lies among the store's entries: a
/// unit's choice and the evidence of its key one entry, the choice first;
/// the measurements kept ([`KEPT`]), and what an earlier build kept of
/// probe answers, until opening carries and removes them
/// ([`FORMER_ARTIFACTS`]); any other file, the programs an earlier build
/// kept among them, an entry of its own.
fn placement(path: &Path) -> Eviction {
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

/// The evidence of one unit's stored choice, the Seismic result the choice
/// was made from, as the store held it when the choice was read
/// ([`KernelCache::tuning`]): its bytes, decoded only by what states it or
/// chooses from it again, on any thread. The store is never read for it
/// again.
#[derive(Clone, PartialEq, Eq)]
pub struct StoredEvidence {
    /// Where the store held it, for what reports it.
    path: PathBuf,
    bytes: Arc<[u8]>,
}

impl std::fmt::Debug for StoredEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StoredEvidence({}, {} bytes)", self.path.display(), self.bytes.len())
    }
}

impl StoredEvidence {
    /// The stored result. A choice names it and the store keeps the two
    /// together, written whole, so an error here is a store that broke
    /// that, not a miss.
    pub fn read(&self) -> Result<TuningResult, String> {
        serde_json::from_slice(&self.bytes).map_err(|error| {
            format!(
                "kernel cache invariant broken: tuning evidence {} that a stored choice names does not parse: {error}",
                self.path.display()
            )
        })
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
    store: DirectoryStore,
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
        Ok(Self { store })
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

    fn evidence_path(key: &TuningCacheKey, digest: &str) -> PathBuf {
        Path::new(EVIDENCE).join(format!("{}.{digest}.json", key.as_str()))
    }

    fn prepared_path(key: &TuningCacheKey) -> PathBuf {
        Path::new(PREPARED).join(format!("{}.json", key.as_str()))
    }

    /// What failed `doing` in the cache: an error of the access, never a
    /// miss.
    fn failed(&self, doing: &str, error: std::io::Error) -> String {
        format!("kernel cache {}: {doing}: {error}", self.root().display())
    }

    /// The record of the unit stored under `key`, if its choice parses: the
    /// choice, all a load decodes of it, and the evidence it names, taken
    /// whole with it under the store's lock held shared, so the record is
    /// one writer's and no remover touches it meanwhile. A choice without
    /// its evidence is a store that broke its invariant, an error, not a
    /// miss; so is a lock or a file that cannot be read.
    pub fn tuning<C: serde::de::DeserializeOwned>(
        &self,
        key: &TuningCacheKey,
    ) -> Result<Option<(C, StoredEvidence)>, String> {
        let records = self.store.reading().map_err(|error| self.failed("cannot take its lock to read a record", error))?;
        let choice = Self::choice(key);
        let Some(entry) = records
            .read(&choice)
            .map_err(|error| self.failed(&format!("cannot read {}", choice.display()), error))?
            .and_then(|bytes| serde_json::from_slice::<ChoiceEntry<C>>(&bytes).ok())
        else {
            return Ok(None);
        };
        let path = Self::evidence_path(key, &entry.evidence);
        let bytes = records
            .read(&path)
            .map_err(|error| self.failed(&format!("cannot read {}", path.display()), error))?
            .ok_or_else(|| {
                format!(
                    "kernel cache invariant broken: tuning evidence {} that a stored choice names is gone",
                    self.root().join(&path).display()
                )
            })?;
        Ok(Some((entry.choice, StoredEvidence { path: self.root().join(path), bytes: bytes.into() })))
    }

    /// Store a unit's record under `key`: the `evidence` its search made,
    /// then the `choice` made from it, which names that evidence, the
    /// store's lock held exclusively. Without its evidence a choice is not
    /// stored, and a writer that fails or dies part way leaves the record
    /// that was there, readable, or none. The evidence the replaced choice
    /// named is removed last, the lock still held: every reader took what it
    /// read of it whole, and none reads it again.
    pub fn store_tuning(
        &self,
        key: &TuningCacheKey,
        choice: &impl serde::Serialize,
        evidence: &TuningResult,
    ) -> Result<(), String> {
        let records = self.store.writing().map_err(|error| self.failed("cannot take its lock to store a record", error))?;
        let bytes = serde_json::to_vec(evidence).expect("a tuning result serializes");
        let digest = Sha256::digest(&bytes)
            .iter()
            .take(8)
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let choice_path = Self::choice(key);
        // What the choice being replaced names, examined: it is replaced,
        // not used. A choice that does not parse names nothing.
        let replaced = records
            .examine(&choice_path)
            .map_err(|error| self.failed(&format!("cannot read {}", choice_path.display()), error))?
            .and_then(|bytes| named_evidence(&bytes))
            .filter(|replaced| *replaced != digest);
        let path = Self::evidence_path(key, &digest);
        records
            .write(&path, &bytes)
            .map_err(|error| self.failed(&format!("cannot write {}", path.display()), error))?;
        let entry = ChoiceEntry {
            evidence: digest,
            choice,
        };
        let bytes = serde_json::to_vec(&entry).expect("a tuning choice serializes");
        records
            .write(&choice_path, &bytes)
            .map_err(|error| self.failed(&format!("cannot write {}", choice_path.display()), error))?;
        let Some(replaced) = replaced else {
            return Ok(());
        };
        // The new record is in place; evidence left behind is what
        // eviction removes as unreachable, so a failed removal is no
        // failure of the store.
        let path = Self::evidence_path(key, &replaced);
        if let Err(error) = records.remove(&path) {
            tracing::warn!("kernel cache: cannot remove {}, which no choice names now: {error}; eviction removes it", path.display());
        }
        Ok(())
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
    pub fn prepared(&self, key: &TuningCacheKey) -> Result<bool, String> {
        let path = Self::prepared_path(key);
        let records = self.store.reading().map_err(|error| self.failed("cannot take its lock to read a record", error))?;
        records
            .read(&path)
            .map(|record| record.is_some())
            .map_err(|error| self.failed(&format!("cannot read {}", path.display()), error))
    }

    /// Record that a Prepare of the load `key` names ran to its end.
    pub fn store_prepared(&self, key: &TuningCacheKey) -> Result<(), String> {
        let path = Self::prepared_path(key);
        let records = self.store.writing().map_err(|error| self.failed("cannot take its lock to store a record", error))?;
        records
            .write(&path, b"{}")
            .map_err(|error| self.failed(&format!("cannot write {}", path.display()), error))
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
        assert!(cache.tuning::<serde_json::Value>(&key).unwrap().is_none());
        cache.store.writing().unwrap().write(&KernelCache::choice(&key), b"rewritten").unwrap();
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
        assert_eq!(placement(&KernelCache::choice(&key)), entry(0));
        assert_eq!(placement(&KernelCache::evidence_path(&key, "0123456789abcdef")), entry(1));
        assert_eq!(placement(&KernelCache::prepared_path(&key)), Eviction::Alone);
        let former_probes = Path::new(FORMER_ARTIFACTS).join(seismic::PROBE_NAMESPACE);
        for directory in KEPT.iter().map(Path::new).chain([former_probes.as_path()]) {
            assert_eq!(placement(directory), Eviction::Kept);
            assert_eq!(placement(&directory.join("digest.json")), Eviction::Kept);
        }
        // An earlier build's programs are entries, evicted whether or not
        // its probe answers have been carried.
        assert_eq!(placement(Path::new(FORMER_ARTIFACTS)), Eviction::Alone);
        assert_eq!(placement(&Path::new(FORMER_ARTIFACTS).join("metal").join("digest")), Eviction::Alone);
    }

    /// A tuning result whose entry is `entry`, so results of different
    /// entries have different evidence.
    fn result(entry: &str) -> TuningResult {
        TuningResult {
            disposition: seismic::TuningDisposition::QualifiedCandidate,
            tuning_identity: "device".into(),
            entry: entry.into(),
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
        }
    }

    /// The evidence files under `root`, temporary files apart.
    fn evidence_files(root: &Path) -> usize {
        fs::read_dir(root.join(EVIDENCE))
            .unwrap()
            .flatten()
            .filter(|item| !seismic_io::unpublished(&item.file_name()))
            .count()
    }

    /// A stored choice names the evidence it was made from; replacing the
    /// record removes the evidence the replaced choice named, storing the
    /// same evidence again removes nothing, and a load that took the first
    /// record still holds it whole.
    #[test]
    fn a_stored_choice_names_its_evidence_and_replacing_it_removes_the_superseded() {
        let root = scratch("pair");
        let cache = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let key = TuningCacheKey::of("material");
        cache.store_tuning(&key, &1u32, &result("first")).unwrap();
        let (choice, taken) = cache.tuning::<u32>(&key).unwrap().unwrap();
        assert_eq!((choice, taken.read().unwrap()), (1, result("first")));
        cache.store_tuning(&key, &2u32, &result("replaced")).unwrap();
        assert_eq!(evidence_files(&root), 1, "the store removes what the replaced choice named");
        cache.store_tuning(&key, &3u32, &result("replaced")).unwrap();
        assert_eq!(evidence_files(&root), 1, "evidence stored again is the new choice's");
        let (choice, evidence) = cache.tuning::<u32>(&key).unwrap().unwrap();
        assert_eq!((choice, evidence.read().unwrap()), (3, result("replaced")));
        assert_eq!(taken.read().unwrap(), result("first"), "a record a load took needs no file");
        fs::remove_dir_all(root).unwrap();
    }

    /// Evidence a dead writer left, which no choice names, is removed by
    /// the next eviction pass, whether its key's record is live or there
    /// is none, however much room there is; the live record stays whole.
    #[test]
    fn eviction_removes_evidence_a_dead_writer_left() {
        let root = scratch("abandoned");
        let cache = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let (live, abandoned) = (TuningCacheKey::of("live"), TuningCacheKey::of("abandoned"));
        cache.store_tuning(&live, &1u32, &result("live")).unwrap();
        let orphans = [KernelCache::evidence_path(&live, "0123456789abcdef"), KernelCache::evidence_path(&abandoned, "0123456789abcdef")];
        let records = cache.store.writing().unwrap();
        for orphan in &orphans {
            records.write(orphan, b"{}").unwrap();
        }
        drop(records);
        assert_eq!(evidence_files(&root), 3);
        // Opening evicts on a thread of its own.
        let _evicting = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while orphans.iter().any(|orphan| root.join(orphan).exists()) {
            assert!(std::time::Instant::now() < deadline, "eviction removes the evidence no choice names");
            std::thread::yield_now();
        }
        assert_eq!(evidence_files(&root), 1);
        let (choice, evidence) = cache.tuning::<u32>(&live).unwrap().expect("the live record stays");
        assert_eq!((choice, evidence.read().unwrap()), (1, result("live")));
        fs::remove_dir_all(root).unwrap();
    }

    /// A choice whose evidence is gone is a broken store, never a miss.
    #[test]
    fn a_choice_without_its_evidence_is_an_invariant_violation() {
        let root = scratch("orphan");
        let cache = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let key = TuningCacheKey::of("material");
        cache.store_tuning(&key, &1u32, &result("first")).unwrap();
        fs::remove_dir_all(root.join(EVIDENCE)).unwrap();
        let broken = cache.tuning::<u32>(&key).unwrap_err();
        assert!(broken.contains("invariant broken"), "{broken}");
        fs::remove_dir_all(root).unwrap();
    }

    /// Where the processes `processes_storing_one_unit_at_once_leave_every_choice_its_evidence`
    /// spawns write; unset, `a_spawned_tuning_writer` writes nothing.
    const WRITER_ROOT: &str = "MAGNITUDE_KERNEL_CACHE_WRITER_ROOT";
    /// Records each spawned writer stores.
    const WRITES: u32 = 400;

    /// A writer process of the test below, through a cache of its own:
    /// it replaces one unit's record, alternating between two whose
    /// evidence differs, so each write replaces a choice whose evidence
    /// another writer may be writing again.
    #[test]
    fn a_spawned_tuning_writer() {
        let Some(root) = std::env::var_os(WRITER_ROOT) else {
            return;
        };
        let cache = KernelCache::open(PathBuf::from(root), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let key = TuningCacheKey::of("raced");
        let (first, second) = (result("1"), result("2"));
        for write in 0..WRITES {
            let stored = match write % 2 {
                0 => cache.store_tuning(&key, &1u32, &first),
                _ => cache.store_tuning(&key, &2u32, &second),
            };
            stored.unwrap();
        }
    }

    /// Two processes replace one unit's record at once while a third reads
    /// it: every choice read names evidence that reads, and is its own, and
    /// once the writers end one choice and its evidence remain, each writer
    /// having removed what it replaced.
    #[test]
    fn processes_storing_one_unit_at_once_leave_every_choice_its_evidence() {
        let root = scratch("raced");
        let cache = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let key = TuningCacheKey::of("raced");
        let spawn = || {
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "kernel_cache::tests::a_spawned_tuning_writer", "--test-threads=1", "--quiet"])
                .env(WRITER_ROOT, &root)
                .stdout(std::process::Stdio::null())
                .spawn()
                .expect("a writer process starts")
        };
        let mut writers = [spawn(), spawn()];
        let mut read = 0u32;
        while writers.iter_mut().any(|writer| writer.try_wait().unwrap().is_none()) {
            if let Some((choice, evidence)) = cache.tuning::<u32>(&key).expect("a choice read names evidence that reads") {
                let evidence = evidence.read().expect("the evidence taken parses");
                assert_eq!(evidence.entry, choice.to_string(), "the evidence a choice names is its own");
                read += 1;
            }
        }
        for writer in &mut writers {
            assert!(writer.wait().unwrap().success(), "each writer ends whole");
        }
        assert!(read > 0, "the reader read while the writers wrote");
        let (choice, evidence) = cache.tuning::<u32>(&key).unwrap().expect("a record remains");
        assert_eq!(evidence.read().unwrap().entry, choice.to_string());
        assert_eq!(evidence_files(&root), 1, "no evidence outlives the choice that named it");
        fs::remove_dir_all(root).unwrap();
    }
}
