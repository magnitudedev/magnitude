//! The model store's files on disk: one [`Store`] at the store's root, which owns how they are
//! published, locked, set aside and swept ([`seismic_io::store`]).
//!
//! ```text
//! hub/models--<owner>--<name>/   a Hugging Face repository: its blobs (each named by its content,
//!                                written as `<key>.incomplete`, resumed in place, and renamed
//!                                onto its name once verified, never rewritten), their integrity
//!                                checkpoints, and the snapshots that link them
//! requested-downloads.json       the downloads asked for and not ended
//! catalog-affiliations.json      which installed packages are which catalog models'
//! readers/<repository>/<commit>/<file>/
//!                                each process's lease among the readers of one stored file
//! removals/<repository>/<commit>/<file>
//!                                the record a removal of one stored file holds
//! removed/<pid>.<n>              a committed removal's garbage, by repository ([`Garbage`])
//! ```
//!
//! A repository's record ([`repository_record`]) is held by its every download and removal, taken
//! only through [`hold_repository`]; each JSON file's by its every update.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use magnitude_service_contracts::{InventoryError, InventoryModel, ModelLocation, ModelSource};
use seismic_io::{Durability, Eviction, Examining, Privacy, Store};

/// The model store at `root`.
pub(crate) fn model_store(root: &Path) -> Store {
    Store::new(root.to_owned(), Privacy::Owner)
}

/// The record of the Hugging Face repository `repository`, under the store's root: the folder of
/// its files, whose lock every download and removal of them holds.
pub(crate) fn repository_record(repository: &str) -> PathBuf {
    Path::new("hub").join(crate::inventory::hf_repo_dir(repository))
}

/// The snapshot entry of `path` of `repository`'s snapshot `commit`, under the store's root.
pub(crate) fn snapshot_entry(repository: &str, commit: &str, path: &Path) -> PathBuf {
    repository_record(repository).join("snapshots").join(commit).join(path)
}

/// Hold the record of a repository ([`repository_record`]), waiting for it or not: the one way
/// it is taken. Its holder alone writes or deletes the repository's files, so taking it first
/// finishes what committed removals left of them ([`finish`]): a download never writes a file
/// that a removal's garbage still names. Refused, leaving the record free, when that garbage
/// cannot be deleted.
pub(crate) fn hold_repository(store: &Store, record: &Path, wait: bool) -> Result<Option<seismic_io::OwnedRecord>, InventoryError> {
    let held = match wait {
        true => Some(store.lock_owned(record).map_err(io_error)?),
        false => store.try_lock_owned(record).map_err(io_error)?,
    };
    let Some(held) = held else {
        return Ok(None);
    };
    finish(store, record)?;
    Ok(Some(held))
}

/// [`hold_repository`] of `repository` in the store at `root`, waiting, across `.await`s: taken
/// on the blocking pool, as another holder may hold it for a whole download.
pub(crate) async fn lock_repository(root: &Path, repository: &str) -> Result<seismic_io::OwnedRecord, InventoryError> {
    let (store, record) = (model_store(root), repository_record(repository));
    magnitude_service_utils::blocking::run(move || {
        Ok(hold_repository(&store, &record, true)?.expect("a waiting hold is taken"))
    })
    .await
}

/// Where committed removals' garbage is recorded, under the store's root ([`Garbage`]).
const REMOVED: &str = "removed";
/// The largest garbage record read back.
const GARBAGE_BYTES: u64 = 16 * 1024 * 1024;
/// Distinguishes the garbage records of this process.
static COMMITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What a removal leaves to delete, by the record of each repository it removes from
/// ([`repository_record`]). Its publication is the removal's commit (T7), one durable write:
/// before it nothing changed; from it on, the store's models it removes are gone, as the store's
/// scan and every hold derive from it ([`committed_entries`]), and what it names is garbage, which
/// the next holder of each repository's record deletes ([`finish`]).
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct Garbage(pub(crate) BTreeMap<PathBuf, RepositoryGarbage>);

/// A repository's part of a removal's garbage, under the store's root: the snapshot entries it
/// removes, and the blobs no other entry publishes.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct RepositoryGarbage {
    pub(crate) entries: BTreeSet<PathBuf>,
    pub(crate) blobs: BTreeSet<PathBuf>,
}

impl Garbage {
    /// Commit the removal (T7). Its caller holds the repositories' records and the removal
    /// records of the files it deletes.
    pub(crate) fn commit(&self, store: &Store) -> Result<(), InventoryError> {
        let record = Path::new(REMOVED).join(format!(
            "{}.{}",
            std::process::id(),
            COMMITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let bytes = serde_json::to_vec(self).map_err(io_error)?;
        store
            .publish(&record, Durability::Durable, |file| std::io::Write::write_all(file, &bytes))
            .map_err(io_error)
    }

    fn parse(record: &Path, bytes: &[u8]) -> Result<Self, InventoryError> {
        serde_json::from_slice(bytes).map_err(|error| {
            InventoryError::Io(format!("the removal record {} does not parse: {error}", record.display()))
        })
    }
}

impl RepositoryGarbage {
    /// Delete it (T8): the bytes its blobs freed. Fails at the first file that cannot be deleted.
    fn delete(&self, root: &Path, repository: &Path) -> Result<u64, InventoryError> {
        let gone = |path: &Path| match fs::remove_file(path) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(InventoryError::Io(format!(
                "removed model garbage {} is not deleted: {error}",
                path.display()
            ))),
            _ => Ok(()),
        };
        for entry in &self.entries {
            gone(&root.join(entry))?;
        }
        let mut freed = 0_u64;
        for blob in &self.blobs {
            let blob = root.join(blob);
            let size = blob.symlink_metadata().map_or(0, |metadata| metadata.len());
            gone(&blob)?;
            freed = freed.saturating_add(size);
            let mut checkpoint = blob.into_os_string();
            checkpoint.push(".integrity");
            gone(Path::new(&checkpoint))?;
        }
        let repo_root = root.join(repository);
        for entry in &self.entries {
            if let Some(parent) = root.join(entry).parent() {
                crate::service::remove_empty_parents(parent, &repo_root.join("snapshots"));
            }
        }
        prune_empty_repository(&repo_root);
        Ok(freed)
    }
}

/// The snapshot entries every committed removal names, under the store's root: models of the
/// store gone, whatever is left of their files on disk.
pub(crate) fn committed_entries(store: &Store) -> Result<BTreeSet<PathBuf>, InventoryError> {
    let mut entries = BTreeSet::new();
    for name in store.list(Path::new(REMOVED)).map_err(io_error)? {
        let record = Path::new(REMOVED).join(name);
        if let Some(bytes) = store.read(&record, GARBAGE_BYTES).map_err(io_error)? {
            for repository in Garbage::parse(&record, &bytes)?.0.into_values() {
                entries.extend(repository.entries);
            }
        }
    }
    Ok(entries)
}

/// Delete what committed removals left of the repository at `record` (T8), whose record its
/// caller holds: the bytes freed. Each removal's record lets the repository go once its part is
/// deleted, and goes with its last; a part that cannot be deleted stays recorded, and fails it.
pub(crate) fn finish(store: &Store, record: &Path) -> Result<u64, InventoryError> {
    let mut freed = 0_u64;
    for name in store.list(Path::new(REMOVED)).map_err(io_error)? {
        let path = Path::new(REMOVED).join(name);
        let held = store.lock(&path).map_err(io_error)?;
        let Some(bytes) = held.read(GARBAGE_BYTES).map_err(io_error)? else {
            continue;
        };
        let mut garbage = Garbage::parse(&path, &bytes)?;
        let Some(repository) = garbage.0.remove(record) else {
            continue;
        };
        freed = freed.saturating_add(repository.delete(store.root(), record)?);
        match garbage.0.is_empty() {
            true => held.remove().map_err(io_error)?,
            false => held
                .write(&serde_json::to_vec(&garbage).map_err(io_error)?, Durability::Durable)
                .map_err(io_error)?,
        }
    }
    Ok(freed)
}

/// Finish every committed removal whose repositories no holder holds now, as a crash leaves
/// them, at the store's opening; a repository held is finished by its holder.
fn finish_removals(store: &Store) {
    let mut repositories = BTreeSet::new();
    let listed = store.list(Path::new(REMOVED)).map_err(io_error).and_then(|names| {
        for name in names {
            let record = Path::new(REMOVED).join(name);
            if let Some(bytes) = store.read(&record, GARBAGE_BYTES).map_err(io_error)? {
                repositories.extend(Garbage::parse(&record, &bytes)?.0.into_keys());
            }
        }
        Ok(())
    });
    if let Err(error) = listed {
        tracing::warn!(%error, "committed removals are not finished at the store's opening");
    }
    for repository in repositories {
        if let Err(error) = hold_repository(store, &repository, false) {
            tracing::warn!(repository = %repository.display(), %error, "a committed removal is not finished");
        }
    }
}

/// One file of Magnitude's store, the unit a removal deletes: `path` of `repository`'s snapshot
/// `commit`. Every resolution of a model served from it holds a lease among its readers
/// ([`Store::lease`]), one per process, for as long as what it resolved lives (a cached
/// configuration, a load); a removal holds its removal record from its claim to its end, which a
/// reader takes shared only while it takes its lease. Each tells the other apart ([`Readers`]): a
/// removal of another file, of the same repository or not, never meets it. Apart from the
/// repository's own record, which a download holds for as long as it runs, so a load never waits
/// for a download.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct StoredFile {
    /// Its snapshot entry.
    entry: PathBuf,
    /// The directory of the leases of its readers.
    readers: PathBuf,
    /// The record a removal of it holds.
    removal: PathBuf,
}

impl StoredFile {
    pub(crate) fn new(repository: &str, commit: &str, path: &Path) -> Self {
        let file = Path::new(&crate::inventory::hf_repo_dir(repository)).join(commit).join(path);
        Self {
            entry: snapshot_entry(repository, commit, path),
            readers: Path::new("readers").join(&file),
            removal: Path::new("removals").join(file),
        }
    }
}

/// The owner of each stored file's lifecycle in this process ([`StoredFile`]). A file is
/// Installed, held by n resolutions here behind one lease of this process (and by any other
/// process's leases); Removing, held by a removal, which nothing acquires; or Removed, named by a
/// committed removal's [`Garbage`]. Every acquirer of a file here takes it through
/// [`Readers::hold`], and these are its transitions:
///
/// - T1 acquire ([`Readers::hold`]): refused while the file is Removing, here or in another
///   process, or Removed.
/// - T2 release ([`ReadersHold`] dropped): the last hold here drops the lease.
/// - T3 begin ([`Readers::begin_removal`]): Installed to Removing, all the removal's files or
///   none, without waiting: refused at once, naming the holder, while a download holds a
///   repository, another process reads a file, or another removal has one. Waiting on another
///   process would be unbounded.
/// - T4 release the resident: the residency lets go of the model loading or serving the files.
/// - T5 roll back ([`Removal`] dropped before T7): Removing to Installed, nothing changed.
/// - T6 drain ([`Removal::drained`]): wait for the holds left here, finite since none begins.
/// - T7 commit ([`Garbage::commit`]): Removing to Removed, one durable write.
/// - T8 delete ([`finish`]): the garbage goes, by the next holder of each repository's record.
#[derive(Debug, Default)]
pub(crate) struct Readers {
    /// Each file ever held or removed here. The map's lock is never held across a file's locks;
    /// each file's own is, against only that file's other acquirers.
    files: std::sync::Mutex<std::collections::BTreeMap<StoredFile, Arc<std::sync::Mutex<FileState>>>>,
    /// Told whenever a file's last hold here goes.
    released: tokio::sync::Notify,
}

/// A stored file's state here.
#[derive(Debug, Default)]
struct FileState {
    /// The resolutions here that hold it.
    holders: usize,
    /// This process's lease among its readers, while `holders` > 0.
    lease: Option<seismic_io::Lease>,
    /// Whether a removal here has it.
    removing: bool,
}

type Entry = Arc<std::sync::Mutex<FileState>>;

/// The state behind `entry`; every holder leaves it whole, so a panic between its steps leaves
/// none half done.
fn state(entry: &Entry) -> std::sync::MutexGuard<'_, FileState> {
    entry.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Readers {
    fn entry(&self, file: &StoredFile) -> Entry {
        let mut files = self.files.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(files.entry(file.clone()).or_default())
    }

    /// A hold of the readers of each file `model` is served from, where Magnitude's store keeps
    /// them (T1), taken on the blocking pool, never waiting: refused, as busy, while a removal,
    /// here or in another process, has one, and as not found once a removal committed one. None
    /// for a model elsewhere, whose files Magnitude never removes.
    pub(crate) async fn hold(
        self: &Arc<Self>,
        root: &Path,
        model: &InventoryModel,
    ) -> Result<Option<ReadersHold>, InventoryError> {
        let (ModelLocation::MagnitudeCache { components, .. }, ModelSource::HuggingFace { repository, commit, .. }) =
            (&model.location, &model.source)
        else {
            return Ok(None);
        };
        let files = components
            .iter()
            .map(|component| StoredFile::new(repository, commit, &component.path))
            .collect::<BTreeSet<_>>();
        let (store, readers, id) = (model_store(root), Arc::clone(self), model.id.0.clone());
        magnitude_service_utils::blocking::run(move || {
            let mut hold = ReadersHold { readers: Arc::clone(&readers), entries: Vec::new() };
            let mut committed = None;
            for file in &files {
                let entry = readers.entry(file);
                let mut state = state(&entry);
                if state.removing {
                    return Err(InventoryError::Busy(id));
                }
                if state.holders == 0 {
                    // The removal record, shared while the lease is taken: a removal in another
                    // process either has it, and this hold is refused, or finds the lease.
                    let _removal = store
                        .try_share_owned(&file.removal)
                        .map_err(io_error)?
                        .ok_or_else(|| InventoryError::Busy(id.clone()))?;
                    // A removal that committed and was not finished holds no record any more.
                    if committed.is_none() {
                        committed = Some(committed_entries(&store)?);
                    }
                    if committed.as_ref().is_some_and(|committed| committed.contains(&file.entry)) {
                        return Err(InventoryError::NotFound(id));
                    }
                    state.lease = Some(store.lease(&file.readers).map_err(io_error)?);
                }
                state.holders += 1;
                drop(state);
                hold.entries.push(entry);
            }
            Ok(Some(hold))
        })
        .await
    }

    /// Take `files`, and the records of `repositories`, from every acquirer for a removal (T3),
    /// on the blocking pool: all at once, or none, refused as busy, naming the holder, while a
    /// download holds a repository, another process reads a file, or another removal has one.
    pub(crate) async fn begin_removal(
        self: &Arc<Self>,
        root: &Path,
        repositories: &BTreeSet<PathBuf>,
        files: &BTreeSet<StoredFile>,
    ) -> Result<Removal, InventoryError> {
        let (store, readers, repositories, files) = (model_store(root), Arc::clone(self), repositories.clone(), files.clone());
        magnitude_service_utils::blocking::run(move || {
            let mut removal = Removal { readers: Arc::clone(&readers), entries: Vec::new(), _records: Vec::new() };
            for repository in &repositories {
                let record = hold_repository(&store, repository, false)?;
                removal._records.push(record.ok_or_else(|| held(repository, "is being downloaded"))?);
            }
            for file in &files {
                let entry = readers.entry(file);
                let mut state = state(&entry);
                if state.removing {
                    return Err(held(&file.removal, "is being removed"));
                }
                state.removing = true;
                let own = state.lease.as_ref().map(|lease| lease.record().to_owned());
                drop(state);
                removal.entries.push(Arc::clone(&entry));
                let record = store.try_lock_owned(&file.removal).map_err(io_error)?;
                removal._records.push(record.ok_or_else(|| held(&file.removal, "is being read or removed by another process"))?);
                let leases = store.leases(&file.readers).map_err(io_error)?;
                if let Some(lease) = leases.iter().find(|lease| Some(*lease) != own.as_ref()) {
                    // A lease is named by its holder's process ([`Store::lease`]).
                    let holder = lease.file_name().and_then(|name| name.to_str()).and_then(|name| name.split('.').next()).unwrap_or("?");
                    return Err(held(&file.removal, &format!("is being read by process {holder}")));
                }
            }
            Ok(removal)
        })
        .await
    }
}

/// A removal's refusal: what `record` names, under the store's root, `holder` says.
fn held(record: &Path, holder: &str) -> InventoryError {
    let named = record.iter().skip(1).collect::<PathBuf>();
    InventoryError::Busy(format!("{} {holder}", named.display()))
}

/// A resolution's hold of the readers of the files it reads ([`Readers::hold`]).
pub(crate) struct ReadersHold {
    readers: Arc<Readers>,
    entries: Vec<Entry>,
}

impl Drop for ReadersHold {
    fn drop(&mut self) {
        for entry in &self.entries {
            let mut state = state(entry);
            state.holders -= 1;
            if state.holders == 0 {
                state.lease = None;
            }
        }
        self.readers.released.notify_waiters();
    }
}

/// A removal's files, Removing ([`Readers::begin_removal`]): no resolution here begins to hold
/// one, and none in another process while their records are held. Dropping it rolls them back to
/// Installed (T5), or, once they are gone, leaves the owner as for a file never held.
#[must_use = "the files are the removal's only while this lives"]
pub(crate) struct Removal {
    readers: Arc<Readers>,
    entries: Vec<Entry>,
    _records: Vec<seismic_io::OwnedRecord>,
}

impl Removal {
    /// Wait until no resolution here holds any of the files (T6): finite, since none begins.
    pub(crate) async fn drained(&self) {
        loop {
            let mut notified = std::pin::pin!(self.readers.released.notified());
            notified.as_mut().enable();
            if self.entries.iter().all(|entry| state(entry).holders == 0) {
                return;
            }
            notified.await;
        }
    }
}

impl Drop for Removal {
    fn drop(&mut self) {
        for entry in &self.entries {
            state(entry).removing = false;
        }
    }
}

/// Publish `bytes` as the file at `path` of the model store, synced, its owner's alone: written
/// beside it under a temporary name, which the store's sweep removes if the writer dies, and
/// renamed into place. A publication takes no lock and names only the file's directory; its
/// caller holds the lock of the record the file is of.
pub(crate) fn publish_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let (directory, name) = beside(path);
    Store::new(directory.to_owned(), Privacy::Owner)
        .publish(Path::new(name), Durability::Durable, |file| std::io::Write::write_all(file, bytes))
}

/// An empty directory staged beside `target`, a folder of the model store, its owner's alone,
/// which `Staged::publish` renames into place whole; a stage a dead writer left is the store's
/// sweep's. Staging names only the target's directory; its caller holds the lock of the record
/// `target` is of.
pub(crate) fn stage(target: &Path) -> std::io::Result<seismic_io::Staged> {
    let (directory, name) = beside(target);
    Store::new(directory.to_owned(), Privacy::Owner).stage(Path::new(name))
}

/// Set what is at `path` of the model store aside, under a temporary name beside it, which the
/// store's sweep removes once abandoned; nothing there is no failure. Its caller holds the lock of
/// the record `path` is of.
pub(crate) fn quarantine_sync(path: &Path) -> std::io::Result<()> {
    if let Err(error) = path.symlink_metadata() {
        return match error.kind() {
            std::io::ErrorKind::NotFound => Ok(()),
            _ => Err(error),
        };
    }
    let (directory, name) = beside(path);
    Store::new(directory.to_owned(), Privacy::Owner).quarantine(Path::new(name))
}

/// [`quarantine_sync`], on the blocking pool.
pub(crate) async fn quarantine(path: &Path) -> std::io::Result<()> {
    let path = path.to_owned();
    magnitude_service_utils::blocking::run(move || quarantine_sync(&path)).await
}

/// The directory `path` lies in, and its name there.
fn beside(path: &Path) -> (&Path, &std::ffi::OsStr) {
    (
        path.parent().expect("a model store file lies in a directory"),
        path.file_name().expect("a model store file has a name"),
    )
}

pub(crate) async fn ensure_store_layout(root: &Path) -> Result<(), InventoryError> {
    match tokio::fs::symlink_metadata(root).await {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
        Ok(_) => {
            return Err(InventoryError::Io(format!(
                "model store root is not a real directory: {}",
                root.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            tokio::fs::create_dir_all(root).await.map_err(io_error)?;
        }
        Err(error) => return Err(io_error(error)),
    }
    ensure_owned_directory(&root.join("hub")).await?;
    let committed = model_store(root);
    magnitude_service_utils::blocking::run(move || finish_removals(&committed)).await;
    sweep(root.to_owned());
    Ok(())
}

/// Sweep the store at `root` on a thread of its own, off the opening path: what dead writers left
/// (temporary files, stages, what was set aside) once abandoned, the locks of records that are
/// gone, and what older builds left that nothing reads ([`legacy`]). It keeps every file of the
/// store, none of them an entry to evict, and every file whose lock a holder holds (a lease of a
/// load's), and runs beside downloads and loads; one that finds the store held passes, and the
/// next opening sweeps.
fn sweep(root: PathBuf) {
    let swept = move || match model_store(&root).evict(u64::MAX, Duration::ZERO, &legacy) {
        Ok(true) => {}
        Ok(false) => tracing::debug!(root = %root.display(), "the model store is held; its next opening sweeps it"),
        Err(error) => tracing::warn!(root = %root.display(), %error, "the model store is not swept"),
    };
    if let Err(error) = seismic_threads::spawn("magnitude-model-store-sweep", swept) {
        tracing::warn!(%error, "the model store's sweep does not start; its next opening sweeps it");
    }
}

/// How the sweep places the store's files: what an older build left that nothing reads is
/// unreachable: `locks/` (its per-repository lock files, before the store's own), what it set
/// aside as `<name>.invalid-<id>`, and its fixed snapshot stages, `.incomplete/snapshot-<commit>`.
/// Everything else is an entry of its own, which a sweep of unbounded capacity keeps, and a
/// directory so placed is walked. A directory emptied of what nothing reads goes with it.
fn legacy(path: &Path, _: &Examining<'_>) -> Eviction {
    let set_aside = path.iter().any(|component| set_aside_name(&component.to_string_lossy()));
    let staged = path
        .iter()
        .zip(path.iter().skip(1))
        .any(|(folder, stage)| folder == ".incomplete" && stage.to_string_lossy().starts_with("snapshot-"));
    match path.starts_with("locks") || set_aside || staged {
        true => Eviction::Unreachable,
        false => Eviction::Alone,
    }
}

/// Whether `name` is one an older build set something aside under: its own name, its extension
/// replaced by `invalid-<nanoseconds>-<sequence>` or `invalid-partial_<32 hex digits>`.
fn set_aside_name(name: &str) -> bool {
    let Some((_, extension)) = name.rsplit_once('.') else {
        return false;
    };
    let Some(mark) = extension.strip_prefix("invalid-") else {
        return false;
    };
    let digits = |text: &str| !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    match mark.strip_prefix("partial_") {
        Some(id) => id.len() == 32 && id.bytes().all(|byte| byte.is_ascii_hexdigit()),
        None => mark.split_once('-').is_some_and(|(time, sequence)| digits(time) && digits(sequence)),
    }
}

pub(crate) async fn ensure_owned_directory(path: &Path) -> Result<(), InventoryError> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => return Ok(()),
        Ok(_) => quarantine(path).await.map_err(io_error)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error(error)),
    }
    // Every folder created on the way is its owner's alone too: an eviction of the store may
    // have removed an emptied parent (`hub`) since the store's layout was made.
    let path = path.to_owned();
    magnitude_service_utils::blocking::run(move || Privacy::Owner.directory(&path))
        .await
        .map_err(io_error)
}

fn io_error(error: impl std::fmt::Display) -> InventoryError {
    InventoryError::Io(error.to_string())
}

/// Removes a repository folder of Magnitude's own store that no longer holds a model: once no
/// snapshot remains, its empty `snapshots`, `.incomplete` and `refs` folders, and a `blobs` folder
/// left with only integrity checkpoints of blobs that are gone, then the folder itself. A partial
/// download, a blob, or anything else unexpected keeps the folder. Callers hold the repository
/// lock.
pub(crate) fn prune_empty_repository(repo_root: &Path) {
    let entries = |path: &Path| -> Option<Vec<std::path::PathBuf>> {
        match fs::read_dir(path) {
            Ok(entries) => Some(entries.filter_map(|entry| entry.ok().map(|entry| entry.path())).collect()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Some(Vec::new()),
            Err(_) => None,
        }
    };
    let Some(snapshots) = entries(&repo_root.join("snapshots")) else {
        return;
    };
    if !snapshots.is_empty() {
        return;
    }
    let blobs = repo_root.join("blobs");
    if let Some(blob_entries) = entries(&blobs) {
        for checkpoint in &blob_entries {
            let orphaned = checkpoint
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| name.strip_suffix(".integrity"))
                .is_some_and(|blob| {
                    !blobs.join(blob).exists() && !blobs.join(format!("{blob}.incomplete")).exists()
                });
            if orphaned {
                let _ = fs::remove_file(checkpoint);
            }
        }
    }
    for folder in ["snapshots", ".incomplete", "refs", "blobs"] {
        let _ = fs::remove_dir(repo_root.join(folder));
    }
    let _ = fs::remove_dir(repo_root);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn owned_layout_replaces_invalid_child_nodes_without_touching_their_targets() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let root = temporary.path().join("models");
        let outside = temporary.path().join("outside");
        fs::create_dir_all(&root).expect("store root");
        fs::write(&outside, b"outside").expect("outside file");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, root.join("hub")).expect("invalid hub link");
        #[cfg(not(unix))]
        fs::write(root.join("hub"), b"invalid").expect("invalid hub file");

        ensure_store_layout(&root).await.expect("reconciled layout");

        assert!(root.join("hub").is_dir());
        assert_eq!(
            fs::read(outside).expect("outside file retained"),
            b"outside"
        );
        let set_aside = model_store(&root).list(Path::new("")).expect("listing");
        assert_eq!(set_aside, ["hub"], "what held the hub's place is set aside under a temporary name");
    }

    /// A removal's garbage is deleted by the next holder of each of its repositories' records,
    /// before the holder writes anything there: a download of the same commit never has what it
    /// writes deleted. Each holder deletes only its repository's part, and the record goes with
    /// the last.
    #[test]
    fn a_repositorys_next_holder_finishes_its_committed_garbage_first() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let root = temporary.path();
        let store = model_store(root);
        let (first, second) = (repository_record("owner/first"), repository_record("owner/second"));
        let mut garbage = Garbage::default();
        for record in [&first, &second] {
            let (entry, blob) = (record.join("snapshots/commit/model.gguf"), record.join("blobs/0123"));
            for path in [&entry, &blob] {
                fs::create_dir_all(root.join(path).parent().unwrap()).unwrap();
                fs::write(root.join(path), b"bytes").unwrap();
            }
            garbage.0.insert(record.clone(), RepositoryGarbage { entries: [entry].into(), blobs: [blob].into() });
        }
        garbage.commit(&store).unwrap();
        assert_eq!(committed_entries(&store).unwrap().len(), 2);

        let held = hold_repository(&store, &first, true).unwrap().expect("a waiting hold");
        assert!(!root.join(&first).join("snapshots/commit/model.gguf").exists());
        assert!(!root.join(&first).join("blobs/0123").exists());
        assert!(root.join(&second).join("blobs/0123").exists(), "another repository's part waits for its holder");
        assert_eq!(committed_entries(&store).unwrap(), [second.join("snapshots/commit/model.gguf")].into());
        drop(held);

        let _held = hold_repository(&store, &second, false).unwrap().expect("a free record");
        assert!(!root.join(&second).join("blobs/0123").exists());
        assert!(store.list(Path::new(REMOVED)).unwrap().is_empty(), "the record goes with its last part");
    }

    /// The sweep removes what older builds left that nothing reads (their lock files, what they
    /// set aside, their fixed snapshot stages), and keeps every file of the store, a partial
    /// download among them.
    #[test]
    fn the_sweep_removes_what_older_builds_left_and_keeps_the_stores_files() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let root = temporary.path();
        let repository = root.join("hub/models--owner--name");
        let kept = [
            repository.join("blobs/abc"),
            repository.join("blobs/def.incomplete"),
            repository.join("blobs/def.integrity"),
            repository.join("snapshots/commit/model.invalid-name.gguf"),
            root.join("requested-downloads.json"),
        ];
        let left = [
            root.join("locks/models--owner--name.lock"),
            repository.join("blobs/abc.invalid-1731800000000000000-3"),
            repository.join(format!("snapshots/commit.invalid-partial_{}/model.gguf", "0a".repeat(16))),
            repository.join(".incomplete/snapshot-commit/model.gguf"),
        ];
        for path in kept.iter().chain(&left) {
            fs::create_dir_all(path.parent().unwrap()).expect("folder");
            fs::write(path, b"bytes").expect("file");
        }
        assert!(model_store(root).evict(u64::MAX, Duration::ZERO, &legacy).expect("swept"));
        for path in &kept {
            assert!(path.exists(), "{} is kept", path.display());
        }
        for path in &left {
            assert!(!path.exists(), "{} is removed", path.display());
            if path.parent() != Some(&repository.join("blobs")) {
                assert!(!path.parent().unwrap().exists(), "the folder it emptied goes with it");
            }
        }
        assert!(repository.join("blobs").exists(), "a folder still holding files stays");
    }

    #[test]
    fn an_empty_repository_folder_is_pruned_and_one_with_a_partial_download_is_kept() {
        let temporary = tempfile::tempdir().unwrap();
        let empty = temporary.path().join("models--owner--empty");
        fs::create_dir_all(empty.join("snapshots")).unwrap();
        fs::create_dir_all(empty.join(".incomplete")).unwrap();
        fs::create_dir_all(empty.join("blobs")).unwrap();
        fs::write(empty.join("blobs").join("abc.integrity"), b"{}").unwrap();
        prune_empty_repository(&empty);
        assert!(!empty.exists());

        let resuming = temporary.path().join("models--owner--resuming");
        fs::create_dir_all(resuming.join("blobs")).unwrap();
        fs::write(resuming.join("blobs").join("abc.incomplete"), b"part").unwrap();
        fs::write(resuming.join("blobs").join("abc.integrity"), b"{}").unwrap();
        prune_empty_repository(&resuming);
        assert!(resuming.join("blobs").join("abc.incomplete").exists());
        assert!(resuming.join("blobs").join("abc.integrity").exists());

        let published = temporary.path().join("models--owner--published");
        fs::create_dir_all(published.join("snapshots").join("commit")).unwrap();
        prune_empty_repository(&published);
        assert!(published.join("snapshots").join("commit").exists());
    }
}
