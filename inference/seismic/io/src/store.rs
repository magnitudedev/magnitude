//! Shared on-disk stores ([`Store`]): the one way any store that several
//! threads or processes write publishes, locks, updates, appends, evicts and
//! opens its files.
//!
//! A store is a directory, its root. Every process that names the same root
//! shares it, and the store owns everything under the root that is not its
//! embedder's own files:
//!
//! ```text
//! .lock              the store's lock ([`Store::shared`], [`Store::exclusive`])
//! .locks/<record>    each record's lock ([`Store::lock`]), named as the record
//! .<name>.<pid>.<n>.tmp
//!                    a write under way beside its target ([`Temporary`]), or a
//!                    directory being staged ([`Store::stage`]) or set aside
//!                    ([`Store::quarantine`])
//! ```
//!
//! **Publication.** A file is written to a temporary file of its own beside
//! its target, synced where the publication is [`Durability::Durable`], and
//! renamed over the target in one step: a reader finds the target as it was
//! or as it now is, never part written. The rename replaces what is there
//! (`rename(2)`; on Windows `std::fs::rename`, which replaces an existing
//! file, with POSIX semantics where the volume has them). A temporary name
//! is the process's id and a counter of the whole process, created
//! exclusively, so no two writers share one, whichever store, thread or
//! process they are; one a writer that died left is removed by the store's
//! next sweep ([`Store::evict`]) once it is [`ABANDONED`]. A directory
//! written over many steps is published the same way, from a stage of its
//! own ([`Staged`]).
//!
//! **Locks.** Two kinds, both `std::fs::File::lock`, which holds across
//! threads (each through a file of its own) and processes alike:
//!
//! - The store's lock: held shared ([`Shared`]) to read a consistent set of
//!   files, exclusively ([`Exclusive`]) to read, write, append and remove
//!   them, and by eviction from its listing to its last removal. Each such
//!   hold is brief: no holder keeps it across work of unbounded length.
//! - A record's lock ([`Record`]), held from the read to the publish of an
//!   update, and by every write and removal of the record, so no path
//!   reaches a record unlocked. It is taken under the store's lock held
//!   shared, which its holder then lets go: no eviction runs while a
//!   record's lock is being taken, and eviction leaves every file whose
//!   record's lock is held. A record's removal removes its lock file under
//!   that lock; a waiter that then takes the removed file finds the name no
//!   longer its and takes the lock again. A record's readers may hold its
//!   lock shared ([`Store::try_share_owned`]), refused while an exclusive
//!   holder holds it, which in turn waits for them or refuses at them
//!   ([`Store::try_lock`]).
//! - A lease ([`Lease`]): a record of a process's own among the holders of
//!   a directory, its lock held for as long as the lease lives, so another
//!   process tells the live holders from the gone ([`Store::leases`]), and
//!   each holder from every other, which one shared lock cannot.
//!
//! No holder waits for the store's lock while it holds it: a record's
//! holder holds none of it, and eviction never waits for it but passes when
//! another holds it. Nothing here depends on whether a platform lets a new
//! shared lock past an exclusive one waiting. A record's holder takes
//! another record's only where no holder of that one ever takes the first's
//! (the model store's repositories before the files it keeps beside them).
//! A lock that cannot be taken is an error of the access that wanted it,
//! never an absent file.
//!
//! **Files that are never rewritten** (content addresses) need no lock to be
//! read or written: [`Store::put_if_absent`] writes one once, and
//! [`Store::hold`] opens one under its own lock held shared, which keeps
//! eviction from removing it while the handle lives: a platform that opens
//! the file again by its path (Metal's archives) finds it there.
//!
//! **Eviction** ([`Store::evict`]) removes the least recently used entries
//! beyond a capacity, as the embedder's [`Layout`] groups the files, under
//! the store's lock held exclusively, and passes over every file whose lock
//! a holder holds; with it go dead writers' temporary files and the lock
//! files no holder holds of records that are gone. It is opportunistic: one
//! that finds the store's lock held does nothing, and the next begins again.
//! Reading a file through the store's lock, fetching it ([`Store::fetch`])
//! or holding it refreshes its modification time, its use time.
//!
//! [`Privacy::Owner`] keeps every directory the store creates readable by
//! its owner alone (`0700`) and every file it writes (`0600`).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Result, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// Whether a publication survives the host's crash once it returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Durability {
    /// The file's bytes and its rename are synced before it returns.
    Durable,
    /// Visible whole or not at all, but a crash may lose it, or leave it
    /// torn on a filesystem that orders a rename before the data it names:
    /// for what every reader checks whole and can form again.
    Volatile,
}

/// Who may read what a store keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Privacy {
    /// As the process's umask leaves it.
    Shared,
    /// Its owner alone: directories `0700`, files `0600`.
    Owner,
}

impl Privacy {
    /// Options that open a file as this privacy says: an owner's is `0600`
    /// when created and never a link to a file elsewhere.
    #[cfg(unix)]
    pub fn options(self) -> OpenOptions {
        use std::os::unix::fs::OpenOptionsExt;
        let mut options = OpenOptions::new();
        if self == Privacy::Owner {
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        options
    }

    /// Options that open a file as this privacy says: on Windows a file
    /// inherits its directory's access control.
    #[cfg(not(unix))]
    pub fn options(self) -> OpenOptions {
        OpenOptions::new()
    }

    /// Create the directory at `path` and every one above it, as this
    /// privacy says: each one created, and `path` whether created or not.
    pub fn directory(self, path: &Path) -> Result<()> {
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        if self == Privacy::Owner {
            use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
            builder.mode(0o700).create(path)?;
            return fs::set_permissions(path, fs::Permissions::from_mode(0o700));
        }
        builder.create(path)
    }
}

/// A temporary file older than this belongs to a writer that died.
pub const ABANDONED: Duration = Duration::from_secs(60 * 60);

/// The suffix of every temporary name.
const UNPUBLISHED: &str = ".tmp";
/// The store's lock, under its root.
const LOCK: &str = ".lock";
/// The directory under the root of every record's lock.
const LOCKS: &str = ".locks";

/// How long a refusal to open or name a lock file is taken for the file's
/// pending deletion, on Windows, before it is reported ([`Pending`]).
const PENDING: Duration = Duration::from_secs(1);

/// Distinguishes every temporary name of this process.
static TEMPORARIES: AtomicU64 = AtomicU64::new(0);

/// Distinguishes every lease of this process.
static LEASES: AtomicU64 = AtomicU64::new(0);

/// Whether the file named `name` is a temporary one, which no reader reads.
fn unpublished(name: &OsStr) -> bool {
    let name = name.to_string_lossy();
    name.starts_with('.') && name.ends_with(UNPUBLISHED)
}

/// Whether `name` is one of the store's own files under its root.
fn owned(name: &OsStr) -> bool {
    name == LOCK || name == LOCKS
}

/// A name in `directory`, of `stem`, no other live temporary holds: the
/// process's id and a counter of the whole process.
fn temporary_name(directory: &Path, stem: &OsStr) -> PathBuf {
    let stem = stem.to_string_lossy();
    directory.join(format!(".{stem}.{}.{}{UNPUBLISHED}", std::process::id(), TEMPORARIES.fetch_add(1, Ordering::Relaxed)))
}

/// The directory `path` lives in.
fn parent(path: &Path) -> &Path {
    path.parent().expect("a stored path lives in a directory")
}

/// Set the use time (the modification time) of the file or directory at
/// `path` to `when`.
fn touched(path: &Path, when: SystemTime) -> Result<()> {
    let mut options = OpenOptions::new();
    #[cfg(not(windows))]
    options.read(true);
    // Setting a time takes `FILE_WRITE_ATTRIBUTES`, and opening a directory
    // `FILE_FLAG_BACKUP_SEMANTICS`.
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.access_mode(0x100).custom_flags(0x0200_0000);
    }
    options.open(path)?.set_modified(when)
}

/// Refresh the use time of the file at `path`: a stale one only makes its
/// entry an earlier eviction candidate.
fn used(path: &Path) {
    let _ = touched(path, SystemTime::now());
}

/// The bytes of the regular file at `path`, none if no file is there. A
/// file larger than `largest`, or anything at `path` that is no regular
/// file, is an error.
fn read_bounded(path: &Path, largest: u64) -> Result<Option<Vec<u8>>> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let invalid = |why: String| std::io::Error::new(ErrorKind::InvalidData, format!("{}: {why}", path.display()));
    if !metadata.is_file() {
        return Err(invalid("not a regular file".into()));
    }
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let mut bytes = Vec::new();
    file.take(largest.saturating_add(1)).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > largest {
        return Err(invalid(format!("larger than the {largest} bytes it is read to")));
    }
    Ok(Some(bytes))
}

/// [`read_bounded`], marking the file used.
fn read_used(path: &Path, largest: u64) -> Result<Option<Vec<u8>>> {
    let bytes = read_bounded(path, largest)?;
    if bytes.is_some() {
        used(path);
    }
    Ok(bytes)
}

/// Remove the file at `path`, none being there no failure.
fn removed(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(error) if error.kind() != ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

/// Sync `directory`, so a rename in it survives a crash.
#[cfg(unix)]
fn synced_directory(directory: &Path) -> Result<()> {
    File::open(directory)?.sync_all()
}

/// Windows journals a rename with the volume's metadata, and opens no
/// directory as a file to sync it.
#[cfg(not(unix))]
fn synced_directory(_: &Path) -> Result<()> {
    Ok(())
}

/// A temporary file: under a name no other live temporary holds, which
/// marks it as one to every store's sweep, removed when dropped. What a
/// store writes before it publishes it, and what a probe of a volume writes
/// and removes.
#[derive(Debug)]
pub struct Temporary {
    path: PathBuf,
}

impl Temporary {
    /// A new empty file in `directory`, named of `stem`, created
    /// exclusively, as `privacy` says.
    pub fn create(directory: &Path, stem: &OsStr, privacy: Privacy) -> Result<(Self, File)> {
        loop {
            let path = temporary_name(directory, stem);
            let mut options = privacy.options();
            options.read(true).write(true).create_new(true);
            // Only a dead writer of the same process id leaves the name taken.
            match options.open(&path) {
                Ok(file) => return Ok((Self { path }, file)),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
    }

    /// A name in `directory`, of `stem`, that nothing holds yet: for what
    /// creates the file itself (a clone), which is removed when this is
    /// dropped.
    pub fn reserve(directory: &Path, stem: &OsStr) -> Self {
        Self { path: temporary_name(directory, stem) }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Rename the file over `target`, which it then is: no longer removed.
    fn rename(self, target: &Path) -> Result<()> {
        fs::rename(&self.path, target)?;
        let mut renamed = std::mem::ManuallyDrop::new(self);
        drop(std::mem::take(&mut renamed.path));
        Ok(())
    }
}

impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// A store: the files under one root, shared by every process that names it.
#[derive(Debug)]
pub struct Store {
    root: PathBuf,
    privacy: Privacy,
    /// Bytes published since eviction last began ([`Store::written`]).
    written: AtomicU64,
}

impl Store {
    /// The store at `root`, which is created, with every directory under it,
    /// as its writes need them.
    pub fn new(root: PathBuf, privacy: Privacy) -> Self {
        Self { root, privacy, written: AtomicU64::new(0) }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Bytes published through the store since eviction last began in this
    /// process: what its embedder decides from when to evict again.
    pub fn written(&self) -> u64 {
        self.written.load(Ordering::Relaxed)
    }

    /// Make `path` (absolute, under the root) hold what `write` writes to a
    /// file, as one atomic step, to `durability`. On any failure `path` is as
    /// it was and no temporary file is left, unless the failure is syncing
    /// the rename itself, of a durable publication, after which the new file
    /// is visible but may not survive a crash.
    fn published(&self, path: &Path, durability: Durability, write: impl FnOnce(&mut File) -> Result<()>) -> Result<()> {
        let directory = parent(path);
        let name = path.file_name().expect("a stored path has a name");
        let (temporary, mut file) = loop {
            self.privacy.directory(directory)?;
            match Temporary::create(directory, name, self.privacy) {
                // An eviction removed the directory, emptied, since it was
                // created: create it again.
                Err(error) if error.kind() == ErrorKind::NotFound => continue,
                created => break created?,
            }
        };
        write(&mut file)?;
        if durability == Durability::Durable {
            file.sync_all()?;
        }
        let length = file.metadata()?.len();
        drop(file);
        temporary.rename(path)?;
        self.written.fetch_add(length, Ordering::Relaxed);
        match durability {
            Durability::Durable => synced_directory(directory),
            Durability::Volatile => Ok(()),
        }
    }

    /// A lock file at `path` (absolute, under the root), opened.
    fn lock_file(&self, path: &Path) -> Result<File> {
        self.privacy.directory(parent(path))?;
        let mut options = self.privacy.options();
        options.create(true).truncate(false).read(true).write(true);
        options.open(path)
    }

    /// The store's lock held shared, waiting until no exclusive holder holds
    /// it: every file is as one writer left it while it lives.
    pub fn shared(&self) -> Result<Shared<'_>> {
        let lock = self.lock_file(&self.root.join(LOCK))?;
        lock.lock_shared()?;
        Ok(Shared { store: self, _lock: lock })
    }

    /// The store's lock held exclusively, waiting until no other holder holds
    /// it: no reader, other writer or eviction is among the files while it
    /// lives, nor is a record's lock being taken. A thread holding the
    /// store's lock takes it again only to wait for itself.
    pub fn exclusive(&self) -> Result<Exclusive<'_>> {
        let lock = self.lock_file(&self.root.join(LOCK))?;
        lock.lock()?;
        Ok(Exclusive { store: self, _lock: lock })
    }

    /// The lock of the record at `record` under the root, held, waiting
    /// until no other holder holds it: what reads, writes and removes the
    /// record, which eviction leaves while it is held. The record may be any file the
    /// store names, or a name for a set of files its holder alone writes (a
    /// directory). A holder that removes the record removes its lock with it
    /// ([`Record::remove`]); the lock of a record that is gone otherwise goes
    /// at the store's next eviction.
    pub fn lock(&self, record: &Path) -> Result<Record<'_>> {
        Ok(Record { store: self, locked: self.waited(record, Holding::Exclusive)? })
    }

    /// [`Store::lock`], none while another holder holds the record, without
    /// waiting: for a holder that refuses, rather than waits for, the
    /// record's readers ([`Store::try_share_owned`]).
    pub fn try_lock(&self, record: &Path) -> Result<Option<Record<'_>>> {
        Ok(self.locked(record, Holding::Try)?.map(|locked| Record { store: self, locked }))
    }

    /// [`Store::try_lock`], the hold borrowing nothing of its store: for a
    /// holder that keeps the record across awaits or threads.
    pub fn try_lock_owned(&self, record: &Path) -> Result<Option<OwnedRecord>> {
        Ok(self.locked(record, Holding::Try)?.map(|locked| OwnedRecord { _locked: locked }))
    }

    /// A lease of this process among the holders of `directory`: the record
    /// `<directory>/<pid>.<n>`, published empty, which its holder may publish
    /// again ([`Lease::record`]) and whose lock it holds until the lease is
    /// dropped, which removes it. [`Store::leases`] finds it meanwhile, from
    /// whichever process, and removes it once its holder is gone unawares.
    pub fn lease(&self, directory: &Path) -> Result<Lease> {
        let record = directory.join(format!("{}.{}", std::process::id(), LEASES.fetch_add(1, Ordering::Relaxed)));
        let locked = self.waited(&record, Holding::Exclusive)?;
        locked.write(self, &[], Durability::Volatile)?;
        Ok(Lease { record, locked: Some(locked) })
    }

    /// The records of the live leases of `directory` ([`Store::lease`]), in
    /// whichever process, this one's among them: each whose lock a holder
    /// holds. A lease no holder holds is a gone holder's, and is removed.
    pub fn leases(&self, directory: &Path) -> Result<Vec<PathBuf>> {
        let mut live = Vec::new();
        for name in self.list(directory)? {
            let record = directory.join(name);
            match self.try_lock(&record)? {
                Some(gone) => gone.remove()?,
                None => live.push(record),
            }
        }
        Ok(live)
    }

    /// [`Store::lock`], the hold borrowing nothing of its store: for a holder
    /// that keeps the record across awaits or threads only to exclude every
    /// other holder.
    pub fn lock_owned(&self, record: &Path) -> Result<OwnedRecord> {
        Ok(OwnedRecord { _locked: self.waited(record, Holding::Exclusive)? })
    }

    /// The record's lock held shared with its other readers, borrowing
    /// nothing of its store, none while an exclusive holder holds it, without
    /// waiting: for a reader that refuses, rather than waits for, a removal
    /// of what the record names, which may hold it for long.
    pub fn try_share_owned(&self, record: &Path) -> Result<Option<OwnedRecord>> {
        Ok(self.locked(record, Holding::TryShared)?.map(|locked| OwnedRecord { _locked: locked }))
    }

    /// [`Store::locked`] as a waiting `holding`, which always takes it.
    fn waited(&self, record: &Path, holding: Holding) -> Result<Locked> {
        Ok(self.locked(record, holding)?.expect("a waiting lock is taken once its holders are gone"))
    }

    /// The record's lock held as `holding` says, none where it is a try and
    /// another holder holds it. It is taken under the store's lock held
    /// shared, which is let go once it is: no eviction is among the record's
    /// lock files while it is taken, and eviction leaves the record while it
    /// is held.
    fn locked(&self, record: &Path, holding: Holding) -> Result<Option<Locked>> {
        let shared = self.lock_file(&self.root.join(LOCK))?;
        shared.lock_shared()?;
        let path = self.root.join(LOCKS).join(record);
        let mut pending = Pending::default();
        loop {
            let file = match self.lock_file(&path) {
                Ok(file) => file,
                // A removed lock file a waiter still holds open.
                Err(error) => {
                    pending.retry(error)?;
                    continue;
                }
            };
            match holding {
                Holding::Exclusive => file.lock()?,
                Holding::Shared => file.lock_shared()?,
                Holding::Try => {
                    if !taken(file.try_lock())? {
                        return Ok(None);
                    }
                }
                Holding::TryShared => {
                    if !taken(file.try_lock_shared())? {
                        return Ok(None);
                    }
                }
            }
            let held = same_file::Handle::from_file(file)?;
            // A holder that removed the record removed this lock file after
            // it was opened: the name is no longer this file's.
            match same_file::Handle::from_path(&path) {
                Ok(named) if named == held => {
                    return Ok(Some(Locked { record: self.root.join(record), lock: path, _held: held }));
                }
                Ok(_) => continue,
                Err(error) if error.kind() == ErrorKind::NotFound => continue,
                // As above, the file open by this waiter among others.
                Err(error) => pending.retry(error)?,
            }
        }
    }

    /// The bytes of the record at `path` under the root, none if none is
    /// there, read without a lock: every record is published whole, so a
    /// reader finds one writer's. Larger than `largest`, or no regular file,
    /// is an error.
    pub fn read(&self, path: &Path, largest: u64) -> Result<Option<Vec<u8>>> {
        read_bounded(&self.root.join(path), largest)
    }

    /// The bytes of the file at `path` under the root, a content address or
    /// a file every reader checks whole, none if none is there, read without
    /// a lock and marked used.
    pub fn fetch(&self, path: &Path) -> Result<Option<Vec<u8>>> {
        read_used(&self.root.join(path), u64::MAX)
    }

    /// The names of the files in `directory` under the root, none if it is
    /// absent; temporary files and the store's own are none of them.
    pub fn list(&self, directory: &Path) -> Result<Vec<String>> {
        let listing = match fs::read_dir(self.root.join(directory)) {
            Ok(listing) => listing,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let mut names = Vec::new();
        for item in listing {
            let name = item?.file_name();
            if !unpublished(&name) && !owned(&name) {
                names.push(name.to_string_lossy().into_owned());
            }
        }
        Ok(names)
    }

    /// Publish what `write` writes at `path` under the root, to
    /// `durability`, without a lock: for a file whose every writer writes
    /// what every reader checks whole and accepts from any of them (a
    /// program store's artifacts and pointers, a probe file, a tool's
    /// outputs). A failure leaves no partial file.
    pub fn publish(&self, path: &Path, durability: Durability, write: impl FnOnce(&mut File) -> Result<()>) -> Result<()> {
        self.published(&self.root.join(path), durability, write)
    }

    /// Publish `bytes` at `path` under the root, a content address, unless a
    /// file of their length is there already, which is then marked used:
    /// whether they were written.
    pub fn put_if_absent(&self, path: &Path, bytes: &[u8], durability: Durability) -> Result<bool> {
        let path = self.root.join(path);
        if fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.is_file() && metadata.len() == bytes.len() as u64) {
            used(&path);
            return Ok(false);
        }
        self.published(&path, durability, |file| file.write_all(bytes))?;
        Ok(true)
    }

    /// The file at `path` under the root, a content address, opened and
    /// marked used, held under its own lock shared: eviction keeps it at its
    /// path while the hold lives. None if no file is there.
    pub fn hold(&self, path: &Path) -> Result<Option<Held>> {
        let locked = self.waited(path, Holding::Shared)?;
        match File::open(&locked.record) {
            Ok(file) => {
                used(&locked.record);
                Ok(Some(Held { file, locked }))
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// An empty directory staged for `target` under the root, beside it, as
    /// the store's privacy says, which [`Staged::publish`] renames into place
    /// whole.
    pub fn stage(&self, target: &Path) -> Result<Staged> {
        let target = self.root.join(target);
        let directory = parent(&target);
        let name = target.file_name().expect("a staged path has a name").to_owned();
        loop {
            self.privacy.directory(directory)?;
            let stage = temporary_name(directory, &name);
            match fs::create_dir(&stage) {
                Ok(()) => {
                    self.privacy.directory(&stage)?;
                    return Ok(Staged { target, stage, published: false });
                }
                // Only a dead writer of the same process id leaves the name
                // taken; an eviction may have removed the directory, emptied.
                Err(error) if matches!(error.kind(), ErrorKind::AlreadyExists | ErrorKind::NotFound) => continue,
                Err(error) => return Err(error),
            }
        }
    }

    /// Set what is at `path` under the root (a file or directory where the
    /// store expects something else) aside, under a temporary name beside
    /// it, which the store's next eviction removes once abandoned.
    pub fn quarantine(&self, path: &Path) -> Result<()> {
        let path = self.root.join(path);
        let aside = temporary_name(parent(&path), path.file_name().expect("a quarantined path has a name"));
        fs::rename(&path, &aside)?;
        // Its age is its quarantine's, so eviction keeps it its full span; a
        // link is aged by nothing, never followed to its target.
        if fs::symlink_metadata(&aside).is_ok_and(|metadata| !metadata.file_type().is_symlink()) {
            let _ = touched(&aside, SystemTime::now());
        }
        Ok(())
    }

    /// Remove the least recently used entries until the rest fit `capacity`
    /// bytes, as `layout` places each file, the store's lock held
    /// exclusively from the listing to the last removal. An entry used
    /// within `grace` stays, counted, whatever the capacity; a file the
    /// layout keeps or pins is never removed, nor what it references; a
    /// pinned entry counts against the capacity first; a directory placed
    /// apart is another store's, never walked. A shared file goes once no
    /// entry kept or file kept references it, and an unreachable one
    /// whatever the capacity, each once not used within `grace`.
    /// With them go temporary files older than [`ABANDONED`], in what the
    /// layout keeps too, the lock files no holder holds of records that are
    /// gone, and every directory these removals left empty. A file whose
    /// lock a holder holds ([`Store::hold`], a record's holder) stays, and
    /// so does the entry it is part of, counted, and what that references; a
    /// file that cannot be removed stays for a later eviction. Whether it
    /// ran: one that finds the store's lock held, by a reader, a writer or a
    /// record's lock being taken, passes without waiting.
    pub fn evict(&self, capacity: u64, grace: Duration, layout: &dyn Layout) -> Result<bool> {
        // Counted from before the lock is tried: an eviction that passes is
        // begun again only once as much more is written.
        self.written.store(0, Ordering::Relaxed);
        let lock = self.lock_file(&self.root.join(LOCK))?;
        if !taken(lock.try_lock())? {
            return Ok(false);
        }
        let _held = Exclusive { store: self, _lock: lock };
        let locks = self.root.join(LOCKS);
        // Whether a holder holds the lock of the file at `path`: no lock is
        // taken while the store's is held exclusively, so one free now stays
        // free until the eviction ends.
        let in_use = |path: &Path| {
            let lock = locks.join(path.strip_prefix(&self.root).expect("eviction walks under its root"));
            match OpenOptions::new().read(true).write(true).open(lock) {
                Ok(file) => !matches!(file.try_lock(), Ok(())),
                Err(error) => error.kind() != ErrorKind::NotFound,
            }
        };
        let now = SystemTime::now();
        // Used within the grace; one used after `now` (a clock that steps)
        // within any grace at all.
        let recent = |used: SystemTime| match now.duration_since(used) {
            Ok(age) => age < grace,
            Err(_) => !grace.is_zero(),
        };
        let examining = Examining { root: &self.root };
        let relative = |path: &Path| path.strip_prefix(&self.root).expect("eviction walks under its root").to_owned();
        let mut entries = HashMap::<Entry, Listed>::new();
        // Each shared file's size, use time, and whether a holder holds it.
        let mut shared = BTreeMap::<PathBuf, (u64, SystemTime, bool)>::new();
        let mut referenced = BTreeSet::<PathBuf>::new();
        // The directories eviction removed something from, which it removes
        // after if that left them empty.
        let mut emptied = BTreeSet::<PathBuf>::new();
        let mut remove = |path: &Path, directory: bool| {
            let removed = match directory {
                true => fs::remove_dir_all(path),
                false => fs::remove_file(path),
            };
            if removed.is_ok() {
                emptied.insert(parent(path).to_owned());
            }
            removed
        };
        // Each directory to walk, and whether the layout keeps it: a kept
        // directory holds no entry, and is walked only for abandoned writes.
        let mut directories = vec![(self.root.clone(), false)];
        while let Some((directory, kept)) = directories.pop() {
            let Ok(listing) = fs::read_dir(&directory) else {
                continue;
            };
            for item in listing.flatten() {
                let Ok(metadata) = item.metadata() else {
                    continue;
                };
                let (name, path) = (item.file_name(), item.path());
                let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                if unpublished(&name) {
                    if now.duration_since(modified).unwrap_or_default() > ABANDONED {
                        let _ = remove(&path, metadata.is_dir());
                    }
                    continue;
                }
                if directory == self.root && owned(&name) {
                    continue;
                }
                let place = |path: &Path| layout.place(&relative(path), &examining);
                if metadata.is_dir() {
                    match place(&path) {
                        Eviction::Apart => {}
                        placed => directories.push((path, kept || placed == Eviction::Kept)),
                    }
                    continue;
                }
                if kept {
                    continue;
                }
                let (entry, order, pinned) = match place(&path) {
                    Eviction::Kept | Eviction::Apart => {
                        referenced.extend(layout.references(&relative(&path), &examining));
                        continue;
                    }
                    Eviction::Unreachable => {
                        if !recent(modified) && !in_use(&path) {
                            if let Err(error) = remove(&path, false) {
                                tracing::warn!("store {}: cannot remove {}, which nothing reaches: {error}", self.root.display(), path.display());
                            }
                        }
                        continue;
                    }
                    Eviction::Shared => {
                        shared.insert(relative(&path), (metadata.len(), modified, in_use(&path)));
                        continue;
                    }
                    Eviction::Alone => (Entry::Alone(path.clone()), 0, false),
                    Eviction::Pinned => (Entry::Alone(path.clone()), 0, true),
                    Eviction::Part { entry, order } => (Entry::Named(entry), order, false),
                };
                let references = layout.references(&relative(&path), &examining);
                let entry = entries.entry(entry).or_insert(Listed { pinned, used: SystemTime::UNIX_EPOCH, bytes: 0, paths: Vec::new(), references: Vec::new() });
                entry.pinned |= in_use(&path);
                entry.used = entry.used.max(modified);
                entry.bytes += metadata.len();
                entry.paths.push((order, path));
                entry.references.extend(references);
            }
        }
        let mut entries = entries.into_values().collect::<Vec<_>>();
        // Pinned entries first, then the most recently used.
        entries.sort_by(|left, right| right.pinned.cmp(&left.pinned).then(right.used.cmp(&left.used)));
        let mut held = 0u64;
        for mut entry in entries {
            let added = entry.bytes
                + entry
                    .references
                    .iter()
                    .filter(|reference| !referenced.contains(*reference))
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .filter_map(|reference| shared.get(reference).map(|(bytes, _, _)| *bytes))
                    .sum::<u64>();
            if entry.pinned || recent(entry.used) || held + added <= capacity {
                held += added;
                referenced.extend(entry.references);
            } else {
                entry.paths.sort();
                for (_, path) in entry.paths {
                    let _ = remove(&path, false);
                }
            }
        }
        for (path, (_, modified, held)) in shared {
            if !referenced.contains(&path) && !recent(modified) && !held {
                let _ = remove(&self.root.join(path), false);
            }
        }
        // No record's lock is taken while the store's is held exclusively:
        // the lock of a record that is gone is no one's once no holder holds
        // it (a record that names no file, as a lease does, is gone).
        let mut directories = vec![locks.clone()];
        while let Some(directory) = directories.pop() {
            let Ok(listing) = fs::read_dir(&directory) else {
                continue;
            };
            for item in listing.flatten() {
                let path = item.path();
                if item.file_type().is_ok_and(|kind| kind.is_dir()) {
                    directories.push(path);
                } else {
                    let record = self.root.join(path.strip_prefix(&locks).expect("a lock lies under the locks"));
                    if fs::symlink_metadata(&record).is_err() && !in_use(&record) {
                        let _ = remove(&path, false);
                    }
                }
            }
        }
        // A directory its removals left empty goes, and then the one it lay
        // in if that left it empty, deepest first: no path under a directory
        // orders before it. One that holds anything stays (`remove_dir`
        // refuses it), as do the root and the directory of the locks. A
        // writer whose directory goes meanwhile creates it again.
        while let Some(directory) = emptied.pop_last() {
            if directory != self.root && directory != locks && fs::remove_dir(&directory).is_ok() {
                emptied.insert(parent(&directory).to_owned());
            }
        }
        Ok(true)
    }
}

/// The files of a store, its lock held shared: every file is as one writer
/// left it until this is dropped.
#[must_use = "the files are held only while this lives"]
#[derive(Debug)]
pub struct Shared<'a> {
    store: &'a Store,
    _lock: File,
}

impl Shared<'_> {
    /// The file at `path` under the root, none if no file is there,
    /// refreshing its use time. Larger than `largest`, or no regular file,
    /// is an error.
    pub fn read(&self, path: &Path, largest: u64) -> Result<Option<Vec<u8>>> {
        read_used(&self.store.root.join(path), largest)
    }
}

/// The files of a store, its lock held exclusively: no reader, record
/// holder, other writer or eviction is among them until this is dropped.
#[must_use = "the files are held only while this lives"]
#[derive(Debug)]
pub struct Exclusive<'a> {
    store: &'a Store,
    _lock: File,
}

impl Exclusive<'_> {
    /// The file at `path` under the root, none if no file is there, its use
    /// time as it was: what maintains the files reads them without using
    /// them. Larger than `largest`, or no regular file, is an error.
    pub fn examine(&self, path: &Path, largest: u64) -> Result<Option<Vec<u8>>> {
        read_bounded(&self.store.root.join(path), largest)
    }

    /// Publish `bytes` at `path` under the root, to `durability`. A failure
    /// leaves no partial file.
    pub fn write(&self, path: &Path, bytes: &[u8], durability: Durability) -> Result<()> {
        self.store.published(&self.store.root.join(path), durability, |file| file.write_all(bytes))
    }

    /// Append `bytes` to the file at `path` under the root, created if
    /// absent: every append of every holder lands whole, in turn.
    pub fn append(&self, path: &Path, bytes: &[u8]) -> Result<()> {
        let path = self.store.root.join(path);
        self.store.privacy.directory(parent(&path))?;
        let mut options = self.store.privacy.options();
        options.create(true).append(true);
        options.open(&path)?.write_all(bytes)?;
        self.store.written.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Remove the file at `path` under the root, if one is there.
    pub fn remove(&self, path: &Path) -> Result<()> {
        removed(&self.store.root.join(path))
    }

    /// [`Store::list`].
    pub fn list(&self, directory: &Path) -> Result<Vec<String>> {
        self.store.list(directory)
    }
}

/// How a record's lock is taken.
#[derive(Clone, Copy)]
enum Holding {
    /// Exclusively, waiting for every other holder.
    Exclusive,
    /// Shared with every other shared holder, waiting for an exclusive one.
    Shared,
    /// Exclusively, or not at all while another holder holds it.
    Try,
    /// Shared, or not at all while an exclusive holder holds it.
    TryShared,
}

/// Whether a try took its lock: not while another holder holds it.
fn taken(tried: std::result::Result<(), std::fs::TryLockError>) -> Result<bool> {
    match tried {
        Ok(()) => Ok(true),
        Err(std::fs::TryLockError::WouldBlock) => Ok(false),
        Err(std::fs::TryLockError::Error(error)) => Err(error),
    }
}

/// A record's lock held.
#[derive(Debug)]
struct Locked {
    /// The record's path, absolute.
    record: PathBuf,
    /// Its lock's path, absolute.
    lock: PathBuf,
    _held: same_file::Handle,
}

/// Windows refuses to open or to name a file removed while a holder still
/// has it open (its deletion pending), with the error it gives a refusal of
/// access, from which nothing the standard library reports tells it apart.
/// A refusal is taken for a pending deletion, which lasts only until its
/// waiter finds the name no longer its own and closes it, for at most
/// [`PENDING`]; past that, and on every other platform, it is reported.
#[derive(Default)]
struct Pending {
    since: Option<std::time::Instant>,
}

impl Pending {
    fn retry(&mut self, error: std::io::Error) -> Result<()> {
        if !(cfg!(windows) && error.kind() == ErrorKind::PermissionDenied) {
            return Err(error);
        }
        if self.since.get_or_insert_with(std::time::Instant::now).elapsed() > PENDING {
            return Err(error);
        }
        std::thread::sleep(Duration::from_millis(1));
        Ok(())
    }
}

impl Locked {
    fn read(&self, largest: u64) -> Result<Option<Vec<u8>>> {
        read_bounded(&self.record, largest)
    }

    fn write(&self, store: &Store, bytes: &[u8], durability: Durability) -> Result<()> {
        store.published(&self.record, durability, |file| file.write_all(bytes))
    }

    fn remove(self) -> Result<()> {
        removed(&self.record)?;
        removed(&self.lock)
    }
}

/// A record's lock, held: the record is this holder's to read, write and
/// remove until it is dropped, and eviction's to leave.
#[must_use = "the record is held only while this lives"]
#[derive(Debug)]
pub struct Record<'a> {
    store: &'a Store,
    locked: Locked,
}

impl Record<'_> {
    /// The record's bytes, none if no file is there. Larger than `largest`,
    /// or no regular file, is an error.
    pub fn read(&self, largest: u64) -> Result<Option<Vec<u8>>> {
        self.locked.read(largest)
    }

    /// Publish `bytes` as the record, to `durability`. A failure leaves the
    /// record as it was.
    pub fn write(&self, bytes: &[u8], durability: Durability) -> Result<()> {
        self.locked.write(self.store, bytes, durability)
    }

    /// Remove the record, if a file is there, and its lock: a waiter for the
    /// lock takes it again, under the name's next file.
    pub fn remove(self) -> Result<()> {
        self.locked.remove()
    }
}

/// A record's lock held, borrowing nothing of the store: exclusively
/// ([`Store::lock_owned`]), so no other holder reaches the record, or shared
/// with its other readers ([`Store::try_share_owned`]), until it is dropped;
/// eviction leaves the record meanwhile.
#[must_use = "the record is held only while this lives"]
#[derive(Debug)]
pub struct OwnedRecord {
    _locked: Locked,
}

/// A process's lease among the holders of a directory ([`Store::lease`]).
#[must_use = "the lease is held only while this lives"]
#[derive(Debug)]
pub struct Lease {
    record: PathBuf,
    locked: Option<Locked>,
}

impl Lease {
    /// The lease's record under the root, whose file its holder may publish.
    pub fn record(&self) -> &Path {
        &self.record
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(locked) = self.locked.take() {
            // One left behind is removed as a gone holder's ([`Store::leases`]).
            let _ = locked.remove();
        }
    }
}

/// A file a store keeps under a content address, open, held under its own
/// lock shared: eviction keeps it at its path while this lives.
#[derive(Debug)]
pub struct Held {
    file: File,
    locked: Locked,
}

impl Held {
    /// Where the store keeps it, which names this file while it is held.
    pub fn path(&self) -> &Path {
        &self.locked.record
    }

    /// Its bytes, read from the open file from its start.
    pub fn bytes(&self) -> Result<Vec<u8>> {
        use std::io::Seek;
        let mut bytes = Vec::new();
        (&self.file).seek(std::io::SeekFrom::Start(0))?;
        (&self.file).read_to_end(&mut bytes)?;
        Ok(bytes)
    }
}

/// A directory being written beside its target, under a temporary name,
/// which [`Staged::publish`] renames into place whole: a reader finds the
/// target as one publication left it, or briefly absent while one replaces
/// another, never part written. Its writer fills it at [`Staged::path`],
/// over as many steps as it needs. Dropped unpublished, it is removed.
/// Nothing of it is synced: a crash may lose it.
#[derive(Debug)]
pub struct Staged {
    target: PathBuf,
    stage: PathBuf,
    published: bool,
}

impl Staged {
    /// Where the directory is written until it is published.
    pub fn path(&self) -> &Path {
        &self.stage
    }

    /// Rename the stage into place. A directory renames over no non-empty
    /// one, so what holds the target, another publication's or an earlier
    /// one's, is first renamed aside under a name of its own and removed
    /// once this one is in place; publications at once each land whole, and
    /// the last to land holds the target. A failure removing what was
    /// replaced is reported after the stage is in place.
    pub fn publish(mut self) -> Result<()> {
        let mut replaced = Vec::new();
        let directory = parent(&self.target).to_owned();
        let name = self.target.file_name().expect("a staged path has a name").to_owned();
        loop {
            let held = match fs::rename(&self.stage, &self.target) {
                Ok(()) => break,
                // Something holds the target (another publication's or an
                // earlier one's; a rename over it fails, with an error kind
                // each platform names its own way), or held it as the rename
                // ran and was moved aside since: move aside what holds it,
                // and again.
                Err(error)
                    if fs::symlink_metadata(&self.target).is_ok()
                        || matches!(error.kind(), ErrorKind::DirectoryNotEmpty | ErrorKind::AlreadyExists) =>
                {
                    error
                }
                // The rename itself failed: what this publication moved
                // aside goes back.
                Err(error) => {
                    if let Some(last) = replaced.pop() {
                        let _ = fs::rename(last, &self.target);
                    }
                    return Err(error);
                }
            };
            let aside = temporary_name(&directory, &name);
            match fs::rename(&self.target, &aside) {
                Ok(()) => replaced.push(aside),
                // Another publication moved it aside first.
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(_) => return Err(held),
            }
        }
        self.published = true;
        for aside in replaced {
            match fs::symlink_metadata(&aside)?.is_dir() {
                true => fs::remove_dir_all(&aside)?,
                false => fs::remove_file(&aside)?,
            }
        }
        Ok(())
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.stage);
        }
    }
}

/// How eviction places a path under the root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Eviction {
    /// Never removed, nor what it references; a directory so placed holds
    /// no entry, and only its abandoned temporary files are: its files are
    /// neither placed nor read for references.
    Kept,
    /// A directory another store owns, under its own lock: never walked.
    Apart,
    /// An entry of its own.
    Alone,
    /// An entry of its own that is never removed: counted first against the
    /// capacity, with what it references.
    Pinned,
    /// Part of the entry named `entry`, with every other file so named,
    /// removed in `order`, lowest first: the file that names the others
    /// goes first.
    Part { entry: String, order: u8 },
    /// A file entries share, by reference ([`Layout::references`]): counted
    /// with the most recently used entry that references it, and removed
    /// once nothing kept references it.
    Shared,
    /// A file nothing the embedder keeps reaches: the walk that places it so
    /// removes it, whatever the capacity, once it is not used within the
    /// grace. Of a directory, the walk reads only whether it is kept.
    Unreachable,
}

/// The embedder's layout: how eviction places each path under the root, a
/// directory before it walks it and each file in one it walks, examining
/// what it needs of the files to judge whether one is reachable, and what
/// each entry's or kept file references.
pub trait Layout {
    fn place(&self, path: &Path, examining: &Examining<'_>) -> Eviction;

    /// The shared files ([`Eviction::Shared`]) the file at `path`, an
    /// entry's or a kept one, references: kept while it is.
    fn references(&self, _path: &Path, _examining: &Examining<'_>) -> Vec<PathBuf> {
        Vec::new()
    }
}

impl<F: Fn(&Path, &Examining<'_>) -> Eviction> Layout for F {
    fn place(&self, path: &Path, examining: &Examining<'_>) -> Eviction {
        self(path, examining)
    }
}

/// The files as eviction's walk lets a layout read them: examined, their use
/// time as it was, under the store's lock the walk holds exclusively, so no
/// writer is between files that name one another.
pub struct Examining<'a> {
    root: &'a Path,
}

impl Examining<'_> {
    /// The file at `path` under the root, none if no file is there. Larger
    /// than `largest`, or no regular file, is an error.
    pub fn examine(&self, path: &Path, largest: u64) -> Result<Option<Vec<u8>>> {
        read_bounded(&self.root.join(path), largest)
    }
}

/// What eviction listed of one entry.
struct Listed {
    /// Whether it is never removed: pinned ([`Eviction::Pinned`]), or a
    /// file of it held.
    pinned: bool,
    /// When the latest of its files was used.
    used: SystemTime,
    bytes: u64,
    /// Its files, each with its order of removal.
    paths: Vec<(u8, PathBuf)>,
    /// The shared files its files reference.
    references: Vec<PathBuf>,
}

/// An entry eviction removes whole.
#[derive(PartialEq, Eq, Hash)]
enum Entry {
    /// A file of its own, by its path.
    Alone(PathBuf),
    /// The files the layout names one entry.
    Named(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("seismic-store-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn names(directory: &Path) -> Vec<String> {
        let mut names = fs::read_dir(directory).unwrap().map(|item| item.unwrap().file_name().to_string_lossy().into_owned()).collect::<Vec<_>>();
        names.sort();
        names
    }

    /// Set the use time of the file or directory at `path` `seconds` back.
    fn aged(path: &Path, seconds: u64) {
        touched(path, SystemTime::now() - Duration::from_secs(seconds)).unwrap();
    }

    fn alone(_: &Path, _: &Examining<'_>) -> Eviction {
        Eviction::Alone
    }

    /// A publication replaces what was there, leaves no temporary file, and
    /// a failed one leaves the target as it was.
    #[test]
    fn a_publication_replaces_whole_or_leaves_the_target() {
        let root = scratch("publish");
        let store = Store::new(root.clone(), Privacy::Owner);
        let path = Path::new("pointer");
        store.publish(path, Durability::Durable, |file| file.write_all(b"first")).unwrap();
        store.lock(path).unwrap().write(b"second", Durability::Volatile).unwrap();
        let failed = store.published(&root.join(path), Durability::Durable, |file| {
            file.write_all(b"torn")?;
            Err(std::io::Error::other("the writer fails"))
        });
        assert!(failed.is_err());
        assert_eq!(fs::read(root.join(path)).unwrap(), b"second");
        assert_eq!(store.list(Path::new("")).unwrap(), ["pointer"], "no temporary file remains, and the store's own are no files of it");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(root.join(path)).unwrap().permissions().mode() & 0o777, 0o600);
            assert_eq!(fs::metadata(&root).unwrap().permissions().mode() & 0o777, 0o700);
        }
        fs::remove_dir_all(root).unwrap();
    }

    /// A temporary file a dead writer of this process id left under the
    /// next name is passed over, not written into.
    #[test]
    fn a_taken_name_is_passed_over() {
        let root = scratch("taken");
        fs::create_dir_all(&root).unwrap();
        let store = Store::new(root.clone(), Privacy::Shared);
        let next = TEMPORARIES.load(Ordering::Relaxed);
        let taken = (next..next + 64).map(|count| root.join(format!(".entry.{}.{count}{UNPUBLISHED}", std::process::id()))).collect::<Vec<_>>();
        for left in &taken {
            fs::write(left, b"abandoned").unwrap();
        }
        store.publish(Path::new("entry"), Durability::Volatile, |file| file.write_all(b"whole")).unwrap();
        assert_eq!(fs::read(root.join("entry")).unwrap(), b"whole");
        for left in &taken {
            assert_eq!(fs::read(left).unwrap(), b"abandoned");
        }
        fs::remove_dir_all(root).unwrap();
    }

    /// Writers updating one record at once, each a store of its own as
    /// another process holds, each read what the others wrote: every one of
    /// their additions is kept. A removal removes the record's lock with it,
    /// and a writer waiting for the lock meanwhile takes it again.
    #[test]
    fn updates_of_a_record_through_its_lock_lose_nothing() {
        let root = scratch("updates");
        let record = Path::new("counts/sum");
        let (writers, additions) = (4u64, 100u64);
        std::thread::scope(|scope| {
            for writer in 0..writers {
                let root = root.clone();
                seismic_threads::spawn_scoped(scope, format!("updating-{writer}"), move || {
                    let store = Store::new(root, Privacy::Shared);
                    for _ in 0..additions {
                        let held = store.lock(record).unwrap();
                        let sum = held.read(32).unwrap().map_or(0, |bytes| String::from_utf8(bytes).unwrap().parse::<u64>().unwrap());
                        held.write((sum + 1).to_string().as_bytes(), Durability::Volatile).unwrap();
                    }
                })
                .unwrap();
            }
        });
        let store = Store::new(root.clone(), Privacy::Shared);
        assert_eq!(store.read(record, 32).unwrap().unwrap(), (writers * additions).to_string().as_bytes());
        assert!(root.join(LOCKS).join(record).exists());
        // Removals interleaved with writers: each writer's lock is the name's
        // current file, so no two hold the record at once. A waiter that took
        // a removed lock file and held it beside the holder of the next would
        // be inside with it, and its addition, read before the other's write,
        // would be lost: the record would not hold every addition since the
        // last removal.
        let (inside, since_removal) = (AtomicU64::new(0), AtomicU64::new(0));
        std::thread::scope(|scope| {
            for writer in 0..writers {
                let (root, inside, since_removal) = (root.clone(), &inside, &since_removal);
                seismic_threads::spawn_scoped(scope, format!("removing-{writer}"), move || {
                    let store = Store::new(root, Privacy::Shared);
                    for addition in 0..additions {
                        let held = store.lock(record).unwrap();
                        assert_eq!(inside.fetch_add(1, Ordering::SeqCst), 0, "two holders of the record at once");
                        std::thread::yield_now();
                        inside.fetch_sub(1, Ordering::SeqCst);
                        match addition % 10 == 9 {
                            true => {
                                since_removal.store(0, Ordering::SeqCst);
                                held.remove().unwrap();
                            }
                            false => {
                                let sum = held.read(32).unwrap().map_or(0, |bytes| String::from_utf8(bytes).unwrap().parse::<u64>().unwrap());
                                since_removal.fetch_add(1, Ordering::SeqCst);
                                held.write((sum + 1).to_string().as_bytes(), Durability::Volatile).unwrap();
                            }
                        }
                    }
                })
                .unwrap();
            }
        });
        let left = store.read(record, 32).unwrap().map_or(0, |bytes| String::from_utf8(bytes).unwrap().parse::<u64>().unwrap());
        assert_eq!(left, since_removal.load(Ordering::SeqCst), "every addition since the last removal is kept");
        store.lock(record).unwrap().remove().unwrap();
        assert!(store.read(record, 32).unwrap().is_none());
        assert!(!root.join(LOCKS).join(record).exists(), "a removed record's lock goes with it");
        fs::remove_dir_all(root).unwrap();
    }

    /// A record's readers share its lock, which an exclusive try refuses while
    /// any of them, in this process or another, holds it, and takes once
    /// they are gone; a reader's try is refused while that one holds it.
    #[test]
    fn readers_share_a_record_a_try_refuses_while_they_hold_it() {
        let root = scratch("readers");
        let (store, other) = (Store::new(root.clone(), Privacy::Shared), Store::new(root.clone(), Privacy::Shared));
        let record = Path::new("hub/repository/.readers");
        let first = store.try_share_owned(record).unwrap().unwrap();
        let second = other.try_share_owned(record).unwrap().unwrap();
        assert!(other.try_lock(record).unwrap().is_none(), "a removal is refused while a reader holds the record");
        drop(first);
        assert!(store.try_lock(record).unwrap().is_none());
        drop(second);
        let removal = other.try_lock(record).unwrap().expect("no reader is left");
        assert!(store.try_share_owned(record).unwrap().is_none(), "a reader that will not wait is refused while a removal holds the record");
        removal.remove().unwrap();
        drop(store.try_share_owned(record).unwrap().expect("no removal is left"));
        fs::remove_dir_all(root).unwrap();
    }

    /// Each holder's lease is its own: every live one is found, from another
    /// store of the root as from another process, one dropped goes with its
    /// holder, and one left by a holder gone unawares is removed once found.
    #[test]
    fn leases_tell_each_live_holder_and_remove_a_gone_ones() {
        let root = scratch("leases");
        let (store, other) = (Store::new(root.clone(), Privacy::Shared), Store::new(root.clone(), Privacy::Shared));
        let directory = Path::new("readers/model");
        let (first, second) = (store.lease(directory).unwrap(), other.lease(directory).unwrap());
        let mut live = other.leases(directory).unwrap();
        live.sort();
        let mut expected = vec![first.record().to_owned(), second.record().to_owned()];
        expected.sort();
        assert_eq!(live, expected);
        drop(first);
        assert_eq!(store.leases(directory).unwrap(), vec![second.record().to_owned()]);
        let abandoned = directory.join("0.0");
        store.publish(&abandoned, Durability::Volatile, |_| Ok(())).unwrap();
        assert_eq!(store.leases(directory).unwrap(), vec![second.record().to_owned()], "a gone holder's lease is not live");
        assert!(!root.join(&abandoned).exists(), "and is removed once found");
        drop(second);
        assert!(store.leases(directory).unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    /// A record larger than it is read to, or no regular file, is an error,
    /// never an absent record.
    #[test]
    fn an_oversized_or_foreign_record_is_an_error() {
        let root = scratch("bounded");
        let store = Store::new(root.clone(), Privacy::Shared);
        store.publish(Path::new("big"), Durability::Volatile, |file| file.write_all(&[0; 10])).unwrap();
        assert!(store.read(Path::new("big"), 9).is_err());
        assert_eq!(store.read(Path::new("big"), 10).unwrap().unwrap().len(), 10);
        fs::create_dir_all(root.join("directory")).unwrap();
        assert!(store.read(Path::new("directory"), 10).is_err());
        assert!(store.read(Path::new("absent"), 10).unwrap().is_none());
        fs::remove_dir_all(root).unwrap();
    }

    /// Appends of holders at once each land whole.
    #[test]
    fn appends_land_whole() {
        let root = scratch("appends");
        std::thread::scope(|scope| {
            for writer in 0..4u8 {
                let root = root.clone();
                seismic_threads::spawn_scoped(scope, format!("appending-{writer}"), move || {
                    let store = Store::new(root, Privacy::Shared);
                    for _ in 0..50 {
                        store.exclusive().unwrap().append(Path::new("ledger"), format!("line {writer}\n").as_bytes()).unwrap();
                    }
                })
                .unwrap();
            }
        });
        let text = fs::read_to_string(root.join("ledger")).unwrap();
        assert_eq!(text.lines().count(), 200);
        assert!(text.lines().all(|line| line.starts_with("line ") && line.len() == 6));
        fs::remove_dir_all(root).unwrap();
    }

    /// Eviction keeps the most recently used entries that fit, each as its
    /// layout groups it, whole; a shared file goes with the last entry kept
    /// that references it; it leaves what the layout keeps, and removes
    /// abandoned temporary files, a stage among them, and the locks of
    /// records that are gone.
    #[test]
    fn eviction_keeps_recent_entries_and_what_they_reference() {
        struct Recipes;
        impl Layout for Recipes {
            fn place(&self, path: &Path, _: &Examining<'_>) -> Eviction {
                match path.iter().next().unwrap().to_str().unwrap() {
                    "kept" => Eviction::Kept,
                    "apart" => Eviction::Apart,
                    "chunks" => Eviction::Shared,
                    "foreign" => Eviction::Unreachable,
                    "pinned" if path != Path::new("pinned") => Eviction::Pinned,
                    _ => Eviction::Alone,
                }
            }
            fn references(&self, path: &Path, examining: &Examining<'_>) -> Vec<PathBuf> {
                let bytes = examining.examine(path, 64).unwrap().unwrap_or_default();
                String::from_utf8(bytes).unwrap().lines().filter(|line| line.starts_with("chunks/")).map(PathBuf::from).collect()
            }
        }
        let root = scratch("evict");
        let store = Store::new(root.clone(), Privacy::Shared);
        let write = |path: &str, bytes: &[u8]| store.publish(Path::new(path), Durability::Volatile, |file| file.write_all(bytes)).unwrap();
        write("chunks/a", &[0; 300]);
        write("chunks/b", &[0; 300]);
        write("chunks/c", &[0; 300]);
        write("recipes/new", b"chunks/a\n");
        write("recipes/old", b"chunks/b\n");
        write("pinned/one", b"chunks/c\n");
        write("kept/measured", &[0; 900]);
        write("foreign/fresh", b"");
        write("foreign/old", b"");
        let apart = root.join("apart");
        fs::create_dir_all(&apart).unwrap();
        let theirs = apart.join(".theirs.1.1.tmp");
        fs::write(&theirs, b"").unwrap();
        store.lock(Path::new("recipes/gone")).unwrap().remove().unwrap();
        drop(store.lock(Path::new("recipes/never")).unwrap());
        let stage = store.stage(Path::new("export")).unwrap();
        fs::write(stage.path().join("file"), b"x").unwrap();
        let abandoned_stage = stage.path().to_owned();
        std::mem::forget(stage);
        let abandoned = root.join(".recipes.1.1.tmp");
        fs::write(&abandoned, b"").unwrap();
        for (path, seconds) in [("chunks/a", 7200), ("chunks/b", 7200), ("chunks/c", 7200), ("recipes/new", 60), ("recipes/old", 3600), ("kept/measured", 9000), ("pinned/one", 9000)] {
            aged(&root.join(path), seconds);
        }
        aged(&abandoned, 2 * ABANDONED.as_secs());
        aged(&abandoned_stage, 2 * ABANDONED.as_secs());
        aged(&theirs, 2 * ABANDONED.as_secs());
        aged(&root.join("foreign/old"), 9000);
        // Room for the pinned entry, counted first, and one more.
        assert!(store.evict(700, Duration::from_secs(600), &Recipes).unwrap());
        assert!(root.join("foreign/fresh").exists() && !root.join("foreign/old").exists(), "an unreachable file goes once past the grace");
        assert!(theirs.exists(), "a directory placed apart is never walked");
        assert!(root.join("recipes/new").exists() && root.join("chunks/a").exists(), "the newest entry that fits stays, with what it references");
        assert!(!root.join("recipes/old").exists() && !root.join("chunks/b").exists(), "what does not fit goes, with what only it referenced");
        assert!(root.join("pinned/one").exists() && root.join("chunks/c").exists(), "a pinned entry stays, with what it references");
        assert!(root.join("kept/measured").exists());
        assert!(!abandoned.exists() && !abandoned_stage.exists(), "abandoned temporaries go");
        assert!(!root.join(LOCKS).join("recipes/never").exists(), "the lock of a record that is gone goes");
        assert!(root.join(LOCK).exists(), "eviction never removes the lock it holds");
        // Used within the grace, an entry stays whatever the capacity.
        assert!(store.evict(0, Duration::from_secs(600), &Recipes).unwrap());
        assert!(root.join("recipes/new").exists() && root.join("chunks/a").exists());
        assert!(store.evict(0, Duration::ZERO, &alone).unwrap());
        assert_eq!(store.list(Path::new("recipes")).unwrap(), Vec::<String>::new());
        fs::remove_dir_all(root).unwrap();
    }

    /// Eviction lists and removes under the store's lock held exclusively,
    /// and passes, without waiting, while another holds it. It leaves a held
    /// content address at its path, and a lease on a record that names no
    /// file, in this process or another, and removes both once let go.
    #[test]
    fn eviction_passes_while_the_store_is_held_and_leaves_what_is_held() {
        let root = scratch("holders");
        let (store, other) = (Store::new(root.clone(), Privacy::Shared), Store::new(root.clone(), Privacy::Shared));
        for path in ["artifact", "unused"] {
            store.publish(Path::new(path), Durability::Volatile, |file| file.write_all(b"compiled")).unwrap();
        }
        assert!(!store.put_if_absent(Path::new("artifact"), b"compiled", Durability::Volatile).unwrap());
        let reading = other.shared().unwrap();
        assert!(!store.evict(0, Duration::ZERO, &alone).unwrap(), "an eviction passes while the store is held");
        assert!(root.join("unused").exists());
        drop(reading);
        let held = store.hold(Path::new("artifact")).unwrap().unwrap();
        assert_eq!(held.bytes().unwrap(), b"compiled");
        let lease = other.try_share_owned(Path::new("readers/model")).unwrap().unwrap();
        assert!(store.evict(0, Duration::ZERO, &alone).unwrap());
        assert!(held.path().exists(), "a held file stays while its hold lives");
        assert!(!root.join("unused").exists(), "what no holder holds goes");
        assert!(root.join(LOCKS).join("readers/model").exists(), "a lease's lock stays while it is held");
        assert!(store.try_lock(Path::new("readers/model")).unwrap().is_none());
        drop((held, lease));
        assert!(store.evict(0, Duration::ZERO, &alone).unwrap());
        assert!(!root.join("artifact").exists());
        assert!(!root.join(LOCKS).join("readers/model").exists() && !root.join(LOCKS).join("artifact").exists(), "the locks no one holds of what is gone go");
        fs::remove_dir_all(root).unwrap();
    }

    /// Staged directories published at once over one target each land
    /// whole: the target is left one publication's files, never a mix, and
    /// no stage or replaced copy remains. One dropped unpublished leaves
    /// nothing. A quarantined path is set aside under a temporary name.
    #[test]
    fn staged_directories_land_whole() {
        let root = scratch("staged");
        let store = Store::new(root.clone(), Privacy::Shared);
        let target = Path::new("export");
        drop(store.stage(target).unwrap());
        std::thread::scope(|scope| {
            for writer in 0..4u8 {
                let store = &store;
                seismic_threads::spawn_scoped(scope, format!("staging-{writer}"), move || {
                    for _ in 0..25 {
                        let staged = store.stage(target).unwrap();
                        for file in ["a", "b", "c"] {
                            fs::write(staged.path().join(file), [writer]).unwrap();
                        }
                        staged.publish().unwrap();
                    }
                })
                .unwrap();
            }
        });
        let found = ["a", "b", "c"].map(|file| fs::read(root.join(target).join(file)).unwrap());
        assert!(found.iter().all(|bytes| bytes == &found[0]), "one publication's files, never a mix: {found:?}");
        assert_eq!(names(&root), ["export"], "no stage or replaced copy remains");
        store.quarantine(target).unwrap();
        assert_eq!(store.list(Path::new("")).unwrap(), Vec::<String>::new());
        assert!(names(&root).iter().all(|name| unpublished(OsStr::new(name))));
        fs::remove_dir_all(root).unwrap();
    }
}
