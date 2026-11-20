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
//! renamed over the target in one step under the target's exclusive record
//! lock: a reader finds the target as it was or as it now is, never part
//! written. A temporary name is the process's id and a counter of the whole
//! process, created exclusively, so no two writers share one, whichever store,
//! thread or process they are; one a writer that died left is removed by the
//! store's next sweep ([`Store::evict`]) once it is [`ABANDONED`]. A directory
//! written over many steps is published the same way, from a stage of its
//! own ([`Staged`]), serialized under the target's exclusive record lock.
//!
//! **Volume contract.** Opening a store ([`Store::open`]) verifies once that
//! its volume supports atomic replacement and unlinking of open files (native
//! on POSIX; NTFS POSIX rename and delete semantics on Windows). Volumes that
//! do not support both fail [`Store::open`] with
//! [`StoreError::UnsupportedVolume`]. This lets readers ([`Store::read`],
//! [`Store::fetch`], [`Shared::read`]) stay lockless on every platform while
//! writers and removals remain serialized under record locks.
//!
//! **Locks.** Two levels, both `std::fs::File::lock`, which holds across
//! threads (each through a file of its own) and processes alike, with strict
//! acquisition order (Level 1 before Level 2):
//!
//! - Level 1 — the store's lock (`.lock`): held shared ([`Shared`]) to read a
//!   consistent set of files or to publish under record locks
//!   ([`Shared::publish`], [`Shared::put_if_absent`]), exclusively
//!   ([`Exclusive`]) to read, write, append and remove them, and by eviction
//!   from its listing to its last removal.
//! - Level 2 — a record's lock (`.locks/<record>`): held exclusively by every
//!   mutation of that record ([`Record`], [`Shared::publish`],
//!   [`Shared::put_if_absent`], [`Staged::publish`], [`Store::quarantine`],
//!   [`Lease::write`]), and shared by readers that must pin the path against
//!   eviction or removal ([`Store::hold`], [`Store::try_share_owned`]).
//!
//! Lock files under `.locks/` are never unlinked while they can be held or
//! awaited: [`Record::remove`] and [`Lease`] drop remove only the record file
//! itself. Orphaned lock files are reclaimed exclusively by [`Store::evict`]
//! while it holds `.lock` exclusively (so no waiter can be opening or awaiting
//! any lock file) and only after closing its own probe handle.
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
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// What a store operation failed with.
#[derive(Debug)]
pub enum StoreError {
    /// A stored file was larger than the caller's read bound.
    TooLarge { path: PathBuf, len: u64, limit: u64 },
    /// A stored path was not a regular file.
    NotAFile { path: PathBuf },
    /// The volume holding the store does not support atomic replacement and
    /// unlinking of open files.
    UnsupportedVolume { path: PathBuf },
    /// An underlying filesystem operation failed.
    Io { operation: &'static str, path: PathBuf, source: std::io::Error },
    /// A staged directory publication failed and restoring the previous
    /// target also failed.
    RestoreFailed {
        target: PathBuf,
        aside: PathBuf,
        publish: std::io::Error,
        restore: std::io::Error,
    },
}

impl StoreError {
    fn io(operation: &'static str, path: &Path, source: std::io::Error) -> Self {
        Self::Io { operation, path: path.to_owned(), source }
    }

    /// The underlying I/O error kind, if this failure came from an OS call.
    pub fn io_kind(&self) -> Option<ErrorKind> {
        match self {
            Self::Io { source, .. } => Some(source.kind()),
            Self::RestoreFailed { publish, .. } => Some(publish.kind()),
            Self::TooLarge { .. } | Self::NotAFile { .. } | Self::UnsupportedVolume { .. } => None,
        }
    }
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge { path, len, limit } => {
                write!(f, "{}: {len} bytes is larger than the {limit} bytes it is read to", path.display())
            }
            Self::NotAFile { path } => write!(f, "{}: not a regular file", path.display()),
            Self::UnsupportedVolume { path } => {
                write!(f, "{}: volume does not support atomic replace and unlink of open files", path.display())
            }
            Self::Io { operation, path, source } => {
                write!(f, "{operation} {}: {source}", path.display())
            }
            Self::RestoreFailed { target, aside, publish, restore } => {
                write!(
                    f,
                    "publish {} failed ({publish}) and restoring {} also failed ({restore})",
                    target.display(),
                    aside.display()
                )
            }
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::RestoreFailed { publish, .. } => Some(publish),
            Self::TooLarge { .. } | Self::NotAFile { .. } | Self::UnsupportedVolume { .. } => None,
        }
    }
}

/// An OS failure classified once at the boundary so callers match on the
/// outcome instead of re-inspecting `std::io::Error`.
enum PathError {
    Absent(std::io::Error),
    Occupied(std::io::Error),
    Other(std::io::Error),
}

impl PathError {
    fn classify(error: std::io::Error) -> Self {
        #[cfg(windows)]
        {
            use windows_sys::Win32::Foundation::{
                ERROR_ALREADY_EXISTS, ERROR_DIR_NOT_EMPTY, ERROR_FILE_EXISTS, ERROR_FILE_NOT_FOUND,
                ERROR_PATH_NOT_FOUND,
            };
            match error.raw_os_error().map(|code| code as u32) {
                Some(ERROR_FILE_NOT_FOUND | ERROR_PATH_NOT_FOUND) => Self::Absent(error),
                Some(ERROR_DIR_NOT_EMPTY | ERROR_ALREADY_EXISTS | ERROR_FILE_EXISTS) => Self::Occupied(error),
                _ => Self::Other(error),
            }
        }
        #[cfg(not(windows))]
        {
            match error.kind() {
                ErrorKind::NotFound => Self::Absent(error),
                ErrorKind::DirectoryNotEmpty | ErrorKind::AlreadyExists | ErrorKind::NotADirectory => Self::Occupied(error),
                _ => Self::Other(error),
            }
        }
    }

    fn into_io(self) -> std::io::Error {
        match self {
            Self::Absent(error) | Self::Occupied(error) | Self::Other(error) => error,
        }
    }
}

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
    pub fn directory(self, path: &Path) -> Result<(), StoreError> {
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        if self == Privacy::Owner {
            use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
            builder.mode(0o700).create(path).map_err(|source| StoreError::io("create_dir", path, source))?;
            return fs::set_permissions(path, fs::Permissions::from_mode(0o700))
                .map_err(|source| StoreError::io("set_permissions", path, source));
        }
        builder.create(path).map_err(|source| StoreError::io("create_dir", path, source))
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

/// Unlink a private temporary file on Windows with NTFS POSIX delete semantics
/// (`FILE_DISPOSITION_FLAG_POSIX_SEMANTICS`).
#[cfg(windows)]
fn dispose_posix(path: &Path) -> Result<(), PathError> {
    use std::os::windows::{fs::OpenOptionsExt, io::AsRawHandle};
    use windows_sys::Win32::Storage::FileSystem::{
        DELETE, FILE_DISPOSITION_FLAG_DELETE, FILE_DISPOSITION_FLAG_POSIX_SEMANTICS,
        FILE_DISPOSITION_INFO_EX, FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_DELETE, FILE_SHARE_READ,
        FILE_SHARE_WRITE, FileDispositionInfoEx, SetFileInformationByHandle,
    };

    let file = OpenOptions::new()
        .access_mode(DELETE)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
        .map_err(PathError::classify)?;

    let info = FILE_DISPOSITION_INFO_EX {
        Flags: FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS,
    };
    // SAFETY: `file` is a valid handle opened with `DELETE` access, and `info`
    // is a valid `FILE_DISPOSITION_INFO_EX` matching its byte size.
    let ok = unsafe {
        SetFileInformationByHandle(
            file.as_raw_handle(),
            FileDispositionInfoEx,
            (&raw const info).cast(),
            std::mem::size_of::<FILE_DISPOSITION_INFO_EX>() as u32,
        )
    };
    if ok == 0 {
        return Err(PathError::classify(std::io::Error::last_os_error()));
    }
    Ok(())
}

/// Unlink the regular file or symlink at `path`. On Windows, renaming `path`
/// to a private temporary name before calling [`dispose_posix`] ensures no
/// lockless reader can ever observe `path` in the window between
/// `SetFileInformationByHandle` and handle close (`STATUS_DELETE_PENDING`).
#[cfg(windows)]
fn unlink_file(path: &Path) -> Result<(), PathError> {
    let name = path.file_name().unwrap_or_else(|| OsStr::new("unlink"));
    if unpublished(name) {
        return dispose_posix(path);
    }
    let aside = temporary_name(parent(path), name);
    fs::rename(path, &aside).map_err(PathError::classify)?;
    dispose_posix(&aside)
}

#[cfg(not(windows))]
fn unlink_file(path: &Path) -> Result<(), PathError> {
    fs::remove_file(path).map_err(PathError::classify)
}

/// Verify once per volume root per process that the volume holding `root`
/// supports atomic POSIX-semantics rename and delete over open lockless
/// readers.
#[cfg(windows)]
fn probe_volume(root: &Path, privacy: Privacy) -> Result<(), StoreError> {
    use std::collections::HashSet;
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use std::sync::{Mutex, OnceLock};
    use windows_sys::Win32::Foundation::{
        ERROR_INVALID_FUNCTION, ERROR_INVALID_PARAMETER, ERROR_NOT_SUPPORTED,
    };
    use windows_sys::Win32::Storage::FileSystem::GetVolumePathNameW;

    static VERIFIED_VOLUMES: OnceLock<Mutex<HashSet<OsString>>> = OnceLock::new();

    let volume_root = {
        let wide: Vec<u16> = root.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let mut out = vec![0u16; wide.len().max(260)];
        // SAFETY: `wide` is NUL-terminated and `out` has capacity `out.len()`.
        let ok = unsafe { GetVolumePathNameW(wide.as_ptr(), out.as_mut_ptr(), out.len() as u32) };
        if ok == 0 {
            return Err(StoreError::io("volume_path", root, std::io::Error::last_os_error()));
        }
        let len = out.iter().position(|&c| c == 0).unwrap_or(out.len());
        OsString::from_wide(&out[..len])
    };

    let verified = VERIFIED_VOLUMES.get_or_init(|| Mutex::new(HashSet::new()));
    if verified.lock().expect("verified volumes lock").contains(&volume_root) {
        return Ok(());
    }

    let unsupported = |error: &std::io::Error| {
        matches!(
            error.raw_os_error().map(|code| code as u32),
            Some(ERROR_INVALID_FUNCTION | ERROR_NOT_SUPPORTED | ERROR_INVALID_PARAMETER)
        )
    };

    let (probe_a, file_a) = Temporary::create(root, OsStr::new("volume-probe"), privacy)?;
    drop(file_a);
    let (probe_b, file_b) = Temporary::create(root, OsStr::new("volume-probe"), privacy)?;
    drop(file_b);

    let reader_a = File::open(probe_a.path()).map_err(|source| StoreError::io("open", probe_a.path(), source))?;
    if let Err(error) = fs::rename(probe_b.path(), probe_a.path()) {
        drop(reader_a);
        if unsupported(&error) {
            return Err(StoreError::UnsupportedVolume { path: root.to_owned() });
        }
        return Err(StoreError::io("rename", probe_a.path(), error));
    }
    let mut moved_b = std::mem::ManuallyDrop::new(probe_b);
    drop(std::mem::take(&mut moved_b.path));
    drop(reader_a);

    let reader_b = File::open(probe_a.path()).map_err(|source| StoreError::io("open", probe_a.path(), source))?;
    if let Err(error) = unlink_file(probe_a.path()) {
        drop(reader_b);
        let io = error.into_io();
        if unsupported(&io) {
            return Err(StoreError::UnsupportedVolume { path: root.to_owned() });
        }
        return Err(StoreError::io("remove_file", probe_a.path(), io));
    }
    let mut unlinked_a = std::mem::ManuallyDrop::new(probe_a);
    drop(std::mem::take(&mut unlinked_a.path));
    drop(reader_b);

    verified.lock().expect("verified volumes lock").insert(volume_root);
    Ok(())
}

#[cfg(not(windows))]
fn probe_volume(_root: &Path, _privacy: Privacy) -> Result<(), StoreError> {
    Ok(())
}

/// Set the use time (the modification time) of the file or directory at
/// `path` to `when`.
fn touched(path: &Path, when: SystemTime) -> Result<(), StoreError> {
    let mut options = OpenOptions::new();
    #[cfg(not(windows))]
    options.read(true);
    // Setting a time takes `FILE_WRITE_ATTRIBUTES`, and opening a directory
    // `FILE_FLAG_BACKUP_SEMANTICS`.
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::{FILE_FLAG_BACKUP_SEMANTICS, FILE_WRITE_ATTRIBUTES};
        options.access_mode(FILE_WRITE_ATTRIBUTES).custom_flags(FILE_FLAG_BACKUP_SEMANTICS);
    }
    options
        .open(path)
        .and_then(|file| file.set_modified(when))
        .map_err(|source| StoreError::io("set_modified", path, source))
}

/// Refresh the use time of the file at `path`: a stale one only makes its
/// entry an earlier eviction candidate.
fn used(path: &Path) {
    let _ = touched(path, SystemTime::now());
}

/// The bytes of the regular file at `path`, none if no file is there. A
/// file larger than `largest`, or anything at `path` that is no regular
/// file, is an error.
fn read_bounded(path: &Path, largest: u64) -> Result<Option<Vec<u8>>, StoreError> {
    let metadata = match fs::symlink_metadata(path).map_err(PathError::classify) {
        Ok(metadata) => metadata,
        Err(PathError::Absent(_)) => return Ok(None),
        Err(error) => return Err(StoreError::io("metadata", path, error.into_io())),
    };
    if !metadata.is_file() {
        return Err(StoreError::NotAFile { path: path.to_owned() });
    }
    let file = match File::open(path).map_err(PathError::classify) {
        Ok(file) => file,
        Err(PathError::Absent(_)) => return Ok(None),
        Err(error) => return Err(StoreError::io("open", path, error.into_io())),
    };
    let mut bytes = Vec::new();
    file.take(largest.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| StoreError::io("read", path, source))?;
    if bytes.len() as u64 > largest {
        return Err(StoreError::TooLarge { path: path.to_owned(), len: bytes.len() as u64, limit: largest });
    }
    Ok(Some(bytes))
}

/// [`read_bounded`], marking the file used.
fn read_used(path: &Path, largest: u64) -> Result<Option<Vec<u8>>, StoreError> {
    let bytes = read_bounded(path, largest)?;
    if bytes.is_some() {
        used(path);
    }
    Ok(bytes)
}

/// Remove the file at `path`, none being there no failure.
fn removed(path: &Path) -> Result<(), StoreError> {
    match unlink_file(path) {
        Ok(()) | Err(PathError::Absent(_)) => Ok(()),
        Err(error) => Err(StoreError::io("remove_file", path, error.into_io())),
    }
}

/// Sync `directory`, so a rename in it survives a crash.
#[cfg(unix)]
fn synced_directory(directory: &Path) -> Result<(), StoreError> {
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|source| StoreError::io("sync_dir", directory, source))
}

/// Windows journals a rename with the volume's metadata, and opens no
/// directory as a file to sync it.
#[cfg(not(unix))]
fn synced_directory(_: &Path) -> Result<(), StoreError> {
    Ok(())
}

/// Open a lock file at `path`, creating parent directories as needed.
fn open_lock_file(privacy: Privacy, path: &Path) -> Result<File, StoreError> {
    privacy.directory(parent(path))?;
    let mut options = privacy.options();
    options.create(true).truncate(false).read(true).write(true);
    options.open(path).map_err(|source| StoreError::io("open_lock", path, source))
}

/// Map a store-relative `record` path to its lock file path under `locks`.
/// Prefixing directory components with `d.` and the leaf with `f.` makes it
/// impossible for a lock on a parent path (e.g. a repository) to collide with
/// a lock on a child path (e.g. a snapshot under that repository).
fn record_lock_path(locks: &Path, record: &Path) -> PathBuf {
    let mut path = locks.to_owned();
    let components: Vec<&OsStr> = record.iter().collect();
    if let Some((leaf, parents)) = components.split_last() {
        for dir in parents {
            let mut name = std::ffi::OsString::from("d.");
            name.push(dir);
            path.push(name);
        }
        let mut name = std::ffi::OsString::from("f.");
        name.push(leaf);
        path.push(name);
    }
    path
}

/// Inverse of [`record_lock_path`]: recover the store-relative record path
/// from a lock file path under `locks`.
fn lock_path_to_record(locks: &Path, lock: &Path) -> Option<PathBuf> {
    let relative = lock.strip_prefix(locks).ok()?;
    let components: Vec<&OsStr> = relative.iter().collect();
    let (leaf, parents) = components.split_last()?;
    let mut record = PathBuf::new();
    for dir in parents {
        record.push(dir.to_str()?.strip_prefix("d.")?);
    }
    record.push(leaf.to_str()?.strip_prefix("f.")?);
    Some(record)
}

/// Acquire `record`'s Level-2 lock under `root`, given that the caller already
/// holds the Level-1 `.lock` (shared or exclusive).
fn lock_record_under_store_lock(root: &Path, privacy: Privacy, record: &Path, holding: Holding) -> Result<Option<Locked>, StoreError> {
    let path = record_lock_path(&root.join(LOCKS), record);
    let file = open_lock_file(privacy, &path)?;
    match holding {
        Holding::Exclusive => file.lock().map_err(|source| StoreError::io("lock", &path, source))?,
        Holding::Shared => file.lock_shared().map_err(|source| StoreError::io("lock_shared", &path, source))?,
        Holding::Try => {
            if !taken(file.try_lock(), &path)? {
                return Ok(None);
            }
        }
        Holding::TryShared => {
            if !taken(file.try_lock_shared(), &path)? {
                return Ok(None);
            }
        }
    }
    Ok(Some(Locked { record: root.join(record), _file: file }))
}

/// Whether `file` still refers to the file currently linked at `path`.
#[cfg(unix)]
fn same_lock_file(file: &File, path: &Path) -> Result<bool, StoreError> {
    use std::os::unix::fs::MetadataExt;
    let held = file.metadata().map_err(|source| StoreError::io("metadata", path, source))?;
    match fs::metadata(path).map_err(PathError::classify) {
        Ok(on_disk) => Ok((held.dev(), held.ino()) == (on_disk.dev(), on_disk.ino())),
        Err(PathError::Absent(_)) => Ok(false),
        Err(error) => Err(StoreError::io("metadata", path, error.into_io())),
    }
}

#[cfg(windows)]
fn file_identity(file: &File, path: &Path) -> Result<(u32, [u32; 2]), StoreError> {
    use std::os::windows::io::{AsRawHandle, RawHandle};
    #[repr(C)]
    #[derive(Default)]
    struct Information {
        attributes: u32,
        times: [u32; 6],
        volume_serial_number: u32,
        size: [u32; 2],
        links: u32,
        index: [u32; 2],
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetFileInformationByHandle(file: RawHandle, information: *mut Information) -> i32;
    }
    let mut information = Information::default();
    // SAFETY: a handle `file` owns, and a structure of the layout the call writes.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) } == 0 {
        return Err(StoreError::io("file_identity", path, std::io::Error::last_os_error()));
    }
    Ok((information.volume_serial_number, information.index))
}

/// Whether `file` still refers to the file currently linked at `path`.
#[cfg(windows)]
fn same_lock_file(file: &File, path: &Path) -> Result<bool, StoreError> {
    let held = file_identity(file, path)?;
    match File::open(path).map_err(PathError::classify) {
        Ok(on_disk) => Ok(held == file_identity(&on_disk, path)?),
        Err(PathError::Absent(_)) => Ok(false),
        Err(error) => Err(StoreError::io("open_lock", path, error.into_io())),
    }
}

/// Whether `file` still refers to the file currently linked at `path`.
#[cfg(not(any(unix, windows)))]
fn same_lock_file(_file: &File, path: &Path) -> Result<bool, StoreError> {
    match fs::metadata(path).map_err(PathError::classify) {
        Ok(_) => Ok(true),
        Err(PathError::Absent(_)) => Ok(false),
        Err(error) => Err(StoreError::io("metadata", path, error.into_io())),
    }
}

/// Acquire `record`'s Level-2 lock without holding Level-1 `.lock` across
/// blocking waits, preventing AB-BA deadlocks with [`Store::exclusive`] and
/// starvation of [`Store::evict`].
fn lock_record(root: &Path, privacy: Privacy, record: &Path, holding: Holding) -> Result<Option<Locked>, StoreError> {
    let store_lock_path = root.join(LOCK);
    match holding {
        Holding::Try | Holding::TryShared => {
            let store_lock = open_lock_file(privacy, &store_lock_path)?;
            store_lock
                .lock_shared()
                .map_err(|source| StoreError::io("lock_shared", &store_lock_path, source))?;
            lock_record_under_store_lock(root, privacy, record, holding)
        }
        Holding::Exclusive | Holding::Shared => {
            let path = record_lock_path(&root.join(LOCKS), record);
            loop {
                let store_lock = open_lock_file(privacy, &store_lock_path)?;
                store_lock
                    .lock_shared()
                    .map_err(|source| StoreError::io("lock_shared", &store_lock_path, source))?;
                let file = open_lock_file(privacy, &path)?;
                let acquired = match holding {
                    Holding::Exclusive => taken(file.try_lock(), &path)?,
                    Holding::Shared => taken(file.try_lock_shared(), &path)?,
                    Holding::Try | Holding::TryShared => unreachable!(),
                };
                if acquired {
                    drop(store_lock);
                    return Ok(Some(Locked { record: root.join(record), _file: file }));
                }
                // Release Level-1 `.lock` before blocking on Level-2 so
                // `Store::exclusive` and `Store::evict` are neither deadlocked
                // nor starved while waiting.
                drop(store_lock);
                match holding {
                    Holding::Exclusive => file.lock().map_err(|source| StoreError::io("lock", &path, source))?,
                    Holding::Shared => file.lock_shared().map_err(|source| StoreError::io("lock_shared", &path, source))?,
                    Holding::Try | Holding::TryShared => unreachable!(),
                }
                // Re-verify under Level-1 `.lock` shared that `path` was not
                // unlinked or replaced while waiting. Use non-blocking try so
                // holding Level-2 never blocks on an exclusive Level-1 holder;
                // if contended, release Level-2 and retry from Level-1 first.
                let store_lock = open_lock_file(privacy, &store_lock_path)?;
                if taken(store_lock.try_lock_shared(), &store_lock_path)? {
                    let same = same_lock_file(&file, &path)?;
                    drop(store_lock);
                    if same {
                        return Ok(Some(Locked { record: root.join(record), _file: file }));
                    }
                }
                drop(file);
            }
        }
    }
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
    fn create_classified(directory: &Path, stem: &OsStr, privacy: Privacy) -> Result<(Self, File), (PathBuf, PathError)> {
        loop {
            let path = temporary_name(directory, stem);
            let mut options = privacy.options();
            options.read(true).write(true).create_new(true);
            // Only a dead writer of the same process id leaves the name taken.
            match options.open(&path).map_err(PathError::classify) {
                Ok(file) => return Ok((Self { path }, file)),
                Err(PathError::Occupied(_)) => continue,
                Err(error) => return Err((path, error)),
            }
        }
    }

    /// A new empty file in `directory`, named of `stem`, created
    /// exclusively, as `privacy` says.
    pub fn create(directory: &Path, stem: &OsStr, privacy: Privacy) -> Result<(Self, File), StoreError> {
        Self::create_classified(directory, stem, privacy)
            .map_err(|(path, error)| StoreError::io("create_temp", &path, error.into_io()))
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
    fn rename(self, target: &Path) -> Result<(), StoreError> {
        fs::rename(&self.path, target).map_err(|source| StoreError::io("rename", target, source))?;
        let mut renamed = std::mem::ManuallyDrop::new(self);
        drop(std::mem::take(&mut renamed.path));
        Ok(())
    }
}

impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = unlink_file(&self.path);
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
    /// Open the store at `root`, creating `root` if absent and verifying once
    /// that its volume supports atomic replacement and unlinking of open files.
    pub fn open(root: PathBuf, privacy: Privacy) -> Result<Self, StoreError> {
        privacy.directory(&root)?;
        probe_volume(&root, privacy)?;
        Ok(Self { root, privacy, written: AtomicU64::new(0) })
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
    /// file, as one atomic step, to `durability`. Caller holds the exclusive
    /// lock serializing `path`. On any failure `path` is as it was and no
    /// temporary file is left, unless the failure is syncing the rename
    /// itself, of a durable publication, after which the new file is visible
    /// but may not survive a crash.
    fn published(&self, path: &Path, durability: Durability, write: impl FnOnce(&mut File) -> std::io::Result<()>) -> Result<(), StoreError> {
        let directory = parent(path);
        let name = path.file_name().expect("a stored path has a name");
        let (temporary, mut file) = loop {
            self.privacy.directory(directory)?;
            match Temporary::create_classified(directory, name, self.privacy) {
                Ok(created) => break created,
                // An eviction removed the directory, emptied, since it was
                // created: create it again.
                Err((_, PathError::Absent(_))) => continue,
                Err((path, error)) => return Err(StoreError::io("create_temp", &path, error.into_io())),
            }
        };
        write(&mut file).map_err(|source| StoreError::io("write", temporary.path(), source))?;
        if durability == Durability::Durable {
            file.sync_all().map_err(|source| StoreError::io("sync_all", temporary.path(), source))?;
        }
        let length = file.metadata().map_err(|source| StoreError::io("metadata", temporary.path(), source))?.len();
        drop(file);
        temporary.rename(path)?;
        self.written.fetch_add(length, Ordering::Relaxed);
        match durability {
            Durability::Durable => synced_directory(directory),
            Durability::Volatile => Ok(()),
        }
    }

    /// The store's lock held shared, waiting until no exclusive holder holds
    /// it: every file is as one writer left it while it lives, and no eviction
    /// runs. Also serves as the Level-1 proof required by [`Shared::publish`]
    /// and [`Shared::put_if_absent`].
    pub fn shared(&self) -> Result<Shared<'_>, StoreError> {
        let lock_path = self.root.join(LOCK);
        let lock = open_lock_file(self.privacy, &lock_path)?;
        lock.lock_shared().map_err(|source| StoreError::io("lock_shared", &lock_path, source))?;
        Ok(Shared { store: self, _lock: lock })
    }

    /// The store's lock held exclusively, waiting until no other holder holds
    /// it: no reader, other writer or eviction is among the files while it
    /// lives, nor is a record's lock being taken. A thread holding the
    /// store's lock takes it again only to wait for itself.
    pub fn exclusive(&self) -> Result<Exclusive<'_>, StoreError> {
        let lock_path = self.root.join(LOCK);
        let lock = open_lock_file(self.privacy, &lock_path)?;
        lock.lock().map_err(|source| StoreError::io("lock", &lock_path, source))?;
        Ok(Exclusive { store: self, _lock: lock })
    }

    /// The lock of the record at `record` under the root, held exclusively,
    /// waiting until no other holder holds it: what reads, writes and removes
    /// the record, which eviction leaves while it is held.
    pub fn lock(&self, record: &Path) -> Result<Record<'_>, StoreError> {
        Ok(Record { store: self, locked: self.waited(record, Holding::Exclusive)? })
    }

    /// [`Store::lock`], none while another holder holds the record, without
    /// waiting.
    pub fn try_lock(&self, record: &Path) -> Result<Option<Record<'_>>, StoreError> {
        Ok(self.locked(record, Holding::Try)?.map(|locked| Record { store: self, locked }))
    }

    /// [`Store::try_lock`], the hold borrowing nothing of its store.
    pub fn try_lock_owned(&self, record: &Path) -> Result<Option<OwnedRecord>, StoreError> {
        Ok(self.locked(record, Holding::Try)?.map(|locked| OwnedRecord { _locked: locked }))
    }

    /// A lease of this process among the holders of `directory`: the record
    /// `<directory>/<pid>.<n>`, published empty, which its holder may publish
    /// again ([`Lease::write`]) and whose lock it holds until the lease is
    /// dropped, which removes the record file. [`Store::leases`] finds it
    /// meanwhile, from whichever process, and removes it once its holder is
    /// gone unawares.
    pub fn lease(&self, directory: &Path) -> Result<Lease, StoreError> {
        let record = directory.join(format!("{}.{}", std::process::id(), LEASES.fetch_add(1, Ordering::Relaxed)));
        let locked = self.waited(&record, Holding::Exclusive)?;
        locked.write(self, &[], Durability::Volatile)?;
        Ok(Lease { record, locked: Some(locked) })
    }

    /// The records of the live leases of `directory` ([`Store::lease`]), in
    /// whichever process, this one's among them: each whose lock a holder
    /// holds. A lease no holder holds is a gone holder's, and is removed.
    pub fn leases(&self, directory: &Path) -> Result<Vec<PathBuf>, StoreError> {
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

    /// [`Store::lock`], the hold borrowing nothing of its store.
    pub fn lock_owned(&self, record: &Path) -> Result<OwnedRecord, StoreError> {
        Ok(OwnedRecord { _locked: self.waited(record, Holding::Exclusive)? })
    }

    /// The record's lock held shared with its other readers, borrowing
    /// nothing of its store, none while an exclusive holder holds it, without
    /// waiting.
    pub fn try_share_owned(&self, record: &Path) -> Result<Option<OwnedRecord>, StoreError> {
        Ok(self.locked(record, Holding::TryShared)?.map(|locked| OwnedRecord { _locked: locked }))
    }

    /// [`Store::locked`] as a waiting `holding`, which always takes it.
    fn waited(&self, record: &Path, holding: Holding) -> Result<Locked, StoreError> {
        Ok(self.locked(record, holding)?.expect("a waiting lock is taken once its holders are gone"))
    }

    /// The record's lock held as `holding` says, none where it is a try and
    /// another holder holds it.
    fn locked(&self, record: &Path, holding: Holding) -> Result<Option<Locked>, StoreError> {
        lock_record(&self.root, self.privacy, record, holding)
    }

    /// The bytes of the record at `path` under the root, none if none is
    /// there, read without a lock: every record is published whole, and the
    /// volume supports atomic replace and unlink over open readers. Larger
    /// than `largest`, or no regular file, is an error.
    pub fn read(&self, path: &Path, largest: u64) -> Result<Option<Vec<u8>>, StoreError> {
        read_bounded(&self.root.join(path), largest)
    }

    /// The bytes of the file at `path` under the root, none if none is there,
    /// read without a lock and marked used.
    pub fn fetch(&self, path: &Path) -> Result<Option<Vec<u8>>, StoreError> {
        read_used(&self.root.join(path), u64::MAX)
    }

    /// The names of the files in `directory` under the root, none if it is
    /// absent; temporary files and the store's own are none of them.
    pub fn list(&self, directory: &Path) -> Result<Vec<String>, StoreError> {
        let full = self.root.join(directory);
        let listing = match fs::read_dir(&full).map_err(PathError::classify) {
            Ok(listing) => listing,
            Err(PathError::Absent(_)) => return Ok(Vec::new()),
            Err(error) => return Err(StoreError::io("read_dir", &full, error.into_io())),
        };
        let mut names = Vec::new();
        for item in listing {
            let name = item.map_err(|source| StoreError::io("read_dir", &full, source))?.file_name();
            if !unpublished(&name) && !owned(&name) {
                names.push(name.to_string_lossy().into_owned());
            }
        }
        Ok(names)
    }

    /// The file at `path` under the root, a content address, opened and
    /// marked used, held under its own lock shared: eviction keeps it at its
    /// path while the hold lives. None if no file is there.
    pub fn hold(&self, path: &Path) -> Result<Option<Held>, StoreError> {
        let locked = self.waited(path, Holding::Shared)?;
        match File::open(&locked.record).map_err(PathError::classify) {
            Ok(file) => {
                used(&locked.record);
                Ok(Some(Held { file, locked }))
            }
            Err(PathError::Absent(_)) => Ok(None),
            Err(error) => Err(StoreError::io("open", &locked.record, error.into_io())),
        }
    }

    /// An empty directory staged for `target` under the root, beside it, as
    /// the store's privacy says, which [`Staged::publish`] renames into place
    /// whole under `target`'s exclusive record lock.
    pub fn stage(&self, target: &Path) -> Result<Staged, StoreError> {
        let full = self.root.join(target);
        let directory = parent(&full);
        let name = full.file_name().expect("a staged path has a name").to_owned();
        loop {
            self.privacy.directory(directory)?;
            let stage = temporary_name(directory, &name);
            match fs::create_dir(&stage).map_err(PathError::classify) {
                Ok(()) => {
                    self.privacy.directory(&stage)?;
                    return Ok(Staged {
                        root: self.root.clone(),
                        privacy: self.privacy,
                        relative: target.to_owned(),
                        target: full,
                        stage,
                        published: false,
                    });
                }
                // Only a dead writer of the same process id leaves the name
                // taken; an eviction may have removed the directory, emptied.
                Err(PathError::Occupied(_) | PathError::Absent(_)) => continue,
                Err(error) => return Err(StoreError::io("create_dir", &stage, error.into_io())),
            }
        }
    }

    /// Set what is at `path` under the root (a file or directory where the
    /// store expects something else) aside, under a temporary name beside
    /// it, serialized under `path`'s exclusive record lock.
    pub fn quarantine(&self, path: &Path) -> Result<(), StoreError> {
        let _held = self.waited(path, Holding::Exclusive)?;
        let full = self.root.join(path);
        let aside = temporary_name(parent(&full), full.file_name().expect("a quarantined path has a name"));
        fs::rename(&full, &aside).map_err(|source| StoreError::io("rename", &full, source))?;
        // Its age is its quarantine's, so eviction keeps it its full span; a
        // link is aged by nothing, never followed to its target.
        if fs::symlink_metadata(&aside).is_ok_and(|metadata| !metadata.file_type().is_symlink()) {
            let _ = touched(&aside, SystemTime::now());
        }
        Ok(())
    }

    /// Remove the least recently used entries until the rest fit `capacity`
    /// bytes, as `layout` places each file, the store's lock held
    /// exclusively from the listing to the last removal.
    pub fn evict(&self, capacity: u64, grace: Duration, layout: &dyn Layout) -> Result<bool, StoreError> {
        // Counted from before the lock is tried: an eviction that passes is
        // begun again only once as much more is written.
        self.written.store(0, Ordering::Relaxed);
        let lock_path = self.root.join(LOCK);
        let lock = open_lock_file(self.privacy, &lock_path)?;
        if !taken(lock.try_lock(), &lock_path)? {
            return Ok(false);
        }
        let _held = Exclusive { store: self, _lock: lock };
        let locks = self.root.join(LOCKS);
        // Whether a holder holds the lock of the file at `path`: no lock is
        // taken while the store's is held exclusively, so one free now stays
        // free until the eviction ends, and the probe handle is dropped before
        // returning so zero handles remain open when unlinking an orphaned lock.
        let in_use = |path: &Path| {
            let lock = record_lock_path(&locks, path.strip_prefix(&self.root).expect("eviction walks under its root"));
            match OpenOptions::new().read(true).write(true).open(&lock).map_err(PathError::classify) {
                Ok(file) => !matches!(file.try_lock(), Ok(())),
                Err(PathError::Absent(_)) => false,
                Err(_) => true,
            }
        };
        let now = SystemTime::now();
        let recent = |used: SystemTime| match now.duration_since(used) {
            Ok(age) => age < grace,
            Err(_) => !grace.is_zero(),
        };
        let examining = Examining { root: &self.root };
        let relative = |path: &Path| path.strip_prefix(&self.root).expect("eviction walks under its root").to_owned();
        let mut entries = HashMap::<Entry, Listed>::new();
        let mut shared = BTreeMap::<PathBuf, (u64, SystemTime, bool)>::new();
        let mut referenced = BTreeSet::<PathBuf>::new();
        let mut emptied = BTreeSet::<PathBuf>::new();
        let mut remove = |path: &Path, directory: bool| -> Result<(), std::io::Error> {
            let result = match directory {
                true => fs::remove_dir_all(path),
                false => unlink_file(path).map_err(PathError::into_io),
            };
            if result.is_ok() {
                emptied.insert(parent(path).to_owned());
            }
            result
        };
        let mut directories = vec![(self.root.clone(), false, false)];
        while let Some((directory, kept, dir_in_use)) = directories.pop() {
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
                    if !dir_in_use && now.duration_since(modified).unwrap_or_default() > ABANDONED {
                        let _ = remove(&path, metadata.is_dir());
                    }
                    continue;
                }
                if directory == self.root && owned(&name) {
                    continue;
                }
                let place = |path: &Path| layout.place(&relative(path), &examining);
                if metadata.is_dir() {
                    let locked_dir = dir_in_use || in_use(&path);
                    match place(&path) {
                        Eviction::Apart => {}
                        placed => directories.push((path, kept || placed == Eviction::Kept, locked_dir)),
                    }
                    continue;
                }
                if kept {
                    continue;
                }
                let used_now = dir_in_use || in_use(&path);
                let (entry, order, pinned) = match place(&path) {
                    Eviction::Kept | Eviction::Apart => {
                        referenced.extend(layout.references(&relative(&path), &examining));
                        continue;
                    }
                    Eviction::Unreachable => {
                        if !recent(modified) && !used_now {
                            if let Err(error) = remove(&path, false) {
                                tracing::warn!("store {}: cannot remove {}, which nothing reaches: {error}", self.root.display(), path.display());
                            }
                        }
                        continue;
                    }
                    Eviction::Shared => {
                        shared.insert(relative(&path), (metadata.len(), modified, used_now));
                        continue;
                    }
                    Eviction::Alone => (Entry::Alone(path.clone()), 0, false),
                    Eviction::Pinned => (Entry::Alone(path.clone()), 0, true),
                    Eviction::Part { entry, order } => (Entry::Named(entry), order, false),
                };
                let references = layout.references(&relative(&path), &examining);
                let entry = entries.entry(entry).or_insert(Listed { pinned, used: SystemTime::UNIX_EPOCH, bytes: 0, paths: Vec::new(), references: Vec::new() });
                entry.pinned |= used_now;
                entry.used = entry.used.max(modified);
                entry.bytes += metadata.len();
                entry.paths.push((order, path));
                entry.references.extend(references);
            }
        }
        let mut entries = entries.into_values().collect::<Vec<_>>();
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
        // the lock of a record that is gone has zero open handles once no
        // holder holds it, so unlinking it cannot enter pending deletion.
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
                    let orphan = match lock_path_to_record(&locks, &path) {
                        Some(relative_record) => {
                            let record = self.root.join(relative_record);
                            fs::symlink_metadata(&record).is_err() && !in_use(&record)
                        }
                        // Pre-upgrade lock file without `d.`/`f.` prefixes:
                        // reclaim once no holder holds it.
                        None => match OpenOptions::new().read(true).write(true).open(&path).map_err(PathError::classify) {
                            Ok(file) => matches!(file.try_lock(), Ok(())),
                            Err(_) => false,
                        },
                    };
                    if orphan {
                        let _ = remove(&path, false);
                    }
                }
            }
        }
        while let Some(directory) = emptied.pop_last() {
            if directory != self.root && directory != locks && fs::remove_dir(&directory).is_ok() {
                emptied.insert(parent(&directory).to_owned());
            }
        }
        Ok(true)
    }
}

/// The store's Level-1 lock held shared: every file is as one writer left it
/// until this is dropped, no eviction runs, and per-record publications
/// ([`Shared::publish`], [`Shared::put_if_absent`]) acquire their Level-2
/// record lock directly under this hold.
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
    pub fn read(&self, path: &Path, largest: u64) -> Result<Option<Vec<u8>>, StoreError> {
        read_used(&self.store.root.join(path), largest)
    }

    /// Publish what `write` writes at `path` under the root, to `durability`,
    /// serialized under `path`'s exclusive record lock. A failure leaves no
    /// partial file.
    pub fn publish(&self, path: &Path, durability: Durability, write: impl FnOnce(&mut File) -> std::io::Result<()>) -> Result<(), StoreError> {
        let _record = lock_record_under_store_lock(&self.store.root, self.store.privacy, path, Holding::Exclusive)?;
        self.store.published(&self.store.root.join(path), durability, write)
    }

    /// Publish `bytes` at `path` under the root, a content address, unless a
    /// file of their length is there already, which is then marked used:
    /// whether they were written.
    pub fn put_if_absent(&self, path: &Path, bytes: &[u8], durability: Durability) -> Result<bool, StoreError> {
        let full = self.store.root.join(path);
        if fs::symlink_metadata(&full).is_ok_and(|metadata| metadata.is_file() && metadata.len() == bytes.len() as u64) {
            used(&full);
            return Ok(false);
        }
        let _record = lock_record_under_store_lock(&self.store.root, self.store.privacy, path, Holding::Exclusive)?;
        if fs::symlink_metadata(&full).is_ok_and(|metadata| metadata.is_file() && metadata.len() == bytes.len() as u64) {
            used(&full);
            return Ok(false);
        }
        self.store.published(&full, durability, |file| file.write_all(bytes))?;
        Ok(true)
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
    pub fn examine(&self, path: &Path, largest: u64) -> Result<Option<Vec<u8>>, StoreError> {
        read_bounded(&self.store.root.join(path), largest)
    }

    /// Publish `bytes` at `path` under the root, to `durability`, serialized
    /// under `path`'s exclusive record lock. A failure leaves no partial file.
    pub fn write(&self, path: &Path, bytes: &[u8], durability: Durability) -> Result<(), StoreError> {
        let _record = lock_record_under_store_lock(&self.store.root, self.store.privacy, path, Holding::Exclusive)?;
        self.store.published(&self.store.root.join(path), durability, |file| file.write_all(bytes))
    }

    /// Append `bytes` to the file at `path` under the root, created if
    /// absent, serialized under `path`'s exclusive record lock: every append
    /// of every holder lands whole, in turn.
    pub fn append(&self, path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
        let _record = lock_record_under_store_lock(&self.store.root, self.store.privacy, path, Holding::Exclusive)?;
        let full = self.store.root.join(path);
        self.store.privacy.directory(parent(&full))?;
        let mut options = self.store.privacy.options();
        options.create(true).append(true);
        options
            .open(&full)
            .and_then(|mut file| file.write_all(bytes))
            .map_err(|source| StoreError::io("append", &full, source))?;
        self.store.written.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Remove the file at `path` under the root, if one is there, serialized
    /// under `path`'s exclusive record lock.
    pub fn remove(&self, path: &Path) -> Result<(), StoreError> {
        let _record = lock_record_under_store_lock(&self.store.root, self.store.privacy, path, Holding::Exclusive)?;
        removed(&self.store.root.join(path))
    }

    /// [`Store::list`].
    pub fn list(&self, directory: &Path) -> Result<Vec<String>, StoreError> {
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
fn taken(tried: std::result::Result<(), std::fs::TryLockError>, path: &Path) -> Result<bool, StoreError> {
    match tried {
        Ok(()) => Ok(true),
        Err(std::fs::TryLockError::WouldBlock) => Ok(false),
        Err(std::fs::TryLockError::Error(error)) => Err(StoreError::io("try_lock", path, error)),
    }
}

/// A record's lock held.
#[derive(Debug)]
struct Locked {
    /// The record's path, absolute.
    record: PathBuf,
    _file: File,
}

impl Locked {
    fn read(&self, largest: u64) -> Result<Option<Vec<u8>>, StoreError> {
        read_bounded(&self.record, largest)
    }

    fn write(&self, store: &Store, bytes: &[u8], durability: Durability) -> Result<(), StoreError> {
        store.published(&self.record, durability, |file| file.write_all(bytes))
    }

    fn remove(self) -> Result<(), StoreError> {
        removed(&self.record)
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
    pub fn read(&self, largest: u64) -> Result<Option<Vec<u8>>, StoreError> {
        self.locked.read(largest)
    }

    /// Publish `bytes` as the record, to `durability`. A failure leaves the
    /// record as it was.
    pub fn write(&self, bytes: &[u8], durability: Durability) -> Result<(), StoreError> {
        self.locked.write(self.store, bytes, durability)
    }

    /// Remove the record file if one is there. Its lock file stays until
    /// [`Store::evict`] reclaims it under the store's exclusive lock.
    pub fn remove(self) -> Result<(), StoreError> {
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
    /// The lease's record under the root.
    pub fn record(&self) -> &Path {
        &self.record
    }

    /// Publish `bytes` as the lease's record, to `durability`, under the
    /// exclusive record lock this lease already holds.
    pub fn write(&self, store: &Store, bytes: &[u8], durability: Durability) -> Result<(), StoreError> {
        let locked = self.locked.as_ref().expect("a live lease holds its lock");
        locked.write(store, bytes, durability)
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
    pub fn bytes(&self) -> Result<Vec<u8>, StoreError> {
        use std::io::Seek;
        let mut bytes = Vec::new();
        (&self.file)
            .seek(std::io::SeekFrom::Start(0))
            .and_then(|_| (&self.file).read_to_end(&mut bytes))
            .map_err(|source| StoreError::io("read", &self.locked.record, source))?;
        Ok(bytes)
    }
}

/// A directory being written beside its target, under a temporary name,
/// which [`Staged::publish`] renames into place whole under the target's
/// exclusive record lock: a reader finds the target as one publication left
/// it, or briefly absent while one replaces another, never part written. Its
/// writer fills it at [`Staged::path`], over as many steps as it needs.
/// Dropped unpublished, it is removed. Nothing of it is synced: a crash may
/// lose it.
#[derive(Debug)]
pub struct Staged {
    root: PathBuf,
    privacy: Privacy,
    relative: PathBuf,
    target: PathBuf,
    stage: PathBuf,
    published: bool,
}

impl Staged {
    /// Where the directory is written until it is published.
    pub fn path(&self) -> &Path {
        &self.stage
    }

    /// Rename the stage into place under the target's exclusive record lock.
    /// A directory renames over no non-empty one, so what holds the target is
    /// first renamed aside under a temporary name and removed once this stage
    /// is in place. If renaming the stage fails and restoring the previous
    /// target also fails, [`StoreError::RestoreFailed`] is returned.
    pub fn publish(mut self) -> Result<(), StoreError> {
        let _held = lock_record(&self.root, self.privacy, &self.relative, Holding::Exclusive)?;
        let directory = parent(&self.target);
        let name = self.target.file_name().expect("a staged path has a name");
        let aside = match fs::symlink_metadata(&self.target).map_err(PathError::classify) {
            Ok(_) => {
                let aside = temporary_name(directory, name);
                fs::rename(&self.target, &aside).map_err(|source| StoreError::io("rename", &self.target, source))?;
                Some(aside)
            }
            Err(PathError::Absent(_)) => None,
            Err(error) => return Err(StoreError::io("metadata", &self.target, error.into_io())),
        };
        if let Err(publish) = fs::rename(&self.stage, &self.target) {
            if let Some(aside) = aside {
                if let Err(restore) = fs::rename(&aside, &self.target) {
                    return Err(StoreError::RestoreFailed {
                        target: self.target.clone(),
                        aside,
                        publish,
                        restore,
                    });
                }
            }
            return Err(StoreError::io("rename", &self.target, publish));
        }
        self.published = true;
        if let Some(aside) = aside {
            let meta = fs::symlink_metadata(&aside).map_err(|source| StoreError::io("metadata", &aside, source))?;
            match meta.is_dir() {
                true => fs::remove_dir_all(&aside).map_err(|source| StoreError::io("remove_dir_all", &aside, source))?,
                false => unlink_file(&aside).map_err(|error| StoreError::io("remove_file", &aside, error.into_io()))?,
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
    pub fn examine(&self, path: &Path, largest: u64) -> Result<Option<Vec<u8>>, StoreError> {
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
        let store = Store::open(root.clone(), Privacy::Owner).unwrap();
        let path = Path::new("pointer");
        store.shared().unwrap().publish(path, Durability::Durable, |file| file.write_all(b"first")).unwrap();
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
        let store = Store::open(root.clone(), Privacy::Shared).unwrap();
        let next = TEMPORARIES.load(Ordering::Relaxed);
        let taken = (next..next + 64).map(|count| root.join(format!(".entry.{}.{count}{UNPUBLISHED}", std::process::id()))).collect::<Vec<_>>();
        for left in &taken {
            fs::write(left, b"abandoned").unwrap();
        }
        store.shared().unwrap().publish(Path::new("entry"), Durability::Volatile, |file| file.write_all(b"whole")).unwrap();
        assert_eq!(fs::read(root.join("entry")).unwrap(), b"whole");
        for left in &taken {
            assert_eq!(fs::read(left).unwrap(), b"abandoned");
        }
        fs::remove_dir_all(root).unwrap();
    }

    /// Writers updating one record at once, each a store of its own as
    /// another process holds, each read what the others wrote: every one of
    /// their additions is kept. Removals never unlink lock files while
    /// waiters can hold them; eviction reclaims orphaned lock files once the
    /// record is gone.
    #[test]
    fn updates_of_a_record_through_its_lock_lose_nothing() {
        let root = scratch("updates");
        let store = Store::open(root.clone(), Privacy::Shared).unwrap();
        let record = Path::new("counts/sum");
        let (writers, additions) = (4u64, 100u64);
        std::thread::scope(|scope| {
            for writer in 0..writers {
                let root = root.clone();
                seismic_threads::spawn_scoped(scope, format!("updating-{writer}"), move || {
                    let store = Store::open(root, Privacy::Shared).unwrap();
                    for _ in 0..additions {
                        let held = store.lock(record).unwrap();
                        let sum = held.read(32).unwrap().map_or(0, |bytes| String::from_utf8(bytes).unwrap().parse::<u64>().unwrap());
                        held.write((sum + 1).to_string().as_bytes(), Durability::Volatile).unwrap();
                    }
                })
                .unwrap();
            }
        });
        assert_eq!(store.read(record, 32).unwrap().unwrap(), (writers * additions).to_string().as_bytes());
        assert!(record_lock_path(&root.join(LOCKS), record).exists());
        let (inside, since_removal) = (AtomicU64::new(0), AtomicU64::new(0));
        std::thread::scope(|scope| {
            for writer in 0..writers {
                let (root, inside, since_removal) = (root.clone(), &inside, &since_removal);
                seismic_threads::spawn_scoped(scope, format!("removing-{writer}"), move || {
                    let store = Store::open(root, Privacy::Shared).unwrap();
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
        assert!(store.evict(u64::MAX, Duration::ZERO, &alone).unwrap());
        assert!(!record_lock_path(&root.join(LOCKS), record).exists(), "eviction reclaims the removed record's lock");
        fs::remove_dir_all(root).unwrap();
    }

    /// A record's readers share its lock, which an exclusive try refuses while
    /// any of them, in this process or another, holds it, and takes once
    /// they are gone; a reader's try is refused while that one holds it.
    #[test]
    fn readers_share_a_record_a_try_refuses_while_they_hold_it() {
        let root = scratch("readers");
        let (store, other) = (Store::open(root.clone(), Privacy::Shared).unwrap(), Store::open(root.clone(), Privacy::Shared).unwrap());
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
        let (store, other) = (Store::open(root.clone(), Privacy::Shared).unwrap(), Store::open(root.clone(), Privacy::Shared).unwrap());
        let directory = Path::new("readers/model");
        let (first, second) = (store.lease(directory).unwrap(), other.lease(directory).unwrap());
        first.write(&store, b"first-pins", Durability::Volatile).unwrap();
        assert_eq!(other.read(first.record(), 64).unwrap().unwrap(), b"first-pins");
        let mut live = other.leases(directory).unwrap();
        live.sort();
        let mut expected = vec![first.record().to_owned(), second.record().to_owned()];
        expected.sort();
        assert_eq!(live, expected);
        drop(first);
        assert_eq!(store.leases(directory).unwrap(), vec![second.record().to_owned()]);
        let abandoned = directory.join("0.0");
        store.shared().unwrap().publish(&abandoned, Durability::Volatile, |_| Ok(())).unwrap();
        assert_eq!(store.leases(directory).unwrap(), vec![second.record().to_owned()], "a gone holder's lease is not live");
        assert!(!root.join(&abandoned).exists(), "and is removed once found");
        drop(second);
        assert!(store.leases(directory).unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    /// A record larger than it is read to, or no regular file, is a typed
    /// error, never an absent record.
    #[test]
    fn an_oversized_or_foreign_record_is_an_error() {
        let root = scratch("bounded");
        let store = Store::open(root.clone(), Privacy::Shared).unwrap();
        store.shared().unwrap().publish(Path::new("big"), Durability::Volatile, |file| file.write_all(&[0; 10])).unwrap();
        assert!(matches!(store.read(Path::new("big"), 9), Err(StoreError::TooLarge { limit: 9, .. })));
        assert_eq!(store.read(Path::new("big"), 10).unwrap().unwrap().len(), 10);
        fs::create_dir_all(root.join("directory")).unwrap();
        assert!(matches!(store.read(Path::new("directory"), 10), Err(StoreError::NotAFile { .. })));
        assert!(store.read(Path::new("absent"), 10).unwrap().is_none());
        fs::remove_dir_all(root).unwrap();
    }

    /// Appends of holders at once each land whole.
    #[test]
    fn appends_land_whole() {
        let root = scratch("appends");
        let _store = Store::open(root.clone(), Privacy::Shared).unwrap();
        std::thread::scope(|scope| {
            for writer in 0..4u8 {
                let root = root.clone();
                seismic_threads::spawn_scoped(scope, format!("appending-{writer}"), move || {
                    let store = Store::open(root, Privacy::Shared).unwrap();
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
        let store = Store::open(root.clone(), Privacy::Shared).unwrap();
        let write = |path: &str, bytes: &[u8]| store.shared().unwrap().publish(Path::new(path), Durability::Volatile, |file| file.write_all(bytes)).unwrap();
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
        let legacy_lock = root.join(LOCKS).join("recipes/legacy");
        fs::create_dir_all(parent(&legacy_lock)).unwrap();
        fs::write(&legacy_lock, b"").unwrap();
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
        assert!(store.evict(700, Duration::from_secs(600), &Recipes).unwrap());
        assert!(root.join("foreign/fresh").exists() && !root.join("foreign/old").exists(), "an unreachable file goes once past the grace");
        assert!(theirs.exists(), "a directory placed apart is never walked");
        assert!(root.join("recipes/new").exists() && root.join("chunks/a").exists(), "the newest entry that fits stays, with what it references");
        assert!(!root.join("recipes/old").exists() && !root.join("chunks/b").exists(), "what does not fit goes, with what only it referenced");
        assert!(root.join("pinned/one").exists() && root.join("chunks/c").exists(), "a pinned entry stays, with what it references");
        assert!(root.join("kept/measured").exists());
        assert!(!abandoned.exists() && !abandoned_stage.exists(), "abandoned temporaries go");
        assert!(!record_lock_path(&root.join(LOCKS), Path::new("recipes/never")).exists(), "the lock of a record that is gone goes");
        assert!(!record_lock_path(&root.join(LOCKS), Path::new("recipes/gone")).exists(), "the lock of a removed record goes on eviction");
        assert!(!legacy_lock.exists(), "a pre-upgrade lock file goes on eviction");
        assert!(root.join(LOCK).exists(), "eviction never removes the lock it holds");
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
        let (store, other) = (Store::open(root.clone(), Privacy::Shared).unwrap(), Store::open(root.clone(), Privacy::Shared).unwrap());
        for path in ["artifact", "unused"] {
            store.shared().unwrap().publish(Path::new(path), Durability::Volatile, |file| file.write_all(b"compiled")).unwrap();
        }
        assert!(!store.shared().unwrap().put_if_absent(Path::new("artifact"), b"compiled", Durability::Volatile).unwrap());
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
        assert!(record_lock_path(&root.join(LOCKS), Path::new("readers/model")).exists(), "a lease's lock stays while it is held");
        assert!(store.try_lock(Path::new("readers/model")).unwrap().is_none());
        drop((held, lease));
        assert!(store.evict(0, Duration::ZERO, &alone).unwrap());
        assert!(!root.join("artifact").exists());
        assert!(
            !record_lock_path(&root.join(LOCKS), Path::new("readers/model")).exists()
                && !record_lock_path(&root.join(LOCKS), Path::new("artifact")).exists(),
            "the locks no one holds of what is gone go"
        );
        fs::remove_dir_all(root).unwrap();
    }

    /// Staged directories published at once over one target each land
    /// whole: the target is left one publication's files, never a mix, and
    /// no stage or replaced copy remains. One dropped unpublished leaves
    /// nothing. A quarantined path is set aside under a temporary name.
    #[test]
    fn staged_directories_land_whole() {
        let root = scratch("staged");
        let store = Store::open(root.clone(), Privacy::Shared).unwrap();
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
        assert_eq!(store.list(Path::new("")).unwrap(), ["export"]);
        assert!(names(&root).iter().all(|name| !unpublished(OsStr::new(name))), "no stage or replaced copy remains");
        store.quarantine(target).unwrap();
        assert_eq!(store.list(Path::new("")).unwrap(), Vec::<String>::new());
        assert!(names(&root).iter().any(|name| unpublished(OsStr::new(name))));
        fs::remove_dir_all(root).unwrap();
    }

    /// Concurrent publications and lockless reads/fetches over one target
    /// each land whole without torn bytes or OS contention errors.
    #[test]
    fn concurrent_publications_and_reads_land_whole() {
        let root = scratch("concurrent-publish");
        let store = Store::open(root.clone(), Privacy::Shared).unwrap();
        let path = Path::new("pointer");
        store.shared().unwrap().publish(path, Durability::Volatile, |file| file.write_all(&[0u8; 64])).unwrap();
        std::thread::scope(|scope| {
            for writer in 0..4u8 {
                let store = &store;
                seismic_threads::spawn_scoped(scope, format!("publishing-{writer}"), move || {
                    let payload = [writer; 64];
                    for _ in 0..40 {
                        store.shared().unwrap().publish(path, Durability::Volatile, |file| file.write_all(&payload)).unwrap();
                    }
                })
                .unwrap();
            }
            for reader in 0..2u8 {
                let store = &store;
                seismic_threads::spawn_scoped(scope, format!("reading-{reader}"), move || {
                    for i in 0..80 {
                        let bytes = match i % 2 == 0 {
                            true => store.read(path, 64).unwrap().unwrap(),
                            false => store.fetch(path).unwrap().unwrap(),
                        };
                        assert_eq!(bytes.len(), 64);
                        assert!(bytes.iter().all(|&byte| byte == bytes[0]), "reader saw torn bytes");
                    }
                })
                .unwrap();
            }
        });
        assert_eq!(store.list(Path::new("")).unwrap(), ["pointer"], "no temporary file remains");
        fs::remove_dir_all(root).unwrap();
    }

    /// Concurrent `put_if_absent` and lockless reads over a shared content
    /// address publish cleanly and read whole bytes.
    #[test]
    fn concurrent_put_if_absent_and_reads_land_whole() {
        let root = scratch("concurrent-put-if-absent");
        let store = Store::open(root.clone(), Privacy::Shared).unwrap();
        std::thread::scope(|scope| {
            for writer in 0..4u8 {
                let store = &store;
                seismic_threads::spawn_scoped(scope, format!("put-absent-{writer}"), move || {
                    let shared = store.shared().unwrap();
                    for key in 0..20u8 {
                        let name = PathBuf::from(format!("chunks/{key}"));
                        let payload = [key; 64];
                        shared.put_if_absent(&name, &payload, Durability::Volatile).unwrap();
                    }
                })
                .unwrap();
            }
            for reader in 0..2u8 {
                let store = &store;
                seismic_threads::spawn_scoped(scope, format!("read-absent-{reader}"), move || {
                    for _ in 0..4 {
                        for key in 0..20u8 {
                            let name = PathBuf::from(format!("chunks/{key}"));
                            if let Some(bytes) = store.read(&name, 64).unwrap() {
                                assert_eq!(bytes, vec![key; 64]);
                            }
                        }
                    }
                })
                .unwrap();
            }
        });
        for key in 0..20u8 {
            assert_eq!(store.read(&PathBuf::from(format!("chunks/{key}")), 64).unwrap().unwrap(), vec![key; 64]);
        }
        fs::remove_dir_all(root).unwrap();
    }

    /// Concurrent `Record::write`, `Record::remove`, and lockless `Store::read`
    /// never fail with pending-deletion errors on Windows NTFS or POSIX.
    #[test]
    fn concurrent_record_write_remove_and_lockless_read() {
        let root = scratch("record-write-remove-read");
        let store = Store::open(root.clone(), Privacy::Shared).unwrap();
        let record = Path::new("state/item");
        std::thread::scope(|scope| {
            for writer in 0..3u8 {
                let store = &store;
                seismic_threads::spawn_scoped(scope, format!("mutating-{writer}"), move || {
                    let payload = [writer + 1; 64];
                    for step in 0..40 {
                        let held = store.lock(record).unwrap();
                        if step % 3 == 0 {
                            held.remove().unwrap();
                        } else {
                            held.write(&payload, Durability::Volatile).unwrap();
                        }
                    }
                })
                .unwrap();
            }
            for reader in 0..2u8 {
                let store = &store;
                seismic_threads::spawn_scoped(scope, format!("lockless-read-{reader}"), move || {
                    for _ in 0..120 {
                        if let Some(bytes) = store.read(record, 64).unwrap() {
                            assert_eq!(bytes.len(), 64);
                            assert!(bytes.iter().all(|&byte| byte == bytes[0]), "reader saw torn bytes");
                        }
                    }
                })
                .unwrap();
            }
        });
        fs::remove_dir_all(root).unwrap();
    }

    /// Concurrent `quarantine` and `Staged::publish` on the same target
    /// serialize cleanly without losing directories or failing on contention.
    #[test]
    fn concurrent_quarantine_and_staged_publish_serialize() {
        let root = scratch("quarantine-staged");
        let store = Store::open(root.clone(), Privacy::Shared).unwrap();
        let target = Path::new("export");
        std::thread::scope(|scope| {
            for writer in 0..2u8 {
                let store = &store;
                seismic_threads::spawn_scoped(scope, format!("q-stage-{writer}"), move || {
                    for _ in 0..20 {
                        let staged = store.stage(target).unwrap();
                        fs::write(staged.path().join("v"), [writer]).unwrap();
                        staged.publish().unwrap();
                    }
                })
                .unwrap();
            }
            for worker in 0..2u8 {
                let store = &store;
                seismic_threads::spawn_scoped(scope, format!("quarantine-{worker}"), move || {
                    for _ in 0..20 {
                        match store.quarantine(target) {
                            Ok(()) => {}
                            Err(error) if error.io_kind() == Some(ErrorKind::NotFound) => {}
                            Err(error) => panic!("unexpected quarantine failure: {error}"),
                        }
                    }
                })
                .unwrap();
            }
        });
        fs::remove_dir_all(root).unwrap();
    }

    /// Concurrent `Record::lock`/`remove`, `Lease::write`/`leases`/`read`,
    /// `Store::hold`, `Shared`/`Exclusive`, and `Store::evict` serialize
    /// without deadlocks or delete-pending errors.
    #[test]
    fn concurrent_evict_records_leases_holds_and_exclusive() {
        let root = scratch("evict-race");
        let store = Store::open(root.clone(), Privacy::Shared).unwrap();
        let lease_dir = Path::new("leases");
        let record = Path::new("records/x");
        let held_path = Path::new("blobs/a");
        store.shared().unwrap().publish(held_path, Durability::Volatile, |file| file.write_all(b"pinned")).unwrap();
        std::thread::scope(|scope| {
            let s = &store;
            seismic_threads::spawn_scoped(scope, "evictor", move || {
                for _ in 0..30 {
                    let _ = s.evict(0, Duration::ZERO, &alone).unwrap();
                }
            })
            .unwrap();
            seismic_threads::spawn_scoped(scope, "record-churn", move || {
                for step in 0..30 {
                    let rec = s.lock(record).unwrap();
                    if step % 2 == 0 {
                        rec.write(b"value", Durability::Volatile).unwrap();
                    } else {
                        rec.remove().unwrap();
                    }
                }
            })
            .unwrap();
            seismic_threads::spawn_scoped(scope, "lease-churn", move || {
                for step in 0..20 {
                    let lease = s.lease(lease_dir).unwrap();
                    lease.write(s, format!("lease-{step}").as_bytes(), Durability::Volatile).unwrap();
                    let live = s.leases(lease_dir).unwrap();
                    for path in live {
                        let _ = s.read(&path, 64).unwrap();
                    }
                }
            })
            .unwrap();
            seismic_threads::spawn_scoped(scope, "holder-and-exclusive", move || {
                for _ in 0..20 {
                    let _ = s.shared().unwrap().put_if_absent(held_path, b"pinned", Durability::Volatile).unwrap();
                    if let Some(held) = s.hold(held_path).unwrap() {
                        assert_eq!(held.bytes().unwrap(), b"pinned");
                    }
                    let excl = s.exclusive().unwrap();
                    excl.append(Path::new("ledger"), b"entry\n").unwrap();
                    let _ = excl.examine(Path::new("ledger"), 4096).unwrap();
                }
            })
            .unwrap();
        });
        fs::remove_dir_all(root).unwrap();
    }

    /// Holding a record lock on a directory protects every file in its
    /// subtree during `Store::evict`, regardless of eviction class, while
    /// unlocked siblings are still evicted.
    #[test]
    fn directory_lock_protects_subtree_during_evict() {
        let root = scratch("dir-lock-evict");
        let store = Store::open(root.clone(), Privacy::Shared).unwrap();
        let nested_alone = Path::new("repo/snap/weights.bin");
        let nested_unreachable = Path::new("repo/snap/sub/orphan.bin");
        let sibling = Path::new("other/old.bin");
        {
            let shared = store.shared().unwrap();
            shared.publish(nested_alone, Durability::Volatile, |f| f.write_all(b"weights")).unwrap();
            shared.publish(nested_unreachable, Durability::Volatile, |f| f.write_all(b"orphan")).unwrap();
            shared.publish(sibling, Durability::Volatile, |f| f.write_all(b"evict-me")).unwrap();
        }
        aged(&root.join(nested_alone), 60);
        aged(&root.join(nested_unreachable), 60);
        aged(&root.join(sibling), 60);

        let dir_lock = store.lock(Path::new("repo")).unwrap();
        let layout = |path: &Path, _: &Examining<'_>| {
            if path.ends_with("orphan.bin") {
                Eviction::Unreachable
            } else {
                Eviction::Alone
            }
        };
        assert!(store.evict(0, Duration::ZERO, &layout).unwrap());
        assert_eq!(store.read(nested_alone, 64).unwrap().as_deref(), Some(&b"weights"[..]));
        assert_eq!(store.read(nested_unreachable, 64).unwrap().as_deref(), Some(&b"orphan"[..]));
        assert!(store.read(sibling, 64).unwrap().is_none(), "unlocked sibling should be evicted");

        drop(dir_lock);
        assert!(store.evict(0, Duration::ZERO, &layout).unwrap());
        assert!(store.read(nested_alone, 64).unwrap().is_none());
        assert!(store.read(nested_unreachable, 64).unwrap().is_none());
        fs::remove_dir_all(root).unwrap();
    }

    /// A thread blocked waiting on a contended Level-2 record lock releases
    /// Level-1 `.lock` before blocking, so `Store::exclusive` and `Store::evict`
    /// proceed without deadlock or starvation.
    #[test]
    fn contended_record_lock_does_not_hold_store_lock() {
        let root = scratch("contended-record-lock");
        let store = Store::open(root.clone(), Privacy::Shared).unwrap();
        let record = Path::new("records/contended");
        let held = store.lock(record).unwrap();
        let started = std::sync::Barrier::new(2);
        let acquired = std::sync::atomic::AtomicBool::new(false);

        std::thread::scope(|scope| {
            let store_ref = &store;
            let started_ref = &started;
            let acquired_ref = &acquired;
            seismic_threads::spawn_scoped(scope, "contended-waiter", move || {
                started_ref.wait();
                let _waiter = store_ref.lock(record).unwrap();
                acquired_ref.store(true, Ordering::Release);
            })
            .unwrap();

            started.wait();
            std::thread::sleep(Duration::from_millis(50));
            assert!(!acquired.load(Ordering::Acquire), "waiter must still be blocked on record lock");

            // Neither `evict` nor `exclusive` may block or deadlock while the
            // waiter is parked on the record lock.
            assert!(store.evict(0, Duration::ZERO, &alone).unwrap());
            let excl = store.exclusive().unwrap();
            drop(excl);

            drop(held);
        });
        assert!(acquired.load(Ordering::Acquire));
        fs::remove_dir_all(root).unwrap();
    }
}
