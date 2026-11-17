//! How a file becomes visible: [`publish`] is the one way every Seismic
//! store writes a file another reader, in this process or another, may
//! read meanwhile.
//!
//! The bytes are written to a temporary file of their own beside the
//! target, synced where the publication is [`Durability::Durable`], and
//! renamed over the target in one step: a reader finds the target as it
//! was or as it now is, never part written. The rename replaces what is
//! there (`rename(2)`; on Windows `std::fs::rename`, which replaces an
//! existing file, with POSIX semantics where the volume has them, so a
//! target a reader holds open is replaced as on Unix).
//!
//! A temporary file is created exclusively, under a name no other live
//! writer holds: the process's id and a counter of the whole process, so
//! no two writers share one, whichever store, thread or process they are.
//! One a writer that died left behind is [`unpublished`], for its
//! directory's owner to remove once it is old enough to be abandoned.
//!
//! A directory written over many steps is published the same way, from a
//! stage of its own beside its target ([`StagedDirectory`]).

use std::ffi::OsStr;
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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

/// The suffix of every temporary file a publication writes.
const UNPUBLISHED: &str = ".tmp";

/// Distinguishes the temporary files of every publication in this process.
static PUBLICATIONS: AtomicU64 = AtomicU64::new(0);

/// Make `path` hold what `write` writes to a file, as one atomic step, to
/// `durability`. On any failure `path` is as it was and no temporary file
/// is left, unless the failure is syncing the rename itself, of a durable
/// publication, after which the new file is visible but may not survive a
/// crash.
pub fn publish(path: &Path, durability: Durability, write: impl FnOnce(&mut File) -> Result<()>) -> Result<()> {
    let directory = path.parent().expect("a published file lives in a directory");
    let (temporary, mut file) = loop {
        let temporary = temporary(path);
        // Only a dead writer of the same process id leaves the name taken.
        match OpenOptions::new().write(true).create_new(true).open(&temporary) {
            Ok(file) => break (temporary, file),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    let written = write(&mut file)
        .and_then(|()| match durability {
            Durability::Durable => file.sync_all(),
            Durability::Volatile => Ok(()),
        })
        .and_then(|()| {
            drop(file);
            std::fs::rename(&temporary, path)
        });
    if let Err(error) = written {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    match durability {
        Durability::Durable => synced_directory(directory),
        Durability::Volatile => Ok(()),
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

/// A name beside `path` that no other live publication holds: the process's
/// id and a counter of the whole process.
fn temporary(path: &Path) -> PathBuf {
    let directory = path.parent().expect("a published path lives in a directory");
    let name = path.file_name().expect("a published path has a name").to_string_lossy();
    directory.join(format!(".{name}.{}.{}{UNPUBLISHED}", std::process::id(), PUBLICATIONS.fetch_add(1, Ordering::Relaxed)))
}

/// A directory being written beside its target, under a name of its own
/// ([`unpublished`]), which [`StagedDirectory::publish`] renames into place
/// whole: a reader finds the target as one publication left it, or briefly
/// absent while one replaces another, never part written. Its writer fills
/// it at [`StagedDirectory::path`], over as many steps as it needs. Dropped
/// unpublished, it is removed. Nothing of it is synced: a crash may lose it.
#[derive(Debug)]
pub struct StagedDirectory {
    target: PathBuf,
    stage: PathBuf,
    published: bool,
}

impl StagedDirectory {
    /// An empty stage for `target`, created exclusively beside it.
    pub fn new(target: &Path) -> Result<Self> {
        loop {
            let stage = temporary(target);
            // Only a dead writer of the same process id leaves the name taken.
            match std::fs::create_dir(&stage) {
                Ok(()) => return Ok(Self { target: target.to_owned(), stage, published: false }),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
    }

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
        loop {
            let held = match std::fs::rename(&self.stage, &self.target) {
                Ok(()) => break,
                // Something holds the target (another publication's or an
                // earlier one's; a rename over it fails, with an error kind
                // each platform names its own way), or held it as the rename
                // ran and was moved aside since: move aside what holds it,
                // and again.
                Err(error)
                    if std::fs::symlink_metadata(&self.target).is_ok()
                        || matches!(error.kind(), ErrorKind::DirectoryNotEmpty | ErrorKind::AlreadyExists) =>
                {
                    error
                }
                // The rename itself failed: what this publication moved
                // aside goes back.
                Err(error) => {
                    if let Some(last) = replaced.pop() {
                        let _ = std::fs::rename(last, &self.target);
                    }
                    return Err(error);
                }
            };
            let aside = temporary(&self.target);
            match std::fs::rename(&self.target, &aside) {
                Ok(()) => replaced.push(aside),
                // Another publication moved it aside first.
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(_) => return Err(held),
            }
        }
        self.published = true;
        for aside in replaced {
            match std::fs::symlink_metadata(&aside)?.is_dir() {
                true => std::fs::remove_dir_all(&aside)?,
                false => std::fs::remove_file(&aside)?,
            }
        }
        Ok(())
    }
}

impl Drop for StagedDirectory {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_dir_all(&self.stage);
        }
    }
}

/// Whether the file named `name` is a publication's temporary file, which
/// no reader reads and which, once old enough, a writer that died left.
pub fn unpublished(name: &OsStr) -> bool {
    let name = name.to_string_lossy();
    name.starts_with('.') && name.ends_with(UNPUBLISHED)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A publication replaces what was there, leaves no temporary file,
    /// and a failed one leaves the target as it was.
    #[test]
    fn a_publication_replaces_whole_or_leaves_the_target() {
        let directory = std::env::temp_dir().join(format!("seismic-publish-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("pointer");
        publish(&path, Durability::Durable, |file| file.write_all(b"first")).unwrap();
        publish(&path, Durability::Volatile, |file| file.write_all(b"second")).unwrap();
        let failed = publish(&path, Durability::Durable, |file| {
            file.write_all(b"torn")?;
            Err(std::io::Error::other("the writer fails"))
        });
        assert!(failed.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"second");
        let names = std::fs::read_dir(&directory).unwrap().map(|item| item.unwrap().file_name()).collect::<Vec<_>>();
        assert_eq!(names, [OsStr::new("pointer")], "no temporary file remains");
        std::fs::remove_dir_all(directory).unwrap();
    }

    /// A temporary file a dead writer of this process id left under the
    /// next name is passed over, not written into.
    #[test]
    fn a_taken_name_is_passed_over() {
        let directory = std::env::temp_dir().join(format!("seismic-publish-taken-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("entry");
        let next = PUBLICATIONS.load(Ordering::Relaxed);
        let taken = (next..next + 64)
            .map(|count| directory.join(format!(".entry.{}.{count}{UNPUBLISHED}", std::process::id())))
            .collect::<Vec<_>>();
        for left in &taken {
            std::fs::write(left, b"abandoned").unwrap();
            assert!(unpublished(left.file_name().unwrap()));
        }
        publish(&path, Durability::Volatile, |file| file.write_all(b"whole")).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"whole");
        for left in &taken {
            assert_eq!(std::fs::read(left).unwrap(), b"abandoned");
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    /// Staged directories published at once over one target each land
    /// whole: the target is left one publication's files, never a mix, and
    /// no stage or replaced copy remains. One dropped unpublished leaves
    /// nothing.
    #[test]
    fn staged_directories_land_whole() {
        let directory = std::env::temp_dir().join(format!("seismic-publish-staged-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let target = directory.join("export");
        drop(StagedDirectory::new(&target).unwrap());
        std::thread::scope(|scope| {
            for writer in 0..4u8 {
                let target = &target;
                seismic_threads::spawn_scoped(scope, format!("staging-{writer}"), move || {
                    for _ in 0..25 {
                        let staged = StagedDirectory::new(target).unwrap();
                        for file in ["a", "b", "c"] {
                            std::fs::write(staged.path().join(file), [writer]).unwrap();
                        }
                        staged.publish().unwrap();
                    }
                })
                .unwrap();
            }
        });
        let found = ["a", "b", "c"].map(|file| std::fs::read(target.join(file)).unwrap());
        assert!(found.iter().all(|bytes| bytes == &found[0]), "one publication's files, never a mix: {found:?}");
        let names = std::fs::read_dir(&directory).unwrap().map(|item| item.unwrap().file_name()).collect::<Vec<_>>();
        assert_eq!(names, [OsStr::new("export")], "no stage or replaced copy remains");
        std::fs::remove_dir_all(directory).unwrap();
    }
}
