//! Resolution of Hugging Face-layout snapshot entries to the repository blobs they publish.
//!
//! A snapshot entry is a symlink into `blobs/` on Unix and a hard link to the blob on Windows.
//! Canonicalizing a hard link yields the snapshot path itself, so a blob is identified by the
//! link target for symlinks and by file identity for every other regular file.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use same_file::Handle;

pub(crate) struct RepositoryBlobs {
    root: Option<PathBuf>,
    by_identity: OnceCell<HashMap<Handle, PathBuf>>,
}

impl RepositoryBlobs {
    /// Observes `<repository_root>/blobs`. A repository without blobs resolves nothing.
    pub(crate) fn open(repository_root: &Path) -> io::Result<Self> {
        let root = match repository_root.join("blobs").canonicalize() {
            Ok(root) => Some(root),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error),
        };
        Ok(Self {
            root,
            by_identity: OnceCell::new(),
        })
    }

    /// The canonical blob path a snapshot entry publishes, or `None` when the entry is not a
    /// link to one of this repository's blobs.
    pub(crate) fn resolve(&self, entry: &Path) -> io::Result<Option<PathBuf>> {
        let Some(root) = &self.root else {
            return Ok(None);
        };
        let kind = fs::symlink_metadata(entry)?.file_type();
        if kind.is_symlink() {
            let target = entry.canonicalize()?;
            return Ok((target.parent() == Some(root.as_path())).then_some(target));
        }
        if !kind.is_file() {
            return Ok(None);
        }
        let identity = Handle::from_path(entry)?;
        Ok(self.identities(root).get(&identity).cloned())
    }

    fn identities(&self, root: &Path) -> &HashMap<Handle, PathBuf> {
        self.by_identity.get_or_init(|| {
            let Ok(entries) = fs::read_dir(root) else {
                return HashMap::new();
            };
            entries
                .filter_map(Result::ok)
                .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
                .filter_map(|entry| {
                    let path = entry.path();
                    Handle::from_path(&path)
                        .ok()
                        .map(|identity| (identity, path))
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_symlinked_and_hard_linked_entries_to_their_blob() {
        let temporary = tempfile::tempdir().expect("temporary root");
        let repository = temporary.path().join("models--owner--model");
        let blobs = repository.join("blobs");
        let snapshot = repository.join("snapshots/commit");
        fs::create_dir_all(&blobs).expect("blobs");
        fs::create_dir_all(&snapshot).expect("snapshot");
        fs::write(blobs.join("first"), b"first").expect("first blob");
        fs::write(blobs.join("second"), b"second").expect("second blob");
        fs::hard_link(blobs.join("first"), snapshot.join("hard.gguf")).expect("hard link");
        fs::write(snapshot.join("copy.gguf"), b"first").expect("unlinked copy");
        #[cfg(unix)]
        std::os::unix::fs::symlink("../../blobs/second", snapshot.join("soft.gguf"))
            .expect("symlink");

        let resolver = RepositoryBlobs::open(&repository).expect("repository blobs");
        let canonical = blobs.canonicalize().expect("canonical blobs");
        assert_eq!(
            resolver
                .resolve(&snapshot.join("hard.gguf"))
                .expect("hard link"),
            Some(canonical.join("first"))
        );
        assert_eq!(
            resolver.resolve(&snapshot.join("copy.gguf")).expect("copy"),
            None
        );
        #[cfg(unix)]
        assert_eq!(
            resolver
                .resolve(&snapshot.join("soft.gguf"))
                .expect("symlink"),
            Some(canonical.join("second"))
        );
        assert!(resolver.resolve(&snapshot.join("missing.gguf")).is_err());
    }

    #[test]
    fn repository_without_blobs_resolves_nothing() {
        let temporary = tempfile::tempdir().expect("temporary root");
        fs::write(temporary.path().join("file"), b"x").expect("file");
        let resolver = RepositoryBlobs::open(temporary.path()).expect("repository");
        assert_eq!(
            resolver
                .resolve(&temporary.path().join("file"))
                .expect("file"),
            None
        );
    }
}
