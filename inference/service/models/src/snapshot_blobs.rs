//! Resolution of Hugging Face-layout snapshot entries to the repository blobs they publish.
//!
//! A snapshot entry is a symlink into `<repository>/blobs/` on Unix and a hard link to the blob on
//! Windows. The repository blob entry is named by the file's content hash, so it is what identifies
//! the content. Classic caches keep the bytes in that entry. Caches written by current
//! `huggingface_hub` make the entry itself a symlink into a store shared by every repository,
//! `<hub>/blobs/<xx>/<hash>`, whose name is not the file's Hub hash. Either way the bytes must stay
//! inside the hub root the repository lives in.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use same_file::Handle;

const MAX_LINK_HOPS: usize = 16;

/// Where a repository blob entry keeps its bytes: in the entry itself, or in a store the hub
/// shares between repositories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlobStorage {
    Repository,
    Shared,
}

/// The repository blob entry a snapshot entry publishes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RepositoryBlob {
    pub(crate) entry: PathBuf,
    pub(crate) storage: BlobStorage,
}

impl RepositoryBlob {
    /// The entry's name, which the Hub layout makes the file's content hash.
    pub(crate) fn name(&self) -> Option<&str> {
        self.entry.file_name().and_then(|value| value.to_str())
    }
}

pub(crate) struct RepositoryBlobs {
    roots: Option<Roots>,
    by_identity: OnceCell<HashMap<Handle, RepositoryBlob>>,
}

struct Roots {
    blobs: PathBuf,
    hub: PathBuf,
}

impl RepositoryBlobs {
    /// Observes `<repository_root>/blobs`, whose bytes may live anywhere under the hub root that
    /// contains the repository. A repository without blobs resolves nothing.
    pub(crate) fn open(repository_root: &Path) -> io::Result<Self> {
        let blobs = match repository_root.join("blobs").canonicalize() {
            Ok(blobs) => blobs,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(Self {
                    roots: None,
                    by_identity: OnceCell::new(),
                });
            }
            Err(error) => return Err(error),
        };
        let hub = repository_root
            .canonicalize()?
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| io::Error::other("Hugging Face repository has no hub root"))?;
        Ok(Self {
            roots: Some(Roots { blobs, hub }),
            by_identity: OnceCell::new(),
        })
    }

    /// The repository blob a snapshot entry publishes, or `None` when the entry is not a link to
    /// one of this repository's blobs whose bytes stay inside the hub.
    pub(crate) fn resolve(&self, entry: &Path) -> io::Result<Option<RepositoryBlob>> {
        let Some(roots) = &self.roots else {
            return Ok(None);
        };
        let kind = fs::symlink_metadata(entry)?.file_type();
        if kind.is_symlink() {
            return roots.follow(entry);
        }
        if !kind.is_file() {
            return Ok(None);
        }
        let identity = Handle::from_path(entry)?;
        Ok(self.identities(roots).get(&identity).cloned())
    }

    fn identities(&self, roots: &Roots) -> &HashMap<Handle, RepositoryBlob> {
        self.by_identity.get_or_init(|| {
            let Ok(entries) = fs::read_dir(&roots.blobs) else {
                return HashMap::new();
            };
            entries
                .filter_map(Result::ok)
                .filter_map(|entry| roots.blob_at(entry.path()).ok().flatten())
                .filter_map(|blob| {
                    Handle::from_path(&blob.entry)
                        .ok()
                        .map(|identity| (identity, blob))
                })
                .collect()
        })
    }
}

impl Roots {
    /// Follows a snapshot entry's link chain one hop at a time until it reaches an entry directly
    /// under the repository's blobs.
    fn follow(&self, entry: &Path) -> io::Result<Option<RepositoryBlob>> {
        let mut current = entry.to_path_buf();
        for _ in 0..MAX_LINK_HOPS {
            let (Some(parent), Some(name)) = (current.parent(), current.file_name()) else {
                return Ok(None);
            };
            let parent = parent.canonicalize()?;
            let located = parent.join(name);
            if parent == self.blobs {
                return self.blob_at(located);
            }
            if !fs::symlink_metadata(&located)?.file_type().is_symlink() {
                return Ok(None);
            }
            current = parent.join(fs::read_link(&located)?);
        }
        Ok(None)
    }

    /// The blob at an entry of the repository's blobs, when its bytes are a regular file inside
    /// the hub.
    fn blob_at(&self, entry: PathBuf) -> io::Result<Option<RepositoryBlob>> {
        let target = entry.canonicalize()?;
        if !target.starts_with(&self.hub) || !target.is_file() {
            return Ok(None);
        }
        let storage = if target == entry {
            BlobStorage::Repository
        } else {
            BlobStorage::Shared
        };
        Ok(Some(RepositoryBlob { entry, storage }))
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
        let owned = |name: &str| RepositoryBlob {
            entry: canonical.join(name),
            storage: BlobStorage::Repository,
        };
        assert_eq!(
            resolver
                .resolve(&snapshot.join("hard.gguf"))
                .expect("hard link"),
            Some(owned("first"))
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
            Some(owned("second"))
        );
        assert!(resolver.resolve(&snapshot.join("missing.gguf")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn resolves_entries_whose_blob_links_into_the_shared_hub_store() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().expect("temporary root");
        let hub = temporary.path().join("hub");
        let repository = hub.join("models--owner--model");
        let blobs = repository.join("blobs");
        let snapshot = repository.join("snapshots/commit");
        let shared = hub.join("blobs/de");
        fs::create_dir_all(&blobs).expect("blobs");
        fs::create_dir_all(&snapshot).expect("snapshot");
        fs::create_dir_all(&shared).expect("shared store");
        let hub_hash = "a".repeat(64);
        let store_hash = format!("de{}", "b".repeat(62));
        fs::write(shared.join(&store_hash), b"weights").expect("shared blob");
        symlink(
            format!("../../blobs/de/{store_hash}"),
            blobs.join(&hub_hash),
        )
        .expect("repository blob link");
        symlink(
            format!("../../blobs/{hub_hash}"),
            snapshot.join("model.gguf"),
        )
        .expect("snapshot link");

        let resolver = RepositoryBlobs::open(&repository).expect("repository blobs");
        let resolved = resolver
            .resolve(&snapshot.join("model.gguf"))
            .expect("shared entry")
            .expect("resolves");
        assert_eq!(
            resolved,
            RepositoryBlob {
                entry: blobs
                    .canonicalize()
                    .expect("canonical blobs")
                    .join(&hub_hash),
                storage: BlobStorage::Shared,
            }
        );
        assert_eq!(resolved.name(), Some(hub_hash.as_str()));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_blob_links_that_leave_the_hub() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().expect("temporary root");
        let hub = temporary.path().join("hub");
        let repository = hub.join("models--owner--model");
        let blobs = repository.join("blobs");
        let snapshot = repository.join("snapshots/commit");
        fs::create_dir_all(&blobs).expect("blobs");
        fs::create_dir_all(&snapshot).expect("snapshot");
        fs::write(temporary.path().join("outside"), b"weights").expect("outside file");
        symlink(temporary.path().join("outside"), blobs.join("hash")).expect("escaping blob");
        symlink("../../blobs/hash", snapshot.join("model.gguf")).expect("snapshot link");
        symlink(
            temporary.path().join("outside"),
            snapshot.join("direct.gguf"),
        )
        .expect("direct link");

        let resolver = RepositoryBlobs::open(&repository).expect("repository blobs");
        assert_eq!(
            resolver
                .resolve(&snapshot.join("model.gguf"))
                .expect("escaping entry"),
            None
        );
        assert_eq!(
            resolver
                .resolve(&snapshot.join("direct.gguf"))
                .expect("direct entry"),
            None
        );
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
