use std::fs::{self, File, OpenOptions};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use magnitude_service_contracts::InventoryError;

static QUARANTINE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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

    for relative in ["hub", "locks"] {
        ensure_owned_directory(&root.join(relative)).await?;
    }
    Ok(())
}

pub(crate) async fn ensure_owned_directory(path: &Path) -> Result<(), InventoryError> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => return Ok(()),
        Ok(_) => quarantine_owned_path(path).await?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error(error)),
    }
    tokio::fs::create_dir_all(path).await.map_err(io_error)?;
    restrict_directory(path).await
}

pub(crate) fn ensure_owned_directory_sync(path: &Path) -> Result<(), InventoryError> {
    match path.symlink_metadata() {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => return Ok(()),
        Ok(_) => quarantine_owned_path_sync(path)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error(error)),
    }
    fs::create_dir_all(path).map_err(io_error)?;
    restrict_directory_sync(path)
}

pub(crate) fn acquire_exclusive_lock(path: &Path) -> Result<File, InventoryError> {
    if let Some(parent) = path.parent() {
        ensure_owned_directory_sync(parent)?;
    }
    match path.symlink_metadata() {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {}
        Ok(_) => quarantine_owned_path_sync(path)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error(error)),
    }

    let mut options = OpenOptions::new();
    options.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
        options.mode(0o600);
    }
    let file = options.open(path).map_err(io_error)?;
    FileExt::lock_exclusive(&file).map_err(io_error)?;
    Ok(file)
}

async fn quarantine_owned_path(path: &Path) -> Result<(), InventoryError> {
    let destination = quarantine_destination(path);
    tokio::fs::rename(path, destination).await.map_err(io_error)
}

pub(crate) fn quarantine_owned_path_sync(path: &Path) -> Result<(), InventoryError> {
    fs::rename(path, quarantine_destination(path)).map_err(io_error)
}

fn quarantine_destination(path: &Path) -> std::path::PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = QUARANTINE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    path.with_extension(format!("invalid-{timestamp}-{sequence}"))
}

async fn restrict_directory(path: &Path) -> Result<(), InventoryError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        tokio::fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .await
            .map_err(io_error)?;
    }
    Ok(())
}

fn restrict_directory_sync(path: &Path) -> Result<(), InventoryError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(io_error)?;
    }
    Ok(())
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
        fs::write(root.join("locks"), b"invalid").expect("invalid locks file");

        ensure_store_layout(&root).await.expect("reconciled layout");

        assert!(root.join("hub").is_dir());
        assert!(root.join("locks").is_dir());
        assert_eq!(
            fs::read(outside).expect("outside file retained"),
            b"outside"
        );
    }

    #[cfg(unix)]
    #[test]
    fn lock_acquisition_replaces_a_symlink_without_following_it() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let lock_path = temporary.path().join("locks/model.lock");
        let outside = temporary.path().join("outside");
        fs::create_dir_all(lock_path.parent().unwrap()).expect("locks");
        fs::write(&outside, b"outside").expect("outside file");
        std::os::unix::fs::symlink(&outside, &lock_path).expect("invalid lock link");

        let lock = acquire_exclusive_lock(&lock_path).expect("exclusive lock");
        drop(lock);

        assert!(lock_path.symlink_metadata().unwrap().file_type().is_file());
        assert_eq!(
            fs::read(outside).expect("outside file retained"),
            b"outside"
        );
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
