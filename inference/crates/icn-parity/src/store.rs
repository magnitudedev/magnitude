use std::path::{Component, Path, PathBuf};

use anyhow::{Context, bail};
use serde::Serialize;
use tokio::io::AsyncWriteExt;

use crate::digest::sha256_bytes;
use crate::provenance::validate_run_id;

#[derive(Debug)]
pub struct RunStore {
    root: PathBuf,
}

#[derive(Clone, Debug)]
pub struct StoredFile {
    pub absolute_path: PathBuf,
    pub relative_path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

impl RunStore {
    pub async fn create(output_root: &Path, run_id: &str) -> anyhow::Result<Self> {
        validate_run_id(run_id)?;
        tokio::fs::create_dir_all(output_root)
            .await
            .with_context(|| format!("failed to create output root {}", output_root.display()))?;
        let root = output_root.join(run_id);
        match tokio::fs::create_dir(&root).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                bail!(
                    "run directory already exists and will not be modified: {}",
                    root.display()
                )
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to create run directory {}", root.display()));
            }
        }
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub async fn write_json<T: Serialize>(
        &self,
        relative: &Path,
        value: &T,
    ) -> anyhow::Result<StoredFile> {
        let mut bytes =
            serde_json::to_vec_pretty(value).context("failed to encode run artifact")?;
        bytes.push(b'\n');
        self.write_bytes(relative, &bytes).await
    }

    pub async fn write_bytes(&self, relative: &Path, bytes: &[u8]) -> anyhow::Result<StoredFile> {
        validate_relative_path(relative)?;
        let path = self.root.join(relative);
        let parent = path.parent().context("run artifact has no parent")?;
        tokio::fs::create_dir_all(parent).await.with_context(|| {
            format!(
                "failed to create run artifact directory {}",
                parent.display()
            )
        })?;
        let temporary = tempfile::Builder::new()
            .prefix(".icn-parity-artifact-")
            .tempfile_in(parent)
            .with_context(|| {
                format!(
                    "failed to create temporary artifact in {}",
                    parent.display()
                )
            })?;
        let std_file = temporary
            .as_file()
            .try_clone()
            .context("failed to clone temporary artifact handle")?;
        let mut file = tokio::fs::File::from_std(std_file);
        file.write_all(bytes)
            .await
            .with_context(|| format!("failed to write {}", path.display()))?;
        file.flush()
            .await
            .with_context(|| format!("failed to flush {}", path.display()))?;
        file.sync_all()
            .await
            .with_context(|| format!("failed to sync {}", path.display()))?;
        drop(file);
        temporary.persist_noclobber(&path).map_err(|error| {
            anyhow::Error::new(error.error).context(format!(
                "refusing to overwrite run artifact {}",
                path.display()
            ))
        })?;
        sync_parent_directory(parent)?;
        Ok(StoredFile {
            absolute_path: path,
            relative_path: relative.to_path_buf(),
            sha256: sha256_bytes(bytes),
            bytes: bytes.len() as u64,
        })
    }
}

pub fn case_file_stem(case_id: &str) -> String {
    case_id
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-') {
                byte as char
            } else {
                '_'
            }
        })
        .collect()
}

fn validate_relative_path(path: &Path) -> anyhow::Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!(
            "run artifact path must be a non-empty safe relative path: {}",
            path.display()
        );
    }
    Ok(())
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> anyhow::Result<()> {
    std::fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("failed to sync run artifact directory {}", parent.display()))
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn store_never_overwrites() {
        let temp = tempfile::tempdir().unwrap();
        let store = RunStore::create(temp.path(), "run-1").await.unwrap();
        store
            .write_bytes(Path::new("a/value.txt"), b"first")
            .await
            .unwrap();
        assert!(
            store
                .write_bytes(Path::new("a/value.txt"), b"second")
                .await
                .is_err()
        );
        assert_eq!(
            tokio::fs::read(store.root().join("a/value.txt"))
                .await
                .unwrap(),
            b"first"
        );
    }

    #[tokio::test]
    async fn duplicate_run_directory_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        RunStore::create(temp.path(), "same").await.unwrap();
        assert!(RunStore::create(temp.path(), "same").await.is_err());
    }
}
