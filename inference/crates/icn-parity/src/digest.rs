use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Context;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

pub fn sha256_json<T: Serialize>(value: &T) -> anyhow::Result<String> {
    let bytes = serde_json::to_vec(value).context("failed to encode value for digest")?;
    Ok(sha256_bytes(&bytes))
}

pub async fn sha256_file(path: &Path) -> anyhow::Result<(String, u64)> {
    let mut file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("failed to open {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .with_context(|| format!("failed while hashing {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        bytes = bytes.saturating_add(read as u64);
    }
    Ok((format!("{:x}", hasher.finalize()), bytes))
}

pub fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[derive(Clone, Debug)]
pub struct SourceTreeDigest {
    pub sha256: String,
    pub file_count: u64,
    pub total_bytes: u64,
}

/// Reproduce `sha256-path-size-content-v1` without consulting version control.
pub async fn sha256_source_tree(
    path: &Path,
    excluded_directory_names: &[&str],
) -> anyhow::Result<SourceTreeDigest> {
    let mut files = Vec::new();
    let excluded = excluded_directory_names
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    collect_source_files(path, path, &excluded, &mut files)?;
    // The source-inventory contract orders normalized relative path strings,
    // not platform Path components. UTF-8 lexical order is explicit so the
    // Rust verifier reproduces the TypeScript builder byte-for-byte.
    let mut normalized_files = files
        .into_iter()
        .map(|relative| {
            let normalized = relative
                .to_str()
                .context("source inventory paths must be valid UTF-8")?
                .replace('\\', "/");
            Ok((normalized, relative))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    normalized_files.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    let mut aggregate = Sha256::new();
    let mut total_bytes = 0_u64;
    for (normalized, relative) in &normalized_files {
        let (digest, bytes) = sha256_file(&path.join(relative)).await?;
        aggregate.update(format!("{normalized}\0{bytes}\0{digest}\n"));
        total_bytes = total_bytes.saturating_add(bytes);
    }
    Ok(SourceTreeDigest {
        sha256: format!("{:x}", aggregate.finalize()),
        file_count: normalized_files.len() as u64,
        total_bytes,
    })
}

fn collect_source_files(
    root: &Path,
    directory: &Path,
    excluded: &BTreeSet<&str>,
    output: &mut Vec<std::path::PathBuf>,
) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(directory)
        .with_context(|| format!("failed to inventory {}", directory.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            let name = entry.file_name();
            if !name.to_str().is_some_and(|name| excluded.contains(name)) {
                collect_source_files(root, &path, excluded, output)?;
            }
        } else if file_type.is_file() {
            output.push(
                path.strip_prefix(root)
                    .context("inventory path escaped root")?
                    .to_path_buf(),
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_bytes() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[tokio::test]
    async fn source_tree_uses_portable_string_order_not_path_component_order() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("a")).unwrap();
        std::fs::write(root.path().join("a-b"), b"dash").unwrap();
        std::fs::write(root.path().join("a/b"), b"slash").unwrap();

        let dash = sha256_bytes(b"dash");
        let slash = sha256_bytes(b"slash");
        let expected =
            sha256_bytes(format!("a-b\0{}\0{dash}\na/b\0{}\0{slash}\n", 4, 5).as_bytes());
        let actual = sha256_source_tree(root.path(), &[]).await.unwrap();

        assert_eq!(actual.sha256, expected);
        assert_eq!(actual.file_count, 2);
        assert_eq!(actual.total_bytes, 9);
    }
}
