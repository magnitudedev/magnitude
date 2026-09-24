//! The engine's persistent cache of formed kernels and tuning results.
//!
//! The host names the directory (ACN passes `--cache-dir
//! <dataDir>/cache/kernels`; `forward_bench` takes the same flag); without
//! one, nothing is cached. Layout:
//!
//! ```text
//! <root>/cuda/<key>.cubin      CUDA images Seismic formed (its ArtifactStore)
//! <root>/tuning/<key>.json     one Seismic TuningResult per tuning key
//! ```
//!
//! Every key is a content address, so nothing is ever invalidated: a changed
//! input gives a new key. Writes go to a temporary file in the same
//! directory and are renamed into place, so concurrent processes never see a
//! partial entry. A file that cannot be read or parsed is a miss (it is
//! rewritten), never an error. Reading an entry refreshes its modification
//! time, and opening the cache evicts the least recently used files above
//! its capacity.

use seismic::{ArtifactKey, ArtifactKind, ArtifactStore, TuningResult};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// Bytes the cache keeps by default.
pub const DEFAULT_KERNEL_CACHE_BYTES: u64 = 1 << 30;

/// Temporary files older than this belong to a writer that died; eviction
/// removes them.
const ABANDONED_WRITE: Duration = Duration::from_secs(60 * 60);

const CUDA: &str = "cuda";
const VULKAN: &str = "vulkan";
const TUNING: &str = "tuning";
/// Every directory of the cache.
const DIRECTORIES: [&str; 3] = [CUDA, VULKAN, TUNING];

#[derive(Debug)]
pub enum KernelCacheError {
    /// The cache directory could not be created.
    Create { path: PathBuf, error: std::io::Error },
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

/// A kernel cache directory.
#[derive(Debug)]
pub struct KernelCache {
    root: PathBuf,
    /// Distinguishes this process's concurrent temporary files.
    writes: AtomicU64,
}

impl KernelCache {
    /// Open the cache at `root`, creating it, and evict the least recently
    /// used entries beyond `capacity` bytes.
    pub fn open(root: PathBuf, capacity: u64) -> Result<Self, KernelCacheError> {
        for directory in DIRECTORIES {
            let path = root.join(directory);
            fs::create_dir_all(&path).map_err(|error| KernelCacheError::Create { path, error })?;
        }
        let cache = Self {
            root,
            writes: AtomicU64::new(0),
        };
        cache.evict(capacity);
        Ok(cache)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn entry(&self, directory: &str, key: &str, extension: &str) -> PathBuf {
        self.root.join(directory).join(format!("{key}.{extension}"))
    }

    /// The bytes at `path`, refreshing its use time; `None` on any failure.
    fn read(&self, path: &Path) -> Option<Vec<u8>> {
        let bytes = fs::read(path).ok()?;
        if let Ok(file) = fs::File::options().write(true).open(path) {
            // Best effort: a stale use time only makes the entry an earlier
            // eviction candidate.
            let _ = file.set_modified(SystemTime::now());
        }
        Some(bytes)
    }

    /// Write `bytes` to `path` through a temporary file in the same
    /// directory. A failure is reported and leaves no partial entry.
    fn write(&self, path: &Path, bytes: &[u8]) {
        let directory = path.parent().expect("cache entries live in a directory");
        let temporary = directory.join(format!(
            ".{}.{}.{}.tmp",
            path.file_name().expect("cache entries have names").to_string_lossy(),
            std::process::id(),
            self.writes.fetch_add(1, Ordering::Relaxed)
        ));
        let written = fs::File::create(&temporary)
            .and_then(|mut file| file.write_all(bytes).and_then(|()| file.sync_all()))
            .and_then(|()| fs::rename(&temporary, path));
        if let Err(error) = written {
            let _ = fs::remove_file(&temporary);
            eprintln!(
                "magnitude-engine: kernel cache: cannot write {}: {error}",
                path.display()
            );
        }
    }

    /// The tuning result stored under `key`, if any parses.
    pub fn tuning(&self, key: &TuningCacheKey) -> Option<TuningResult> {
        let bytes = self.read(&self.entry(TUNING, key.as_str(), "json"))?;
        serde_json::from_slice(&bytes).ok()
    }

    pub fn store_tuning(&self, key: &TuningCacheKey, result: &TuningResult) {
        let bytes = serde_json::to_vec(result).expect("a tuning result serializes");
        self.write(&self.entry(TUNING, key.as_str(), "json"), &bytes);
    }

    /// Remove abandoned temporary files, then the least recently used
    /// entries until the rest fit `capacity`.
    fn evict(&self, capacity: u64) {
        let now = SystemTime::now();
        let mut entries = Vec::new();
        for directory in DIRECTORIES {
            let Ok(listing) = fs::read_dir(self.root.join(directory)) else {
                continue;
            };
            for item in listing.flatten() {
                let Ok(metadata) = item.metadata() else {
                    continue;
                };
                let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                let path = item.path();
                if item.file_name().to_string_lossy().starts_with('.') {
                    if now.duration_since(modified).unwrap_or_default() > ABANDONED_WRITE {
                        let _ = fs::remove_file(&path);
                    }
                    continue;
                }
                entries.push((modified, metadata.len(), path));
            }
        }
        entries.sort_by(|left, right| right.0.cmp(&left.0));
        let mut kept = 0u64;
        for (_, bytes, path) in entries {
            kept += bytes;
            if kept > capacity {
                let _ = fs::remove_file(&path);
            }
        }
    }
}

impl ArtifactStore for KernelCache {
    fn get(&self, kind: ArtifactKind, key: &ArtifactKey) -> Option<Vec<u8>> {
        match kind {
            ArtifactKind::CudaImage => self.read(&self.entry(CUDA, key.as_str(), "cubin")),
            ArtifactKind::SpirV => self.read(&self.entry(VULKAN, key.as_str(), "spv")),
        }
    }

    fn put(&self, kind: ArtifactKind, key: &ArtifactKey, bytes: &[u8]) {
        match kind {
            ArtifactKind::CudaImage => self.write(&self.entry(CUDA, key.as_str(), "cubin"), bytes),
            ArtifactKind::SpirV => self.write(&self.entry(VULKAN, key.as_str(), "spv"), bytes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "magnitude-kernel-cache-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn a_corrupt_entry_is_a_miss_and_is_rewritten() {
        let root = scratch("corrupt");
        let cache = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let key = TuningCacheKey::of("material");
        fs::write(root.join(TUNING).join(format!("{}.json", key.as_str())), b"{ not json").unwrap();
        assert!(cache.tuning(&key).is_none());
        let path = cache.entry(TUNING, key.as_str(), "json");
        cache.write(&path, b"rewritten");
        assert_eq!(fs::read(&path).unwrap(), b"rewritten");
        let leftovers = fs::read_dir(root.join(TUNING))
            .unwrap()
            .flatten()
            .filter(|item| item.file_name().to_string_lossy().starts_with('.'))
            .count();
        assert_eq!(leftovers, 0, "no temporary file remains");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn opening_evicts_the_least_recently_used_beyond_capacity() {
        let root = scratch("evict");
        let cache = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let old = cache.entry(CUDA, "old", "cubin");
        let recent = cache.entry(CUDA, "recent", "cubin");
        cache.write(&old, &[0; 600]);
        cache.write(&recent, &[0; 600]);
        let file = fs::File::options().write(true).open(&old).unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(3600)).unwrap();
        drop(KernelCache::open(root.clone(), 1000).unwrap());
        assert!(!old.exists());
        assert!(recent.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
