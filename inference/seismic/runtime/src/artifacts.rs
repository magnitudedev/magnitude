//! Storage for compiled program artifacts, supplied by the embedder.
//!
//! Seismic compiles rendered programs with a backend toolchain at run time.
//! An embedder that keeps compiled artifacts between processes passes an
//! [`ArtifactStore`] when it opens a device; the device's program former
//! alone reads and writes it: each compile's artifact under its content
//! address, and for each entry a pointer naming the artifact that holds it.
//! Seismic performs no file I/O and knows no locations: where and how
//! artifacts are kept, and for how long, is the store's concern. Each
//! toolchain keeps its artifacts in its own namespace; the store knows no
//! backends.

use sha2::{Digest, Sha256};
use std::sync::Arc;

/// A key of the store: the SHA-256 (lowercase hex) of what it names. An
/// artifact's key is a content address, of everything that determines its
/// bytes: a changed input gives a new key, so an artifact is never
/// invalidated. An entry pointer's key names the entry, and its value, the
/// artifact that holds the entry, may be replaced.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactKey(String);

impl ArtifactKey {
    /// The key of the artifact determined by `parts`, in order.
    pub(crate) fn of(parts: &[&[u8]]) -> Self {
        let mut digest = Sha256::new();
        for part in parts {
            digest.update((part.len() as u64).to_le_bytes());
            digest.update(part);
        }
        Self(crate::telemetry::hex(&digest.finalize()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The key `text` names, as [`ArtifactKey::as_str`] wrote it.
    pub(crate) fn parse(text: &str) -> Option<Self> {
        (text.len() == 64
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
            .then(|| Self(text.to_owned()))
    }
}

/// Storage for compiled artifacts and the entry pointers that name them,
/// supplied by the embedder. `namespace` separates toolchains; a store
/// treats it as an opaque name.
pub trait ArtifactStore: Send + Sync {
    /// The stored bytes for `key`, or `None` on a miss or any failure.
    fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>>;
    /// Whether the store keeps bytes for `key`, without reading them: what
    /// a formation's class is read from. A store that cannot tell more
    /// cheaply reads them.
    fn contains(&self, namespace: &str, key: &ArtifactKey) -> bool {
        self.get(namespace, key).is_some()
    }
    /// Store `bytes` under `key`, replacing what was there. Failures are
    /// the store's to report; forming a program never fails for them.
    fn put(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]);
}

/// Artifacts held in memory for the life of the store: a store that keeps
/// nothing between processes. Shared by every device opened with it, it
/// forms each program once in the process.
#[derive(Default)]
pub struct MemoryStore(std::sync::Mutex<MemoryEntries>);

/// What a [`MemoryStore`] holds, by namespace and key.
pub type MemoryEntries = std::collections::HashMap<(String, ArtifactKey), Vec<u8>>;

impl MemoryStore {
    /// What the store holds, to read or change in place (a test's damage).
    pub fn entries(&self) -> std::sync::MutexGuard<'_, MemoryEntries> {
        self.0.lock().expect("the memory store")
    }
}

impl ArtifactStore for MemoryStore {
    fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>> {
        self.entries().get(&(namespace.to_owned(), key.clone())).cloned()
    }

    fn put(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]) {
        self.entries().insert((namespace.to_owned(), key.clone()), bytes.to_vec());
    }
}

/// How a device is opened.
#[derive(Clone, Default)]
pub struct DeviceOptions {
    /// Where compiled artifacts are looked up and kept. Without a store,
    /// every program is compiled.
    pub artifacts: Option<Arc<dyn ArtifactStore>>,
    /// What its recorded graphs hold, as records hold it. Without one its
    /// backend measures it on opening, once a process.
    pub graph_cost: Option<seismic_compiler::graph_cost::GraphCost>,
}
