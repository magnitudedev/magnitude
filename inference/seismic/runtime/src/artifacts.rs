//! Storage for compiled program artifacts, supplied by the embedder.
//!
//! Seismic compiles rendered programs with a backend toolchain at run time.
//! An embedder that keeps compiled artifacts between processes passes an
//! [`ArtifactStore`] when it opens a device; each program's toolchain reads
//! and writes the bytes it keeps under the program's content address.
//! Seismic performs no file I/O and knows no locations: where and how
//! artifacts are kept, and for how long, is the store's concern. Each
//! toolchain keeps its artifacts in its own namespace; the store knows no
//! backends.

use sha2::{Digest, Sha256};
use std::sync::Arc;

/// A content address: the SHA-256 (lowercase hex) of everything that
/// determines an artifact's bytes. A changed input gives a new key, so a
/// store never needs to invalidate an entry.
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
}

/// Content-addressed storage for compiled artifacts, supplied by the
/// embedder. `namespace` separates toolchains; a store treats it as an
/// opaque name.
pub trait ArtifactStore: Send + Sync {
    /// The stored bytes for `key`, or `None` on a miss or any failure.
    fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>>;
    /// Store `bytes` under `key`, replacing what was there. Failures are
    /// the store's to report; forming a program never fails for them.
    fn put(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]);
}

/// How a device is opened.
#[derive(Clone, Default)]
pub struct DeviceOptions {
    /// Where compiled artifacts are looked up and kept. Without a store,
    /// every program is compiled.
    pub artifacts: Option<Arc<dyn ArtifactStore>>,
}
