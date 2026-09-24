//! Storage for formed native artifacts, supplied by the embedder.
//!
//! Seismic forms some native implementations with a toolchain at run time
//! (NVRTC for CUDA). An embedder that keeps formed artifacts between
//! processes passes an [`ArtifactStore`] when it opens a device; Seismic
//! computes each artifact's content address and asks the store before
//! forming, and hands it every newly formed artifact. Seismic performs no
//! file I/O and knows no locations: where and how artifacts are kept, and
//! for how long, is the store's concern.

use sha2::{Digest, Sha256};
use std::sync::Arc;

/// The kinds of formed artifacts Seismic asks a store for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArtifactKind {
    /// A CUDA CUBIN image formed by NVRTC.
    CudaImage,
}

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

/// Content-addressed storage for formed native artifacts, supplied by the
/// embedder.
pub trait ArtifactStore: Send + Sync {
    /// The stored bytes for `key`, or `None` on a miss or any failure.
    fn get(&self, kind: ArtifactKind, key: &ArtifactKey) -> Option<Vec<u8>>;
    /// Store `bytes` under `key`. Failures are the store's to report;
    /// formation never fails for them.
    fn put(&self, kind: ArtifactKind, key: &ArtifactKey, bytes: &[u8]);
}

/// How a device is opened.
#[derive(Clone, Default)]
pub struct DeviceOptions {
    /// Where formed artifacts are looked up and kept. Without a store,
    /// formation always runs the toolchain.
    pub artifacts: Option<Arc<dyn ArtifactStore>>,
}
