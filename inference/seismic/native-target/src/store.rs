//! What a program store keeps: bytes under content-addressed keys, by
//! namespace. The runtime's program store is its one owner
//! (`seismic_runtime::artifacts::ProgramStore`); backends see it only as an
//! [`ArtifactStore`], to keep what a formation determines (Metal's probe
//! answers) and as the [`Programs`] of a device.
//!
//! [`Programs`]: crate::Programs

use sha2::{Digest, Sha256};

/// A key of the store: the SHA-256 (lowercase hex) of what it names. An
/// artifact's key is a content address, of everything that determines its
/// bytes: a changed input gives a new key, so an artifact is never
/// invalidated. An entry pointer's key names the entry, and its value, the
/// artifact that holds the entry, may be replaced.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactKey(String);

impl ArtifactKey {
    /// The key of the artifact determined by `parts`, in order.
    pub fn of(parts: &[&[u8]]) -> Self {
        let mut digest = Sha256::new();
        for part in parts {
            digest.update((part.len() as u64).to_le_bytes());
            digest.update(part);
        }
        Self(digest.finalize().iter().map(|byte| format!("{byte:02x}")).collect())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The key `text` names, as [`ArtifactKey::as_str`] wrote it.
    pub fn parse(text: &str) -> Option<Self> {
        (text.len() == 64
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
            .then(|| Self(text.to_owned()))
    }
}

/// The file a store keeps an artifact's bytes in, held
/// ([`ArtifactStore::held`]): its path names those bytes for as long as this
/// lives, whatever the store evicts meanwhile, so a platform that opens the
/// artifact by path ([`Artifact::IN_PLACE`]) finds it there.
///
/// [`Artifact::IN_PLACE`]: crate::Artifact::IN_PLACE
pub struct HeldFile {
    path: std::path::PathBuf,
    /// What keeps the file at its path: the store's hold of it.
    _hold: Box<dyn Send + Sync>,
}

impl HeldFile {
    /// The file at `path`, kept there while `hold` lives.
    pub fn new(path: std::path::PathBuf, hold: Box<dyn Send + Sync>) -> Self {
        Self { path, _hold: hold }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl std::fmt::Debug for HeldFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("HeldFile").field(&self.path).finish()
    }
}

/// Storage for compiled artifacts, the entry pointers that name them, and
/// probe answers. `namespace` separates toolchains and the probes; a store
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
    /// The stored bytes for `key` and the file that holds them, held, where
    /// the store keeps them in one: what a toolchain whose platform opens
    /// its artifacts by path opens in place ([`Artifact::decode`]). The
    /// bytes are the held file's, read under the hold. None on a miss, any
    /// failure, or where the store keeps no file for it.
    ///
    /// [`Artifact::decode`]: crate::Artifact::decode
    fn held(&self, _namespace: &str, _key: &ArtifactKey) -> Option<(Vec<u8>, HeldFile)> {
        None
    }
    /// Store `bytes` under `key`, replacing what was there. Failures are
    /// the store's to report; forming a program never fails for them.
    fn put(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]);
}


/// The namespace of probe answers: what a device's probes found (Metal's
/// capability probes at discovery and open, the measurement job's probes),
/// kept by every store whatever its use time, since a lost answer runs its
/// probes again.
pub const PROBES: &str = "probes";

/// Artifacts held in memory for the life of the store: a store that keeps
/// nothing between processes. Shared by every device opened with it, it
/// forms each program once in the process.
#[derive(Default)]
pub struct MemoryStore(std::sync::Mutex<MemoryEntries>);

/// What a [`MemoryStore`] holds, by namespace and key.
type MemoryEntries = std::collections::HashMap<(String, ArtifactKey), Vec<u8>>;

impl MemoryStore {
    fn entries(&self) -> std::sync::MutexGuard<'_, MemoryEntries> {
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

