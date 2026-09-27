use serde::Serialize;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_COMPONENT: AtomicU64 = AtomicU64::new(1);

/// Process-local identity of one opened artifact component.
///
/// This value does not attest file contents and must not key persistent caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactIdentity(pub [u8; 32]);

impl ArtifactIdentity {
    pub(crate) fn for_open() -> Self {
        let mut value = [0; 32];
        value[..8].copy_from_slice(&NEXT_COMPONENT.fetch_add(1, Ordering::Relaxed).to_le_bytes());
        Self(value)
    }
}

impl fmt::Display for ArtifactIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Serialize for ArtifactIdentity {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

/// Process-local identity of the package admitted by the engine.
///
/// Component boundaries remain distinct so in-process consumers can identify
/// target and projector resources separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackageIdentity {
    pub target: ArtifactIdentity,
    pub projector: Option<ArtifactIdentity>,
}

impl fmt::Display for PackageIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.target)?;
        if let Some(projector) = self.projector {
            write!(f, ":{projector}")?;
        }
        Ok(())
    }
}

impl Serialize for PackageIdentity {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}
