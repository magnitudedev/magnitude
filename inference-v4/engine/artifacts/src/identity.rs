use serde::Serialize;
use std::fmt;

/// Content identity of one immutable artifact component.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactIdentity(pub [u8; 32]);

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

/// Identity of the package admitted by the engine.
///
/// Component boundaries are retained instead of hashing the concatenated bytes,
/// so diagnostics and cache keys can identify which component changed.
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
