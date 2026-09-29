use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
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

/// Parses exactly the [`fmt::Display`] form.
impl FromStr for ArtifactIdentity {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, String> {
        let invalid = || format!("invalid artifact identity `{text}`");
        if text.len() != 64 || !text.is_ascii() {
            return Err(invalid());
        }
        let mut value = [0u8; 32];
        for (index, byte) in value.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).map_err(|_| invalid())?;
        }
        Ok(Self(value))
    }
}

/// Parses exactly the [`fmt::Display`] form.
impl FromStr for PackageIdentity {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, String> {
        let (target, projector) = match text.split_once(':') {
            Some((target, projector)) => (target, Some(projector.parse()?)),
            None => (text, None),
        };
        Ok(Self {
            target: target.parse()?,
            projector,
        })
    }
}

impl<'de> Deserialize<'de> for ArtifactIdentity {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for PackageIdentity {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}
