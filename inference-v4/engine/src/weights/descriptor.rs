//! Logical roles and stored values, independent of container interpretation.
use super::{source::FileSource, Error};
use seismic::DType;
use std::sync::Arc;
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Transform {
    Identity,
    NegativeExp,
}
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct WeightDescriptor {
    pub name: String,
    pub shape: Vec<u64>,
    pub transform: Transform,
}
#[derive(Clone, Debug)]
pub struct StoredTensor {
    pub source: Arc<FileSource>,
    pub offset: u64,
    pub nbytes: u64,
    pub dtype: DType,
    pub shape: Vec<u64>,
}
impl StoredTensor {
    pub fn read(&self) -> Result<Vec<u8>, Error> {
        self.source.read(
            self.offset,
            usize::try_from(self.nbytes)
                .map_err(|_| Error::Invalid("tensor exceeds host address range".into()))?,
        )
    }
}
#[derive(Clone, Debug)]
pub enum Stored {
    Dense(StoredTensor),
    /// Source codec identity. Container metadata stays outside numerical import.
    GgmlBlocks {
        source: Arc<FileSource>,
        offset: u64,
        nbytes: u64,
        shape: Vec<u64>,
        encoding: super::gguf::Encoding,
    },
    AffinePlanes {
        bits: u32,
        group: u64,
        codes: StoredTensor,
        scales: StoredTensor,
        biases: StoredTensor,
    },
}
/// Content identity. Each format defines its canonical V3 hash composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactIdentity(pub [u8; 32]);
impl std::fmt::Display for ArtifactIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl serde::Serialize for ArtifactIdentity {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}
