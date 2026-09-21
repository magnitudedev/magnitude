//! Device-free interpretation of V3's uniform affine Q4 group-64 artifacts.
use super::{
    descriptor::{ArtifactIdentity, Stored, StoredTensor, Transform, WeightDescriptor},
    safetensors,
    source::FileSource,
    Error,
};
use seismic::DType;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, sync::Arc};

pub struct MlxArtifact {
    config: serde_json::Value,
    identity: ArtifactIdentity,
    tensors: BTreeMap<String, StoredTensor>,
}
fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
impl MlxArtifact {
    pub fn config(&self) -> &serde_json::Value {
        &self.config
    }
    pub fn identity(&self) -> ArtifactIdentity {
        self.identity
    }
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let config_source = FileSource::open(path.join("config.json"))?;
        let config_bytes = config_source.read(
            0,
            usize::try_from(config_source.size())
                .map_err(|_| invalid("configuration exceeds host address range"))?,
        )?;
        let config: serde_json::Value = serde_json::from_slice(&config_bytes)
            .map_err(|e| invalid(format!("invalid MLX configuration: {e}")))?;
        if config.get("quantization")
            != Some(&serde_json::json!({"group_size":64,"bits":4,"mode":"affine"}))
        {
            return Err(invalid(
                "MLX artifact requires uniform affine Q4 group-64 weights",
            ));
        }
        let mut paths = std::fs::read_dir(path)?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()?;
        paths.retain(|p| p.extension().is_some_and(|e| e == "safetensors"));
        paths.sort();
        let mut digest = Sha256::new();
        digest.update(&config_bytes);
        let mut tensors = BTreeMap::new();
        for path in paths {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| invalid("non-UTF8 artifact shard name"))?;
            let source = Arc::new(FileSource::open(&path)?);
            let directory = safetensors::read_directory(&mut source.reader())?;
            digest.update(name.as_bytes());
            digest.update([0]);
            digest.update(ArtifactIdentity(source.digest()?).to_string().as_bytes());
            for t in directory.tensors {
                let name = t.name;
                let stored = StoredTensor {
                    source: source.clone(),
                    offset: t.offset,
                    nbytes: t.nbytes,
                    dtype: t.dtype,
                    shape: t.shape,
                };
                if tensors.insert(name.clone(), stored).is_some() {
                    return Err(invalid(format!("duplicate Safetensors tensor {name:?}")));
                }
            }
        }
        if tensors.is_empty() {
            return Err(invalid("MLX artifact has no tensors"));
        }
        Ok(Self {
            config,
            identity: ArtifactIdentity(digest.finalize().into()),
            tensors,
        })
    }
    pub fn tensors(&self) -> &BTreeMap<String, StoredTensor> {
        &self.tensors
    }
    pub fn descriptor(&self, name: &str, shape: &[u64]) -> Result<WeightDescriptor, Error> {
        self.validate(name, shape)?;
        Ok(WeightDescriptor {
            name: name.into(),
            shape: shape.into(),
            transform: Transform::Identity,
        })
    }
    fn tensor(&self, name: &str) -> Result<&StoredTensor, Error> {
        self.tensors
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor {name:?}")))
    }
    fn validate(&self, name: &str, shape: &[u64]) -> Result<(), Error> {
        if shape.contains(&0) {
            return Err(invalid("weight role requires positive dimensions"));
        }
        let elements = shape
            .iter()
            .try_fold(1u64, |a, n| a.checked_mul(*n))
            .ok_or_else(|| invalid("weight role shape overflows"))?;
        let entry = self.tensor(name)?;
        if entry.dtype == DType::U32 {
            let [n, k] = shape else {
                return Err(invalid(format!("invalid affine geometry {name:?}")));
            };
            if k % 64 != 0 || entry.shape != [*n, k / 8] || !name.ends_with(".weight") {
                return Err(invalid(format!("affine code shape differs {name:?}")));
            }
            for suffix in ["scales", "biases"] {
                let coefficient =
                    self.tensor(&format!("{}{suffix}", name.strip_suffix("weight").unwrap()))?;
                if coefficient.dtype != DType::BF16 || coefficient.shape != [*n, k / 64] {
                    return Err(invalid(format!(
                        "affine coefficient shape/dtype differs {name:?}"
                    )));
                }
            }
        } else if elements != entry.nbytes / u64::from(entry.dtype.bytes()) {
            return Err(invalid(format!(
                "floating parameter shape differs {name:?}"
            )));
        }
        Ok(())
    }
    pub fn stored(&self, descriptor: &WeightDescriptor) -> Result<Stored, Error> {
        self.validate(&descriptor.name, &descriptor.shape)?;
        let entry = self.tensor(&descriptor.name)?;
        if entry.dtype != DType::U32 {
            return Ok(Stored::Dense(entry.clone()));
        }
        let prefix = descriptor.name.strip_suffix("weight").unwrap();
        Ok(Stored::AffinePlanes {
            bits: 4,
            group: 64,
            codes: entry.clone(),
            scales: self.tensor(&format!("{prefix}scales"))?.clone(),
            biases: self.tensor(&format!("{prefix}biases"))?.clone(),
        })
    }
}
