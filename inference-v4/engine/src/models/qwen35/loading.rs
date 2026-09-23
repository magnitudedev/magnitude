//! Artifact interpretation is device-free. Loading imports the interpreted weight
//! roles through ordinary automatic selection on the caller's execution owner.
use super::{decoder::Decoder, gguf, mlx, Description};
use crate::{
    weights::{
        gguf::GgufArtifact,
        mlx::MlxArtifact,
        residency::{Importer, ResidentWeight},
        Error as WeightError,
    },
    Error,
};
use seismic::{Device, PreparationOptions};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    rc::Rc,
};

enum Artifact {
    Mlx {
        artifact: MlxArtifact,
        directory: PathBuf,
    },
    Gguf {
        artifact: GgufArtifact,
        path: PathBuf,
    },
}
pub struct Model {
    artifact: Artifact,
    description: Description,
}
impl Model {
    /// Local directories are MLX/Safetensors artifacts; local files are GGUF.
    /// Interpretation retains open tensor sources and performs no acquisition.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (artifact, description) = if metadata.is_dir() {
            let artifact = MlxArtifact::open(path).map_err(|e| e.to_string())?;
            let description = mlx::describe(&artifact).map_err(|e| e.to_string())?;
            (
                Artifact::Mlx {
                    artifact,
                    directory: path.to_owned(),
                },
                description,
            )
        } else if metadata.is_file() {
            let artifact = GgufArtifact::open(path).map_err(|e| e.to_string())?;
            let description = gguf::inspect(artifact.directory(), artifact.identity())
                .map_err(|e| e.to_string())?;
            (
                Artifact::Gguf {
                    artifact,
                    path: path.to_owned(),
                },
                description,
            )
        } else {
            return Err("model artifact must be a regular file or directory".into());
        };
        Ok(Self {
            artifact,
            description,
        })
    }
    pub fn tokenizer_config(&self) -> Result<crate::inputs::BpeConfig, String> {
        let identity = self.description.artifact_identity.to_string();
        match &self.artifact {
            Artifact::Mlx { directory, .. } => {
                crate::inputs::artifacts::mlx_tokenizer(directory, identity)
            }
            Artifact::Gguf { artifact, .. } => {
                crate::inputs::artifacts::qwen35_gguf(artifact.directory(), identity)
            }
        }
    }
    pub fn templates(&self) -> Result<crate::chat::TemplateBundle, String> {
        match &self.artifact {
            Artifact::Mlx { directory, .. } => {
                crate::inputs::artifacts::directory_templates(directory)
            }
            Artifact::Gguf { artifact, path } => crate::inputs::artifacts::gguf_templates(
                artifact.directory(),
                &path.display().to_string(),
            ),
        }
    }
    /// Image-tower interpretation is opt-in; text-only callers retain no image
    /// processor or numerical encoder resources.
    pub fn vision_description(&self) -> Result<super::vision::Description, String> {
        match &self.artifact {
            Artifact::Mlx { artifact, .. } => super::vision::describe(artifact),
            Artifact::Gguf { .. } => Err("GGUF Qwen vision artifacts are not supported".into()),
        }
    }
    pub fn description(&self) -> &Description {
        &self.description
    }
    /// Load the optional image tower on the same budgeted execution owner as text.
    pub fn load_vision(
        &self,
        device: Rc<Device>,
        preparation: PreparationOptions,
    ) -> Result<super::vision_runtime::Encoder, Error> {
        let description = self.vision_description().map_err(Error::from)?;
        let Artifact::Mlx { artifact, .. } = &self.artifact else {
            return Err(Error::Request(
                "GGUF Qwen vision artifacts are not supported".into(),
            ));
        };
        let mut importer = Importer::new(device.clone(), preparation.clone());
        super::vision_runtime::Encoder::load(
            device,
            &description,
            preparation,
            |descriptor, dtype| {
                let stored = artifact.stored(descriptor).map_err(weight_error)?;
                importer
                    .import(descriptor, &stored, dtype)
                    .map_err(weight_error)
            },
        )
    }
    pub fn load(
        self,
        device: Rc<Device>,
        preparation: PreparationOptions,
        context: usize,
        sequences: usize,
    ) -> Result<Decoder, Error> {
        let mut importer = Importer::new(device.clone(), preparation.clone());
        let mut resident: HashMap<(String, String, String), ResidentWeight> = HashMap::new();
        let workload = super::decoder::DecoderCapacity {
            context_capacity: context,
            max_sequences: sequences,
        };
        Decoder::compile(
            device,
            &self.description,
            |descriptor, dtype| {
                let key = (
                    descriptor.name.clone(),
                    format!("{:?}", descriptor.transform),
                    dtype.name().to_string(),
                );
                if let Some(weight) = resident.get(&key) {
                    return Ok(weight.clone());
                }
                let stored = match &self.artifact {
                    Artifact::Mlx { artifact, .. } => artifact.stored(descriptor),
                    Artifact::Gguf { artifact, .. } => artifact.stored(descriptor),
                }
                .map_err(weight_error)?;
                let weight = importer
                    .import(descriptor, &stored, dtype)
                    .map_err(weight_error)?;
                resident.insert(key, weight.clone());
                Ok(weight)
            },
            preparation,
            workload,
        )
    }
}

fn weight_error(error: WeightError) -> Error {
    match error {
        WeightError::Seismic(error) => error,
        WeightError::Io(error) => Error::Request(format!("weight source: {error}")),
        WeightError::Invalid(message) => Error::Request(message),
    }
}
