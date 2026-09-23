use crate::{
    gguf::{GgufArtifact, TensorDescriptor},
    ArtifactIdentity, Error, PackageIdentity, TemplatePayload, TokenizerPayload,
};
use std::path::{Path, PathBuf};

/// Device-free description of one immutable package component.
///
/// The canonical path is transport, not identity. A worker must reopen the
/// path and prove that the content identity still matches before using it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentManifest {
    pub path: PathBuf,
    pub identity: ArtifactIdentity,
    pub size: u64,
    pub tensors: Vec<TensorDescriptor>,
}

/// Exact package admitted by the host and safe to move to the numerical
/// worker. It contains no open file, mapping, device, or other live resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageManifest {
    pub identity: PackageIdentity,
    pub target: ComponentManifest,
    pub projector: Option<ComponentManifest>,
}

/// A target GGUF and its optional projector component.
///
/// The package establishes component ownership and identity only. A model-family
/// adapter decides whether the metadata describes a supported target/projector.
#[derive(Debug)]
pub struct Package {
    target: GgufArtifact,
    projector: Option<GgufArtifact>,
    identity: PackageIdentity,
    tokenizer: TokenizerPayload,
    templates: TemplatePayload,
}

impl Package {
    /// Opens a target and discovers an unambiguous `mmproj-*.gguf` sibling.
    pub fn open(target: impl AsRef<Path>) -> Result<Self, Error> {
        let target = target.as_ref();
        let projector = discover_projector(target)?;
        Self::open_paths(target, projector.as_deref())
    }

    /// Opens a target without attempting projector discovery.
    pub fn open_without_projector(target: impl AsRef<Path>) -> Result<Self, Error> {
        Self::open_paths(target.as_ref(), None)
    }

    /// Opens a target with an explicitly configured projector component.
    pub fn open_with_projector(
        target: impl AsRef<Path>,
        projector: impl AsRef<Path>,
    ) -> Result<Self, Error> {
        Self::open_paths(target.as_ref(), Some(projector.as_ref()))
    }

    fn open_paths(target_path: &Path, projector_path: Option<&Path>) -> Result<Self, Error> {
        if let Some(projector_path) = projector_path {
            if target_path.canonicalize()? == projector_path.canonicalize()? {
                return Err(Error::Invalid(
                    "target and projector must be distinct package components".into(),
                ));
            }
        }
        let target = GgufArtifact::open(target_path)?;
        let projector = projector_path.map(GgufArtifact::open).transpose()?;
        if projector
            .as_ref()
            .is_some_and(|projector| projector.identity() == target.identity())
        {
            return Err(Error::Invalid(
                "target and projector components have identical content identity".into(),
            ));
        }
        let identity = PackageIdentity {
            target: target.identity(),
            projector: projector.as_ref().map(GgufArtifact::identity),
        };
        let tokenizer = TokenizerPayload::from_directory(target.directory());
        let templates = TemplatePayload::from_directory(
            target.directory(),
            &target.source().path().display().to_string(),
        )?;
        Ok(Self {
            target,
            projector,
            identity,
            tokenizer,
            templates,
        })
    }

    pub fn target(&self) -> &GgufArtifact {
        &self.target
    }

    pub fn projector(&self) -> Option<&GgufArtifact> {
        self.projector.as_ref()
    }

    pub fn identity(&self) -> PackageIdentity {
        self.identity
    }

    pub fn tokenizer(&self) -> &TokenizerPayload {
        &self.tokenizer
    }

    pub fn templates(&self) -> &TemplatePayload {
        &self.templates
    }

    pub fn manifest(&self) -> PackageManifest {
        PackageManifest {
            identity: self.identity,
            target: component_manifest(&self.target),
            projector: self.projector.as_ref().map(component_manifest),
        }
    }

    /// Reopen the exact components selected by the host and reject a path
    /// replacement or inventory change before numerical construction begins.
    pub fn reopen(manifest: &PackageManifest) -> Result<Self, Error> {
        let package = Self::open_paths(
            &manifest.target.path,
            manifest
                .projector
                .as_ref()
                .map(|component| component.path.as_path()),
        )?;
        let reopened = package.manifest();
        if reopened != *manifest {
            return Err(Error::Invalid(
                "package components changed after host admission".into(),
            ));
        }
        Ok(package)
    }
}

fn component_manifest(artifact: &GgufArtifact) -> ComponentManifest {
    ComponentManifest {
        path: artifact.source().path().to_path_buf(),
        identity: artifact.identity(),
        size: artifact.source().size(),
        tensors: artifact.directory().tensors.clone(),
    }
}

fn discover_projector(target: &Path) -> Result<Option<PathBuf>, Error> {
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let canonical_target = target.canonicalize()?;
    let mut projectors = Vec::new();
    for entry in std::fs::read_dir(parent)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if path.is_file()
            && name.starts_with("mmproj-")
            && name.ends_with(".gguf")
            && path
                .canonicalize()
                .is_ok_and(|path| path != canonical_target)
        {
            projectors.push(path);
        }
    }
    projectors.sort();
    match projectors.len() {
        0 => Ok(None),
        1 => Ok(projectors.pop()),
        _ => Err(Error::AmbiguousProjectors(projectors)),
    }
}
