use crate::{
    gguf::{self, Directory, GgufArtifact, TensorDescriptor},
    ArtifactIdentity, Error, PackageIdentity, TemplatePayload, TokenizerPayload,
};
use std::path::{Path, PathBuf};

/// Device-free description of one opened package component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentManifest {
    pub path: PathBuf,
    pub identity: ArtifactIdentity,
    pub size: u64,
    pub tensors: Vec<TensorDescriptor>,
}

/// Description of the package admitted by the host. The opened package itself
/// is shared with the numerical worker so all reads use the admitted files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageManifest {
    pub identity: PackageIdentity,
    pub target: ComponentManifest,
    pub projector: Option<ComponentManifest>,
}

/// Validated package metadata when tensor payloads have not been downloaded.
/// It cannot be used as an import source: the open `Package` remains the
/// authority for payload-backed execution.
#[derive(Debug)]
pub struct PackageHeaders {
    target: Directory,
    projector: Option<Directory>,
    identity: PackageIdentity,
}

impl PackageHeaders {
    /// Inspect a target header and an optional explicit projector header.
    /// Neither tensor payload is read or required to exist.
    pub fn open(target: impl AsRef<Path>, projector: Option<&Path>) -> Result<Self, Error> {
        let target_path = target.as_ref();
        if let Some(projector_path) = projector {
            if target_path.canonicalize()? == projector_path.canonicalize()? {
                return Err(Error::Invalid(
                    "target and projector must be distinct package components".into(),
                ));
            }
        }
        let target = gguf::inspect_header(target_path)?;
        let projector = projector.map(gguf::inspect_header).transpose()?;
        let identity = PackageIdentity {
            target: ArtifactIdentity::for_open(),
            projector: projector.as_ref().map(|_| ArtifactIdentity::for_open()),
        };
        Ok(Self {
            target,
            projector,
            identity,
        })
    }

    pub fn target(&self) -> &Directory {
        &self.target
    }

    pub fn projector(&self) -> Option<&Directory> {
        self.projector.as_ref()
    }

    pub fn identity(&self) -> PackageIdentity {
        self.identity
    }
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
        if let Some(projector) = &projector {
            if same_file(target.source(), projector.source())? {
                return Err(Error::Invalid(
                    "target and projector must be distinct package components".into(),
                ));
            }
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
}

fn same_file(
    left: &std::sync::Arc<crate::FileSource>,
    right: &std::sync::Arc<crate::FileSource>,
) -> Result<bool, Error> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let left = left.metadata()?;
        let right = right.metadata()?;
        Ok(left.dev() == right.dev() && left.ino() == right.ino())
    }
    #[cfg(not(unix))]
    {
        Ok(left.path() == right.path())
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
