use crate::{
    gguf::{self, Directory, GgufArtifact, TensorDescriptor},
    ArtifactIdentity, Error, PackageIdentity, TemplatePayload, TokenizerPayload,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// One file of a package component.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentFile {
    pub path: PathBuf,
    pub size: u64,
}

/// Device-free description of one opened package component.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawComponentManifest")]
pub struct ComponentManifest {
    /// The component's files in shard order: one for an unsplit GGUF, the
    /// first naming the component. Never empty.
    pub files: Vec<ComponentFile>,
    pub identity: ArtifactIdentity,
    /// The component directory's tensors (see [`gguf::Directory`]).
    pub tensors: Vec<TensorDescriptor>,
}

/// Deserialization requires at least one file.
#[derive(Deserialize)]
struct RawComponentManifest {
    files: Vec<ComponentFile>,
    identity: ArtifactIdentity,
    tensors: Vec<TensorDescriptor>,
}

impl TryFrom<RawComponentManifest> for ComponentManifest {
    type Error = String;
    fn try_from(raw: RawComponentManifest) -> Result<Self, String> {
        if raw.files.is_empty() {
            return Err("a package component has at least one file".into());
        }
        Ok(Self {
            files: raw.files,
            identity: raw.identity,
            tensors: raw.tensors,
        })
    }
}

impl ComponentManifest {
    /// The path that names the component: its file, or a split GGUF's first
    /// shard.
    pub fn path(&self) -> &Path {
        &self.files[0].path
    }
}

/// Description of the package admitted by the host. The opened package itself
/// is shared with the numerical worker so all reads use the admitted files.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    target: HeaderComponent,
    projector: Option<HeaderComponent>,
    identity: PackageIdentity,
}

/// One component's headers and its files' logical sizes.
#[derive(Debug)]
struct HeaderComponent {
    directory: Directory,
    files: Vec<ComponentFile>,
}

impl HeaderComponent {
    fn inspect(path: &Path) -> Result<Self, Error> {
        let (directory, sources) = gguf::inspect_component_header(path)?;
        Ok(Self {
            directory,
            files: sources
                .iter()
                .map(|source| ComponentFile {
                    path: source.path().to_path_buf(),
                    size: source.size(),
                })
                .collect(),
        })
    }

    fn manifest(&self, identity: ArtifactIdentity) -> ComponentManifest {
        ComponentManifest {
            files: self.files.clone(),
            identity,
            tensors: self.directory.tensors.clone(),
        }
    }
}

impl PackageHeaders {
    /// Inspect a target's headers (every shard when `target` is the first
    /// shard of a split GGUF) and an optional explicit projector header.
    /// No tensor payload is read or required to exist.
    pub fn open(target: impl AsRef<Path>, projector: Option<&Path>) -> Result<Self, Error> {
        let target_path = target.as_ref();
        if let Some(projector_path) = projector {
            if target_path.canonicalize()? == projector_path.canonicalize()? {
                return Err(Error::Invalid(
                    "target and projector must be distinct package components".into(),
                ));
            }
        }
        let target = HeaderComponent::inspect(target_path)?;
        let projector = projector.map(HeaderComponent::inspect).transpose()?;
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
        &self.target.directory
    }

    pub fn projector(&self) -> Option<&Directory> {
        self.projector
            .as_ref()
            .map(|projector| &projector.directory)
    }

    pub fn identity(&self) -> PackageIdentity {
        self.identity
    }

    /// The manifest a payload-backed open of the same files would admit:
    /// planning reads only component files, identities and tensor
    /// directories, so header-only material plans exactly as a load does.
    pub fn manifest(&self) -> PackageManifest {
        PackageManifest {
            identity: self.identity,
            target: self.target.manifest(self.identity.target),
            projector: self
                .projector
                .as_ref()
                .zip(self.identity.projector)
                .map(|(projector, identity)| projector.manifest(identity)),
        }
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

    /// Opens exactly the package a host admitted, in another process: each
    /// component must still have the admitted size and tensor directory, and
    /// the opened package carries the host's identities so both sides name
    /// the same package.
    pub fn open_manifest(manifest: &PackageManifest) -> Result<Self, Error> {
        let mut package = Self::open_paths(
            manifest.target.path(),
            manifest.projector.as_ref().map(ComponentManifest::path),
        )?;
        let admitted = |artifact: &mut GgufArtifact, component: &ComponentManifest| {
            if component_files(artifact) != component.files
                || artifact.directory().tensors != component.tensors
            {
                return Err(Error::Invalid(format!(
                    "{} differs from the admitted package component",
                    component.path().display()
                )));
            }
            artifact.adopt_identity(component.identity);
            Ok(())
        };
        admitted(&mut package.target, &manifest.target)?;
        if let (Some(artifact), Some(component)) =
            (package.projector.as_mut(), manifest.projector.as_ref())
        {
            admitted(artifact, component)?;
        }
        package.identity = manifest.identity;
        Ok(package)
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
            for target in target.sources() {
                for projector in projector.sources() {
                    if same_file(target, projector)? {
                        return Err(Error::Invalid(
                            "target and projector must be distinct package components".into(),
                        ));
                    }
                }
            }
        }
        let identity = PackageIdentity {
            target: target.identity(),
            projector: projector.as_ref().map(GgufArtifact::identity),
        };
        let tokenizer = TokenizerPayload::from_directory(target.directory());
        let templates = TemplatePayload::from_directory(
            target.directory(),
            &component_files(&target)[0].path.display().to_string(),
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

fn component_files(artifact: &GgufArtifact) -> Vec<ComponentFile> {
    artifact
        .sources()
        .map(|source| ComponentFile {
            path: source.path().to_path_buf(),
            size: source.size(),
        })
        .collect()
}

fn component_manifest(artifact: &GgufArtifact) -> ComponentManifest {
    ComponentManifest {
        files: component_files(artifact),
        identity: artifact.identity(),
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
