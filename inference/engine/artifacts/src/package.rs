use crate::{
    gguf::{self, Directory, FileHeaders, GgufArtifact, HeaderSource, TensorDescriptor},
    ArtifactIdentity, Error, PackageIdentity, SplitError, TemplatePayload, TokenizerPayload,
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
    /// A separate draft model, carrying its own component identity.
    pub draft: Option<ComponentManifest>,
}

/// Validated package metadata when tensor payloads have not been downloaded.
/// It cannot be used as an import source: the open `Package` remains the
/// authority for payload-backed execution.
#[derive(Debug)]
pub struct PackageHeaders {
    target: HeaderComponent,
    projector: Option<HeaderComponent>,
    draft: Option<(HeaderComponent, ArtifactIdentity)>,
    identity: PackageIdentity,
}

/// One component's headers and its files' logical sizes.
#[derive(Debug)]
struct HeaderComponent {
    directory: Directory,
    files: Vec<ComponentFile>,
}

impl HeaderComponent {
    fn inspect(path: &Path, headers: &dyn HeaderSource) -> Result<Self, Error> {
        Self::inspect_shards(&[path.to_path_buf()], headers)
    }

    fn inspect_shards(paths: &[PathBuf], headers: &dyn HeaderSource) -> Result<Self, Error> {
        let (directory, files) = gguf::inspect_component_headers(paths, headers)?;
        Ok(Self {
            directory,
            files: files
                .into_iter()
                .map(|(path, size)| ComponentFile { path, size })
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

    fn into_manifest(self, identity: ArtifactIdentity) -> ComponentManifest {
        ComponentManifest {
            files: self.files,
            identity,
            tensors: self.directory.tensors,
        }
    }
}

impl PackageHeaders {
    /// Inspect a target's headers (every shard when `target` is the first
    /// shard of a split GGUF) and an optional explicit projector header.
    /// No tensor payload is read or required to exist.
    pub fn open(target: impl AsRef<Path>, projector: Option<&Path>) -> Result<Self, Error> {
        Self::open_from(target, projector, &FileHeaders)
    }

    /// [`Self::open`] with headers from `headers`.
    pub fn open_from(
        target: impl AsRef<Path>,
        projector: Option<&Path>,
        headers: &dyn HeaderSource,
    ) -> Result<Self, Error> {
        Self::open_shards_from(&[target.as_ref().to_path_buf()], projector, headers)
    }

    /// Inspect a target's headers from its explicit shard paths (in shard
    /// order) and an optional explicit projector header. No tensor payload is
    /// read or required to exist.
    pub fn open_shards(
        target_shards: &[PathBuf],
        projector: Option<&Path>,
    ) -> Result<Self, Error> {
        Self::open_shards_from(target_shards, projector, &FileHeaders)
    }

    /// [`Self::open_shards`] with headers from `headers`.
    pub fn open_shards_from(
        target_shards: &[PathBuf],
        projector: Option<&Path>,
        headers: &dyn HeaderSource,
    ) -> Result<Self, Error> {
        require_distinct_files(
            target_shards.iter().map(PathBuf::as_path),
            projector,
            headers,
            DISTINCT_PROJECTOR,
        )?;
        let target = HeaderComponent::inspect_shards(target_shards, headers)?;
        require_distinct_files(
            target.files.iter().map(|file| file.path.as_path()),
            projector,
            headers,
            DISTINCT_PROJECTOR,
        )?;
        let projector = projector
            .map(|projector| HeaderComponent::inspect(projector, headers))
            .transpose()?;
        let identity = PackageIdentity {
            target: ArtifactIdentity::for_open(),
            projector: projector.as_ref().map(|_| ArtifactIdentity::for_open()),
        };
        Ok(Self {
            target,
            projector,
            draft: None,
            identity,
        })
    }

    /// Inspect one component's headers from disk without attaching it to a
    /// package yet, so a caller can overlap draft inspection with target
    /// inspection and attach the result via [`Self::add_preopened_draft`].
    pub fn inspect_component(path: &Path) -> Result<(Directory, Vec<ComponentFile>), Error> {
        let component = HeaderComponent::inspect(path, &FileHeaders)?;
        Ok((component.directory, component.files))
    }

    /// Add a separate draft model's header as a component of the package.
    /// On failure the package keeps its target and projector.
    pub fn add_draft(&mut self, draft: &Path) -> Result<(), Error> {
        self.add_draft_from(draft, &FileHeaders)
    }

    /// [`Self::add_draft`] with the draft's header from `headers`.
    pub fn add_draft_from(&mut self, draft: &Path, headers: &dyn HeaderSource) -> Result<(), Error> {
        let component = HeaderComponent::inspect(draft, headers)?;
        self.add_preopened_draft_with(draft, component.directory, component.files, headers)
    }

    /// Attach a previously inspected draft component ([`Self::inspect_component`])
    /// to this package after verifying it is distinct from the target and
    /// projector.
    pub fn add_preopened_draft(
        &mut self,
        draft: &Path,
        directory: Directory,
        files: Vec<ComponentFile>,
    ) -> Result<(), Error> {
        self.add_preopened_draft_with(draft, directory, files, &FileHeaders)
    }

    fn add_preopened_draft_with(
        &mut self,
        draft: &Path,
        directory: Directory,
        files: Vec<ComponentFile>,
        headers: &dyn HeaderSource,
    ) -> Result<(), Error> {
        let others = self
            .target
            .files
            .iter()
            .chain(self.projector.iter().flat_map(|projector| &projector.files))
            .map(|file| file.path.as_path());
        require_distinct_files(others, [draft], headers, DISTINCT_DRAFT)?;
        self.draft = Some((HeaderComponent { directory, files }, ArtifactIdentity::for_open()));
        Ok(())
    }

    pub fn target(&self) -> &Directory {
        &self.target.directory
    }

    pub fn projector(&self) -> Option<&Directory> {
        self.projector
            .as_ref()
            .map(|projector| &projector.directory)
    }

    pub fn draft(&self) -> Option<&Directory> {
        self.draft.as_ref().map(|(draft, _)| &draft.directory)
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
            draft: self
                .draft
                .as_ref()
                .map(|(draft, identity)| draft.manifest(*identity)),
        }
    }

    /// Consume these headers into a [`PackageManifest`] without cloning the
    /// component tensor directories.
    pub fn into_manifest(self) -> PackageManifest {
        PackageManifest {
            identity: self.identity,
            target: self.target.into_manifest(self.identity.target),
            projector: self
                .projector
                .zip(self.identity.projector)
                .map(|(projector, identity)| projector.into_manifest(identity)),
            draft: self
                .draft
                .map(|(draft, identity)| draft.into_manifest(identity)),
        }
    }
}

/// A target GGUF and its optional projector and draft components.
///
/// The package establishes component ownership and identity only. A model-family
/// adapter decides whether the metadata describes a supported target/projector.
#[derive(Debug)]
pub struct Package {
    target: GgufArtifact,
    /// The model store: the directory the target's file lies in, its links
    /// followed ([`model_store`]), resolved as the package opens.
    store: PathBuf,
    projector: Option<GgufArtifact>,
    draft: Option<GgufArtifact>,
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

    /// Opens a target from its explicit shard paths (in shard order) with an
    /// optional explicit projector component.
    pub fn open_target_shards(
        target_shards: &[PathBuf],
        projector: Option<&Path>,
    ) -> Result<Self, Error> {
        require_distinct_files(
            target_shards.iter().map(PathBuf::as_path),
            projector,
            &FileHeaders,
            DISTINCT_PROJECTOR,
        )?;
        let target = GgufArtifact::open_shards(target_shards)?;
        let projector = projector.map(GgufArtifact::open).transpose()?;
        require_distinct_files(
            target.sources().map(|source| source.path()),
            projector
                .iter()
                .flat_map(GgufArtifact::sources)
                .map(|source| source.path()),
            &FileHeaders,
            DISTINCT_PROJECTOR,
        )?;
        let identity = PackageIdentity {
            target: target.identity(),
            projector: projector.as_ref().map(GgufArtifact::identity),
        };
        let first = target
            .sources()
            .next()
            .expect("open_shards returns at least one shard")
            .path();
        let store = model_store(first)?;
        let tokenizer = TokenizerPayload::from_directory(target.directory());
        let templates = TemplatePayload::from_directory(
            target.directory(),
            &component_files(&target)[0].path.display().to_string(),
        )?;
        Ok(Self {
            target,
            store,
            projector,
            draft: None,
            identity,
            tokenizer,
            templates,
        })
    }

    /// Opens a target from its explicit shard paths (in shard order) and
    /// discovers an unambiguous `mmproj-*.gguf` sibling beside the first shard.
    pub fn open_target_shards_discovering_projector(
        target_shards: &[PathBuf],
    ) -> Result<Self, Error> {
        let first = target_shards.first().ok_or(SplitError::Empty)?;
        let projector = discover_projector(first)?;
        Self::open_target_shards(target_shards, projector.as_deref())
    }

    /// Opens exactly the package a host admitted, in another process: each
    /// component must still have the admitted size and tensor directory, and
    /// the opened package carries the host's identities so both sides name
    /// the same package.
    pub fn open_manifest(manifest: &PackageManifest) -> Result<Self, Error> {
        let target_shards: Vec<PathBuf> = manifest
            .target
            .files
            .iter()
            .map(|file| file.path.clone())
            .collect();
        let mut package = Self::open_target_shards(
            &target_shards,
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
        if let Some(draft) = &manifest.draft {
            package = package.with_draft(draft.path())?;
        }
        admitted(&mut package.target, &manifest.target)?;
        if let (Some(artifact), Some(component)) =
            (package.projector.as_mut(), manifest.projector.as_ref())
        {
            admitted(artifact, component)?;
        }
        if let (Some(artifact), Some(component)) =
            (package.draft.as_mut(), manifest.draft.as_ref())
        {
            admitted(artifact, component)?;
        }
        package.identity = manifest.identity;
        Ok(package)
    }

    /// Add a separate draft model (a DFlash or DSpark GGUF) as a component
    /// of the package.
    pub fn with_draft(mut self, draft: impl AsRef<Path>) -> Result<Self, Error> {
        let draft = GgufArtifact::open(draft.as_ref())?;
        let others = self
            .target
            .sources()
            .chain(self.projector.iter().flat_map(GgufArtifact::sources))
            .map(|source| source.path());
        require_distinct_files(
            others,
            draft.sources().map(|source| source.path()),
            &FileHeaders,
            DISTINCT_DRAFT,
        )?;
        self.draft = Some(draft);
        Ok(self)
    }

    fn open_paths(target_path: &Path, projector_path: Option<&Path>) -> Result<Self, Error> {
        Self::open_target_shards(&[target_path.to_path_buf()], projector_path)
    }

    pub fn target(&self) -> &GgufArtifact {
        &self.target
    }

    /// The model store the target was opened from ([`model_store`]).
    pub fn store(&self) -> &Path {
        &self.store
    }

    pub fn projector(&self) -> Option<&GgufArtifact> {
        self.projector.as_ref()
    }

    pub fn draft(&self) -> Option<&GgufArtifact> {
        self.draft.as_ref()
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
            draft: self.draft.as_ref().map(component_manifest),
        }
    }
}

const DISTINCT_PROJECTOR: &str = "target and projector must be distinct package components";
const DISTINCT_DRAFT: &str = "the draft must be a distinct package component";

#[derive(PartialEq, Eq)]
enum FileIdentity {
    #[cfg(unix)]
    Inode(u64, u64),
    #[cfg(not(unix))]
    Canonical(PathBuf),
}

impl FileIdentity {
    fn of(path: &Path) -> Result<Self, Error> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let metadata = std::fs::metadata(path)?;
            Ok(Self::Inode(metadata.dev(), metadata.ino()))
        }
        #[cfg(not(unix))]
        {
            Ok(Self::Canonical(path.canonicalize()?))
        }
    }
}

pub(crate) fn same_disk_file(left: &Path, right: &Path) -> Result<bool, Error> {
    if left == right {
        return Ok(true);
    }
    Ok(FileIdentity::of(left)? == FileIdentity::of(right)?)
}

/// Verify that no file in `left` refers to the same underlying file as any
/// file in `right`, using `headers`'s file-identity check.
fn require_distinct_files<'a>(
    left: impl IntoIterator<Item = &'a Path>,
    right: impl IntoIterator<Item = &'a Path>,
    headers: &dyn HeaderSource,
    message: &'static str,
) -> Result<(), Error> {
    let right: Vec<&Path> = right.into_iter().collect();
    if right.is_empty() {
        return Ok(());
    }
    for path in left {
        for other in &right {
            if headers.same_file(path, other)? {
                return Err(Error::Invalid(message.into()));
            }
        }
    }
    Ok(())
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

/// The model store of `target`: the directory its file lies in, its links
/// followed. A content-addressed snapshot names its files by links into a
/// blob store that may lie in another directory, on another volume, and
/// that store is where its reads are served from: what the measurement job
/// measures a read line of (writing its probe there) and a streamed load
/// prices its reads on.
pub fn model_store(target: &Path) -> Result<PathBuf, Error> {
    let file = target.canonicalize()?;
    file.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| Error::Invalid(format!("{} has no directory", file.display())))
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

#[cfg(test)]
mod tests {
    use super::{Package, PackageHeaders};
    use crate::gguf::{Encoding, Metadata, Scalar, StoredHeader, StoredMetadata, StoredType, Value, ValueType};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temporary_directory(tag: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "magnitude-package-{tag}-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_shard(
        path: &Path,
        split_no: u64,
        split_count: u64,
        split_tensors: Option<u64>,
        name: Option<&str>,
        tensors: &[(&str, Vec<u64>, Encoding)],
        fill: u8,
    ) -> PathBuf {
        let mut metadata = vec![
            StoredMetadata {
                metadata: Metadata {
                    name: "split.no".into(),
                    value: Value::Scalar(Scalar::Unsigned(split_no)),
                },
                stored: StoredType::Scalar(ValueType::U16),
            },
            StoredMetadata {
                metadata: Metadata {
                    name: "split.count".into(),
                    value: Value::Scalar(Scalar::Unsigned(split_count)),
                },
                stored: StoredType::Scalar(ValueType::U16),
            },
        ];
        if let Some(count) = split_tensors {
            metadata.push(StoredMetadata {
                metadata: Metadata {
                    name: "split.tensors.count".into(),
                    value: Value::Scalar(Scalar::Unsigned(count)),
                },
                stored: StoredType::Scalar(ValueType::U32),
            });
        }
        if let Some(name) = name {
            metadata.push(StoredMetadata {
                metadata: Metadata {
                    name: "general.name".into(),
                    value: Value::Scalar(Scalar::String(name.into())),
                },
                stored: StoredType::Scalar(ValueType::String),
            });
        }
        let header = StoredHeader::new(
            3,
            metadata,
            tensors
                .iter()
                .map(|(name, shape, encoding)| ((*name).to_owned(), shape.clone(), *encoding)),
        )
        .unwrap();
        let mut bytes = header.encode().unwrap();
        let alignment = header.alignment().unwrap();
        let data_offset = bytes.len() as u64;
        for tensor in &header.tensors {
            let start = (data_offset + tensor.offset) as usize;
            if bytes.len() < start {
                bytes.resize(start, 0);
            }
            bytes.resize(start + tensor.nbytes as usize, fill);
        }
        let end = (bytes.len() as u64).next_multiple_of(alignment) as usize;
        bytes.resize(end, 0);
        std::fs::write(path, bytes).unwrap();
        path.to_path_buf()
    }

    #[test]
    fn open_shards_and_open_manifest_support_cross_directory_non_standard_splits() {
        let root = temporary_directory("cross-dir-split");
        let sub = root.join("Q4_K_M");
        std::fs::create_dir_all(&sub).unwrap();

        let shard0 = write_shard(
            &sub.join("model-00001-of-00003.gguf"),
            0,
            3,
            Some(2),
            Some("cross-dir-split"),
            &[],
            0,
        );
        let shard1 = write_shard(
            &sub.join("model-00002-of-00003.gguf"),
            1,
            3,
            None,
            None,
            &[("first", vec![2, 256], Encoding::Q4K)],
            0x33,
        );
        let shard2 = write_shard(
            &root.join("shared-non-standard-shard.gguf"),
            2,
            3,
            None,
            None,
            &[("second", vec![3, 32], Encoding::Q8_0)],
            0x44,
        );
        let shards = vec![shard0, shard1, shard2];

        let headers = PackageHeaders::open_shards(&shards, None).unwrap();
        let header_manifest = headers.manifest();
        assert_eq!(header_manifest.target.files.len(), 3);
        assert_eq!(header_manifest.target.tensors.len(), 2);

        let package = Package::open_target_shards(&shards, None).unwrap();
        let manifest = package.manifest();
        assert_eq!(manifest.target.files.len(), 3);
        assert_eq!(manifest.target.files, header_manifest.target.files);
        assert_eq!(manifest.target.tensors, header_manifest.target.tensors);

        let reopened = Package::open_manifest(&manifest).unwrap();
        assert_eq!(reopened.identity(), package.identity());
        assert_eq!(reopened.manifest(), manifest);
        assert_eq!(
            reopened.target().tensor("first").unwrap().read().unwrap(),
            vec![0x33; 288]
        );
        assert_eq!(
            reopened.target().tensor("second").unwrap().read().unwrap(),
            vec![0x44; 102]
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    /// A snapshot's link into a blob store in another directory is stored
    /// where the blob lies, not where the link does.
    #[cfg(unix)]
    #[test]
    fn a_linked_target_is_stored_where_its_file_lies() {
        let root = std::env::temp_dir().join(format!("model-store-{}", std::process::id()));
        let (blobs, snapshot) = (root.join("blobs"), root.join("snapshots/main"));
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(blobs.join("sha256-model"), b"GGUF").unwrap();
        let link = snapshot.join("model.gguf");
        std::os::unix::fs::symlink(blobs.join("sha256-model"), &link).unwrap();
        let stored = super::model_store(&link);
        let expected = blobs.canonicalize().unwrap();
        std::fs::remove_dir_all(&root).unwrap();
        assert_eq!(stored.unwrap(), expected);
    }
}
