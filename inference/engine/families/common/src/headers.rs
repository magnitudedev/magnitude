//! GGUF headers for family and assessment tests: the release catalog's
//! components at their locked revisions, and the pinned artifacts outside the
//! catalog the tests read ([`catalog_headers::artifacts`]).
//!
//! The catalog's headers are exported from `inference/catalog/models.lock.json`
//! by `magnitude-service-catalog-tool headers --output DIR --model-store STORE`
//! (the evaluation flow's `--catalog-headers` runs it on the host), and
//! `MAGNITUDE_CATALOG_HEADERS` names the export (`DIR/headers`). Each is the
//! exact header, read as the engine reads a component before its payload
//! exists ([`PackageHeaders`]): a split target is all its shards. An export
//! of another lock or other artifacts is refused.

use magnitude_artifacts::catalog_headers::{
    self, lock_digest, CatalogComponent, ComponentRole, HeaderIndex, INDEX,
};
use magnitude_artifacts::gguf::{Directory, Encoding, Metadata, TensorDescriptor, Value};
use magnitude_artifacts::PackageHeaders;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The environment variable naming the catalog header export.
pub const ROOT: &str = "MAGNITUDE_CATALOG_HEADERS";

/// The catalog lock, of which the export must be.
const LOCK: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../catalog/models.lock.json"
);

/// The catalog header export.
pub fn root() -> PathBuf {
    std::env::var_os(ROOT)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!(
                "{ROOT} must name the catalog header export: \
             `cargo run -p magnitude-service-catalog-tool -- headers --output DIR \
             --model-store STORE` writes DIR/headers"
            )
        })
}

/// Every catalog component, from an export of this tree's lock.
pub fn index() -> &'static [CatalogComponent] {
    &export().components
}

/// The export's index, of this tree's lock and artifacts.
fn export() -> &'static HeaderIndex {
    static EXPORT: OnceLock<HeaderIndex> = OnceLock::new();
    EXPORT.get_or_init(|| {
        let path = root().join(INDEX);
        let index: HeaderIndex = serde_json::from_slice(
            &std::fs::read(&path).unwrap_or_else(|error| panic!("{path:?}: {error}")),
        )
        .unwrap_or_else(|error| panic!("{path:?}: {error}"));
        let lock = lock_digest(&std::fs::read(LOCK).unwrap());
        assert_eq!(
            index.lock, lock,
            "{path:?} is an export of another catalog lock; export this tree's"
        );
        assert_eq!(
            index.artifacts,
            catalog_headers::artifacts(),
            "{path:?} is an export of other artifacts; export this tree's"
        );
        index
    })
}

/// The directory of one catalog component.
pub fn component(component: &CatalogComponent) -> Directory {
    open(&component.file(&root()))
}

/// The directory of the pinned artifact `name` ([`catalog_headers::artifacts`]).
pub fn artifact(name: &str) -> Directory {
    let artifact = export()
        .artifacts
        .iter()
        .find(|artifact| artifact.name == name)
        .unwrap_or_else(|| panic!("no pinned artifact is named {name}"));
    open(&artifact.file(&root()))
}

fn open(file: &Path) -> Directory {
    PackageHeaders::open(file, None)
        .unwrap_or_else(|error| panic!("{file:?}: {error}"))
        .target()
        .clone()
}

/// The directory of catalog model `model`'s (`qwen3.5-4b:gguf:q4`) component
/// in `role`.
pub fn catalog(model: &str, role: ComponentRole) -> Directory {
    component(
        index()
            .iter()
            .find(|component| component.model == model && component.role == role)
            .unwrap_or_else(|| panic!("the catalog has no {model} {role}")),
    )
}

/// The directory of catalog model `model`'s target.
pub fn target(model: &str) -> Directory {
    catalog(model, ComponentRole::Target)
}

/// The directory of catalog model `model`'s separately packaged draft.
pub fn draft(model: &str) -> Directory {
    catalog(model, ComponentRole::Draft)
}

/// The directory of catalog model `model`'s projector.
pub fn projector(model: &str) -> Directory {
    catalog(model, ComponentRole::Projector)
}

/// A header a test names: a catalog component, or a pinned artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Header {
    /// Catalog model `.0`'s (`qwen3.5-4b:gguf:q4`) component in role `.1`.
    Catalog(&'static str, ComponentRole),
    /// The pinned artifact named `.0` ([`catalog_headers::artifacts`]).
    Artifact(&'static str),
}

impl Header {
    pub fn directory(self) -> Directory {
        match self {
            Self::Catalog(model, role) => catalog(model, role),
            Self::Artifact(name) => artifact(name),
        }
    }
}

impl std::fmt::Display for Header {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Catalog(model, role) => write!(formatter, "{model} {role}"),
            Self::Artifact(name) => formatter.write_str(name),
        }
    }
}

/// Sets metadata `name` to `value`, replacing any recorded value.
pub fn set(directory: &mut Directory, name: &str, value: Value) {
    directory.metadata.retain(|entry| entry.name != name);
    directory.metadata.push(Metadata {
        name: name.into(),
        value,
    });
}

pub fn remove_tensor(directory: &mut Directory, name: &str) {
    let before = directory.tensors.len();
    directory.tensors.retain(|tensor| tensor.name != name);
    assert_eq!(
        directory.tensors.len() + 1,
        before,
        "{name} is recorded once"
    );
}

pub fn reshape_tensor(directory: &mut Directory, name: &str, shape: &[u64]) {
    directory
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("{name} is recorded"))
        .shape = shape.into();
}

/// Adds an F32 tensor.
pub fn add_tensor(directory: &mut Directory, name: &str, shape: &[u64]) {
    directory.tensors.push(TensorDescriptor {
        name: name.into(),
        shape: shape.into(),
        encoding: Encoding::F32,
        offset: 0,
        nbytes: shape.iter().product::<u64>() * 4,
    });
}
