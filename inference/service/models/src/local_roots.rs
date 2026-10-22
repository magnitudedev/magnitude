//! Scanners for the read-only model locations ACN configures, and for drafter files the
//! configuration names in place. Each scanner reads directory layouts, manifests and GGUF headers
//! only; it never writes, moves or hashes a file in full, and partial downloads never form a group.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use magnitude_service_contracts::models::{
    DrafterChoice, DrafterReference, HuggingFaceArtifactSelector, HuggingFaceRepositoryId,
    LocalModelsConfiguration, ModelRoot, ModelRootKind,
};
use magnitude_service_contracts::{
    ComponentRelationship, ComponentRole, ContentIdentity, ExternalOrigin, HubCacheOwner, HubFile,
    Integrity, InventoryError, LocalDeclaration, ModelComponent, ModelLocation, ModelSource,
};

use crate::cache::{ModelCache, ModelIndexKind};
use crate::candidate_files::RecognizedArchitectures;
use crate::identity::{content_id, inventory_entry_id};
use crate::inventory::{
    ArtifactCandidate, DeclaredProjectors, DiscoveryCandidate, LinkedFiles, LocationScan,
    ModelGroup, UnreadableDirectories, canonical_or_self, components_for_group, discover_groups,
    inspect_cached, io_error, modified_seconds, now, primary_path, read_dir_sorted,
    readable_entries, scan_hf_cache, split_shard_name,
};

const OLLAMA_MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";
const OLLAMA_PROJECTOR_MEDIA_TYPE: &str = "application/vnd.ollama.image.projector";
const OLLAMA_TENSOR_MEDIA_TYPE: &str = "application/vnd.ollama.image.tensor";
const OLLAMA_MANIFEST_LIST_MEDIA_TYPE: &str = "application/vnd.ollama.manifest.list.v2+json";
const OLLAMA_GGUF_FORMAT: &str = "gguf";
const OLLAMA_REGISTRY: &str = "registry.ollama.ai";
const OLLAMA_REGISTRY_HOST: &str = "ollama.com";
const OLLAMA_MANIFEST_TREES: [&str; 2] = ["manifests", "manifests-v2"];
const MAX_OLLAMA_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_LLAMA_CPP_MANIFEST_BYTES: u64 = 1024 * 1024;

/// Name suffixes and infixes download tools give files that are still being written.
pub(crate) fn is_partial_download(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    [
        ".incomplete",
        ".part",
        ".partial",
        ".tmp",
        ".download",
        ".downloadinprogress",
        ".lock",
    ]
    .iter()
    .any(|suffix| lower.ends_with(suffix))
        || lower.contains("-partial")
}

/// One location's models. A location that can't be read at all fails, and is reported by its own
/// state; a directory inside it that can't be read is skipped and recorded in the scan, so the rest
/// of the location is still listed.
pub(crate) fn scan_root(
    root: &ModelRoot,
    declared: &DeclaredProjectors,
    cache: &ModelCache,
    architectures: &RecognizedArchitectures,
) -> Result<LocationScan, InventoryError> {
    let mut output = LocationScan::default();
    if root.kind == ModelRootKind::Folder && root.path.is_file() {
        scan_folder_file(&root.path, cache, architectures, &mut output)?;
        return Ok(output);
    }
    if !root.path.is_dir() {
        return Ok(output);
    }
    fs::read_dir(&root.path).map_err(io_error)?;
    match root.kind {
        ModelRootKind::HuggingFaceCache => scan_hf_cache(
            HubCacheOwner::HuggingFace,
            &root.path,
            declared,
            cache,
            architectures,
            &mut output,
        ),
        ModelRootKind::LmStudio => scan_lm_studio(&root.path, cache, architectures, &mut output),
        ModelRootKind::LlamaCpp => {
            scan_llama_cpp(&root.path, declared, cache, architectures, &mut output)
        }
        ModelRootKind::Ollama => scan_ollama(&root.path, cache, &mut output),
        ModelRootKind::Folder => scan_folder(&root.path, cache, architectures, &mut output),
    }?;
    Ok(output)
}

fn push_external(
    root: &Path,
    kind: &str,
    origin: ExternalOrigin,
    declared_by: LocalDeclaration,
    components: Vec<ModelComponent>,
    unsupported_format: Option<String>,
    output: &mut Vec<DiscoveryCandidate>,
) {
    let Some(primary) = primary_path(root, &components) else {
        return;
    };
    let content = content_id(&components);
    let id = inventory_entry_id(kind, root, &content);
    let timestamp = modified_seconds(&primary).unwrap_or_else(now);
    output.push(DiscoveryCandidate::Artifact(Box::new(ArtifactCandidate {
        id,
        content_id: content,
        created: timestamp,
        ready_at: timestamp,
        source: ModelSource::Local { declared_by },
        location: ModelLocation::External {
            origin,
            root: root.to_path_buf(),
            total_bytes: components.iter().map(|item| item.size_bytes).sum(),
            components,
            integrity: Integrity::Unverified {
                reason: "external_location".to_owned(),
            },
        },
        primary,
        deletable: false,
        unsupported_format,
    })));
}

fn push_group(
    root: &Path,
    kind: &str,
    origin: ExternalOrigin,
    group: &ModelGroup,
    output: &mut Vec<DiscoveryCandidate>,
) -> Result<(), InventoryError> {
    let components = components_for_group(root, group, None)?;
    push_external(
        root,
        kind,
        origin,
        LocalDeclaration::Discovery,
        components,
        None,
        output,
    );
    Ok(())
}

fn hub_file(repository: String, selector: String) -> Option<HubFile> {
    HuggingFaceRepositoryId::new(repository.clone()).ok()?;
    HuggingFaceArtifactSelector::new(selector.clone()).ok()?;
    Some(HubFile {
        repository,
        selector,
    })
}

fn relative_parts(root: &Path, path: &Path) -> Option<Vec<String>> {
    path.strip_prefix(root)
        .ok()?
        .components()
        .map(|component| match component {
            Component::Normal(part) => part.to_str().map(str::to_owned),
            _ => None,
        })
        .collect()
}

/// LM Studio keeps models as `<publisher>/<repository>/<file>.gguf`, the Hub's own names.
fn scan_lm_studio(
    root: &Path,
    cache: &ModelCache,
    architectures: &RecognizedArchitectures,
    output: &mut LocationScan,
) -> Result<(), InventoryError> {
    for group in discover_groups(
        root,
        &canonical_or_self(root),
        LinkedFiles::Anywhere,
        cache,
        architectures,
        &BTreeMap::new(),
        &mut output.unreadable,
    )? {
        let hub = lm_studio_hub_file(root, &group.paths[0]);
        push_group(
            root,
            "lm-studio",
            ExternalOrigin::LmStudio { hub },
            &group,
            &mut output.candidates,
        )?;
    }
    Ok(())
}

fn lm_studio_hub_file(root: &Path, path: &Path) -> Option<HubFile> {
    let parts = relative_parts(root, path)?;
    let [publisher, repository, selector @ ..] = parts.as_slice() else {
        return None;
    };
    if selector.is_empty() {
        return None;
    }
    hub_file(format!("{publisher}/{repository}"), selector.join("/"))
}

struct LlamaCppManifest {
    hub: HubFile,
    projector: Option<String>,
}

/// llama.cpp's `-hf` cache. Current releases write the Hugging Face hub layout
/// (`models--<owner>--<repository>/snapshots/<commit>/…`), in `LLAMA_CACHE` itself or its `hub`
/// folder; those repositories are scanned as a hub cache that llama.cpp owns. Older releases store
/// files flat as `<owner>_<repository>_<file>` beside `manifest=<owner>=<repository>=<tag>.json`,
/// which names the exact Hub file and its projector.
fn scan_llama_cpp(
    root: &Path,
    declared: &DeclaredProjectors,
    cache: &ModelCache,
    architectures: &RecognizedArchitectures,
    output: &mut LocationScan,
) -> Result<(), InventoryError> {
    for hub in [root.to_path_buf(), root.join("hub")] {
        scan_hf_cache(
            HubCacheOwner::LlamaCpp,
            &hub,
            declared,
            cache,
            architectures,
            output,
        )?;
    }
    let manifests = llama_cpp_manifests(root)?;
    let declared = manifests
        .iter()
        .filter_map(|(name, manifest)| {
            let projector = manifest.projector.as_ref()?;
            Some((root.join(name), root.join(projector)))
        })
        .collect::<BTreeMap<_, _>>();
    let hub_layout = |path: &Path| {
        relative_parts(root, path).is_some_and(|parts| {
            let mut parts = parts.iter();
            let first = parts.next();
            first.is_some_and(|part| part.starts_with("models--"))
                || (first.is_some_and(|part| part == "hub")
                    && parts
                        .next()
                        .is_some_and(|part| part.starts_with("models--")))
        })
    };
    for group in discover_groups(
        root,
        &canonical_or_self(root),
        LinkedFiles::WithinRoot,
        cache,
        architectures,
        &declared,
        &mut output.unreadable,
    )? {
        if hub_layout(&group.paths[0]) {
            continue;
        }
        let name = group.paths[0]
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        let hub = manifests
            .get(name)
            .map(|manifest| manifest.hub.clone())
            .or_else(|| parse_llama_cpp_name(name));
        push_group(
            root,
            "llama-cpp",
            ExternalOrigin::LlamaCpp { hub },
            &group,
            &mut output.candidates,
        )?;
    }
    Ok(())
}

fn llama_cpp_local_name(owner: &str, repository: &str, remote: &str) -> String {
    format!(
        "{}{}",
        llama_cpp_flat_prefix(owner, repository),
        remote.replace('/', "_")
    )
}

/// The prefix llama.cpp gives every file of `owner/repository` it keeps flat in its cache root.
#[must_use]
pub fn llama_cpp_flat_prefix(owner: &str, repository: &str) -> String {
    format!("{owner}_{repository}_")
}

fn llama_cpp_manifests(root: &Path) -> Result<BTreeMap<String, LlamaCppManifest>, InventoryError> {
    let mut manifests = BTreeMap::new();
    for entry in read_dir_sorted(root)? {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(key) = name
            .strip_prefix("manifest=")
            .and_then(|key| key.strip_suffix(".json"))
        else {
            continue;
        };
        let mut parts = key.splitn(3, '=');
        let (Some(owner), Some(repository)) = (parts.next(), parts.next()) else {
            continue;
        };
        let Some(document) = read_bounded_json(&entry.path(), MAX_LLAMA_CPP_MANIFEST_BYTES) else {
            continue;
        };
        let remote = |field: &str| {
            document
                .get(field)
                .and_then(|file| file.get("rfilename"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        };
        let Some(model) = remote("ggufFile") else {
            continue;
        };
        let Some(hub) = hub_file(format!("{owner}/{repository}"), model.clone()) else {
            continue;
        };
        manifests.insert(
            llama_cpp_local_name(owner, repository, &model),
            LlamaCppManifest {
                hub,
                projector: remote("mmprojFile")
                    .map(|projector| llama_cpp_local_name(owner, repository, &projector)),
            },
        );
    }
    Ok(manifests)
}

/// A flat cache name without a manifest parses only when its repository segment ends in `GGUF`,
/// the convention that keeps an owner or file name containing `_` from being misread.
fn parse_llama_cpp_name(name: &str) -> Option<HubFile> {
    let (owner, rest) = name.split_once('_')?;
    let split = rest
        .match_indices('_')
        .map(|(index, _)| index)
        .find(|index| rest[..*index].to_ascii_lowercase().ends_with("gguf"))?;
    hub_file(
        format!("{owner}/{}", &rest[..split]),
        rest[split + 1..].to_owned(),
    )
}

#[derive(serde::Deserialize)]
struct OllamaManifest {
    layers: Vec<OllamaLayer>,
}

/// A `manifests-v2` entry Ollama 0.40 and later write for a tag: one manifest per runner format,
/// each stored as a blob named by its digest.
#[derive(serde::Deserialize)]
struct OllamaManifestList {
    manifests: Vec<OllamaManifestListEntry>,
}

#[derive(serde::Deserialize)]
struct OllamaManifestListEntry {
    digest: String,
    format: Option<String>,
}

#[derive(serde::Deserialize)]
struct OllamaLayer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
    size: u64,
}

/// One Ollama model: its registry path, the name Ollama shows for it, and its parsed manifest.
struct OllamaPull {
    registry: String,
    namespace: String,
    model: String,
    name: String,
    manifest: OllamaManifest,
}

/// Every model under an Ollama root. Ollama writes each pull's manifest to
/// `manifests/<registry>/<namespace>/<model>/<tag>`, and newer releases also to
/// `manifests-v2/<registry>/<namespace>/<model>/<tag>`, often as a symlink to the manifest stored as
/// a blob, naming Ollama's own registry `ollama.com` rather than `registry.ollama.ai`. A manifest is
/// read through links wherever they lead. The same model in both trees is one pull, the
/// manifest whose blobs are all present when only one is complete. A manifest directory that can't
/// be read is recorded in `unreadable`, and every other pull is still listed.
fn ollama_pulls(root: &Path, unreadable: &mut UnreadableDirectories) -> Vec<OllamaPull> {
    let mut pulls = BTreeMap::<String, OllamaPull>::new();
    for tree in OLLAMA_MANIFEST_TREES {
        for registry in readable_entries(&root.join(tree), unreadable) {
            let registry_name = ollama_registry(&registry.file_name().to_string_lossy());
            for namespace in readable_entries(&registry.path(), unreadable) {
                let namespace_name = namespace.file_name().to_string_lossy().into_owned();
                for model in readable_entries(&namespace.path(), unreadable) {
                    let model_name = model.file_name().to_string_lossy().into_owned();
                    for tag in readable_entries(&model.path(), unreadable) {
                        let Some(manifest) = read_ollama_manifest(root, &tag.path())
                        else {
                            continue;
                        };
                        let tag_name = tag.file_name().to_string_lossy().into_owned();
                        let name =
                            ollama_name(&registry_name, &namespace_name, &model_name, &tag_name);
                        let pull = OllamaPull {
                            registry: registry_name.clone(),
                            namespace: namespace_name.clone(),
                            model: model_name.clone(),
                            name: name.clone(),
                            manifest,
                        };
                        let keep_existing = pulls.get(&name).is_some_and(|existing| {
                            ollama_model(root, &existing.manifest).is_some()
                                || ollama_model(root, &pull.manifest).is_none()
                        });
                        if !keep_existing {
                            pulls.insert(name, pull);
                        }
                    }
                }
            }
        }
    }
    pulls.into_values().collect()
}

/// The manifest a tag names, read only when it resolves inside the root. A manifest list names one
/// manifest per runner format, stored as a blob only for the format Ollama pulled; it resolves to
/// its GGUF manifest when that is on disk, and otherwise to the one that is.
fn read_ollama_manifest(root: &Path, tag: &Path) -> Option<OllamaManifest> {
    let read = |path: &Path| read_bounded_json(path, MAX_OLLAMA_MANIFEST_BYTES);
    let document = read(tag)?;
    if document.get("mediaType").and_then(|value| value.as_str())
        != Some(OLLAMA_MANIFEST_LIST_MEDIA_TYPE)
    {
        return serde_json::from_value(document).ok();
    }
    let mut list = serde_json::from_value::<OllamaManifestList>(document).ok()?;
    list.manifests
        .sort_by_key(|entry| entry.format.as_deref() != Some(OLLAMA_GGUF_FORMAT));
    list.manifests.iter().find_map(|entry| {
        let hex = sha256_hex(&entry.digest)?;
        let manifest = read(&root.join("blobs").join(format!("sha256-{hex}")))?;
        serde_json::from_value(manifest).ok()
    })
}

/// The hex of a `sha256:<hex>` digest, when well formed.
fn sha256_hex(digest: &str) -> Option<&str> {
    digest.strip_prefix("sha256:").filter(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

/// Ollama's own registry under the name its older manifests use.
fn ollama_registry(name: &str) -> String {
    if name == OLLAMA_REGISTRY_HOST {
        OLLAMA_REGISTRY.to_owned()
    } else {
        name.to_owned()
    }
}

impl OllamaPull {
    /// The repository of a `hf.co/<owner>/<repository>:<tag>` pull.
    fn hub_repository(&self) -> Option<String> {
        matches!(self.registry.as_str(), "hf.co" | "huggingface.co")
            .then(|| format!("{}/{}", self.namespace, self.model))
            .filter(|repository| HuggingFaceRepositoryId::new(repository.clone()).is_ok())
    }

    /// The SHA-256 of the model layer, which is the pulled file's own.
    fn model_sha256(&self) -> Option<&str> {
        self.manifest
            .layers
            .iter()
            .find(|layer| layer.media_type == OLLAMA_MODEL_MEDIA_TYPE)?
            .digest
            .strip_prefix("sha256:")
    }
}

/// Which file of a Hub repository a `hf.co` pull's bytes are, once resolved: `file` is `None` when
/// no file of the repository has those bytes.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub(crate) struct OllamaHubFile {
    pub(crate) file: Option<String>,
}

fn ollama_hub_file_evidence(repository: &str, sha256: &str) -> String {
    format!("ollama-hub-file-v1:{}@{sha256}", repository.to_lowercase())
}

pub(crate) fn read_ollama_hub_file(
    cache: &ModelCache,
    repository: &str,
    sha256: &str,
) -> Option<OllamaHubFile> {
    cache.read_index(
        ModelIndexKind::OllamaHubFile,
        &ollama_hub_file_evidence(repository, sha256),
    )
}

pub(crate) fn write_ollama_hub_file(
    cache: &ModelCache,
    repository: &str,
    sha256: &str,
    file: &OllamaHubFile,
) {
    cache.write_index(
        ModelIndexKind::OllamaHubFile,
        &ollama_hub_file_evidence(repository, sha256),
        file,
    );
}

/// `hf.co` pulls under the enabled Ollama roots whose Hub file is not yet resolved, as
/// `(repository, sha256)`.
pub(crate) fn unresolved_ollama_hub_pulls(
    roots: &[ModelRoot],
    cache: &ModelCache,
) -> Vec<(String, String)> {
    let mut unresolved = roots
        .iter()
        .filter(|root| root.enabled && root.kind == ModelRootKind::Ollama)
        .flat_map(|root| ollama_pulls(&root.path, &mut UnreadableDirectories::new()))
        .filter_map(|pull| {
            let repository = pull.hub_repository()?;
            let sha256 = pull.model_sha256()?.to_owned();
            read_ollama_hub_file(cache, &repository, &sha256)
                .is_none()
                .then_some((repository, sha256))
        })
        .collect::<Vec<_>>();
    unresolved.sort();
    unresolved.dedup();
    unresolved
}

/// Each Ollama manifest's model blob. Ollama records only the tag of a `hf.co` pull, so its Hub
/// identity is the repository file with the same bytes, resolved once and cached; until then, or
/// when no such file exists, the blob has no Hub identity and is identified by its header. Either
/// way the same bytes have the same identity whichever locations are enabled.
fn scan_ollama(
    root: &Path,
    cache: &ModelCache,
    output: &mut LocationScan,
) -> Result<(), InventoryError> {
    for pull in ollama_pulls(root, &mut output.unreadable) {
        let hub = pull.hub_repository().and_then(|repository| {
            let file = read_ollama_hub_file(cache, &repository, pull.model_sha256()?)?.file?;
            hub_file(repository, file)
        });
        let (components, unsupported_format) = match ollama_model(root, &pull.manifest) {
            Some(OllamaModel::Gguf(components)) => (components, None),
            Some(OllamaModel::Tensors(components)) => {
                let detail = format!(
                    "Ollama stores {} as {} safetensors tensor files for its MLX runner, not as a \
                     GGUF",
                    pull.name,
                    components.len()
                );
                (components, Some(detail))
            }
            None => continue,
        };
        push_external(
            root,
            "ollama",
            ExternalOrigin::Ollama {
                name: pull.name,
                hub,
            },
            LocalDeclaration::Discovery,
            components,
            unsupported_format,
            &mut output.candidates,
        );
    }
    Ok(())
}

/// The name Ollama shows for a model.
fn ollama_name(registry: &str, namespace: &str, model: &str, tag: &str) -> String {
    match (registry, namespace) {
        (OLLAMA_REGISTRY, "library") => format!("{model}:{tag}"),
        (OLLAMA_REGISTRY, _) => format!("{namespace}/{model}:{tag}"),
        _ => format!("{registry}/{namespace}/{model}:{tag}"),
    }
}

/// A layer's blob once it is on disk, read through any link wherever it leads: people move
/// Ollama's storage to another drive by linking a blob, the blobs folder or the whole store. A
/// blob still being pulled has a different size, or exists only as `-partial` files.
fn ollama_blob(root: &Path, layer: &OllamaLayer) -> Option<(PathBuf, String)> {
    let hex = sha256_hex(&layer.digest)?;
    let relative = PathBuf::from("blobs").join(format!("sha256-{hex}"));
    fs::metadata(root.join(&relative))
        .ok()
        .filter(|metadata| metadata.is_file() && metadata.len() == layer.size)?;
    Some((relative, hex.to_owned()))
}

/// A layer's blob that is a link whose target can't be read, such as a file on a drive that was
/// removed. Its model is listed as unreadable rather than left out as if never pulled.
fn ollama_broken_link(root: &Path, layer: &OllamaLayer) -> Option<(PathBuf, String)> {
    let hex = sha256_hex(&layer.digest)?;
    let relative = PathBuf::from("blobs").join(format!("sha256-{hex}"));
    let path = root.join(&relative);
    (fs::metadata(&path).is_err()
        && fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.file_type().is_symlink()))
    .then(|| (relative, hex.to_owned()))
}

/// What a manifest holds once every blob it names for the model is on disk.
enum OllamaModel {
    /// One GGUF blob, with its projector when that is on disk too.
    Gguf(Vec<ModelComponent>),
    /// One safetensors file per tensor, as Ollama stores a model for its MLX runner. Nothing in
    /// them is GGUF, so the engine can't read them in place.
    Tensors(Vec<ModelComponent>),
}

fn ollama_model(root: &Path, manifest: &OllamaManifest) -> Option<OllamaModel> {
    if let Some(components) = ollama_components(root, manifest) {
        return Some(OllamaModel::Gguf(components));
    }
    let tensors = manifest
        .layers
        .iter()
        .filter(|layer| layer.media_type == OLLAMA_TENSOR_MEDIA_TYPE)
        .map(|layer| {
            let (path, hex) = ollama_blob(root, layer)?;
            Some(ModelComponent {
                path,
                role: ComponentRole::Weights,
                size_bytes: layer.size,
                content: ContentIdentity::Sha256 { value: hex },
                shard_index: None,
                relationship: None,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    (!tensors.is_empty()).then_some(OllamaModel::Tensors(tensors))
}

fn ollama_components(root: &Path, manifest: &OllamaManifest) -> Option<Vec<ModelComponent>> {
    let layer = |media_type: &str| {
        manifest
            .layers
            .iter()
            .find(|layer| layer.media_type == media_type)
    };
    let model_layer = layer(OLLAMA_MODEL_MEDIA_TYPE)?;
    let (model_path, model_hex) =
        ollama_blob(root, model_layer).or_else(|| ollama_broken_link(root, model_layer))?;
    let mut components = vec![ModelComponent {
        path: model_path.clone(),
        role: ComponentRole::Weights,
        size_bytes: model_layer.size,
        content: ContentIdentity::Sha256 { value: model_hex },
        shard_index: None,
        relationship: None,
    }];
    if let Some(projector_layer) = layer(OLLAMA_PROJECTOR_MEDIA_TYPE)
        && let Some((projector_path, projector_hex)) = ollama_blob(root, projector_layer)
    {
        components.push(ModelComponent {
            path: projector_path.clone(),
            role: ComponentRole::Projector,
            size_bytes: projector_layer.size,
            content: ContentIdentity::Sha256 {
                value: projector_hex,
            },
            shard_index: None,
            relationship: Some(ComponentRelationship::ProjectorFor {
                projector: projector_path,
                model: model_path,
            }),
        });
    }
    Some(components)
}

fn scan_folder(
    root: &Path,
    cache: &ModelCache,
    architectures: &RecognizedArchitectures,
    output: &mut LocationScan,
) -> Result<(), InventoryError> {
    for group in discover_groups(
        root,
        &canonical_or_self(root),
        LinkedFiles::Anywhere,
        cache,
        architectures,
        &BTreeMap::new(),
        &mut output.unreadable,
    )? {
        push_group(
            root,
            "folder",
            ExternalOrigin::Folder,
            &group,
            &mut output.candidates,
        )?;
    }
    Ok(())
}

/// A single GGUF file listed as a model location: the file, with its sibling shards when it is a
/// split model's first shard (a set missing a shard contributes nothing), and the drafters in its
/// own folder, so its automatic drafter resolves wherever the folder is. Shards past the first
/// listed on their own are no model.
fn scan_folder_file(
    file: &Path,
    cache: &ModelCache,
    architectures: &RecognizedArchitectures,
    output: &mut LocationScan,
) -> Result<(), InventoryError> {
    let Some(folder) = file.parent() else {
        return Ok(());
    };
    let paths = match split_shard_name(file) {
        None => vec![file.to_path_buf()],
        Some(_) => match file
            .file_name()
            .and_then(|name| name.to_str())
            .map(crate::candidate_files::split_set)
        {
            Some(shards) if shards.len() > 1 => {
                shards.into_iter().map(|name| folder.join(name)).collect()
            }
            _ => return Ok(()),
        },
    };
    if paths
        .iter()
        .any(|path| !path.is_file() || is_partial_download(path))
    {
        return Ok(());
    }
    let model = ModelGroup {
        paths,
        projector: None,
    };
    let beside = folder_drafters(folder, cache, architectures)
        .into_iter()
        .filter(|path| path != file)
        .map(|path| ModelGroup {
            paths: vec![path],
            projector: None,
        });
    for group in std::iter::once(model).chain(beside) {
        push_group(
            folder,
            "folder",
            ExternalOrigin::Folder,
            &group,
            &mut output.candidates,
        )?;
    }
    Ok(())
}

/// Drafters attached by local path, referenced in place wherever they are. A missing file simply
/// contributes nothing.
pub(crate) fn scan_attached_drafters(
    configuration: &LocalModelsConfiguration,
    output: &mut Vec<DiscoveryCandidate>,
) {
    let attached =
        configuration
            .drafters
            .iter()
            .filter_map(|attachment| match &attachment.drafter {
                DrafterChoice::Reference {
                    reference: DrafterReference::Local { path },
                } => Some(path),
                _ => None,
            });
    for path in attached {
        if let Err(error) = push_attached_drafter(path, output) {
            tracing::warn!(
                path = %path.display(),
                %error,
                "attached drafter file could not be observed"
            );
        }
    }
}

/// The drafters "in this folder" for a model listed by its file in `folder`: its own single-file
/// GGUFs, never a subfolder's, whose header declares a drafter role.
fn folder_drafters(
    folder: &Path,
    cache: &ModelCache,
    architectures: &RecognizedArchitectures,
) -> Vec<PathBuf> {
    let Ok(entries) = read_dir_sorted(folder) else {
        return Vec::new();
    };
    entries
        .into_iter()
        .filter(|entry| !entry.file_name().to_string_lossy().starts_with('.'))
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
                && !is_partial_download(path)
                && split_shard_name(path).is_none()
                && path.is_file()
        })
        .filter(|path| {
            inspect_cached(cache, path).is_some_and(|inspection| {
                inspection.role(architectures) == crate::gguf::GgufRole::Drafter
            })
        })
        .collect()
}

fn push_attached_drafter(
    path: &Path,
    output: &mut Vec<DiscoveryCandidate>,
) -> Result<(), InventoryError> {
    if !path.is_file() || is_partial_download(path) {
        return Ok(());
    }
    let Some(root) = path.parent() else {
        return Ok(());
    };
    let group = ModelGroup {
        paths: vec![path.to_path_buf()],
        projector: None,
    };
    let components = components_for_group(root, &group, None)?;
    push_external(
        root,
        "attached",
        ExternalOrigin::Folder,
        LocalDeclaration::Configuration,
        components,
        None,
        output,
    );
    Ok(())
}

fn read_bounded_json(path: &Path, limit: u64) -> Option<serde_json::Value> {
    let metadata = fs::metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() > limit {
        return None;
    }
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

/// Writes an Ollama 0.40 pull of a model stored for its MLX runner, as `ollama pull qwen3.5:4b`
/// leaves it on Apple silicon: every tensor its own safetensors blob named in the manifest, the
/// tag in `manifests-v2/ollama.com/library/<model>/<tag>` a symlink to a manifest list whose GGUF
/// entry was never downloaded, and the MLX manifest both as a blob and as the `manifests/` copy.
/// Returns the tensor blobs' total size.
#[cfg(all(test, unix))]
pub(crate) fn write_ollama_mlx_pull(root: &Path, model: &str, tag: &str, seed: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let blobs = root.join("blobs");
    fs::create_dir_all(&blobs).unwrap();
    let blob = |bytes: &[u8]| {
        let digest = format!("{:x}", Sha256::digest(bytes));
        fs::write(blobs.join(format!("sha256-{digest}")), bytes).unwrap();
        serde_json::json!({ "digest": format!("sha256:{digest}"), "size": bytes.len() })
    };
    let safetensors = |name: &str, dtype: &str, shape: &[u64], data: &[u8]| {
        let header = serde_json::to_vec(&serde_json::json!({
            "__metadata__": { "quant_type": "nvfp4", "group_size": "16" },
            name: { "dtype": dtype, "shape": shape, "data_offsets": [0, data.len()] },
        }))
        .unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        bytes
    };
    let tensors = [
        (
            "language_model.model.embed_tokens.weight",
            "U32",
            &[4u64, 2][..],
        ),
        (
            "language_model.model.layers.0.linear_attn.dt_bias",
            "BF16",
            &[4][..],
        ),
        (
            "language_model.model.layers.0.mlp.down_proj.weight",
            "U32",
            &[2, 2][..],
        ),
        ("vision_tower.blocks.0.attn.qkv.weight", "BF16", &[2, 2][..]),
    ];
    let mut total = 0;
    let mut layers = Vec::new();
    for (name, dtype, shape) in tensors {
        let data = format!("{seed}:{name}").into_bytes();
        let bytes = safetensors(name, dtype, shape, &data);
        total += bytes.len() as u64;
        let mut layer = blob(&bytes);
        layer["mediaType"] = OLLAMA_TENSOR_MEDIA_TYPE.into();
        layer["name"] = name.into();
        layers.push(layer);
    }
    for (name, bytes) in [
        ("config.json", br#"{"model_type":"qwen3_5"}"#.as_slice()),
        ("tokenizer.json", br#"{"version":"1.0"}"#.as_slice()),
    ] {
        let mut layer = blob(bytes);
        layer["mediaType"] = "application/vnd.ollama.image.json".into();
        layer["name"] = name.into();
        layers.push(layer);
    }
    let mut license = blob(b"Apache License 2.0");
    license["mediaType"] = "application/vnd.ollama.image.license".into();
    layers.push(license);
    let mut params = blob(br#"{"temperature":1,"top_k":20}"#);
    params["mediaType"] = "application/vnd.ollama.image.params".into();
    layers.push(params);
    let mut config = blob(
        br#"{"model_format":"safetensors","file_type":"nvfp4","renderer":"qwen3.5","capabilities":["completion","vision","thinking"]}"#,
    );
    config["mediaType"] = "application/vnd.docker.container.image.v1+json".into();
    let manifest = serde_json::to_vec(&serde_json::json!({
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": config,
        "layers": layers,
        "runner": "mlx",
        "format": "safetensors",
    }))
    .unwrap();
    let mlx = blob(&manifest);
    let gguf = format!(
        "sha256:{:x}",
        Sha256::digest(format!("{seed} gguf manifest"))
    );
    let list = serde_json::to_vec(&serde_json::json!({
        "schemaVersion": 2,
        "mediaType": OLLAMA_MANIFEST_LIST_MEDIA_TYPE,
        "manifests": [
            {
                "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                "digest": gguf,
                "runner": "llamacpp",
                "format": "gguf",
            },
            {
                "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                "digest": mlx["digest"],
                "runner": "mlx",
                "format": "safetensors",
            },
        ],
    }))
    .unwrap();
    let list = blob(&list);
    let v1 = root
        .join("manifests")
        .join(OLLAMA_REGISTRY)
        .join("library")
        .join(model)
        .join(tag);
    fs::create_dir_all(v1.parent().unwrap()).unwrap();
    fs::write(v1, &manifest).unwrap();
    let v2 = root
        .join("manifests-v2")
        .join(OLLAMA_REGISTRY_HOST)
        .join("library")
        .join(model)
        .join(tag);
    fs::create_dir_all(v2.parent().unwrap()).unwrap();
    let list_blob = format!(
        "sha256-{}",
        list["digest"]
            .as_str()
            .unwrap()
            .trim_start_matches("sha256:")
    );
    std::os::unix::fs::symlink(Path::new("../../../../blobs").join(list_blob), v2).unwrap();
    total
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::symlink;

    use sha2::{Digest, Sha256};

    use super::*;
    use crate::test_gguf::LockedDirectory;

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn manifest_path(root: &Path, tree: &str, registry: &str, path: &str) -> PathBuf {
        let manifest = root.join(tree).join(registry).join(path);
        fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        manifest
    }

    /// Writes the model blob and returns the manifest naming it.
    fn manifest(root: &Path, model: &[u8], stored_size: usize) -> Vec<u8> {
        let blobs = root.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let digest = sha256(model);
        fs::write(
            blobs.join(format!("sha256-{digest}")),
            &model[..stored_size],
        )
        .unwrap();
        serde_json::to_vec(&serde_json::json!({
            "schemaVersion": 2,
            "layers": [{
                "mediaType": OLLAMA_MODEL_MEDIA_TYPE,
                "digest": format!("sha256:{digest}"),
                "size": model.len(),
            }],
        }))
        .unwrap()
    }

    /// `manifests/<registry>/<path>` as a regular file, as every Ollama release writes it.
    fn write_v1(root: &Path, registry: &str, path: &str, model: &[u8]) {
        let document = manifest(root, model, model.len());
        fs::write(manifest_path(root, "manifests", registry, path), document).unwrap();
    }

    /// `manifests-v2/<registry>/<path>` as a symlink to the manifest stored as a blob.
    fn write_v2(root: &Path, registry: &str, path: &str, document: &[u8]) {
        let blob = format!("sha256-{}", sha256(document));
        fs::write(root.join("blobs").join(&blob), document).unwrap();
        symlink(
            Path::new("../../../../blobs").join(blob),
            manifest_path(root, "manifests-v2", registry, path),
        )
        .unwrap();
    }

    fn names(root: &Path) -> Vec<String> {
        ollama_pulls(root, &mut UnreadableDirectories::new())
            .into_iter()
            .map(|pull| pull.name)
            .collect()
    }

    #[test]
    fn ollama_reads_symlinked_v2_manifests_and_names_ollama_com_as_its_registry() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let qwen = manifest(root, b"qwen weights", 12);
        write_v2(root, "ollama.com", "library/qwen3.5/0.8b", &qwen);
        let shared = manifest(root, b"shared namespace", 16);
        write_v2(root, "ollama.com", "someone/tool/latest", &shared);
        let hub = manifest(root, b"hub weights", 11);
        write_v2(root, "hf.co", "unsloth/Qwen3.5-0.8B-GGUF/Q4_K_M", &hub);

        assert_eq!(
            names(root),
            [
                "hf.co/unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
                "qwen3.5:0.8b",
                "someone/tool:latest",
            ]
        );
        let pulls = ollama_pulls(root, &mut UnreadableDirectories::new());
        assert!(
            pulls
                .iter()
                .all(|pull| ollama_components(root, &pull.manifest).is_some())
        );
        assert_eq!(
            pulls[0].hub_repository().as_deref(),
            Some("unsloth/Qwen3.5-0.8B-GGUF")
        );
    }

    /// Ollama 0.40 writes only `manifests-v2/ollama.com/...` for a library pull, and the tag
    /// links to a manifest list whose GGUF entry names the real manifest by digest. The
    /// safetensors entry's manifest is not on disk.
    #[test]
    fn ollama_reads_a_v2_only_store_whose_tag_links_to_a_manifest_list() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let blobs = root.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let blob = |bytes: &[u8]| {
            let digest = sha256(bytes);
            fs::write(blobs.join(format!("sha256-{digest}")), bytes).unwrap();
            digest
        };
        let model = blob(b"qwen weights");
        let draft = blob(b"qwen draft weights");
        let template = blob(b"{{ .Prompt }}");
        let config = blob(br#"{"model_format":"gguf","model_family":"qwen35"}"#);
        let gguf_manifest = serde_json::to_vec(&serde_json::json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": format!("sha256:{config}"),
                "size": 47,
            },
            "layers": [
                {"mediaType": OLLAMA_MODEL_MEDIA_TYPE, "digest": format!("sha256:{model}"), "size": 12},
                {"mediaType": "application/vnd.ollama.image.draft", "digest": format!("sha256:{draft}"), "size": 18},
                {"mediaType": "application/vnd.ollama.image.template", "digest": format!("sha256:{template}"), "size": 13},
            ],
        }))
        .unwrap();
        let gguf = blob(&gguf_manifest);
        let list = format!(
            r#"{{"schemaVersion":2,"mediaType":"application/vnd.ollama.manifest.list.v2+json","manifests":[{{"mediaType":"application/vnd.docker.distribution.manifest.v2+json","digest":"sha256:{gguf}","runner":"llamacpp","format":"gguf"}},{{"mediaType":"application/vnd.docker.distribution.manifest.v2+json","digest":"sha256:{}","runner":"mlx","format":"safetensors"}}]}}"#,
            sha256(b"mlx manifest not pulled"),
        );
        let list = blob(list.as_bytes());
        symlink(
            Path::new("../../../../blobs").join(format!("sha256-{list}")),
            manifest_path(root, "manifests-v2", "ollama.com", "library/qwen3.5/0.8b"),
        )
        .unwrap();
        assert!(!root.join("manifests").exists());

        let pulls = ollama_pulls(root, &mut UnreadableDirectories::new());
        assert_eq!(
            pulls
                .iter()
                .map(|pull| pull.name.as_str())
                .collect::<Vec<_>>(),
            ["qwen3.5:0.8b"]
        );
        assert_eq!(pulls[0].model_sha256(), Some(model.as_str()));
        let components = ollama_components(root, &pulls[0].manifest).unwrap();
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn ollama_lists_a_pull_in_both_manifest_trees_once() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        write_v1(
            root,
            "registry.ollama.ai",
            "library/qwen3.5/0.8b",
            b"qwen weights",
        );
        let qwen = manifest(root, b"qwen weights", 12);
        write_v2(root, "ollama.com", "library/qwen3.5/0.8b", &qwen);
        write_v1(
            root,
            "hf.co",
            "unsloth/Qwen3.5-0.8B-GGUF/Q4_K_M",
            b"hub weights",
        );
        let hub = manifest(root, b"hub weights", 11);
        write_v2(root, "hf.co", "unsloth/Qwen3.5-0.8B-GGUF/Q4_K_M", &hub);

        assert_eq!(
            names(root),
            ["hf.co/unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M", "qwen3.5:0.8b"]
        );
        let unresolved = unresolved_ollama_hub_pulls(
            &[ModelRoot {
                kind: ModelRootKind::Ollama,
                path: root.to_path_buf(),
                enabled: true,
            }],
            &ModelCache::new(&temporary.path().join("cache")),
        );
        assert_eq!(
            unresolved,
            [(
                "unsloth/Qwen3.5-0.8B-GGUF".to_owned(),
                sha256(b"hub weights")
            )]
        );
    }

    #[test]
    fn ollama_prefers_the_manifest_whose_blobs_are_complete() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let stale = manifest(root, b"stale weights", 5);
        fs::write(
            manifest_path(
                root,
                "manifests",
                "registry.ollama.ai",
                "library/qwen3.5/0.8b",
            ),
            stale,
        )
        .unwrap();
        let complete = manifest(root, b"complete weights", 16);
        write_v2(root, "ollama.com", "library/qwen3.5/0.8b", &complete);

        let pulls = ollama_pulls(root, &mut UnreadableDirectories::new());
        assert_eq!(pulls.len(), 1);
        assert_eq!(
            pulls[0].model_sha256(),
            Some(sha256(b"complete weights").as_str())
        );
    }

    /// A store moved to another drive by links is read through them, manifests included.
    #[test]
    fn ollama_reads_a_v2_manifest_linked_outside_the_root() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("ollama");
        let outside = temporary.path().join("outside.json");
        fs::write(&outside, manifest(&root, b"elsewhere", 9)).unwrap();
        symlink(
            &outside,
            manifest_path(&root, "manifests-v2", "ollama.com", "library/moved/latest"),
        )
        .unwrap();

        assert_eq!(names(&root), ["moved:latest"]);
    }

    #[test]
    fn ollama_scans_a_distribution_service_root() {
        // Arch's package sets OLLAMA_MODELS=/var/lib/ollama: both trees sit directly in the root,
        // beside the service user's `.ollama` and Ollama's `metadata`.
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        fs::create_dir_all(root.join(".ollama/cache")).unwrap();
        fs::create_dir_all(root.join("metadata")).unwrap();
        fs::write(root.join("metadata/sha256-0.json"), b"{}").unwrap();
        write_v1(
            root,
            "registry.ollama.ai",
            "library/llama3.2/1b",
            b"llama weights",
        );
        let llama = manifest(root, b"llama weights", 13);
        write_v2(root, "ollama.com", "library/llama3.2/1b", &llama);
        write_v1(
            root,
            "hf.co",
            "unsloth/Qwen3.5-0.8B-GGUF/Q4_K_M",
            b"hub weights",
        );

        let cache = ModelCache::new(&temporary.path().join("cache"));
        let mut output = LocationScan::default();
        scan_ollama(root, &cache, &mut output).unwrap();
        let names = output
            .candidates
            .iter()
            .filter_map(
                |DiscoveryCandidate::Artifact(artifact)| match &artifact.location {
                    ModelLocation::External {
                        origin: ExternalOrigin::Ollama { name, .. },
                        ..
                    } => Some(name.clone()),
                    _ => None,
                },
            )
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            ["hf.co/unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M", "llama3.2:1b"]
        );
    }

    /// Ollama 0.40 on Apple silicon pulls a library model's MLX manifest: one safetensors blob
    /// per tensor and no GGUF. Both trees resolve to that one pull, a store with only
    /// `manifests-v2` too, and it is found with a format the engine can't read, never dropped.
    #[test]
    fn ollama_finds_a_model_stored_as_mlx_tensors_as_an_unsupported_format() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let total = write_ollama_mlx_pull(root, "qwen3.5", "4b", "qwen");
        let llama = manifest(root, b"llama weights", 13);
        write_v2(root, "ollama.com", "library/llama3.2/1b", &llama);

        let scan = |root: &Path| {
            let mut output = LocationScan::default();
            scan_ollama(
                root,
                &ModelCache::new(&temporary.path().join("cache")),
                &mut output,
            )
            .unwrap();
            output
                .candidates
                .into_iter()
                .map(|DiscoveryCandidate::Artifact(artifact)| *artifact)
                .collect::<Vec<_>>()
        };
        for pass in ["both trees", "manifests-v2 only"] {
            let found = scan(root);
            let names = found
                .iter()
                .map(|artifact| match &artifact.location {
                    ModelLocation::External {
                        origin: ExternalOrigin::Ollama { name, .. },
                        ..
                    } => name.as_str(),
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            assert_eq!(names, ["llama3.2:1b", "qwen3.5:4b"], "{pass}");
            assert_eq!(found[0].unsupported_format, None);
            let qwen = &found[1];
            assert_eq!(
                qwen.unsupported_format.as_deref(),
                Some(
                    "Ollama stores qwen3.5:4b as 4 safetensors tensor files for its MLX runner, \
                     not as a GGUF"
                ),
                "{pass}"
            );
            let ModelLocation::External {
                components,
                total_bytes,
                ..
            } = &qwen.location
            else {
                unreachable!()
            };
            assert_eq!(*total_bytes, total);
            assert!(components.iter().all(|component| {
                component.role == ComponentRole::Weights && component.path.starts_with("blobs")
            }));
            fs::remove_dir_all(root.join("manifests")).ok();
        }
        assert!(!root.join("manifests").exists());
    }

    /// A tensor blob still being pulled leaves the model unlisted until it completes, as a GGUF
    /// blob does.
    #[test]
    fn ollama_skips_an_mlx_pull_with_a_missing_tensor_blob() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        write_ollama_mlx_pull(root, "qwen3.5", "4b", "qwen");
        let pulls = ollama_pulls(root, &mut UnreadableDirectories::new());
        let layer = pulls[0]
            .manifest
            .layers
            .iter()
            .find(|layer| layer.media_type == OLLAMA_TENSOR_MEDIA_TYPE)
            .unwrap();
        let blob = root
            .join("blobs")
            .join(format!("sha256-{}", sha256_hex(&layer.digest).unwrap()));
        fs::rename(&blob, blob.with_extension("partial")).unwrap();

        let mut output = LocationScan::default();
        scan_ollama(
            root,
            &ModelCache::new(&temporary.path().join("cache")),
            &mut output,
        )
        .unwrap();
        assert!(output.candidates.is_empty());
    }

    fn unreadable(scan: &LocationScan) -> Vec<PathBuf> {
        scan.unreadable.keys().cloned().collect()
    }

    fn scan(kind: ModelRootKind, path: &Path, cache: &ModelCache) -> LocationScan {
        scan_root(
            &ModelRoot {
                kind,
                path: path.to_path_buf(),
                enabled: true,
            },
            &DeclaredProjectors::new(&LocalModelsConfiguration::default()),
            cache,
            &RecognizedArchitectures::default(),
        )
        .unwrap()
    }

    fn primaries(scan: &LocationScan) -> Vec<PathBuf> {
        scan.candidates
            .iter()
            .map(|DiscoveryCandidate::Artifact(artifact)| artifact.primary.clone())
            .collect()
    }

    /// A subfolder Magnitude can't read, empty (a disk's `lost+found`) or holding a GGUF, is
    /// recorded and skipped; every other model of the location is still found.
    #[test]
    fn an_unreadable_subfolder_never_hides_the_rest_of_its_location() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = ModelCache::new(&temporary.path().join("cache"));
        let folder = temporary.path().join("models");
        crate::test_gguf::write_model(&folder.join("nested/model.gguf"), "nested");
        crate::test_gguf::write_model(&folder.join("top.gguf"), "top");
        crate::test_gguf::write_model(&folder.join("locked/hidden.gguf"), "hidden");
        let Some(_locked) = LockedDirectory::lock(&folder.join("locked")) else {
            return;
        };
        let Some(_empty) = LockedDirectory::lock(&folder.join("empty-locked")) else {
            return;
        };

        for kind in [
            ModelRootKind::Folder,
            ModelRootKind::LmStudio,
            ModelRootKind::LlamaCpp,
        ] {
            let found = scan(kind, &folder, &cache);
            assert_eq!(
                primaries(&found),
                [folder.join("nested/model.gguf"), folder.join("top.gguf")],
                "{kind:?}"
            );
            assert_eq!(
                unreadable(&found),
                [folder.join("empty-locked"), folder.join("locked")],
                "{kind:?}"
            );
        }
    }

    /// A location that can't be read at all fails its scan, and is reported by its own state
    /// rather than as an unreadable directory inside it.
    #[test]
    fn an_unreadable_location_fails_its_scan() {
        let temporary = tempfile::tempdir().unwrap();
        let folder = temporary.path().join("models");
        let Some(_locked) = LockedDirectory::lock(&folder) else {
            return;
        };
        assert!(
            scan_root(
                &ModelRoot {
                    kind: ModelRootKind::Folder,
                    path: folder.clone(),
                    enabled: true,
                },
                &DeclaredProjectors::new(&LocalModelsConfiguration::default()),
                &ModelCache::new(&temporary.path().join("cache")),
                &RecognizedArchitectures::default(),
            )
            .is_err()
        );
    }

    #[test]
    fn an_unreadable_hub_repository_never_hides_the_rest_of_the_cache() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = ModelCache::new(&temporary.path().join("cache"));
        let hub = temporary.path().join("hub");
        let model = hub.join("models--owner--good/snapshots/abc/model.gguf");
        crate::test_gguf::write_model(&model, "good");
        crate::test_gguf::write_model(
            &hub.join("models--owner--locked/snapshots/def/model.gguf"),
            "locked",
        );
        let Some(_locked) = LockedDirectory::lock(&hub.join("models--owner--locked")) else {
            return;
        };

        let found = scan(ModelRootKind::HuggingFaceCache, &hub, &cache);

        assert_eq!(primaries(&found), [model]);
        assert_eq!(
            unreadable(&found),
            [hub.join("models--owner--locked/snapshots")]
        );
    }

    #[test]
    fn an_unreadable_ollama_namespace_never_hides_the_other_pulls() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        write_v1(root, OLLAMA_REGISTRY, "library/qwen3.5/0.8b", b"qwen weights");
        write_v1(root, OLLAMA_REGISTRY, "someone/tool/latest", b"tool weights");
        let namespace = root.join("manifests").join(OLLAMA_REGISTRY).join("someone");
        let Some(_locked) = LockedDirectory::lock(&namespace) else {
            return;
        };

        let mut unreadable = UnreadableDirectories::new();
        let names = ollama_pulls(root, &mut unreadable)
            .into_iter()
            .map(|pull| pull.name)
            .collect::<Vec<_>>();

        assert_eq!(names, ["qwen3.5:0.8b"]);
        assert_eq!(unreadable.into_keys().collect::<Vec<_>>(), [namespace]);
    }
}
