//! Scanners for the read-only model locations ACN configures, and for files declarations name in
//! place. Each scanner reads directory layouts, manifests and GGUF headers only; it never writes,
//! moves or hashes a file in full, and partial downloads never form a group.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use magnitude_service_contracts::models::{
    CustomModelDeclarationOrigin, DrafterChoice, DrafterReference, HuggingFaceArtifactSelector,
    HuggingFaceRepositoryId, LocalModelsConfiguration, ModelRoot, ModelRootKind,
};
use magnitude_service_contracts::{
    ComponentRelationship, ComponentRole, ContentIdentity, ExternalOrigin, HubFile, Integrity,
    InventoryError, LocalDeclaration, ModelComponent, ModelLocation, ModelSource,
};

use crate::cache::{ModelCache, ModelIndexKind};
use crate::identity::{content_id, inventory_entry_id};
use crate::inventory::{
    ArtifactCandidate, DeclaredProjectors, DiscoveryCandidate, ModelGroup, canonical_or_self,
    components_for_group, discover_groups, modified_seconds, now, primary_path, read_dir_sorted,
    scan_hf_cache,
};

const OLLAMA_MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";
const OLLAMA_PROJECTOR_MEDIA_TYPE: &str = "application/vnd.ollama.image.projector";
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

pub(crate) fn scan_root(
    root: &ModelRoot,
    declared: &DeclaredProjectors,
    cache: &ModelCache,
    output: &mut Vec<DiscoveryCandidate>,
) -> Result<(), InventoryError> {
    if !root.path.is_dir() {
        return Ok(());
    }
    match root.kind {
        ModelRootKind::HuggingFaceCache => scan_hf_cache(&root.path, declared, cache, output),
        ModelRootKind::LmStudio => scan_lm_studio(&root.path, cache, output),
        ModelRootKind::LlamaCpp => scan_llama_cpp(&root.path, cache, output),
        ModelRootKind::Ollama => scan_ollama(&root.path, cache, output),
        ModelRootKind::Folder => scan_folder(&root.path, cache, output),
    }
}

fn push_external(
    root: &Path,
    kind: &str,
    origin: ExternalOrigin,
    declared_by: LocalDeclaration,
    components: Vec<ModelComponent>,
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
    output: &mut Vec<DiscoveryCandidate>,
) -> Result<(), InventoryError> {
    for group in discover_groups(root, &canonical_or_self(root), cache, &BTreeMap::new())? {
        let hub = lm_studio_hub_file(root, &group.paths[0]);
        push_group(
            root,
            "lm-studio",
            ExternalOrigin::LmStudio { hub },
            &group,
            output,
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

/// llama.cpp's `-hf` cache stores files flat as `<owner>_<repository>_<file>` beside
/// `manifest=<owner>=<repository>=<tag>.json`, which names the exact Hub file and its projector.
fn scan_llama_cpp(
    root: &Path,
    cache: &ModelCache,
    output: &mut Vec<DiscoveryCandidate>,
) -> Result<(), InventoryError> {
    let manifests = llama_cpp_manifests(root)?;
    let declared = manifests
        .iter()
        .filter_map(|(name, manifest)| {
            let projector = manifest.projector.as_ref()?;
            Some((root.join(name), root.join(projector)))
        })
        .collect::<BTreeMap<_, _>>();
    for group in discover_groups(root, &canonical_or_self(root), cache, &declared)? {
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
            output,
        )?;
    }
    Ok(())
}

fn llama_cpp_local_name(owner: &str, repository: &str, remote: &str) -> String {
    format!("{owner}_{repository}_{}", remote.replace('/', "_"))
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
/// read only when it resolves inside the root. The same model in both trees is one pull, the
/// manifest whose blobs are all present when only one is complete.
fn ollama_pulls(root: &Path) -> Result<Vec<OllamaPull>, InventoryError> {
    let contained = canonical_or_self(root);
    let mut pulls = BTreeMap::<String, OllamaPull>::new();
    for tree in OLLAMA_MANIFEST_TREES {
        for registry in read_dir_sorted(&root.join(tree))? {
            let registry_name = ollama_registry(&registry.file_name().to_string_lossy());
            for namespace in read_dir_sorted(&registry.path())? {
                let namespace_name = namespace.file_name().to_string_lossy().into_owned();
                for model in read_dir_sorted(&namespace.path())? {
                    let model_name = model.file_name().to_string_lossy().into_owned();
                    for tag in read_dir_sorted(&model.path())? {
                        let Some(manifest) = fs::canonicalize(tag.path())
                            .ok()
                            .filter(|path| path.starts_with(&contained))
                            .and_then(|path| read_bounded_json(&path, MAX_OLLAMA_MANIFEST_BYTES))
                            .and_then(|document| {
                                serde_json::from_value::<OllamaManifest>(document).ok()
                            })
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
                            ollama_components(root, &existing.manifest).is_some()
                                || ollama_components(root, &pull.manifest).is_none()
                        });
                        if !keep_existing {
                            pulls.insert(name, pull);
                        }
                    }
                }
            }
        }
    }
    Ok(pulls.into_values().collect())
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
        .flat_map(|root| ollama_pulls(&root.path).unwrap_or_default())
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
    output: &mut Vec<DiscoveryCandidate>,
) -> Result<(), InventoryError> {
    for pull in ollama_pulls(root)? {
        let hub = pull.hub_repository().and_then(|repository| {
            let file = read_ollama_hub_file(cache, &repository, pull.model_sha256()?)?.file?;
            hub_file(repository, file)
        });
        if let Some(components) = ollama_components(root, &pull.manifest) {
            push_external(
                root,
                "ollama",
                ExternalOrigin::Ollama {
                    name: pull.name,
                    hub,
                },
                LocalDeclaration::Discovery,
                components,
                output,
            );
        }
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

fn ollama_blob(root: &Path, layer: &OllamaLayer) -> Option<(PathBuf, String)> {
    let hex = layer.digest.strip_prefix("sha256:")?;
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    let relative = PathBuf::from("blobs").join(format!("sha256-{hex}"));
    // A blob still being pulled has a different size, or exists only as `-partial` files.
    fs::metadata(root.join(&relative))
        .ok()
        .filter(|metadata| metadata.is_file() && metadata.len() == layer.size)?;
    Some((relative, hex.to_owned()))
}

fn ollama_components(root: &Path, manifest: &OllamaManifest) -> Option<Vec<ModelComponent>> {
    let layer = |media_type: &str| {
        manifest
            .layers
            .iter()
            .find(|layer| layer.media_type == media_type)
    };
    let model_layer = layer(OLLAMA_MODEL_MEDIA_TYPE)?;
    let (model_path, model_hex) = ollama_blob(root, model_layer)?;
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
    output: &mut Vec<DiscoveryCandidate>,
) -> Result<(), InventoryError> {
    for group in discover_groups(root, &canonical_or_self(root), cache, &BTreeMap::new())? {
        push_group(root, "folder", ExternalOrigin::Folder, &group, output)?;
    }
    Ok(())
}

/// Files declarations reference in place: imported models, and drafters attached by local path.
/// A declared file that is missing simply contributes nothing.
pub(crate) fn scan_declared_files(
    configuration: &LocalModelsConfiguration,
    output: &mut Vec<DiscoveryCandidate>,
) {
    let imported = configuration
        .declarations
        .iter()
        .filter(|declaration| {
            matches!(
                declaration.origin,
                CustomModelDeclarationOrigin::Imported { .. }
            )
        })
        .map(|declaration| ModelGroup {
            paths: declaration.target.iter().map(PathBuf::from).collect(),
            projector: declaration.projector.as_ref().map(PathBuf::from),
        });
    let drafters =
        configuration
            .drafters
            .iter()
            .filter_map(|attachment| match &attachment.drafter {
                DrafterChoice::Reference {
                    reference: DrafterReference::Local { path },
                    ..
                } => Some(ModelGroup {
                    paths: vec![path.clone()],
                    projector: None,
                }),
                _ => None,
            });
    for group in imported.chain(drafters) {
        if let Err(error) = push_declared(&group, output) {
            tracing::warn!(
                path = %group.paths[0].display(),
                %error,
                "declared model file could not be observed"
            );
        }
    }
}

fn push_declared(
    group: &ModelGroup,
    output: &mut Vec<DiscoveryCandidate>,
) -> Result<(), InventoryError> {
    let files = group.paths.iter().chain(group.projector.iter());
    if files
        .clone()
        .any(|path| !path.is_file() || is_partial_download(path))
    {
        return Ok(());
    }
    let Some(root) = common_directory(files) else {
        return Ok(());
    };
    let components = components_for_group(&root, group, None)?;
    push_external(
        &root,
        "imported",
        ExternalOrigin::Imported,
        LocalDeclaration::Configuration,
        components,
        output,
    );
    Ok(())
}

fn common_directory<'a>(mut files: impl Iterator<Item = &'a PathBuf>) -> Option<PathBuf> {
    let mut common = files.next()?.parent()?.to_path_buf();
    for file in files {
        while !file.starts_with(&common) {
            common = common.parent()?.to_path_buf();
        }
    }
    Some(common)
}

fn read_bounded_json(path: &Path, limit: u64) -> Option<serde_json::Value> {
    let metadata = fs::metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() > limit {
        return None;
    }
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::symlink;

    use sha2::{Digest, Sha256};

    use super::*;

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
        ollama_pulls(root)
            .unwrap()
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
        let pulls = ollama_pulls(root).unwrap();
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

        let pulls = ollama_pulls(root).unwrap();
        assert_eq!(pulls.len(), 1);
        assert_eq!(
            pulls[0].model_sha256(),
            Some(sha256(b"complete weights").as_str())
        );
    }

    #[test]
    fn ollama_ignores_a_v2_manifest_resolving_outside_the_root() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("ollama");
        let outside = temporary.path().join("outside.json");
        fs::write(&outside, manifest(&root, b"elsewhere", 9)).unwrap();
        symlink(
            &outside,
            manifest_path(&root, "manifests-v2", "ollama.com", "library/escape/latest"),
        )
        .unwrap();

        assert!(names(&root).is_empty());
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
        let mut output = Vec::new();
        scan_ollama(root, &cache, &mut output).unwrap();
        let names = output
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
}
