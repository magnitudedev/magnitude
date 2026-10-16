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

use crate::cache::ModelCache;
use crate::identity::{content_id, inventory_entry_id};
use crate::inventory::{
    ArtifactCandidate, DeclaredProjectors, DiscoveryCandidate, ModelGroup, canonical_or_self,
    components_for_group, discover_groups, io_error, modified_seconds, now, primary_path,
    read_dir_sorted, scan_hf_cache,
};

const OLLAMA_MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";
const OLLAMA_PROJECTOR_MEDIA_TYPE: &str = "application/vnd.ollama.image.projector";
const OLLAMA_REGISTRY: &str = "registry.ollama.ai";
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
        ModelRootKind::Ollama => scan_ollama(&root.path, output),
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

/// Ollama names models in `manifests/<registry>/<namespace>/<model>/<tag>`; each manifest's model
/// layer is a content-addressed blob in `blobs/sha256-<hex>`, whose name is its full SHA-256.
fn scan_ollama(root: &Path, output: &mut Vec<DiscoveryCandidate>) -> Result<(), InventoryError> {
    let manifests = root.join("manifests");
    for registry in read_dir_sorted(&manifests)? {
        let registry_name = registry.file_name().to_string_lossy().into_owned();
        for namespace in read_dir_sorted(&registry.path())? {
            let namespace_name = namespace.file_name().to_string_lossy().into_owned();
            for model in read_dir_sorted(&namespace.path())? {
                let model_name = model.file_name().to_string_lossy().into_owned();
                for tag in read_dir_sorted(&model.path())? {
                    if !tag.file_type().map_err(io_error)?.is_file() {
                        continue;
                    }
                    let tag_name = tag.file_name().to_string_lossy().into_owned();
                    let Some(manifest) = read_bounded_json(&tag.path(), MAX_OLLAMA_MANIFEST_BYTES)
                        .and_then(|document| {
                            serde_json::from_value::<OllamaManifest>(document).ok()
                        })
                    else {
                        continue;
                    };
                    let (name, hub) =
                        ollama_identity(&registry_name, &namespace_name, &model_name, &tag_name);
                    if let Some(components) = ollama_components(root, &manifest) {
                        push_external(
                            root,
                            "ollama",
                            ExternalOrigin::Ollama { name, hub },
                            LocalDeclaration::Discovery,
                            components,
                            output,
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// The name Ollama shows and, for a `hf.co/<owner>/<repository>:<tag>` pull, the Hub identity.
/// Ollama records only the tag, so the selector is `<tag>.gguf`.
fn ollama_identity(
    registry: &str,
    namespace: &str,
    model: &str,
    tag: &str,
) -> (String, Option<HubFile>) {
    let name = match (registry, namespace) {
        (OLLAMA_REGISTRY, "library") => format!("{model}:{tag}"),
        (OLLAMA_REGISTRY, _) => format!("{namespace}/{model}:{tag}"),
        _ => format!("{registry}/{namespace}/{model}:{tag}"),
    };
    let hub = matches!(registry, "hf.co" | "huggingface.co")
        .then(|| hub_file(format!("{namespace}/{model}"), format!("{tag}.gguf")))
        .flatten();
    (name, hub)
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
