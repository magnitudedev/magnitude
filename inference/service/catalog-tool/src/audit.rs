//! The tensor-type audit of the locked release catalog: every catalog component's files, as the
//! catalog resolves them at their locked revisions, with the tensor encodings each file's GGUF
//! header declares. It reads the header export (`headers`), so the files audited are exactly the
//! ones the catalog loads.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::Context;
use magnitude_artifacts::catalog_headers::{self, ComponentRole, HeaderIndex};
use magnitude_artifacts::gguf::{Directory, inspect_header, split_paths};
use magnitude_service_models::{catalog_lock_digest, catalog_source_digest};
use serde_json::json;

/// Encodings the engine's GGUF reader accepted when the catalog was first audited (2026-09-21,
/// dependency row D11 of the blueprint); the audit reports the catalog's types outside them.
const D11_ENGINE_ENCODINGS: [&str; 7] = ["F32", "F16", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS"];

/// One audited file at its locked source.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct File {
    repository: String,
    revision: String,
    path: PathBuf,
}

/// The tensor encodings one file's header declares.
struct Facts {
    version: u32,
    /// Tensor count by type id, with each id's ggml name (`UNKNOWN_<id>` for an id the parser
    /// doesn't know).
    types: BTreeMap<u32, (String, u64)>,
}

fn facts(path: &Path) -> anyhow::Result<(Directory, Facts)> {
    let (directory, unknown) = match inspect_header(path) {
        Err(magnitude_artifacts::Error::UnknownTensorTypes { types, directory }) => {
            (*directory, types)
        }
        parsed => (
            parsed.with_context(|| format!("failed to read {}", path.display()))?,
            Vec::new(),
        ),
    };
    let mut types = BTreeMap::<u32, (String, u64)>::new();
    for tensor in &directory.tensors {
        types
            .entry(tensor.encoding as u32)
            .or_insert_with(|| (tensor.encoding.ggml_name().to_owned(), 0))
            .1 += 1;
    }
    for (id, count) in unknown {
        types.insert(id, (format!("UNKNOWN_{id}"), u64::from(count)));
    }
    let version = directory.version;
    Ok((directory, Facts { version, types }))
}

fn role_name(role: &ComponentRole) -> (&str, &str) {
    match role {
        ComponentRole::Target { variant } => ("target", variant),
        ComponentRole::Draft => ("draft", "all"),
        ComponentRole::Projector => ("projector", "all"),
    }
}

/// Writes the audit of the header export `exported` to `output`, and fails when a file declares
/// a tensor type the parser doesn't know.
pub fn run(exported: &Path, output: &Path) -> anyhow::Result<()> {
    let index: HeaderIndex = serde_json::from_slice(
        &std::fs::read(exported.join(catalog_headers::INDEX)).context("no header export index")?,
    )?;
    // Each file's exported header and its catalog uses.
    let mut uses = BTreeMap::<File, (PathBuf, Vec<String>)>::new();
    let mut resolutions = Vec::new();
    for component in &index.components {
        let (role, variant) = role_name(&component.role);
        let use_name = match &component.role {
            ComponentRole::Target { variant } => format!("{}/{variant}:target", component.model),
            ComponentRole::Draft | ComponentRole::Projector => {
                format!("{}:{role}", component.model)
            }
        };
        let first = component.file(exported);
        let (directory, _) = facts(&first)?;
        let shards = split_paths(&first, &directory)?.unwrap_or_else(|| vec![first.clone()]);
        for shard in shards {
            let name = shard.file_name().expect("a shard file name");
            uses.entry(File {
                repository: component.repository.clone(),
                revision: component.revision.clone(),
                path: component.path.with_file_name(name),
            })
            .or_insert_with(|| (shard.clone(), Vec::new()))
            .1
            .push(use_name.clone());
        }
        resolutions.push(json!({
            "modelId": component.model,
            "variantId": variant,
            "role": role,
            "repository": component.repository,
            "revision": component.revision,
            "primary": component.path,
        }));
    }
    let mut observed = BTreeSet::new();
    let mut artifacts = Vec::new();
    for (file, (exported_file, mut file_uses)) in uses {
        let (_, facts) = facts(&exported_file)?;
        file_uses.sort();
        observed.extend(facts.types.values().map(|(name, _)| name.clone()));
        let tensor_types = facts
            .types
            .iter()
            .map(|(id, (name, count))| json!({ "id": id, "name": name, "tensorCount": count }))
            .collect::<Vec<_>>();
        artifacts.push(json!({
            "repository": file.repository,
            "revision": file.revision,
            "path": file.path,
            "uses": file_uses,
            "version": facts.version,
            "tensorCount": facts.types.values().map(|(_, count)| count).sum::<u64>(),
            "tensorTypes": tensor_types,
        }));
    }
    let unknown = observed
        .iter()
        .filter(|name| name.starts_with("UNKNOWN_"))
        .cloned()
        .collect::<Vec<_>>();
    let missing = observed
        .iter()
        .filter(|name| !D11_ENGINE_ENCODINGS.contains(&name.as_str()))
        .collect::<Vec<_>>();
    let scope = json!({
        "catalog": "inference/catalog/models.json",
        "lock": "inference/catalog/models.lock.json",
        "catalogSha256": catalog_source_digest(),
        "lockSha256": catalog_lock_digest(),
    });
    let mut baseline = D11_ENGINE_ENCODINGS.to_vec();
    baseline.sort_unstable();
    let summary = json!({
        "scope": scope,
        "observedTensorTypes": observed,
        "d11MissingTensorTypes": missing,
    });
    let report = json!({
        "schemaVersion": 2,
        "scope": scope,
        "method": "the release catalog's resolution of the locked revisions and the GGUF headers it exports",
        "engineEncodingBaseline": baseline,
        "observedTensorTypes": observed,
        "d11MissingTensorTypes": missing,
        "unknownTensorTypes": unknown,
        "resolutions": resolutions,
        "artifacts": artifacts,
    });
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    anyhow::ensure!(
        unknown.is_empty(),
        "unknown tensor types {}",
        unknown.join(", ")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use magnitude_artifacts::gguf::Encoding;

    use super::*;

    #[test]
    fn the_baseline_names_engine_encodings() {
        for name in D11_ENGINE_ENCODINGS {
            assert!(
                Encoding::ALL
                    .iter()
                    .any(|encoding| encoding.ggml_name() == name),
                "{name}"
            );
        }
    }
}
