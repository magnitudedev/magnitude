//! The release bundle's GGUF headers as the JSON dumps engine family tests read
//! (`magnitude_family_common::headers`): each component's metadata and tensor
//! directory, with long metadata arrays reduced to their head, plus an index.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, bail, ensure};
use magnitude_artifacts::gguf::{self, Array, Directory, Scalar, TensorDescriptor, Value};
use magnitude_service_contracts::ComponentRole;
use magnitude_service_models::{ReleasePackageRole, load_release_catalog};
use serde_json::{Map, Value as Json, json};

/// Arrays longer than this (tokenizer vocabularies and merges) keep only
/// their first `TRUNCATED_HEAD` entries; per-layer arrays stay whole.
const MAX_WHOLE_ARRAY: usize = 1024;
const TRUNCATED_HEAD: usize = 64;

struct Dump {
    model: String,
    role: String,
    header: Json,
}

/// Writes every component header of the bundle at `bundle` into `output`,
/// replacing its contents, and returns the number of indexed components.
pub fn dump(bundle: &Path, output: &Path) -> anyhow::Result<usize> {
    let catalog = load_release_catalog(bundle)
        .with_context(|| format!("failed to load planner bundle {}", bundle.display()))?;
    let mut dumps = BTreeMap::<String, Dump>::new();
    let mut targets = BTreeMap::<(String, String), Vec<Directory>>::new();
    for item in catalog.component_headers() {
        let item = item?;
        let model = item.model_id.as_str().to_owned();
        let directory = gguf::read_header(&item.header).with_context(|| {
            format!(
                "{model} {}: invalid GGUF header",
                item.component.path.display()
            )
        })?;
        match (item.package, &item.component.role) {
            (ReleasePackageRole::Target, ComponentRole::Weights | ComponentRole::Shard) => {
                targets
                    .entry((model, item.variant_id.as_str().replace(':', "_")))
                    .or_default()
                    .push(directory);
            }
            (ReleasePackageRole::Target, ComponentRole::Projector) => {
                insert_shared(&mut dumps, &model, "projector", &directory)?;
            }
            (ReleasePackageRole::Draft, ComponentRole::Weights) => {
                insert_shared(&mut dumps, &model, "draft", &directory)?;
            }
            (package, role) => bail!(
                "{model} {}: no dump role for a {role:?} component of the {package:?} package",
                item.component.path.display()
            ),
        }
    }
    let mut all_shards = BTreeMap::new();
    for ((model, variant), mut shards) in targets {
        shards.sort_by_key(|shard| shard.value("split.no").and_then(Value::unsigned));
        let role = format!("target-{variant}");
        let file = format!("{model}__{role}.json");
        if shards.len() > 1 {
            let tensors = shards.iter().flat_map(|shard| &shard.tensors);
            all_shards.insert(
                file.replace(".json", "-allshards.json"),
                header_json(&shards[0], tensors)?,
            );
        }
        dumps.insert(
            file,
            Dump {
                model,
                role,
                header: header_json(&shards[0], &shards[0].tensors)?,
            },
        );
    }

    let staging = output.with_extension("tmp");
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    std::fs::create_dir_all(&staging)?;
    let index = dumps
        .iter()
        .map(|(file, dump)| json!({ "model": dump.model, "role": dump.role, "file": file }))
        .collect::<Vec<_>>();
    let files = dumps
        .iter()
        .map(|(file, dump)| (file, &dump.header))
        .chain(all_shards.iter());
    for (file, header) in files {
        write_json(&staging.join(file), header)?;
    }
    write_json(&staging.join("index.json"), &Json::Array(index))?;
    if output.exists() {
        std::fs::remove_dir_all(output)?;
    }
    std::fs::rename(&staging, output)?;
    Ok(dumps.len())
}

/// A per-model component (projector or draft) every variant shares: each
/// variant must bundle the same header.
fn insert_shared(
    dumps: &mut BTreeMap<String, Dump>,
    model: &str,
    role: &str,
    directory: &Directory,
) -> anyhow::Result<()> {
    let file = format!("{model}__{role}.json");
    let header = header_json(directory, &directory.tensors)?;
    if let Some(existing) = dumps.get(&file) {
        ensure!(
            existing.header == header,
            "{model} bundles different {role} headers across its variants"
        );
        return Ok(());
    }
    dumps.insert(
        file,
        Dump {
            model: model.to_owned(),
            role: role.to_owned(),
            header,
        },
    );
    Ok(())
}

fn header_json<'a>(
    directory: &Directory,
    tensors: impl IntoIterator<Item = &'a TensorDescriptor>,
) -> anyhow::Result<Json> {
    let metadata = directory
        .metadata
        .iter()
        .map(|entry| {
            let value = value_json(&entry.value).with_context(|| entry.name.clone())?;
            Ok((entry.name.clone(), value))
        })
        .collect::<anyhow::Result<Map<_, _>>>()?;
    let tensors = tensors
        .into_iter()
        .map(|tensor| {
            let mut shape = tensor.shape.clone();
            shape.reverse();
            json!({ "name": tensor.name, "type": tensor.encoding.ggml_name(), "shape": shape })
        })
        .collect::<Vec<_>>();
    Ok(json!({ "metadata": metadata, "tensors": tensors }))
}

fn value_json(value: &Value) -> anyhow::Result<Json> {
    let array = |values: &Array, length: usize| {
        values
            .iter()
            .take(length)
            .map(|value| scalar_json(&value.to_scalar()))
            .collect::<anyhow::Result<Vec<_>>>()
    };
    Ok(match value {
        Value::Scalar(scalar) => scalar_json(scalar)?,
        Value::Array(values) if values.len() > MAX_WHOLE_ARRAY => json!({
            "head": array(values, TRUNCATED_HEAD)?,
            "length": values.len(),
        }),
        Value::Array(values) => Json::Array(array(values, values.len())?),
    })
}

fn scalar_json(scalar: &Scalar) -> anyhow::Result<Json> {
    Ok(match scalar {
        Scalar::String(value) => Json::String(value.clone()),
        Scalar::Bool(value) => Json::Bool(*value),
        Scalar::Unsigned(value) => json!(value),
        Scalar::Signed(value) => json!(value),
        Scalar::Float(value) => Json::Number(
            serde_json::Number::from_f64(*value)
                .with_context(|| format!("non-finite metadata value {value} has no JSON form"))?,
        ),
    })
}

fn write_json(path: &Path, value: &Json) -> anyhow::Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    std::fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_arrays_keep_their_head_and_length() {
        let long = Value::Array((0..2000).collect::<Vec<u64>>().into());
        let dumped = value_json(&long).unwrap();
        assert_eq!(dumped["length"], 2000);
        assert_eq!(dumped["head"].as_array().unwrap().len(), TRUNCATED_HEAD);
        let short = Value::Array((0..MAX_WHOLE_ARRAY as u64).collect::<Vec<_>>().into());
        assert_eq!(
            value_json(&short).unwrap().as_array().unwrap().len(),
            MAX_WHOLE_ARRAY
        );
    }

    #[test]
    fn floats_stay_floats() {
        let one = scalar_json(&Scalar::Float(1.0)).unwrap();
        assert!(one.as_u64().is_none());
        assert_eq!(one.as_f64(), Some(1.0));
        assert!(scalar_json(&Scalar::Float(f64::NAN)).is_err());
    }
}
