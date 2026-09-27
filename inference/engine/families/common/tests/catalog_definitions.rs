//! Every catalog component's definition, pinned. For each target header in
//! `inference/validation/results/catalog-headers/` (all shards of a split
//! target) the family that
//! recognizes it builds its `ModelDefinition` (with the model's projector,
//! when the catalog has one), and the `dflash` family builds each of the
//! model's drafts against it. The serialized definitions' SHA-256 digests
//! must equal `fixtures/catalog-definitions.json`, first recorded from the
//! family code before it moved onto this crate. A change that means to alter
//! a definition re-records the fixture in the same change.
//!
//! `CATALOG_DEFINITIONS=record` rewrites the fixture;
//! `CATALOG_DEFINITIONS_DUMP=<dir>` also writes each serialized definition.

use magnitude_artifacts::{ArtifactIdentity, PackageIdentity};
use magnitude_family_common::headers;
use magnitude_family_contracts::ModelFamily;
use magnitude_family_gemma4::Gemma4Family;
use magnitude_family_lfm2::Lfm2Family;
use magnitude_family_llama::LlamaFamily;
use magnitude_family_muse_glimmer::MuseGlimmerFamily;
use magnitude_family_nemotron_h::NemotronHFamily;
use magnitude_family_qwen35::Qwen35Family;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf};

const FAMILIES: [&dyn ModelFamily; 6] = [
    &Qwen35Family,
    &LlamaFamily,
    &NemotronHFamily,
    &Lfm2Family,
    &Gemma4Family,
    &MuseGlimmerFamily,
];

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/catalog-definitions.json")
}

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    }
}

/// A built definition's digest, or `rejected`.
fn outcome<T: Serialize, E>(key: &str, built: &Result<T, E>) -> String {
    let Ok(definition) = built else {
        return "rejected".into();
    };
    let serialized = serde_json::to_vec(definition).unwrap();
    if let Some(dump) = std::env::var_os("CATALOG_DEFINITIONS_DUMP") {
        std::fs::write(PathBuf::from(dump).join(format!("{key}.json")), &serialized).unwrap();
    }
    Sha256::digest(&serialized)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn outcomes() -> BTreeMap<String, String> {
    let index = headers::index();
    let component = |model: &str, role: &str| -> Vec<&String> {
        index
            .iter()
            .filter(|(m, r, _)| m == model && r == role)
            .map(|(_, _, file)| file)
            .collect()
    };
    let mut outcomes = BTreeMap::new();
    for (model, role, file) in &index {
        if !role.starts_with("target") {
            continue;
        }
        // A split target's full tensor directory is dumped beside its first
        // shard's.
        let all_shards = file.replace(".json", "-allshards.json");
        let file = if headers::root().join(&all_shards).exists() {
            &all_shards
        } else {
            file
        };
        let target = headers::directory(file);
        let Some(family) = FAMILIES.iter().find(|family| family.recognizes(&target)) else {
            outcomes.insert(file.clone(), "unrecognized".into());
            continue;
        };
        let projector = component(model, "projector")
            .first()
            .map(|file| headers::directory(file));
        let built = family.inspect(&target, projector.as_ref(), identity());
        outcomes.insert(file.clone(), outcome(file, &built));
        let Ok(definition) = built else { continue };
        for draft in component(model, "draft") {
            let key = format!("{draft}@{file}");
            let built = magnitude_family_dflash::inspect(
                &headers::directory(draft),
                &definition,
                &|layer| family.layer_entry(&definition, layer),
            );
            outcomes.insert(key.clone(), outcome(&key, &built));
        }
    }
    outcomes
}

#[test]
fn every_catalog_definition_is_pinned() {
    let outcomes = outcomes();
    if std::env::var("CATALOG_DEFINITIONS").as_deref() == Ok("record") {
        let mut text = serde_json::to_string_pretty(&outcomes).unwrap();
        text.push('\n');
        std::fs::write(fixture(), text).unwrap();
        return;
    }
    let pinned: BTreeMap<String, String> =
        serde_json::from_slice(&std::fs::read(fixture()).unwrap()).unwrap();
    let differing = pinned
        .keys()
        .chain(outcomes.keys())
        .filter(|key| pinned.get(*key) != outcomes.get(*key))
        .collect::<std::collections::BTreeSet<_>>();
    assert!(
        differing.is_empty(),
        "definitions differ from the pinned ones: {differing:#?}"
    );
}
