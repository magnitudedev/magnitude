//! Every catalog component's definition, pinned. For each catalog target
//! (`headers::index`; all shards of a split target) the family that
//! recognizes it builds its `ModelDefinition` (with the model's projector,
//! when the catalog has one), and the `dflash` family builds the model's
//! draft against it. The serialized definitions' SHA-256 digests
//! must equal `fixtures/catalog-definitions.json`, first recorded from the
//! family code before it moved onto this crate. A change that means to alter
//! a definition re-records the fixture in the same change.
//!
//! `CATALOG_DEFINITIONS=record` rewrites the fixture;
//! `CATALOG_DEFINITIONS_DUMP=<dir>` also writes each serialized definition.

use magnitude_artifacts::{catalog_headers::ComponentRole, ArtifactIdentity, PackageIdentity};
use magnitude_family_common::headers;
use magnitude_family_contracts::ModelFamily;
use magnitude_family_gemma4::Gemma4Family;
use magnitude_family_lfm2::Lfm2Family;
use magnitude_family_llama::LlamaFamily;
use magnitude_family_muse_glimmer::MuseGlimmerFamily;
use magnitude_family_nemotron_h::NemotronHFamily;
use magnitude_family_qwen4exp::Qwen4ExpFamily;
use magnitude_family_qwen35::Qwen35Family;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf};

const FAMILIES: [&dyn ModelFamily; 7] = [
    &Qwen35Family,
    &Qwen4ExpFamily,
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
    let component = |model: &str, role: ComponentRole| {
        index
            .iter()
            .find(|component| component.model == model && component.role == role)
    };
    let mut outcomes = BTreeMap::new();
    for target in index {
        let ComponentRole::Target { .. } = target.role else {
            continue;
        };
        let name = headers::key(target);
        let directory = headers::component(target);
        let Some(family) = FAMILIES.iter().find(|family| family.recognizes(&directory)) else {
            outcomes.insert(name, "unrecognized".into());
            continue;
        };
        let projector = component(&target.model, ComponentRole::Projector).map(headers::component);
        let built = family.inspect(&directory, projector.as_ref(), identity());
        outcomes.insert(name.clone(), outcome(&name, &built));
        let Ok(definition) = built else { continue };
        if let Some(draft) = component(&target.model, ComponentRole::Draft) {
            let key = format!("{}@{name}", headers::key(draft));
            let built = magnitude_family_dflash::inspect(
                &headers::component(draft),
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
