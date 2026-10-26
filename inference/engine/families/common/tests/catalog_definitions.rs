//! Every catalog component's definition, pinned. For each catalog variant's
//! target (`headers::index`; all shards of a split target) the family that
//! recognizes it builds its `ModelDefinition` (with the variant's projector,
//! when it has one), and the `dflash` family builds the variant's draft
//! against it. The serialized definitions' SHA-256 digests, by catalog model
//! ID, must equal `fixtures/catalog-definitions.json`, first recorded from the
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
use serde::{Deserialize, Serialize};
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

/// A catalog variant's pinned outcomes: its target's definition and, when it
/// has a separate draft, the draft's against it.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Outcomes {
    target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    draft: Option<String>,
}

fn outcomes() -> BTreeMap<String, Outcomes> {
    let index = headers::index();
    let mut outcomes = BTreeMap::new();
    for target in index.iter().filter(|component| component.role == ComponentRole::Target) {
        let model = target.model.as_str();
        let component = |role| {
            index
                .iter()
                .find(|component| component.model == model && component.role == role)
        };
        let directory = headers::component(target);
        let Some(family) = FAMILIES.iter().find(|family| family.recognizes(&directory)) else {
            outcomes.insert(
                model.to_owned(),
                Outcomes {
                    target: "unrecognized".into(),
                    draft: None,
                },
            );
            continue;
        };
        let projector = component(ComponentRole::Projector).map(headers::component);
        let built = family.inspect(&directory, projector.as_ref(), identity());
        let draft = match (&built, component(ComponentRole::Draft)) {
            (Ok(definition), Some(draft)) => {
                let built = magnitude_family_dflash::inspect(
                    &headers::component(draft),
                    definition,
                    &|layer| family.layer_entry(definition, layer),
                );
                Some(outcome(&format!("{model} draft"), &built))
            }
            _ => None,
        };
        outcomes.insert(
            model.to_owned(),
            Outcomes {
                target: outcome(&format!("{model} target"), &built),
                draft,
            },
        );
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
    let pinned: BTreeMap<String, Outcomes> =
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
