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
//! `CATALOG_DEFINITIONS_DUMP=<dir>` also writes each serialized definition; a rejection's refusal is printed
//! to stderr (`--nocapture`).
//!
//! Every definition the families build from a real header, these and those
//! of each pinned artifact and each assessment header fixture, also crosses
//! the engine's worker wire unchanged: postcard, which reads fields by
//! position alone (`HostMessage::Load` and `Prepare` carry the definition in
//! their `ExecutionManifest`).

use magnitude_artifacts::{
    catalog_headers::{self, ComponentRole},
    gguf::{self, Directory},
    ArtifactIdentity, PackageIdentity,
};
use magnitude_family_common::headers;
use magnitude_family_contracts::{DraftDefinition, FamilyError, ModelDefinition};
use magnitude_family_registry::FAMILIES;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    io::Read,
    path::PathBuf,
};

/// This package's directory, read at run time, as cargo sets it for a test.
fn package() -> PathBuf {
    PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"))
}

fn fixture() -> PathBuf {
    package().join("tests/fixtures/catalog-definitions.json")
}

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    }
}

/// A built definition's digest, or `rejected`.
fn outcome<T: Serialize, E: Debug>(key: &str, built: &Result<T, E>) -> String {
    let definition = match built {
        Ok(definition) => definition,
        Err(refusal) => {
            eprintln!("{key} rejected: {refusal:#?}");
            return "rejected".into();
        }
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

/// What the families build of one catalog model's components: nothing when
/// no family recognizes its target; else its target's definition and, when
/// the target built and the model has a separate draft, the draft's.
type Built = Option<(
    Result<ModelDefinition, FamilyError>,
    Option<Result<DraftDefinition, magnitude_family_dflash::Error>>,
)>;

/// Each catalog model's [`Built`], by catalog model ID.
fn catalog() -> BTreeMap<String, Built> {
    let index = headers::index();
    let mut catalog = BTreeMap::new();
    for target in index.iter().filter(|component| component.role == ComponentRole::Target) {
        let model = target.model.as_str();
        let component = |role| {
            index
                .iter()
                .find(|component| component.model == model && component.role == role)
        };
        let directory = headers::component(target);
        let built = FAMILIES.iter().find(|family| family.recognizes(&directory)).map(|family| {
            let projector = component(ComponentRole::Projector).map(headers::component);
            let built = family.inspect(&directory, projector.as_ref(), identity());
            let draft = match (&built, component(ComponentRole::Draft)) {
                (Ok(definition), Some(draft)) => Some(magnitude_family_dflash::inspect(
                    &headers::component(draft),
                    definition,
                    &|layer| family.layer_entry(definition, layer),
                )),
                _ => None,
            };
            (built, draft)
        });
        catalog.insert(model.to_owned(), built);
    }
    catalog
}

fn outcomes() -> BTreeMap<String, Outcomes> {
    catalog()
        .into_iter()
        .map(|(model, built)| {
            let outcomes = match built {
                None => Outcomes {
                    target: "unrecognized".into(),
                    draft: None,
                },
                Some((built, draft)) => Outcomes {
                    target: outcome(&format!("{model} target"), &built),
                    draft: draft.map(|draft| outcome(&format!("{model} draft"), &draft)),
                },
            };
            (model, outcomes)
        })
        .collect()
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
        .map(|key| (key, (pinned.get(key), outcomes.get(key))))
        .collect::<BTreeMap<_, _>>();
    assert!(
        differing.is_empty(),
        "definitions differ from the pinned ones, as (pinned, built): {differing:#?}"
    );
}

fn crosses<T: Serialize + DeserializeOwned + PartialEq + Debug>(source: &str, value: &T) {
    let bytes = postcard::to_allocvec(value).unwrap_or_else(|error| panic!("{source}: {error}"));
    let decoded: T = postcard::from_bytes(&bytes).unwrap_or_else(|error| panic!("{source}: {error}"));
    assert_eq!(&decoded, value, "{source} changed crossing the wire");
}

/// The definition the family that recognizes `target` builds, if any does
/// and it builds one.
fn inspect(target: &Directory) -> Option<ModelDefinition> {
    let family = FAMILIES.iter().find(|family| family.recognizes(target))?;
    family.inspect(target, None, identity()).ok()
}

/// The pooled-row families, DeepSeek V4.1 (its fixture slices), Qwen4exp
/// (the catalog's) and GLM-5.3-Flash (its pinned artifact), are among the
/// definitions that cross.
#[test]
fn every_built_definition_crosses_the_worker_wire() {
    let mut families = BTreeSet::new();
    let mut cross = |source: &str, definition: &ModelDefinition| {
        crosses(source, definition);
        families.insert(definition.family.0.clone());
    };
    for (model, built) in catalog() {
        let Some((Ok(definition), draft)) = built else { continue };
        cross(&format!("{model} target"), &definition);
        if let Some(Ok(draft)) = draft {
            crosses(&format!("{model} draft"), &draft);
        }
    }
    for artifact in catalog_headers::artifacts() {
        if let Some(definition) = inspect(&headers::artifact(&artifact.name)) {
            cross(&artifact.name, &definition);
        }
    }
    let fixtures = package().join("../../fixtures/assessment/headers");
    for entry in std::fs::read_dir(&fixtures).unwrap() {
        let path = entry.unwrap().path();
        if !path.to_string_lossy().ends_with(".gguf.gz") {
            continue;
        }
        let mut header = Vec::new();
        flate2::read::GzDecoder::new(std::fs::File::open(&path).unwrap())
            .read_to_end(&mut header)
            .unwrap();
        if let Some(definition) = inspect(&gguf::read_header(header).unwrap()) {
            cross(&path.display().to_string(), &definition);
        }
    }
    for pooled in ["deepseek41", "qwen4exp", "glm5-next"] {
        assert!(families.contains(pooled), "no {pooled} definition crossed: {families:?}");
    }
}
