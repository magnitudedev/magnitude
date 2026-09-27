//! Every catalog target the executor admits derives complete assessment
//! terms on every backend, and the fixed measurement plan covers them: its
//! decode demand under both native codecs (every term's cost key timed, its
//! weight format timed and its exact representation binding formed) and its
//! memory terms. A form the executor does not run is refused at admission,
//! with a typed reason, before any term is derived; a target past admission
//! whose terms fail, or whose demand the plan does not cover, is a failure.

use super::basis::MeasurementKey;
use super::demand::DecodeDemand;
use super::plan::{activation, measurement_plan, PlannedKey};
use crate::{
    resident_layout, AssessmentMemoryTerms, ComponentSelection, ExecutionPath, ModelLoadPlan,
    PlannedMethod, ResourceLimits,
};
use magnitude_artifacts::{
    ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
};
use magnitude_family_common::headers;
use magnitude_family_contracts::ModelFamily;
use magnitude_family_gemma4::Gemma4Family;
use magnitude_family_lfm2::Lfm2Family;
use magnitude_family_llama::LlamaFamily;
use magnitude_family_muse_glimmer::MuseGlimmerFamily;
use magnitude_family_nemotron_h::NemotronHFamily;
use magnitude_family_qwen35::Qwen35Family;
use magnitude_state::KvCodec;
use seismic::BackendName;

const FAMILIES: [&dyn ModelFamily; 6] = [
    &Qwen35Family,
    &LlamaFamily,
    &NemotronHFamily,
    &Lfm2Family,
    &Gemma4Family,
    &MuseGlimmerFamily,
];

const BACKENDS: [BackendName; 4] = [
    BackendName::Metal,
    BackendName::Cuda,
    BackendName::Vulkan,
    BackendName::Cpu,
];

/// The standard service's limits, as assessment plans them.
const LIMITS: ResourceLimits = ResourceLimits {
    max_launch_rows: 512,
    max_launch_slots: 8,
    max_projected_rows: 8,
    max_images_per_request: 1,
    lookahead: true,
};

#[test]
fn every_admitted_catalog_target_derives_complete_terms() {
    let identity = PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    };
    let mut admitted = Vec::new();
    let mut refused = Vec::new();
    let mut failures = Vec::new();
    for (model, role, file) in headers::index() {
        if !role.starts_with("target") || file.ends_with("-allshards.json") {
            continue;
        }
        // A split target's full tensor directory is dumped beside its first
        // shard's.
        let all_shards = file.replace(".json", "-allshards.json");
        let file = if headers::root().join(&all_shards).exists() {
            all_shards
        } else {
            file
        };
        let target = headers::directory(&file);
        let Some(family) = FAMILIES.iter().find(|family| family.recognizes(&target)) else {
            continue;
        };
        let Ok(definition) = family.inspect(&target, None, identity) else {
            continue;
        };
        if let Err(error) = crate::operators::admit(&definition, false) {
            refused.push(format!("{model} {role}: {error}"));
            continue;
        }
        let manifest = PackageManifest {
            identity,
            target: ComponentManifest {
                files: vec![ComponentFile {
                    path: file.clone().into(),
                    size: target.tensors.iter().map(|tensor| tensor.nbytes).sum(),
                }],
                identity: identity.target,
                tensors: target.tensors.clone(),
            },
            projector: None,
            draft: None,
        };
        let selection = ComponentSelection {
            head: false,
            vision: false,
        };
        for backend in BACKENDS {
            let layout = resident_layout(ExecutionPath::Native, backend);
            let load = match ModelLoadPlan::derive(&manifest, &definition, selection, layout) {
                Ok(load) => load,
                // A representation the backend has no resident form for.
                Err(error) => {
                    refused.push(format!("{model} {role} {}: {error}", backend.as_str()));
                    continue;
                }
            };
            let plan = measurement_plan(backend);
            let covered = |key: &MeasurementKey, timed: bool| {
                plan.contains(&if timed {
                    PlannedKey::Timed(key.clone())
                } else {
                    PlannedKey::Formed(key.clone())
                })
            };
            let terms = [KvCodec::Dense, KvCodec::AffineK8V4]
                .into_iter()
                .map(|codec| DecodeDemand::from_model(&definition, &load, codec))
                .collect::<Result<Vec<_>, _>>()
                .map(|demands| {
                    let terms = demands.iter().flat_map(|demand| &demand.terms);
                    let outside = terms
                        .clone()
                        .filter(|term| {
                            !covered(&term.key.cost(), true)
                                || (term.key.class.binds_representation()
                                    && !covered(&term.key, false))
                                || term.weight().is_some_and(|weight| {
                                    !covered(
                                        &MeasurementKey::weight_format(weight, activation()),
                                        true,
                                    )
                                })
                        })
                        .map(|term| term.key.to_string())
                        .collect::<Vec<_>>();
                    (terms.count(), outside)
                });
            let memory = AssessmentMemoryTerms::derive(
                &definition,
                &load,
                selection,
                KvCodec::AffineK8V4,
                PlannedMethod::Plain,
                LIMITS,
            );
            match (terms, memory) {
                (Ok((terms, outside)), Ok(_)) if outside.is_empty() => {
                    admitted.push(format!("{model} {role} {}: {terms} terms", backend.as_str()))
                }
                (Ok((_, outside)), Ok(_)) => failures.push(format!(
                    "{model} {role} {}: outside the plan: {}",
                    backend.as_str(),
                    outside.join(", ")
                )),
                (terms, memory) => failures.push(format!(
                    "{model} {role} {}: {:?} / {:?}",
                    backend.as_str(),
                    terms.err(),
                    memory.err()
                )),
            }
        }
    }
    println!("admitted:\n  {}", admitted.join("\n  "));
    println!("refused:\n  {}", refused.join("\n  "));
    assert!(failures.is_empty(), "missing terms:\n  {}", failures.join("\n  "));
    assert!(!admitted.is_empty());
}
