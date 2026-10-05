//! Every catalog target the executor admits derives complete assessment
//! terms on every backend: its decode demand under both native codecs, its
//! memory terms, and every graph a load prepares, so no call lies outside
//! its kernel's domain. A form the executor does not run is refused at
//! admission, with a typed reason, before any term is derived; a target past
//! admission whose terms or graphs fail is a failure.

use super::demand::DecodeDemand;
use crate::{
    resident_layout, AssessmentGraphResourceBounds, AssessmentMemoryTerms, ComponentSelection,
    ExecutionPath, ModelLoadPlan, PlannedMethod, ResourceCapacity, ResourceLimits, ResourcePlanner,
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
    max_selected_rows: 8,
    max_drafting_slots: 8,
    exported_logits_rows: 0,
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
            // Admission binds every weight to its entries' ports, a
            // second-level scale included (`PlanError::UnportedScale`).
            if let Err(error) = load.program_plan(&definition, KvCodec::AffineK8V4) {
                failures.push(format!("{model} {role} {}: {error}", backend.as_str()));
                continue;
            }
            let demands = [KvCodec::Dense, KvCodec::AffineK8V4]
                .into_iter()
                .map(|codec| DecodeDemand::from_model(&definition, &load, codec))
                .collect::<Result<Vec<_>, _>>();
            let memory = AssessmentMemoryTerms::derive(
                &definition,
                &load,
                selection,
                KvCodec::AffineK8V4,
                PlannedMethod::Plain,
                LIMITS,
            );
            // Building every graph the load prepares proves no call lies
            // outside its kernel's domain on the backend.
            let graphs = ResourcePlanner::state_plan(
                &definition,
                &load,
                PlannedMethod::Plain,
                KvCodec::AffineK8V4,
                LIMITS,
                ResourceCapacity {
                    domain_bytes: 64 << 30,
                    tensor_operations: crate::TensorOperations::Absent,
                },
            )
            .and_then(|state| {
                AssessmentGraphResourceBounds::derive(
                    &definition,
                    &load,
                    &state,
                    PlannedMethod::Plain,
                    KvCodec::AffineK8V4,
                    LIMITS,
                    backend,
                )
                .map_err(|error| error.to_string())
            });
            match (demands, memory, graphs) {
                // A decode step streams its weights and launches its entries.
                (Ok(demands), Ok(_), Ok(_))
                    if demands
                        .iter()
                        .all(|demand| demand.streamed_bytes > 0 && demand.launches > 0) =>
                {
                    admitted.push(format!(
                        "{model} {role} {}: {} launches, {} bytes",
                        backend.as_str(),
                        demands[0].launches,
                        demands[0].streamed_bytes
                    ))
                }
                (Ok(demands), Ok(_), Ok(_)) => failures.push(format!(
                    "{model} {role} {}: an empty decode step: {demands:?}",
                    backend.as_str()
                )),
                (demands, memory, graphs) => failures.push(format!(
                    "{model} {role} {}: {:?} / {:?} / {:?}",
                    backend.as_str(),
                    demands.err(),
                    memory.err(),
                    graphs.err()
                )),
            }
        }
    }
    println!("admitted:\n  {}", admitted.join("\n  "));
    println!("refused:\n  {}", refused.join("\n  "));
    assert!(failures.is_empty(), "failing targets:\n  {}", failures.join("\n  "));
    assert!(!admitted.is_empty());
}

/// Separate drafts of the qualification artifacts (target header, draft
/// header) and their target families. Each plans its draft program and
/// charges its weights, state and every draft graph class from headers
/// alone, on every backend: the draft graphs are sealed exactly, in
/// metadata, for the planned method's proposals.
const SEPARATE_DRAFTS: [(&str, &str, &dyn ModelFamily); 6] = [
    ("qwen3.6-35b-a3b__target-gguf_q4.json", "qwen3.6-35b-a3b__draft.json", &Qwen35Family),
    ("lfm2.5-2.6b__target-gguf_q4.json", "lfm2.5-2.6b__draft.json", &Lfm2Family),
    ("qwen3.8-27b__target-gguf_q4.json", "qwen3.8-27b__draft-dspark.json", &Qwen35Family),
    ("qwen3.8-27b__target-gguf_q4.json", "qwen3.8-27b__draft-dflash2.json", &Qwen35Family),
    // An NVFP4 draft (scaled fusion and feed-forward) beside a Q4_K_M
    // target, and beside the NVFP4 target (a scaled vocabulary projection).
    (
        "nemotron-3.5-lightning-30b-a3b__target-gguf_q4.json",
        "nemotron-3.5-lightning-30b-a3b__draft.json",
        &NemotronHFamily,
    ),
    (
        "nemotron-3.5-lightning-30b-a3b__target-gguf_nvfp4-qat.json",
        "nemotron-3.5-lightning-30b-a3b__draft.json",
        &NemotronHFamily,
    ),
];

#[test]
fn separate_drafts_plan_and_charge_every_graph_class_from_headers() {
    let identity = PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    };
    let component = |file: &str, directory: &magnitude_artifacts::gguf::Directory, identity| {
        ComponentManifest {
            files: vec![ComponentFile {
                path: file.into(),
                size: directory.tensors.iter().map(|tensor| tensor.nbytes).sum(),
            }],
            identity,
            tensors: directory.tensors.clone(),
        }
    };
    for (target_file, draft_file, family) in SEPARATE_DRAFTS {
        let target = headers::directory(target_file);
        let draft = headers::directory(draft_file);
        let declared = family.inspect(&target, None, identity).unwrap();
        let bound = magnitude_family_dflash::inspect(&draft, &declared, &|layer| {
            family.layer_entry(&declared, layer)
        })
        .unwrap_or_else(|error| panic!("{draft_file}: {error}"));
        let proposals = u8::try_from(bound.max_proposals()).unwrap();
        // A separate-draft load executes the draft, never an embedded head.
        let definition = magnitude_family_contracts::ModelDefinition {
            head: None,
            draft: Some(bound),
            ..declared
        };
        crate::operators::admit(&definition, true)
            .unwrap_or_else(|error| panic!("{draft_file}: {error}"));
        let manifest = PackageManifest {
            identity,
            target: component(target_file, &target, identity.target),
            projector: None,
            draft: Some(component(draft_file, &draft, ArtifactIdentity([9; 32]))),
        };
        let selection = ComponentSelection {
            head: true,
            vision: false,
        };
        let method = PlannedMethod::DFlash { proposals };
        for backend in BACKENDS {
            let context = format!("{draft_file} {}", backend.as_str());
            let layout = resident_layout(ExecutionPath::Native, backend);
            let load = ModelLoadPlan::derive(&manifest, &definition, selection, layout)
                .unwrap_or_else(|error| panic!("{context}: {error}"));
            let terms = AssessmentMemoryTerms::derive(
                &definition,
                &load,
                selection,
                KvCodec::AffineK8V4,
                method,
                LIMITS,
            )
            .unwrap_or_else(|error| panic!("{context}: {error}"));
            assert!(terms.head_weights > 0, "{context}: the draft's weights are charged");
            let state = ResourcePlanner::state_plan(
                &definition,
                &load,
                method,
                KvCodec::AffineK8V4,
                LIMITS,
                ResourceCapacity {
                    domain_bytes: 64 << 30,
                    tensor_operations: crate::TensorOperations::Absent,
                },
            )
            .unwrap_or_else(|error| panic!("{context}: {error}"));
            let graph = AssessmentGraphResourceBounds::derive(
                &definition,
                &load,
                &state,
                method,
                KvCodec::AffineK8V4,
                LIMITS,
                backend,
            )
            .unwrap_or_else(|error| panic!("{context}: {error}"));
            assert!(graph.total_bytes > 0, "{context}");
        }
    }
}
