//! Every catalog target the executor admits derives complete assessment
//! terms on every backend: its decode demand under both native codecs, its
//! memory terms, and every graph a load prepares, so no call lies outside
//! its kernel's domain. A form the executor does not run is refused at
//! admission, with a typed reason, before any term is derived; a target past
//! admission whose terms or graphs fail is a failure.

use super::demand::DecodeDemand;
use super::launches::Declarations;
use super::AssessmentError;
use crate::{
    resident_layout, AssessmentGraphResourceBounds, AssessmentMemoryTerms, ComponentSelection,
    ExecutionPath, ModelLoadPlan, PlannedMethod, ResourceCapacity, ResourceLimits, ResourcePlanner,
};
use magnitude_artifacts::catalog_headers::{CatalogComponent, ComponentRole};
use magnitude_artifacts::{
    ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
};
use magnitude_family_common::headers::{self, Header};
use magnitude_family_contracts::ModelFamily;
use magnitude_family_gemma4::Gemma4Family;
use magnitude_family_lfm2::Lfm2Family;
use magnitude_family_llama::LlamaFamily;
use magnitude_family_muse_glimmer::MuseGlimmerFamily;
use magnitude_family_nemotron_h::NemotronHFamily;
use magnitude_family_qwen35::Qwen35Family;
use magnitude_family_qwen4exp::Qwen4ExpFamily;
use magnitude_state::KvCodec;
use seismic::BackendName;
use std::collections::BTreeMap;
use std::path::Path;

const FAMILIES: [&dyn ModelFamily; 7] = [
    &Qwen35Family,
    &LlamaFamily,
    &NemotronHFamily,
    &Lfm2Family,
    &Gemma4Family,
    &MuseGlimmerFamily,
    &Qwen4ExpFamily,
];

/// Catalog models whose target's load must derive whatever their catalog
/// support level.
const REQUIRED: [&str; 1] = ["qwen3.8-flash-next:gguf:q4"];

/// Each catalog model's support level (`catalog/models.json`), by catalog
/// model ID.
fn support_levels() -> BTreeMap<String, String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../catalog/models.json");
    let catalog: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path).unwrap_or_else(|error| panic!("{path:?}: {error}")),
    )
    .unwrap();
    catalog["models"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|model| {
            let id = model["id"].as_str().unwrap();
            let level = model["support"]["level"].as_str().unwrap();
            model["sources"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|source| source["variants"].as_array().unwrap())
                .map(move |variant| {
                    (
                        format!("{id}:{}", variant["variantId"].as_str().unwrap()),
                        level.to_owned(),
                    )
                })
        })
        .collect()
}

/// The standard service's limits, as assessment plans them.
const LIMITS: ResourceLimits = ResourceLimits {
    max_launch_rows: 512,
    max_launch_slots: 8,
    max_selected_rows: 8,
    max_drafting_slots: 8,
    exported_logits_rows: 0,
    max_images_per_request: 1,
    max_image_cells: 0,
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
    let levels = support_levels();
    let mut required_seen = Vec::new();
    for component in headers::index() {
        if component.role != ComponentRole::Target {
            continue;
        }
        let name = component.model.as_str();
        let target = headers::component(component);
        // A target that must serve: its catalog model is neither disabled
        // nor deprecated, or it is named in `REQUIRED`.
        let required = REQUIRED.contains(&name);
        if required {
            required_seen.push(name);
        }
        let served = required
            || !matches!(
                levels.get(name).map(String::as_str),
                Some("disabled" | "deprecated")
            );
        let Some(family) = FAMILIES.iter().find(|family| family.recognizes(&target)) else {
            if served {
                failures.push(format!("{name}: no family recognizes the target"));
            }
            continue;
        };
        let definition = match family.inspect(&target, None, identity) {
            Ok(definition) => definition,
            Err(error) => {
                let line = format!("{name}: its family fails to inspect it: {error}");
                if served {
                    failures.push(line);
                } else {
                    refused.push(line);
                }
                continue;
            }
        };
        if let Err(error) = crate::operators::admit(&definition, false) {
            refused.push(format!("{name}: {error}"));
            continue;
        }
        let manifest = PackageManifest {
            identity,
            target: ComponentManifest {
                files: vec![ComponentFile {
                    path: component.path.clone(),
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
        for backend in BackendName::ALL {
            let layout = resident_layout(ExecutionPath::Native, backend);
            let load = match ModelLoadPlan::derive(&manifest, &definition, selection, layout)
                .and_then(|load| load.with_selected_reads(&definition, PlannedMethod::Plain, LIMITS))
            {
                Ok(load) => load,
                // A target that must serve derives its load on every
                // backend; another may lack a resident form.
                Err(error) => {
                    let line = format!("{name} {}: {error}", backend.as_str());
                    if served {
                        failures.push(line);
                    } else {
                        refused.push(line);
                    }
                    continue;
                }
            };
            // Admission binds every weight to its entries' ports, a
            // second-level scale included (`PlanError::UnportedScale`).
            if let Err(error) = load.program_plan(&definition, KvCodec::AffineK8V4) {
                failures.push(format!("{name} {}: {error}", backend.as_str()));
                continue;
            }
            // Each call's launches are its entry's declaration's at the
            // statics the load prepares it with.
            let demands = [KvCodec::Dense, KvCodec::AffineK8V4]
                .into_iter()
                .map(|codec| {
                    let program = load
                        .program_plan(&definition, codec)
                        .map_err(|error| AssessmentError::Demand(error.to_string()))?;
                    let declarations = Declarations::of_load(
                        backend,
                        &program,
                        &definition,
                        &load,
                        crate::native::TuningLimits::of(LIMITS, &definition),
                    )?;
                    let demand =
                        DecodeDemand::from_model(&definition, &load, codec, &declarations)?;
                    // Every call's launches at one and at the decode rows'
                    // most.
                    for call in &demand.calls {
                        for rows in [1, crate::operators::attention::graph::DECODE_ROWS] {
                            call.launches.at(rows)?;
                        }
                    }
                    Ok(demand)
                })
                .collect::<Result<Vec<_>, AssessmentError>>();
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
                    if demands.iter().all(|demand| {
                        !demand.calls.is_empty()
                            && demand
                                .calls
                                .iter()
                                .any(|call| call.work.weights.iter().any(|bytes| bytes.bytes > 0))
                    }) =>
                {
                    admitted.push(format!(
                        "{name} {}: {} calls, {} weight bytes",
                        backend.as_str(),
                        demands[0].calls.len(),
                        demands[0]
                            .calls
                            .iter()
                            .flat_map(|call| &call.work.weights)
                            .map(|bytes| bytes.bytes)
                            .sum::<u64>()
                    ))
                }
                (Ok(demands), Ok(_), Ok(_)) => failures.push(format!(
                    "{name} {}: an empty decode step: {demands:?}",
                    backend.as_str()
                )),
                (demands, memory, graphs) => failures.push(format!(
                    "{name} {}: {:?} / {:?} / {:?}",
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
    assert!(
        failures.is_empty(),
        "failing targets:\n  {}",
        failures.join("\n  ")
    );
    assert!(!admitted.is_empty());
    // Every required target is in the catalog's headers and admitted on
    // every backend.
    for name in REQUIRED {
        assert!(
            required_seen.contains(&name),
            "required target {name} has no catalog header"
        );
        for backend in BackendName::ALL {
            let prefix = format!("{name} {}:", backend.as_str());
            assert!(
                admitted.iter().any(|line| line.starts_with(&prefix)),
                "required target {name} is not admitted on {}",
                backend.as_str()
            );
        }
    }
}

/// The catalog's separate draft of catalog model `model`.
const fn catalog_draft(model: &'static str) -> Header {
    Header::Catalog(model, ComponentRole::Draft)
}

/// Separate drafts of the qualification artifacts (target's catalog model,
/// draft header) and their target families. Each plans its draft program and
/// charges its weights, state and every draft graph class from headers
/// alone, on every backend: the draft graphs are sealed exactly, in
/// metadata, for the planned method's proposals.
const SEPARATE_DRAFTS: [(&str, Header, &dyn ModelFamily); 6] = [
    (
        "qwen3.6-35b-a3b:gguf:q4",
        catalog_draft("qwen3.6-35b-a3b:gguf:q4"),
        &Qwen35Family,
    ),
    (
        "lfm2.5-2.6b:gguf:q4",
        catalog_draft("lfm2.5-2.6b:gguf:q4"),
        &Lfm2Family,
    ),
    (
        "qwen3.8-27b:gguf:q4",
        Header::Artifact("qwen3.8-27b-dspark"),
        &Qwen35Family,
    ),
    // The catalog's DFlash2 draft.
    (
        "qwen3.8-27b:gguf:q4",
        catalog_draft("qwen3.8-27b:gguf:q4"),
        &Qwen35Family,
    ),
    // An NVFP4 draft (scaled fusion and feed-forward) beside a Q4_K_M
    // target, and beside the NVFP4 target (a scaled vocabulary projection).
    (
        "nemotron-3.5-lightning-30b-a3b:gguf:q4",
        catalog_draft("nemotron-3.5-lightning-30b-a3b:gguf:q4"),
        &NemotronHFamily,
    ),
    (
        "nemotron-3.5-lightning-30b-a3b:gguf:nvfp4-qat",
        catalog_draft("nemotron-3.5-lightning-30b-a3b:gguf:nvfp4-qat"),
        &NemotronHFamily,
    ),
];

/// Synthetic graph-planning fixtures retain captured tensor geometry and supply
/// explicit attention semantics for each named draft. These do not establish
/// admission of the captured artifacts; binary-header admission is qualified
/// separately by the DFlash family's publication test.
fn synthetic_separate_draft_directory(draft: Header) -> magnitude_artifacts::gguf::Directory {
    use magnitude_artifacts::gguf::{Metadata, Value};
    let causal: &[bool] = match draft {
        Header::Catalog("qwen3.6-35b-a3b:gguf:q4", ComponentRole::Draft) => {
            &[true, true, true, true, true, false]
        }
        Header::Catalog("lfm2.5-2.6b:gguf:q4", ComponentRole::Draft) => &[false; 5],
        Header::Artifact("qwen3.8-27b-dspark") => &[false; 5],
        Header::Catalog("qwen3.8-27b:gguf:q4", ComponentRole::Draft) => &[false; 5],
        Header::Catalog(
            "nemotron-3.5-lightning-30b-a3b:gguf:q4"
            | "nemotron-3.5-lightning-30b-a3b:gguf:nvfp4-qat",
            ComponentRole::Draft,
        ) => &[false; 6],
        draft => panic!("no synthetic attention semantics for {draft}"),
    };
    let mut directory = draft.directory();
    directory
        .metadata
        .retain(|entry| entry.name != "dflash.attention.causal");
    directory.metadata.push(Metadata {
        name: "dflash.attention.causal".into(),
        value: Value::Array(causal.to_vec().into()),
    });
    directory
}

#[test]
fn synthetic_separate_drafts_plan_and_charge_every_graph_class() {
    for (target, draft, family) in SEPARATE_DRAFTS {
        plan_and_charge_separate_draft(
            (target, &headers::target(target)),
            (&draft.to_string(), &synthetic_separate_draft_directory(draft)),
            family,
        );
    }
}

/// The Nemotron 3.5 Bolt and Super-VL drafts at their exact headers, each
/// read against its own target: LiLiCorr (Bolt with attention sinks,
/// Super-VL with its own NVFP4 projection) and DSpark (Bolt with an NVFP4
/// Markov projection). Every graph class of their backbones, corrector and
/// Markov head plans and charges on every backend.
const NEMOTRON_DRAFTS: [(&str, Header); 4] = [
    (
        "nemotron-3.5-bolt-74b-a9b:gguf:iq2-qat",
        catalog_draft("nemotron-3.5-bolt-74b-a9b:gguf:iq2-qat"),
    ),
    (
        "nemotron-3.5-bolt-74b-a9b:gguf:iq2-qat",
        Header::Artifact("nemotron-3.5-bolt-74b-a9b-dspark"),
    ),
    (
        "nemotron-3.5-super-vl-120b-a12b:gguf:q4",
        catalog_draft("nemotron-3.5-super-vl-120b-a12b:gguf:q4"),
    ),
    (
        "nemotron-3.5-super-vl-120b-a12b:gguf:q4",
        Header::Artifact("nemotron-3.5-super-vl-120b-a12b-dspark"),
    ),
];

#[test]
fn nemotron_drafts_plan_and_charge_every_graph_class() {
    for (target, draft) in NEMOTRON_DRAFTS {
        plan_and_charge_separate_draft(
            (target, &headers::target(target)),
            (&draft.to_string(), &draft.directory()),
            &NemotronHFamily,
        );
    }
}

/// A separate draft read against its target admits, prepares only its
/// load's generated width, and derives its memory terms, state plan and
/// graph bounds on every backend.
fn plan_and_charge_separate_draft(
    (target_file, target): (&str, &magnitude_artifacts::gguf::Directory),
    (draft_file, draft): (&str, &magnitude_artifacts::gguf::Directory),
    family: &dyn ModelFamily,
) {
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
    {
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
        for generated in [0, 1, proposals] {
            let classes = crate::programs::native_draft::draft_graph_classes(
                LIMITS,
                proposals as usize,
                &[generated][..usize::from(generated > 0)],
                definition.draft.as_ref().unwrap(),
            )
            .unwrap();
            let actual: std::collections::BTreeSet<_> = classes
                .iter()
                .filter(|class| class.slots > 0)
                .map(|class| class.proposals as u8)
                .collect();
            assert_eq!(
                actual,
                (generated > 0).then_some(generated).into_iter().collect(),
                "only the load's generated width is prepared"
            );
        }
        let method = PlannedMethod::DFlash {
            proposals,
            widths: crate::DraftWidths::Adaptive,
        };
        for backend in BackendName::ALL {
            let context = format!("{draft_file} {}", backend.as_str());
            let layout = resident_layout(ExecutionPath::Native, backend);
            let load = ModelLoadPlan::derive(&manifest, &definition, selection, layout)
                .and_then(|load| load.with_selected_reads(&definition, method, LIMITS))
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
            assert!(
                terms.head_weights > 0,
                "{context}: the draft's weights are charged"
            );
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
            // Every draft class's exact graph seals into its certified layout.
            let programs = load
                .program_plan(&definition, KvCodec::AffineK8V4)
                .unwrap_or_else(|error| panic!("{context}: {error}"));
            let geometry = crate::programs::native_draft::DraftGeometry::new(
                &definition,
                state.head_state().unwrap(),
            )
            .unwrap_or_else(|error| panic!("{context}: {error}"));
            let classes = crate::programs::native_draft::draft_graph_classes(
                LIMITS,
                method.draft_rows(),
                &method.block_widths(),
                definition.draft.as_ref().unwrap(),
            )
            .unwrap_or_else(|error| panic!("{context}: {error}"));
            crate::programs::native_draft::verify_draft_family_certificates(
                backend,
                &load,
                programs.draft().unwrap(),
                &geometry,
                &classes,
            )
            .unwrap_or_else(|error| panic!("{context}: {error}"));
        }
    }
}

/// A draft header bound against its target as the host binds it
/// (`magnitude_engine::host::bind_draft`): a block drafter as a separate
/// draft, a prediction-head or assistant file as a head in the draft
/// component.
fn bind_catalog_draft(
    family: &dyn ModelFamily,
    declared: &magnitude_family_contracts::ModelDefinition,
    draft: &magnitude_artifacts::gguf::Directory,
) -> Result<magnitude_family_contracts::ModelDefinition, String> {
    use magnitude_artifacts::gguf::Value;
    let declared = declared.clone();
    match draft.value("general.architecture").and_then(Value::string) {
        Some(magnitude_family_dflash::ARCHITECTURE) => {
            magnitude_family_dflash::inspect(draft, &declared, &|layer| {
                family.layer_entry(&declared, layer)
            })
            .map(|separate| magnitude_family_contracts::ModelDefinition {
                head: None,
                draft: Some(separate),
                ..declared.clone()
            })
            .map_err(|error| error.to_string())
        }
        Some(magnitude_family_qwen4exp::FAMILY_ID) => {
            magnitude_family_qwen4exp::inspect_head(draft, &declared)
                .map(|head| magnitude_family_contracts::ModelDefinition {
                    head: Some(head),
                    ..declared.clone()
                })
                .map_err(|error| error.to_string())
        }
        Some(magnitude_family_gemma4::ASSISTANT_ARCHITECTURE) => {
            magnitude_family_gemma4::inspect_assistant(draft, &declared)
                .map(|head| magnitude_family_contracts::ModelDefinition {
                    head: Some(head),
                    ..declared.clone()
                })
                .map_err(|error| error.to_string())
        }
        architecture => Err(format!("no drafter family for {architecture:?}")),
    }
}

/// Every catalog variant lists its kernels over its finalized readout in
/// every method it loads with: plain (its drafter off), its embedded MTP
/// heads, and its drafter file, with its projector where it has one, on
/// every backend that reads progressive heads. Every view the load offers
/// is selected: a scope stored as views keeps its packed projection planned
/// and not placed, and the readers still bound to it (the target's packed
/// readout, a head's prepared readout) resolve it. A load that does not
/// derive or list its kernels with no view selected is outside this test.
#[test]
fn every_catalog_variant_lists_kernels_over_every_finalized_readout() {
    let mut models = BTreeMap::<&str, Vec<&CatalogComponent>>::new();
    for component in headers::index() {
        models.entry(&component.model).or_default().push(component);
    }
    let component = |file: &Path, directory: &magnitude_artifacts::gguf::Directory, identity| {
        ComponentManifest {
            files: vec![ComponentFile {
                path: file.into(),
                size: directory.tensors.iter().map(|tensor| tensor.nbytes).sum(),
            }],
            identity,
            tensors: directory.tensors.clone(),
        }
    };
    let mut finalized = Vec::new();
    let mut outside = Vec::new();
    let mut failures = Vec::new();
    for components in models.values() {
        let file = |role: ComponentRole| {
            components
                .iter()
                .find(|component| component.role == role)
                .map(|component| (component.path.clone(), headers::component(component)))
        };
        let projector = file(ComponentRole::Projector);
        let draft = file(ComponentRole::Draft);
        let identity = PackageIdentity {
            target: ArtifactIdentity([7; 32]),
            projector: projector.as_ref().map(|_| ArtifactIdentity([8; 32])),
        };
        for target_component in components {
            if target_component.role != ComponentRole::Target {
                continue;
            }
            let target_file = &target_component.path;
            let target = headers::component(target_component);
            let context = &target_component.model;
            let Some(family) = FAMILIES.iter().find(|family| family.recognizes(&target)) else {
                outside.push(format!("{context}: no family recognizes the target"));
                continue;
            };
            let declared = match family.inspect(
                &target,
                projector.as_ref().map(|(_, directory)| directory),
                identity,
            ) {
                Ok(definition) => definition,
                Err(error) => {
                    outside.push(format!("{context}: {error}"));
                    continue;
                }
            };
            // A plain load keeps an embedded head it never selects
            // (`ResolvedMethod::executed`).
            let mut loads = vec![("plain", false, declared.clone(), None)];
            if declared.head.is_some() {
                loads.push(("mtp", true, declared.clone(), None));
            }
            if let Some((draft_file, draft)) = &draft {
                match bind_catalog_draft(*family, &declared, draft) {
                    Ok(definition) => loads.push((
                        "drafter",
                        true,
                        definition,
                        Some(component(draft_file, draft, ArtifactIdentity([9; 32]))),
                    )),
                    Err(error) => outside.push(format!("{context} drafter: {error}")),
                }
            }
            for (method, head, definition, draft_component) in loads {
                let manifest = PackageManifest {
                    identity,
                    target: component(&target_file, &target, identity.target),
                    projector: projector.as_ref().map(|(file, directory)| {
                        component(file, directory, identity.projector.unwrap())
                    }),
                    draft: draft_component,
                };
                let selection = ComponentSelection {
                    head,
                    vision: projector.is_some(),
                };
                for backend in BackendName::ALL {
                    let context = format!("{context} {method} {}", backend.as_str());
                    let progressive =
                        crate::programs::graph::readout::reads_progressive_heads(backend).unwrap();
                    let listed = |load: ModelLoadPlan, choices: &[crate::ReadoutClassChoice]| {
                        let load = load.finalize_readout(choices)?;
                        let plan = load
                            .program_plan(&definition, KvCodec::AffineK8V4)
                            .map_err(|error| error.to_string())?;
                        crate::HostProcessBytes::derive(&definition, &load, &plan, LIMITS, backend)
                    };
                    let load = match crate::operators::admit(&definition, head)
                        .map_err(|error| error.to_string())
                        .and_then(|()| {
                            ModelLoadPlan::derive(
                                &manifest,
                                &definition,
                                selection,
                                resident_layout(ExecutionPath::Native, backend),
                            )
                        })
                        .and_then(|load| {
                            if progressive {
                                load.with_readout_candidates(&definition)
                            } else {
                                Ok(load)
                            }
                        }) {
                        Ok(load) => load,
                        Err(error) => {
                            outside.push(format!("{context}: {error}"));
                            continue;
                        }
                    };
                    if let Err(error) = listed(load.clone(), &[]) {
                        outside.push(format!("{context}: {error}"));
                        continue;
                    }
                    let mut scopes = Vec::new();
                    for candidate in load.readout_candidates() {
                        if !scopes.contains(&candidate.role.scope) {
                            scopes.push(candidate.role.scope);
                        }
                    }
                    if scopes.is_empty() {
                        continue;
                    }
                    let choices: Vec<_> = scopes
                        .iter()
                        .map(|&scope| crate::ReadoutClassChoice {
                            scope,
                            norm_scope: scope,
                            prepared: false,
                            rows: vec![1],
                        })
                        .collect();
                    match listed(load, &choices) {
                        Ok(_) => finalized.push(format!("{context}: views of {scopes:?}")),
                        Err(error) => failures.push(format!("{context}: {error}")),
                    }
                }
            }
        }
    }
    println!("finalized with views:\n{}", finalized.join("\n"));
    println!("outside this test:\n{}", outside.join("\n"));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(!finalized.is_empty(), "no catalog load offers views");
}
