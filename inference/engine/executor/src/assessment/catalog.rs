//! Every catalog target the executor admits derives complete assessment
//! terms on every backend: its decode demand under both native codecs, its
//! memory terms, and every graph a load prepares, so no call lies outside
//! its kernel's domain; where its experts stream on a backend, the same of
//! its load with them in an expert bank at its floor. A form the executor
//! does not run is refused with a typed reason before any term is derived;
//! a served target past admission whose terms or graphs fail, or whose
//! experts a backend that streams experts refuses to stream, is a failure.
//! A target that need not serve may lack a form on any backend: what it
//! fails is listed as refused.

use super::demand::DecodeDemand;
use super::launches::Declarations;
use super::AssessmentError;
use crate::{
    resident_layout, AssessmentGraphResourceBounds, AssessmentMemoryTerms, ComponentSelection,
    ExecutionPath, ModelLoadPlan, PlanError, PlannedMethod, ResourceCapacity, ResourceLimits,
    ResourcePlanner, UnstreamedExperts,
};
use magnitude_artifacts::catalog_headers::{CatalogComponent, ComponentRole};
use magnitude_artifacts::{
    ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
};
use magnitude_family_common::headers::{self, Header};
use magnitude_family_contracts::ModelFamily;
use magnitude_family_lfm2::Lfm2Family;
use magnitude_family_nemotron_h::NemotronHFamily;
use magnitude_family_qwen35::Qwen35Family;
use magnitude_family_registry::FAMILIES;
use magnitude_state::KvCodec;
use seismic::BackendName;
use std::collections::BTreeMap;
use std::path::Path;

/// Catalog models whose target's load must derive whatever their catalog
/// support level.
const REQUIRED: [&str; 1] = ["qwen3.8-flash-next:gguf:q4"];

/// Each catalog model's support level (`catalog/models.json`), by catalog
/// model ID.
fn support_levels() -> BTreeMap<String, String> {
    let catalog: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../../../catalog/models.json")).unwrap();
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
    chain_capacity: std::num::NonZeroUsize::MIN,
    conversations: std::num::NonZeroUsize::MIN,
};

/// Every catalog target that must serve (its catalog model neither disabled
/// nor deprecated, or named in `REQUIRED`) and the executor admits, by
/// catalog model ID, with its definition and its package alone.
pub(super) fn served_targets() -> Vec<(String, magnitude_family_contracts::ModelDefinition, PackageManifest)> {
    let identity = PackageIdentity { target: ArtifactIdentity([7; 32]), projector: None };
    let levels = support_levels();
    let mut targets = Vec::new();
    for component in headers::index() {
        let name = component.model.as_str();
        let served = REQUIRED.contains(&name) || !matches!(levels.get(name).map(String::as_str), Some("disabled" | "deprecated"));
        if component.role != ComponentRole::Target || !served {
            continue;
        }
        let target = headers::component(component);
        let family = FAMILIES
            .iter()
            .find(|family| family.recognizes(&target))
            .unwrap_or_else(|| panic!("{name}: a family recognizes the served target"));
        let definition = family
            .inspect(&target, None, identity)
            .unwrap_or_else(|error| panic!("{name}: its family inspects it: {error}"));
        if crate::operators::admit(&definition, false).is_err() {
            continue;
        }
        targets.push((name.to_owned(), definition, target_manifest(component, &target, identity)));
    }
    targets
}

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
                sort(format!("{name}: its family fails to inspect it: {error}"), served, &mut failures, &mut refused);
                continue;
            }
        };
        if let Err(error) = crate::operators::admit(&definition, false) {
            refused.push(format!("{name}: {error}"));
            continue;
        }
        let manifest = target_manifest(component, &target, identity);
        for backend in BackendName::ALL {
            let load = match plain_load(&manifest, &definition, backend) {
                Ok(load) => load,
                // A target that must serve derives its load on every
                // backend; another may lack a resident form.
                Err(error) => {
                    sort(format!("{name} {}: {error}", backend.as_str()), served, &mut failures, &mut refused);
                    continue;
                }
            };
            // Experts that stream on the backend are planned in its bank too,
            // at its floor: what the most constrained streamed load holds.
            // Experts that do not stream are refused with their reason.
            let banked = match bank_reservations(&definition, backend) {
                Ok((reservation, host)) => Some(load.bank_floor(&definition, 1, LIMITS.flights(), reservation, host).and_then(|floor| {
                    load.clone().with_expert_bank(&definition, floor, 1, LIMITS.flights(), reservation, host)
                })),
                // A served target that routes experts streams them on every
                // backend whose recorded device streams experts at all: only
                // a fact of the device or its backend refuses it.
                Err(PlanError::Unstreamed(reason)) => {
                    let line = format!("{name} {} banked: {reason}", backend.as_str());
                    match served && !matches!(reason, UnstreamedExperts::Unrouted) && !refused_by_the_device(&reason) {
                        true => failures.push(line),
                        false => refused.push(line),
                    }
                    None
                }
                Err(error) => {
                    sort(format!("{name} {} banked: {error}", backend.as_str()), served, &mut failures, &mut refused);
                    None
                }
            };
            // Every routed form streams: a bank that refuses one fails.
            let loads = std::iter::once(("resident", Ok(load))).chain(banked.map(|banked| ("banked", banked)));
            for (form, load) in loads {
                match load
                    .map_err(|error| error.to_string())
                    .and_then(|load| derive_terms(&definition, &load, SELECTION, backend))
                {
                    Ok(terms) => admitted.push(format!("{name} {} {form}: {terms}", backend.as_str())),
                    Err(error) => sort(format!("{name} {} {form}: {error}", backend.as_str()), served, &mut failures, &mut refused),
                }
            }
        }
    }
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
            let prefix = format!("{name} {} resident:", backend.as_str());
            assert!(
                admitted.iter().any(|line| line.starts_with(&prefix)),
                "required target {name} is not admitted on {}",
                backend.as_str()
            );
        }
    }
}

/// Lists what a target fails: a failure where it must serve, a refusal
/// where it need not.
fn sort(line: String, served: bool, failures: &mut Vec<String>, refused: &mut Vec<String>) {
    match served {
        true => failures.push(line),
        false => refused.push(line),
    }
}

/// Whether `reason` is a fact of the device or its backend, which refuses
/// every model alike, rather than of the model.
fn refused_by_the_device(reason: &UnstreamedExperts) -> bool {
    use UnstreamedExperts::*;
    matches!(reason, Backend(_) | Unreserved(_) | Unreservable { .. } | Untiered { .. })
}

/// The components a plain load of a target selects: neither an embedded
/// head nor vision.
const SELECTION: ComponentSelection = ComponentSelection {
    head: false,
    vision: false,
};

/// The package of catalog target `component` alone, its directory
/// `target`.
fn target_manifest(
    component: &CatalogComponent,
    target: &magnitude_artifacts::gguf::Directory,
    identity: PackageIdentity,
) -> PackageManifest {
    PackageManifest {
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
    }
}

/// The plain load of `manifest` on `backend`, as the standard service
/// plans it.
fn plain_load(
    manifest: &PackageManifest,
    definition: &magnitude_family_contracts::ModelDefinition,
    backend: BackendName,
) -> Result<ModelLoadPlan, String> {
    let layout = resident_layout(ExecutionPath::Native, backend);
    ModelLoadPlan::derive(manifest, definition, SELECTION, layout, PlannedMethod::Plain, LIMITS)
}

/// Where the experts of `definition` stream on `backend`: the device's and
/// the host tier's reservations of the recorded machine of the backend.
/// Experts that do not stream are refused with their reason.
fn bank_reservations(
    definition: &magnitude_family_contracts::ModelDefinition,
    backend: BackendName,
) -> Result<(seismic::Reservation, crate::planning::HostTier), PlanError> {
    let recorded = recorded_reservation(backend).map_err(|why| why.to_string());
    let substrate = crate::planning::bank_substrate(
        backend,
        recorded.as_ref().map(|recorded| recorded.map(|(reservation, _)| reservation)).map_err(String::as_str),
        Ok(recorded.as_ref().ok().copied().flatten().and_then(|(_, host)| host)),
        definition,
    )?;
    crate::planning::routed_experts(definition)?;
    Ok(substrate)
}

/// Every expert record form a served catalog target streams on each backend
/// whose recorded device streams experts (each routed tensor's stored and
/// resident elements and matrix, as its bank on that backend plans it) is
/// resident as stored, or converts on the host, piece by piece
/// as a landing converts it (`RowConversion::piece`), into exactly the
/// registry's reference bytes:
/// one record of every form, in its conversion's domain, as pieces of one
/// tile and as pieces of the most whole tiles of 128 KiB of its source (a
/// landing's read unit).
#[test]
fn every_streamed_record_converts_on_the_host_as_its_reference() {
    let identity = PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    };
    let levels = support_levels();
    // Each form on each backend, and the targets that stream it.
    let mut forms = BTreeMap::<(BackendName, String, String, [u64; 2]), Vec<String>>::new();
    for component in headers::index() {
        if component.role != ComponentRole::Target {
            continue;
        }
        let name = component.model.as_str();
        let served = REQUIRED.contains(&name)
            || !matches!(
                levels.get(name).map(String::as_str),
                Some("disabled" | "deprecated")
            );
        if !served {
            continue;
        }
        let target = headers::component(component);
        let family = FAMILIES
            .iter()
            .find(|family| family.recognizes(&target))
            .unwrap_or_else(|| panic!("{name}: a family recognizes the served target"));
        let definition = family
            .inspect(&target, None, identity)
            .unwrap_or_else(|error| panic!("{name}: its family inspects it: {error}"));
        crate::operators::admit(&definition, false).unwrap_or_else(|error| panic!("{name}: admitted: {error}"));
        // Only a target that routes experts streams them.
        let routes = definition
            .decoder
            .sublayers()
            .any(|(_, sublayer)| crate::operators::routed::routed_ffn(&sublayer.op).is_some());
        if !routes {
            continue;
        }
        let manifest = target_manifest(component, &target, identity);
        for backend in BackendName::ALL {
            let (reservation, host) = match bank_reservations(&definition, backend) {
                Ok(substrate) => substrate,
                Err(PlanError::Unstreamed(reason)) if refused_by_the_device(&reason) => continue,
                Err(error) => panic!("{name}: its bank's substrate on {}: {error}", backend.as_str()),
            };
            let load = plain_load(&manifest, &definition, backend)
                .unwrap_or_else(|error| panic!("{name}: its plain load plans on {}: {error}", backend.as_str()));
            let banked = load
                .bank_floor(&definition, 1, LIMITS.flights(), reservation, host)
                .and_then(|floor| load.with_expert_bank(&definition, floor, 1, LIMITS.flights(), reservation, host))
                .unwrap_or_else(|error| panic!("{name}: its bank plans on {}: {error}", backend.as_str()));
            let bank = banked.expert_bank().expect("a banked load has a bank");
            for group in bank.groups() {
                for (_, tensor) in group.record.iter() {
                    let form = (
                        backend,
                        tensor.source.name().to_owned(),
                        tensor.resident.name().to_owned(),
                        tensor.matrix,
                    );
                    forms.entry(form).or_default().push(name.to_owned());
                }
            }
        }
    }
    assert!(
        forms.keys().any(|(backend, _, resident, _)| *backend == BackendName::Cuda && resident.ends_with("@mma16")),
        "some catalog target streams mma16 records on CUDA: {forms:?}"
    );
    for ((backend, source, resident, [n, k]), targets) in &forms {
        let (source, resident) = (
            seismic::Element::named(source).unwrap(),
            seismic::Element::named(resident).unwrap(),
        );
        let context = format!("{} {} -> {} [{n}, {k}] of {targets:?}", backend.as_str(), source.name(), resident.name());
        // A record resident as stored lands as its bytes.
        if source == resident {
            println!("{context}: resident as stored");
            continue;
        }
        let conversion = seismic::cpu::repack::resolve(source.name(), resident.name())
            .unwrap_or_else(|| panic!("{context}: no host conversion into a row layout"));
        let (n, k) = (*n as usize, *k as usize);
        let (tile, row) = (conversion.tile(), conversion.source_row_bytes(k));
        let bytes = (0..n * row)
            .map(|index| (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29) as u8)
            .collect();
        let stored = resident.repack_source(source, bytes).unwrap();
        let expected = resident
            .repack_reference(source, &[n as u64, k as u64], &stored)
            .unwrap_or_else(|| panic!("{context}: no reference"));
        let largest = ((128 << 10) / (tile * row)).max(1);
        for tiles in [1, largest] {
            let mut converted = vec![0xa5; expected.len()];
            let pieces = std::num::NonZeroUsize::new(tiles).unwrap();
            for (from, to) in conversion.pieces(k, n, pieces).expect("a stored row") {
                conversion
                    .piece(k, &stored[from], &mut converted[to])
                    .unwrap_or_else(|| panic!("{context}: a piece of {tiles} tiles"));
            }
            assert!(
                converted == expected,
                "{context}: pieces of {tiles} tiles differ from the reference"
            );
        }
        println!("{context}: equal");
    }
}

/// The address reservation a recorded machine's executor of `backend`
/// makes for an expert bank (`seismic::fixtures::recorded`): the first that
/// reserves; none where none reserves, and why where one would but cannot.
fn recorded_reservation(
    backend: BackendName,
) -> Result<Option<(seismic::Reservation, Option<crate::planning::HostTier>)>, seismic::Unreservable> {
    use seismic::fixtures::recorded;
    for topology in [recorded::m4_pro(), recorded::rtx3090()] {
        for executor in topology.executors().iter().filter(|executor| executor.backend() == backend) {
            let contract = crate::planning::BankTier::Bank.contract(&topology, executor);
            if let Some(reservation) = executor.reserving(contract).map_err(Clone::clone)? {
                let host = crate::platform::host_tier(&topology, executor).expect("a recorded machine reads how it reserves host storage");
                return Ok(Some((reservation, host)));
            }
        }
    }
    Ok(None)
}

/// The assessment terms `load` of `definition` derives on `backend`: its
/// decode demand under both native codecs, its memory terms and every graph
/// it prepares, summarized; or why one fails. Admission binds every weight
/// to its entries' ports, a second-level scale included
/// (`PlanError::UnportedScale`).
fn derive_terms(
    definition: &magnitude_family_contracts::ModelDefinition,
    load: &ModelLoadPlan,
    selection: ComponentSelection,
    backend: BackendName,
) -> Result<String, String> {
    load.program_plan(definition, KvCodec::AffineK8V4)
        .map_err(|error| error.to_string())?;
    // Each call's launches are its entry's declaration's at the statics the
    // load prepares it with.
    let demands = [KvCodec::Dense, KvCodec::AffineK8V4]
        .into_iter()
        .map(|codec| {
            let program = load
                .program_plan(definition, codec)
                .map_err(|error| AssessmentError::Demand(error.to_string()))?;
            let declarations = Declarations::of_load(
                backend,
                &program,
                definition,
                load,
                crate::native::TuningLimits::of(LIMITS, definition),
            )?;
            let demand = DecodeDemand::from_model(definition, load, codec, &declarations)?;
            // Every call's launches at one and at the decode rows' most.
            for call in &demand.calls {
                for rows in [1, crate::operators::attention::graph::DECODE_ROWS] {
                    call.launches.at(rows)?;
                }
            }
            Ok(demand)
        })
        .collect::<Result<Vec<_>, AssessmentError>>();
    let memory = AssessmentMemoryTerms::derive(
        definition,
        load,
        selection,
        KvCodec::AffineK8V4,
        PlannedMethod::Plain,
        LIMITS,
    );
    // Building every graph the load prepares proves no call lies outside
    // its kernel's domain on the backend.
    let graphs = ResourcePlanner::state_plan(
        definition,
        load,
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
            definition,
            load,
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
            Ok(format!(
                "{} calls, {} weight bytes",
                demands[0].calls.len(),
                demands[0]
                    .calls
                    .iter()
                    .flat_map(|call| &call.work.weights)
                    .map(|bytes| bytes.bytes)
                    .sum::<u64>()
            ))
        }
        (Ok(demands), Ok(_), Ok(_)) => Err(format!("an empty decode step: {demands:?}")),
        (demands, memory, graphs) => Err(format!(
            "{:?} / {:?} / {:?}",
            demands.err(),
            memory.err(),
            graphs.err()
        )),
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
            let load = ModelLoadPlan::derive(&manifest, &definition, selection, layout, method, LIMITS)
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
                                PlannedMethod::Plain,
                                LIMITS,
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
