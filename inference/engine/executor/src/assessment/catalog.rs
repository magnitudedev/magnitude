//! Every catalog target the executor admits derives complete assessment
//! terms on every backend: its decode demand under both native codecs, its
//! memory terms, and every graph a load prepares, so no call lies outside
//! its kernel's domain; where its experts stream on a backend, the same of
//! its load with them in an expert bank at its floor. A form the executor
//! does not run is refused with a typed reason before any term is derived;
//! a required target (one the catalog serves, or `REQUIRED` names) past
//! admission whose terms or graphs fail, or whose experts a backend that
//! streams experts refuses to stream, is a failure. Any other target may
//! lack a form on any backend: its failed
//! inspection, load or bank is listed as refused, and its terms only where
//! the backend has no native of an entry the load prepares; any other
//! failure of its terms fails.

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
/// support level. Only the derivation test reads it: what the catalog serves
/// ([`serves`]), with its drafts and projectors, is the support level's
/// alone. DeepSeek V4.1 Flash is disabled for users until it passes on real
/// text, but its loads plan on every backend, so a call outside a
/// declaration's domain at its shapes fails here, not at a device's load.
const REQUIRED: [&str; 2] = ["qwen3.8-flash-next:gguf:q4", "deepseek-v4.1-flash:gguf:mxfp4"];

/// Whether the catalog serves a model: its support level
/// (`catalog/models.json`) is neither disabled nor deprecated.
pub(super) fn serves() -> impl Fn(&str) -> bool + Sync {
    let levels = support_levels();
    move |name| !matches!(levels.get(name).map(String::as_str), Some("disabled" | "deprecated"))
}

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
pub(super) const LIMITS: ResourceLimits = ResourceLimits {
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

/// Every catalog target the catalog serves ([`serves`]) and the executor
/// admits, by catalog model ID, with its definition and its package alone.
pub(super) fn served_targets() -> Vec<(String, magnitude_family_contracts::ModelDefinition, PackageManifest)> {
    let identity = PackageIdentity { target: ArtifactIdentity([7; 32]), projector: None };
    let served = serves();
    let mut targets = Vec::new();
    for component in headers::index() {
        let name = component.model.as_str();
        if component.role != ComponentRole::Target || !served(name) {
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
        targets.push((name.to_owned(), definition, target_manifest(&component.path, &target, identity)));
    }
    targets
}

/// Pinned artifact `name`'s target, as [`served_targets`] gives a catalog
/// target's.
pub(super) fn artifact_target(name: &str) -> (String, magnitude_family_contracts::ModelDefinition, PackageManifest) {
    let identity = PackageIdentity { target: ArtifactIdentity([7; 32]), projector: None };
    let target = headers::artifact(name);
    let family = FAMILIES
        .iter()
        .find(|family| family.recognizes(&target))
        .unwrap_or_else(|| panic!("{name}: a family recognizes the target"));
    let definition = family
        .inspect(&target, None, identity)
        .unwrap_or_else(|error| panic!("{name}: its family inspects it: {error}"));
    (name.to_owned(), definition, target_manifest(Path::new(name), &target, identity))
}

/// The catalog's targets on `backend`, each derived on its own thread: a
/// backend's test names it (`…_on_vulkan`), so a run selects its backend,
/// and its targets cost the slowest target's derivation, not their sum.
fn every_admitted_catalog_target_derives_complete_terms_on(backend: BackendName) {
    let served = serves();
    let required = |name: &str| served(name) || REQUIRED.contains(&name);
    let targets = headers::index()
        .iter()
        .filter(|component| component.role == ComponentRole::Target)
        .collect::<Vec<_>>();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
    let derived = std::thread::scope(|scope| {
        let workers = (0..threads.min(targets.len()))
            .map(|_| {
                scope.spawn(|| {
                    let mut derived = Vec::new();
                    while let Some(component) = targets.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) {
                        derived.push((component.model.as_str(), derive_target(component, backend, required(component.model.as_str()))));
                    }
                    derived
                })
            })
            .collect::<Vec<_>>();
        workers.into_iter().flat_map(|worker| worker.join().unwrap()).collect::<BTreeMap<_, _>>()
    });
    let lines = |pick: fn(&Derived) -> &Vec<String>| derived.values().flat_map(pick).cloned().collect::<Vec<_>>();
    let (admitted, refused, failures) = (lines(|d| &d.admitted), lines(|d| &d.refused), lines(|d| &d.failures));
    println!("refused:\n  {}", refused.join("\n  "));
    assert!(failures.is_empty(), "failing targets:\n  {}", failures.join("\n  "));
    assert!(!admitted.is_empty());
    // Every required target is in the catalog's headers and admitted on the
    // backend.
    for name in REQUIRED {
        assert!(derived.contains_key(name), "required target {name} has no catalog header");
        let prefix = format!("{name} {} resident:", backend.as_str());
        assert!(
            admitted.iter().any(|line| line.starts_with(&prefix)),
            "required target {name} is not admitted on {}",
            backend.as_str()
        );
    }
}

#[test]
fn every_admitted_catalog_target_derives_complete_terms_on_cpu() {
    every_admitted_catalog_target_derives_complete_terms_on(BackendName::Cpu);
}

#[test]
fn every_admitted_catalog_target_derives_complete_terms_on_metal() {
    every_admitted_catalog_target_derives_complete_terms_on(BackendName::Metal);
}

#[test]
fn every_admitted_catalog_target_derives_complete_terms_on_cuda() {
    every_admitted_catalog_target_derives_complete_terms_on(BackendName::Cuda);
}

#[test]
fn every_admitted_catalog_target_derives_complete_terms_on_vulkan() {
    every_admitted_catalog_target_derives_complete_terms_on(BackendName::Vulkan);
}

/// What one target's loads on one backend came to: lines admitted, refused
/// and failing.
#[derive(Default)]
struct Derived {
    admitted: Vec<String>,
    refused: Vec<String>,
    failures: Vec<String>,
}

/// Catalog target `component`'s resident load on `backend`, and its banked
/// load where its experts stream there, derived to complete terms; `required`
/// where its terms must derive (the catalog serves it, or `REQUIRED` names
/// it).
fn derive_target(component: &CatalogComponent, backend: BackendName, required: bool) -> Derived {
    let identity = PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    };
    let mut derived = Derived::default();
    let Derived { admitted, refused, failures } = &mut derived;
    let name = component.model.as_str();
    let target = headers::component(component);
    let Some(family) = FAMILIES.iter().find(|family| family.recognizes(&target)) else {
        if required {
            failures.push(format!("{name}: no family recognizes the target"));
        }
        return derived;
    };
    let definition = match family.inspect(&target, None, identity) {
        Ok(definition) => definition,
        Err(error) => {
            sort(format!("{name}: its family fails to inspect it: {error}"), required, failures, refused);
            return derived;
        }
    };
    if let Err(error) = crate::operators::admit(&definition, false) {
        refused.push(format!("{name}: {error}"));
        return derived;
    }
    let manifest = target_manifest(&component.path, &target, identity);
    let load = match plain_load(&manifest, &definition, backend) {
        Ok(load) => load,
        // A required target derives its load on every backend; another
        // may lack a resident form.
        Err(error) => {
            sort(format!("{name} {}: {error}", backend.as_str()), required, failures, refused);
            return derived;
        }
    };
    // Experts that stream on the backend are planned in its bank too, at its
    // floor: what the most constrained streamed load holds. Experts that do
    // not stream are refused with their reason.
    let banked = match bank_reservations(&definition, backend) {
        Ok(substrate) => Some(load.bank_floor(&definition, 1, LIMITS.flights(), substrate).and_then(|floor| {
            load.clone().with_expert_bank(&definition, floor, 1, LIMITS.flights(), substrate)
        })),
        // A required target that routes experts streams them on every backend
        // whose recorded device streams experts at all: only a fact of the
        // device or its backend refuses it.
        Err(PlanError::Unstreamed(reason)) => {
            let line = format!("{name} {} banked: {reason}", backend.as_str());
            match required && !matches!(reason, UnstreamedExperts::Unrouted) && !refused_by_the_device(&reason) {
                true => failures.push(line),
                false => refused.push(line),
            }
            None
        }
        Err(error) => {
            sort(format!("{name} {} banked: {error}", backend.as_str()), required, failures, refused);
            None
        }
    };
    // Every routed form streams: a bank that refuses one fails.
    let loads = std::iter::once(("resident", Ok(load))).chain(banked.map(|banked| ("banked", banked)));
    for (form, load) in loads {
        let line = |error: &dyn std::fmt::Display| format!("{name} {} {form}: {error}", backend.as_str());
        match load.map(|load| derive_terms(&definition, &load, SELECTION, backend)) {
            Err(error) => sort(line(&error), required, failures, refused),
            Ok(Ok(terms)) => admitted.push(line(&terms)),
            // Only a missing native refuses a target that is not required;
            // any other failure of its terms fails.
            Ok(Err(underived)) => match !required && underived.unoffered {
                true => refused.push(line(&underived.line)),
                false => failures.push(line(&underived.line)),
            },
        }
    }
    derived
}

/// Lists what a target fails: a failure where it is required, a refusal
/// where it is not.
fn sort(line: String, required: bool, failures: &mut Vec<String>, refused: &mut Vec<String>) {
    match required {
        true => failures.push(line),
        false => refused.push(line),
    }
}

/// Whether `reason` is a fact of the device or its backend, which refuses
/// every model alike, rather than of the model.
pub(super) fn refused_by_the_device(reason: &UnstreamedExperts) -> bool {
    use UnstreamedExperts::*;
    matches!(reason, Backend(_) | Unreserved(_) | Unreservable { .. } | Untiered { .. })
}

/// The components a plain load of a target selects: neither an embedded
/// head nor vision.
pub(super) const SELECTION: ComponentSelection = ComponentSelection {
    head: false,
    vision: false,
};

/// The target with its draft head, as a load that drafts through it selects
/// it.
const HEAD_SELECTION: ComponentSelection = ComponentSelection { head: true, vision: false };

/// The package of catalog target `component` alone, its directory
/// `target`.
pub(super) fn target_manifest(
    path: &Path,
    target: &magnitude_artifacts::gguf::Directory,
    identity: PackageIdentity,
) -> PackageManifest {
    PackageManifest {
        identity,
        target: ComponentManifest {
            files: vec![ComponentFile {
                path: path.to_owned(),
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
pub(super) fn plain_load(
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
pub(super) fn bank_reservations(
    definition: &magnitude_family_contracts::ModelDefinition,
    backend: BackendName,
) -> Result<crate::planning::BankSubstrate, PlanError> {
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
    let served = serves();
    // Each form on each backend, and the targets that stream it.
    let mut forms = BTreeMap::<(BackendName, String, String, [u64; 2]), Vec<String>>::new();
    for component in headers::index() {
        if component.role != ComponentRole::Target {
            continue;
        }
        let name = component.model.as_str();
        if !served(name) {
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
        let manifest = target_manifest(&component.path, &target, identity);
        for backend in BackendName::ALL {
            let substrate = match bank_reservations(&definition, backend) {
                Ok(substrate) => substrate,
                Err(PlanError::Unstreamed(reason)) if refused_by_the_device(&reason) => continue,
                Err(error) => panic!("{name}: its bank's substrate on {}: {error}", backend.as_str()),
            };
            let load = plain_load(&manifest, &definition, backend)
                .unwrap_or_else(|error| panic!("{name}: its plain load plans on {}: {error}", backend.as_str()));
            let banked = load
                .bank_floor(&definition, 1, LIMITS.flights(), substrate)
                .and_then(|floor| load.with_expert_bank(&definition, floor, 1, LIMITS.flights(), substrate))
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
        let (tile, record) = (conversion.tile(), conversion.source_bytes(n, k));
        let bytes = (0..record)
            .map(|index| (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29) as u8)
            .collect();
        let stored = resident.repack_source(source, &[n as u64, k as u64], bytes).unwrap();
        let expected = resident
            .repack_reference(source, &[n as u64, k as u64], &stored)
            .unwrap_or_else(|| panic!("{context}: no reference"));
        let largest = ((128 << 10) / conversion.source_bytes(tile, k)).max(1);
        for tiles in [1, largest] {
            let mut converted = vec![0xa5; expected.len()];
            let pieces = std::num::NonZeroUsize::new(tiles).unwrap();
            for (rows, to) in conversion.pieces(k, n, pieces).expect("a stored row") {
                conversion
                    .piece(k, n, rows, &stored, &mut converted[to])
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
) -> Result<String, Underived> {
    load.program_plan(definition, KvCodec::AffineK8V4)
        .map_err(|error| Underived::failed(error.to_string()))?;
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
        (Ok(demands), Ok(_), Ok(_)) => Err(Underived::failed(format!("an empty decode step: {demands:?}"))),
        (demands, memory, graphs) => {
            let demands = demands.err();
            // The memory and graph terms read the same kernel inventory, so
            // a missing native fails them too.
            let unoffered = matches!(
                demands,
                Some(
                    AssessmentError::Unimplemented { .. }
                        | AssessmentError::Catalog(crate::CatalogError {
                            failure: crate::CatalogFailure::KernelDomain { .. },
                            ..
                        })
                )
            );
            Err(Underived {
                line: format!("{demands:?} / {:?} / {:?}", memory.err(), graphs.err()),
                unoffered,
            })
        }
    }
}

/// Why a load's terms did not derive.
struct Underived {
    line: String,
    /// The load's demand names an entry the backend has no native of: no
    /// configuration admits its statics, or the backend implements none.
    unoffered: bool,
}

impl Underived {
    fn failed(line: String) -> Self {
        Self { line, unoffered: false }
    }
}

/// The catalog's separate draft of catalog model `model`.
pub(super) const fn catalog_draft(model: &'static str) -> Header {
    Header::Catalog(model, ComponentRole::Draft)
}

/// Catalog model `model`'s target.
const fn catalog_target(model: &'static str) -> Header {
    Header::Catalog(model, ComponentRole::Target)
}

/// Separate drafts of the qualification artifacts (target header, draft
/// header) and their target families: the catalog's, and pinned artifacts of
/// architectures the catalog does not list. Each plans its draft program and
/// charges its weights, state and every draft graph class from headers
/// alone, on every backend: the draft graphs are sealed exactly, in
/// metadata, for the planned method's proposals.
const SEPARATE_DRAFTS: [(Header, Header, &dyn ModelFamily); 6] = [
    (
        catalog_target("qwen3.6-35b-a3b:gguf:q4"),
        catalog_draft("qwen3.6-35b-a3b:gguf:q4"),
        &Qwen35Family,
    ),
    (
        Header::Artifact("lfm2.5-2.6b-q4"),
        Header::Artifact("lfm2.5-2.6b-dspark"),
        &Lfm2Family,
    ),
    (
        catalog_target("qwen3.8-27b:gguf:q4"),
        Header::Artifact("qwen3.8-27b-dspark"),
        &Qwen35Family,
    ),
    // The catalog's DFlash2 draft.
    (
        catalog_target("qwen3.8-27b:gguf:q4"),
        catalog_draft("qwen3.8-27b:gguf:q4"),
        &Qwen35Family,
    ),
    // An NVFP4 draft (scaled fusion and feed-forward) beside a Q4_K_M
    // target, and beside the NVFP4 target (a scaled vocabulary projection).
    (
        Header::Artifact("nemotron-3.5-lightning-30b-a3b-q4"),
        Header::Artifact("nemotron-3.5-lightning-30b-a3b-dflash"),
        &NemotronHFamily,
    ),
    (
        Header::Artifact("nemotron-3.5-lightning-30b-a3b-nvfp4"),
        Header::Artifact("nemotron-3.5-lightning-30b-a3b-dflash"),
        &NemotronHFamily,
    ),
];

/// Synthetic graph-planning fixtures retain captured tensor geometry and supply
/// explicit attention semantics for each named draft. These do not establish
/// admission of the captured artifacts; binary-header admission is qualified
/// separately by the DFlash family's publication test.
pub(super) fn synthetic_separate_draft_directory(draft: Header) -> magnitude_artifacts::gguf::Directory {
    use magnitude_artifacts::gguf::{Metadata, Value};
    let causal: &[bool] = match draft {
        Header::Catalog("qwen3.6-35b-a3b:gguf:q4", ComponentRole::Draft) => {
            &[true, true, true, true, true, false]
        }
        Header::Artifact("lfm2.5-2.6b-dspark") => &[false; 5],
        Header::Artifact("qwen3.8-27b-dspark") => &[false; 5],
        Header::Catalog("qwen3.8-27b:gguf:q4", ComponentRole::Draft) => &[false; 5],
        Header::Artifact("nemotron-3.5-lightning-30b-a3b-dflash") => &[false; 6],
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
            (&target.to_string(), &target.directory()),
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
/// Every served target that has a catalog draft ([`served_targets`]), with
/// its definition bound with the draft ([`bind_catalog_draft`]: a separate
/// draft or a draft head) and its package carrying the draft's file.
pub(super) fn served_drafted_targets() -> Vec<(String, magnitude_family_contracts::ModelDefinition, PackageManifest)> {
    served_targets()
        .into_iter()
        .filter_map(|(name, declared, manifest)| {
            let component = headers::index()
                .iter()
                .find(|component| component.model.as_str() == name && component.role == ComponentRole::Draft)?;
            let draft = headers::component(component);
            let family = FAMILIES
                .iter()
                .find(|family| family.recognizes(&headers::target(&name)))
                .unwrap_or_else(|| panic!("{name}: a family recognizes the served target"));
            let definition = bind_catalog_draft(*family, &declared, &draft)
                .unwrap_or_else(|error| panic!("{name}: its draft binds: {error}"));
            let manifest = PackageManifest {
                draft: Some(ComponentManifest {
                    files: vec![ComponentFile {
                        path: component.path.clone(),
                        size: draft.tensors.iter().map(|tensor| tensor.nbytes).sum(),
                    }],
                    identity: ArtifactIdentity([9; 32]),
                    tensors: draft.tensors.clone(),
                }),
                ..manifest
            };
            Some((name, definition, manifest))
        })
        .collect()
}

pub(super) fn bind_catalog_draft(
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

/// How the outlines of a test of graph traces were answered, and how long
/// each took: from a trace recorded before, or by recording one.
#[derive(Default)]
struct TraceTally {
    answered: (u32, std::time::Duration),
    recorded: (u32, std::time::Duration),
}

impl TraceTally {
    /// `outline`'s result, tallied as answered where the family's count of
    /// traces recorded (`traces`) did not move, else as recorded.
    fn outline<T>(&mut self, traces: impl Fn() -> usize, outline: impl FnOnce() -> T) -> T {
        let before = traces();
        let began = std::time::Instant::now();
        let outlined = outline();
        let elapsed = began.elapsed();
        let tally = if traces() == before { &mut self.answered } else { &mut self.recorded };
        *tally = (tally.0 + 1, tally.1 + elapsed);
        outlined
    }

    /// Print the tally over `rows` rows, `traces` traces recorded in all.
    fn report(&self, rows: usize, traces: usize) {
        let mean = |(count, total): (u32, std::time::Duration)| total.checked_div(count.max(1)).unwrap_or_default();
        println!(
            "{rows} rows; {traces} traces recorded for {} outlines ({:?} each), {} outlines answered by a trace ({:?} each)",
            self.recorded.0,
            mean(self.recorded),
            self.answered.0,
            mean(self.answered)
        );
    }
}

/// The state `load` of `definition` plans for `method` at the standard
/// service's limits, its domain unbounded for the purpose.
fn state_at_limits(
    definition: &magnitude_family_contracts::ModelDefinition,
    load: &ModelLoadPlan,
    method: PlannedMethod,
) -> Result<crate::StateResourcePlan, String> {
    ResourcePlanner::state_plan(
        definition,
        load,
        method,
        KvCodec::AffineK8V4,
        LIMITS,
        ResourceCapacity { domain_bytes: 64 << 30, tensor_operations: crate::TensorOperations::Absent },
    )
}

/// What a test of graph traces compares of each graph class: its storage,
/// plan heap and bound heap.
fn class_terms<'a, C: crate::programs::graph::draft::Certified + 'a>(
    certified: impl IntoIterator<Item = &'a C>,
) -> Vec<(seismic::NativeGraphStorageBytes, seismic::HostHeap, seismic::HostHeap)> {
    certified.into_iter().map(|class| (class.storage_bytes(), class.plan_heap(), class.bound_heap())).collect()
}

/// A served draft head's load at the standard service's limits on
/// `backend`, its state and its program plan, for the tests of its graph
/// traces; none where the target has no resident form there.
fn head_plans(
    name: &str,
    definition: &magnitude_family_contracts::ModelDefinition,
    manifest: &PackageManifest,
    backend: BackendName,
    method: PlannedMethod,
) -> Option<(ModelLoadPlan, crate::StateResourcePlan, crate::ProgramPlan)> {
    let layout = resident_layout(ExecutionPath::Native, backend);
    let load = ModelLoadPlan::derive(manifest, definition, HEAD_SELECTION, layout, method, LIMITS).ok()?;
    let line = format!("{name} {}", backend.as_str());
    let state = state_at_limits(definition, &load, method).unwrap_or_else(|error| panic!("{line}: its state plans: {error}"));
    let program = load
        .program_plan(definition, KvCodec::AffineK8V4)
        .unwrap_or_else(|error| panic!("{line}: its program plan: {error}"));
    Some((load, state, program))
}

/// Every served target's heads: each target's own (from its file alone)
/// and its catalog draft's ([`served_drafted_targets`]), each named by the
/// component it comes from.
fn served_heads() -> impl Iterator<Item = (&'static str, (String, magnitude_family_contracts::ModelDefinition, PackageManifest))> {
    (served_targets().into_iter().map(|target| ("target", target)))
        .chain(served_drafted_targets().into_iter().map(|drafted| ("draft", drafted)))
}

/// A draft head's method in the tests of its graph traces.
const HEAD_METHOD: PlannedMethod = PlannedMethod::Mtp {
    greedy_proposals: 1,
    sampled_proposals: 1,
    widths: crate::DraftWidths::Adaptive,
};

/// Every served target's state copy graphs, outlined from the family's graph
/// traces (`seismic::trace`: recorded once per topology, answering every
/// later model its guards and shapes admit), hold exactly what the load's
/// layouts of the same classes hold, layout by layout, on `backend` at
/// the standard service's limits: storage, census and both host heaps. How
/// many traces were recorded for how many outlines, and the time of an
/// outline a trace answered against one it recorded, are printed.
fn every_served_target_outlines_its_state_copies_from_traces(backend: BackendName) {
    use crate::programs::native_state::{certify_copy_family, copy_traces_recorded, state_copy_classes};
    let row_classes = magnitude_batching::row_classes(LIMITS.max_launch_rows)
        .into_iter()
        .map(|rows| rows as u64)
        .collect::<Vec<_>>();
    let (mut rows, mut failures, mut tally) = (0, Vec::new(), TraceTally::default());
    for (name, definition, manifest) in served_targets() {
        // A target need not have a resident form on every backend.
        let Ok(load) = plain_load(&manifest, &definition, backend) else {
            continue;
        };
        let line = format!("{name} {}", backend.as_str());
        let state = state_at_limits(&definition, &load, PlannedMethod::Plain)
            .unwrap_or_else(|error| panic!("{line}: its state plans: {error}"));
        let classes = state_copy_classes(state.target_state(), state.head_state(), &row_classes)
            .unwrap_or_else(|error| panic!("{line}: its copy classes: {error}"));
        let outlined = tally.outline(copy_traces_recorded, || certify_copy_family::<seismic::NativeGraphOutline>(backend, &classes));
        let certified = certify_copy_family::<seismic::NativeGraphLayout>(backend, &classes);
        rows += 1;
        match (outlined, certified) {
            (Ok((storage, graphs, outlines)), Ok((held, recorded_graphs, layouts))) => {
                let (outlined, laid) = (class_terms(&outlines), class_terms(&layouts));
                if storage != held || graphs != recorded_graphs || outlined != laid {
                    failures.push(format!("{line}: outlined {storage:?} {outlined:?}, laid out {held:?} {laid:?}"));
                }
            }
            (outlined, certified) => failures.push(format!(
                "{line}: outlined {:?}, laid out {:?}",
                outlined.err(),
                certified.err()
            )),
        }
    }
    tally.report(rows, copy_traces_recorded());
    assert!(rows > 0);
    assert!(failures.is_empty(), "{} of {rows} rows differ:\n{}", failures.len(), failures.join("\n"));
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_target_outlines_its_state_copies_from_traces_on_cpu() {
    every_served_target_outlines_its_state_copies_from_traces(BackendName::Cpu);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_target_outlines_its_state_copies_from_traces_on_metal() {
    every_served_target_outlines_its_state_copies_from_traces(BackendName::Metal);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_target_outlines_its_state_copies_from_traces_on_cuda() {
    every_served_target_outlines_its_state_copies_from_traces(BackendName::Cuda);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_target_outlines_its_state_copies_from_traces_on_vulkan() {
    every_served_target_outlines_its_state_copies_from_traces(BackendName::Vulkan);
}

/// Every causal draft head of a served target ([`served_heads`]: its own
/// or its catalog draft's) outlines its head graphs
/// from the head's traces exactly as its own recording lays them out, on
/// `backend` where it loads: per row, the family's storage and each class's
/// storage, plan heap and bound heap. The counts of traces recorded, and the
/// time of an outline a trace answered against one it recorded, are printed.
fn every_served_head_outlines_its_graphs_from_traces(backend: BackendName) {
    use crate::programs::native_head::{certify_head_family, head_graph_classes, head_traces_recorded};
    let (mut rows, mut failures, mut tally) = (0, Vec::new(), TraceTally::default());
    for (component, (name, definition, manifest)) in served_heads() {
        let Some(head) = definition.head.as_ref() else {
            continue;
        };
        if !matches!(crate::operators::head_form(head, &definition.decoder), Ok(crate::operators::HeadForm::Causal)) {
            continue;
        }
        let head_block = crate::operators::CausalHeadBlock::of(head, &head.blocks[0])
            .unwrap_or_else(|error| panic!("{name}: its head block: {error}"));
        // A target need not have a resident form on every backend.
        let Some((load, state, program)) = head_plans(&name, &definition, &manifest, backend, HEAD_METHOD) else {
            continue;
        };
        let line = format!("{name} {component} {}", backend.as_str());
        let (Some(head_program), Some(head_state)) = (program.head(), state.head_state()) else {
            panic!("{line}: a selected causal head plans no head program or state");
        };
        let history = head_state.sole_history().unwrap_or_else(|error| panic!("{line}: {error}"));
        let classes = head_graph_classes(
            LIMITS,
            history.rows as u64,
            history.slab_rows,
            head_state.span_limit(),
            HEAD_METHOD.draft_rows(),
        )
        .unwrap_or_else(|error| panic!("{line}: its head classes: {error}"));
        let binding = head_program.blocks()[0];
        let outlined = tally.outline(head_traces_recorded, || {
            certify_head_family::<seismic::NativeGraphOutline>(backend, &load, &definition.decoder, head_block, binding, &classes)
                .map(|(storage, outlines)| (storage, class_terms(outlines.values())))
        });
        let laid = certify_head_family::<seismic::NativeGraphLayout>(backend, &load, &definition.decoder, head_block, binding, &classes)
            .map(|(storage, layouts)| (storage, class_terms(layouts.values())));
        rows += 1;
        match (outlined, laid) {
            (Ok(outlined), Ok(laid)) if outlined == laid => {}
            (outlined, laid) => failures.push(format!("{line}: outlined {outlined:?}, laid out {laid:?}")),
        }
    }
    tally.report(rows, head_traces_recorded());
    assert!(rows > 0);
    assert!(failures.is_empty(), "{} of {rows} rows differ:\n{}", failures.len(), failures.join("\n"));
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_head_outlines_its_graphs_from_traces_on_cpu() {
    every_served_head_outlines_its_graphs_from_traces(BackendName::Cpu);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_head_outlines_its_graphs_from_traces_on_metal() {
    every_served_head_outlines_its_graphs_from_traces(BackendName::Metal);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_head_outlines_its_graphs_from_traces_on_cuda() {
    every_served_head_outlines_its_graphs_from_traces(BackendName::Cuda);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_head_outlines_its_graphs_from_traces_on_vulkan() {
    every_served_head_outlines_its_graphs_from_traces(BackendName::Vulkan);
}

/// Every shared draft head of a served target ([`served_heads`]: a shared
/// head is a draft component's, so it comes from a catalog draft)
/// outlines its head graphs
/// from the shared head's traces exactly as its own recording lays them
/// out, on `backend` where it loads (as
/// `every_served_head_outlines_its_graphs_from_traces`).
fn every_served_shared_head_outlines_its_graphs_from_traces(backend: BackendName) {
    use crate::programs::native_shared_head::{
        SharedHeadGeometry, certify_shared_head_family, shared_head_graph_classes, shared_head_traces_recorded,
    };
    let (mut rows, mut failures, mut tally) = (0, Vec::new(), TraceTally::default());
    for (component, (name, definition, manifest)) in served_heads() {
        let Some(head) = definition.head.as_ref() else {
            continue;
        };
        if !matches!(crate::operators::head_form(head, &definition.decoder), Ok(crate::operators::HeadForm::Shared(_))) {
            continue;
        }
        // A target need not have a resident form on every backend.
        let Some((load, state, program)) = head_plans(&name, &definition, &manifest, backend, HEAD_METHOD) else {
            continue;
        };
        let line = format!("{name} {component} {}", backend.as_str());
        let shared = program
            .shared_head()
            .unwrap_or_else(|| panic!("{line}: a selected shared head plans no program"));
        let geometry = SharedHeadGeometry::new(&definition, &state, shared)
            .unwrap_or_else(|error| panic!("{line}: its shared head geometry: {error}"));
        let classes = shared_head_graph_classes(LIMITS, HEAD_METHOD.draft_rows())
            .unwrap_or_else(|error| panic!("{line}: its shared head classes: {error}"));
        let outlined = tally.outline(shared_head_traces_recorded, || {
            certify_shared_head_family::<seismic::NativeGraphOutline>(backend, &load, &geometry, classes.iter().copied())
                .map(|(storage, outlines)| (storage, class_terms(outlines.values())))
        });
        let laid = certify_shared_head_family::<seismic::NativeGraphLayout>(backend, &load, &geometry, classes.iter().copied())
            .map(|(storage, layouts)| (storage, class_terms(layouts.values())));
        rows += 1;
        match (outlined, laid) {
            (Ok(outlined), Ok(laid)) if outlined == laid => {}
            (outlined, laid) => failures.push(format!("{line}: outlined {outlined:?}, laid out {laid:?}")),
        }
    }
    tally.report(rows, shared_head_traces_recorded());
    assert!(rows > 0);
    assert!(failures.is_empty(), "{} of {rows} rows differ:\n{}", failures.len(), failures.join("\n"));
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_shared_head_outlines_its_graphs_from_traces_on_cpu() {
    every_served_shared_head_outlines_its_graphs_from_traces(BackendName::Cpu);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_shared_head_outlines_its_graphs_from_traces_on_metal() {
    every_served_shared_head_outlines_its_graphs_from_traces(BackendName::Metal);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_shared_head_outlines_its_graphs_from_traces_on_cuda() {
    every_served_shared_head_outlines_its_graphs_from_traces(BackendName::Cuda);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_shared_head_outlines_its_graphs_from_traces_on_vulkan() {
    every_served_shared_head_outlines_its_graphs_from_traces(BackendName::Vulkan);
}
