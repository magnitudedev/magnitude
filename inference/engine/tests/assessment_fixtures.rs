//! The shared real-header fixtures (`fixtures/assessment/`): every
//! fixture's expected verdict, asserted through the engine's public
//! assessment entry point on the CPU backend, which every build has. The
//! manifest format is described in the fixtures' README.

use flate2::read::GzDecoder;
use magnitude_artifacts::gguf::{Array, Directory, Metadata, Scalar, Value, inspect_header};
use magnitude_engine::{
    assessment::{
        AssessmentSetup, AssessmentStructure, DrafterVerdict, Measurements, ModelAssessment,
        ModelAssessmentError, ModelExecution, ModelPackagePaths, ModelResidency, Performance, PreparedModelAssessment,
        SpeculativeMethod, assess_model, finish_model_assessment,
        prepare_model_assessment,
    },
    error::UnsupportedReason,
    fingerprint::ModelFingerprint,
    options::{ModelMethod, ModelPolicy, PROFILE_SEQUENCES, standard_service_limits},
};
use magnitude_executor::{
    AssessmentFitVerdict, AssessmentMemoryCharge, ConveyorCarrier, HeldBound, PlanError, PlanPrices, PlanRoom, StableRoom, WeightBound,
    assessment::{AssessmentError, ColdSource, ExecutionFit, ExpertShares, Priced, Residency, execution_memory_charge, synthetic},
    platform::{DeviceRequest, DomainRole, MemoryReserves, StoreMotion},
};
use magnitude_scheduler::ServiceLimits;
use seismic::BackendName;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod recorded;
use recorded::{m4_pro_setup, rtx3090_setup};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    fixtures: Vec<Fixture>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    name: String,
    #[allow(dead_code)]
    description: String,
    target: Vec<Component>,
    #[serde(default)]
    draft: Option<Component>,
    #[serde(default)]
    method: Method,
    expected: Expected,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Component {
    header: String,
    #[allow(dead_code)]
    repository: String,
    #[allow(dead_code)]
    revision: String,
    path: String,
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Method {
    #[default]
    Auto,
    Plain,
    Mtp,
    DFlash,
    DSpark,
    DFlash2,
}

impl From<Method> for ModelMethod {
    fn from(method: Method) -> Self {
        match method {
            Method::Auto => Self::Auto,
            Method::Plain => Self::Plain,
            Method::Mtp => Self::Mtp,
            Method::DFlash => Self::DFlash,
            Method::DSpark => Self::DSpark,
            Method::DFlash2 => Self::DFlash2,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    target: TargetVerdict,
    #[serde(default)]
    drafter: Option<Drafter>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum TargetVerdict {
    Supported,
    Unsupported(Vec<UnsupportedReason>),
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Drafter {
    None,
    Bound(Bound),
    Rejected(Vec<UnsupportedReason>),
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Bound {
    Mtp,
    DFlash,
    DSpark,
    DFlash2,
    LiLiCorr,
}

impl From<SpeculativeMethod> for Bound {
    fn from(method: SpeculativeMethod) -> Self {
        match method {
            SpeculativeMethod::Mtp => Self::Mtp,
            SpeculativeMethod::DFlash => Self::DFlash,
            SpeculativeMethod::DSpark => Self::DSpark,
            SpeculativeMethod::DFlash2 => Self::DFlash2,
            SpeculativeMethod::LiLiCorr => Self::LiLiCorr,
        }
    }
}

/// Equal reasons, except that a `TargetMismatch` detail is developer text:
/// the manifest's detail need only occur in it.
fn reasons_match(actual: &[UnsupportedReason], expected: &[UnsupportedReason]) -> bool {
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|pair| match pair {
            (
                UnsupportedReason::TargetMismatch { detail },
                UnsupportedReason::TargetMismatch { detail: expected },
            ) => detail.contains(expected.as_str()),
            (actual, expected) => actual == expected,
        })
}

fn fixtures() -> PathBuf {
    Path::new(&std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"))
        .join("fixtures/assessment")
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "magnitude-assessment-fixtures-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    /// The component's header, decompressed under its repository file name
    /// so a split target's shards are found beside its first.
    fn materialize(&self, component: &Component) -> PathBuf {
        let source = fixtures().join(&component.header);
        let mut decoder = GzDecoder::new(
            std::fs::File::open(&source)
                .unwrap_or_else(|error| panic!("{}: {error}", source.display())),
        );
        let name = Path::new(&component.path).file_name().unwrap();
        let path = self.0.join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        std::io::copy(&mut decoder, &mut file).unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn cpu_structure() -> AssessmentStructure {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    AssessmentStructure::discover(
        &catalog,
        DeviceRequest::Backend(BackendName::Cpu),
        None,
        MemoryReserves::of_machine(None),
        ModelPolicy::default(),
        standard_service_limits(),
    )
    .unwrap_or_else(|error| panic!("the CPU backend is selectable: {error}"))
}

/// The CPU backend before any measurement: fit is assessed, speed is pending.
fn cpu_setup() -> AssessmentSetup {
    AssessmentSetup {
        structure: Arc::new(cpu_structure()),
        measurements: Measurements::Pending,
    }
}

/// Every Unsupported carries at least one reason, and so does a rejected
/// drafter.
fn check(fixture: &Fixture, assessment: &ModelAssessment) -> Result<(), String> {
    let mismatch = |actual: &dyn std::fmt::Debug| {
        Err(format!(
            "{}: expected {:?}, assessed {actual:?}",
            fixture.name, fixture.expected
        ))
    };
    match (&fixture.expected.target, assessment) {
        (
            TargetVerdict::Unsupported(expected),
            ModelAssessment::Unsupported { unsupported, .. },
        ) => {
            if !reasons_match(&unsupported.reasons, expected) {
                return mismatch(unsupported);
            }
            if fixture.expected.drafter.is_some() {
                return Err(format!(
                    "{}: an unsupported target has no drafter verdict",
                    fixture.name
                ));
            }
            Ok(())
        }
        (TargetVerdict::Supported, ModelAssessment::Assessed { facts, .. }) => {
            match (&fixture.expected.drafter, &facts.drafter) {
                (None | Some(Drafter::None), DrafterVerdict::None) => Ok(()),
                (Some(Drafter::Bound(expected)), DrafterVerdict::Bound { method, .. })
                    if Bound::from(*method) == *expected =>
                {
                    Ok(())
                }
                (Some(Drafter::Rejected(expected)), DrafterVerdict::Rejected(rejection))
                    if reasons_match(&rejection.reasons, expected) =>
                {
                    Ok(())
                }
                (_, actual) => mismatch(actual),
            }
        }
        (_, ModelAssessment::Unsupported { unsupported, .. }) => mismatch(unsupported),
        (_, ModelAssessment::Assessed { facts, .. }) => mismatch(&facts.drafter),
    }
}

#[test]
fn every_fixture_has_its_expected_verdict() {
    let manifest = manifest();
    assert!(!manifest.fixtures.is_empty());
    let setup = cpu_setup();
    let mut failures = Vec::new();
    for fixture in &manifest.fixtures {
        let scratch = Scratch::new(&fixture.name);
        let mut target = fixture
            .target
            .iter()
            .map(|component| scratch.materialize(component));
        let package = ModelPackagePaths {
            target: target.next().expect("a fixture has a target"),
            projector: None,
            draft: fixture
                .draft
                .as_ref()
                .map(|component| scratch.materialize(component)),
            method: fixture.method.into(),
        };
        target.for_each(drop);
        let assessment = assess_model(&package, &setup, &[], PROFILE_SEQUENCES)
            .unwrap_or_else(|error| panic!("{}: operational failure: {error}", fixture.name));
        if let ModelAssessment::Unsupported { unsupported, .. } = &assessment {
            assert!(
                !unsupported.reasons.is_empty(),
                "{}: {unsupported:?}",
                fixture.name
            );
        }
        if let ModelAssessment::Assessed { facts, .. } = &assessment {
            if let DrafterVerdict::Rejected(rejection) = &facts.drafter {
                assert!(
                    !rejection.reasons.is_empty(),
                    "{}: {rejection:?}",
                    fixture.name
                );
            }
        }
        if let Err(failure) = check(fixture, &assessment) {
            failures.push(failure);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn manifest() -> Manifest {
    serde_json::from_slice(
        &std::fs::read(fixtures().join("manifest.json")).expect("the manifest is committed"),
    )
    .expect("the manifest is well formed")
}

/// `directory` as a GGUF header without tensor payloads: unsigned, signed
/// and float values widened to 64 bits, tensors laid out in order.
fn gguf_header(directory: &Directory) -> Vec<u8> {
    fn string(out: &mut Vec<u8>, value: &str) {
        out.extend((value.len() as u64).to_le_bytes());
        out.extend(value.as_bytes());
    }
    fn kind(scalar: &Scalar) -> u32 {
        match scalar {
            Scalar::Bool(_) => 7,
            Scalar::String(_) => 8,
            Scalar::Unsigned(_) => 10,
            Scalar::Signed(_) => 11,
            Scalar::Float(_) => 12,
        }
    }
    fn scalar(out: &mut Vec<u8>, scalar: &Scalar) {
        match scalar {
            Scalar::Bool(value) => out.push(u8::from(*value)),
            Scalar::String(value) => string(out, value),
            Scalar::Unsigned(value) => out.extend(value.to_le_bytes()),
            Scalar::Signed(value) => out.extend(value.to_le_bytes()),
            Scalar::Float(value) => out.extend(value.to_le_bytes()),
        }
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((directory.tensors.len() as u64).to_le_bytes());
    out.extend((directory.metadata.len() as u64).to_le_bytes());
    for entry in &directory.metadata {
        string(&mut out, &entry.name);
        match &entry.value {
            Value::Scalar(value) => {
                out.extend(kind(value).to_le_bytes());
                scalar(&mut out, value);
            }
            Value::Array(values) => {
                out.extend(9u32.to_le_bytes());
                let element = values.iter().next().map(|value| kind(&value.to_scalar()));
                out.extend(element.unwrap_or(8).to_le_bytes());
                out.extend((values.len() as u64).to_le_bytes());
                for value in values.iter() {
                    scalar(&mut out, &value.to_scalar());
                }
            }
        }
    }
    let alignment = directory.alignment;
    let mut offset = 0u64;
    for tensor in &directory.tensors {
        string(&mut out, &tensor.name);
        out.extend((tensor.shape.len() as u32).to_le_bytes());
        for dimension in tensor.shape.iter().rev() {
            out.extend(dimension.to_le_bytes());
        }
        out.extend((tensor.encoding as u32).to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset += tensor.nbytes.div_ceil(alignment) * alignment;
    }
    out.resize(
        out.len().div_ceil(alignment as usize) * alignment as usize,
        0,
    );
    out
}

fn set(directory: &mut Directory, name: &str, value: Value) {
    let entry = directory
        .metadata
        .iter_mut()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("no metadata {name}"));
    entry.value = value;
}

/// A Qwen4 target that stores its prediction block after its decoder's, as
/// one file: the catalog target's shards joined with its separately packaged
/// head's block, declaring `nextn_predict_layers`. It is assessed as a target
/// whose built-in head drafts, with the fingerprint of the target alone.
#[test]
fn a_qwen4exp_target_with_its_prediction_block_is_assessed_with_built_in_heads() {
    let manifest = manifest();
    let fixture = |name: &str| {
        manifest
            .fixtures
            .iter()
            .find(|fixture| fixture.name == name)
            .unwrap_or_else(|| panic!("no fixture {name}"))
    };
    let separate = fixture("qwen4exp-mtp-head");
    let scratch = Scratch::new("qwen4exp-built-in-heads");
    let shards = separate
        .target
        .iter()
        .map(|component| inspect_header(scratch.materialize(component)).unwrap())
        .collect::<Vec<_>>();
    let head = inspect_header(scratch.materialize(separate.draft.as_ref().unwrap())).unwrap();

    let mut combined = shards[0].clone();
    combined
        .metadata
        .retain(|entry| !entry.name.starts_with("split."));
    combined.tensors = shards
        .iter()
        .flat_map(|shard| shard.tensors.clone())
        .collect();
    let blocks = combined
        .value("qwen4exp.block_count")
        .and_then(Value::unsigned)
        .unwrap();
    let prefix = format!("blk.{blocks}.");
    combined.tensors.extend(
        head.tensors
            .iter()
            .filter(|tensor| tensor.name.starts_with(&prefix))
            .cloned(),
    );
    set(
        &mut combined,
        "qwen4exp.block_count",
        Value::Scalar(Scalar::Unsigned(blocks + 1)),
    );
    combined.metadata.push(Metadata {
        name: "qwen4exp.nextn_predict_layers".into(),
        value: Value::Scalar(Scalar::Unsigned(1)),
    });
    let ratios = match combined.value("qwen4exp.attention.compress_ratios").cloned() {
        Some(Value::Array(Array::Signed(mut ratios))) => {
            ratios.push(0);
            Array::Signed(ratios)
        }
        Some(Value::Array(Array::Unsigned(mut ratios))) => {
            ratios.push(0);
            Array::Unsigned(ratios)
        }
        _ => panic!("the target names its compression ratios"),
    };
    set(
        &mut combined,
        "qwen4exp.attention.compress_ratios",
        Value::Array(ratios),
    );
    let target = scratch.0.join("Qwen3.8-Flash-Next-with-heads.gguf");
    std::fs::write(&target, gguf_header(&combined)).unwrap();

    let separate_target = scratch.materialize(&separate.target[0]);
    let ModelAssessment::Assessed { facts, .. } = assess_model(
        &ModelPackagePaths {
            target,
            projector: None,
            draft: None,
            method: ModelMethod::Auto,
        },
        &cpu_setup(),
        &[],
        PROFILE_SEQUENCES,
    )
    .unwrap_or_else(|error| panic!("operational failure: {error}")) else {
        panic!("a target with built-in heads is assessed");
    };
    assert!(
        matches!(
            facts.drafter,
            DrafterVerdict::Bound {
                method: SpeculativeMethod::Mtp,
                ..
            }
        ),
        "{:?}",
        facts.drafter
    );
    assert_eq!(
        facts.fingerprint.map(|fingerprint| fingerprint.block_count),
        ModelFingerprint::from_path(&separate_target)
            .ok()
            .map(|fingerprint| fingerprint.block_count)
    );
}

/// A separate Nemotron-H prediction-head file (as `mtp-*.gguf` beside a
/// Nemotron 3.5 repository's versions): a `nemotron_h_moe` file holding only
/// its last block. The engine runs the architecture but binds no separate
/// drafter of it, which is the reason, never the architecture itself.
#[test]
fn a_separate_nemotron_head_names_its_drafter_architecture() {
    let manifest = manifest();
    let fixture = manifest
        .fixtures
        .iter()
        .find(|fixture| fixture.name == "nemotron-h-supported")
        .expect("the Nemotron fixture");
    let scratch = Scratch::new("nemotron-separate-head");
    let target = scratch.materialize(&fixture.target[0]);
    let mut head = inspect_header(&target).unwrap();
    let blocks = head
        .value("nemotron_h_moe.block_count")
        .and_then(Value::unsigned)
        .unwrap();
    let last = format!("blk.{}.", blocks - 1);
    head.tensors.retain(|tensor| tensor.name.starts_with(&last));
    let draft = scratch.0.join("mtp-nemotron.gguf");
    std::fs::write(&draft, gguf_header(&head)).unwrap();
    let ModelAssessment::Assessed { facts, .. } = assess_model(
        &ModelPackagePaths {
            target,
            projector: None,
            draft: Some(draft),
            method: ModelMethod::Auto,
        },
        &cpu_setup(),
        &[],
        PROFILE_SEQUENCES,
    )
    .unwrap_or_else(|error| panic!("operational failure: {error}")) else {
        panic!("the Nemotron target is assessed");
    };
    let DrafterVerdict::Rejected(rejection) = facts.drafter else {
        panic!("a separate Nemotron head is rejected: {:?}", facts.drafter);
    };
    assert_eq!(
        rejection.reasons,
        [UnsupportedReason::DraftArchitecture {
            architecture: "nemotron_h_moe".into()
        }]
    );
}

/// The catalog's Qwen3.6 35B-A3B target, served plain.
fn qwen36_35b(scratch: &Scratch) -> ModelPackagePaths {
    let manifest = manifest();
    let fixture = manifest
        .fixtures
        .iter()
        .find(|fixture| fixture.name == "qwen35moe-supported")
        .expect("the Qwen3.6 35B fixture");
    ModelPackagePaths {
        target: scratch.materialize(&fixture.target[0]),
        projector: None,
        draft: None,
        method: ModelMethod::Plain,
    }
}

/// On a discrete device, whose bank records the host does not read into,
/// a group's reads land in the host tier's records (its lender's bottom
/// ones), which the rounds convert from over the link: no weight byte is a
/// conveyor's, so a weight cap's bytes go to slots. Qwen3.6 35B on the
/// recorded RTX 3090 under a 6,890,894,474-byte cap.
#[test]
fn a_discrete_banks_conveyor_lies_in_host_storage() {
    let scratch = Scratch::new("qwen36-discrete-conveyor");
    let package = qwen36_35b(&scratch);
    let cap = 6_890_894_474;
    let setup = rtx3090_setup(
        BackendName::Cuda,
        MemoryReserves::of_machine(None),
        ModelPolicy {
            weight_bound: Some(WeightBound::Cap(cap)),
            ..ModelPolicy::default()
        },
        standard_service_limits(),
    );
    let PreparedModelAssessment::Planned { definition, execution, .. } = prepare_model_assessment(&package, &setup.structure)
        .unwrap_or_else(|error| panic!("operational failure: {error}"))
    else {
        panic!("Qwen3.6 35B is planned streamed on the RTX 3090 under the cap");
    };
    let draft = execution.draft();
    let bank = draft.load().expert_bank().expect("a capped load streams").clone();
    assert!(!bank.reservation().host_addressed);
    let conveyor = bank.conveyor();
    let ConveyorCarrier::HostTier(lender) = conveyor.carrier else {
        panic!("a discrete device's conveyor lies in its host tier");
    };
    assert_eq!(conveyor.conveyor.records, bank.conveyor_records());
    assert!(bank.host_records(lender.group, 0) >= bank.landing_records() + lender.records);
    let weights = |slots: u64| {
        draft
            .load()
            .clone()
            .unbanked()
            .with_expert_bank(&definition, slots, bank.places, bank.flights, bank.reservation(), bank.host())
            .unwrap()
            .weight_bytes()
            .unwrap()
    };
    assert!(weights(bank.slots) <= cap && weights(bank.slots + 1) > cap);
    // The conveyor lies in host storage, which the weight cap does not
    // bound, and the scratch in the lender's records, one widest layer for
    // every group, which the lender's slots already hold, so it commits no
    // record beyond them. A conveyor charged as weights would cost the
    // slots its bytes hold, whatever else the bank charges besides.
    // The scratch is lent from the lender's bottom records, so it lies
    // within the records the lender's slots commit.
    let scratch = bank.scratch().lender;
    let committed = bank.committed_records(scratch.group);
    assert!(scratch.records <= committed, "{} scratch records beyond the lender's {committed}", scratch.records);
    let arena = conveyor.conveyor.bytes.iter().map(|(_, &bytes)| bytes).sum::<u64>();
    let charged_arena = (0..=bank.slots).rev().find(|&slots| weights(slots) + arena <= cap).unwrap();
    assert!(bank.slots - charged_arena >= arena / bank.slot_bytes().unwrap(), "{charged_arena} slots beside a charged conveyor");
}

/// A capped streamed load fits the device it is planned on, its graphs
/// charged as its slots hold them: one graph of each layout for each run of
/// it a launch makes, whatever class it is. Qwen3.6 35B on the recorded RTX
/// 3090 under a 6,890,894,474-byte cap, its graphs priced as the device
/// measured them, on CUDA and Vulkan: planned streamed, its startup graphs
/// under an eighth of its allocation (a charge of every launch class's
/// graphs apart took 54 GB on CUDA and 30 GB on Vulkan, and was refused at
/// planning).
#[test]
fn a_capped_streamed_load_fits_with_its_graphs_priced() {
    let scratch = Scratch::new("qwen36-priced-graphs");
    let package = qwen36_35b(&scratch);
    for backend in [BackendName::Cuda, BackendName::Vulkan] {
        let policy = ModelPolicy { weight_bound: Some(WeightBound::Cap(6_890_894_474)), ..ModelPolicy::default() };
        let setup = rtx3090_setup(backend, MemoryReserves::of_machine(None), policy, standard_service_limits());
        let (definition, execution) = match prepare_model_assessment(&package, &setup.structure) {
            Ok(PreparedModelAssessment::Planned { definition, execution, .. }) => (definition, execution),
            Ok(PreparedModelAssessment::Unsupported { unsupported, .. }) => {
                panic!("{backend:?}: Qwen3.6 35B streams on the RTX 3090 under the cap: {unsupported:?}")
            }
            Err(error) => panic!("{backend:?}: Qwen3.6 35B is planned on the RTX 3090 under the cap: {error}"),
        };
        assert!(execution.draft().load().expert_bank().is_some(), "{backend:?}: a capped load streams");
        let charge = execution.memory_charge(&definition, setup.structure.selected.graph_cost).unwrap();
        let graphs = charge.bounds.recorded_graph_bytes;
        assert!(0 < graphs && graphs < charge.allocation_bytes / 8, "{backend:?}: startup graphs {graphs} bytes");
    }
}

/// A weight cap grants the largest expert bank its weights hold, each
/// bank's weights the plan's own at its slots: a cap of them grants that
/// bank, a byte fewer the bank a slot smaller, and a byte below the floor's
/// none, a configuration refusal. Qwen3.6 35B on the M4 Pro's Metal, at the floor and at 128 and
/// 240 slots.
#[test]
fn a_weight_cap_grants_the_largest_bank_it_holds() {
    let scratch = Scratch::new("qwen36-weight-cap");
    let package = qwen36_35b(&scratch);
    let assess = |weight_cap: Option<u64>| {
        let setup = m4_pro_setup(
            MemoryReserves::of_machine(None),
            ModelPolicy {
                weight_bound: weight_cap.map(WeightBound::Cap),
                ..ModelPolicy::default()
            },
            // One decode row a step: the selected reads, and so the
            // weights, follow the launch rows.
            ServiceLimits {
                decode_tokens: 1,
                ..standard_service_limits()
            },
        );
        prepare_model_assessment(&package, &setup.structure)
    };
    let prepare = |weight_cap| assess(weight_cap).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let PreparedModelAssessment::Planned {
        definition, execution, ..
    } = prepare(None)
    else {
        panic!("Qwen3.6 35B is planned resident");
    };
    let draft = execution.draft();
    let Err(PlanError::WeightBound {
        needed: floor_weights,
        ..
    }) = draft.clone().with_bank::<AssessmentMemoryCharge>(&definition, Some(WeightBound::Cap(0)), &synthetic_prices())
    else {
        panic!("no bank holds no weights");
    };
    let floor = draft.clone().with_bank::<AssessmentMemoryCharge>(&definition, Some(WeightBound::Cap(floor_weights)), &synthetic_prices()).unwrap().into_draft();
    let bank = floor.load().expert_bank().unwrap().clone();
    assert_eq!(bank.slots, bank.floor());
    // A bank's weights at `slots`, its record tensors reserved as the
    // device reserves.
    let weights = |slots: u64| {
        draft
            .load()
            .clone()
            .unbanked()
            .with_expert_bank(&definition, slots, bank.places, bank.flights, bank.reservation(), bank.host())
            .unwrap()
            .weight_bytes()
            .unwrap()
    };
    assert_eq!(weights(bank.floor()), floor_weights);
    let granted = |cap: u64| match prepare(Some(cap)) {
        PreparedModelAssessment::Planned { execution, .. } => {
            execution.draft().load().expert_bank().expect("a capped load streams").slots
        }
        PreparedModelAssessment::Unsupported { unsupported, .. } => {
            panic!("a cap of {cap} bytes: {unsupported:?}")
        }
    };
    // The caller's cap is a configuration it cannot have, naming the
    // floor's weights, not a model the backend does not support.
    assert!(matches!(
        assess(Some(floor_weights - 1)),
        Err(ModelAssessmentError::WeightBound(error)) if error.needed == floor_weights
    ));
    for slots in [bank.floor(), 128, 240] {
        let cap = weights(slots);
        assert_eq!(granted(cap), slots, "a cap of {cap} bytes");
        if slots > bank.floor() {
            assert_eq!(granted(cap - 1), slots - 1, "a cap of {} bytes", cap - 1);
        }
    }
}

/// Every file miss of a discrete device lands in a host tier record, so a
/// linked bank streams only where host memory admits the tier's floor, its
/// landing reserve: Qwen3.6 35B on the recorded RTX 3090 under a
/// 6,890,894,474-byte cap streams beside a tier of at least that floor with
/// the standard reserves, and is unsupported, saying why, when host memory
/// keeps no room for a load.
#[test]
fn a_linked_bank_streams_only_beside_its_tiers_landing_reserve() {
    let scratch = Scratch::new("qwen36-tier-floor");
    let package = qwen36_35b(&scratch);
    let prepare = |reserves| {
        let policy = ModelPolicy { weight_bound: Some(WeightBound::Cap(6_890_894_474)), ..ModelPolicy::default() };
        let setup = rtx3090_setup(BackendName::Cuda, reserves, policy, standard_service_limits());
        prepare_model_assessment(&package, &setup.structure).unwrap_or_else(|error| panic!("operational failure: {error}"))
    };
    let draft = match prepare(MemoryReserves::of_machine(None)) {
        PreparedModelAssessment::Planned { execution, .. } => execution.draft().clone(),
        PreparedModelAssessment::Unsupported { unsupported, .. } => panic!("Qwen3.6 35B streams on the RTX 3090 under the cap: {unsupported:?}"),
    };
    let bank = draft.load().expert_bank().expect("a capped load streams");
    assert!((0..bank.groups().len()).all(|group| bank.host_records(group, bank.host_slots) >= bank.landing_records()));
    // The structure is prepared whatever the room; a plan of it settles
    // its tier into the host's room, here a fluctuation of the whole store.
    let host = seismic::fixtures::recorded::rtx3090().host_ram().capacity().value;
    let roomless = prepare(host_room(host, 0));
    let PreparedModelAssessment::Planned { definition, execution, .. } = roomless else {
        panic!("a capped load is prepared from its structure");
    };
    let draft = execution.draft().clone();
    let Err(PlanError::Unstreamed(reason)) = draft.with_bank::<AssessmentMemoryCharge>(&definition, Some(WeightBound::Cap(6_890_894_474)), &synthetic_prices()) else {
        panic!("no host room leaves the experts unstreamed");
    };
    assert!(format!("{reason:?}").contains("landing reserve"), "{reason:?}");
}

/// A linked host tier is sized at load by the plan's settle into the host
/// memory the reading admits beside the rest of the load, up to the ceiling
/// its weight cap leaves it (`ExpertBankPlan::host_cap`), which an
/// assessment's re-plan keeps: Qwen3.6 35B on the recorded RTX 3090 under
/// its 6,890,894,474-byte cap holds its tier's ceiling where host RAM has
/// room for it, and where it has room for half of them, the most slots
/// whose growth past the tier's floor that room admits, up to the ceiling:
/// of the cap's ceiling and the host's room, the smaller bound wins.
#[test]
fn a_linked_host_tier_holds_the_slots_host_memory_admits() {
    let scratch = Scratch::new("qwen36-tier-size");
    let package = qwen36_35b(&scratch);
    let prepare = |reserves| {
        let policy = ModelPolicy { weight_bound: Some(WeightBound::Cap(6_890_894_474)), ..ModelPolicy::default() };
        let setup = rtx3090_setup(BackendName::Cuda, reserves, policy, standard_service_limits());
        let prepared = prepare_model_assessment(&package, &setup.structure)
            .unwrap_or_else(|error| panic!("operational failure: {error}"));
        match prepared {
            PreparedModelAssessment::Planned { definition, execution, .. } => (setup, definition, execution.draft().clone()),
            PreparedModelAssessment::Unsupported { unsupported, .. } => {
                panic!("Qwen3.6 35B streams on the RTX 3090 under the cap: {unsupported:?}")
            }
        }
    };
    let (setup, definition, draft) = prepare(MemoryReserves::of_machine(None));
    let bank = draft.load().expert_bank().expect("a capped load streams").clone();
    let ceiling = bank.host_ceiling().expect("a discrete bank's tier is linked");
    // The assessment's fit holds the tier's ceiling where host RAM has
    // room for it, and reports it as where the cold experts lie.
    let prepared = prepare_model_assessment(&package, &setup.structure).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let assessed = finish_model_assessment(&prepared, &setup, &[], PROFILE_SEQUENCES).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let ModelAssessment::Assessed { execution: ModelExecution::Fits { residency: ModelResidency::Streamed { shares, .. }, .. }, .. } = assessed else {
        panic!("Qwen3.6 35B fits the RTX 3090 streamed under the cap: {assessed:?}");
    };
    assert_eq!(shares.cold(), &ColdSource::HostTier { slots: ceiling }, "host RAM holds the tier's ceiling");
    // A load's re-plan of the draft keeps its tier at the cap's ceiling
    // where host RAM has room for it, which is then what its fit reports the cold experts
    // lie in.
    let replan = |draft: &magnitude_executor::ExecutionPlanDraft| {
        draft
            .with_held_experts(&definition, HeldBound::of(None, draft.device().assessment_capacity_bytes()), &synthetic_prices(), unbounded, |_, _| {
                Ok::<_, AssessmentError>(Ok(()))
            })
            .unwrap()
            .unwrap()
            .0
    };
    let replanned = replan(&draft);
    assert_eq!(replanned.draft().load().expert_bank().unwrap().host_slots, ceiling);
    let Residency::Streamed(bank) = Residency::of(replanned.draft()) else {
        panic!("a capped load streams");
    };
    let shares = ExpertShares::of(&bank);
    assert_eq!(shares.cold(), &ColdSource::HostTier { slots: ceiling });
    assert_eq!(shares.held(shares.floor()).host as u64, ceiling);
    // The host RAM the load holds beside the tier's floor: its own plan's
    // charge, which holds the tier at its floor (`with_host_tier`), not the
    // assessment's floor form, whose pass is the floor's.
    let structure = &setup.structure;
    let device = structure.topology.resolve(replanned.draft().device().selector()).unwrap();
    let fit = replanned
        .charge()
        .assess_fit(&structure.topology, device, &structure.status, &structure.reserves)
        .unwrap();
    let host = fit.domains.iter().find(|domain| domain.role == DomainRole::Staging).unwrap();
    let growth = |slots| bank.host_charge(slots).unwrap() - bank.host_charge(0).unwrap();
    let half = ceiling / 2;
    let (_, _, draft) = prepare(host_room(host.capacity_bytes, host.required_bytes + growth(half)));
    let replanned = replan(&draft);
    let planned = replanned.draft().load().expert_bank().unwrap();
    assert_eq!(planned.slots, bank.slots, "host room leaves the device's bank as it is");
    let slots = planned.host_slots;
    assert!((half..=ceiling).contains(&slots), "{slots} host slots of {ceiling}");
    assert!(growth(slots) <= growth(half), "{slots} host slots grow past the room");
    assert!(slots == ceiling || growth(slots + 1) > growth(half), "{slots} host slots leave room for another");
}

/// A load assessment shrinks to fit is the load `with_bank` plans, tier and
/// all: Qwen3.6 35B on the recorded RTX 3090 with 8 GB of its device
/// reserved fits streamed, its cold experts in host RAM, not read from file.
#[test]
fn a_bank_shrunk_to_fit_keeps_its_host_tier() {
    let scratch = Scratch::new("qwen36-shrunk-tier");
    let package = qwen36_35b(&scratch);
    // A device whose measured motion holds 8 GB over a reading interval.
    let reserves = MemoryReserves::of_machine(None).measured(None, Some(StoreMotion::new(8_000_000_000 - 1, 10.0)));
    let mut setup = rtx3090_setup(BackendName::Cuda, reserves, ModelPolicy::default(), standard_service_limits());
    setup.measurements = synthetic_measurements();
    let prepared = prepare_model_assessment(&package, &setup.structure)
        .unwrap_or_else(|error| panic!("operational failure: {error}"));
    let ModelAssessment::Assessed { execution, .. } =
        finish_model_assessment(&prepared, &setup, &[], PROFILE_SEQUENCES)
            .unwrap_or_else(|error| panic!("operational failure: {error}"))
    else {
        panic!("Qwen3.6 35B is assessed");
    };
    let ModelExecution::Fits { residency: ModelResidency::Streamed { shares, .. }, .. } = &execution else {
        panic!("Qwen3.6 35B fits the RTX 3090 streamed: {execution:?}");
    };
    assert!(shares.slots() < shares.experts().get(), "{shares:?}");
    assert!(matches!(shares.cold(), ColdSource::HostTier { .. }), "{shares:?}");
}

/// A 4-bit MoE on a 24 GB device holds as much of its bank as the room
/// holds and merges its prompt passes, the plan's settle pricing each
/// holding by what it saves: Qwen3.6 35B on the recorded RTX 3090,
/// uncapped, planned on synthetic prices into the device's stable room
/// beside its mandatory part, which holds one conversation's state at the
/// model's whole 262,144-token context, and CUDA's blocks charged only the
/// tails no allocation reuses: 240 of its 256 slots, the most that room
/// admits that save anything on these prices (a load lends them the state
/// no request has committed), and grows to no more than the room's 243; a
/// prompt pass covers at least 512 rows.
#[test]
fn a_q4_moe_on_24_gb_holds_its_bank_and_a_merged_pass() {
    let scratch = Scratch::new("qwen36-3090-settle");
    let package = qwen36_35b(&scratch);
    let mut setup = rtx3090_setup(BackendName::Cuda, MemoryReserves::of_machine(None), ModelPolicy::default(), standard_service_limits());
    setup.measurements = synthetic_measurements();
    let prepared = prepare_model_assessment(&package, &setup.structure).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let PreparedModelAssessment::Planned { definition, execution, .. } = &prepared else {
        panic!("Qwen3.6 35B is planned on the RTX 3090");
    };
    let draft = execution.draft();
    let structure = &setup.structure;
    let device = structure.topology.resolve(draft.device().selector()).unwrap();
    let fit = execution_memory_charge(definition, draft).unwrap().assess_fit(&structure.topology, device, &structure.status, &structure.reserves).unwrap();
    let allocation = fit.domains.iter().find(|domain| domain.role == DomainRole::Allocation).unwrap();
    let bound = allocation.capacity_bytes - allocation.reserve_bytes;
    let room = |mandatory: u64| {
        let left = bound.saturating_sub(mandatory);
        PlanRoom { free: left, ceiling: left }
    };
    let prices = magnitude_engine::planning::plan_prices(&setup.measurements, draft);
    let (planned, ()) = draft
        .with_held_experts(definition, HeldBound::room(bound), &prices, room, |_, _| {
            Ok::<_, AssessmentError>(Ok(()))
        })
        .unwrap()
        .unwrap_or_else(|no| panic!("Qwen3.6 35B streams on the RTX 3090: {no:?}"));
    let bank = planned.draft().load().expert_bank().expect("a routed load banks its experts");
    assert_eq!(bank.slots, 240, "the bank holds the slots its room admits that save anything");
    assert!((240..=243).contains(&bank.ceiling), "a ceiling of {} slots beyond the room's 243", bank.ceiling);
    let limits = planned.draft().policy().limits();
    let region = *magnitude_executor::batching::row_classes(limits.max_launch_rows).last().unwrap() as u64;
    assert!(bank.pass.get() as u64 * region >= 512, "{:?} of {region} rows", bank.pass);
}

/// A streamed fit is priced on the expert bank it was classed streamed
/// by: Qwen3.6 35B prepared on the M4 Pro with 1 GB of room (its structure,
/// whatever the room), then fitted with 4 GB less room than its resident
/// plan, measured on synthetic costs and figures: streamed through the bank
/// the fit's settle holds (`with_held_experts`), its speed estimated on
/// that bank.
#[test]
fn a_streamed_fit_is_priced_on_the_bank_it_was_held_in() {
    use magnitude_executor::assessment::{Priced, ReadLine, StreamedFigures, synthetic};
    let scratch = Scratch::new("streamed-own-bank");
    let package = qwen36_35b(&scratch);
    let (_, _, resident) = assess_qwen36_35b(&package, MemoryReserves::of_machine(None));
    let ModelExecution::Fits { fit: ExecutionFit { domains, .. }, .. } = resident else {
        panic!("Qwen3.6 35B fits the M4 Pro resident: {resident:?}");
    };
    let allocation = domains.iter().find(|domain| domain.role == DomainRole::Allocation).unwrap();
    let room = |bytes| m4_pro_setup(host_room(allocation.capacity_bytes, bytes), ModelPolicy::default(), standard_service_limits());
    let prepared = prepare_model_assessment(&package, &room(1_000_000_000).structure).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let PreparedModelAssessment::Planned { .. } = &prepared else {
        panic!("Qwen3.6 35B is planned");
    };
    let mut setup = room(allocation.required_bytes - 4_000_000_000);
    let figures = StreamedFigures { read: ReadLine::new(1e-4, 1. / 3e9).unwrap(), memory: None, site: None, convert: 1. / 100e9 };
    setup.measurements = Measurements::Measured { costs: Arc::new(synthetic::costs()), records: Arc::default(), pending: false, streamed: Priced::Measured(figures) };
    let assessed = finish_model_assessment(&prepared, &setup, &[], PROFILE_SEQUENCES).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let ModelAssessment::Assessed { execution: ModelExecution::Fits { residency: ModelResidency::Streamed { performance, floor_performance, .. }, .. }, .. } = assessed else {
        panic!("Qwen3.6 35B fits streamed with 4 GB less room: {assessed:?}");
    };
    assert!(matches!(performance, Performance::Estimated(_)), "{performance:?}");
    assert!(matches!(floor_performance, Performance::Estimated(_)), "{floor_performance:?}");
}

/// Host RAM of stable `capacity` (a domain's room with its reserve)
/// keeping only `room` bytes for a load: its planning reserve, the margin
/// of a motion that holds it over a reading interval (one byte taken, the
/// rest fluctuation) above the recorded host's line (it reports none), is
/// the rest.
fn host_room(capacity: u64, room: u64) -> MemoryReserves {
    MemoryReserves::new(StoreMotion::new(capacity - room - 1, 10.0), magnitude_executor::platform::ReleaseRate::new(1e9))
}

/// The plan's prices on synthetic costs, figures and resident step.
fn synthetic_prices() -> Priced<PlanPrices> {
    Priced::Measured(synthetic::plan_prices())
}

/// The machine measured as `synthetic` gives it: its costs and streamed
/// figures stored.
fn synthetic_measurements() -> Measurements {
    Measurements::Measured {
        costs: Arc::new(synthetic::costs()),
        records: Arc::default(),
        pending: false,
        streamed: Priced::Measured(synthetic::streamed_figures()),
    }
}

/// Room for every holding a plan offers.
fn unbounded(_: u64) -> PlanRoom {
    PlanRoom { free: u64::MAX, ceiling: u64::MAX }
}

/// Qwen3.6 35B on the M4 Pro under `reserves`, prepared and assessed.
fn assess_qwen36_35b(
    package: &ModelPackagePaths,
    reserves: MemoryReserves,
) -> (AssessmentSetup, PreparedModelAssessment, ModelExecution) {
    let mut setup = m4_pro_setup(reserves, ModelPolicy::default(), standard_service_limits());
    setup.measurements = synthetic_measurements();
    let prepared = prepare_model_assessment(package, &setup.structure)
        .unwrap_or_else(|error| panic!("operational failure: {error}"));
    let ModelAssessment::Assessed { execution, .. } =
        finish_model_assessment(&prepared, &setup, &[], PROFILE_SEQUENCES)
            .unwrap_or_else(|error| panic!("operational failure: {error}"))
    else {
        panic!("Qwen3.6 35B is assessed");
    };
    (setup, prepared, execution)
}

/// Qwen3.6 35B fits the M4 Pro resident, unchanged; with too little room
/// for every expert it fits streamed through the largest expert bank the
/// room holds, its speed unmeasured; with too little for the bank's floor
/// it does not fit.
#[test]
fn a_model_fits_resident_streamed_or_not_at_all() {
    let scratch = Scratch::new("qwen36-streamed-fit");
    let package = qwen36_35b(&scratch);

    let (_, whole, resident) = assess_qwen36_35b(&package, MemoryReserves::of_machine(None));
    let ModelExecution::Fits {
        fit: ExecutionFit { domains, .. },
        residency:
            ModelResidency::Resident {
                performance: Performance::Estimated(_),
            },
    } = resident
    else {
        panic!("Qwen3.6 35B fits the M4 Pro resident: {resident:?}");
    };
    let allocation = domains
        .iter()
        .find(|domain| domain.role == DomainRole::Allocation)
        .unwrap();
    let (capacity, resident_bytes) = (allocation.capacity_bytes, allocation.required_bytes);

    // Room for the whole weights, every expert resident, and nothing beside
    // them: a whole bank charges those weights and the plan's mandatory part
    // besides at any pass, so none fits, and the load streams through the
    // holdings the fit's settle grants on the machine's prices. Room short
    // of the resident plan alone would still hold the bank whole beside a
    // smaller pass, which misses nothing.
    let PreparedModelAssessment::Planned { execution: whole, .. } = &whole else {
        unreachable!("the model was assessed");
    };
    let whole = whole.draft();
    let whole_weights = whole.load().clone().unbanked().weight_bytes().unwrap();
    assert!(whole_weights < resident_bytes);
    let reserves = host_room(capacity, whole_weights);
    let (setup, prepared, streamed) = assess_qwen36_35b(&package, reserves);
    let ModelExecution::Fits { fit: ExecutionFit { domains, .. }, residency: ModelResidency::Streamed { shares, .. } } = streamed else {
        panic!("Qwen3.6 35B fits streamed: {streamed:?}");
    };
    assert!(domains.iter().all(|domain| domain.remaining_bytes >= 0));
    assert!(shares.floor() <= shares.slots() && shares.slots() < shares.experts().get());
    assert!(shares.floor_share() <= shares.share() && shares.share() < 1.0);
    let PreparedModelAssessment::Planned { definition, execution, .. } = &prepared else {
        unreachable!("the model was assessed");
    };
    let draft = execution.draft();
    let Err(PlanError::WeightBound { needed: floor_weights, .. }) =
        draft.clone().with_bank::<AssessmentMemoryCharge>(definition, Some(WeightBound::Cap(0)), &synthetic_prices())
    else {
        panic!("no bank holds no weights");
    };

    // The fit holds the most slots the room admits at the floor of places,
    // and a load's settle of the fit's room on the machine's prices,
    // confirmed by the same stable fit, holds no more (it may trade slots
    // for prompt places); nor does a plan of half that room, as a release
    // slower than another program's take leaves it while it serves.
    let allocation = domains.iter().find(|domain| domain.role == DomainRole::Allocation).unwrap();
    let bound = allocation.capacity_bytes - allocation.reserve_bytes;
    let prices = magnitude_engine::planning::plan_prices(&setup.measurements, draft);
    let structure = &setup.structure;
    let stable = StableRoom {
        topology: &structure.topology,
        executor: structure.topology.resolve(draft.device().selector()).unwrap(),
        status: &structure.status,
        reserves: &structure.reserves,
    };
    let settled = |share: u64| {
        let room = |mandatory: u64| {
            let left = bound.saturating_sub(mandatory) / share;
            PlanRoom { free: left, ceiling: left }
        };
        let (planned, ()) = draft
            .with_held_experts(definition, HeldBound::room(bound), &prices, room, |candidate, charge| {
                Ok::<_, AssessmentError>(match stable.fit(candidate, charge).unwrap().verdict {
                    AssessmentFitVerdict::Fits => Ok(()),
                    AssessmentFitVerdict::DoesNotFit { deficit_bytes, .. } => Err(deficit_bytes),
                })
            })
            .unwrap()
            .unwrap();
        planned.draft().load().expert_bank().unwrap().slots
    };
    assert!(settled(1) <= shares.slots());
    assert!(settled(2) <= settled(1));

    // A machine whose figures are pending leaves a plan that must stream
    // pending: no plan is sized unpriced.
    let Err(PlanError::Unpriced { pending: true, .. }) =
        draft.clone().with_bank::<AssessmentMemoryCharge>(definition, None, &Priced::Pending("the job has not run".into()))
    else {
        panic!("a streamed plan with pending prices is pending");
    };

    // Room for less than the floor's weights alone.
    let (_, _, short) = assess_qwen36_35b(&package, host_room(capacity, floor_weights / 2));
    assert!(
        matches!(short, ModelExecution::DoesNotFit(_)),
        "{short:?}"
    );
}

/// Each tier of a fit's placement is the plan's: Qwen3.6 35B on the M4 Pro
/// resident holds every byte it requires in unified memory and reads from
/// disk only its streamed token table's rows beyond the table's slots;
/// with room for its whole weights and nothing beside them it streams, its
/// device bytes still what its fit requires and its disk bytes beside the
/// table's every expert its bank does not hold, as stored; on the recorded
/// RTX 3090 with 8 GB of its device reserved, host RAM holds what the fit
/// requires there and its host tier's growth past the landing reserve the
/// fit charges, and disk the experts neither tier holds beside the table's
/// rows.
#[test]
fn a_fits_placement_is_its_plans_tiers() {
    let scratch = Scratch::new("qwen36-placement");
    let package = qwen36_35b(&scratch);
    let required = |fit: &ExecutionFit| fit.breakdown.weights_bytes + fit.breakdown.context_bytes + fit.breakdown.overhead_bytes;
    // Every routed expert as stored, less those each layer holds in its
    // bank's slots or its host tier's.
    let unheld = |fit: &ExecutionFit| {
        let bank = fit.bank.as_ref().expect("a routed load banks its experts");
        bank.layers()
            .iter()
            .map(|layer| {
                let stored = layer.resident.iter().map(|(_, weight)| weight.source_bytes).sum::<u64>();
                let held = bank.slots.max(bank.host_slots).min(layer.experts);
                stored / layer.experts * (layer.experts - held)
            })
            .sum::<u64>()
    };

    let (_, whole, resident) = assess_qwen36_35b(&package, MemoryReserves::of_machine(None));
    let ModelExecution::Fits { fit, residency: ModelResidency::Resident { .. } } = resident else {
        panic!("Qwen3.6 35B fits the M4 Pro resident: {resident:?}");
    };
    let PreparedModelAssessment::Planned { draft: whole, .. } = &whole else {
        unreachable!("the model was assessed");
    };
    let streamed_tables = whole.load().target().iter().filter(|weight| weight.streamed().is_some()).collect::<Vec<_>>();
    assert!(!streamed_tables.is_empty(), "the token table is streamed by rows");
    let tables = streamed_tables.iter().map(|weight| weight.unheld_bytes().unwrap()).sum::<u64>();
    assert!(tables > 0);
    assert_eq!(fit.placement.device_bytes, required(&fit));
    assert_eq!((fit.placement.host_bytes, fit.placement.disk_bytes), (0, tables));

    let allocation = fit.domains.iter().find(|domain| domain.role == DomainRole::Allocation).unwrap();
    let whole_weights = whole.load().clone().unbanked().weight_bytes().unwrap();
    let (_, _, streamed) = assess_qwen36_35b(&package, host_room(allocation.capacity_bytes, whole_weights));
    let ModelExecution::Fits { fit, residency: ModelResidency::Streamed { .. } } = streamed else {
        panic!("Qwen3.6 35B fits streamed: {streamed:?}");
    };
    assert_eq!(fit.placement.device_bytes, required(&fit));
    assert_eq!(fit.placement.host_bytes, 0, "unified memory has no host tier");
    assert!(unheld(&fit) > 0);
    assert_eq!(fit.placement.disk_bytes, unheld(&fit) + tables);

    let reserves = MemoryReserves::of_machine(None).measured(None, Some(StoreMotion::new(8_000_000_000 - 1, 10.0)));
    let mut setup = rtx3090_setup(BackendName::Cuda, reserves, ModelPolicy::default(), standard_service_limits());
    setup.measurements = synthetic_measurements();
    let prepared = prepare_model_assessment(&package, &setup.structure).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let ModelAssessment::Assessed { execution: ModelExecution::Fits { fit, residency: ModelResidency::Streamed { .. } }, .. } =
        finish_model_assessment(&prepared, &setup, &[], PROFILE_SEQUENCES).unwrap_or_else(|error| panic!("operational failure: {error}"))
    else {
        panic!("Qwen3.6 35B fits the RTX 3090 streamed");
    };
    let bank = fit.bank.as_ref().unwrap();
    let held = |role| fit.domains.iter().filter(|domain| domain.role == role).map(|domain| domain.required_bytes).sum::<u64>();
    assert_eq!(fit.placement.device_bytes, held(DomainRole::Allocation));
    let tier = bank.host_charge(bank.host_slots).unwrap() - bank.host_charge(0).unwrap();
    assert!(tier > 0, "host RAM holds a host tier past its landing reserve");
    assert_eq!(fit.placement.host_bytes, held(DomainRole::Staging) + tier);
    assert_eq!(fit.placement.device_bytes + fit.placement.host_bytes, required(&fit) + tier);
    let PreparedModelAssessment::Planned { draft, .. } = &prepared else {
        unreachable!("the model was assessed");
    };
    let tables = draft.load().target().iter().map(|weight| weight.unheld_bytes().unwrap()).sum::<u64>();
    assert_eq!(fit.placement.disk_bytes, unheld(&fit) + tables);
}
