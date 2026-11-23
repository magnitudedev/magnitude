//! The shared real-header fixtures (`fixtures/assessment/`): every
//! fixture's expected verdict, asserted through the engine's public
//! assessment entry point on the CPU backend, which every build has. The
//! manifest format is described in the fixtures' README.

use flate2::read::GzDecoder;
use magnitude_artifacts::gguf::{
    inspect_header, Array, Directory, Elements, Metadata, Scalar, Value,
};
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
    AssessmentFitVerdict, AssessmentMemoryCharge, ConveyorCarrier, ExpertBankPlan, HeldBound, MemoryBudget, PlanError, PlanPrices,
    WeightBound,
    assessment::{AssessmentError, ColdSource, ExecutionFit, ExpertShares, Priced, Residency, execution_memory_charge, synthetic},
    platform::{DeviceRequest, DomainRole, ReleaseRate},
};
use magnitude_scheduler::ServiceLimits;
use seismic::BackendName;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod recorded;
use recorded::{m4_pro_setup, rtx3090_granting_setup, rtx3090_setup};

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
        ReleaseRate::of_machine(None),
        ModelPolicy::default(),
        standard_service_limits(),
    )
    .unwrap_or_else(|error| panic!("the CPU backend is selectable: {error}"))
}

/// The CPU backend before any measurement: fit is assessed, speed is pending.
fn cpu_setup() -> AssessmentSetup {
    AssessmentSetup {
        structure: Arc::new(cpu_structure()),
        measurements: Measurements::Pending(magnitude_executor::assessment::PendingPrices::standing(
            magnitude_executor::assessment::PointRequest::Costs(magnitude_executor::assessment::CostWalk::Required),
        )),
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
        let target_shards = fixture
            .target
            .iter()
            .map(|component| scratch.materialize(component))
            .collect::<Vec<_>>();
        assert!(!target_shards.is_empty(), "a fixture has a target");
        let package = ModelPackagePaths {
            target_shards,
            projector: None,
            draft: fixture
                .draft
                .as_ref()
                .map(|component| scratch.materialize(component)),
            method: fixture.method.into(),
        };
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
                let values = values.elements().unwrap();
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
    let ratios = match combined
        .value("qwen4exp.attention.compress_ratios")
        .map(|value| match value {
            Value::Array(ratios) => ratios.elements().cloned().ok(),
            Value::Scalar(_) => None,
        }) {
        Some(Some(Elements::Signed(mut ratios))) => {
            ratios.push(0);
            Array::from(ratios)
        }
        Some(Some(Elements::Unsigned(mut ratios))) => {
            ratios.push(0);
            Array::from(ratios)
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
        &ModelPackagePaths::single(target, None, None, ModelMethod::Auto),
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
        &ModelPackagePaths::single(target, None, Some(draft), ModelMethod::Auto),
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
    ModelPackagePaths::single(
        scratch.materialize(&fixture.target[0]),
        None,
        None,
        ModelMethod::Plain,
    )
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
        ReleaseRate::of_machine(None),
        ModelPolicy {
            weight_bound: Some(WeightBound::Cap(cap)),
            ..ModelPolicy::default()
        },
        standard_service_limits(),
    );
    let PreparedModelAssessment::Planned { execution, .. } = prepare_model_assessment(&package, &setup.structure)
        .unwrap_or_else(|error| panic!("operational failure: {error}"))
    else {
        panic!("Qwen3.6 35B is planned streamed on the RTX 3090 under the cap");
    };
    let definition = execution.definition();
    let draft = execution.draft();
    let bank = draft.load().expert_bank().expect("a capped load streams").clone();
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
            .with_expert_bank(&definition, slots, Some(bank.window_slots()), bank.places, bank.flights, bank.substrate())
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
    let slot_granules: u64 = bank.layers().iter().map(|layer| bank.group_granule(layer.group)).sum();
    assert!(bank.slots - charged_arena >= arena / slot_granules, "{charged_arena} slots beside a charged conveyor");
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
        let setup = rtx3090_setup(backend, ReleaseRate::of_machine(None), policy, standard_service_limits());
        let execution = match prepare_model_assessment(&package, &setup.structure) {
            Ok(PreparedModelAssessment::Planned { execution, .. }) => execution,
            Ok(PreparedModelAssessment::Unsupported { unsupported, .. }) => {
                panic!("{backend:?}: Qwen3.6 35B streams on the RTX 3090 under the cap: {unsupported:?}")
            }
            Err(error) => panic!("{backend:?}: Qwen3.6 35B is planned on the RTX 3090 under the cap: {error}"),
        };
        assert!(execution.draft().load().expert_bank().is_some(), "{backend:?}: a capped load streams");
        let charge = execution.memory_charge(execution.definition(), setup.structure.selected.graph_cost).unwrap();
        let graphs = charge.bounds.recorded_graph_bytes;
        assert!(0 < graphs && graphs < charge.allocation_bytes / 8, "{backend:?}: startup graphs {graphs} bytes");
    }
}

/// A weight cap grants the largest expert bank its weights hold at any
/// scratch window of the bank's ladder, each bank's weights the plan's own
/// at its slots and window: the granted bank holds at least its window's
/// floor and its weights are within the cap, and no rung holds a slot more
/// within it (a rung whose floor is above that, or a bank of every expert,
/// holds none); a byte below the floor's weights grants none, a
/// configuration refusal. Qwen3.6 35B on the M4 Pro's Metal, under caps of
/// the floor bank's weights at the floor and at 128 and 240 slots, and a
/// byte fewer.
#[test]
fn a_weight_cap_grants_the_largest_bank_it_holds() {
    let scratch = Scratch::new("qwen36-weight-cap");
    let package = qwen36_35b(&scratch);
    let assess = |weight_cap: Option<u64>| {
        let setup = m4_pro_setup(
            ReleaseRate::of_machine(None),
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
    let PreparedModelAssessment::Planned { execution, .. } = prepare(None) else {
        panic!("Qwen3.6 35B is planned resident");
    };
    let definition = execution.definition();
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
    assert_eq!(floor.load().weight_bytes().unwrap(), floor_weights);
    assert!(matches!(
        draft.clone().with_bank::<AssessmentMemoryCharge>(&definition, Some(WeightBound::Cap(floor_weights - 1)), &synthetic_prices()),
        Err(PlanError::WeightBound { needed, .. }) if needed == floor_weights
    ));
    // The load with `bank`'s structure at `slots` and `window_slots`, its
    // record tensors reserved as the device reserves, and its weights.
    // `prepare_model_assessment` runs before measurements (`BankPaths::Unpriced`),
    // matching `with_expert_bank`, so its floor is the unpriced floor.
    let Err(ModelAssessmentError::WeightBound(unpriced)) = assess(Some(0)) else {
        panic!("no unpriced bank holds no weights");
    };
    let unpriced_floor = unpriced.needed;
    let banked = |bank: &ExpertBankPlan, slots: u64, window_slots: u64| {
        draft
            .load()
            .clone()
            .unbanked()
            .with_expert_bank(&definition, slots, Some(window_slots), bank.places, bank.flights, bank.substrate())
            .unwrap()
    };
    let weights = |bank: &ExpertBankPlan, slots: u64, window_slots: u64| banked(bank, slots, window_slots).weight_bytes().unwrap();
    // Every rung of the ladder the plan walks, from the window of a bank of
    // every expert, with that rung's floor.
    let experts = bank.widest().get();
    let rungs = banked(&bank, experts, experts)
        .expert_bank()
        .unwrap()
        .window_ladder()
        .into_iter()
        .map(|window| (window, banked(&bank, experts, window).expert_bank().unwrap().floor()))
        .collect::<Vec<_>>();
    assert_eq!(weights(&bank, bank.floor(), bank.window_slots()), unpriced_floor);
    // The bank a cap grants, at the window the plan settles on for it.
    let granted = |cap: u64| match prepare(Some(cap)) {
        PreparedModelAssessment::Planned { execution, .. } => {
            execution.draft().load().expert_bank().expect("a capped load streams").clone()
        }
        PreparedModelAssessment::Unsupported { unsupported, .. } => {
            panic!("a cap of {cap} bytes: {unsupported:?}")
        }
    };
    // The caller's cap is a configuration it cannot have, naming the
    // floor's weights, not a model the backend does not support.
    assert!(matches!(
        assess(Some(unpriced_floor - 1)),
        Err(ModelAssessmentError::WeightBound(error)) if error.needed == unpriced_floor
    ));
    for slots in [bank.floor(), 128, 240] {
        let cap = weights(&bank, slots, bank.window_slots());
        for cap in [cap, cap - 1].into_iter().filter(|&cap| cap >= unpriced_floor) {
            let grant = granted(cap);
            let (held, window) = (grant.slots, grant.window_slots());
            assert!(rungs.iter().any(|&(rung, _)| rung == window), "a cap of {cap} bytes settles on window {window}, a rung");
            assert!(grant.floor() <= held && held <= grant.ceiling, "a cap of {cap} bytes grants {held} slots");
            assert!(weights(&grant, held, window) <= cap, "a cap of {cap} bytes grants {held} slots beyond it");
            for &(rung, rung_floor) in &rungs {
                let more = held + 1;
                // A bank of every expert holds no slot more, and a rung
                // whose floor is above it holds none of that size.
                if more > experts || more < rung_floor {
                    continue;
                }
                assert!(
                    weights(&grant, more, rung) > cap,
                    "a cap of {cap} bytes grants {held} slots at window {window}, but window {rung} holds {more}"
                );
            }
        }
    }
}

/// Every file miss of a discrete device lands in a host tier record, so a
/// linked bank streams only where host memory admits the tier's floor, its
/// landing reserve: Qwen3.6 35B on the recorded RTX 3090 under a
/// 6,890,894,474-byte cap streams beside a tier of at least that floor with
/// the standard limits, and is unsupported, saying why, when host memory
/// keeps no room for a load.
#[test]
fn a_linked_bank_streams_only_beside_its_tiers_landing_reserve() {
    let scratch = Scratch::new("qwen36-tier-floor");
    let package = qwen36_35b(&scratch);
    let prepare = |room: Option<u64>| {
        let policy = ModelPolicy { weight_bound: Some(WeightBound::Cap(6_890_894_474)), ..ModelPolicy::default() };
        let setup = within_host(rtx3090_setup(BackendName::Cuda, ReleaseRate::of_machine(None), policy, standard_service_limits()), room);
        prepare_model_assessment(&package, &setup.structure).unwrap_or_else(|error| panic!("operational failure: {error}"))
    };
    let draft = match prepare(None) {
        PreparedModelAssessment::Planned { execution, .. } => execution.draft().clone(),
        PreparedModelAssessment::Unsupported { unsupported, .. } => panic!("Qwen3.6 35B streams on the RTX 3090 under the cap: {unsupported:?}"),
    };
    let bank = draft.load().expert_bank().expect("a capped load streams");
    assert!((0..bank.groups().len()).all(|group| bank.host_records(group, bank.host_slots) >= bank.landing_records()));
    // The structure is prepared whatever the room; a plan of it settles
    // its tier into the host's room, here none.
    let roomless = prepare(Some(0));
    let PreparedModelAssessment::Planned { execution, .. } = roomless else {
        panic!("a capped load is prepared from its structure");
    };
    let definition = execution.definition();
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
    let prepare = |room: Option<u64>| {
        let policy = ModelPolicy { weight_bound: Some(WeightBound::Cap(6_890_894_474)), ..ModelPolicy::default() };
        let setup = within_host(rtx3090_setup(BackendName::Cuda, ReleaseRate::of_machine(None), policy, standard_service_limits()), room);
        let prepared = prepare_model_assessment(&package, &setup.structure)
            .unwrap_or_else(|error| panic!("operational failure: {error}"));
        match prepared {
            PreparedModelAssessment::Planned { execution, .. } => (setup, execution.definition().clone(), execution.draft().clone()),
            PreparedModelAssessment::Unsupported { unsupported, .. } => {
                panic!("Qwen3.6 35B streams on the RTX 3090 under the cap: {unsupported:?}")
            }
        }
    };
    let (setup, definition, draft) = prepare(None);
    let bank = draft.load().expert_bank().expect("a capped load streams").clone();
    let ceiling = bank.host_ceiling().expect("a discrete bank's tier is linked");
    // The assessment's fit holds the tier's ceiling where host RAM has
    // room for it, and reports it as where the cold experts lie.
    let prepared = prepare_model_assessment(&package, &setup.structure).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let assessed = finish_model_assessment(&prepared, &setup, &[], PROFILE_SEQUENCES).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let ModelAssessment::Assessed { execution: ModelExecution::Fits { residency: ModelResidency::Streamed { shares, .. }, .. }, .. } = assessed else {
        panic!("Qwen3.6 35B fits the RTX 3090 streamed under the cap: {assessed:?}");
    };
    assert!(matches!(shares.cold(), &ColdSource::HostTier { slots, .. } if slots == ceiling), "host RAM holds the tier's ceiling");
    // A load's re-plan of the draft keeps its tier at the cap's ceiling
    // where host RAM has room for it, which is then what its fit reports the cold experts
    // lie in. Uses `StableRoom::of(draft.device())` so `with_host_tier` reads
    // `within_host`'s host RAM limit (`unbounded`'s `host_left` returns `i64::MAX`,
    // which previously passed only because `landing_records` was 80 >= 40 * 2).
    let replan = |draft: &magnitude_executor::ExecutionPlanDraft| {
        let budget = MemoryBudget::of_stable(draft.device()).unwrap();
        draft
            .with_held_experts(
                &definition,
                HeldBound::of(None, budget.capacity()),
                &synthetic_prices(),
                &budget,
                |_, _| Ok::<_, AssessmentError>(Ok(())),
            )
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
    assert_eq!(shares.cold(), &ColdSource::HostTier { slots: ceiling, cache_bytes: bank.host_cache(ceiling).unwrap() });
    assert_eq!(shares.held(shares.floor()).host as u64, ceiling);
    // The host RAM the load holds beside the tier's floor: its own plan's
    // charge, which holds the tier at its floor (`with_host_tier`), not the
    // assessment's floor form, whose pass is the floor's.
    let structure = &setup.structure;
    let device = structure.topology.resolve(replanned.draft().device().selector()).unwrap();
    let fit = replanned
        .charge()
        .assess_fit(&structure.topology, device, &structure.status)
        .unwrap();
    let host = fit.domains.iter().find(|domain| domain.role == DomainRole::Staging).unwrap();
    let priced_bank = replanned.draft().load().expert_bank().unwrap().clone();
    let growth = |slots| priced_bank.host_cache(slots).unwrap();
    let half = ceiling / 2;
    let (_, _, draft) = prepare(Some(host.required_bytes + growth(half)));
    let replanned = replan(&draft);
    let planned = replanned.draft().load().expert_bank().unwrap();
    assert_eq!(planned.slots, bank.slots, "host room leaves the device's bank as it is");
    let slots = planned.host_slots;
    assert!((half..=ceiling).contains(&slots), "{slots} host slots of {ceiling}");
    assert!(growth(slots) <= growth(half), "{slots} host slots grow past the room");
    assert!(slots == ceiling || growth(slots + 1) > growth(half), "{slots} host slots leave room for another");
}

/// A streamed fit's figures count each byte where the load needs it: its
/// stores' required bytes hold the host tier at its floor, each expert
/// counted once, and the tier's growth past it is reported apart as cache
/// (`ColdSource::HostTier::cache_bytes`), the growth the plan chose
/// (`ExpertBankPlan::host_cache`); required and cache together stay within
/// the host room the plan admitted. Qwen3.6 35B on the recorded RTX 3090,
/// uncapped, so its tier's ceiling is every expert (a weight cap bounds the
/// tier too, and the 6,890,894,474-byte cap leaves it no growth past its
/// floor): with room for the tier's ceiling, and with room for its floor
/// alone, the stores require the same bytes.
#[test]
fn a_streamed_fit_reports_its_tier_growth_as_cache() {
    let scratch = Scratch::new("qwen36-tier-cache");
    let package = qwen36_35b(&scratch);
    let assess = |room: Option<u64>| {
        let mut setup = within_host(
            rtx3090_setup(BackendName::Cuda, ReleaseRate::of_machine(None), ModelPolicy::default(), standard_service_limits()),
            room,
        );
        setup.measurements = synthetic_measurements();
        let prepared = prepare_model_assessment(&package, &setup.structure)
            .unwrap_or_else(|error| panic!("operational failure: {error}"));
        match finish_model_assessment(&prepared, &setup, &[], PROFILE_SEQUENCES)
            .unwrap_or_else(|error| panic!("operational failure: {error}"))
        {
            ModelAssessment::Assessed { execution: ModelExecution::Fits { fit, residency: ModelResidency::Streamed { shares, .. } }, .. } => {
                (fit, shares)
            }
            assessed => panic!("Qwen3.6 35B fits the RTX 3090 streamed: {assessed:?}"),
        }
    };
    let host_of = |fit: &ExecutionFit| fit.domains.iter().find(|domain| domain.role == DomainRole::Staging).cloned().unwrap();
    let (fit, shares) = assess(None);
    let bank = fit.bank.as_ref().expect("a streamed fit reports its bank");
    let &ColdSource::HostTier { slots, cache_bytes } = shares.cold() else {
        panic!("host RAM holds a tier: {shares:?}");
    };
    assert_eq!(slots, bank.host_slots);
    assert_eq!(cache_bytes, bank.host_cache(slots).unwrap(), "the cache is the growth the plan chose");
    assert!(cache_bytes > 0, "room for the ceiling grows the tier past its floor");
    let host = host_of(&fit);
    assert!(
        host.required_bytes + cache_bytes <= host.capacity_bytes,
        "{} required and {cache_bytes} cache exceed the host room {}",
        host.required_bytes,
        host.capacity_bytes
    );
    // Room for the tier's floor alone: no cache, the same required bytes.
    let (floor_fit, floor_shares) = assess(Some(host.required_bytes));
    assert!(
        !matches!(floor_shares.cold(), ColdSource::HostTier { cache_bytes, .. } if *cache_bytes > 0),
        "a host room of the floor grows no cache: {floor_shares:?}"
    );
    let required = |fit: &ExecutionFit| fit.domains.iter().map(|domain| (domain.role, domain.required_bytes)).collect::<Vec<_>>();
    assert_eq!(required(&floor_fit), required(&fit), "the tier's growth is no store's required bytes");
    assert_eq!(floor_fit.breakdown, fit.breakdown);
}

/// A load assessment shrinks to fit is the load `with_bank` plans, tier and
/// all: Qwen3.6 35B on the recorded RTX 3090 granting CUDA 8 GB less than
/// its whole memory fits streamed, its cold experts in host RAM, not read
/// from file.
#[test]
fn a_bank_shrunk_to_fit_keeps_its_host_tier() {
    let scratch = Scratch::new("qwen36-shrunk-tier");
    let package = qwen36_35b(&scratch);
    let mut setup = rtx3090_less_8_gb();
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
/// holds and merges its prompt passes: Qwen3.6 35B on the recorded RTX
/// 3090, uncapped, planned into the device's stable room beside its
/// mandatory part (its arena among it), which holds one conversation's
/// state at the model's whole 262,144-token context, and CUDA's blocks
/// charged only the tails no allocation reuses: its ceiling the most slots
/// the card's weights hold at the records pool's uniform granule, and its
/// slots the most the room admits beside an arena holding
/// a group of one place (a group's class beyond it is claimed as the group opens); a
/// prompt pass covers at least 512 rows.
#[test]
fn a_q4_moe_on_24_gb_holds_its_bank_and_a_merged_pass() {
    let scratch = Scratch::new("qwen36-3090-settle");
    let package = qwen36_35b(&scratch);
    let mut setup = rtx3090_setup(BackendName::Cuda, ReleaseRate::of_machine(None), ModelPolicy::default(), standard_service_limits());
    setup.measurements = synthetic_measurements();
    let prepared = prepare_model_assessment(&package, &setup.structure).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let PreparedModelAssessment::Planned { execution, .. } = &prepared else {
        panic!("Qwen3.6 35B is planned on the RTX 3090");
    };
    let definition = execution.definition();
    let draft = execution.draft();
    let structure = &setup.structure;
    let device = structure.topology.resolve(draft.device().selector()).unwrap();
    let fit = execution_memory_charge(definition, draft).unwrap().assess_fit(&structure.topology, device, &structure.status).unwrap();
    let allocation = fit.domains.iter().find(|domain| domain.role == DomainRole::Allocation).unwrap();
    let bound = allocation.capacity_bytes;
    let budget = MemoryBudget::fixed(bound, None);
    let prices = magnitude_engine::planning::plan_prices(&setup.measurements, draft);
    let (planned, ()) = draft
        .with_held_experts(definition, HeldBound::room(bound), &prices, &budget, |_, _| {
            Ok::<_, AssessmentError>(Ok(()))
        })
        .unwrap()
        .unwrap_or_else(|no| panic!("Qwen3.6 35B streams on the RTX 3090: {no:?}"));
    let bank = planned.draft().load().expert_bank().expect("a routed load banks its experts");
    let device_weights = |slots: u64| {
        let load = planned.draft().load().clone().with_bank_slots(slots, slots).unwrap();
        load.weight_bytes().unwrap() - load.host_tier_bytes().unwrap()
    };
    assert!(
        bank.ceiling < bank.widest().get()
            && device_weights(bank.ceiling) <= bound
            && device_weights(bank.ceiling + 1) > bound,
        "ceiling {} of {} is not the most slots whose device weights fit {bound} bytes ({} / {})",
        bank.ceiling,
        bank.widest().get(),
        device_weights(bank.ceiling),
        device_weights(bank.ceiling + 1),
    );
    assert!(bank.ceiling >= 230, "24 GB card holds almost every expert: {} of {}", bank.ceiling, bank.widest().get());
    assert!((186..=192).contains(&bank.slots), "{} slots: the bank holds the most slots the room admits", bank.slots);
    let limits = planned.draft().policy().limits();
    let region = *magnitude_executor::batching::row_classes(limits.max_launch_rows).last().unwrap() as u64;
    assert!(bank.pass.get() as u64 * region >= 512, "{:?} of {region} rows", bank.pass);
}

/// A streamed fit is priced on the expert bank it was classed streamed
/// by: Qwen3.6 35B prepared on the M4 Pro with all its room (its structure,
/// whatever the room, which no load's own charge can exceed), then fitted
/// with 4 GB less room than its resident plan, measured on synthetic costs and figures: streamed through the bank
/// the fit's settle holds (`with_held_experts`), its speed estimated on
/// that bank.
#[test]
fn a_streamed_fit_is_priced_on_the_bank_it_was_held_in() {
    use magnitude_cost_model::ReadLine;
    use magnitude_executor::assessment::{Priced, StreamedFigures, synthetic};
    let scratch = Scratch::new("streamed-own-bank");
    let package = qwen36_35b(&scratch);
    let (_, _, resident) = assess_qwen36_35b(&package, None);
    let ModelExecution::Fits { fit: ExecutionFit { domains, .. }, .. } = resident else {
        panic!("Qwen3.6 35B fits the M4 Pro resident: {resident:?}");
    };
    let allocation = domains.iter().find(|domain| domain.role == DomainRole::Allocation).unwrap();
    let room = |bytes| within_host(m4_pro_setup(ReleaseRate::of_machine(None), ModelPolicy::default(), standard_service_limits()), Some(bytes));
    let prepared = prepare_model_assessment(&package, &room(allocation.capacity_bytes - 1).structure).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let PreparedModelAssessment::Planned { execution: whole, .. } = &prepared else {
        panic!("Qwen3.6 35B is planned");
    };
    let whole_weights = whole.draft().load().clone().unbanked().weight_bytes().unwrap();
    let mut setup = room(whole_weights - 4_000_000_000);
    let figures = StreamedFigures { read: ReadLine::new(1e-4, 1. / 3e9).unwrap(), memory: None, site: None, convert: 1. / 100e9 };
    setup.measurements = Measurements::Measured { costs: Arc::new(synthetic::costs()), records: Arc::default(), pending: false, streamed: Priced::Measured(figures) };
    let assessed = finish_model_assessment(&prepared, &setup, &[], PROFILE_SEQUENCES).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let ModelAssessment::Assessed { execution: ModelExecution::Fits { residency: ModelResidency::Streamed { performance, floor_performance, .. }, .. }, .. } = assessed else {
        panic!("Qwen3.6 35B fits streamed with 4 GB less room: {assessed:?}");
    };
    assert!(matches!(performance, Performance::Estimated(_)), "{performance:?}");
    assert!(matches!(floor_performance, Performance::Estimated(_)), "{floor_performance:?}");
}

/// The recorded RTX 3090 granting CUDA 8 GB less than its whole memory.
fn rtx3090_less_8_gb() -> AssessmentSetup {
    rtx3090_granting_setup(
        BackendName::Cuda,
        seismic::fixtures::RTX3090_CUDA_TOTAL - 8_000_000_000,
        ReleaseRate::of_machine(None),
        ModelPolicy::default(),
        standard_service_limits(),
    )
}

/// `setup` with its host RAM bound, where `room` is given, by a limit of
/// `room` bytes, as a cgroup or job binds it, so every store on host RAM
/// holds no more for a load; its device selected again within the limit.
fn within_host(mut setup: AssessmentSetup, room: Option<u64>) -> AssessmentSetup {
    let Some(room) = room else {
        return setup;
    };
    let structure = Arc::get_mut(&mut setup.structure).expect("a fresh setup's structure is its own");
    structure.status.sample.host.total = room;
    let selected = &structure.selected;
    structure.selected = magnitude_executor::platform::selected(
        &structure.topology,
        &structure.status,
        &structure.release,
        selected.info.clone(),
        selected.graph_cost,
    );
    setup
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

/// Qwen3.6 35B on the M4 Pro, its host RAM bound to `room` where given
/// ([`within_host`]), prepared and assessed.
fn assess_qwen36_35b(
    package: &ModelPackagePaths,
    room: Option<u64>,
) -> (AssessmentSetup, PreparedModelAssessment, ModelExecution) {
    let mut setup = within_host(m4_pro_setup(ReleaseRate::of_machine(None), ModelPolicy::default(), standard_service_limits()), room);
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

    let (_, whole, resident) = assess_qwen36_35b(&package, None);
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
    let resident_bytes = allocation.required_bytes;

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
    let (setup, prepared, streamed) = assess_qwen36_35b(&package, Some(whole_weights));
    let ModelExecution::Fits { fit: ExecutionFit { domains, .. }, residency: ModelResidency::Streamed { shares, .. } } = streamed else {
        panic!("Qwen3.6 35B fits streamed: {streamed:?}");
    };
    assert!(domains.iter().all(|domain| domain.remaining_bytes >= 0));
    assert!(shares.floor() <= shares.slots() && shares.slots() < shares.experts().get());
    assert!(shares.floor_share() <= shares.share() && shares.share() < 1.0);
    let PreparedModelAssessment::Planned { execution, .. } = &prepared else {
        unreachable!("the model was assessed");
    };
    let definition = execution.definition();
    let draft = execution.draft();
    let Err(PlanError::WeightBound { needed: floor_weights, .. }) =
        draft.clone().with_bank::<AssessmentMemoryCharge>(definition, Some(WeightBound::Cap(0)), &synthetic_prices())
    else {
        panic!("no bank holds no weights");
    };

    // The fit holds the most slots the room admits at the floor of places,
    // and a load's plan of the fit's room beside its arena, confirmed by
    // the same stable fit, holds no more; nor does a plan of half that
    // room, as a release slower than another program's take leaves it
    // while it serves.
    let allocation = domains.iter().find(|domain| domain.role == DomainRole::Allocation).unwrap();
    let bound = allocation.capacity_bytes;
    let structure = &setup.structure;
    let executor = structure.topology.resolve(draft.device().selector()).unwrap();
    let stable = MemoryBudget::stable(&structure.topology, executor, &structure.status).unwrap();
    let prices = magnitude_engine::planning::plan_prices(&setup.measurements, draft);
    let settled = |budget_bytes: u64| {
        let budget = MemoryBudget::fixed(budget_bytes, None);
        let (planned, ()) = draft
            .with_held_experts(definition, HeldBound::room(bound), &prices, &budget, |_, charge| {
                Ok::<_, AssessmentError>(match stable.fit(&structure.topology, executor, charge).unwrap().verdict {
                    AssessmentFitVerdict::Fits => Ok(()),
                    AssessmentFitVerdict::DoesNotFit { deficit_bytes, .. } => Err(deficit_bytes),
                })
            })
            .unwrap()
            .unwrap();
        planned.draft().load().expert_bank().unwrap().slots
    };
    let full = settled(bound);
    assert!(full <= shares.slots());
    let floor_bytes = execution_memory_charge(definition, &draft.clone().streaming::<AssessmentMemoryCharge>(definition, None, &stable).unwrap().streamed(definition, &prices, &MemoryBudget::fixed(bound, None)).unwrap().into_draft())
        .unwrap()
        .store_demands(&structure.topology, executor)
        .unwrap()
        .allocation_bytes();
    let half_extra = bound.saturating_sub(floor_bytes) / 2;
    assert!(settled(floor_bytes + half_extra) <= full);

    // A machine whose figures are pending leaves a plan that must stream
    // pending: no plan is sized unpriced.
    let Err(PlanError::Pending(_)) = draft.clone().with_bank::<AssessmentMemoryCharge>(
        definition,
        None,
        &Priced::Pending(magnitude_executor::assessment::PendingPrices::standing(
            magnitude_executor::assessment::PointRequest::Costs(magnitude_executor::assessment::CostWalk::Required),
        )),
    )
    else {
        panic!("a streamed plan with pending prices is pending");
    };

    // Room for less than the floor's weights alone.
    let (_, _, short) = assess_qwen36_35b(&package, Some(floor_weights / 2));
    assert!(
        matches!(short, ModelExecution::DoesNotFit(_)),
        "{short:?}"
    );
}

/// A fit's placement is where its weights are held as they run, as stored,
/// so every plan of the same weights places the same bytes. Qwen3.6 35B's
/// token table is read by row: its rows beyond its slots are held in host
/// RAM's file cache when host RAM has room left for them (unified memory on
/// the M4 Pro, so in memory), else looked up on disk, and never counted as
/// streamed. On the M4 Pro resident it holds everything in unified memory;
/// with room for its whole weights and nothing beside them it streams,
/// reading from disk exactly the experts its bank does not hold; on the
/// recorded RTX 3090 granting CUDA 8 GB less than its whole memory, host
/// RAM holds the experts only its host tier holds, and disk those neither
/// holds.
#[test]
fn a_fits_placement_is_where_its_weights_are_held() {
    let scratch = Scratch::new("qwen36-placement");
    let package = qwen36_35b(&scratch);
    let total = |fit: &ExecutionFit| fit.placement.device_bytes + fit.placement.host_bytes + fit.placement.disk_bytes + fit.placement.lookup_disk_bytes;
    // Each layer's routed experts as stored that fall in `[from, to)` of its
    // expert order, given in slots.
    let experts = |fit: &ExecutionFit, held: &dyn Fn(u64, u64) -> u64| {
        let bank = fit.bank.as_ref().expect("a routed load banks its experts");
        bank.layers()
            .iter()
            .map(|layer| {
                let stored = layer.resident.iter().map(|(_, weight)| weight.source_bytes).sum::<u64>();
                stored / layer.experts * held(layer.experts, bank.slots.min(layer.experts))
            })
            .sum::<u64>()
    };

    let (_, whole, resident) = assess_qwen36_35b(&package, None);
    let ModelExecution::Fits { fit: resident, residency: ModelResidency::Resident { .. } } = resident else {
        panic!("Qwen3.6 35B fits the M4 Pro resident: {resident:?}");
    };
    let PreparedModelAssessment::Planned { execution: whole, .. } = &whole else {
        unreachable!("the model was assessed");
    };
    let whole = whole.draft();
    assert!(whole.load().target().iter().any(|weight| weight.streamed().is_some()), "the token table is streamed by rows");
    let rows = whole.load().unheld_table_bytes().unwrap();
    assert!(rows > 0, "the token table's slots hold some of its rows");
    // Where a fit's table rows go: RAM's file cache when its room takes them.
    let looked_up = |fit: &ExecutionFit| {
        let room = fit.domains.iter().find(|domain| domain.role == DomainRole::Allocation).unwrap().remaining_bytes;
        if room >= rows as i64 { 0 } else { rows }
    };
    assert_eq!(looked_up(&resident), 0, "unified memory has room for the table beside the model");
    assert_eq!((resident.placement.host_bytes, resident.placement.disk_bytes, resident.placement.lookup_disk_bytes), (0, 0, 0));

    let whole_weights = whole.load().clone().unbanked().weight_bytes().unwrap();
    let (_, _, streamed) = assess_qwen36_35b(&package, Some(whole_weights));
    let ModelExecution::Fits { fit, residency: ModelResidency::Streamed { .. } } = streamed else {
        panic!("Qwen3.6 35B fits streamed: {streamed:?}");
    };
    let bank = fit.bank.as_ref().unwrap();
    let unheld = experts(&fit, &|experts, _| experts - bank.slots.max(bank.host_slots).min(experts));
    assert!(unheld > 0);
    assert_eq!(fit.placement.host_bytes, 0, "unified memory has no host tier");
    assert_eq!(fit.placement.disk_bytes, unheld, "only experts stream");
    assert_eq!(fit.placement.lookup_disk_bytes, looked_up(&fit));
    assert_eq!(total(&fit), total(&resident));

    let mut setup = rtx3090_less_8_gb();
    setup.measurements = synthetic_measurements();
    let prepared = prepare_model_assessment(&package, &setup.structure).unwrap_or_else(|error| panic!("operational failure: {error}"));
    let ModelAssessment::Assessed { execution: ModelExecution::Fits { fit, residency: ModelResidency::Streamed { .. } }, .. } =
        finish_model_assessment(&prepared, &setup, &[], PROFILE_SEQUENCES).unwrap_or_else(|error| panic!("operational failure: {error}"))
    else {
        panic!("Qwen3.6 35B fits the RTX 3090 streamed");
    };
    let bank = fit.bank.as_ref().unwrap();
    let tier_only = experts(&fit, &|experts, slots| bank.host_slots.min(experts).saturating_sub(slots));
    assert!(tier_only > 0, "host RAM holds experts the bank does not");
    // Beside a discrete device the table's rows are cached in host RAM or looked up on disk.
    let cached = if fit.placement.lookup_disk_bytes == 0 { rows } else { assert_eq!(fit.placement.lookup_disk_bytes, rows); 0 };
    assert_eq!(fit.placement.host_bytes, tier_only + cached);
    assert_eq!(fit.placement.disk_bytes, experts(&fit, &|experts, _| experts - bank.slots.max(bank.host_slots).min(experts)));
    assert_eq!(total(&fit), total(&resident));
}

#[test]
fn assessment_timing_gate_m4_pro() {
    let manifest = manifest();
    assert!(!manifest.fixtures.is_empty());
    let mut setup = m4_pro_setup(
        ReleaseRate::of_machine(None),
        ModelPolicy::default(),
        standard_service_limits(),
    );
    setup.measurements = synthetic_measurements();

    let mut timings_ms: Vec<(String, f64, f64, f64)> = Vec::new();
    for fixture in &manifest.fixtures {
        let scratch = Scratch::new(&format!("timing-{}", fixture.name));
        let target_shards = fixture
            .target
            .iter()
            .map(|component| scratch.materialize(component))
            .collect::<Vec<_>>();
        assert!(!target_shards.is_empty(), "a fixture has a target");
        let package = ModelPackagePaths {
            target_shards,
            projector: None,
            draft: fixture
                .draft
                .as_ref()
                .map(|component| scratch.materialize(component)),
            method: fixture.method.into(),
        };

        let t0 = std::time::Instant::now();
        let prepared = prepare_model_assessment(&package, &setup.structure)
            .unwrap_or_else(|error| panic!("{}: prepare failed: {error}", fixture.name));
        let t1 = std::time::Instant::now();
        let _assessment = finish_model_assessment(&prepared, &setup, &[], PROFILE_SEQUENCES)
            .unwrap_or_else(|error| panic!("{}: finish failed: {error}", fixture.name));
        let t2 = std::time::Instant::now();

        let prepare_ms = t1.duration_since(t0).as_secs_f64() * 1_000.0;
        let finish_ms = t2.duration_since(t1).as_secs_f64() * 1_000.0;
        let total_ms = t2.duration_since(t0).as_secs_f64() * 1_000.0;
        timings_ms.push((fixture.name.clone(), prepare_ms, finish_ms, total_ms));
    }

    timings_ms.sort_by(|a, b| a.3.total_cmp(&b.3));
    let n = timings_ms.len();
    let p50 = timings_ms[n / 2].3;
    let p90 = timings_ms[(n * 9) / 10].3;
    let max = timings_ms[n - 1].3;
    eprintln!("=== assessment_timing_gate_m4_pro ({n} fixtures) ===");
    for (name, prep, fin, tot) in &timings_ms {
        eprintln!("  {name:<48} prepare={prep:6.2}ms  finish={fin:6.2}ms  total={tot:6.2}ms");
    }
    eprintln!("=== summary: p50={p50:.2}ms  p90={p90:.2}ms  max={max:.2}ms ===");

    // Held-out (cold unseen template) protocol: clear the template inspection
    // cache before every fixture so no fixture benefits from an earlier
    // fixture's template inspection.
    let mut cold_ms = Vec::with_capacity(manifest.fixtures.len());
    for fixture in &manifest.fixtures {
        let scratch = Scratch::new(&format!("timing-cold-{}", fixture.name));
        let target_shards: Vec<PathBuf> = fixture
            .target
            .iter()
            .map(|component| scratch.materialize(component))
            .collect();
        let package = ModelPackagePaths {
            target_shards,
            projector: None,
            draft: fixture
                .draft
                .as_ref()
                .map(|component| scratch.materialize(component)),
            method: fixture.method.into(),
        };
        magnitude_chat::TemplateBundle::clear_inspection_cache();
        let t0 = std::time::Instant::now();
        let prepared = prepare_model_assessment(&package, &setup.structure)
            .unwrap_or_else(|error| panic!("{}: cold prepare failed: {error}", fixture.name));
        let t1 = std::time::Instant::now();
        let _assessment = finish_model_assessment(&prepared, &setup, &[], PROFILE_SEQUENCES)
            .unwrap_or_else(|error| panic!("{}: cold finish failed: {error}", fixture.name));
        let t2 = std::time::Instant::now();
        let prepare_ms = t1.duration_since(t0).as_secs_f64() * 1_000.0;
        let finish_ms = t2.duration_since(t1).as_secs_f64() * 1_000.0;
        let total_ms = t2.duration_since(t0).as_secs_f64() * 1_000.0;
        cold_ms.push((fixture.name.clone(), prepare_ms, finish_ms, total_ms));
    }
    cold_ms.sort_by(|a, b| a.3.total_cmp(&b.3));
    let cold_p50 = cold_ms[n / 2].3;
    let cold_p90 = cold_ms[(n * 9) / 10].3;
    let cold_max = cold_ms[n - 1].3;
    eprintln!("=== assessment_timing_gate_m4_pro [held-out cold template] ({n} fixtures) ===");
    for (name, prep, fin, tot) in &cold_ms {
        eprintln!("  {name:<48} prepare={prep:6.2}ms  finish={fin:6.2}ms  total={tot:6.2}ms");
    }
    eprintln!("=== held-out summary: p50={cold_p50:.2}ms  p90={cold_p90:.2}ms  max={cold_max:.2}ms ===");

    #[cfg(not(debug_assertions))]
    {
        assert!(p90 < 10.0, "p90 assessment latency {p90:.2}ms exceeds 10.0ms bar");
        assert!(max < 50.0, "max assessment latency {max:.2}ms exceeds 50.0ms bar");
    }
}

#[test]
fn assessment_stage_breakdown_m4_pro() {
    const TARGETS: [&str; 5] = [
        "llama-supported",
        "qwen35-supported",
        "qwen35moe-supported",
        "qwen4exp-mtp-head",
        "qwen38-27b-dflash2",
    ];
    let manifest = manifest();
    let mut setup = m4_pro_setup(
        ReleaseRate::of_machine(None),
        ModelPolicy::default(),
        standard_service_limits(),
    );
    setup.measurements = synthetic_measurements();

    let ms = |ns: u64| ns as f64 / 1_000_000.0;
    eprintln!("=== assessment_stage_breakdown_m4_pro ===");
    for name in TARGETS {
        let fixture = manifest
            .fixtures
            .iter()
            .find(|fixture| fixture.name == name)
            .unwrap_or_else(|| panic!("fixture `{name}` not found in manifest"));
        let scratch = Scratch::new(&format!("breakdown-{name}"));
        let target_shards = fixture
            .target
            .iter()
            .map(|component| scratch.materialize(component))
            .collect::<Vec<_>>();
        assert!(!target_shards.is_empty(), "a fixture has a target");
        let package = ModelPackagePaths {
            target_shards,
            projector: None,
            draft: fixture
                .draft
                .as_ref()
                .map(|component| scratch.materialize(component)),
            method: fixture.method.into(),
        };

        let t0 = std::time::Instant::now();
        let (prepared, stages) =
            magnitude_executor::assessment::stage_timer::measure_stages(|| {
                prepare_model_assessment(&package, &setup.structure)
            });
        let prepare_ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let prepared =
            prepared.unwrap_or_else(|error| panic!("{}: prepare failed: {error}", fixture.name));

        let t1 = std::time::Instant::now();
        let _assessment = finish_model_assessment(&prepared, &setup, &[], PROFILE_SEQUENCES)
            .unwrap_or_else(|error| panic!("{}: finish failed: {error}", fixture.name));
        let finish_ns = u64::try_from(t1.elapsed().as_nanos()).unwrap_or(u64::MAX);

        let accounted_ns = stages.header_open_ns
            + stages.chat_inspect_ns
            + stages.admit_other_ns
            + stages.plan_resident_ns
            + stages.target_ns
            + stages.readout_ns
            + stages.native_head_ns
            + stages.other_drafter_ns
            + stages.vision_state_ns
            + stages.host_process_ns
            + stages.decode_demand_ns;
        let other_ms = (prepare_ns as f64 - accounted_ns as f64) / 1_000_000.0;

        eprintln!("--- {name} ---");
        eprintln!("  header_open:        {:8.3} ms", ms(stages.header_open_ns));
        eprintln!("  chat_inspect:       {:8.3} ms", ms(stages.chat_inspect_ns));
        eprintln!("  admit_other:        {:8.3} ms", ms(stages.admit_other_ns));
        eprintln!("  plan_resident:      {:8.3} ms", ms(stages.plan_resident_ns));
        eprintln!("  target:             {:8.3} ms", ms(stages.target_ns));
        eprintln!("  readout:            {:8.3} ms", ms(stages.readout_ns));
        eprintln!(
            "  native_head:        {:8.3} ms  (groups={})",
            ms(stages.native_head_ns),
            stages.native_head_groups
        );
        eprintln!("  other_drafter:      {:8.3} ms", ms(stages.other_drafter_ns));
        eprintln!("  vision_state:       {:8.3} ms", ms(stages.vision_state_ns));
        eprintln!("  host_process:       {:8.3} ms", ms(stages.host_process_ns));
        eprintln!("  decode_demand:      {:8.3} ms", ms(stages.decode_demand_ns));
        eprintln!("  other:              {:8.3} ms", other_ms);
        eprintln!("  prepare_total:      {:8.3} ms", ms(prepare_ns));
        eprintln!("  finish_total:       {:8.3} ms", ms(finish_ns));
        eprintln!("  [admit_inner deep]");
        eprintln!("    fingerprint:        {:8.3} ms", ms(stages.admit_fingerprint_ns));
        eprintln!("    unsupported_weights:{:8.3} ms", ms(stages.admit_unsupported_weights_ns));
        eprintln!("    interpret_target:   {:8.3} ms", ms(stages.admit_interpret_target_ns));
        eprintln!("    tokenizer_vocab:    {:8.3} ms", ms(stages.admit_tokenizer_vocab_ns));
        eprintln!("    gguf_templates:     {:8.3} ms", ms(stages.admit_gguf_templates_ns));
        eprintln!("    admit_drafter:      {:8.3} ms", ms(stages.admit_drafter_ns));
        eprintln!("  [plan_resident deep]");
        eprintln!("    load_derive:        {:8.3} ms", ms(stages.pr_load_derive_ns));
        eprintln!("    record_ports:       {:8.3} ms", ms(stages.pr_record_ports_ns));
        eprintln!("    readout_candidates: {:8.3} ms", ms(stages.pr_readout_candidates_ns));
        eprintln!("    program_plan:       {:8.3} ms", ms(stages.pr_program_plan_ns));
        eprintln!("  [floor_forms deep]");
        eprintln!("    bank_ceiling:       {:8.3} ms", ms(stages.ff_bank_ceiling_ns));
        eprintln!("    rising:             {:8.3} ms", ms(stages.ff_rising_ns));
        eprintln!("    whole_chains:       {:8.3} ms", ms(stages.ff_whole_chains_ns));
        eprintln!("  [SizingDraft deep]");
        eprintln!(
            "    counts:             drafts={}  nodes={}  slice_sets={}",
            stages.sd_drafts_built, stages.sd_nodes_enqueued, stages.sd_slice_sets
        );
        eprintln!("    port_param_spec:    {:8.3} ms", ms(stages.sd_port_param_spec_ns));
        eprintln!("    enqueue:            {:8.3} ms", ms(stages.sd_enqueue_ns));
        eprintln!("    operand_links:      {:8.3} ms", ms(stages.sd_operand_links_ns));
        eprintln!("    storage_compile:    {:8.3} ms", ms(stages.sd_storage_compile_ns));
        eprintln!("    cover_ports:        {:8.3} ms", ms(stages.sd_cover_ports_ns));
        eprintln!(
            "    scratch_configs:    hit={:8.3} ms ({})  miss={:8.3} ms ({})",
            ms(stages.sd_scratch_configs_hit_ns),
            stages.sd_scratch_configs_hits,
            ms(stages.sd_scratch_configs_miss_ns),
            stages.sd_scratch_configs_misses
        );
        eprintln!(
            "    sizing_point:       hit={:8.3} ms ({})  miss={:8.3} ms ({})",
            ms(stages.sd_sizing_point_hit_ns),
            stages.sd_sizing_point_hits,
            ms(stages.sd_sizing_point_miss_ns),
            stages.sd_sizing_point_misses
        );
        eprintln!("    storage_eval:       {:8.3} ms", ms(stages.sd_storage_eval_ns));

        let mut top_configs: Vec<(&'static str, u64, u64)> = stages
            .sd_configs_by_entry
            .iter()
            .map(|(&entry, &(ns, calls))| (entry, ns, calls))
            .collect();
        top_configs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        if !top_configs.is_empty() {
            eprintln!("    top-10 sd_configs_by_entry (miss):");
            for (entry, ns, calls) in top_configs.into_iter().take(10) {
                eprintln!("      {:<40} {:8.3} ms ({} misses)", entry, ms(ns), calls);
            }
        }

        let mut top_points: Vec<(&'static str, u64, u64)> = stages
            .sd_point_by_entry
            .iter()
            .map(|(&entry, &(ns, calls))| (entry, ns, calls))
            .collect();
        top_points.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        if !top_points.is_empty() {
            eprintln!("    top-10 sd_point_by_entry (miss):");
            for (entry, ns, calls) in top_points.into_iter().take(10) {
                eprintln!("      {:<40} {:8.3} ms ({} misses)", entry, ms(ns), calls);
            }
        }
    }
}

#[test]
fn chat_template_inspection_pruning_matches_unpruned_across_manifest() {
    let manifest = manifest();
    let mut checked = 0usize;
    for fixture in &manifest.fixtures {
        let scratch = Scratch::new(&format!("chat-parity-{}", fixture.name));
        let target_path = scratch.materialize(
            fixture
                .target
                .first()
                .expect("a fixture has a target"),
        );
        let Ok(directory) = inspect_header(&target_path) else {
            continue;
        };
        let tokenizer = magnitude_artifacts::TokenizerPayload::from_directory(&directory);
        let Ok(templates) =
            magnitude_artifacts::TemplatePayload::from_directory(&directory, &fixture.name)
        else {
            continue;
        };
        let Ok(bundle) = magnitude_chat::artifacts::gguf_templates(&templates, &tokenizer) else {
            continue;
        };
        let pruned = bundle.inspect();
        let unpruned = bundle.inspect_unpruned();
        assert_eq!(
            pruned, unpruned,
            "pruned TemplateInspection diverged from unpruned on fixture `{}`",
            fixture.name
        );
        checked += 1;
    }
    assert!(checked > 0, "expected at least one fixture with a chat template");
}
