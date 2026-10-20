//! The shared real-header fixtures (`fixtures/assessment/`): every
//! fixture's expected verdict, asserted through the engine's public
//! assessment entry point on the CPU backend, which every build has. The
//! manifest format is described in the fixtures' README.

use flate2::read::GzDecoder;
use magnitude_artifacts::gguf::{Directory, Metadata, Scalar, Value, inspect_header};
use magnitude_engine::{
    assessment::{
        AssessmentSetup, DrafterVerdict, ModelAssessment, ModelPackagePaths, SpeculativeMethod,
        assess_model,
    },
    error::UnsupportedReason,
    fingerprint::ModelFingerprint,
    options::{ModelMethod, ModelPolicy, standard_service_limits},
};
use magnitude_executor::platform::{DeviceRequest, MemoryReserves};
use seismic::BackendName;
use serde::Deserialize;
use std::path::{Path, PathBuf};

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
}

impl From<SpeculativeMethod> for Bound {
    fn from(method: SpeculativeMethod) -> Self {
        match method {
            SpeculativeMethod::Mtp => Self::Mtp,
            SpeculativeMethod::DFlash => Self::DFlash,
            SpeculativeMethod::DSpark => Self::DSpark,
            SpeculativeMethod::DFlash2 => Self::DFlash2,
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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/assessment")
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

fn cpu_setup() -> AssessmentSetup {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    AssessmentSetup::discover(
        &catalog,
        DeviceRequest::Backend(BackendName::Cpu),
        MemoryReserves::standard(),
        ModelPolicy::default(),
        standard_service_limits(),
    )
    .unwrap_or_else(|error| panic!("the CPU backend is selectable: {error}"))
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
        let assessment = assess_model(&package, &setup, &[])
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
                out.extend(values.first().map_or(8, kind).to_le_bytes());
                out.extend((values.len() as u64).to_le_bytes());
                for value in values {
                    scalar(&mut out, value);
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
    let Some(Value::Array(mut ratios)) = combined
        .value("qwen4exp.attention.compress_ratios")
        .cloned()
    else {
        panic!("the target names its compression ratios");
    };
    ratios.push(Scalar::Unsigned(0));
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
