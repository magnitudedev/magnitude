use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component as PathComponent, Path, PathBuf};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::digest::valid_sha256;

pub const SCHEMA_VERSION: &str = "1";

fn default_true() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Category {
    Correctness,
    Performance,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum AdapterKind {
    UpstreamTest,
    UpstreamQualification,
    UpstreamTool,
    Differential,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum CaseStatus {
    Active,
    Planned,
    Disabled,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SourceDefinition {
    pub kind: SourceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_case: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SourceKind {
    UpstreamDerived,
    UpstreamDefined,
    MagnitudeDefined,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FixtureRef {
    pub id: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct ModelSelector {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub all_tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub any_tags: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Requirements {
    #[serde(default = "default_true")]
    pub offline: bool,
    #[serde(default)]
    pub fixtures: Vec<FixtureRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelSelector>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backends: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub architecture_tags: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum InvocationKind {
    UpstreamTest,
    UpstreamTool,
    NativeOracle,
    IcnProbe,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Invocation {
    pub kind: InvocationKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Invocations {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<Invocation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<Invocation>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Statistic {
    Median,
    Mean,
    Minimum,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RatioDirection {
    #[default]
    CandidateOverReference,
    ReferenceOverCandidate,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Comparator {
    Exact {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        paths: Vec<String>,
    },
    Structural {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        paths: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        ignore_paths: Vec<String>,
    },
    Numeric {
        paths: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        exact_paths: Vec<String>,
        absolute_tolerance: f64,
        relative_tolerance: f64,
        #[serde(default)]
        nan_equal: bool,
    },
    OutcomeAgreement {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        accepted_classes: Vec<OutcomeClass>,
    },
    PerformanceRatio {
        metric: String,
        statistic: Statistic,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        minimum_ratio: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        maximum_ratio: Option<f64>,
        #[serde(default)]
        direction: RatioDirection,
        /// Exact output projections that must match before a timing ratio can
        /// pass. Differential microbenchmarks use this to prove the timed
        /// operation still produced the same semantic result.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        exact_output_paths: Vec<String>,
    },
}

impl Comparator {
    pub fn paths(&self) -> &[String] {
        match self {
            Self::Exact { paths }
            | Self::Structural { paths, .. }
            | Self::Numeric { paths, .. } => paths,
            Self::PerformanceRatio {
                exact_output_paths, ..
            } => exact_output_paths,
            Self::OutcomeAgreement { .. } => &[],
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TimingContract {
    pub included: Vec<String>,
    pub excluded: Vec<String>,
    pub warmup_iterations: u64,
    pub measurement_iterations: u64,
    pub synchronize: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CaseDefinition {
    #[serde(rename = "$schema", default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    pub schema_version: String,
    pub id: String,
    pub title: String,
    pub description: String,
    pub category: Category,
    pub primitive: String,
    pub adapter: AdapterKind,
    pub operation: String,
    pub status: CaseStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disabled_reason: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub source: SourceDefinition,
    pub inputs: serde_json::Map<String, Value>,
    pub requirements: Requirements,
    pub invocations: Invocations,
    pub comparison: Comparator,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prerequisites: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingContract>,
}

impl CaseDefinition {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            bail!(
                "case {} has unsupported schema_version {}",
                self.id,
                self.schema_version
            );
        }
        validate_case_id(&self.id)?;
        validate_operation(&self.operation)
            .with_context(|| format!("case {} operation", self.id))?;
        if self.title.trim().is_empty() || self.description.trim().is_empty() {
            bail!("case {} title and description must be non-empty", self.id);
        }
        let expected_category = if self.primitive.starts_with('C') {
            Category::Correctness
        } else if self.primitive.starts_with('P') {
            Category::Performance
        } else {
            bail!("case {} has invalid primitive {}", self.id, self.primitive);
        };
        if self.category != expected_category || !valid_primitive(&self.primitive) {
            bail!(
                "case {} category and primitive {} are inconsistent",
                self.id,
                self.primitive
            );
        }
        if self.category == Category::Performance && self.timing.is_none() {
            bail!("performance case {} requires a timing contract", self.id);
        }
        if self.category == Category::Performance && self.prerequisites.is_empty() {
            bail!(
                "performance case {} requires explicit correctness prerequisites",
                self.id
            );
        }
        if self.category == Category::Correctness && !self.prerequisites.is_empty() {
            bail!(
                "correctness case {} cannot declare performance prerequisites",
                self.id
            );
        }
        let mut prerequisites = std::collections::BTreeSet::new();
        for prerequisite in &self.prerequisites {
            validate_case_id(prerequisite)?;
            if !prerequisite.starts_with("correctness.") || !prerequisites.insert(prerequisite) {
                bail!(
                    "case {} has invalid or duplicate prerequisite {}",
                    self.id,
                    prerequisite
                );
            }
        }
        if self.status == CaseStatus::Disabled
            && self
                .disabled_reason
                .as_deref()
                .is_none_or(|reason| reason.trim().is_empty())
        {
            bail!("disabled case {} requires disabled_reason", self.id);
        }
        if !self.requirements.offline {
            bail!("case {} must declare offline=true", self.id);
        }
        validate_requirements(&self.requirements)
            .with_context(|| format!("case {} requirements", self.id))?;
        validate_source(&self.source).with_context(|| format!("case {} source", self.id))?;
        let reference = self.invocations.reference.as_ref();
        let candidate = self.invocations.candidate.as_ref();
        match self.adapter {
            AdapterKind::UpstreamTest => {
                if reference.map(|value| value.kind) != Some(InvocationKind::UpstreamTest) {
                    bail!(
                        "upstream-test case {} requires an upstream-test reference",
                        self.id
                    );
                }
                if candidate.is_some() {
                    bail!("upstream-test case {} cannot declare a candidate", self.id);
                }
            }
            AdapterKind::UpstreamQualification => {
                if reference.map(|value| value.kind) != Some(InvocationKind::UpstreamTool) {
                    bail!(
                        "upstream-qualification case {} requires an upstream-tool reference",
                        self.id
                    );
                }
                if candidate.is_some() {
                    bail!(
                        "upstream-qualification case {} cannot declare a candidate",
                        self.id
                    );
                }
            }
            AdapterKind::UpstreamTool => {
                if reference.map(|value| value.kind) != Some(InvocationKind::UpstreamTool) {
                    bail!(
                        "upstream-tool case {} requires an upstream-tool reference",
                        self.id
                    );
                }
                if candidate.is_none() {
                    bail!(
                        "upstream-tool case {} requires a candidate invocation",
                        self.id
                    );
                }
            }
            AdapterKind::Differential => {
                if reference.map(|value| value.kind) != Some(InvocationKind::NativeOracle)
                    || candidate.map(|value| value.kind) != Some(InvocationKind::IcnProbe)
                {
                    bail!(
                        "differential case {} requires native-oracle reference and icn-probe candidate",
                        self.id
                    );
                }
            }
        }
        if self.status == CaseStatus::Active {
            for (role, invocation) in [("reference", reference), ("candidate", candidate)] {
                if let Some(invocation) = invocation
                    && invocation.target.as_deref().is_none_or(str::is_empty)
                {
                    bail!("active case {} {role} invocation requires target", self.id);
                }
            }
        }
        validate_comparator(&self.comparison)
            .with_context(|| format!("case {} comparator", self.id))?;
        Ok(())
    }
}

const SUPPORTED_BACKENDS: &[&str] = &[
    "cpu", "metal", "cuda", "vulkan", "kompute", "sycl", "opencl",
];

fn validate_requirements(requirements: &Requirements) -> anyhow::Result<()> {
    if requirements.capabilities.is_empty() {
        bail!("at least one required capability must be declared");
    }
    let mut capabilities = BTreeSet::new();
    for capability in &requirements.capabilities {
        validate_operation(capability)
            .with_context(|| format!("invalid required capability {capability:?}"))?;
        if !capabilities.insert(capability) {
            bail!("duplicate required capability {capability:?}");
        }
    }

    let mut backends = BTreeSet::new();
    for backend in &requirements.backends {
        if !SUPPORTED_BACKENDS.contains(&backend.as_str()) {
            bail!("unsupported required backend {backend:?}");
        }
        if !backends.insert(backend) {
            bail!("duplicate required backend {backend:?}");
        }
    }

    let mut fixture_ids = BTreeSet::new();
    let mut fixture_paths = BTreeSet::new();
    for fixture in &requirements.fixtures {
        validate_operation(&fixture.id)
            .with_context(|| format!("invalid fixture id {:?}", fixture.id))?;
        validate_fixture_path(&fixture.path)
            .with_context(|| format!("invalid fixture path {:?}", fixture.path))?;
        let digest = fixture
            .sha256
            .as_deref()
            .context("fixture sha256 is required")?;
        if !valid_sha256(digest) {
            bail!("fixture {} has an invalid sha256", fixture.id);
        }
        if !fixture_ids.insert(&fixture.id) {
            bail!("duplicate fixture id {:?}", fixture.id);
        }
        if !fixture_paths.insert(&fixture.path) {
            bail!("duplicate fixture path {:?}", fixture.path);
        }
    }
    Ok(())
}

fn validate_source(source: &SourceDefinition) -> anyhow::Result<()> {
    match source.kind {
        SourceKind::UpstreamDerived | SourceKind::UpstreamDefined => {
            let upstream_file = source
                .upstream_file
                .as_deref()
                .context("upstream-defined/derived source requires upstream_file")?;
            validate_portable_relative_path(upstream_file)
                .with_context(|| format!("invalid upstream_file {upstream_file:?}"))?;
        }
        SourceKind::MagnitudeDefined if source.upstream_file.is_some() => {
            bail!("magnitude-defined source cannot declare upstream_file");
        }
        SourceKind::MagnitudeDefined => {}
    }
    Ok(())
}

fn validate_fixture_path(value: &str) -> anyhow::Result<()> {
    validate_portable_relative_path(value)?;
    let mut components = value.split('/');
    if components.next() != Some("fixtures") || components.next().is_none() {
        bail!("fixture path must be below fixtures/");
    }
    Ok(())
}

fn validate_portable_relative_path(value: &str) -> anyhow::Result<()> {
    let path = Path::new(value);
    if value.is_empty()
        || value.contains('\\')
        || path.is_absolute()
        || value
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."))
        || path
            .components()
            .any(|component| !matches!(component, PathComponent::Normal(_)))
    {
        bail!("path must be a non-empty normalized relative path using '/' separators");
    }
    Ok(())
}

fn validate_comparator(comparator: &Comparator) -> anyhow::Result<()> {
    for path in comparator.paths() {
        validate_json_pointer(path)?;
    }
    match comparator {
        Comparator::Structural { ignore_paths, .. } => {
            for path in ignore_paths {
                validate_json_pointer(path)?;
            }
        }
        Comparator::Numeric {
            paths,
            exact_paths,
            absolute_tolerance,
            relative_tolerance,
            ..
        } => {
            if paths.is_empty() {
                bail!("numeric comparator requires at least one path");
            }
            for path in exact_paths {
                validate_json_pointer(path)?;
            }
            if !absolute_tolerance.is_finite()
                || !relative_tolerance.is_finite()
                || *absolute_tolerance < 0.0
                || *relative_tolerance < 0.0
            {
                bail!("numeric tolerances must be finite and non-negative");
            }
        }
        Comparator::PerformanceRatio {
            minimum_ratio,
            maximum_ratio,
            ..
        } => {
            if minimum_ratio.is_none() && maximum_ratio.is_none()
                || minimum_ratio.is_some_and(|value| !value.is_finite() || value <= 0.0)
                || maximum_ratio.is_some_and(|value| !value.is_finite() || value <= 0.0)
                || minimum_ratio
                    .zip(*maximum_ratio)
                    .is_some_and(|(minimum, maximum)| minimum > maximum)
            {
                bail!(
                    "at least one finite, positive performance ratio bound is required; bounds must be ordered"
                );
            }
        }
        Comparator::Exact { .. } | Comparator::OutcomeAgreement { .. } => {}
    }
    Ok(())
}

pub(crate) fn valid_primitive(value: &str) -> bool {
    match value.strip_prefix('C') {
        Some(number) => number.parse::<u8>().is_ok_and(|number| number <= 13),
        None => value
            .strip_prefix('P')
            .and_then(|number| number.parse::<u8>().ok())
            .is_some_and(|number| number <= 8),
    }
}

pub fn validate_case_id(value: &str) -> anyhow::Result<()> {
    let Some((category, suffix)) = value.split_once('.') else {
        bail!("case id {value:?} must start with correctness. or performance.");
    };
    if !matches!(category, "correctness" | "performance")
        || suffix.is_empty()
        || !suffix
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || b".-".contains(&byte))
    {
        bail!("invalid case id {value:?}");
    }
    Ok(())
}

pub fn validate_operation(value: &str) -> anyhow::Result<()> {
    if value.is_empty()
        || !value.split(['.', '-']).all(|component| {
            !component.is_empty()
                && component
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        })
    {
        bail!("invalid operation {value:?}");
    }
    Ok(())
}

pub fn validate_json_pointer(value: &str) -> anyhow::Result<()> {
    if value.is_empty() {
        return Ok(());
    }
    if !value.starts_with('/') {
        bail!("invalid JSON pointer {value:?}");
    }
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'~' {
            if bytes
                .get(index + 1)
                .is_none_or(|next| !matches!(next, b'0' | b'1'))
            {
                bail!("invalid JSON pointer escape in {value:?}");
            }
            index += 1;
        }
        index += 1;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ProducerRole {
    Qualification,
    Reference,
    Candidate,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Producer {
    pub role: ProducerRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<InvocationKind>,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum OutcomeClass {
    Success,
    Unsupported,
    InvalidInput,
    Cancelled,
    RuntimeError,
    Skipped,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub class: OutcomeClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WorkDefinition {
    pub parameters: serde_json::Map<String, Value>,
    pub included: Vec<String>,
    pub excluded: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub name: String,
    pub unit: String,
    pub samples: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub kind: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tree_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dirty: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BuildInfo {
    pub build_type: String,
    pub compiler: String,
    pub compiler_version: String,
    pub flags: Vec<String>,
    pub assertions: Option<bool>,
    pub sanitizers: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HostInfo {
    pub os: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub os_version: Option<String>,
    pub arch: String,
    pub cpu: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_cpus: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeviceInfo {
    pub backend: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactInfo {
    pub kind: String,
    pub id: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub components: Vec<Component>,
    pub build: BuildInfo,
    pub host: HostInfo,
    pub devices: Vec<DeviceInfo>,
    pub artifacts: Vec<ArtifactInfo>,
    pub effective_configuration: serde_json::Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRecord {
    pub schema_version: String,
    pub run_id: String,
    pub case_id: String,
    pub category: Category,
    pub primitive: String,
    pub operation: String,
    pub recorded_at: String,
    pub producer: Producer,
    pub outcome: Outcome,
    pub work: WorkDefinition,
    pub output: Value,
    pub measurements: Vec<Measurement>,
    pub provenance: Provenance,
    pub warnings: Vec<String>,
}

impl EvidenceRecord {
    pub fn validate_for(&self, case: &CaseDefinition, role: ProducerRole) -> anyhow::Result<()> {
        if self.schema_version != SCHEMA_VERSION
            || self.case_id != case.id
            || self.category != case.category
            || self.primitive != case.primitive
            || self.operation != case.operation
            || self.producer.role != role
        {
            bail!(
                "evidence envelope does not match case {} and role {role:?}",
                case.id
            );
        }
        if self.producer.name.trim().is_empty() || self.provenance.components.is_empty() {
            bail!("evidence {} has incomplete producer provenance", case.id);
        }
        for measurement in &self.measurements {
            if measurement.samples.is_empty()
                || measurement.samples.iter().any(|sample| !sample.is_finite())
            {
                bail!(
                    "evidence {} measurement {} has invalid samples",
                    case.id,
                    measurement.name
                );
            }
        }
        if case.category == Category::Performance
            && self.outcome.class == OutcomeClass::Success
            && self.measurements.is_empty()
        {
            bail!("performance evidence {} requires measurements", case.id);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRef {
    pub path: String,
    pub sha256: String,
    pub producer_role: ProducerRole,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Validity {
    pub valid: bool,
    pub reasons: Vec<String>,
    pub matched_work: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_order: Option<EngineOrder>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contamination_warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum EngineOrder {
    ReferenceFirst,
    CandidateFirst,
    AlternatingPairs,
    NotApplicable,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ComparisonStatus {
    Pass,
    Fail,
    Invalid,
    Skipped,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum MismatchKind {
    Missing,
    Unexpected,
    Type,
    Value,
    NumericTolerance,
    Outcome,
    PerformanceBound,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Mismatch {
    pub path: String,
    pub kind: MismatchKind,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub absolute_difference: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relative_difference: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparisonMetric {
    pub name: String,
    pub unit: String,
    pub value: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_value: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparisonRecord {
    pub schema_version: String,
    pub run_id: String,
    pub case_id: String,
    pub category: Category,
    pub primitive: String,
    pub recorded_at: String,
    pub reference: EvidenceRef,
    pub candidate: EvidenceRef,
    pub comparator: Comparator,
    pub validity: Validity,
    pub status: ComparisonStatus,
    pub mismatches: Vec<Mismatch>,
    pub metrics: Vec<ComparisonMetric>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DecoderKind {
    ExitStatus,
    #[default]
    Json,
    Jsonl,
    EvidenceJson,
    EvidenceJsonl,
    ProbeJsonl,
    LlamaBenchJson,
    BatchedBenchJsonl,
    PerplexityText,
    BackendOpsSql,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum StdinKind {
    #[default]
    None,
    ProbeJsonl,
    CaseJson,
    CaseJsonl,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CommandSpec {
    pub program: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub clear_env: bool,
    #[serde(default)]
    pub stdin: StdinKind,
    #[serde(default)]
    pub decoder: DecoderKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_stdout_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_stderr_bytes: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(id: &str, path: &str) -> FixtureRef {
        FixtureRef {
            id: id.to_owned(),
            path: path.to_owned(),
            sha256: Some("0".repeat(64)),
        }
    }

    fn requirements(fixtures: Vec<FixtureRef>) -> Requirements {
        Requirements {
            offline: true,
            fixtures,
            model: None,
            capabilities: vec!["test.capability".to_owned()],
            backends: Vec::new(),
            architecture_tags: Vec::new(),
        }
    }

    #[test]
    fn validates_json_pointers() {
        assert!(validate_json_pointer("").is_ok());
        assert!(validate_json_pointer("/items/0/a~1b").is_ok());
        assert!(validate_json_pointer("items").is_err());
        assert!(validate_json_pointer("/~2").is_err());
    }

    #[test]
    fn primitive_ranges_are_bounded() {
        assert!(valid_primitive("C0"));
        assert!(valid_primitive("C13"));
        assert!(valid_primitive("P8"));
        assert!(!valid_primitive("C14"));
        assert!(!valid_primitive("P9"));
    }

    #[test]
    fn requirements_require_pinned_unique_fixtures() {
        let mut missing_digest = requirements(vec![fixture("input", "fixtures/input.json")]);
        missing_digest.fixtures[0].sha256 = None;
        assert!(validate_requirements(&missing_digest).is_err());

        let duplicate_id = requirements(vec![
            fixture("input", "fixtures/input.json"),
            fixture("input", "fixtures/other.json"),
        ]);
        assert!(validate_requirements(&duplicate_id).is_err());

        let duplicate_path = requirements(vec![
            fixture("input", "fixtures/input.json"),
            fixture("other", "fixtures/input.json"),
        ]);
        assert!(validate_requirements(&duplicate_path).is_err());
    }

    #[test]
    fn requirements_reject_unsafe_fixture_paths() {
        for path in [
            "../input.json",
            "fixtures/../input.json",
            "/fixtures/input.json",
            "fixtures//input.json",
            "fixtures\\input.json",
            "input.json",
        ] {
            assert!(
                validate_requirements(&requirements(vec![fixture("input", path)])).is_err(),
                "unsafe path was accepted: {path}"
            );
        }
    }

    #[test]
    fn requirements_validate_capability_and_backend_vocabulary() {
        let mut malformed_capability = requirements(Vec::new());
        malformed_capability.capabilities = vec!["bad..capability".to_owned()];
        assert!(validate_requirements(&malformed_capability).is_err());

        let mut duplicate_capability = requirements(Vec::new());
        duplicate_capability.capabilities = vec!["cpu".to_owned(), "cpu".to_owned()];
        assert!(validate_requirements(&duplicate_capability).is_err());

        let mut unsupported_backend = requirements(Vec::new());
        unsupported_backend.backends = vec!["quantum".to_owned()];
        assert!(validate_requirements(&unsupported_backend).is_err());

        let mut duplicate_backend = requirements(Vec::new());
        duplicate_backend.backends = vec!["cpu".to_owned(), "cpu".to_owned()];
        assert!(validate_requirements(&duplicate_backend).is_err());
    }

    #[test]
    fn source_files_are_kind_appropriate_and_safe() {
        let upstream = |upstream_file: Option<&str>| SourceDefinition {
            kind: SourceKind::UpstreamDefined,
            upstream_file: upstream_file.map(str::to_owned),
            upstream_case: None,
            note: None,
        };
        assert!(validate_source(&upstream(Some("tests/test.cpp"))).is_ok());
        assert!(validate_source(&upstream(None)).is_err());
        assert!(validate_source(&upstream(Some("../test.cpp"))).is_err());

        let magnitude = SourceDefinition {
            kind: SourceKind::MagnitudeDefined,
            upstream_file: Some("tests/test.cpp".to_owned()),
            upstream_case: None,
            note: None,
        };
        assert!(validate_source(&magnitude).is_err());
    }
}
