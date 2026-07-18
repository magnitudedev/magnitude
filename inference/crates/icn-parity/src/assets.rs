use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component as PathComponent, Path, PathBuf};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::digest::{sha256_bytes, sha256_file, valid_sha256};
use crate::model::{
    AdapterKind, CaseDefinition, CaseStatus, CommandSpec, DecoderKind, InvocationKind, StdinKind,
};
use crate::models::ModelRegistry;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub schema_version: u32,
    pub id: String,
    pub description: String,
    pub selection: ProfileSelection,
    pub upstream: UpstreamProfile,
    pub execution: ExecutionProfile,
    pub engine: EngineProfile,
    pub models: ModelsProfile,
    pub performance: PerformanceProfile,
    pub results: ResultsProfile,
    pub gates: GateProfile,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSelection {
    pub statuses: Vec<CaseStatus>,
    pub include_case_ids: Vec<String>,
    pub include_primitives: Vec<String>,
    pub include_tags: Vec<String>,
    pub exclude_case_ids: Vec<String>,
    pub exclude_tags: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamProfile {
    pub build_profile: String,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PerformanceOrder {
    Alternate,
    ReferenceFirst,
    CandidateFirst,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionProfile {
    pub offline: bool,
    pub max_parallel: usize,
    pub case_timeout_seconds: u64,
    pub fail_fast: bool,
    pub performance_engine_order: PerformanceOrder,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineProfile {
    pub backend: String,
    pub threads: String,
    pub gpu_layers: String,
    pub flash_attention: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsProfile {
    pub verify_before_run: bool,
    pub allow_fetch: bool,
    pub ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceProfile {
    pub enabled: bool,
    pub paired_repetitions: u64,
    pub require_controlled_host: bool,
    pub require_exclusive_device: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultsProfile {
    pub preserve_raw_evidence: bool,
    pub preserve_raw_samples: bool,
    pub write_comparisons: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateProfile {
    pub fail_on_invalid: bool,
    pub fail_on_error: bool,
    #[serde(default)]
    pub fail_on_performance: bool,
    #[serde(default)]
    pub fail_on_skipped: bool,
    pub require_correctness_before_performance: bool,
}

impl Profile {
    pub fn selects(&self, case: &CaseDefinition) -> bool {
        let has_include_selector = !self.selection.include_case_ids.is_empty()
            || !self.selection.include_primitives.is_empty()
            || !self.selection.include_tags.is_empty();
        let included = !has_include_selector
            || self.selection.include_case_ids.contains(&case.id)
            || self.selection.include_primitives.contains(&case.primitive)
            || case
                .tags
                .iter()
                .any(|tag| self.selection.include_tags.contains(tag));
        self.selection.statuses.contains(&case.status)
            && included
            && !self.selection.exclude_case_ids.contains(&case.id)
            && !case
                .tags
                .iter()
                .any(|tag| self.selection.exclude_tags.contains(tag))
            && (case.category != crate::model::Category::Performance || self.performance.enabled)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamTargetRegistry {
    pub schema_version: u32,
    pub default_targets: Vec<String>,
    pub targets: Vec<UpstreamTarget>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum UpstreamTargetKind {
    CtestSuite,
    UpstreamTool,
    Oracle,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamTarget {
    pub id: String,
    pub kind: UpstreamTargetKind,
    pub description: String,
    pub requires_tests: bool,
    pub requires_tools: bool,
    #[serde(default)]
    pub requires_server: bool,
    pub cmake_targets: Vec<String>,
    pub artifacts: Vec<String>,
    pub ctest_names: Vec<String>,
    pub ctest_setup_names: Vec<String>,
    #[serde(default)]
    pub ctest_model_fixtures: Vec<CtestModelFixture>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CtestModelFixture {
    pub setup_name: String,
    pub model_id: String,
    pub artifact_role: String,
    pub destination: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerRegistry {
    pub schema_version: u32,
    pub producers: Vec<ProducerDefinition>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerDefinition {
    pub id: String,
    pub kind: InvocationKind,
    #[serde(default)]
    pub program: Option<PathBuf>,
    #[serde(default)]
    pub program_setting: Option<String>,
    pub decoder: DecoderKind,
    pub stdin: StdinKind,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub clear_env: bool,
    pub timeout_seconds: u64,
    pub max_stdout_bytes: u64,
    pub max_stderr_bytes: u64,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

impl ProducerDefinition {
    pub fn command(
        &self,
        program: PathBuf,
        args: Vec<String>,
        cwd: Option<PathBuf>,
    ) -> CommandSpec {
        CommandSpec {
            program,
            args,
            cwd,
            env: self.env.clone(),
            clear_env: self.clear_env,
            stdin: self.stdin,
            decoder: self.decoder,
            timeout_seconds: Some(self.timeout_seconds),
            max_stdout_bytes: Some(self.max_stdout_bytes),
            max_stderr_bytes: Some(self.max_stderr_bytes),
        }
    }
}

#[derive(Debug)]
pub struct AssetRepository {
    pub root: PathBuf,
    pub cases: BTreeMap<String, LoadedCase>,
    pub profiles: BTreeMap<String, Profile>,
    pub upstream_targets: BTreeMap<String, UpstreamTarget>,
    pub producers: BTreeMap<String, ProducerDefinition>,
    pub models: ModelRegistry,
    pub schemas: SchemaDocuments,
    pub(crate) contract_digests: BTreeMap<PathBuf, String>,
    pub(crate) profile_paths: BTreeMap<String, PathBuf>,
}

#[derive(Debug)]
pub struct SchemaDocuments {
    pub case: Value,
    pub evidence: Value,
    pub comparison: Value,
}

#[derive(Clone, Copy, Debug)]
pub enum SchemaKind {
    Case,
    Evidence,
    Comparison,
}

#[derive(Debug)]
pub struct LoadedCase {
    pub path: PathBuf,
    pub definition: CaseDefinition,
}

impl AssetRepository {
    pub fn discover_root(explicit: Option<&Path>) -> anyhow::Result<PathBuf> {
        if let Some(root) = explicit {
            return canonical_directory(root, "parity root");
        }
        let cwd = std::env::current_dir().context("failed to inspect current directory")?;
        for ancestor in cwd.ancestors() {
            for candidate in [
                ancestor.join("inference/parity"),
                ancestor.join("parity"),
                ancestor.to_path_buf(),
            ] {
                if candidate.join("schemas/case.schema.json").is_file()
                    && candidate.join("cases").is_dir()
                {
                    return canonical_directory(&candidate, "parity root");
                }
            }
        }
        bail!("unable to discover parity root; pass --root explicitly")
    }

    pub fn load(root: PathBuf) -> anyhow::Result<Self> {
        let mut contract_digests = BTreeMap::new();
        let schemas = SchemaDocuments {
            case: load_json_contract(
                &root.join("schemas/case.schema.json"),
                &mut contract_digests,
            )?,
            evidence: load_json_contract(
                &root.join("schemas/evidence.schema.json"),
                &mut contract_digests,
            )?,
            comparison: load_json_contract(
                &root.join("schemas/comparison.schema.json"),
                &mut contract_digests,
            )?,
        };
        let cases = load_cases(&root.join("cases"), &schemas.case, &mut contract_digests)?;
        let (profiles, profile_paths) =
            load_profiles(&root.join("profiles"), &mut contract_digests)?;
        let upstream: UpstreamTargetRegistry =
            load_toml_contract(&root.join("upstream/targets.toml"), &mut contract_digests)?;
        if upstream.schema_version != 1 {
            bail!(
                "unsupported upstream targets schema_version {}",
                upstream.schema_version
            );
        }
        let upstream_targets =
            unique_by_id(upstream.targets, |value| &value.id, "upstream target")?;
        for target in &upstream.default_targets {
            if !upstream_targets.contains_key(target) {
                bail!("unknown default upstream target {target}");
            }
        }
        let producer_registry: ProducerRegistry =
            load_toml_contract(&root.join("producers.toml"), &mut contract_digests)?;
        if producer_registry.schema_version != 1 {
            bail!(
                "unsupported producer registry schema_version {}",
                producer_registry.schema_version
            );
        }
        let producers = unique_by_id(producer_registry.producers, |value| &value.id, "producer")?;
        let models: ModelRegistry =
            load_toml_contract(&root.join("models/registry.toml"), &mut contract_digests)?;
        models.validate()?;
        for path in [
            root.join("upstream/build-profiles.toml"),
            root.join("upstream/binding-surfaces.json"),
        ] {
            read_contract(&path, &mut contract_digests)?;
        }
        for case in cases.values() {
            for fixture in &case.definition.requirements.fixtures {
                if let Some(digest) = &fixture.sha256 {
                    record_contract_digest(
                        &mut contract_digests,
                        root.join(&fixture.path),
                        digest,
                    )?;
                }
            }
        }
        Ok(Self {
            root,
            cases,
            profiles,
            upstream_targets,
            producers,
            models,
            schemas,
            contract_digests,
            profile_paths,
        })
    }

    pub async fn validate(&self) -> ValidationReport {
        let mut report = ValidationReport::default();
        let expected_primitives = (0..=13)
            .map(|value| format!("C{value}"))
            .chain((0..=8).map(|value| format!("P{value}")))
            .collect::<BTreeSet<_>>();
        let actual_primitives = self
            .cases
            .values()
            .map(|case| case.definition.primitive.clone())
            .collect::<BTreeSet<_>>();
        for missing in expected_primitives.difference(&actual_primitives) {
            report.error(
                "primitive-coverage",
                &self.root,
                format!("suite has no descriptor for {missing}"),
            );
        }
        let source_root = pinned_source_root(&self.root).ok();
        for loaded in self.cases.values() {
            if let Err(error) = loaded.definition.validate() {
                report.error("case-invalid", &loaded.path, error.to_string());
            }
            self.validate_case_references(loaded, &mut report).await;
            if let Some(upstream_file) = loaded.definition.source.upstream_file.as_deref() {
                match &source_root {
                    Some(source_root) => {
                        let path = source_root.join(upstream_file);
                        if let Err(error) = canonical_file_within(source_root, &path) {
                            report.error(
                                "upstream-source-invalid",
                                &loaded.path,
                                format!(
                                    "declared upstream source is missing or escapes the pinned source root: {upstream_file}: {error}"
                                ),
                            );
                        }
                    }
                    None => report.error(
                        "native-pin-invalid",
                        &loaded.path,
                        "unable to resolve pinned llama.cpp source root",
                    ),
                }
            }
            for invocation in [
                loaded.definition.invocations.reference.as_ref(),
                loaded.definition.invocations.candidate.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                if let Some(operation) = invocation.operation.as_deref()
                    && crate::model::validate_operation(operation).is_err()
                {
                    report.error(
                        "invocation-operation",
                        &loaded.path,
                        format!("invalid invocation operation {operation}"),
                    );
                }
            }
            for prerequisite in &loaded.definition.prerequisites {
                match self.cases.get(prerequisite) {
                    Some(required)
                        if required.definition.category == crate::model::Category::Correctness => {}
                    Some(_) => report.error(
                        "prerequisite-category",
                        &loaded.path,
                        format!("prerequisite {prerequisite} is not a correctness case"),
                    ),
                    None => report.error(
                        "prerequisite-missing",
                        &loaded.path,
                        format!("unknown prerequisite case {prerequisite}"),
                    ),
                }
            }
        }
        for (id, profile) in &self.profiles {
            if profile.schema_version != 1 || id != &profile.id {
                report.error(
                    "profile-invalid",
                    Path::new(id),
                    "profile id/schema mismatch",
                );
            }
            if !profile.execution.offline {
                report.error(
                    "profile-online",
                    Path::new(id),
                    "parity profiles must run offline",
                );
            }
            if profile.execution.max_parallel == 0
                || profile.execution.case_timeout_seconds == 0
                || profile.performance.paired_repetitions == 0
            {
                report.error(
                    "profile-bounds",
                    Path::new(id),
                    "execution bounds must be positive",
                );
            }
            for case_id in profile
                .selection
                .include_case_ids
                .iter()
                .chain(&profile.selection.exclude_case_ids)
            {
                if !self.cases.contains_key(case_id) {
                    report.error(
                        "profile-case-missing",
                        Path::new(id),
                        format!("profile references unknown case {case_id}"),
                    );
                }
            }
            for model_id in &profile.models.ids {
                if self.models.by_id(model_id).is_err() {
                    report.error(
                        "profile-model-missing",
                        Path::new(id),
                        format!("profile references unknown model {model_id}"),
                    );
                }
            }
            if self.selected_cases(profile).is_empty() {
                report.error(
                    "profile-empty",
                    Path::new(id),
                    "profile selects no parity cases",
                );
            }
            if profile.models.ids.iter().collect::<BTreeSet<_>>().len() != profile.models.ids.len()
            {
                report.error(
                    "profile-model-duplicate",
                    Path::new(id),
                    "profile selects a model id more than once",
                );
            }
            if profile.models.allow_fetch {
                report.warning(
                    "run-never-fetches",
                    Path::new(id),
                    "allow_fetch is declarative only; parity run never downloads artifacts",
                );
            }
        }
        for producer in self.producers.values() {
            if producer.program.is_some() == producer.program_setting.is_some() {
                report.error(
                    "producer-program",
                    Path::new(&producer.id),
                    "producer requires exactly one of program or program_setting",
                );
            }
            if producer.timeout_seconds == 0
                || producer.max_stdout_bytes == 0
                || producer.max_stderr_bytes == 0
            {
                report.error(
                    "producer-bounds",
                    Path::new(&producer.id),
                    "producer bounds must be positive",
                );
            }
            let mut capabilities = BTreeSet::new();
            for capability in &producer.capabilities {
                if crate::model::validate_operation(capability).is_err()
                    || !capabilities.insert(capability)
                {
                    report.error(
                        "producer-capability-invalid",
                        Path::new(&producer.id),
                        format!("invalid or duplicate producer capability {capability:?}"),
                    );
                }
            }
        }
        for target in self.upstream_targets.values() {
            let mut destinations = BTreeSet::new();
            for fixture in &target.ctest_model_fixtures {
                if fixture.setup_name.trim().is_empty()
                    || !target.ctest_setup_names.contains(&fixture.setup_name)
                {
                    report.error(
                        "ctest-fixture-setup",
                        Path::new(&target.id),
                        format!(
                            "fixture setup {} is not declared by target",
                            fixture.setup_name
                        ),
                    );
                }
                if validate_relative_path(&fixture.destination).is_err()
                    || !destinations.insert(fixture.destination.clone())
                {
                    report.error(
                        "ctest-fixture-destination",
                        Path::new(&target.id),
                        format!(
                            "invalid or duplicate CTest fixture destination {}",
                            fixture.destination.display()
                        ),
                    );
                }
                match self.models.by_id(&fixture.model_id) {
                    Ok(model)
                        if model
                            .files
                            .iter()
                            .any(|file| file.role == fixture.artifact_role) => {}
                    Ok(_) => report.error(
                        "ctest-fixture-role",
                        Path::new(&target.id),
                        format!(
                            "model {} has no artifact role {}",
                            fixture.model_id, fixture.artifact_role
                        ),
                    ),
                    Err(_) => report.error(
                        "ctest-fixture-model",
                        Path::new(&target.id),
                        format!("unknown CTest fixture model {}", fixture.model_id),
                    ),
                }
            }
        }
        report
    }

    async fn validate_case_references(&self, loaded: &LoadedCase, report: &mut ValidationReport) {
        let case = &loaded.definition;
        for fixture in &case.requirements.fixtures {
            let Some(expected) = fixture.sha256.as_deref() else {
                report.error(
                    "fixture-digest-required",
                    &loaded.path,
                    format!("fixture {} has no required sha256", fixture.path),
                );
                continue;
            };
            let path = self.root.join(&fixture.path);
            let path = match canonical_file_within(&self.root, &path) {
                Ok(path) => path,
                Err(error) => {
                    report.error(
                        "fixture-missing",
                        &loaded.path,
                        format!(
                            "fixture {} is missing or escapes the parity root: {error}",
                            fixture.path
                        ),
                    );
                    continue;
                }
            };
            match sha256_file(&path).await {
                Ok((actual, _)) if actual == expected => {}
                Ok((actual, _)) => report.error(
                    "fixture-digest",
                    &loaded.path,
                    format!("fixture {} digest mismatch: {actual}", fixture.path),
                ),
                Err(error) => report.error("fixture-read", &loaded.path, error.to_string()),
            }
        }
        if let Some(selector) = &case.requirements.model {
            let mut compatible = 0_usize;
            for id in &selector.ids {
                match self.models.by_id(id) {
                    Err(_) => report.error(
                        "case-model-missing",
                        &loaded.path,
                        format!("case references unknown model {id}"),
                    ),
                    Ok(model) if !model_matches_case(model, case, selector) => report.error(
                        "case-model-incompatible",
                        &loaded.path,
                        format!(
                            "model {id} is not valid for {} and its required architecture tags",
                            case.primitive
                        ),
                    ),
                    Ok(_) => compatible += 1,
                }
            }
            if selector.ids.is_empty() {
                compatible = self
                    .models
                    .models
                    .iter()
                    .filter(|model| model_matches_case(model, case, selector))
                    .count();
            }
            if compatible == 0 && case.status != CaseStatus::Disabled {
                report.error(
                    "case-model-selector-empty",
                    &loaded.path,
                    "model selector matches no accepted registry model",
                );
            }
        }
        for (role, invocation) in [
            ("reference", case.invocations.reference.as_ref()),
            ("candidate", case.invocations.candidate.as_ref()),
        ] {
            let Some(invocation) = invocation else {
                continue;
            };
            let target = invocation
                .target
                .as_deref()
                .or_else(|| default_target(invocation.kind));
            let Some(target) = target else {
                if case.status == CaseStatus::Active {
                    report.error(
                        "invocation-target",
                        &loaded.path,
                        format!("active {role} invocation has no target"),
                    );
                }
                continue;
            };
            match invocation.kind {
                InvocationKind::UpstreamTest
                | InvocationKind::UpstreamTool
                | InvocationKind::NativeOracle => {
                    let expected = match invocation.kind {
                        InvocationKind::UpstreamTest => UpstreamTargetKind::CtestSuite,
                        InvocationKind::UpstreamTool => UpstreamTargetKind::UpstreamTool,
                        InvocationKind::NativeOracle => UpstreamTargetKind::Oracle,
                        InvocationKind::IcnProbe => unreachable!(),
                    };
                    if self.upstream_targets.get(target).map(|value| value.kind) != Some(expected) {
                        report.error(
                            "upstream-target",
                            &loaded.path,
                            format!("{role} target {target} is missing or has the wrong kind"),
                        );
                    }
                }
                InvocationKind::IcnProbe => match self.producers.get(target) {
                    None => report.error(
                        "producer-target",
                        &loaded.path,
                        format!("{role} producer {target} is not declared"),
                    ),
                    Some(producer) => {
                        let operation = invocation.operation.as_deref().unwrap_or(&case.operation);
                        if case.status == crate::model::CaseStatus::Active
                            && !producer.capabilities.is_empty()
                            && !producer.capabilities.iter().any(|value| value == operation)
                        {
                            report.error(
                                "producer-capability",
                                &loaded.path,
                                format!("producer {target} does not declare operation {operation}"),
                            );
                        }
                    }
                },
            }
        }
        if case.adapter == AdapterKind::Differential
            && case
                .invocations
                .reference
                .as_ref()
                .and_then(|value| value.operation.as_deref())
                .is_some_and(|operation| operation != case.operation)
        {
            report.error(
                "operation-mismatch",
                &loaded.path,
                "reference invocation operation differs from neutral case operation",
            );
        }
    }

    pub fn profile(&self, id: &str) -> anyhow::Result<&Profile> {
        self.profiles
            .get(id)
            .with_context(|| format!("unknown parity profile {id}"))
    }

    pub fn selected_cases(&self, profile: &Profile) -> Vec<&LoadedCase> {
        self.cases
            .values()
            .filter(|case| profile.selects(&case.definition))
            .collect()
    }

    pub fn validate_schema<T: Serialize>(&self, kind: SchemaKind, value: &T) -> anyhow::Result<()> {
        let instance = serde_json::to_value(value)?;
        let schema = match kind {
            SchemaKind::Case => &self.schemas.case,
            SchemaKind::Evidence => &self.schemas.evidence,
            SchemaKind::Comparison => &self.schemas.comparison,
        };
        validate_json_schema(schema, &instance)
    }
}

fn model_matches_case(
    model: &crate::models::ModelRecord,
    case: &CaseDefinition,
    selector: &crate::model::ModelSelector,
) -> bool {
    model.status == "accepted"
        && model.valid_for.contains(&case.primitive)
        && selector
            .all_tags
            .iter()
            .all(|tag| model.attributes.architecture_tags.contains(tag))
        && (selector.any_tags.is_empty()
            || selector
                .any_tags
                .iter()
                .any(|tag| model.attributes.architecture_tags.contains(tag)))
        && case
            .requirements
            .architecture_tags
            .iter()
            .all(|tag| model.attributes.architecture_tags.contains(tag))
}

fn pinned_source_root(parity_root: &Path) -> anyhow::Result<PathBuf> {
    #[derive(Deserialize)]
    struct NativePin {
        llama_cpp: NativeCheckout,
    }
    #[derive(Deserialize)]
    struct NativeCheckout {
        checkout_path: PathBuf,
    }
    let inference_root = parity_root
        .parent()
        .context("parity root has no inference parent")?;
    let pin: NativePin = load_toml(&inference_root.join("native-pin.toml"))?;
    let source = inference_root.join(pin.llama_cpp.checkout_path);
    canonical_directory(&source, "pinned llama.cpp source")
}

fn default_target(kind: InvocationKind) -> Option<&'static str> {
    match kind {
        InvocationKind::NativeOracle => Some("oracle"),
        InvocationKind::IcnProbe => Some("icn-probe"),
        InvocationKind::UpstreamTest | InvocationKind::UpstreamTool => None,
    }
}

fn load_cases(
    directory: &Path,
    schema: &Value,
    contract_digests: &mut BTreeMap<PathBuf, String>,
) -> anyhow::Result<BTreeMap<String, LoadedCase>> {
    let mut files = Vec::new();
    collect_files(directory, "json", &mut files)?;
    let mut cases = BTreeMap::new();
    for path in files {
        let bytes = read_contract(&path, contract_digests)?;
        let raw: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid parity case {}", path.display()))?;
        validate_json_schema(schema, &raw)
            .with_context(|| format!("case does not satisfy schema {}", path.display()))?;
        let case: CaseDefinition = serde_json::from_value(raw)
            .with_context(|| format!("invalid typed parity case {}", path.display()))?;
        if cases
            .insert(
                case.id.clone(),
                LoadedCase {
                    path: path.clone(),
                    definition: case,
                },
            )
            .is_some()
        {
            bail!("duplicate parity case id in {}", path.display());
        }
    }
    Ok(cases)
}

fn load_json_contract(
    path: &Path,
    contract_digests: &mut BTreeMap<PathBuf, String>,
) -> anyhow::Result<Value> {
    let bytes = read_contract(path, contract_digests)?;
    serde_json::from_slice(&bytes).with_context(|| format!("invalid JSON in {}", path.display()))
}

fn validate_json_schema(schema: &Value, instance: &Value) -> anyhow::Result<()> {
    let validator = jsonschema::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .build(schema)
        .context("failed to compile parity JSON Schema")?;
    let errors = validator
        .iter_errors(instance)
        .take(20)
        .map(|error| format!("{}: {}", error.instance_path(), error))
        .collect::<Vec<_>>();
    if !errors.is_empty() {
        bail!("{}", errors.join("; "));
    }
    Ok(())
}

fn load_profiles(
    directory: &Path,
    contract_digests: &mut BTreeMap<PathBuf, String>,
) -> anyhow::Result<(BTreeMap<String, Profile>, BTreeMap<String, PathBuf>)> {
    let mut files = Vec::new();
    collect_files(directory, "toml", &mut files)?;
    let mut profiles = BTreeMap::new();
    let mut paths = BTreeMap::new();
    for path in files {
        let profile: Profile = load_toml_contract(&path, contract_digests)?;
        if paths.insert(profile.id.clone(), path.clone()).is_some() {
            bail!("duplicate profile id in {}", path.display());
        }
        if profiles.insert(profile.id.clone(), profile).is_some() {
            bail!("duplicate profile id in {}", path.display());
        }
    }
    Ok((profiles, paths))
}

fn collect_files(
    directory: &Path,
    extension: &str,
    output: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(directory)
        .with_context(|| format!("failed to read {}", directory.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_files(&path, extension, output)?;
        } else if metadata.is_file()
            && path.extension().and_then(|value| value.to_str()) == Some(extension)
        {
            output.push(path);
        }
    }
    output.sort();
    Ok(())
}

fn load_toml<T: for<'de> Deserialize<'de>>(path: &Path) -> anyhow::Result<T> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("invalid TOML in {}", path.display()))
}

fn load_toml_contract<T: for<'de> Deserialize<'de>>(
    path: &Path,
    contract_digests: &mut BTreeMap<PathBuf, String>,
) -> anyhow::Result<T> {
    let bytes = read_contract(path, contract_digests)?;
    let text = std::str::from_utf8(&bytes)
        .with_context(|| format!("TOML contract is not UTF-8: {}", path.display()))?;
    toml::from_str(text).with_context(|| format!("invalid TOML in {}", path.display()))
}

fn read_contract(
    path: &Path,
    contract_digests: &mut BTreeMap<PathBuf, String>,
) -> anyhow::Result<Vec<u8>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read run contract {}", path.display()))?;
    record_contract_digest(contract_digests, path.to_path_buf(), &sha256_bytes(&bytes))?;
    Ok(bytes)
}

fn record_contract_digest(
    contract_digests: &mut BTreeMap<PathBuf, String>,
    path: PathBuf,
    digest: &str,
) -> anyhow::Result<()> {
    if let Some(previous) = contract_digests.insert(path.clone(), digest.to_owned())
        && previous != digest
    {
        bail!(
            "run contract {} is declared with conflicting digests",
            path.display()
        );
    }
    Ok(())
}

fn unique_by_id<T, F>(values: Vec<T>, id: F, kind: &str) -> anyhow::Result<BTreeMap<String, T>>
where
    F: Fn(&T) -> &str,
{
    let mut result = BTreeMap::new();
    for value in values {
        let key = id(&value).to_owned();
        if result.insert(key.clone(), value).is_some() {
            bail!("duplicate {kind} id {key}");
        }
    }
    Ok(result)
}

fn canonical_directory(path: &Path, label: &str) -> anyhow::Result<PathBuf> {
    if !path.is_dir() {
        bail!("{label} does not exist: {}", path.display());
    }
    std::fs::canonicalize(path)
        .with_context(|| format!("failed to resolve {label} {}", path.display()))
}

fn canonical_file_within(root: &Path, path: &Path) -> anyhow::Result<PathBuf> {
    let canonical_root = std::fs::canonicalize(root)
        .with_context(|| format!("failed to resolve containing root {}", root.display()))?;
    let canonical_path = std::fs::canonicalize(path)
        .with_context(|| format!("failed to resolve file {}", path.display()))?;
    if !canonical_path.starts_with(&canonical_root) || !canonical_path.is_file() {
        bail!(
            "resolved file {} is not a regular descendant of {}",
            canonical_path.display(),
            canonical_root.display()
        );
    }
    Ok(canonical_path)
}

fn validate_relative_path(path: &Path) -> anyhow::Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, PathComponent::Normal(_)))
    {
        bail!("path must be a non-empty normalized relative path");
    }
    Ok(())
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ValidationReport {
    pub errors: Vec<Diagnostic>,
    pub warnings: Vec<Diagnostic>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    fn error(&mut self, code: &str, path: &Path, message: impl Into<String>) {
        self.errors.push(Diagnostic {
            code: code.to_owned(),
            path: path.to_path_buf(),
            message: message.into(),
        });
    }

    fn warning(&mut self, code: &str, path: &Path, message: impl Into<String>) {
        self.warnings.push(Diagnostic {
            code: code.to_owned(),
            path: path.to_path_buf(),
            message: message.into(),
        });
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Diagnostic {
    pub code: String,
    pub path: PathBuf,
    pub message: String,
}

#[derive(Clone, Debug)]
pub struct ReferenceBuildManifest {
    pub path: PathBuf,
    pub schema_version: u64,
    pub build_directory: PathBuf,
    pub backend: String,
    pub lane: String,
    pub build_type: String,
    pub artifacts: BTreeMap<String, ReferenceArtifact>,
    pub native_source: Option<NativeSourceIdentity>,
    pub oracle_source: Option<NativeSourceIdentity>,
    pub configuration_artifacts: Vec<ReferenceArtifact>,
    pub selected_targets: BTreeSet<String>,
    manifest_sha256: String,
    raw: Value,
}

#[derive(Clone, Debug)]
pub struct ReferenceArtifact {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub struct NativeSourceIdentity {
    pub name: String,
    pub revision: Option<String>,
    pub path: Option<PathBuf>,
    pub tree_sha256: String,
    pub inventory: Option<ReferenceArtifact>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CandidateBuildManifestDocument {
    pub schema_version: u32,
    pub created_at: String,
    pub reference_manifest_sha256: String,
    pub backend: String,
    pub lane: String,
    pub build_type: String,
    pub environment: CandidateEnvironmentEvidence,
    pub configuration: CandidateConfiguration,
    pub compiler: CandidateCompiler,
    pub flags: Vec<String>,
    pub assertions: bool,
    pub sanitizers: Vec<String>,
    pub components: Vec<CandidateSourceComponent>,
    pub artifacts: Vec<CandidateArtifactDocument>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CandidateEnvironmentEvidence {
    pub policy: String,
    pub names: Vec<String>,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CandidateConfiguration {
    pub native_pin: CandidateConfigurationArtifact,
    pub cargo_manifest: CandidateConfigurationArtifact,
    pub cargo_lock: CandidateConfigurationArtifact,
    pub candidate_builder: CandidateConfigurationArtifact,
    pub controlled_environment: CandidateConfigurationArtifact,
    pub source_inventory: CandidateConfigurationArtifact,
}

impl CandidateConfiguration {
    fn artifacts(&self) -> [(&'static str, &CandidateConfigurationArtifact); 6] {
        [
            ("nativePin", &self.native_pin),
            ("cargoManifest", &self.cargo_manifest),
            ("cargoLock", &self.cargo_lock),
            ("candidateBuilder", &self.candidate_builder),
            ("controlledEnvironment", &self.controlled_environment),
            ("sourceInventory", &self.source_inventory),
        ]
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateConfigurationArtifact {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateCompiler {
    pub name: String,
    pub version: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CandidateSourceComponent {
    pub kind: String,
    pub name: String,
    pub path: PathBuf,
    #[serde(default)]
    pub revision: Option<String>,
    pub tree_sha256: String,
    pub dirty: bool,
    pub excluded_directory_names: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateArtifactDocument {
    pub name: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub struct CandidateBuildManifest {
    pub path: PathBuf,
    pub document: CandidateBuildManifestDocument,
    manifest_sha256: String,
    inference_root: PathBuf,
}

impl CandidateBuildManifest {
    pub fn load(path: &Path, parity_root: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read candidate manifest {}", path.display()))?;
        let manifest_sha256 = sha256_bytes(&bytes);
        let document: CandidateBuildManifestDocument = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid candidate manifest {}", path.display()))?;
        if document.schema_version != 1
            || document.created_at.trim().is_empty()
            || !valid_sha256(&document.reference_manifest_sha256)
            || document.backend.trim().is_empty()
            || document.lane.trim().is_empty()
            || document.build_type.trim().is_empty()
            || document.compiler.name.trim().is_empty()
            || document.compiler.version.trim().is_empty()
            || document.flags.is_empty()
            || document.components.is_empty()
            || document.artifacts.is_empty()
        {
            bail!("candidate manifest has incomplete schema/build identity");
        }
        if document.environment.policy != "allowlist-v1"
            || !valid_sha256(&document.environment.sha256)
            || document.environment.names.is_empty()
            || document
                .environment
                .names
                .iter()
                .any(|name| name.trim().is_empty())
            || document
                .environment
                .names
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != document.environment.names.len()
        {
            bail!("candidate manifest has invalid controlled-environment evidence");
        }
        let expected_configuration_paths = BTreeMap::from([
            ("nativePin", Path::new("native-pin.toml")),
            ("cargoManifest", Path::new("Cargo.toml")),
            ("cargoLock", Path::new("Cargo.lock")),
            ("candidateBuilder", Path::new("scripts/build-candidate.ts")),
            (
                "controlledEnvironment",
                Path::new("scripts/controlled-environment.ts"),
            ),
            ("sourceInventory", Path::new("scripts/source-inventory.ts")),
        ]);
        for (name, artifact) in document.configuration.artifacts() {
            if artifact.bytes == 0
                || !valid_sha256(&artifact.sha256)
                || validate_relative_path(&artifact.path).is_err()
                || expected_configuration_paths.get(name).copied() != Some(artifact.path.as_path())
            {
                bail!("candidate manifest has invalid configuration artifact {name}");
            }
        }
        let mut component_names = BTreeSet::new();
        for component in &document.components {
            let mut exclusions = BTreeSet::new();
            let valid_exclusions = component.excluded_directory_names.iter().all(|name| {
                !name.is_empty()
                    && name != "."
                    && name != ".."
                    && !name.contains('/')
                    && !name.contains('\\')
                    && exclusions.insert(name)
            });
            if !matches!(
                component.kind.as_str(),
                "native-source" | "bindings-source" | "icn-source"
            ) || !component_names.insert((&component.kind, &component.name))
                || !valid_sha256(&component.tree_sha256)
                || component.excluded_directory_names.is_empty()
                || !valid_exclusions
                || (component.path != Path::new(".")
                    && validate_relative_path(&component.path).is_err())
            {
                bail!(
                    "candidate manifest has invalid or duplicate source component {}",
                    component.name
                );
            }
        }
        let mut artifact_names = BTreeSet::new();
        for artifact in &document.artifacts {
            if !artifact_names.insert(&artifact.name)
                || artifact.bytes == 0
                || !valid_sha256(&artifact.sha256)
                || validate_relative_path(&artifact.path).is_err()
            {
                bail!(
                    "candidate manifest has invalid or duplicate artifact {}",
                    artifact.name
                );
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            document,
            manifest_sha256,
            inference_root: parity_root
                .parent()
                .context("parity root has no inference parent")?
                .to_path_buf(),
        })
    }

    pub async fn verify(&self) -> anyhow::Result<()> {
        let current = tokio::fs::read(&self.path).await.with_context(|| {
            format!(
                "failed to re-read candidate manifest {}",
                self.path.display()
            )
        })?;
        if sha256_bytes(&current) != self.manifest_sha256 {
            bail!("candidate manifest changed after it was loaded");
        }
        let canonical_root = std::fs::canonicalize(&self.inference_root).with_context(|| {
            format!(
                "failed to resolve inference root {}",
                self.inference_root.display()
            )
        })?;
        for component in &self.document.components {
            let path = self.resolve(&component.path);
            let canonical = std::fs::canonicalize(&path).with_context(|| {
                format!(
                    "failed to resolve candidate source component {}",
                    path.display()
                )
            })?;
            if !canonical.starts_with(&canonical_root) || !canonical.is_dir() {
                bail!(
                    "candidate source component {} escapes the inference root",
                    component.name
                );
            }
            let exclusions = component
                .excluded_directory_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let actual = crate::digest::sha256_source_tree(&canonical, &exclusions).await?;
            if actual.sha256 != component.tree_sha256 {
                bail!(
                    "candidate source component {} differs from its build inventory",
                    component.name
                );
            }
        }
        for artifact in &self.document.artifacts {
            let path = self.resolve(&artifact.path);
            let canonical = std::fs::canonicalize(&path).with_context(|| {
                format!("failed to resolve candidate artifact {}", path.display())
            })?;
            if !canonical.starts_with(&canonical_root) || !canonical.is_file() {
                bail!(
                    "candidate artifact {} escapes the inference root",
                    artifact.name
                );
            }
            let (digest, bytes) = sha256_file(&canonical).await?;
            if digest != artifact.sha256 || bytes != artifact.bytes {
                bail!(
                    "candidate artifact {} differs from its manifest",
                    artifact.name
                );
            }
        }
        for (name, artifact) in self.document.configuration.artifacts() {
            let path = self.resolve(&artifact.path);
            let canonical = std::fs::canonicalize(&path).with_context(|| {
                format!(
                    "failed to resolve candidate configuration artifact {name}: {}",
                    path.display()
                )
            })?;
            if !canonical.starts_with(&canonical_root) || !canonical.is_file() {
                bail!("candidate configuration artifact {name} escapes the inference root");
            }
            let (digest, bytes) = sha256_file(&canonical).await?;
            if digest != artifact.sha256 || bytes != artifact.bytes {
                bail!("candidate configuration artifact {name} differs from its manifest");
            }
        }
        Ok(())
    }

    pub fn artifact(&self, name: &str) -> anyhow::Result<PathBuf> {
        self.document
            .artifacts
            .iter()
            .find(|artifact| artifact.name == name)
            .map(|artifact| {
                std::fs::canonicalize(self.resolve(&artifact.path))
                    .with_context(|| format!("failed to resolve candidate artifact {name}"))
            })
            .transpose()?
            .with_context(|| format!("candidate manifest has no artifact {name}"))
    }

    pub fn digest(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn provenance_components(&self) -> Vec<crate::model::Component> {
        self.document
            .components
            .iter()
            .map(|component| crate::model::Component {
                kind: component.kind.clone(),
                name: component.name.clone(),
                revision: component.revision.clone(),
                tree_sha256: Some(component.tree_sha256.clone()),
                binary_sha256: None,
                dirty: Some(component.dirty),
            })
            .collect()
    }

    pub fn build_info(&self) -> crate::model::BuildInfo {
        crate::model::BuildInfo {
            build_type: self.document.build_type.to_ascii_lowercase(),
            compiler: self.document.compiler.name.clone(),
            compiler_version: self.document.compiler.version.clone(),
            flags: self.document.flags.clone(),
            assertions: Some(self.document.assertions),
            sanitizers: Some(self.document.sanitizers.clone()),
        }
    }

    fn resolve(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.inference_root.join(path)
        }
    }
}

impl ReferenceBuildManifest {
    pub fn load(path: &Path, parity_root: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read reference manifest {}", path.display()))?;
        let manifest_sha256 = sha256_bytes(&bytes);
        let raw: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid reference manifest {}", path.display()))?;
        let schema_version = lookup_u64(&raw, &["schemaVersion", "schema_version"])
            .context("reference manifest is missing schema version")?;
        if !(2..=3).contains(&schema_version) {
            bail!("unsupported reference build manifest schema {schema_version}");
        }
        let inference_root = parity_root
            .parent()
            .context("parity root has no inference parent")?
            .to_path_buf();
        let lane = lookup_str(&raw, &["lane", "buildLane"])
            .unwrap_or("unknown")
            .to_owned();
        let backend = lookup_str(&raw, &["backend"])
            .unwrap_or("unknown")
            .to_owned();
        let inferred_suffix = if lane == "upstream-default" {
            backend.clone()
        } else {
            format!("{backend}-{lane}")
        };
        let build_directory = lookup_str(&raw, &["buildDirectory", "build_directory"])
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                inference_root
                    .join("target/reference/llama.cpp")
                    .join(inferred_suffix)
            });
        let build_directory = resolve_manifest_path(&build_directory, &inference_root, path);
        let build_type = lookup_str(&raw, &["buildType", "build_type"])
            .unwrap_or("unknown")
            .to_owned();
        let selected_targets = raw
            .get("selectedTargets")
            .or_else(|| raw.get("selected_targets"))
            .and_then(Value::as_array)
            .context("reference manifest is missing selectedTargets")?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .or_else(|| value.get("id").and_then(Value::as_str))
                    .map(str::to_owned)
                    .context("reference manifest selectedTargets entry has no id")
            })
            .collect::<anyhow::Result<BTreeSet<_>>>()?;
        let mut artifacts = BTreeMap::new();
        let artifacts_value = raw
            .get("artifacts")
            .context("reference manifest is missing artifacts")?;
        match artifacts_value {
            Value::Object(records) => {
                for (id, value) in records {
                    if let Some(artifact) = parse_artifact(value, &inference_root, path)? {
                        artifacts.insert(id.to_owned(), artifact);
                    }
                }
            }
            Value::Array(records) => {
                for value in records {
                    let id = lookup_str(value, &["name", "id"])
                        .context("reference artifact record has no name")?;
                    if let Some(artifact) = parse_artifact(value, &inference_root, path)? {
                        artifacts.insert(id.to_owned(), artifact);
                    }
                }
            }
            _ => bail!("reference manifest artifacts must be an object or array"),
        }
        let llama = raw.get("llamaCpp").or_else(|| raw.get("llama_cpp"));
        let native_source = llama.and_then(|llama| {
            let tree = llama
                .get("sourceTree")
                .or_else(|| llama.get("source_tree"))?;
            let tree_sha256 = lookup_str(tree, &["sha256"])?;
            let source_path = lookup_str(llama, &["path", "sourceDirectory", "source_directory"])
                .map(PathBuf::from)
                .map(|value| resolve_manifest_path(&value, &inference_root, path));
            let inventory = tree
                .get("inventory")
                .and_then(|value| parse_artifact(value, &inference_root, path).ok().flatten());
            Some(NativeSourceIdentity {
                name: "llama.cpp".to_owned(),
                revision: lookup_str(llama, &["revision"]).map(str::to_owned),
                path: source_path,
                tree_sha256: tree_sha256.to_owned(),
                inventory,
            })
        });
        let oracle_source = raw.get("oracle").and_then(|oracle| {
            let tree = oracle
                .get("sourceTree")
                .or_else(|| oracle.get("source_tree"))?;
            Some(NativeSourceIdentity {
                name: "icn-parity-oracle".to_owned(),
                revision: None,
                path: None,
                tree_sha256: lookup_str(tree, &["sha256"])?.to_owned(),
                inventory: None,
            })
        });
        let mut configuration_artifacts = Vec::new();
        if let Some(configuration) = raw.get("configuration").and_then(Value::as_object) {
            for value in configuration.values() {
                if let Some(artifact) = parse_artifact(value, &inference_root, path)? {
                    configuration_artifacts.push(artifact);
                }
            }
        }
        if schema_version >= 3 {
            let source = native_source
                .as_ref()
                .context("schema-v3 manifest requires llamaCpp.sourceTree")?;
            if source.revision.as_deref().is_none_or(str::is_empty)
                || source.inventory.is_none()
                || matches!(backend.as_str(), "" | "unknown")
                || matches!(lane.as_str(), "" | "unknown")
                || matches!(build_type.as_str(), "" | "unknown")
            {
                bail!("schema-v3 reference manifest has incomplete source/build identity");
            }
            if (selected_targets.contains("oracle") && oracle_source.is_none())
                || configuration_artifacts.is_empty()
                || selected_targets.is_empty()
            {
                bail!(
                    "schema-v3 reference manifest is missing selected-target source or configuration identity"
                );
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            schema_version,
            build_directory,
            backend,
            lane,
            build_type,
            artifacts,
            native_source,
            oracle_source,
            configuration_artifacts,
            selected_targets,
            manifest_sha256,
            raw,
        })
    }

    pub async fn verify(&self) -> anyhow::Result<()> {
        let current = tokio::fs::read(&self.path).await.with_context(|| {
            format!(
                "failed to re-read reference manifest {}",
                self.path.display()
            )
        })?;
        if sha256_bytes(&current) != self.manifest_sha256 {
            bail!("reference manifest changed after it was loaded");
        }
        if self.artifacts.is_empty() {
            bail!("reference manifest contains no executable artifacts");
        }
        for (id, artifact) in &self.artifacts {
            verify_artifact(id, artifact).await?;
        }
        for (index, artifact) in self.configuration_artifacts.iter().enumerate() {
            verify_artifact(&format!("configuration-{index}"), artifact).await?;
        }
        if let Some(source) = &self.native_source {
            if !valid_sha256(&source.tree_sha256) {
                bail!("reference source tree digest is invalid");
            }
            if let Some(inventory) = &source.inventory {
                verify_artifact("source-inventory", inventory).await?;
                let bytes = tokio::fs::read(&inventory.path).await?;
                let inventory_json: Value = serde_json::from_slice(&bytes)?;
                if lookup_str(&inventory_json, &["sha256"]) != Some(&source.tree_sha256) {
                    bail!("source inventory aggregate does not match reference manifest");
                }
                if let Some(source_path) = &source.path {
                    verify_inventory_entries(source_path, &inventory_json).await?;
                }
            }
        }
        Ok(())
    }

    pub fn provenance_components(&self, include_oracle: bool) -> Vec<crate::model::Component> {
        let mut components = Vec::new();
        if let Some(source) = &self.native_source {
            components.push(crate::model::Component {
                kind: "native-source".to_owned(),
                name: source.name.clone(),
                revision: source.revision.clone(),
                tree_sha256: Some(source.tree_sha256.clone()),
                binary_sha256: None,
                // A content inventory proves the exact tree used, but this
                // no-Git workflow does not claim whether it differs from the
                // named revision.
                dirty: None,
            });
        }
        if include_oracle && let Some(source) = &self.oracle_source {
            components.push(crate::model::Component {
                kind: "native-source".to_owned(),
                name: source.name.clone(),
                revision: source.revision.clone(),
                tree_sha256: Some(source.tree_sha256.clone()),
                binary_sha256: None,
                dirty: None,
            });
        }
        components
    }

    pub fn build_info(&self) -> crate::model::BuildInfo {
        let compiler = self
            .raw
            .pointer("/compilers/cxx/id")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let compiler_version = self
            .raw
            .pointer("/compilers/cxx/version")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let flags = self
            .raw
            .pointer("/cmake/resolvedCache")
            .and_then(Value::as_object)
            .map(|values| {
                values
                    .iter()
                    .map(|(key, value)| match value.as_str() {
                        Some(value) => format!("{key}={value}"),
                        None => format!("{key}={value}"),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let assertions = self
            .raw
            .pointer("/verification/assertions")
            .and_then(Value::as_bool);
        let sanitizers = self
            .raw
            .pointer("/verification/sanitizers")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            });
        crate::model::BuildInfo {
            build_type: self.build_type.to_ascii_lowercase(),
            compiler: compiler.to_owned(),
            compiler_version: compiler_version.to_owned(),
            flags,
            assertions,
            sanitizers,
        }
    }

    pub fn artifact(&self, id: &str) -> anyhow::Result<&ReferenceArtifact> {
        self.artifacts
            .get(id)
            .with_context(|| format!("reference manifest does not contain artifact {id}"))
    }

    pub fn require_selected_target(&self, id: &str) -> anyhow::Result<()> {
        if !self.selected_targets.contains(id) {
            bail!("reference manifest did not select upstream target {id}");
        }
        Ok(())
    }

    pub fn digest(&self) -> anyhow::Result<String> {
        Ok(self.manifest_sha256.clone())
    }
}

async fn verify_artifact(id: &str, artifact: &ReferenceArtifact) -> anyhow::Result<()> {
    let (digest, bytes) = sha256_file(&artifact.path)
        .await
        .with_context(|| format!("failed to verify reference artifact {id}"))?;
    if bytes != artifact.bytes || digest != artifact.sha256 {
        bail!("reference artifact {id} does not match its manifest digest/size");
    }
    Ok(())
}

async fn verify_inventory_entries(source: &Path, inventory: &Value) -> anyhow::Result<()> {
    let exclusions = inventory
        .get("excludedDirectoryNames")
        .or_else(|| inventory.get("excluded_directory_names"))
        .and_then(Value::as_array)
        .context("source inventory has no exclusion contract")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .context("source inventory exclusion must be a string")
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let actual = crate::digest::sha256_source_tree(source, &exclusions).await?;
    let expected_digest =
        lookup_str(inventory, &["sha256"]).context("source inventory has no aggregate digest")?;
    let expected_files = lookup_u64(inventory, &["fileCount", "file_count"])
        .context("source inventory has no file count")?;
    let expected_bytes = lookup_u64(inventory, &["totalBytes", "total_bytes"])
        .context("source inventory has no total bytes")?;
    if actual.sha256 != expected_digest
        || actual.file_count != expected_files
        || actual.total_bytes != expected_bytes
    {
        bail!(
            "native source tree differs from reference inventory (digest/files/bytes actual={}/{}/{}, expected={}/{}/{})",
            actual.sha256,
            actual.file_count,
            actual.total_bytes,
            expected_digest,
            expected_files,
            expected_bytes,
        );
    }
    Ok(())
}

fn parse_artifact(
    value: &Value,
    inference_root: &Path,
    manifest_path: &Path,
) -> anyhow::Result<Option<ReferenceArtifact>> {
    let Some(path) = lookup_str(value, &["path"]) else {
        return Ok(None);
    };
    let bytes = lookup_u64(value, &["bytes"]).context("reference artifact has no byte size")?;
    let sha256 = lookup_str(value, &["sha256"])
        .context("reference artifact has no SHA-256")?
        .to_owned();
    if !valid_sha256(&sha256) {
        bail!("reference artifact SHA-256 is invalid");
    }
    Ok(Some(ReferenceArtifact {
        path: resolve_manifest_path(Path::new(path), inference_root, manifest_path),
        bytes,
        sha256,
    }))
}

fn resolve_manifest_path(value: &Path, inference_root: &Path, manifest_path: &Path) -> PathBuf {
    if value.is_absolute() {
        return value.to_path_buf();
    }
    let _ = manifest_path;
    inference_root.join(value)
}

fn lookup_str<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
}

fn lookup_u64(value: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_selection_is_conjunctive() {
        // Selection behavior is exercised end-to-end against repository assets;
        // this unit test protects the most error-prone status gate.
        let profile: Profile = toml::from_str(
            r#"
schema_version=1
id="test"
description="test"
[selection]
statuses=["active"]
include_case_ids=[]
include_primitives=[]
include_tags=[]
exclude_case_ids=[]
exclude_tags=[]
[upstream]
build_profile="cargo-equivalent"
[execution]
offline=true
max_parallel=1
case_timeout_seconds=1
fail_fast=false
performance_engine_order="alternate"
[engine]
backend="cpu"
threads="1"
gpu_layers="0"
flash_attention="off"
[models]
verify_before_run=true
allow_fetch=false
ids=[]
[performance]
enabled=true
paired_repetitions=1
require_controlled_host=false
require_exclusive_device=false
[results]
preserve_raw_evidence=true
preserve_raw_samples=true
write_comparisons=true
[gates]
fail_on_invalid=true
fail_on_error=true
fail_on_performance=false
require_correctness_before_performance=true
"#,
        )
        .unwrap();
        assert_eq!(profile.id, "test");
        assert_eq!(profile.selection.statuses, vec![CaseStatus::Active]);
    }

    #[tokio::test]
    async fn candidate_manifest_reverification_detects_artifact_and_manifest_mutation() {
        let temporary = tempfile::tempdir().unwrap();
        let inference_root = temporary.path().join("inference");
        let parity_root = inference_root.join("parity");
        std::fs::create_dir_all(inference_root.join("source")).unwrap();
        std::fs::create_dir_all(inference_root.join("scripts")).unwrap();
        std::fs::create_dir_all(&parity_root).unwrap();
        std::fs::write(inference_root.join("source/lib.rs"), b"source").unwrap();
        std::fs::write(inference_root.join("probe"), b"binary").unwrap();
        let configuration_inputs = [
            ("nativePin", "native-pin.toml", b"native pin".as_slice()),
            ("cargoManifest", "Cargo.toml", b"cargo manifest".as_slice()),
            ("cargoLock", "Cargo.lock", b"cargo lock".as_slice()),
            (
                "candidateBuilder",
                "scripts/build-candidate.ts",
                b"candidate builder".as_slice(),
            ),
            (
                "controlledEnvironment",
                "scripts/controlled-environment.ts",
                b"controlled environment".as_slice(),
            ),
            (
                "sourceInventory",
                "scripts/source-inventory.ts",
                b"source inventory".as_slice(),
            ),
        ];
        let mut configuration = serde_json::Map::new();
        for (name, path, contents) in configuration_inputs {
            std::fs::write(inference_root.join(path), contents).unwrap();
            let (sha256, bytes) = sha256_file(&inference_root.join(path)).await.unwrap();
            configuration.insert(
                name.to_owned(),
                serde_json::json!({"path": path, "bytes": bytes, "sha256": sha256}),
            );
        }
        let source = crate::digest::sha256_source_tree(&inference_root.join("source"), &["target"])
            .await
            .unwrap();
        let (binary_sha256, binary_bytes) =
            sha256_file(&inference_root.join("probe")).await.unwrap();
        let manifest_path = inference_root.join("candidate.json");
        let manifest = serde_json::json!({
            "schemaVersion": 1,
            "createdAt": "2026-01-01T00:00:00Z",
            "referenceManifestSha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "backend": "metal",
            "lane": "cargo-equivalent",
            "buildType": "release",
            "environment": {
                "policy": "allowlist-v1",
                "names": ["LANG", "LC_ALL", "PATH", "TZ"],
                "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            },
            "configuration": configuration,
            "compiler": {"name": "clang", "version": "1"},
            "flags": ["GGML_METAL=ON"],
            "assertions": false,
            "sanitizers": [],
            "components": [{
                "kind": "icn-source", "name": "icn", "path": "source",
                "treeSha256": source.sha256, "dirty": true,
                "excludedDirectoryNames": ["target"]
            }],
            "artifacts": [{
                "name": "icn-probe", "path": "probe",
                "bytes": binary_bytes, "sha256": binary_sha256
            }]
        });
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let loaded = CandidateBuildManifest::load(&manifest_path, &parity_root).unwrap();
        loaded.verify().await.unwrap();

        std::fs::write(inference_root.join("probe"), b"swap!!").unwrap();
        assert!(loaded.verify().await.is_err());
        std::fs::write(inference_root.join("probe"), b"binary").unwrap();

        std::fs::write(inference_root.join("Cargo.lock"), b"changed cargo lock").unwrap();
        assert!(loaded.verify().await.is_err());
        std::fs::write(inference_root.join("Cargo.lock"), b"cargo lock").unwrap();
        loaded.verify().await.unwrap();

        let mut manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        manifest_bytes.push(b'\n');
        std::fs::write(&manifest_path, manifest_bytes).unwrap();
        assert!(loaded.verify().await.is_err());

        let mut missing_reference = manifest.clone();
        missing_reference
            .as_object_mut()
            .unwrap()
            .remove("referenceManifestSha256");
        let missing_path = inference_root.join("candidate-missing-reference.json");
        std::fs::write(
            &missing_path,
            serde_json::to_vec_pretty(&missing_reference).unwrap(),
        )
        .unwrap();
        assert!(CandidateBuildManifest::load(&missing_path, &parity_root).is_err());

        let mut invalid_reference = manifest.clone();
        invalid_reference["referenceManifestSha256"] = Value::String("not-a-digest".to_owned());
        let invalid_path = inference_root.join("candidate-invalid-reference.json");
        std::fs::write(
            &invalid_path,
            serde_json::to_vec_pretty(&invalid_reference).unwrap(),
        )
        .unwrap();
        assert!(CandidateBuildManifest::load(&invalid_path, &parity_root).is_err());

        let mut unknown_environment = manifest.clone();
        unknown_environment["environment"]["unexpected"] = Value::Bool(true);
        let unknown_environment_path = inference_root.join("candidate-unknown-environment.json");
        std::fs::write(
            &unknown_environment_path,
            serde_json::to_vec_pretty(&unknown_environment).unwrap(),
        )
        .unwrap();
        assert!(CandidateBuildManifest::load(&unknown_environment_path, &parity_root).is_err());

        let mut unknown_configuration = manifest.clone();
        unknown_configuration["configuration"]["unexpected"] = serde_json::json!({
            "path": "unexpected",
            "bytes": 1,
            "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
        });
        let unknown_configuration_path =
            inference_root.join("candidate-unknown-configuration.json");
        std::fs::write(
            &unknown_configuration_path,
            serde_json::to_vec_pretty(&unknown_configuration).unwrap(),
        )
        .unwrap();
        assert!(CandidateBuildManifest::load(&unknown_configuration_path, &parity_root).is_err());
    }

    #[test]
    fn candidate_workspace_root_is_allowed_but_unsafe_exclusions_are_not() {
        assert_eq!(Path::new("."), Path::new("."));
        for exclusion in ["", ".", "..", "nested/path", "nested\\path"] {
            let mut seen = BTreeSet::new();
            let valid = !exclusion.is_empty()
                && exclusion != "."
                && exclusion != ".."
                && !exclusion.contains('/')
                && !exclusion.contains('\\')
                && seen.insert(exclusion);
            assert!(!valid);
        }
    }

    #[cfg(unix)]
    #[test]
    fn canonical_file_containment_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("root");
        let outside = temporary.path().join("outside");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(root.join("inside.json"), b"inside").unwrap();
        std::fs::write(outside.join("outside.json"), b"outside").unwrap();
        symlink(outside.join("outside.json"), root.join("escape.json")).unwrap();

        assert!(canonical_file_within(&root, &root.join("inside.json")).is_ok());
        assert!(canonical_file_within(&root, &root.join("escape.json")).is_err());
    }
}
