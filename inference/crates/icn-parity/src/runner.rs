use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail};
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::assets::{
    AssetRepository, CandidateBuildManifest, PerformanceOrder, ProducerDefinition, Profile,
    ReferenceBuildManifest, SchemaKind,
};
use crate::compare::{ComparisonContext, compare_evidence};
use crate::decode::{decode_process_evidence, encode_case_stdin};
use crate::digest::{sha256_bytes, sha256_file};
use crate::model::{
    ArtifactInfo, CaseDefinition, Category, CommandSpec, Comparator, ComparisonStatus, Component,
    DecoderKind, EngineOrder, EvidenceRecord, Invocation, InvocationKind, OutcomeClass,
    ProducerRole, StdinKind,
};
use crate::models::{ModelFileState, all_verified};
use crate::process::{ProcessLimits, ProcessOutput, run_bounded};
use crate::protocol::{
    DESCRIBE_OPERATION, ProtocolDescription, describe_request_jsonl, validate_protocol_description,
};
use crate::provenance::{new_run_id, now_rfc3339, resolve_program};
use crate::store::{RunStore, StoredFile, case_file_stem};
use crate::tools::{ResolvedEngine, ctest_inventory_command, upstream_command};

const CTEST_SOURCE_RESIDUE_FILES: [&str; 2] =
    ["test-grammar-output.tmp", "test-json-schema-input.tmp"];

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub profile_id: String,
    pub output_root: Option<PathBuf>,
    pub run_id: Option<String>,
    pub reference_manifest: Option<PathBuf>,
    pub candidate_manifest: Option<PathBuf>,
    pub model_root: Option<PathBuf>,
    pub model_ids: Vec<String>,
    pub settings: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RunStatus {
    Pass,
    Fail,
    Blocked,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum CaseExecutionStatus {
    Pass,
    Fail,
    Invalid,
    Skipped,
    Blocked,
    Error,
}

#[derive(Clone, Debug, Serialize)]
pub struct CaseExecutionResult {
    pub case_id: String,
    pub category: Category,
    pub model_id: Option<String>,
    pub status: CaseExecutionStatus,
    pub reason: Option<String>,
    pub reference_evidence: Option<String>,
    pub candidate_evidence: Option<String>,
    pub comparison: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunCounts {
    pub pass: usize,
    pub fail: usize,
    pub invalid: usize,
    pub skipped: usize,
    pub blocked: usize,
    pub error: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunSummary {
    pub schema_version: String,
    pub run_id: String,
    pub profile: String,
    pub started_at: String,
    pub completed_at: String,
    pub status: RunStatus,
    pub output_directory: PathBuf,
    pub cases: Vec<CaseExecutionResult>,
    pub counts: RunCounts,
}

impl RunSummary {
    pub fn gate_failed(&self) -> bool {
        self.status != RunStatus::Pass
    }
}

#[derive(Clone, Debug)]
struct CaseInstance {
    case: CaseDefinition,
    model_id: Option<String>,
}

struct RunContext<'a> {
    repository: &'a AssetRepository,
    profile: &'a Profile,
    options: &'a RunOptions,
    store: &'a RunStore,
    run_id: &'a str,
    reference: Option<&'a ReferenceBuildManifest>,
    candidate: Option<&'a CandidateBuildManifest>,
    model_root: PathBuf,
    engine: Option<ResolvedEngine>,
}

pub async fn run_profile(
    repository: &AssetRepository,
    options: RunOptions,
) -> anyhow::Result<RunSummary> {
    let validation = repository.validate().await;
    if !validation.is_valid() {
        let details = validation
            .errors
            .iter()
            .take(20)
            .map(|diagnostic| {
                format!(
                    "{} [{}]: {}",
                    diagnostic.path.display(),
                    diagnostic.code,
                    diagnostic.message
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        bail!("parity assets are invalid:\n{details}");
    }
    let profile = repository.profile(&options.profile_id)?;
    let started_at = now_rfc3339();
    let run_id = options.run_id.clone().unwrap_or_else(new_run_id);
    let output_root = options.output_root.clone().unwrap_or_else(|| {
        repository
            .root
            .parent()
            .unwrap_or(&repository.root)
            .join("results/parity")
    });

    let reference_path = options.reference_manifest.clone().or_else(|| {
        options
            .settings
            .get("reference_manifest")
            .map(PathBuf::from)
    });
    let reference = match reference_path {
        Some(path) => {
            let manifest = ReferenceBuildManifest::load(&path, &repository.root)?;
            require_execution_reference_schema(manifest.schema_version)?;
            if let Some(source_root) = manifest
                .native_source
                .as_ref()
                .and_then(|source| source.path.as_deref())
            {
                cleanup_ctest_source_residue(source_root).await.context(
                    "failed to clean CTest source residue before reference verification",
                )?;
            }
            manifest.verify().await?;
            if manifest.lane != profile.upstream.build_profile {
                bail!(
                    "reference manifest lane {} does not match profile lane {}",
                    manifest.lane,
                    profile.upstream.build_profile
                );
            }
            Some(manifest)
        }
        None => None,
    };
    let candidate_path = options.candidate_manifest.clone().or_else(|| {
        options
            .settings
            .get("candidate_manifest")
            .map(PathBuf::from)
    });
    let candidate = match candidate_path {
        Some(path) => {
            let manifest = CandidateBuildManifest::load(&path, &repository.root)?;
            manifest.verify().await?;
            Some(manifest)
        }
        None => None,
    };
    if let (Some(reference), Some(candidate)) = (&reference, &candidate) {
        validate_candidate_reference_identity(candidate, reference)?;
    }
    let mut effective_settings = options.settings.clone();
    if !effective_settings.contains_key("backend")
        && let Some(reference) = &reference
    {
        effective_settings.insert("backend".to_owned(), reference.backend.clone());
    }
    let needs_engine = repository
        .selected_cases(profile)
        .iter()
        .any(|loaded| case_consumes_engine(repository, &loaded.definition));
    let engine = if needs_engine {
        Some(ResolvedEngine::resolve(profile, &effective_settings)?)
    } else {
        None
    };
    let model_root = repository
        .models
        .artifact_root(&repository.root, options.model_root.as_deref());
    let model_root = absolute_model_root(&model_root)?;
    let instances = expand_cases(repository, profile, &options.model_ids)?;
    if instances.is_empty() {
        bail!("profile {} expanded to zero case instances", profile.id);
    }
    let selected_model_ids = instances
        .iter()
        .filter_map(|instance| instance.model_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if profile.models.verify_before_run && !selected_model_ids.is_empty() {
        let statuses = repository
            .models
            .verify(&selected_model_ids, &model_root)
            .await
            .context("model preflight verification failed")?;
        require_verified_models("preflight", &statuses)?;
    }
    let store = RunStore::create(&output_root, &run_id).await?;
    let contracts = snapshot_run_contracts(repository, profile, &store).await?;
    let descriptor = json!({
        "schema_version": crate::model::SCHEMA_VERSION,
        "run_id": run_id,
        "profile": profile.id,
        "started_at": started_at,
        "reference_manifest": reference.as_ref().map(|manifest| manifest.path.clone()),
        "reference_manifest_sha256": reference.as_ref().map(ReferenceBuildManifest::digest).transpose()?,
        "candidate_manifest": candidate.as_ref().map(|manifest| manifest.path.clone()),
        "candidate_manifest_sha256": candidate.as_ref().map(|manifest| manifest.digest()),
        "candidate_reference_manifest_sha256": candidate.as_ref().map(|manifest| manifest.document.reference_manifest_sha256.clone()),
        "model_root": model_root.clone(),
        "settings": redacted_settings(&effective_settings),
        "contracts": contracts,
        "case_instances": instances.iter().map(|instance| json!({"case_id": instance.case.id, "model_id": instance.model_id})).collect::<Vec<_>>(),
    });
    store.write_json(Path::new("run.json"), &descriptor).await?;

    let effective_options = RunOptions {
        settings: effective_settings,
        ..options.clone()
    };
    let context = RunContext {
        repository,
        profile,
        options: &effective_options,
        store: &store,
        run_id: &run_id,
        reference: reference.as_ref(),
        candidate: candidate.as_ref(),
        model_root,
        engine,
    };
    preflight_jsonl_producers(&context, &instances)
        .await
        .context("JSONL producer capability preflight failed")?;
    let mut results = Vec::new();
    let mut correctness_failed = false;
    for instance in instances {
        if instance.case.category == Category::Performance
            && let Some(reason) = unmet_prerequisite(&instance, &results, repository)
        {
            results.push(blocked(&instance, reason));
            continue;
        }
        if instance.case.category == Category::Performance
            && profile.gates.require_correctness_before_performance
            && correctness_failed
        {
            results.push(skipped(
                &instance,
                "correctness gate failed before performance execution",
            ));
            continue;
        }
        let result = execute_instance(&context, &instance)
            .await
            .unwrap_or_else(|error| CaseExecutionResult {
                case_id: instance.case.id.clone(),
                category: instance.case.category,
                model_id: instance.model_id.clone(),
                status: CaseExecutionStatus::Error,
                reason: Some(error.to_string()),
                reference_evidence: None,
                candidate_evidence: None,
                comparison: None,
            });
        if instance.case.category == Category::Correctness
            && result.status != CaseExecutionStatus::Pass
        {
            correctness_failed = true;
        }
        let stop = profile.execution.fail_fast && result_is_gating_failure(&result, profile);
        results.push(result);
        if stop {
            break;
        }
    }
    // Re-verify after the timed schedule so manifest/source/artifact mutation
    // is detected without asymmetrically contaminating either engine's turn.
    if let Some(reference) = &reference {
        reference
            .verify()
            .await
            .context("reference postflight verification failed")?;
    }
    if let Some(candidate) = &candidate {
        candidate
            .verify()
            .await
            .context("candidate postflight verification failed")?;
    }
    if !selected_model_ids.is_empty() {
        let statuses = repository
            .models
            .verify(&selected_model_ids, &context.model_root)
            .await
            .context("model postflight verification failed")?;
        require_verified_models("postflight", &statuses)?;
    }
    verify_run_contracts(repository, profile)
        .await
        .context("run contract postflight verification failed")?;
    let counts = count_results(&results);
    let correctness_failed = results.iter().any(|result| {
        result.category == Category::Correctness && result.status == CaseExecutionStatus::Fail
    });
    let performance_failed = results.iter().any(|result| {
        result.category == Category::Performance && result.status == CaseExecutionStatus::Fail
    });
    let status = if counts.blocked > 0 {
        RunStatus::Blocked
    } else if correctness_failed
        || (profile.gates.fail_on_performance && performance_failed)
        || (profile.gates.fail_on_invalid && counts.invalid > 0)
        || (profile.gates.fail_on_error && counts.error > 0)
        || (profile.gates.fail_on_skipped && counts.skipped > 0)
    {
        RunStatus::Fail
    } else {
        RunStatus::Pass
    };
    let summary = RunSummary {
        schema_version: crate::model::SCHEMA_VERSION.to_owned(),
        run_id,
        profile: profile.id.clone(),
        started_at,
        completed_at: now_rfc3339(),
        status,
        output_directory: store.root().to_path_buf(),
        cases: results,
        counts,
    };
    store
        .write_json(Path::new("summary.json"), &summary)
        .await?;
    Ok(summary)
}

fn absolute_model_root(path: &Path) -> anyhow::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to resolve the current directory for --model-root")?
            .join(path)
    };
    if absolute.exists() {
        std::fs::canonicalize(&absolute)
            .with_context(|| format!("failed to canonicalize model root {}", absolute.display()))
    } else {
        Ok(absolute)
    }
}

fn require_execution_reference_schema(schema_version: u64) -> anyhow::Result<()> {
    if schema_version != 3 {
        bail!(
            "parity execution requires a schema-v3 reference manifest; schema {schema_version} does not prove the required source/configuration identity"
        );
    }
    Ok(())
}

fn reference_source_root(manifest: &ReferenceBuildManifest) -> anyhow::Result<&Path> {
    manifest
        .native_source
        .as_ref()
        .and_then(|source| source.path.as_deref())
        .context("CTest execution requires a reference manifest with llamaCpp.path")
}

async fn cleanup_ctest_source_residue(source_root: &Path) -> anyhow::Result<()> {
    let mut failures = Vec::new();
    for name in CTEST_SOURCE_RESIDUE_FILES {
        let path = source_root.join(name);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => failures.push(format!("{}: {error}", path.display())),
        }
    }
    if !failures.is_empty() {
        bail!(
            "failed to remove scoped CTest source residue: {}",
            failures.join("; ")
        );
    }
    Ok(())
}

async fn with_ctest_source_cleanup<T, F>(source_root: &Path, operation: F) -> anyhow::Result<T>
where
    F: Future<Output = anyhow::Result<T>>,
{
    cleanup_ctest_source_residue(source_root)
        .await
        .context("failed to prepare a clean CTest source root")?;
    let operation_result = operation.await;
    let cleanup_result = cleanup_ctest_source_residue(source_root).await;
    match (operation_result, cleanup_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup_error)) => Err(cleanup_error)
            .context("CTest completed but its scoped source residue could not be removed"),
        (Err(operation_error), Err(cleanup_error)) => Err(operation_error).context(format!(
            "CTest invocation failed and scoped source cleanup also failed: {cleanup_error:#}"
        )),
    }
}

fn require_verified_models(
    phase: &str,
    statuses: &[crate::models::ModelFileStatus],
) -> anyhow::Result<()> {
    if all_verified(statuses) {
        return Ok(());
    }
    let failures = statuses
        .iter()
        .filter(|status| status.state != ModelFileState::Verified)
        .map(|status| format!("{} ({:?})", status.path.display(), status.state))
        .collect::<Vec<_>>()
        .join(", ");
    bail!("model {phase} verification failed: {failures}")
}

async fn snapshot_run_contracts(
    repository: &AssetRepository,
    profile: &Profile,
    store: &RunStore,
) -> anyhow::Result<Vec<Value>> {
    let paths = run_contract_paths(repository, profile)?;

    let mut records = Vec::with_capacity(paths.len());
    for source in paths {
        let expected = repository.contract_digests.get(&source).with_context(|| {
            format!("no load-time digest for run contract {}", source.display())
        })?;
        let bytes = read_contract_matching_digest(&source, expected).await?;
        let relative = source.strip_prefix(&repository.root).with_context(|| {
            format!(
                "run contract path escapes parity root: {}",
                source.display()
            )
        })?;
        let destination = Path::new("contracts/assets").join(relative);
        let stored = store.write_bytes(&destination, &bytes).await?;
        records.push(json!({
            "source": relative,
            "snapshot": stored.relative_path,
            "sha256": stored.sha256,
            "bytes": stored.bytes,
        }));
    }
    Ok(records)
}

fn run_contract_paths(
    repository: &AssetRepository,
    profile: &Profile,
) -> anyhow::Result<BTreeSet<PathBuf>> {
    let profile_path = repository
        .profile_paths
        .get(&profile.id)
        .with_context(|| format!("profile {} has no bound source path", profile.id))?;
    let mut paths = BTreeSet::from([
        profile_path.clone(),
        repository.root.join("models/registry.toml"),
        repository.root.join("producers.toml"),
        repository.root.join("upstream/targets.toml"),
        repository.root.join("upstream/build-profiles.toml"),
        repository.root.join("upstream/binding-surfaces.json"),
        repository.root.join("schemas/case.schema.json"),
        repository.root.join("schemas/evidence.schema.json"),
        repository.root.join("schemas/comparison.schema.json"),
    ]);
    for loaded in repository.selected_cases(profile) {
        paths.insert(loaded.path.clone());
        for fixture in &loaded.definition.requirements.fixtures {
            paths.insert(repository.root.join(&fixture.path));
        }
    }
    Ok(paths)
}

async fn read_contract_matching_digest(path: &Path, expected: &str) -> anyhow::Result<Vec<u8>> {
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("failed to read run contract {}", path.display()))?;
    let actual = sha256_bytes(&bytes);
    if actual != expected {
        bail!(
            "run contract {} changed after it was loaded (expected {expected}, actual {actual})",
            path.display()
        );
    }
    Ok(bytes)
}

async fn verify_run_contracts(
    repository: &AssetRepository,
    profile: &Profile,
) -> anyhow::Result<()> {
    for path in run_contract_paths(repository, profile)? {
        let expected = repository
            .contract_digests
            .get(&path)
            .with_context(|| format!("no load-time digest for run contract {}", path.display()))?;
        read_contract_matching_digest(&path, expected).await?;
    }
    Ok(())
}

fn backend_requirement_block(
    allowed_backends: &[String],
    reference_backend: Option<&str>,
) -> Option<String> {
    if allowed_backends.is_empty() {
        return None;
    }
    let Some(reference_backend) = reference_backend else {
        return Some("backend-constrained case requires a verified reference manifest".to_owned());
    };
    (!allowed_backends
        .iter()
        .any(|backend| backend == reference_backend))
    .then(|| {
        format!(
            "reference backend {reference_backend} is outside the case's allowed backends {allowed_backends:?}"
        )
    })
}

async fn execute_instance(
    context: &RunContext<'_>,
    instance: &CaseInstance,
) -> anyhow::Result<CaseExecutionResult> {
    let mut prepared_case = instance.case.clone();
    if prepared_case.operation == "backend-ops.perf" {
        let device = context.options.settings.get("backend_device").map(String::as_str)
            .context("backend-ops qualification requires --set backend_device=<exact upstream device name>")?;
        prepared_case
            .inputs
            .insert("backend_device".to_owned(), json!(device));
    }
    let case = &prepared_case;
    if case.status != crate::model::CaseStatus::Active {
        return Ok(skipped(
            instance,
            match case.status {
                crate::model::CaseStatus::Planned => {
                    "case is planned and has no accepted producer pair"
                }
                crate::model::CaseStatus::Disabled => case
                    .disabled_reason
                    .as_deref()
                    .unwrap_or("case is disabled"),
                crate::model::CaseStatus::Active => unreachable!(),
            },
        ));
    }
    if let Some(reason) = backend_requirement_block(
        &case.requirements.backends,
        context
            .reference
            .map(|reference| reference.backend.as_str()),
    ) {
        return Ok(blocked(instance, reason));
    }
    if case.category == Category::Performance {
        if case.invocations.candidate.is_some() && context.candidate.is_none() {
            return Ok(blocked(
                instance,
                "two-sided performance parity requires a verified --candidate-manifest",
            ));
        }
        if context.profile.performance.require_controlled_host
            && !setting_is_true(&context.options.settings, "controlled_host")
        {
            return Ok(blocked(
                instance,
                "performance profile requires explicit --set controlled_host=true attestation",
            ));
        }
        if context.profile.performance.require_exclusive_device
            && !setting_is_true(&context.options.settings, "exclusive_device")
        {
            return Ok(blocked(
                instance,
                "performance profile requires explicit --set exclusive_device=true attestation",
            ));
        }
    }
    let (model_paths, model_artifacts) = match resolve_model(context, instance).await? {
        ModelResolution::Ready { paths, artifacts } => (paths, artifacts),
        ModelResolution::Blocked(reason) => return Ok(blocked(instance, reason)),
    };
    let fixture_paths = resolve_fixtures(context, case).await?;
    let base_runtime_input = runtime_input(
        case,
        instance.model_id.as_deref(),
        &model_paths,
        &fixture_paths,
        runtime_engine_for_case(context.repository, case, context.engine.as_ref()),
    )?;

    if case.adapter == crate::model::AdapterKind::UpstreamTest
        || case.adapter == crate::model::AdapterKind::UpstreamQualification
    {
        let Some(reference_manifest) = context.reference else {
            return Ok(blocked(
                instance,
                "reference build manifest is unavailable; pass --reference-manifest",
            ));
        };
        let invocation = case.invocations.reference.as_ref().unwrap();
        let target_id = invocation
            .target
            .as_deref()
            .context("active qualification has no target")?;
        reference_manifest.require_selected_target(target_id)?;
        let target = context
            .repository
            .upstream_targets
            .get(target_id)
            .with_context(|| format!("unknown upstream target {target_id}"))?;
        stage_ctest_fixtures(target, reference_manifest, context, &model_paths).await?;
        let stem = instance_stem(instance);
        let ctest_inventory = if target.kind == crate::assets::UpstreamTargetKind::CtestSuite {
            Some(preflight_ctest(context, &stem, target, reference_manifest).await?)
        } else {
            None
        };
        let dummy_engine = ResolvedEngine {
            backend: reference_manifest.backend.clone(),
            backend_device: None,
            threads: 1,
            reference_gpu_layers: 0,
            candidate_gpu_layers: 0,
            flash_attention: "auto".to_owned(),
            cpu_strict: false,
            threadpool_poll: 50,
        };
        let mut command = upstream_command(
            target,
            reference_manifest,
            case,
            &dummy_engine,
            &model_paths,
            &fixture_paths,
            None,
        )?;
        let junit_directory = (target.kind == crate::assets::UpstreamTargetKind::CtestSuite)
            .then(tempfile::tempdir)
            .transpose()?;
        let junit_path = junit_directory
            .as_ref()
            .map(|directory| directory.path().join("ctest.junit.xml"));
        if let Some(path) = &junit_path {
            command
                .args
                .extend(["--output-junit".to_owned(), path.display().to_string()]);
        }
        let invocation_result = run_invocation(
            context,
            case,
            ProducerRole::Qualification,
            invocation,
            command,
            &base_runtime_input,
            reference_manifest.provenance_components(false),
            &stem,
            0,
        );
        let (mut evidence, process) =
            if target.kind == crate::assets::UpstreamTargetKind::CtestSuite {
                with_ctest_source_cleanup(
                    reference_source_root(reference_manifest)?,
                    invocation_result,
                )
                .await?
            } else {
                invocation_result.await?
            };
        let junit = if let Some(path) = &junit_path {
            let bytes = tokio::fs::read(path).await.with_context(|| {
                format!(
                    "CTest did not produce required JUnit evidence at {}",
                    path.display()
                )
            })?;
            let summary = parse_junit_summary(&bytes)?;
            if process.success()
                && (summary.tests != target.ctest_names.len() as u64
                    || summary.failures != 0
                    || summary.errors != 0
                    || summary.disabled != 0
                    || summary.skipped != 0
                    || summary.test_names != target.ctest_names.iter().cloned().collect())
            {
                bail!("CTest JUnit counts do not match the selected passing test inventory");
            }
            let stored = context
                .store
                .write_bytes(
                    &PathBuf::from(format!("raw/{stem}/qualification.junit.xml")),
                    &bytes,
                )
                .await?;
            Some(json!({
                "path": path_string(&stored), "sha256": stored.sha256, "bytes": stored.bytes,
                "tests": summary.tests, "failures": summary.failures,
                "errors": summary.errors, "disabled": summary.disabled,
                "skipped": summary.skipped,
            }))
        } else {
            None
        };
        let staged_fixtures = if target.kind == crate::assets::UpstreamTargetKind::CtestSuite {
            Some(verify_staged_ctest_fixtures(target, reference_manifest, context).await?)
        } else {
            None
        };
        if let Some(inventory) = ctest_inventory {
            evidence.output = json!({
                "exit_code": process.exit_code,
                "inventory": inventory,
                "junit": junit,
                "staged_fixtures": staged_fixtures,
            });
        }
        decorate_reference_evidence(
            &mut evidence,
            reference_manifest,
            context,
            &model_artifacts,
            &fixture_paths,
            true,
        )
        .await?;
        verify_input_artifacts_unchanged(
            context,
            instance.model_id.as_deref(),
            &fixture_paths,
            &evidence,
        )
        .await?;
        let stored =
            write_evidence_and_raw(context, &stem, "qualification", &evidence, &process, 0).await?;
        let status = if evidence.outcome.class == OutcomeClass::Success {
            CaseExecutionStatus::Pass
        } else {
            CaseExecutionStatus::Fail
        };
        return Ok(CaseExecutionResult {
            case_id: case.id.clone(),
            category: case.category,
            model_id: instance.model_id.clone(),
            status,
            reason: evidence.outcome.message.clone(),
            reference_evidence: Some(path_string(&stored)),
            candidate_evidence: None,
            comparison: None,
        });
    }

    let Some(reference_manifest) = context.reference else {
        return Ok(blocked(
            instance,
            "reference build manifest is unavailable; pass --reference-manifest",
        ));
    };
    let reference_invocation = case.invocations.reference.as_ref().unwrap();
    let candidate_invocation = case.invocations.candidate.as_ref().unwrap();
    if let Err(reason) = producer_available(context, candidate_invocation) {
        return Ok(blocked(instance, reason));
    }
    let repetitions = if case.category == Category::Performance {
        context.profile.performance.paired_repetitions
    } else {
        1
    };
    let mut reference_evidence = None;
    let mut candidate_evidence = None;
    let stem = instance_stem(instance);
    for pair in 0..repetitions {
        let reference_first = match context.profile.execution.performance_engine_order {
            PerformanceOrder::Alternate => pair % 2 == 0,
            PerformanceOrder::ReferenceFirst => true,
            PerformanceOrder::CandidateFirst => false,
        };
        let order = if case.category == Category::Performance && !reference_first {
            [ProducerRole::Candidate, ProducerRole::Reference]
        } else {
            [ProducerRole::Reference, ProducerRole::Candidate]
        };
        for role in order {
            let invocation_case = paired_invocation_case(case);
            let invocation_runtime_input = runtime_input(
                &invocation_case,
                instance.model_id.as_deref(),
                &model_paths,
                &fixture_paths,
                runtime_engine_for_case(
                    context.repository,
                    &invocation_case,
                    context.engine.as_ref(),
                ),
            )?;
            let (invocation, command, extra_components) = match role {
                ProducerRole::Reference => {
                    let target_id = reference_invocation
                        .target
                        .as_deref()
                        .or_else(|| {
                            (reference_invocation.kind == InvocationKind::NativeOracle)
                                .then_some("oracle")
                        })
                        .context("reference invocation has no target")?;
                    reference_manifest.require_selected_target(target_id)?;
                    let target = context
                        .repository
                        .upstream_targets
                        .get(target_id)
                        .with_context(|| format!("unknown upstream target {target_id}"))?;
                    let fallback;
                    let engine = if let Some(engine) = context.engine.as_ref() {
                        engine
                    } else {
                        fallback = ResolvedEngine {
                            backend: reference_manifest.backend.clone(),
                            backend_device: None,
                            threads: 1,
                            reference_gpu_layers: 0,
                            candidate_gpu_layers: 0,
                            flash_attention: "auto".to_owned(),
                            cpu_strict: false,
                            threadpool_poll: 50,
                        };
                        &fallback
                    };
                    (
                        reference_invocation,
                        upstream_command(
                            target,
                            reference_manifest,
                            &invocation_case,
                            engine,
                            &model_paths,
                            &fixture_paths,
                            (case.category == Category::Performance).then_some(1),
                        )?,
                        reference_manifest.provenance_components(
                            reference_invocation.kind == InvocationKind::NativeOracle,
                        ),
                    )
                }
                ProducerRole::Candidate => {
                    let producer = resolve_producer(context, candidate_invocation)?;
                    (
                        candidate_invocation,
                        producer_command(
                            context,
                            producer,
                            &invocation_case,
                            instance.model_id.as_deref(),
                            &model_paths,
                        )?,
                        candidate_source_components(context).await?,
                    )
                }
                ProducerRole::Qualification => unreachable!(),
            };
            let (mut evidence, process) = run_invocation(
                context,
                &invocation_case,
                role,
                invocation,
                command,
                &invocation_runtime_input,
                extra_components,
                &stem,
                pair,
            )
            .await?;
            match role {
                ProducerRole::Reference => {
                    decorate_reference_evidence(
                        &mut evidence,
                        reference_manifest,
                        context,
                        &model_artifacts,
                        &fixture_paths,
                        true,
                    )
                    .await?
                }
                ProducerRole::Candidate => {
                    decorate_candidate_evidence(
                        &mut evidence,
                        reference_manifest,
                        context,
                        &model_artifacts,
                        &fixture_paths,
                        candidate_invocation
                            .target
                            .as_deref()
                            .unwrap_or("icn-probe"),
                    )
                    .await?
                }
                ProducerRole::Qualification => unreachable!(),
            }
            write_raw(context, &stem, role_name(role), &process, pair).await?;
            let accumulator = if role == ProducerRole::Reference {
                &mut reference_evidence
            } else {
                &mut candidate_evidence
            };
            merge_evidence(case, accumulator, evidence)?;
        }
    }
    let reference_evidence = reference_evidence.context("reference produced no evidence")?;
    let candidate_evidence = candidate_evidence.context("candidate produced no evidence")?;
    verify_input_artifacts_unchanged(
        context,
        instance.model_id.as_deref(),
        &fixture_paths,
        &reference_evidence,
    )
    .await?;
    if case.category == Category::Performance {
        validate_paired_cardinality(&reference_evidence, repetitions, "reference")?;
        validate_paired_cardinality(&candidate_evidence, repetitions, "candidate")?;
    }
    context
        .repository
        .validate_schema(SchemaKind::Evidence, &reference_evidence)?;
    context
        .repository
        .validate_schema(SchemaKind::Evidence, &candidate_evidence)?;
    let reference_file = context
        .store
        .write_json(
            &PathBuf::from(format!("evidence/{stem}/reference.json")),
            &reference_evidence,
        )
        .await?;
    let candidate_file = context
        .store
        .write_json(
            &PathBuf::from(format!("evidence/{stem}/candidate.json")),
            &candidate_evidence,
        )
        .await?;
    let engine_order = if case.category == Category::Correctness {
        EngineOrder::ReferenceFirst
    } else {
        match context.profile.execution.performance_engine_order {
            PerformanceOrder::Alternate => EngineOrder::AlternatingPairs,
            PerformanceOrder::ReferenceFirst => EngineOrder::ReferenceFirst,
            PerformanceOrder::CandidateFirst => EngineOrder::CandidateFirst,
        }
    };
    let comparison = compare_evidence(
        case,
        &reference_evidence,
        &candidate_evidence,
        ComparisonContext {
            run_id: context.run_id,
            reference_path: &path_string(&reference_file),
            reference_sha256: &reference_file.sha256,
            candidate_path: &path_string(&candidate_file),
            candidate_sha256: &candidate_file.sha256,
            engine_order,
        },
    )?;
    context
        .repository
        .validate_schema(SchemaKind::Comparison, &comparison)?;
    let comparison_file = if context.profile.results.write_comparisons {
        Some(
            context
                .store
                .write_json(
                    &PathBuf::from(format!("comparisons/{stem}.json")),
                    &comparison,
                )
                .await?,
        )
    } else {
        None
    };
    let status = match comparison.status {
        ComparisonStatus::Pass => CaseExecutionStatus::Pass,
        ComparisonStatus::Fail => CaseExecutionStatus::Fail,
        ComparisonStatus::Invalid => CaseExecutionStatus::Invalid,
        ComparisonStatus::Skipped => CaseExecutionStatus::Skipped,
    };
    Ok(CaseExecutionResult {
        case_id: case.id.clone(),
        category: case.category,
        model_id: instance.model_id.clone(),
        status,
        reason: comparison.validity.reasons.first().cloned(),
        reference_evidence: Some(path_string(&reference_file)),
        candidate_evidence: Some(path_string(&candidate_file)),
        comparison: comparison_file.as_ref().map(path_string),
    })
}

#[allow(clippy::too_many_arguments)]
async fn run_invocation(
    context: &RunContext<'_>,
    case: &CaseDefinition,
    role: ProducerRole,
    invocation: &Invocation,
    command: crate::model::CommandSpec,
    runtime_input: &Map<String, Value>,
    extra_components: Vec<Component>,
    stem: &str,
    repetition: u64,
) -> anyhow::Result<(EvidenceRecord, ProcessOutput)> {
    let stdin = encode_case_stdin(case, command.stdin, Some(runtime_input))?;
    let command = sandbox_command(command, &context.options.settings)?;
    let invocation_record = json!({
        "schema_version": crate::model::SCHEMA_VERSION,
        "run_id": context.run_id,
        "case_id": case.id,
        "producer_role": role,
        "repetition": repetition,
        "program": &command.program,
        "argv": &command.args,
        "cwd": &command.cwd,
        "clear_env": command.clear_env,
        "environment": &command.env,
        "stdin_sha256": stdin.as_ref().map(|bytes| sha256_bytes(bytes)),
        "stdin_bytes": stdin.as_ref().map_or(0, Vec::len),
    });
    context
        .store
        .write_json(
            &PathBuf::from(format!(
                "invocations/{stem}/{}-{repetition:04}.json",
                role_name(role)
            )),
            &invocation_record,
        )
        .await?;
    let limits = ProcessLimits {
        timeout: Duration::from_secs(context.profile.execution.case_timeout_seconds),
        ..ProcessLimits::default()
    };
    let output = run_bounded(&command, stdin.as_deref(), limits).await?;
    let evidence = decode_process_evidence(
        context.run_id,
        case,
        role,
        invocation.kind,
        &command,
        &output,
        extra_components,
    )
    .await?;
    Ok((evidence, output))
}

fn sandbox_command(
    mut command: crate::model::CommandSpec,
    settings: &BTreeMap<String, String>,
) -> anyhow::Result<crate::model::CommandSpec> {
    const ALLOWED: &[&str] = &[
        "LC_ALL",
        "LANG",
        "TZ",
        "TMPDIR",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
        "GGML_VK_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "GGML_METAL_PATH_RESOURCES",
        "GGML_METAL_DEVICE",
        "GGML_CUDA_NO_PINNED",
    ];
    for name in command.env.keys() {
        if !ALLOWED.contains(&name.as_str()) {
            bail!("producer declares non-allowlisted environment variable {name}");
        }
    }
    command.env.insert("LC_ALL".to_owned(), "C".to_owned());
    command.env.insert("LANG".to_owned(), "C".to_owned());
    command.env.insert("TZ".to_owned(), "UTC".to_owned());
    for (key, value) in settings.iter().filter(|(key, _)| key.starts_with("env.")) {
        let name = &key[4..];
        if !ALLOWED.contains(&name) {
            bail!("--set {key}=... is not an allowlisted parity subprocess environment variable");
        }
        command.env.insert(name.to_owned(), value.clone());
    }
    command.clear_env = true;
    command.program = resolve_program(&command.program)?;
    if let Some(cwd) = &command.cwd {
        command.cwd = Some(std::fs::canonicalize(cwd).with_context(|| {
            format!(
                "failed to resolve producer working directory {}",
                cwd.display()
            )
        })?);
    }
    Ok(command)
}

fn expand_cases(
    repository: &AssetRepository,
    profile: &Profile,
    model_overrides: &[String],
) -> anyhow::Result<Vec<CaseInstance>> {
    let mut instances = Vec::new();
    for loaded in repository.selected_cases(profile) {
        let case = &loaded.definition;
        let Some(selector) = &case.requirements.model else {
            instances.push(CaseInstance {
                case: case.clone(),
                model_id: None,
            });
            continue;
        };
        let fixed = !selector.ids.is_empty();
        let requested = if fixed {
            selector.ids.clone()
        } else if !model_overrides.is_empty() {
            model_overrides.to_vec()
        } else {
            profile.models.ids.clone()
        };
        let before = instances.len();
        let mut seen = BTreeSet::new();
        for model_id in requested {
            if !seen.insert(model_id.clone()) {
                continue;
            }
            let model = repository.models.by_id(&model_id)?;
            if !model.valid_for.contains(&case.primitive)
                || !selector
                    .all_tags
                    .iter()
                    .all(|tag| model.attributes.architecture_tags.contains(tag))
                || (!selector.any_tags.is_empty()
                    && !selector
                        .any_tags
                        .iter()
                        .any(|tag| model.attributes.architecture_tags.contains(tag)))
                || !case
                    .requirements
                    .architecture_tags
                    .iter()
                    .all(|tag| model.attributes.architecture_tags.contains(tag))
            {
                continue;
            }
            let mut expanded = case.clone();
            expanded
                .inputs
                .insert("model_id".to_owned(), json!(model_id));
            instances.push(CaseInstance {
                case: expanded,
                model_id: Some(model_id),
            });
        }
        if instances.len() == before {
            bail!(
                "selected case {} has no compatible model instance after applying registry validity/tags and model selection",
                case.id
            );
        }
    }
    Ok(instances)
}

enum ModelResolution {
    Ready {
        paths: BTreeMap<String, PathBuf>,
        artifacts: Vec<ArtifactInfo>,
    },
    Blocked(String),
}

async fn resolve_model(
    context: &RunContext<'_>,
    instance: &CaseInstance,
) -> anyhow::Result<ModelResolution> {
    let Some(model_id) = &instance.model_id else {
        return Ok(ModelResolution::Ready {
            paths: BTreeMap::new(),
            artifacts: Vec::new(),
        });
    };
    let statuses = context
        .repository
        .models
        .verify(std::slice::from_ref(model_id), &context.model_root)
        .await?;
    if !all_verified(&statuses) {
        let failures = statuses
            .iter()
            .filter(|status| status.state != ModelFileState::Verified)
            .map(|status| format!("{} ({:?})", status.path.display(), status.state))
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(ModelResolution::Blocked(format!(
            "model {model_id} is not verified: {failures}; run `icn-parity models fetch --id {model_id}` explicitly"
        )));
    }
    let paths_by_role = context
        .repository
        .models
        .model_paths(model_id, &context.model_root)?;
    let primary = paths_by_role
        .get("model")
        .or_else(|| paths_by_role.values().next())
        .context("model has no primary file")?
        .clone();
    let mut paths = paths_by_role.clone();
    paths.insert(model_id.clone(), primary.clone());
    paths.insert("primary".to_owned(), primary);
    let model = context.repository.models.by_id(model_id)?;
    let artifacts = model
        .files
        .iter()
        .map(|file| ArtifactInfo {
            kind: artifact_kind_for_role(&file.role).to_owned(),
            id: format!("{model_id}:{}", file.role),
            sha256: file.sha256.clone(),
            bytes: file.bytes,
        })
        .collect();
    Ok(ModelResolution::Ready { paths, artifacts })
}

async fn resolve_fixtures(
    context: &RunContext<'_>,
    case: &CaseDefinition,
) -> anyhow::Result<BTreeMap<String, PathBuf>> {
    let mut paths = BTreeMap::new();
    let canonical_root = std::fs::canonicalize(&context.repository.root).with_context(|| {
        format!(
            "failed to resolve parity root {}",
            context.repository.root.display()
        )
    })?;
    for fixture in &case.requirements.fixtures {
        let expected = fixture
            .sha256
            .as_deref()
            .context("validated fixture is missing its required sha256")?;
        let declared_path = context.repository.root.join(&fixture.path);
        let path = std::fs::canonicalize(&declared_path)
            .with_context(|| format!("failed to resolve fixture {}", declared_path.display()))?;
        if !path.starts_with(&canonical_root) || !path.is_file() {
            bail!(
                "fixture {} is not a regular file within the parity root",
                fixture.path
            );
        }
        let (digest, _) = sha256_file(&path).await?;
        if expected != digest {
            bail!("fixture {} changed after validation", fixture.path);
        }
        paths.insert(fixture.id.clone(), path.clone());
        paths.insert(fixture.path.clone(), path);
    }
    Ok(paths)
}

fn runtime_input(
    case: &CaseDefinition,
    model_id: Option<&str>,
    model_paths: &BTreeMap<String, PathBuf>,
    fixture_paths: &BTreeMap<String, PathBuf>,
    engine: Option<&ResolvedEngine>,
) -> anyhow::Result<Map<String, Value>> {
    let mut input = case.inputs.clone();
    if let Some(model_id) = model_id {
        if let Some(declared) = input.remove("model_id")
            && declared.as_str() != Some(model_id)
        {
            bail!(
                "expanded case model_id does not match its selected model instance: expected {model_id}, found {declared}"
            );
        }
        let path = model_paths
            .get("primary")
            .context("selected model has no primary path")?;
        input.insert("modelId".to_owned(), json!(model_id));
        input.insert("modelPath".to_owned(), json!(path));
    }
    let fixtures = fixture_paths
        .iter()
        .filter(|(key, _)| !key.starts_with("fixtures/"))
        .map(|(id, path)| (id.clone(), json!(path)))
        .collect::<Map<_, _>>();
    if !fixtures.is_empty() {
        input.insert("fixturePaths".to_owned(), Value::Object(fixtures));
    }
    if let Some(engine) = engine {
        let gpu_layers =
            if engine.reference_gpu_layers == -2 && engine.candidate_gpu_layers == u32::MAX {
                json!("all")
            } else {
                json!(engine.candidate_gpu_layers)
            };
        input.insert(
            "engineConfiguration".to_owned(),
            json!({
                "backend": engine.backend,
                "threads": engine.threads,
                "gpuLayers": gpu_layers,
                "flashAttention": engine.flash_attention,
                "cpuStrict": engine.cpu_strict,
                "threadpoolPoll": engine.threadpool_poll,
            }),
        );
    }
    Ok(input)
}

fn paired_invocation_case(case: &CaseDefinition) -> CaseDefinition {
    let mut invocation = case.clone();
    if case.category == Category::Performance && invocation.inputs.contains_key("repetitions") {
        // Official benchmark tools expose one sample per `repetitions` entry.
        // The runner owns cross-process paired repetitions, so those tools run
        // once per pair. Custom microbench `iterations` remain part of the timed
        // work and must never be rewritten.
        invocation.inputs.insert("repetitions".to_owned(), json!(1));
        if let Some(timing) = &mut invocation.timing {
            timing.measurement_iterations = 1;
        }
    }
    invocation
}

#[derive(Debug)]
struct ProtocolPreflightTarget {
    role: ProducerRole,
    producer_id: String,
    command: CommandSpec,
    required_operations: BTreeSet<String>,
}

async fn preflight_jsonl_producers(
    context: &RunContext<'_>,
    instances: &[CaseInstance],
) -> anyhow::Result<()> {
    let mut targets = Vec::new();
    for instance in instances
        .iter()
        .filter(|instance| instance.case.status == crate::model::CaseStatus::Active)
    {
        let uses_jsonl_protocol = instance
            .case
            .invocations
            .reference
            .as_ref()
            .is_some_and(|invocation| invocation.kind == InvocationKind::NativeOracle)
            || instance
                .case
                .invocations
                .candidate
                .as_ref()
                .is_some_and(|invocation| invocation.kind == InvocationKind::IcnProbe);
        if !uses_jsonl_protocol {
            continue;
        }
        let model_paths = preflight_model_paths(context, instance)?;
        if let Some(invocation) = instance.case.invocations.reference.as_ref()
            && invocation.kind == InvocationKind::NativeOracle
            && let Some(reference) = context.reference
        {
            let target_id = invocation.target.as_deref().unwrap_or("oracle");
            reference.require_selected_target(target_id)?;
            let target = context
                .repository
                .upstream_targets
                .get(target_id)
                .with_context(|| format!("unknown native oracle target {target_id}"))?;
            let fallback = fallback_engine(reference);
            let engine = context.engine.as_ref().unwrap_or(&fallback);
            let command = upstream_command(
                target,
                reference,
                &instance.case,
                engine,
                &model_paths,
                &BTreeMap::new(),
                None,
            )?;
            add_protocol_preflight_target(
                &mut targets,
                ProducerRole::Reference,
                target_id,
                command,
                &instance.case.operation,
            );
        }

        if let Some(invocation) = instance.case.invocations.candidate.as_ref()
            && invocation.kind == InvocationKind::IcnProbe
            // An unavailable producer is classified as a blocked case by the normal
            // execution path. A producer that resolves must prove its live protocol.
            && producer_available(context, invocation).is_ok()
        {
            let producer = resolve_producer(context, invocation)?;
            let command = producer_command(
                context,
                producer,
                &instance.case,
                instance.model_id.as_deref(),
                &model_paths,
            )?;
            add_protocol_preflight_target(
                &mut targets,
                ProducerRole::Candidate,
                &producer.id,
                command,
                &instance.case.operation,
            );
        }
    }

    let request = describe_request_jsonl()?;
    let limits = ProcessLimits {
        timeout: Duration::from_secs(context.profile.execution.case_timeout_seconds.clamp(1, 60)),
        max_stdout_bytes: 1024 * 1024,
        max_stderr_bytes: 1024 * 1024,
    };
    let mut summaries = Vec::with_capacity(targets.len());
    for (index, target) in targets.into_iter().enumerate() {
        let label = format!("{} producer {}", role_name(target.role), target.producer_id);
        if target.command.stdin != StdinKind::ProbeJsonl
            || target.command.decoder != DecoderKind::ProbeJsonl
        {
            bail!(
                "{label} is used as a JSONL primitive producer but resolves to stdin={:?}, decoder={:?}",
                target.command.stdin,
                target.command.decoder
            );
        }
        let command = sandbox_command(target.command, &context.options.settings)
            .with_context(|| format!("failed to sandbox {label} for protocol preflight"))?;
        let stem = format!(
            "{:02}-{}-{}",
            index + 1,
            role_name(target.role),
            case_file_stem(&target.producer_id)
        );
        let directory = PathBuf::from("preflight/producers").join(&stem);
        context
            .store
            .write_json(
                &directory.join("invocation.json"),
                &json!({
                    "schema_version": crate::model::SCHEMA_VERSION,
                    "producer_role": target.role,
                    "producer_id": target.producer_id,
                    "program": &command.program,
                    "argv": &command.args,
                    "cwd": &command.cwd,
                    "clear_env": command.clear_env,
                    "environment": &command.env,
                    "required_operations": &target.required_operations,
                    "stdin_sha256": sha256_bytes(&request),
                    "stdin_bytes": request.len(),
                }),
            )
            .await?;
        context
            .store
            .write_bytes(&directory.join("request.jsonl"), &request)
            .await?;
        let output = run_bounded(&command, Some(&request), limits)
            .await
            .with_context(|| format!("failed to execute {label} protocol preflight"))?;
        context
            .store
            .write_bytes(&directory.join("stdout.jsonl"), &output.stdout)
            .await?;
        context
            .store
            .write_bytes(&directory.join("stderr.txt"), &output.stderr)
            .await?;
        let description =
            validate_protocol_description(&label, &output, &target.required_operations)?;
        if target.role == ProducerRole::Candidate {
            let producer = context
                .repository
                .producers
                .get(&target.producer_id)
                .with_context(|| {
                    format!("candidate producer {} is not declared", target.producer_id)
                })?;
            validate_declared_protocol_operations(
                &label,
                &producer.capabilities,
                &description.operations,
            )?;
        }
        let summary = ProtocolPreflightSummary {
            producer_role: target.role,
            producer_id: target.producer_id,
            required_operations: target.required_operations,
            description,
        };
        context
            .store
            .write_json(&directory.join("description.json"), &summary)
            .await?;
        summaries.push(summary);
    }
    context
        .store
        .write_json(Path::new("preflight/producers.json"), &summaries)
        .await?;
    Ok(())
}

fn validate_declared_protocol_operations(
    label: &str,
    declared_capabilities: &[String],
    live_operations: &BTreeSet<String>,
) -> anyhow::Result<()> {
    let mut declared = declared_capabilities
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    declared.insert(DESCRIBE_OPERATION.to_owned());
    if &declared == live_operations {
        return Ok(());
    }
    let missing_live = declared
        .difference(live_operations)
        .cloned()
        .collect::<Vec<_>>();
    let undeclared_live = live_operations
        .difference(&declared)
        .cloned()
        .collect::<Vec<_>>();
    bail!(
        "{label} static capabilities and live protocol.describe operations differ; declared-but-missing-live=[{}], live-but-undeclared=[{}]",
        missing_live.join(", "),
        undeclared_live.join(", ")
    )
}

#[derive(Debug, Serialize)]
struct ProtocolPreflightSummary {
    producer_role: ProducerRole,
    producer_id: String,
    required_operations: BTreeSet<String>,
    description: ProtocolDescription,
}

fn add_protocol_preflight_target(
    targets: &mut Vec<ProtocolPreflightTarget>,
    role: ProducerRole,
    producer_id: &str,
    command: CommandSpec,
    operation: &str,
) {
    if let Some(target) = targets.iter_mut().find(|target| {
        target.role == role && target.producer_id == producer_id && target.command == command
    }) {
        target.required_operations.insert(operation.to_owned());
    } else {
        targets.push(ProtocolPreflightTarget {
            role,
            producer_id: producer_id.to_owned(),
            command,
            required_operations: BTreeSet::from([operation.to_owned()]),
        });
    }
}

fn preflight_model_paths(
    context: &RunContext<'_>,
    instance: &CaseInstance,
) -> anyhow::Result<BTreeMap<String, PathBuf>> {
    let Some(model_id) = instance.model_id.as_deref() else {
        return Ok(BTreeMap::new());
    };
    let mut paths = context
        .repository
        .models
        .model_paths(model_id, &context.model_root)?;
    let primary = paths
        .get("model")
        .or_else(|| paths.values().next())
        .context("model has no primary file")?
        .clone();
    paths.insert(model_id.to_owned(), primary.clone());
    paths.insert("primary".to_owned(), primary);
    Ok(paths)
}

fn fallback_engine(reference: &ReferenceBuildManifest) -> ResolvedEngine {
    ResolvedEngine {
        backend: reference.backend.clone(),
        backend_device: None,
        threads: 1,
        reference_gpu_layers: 0,
        candidate_gpu_layers: 0,
        flash_attention: "auto".to_owned(),
        cpu_strict: false,
        threadpool_poll: 50,
    }
}

fn producer_available(context: &RunContext<'_>, invocation: &Invocation) -> Result<(), String> {
    let producer = resolve_producer(context, invocation).map_err(|error| error.to_string())?;
    if let Some(candidate) = context.candidate {
        return candidate
            .artifact(&producer.id)
            .map(|_| ())
            .map_err(|error| error.to_string());
    }
    if let Some(setting) = &producer.program_setting
        && !context.options.settings.contains_key(setting)
    {
        return Err(format!(
            "ICN producer {} is unavailable; provide --set {setting}=/absolute/path after the binding build is finalized",
            producer.id
        ));
    }
    Ok(())
}

fn resolve_producer<'a>(
    context: &'a RunContext<'_>,
    invocation: &Invocation,
) -> anyhow::Result<&'a ProducerDefinition> {
    let id = invocation.target.as_deref().unwrap_or("icn-probe");
    context
        .repository
        .producers
        .get(id)
        .with_context(|| format!("producer {id} is not declared"))
}

fn producer_command(
    context: &RunContext<'_>,
    producer: &ProducerDefinition,
    case: &CaseDefinition,
    model_id: Option<&str>,
    model_paths: &BTreeMap<String, PathBuf>,
) -> anyhow::Result<crate::model::CommandSpec> {
    let program = if let Some(candidate) = context.candidate {
        candidate.artifact(&producer.id)?
    } else if let Some(program) = &producer.program {
        if program.is_absolute() {
            program.clone()
        } else {
            context.repository.root.join(program)
        }
    } else {
        let key = producer
            .program_setting
            .as_deref()
            .context("producer has no program resolver")?;
        PathBuf::from(
            context
                .options
                .settings
                .get(key)
                .with_context(|| format!("missing producer setting {key}"))?,
        )
    };
    let args = producer
        .args
        .iter()
        .map(|argument| render_template(argument, context, case, model_id, model_paths))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let cwd = producer.cwd.as_ref().map(|cwd| {
        if cwd.is_absolute() {
            cwd.clone()
        } else {
            context.repository.root.join(cwd)
        }
    });
    Ok(producer.command(program, args, cwd))
}

fn render_template(
    template: &str,
    context: &RunContext<'_>,
    case: &CaseDefinition,
    model_id: Option<&str>,
    model_paths: &BTreeMap<String, PathBuf>,
) -> anyhow::Result<String> {
    let pattern = regex::Regex::new(r"\{([^{}]+)\}")?;
    let mut rendered = String::new();
    let mut end = 0;
    for capture in pattern.captures_iter(template) {
        let matched = capture.get(0).unwrap();
        rendered.push_str(&template[end..matched.start()]);
        let key = &capture[1];
        let value = if let Some(key) = key.strip_prefix("setting.") {
            context
                .options
                .settings
                .get(key)
                .with_context(|| format!("missing setting {key}"))?
                .clone()
        } else if let Some(key) = key.strip_prefix("input.") {
            scalar(
                case.inputs
                    .get(key)
                    .with_context(|| format!("case input {key} is missing"))?,
            )?
        } else if let Some(key) = key.strip_prefix("model.") {
            let selected =
                model_id.context("producer template requests a model for a model-free case")?;
            if key != "primary" && key != selected {
                bail!(
                    "producer template hard-codes model {key}, but expanded instance selected {selected}"
                );
            }
            model_paths
                .get("primary")
                .context("model primary path is missing")?
                .display()
                .to_string()
        } else {
            match key {
                "operation" => case.operation.clone(),
                "parity_root" => context.repository.root.display().to_string(),
                "profile.engine.threads" => context
                    .engine
                    .as_ref()
                    .context("engine is unresolved")?
                    .threads
                    .to_string(),
                "profile.engine.gpu_layers" => context
                    .engine
                    .as_ref()
                    .context("engine is unresolved")?
                    .candidate_gpu_layers
                    .to_string(),
                "profile.engine.flash_attention" => context
                    .engine
                    .as_ref()
                    .context("engine is unresolved")?
                    .flash_attention
                    .clone(),
                "profile.engine.backend" => context
                    .engine
                    .as_ref()
                    .context("engine is unresolved")?
                    .backend
                    .clone(),
                _ => bail!("unknown producer template placeholder {{{key}}}"),
            }
        };
        rendered.push_str(&value);
        end = matched.end();
    }
    rendered.push_str(&template[end..]);
    Ok(rendered)
}

fn scalar(value: &Value) -> anyhow::Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string()),
        Value::Bool(value) => Ok(value.to_string()),
        _ => bail!("producer template values must be scalar"),
    }
}

async fn stage_ctest_fixtures(
    target: &crate::assets::UpstreamTarget,
    manifest: &ReferenceBuildManifest,
    context: &RunContext<'_>,
    _selected_model_paths: &BTreeMap<String, PathBuf>,
) -> anyhow::Result<()> {
    let canonical_build = std::fs::canonicalize(&manifest.build_directory).with_context(|| {
        format!(
            "failed to resolve reference build directory {}",
            manifest.build_directory.display()
        )
    })?;
    for fixture in &target.ctest_model_fixtures {
        let statuses = context
            .repository
            .models
            .verify(std::slice::from_ref(&fixture.model_id), &context.model_root)
            .await?;
        if !all_verified(&statuses) {
            bail!(
                "CTest fixture model {} is not present and verified; fetch it explicitly before the offline run",
                fixture.model_id
            );
        }
        let paths = context
            .repository
            .models
            .model_paths(&fixture.model_id, &context.model_root)?;
        let source = paths
            .get(&fixture.artifact_role)
            .with_context(|| {
                format!(
                    "model {} has no role {}",
                    fixture.model_id, fixture.artifact_role
                )
            })?
            .clone();
        let destination = canonical_build.join(&fixture.destination);
        if let Ok(metadata) = std::fs::symlink_metadata(&destination)
            && metadata.file_type().is_symlink()
        {
            bail!(
                "refusing symlinked CTest fixture destination {}",
                destination.display()
            );
        }
        let parent = destination
            .parent()
            .context("CTest fixture destination has no parent")?;
        tokio::fs::create_dir_all(parent).await?;
        let canonical_parent = std::fs::canonicalize(parent)?;
        if !canonical_parent.starts_with(&canonical_build) {
            bail!("CTest fixture destination escapes the verified build directory");
        }
        let (source_digest, source_bytes) = sha256_file(&source).await?;
        if destination.exists() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                let source_metadata = std::fs::metadata(&source)?;
                let destination_metadata = std::fs::metadata(&destination)?;
                if source_metadata.dev() == destination_metadata.dev()
                    && source_metadata.ino() == destination_metadata.ino()
                {
                    bail!(
                        "refusing hard-linked CTest fixture {}; rebuild/stage it as an independent copy",
                        destination.display()
                    );
                }
            }
            let (destination_digest, destination_bytes) = sha256_file(&destination).await?;
            if source_digest != destination_digest || source_bytes != destination_bytes {
                bail!(
                    "refusing to replace mismatched staged CTest fixture {}",
                    destination.display()
                );
            }
            continue;
        }
        // Never hard-link a content-addressed cache artifact into a mutable
        // CMake build tree: an upstream test/build must not be able to alter
        // the verified cache inode.
        let mut input = std::fs::File::open(&source)?;
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)?;
        std::io::copy(&mut input, &mut output)?;
        output.sync_all()?;
    }
    Ok(())
}

async fn verify_staged_ctest_fixtures(
    target: &crate::assets::UpstreamTarget,
    manifest: &ReferenceBuildManifest,
    context: &RunContext<'_>,
) -> anyhow::Result<Vec<Value>> {
    let canonical_build = std::fs::canonicalize(&manifest.build_directory)?;
    let mut records = Vec::new();
    for fixture in &target.ctest_model_fixtures {
        let source = context
            .repository
            .models
            .model_paths(&fixture.model_id, &context.model_root)?
            .remove(&fixture.artifact_role)
            .with_context(|| {
                format!(
                    "model {} has no role {}",
                    fixture.model_id, fixture.artifact_role
                )
            })?;
        let destination = canonical_build.join(&fixture.destination);
        let metadata = std::fs::symlink_metadata(&destination).with_context(|| {
            format!(
                "staged CTest fixture disappeared: {}",
                destination.display()
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            bail!(
                "staged CTest fixture is no longer a regular independent file: {}",
                destination.display()
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let source_metadata = std::fs::metadata(&source)?;
            if source_metadata.dev() == metadata.dev() && source_metadata.ino() == metadata.ino() {
                bail!("staged CTest fixture became hard-linked to the model cache");
            }
        }
        let (source_sha256, source_bytes) = sha256_file(&source).await?;
        let (destination_sha256, destination_bytes) = sha256_file(&destination).await?;
        if source_sha256 != destination_sha256 || source_bytes != destination_bytes {
            bail!(
                "CTest changed staged fixture {} during execution",
                destination.display()
            );
        }
        records.push(json!({
            "setup_name": fixture.setup_name,
            "model_id": fixture.model_id,
            "artifact_role": fixture.artifact_role,
            "destination": fixture.destination,
            "sha256": destination_sha256,
            "bytes": destination_bytes,
        }));
    }
    Ok(records)
}

async fn verify_input_artifacts_unchanged(
    context: &RunContext<'_>,
    model_id: Option<&str>,
    fixture_paths: &BTreeMap<String, PathBuf>,
    evidence: &EvidenceRecord,
) -> anyhow::Result<()> {
    if let Some(model_id) = model_id {
        let statuses = context
            .repository
            .models
            .verify(&[model_id.to_owned()], &context.model_root)
            .await?;
        if !all_verified(&statuses) {
            bail!("model {model_id} changed during parity execution");
        }
    }
    for (id, path) in fixture_paths {
        if id.starts_with("fixtures/") {
            continue;
        }
        let (sha256, bytes) = sha256_file(path).await?;
        if let Some(recorded) = evidence
            .provenance
            .artifacts
            .iter()
            .find(|artifact| artifact.id == *id)
            && (recorded.sha256 != sha256 || recorded.bytes != bytes)
        {
            bail!("fixture/corpus {id} changed during parity execution");
        }
    }
    Ok(())
}

async fn preflight_ctest(
    context: &RunContext<'_>,
    stem: &str,
    target: &crate::assets::UpstreamTarget,
    manifest: &ReferenceBuildManifest,
) -> anyhow::Result<Value> {
    let command = sandbox_command(
        ctest_inventory_command(target, manifest)?,
        &context.options.settings,
    )?;
    let invocation = json!({
        "schema_version": crate::model::SCHEMA_VERSION,
        "run_id": context.run_id,
        "case_id": "ctest-inventory-preflight",
        "producer_role": "qualification",
        "program": &command.program,
        "argv": &command.args,
        "cwd": &command.cwd,
        "clear_env": command.clear_env,
        "environment": &command.env,
        "stdin_sha256": Value::Null,
        "stdin_bytes": 0,
    });
    context
        .store
        .write_json(
            &PathBuf::from(format!("invocations/{stem}/qualification-preflight.json")),
            &invocation,
        )
        .await?;
    let output = run_bounded(
        &command,
        None,
        ProcessLimits {
            timeout: Duration::from_secs(context.profile.execution.case_timeout_seconds),
            ..ProcessLimits::default()
        },
    )
    .await?;
    write_raw(context, stem, "qualification-preflight", &output, 0).await?;
    if !output.success() || output.stdout_truncated || output.stderr_truncated {
        bail!("CTest inventory preflight failed or exceeded its output bounds");
    }
    let document: Value = serde_json::from_slice(&output.stdout)
        .context("CTest --show-only=json-v1 emitted invalid JSON")?;
    let tests = document
        .get("tests")
        .and_then(Value::as_array)
        .context("CTest inventory has no tests array")?;
    let expected = target.ctest_names.iter().cloned().collect::<BTreeSet<_>>();
    let mut observed = BTreeSet::new();
    let mut allowed_programs = BTreeMap::new();
    for artifact_id in &target.artifacts {
        let artifact = manifest.artifact(artifact_id)?;
        let program = std::fs::canonicalize(&artifact.path).with_context(|| {
            format!(
                "failed to resolve CTest artifact {}",
                artifact.path.display()
            )
        })?;
        allowed_programs.insert(program, (artifact_id.clone(), artifact.sha256.clone()));
    }
    let mut normalized = Vec::new();
    for test in tests {
        let name = test
            .get("name")
            .and_then(Value::as_str)
            .context("CTest inventory test has no name")?;
        if !observed.insert(name.to_owned()) {
            bail!("CTest inventory contains duplicate test {name}");
        }
        if test
            .get("config")
            .and_then(Value::as_str)
            .is_some_and(|config| !config.eq_ignore_ascii_case(&manifest.build_type))
        {
            bail!("CTest inventory test {name} uses an unexpected build configuration");
        }
        let command = test
            .get("command")
            .and_then(Value::as_array)
            .context("CTest inventory test has no command")?;
        let program = command
            .first()
            .and_then(Value::as_str)
            .context("CTest inventory test command has no executable")?;
        let program = std::fs::canonicalize(program)
            .with_context(|| format!("failed to resolve CTest program {program}"))?;
        let (artifact_id, artifact_sha256) = allowed_programs.get(&program)
            .with_context(|| format!("CTest {name} resolves to an executable outside the verified target artifact inventory"))?;
        normalized.push(
            json!({ "name": name, "artifact": artifact_id, "binary_sha256": artifact_sha256 }),
        );
    }
    if observed != expected {
        let missing = expected.difference(&observed).cloned().collect::<Vec<_>>();
        let unexpected = observed.difference(&expected).cloned().collect::<Vec<_>>();
        bail!(
            "CTest inventory differs from target declaration; missing={missing:?}, unexpected={unexpected:?}"
        );
    }
    let stored = context
        .store
        .write_json(
            &PathBuf::from(format!("qualification/{stem}/ctest-inventory.json")),
            &normalized,
        )
        .await?;
    Ok(json!({
        "path": path_string(&stored),
        "sha256": stored.sha256,
        "tests": normalized,
    }))
}

struct JunitSummary {
    tests: u64,
    failures: u64,
    errors: u64,
    disabled: u64,
    skipped: u64,
    test_names: BTreeSet<String>,
}

fn parse_junit_summary(bytes: &[u8]) -> anyhow::Result<JunitSummary> {
    let text = std::str::from_utf8(bytes).context("CTest JUnit output is not UTF-8")?;
    let opening = regex::Regex::new(r#"<testsuite\b[^>]*>"#)?
        .find(text)
        .context("CTest JUnit output has no testsuite element")?
        .as_str();
    let attribute = |name: &str| -> anyhow::Result<u64> {
        let pattern = regex::Regex::new(&format!(r#"\b{}="([0-9]+)""#, regex::escape(name)))?;
        Ok(pattern
            .captures(opening)
            .and_then(|captures| captures.get(1))
            .map(|value| value.as_str().parse())
            .transpose()?
            .unwrap_or(0))
    };
    let testcase = regex::Regex::new(r#"<testcase\b[^>]*\bname="([^"]+)"[^>]*>"#)?;
    let mut test_names = BTreeSet::new();
    let mut testcase_count = 0_u64;
    for captures in testcase.captures_iter(text) {
        testcase_count += 1;
        let name = captures.get(1).unwrap().as_str().to_owned();
        if !test_names.insert(name.clone()) {
            bail!("CTest JUnit output repeats testcase {name}");
        }
    }
    let tests = attribute("tests")?;
    if testcase_count != tests {
        bail!("CTest JUnit testcase count {testcase_count} does not match aggregate tests={tests}");
    }
    Ok(JunitSummary {
        tests,
        failures: attribute("failures")?,
        errors: attribute("errors")?,
        disabled: attribute("disabled")?,
        skipped: attribute("skipped")?,
        test_names,
    })
}

async fn decorate_reference_evidence(
    evidence: &mut EvidenceRecord,
    manifest: &ReferenceBuildManifest,
    context: &RunContext<'_>,
    model_artifacts: &[ArtifactInfo],
    fixture_paths: &BTreeMap<String, PathBuf>,
    _linkage_proven: bool,
) -> anyhow::Result<()> {
    evidence.provenance.build = manifest.build_info();
    evidence
        .provenance
        .artifacts
        .extend_from_slice(model_artifacts);
    normalize_model_identity(evidence, model_artifacts);
    append_fixture_artifacts(evidence, fixture_paths).await?;
    evidence
        .provenance
        .effective_configuration
        .insert("backend".to_owned(), json!(manifest.backend));
    evidence
        .provenance
        .effective_configuration
        .insert("build_lane".to_owned(), json!(manifest.lane));
    append_performance_attestations(evidence, context);
    Ok(())
}

async fn decorate_candidate_evidence(
    evidence: &mut EvidenceRecord,
    manifest: &ReferenceBuildManifest,
    context: &RunContext<'_>,
    model_artifacts: &[ArtifactInfo],
    fixture_paths: &BTreeMap<String, PathBuf>,
    producer_id: &str,
) -> anyhow::Result<()> {
    evidence
        .provenance
        .artifacts
        .extend_from_slice(model_artifacts);
    normalize_model_identity(evidence, model_artifacts);
    append_fixture_artifacts(evidence, fixture_paths).await?;
    if let Some(candidate) = context.candidate {
        let artifact = candidate
            .document
            .artifacts
            .iter()
            .find(|artifact| artifact.name == producer_id)
            .with_context(|| {
                format!("candidate manifest has no authorized producer artifact {producer_id}")
            })?;
        if evidence.producer.binary_sha256.as_deref() != Some(artifact.sha256.as_str()) {
            bail!(
                "executed candidate binary digest does not match its authorized manifest artifact"
            );
        }
        evidence.provenance.build = candidate.build_info();
        evidence.producer.version =
            Some(format!("candidate-manifest-sha256:{}", candidate.digest()));
        evidence
            .provenance
            .effective_configuration
            .insert("backend".to_owned(), json!(candidate.document.backend));
        evidence
            .provenance
            .effective_configuration
            .insert("build_lane".to_owned(), json!(candidate.document.lane));
    } else {
        evidence
            .provenance
            .effective_configuration
            .entry("backend".to_owned())
            .or_insert_with(|| {
                json!(
                    context
                        .engine
                        .as_ref()
                        .map(|engine| &engine.backend)
                        .unwrap_or(&manifest.backend)
                )
            });
        evidence
            .provenance
            .effective_configuration
            .entry("build_lane".to_owned())
            .or_insert_with(|| json!(manifest.lane));
        evidence.warnings.push("candidate source trees were inventoried at run time, but no candidate build manifest proves they produced this binary".to_owned());
    }
    append_performance_attestations(evidence, context);
    Ok(())
}

fn normalize_model_identity(evidence: &mut EvidenceRecord, model_artifacts: &[ArtifactInfo]) {
    evidence.provenance.effective_configuration.remove("model");
    evidence
        .provenance
        .effective_configuration
        .remove("model_bytes");
    if let Some(model) = model_artifacts
        .iter()
        .find(|artifact| artifact.kind == "model")
    {
        let id = model
            .id
            .split_once(':')
            .map_or(model.id.as_str(), |(id, _)| id);
        evidence
            .provenance
            .effective_configuration
            .insert("model_id".to_owned(), json!(id));
        evidence
            .provenance
            .effective_configuration
            .insert("model_sha256".to_owned(), json!(model.sha256));
    }
}

async fn append_fixture_artifacts(
    evidence: &mut EvidenceRecord,
    paths: &BTreeMap<String, PathBuf>,
) -> anyhow::Result<()> {
    let mut seen = BTreeSet::new();
    for (id, path) in paths {
        if id.starts_with("fixtures/") || !seen.insert(path.clone()) {
            continue;
        }
        let (sha256, bytes) = sha256_file(path).await?;
        let kind = if id.contains("corpus")
            || path
                .components()
                .any(|component| component.as_os_str() == "corpora")
        {
            "corpus"
        } else {
            "fixture"
        };
        evidence.provenance.artifacts.push(ArtifactInfo {
            kind: kind.to_owned(),
            id: id.clone(),
            sha256,
            bytes,
        });
    }
    Ok(())
}

async fn candidate_source_components(context: &RunContext<'_>) -> anyhow::Result<Vec<Component>> {
    let inference_root = context
        .repository
        .root
        .parent()
        .context("parity root has no inference parent")?;
    if let Some(candidate) = context.candidate {
        return Ok(candidate.provenance_components());
    }
    let mut components = Vec::new();
    for (kind, name, path, exclusions) in [
        (
            "bindings-source",
            "llama-cpp-rs",
            inference_root.join("native/llama-cpp-rs"),
            vec![".git", "target", "llama.cpp"],
        ),
        (
            "icn-source",
            "icn-engine",
            inference_root.join("crates/icn-engine/src"),
            vec![".git", "target"],
        ),
    ] {
        if path.is_dir() {
            let digest = crate::digest::sha256_source_tree(&path, &exclusions).await?;
            components.push(Component {
                kind: kind.to_owned(),
                name: name.to_owned(),
                revision: None,
                tree_sha256: Some(digest.sha256),
                binary_sha256: None,
                dirty: Some(true),
            });
        }
    }
    Ok(components)
}

fn merge_evidence(
    case: &CaseDefinition,
    accumulator: &mut Option<EvidenceRecord>,
    evidence: EvidenceRecord,
) -> anyhow::Result<()> {
    let Some(existing) = accumulator else {
        *accumulator = Some(evidence);
        return Ok(());
    };
    if existing.outcome.class != evidence.outcome.class
        || existing.work != evidence.work
        || existing.provenance.effective_configuration
            != evidence.provenance.effective_configuration
    {
        bail!("paired producer repetitions reported inconsistent outcome/work/configuration");
    }
    if let Comparator::PerformanceRatio {
        exact_output_paths, ..
    } = &case.comparison
    {
        for path in exact_output_paths {
            let existing_value = if path.is_empty() {
                Some(&existing.output)
            } else {
                existing.output.pointer(path)
            };
            let repetition_value = if path.is_empty() {
                Some(&evidence.output)
            } else {
                evidence.output.pointer(path)
            };
            if existing_value != repetition_value {
                bail!(
                    "paired producer repetitions reported inconsistent output at JSON pointer {path:?}"
                );
            }
        }
    }
    for measurement in evidence.measurements {
        let target = existing
            .measurements
            .iter_mut()
            .find(|candidate| {
                candidate.name == measurement.name && candidate.unit == measurement.unit
            })
            .with_context(|| {
                format!(
                    "paired repetition introduced measurement {}",
                    measurement.name
                )
            })?;
        target.samples.extend(measurement.samples);
    }
    existing.warnings.extend(evidence.warnings);
    Ok(())
}

async fn write_evidence_and_raw(
    context: &RunContext<'_>,
    stem: &str,
    role: &str,
    evidence: &EvidenceRecord,
    process: &ProcessOutput,
    repetition: u64,
) -> anyhow::Result<StoredFile> {
    context
        .repository
        .validate_schema(SchemaKind::Evidence, evidence)?;
    write_raw(context, stem, role, process, repetition).await?;
    context
        .store
        .write_json(
            &PathBuf::from(format!("evidence/{stem}/{role}.json")),
            evidence,
        )
        .await
}

async fn write_raw(
    context: &RunContext<'_>,
    stem: &str,
    role: &str,
    process: &ProcessOutput,
    repetition: u64,
) -> anyhow::Result<()> {
    if !context.profile.results.preserve_raw_evidence {
        return Ok(());
    }
    context
        .store
        .write_bytes(
            &PathBuf::from(format!("raw/{stem}/{role}-{repetition:04}.stdout")),
            &process.stdout,
        )
        .await?;
    context
        .store
        .write_bytes(
            &PathBuf::from(format!("raw/{stem}/{role}-{repetition:04}.stderr")),
            &process.stderr,
        )
        .await?;
    Ok(())
}

fn instance_stem(instance: &CaseInstance) -> String {
    let case = case_file_stem(&instance.case.id);
    instance.model_id.as_ref().map_or(case.clone(), |model| {
        format!("{case}--{}", case_file_stem(model))
    })
}

fn path_string(file: &StoredFile) -> String {
    file.relative_path.to_string_lossy().replace('\\', "/")
}

fn role_name(role: ProducerRole) -> &'static str {
    match role {
        ProducerRole::Qualification => "qualification",
        ProducerRole::Reference => "reference",
        ProducerRole::Candidate => "candidate",
    }
}

fn blocked(instance: &CaseInstance, reason: impl Into<String>) -> CaseExecutionResult {
    CaseExecutionResult {
        case_id: instance.case.id.clone(),
        category: instance.case.category,
        model_id: instance.model_id.clone(),
        status: CaseExecutionStatus::Blocked,
        reason: Some(reason.into()),
        reference_evidence: None,
        candidate_evidence: None,
        comparison: None,
    }
}

fn skipped(instance: &CaseInstance, reason: impl Into<String>) -> CaseExecutionResult {
    CaseExecutionResult {
        case_id: instance.case.id.clone(),
        category: instance.case.category,
        model_id: instance.model_id.clone(),
        status: CaseExecutionStatus::Skipped,
        reason: Some(reason.into()),
        reference_evidence: None,
        candidate_evidence: None,
        comparison: None,
    }
}

fn count_results(results: &[CaseExecutionResult]) -> RunCounts {
    let count = |status| {
        results
            .iter()
            .filter(|result| result.status == status)
            .count()
    };
    RunCounts {
        pass: count(CaseExecutionStatus::Pass),
        fail: count(CaseExecutionStatus::Fail),
        invalid: count(CaseExecutionStatus::Invalid),
        skipped: count(CaseExecutionStatus::Skipped),
        blocked: count(CaseExecutionStatus::Blocked),
        error: count(CaseExecutionStatus::Error),
    }
}

fn validate_candidate_reference_identity(
    candidate: &CandidateBuildManifest,
    reference: &ReferenceBuildManifest,
) -> anyhow::Result<()> {
    validate_candidate_reference_digest(
        &candidate.document.reference_manifest_sha256,
        &reference.digest()?,
    )?;
    if candidate.document.backend != reference.backend {
        bail!(
            "candidate backend {} does not match reference backend {}",
            candidate.document.backend,
            reference.backend
        );
    }
    if candidate.document.lane != reference.lane {
        bail!(
            "candidate build lane {} does not match reference lane {}",
            candidate.document.lane,
            reference.lane
        );
    }
    let candidate_native = candidate
        .document
        .components
        .iter()
        .find(|component| component.kind == "native-source" && component.name == "llama.cpp")
        .context("candidate manifest does not bind the llama.cpp native source")?;
    let reference_native = reference
        .native_source
        .as_ref()
        .context("reference manifest does not bind the llama.cpp native source")?;
    if candidate_native.tree_sha256 != reference_native.tree_sha256
        || candidate_native.revision.as_deref() != reference_native.revision.as_deref()
    {
        bail!("candidate and reference manifests bind different llama.cpp source identities");
    }
    Ok(())
}

fn validate_candidate_reference_digest(
    candidate_reference_digest: &str,
    loaded_reference_digest: &str,
) -> anyhow::Result<()> {
    if candidate_reference_digest != loaded_reference_digest {
        bail!(
            "candidate manifest was built against reference manifest {candidate_reference_digest}, not the loaded reference manifest {loaded_reference_digest}"
        );
    }
    Ok(())
}

fn case_consumes_engine(repository: &AssetRepository, case: &CaseDefinition) -> bool {
    let reference_target = case
        .invocations
        .reference
        .as_ref()
        .and_then(|invocation| invocation.target.as_deref());
    if matches!(
        reference_target,
        Some("llama-bench" | "llama-batched-bench" | "llama-perplexity")
    ) {
        return true;
    }
    if case.category == Category::Performance && case.requirements.model.is_some() {
        return true;
    }
    case.invocations
        .candidate
        .as_ref()
        .and_then(|invocation| invocation.target.as_deref())
        .and_then(|target| repository.producers.get(target))
        .is_some_and(|producer| {
            producer
                .args
                .iter()
                .any(|argument| argument.contains("{profile.engine."))
        })
}

fn runtime_engine_for_case<'a>(
    repository: &AssetRepository,
    case: &CaseDefinition,
    engine: Option<&'a ResolvedEngine>,
) -> Option<&'a ResolvedEngine> {
    engine.filter(|_| case_consumes_engine(repository, case))
}

fn unmet_prerequisite(
    instance: &CaseInstance,
    results: &[CaseExecutionResult],
    repository: &AssetRepository,
) -> Option<String> {
    for prerequisite in &instance.case.prerequisites {
        let required = repository.cases.get(prerequisite)?;
        let model_specific = required
            .definition
            .requirements
            .model
            .is_some()
            .then_some(instance.model_id.as_deref())
            .flatten();
        let observed = results
            .iter()
            .filter(|result| {
                result.case_id == *prerequisite
                    && model_specific.is_none_or(|model| result.model_id.as_deref() == Some(model))
            })
            .collect::<Vec<_>>();
        if observed.is_empty() {
            return Some(format!(
                "prerequisite {prerequisite} was not selected/executed{}",
                model_specific.map_or_else(String::new, |model| format!(" for model {model}")),
            ));
        }
        if let Some(result) = observed
            .iter()
            .find(|result| result.status != CaseExecutionStatus::Pass)
        {
            return Some(format!(
                "prerequisite {prerequisite} did not pass{} (status {:?})",
                result
                    .model_id
                    .as_deref()
                    .map_or_else(String::new, |model| format!(" for model {model}")),
                result.status,
            ));
        }
    }
    None
}

fn result_is_gating_failure(result: &CaseExecutionResult, profile: &Profile) -> bool {
    match result.status {
        CaseExecutionStatus::Pass => false,
        CaseExecutionStatus::Fail => {
            result.category == Category::Correctness || profile.gates.fail_on_performance
        }
        CaseExecutionStatus::Invalid => profile.gates.fail_on_invalid,
        CaseExecutionStatus::Skipped => profile.gates.fail_on_skipped,
        CaseExecutionStatus::Blocked | CaseExecutionStatus::Error => true,
    }
}

fn setting_is_true(settings: &BTreeMap<String, String>, name: &str) -> bool {
    settings
        .get(name)
        .is_some_and(|value| matches!(value.as_str(), "true" | "1"))
}

fn append_performance_attestations(evidence: &mut EvidenceRecord, context: &RunContext<'_>) {
    if evidence.category != Category::Performance {
        return;
    }
    evidence.provenance.effective_configuration.insert(
        "controlled_host".to_owned(),
        json!(setting_is_true(
            &context.options.settings,
            "controlled_host"
        )),
    );
    evidence.provenance.effective_configuration.insert(
        "exclusive_device".to_owned(),
        json!(setting_is_true(
            &context.options.settings,
            "exclusive_device"
        )),
    );
}

fn validate_paired_cardinality(
    evidence: &EvidenceRecord,
    repetitions: u64,
    role: &str,
) -> anyhow::Result<()> {
    let expected =
        usize::try_from(repetitions).context("paired repetition count exceeds this platform")?;
    if evidence.measurements.is_empty() {
        bail!("{role} performance evidence contains no measurements");
    }
    for measurement in &evidence.measurements {
        if measurement.samples.len() != expected {
            bail!(
                "{role} measurement {} has {} samples after pairing, expected exactly {expected}",
                measurement.name,
                measurement.samples.len(),
            );
        }
    }
    Ok(())
}

fn artifact_kind_for_role(role: &str) -> &'static str {
    match role {
        "model" => "model",
        "vocab" | "vocabulary" | "tokenizer" => "vocabulary",
        "projector" | "mmproj" => "projector",
        "media" | "image" | "audio" => "media",
        _ => "fixture",
    }
}

fn redacted_settings(settings: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    settings
        .iter()
        .map(|(key, value)| {
            let redacted = if key.to_ascii_lowercase().contains("token")
                || key.to_ascii_lowercase().contains("secret")
            {
                "<redacted>".to_owned()
            } else {
                value.clone()
            };
            (key.clone(), redacted)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        BuildInfo, HostInfo, Measurement, Outcome, Producer, Provenance, WorkDefinition,
    };

    fn repository() -> AssetRepository {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../parity");
        AssetRepository::load(std::fs::canonicalize(root).unwrap()).unwrap()
    }

    #[test]
    fn model_backed_correctness_satisfies_model_free_performance_prerequisite() {
        let repository = repository();
        let case = repository.cases["performance.backend-ops.native-perf-qualification"]
            .definition
            .clone();
        let instance = CaseInstance {
            case,
            model_id: None,
        };
        let results = vec![CaseExecutionResult {
            case_id: "correctness.baseline.focused-upstream-tests".to_owned(),
            category: Category::Correctness,
            model_id: Some("stories15m-q4-0".to_owned()),
            status: CaseExecutionStatus::Pass,
            reason: None,
            reference_evidence: None,
            candidate_evidence: None,
            comparison: None,
        }];
        assert!(unmet_prerequisite(&instance, &results, &repository).is_none());
    }

    #[test]
    fn backend_ops_qualification_does_not_force_full_engine_resolution() {
        let repository = repository();
        let p0 = &repository.cases["performance.backend-ops.native-perf-qualification"].definition;
        assert!(!case_consumes_engine(&repository, p0));
        let p1 = repository
            .cases
            .values()
            .find(|case| case.definition.primitive == "P1")
            .unwrap();
        assert!(case_consumes_engine(&repository, &p1.definition));
    }

    #[test]
    fn runtime_payload_keeps_model_routing_fields_out_of_the_neutral_case() {
        let repository = repository();
        let mut case = repository.cases["correctness.config.context-defaults"]
            .definition
            .clone();
        case.inputs
            .insert("model_id".to_owned(), json!("stories15m-q4-0"));
        let paths = BTreeMap::from([(
            "primary".to_owned(),
            PathBuf::from("/verified-cache/stories.gguf"),
        )]);

        let payload = runtime_input(
            &case,
            Some("stories15m-q4-0"),
            &paths,
            &BTreeMap::new(),
            None,
        )
        .unwrap();

        assert!(!payload.contains_key("model_id"));
        assert_eq!(payload.get("modelId"), Some(&json!("stories15m-q4-0")));
        assert_eq!(
            payload.get("modelPath"),
            Some(&json!(PathBuf::from("/verified-cache/stories.gguf")))
        );
    }

    #[test]
    fn runtime_engine_is_injected_only_for_cases_that_consume_it() {
        let repository = repository();
        let correctness = &repository.cases["correctness.config.context-defaults"].definition;
        let performance = repository
            .cases
            .values()
            .find(|case| case.definition.primitive == "P1")
            .unwrap();
        let engine = ResolvedEngine {
            backend: "metal".to_owned(),
            backend_device: None,
            threads: 12,
            reference_gpu_layers: -2,
            candidate_gpu_layers: u32::MAX,
            flash_attention: "auto".to_owned(),
            cpu_strict: false,
            threadpool_poll: 50,
        };

        assert!(runtime_engine_for_case(&repository, correctness, Some(&engine)).is_none());
        assert!(
            runtime_engine_for_case(&repository, &performance.definition, Some(&engine)).is_some()
        );
    }

    #[test]
    fn paired_invocations_preserve_custom_microbenchmark_work() {
        let repository = repository();
        let p1 = repository
            .cases
            .values()
            .find(|case| case.definition.primitive == "P1")
            .unwrap();
        let p6 = repository
            .cases
            .values()
            .find(|case| case.definition.primitive == "P6")
            .unwrap();

        let llama_bench = paired_invocation_case(&p1.definition);
        assert_eq!(llama_bench.inputs["repetitions"], 1);
        assert_eq!(
            llama_bench.timing.as_ref().unwrap().measurement_iterations,
            1
        );

        let sampler = paired_invocation_case(&p6.definition);
        assert!(!sampler.inputs.contains_key("repetitions"));
        assert_eq!(
            sampler.inputs["iterations"],
            p6.definition.inputs["iterations"]
        );
        assert_eq!(sampler.timing, p6.definition.timing);
    }

    #[test]
    fn protocol_preflight_groups_operations_only_for_the_same_resolved_command() {
        let command = CommandSpec {
            program: PathBuf::from("icn-probe"),
            args: Vec::new(),
            cwd: None,
            env: BTreeMap::new(),
            clear_env: false,
            stdin: StdinKind::ProbeJsonl,
            decoder: DecoderKind::ProbeJsonl,
            timeout_seconds: Some(30),
            max_stdout_bytes: Some(1024),
            max_stderr_bytes: Some(1024),
        };
        let mut targets = Vec::new();
        add_protocol_preflight_target(
            &mut targets,
            ProducerRole::Candidate,
            "icn-probe",
            command.clone(),
            "sampler.apply",
        );
        add_protocol_preflight_target(
            &mut targets,
            ProducerRole::Candidate,
            "icn-probe",
            command.clone(),
            "chat-template.render",
        );
        let mut distinct_command = command;
        distinct_command.args.push("--isolated-mode".to_owned());
        add_protocol_preflight_target(
            &mut targets,
            ProducerRole::Candidate,
            "icn-probe",
            distinct_command,
            "sampler.apply",
        );

        assert_eq!(targets.len(), 2);
        assert_eq!(
            targets[0].required_operations,
            BTreeSet::from([
                "chat-template.render".to_owned(),
                "sampler.apply".to_owned(),
            ])
        );
        assert_eq!(
            targets[1].required_operations,
            BTreeSet::from(["sampler.apply".to_owned()])
        );
    }

    #[test]
    fn candidate_static_capabilities_must_equal_live_protocol_operations() {
        let declared = vec!["sampler.apply".to_owned()];
        let matching = BTreeSet::from([DESCRIBE_OPERATION.to_owned(), "sampler.apply".to_owned()]);
        validate_declared_protocol_operations("candidate", &declared, &matching).unwrap();

        let extra = BTreeSet::from([
            DESCRIBE_OPERATION.to_owned(),
            "sampler.apply".to_owned(),
            "sampler.bench".to_owned(),
        ]);
        assert!(validate_declared_protocol_operations("candidate", &declared, &extra).is_err());
    }

    #[test]
    fn parses_ctest_junit_counts() {
        // This mirrors CTest's aggregate shape: `errors` is commonly absent,
        // while `disabled` and `skipped` are explicit.
        let summary = parse_junit_summary(br#"<?xml version="1.0"?><testsuite name="x" tests="1" failures="0" disabled="0" skipped="0"><testcase name="test-sampling"></testcase></testsuite>"#).unwrap();
        assert_eq!(summary.tests, 1);
        assert_eq!(summary.failures, 0);
        assert_eq!(summary.errors, 0);
        assert_eq!(summary.disabled, 0);
        assert_eq!(summary.skipped, 0);
        assert!(summary.test_names.contains("test-sampling"));
    }

    #[tokio::test]
    async fn ctest_source_residue_is_cleaned_when_invocation_fails() {
        let temporary = tempfile::tempdir().unwrap();
        let source_root = temporary.path().to_path_buf();
        tokio::fs::write(source_root.join(CTEST_SOURCE_RESIDUE_FILES[0]), b"stale")
            .await
            .unwrap();
        tokio::fs::write(source_root.join("unrelated.tmp"), b"keep")
            .await
            .unwrap();
        let operation_root = source_root.clone();

        let result: anyhow::Result<()> = with_ctest_source_cleanup(&source_root, async move {
            assert!(!operation_root.join(CTEST_SOURCE_RESIDUE_FILES[0]).exists());
            for name in CTEST_SOURCE_RESIDUE_FILES {
                tokio::fs::write(operation_root.join(name), b"generated").await?;
            }
            Err(anyhow::anyhow!("synthetic invocation failure"))
        })
        .await;

        assert!(result.is_err());
        for name in CTEST_SOURCE_RESIDUE_FILES {
            assert!(!source_root.join(name).exists());
        }
        assert!(source_root.join("unrelated.tmp").is_file());
        cleanup_ctest_source_residue(&source_root).await.unwrap();
    }

    #[test]
    fn execution_rejects_legacy_reference_manifests() {
        assert!(require_execution_reference_schema(2).is_err());
        assert!(require_execution_reference_schema(3).is_ok());
    }

    #[test]
    fn candidate_must_bind_the_exact_loaded_reference_manifest() {
        let first = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let second = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        assert!(validate_candidate_reference_digest(first, first).is_ok());
        assert!(validate_candidate_reference_digest(first, second).is_err());
    }

    #[test]
    fn backend_requirements_gate_the_verified_reference_backend() {
        assert!(backend_requirement_block(&[], None).is_none());
        let allowed = vec!["cpu".to_owned(), "metal".to_owned()];
        assert!(backend_requirement_block(&allowed, None).is_some());
        assert!(backend_requirement_block(&allowed, Some("metal")).is_none());
        assert!(backend_requirement_block(&allowed, Some("cuda")).is_some());
    }

    #[test]
    fn relative_model_roots_are_resolved_for_producers_with_distinct_working_directories() {
        let current = std::env::current_dir().unwrap();
        let temporary = tempfile::Builder::new()
            .prefix("model-root-test-")
            .tempdir_in(&current)
            .unwrap();
        let relative = temporary.path().strip_prefix(&current).unwrap();
        let resolved = absolute_model_root(relative).unwrap();

        assert!(resolved.is_absolute());
        assert_eq!(resolved, std::fs::canonicalize(temporary.path()).unwrap());
    }

    #[test]
    fn paired_evidence_requires_exact_final_sample_count() {
        let mut evidence = performance_evidence();
        assert!(validate_paired_cardinality(&evidence, 2, "candidate").is_err());
        evidence.measurements[0].samples.push(2.0);
        validate_paired_cardinality(&evidence, 2, "candidate").unwrap();
    }

    #[test]
    fn paired_performance_evidence_rejects_semantic_output_drift() {
        let repository = repository();
        let case = &repository.cases["performance.sampler.top-k-large-vector"].definition;
        let mut first = performance_evidence();
        first.output = json!({"resultTokenIds": [3, 2, 1]});
        let mut second = first.clone();
        second.output = json!({"resultTokenIds": [3, 2, 0]});

        let mut accumulator = None;
        merge_evidence(case, &mut accumulator, first).unwrap();
        let error = merge_evidence(case, &mut accumulator, second).unwrap_err();

        assert!(error.to_string().contains("inconsistent output"));
        assert_eq!(accumulator.unwrap().measurements[0].samples, vec![1.0]);
    }

    #[tokio::test]
    async fn snapshots_the_exact_selected_run_contract() {
        let repository = repository();
        let profile = repository.profile("pr").unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let store = RunStore::create(temporary.path(), "contract-run")
            .await
            .unwrap();

        let records = snapshot_run_contracts(&repository, profile, &store)
            .await
            .unwrap();

        assert!(records.iter().any(|record| {
            record.pointer("/source").and_then(Value::as_str) == Some("profiles/pr.toml")
        }));
        assert!(records.iter().any(|record| {
            record.pointer("/source").and_then(Value::as_str) == Some("models/registry.toml")
        }));
        assert!(records.iter().any(|record| {
            record.pointer("/source").and_then(Value::as_str)
                == Some("cases/correctness/baseline/focused-upstream-tests.json")
        }));
        for record in records {
            let source = record.pointer("/source").and_then(Value::as_str).unwrap();
            let expected = &repository.contract_digests[&repository.root.join(source)];
            assert_eq!(
                record.pointer("/sha256").and_then(Value::as_str),
                Some(expected.as_str())
            );
            let snapshot = record.pointer("/snapshot").and_then(Value::as_str).unwrap();
            assert!(store.root().join(snapshot).is_file());
            assert!(
                record
                    .pointer("/sha256")
                    .and_then(Value::as_str)
                    .is_some_and(|value| value.len() == 64)
            );
        }
    }

    #[tokio::test]
    async fn contract_verification_rejects_mutation_after_load() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("case.json");
        tokio::fs::write(&path, b"parsed contract").await.unwrap();
        let expected = sha256_bytes(b"parsed contract");

        assert_eq!(
            read_contract_matching_digest(&path, &expected)
                .await
                .unwrap(),
            b"parsed contract"
        );
        tokio::fs::write(&path, b"mutated contract").await.unwrap();
        assert!(
            read_contract_matching_digest(&path, &expected)
                .await
                .is_err()
        );
    }

    #[test]
    fn performance_errors_may_omit_measurements_but_success_may_not() {
        let repository = repository();
        let mut evidence = performance_evidence();
        evidence.measurements.clear();
        assert!(
            repository
                .validate_schema(SchemaKind::Evidence, &evidence)
                .is_err()
        );
        evidence.outcome.class = OutcomeClass::RuntimeError;
        assert!(
            repository
                .validate_schema(SchemaKind::Evidence, &evidence)
                .is_ok()
        );
    }

    fn performance_evidence() -> EvidenceRecord {
        EvidenceRecord {
            schema_version: crate::model::SCHEMA_VERSION.to_owned(),
            run_id: "run".to_owned(),
            case_id: "performance.test".to_owned(),
            category: Category::Performance,
            primitive: "P1".to_owned(),
            operation: "test.run".to_owned(),
            recorded_at: "2026-01-01T00:00:00Z".to_owned(),
            producer: Producer {
                role: ProducerRole::Candidate,
                kind: None,
                name: "test".to_owned(),
                version: None,
                binary_sha256: None,
            },
            outcome: Outcome {
                class: OutcomeClass::Success,
                code: None,
                message: None,
            },
            work: WorkDefinition {
                parameters: Map::new(),
                included: Vec::new(),
                excluded: Vec::new(),
                item_count: None,
                plan_sha256: None,
            },
            output: Value::Null,
            measurements: vec![Measurement {
                name: "duration".to_owned(),
                unit: "ns".to_owned(),
                samples: vec![1.0],
            }],
            provenance: Provenance {
                components: vec![Component {
                    kind: "runner".to_owned(),
                    name: "test".to_owned(),
                    revision: None,
                    tree_sha256: None,
                    binary_sha256: None,
                    dirty: None,
                }],
                build: BuildInfo {
                    build_type: "release".to_owned(),
                    compiler: "test".to_owned(),
                    compiler_version: "1".to_owned(),
                    flags: vec!["x".to_owned()],
                    assertions: Some(false),
                    sanitizers: Some(Vec::new()),
                },
                host: HostInfo {
                    os: "test".to_owned(),
                    os_version: None,
                    arch: "test".to_owned(),
                    cpu: "test".to_owned(),
                    logical_cpus: None,
                    memory_bytes: None,
                },
                devices: Vec::new(),
                artifacts: Vec::new(),
                effective_configuration: Map::new(),
                environment_sha256: None,
            },
            warnings: Vec::new(),
        }
    }
}
