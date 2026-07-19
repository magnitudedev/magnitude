use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use futures_util::stream::{FuturesUnordered, StreamExt};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

use crate::client::{ChatRequestPlan, ClientProgress, EndpointClient, output_digest};
use crate::model::{
    BenchmarkComparison, BenchmarkError, BenchmarkRun, BenchmarkRunConfig, ExperimentResult,
    FixtureCatalog, HostEvidence, MemoryEvidence, MetricComparison, ProgressEvent,
    RequestObservation, RunOutcome, SUITE_VERSION, ScenarioSample, TargetResult,
    ToolCallObservation, WorkSignature,
};
use crate::report::{comparison_markdown, run_markdown};
use crate::stats::{paired_ratio_interval, summarize_samples};
use crate::suite::{load_fixtures, load_profile};

pub type ProgressCallback = Arc<dyn Fn(ProgressEvent) + Send + Sync>;

pub fn compare_evidence(
    candidate: &BenchmarkRun,
    reference: &BenchmarkRun,
) -> Result<BenchmarkComparison, BenchmarkError> {
    if candidate.suite_version != reference.suite_version
        || candidate.fixtures_sha256 != reference.fixtures_sha256
        || serde_json::to_value(&candidate.profile)? != serde_json::to_value(&reference.profile)?
    {
        return Err(BenchmarkError::InvalidConfig(
            "evidence uses different suite, fixture, or profile definitions".into(),
        ));
    }
    if candidate.targets.len() != 1 || reference.targets.len() != 1 {
        return Err(BenchmarkError::InvalidConfig(
            "evidence comparison requires one standalone target in each run".into(),
        ));
    }
    let joined = BenchmarkRun {
        schema_version: 1,
        suite_version: candidate.suite_version.clone(),
        run_id: format!("{}-vs-{}", candidate.run_id, reference.run_id),
        recorded_at_unix_ms: candidate
            .recorded_at_unix_ms
            .max(reference.recorded_at_unix_ms),
        profile: candidate.profile.clone(),
        fixtures_sha256: candidate.fixtures_sha256.clone(),
        host: candidate.host.clone(),
        targets: vec![candidate.targets[0].clone(), reference.targets[0].clone()],
    };
    let mut comparison = compare_run(&joined);
    comparison.warnings.push(format!(
        "joined standalone evidence {} and {}",
        candidate.run_id, reference.run_id
    ));
    Ok(comparison)
}

pub struct BenchmarkRunner {
    progress: Option<ProgressCallback>,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

fn selected_experiments(profile: &crate::model::Profile) -> Vec<String> {
    let mut experiments = Vec::new();
    for case in &profile.cases {
        let experiment = case
            .split_once('/')
            .expect("profile validation requires EXPERIMENT/ARM syntax")
            .0;
        if !experiments.iter().any(|selected| selected == experiment) {
            experiments.push(experiment.to_owned());
        }
    }
    experiments
}

fn case_selected(profile: &crate::model::Profile, experiment: &str, arm: &str) -> bool {
    profile.cases.iter().any(|case| {
        case.split_once('/')
            .is_some_and(|(selected_experiment, selected_arm)| {
                selected_experiment == experiment && (selected_arm == "*" || selected_arm == arm)
            })
    })
}

fn e2_concurrency(profile: &crate::model::Profile) -> Vec<usize> {
    if profile.cases.iter().any(|case| case == "E2/*") {
        return profile.concurrency.clone();
    }
    profile
        .cases
        .iter()
        .filter_map(|case| {
            let (experiment, arm) = case.split_once('/')?;
            (experiment == "E2")
                .then(|| arm.rsplit_once(".c")?.1.parse::<usize>().ok())
                .flatten()
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

impl BenchmarkRunner {
    #[must_use]
    pub const fn new() -> Self {
        Self { progress: None }
    }

    #[must_use]
    pub fn with_progress(mut self, progress: ProgressCallback) -> Self {
        self.progress = Some(progress);
        self
    }

    pub async fn run(
        &self,
        config: BenchmarkRunConfig,
    ) -> Result<(BenchmarkRun, Option<BenchmarkComparison>), BenchmarkError> {
        if config.targets.is_empty() || config.targets.len() > 2 {
            return Err(BenchmarkError::InvalidConfig(
                "a run requires one or two targets".into(),
            ));
        }
        let mut profile = load_profile(&config.root, &config.profile).await?;
        let (fixtures, fixtures_sha256) = load_fixtures(&config.root).await?;
        if profile.paired && config.targets.len() != 2 {
            return Err(BenchmarkError::InvalidConfig(
                "a paired profile requires exactly two targets".into(),
            ));
        }
        if profile.controlled
            && (!config.controlled_host
                || (profile.require_exclusive_device && !config.exclusive_device))
        {
            return Err(BenchmarkError::InvalidConfig(
                "the controlled profile requires --controlled-host and --exclusive-device attestations"
                    .into(),
            ));
        }
        if profile.require_exclusive_device && !config.exclusive_device {
            return Err(BenchmarkError::InvalidConfig(format!(
                "profile {} requires --exclusive-device",
                profile.id
            )));
        }
        let run_id = config.run_id.unwrap_or_else(default_run_id);
        self.emit(ProgressEvent::RunStarted {
            run_id: run_id.clone(),
        });

        let target_configs = config.targets;
        let clients = target_configs
            .iter()
            .cloned()
            .map(|target| EndpointClient::new(target, profile.request_timeout_seconds))
            .collect::<Result<Vec<_>, _>>()?;
        let targets = if profile.paired {
            let mut accumulated = vec![None, None];
            let mut completed_repetitions = 0;
            let maximum = if profile.controlled {
                profile.max_paired_repetitions
            } else {
                profile.repetitions
            };
            for block in 0..maximum {
                let order = if block % 2 == 0 { [0, 1] } else { [1, 0] };
                let mut block_profile = profile.clone();
                block_profile.repetitions = 1;
                block_profile.warmups = usize::from(block == 0) * profile.warmups;
                for index in order {
                    let client = &clients[index];
                    self.emit(ProgressEvent::TargetStarted {
                        target: client.target().name.clone(),
                    });
                    let result = self
                        .run_target(client, &block_profile, &fixtures, block)
                        .await;
                    merge_target_result(&mut accumulated[index], result);
                    self.emit(ProgressEvent::TargetCompleted {
                        target: client.target().name.clone(),
                    });
                }
                completed_repetitions = block + 1;
                if profile.controlled && completed_repetitions >= profile.min_paired_repetitions {
                    let interim_targets = accumulated
                        .iter()
                        .filter_map(Clone::clone)
                        .collect::<Vec<_>>();
                    let interim = benchmark_run(
                        run_id.clone(),
                        profile.clone(),
                        fixtures_sha256.clone(),
                        interim_targets,
                        config.controlled_host,
                        config.exclusive_device,
                    );
                    let comparison = compare_run(&interim);
                    let primary = comparison
                        .metrics
                        .iter()
                        .filter(|metric| is_primary_metric(&metric.experiment, &metric.metric))
                        .collect::<Vec<_>>();
                    if !primary.is_empty() && primary.iter().all(|metric| metric.stable) {
                        break;
                    }
                }
            }
            profile.repetitions = completed_repetitions;
            accumulated
                .into_iter()
                .map(|result| result.expect("both paired targets execute each block"))
                .collect()
        } else {
            let mut results = Vec::with_capacity(clients.len());
            for client in &clients {
                self.emit(ProgressEvent::TargetStarted {
                    target: client.target().name.clone(),
                });
                let result = self.run_target(client, &profile, &fixtures, 0).await;
                self.emit(ProgressEvent::TargetCompleted {
                    target: client.target().name.clone(),
                });
                results.push(result);
            }
            results
        };

        let run = benchmark_run(
            run_id.clone(),
            profile,
            fixtures_sha256,
            targets,
            config.controlled_host,
            config.exclusive_device,
        );
        let comparison = (run.targets.len() == 2).then(|| compare_run(&run));
        persist(&config.output_dir, &run, comparison.as_ref()).await?;
        self.emit(ProgressEvent::RunCompleted { run_id });
        Ok((run, comparison))
    }

    async fn run_target(
        &self,
        client: &EndpointClient,
        profile: &crate::model::Profile,
        fixtures: &FixtureCatalog,
        repetition_offset: usize,
    ) -> TargetResult {
        let memory_monitor = MemoryMonitor::start(client.target().process_id).await;
        let probe = client.probe().await;
        let mut experiments = Vec::new();
        for id in selected_experiments(profile) {
            self.emit(ProgressEvent::ExperimentStarted {
                target: client.target().name.clone(),
                experiment: id.clone(),
            });
            let result = match id.as_str() {
                "E1" => self.e1(client, profile, fixtures, repetition_offset).await,
                "E2" => self.e2(client, profile, fixtures, repetition_offset).await,
                "E3" => self.e3(client, profile, fixtures, repetition_offset).await,
                "E4" => self.e4(client, profile, fixtures, repetition_offset).await,
                "E5" => self.e5(client, profile, fixtures, repetition_offset).await,
                "E6" => self.e6(client, profile, fixtures, repetition_offset).await,
                "E7" => self.e7(client, profile, fixtures, repetition_offset).await,
                _ => unreachable!("profile validation rejects unknown experiments"),
            };
            experiments.push(result);
        }
        let memory = match memory_monitor {
            Some(monitor) => monitor.finish().await,
            None => MemoryEvidence::default(),
        };
        TargetResult {
            target: client.target().clone(),
            probe,
            experiments,
            memory,
            warnings: Vec::new(),
        }
    }

    async fn e1(
        &self,
        client: &EndpointClient,
        profile: &crate::model::Profile,
        fixtures: &FixtureCatalog,
        repetition_offset: usize,
    ) -> ExperimentResult {
        self.warmup(client, profile, fixtures, "E1").await;
        let arms = [
            (
                "ss",
                fixtures.prompt_short_tokens,
                fixtures.output_short_tokens,
            ),
            (
                "ls",
                fixtures.prompt_long_tokens,
                fixtures.output_short_tokens,
            ),
            (
                "sl",
                fixtures.prompt_short_tokens,
                fixtures.output_long_tokens,
            ),
            (
                "ll",
                fixtures.prompt_long_tokens,
                fixtures.output_long_tokens,
            ),
        ];
        let mut samples = Vec::new();
        for local_repetition in 0..profile.repetitions {
            let repetition = repetition_offset + local_repetition;
            for (arm, prompt, output) in arms {
                if !case_selected(profile, "E1", arm) {
                    continue;
                }
                let plan = load_plan(
                    fixtures,
                    format!("E1-{arm}-{repetition}"),
                    prompt,
                    output,
                    arm,
                );
                let started = Instant::now();
                let request = client.execute(plan, None).await;
                let sample = single_sample(
                    repetition,
                    arm,
                    request,
                    output,
                    answer_for(fixtures, output),
                    started.elapsed(),
                );
                self.sample_progress(client, "E1", &sample);
                samples.push(sample);
            }
        }
        experiment(
            "E1",
            "Single-request phase surface",
            "What are the fixed, prefill, decode, and context-depth characteristics?",
            samples,
            Vec::new(),
        )
    }

    async fn e2(
        &self,
        client: &EndpointClient,
        profile: &crate::model::Profile,
        fixtures: &FixtureCatalog,
        repetition_offset: usize,
    ) -> ExperimentResult {
        self.warmup(client, profile, fixtures, "E2").await;
        let phases = [
            (
                "prefill",
                fixtures.prompt_long_tokens,
                fixtures.output_short_tokens,
            ),
            (
                "decode",
                fixtures.prompt_short_tokens,
                fixtures.output_long_tokens,
            ),
        ];
        let mut samples = Vec::new();
        for local_repetition in 0..profile.repetitions {
            let repetition = repetition_offset + local_repetition;
            for (phase, prompt, output) in phases {
                for concurrency in e2_concurrency(profile) {
                    let arm = format!("{phase}.c{concurrency}");
                    if !case_selected(profile, "E2", &arm) {
                        continue;
                    }
                    let count = concurrency.saturating_mul(profile.closed_loop_multiplier);
                    let plans = (0..count)
                        .map(|index| {
                            load_plan(
                                fixtures,
                                format!("E2-{phase}-c{concurrency}-r{repetition}-q{index}"),
                                prompt,
                                output,
                                &format!("{phase}-{index:04}"),
                            )
                        })
                        .collect();
                    let sample = population_sample(
                        client,
                        repetition,
                        &arm,
                        concurrency,
                        plans,
                        output,
                        answer_for(fixtures, output),
                    )
                    .await;
                    self.sample_progress(client, "E2", &sample);
                    samples.push(sample);
                }
            }
        }
        experiment(
            "E2",
            "Phase-specific concurrency",
            "How efficiently does the endpoint continuously batch prefill-heavy and decode-heavy work?",
            samples,
            Vec::new(),
        )
    }

    async fn e3(
        &self,
        client: &EndpointClient,
        profile: &crate::model::Profile,
        fixtures: &FixtureCatalog,
        repetition_offset: usize,
    ) -> ExperimentResult {
        self.warmup(client, profile, fixtures, "E3").await;
        let mut samples = Vec::new();
        if !case_selected(profile, "E3", "prefill-arrives-during-decode") {
            return experiment(
                "E3",
                "Mixed prefill/decode interference",
                "What happens when long prefill is introduced during active decode?",
                samples,
                Vec::new(),
            );
        }
        for local_repetition in 0..profile.repetitions {
            let repetition = repetition_offset + local_repetition;
            let a_plan = load_plan(
                fixtures,
                format!("E3-A-{repetition}"),
                fixtures.prompt_short_tokens,
                fixtures.output_long_tokens,
                "mixed-a",
            );
            let b_plan = load_plan(
                fixtures,
                format!("E3-B-{repetition}"),
                fixtures.prompt_long_tokens,
                fixtures.output_short_tokens,
                "mixed-b",
            );
            let (progress_tx, mut progress_rx) = mpsc::unbounded_channel();
            let a_client = client.clone();
            let started = Instant::now();
            let a = tokio::spawn(async move { a_client.execute(a_plan, Some(progress_tx)).await });
            let aligned = tokio::time::timeout(
                std::time::Duration::from_secs(profile.request_timeout_seconds),
                async {
                    while let Some(event) = progress_rx.recv().await {
                        if matches!(event, ClientProgress::FirstSemantic) {
                            return true;
                        }
                    }
                    false
                },
            )
            .await
            .unwrap_or(false);
            let b = client.execute(b_plan, None).await;
            let a = match a.await {
                Ok(value) => value,
                Err(error) => task_error("E3-A", error),
            };
            let makespan = started.elapsed();
            let mut warnings = Vec::new();
            if !aligned {
                warnings.push("request B could not be aligned to A's first semantic output".into());
            }
            let mut metrics = BTreeMap::new();
            insert_request_metrics(&mut metrics, "a", &a);
            insert_request_metrics(&mut metrics, "b", &b);
            insert_goodput(&mut metrics, "total", &[a.clone(), b.clone()], makespan);
            let outcome = if aligned {
                validate_requests(
                    &[a.clone(), b.clone()],
                    &[fixtures.output_long_tokens, fixtures.output_short_tokens],
                    &[&fixtures.answer_long, &fixtures.answer_short],
                )
            } else {
                RunOutcome::Invalid
            };
            let sample = ScenarioSample {
                repetition,
                arm: "prefill-arrives-during-decode".into(),
                outcome,
                makespan_ms: makespan.as_secs_f64() * 1_000.0,
                metrics: prefix_metrics("mixed", metrics),
                requests: vec![a, b],
                warnings,
            };
            self.sample_progress(client, "E3", &sample);
            samples.push(sample);
        }
        experiment(
            "E3",
            "Mixed prefill/decode interference",
            "What happens when long prefill is introduced during active decode?",
            samples,
            Vec::new(),
        )
    }

    async fn e4(
        &self,
        client: &EndpointClient,
        profile: &crate::model::Profile,
        fixtures: &FixtureCatalog,
        repetition_offset: usize,
    ) -> ExperimentResult {
        let mut samples = Vec::new();
        let arms = ["exact", "partial", "unrelated"];
        for local_repetition in 0..profile.repetitions {
            let repetition = repetition_offset + local_repetition;
            for arm in arms {
                if !case_selected(profile, "E4", arm) {
                    continue;
                }
                let mut warnings = Vec::new();
                match client.reset_cache().await {
                    Ok(true) => {}
                    Ok(false) => warnings.push(
                        "target cannot explicitly reset cache; result is observational".into(),
                    ),
                    Err(error) => warnings.push(format!("cache reset failed: {error}")),
                }
                let (first_variant, second_variant) = match arm {
                    "exact" => ("prefix-a", "prefix-a"),
                    "partial" => ("prefix-a-tail-a", "prefix-a-tail-b"),
                    "unrelated" => ("unrelated-u", "unrelated-v"),
                    _ => unreachable!(),
                };
                let first = cache_plan(
                    fixtures,
                    format!("E4-{arm}-first-{repetition}"),
                    first_variant,
                    arm != "unrelated",
                );
                let second = cache_plan(
                    fixtures,
                    format!("E4-{arm}-second-{repetition}"),
                    second_variant,
                    arm != "unrelated",
                );
                let started = Instant::now();
                let first = client.execute(first, None).await;
                let second = client.execute(second, None).await;
                let makespan = started.elapsed();
                let outcome = validate_requests(
                    &[first.clone(), second.clone()],
                    &[fixtures.output_short_tokens, fixtures.output_short_tokens],
                    &[&fixtures.answer_short, &fixtures.answer_short],
                );
                let mut metrics = BTreeMap::new();
                insert_request_metrics(&mut metrics, "first", &first);
                insert_request_metrics(&mut metrics, "second", &second);
                if let (Some(total), Some(cache)) = (
                    prompt_total(&second),
                    second.timings.as_ref().and_then(|timings| timings.cache_n),
                ) {
                    metrics.insert("second.cache_fraction".into(), ratio(cache, total));
                }
                let sample = ScenarioSample {
                    repetition,
                    arm: arm.into(),
                    outcome,
                    makespan_ms: makespan.as_secs_f64() * 1_000.0,
                    metrics: prefix_metrics(arm, metrics),
                    requests: vec![first, second],
                    warnings,
                };
                self.sample_progress(client, "E4", &sample);
                samples.push(sample);
            }
        }
        experiment(
            "E4",
            "Sequential prefix reuse",
            "Does a completed request make exact or partial-prefix requests cheaper?",
            samples,
            Vec::new(),
        )
    }

    async fn e5(
        &self,
        client: &EndpointClient,
        profile: &crate::model::Profile,
        fixtures: &FixtureCatalog,
        repetition_offset: usize,
    ) -> ExperimentResult {
        let mut samples = Vec::new();
        for local_repetition in 0..profile.repetitions {
            let repetition = repetition_offset + local_repetition;
            for shared in [true, false] {
                let arm = if shared { "shared" } else { "independent" };
                if !case_selected(profile, "E5", arm) {
                    continue;
                }
                let mut warnings = Vec::new();
                match client.reset_cache().await {
                    Ok(true) => {}
                    Ok(false) => warnings.push(
                        "target cannot explicitly reset cache; result is observational".into(),
                    ),
                    Err(error) => warnings.push(format!("cache reset failed: {error}")),
                }
                let plans = (0..4)
                    .map(|index| {
                        fanout_plan(
                            fixtures,
                            format!("E5-{arm}-{repetition}-{index}"),
                            shared,
                            index,
                        )
                    })
                    .collect();
                let mut sample = population_sample(
                    client,
                    repetition,
                    arm,
                    4,
                    plans,
                    fixtures.output_short_tokens,
                    &fixtures.answer_short,
                )
                .await;
                sample.warnings.extend(warnings);
                self.sample_progress(client, "E5", &sample);
                samples.push(sample);
            }
        }
        experiment(
            "E5",
            "Concurrent shared-prefix fan-out",
            "Can the endpoint avoid or efficiently share prompt work across resident requests?",
            samples,
            Vec::new(),
        )
    }

    async fn e6(
        &self,
        client: &EndpointClient,
        profile: &crate::model::Profile,
        fixtures: &FixtureCatalog,
        repetition_offset: usize,
    ) -> ExperimentResult {
        self.warmup(client, profile, fixtures, "E6").await;
        let mut samples = Vec::new();
        if !case_selected(profile, "E6", "forced-edit") {
            return experiment(
                "E6",
                "Deterministic tool transaction",
                "Do phase results transfer to the complete tool-call and multi-turn path?",
                samples,
                Vec::new(),
            );
        }
        for local_repetition in 0..profile.repetitions {
            let repetition = repetition_offset + local_repetition;
            let tool = &fixtures.tool_fixture;
            let first_plan = tool_plan(fixtures, format!("E6-tool-{repetition}"));
            let started = Instant::now();
            let first = client.execute(first_plan, None).await;
            let semantic_tool_match = validate_tool_call(&first.tool_calls, tool);
            let second_plan =
                tool_followup_plan(fixtures, format!("E6-final-{repetition}"), &first);
            let second = client.execute(second_plan, None).await;
            let makespan = started.elapsed();
            let acknowledgement_match = second.content.trim() == tool.final_acknowledgement;
            let outcome = if matches!(first.outcome, RunOutcome::Valid)
                && matches!(second.outcome, RunOutcome::Valid)
                && semantic_tool_match
                && acknowledgement_match
            {
                RunOutcome::Valid
            } else if matches!(first.outcome, RunOutcome::Error)
                || matches!(second.outcome, RunOutcome::Error)
            {
                RunOutcome::Unsupported
            } else {
                RunOutcome::Invalid
            };
            let mut metrics = BTreeMap::new();
            insert_request_metrics(&mut metrics, "tool", &first);
            insert_request_metrics(&mut metrics, "final", &second);
            metrics.insert("transaction_ms".into(), makespan.as_secs_f64() * 1_000.0);
            let mut warnings = Vec::new();
            if !semantic_tool_match {
                warnings.push("tool call did not match the deterministic edit contract".into());
            }
            if !acknowledgement_match {
                warnings.push("final acknowledgement did not match the fixture contract".into());
            }
            let sample = ScenarioSample {
                repetition,
                arm: "forced-edit".into(),
                outcome,
                makespan_ms: makespan.as_secs_f64() * 1_000.0,
                metrics,
                requests: vec![first, second],
                warnings,
            };
            self.sample_progress(client, "E6", &sample);
            samples.push(sample);
        }
        experiment(
            "E6",
            "Deterministic tool transaction",
            "Do phase results transfer to the complete tool-call and multi-turn path?",
            samples,
            Vec::new(),
        )
    }

    async fn e7(
        &self,
        client: &EndpointClient,
        profile: &crate::model::Profile,
        fixtures: &FixtureCatalog,
        repetition_offset: usize,
    ) -> ExperimentResult {
        self.warmup(client, profile, fixtures, "E7").await;
        let mut samples = Vec::new();
        if !case_selected(profile, "E7", "decode-cancel-recovery") {
            return experiment(
                "E7",
                "Cancellation and recovery",
                "Does abandoned decode work release capacity for the next request?",
                samples,
                Vec::new(),
            );
        }
        for local_repetition in 0..profile.repetitions {
            let repetition = repetition_offset + local_repetition;
            let mut cancel_plan = load_plan(
                fixtures,
                format!("E7-cancel-{repetition}"),
                fixtures.prompt_short_tokens,
                fixtures.output_long_tokens,
                "cancel",
            );
            cancel_plan.timings_per_token = true;
            let (tx, mut rx) = mpsc::unbounded_channel();
            let cancel_client = client.clone();
            let handle =
                tokio::spawn(async move { cancel_client.execute(cancel_plan, Some(tx)).await });
            let mut progress_kind = "none";
            let observed = tokio::time::timeout(
                std::time::Duration::from_secs(profile.request_timeout_seconds),
                async {
                    while let Some(event) = rx.recv().await {
                        match event {
                            ClientProgress::Predicted(count) if count >= 8 => return "predicted",
                            ClientProgress::FirstSemantic => progress_kind = "semantic",
                            _ => {}
                        }
                    }
                    progress_kind
                },
            )
            .await
            .unwrap_or("none");
            handle.abort();
            let _ = handle.await;
            let recovery_plan = load_plan(
                fixtures,
                format!("E7-recovery-{repetition}"),
                fixtures.prompt_short_tokens,
                fixtures.output_short_tokens,
                "recovery",
            );
            let started = Instant::now();
            let recovery = client.execute(recovery_plan, None).await;
            let makespan = started.elapsed();
            let strict_progress = observed == "predicted";
            let outcome = if matches!(recovery.outcome, RunOutcome::Valid)
                && observed != "none"
                && completion_tokens(&recovery) == Some(fixtures.output_short_tokens as u64)
                && recovery.content == fixtures.answer_short
            {
                RunOutcome::Valid
            } else {
                RunOutcome::Invalid
            };
            let mut metrics = BTreeMap::new();
            insert_request_metrics(&mut metrics, "recovery", &recovery);
            let mut warnings = Vec::new();
            if !strict_progress {
                warnings.push("cancellation used semantic-event fallback rather than generated-token progress".into());
            }
            let cancelled = cancelled_observation(format!("E7-cancel-{repetition}"), observed);
            let sample = ScenarioSample {
                repetition,
                arm: "decode-cancel-recovery".into(),
                outcome,
                makespan_ms: makespan.as_secs_f64() * 1_000.0,
                metrics,
                requests: vec![cancelled, recovery],
                warnings,
            };
            self.sample_progress(client, "E7", &sample);
            samples.push(sample);
        }
        experiment(
            "E7",
            "Cancellation and recovery",
            "Does abandoned decode work release capacity for the next request?",
            samples,
            Vec::new(),
        )
    }

    async fn warmup(
        &self,
        client: &EndpointClient,
        profile: &crate::model::Profile,
        fixtures: &FixtureCatalog,
        experiment: &str,
    ) {
        for index in 0..profile.warmups {
            let plan = load_plan(
                fixtures,
                format!("warmup-{experiment}-{index}"),
                fixtures.prompt_short_tokens,
                fixtures.output_short_tokens,
                &format!("warmup-{experiment}-{index}"),
            );
            let _ = client.execute(plan, None).await;
        }
    }

    fn sample_progress(&self, client: &EndpointClient, experiment: &str, sample: &ScenarioSample) {
        self.emit(ProgressEvent::SampleCompleted {
            target: client.target().name.clone(),
            experiment: experiment.into(),
            arm: sample.arm.clone(),
            repetition: sample.repetition,
        });
    }

    fn emit(&self, event: ProgressEvent) {
        if let Some(progress) = &self.progress {
            progress(event);
        }
    }
}

fn load_plan(
    fixtures: &FixtureCatalog,
    id: String,
    prompt_tokens: usize,
    output_tokens: u32,
    variant: &str,
) -> ChatRequestPlan {
    let answer = answer_for(fixtures, output_tokens);
    let prompt = carrier_prompt(fixtures, prompt_tokens, variant, None, answer);
    ChatRequestPlan {
        id,
        messages: vec![json!({"role": "user", "content": prompt})],
        tools: None,
        tool_choice: None,
        max_tokens: output_tokens,
        cache_prompt: false,
        ignore_eos: true,
        timings_per_token: false,
        slot_id: None,
    }
}

fn cache_plan(
    fixtures: &FixtureCatalog,
    id: String,
    variant: &str,
    shared: bool,
) -> ChatRequestPlan {
    let prefix = shared.then_some("shared-prefix");
    let prompt = carrier_prompt(
        fixtures,
        fixtures.prompt_long_tokens,
        variant,
        prefix,
        &fixtures.answer_short,
    );
    ChatRequestPlan {
        id,
        messages: vec![json!({"role": "user", "content": prompt})],
        tools: None,
        tool_choice: None,
        max_tokens: fixtures.output_short_tokens,
        cache_prompt: true,
        ignore_eos: true,
        timings_per_token: false,
        slot_id: Some(0),
    }
}

fn fanout_plan(
    fixtures: &FixtureCatalog,
    id: String,
    shared: bool,
    index: usize,
) -> ChatRequestPlan {
    let variant = if shared {
        format!("shared-tail-{index}")
    } else {
        format!("independent-{index}")
    };
    let prefix = shared.then_some("fanout-shared-prefix");
    let prompt = carrier_prompt(
        fixtures,
        fixtures.prompt_long_tokens,
        &variant,
        prefix,
        &fixtures.answer_short,
    );
    ChatRequestPlan {
        id,
        messages: vec![json!({"role": "user", "content": prompt})],
        tools: None,
        tool_choice: None,
        max_tokens: fixtures.output_short_tokens,
        cache_prompt: true,
        ignore_eos: true,
        timings_per_token: false,
        slot_id: None,
    }
}

fn carrier_prompt(
    fixtures: &FixtureCatalog,
    approximate_tokens: usize,
    variant: &str,
    shared_prefix: Option<&str>,
    answer: &str,
) -> String {
    let target_chars = approximate_tokens.saturating_mul(4);
    let early = shared_prefix.unwrap_or(variant);
    let mut prompt = format!("BENCHMARK-ID:{early}\nBEGIN-CONTEXT\n");
    while prompt.len() + fixtures.carrier_block.len() + answer.len() + 256 < target_chars {
        prompt.push_str(&fixtures.carrier_block);
        prompt.push('\n');
    }
    prompt.push_str("END-CONTEXT\nREQUEST-VARIANT:");
    prompt.push_str(variant);
    prompt.push('\n');
    prompt.push_str("ANSWER_BLOCK_BEGIN\n");
    prompt.push_str(answer);
    prompt.push_str("\nANSWER_BLOCK_END\n");
    prompt.push_str(
        "Copy exactly the bytes between ANSWER_BLOCK_BEGIN and ANSWER_BLOCK_END. Output no marker \
         and no additional text.",
    );
    prompt
}

fn answer_for(fixtures: &FixtureCatalog, output_tokens: u32) -> &str {
    if output_tokens == fixtures.output_long_tokens {
        &fixtures.answer_long
    } else {
        &fixtures.answer_short
    }
}

fn tool_plan(fixtures: &FixtureCatalog, id: String) -> ChatRequestPlan {
    let tool = &fixtures.tool_fixture;
    let parameters = json!({
        "type": "object",
        "properties": {
            "path": {"type": "string", "enum": [tool.path]},
            "old_text": {"type": "string", "enum": [tool.old_text]},
            "new_text": {"type": "string", "enum": [tool.new_text]}
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": false
    });
    let prompt = format!(
        "Edit the following file by replacing the one exact occurrence of the old text with the new text. Use the required tool exactly once.\nPATH: {}\nFILE:\n{}\nOLD: {}\nNEW: {}",
        tool.path, tool.before, tool.old_text, tool.new_text
    );
    ChatRequestPlan {
        id,
        messages: vec![json!({"role": "user", "content": prompt})],
        tools: Some(vec![json!({
            "type": "function",
            "function": {
                "name": tool.tool_name,
                "description": "Apply the single requested deterministic text replacement.",
                "parameters": parameters
            }
        })]),
        tool_choice: Some(json!({"type": "function", "function": {"name": tool.tool_name}})),
        max_tokens: 128,
        cache_prompt: false,
        ignore_eos: false,
        timings_per_token: false,
        slot_id: None,
    }
}

fn tool_followup_plan(
    fixtures: &FixtureCatalog,
    id: String,
    first: &RequestObservation,
) -> ChatRequestPlan {
    let tool = &fixtures.tool_fixture;
    let observed = first
        .tool_calls
        .first()
        .cloned()
        .unwrap_or(ToolCallObservation {
            index: 0,
            id: "benchmark-call".into(),
            name: tool.tool_name.clone(),
            arguments: serde_json::to_string(&json!({
                "path": tool.path,
                "old_text": tool.old_text,
                "new_text": tool.new_text
            }))
            .unwrap_or_default(),
        });
    let call_id = if observed.id.is_empty() {
        "benchmark-call".to_owned()
    } else {
        observed.id.clone()
    };
    let first_prompt = format!(
        "Edit the following file by replacing the one exact occurrence of the old text with the new text. Use the required tool exactly once.\nPATH: {}\nFILE:\n{}\nOLD: {}\nNEW: {}",
        tool.path, tool.before, tool.old_text, tool.new_text
    );
    ChatRequestPlan {
        id,
        messages: vec![
            json!({"role": "user", "content": first_prompt}),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": observed.name, "arguments": observed.arguments}
                }]
            }),
            json!({"role": "tool", "tool_call_id": call_id, "content": tool.tool_result}),
            json!({"role": "user", "content": format!("Reply with exactly: {}", tool.final_acknowledgement)}),
        ],
        tools: None,
        tool_choice: None,
        max_tokens: 32,
        cache_prompt: false,
        ignore_eos: false,
        timings_per_token: false,
        slot_id: None,
    }
}

fn validate_tool_call(calls: &[ToolCallObservation], tool: &crate::model::ToolFixture) -> bool {
    if calls.len() != 1 || calls[0].name != tool.tool_name {
        return false;
    }
    let Ok(arguments) = serde_json::from_str::<Value>(&calls[0].arguments) else {
        return false;
    };
    arguments
        == json!({
            "path": tool.path,
            "old_text": tool.old_text,
            "new_text": tool.new_text
        })
}

async fn population_sample(
    client: &EndpointClient,
    repetition: usize,
    arm: &str,
    concurrency: usize,
    plans: Vec<ChatRequestPlan>,
    expected_output: u32,
    expected_answer: &str,
) -> ScenarioSample {
    let started = Instant::now();
    let mut pending = plans.into_iter();
    let mut active =
        FuturesUnordered::<Pin<Box<dyn Future<Output = RequestObservation> + Send>>>::new();
    for _ in 0..concurrency {
        if let Some(plan) = pending.next() {
            let client = client.clone();
            active.push(Box::pin(async move { client.execute(plan, None).await }));
        }
    }
    let mut requests = Vec::new();
    while let Some(request) = active.next().await {
        requests.push(request);
        if let Some(plan) = pending.next() {
            let client = client.clone();
            active.push(Box::pin(async move { client.execute(plan, None).await }));
        }
    }
    let makespan = started.elapsed();
    let expected = vec![expected_output; requests.len()];
    let answers = vec![expected_answer; requests.len()];
    let outcome = validate_requests(&requests, &expected, &answers);
    let mut metrics = BTreeMap::new();
    insert_goodput(&mut metrics, "aggregate", &requests, makespan);
    if !requests.is_empty() {
        metrics.insert(
            "aggregate.mean_request_ms".into(),
            requests
                .iter()
                .map(|request| request.completed_ms)
                .sum::<f64>()
                / requests.len() as f64,
        );
        metrics.insert(
            "aggregate.max_request_ms".into(),
            requests
                .iter()
                .map(|request| request.completed_ms)
                .fold(0.0, f64::max),
        );
    }
    ScenarioSample {
        repetition,
        arm: arm.into(),
        outcome,
        makespan_ms: makespan.as_secs_f64() * 1_000.0,
        metrics: prefix_metrics(arm, metrics),
        requests,
        warnings: Vec::new(),
    }
}

fn single_sample(
    repetition: usize,
    arm: &str,
    request: RequestObservation,
    expected_output: u32,
    expected_answer: &str,
    makespan: std::time::Duration,
) -> ScenarioSample {
    let outcome = validate_requests(
        std::slice::from_ref(&request),
        &[expected_output],
        &[expected_answer],
    );
    let mut metrics = BTreeMap::new();
    insert_request_metrics(&mut metrics, "request", &request);
    ScenarioSample {
        repetition,
        arm: arm.into(),
        outcome,
        makespan_ms: makespan.as_secs_f64() * 1_000.0,
        metrics: prefix_metrics(arm, metrics),
        requests: vec![request],
        warnings: Vec::new(),
    }
}

fn validate_requests(
    requests: &[RequestObservation],
    expected_outputs: &[u32],
    expected_answers: &[&str],
) -> RunOutcome {
    if requests.len() != expected_outputs.len() || requests.len() != expected_answers.len() {
        return RunOutcome::Invalid;
    }
    if requests
        .iter()
        .any(|request| matches!(request.outcome, RunOutcome::Error))
    {
        return RunOutcome::Error;
    }
    if requests
        .iter()
        .zip(expected_outputs.iter().zip(expected_answers))
        .any(|(request, (expected_tokens, expected_answer))| {
            completion_tokens(request) != Some(*expected_tokens as u64)
                || request.content != *expected_answer
        })
    {
        return RunOutcome::Invalid;
    }
    RunOutcome::Valid
}

fn completion_tokens(request: &RequestObservation) -> Option<u64> {
    request
        .usage
        .as_ref()
        .map(|usage| usage.completion_tokens)
        .or_else(|| {
            request
                .timings
                .as_ref()
                .and_then(|timings| timings.predicted_n)
        })
}

fn prompt_total(request: &RequestObservation) -> Option<u64> {
    request
        .usage
        .as_ref()
        .map(|usage| usage.prompt_tokens)
        .or_else(|| {
            request
                .timings
                .as_ref()
                .and_then(|timings| Some(timings.cache_n.unwrap_or(0) + timings.prompt_n?))
        })
}

fn insert_request_metrics(
    metrics: &mut BTreeMap<String, f64>,
    prefix: &str,
    request: &RequestObservation,
) {
    metrics.insert(format!("{prefix}.e2e_ms"), request.completed_ms);
    if let Some(value) = request.first_semantic_ms {
        metrics.insert(format!("{prefix}.ttfs_ms"), value);
    }
    if let Some(usage) = &request.usage {
        metrics.insert(
            format!("{prefix}.prompt_tokens"),
            usage.prompt_tokens as f64,
        );
        metrics.insert(
            format!("{prefix}.completion_tokens"),
            usage.completion_tokens as f64,
        );
    }
    if let Some(timings) = &request.timings {
        for (name, value) in [
            ("cache_tokens", timings.cache_n.map(|value| value as f64)),
            (
                "evaluated_prompt_tokens",
                timings.prompt_n.map(|value| value as f64),
            ),
            ("prompt_ms", timings.prompt_ms),
            ("prompt_tokens_per_second", timings.prompt_per_second),
            ("decode_ms", timings.predicted_ms),
            ("decode_tokens_per_second", timings.predicted_per_second),
        ] {
            if let Some(value) = value {
                metrics.insert(format!("{prefix}.{name}"), value);
            }
        }
    }
}

fn insert_goodput(
    metrics: &mut BTreeMap<String, f64>,
    prefix: &str,
    requests: &[RequestObservation],
    makespan: std::time::Duration,
) {
    let seconds = makespan.as_secs_f64();
    if seconds <= 0.0 {
        return;
    }
    let completion = requests.iter().filter_map(completion_tokens).sum::<u64>();
    let prompt = requests.iter().filter_map(prompt_total).sum::<u64>();
    metrics.insert(
        format!("{prefix}.completion_goodput_tokens_per_second"),
        completion as f64 / seconds,
    );
    metrics.insert(
        format!("{prefix}.prompt_goodput_tokens_per_second"),
        prompt as f64 / seconds,
    );
    metrics.insert(
        format!("{prefix}.requests_per_second"),
        requests.len() as f64 / seconds,
    );
}

fn prefix_metrics(prefix: &str, metrics: BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    metrics
        .into_iter()
        .map(|(name, value)| (format!("{prefix}.{name}"), value))
        .collect()
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn experiment(
    id: &str,
    title: &str,
    question: &str,
    samples: Vec<ScenarioSample>,
    warnings: Vec<String>,
) -> ExperimentResult {
    let outcome = aggregate_outcome(samples.iter().map(|sample| &sample.outcome));
    let summaries = summarize_samples(&samples);
    let mut work_signatures = samples
        .iter()
        .flat_map(|sample| sample.requests.iter())
        .map(|request| WorkSignature {
            request_id: request.id.clone(),
            prompt_tokens: prompt_total(request),
            completion_tokens: completion_tokens(request),
            finish_reason: request.finish_reason.clone(),
            output_sha256: request.output_sha256.clone(),
        })
        .collect::<Vec<_>>();
    work_signatures.sort_by(|left, right| left.request_id.cmp(&right.request_id));
    ExperimentResult {
        id: id.into(),
        title: title.into(),
        question: question.into(),
        outcome,
        samples,
        summaries,
        work_signatures,
        warnings,
    }
}

fn aggregate_outcome<'a>(outcomes: impl Iterator<Item = &'a RunOutcome>) -> RunOutcome {
    let values = outcomes.collect::<Vec<_>>();
    if values.is_empty() {
        return RunOutcome::Unsupported;
    }
    if values
        .iter()
        .all(|value| matches!(value, RunOutcome::Valid))
    {
        RunOutcome::Valid
    } else if values
        .iter()
        .any(|value| matches!(value, RunOutcome::Error))
    {
        RunOutcome::Error
    } else if values
        .iter()
        .all(|value| matches!(value, RunOutcome::Unsupported))
    {
        RunOutcome::Unsupported
    } else {
        RunOutcome::Invalid
    }
}

fn compare_run(run: &BenchmarkRun) -> BenchmarkComparison {
    let candidate = &run.targets[0];
    let reference = &run.targets[1];
    let mut warnings = Vec::new();
    let mut metrics = Vec::new();
    let mut work_match = true;
    let mut semantic_match = true;
    let mut experiments_valid = true;
    let candidate_template = target_template_digest(candidate);
    let reference_template = target_template_digest(reference);
    let provenance_match = !run.profile.controlled
        || (candidate.target.model_sha256.is_some()
            && candidate.target.model_sha256 == reference.target.model_sha256
            && candidate_template.is_some()
            && candidate_template == reference_template
            && !candidate.target.configuration.is_empty()
            && candidate.target.configuration == reference.target.configuration);

    for candidate_experiment in &candidate.experiments {
        let Some(reference_experiment) = reference
            .experiments
            .iter()
            .find(|experiment| experiment.id == candidate_experiment.id)
        else {
            work_match = false;
            warnings.push(format!(
                "reference is missing experiment {}",
                candidate_experiment.id
            ));
            continue;
        };
        experiments_valid &= matches!(candidate_experiment.outcome, RunOutcome::Valid)
            && matches!(reference_experiment.outcome, RunOutcome::Valid);
        if candidate_experiment.work_signatures.len() != reference_experiment.work_signatures.len()
        {
            work_match = false;
        }
        for (candidate_work, reference_work) in candidate_experiment
            .work_signatures
            .iter()
            .zip(&reference_experiment.work_signatures)
        {
            work_match &= candidate_work.prompt_tokens == reference_work.prompt_tokens
                && candidate_work.completion_tokens == reference_work.completion_tokens
                && candidate_work.finish_reason == reference_work.finish_reason;
        }
        semantic_match &=
            semantic_outputs(candidate_experiment) == semantic_outputs(reference_experiment);
        for (name, candidate_summary) in &candidate_experiment.summaries {
            let Some(reference_summary) = reference_experiment.summaries.get(name) else {
                continue;
            };
            if reference_summary.median <= 0.0 {
                continue;
            }
            let candidate_values = metric_values(candidate_experiment, name);
            let reference_values = metric_values(reference_experiment, name);
            let interval = paired_ratio_interval(&candidate_values, &reference_values, 2_000);
            let ratio = candidate_summary.median / reference_summary.median;
            let stable = interval.is_some_and(|(low, high)| {
                let midpoint = (low + high) / 2.0;
                midpoint > 0.0
                    && (high - low) / 2.0 / midpoint <= run.profile.confidence_half_width_ratio
            });
            metrics.push(MetricComparison {
                experiment: candidate_experiment.id.clone(),
                metric: name.clone(),
                candidate_median: candidate_summary.median,
                reference_median: reference_summary.median,
                candidate_reference_ratio: ratio,
                confidence_low: interval.map(|value| value.0),
                confidence_high: interval.map(|value| value.1),
                stable,
            });
        }
    }
    if !semantic_match {
        warnings.push("one or more response digests differed across targets".into());
    }
    if !work_match {
        warnings.push("one or more prompt/completion work signatures differed".into());
    }
    if !experiments_valid {
        warnings.push("one or more experiments failed its validity contract".into());
    }
    if !provenance_match {
        warnings.push(
            "controlled provenance requires equal model digests, template digests, and explicit effective configurations"
                .into(),
        );
    }
    let primary_metrics = metrics
        .iter()
        .filter(|metric| is_primary_metric(&metric.experiment, &metric.metric))
        .collect::<Vec<_>>();
    let stable = !run.profile.controlled
        || (!primary_metrics.is_empty() && primary_metrics.iter().all(|metric| metric.stable));
    let outcome = if !work_match
        || !experiments_valid
        || !provenance_match
        || (run.profile.controlled && !semantic_match)
    {
        RunOutcome::Invalid
    } else if !stable {
        RunOutcome::Unstable
    } else {
        RunOutcome::Valid
    };
    BenchmarkComparison {
        schema_version: 1,
        run_id: run.run_id.clone(),
        candidate: candidate.target.name.clone(),
        reference: reference.target.name.clone(),
        outcome,
        semantic_match,
        work_match,
        metrics,
        warnings,
    }
}

fn semantic_outputs(experiment: &ExperimentResult) -> Vec<(usize, String, String, String)> {
    let mut outputs = experiment
        .samples
        .iter()
        .flat_map(|sample| {
            sample.requests.iter().map(|request| {
                (
                    sample.repetition,
                    sample.arm.clone(),
                    request.id.clone(),
                    output_digest(&request.content, &request.reasoning, &request.tool_calls),
                )
            })
        })
        .collect::<Vec<_>>();
    outputs.sort();
    outputs
}

fn target_template_digest(target: &TargetResult) -> Option<String> {
    let template = target
        .probe
        .properties
        .as_ref()?
        .get("chat_template")?
        .as_str()?;
    Some(format!("{:x}", Sha256::digest(template.as_bytes())))
}

fn is_primary_metric(experiment: &str, metric: &str) -> bool {
    match experiment {
        "E1" => metric.ends_with("request.e2e_ms"),
        "E2" => metric.ends_with("aggregate.completion_goodput_tokens_per_second"),
        "E3" => metric == "mixed.total.completion_goodput_tokens_per_second",
        "E4" => metric.ends_with("second.evaluated_prompt_tokens"),
        "E5" => metric.ends_with("aggregate.prompt_goodput_tokens_per_second"),
        "E6" => metric == "transaction_ms",
        "E7" => metric == "recovery.e2e_ms",
        _ => false,
    }
}

fn metric_values(experiment: &ExperimentResult, name: &str) -> Vec<f64> {
    experiment
        .samples
        .iter()
        .filter(|sample| matches!(sample.outcome, RunOutcome::Valid))
        .filter_map(|sample| sample.metrics.get(name).copied())
        .collect()
}

struct MemoryMonitor {
    baseline_bytes: u64,
    stop: tokio::sync::oneshot::Sender<()>,
    task: tokio::task::JoinHandle<u64>,
    process_id: u32,
}

impl MemoryMonitor {
    async fn start(process_id: Option<u32>) -> Option<Self> {
        let process_id = process_id?;
        let baseline_bytes = process_rss_bytes(process_id).await?;
        let (stop, mut stopped) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let mut peak = baseline_bytes;
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(25));
            loop {
                tokio::select! {
                    _ = &mut stopped => return peak,
                    _ = interval.tick() => {
                        if let Some(current) = process_rss_bytes(process_id).await {
                            peak = peak.max(current);
                        }
                    }
                }
            }
        });
        Some(Self {
            baseline_bytes,
            stop,
            task,
            process_id,
        })
    }

    async fn finish(self) -> MemoryEvidence {
        let _ = self.stop.send(());
        let peak_bytes = self.task.await.ok();
        MemoryEvidence {
            source: Some("process-rss-ps".into()),
            baseline_bytes: Some(self.baseline_bytes),
            peak_bytes,
            retained_bytes: process_rss_bytes(self.process_id).await,
        }
    }
}

async fn process_rss_bytes(process_id: u32) -> Option<u64> {
    let output = tokio::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &process_id.to_string()])
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    std::str::from_utf8(&output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .and_then(|kilobytes| kilobytes.checked_mul(1024))
}

fn merge_target_result(destination: &mut Option<TargetResult>, mut incoming: TargetResult) {
    for experiment in &mut incoming.experiments {
        *experiment = experiment_from_parts(experiment);
    }
    let Some(existing) = destination else {
        *destination = Some(incoming);
        return;
    };
    existing.probe.warnings.extend(incoming.probe.warnings);
    existing.memory.peak_bytes = existing
        .memory
        .peak_bytes
        .into_iter()
        .chain(incoming.memory.peak_bytes)
        .max();
    existing.memory.retained_bytes = incoming.memory.retained_bytes;
    if existing.memory.source.is_none() {
        existing.memory = incoming.memory;
    }
    existing.warnings.extend(incoming.warnings);
    for incoming_experiment in incoming.experiments {
        if let Some(existing_experiment) = existing
            .experiments
            .iter_mut()
            .find(|experiment| experiment.id == incoming_experiment.id)
        {
            existing_experiment
                .samples
                .extend(incoming_experiment.samples);
            existing_experiment
                .warnings
                .extend(incoming_experiment.warnings);
            *existing_experiment = experiment_from_parts(existing_experiment);
        } else {
            existing.experiments.push(incoming_experiment);
        }
    }
}

fn experiment_from_parts(source: &ExperimentResult) -> ExperimentResult {
    experiment(
        &source.id,
        &source.title,
        &source.question,
        source.samples.clone(),
        source.warnings.clone(),
    )
}

fn benchmark_run(
    run_id: String,
    profile: crate::model::Profile,
    fixtures_sha256: String,
    targets: Vec<TargetResult>,
    controlled_host: bool,
    exclusive_device: bool,
) -> BenchmarkRun {
    BenchmarkRun {
        schema_version: 1,
        suite_version: SUITE_VERSION.into(),
        run_id,
        recorded_at_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        profile,
        fixtures_sha256,
        host: HostEvidence {
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            logical_cpus: std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(1),
            controlled_host,
            exclusive_device,
        },
        targets,
    }
}

async fn persist(
    output_dir: &Path,
    run: &BenchmarkRun,
    comparison: Option<&BenchmarkComparison>,
) -> Result<(), BenchmarkError> {
    tokio::fs::create_dir_all(output_dir).await?;
    let run_json = serde_json::to_vec_pretty(run)?;
    tokio::fs::write(output_dir.join("run.json"), run_json).await?;
    tokio::fs::write(output_dir.join("summary.md"), run_markdown(run)).await?;
    for target in &run.targets {
        tokio::fs::write(
            output_dir.join(format!("target-{}.json", safe_name(&target.target.name))),
            serde_json::to_vec_pretty(target)?,
        )
        .await?;
    }
    if let Some(comparison) = comparison {
        tokio::fs::write(
            output_dir.join("comparison.json"),
            serde_json::to_vec_pretty(comparison)?,
        )
        .await?;
        tokio::fs::write(
            output_dir.join("comparison.md"),
            comparison_markdown(comparison),
        )
        .await?;
    }
    Ok(())
}

fn safe_name(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect()
}

fn default_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("benchmark-{millis}")
}

fn task_error(id: &str, error: tokio::task::JoinError) -> RequestObservation {
    RequestObservation {
        id: id.into(),
        outcome: RunOutcome::Error,
        status: None,
        headers_ms: None,
        first_event_ms: None,
        first_semantic_ms: None,
        completed_ms: 0.0,
        finish_reason: None,
        content: String::new(),
        reasoning: String::new(),
        tool_calls: Vec::new(),
        usage: None,
        timings: None,
        output_sha256: empty_digest(),
        raw_events: Vec::new(),
        error: Some(format!("request task failed: {error}")),
    }
}

fn cancelled_observation(id: String, progress: &str) -> RequestObservation {
    RequestObservation {
        id,
        outcome: RunOutcome::Cancelled,
        status: Some(200),
        headers_ms: None,
        first_event_ms: None,
        first_semantic_ms: None,
        completed_ms: 0.0,
        finish_reason: None,
        content: String::new(),
        reasoning: String::new(),
        tool_calls: Vec::new(),
        usage: None,
        timings: None,
        output_sha256: empty_digest(),
        raw_events: Vec::new(),
        error: Some(format!(
            "client intentionally cancelled after {progress} progress"
        )),
    }
}

fn empty_digest() -> String {
    format!("{:x}", Sha256::digest([]))
}
