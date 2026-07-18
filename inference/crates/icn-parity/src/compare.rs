use std::cmp::Ordering;
use std::collections::BTreeSet;

use anyhow::{Context, bail};
use serde_json::{Value, json};

use crate::model::{
    CaseDefinition, Comparator, ComparisonMetric, ComparisonRecord, ComparisonStatus, EngineOrder,
    EvidenceRecord, EvidenceRef, Measurement, Mismatch, MismatchKind, OutcomeClass, ProducerRole,
    RatioDirection, Statistic, Validity,
};
use crate::provenance::now_rfc3339;

pub struct ComparisonContext<'a> {
    pub run_id: &'a str,
    pub reference_path: &'a str,
    pub reference_sha256: &'a str,
    pub candidate_path: &'a str,
    pub candidate_sha256: &'a str,
    pub engine_order: EngineOrder,
}

pub fn compare_evidence(
    case: &CaseDefinition,
    reference: &EvidenceRecord,
    candidate: &EvidenceRecord,
    context: ComparisonContext<'_>,
) -> anyhow::Result<ComparisonRecord> {
    reference.validate_for(case, ProducerRole::Reference)?;
    candidate.validate_for(case, ProducerRole::Candidate)?;
    if reference.run_id != context.run_id || candidate.run_id != context.run_id {
        bail!("evidence run IDs do not match the comparison run");
    }

    let matched_work = reference.work == candidate.work
        && reference.provenance.effective_configuration
            == candidate.provenance.effective_configuration;
    let mut validity_reasons = Vec::new();
    if !matched_work {
        validity_reasons.push(
            "reference and candidate did not report identical work and effective configuration"
                .to_owned(),
        );
    }
    let mut valid = matched_work;
    if reference.provenance.artifacts != candidate.provenance.artifacts {
        valid = false;
        validity_reasons.push(
            "reference and candidate used different model/fixture/corpus artifact identities"
                .to_owned(),
        );
    }
    let reference_source = native_source_identity(reference);
    let candidate_source = native_source_identity(candidate);
    if reference_source.is_none()
        || candidate_source.is_none()
        || reference_source != candidate_source
    {
        valid = false;
        validity_reasons.push(
            "reference and candidate do not prove the same llama.cpp source inventory/revision"
                .to_owned(),
        );
    }
    if case.category == crate::model::Category::Performance {
        let configuration = &reference.provenance.effective_configuration;
        if configuration.is_empty() {
            valid = false;
            validity_reasons.push(
                "performance evidence has no producer-reported effective configuration".to_owned(),
            );
        }
        let required_fields: &[&str] = match case.operation.as_str() {
            "llama-bench.prompt-processing"
            | "llama-bench.token-generation"
            | "llama-bench.prompt-generation" => &[
                "model_id",
                "model_sha256",
                "threads",
                "batch_size",
                "ubatch_size",
                "requested_batch_size",
                "requested_ubatch_size",
                "n_gpu_layers",
                "cpu_strict",
                "threadpool_poll",
                "kv_type_k",
                "kv_type_v",
                "flash_attention",
                "n_prompt",
                "n_gen",
                "context_depth",
                "requested_n_ctx",
                "effective_n_ctx",
                "split_mode",
                "main_gpu",
                "offload_kqv",
                "devices",
                "tensor_split",
                "use_mmap",
                "use_direct_io",
                "embeddings",
                "operation_offload",
                "no_host",
                "kv_unified",
                "swa_full",
                "memory_clear_data",
                "warmup",
                "threadpool_contract",
                "backend",
                "build_lane",
                "actual_backends",
                "device_identity",
            ],
            "llama-batched-bench.sequence-throughput" => &[
                "n_kv_max",
                "n_batch",
                "n_ubatch",
                "flash_attn",
                "is_pp_shared",
                "n_gpu_layers",
                "n_threads",
                "n_threads_batch",
                "pp",
                "tg",
                "pl",
                "n_kv",
                "backend",
                "build_lane",
                "kv_unified",
            ],
            _ => &["backend", "build_lane"],
        };
        let missing = required_fields
            .iter()
            .filter(|field| !configuration.contains_key(**field))
            .copied()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            valid = false;
            validity_reasons.push(format!(
                "effective configuration is missing required fields: {}",
                missing.join(", ")
            ));
        }
        if reference.provenance.build.compiler == "unknown"
            || candidate.provenance.build.compiler == "unknown"
            || reference.provenance.build.flags.is_empty()
            || candidate.provenance.build.flags.is_empty()
        {
            valid = false;
            validity_reasons
                .push("performance evidence lacks complete compiler/native build flags".to_owned());
        }
        if reference.provenance.build.build_type != "release"
            || candidate.provenance.build.build_type != "release"
            || reference.provenance.build.assertions != Some(false)
            || candidate.provenance.build.assertions != Some(false)
            || reference
                .provenance
                .build
                .sanitizers
                .as_ref()
                .is_none_or(|sanitizers| !sanitizers.is_empty())
            || candidate
                .provenance
                .build
                .sanitizers
                .as_ref()
                .is_none_or(|sanitizers| !sanitizers.is_empty())
        {
            valid = false;
            validity_reasons.push(
                "performance evidence requires verified release builds without assertions or sanitizers"
                    .to_owned(),
            );
        }
    }
    let mut mismatches = Vec::new();
    let mut metrics = Vec::new();

    let both_skipped = reference.outcome.class == OutcomeClass::Skipped
        && candidate.outcome.class == OutcomeClass::Skipped
        && reference.outcome.code.is_some()
        && reference.outcome.code == candidate.outcome.code;
    if valid && !both_skipped {
        compare_outcomes(case, reference, candidate, &mut mismatches)?;
        if reference.outcome.class == OutcomeClass::Success
            && candidate.outcome.class == OutcomeClass::Success
        {
            match &case.comparison {
                Comparator::Exact { paths } => {
                    compare_paths(
                        &reference.output,
                        &candidate.output,
                        paths,
                        &[],
                        false,
                        &mut mismatches,
                    );
                }
                Comparator::Structural {
                    paths,
                    ignore_paths,
                } => {
                    compare_paths(
                        &reference.output,
                        &candidate.output,
                        paths,
                        ignore_paths,
                        false,
                        &mut mismatches,
                    );
                }
                Comparator::Numeric {
                    paths,
                    exact_paths,
                    absolute_tolerance,
                    relative_tolerance,
                    nan_equal,
                } => {
                    for path in paths {
                        let reference_value = pointer(&reference.output, path);
                        let candidate_value = pointer(&candidate.output, path);
                        if reference_value.is_none() && candidate_value.is_none() {
                            push_value_mismatch(
                                path,
                                MismatchKind::Missing,
                                "selected numeric path is missing from both evidence outputs",
                                None,
                                None,
                                &mut mismatches,
                            );
                            continue;
                        }
                        compare_numeric_value(
                            path,
                            reference_value,
                            candidate_value,
                            *absolute_tolerance,
                            *relative_tolerance,
                            *nan_equal,
                            &mut mismatches,
                        );
                    }
                    if !exact_paths.is_empty() {
                        compare_paths(
                            &reference.output,
                            &candidate.output,
                            exact_paths,
                            &[],
                            false,
                            &mut mismatches,
                        );
                    }
                }
                Comparator::OutcomeAgreement { .. } => {}
                Comparator::PerformanceRatio {
                    metric,
                    statistic,
                    minimum_ratio,
                    maximum_ratio,
                    direction,
                    exact_output_paths,
                } => {
                    if !exact_output_paths.is_empty() {
                        compare_paths(
                            &reference.output,
                            &candidate.output,
                            exact_output_paths,
                            &[],
                            false,
                            &mut mismatches,
                        );
                    }
                    compare_performance(
                        metric,
                        *statistic,
                        *minimum_ratio,
                        *maximum_ratio,
                        *direction,
                        &reference.measurements,
                        &candidate.measurements,
                        &mut mismatches,
                        &mut metrics,
                    )?;
                }
            }
        }
    }

    let status = if both_skipped {
        ComparisonStatus::Skipped
    } else if !valid {
        ComparisonStatus::Invalid
    } else if mismatches.is_empty() {
        ComparisonStatus::Pass
    } else {
        ComparisonStatus::Fail
    };
    Ok(ComparisonRecord {
        schema_version: crate::model::SCHEMA_VERSION.to_owned(),
        run_id: context.run_id.to_owned(),
        case_id: case.id.clone(),
        category: case.category,
        primitive: case.primitive.clone(),
        recorded_at: now_rfc3339(),
        reference: EvidenceRef {
            path: context.reference_path.to_owned(),
            sha256: context.reference_sha256.to_owned(),
            producer_role: ProducerRole::Reference,
        },
        candidate: EvidenceRef {
            path: context.candidate_path.to_owned(),
            sha256: context.candidate_sha256.to_owned(),
            producer_role: ProducerRole::Candidate,
        },
        comparator: case.comparison.clone(),
        validity: Validity {
            valid,
            reasons: validity_reasons,
            matched_work,
            engine_order: Some(context.engine_order),
            contamination_warnings: Vec::new(),
        },
        status,
        mismatches,
        metrics,
        warnings: Vec::new(),
    })
}

fn native_source_identity(evidence: &EvidenceRecord) -> Option<(Option<&str>, Option<&str>)> {
    evidence
        .provenance
        .components
        .iter()
        .find(|component| component.kind == "native-source" && component.name == "llama.cpp")
        .map(|component| {
            (
                component.revision.as_deref(),
                component.tree_sha256.as_deref(),
            )
        })
}

fn compare_outcomes(
    case: &CaseDefinition,
    reference: &EvidenceRecord,
    candidate: &EvidenceRecord,
    mismatches: &mut Vec<Mismatch>,
) -> anyhow::Result<()> {
    let accepted: &[OutcomeClass] = match &case.comparison {
        Comparator::OutcomeAgreement { accepted_classes } => accepted_classes.as_slice(),
        _ => &[],
    };
    if reference.outcome.class == OutcomeClass::Skipped
        && candidate.outcome.class == OutcomeClass::Skipped
        && (reference.outcome.code.is_none() || reference.outcome.code != candidate.outcome.code)
    {
        mismatches.push(Mismatch {
            path: "/outcome/code".to_owned(),
            kind: MismatchKind::Outcome,
            message:
                "skips establish parity only when both producers report the same explicit skip code"
                    .to_owned(),
            reference: Some(json!(reference.outcome.code)),
            candidate: Some(json!(candidate.outcome.code)),
            absolute_difference: None,
            relative_difference: None,
            tolerance: None,
        });
    } else if reference.outcome.class == OutcomeClass::RuntimeError
        || candidate.outcome.class == OutcomeClass::RuntimeError
    {
        mismatches.push(Mismatch {
            path: "/outcome/class".to_owned(),
            kind: MismatchKind::Outcome,
            message: "runtime-error evidence can never establish parity".to_owned(),
            reference: Some(json!(reference.outcome.class)),
            candidate: Some(json!(candidate.outcome.class)),
            absolute_difference: None,
            relative_difference: None,
            tolerance: None,
        });
    } else if reference.outcome.class != candidate.outcome.class {
        mismatches.push(Mismatch {
            path: "/outcome/class".to_owned(),
            kind: MismatchKind::Outcome,
            message: "reference and candidate outcome classes differ".to_owned(),
            reference: Some(json!(reference.outcome.class)),
            candidate: Some(json!(candidate.outcome.class)),
            absolute_difference: None,
            relative_difference: None,
            tolerance: None,
        });
    } else if !matches!(case.comparison, Comparator::OutcomeAgreement { .. })
        && reference.outcome.class != OutcomeClass::Success
    {
        mismatches.push(Mismatch {
            path: "/outcome/class".to_owned(),
            kind: MismatchKind::Outcome,
            message: "this comparator requires successful evidence from both producers".to_owned(),
            reference: Some(json!(reference.outcome.class)),
            candidate: Some(json!(candidate.outcome.class)),
            absolute_difference: None,
            relative_difference: None,
            tolerance: None,
        });
    } else if !accepted.is_empty() && !accepted.contains(&reference.outcome.class) {
        mismatches.push(Mismatch {
            path: "/outcome/class".to_owned(),
            kind: MismatchKind::Outcome,
            message: "agreed outcome class is not accepted by this case".to_owned(),
            reference: Some(json!(reference.outcome.class)),
            candidate: Some(json!(candidate.outcome.class)),
            absolute_difference: None,
            relative_difference: None,
            tolerance: None,
        });
    }
    Ok(())
}

fn compare_paths(
    reference: &Value,
    candidate: &Value,
    paths: &[String],
    ignored: &[String],
    _exact_bytes: bool,
    mismatches: &mut Vec<Mismatch>,
) {
    if paths.is_empty() {
        compare_structure("", Some(reference), Some(candidate), ignored, mismatches);
    } else {
        for path in paths {
            let reference_value = pointer(reference, path);
            let candidate_value = pointer(candidate, path);
            if reference_value.is_none() && candidate_value.is_none() {
                push_value_mismatch(
                    path,
                    MismatchKind::Missing,
                    "selected path is missing from both evidence outputs",
                    None,
                    None,
                    mismatches,
                );
            } else {
                compare_structure(path, reference_value, candidate_value, ignored, mismatches);
            }
        }
    }
}

fn compare_structure(
    path: &str,
    reference: Option<&Value>,
    candidate: Option<&Value>,
    ignored: &[String],
    mismatches: &mut Vec<Mismatch>,
) {
    if ignored
        .iter()
        .any(|ignored| path == ignored || is_descendant(path, ignored))
    {
        return;
    }
    match (reference, candidate) {
        (None, None) => {}
        (Some(reference), None) => push_value_mismatch(
            path,
            MismatchKind::Missing,
            "candidate value is missing",
            Some(reference),
            None,
            mismatches,
        ),
        (None, Some(candidate)) => push_value_mismatch(
            path,
            MismatchKind::Unexpected,
            "candidate contains an unexpected value",
            None,
            Some(candidate),
            mismatches,
        ),
        (Some(Value::Object(reference)), Some(Value::Object(candidate))) => {
            let keys = reference
                .keys()
                .chain(candidate.keys())
                .collect::<BTreeSet<_>>();
            for key in keys {
                let child = join_pointer(path, key);
                compare_structure(
                    &child,
                    reference.get(key),
                    candidate.get(key),
                    ignored,
                    mismatches,
                );
            }
        }
        (Some(Value::Array(reference)), Some(Value::Array(candidate))) => {
            let length = reference.len().max(candidate.len());
            for index in 0..length {
                let child = join_pointer(path, &index.to_string());
                compare_structure(
                    &child,
                    reference.get(index),
                    candidate.get(index),
                    ignored,
                    mismatches,
                );
            }
        }
        (Some(reference), Some(candidate)) if value_kind(reference) != value_kind(candidate) => {
            push_value_mismatch(
                path,
                MismatchKind::Type,
                "JSON value types differ",
                Some(reference),
                Some(candidate),
                mismatches,
            );
        }
        (Some(reference), Some(candidate)) if reference != candidate => {
            push_value_mismatch(
                path,
                MismatchKind::Value,
                "JSON values differ",
                Some(reference),
                Some(candidate),
                mismatches,
            );
        }
        _ => {}
    }
}

fn compare_numeric_value(
    path: &str,
    reference: Option<&Value>,
    candidate: Option<&Value>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    nan_equal: bool,
    mismatches: &mut Vec<Mismatch>,
) {
    match (reference, candidate) {
        (Some(Value::Array(reference)), Some(Value::Array(candidate))) => {
            for index in 0..reference.len().max(candidate.len()) {
                compare_numeric_value(
                    &join_pointer(path, &index.to_string()),
                    reference.get(index),
                    candidate.get(index),
                    absolute_tolerance,
                    relative_tolerance,
                    nan_equal,
                    mismatches,
                );
            }
        }
        (Some(Value::Object(reference)), Some(Value::Object(candidate))) => {
            let keys = reference
                .keys()
                .chain(candidate.keys())
                .collect::<BTreeSet<_>>();
            for key in keys {
                compare_numeric_value(
                    &join_pointer(path, key),
                    reference.get(key),
                    candidate.get(key),
                    absolute_tolerance,
                    relative_tolerance,
                    nan_equal,
                    mismatches,
                );
            }
        }
        (Some(reference), Some(candidate)) => {
            if nan_equal && reference == "NaN" && candidate == "NaN" {
                return;
            }
            let (Some(reference_number), Some(candidate_number)) =
                (reference.as_f64(), candidate.as_f64())
            else {
                compare_structure(path, Some(reference), Some(candidate), &[], mismatches);
                return;
            };
            let absolute_difference = (reference_number - candidate_number).abs();
            let scale = reference_number.abs().max(candidate_number.abs());
            let relative_difference = if scale == 0.0 {
                0.0
            } else {
                absolute_difference / scale
            };
            let tolerance = absolute_tolerance.max(relative_tolerance * scale);
            if absolute_difference > tolerance {
                mismatches.push(Mismatch {
                    path: path.to_owned(),
                    kind: MismatchKind::NumericTolerance,
                    message: "numeric values exceed the declared absolute/relative tolerance"
                        .to_owned(),
                    reference: Some(reference.clone()),
                    candidate: Some(candidate.clone()),
                    absolute_difference: Some(absolute_difference),
                    relative_difference: Some(relative_difference),
                    tolerance: Some(tolerance),
                });
            }
        }
        (reference, candidate) => compare_structure(path, reference, candidate, &[], mismatches),
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_performance(
    metric_name: &str,
    statistic: Statistic,
    minimum_ratio: Option<f64>,
    maximum_ratio: Option<f64>,
    direction: RatioDirection,
    reference: &[Measurement],
    candidate: &[Measurement],
    mismatches: &mut Vec<Mismatch>,
    metrics: &mut Vec<ComparisonMetric>,
) -> anyhow::Result<()> {
    let reference = unique_measurement(reference, metric_name, "reference")?;
    let candidate = unique_measurement(candidate, metric_name, "candidate")?;
    if reference.unit != candidate.unit {
        bail!("performance measurement units differ for {metric_name}");
    }
    let reference_value = statistic_value(&reference.samples, statistic)?;
    let candidate_value = statistic_value(&candidate.samples, statistic)?;
    if reference_value <= 0.0 || candidate_value <= 0.0 {
        bail!("performance ratio inputs must be greater than zero");
    }
    let ratio = match direction {
        RatioDirection::CandidateOverReference => candidate_value / reference_value,
        RatioDirection::ReferenceOverCandidate => reference_value / candidate_value,
    };
    metrics.push(ComparisonMetric {
        name: format!("{metric_name}.ratio"),
        unit: "ratio".to_owned(),
        value: ratio,
        reference_value: Some(reference_value),
        candidate_value: Some(candidate_value),
    });
    let reference_median = statistic_value(&reference.samples, Statistic::Median)?;
    let candidate_median = statistic_value(&candidate.samples, Statistic::Median)?;
    let reference_mad = median_absolute_deviation(&reference.samples, reference_median)?;
    let candidate_mad = median_absolute_deviation(&candidate.samples, candidate_median)?;
    for (name, unit, value, reference_value, candidate_value) in [
        (
            format!("{metric_name}.reference_median"),
            reference.unit.clone(),
            reference_median,
            Some(reference_median),
            None,
        ),
        (
            format!("{metric_name}.candidate_median"),
            candidate.unit.clone(),
            candidate_median,
            None,
            Some(candidate_median),
        ),
        (
            format!("{metric_name}.reference_mad"),
            reference.unit.clone(),
            reference_mad,
            Some(reference_mad),
            None,
        ),
        (
            format!("{metric_name}.candidate_mad"),
            candidate.unit.clone(),
            candidate_mad,
            None,
            Some(candidate_mad),
        ),
        (
            format!("{metric_name}.reference_relative_mad"),
            "ratio".to_owned(),
            reference_mad / reference_median.abs(),
            None,
            None,
        ),
        (
            format!("{metric_name}.candidate_relative_mad"),
            "ratio".to_owned(),
            candidate_mad / candidate_median.abs(),
            None,
            None,
        ),
    ] {
        metrics.push(ComparisonMetric {
            name,
            unit,
            value,
            reference_value,
            candidate_value,
        });
    }
    if minimum_ratio.is_some_and(|minimum| ratio < minimum)
        || maximum_ratio.is_some_and(|maximum| ratio > maximum)
    {
        let bounds = match (minimum_ratio, maximum_ratio) {
            (Some(minimum), Some(maximum)) => format!("[{minimum:.6}, {maximum:.6}]"),
            (Some(minimum), None) => format!("[{minimum:.6}, +infinity)"),
            (None, Some(maximum)) => format!("(0, {maximum:.6}]"),
            (None, None) => unreachable!("comparator validation requires a bound"),
        };
        mismatches.push(Mismatch {
            path: format!("/measurements/{metric_name}"),
            kind: MismatchKind::PerformanceBound,
            message: format!("performance ratio {ratio:.6} is outside {bounds}"),
            reference: Some(json!(reference_value)),
            candidate: Some(json!(candidate_value)),
            absolute_difference: None,
            relative_difference: None,
            tolerance: None,
        });
    }
    Ok(())
}

fn median_absolute_deviation(samples: &[f64], median: f64) -> anyhow::Result<f64> {
    let deviations = samples
        .iter()
        .map(|sample| (sample - median).abs())
        .collect::<Vec<_>>();
    statistic_value(&deviations, Statistic::Median)
}

fn unique_measurement<'a>(
    measurements: &'a [Measurement],
    name: &str,
    role: &str,
) -> anyhow::Result<&'a Measurement> {
    let matches = measurements
        .iter()
        .filter(|measurement| measurement.name == name)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [measurement] => Ok(measurement),
        [] => bail!("{role} evidence is missing measurement {name}"),
        _ => bail!("{role} evidence repeats measurement {name}"),
    }
}

fn statistic_value(samples: &[f64], statistic: Statistic) -> anyhow::Result<f64> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        bail!("cannot summarize empty or non-finite samples");
    }
    Ok(match statistic {
        Statistic::Mean => samples.iter().sum::<f64>() / samples.len() as f64,
        Statistic::Minimum => samples
            .iter()
            .copied()
            .reduce(f64::min)
            .context("missing minimum")?,
        Statistic::Median => {
            let mut sorted = samples.to_vec();
            sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
            let middle = sorted.len() / 2;
            if sorted.len().is_multiple_of(2) {
                (sorted[middle - 1] + sorted[middle]) / 2.0
            } else {
                sorted[middle]
            }
        }
    })
}

fn pointer<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        Some(value)
    } else {
        value.pointer(path)
    }
}

fn join_pointer(parent: &str, child: &str) -> String {
    let child = child.replace('~', "~0").replace('/', "~1");
    format!("{parent}/{child}")
}

fn is_descendant(path: &str, parent: &str) -> bool {
    !parent.is_empty()
        && path
            .strip_prefix(parent)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn push_value_mismatch(
    path: &str,
    kind: MismatchKind,
    message: &str,
    reference: Option<&Value>,
    candidate: Option<&Value>,
    mismatches: &mut Vec<Mismatch>,
) {
    mismatches.push(Mismatch {
        path: path.to_owned(),
        kind,
        message: message.to_owned(),
        reference: reference.cloned(),
        candidate: candidate.cloned(),
        absolute_difference: None,
        relative_difference: None,
        tolerance: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        AdapterKind, CaseStatus, Category, Invocations, Outcome, Requirements, SourceDefinition,
        SourceKind,
    };

    fn outcome_case(comparison: Comparator) -> CaseDefinition {
        CaseDefinition {
            schema: None,
            schema_version: crate::model::SCHEMA_VERSION.to_owned(),
            id: "correctness.test.outcome".to_owned(),
            title: "test".to_owned(),
            description: "test".to_owned(),
            category: Category::Correctness,
            primitive: "C1".to_owned(),
            adapter: AdapterKind::Differential,
            operation: "test.outcome".to_owned(),
            status: CaseStatus::Planned,
            disabled_reason: None,
            tags: Vec::new(),
            source: SourceDefinition {
                kind: SourceKind::MagnitudeDefined,
                upstream_file: None,
                upstream_case: None,
                note: None,
            },
            inputs: serde_json::Map::new(),
            requirements: Requirements {
                offline: true,
                fixtures: Vec::new(),
                model: None,
                capabilities: Vec::new(),
                backends: Vec::new(),
                architecture_tags: Vec::new(),
            },
            invocations: Invocations {
                reference: None,
                candidate: None,
            },
            comparison,
            prerequisites: Vec::new(),
            timing: None,
        }
    }

    #[test]
    fn median_handles_even_and_odd_samples() {
        assert_eq!(
            statistic_value(&[3.0, 1.0, 2.0], Statistic::Median).unwrap(),
            2.0
        );
        assert_eq!(
            statistic_value(&[4.0, 1.0, 3.0, 2.0], Statistic::Median).unwrap(),
            2.5
        );
    }

    #[test]
    fn numeric_comparison_reports_precise_path() {
        let reference = json!({ "logits": [1.0, 2.0] });
        let candidate = json!({ "logits": [1.0, 2.1] });
        let mut mismatches = Vec::new();
        compare_numeric_value(
            "/logits",
            reference.pointer("/logits"),
            candidate.pointer("/logits"),
            0.01,
            0.0,
            false,
            &mut mismatches,
        );
        assert_eq!(mismatches.len(), 1);
        assert_eq!(mismatches[0].path, "/logits/1");
    }

    #[test]
    fn structural_ignore_applies_to_descendants() {
        let mut mismatches = Vec::new();
        compare_structure(
            "",
            Some(&json!({"stable": 1, "volatile": {"a": 1}})),
            Some(&json!({"stable": 1, "volatile": {"a": 2}})),
            &["/volatile".to_owned()],
            &mut mismatches,
        );
        assert!(mismatches.is_empty());
    }

    #[test]
    fn explicitly_selected_path_missing_on_both_sides_is_a_mismatch() {
        let mut mismatches = Vec::new();
        compare_paths(
            &json!({}),
            &json!({}),
            &["/required".to_owned()],
            &[],
            false,
            &mut mismatches,
        );
        assert_eq!(mismatches.len(), 1);
        assert_eq!(mismatches[0].kind, MismatchKind::Missing);
    }

    #[test]
    fn mixed_skip_and_success_never_passes() {
        let case = outcome_case(Comparator::OutcomeAgreement {
            accepted_classes: vec![OutcomeClass::Skipped, OutcomeClass::Success],
        });
        let reference = EvidenceRecord {
            outcome: Outcome {
                class: OutcomeClass::Skipped,
                code: Some("no-device".to_owned()),
                message: None,
            },
            ..dummy_evidence()
        };
        let candidate = EvidenceRecord {
            outcome: Outcome {
                class: OutcomeClass::Success,
                code: None,
                message: None,
            },
            ..dummy_evidence()
        };
        let mut mismatches = Vec::new();
        compare_outcomes(&case, &reference, &candidate, &mut mismatches).unwrap();
        assert_eq!(mismatches.len(), 1);
    }

    #[test]
    fn runtime_error_is_a_mismatch_even_when_symmetric() {
        let case = outcome_case(Comparator::OutcomeAgreement {
            accepted_classes: vec![OutcomeClass::RuntimeError],
        });
        let reference = EvidenceRecord {
            outcome: Outcome {
                class: OutcomeClass::RuntimeError,
                code: Some("boom".to_owned()),
                message: None,
            },
            ..dummy_evidence()
        };
        let candidate = reference.clone();
        let mut mismatches = Vec::new();
        compare_outcomes(&case, &reference, &candidate, &mut mismatches).unwrap();
        assert_eq!(mismatches.len(), 1);
        assert!(mismatches[0].message.contains("never establish parity"));
    }

    #[test]
    fn performance_ratio_fails_when_declared_semantic_output_differs() {
        let mut case = outcome_case(Comparator::PerformanceRatio {
            metric: "duration".to_owned(),
            statistic: Statistic::Median,
            minimum_ratio: Some(0.5),
            maximum_ratio: Some(2.0),
            direction: RatioDirection::CandidateOverReference,
            exact_output_paths: vec!["".to_owned()],
        });
        case.id = "performance.test.semantic-output".to_owned();
        case.category = Category::Performance;
        case.primitive = "P6".to_owned();
        case.operation = "sampler.apply-performance".to_owned();
        case.prerequisites = vec!["correctness.test.outcome".to_owned()];
        case.timing = Some(crate::model::TimingContract {
            included: vec!["sampler apply".to_owned()],
            excluded: vec!["fixture construction".to_owned()],
            warmup_iterations: 1,
            measurement_iterations: 1,
            synchronize: false,
        });

        let mut reference = performance_evidence(ProducerRole::Reference);
        reference.output = json!({"resultTokenIds": [3, 2, 1]});
        let mut candidate = performance_evidence(ProducerRole::Candidate);
        candidate.output = json!({"resultTokenIds": [3, 2, 0]});

        let comparison = compare_evidence(
            &case,
            &reference,
            &candidate,
            ComparisonContext {
                run_id: "run",
                reference_path: "reference.json",
                reference_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                candidate_path: "candidate.json",
                candidate_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                engine_order: EngineOrder::AlternatingPairs,
            },
        )
        .unwrap();

        assert_eq!(comparison.status, ComparisonStatus::Fail);
        assert_eq!(comparison.mismatches.len(), 1);
        assert_eq!(comparison.mismatches[0].path, "/resultTokenIds/2");
    }

    fn performance_evidence(role: ProducerRole) -> EvidenceRecord {
        let mut evidence = dummy_evidence();
        evidence.case_id = "performance.test.semantic-output".to_owned();
        evidence.category = Category::Performance;
        evidence.primitive = "P6".to_owned();
        evidence.operation = "sampler.apply-performance".to_owned();
        evidence.producer.role = role;
        evidence.measurements = vec![Measurement {
            name: "duration".to_owned(),
            unit: "ns".to_owned(),
            samples: vec![10.0],
        }];
        evidence.provenance.build.build_type = "release".to_owned();
        evidence.provenance.build.compiler = "clang".to_owned();
        evidence.provenance.build.compiler_version = "test".to_owned();
        evidence.provenance.build.flags = vec!["-O3".to_owned()];
        evidence.provenance.build.assertions = Some(false);
        evidence.provenance.build.sanitizers = Some(Vec::new());
        evidence
            .provenance
            .effective_configuration
            .insert("backend".to_owned(), json!("cpu"));
        evidence
            .provenance
            .effective_configuration
            .insert("build_lane".to_owned(), json!("cargo-equivalent"));
        evidence
    }

    fn dummy_evidence() -> EvidenceRecord {
        serde_json::from_value(json!({
            "schema_version": "1", "run_id": "run", "case_id": "correctness.test.outcome",
            "category": "correctness", "primitive": "C1", "operation": "test.outcome",
            "recorded_at": "2026-01-01T00:00:00Z",
            "producer": {"role": "reference", "name": "test"},
            "outcome": {"class": "success"},
            "work": {"parameters": {}, "included": [], "excluded": []},
            "output": null, "measurements": [],
            "provenance": {
                "components": [{"kind": "native-source", "name": "llama.cpp", "tree_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],
                "build": {"build_type": "unknown", "compiler": "unknown", "compiler_version": "unknown", "flags": [], "assertions": false, "sanitizers": []},
                "host": {"os": "test", "arch": "test", "cpu": "test"},
                "devices": [], "artifacts": [], "effective_configuration": {}
            },
            "warnings": []
        })).unwrap()
    }
}
