use std::fmt::Write;

use crate::model::{BenchmarkComparison, BenchmarkRun, RunOutcome};

pub fn run_markdown(run: &BenchmarkRun) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "# Inference benchmark {}\n", run.run_id);
    let _ = writeln!(output, "- Suite version: `{}`", run.suite_version);
    let _ = writeln!(output, "- Profile: `{}`", run.profile.id);
    let _ = writeln!(output, "- Fixture SHA-256: `{}`", run.fixtures_sha256);
    let _ = writeln!(
        output,
        "- Host: `{}` `{}` ({} logical CPUs)",
        run.host.os, run.host.arch, run.host.logical_cpus
    );
    let _ = writeln!(
        output,
        "- Controlled host attested: `{}`",
        run.host.controlled_host
    );
    let _ = writeln!(
        output,
        "- Exclusive device attested: `{}`\n",
        run.host.exclusive_device
    );
    for target in &run.targets {
        let _ = writeln!(output, "## {}\n", target.target.name);
        let _ = writeln!(output, "- Endpoint: `{}`", target.target.endpoint);
        let _ = writeln!(output, "- Model: `{}`", target.target.model);
        let _ = writeln!(output, "- Adapter: `{:?}`", target.target.kind);
        if let Some(source) = &target.memory.source {
            let _ = writeln!(
                output,
                "- Memory ({source}): baseline `{}`, peak `{}`, retained `{}` bytes",
                optional_u64(target.memory.baseline_bytes),
                optional_u64(target.memory.peak_bytes),
                optional_u64(target.memory.retained_bytes)
            );
        }
        output.push('\n');
        let _ = writeln!(
            output,
            "| Experiment | Outcome | Valid samples | Total samples |"
        );
        let _ = writeln!(output, "| --- | --- | ---: | ---: |");
        for experiment in &target.experiments {
            let valid = experiment
                .samples
                .iter()
                .filter(|sample| matches!(sample.outcome, RunOutcome::Valid))
                .count();
            let _ = writeln!(
                output,
                "| {} — {} | `{:?}` | {} | {} |",
                experiment.id,
                experiment.title,
                experiment.outcome,
                valid,
                experiment.samples.len()
            );
        }
        output.push('\n');
        for experiment in &target.experiments {
            let _ = writeln!(output, "### {} — {}\n", experiment.id, experiment.title);
            let _ = writeln!(output, "{}\n", experiment.question);
            if experiment.summaries.is_empty() {
                output.push_str("No valid measurements.\n\n");
            } else {
                let _ = writeln!(output, "| Metric | N | Median | Mean | MAD | Min | Max |");
                let _ = writeln!(output, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
                for (name, summary) in &experiment.summaries {
                    let _ = writeln!(
                        output,
                        "| `{}` | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
                        name,
                        summary.samples,
                        summary.median,
                        summary.mean,
                        summary.median_absolute_deviation,
                        summary.min,
                        summary.max
                    );
                }
                output.push('\n');
            }
            for sample in &experiment.samples {
                for warning in &sample.warnings {
                    let _ = writeln!(
                        output,
                        "- {} repetition {}: {}",
                        sample.arm, sample.repetition, warning
                    );
                }
            }
            if experiment
                .samples
                .iter()
                .any(|sample| !sample.warnings.is_empty())
            {
                output.push('\n');
            }
        }
    }
    output
}

fn optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "unavailable".into(), |value| value.to_string())
}

pub fn comparison_markdown(comparison: &BenchmarkComparison) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "# Benchmark comparison {}\n", comparison.run_id);
    let _ = writeln!(output, "- Candidate: `{}`", comparison.candidate);
    let _ = writeln!(output, "- Reference: `{}`", comparison.reference);
    let _ = writeln!(output, "- Outcome: `{:?}`", comparison.outcome);
    let _ = writeln!(output, "- Work matched: `{}`", comparison.work_match);
    let _ = writeln!(
        output,
        "- Responses matched: `{}`\n",
        comparison.semantic_match
    );
    let _ = writeln!(
        output,
        "| Experiment | Metric | Candidate | Reference | Ratio | 95% interval | Stable |"
    );
    let _ = writeln!(output, "| --- | --- | ---: | ---: | ---: | --- | --- |");
    for metric in &comparison.metrics {
        let interval = match (metric.confidence_low, metric.confidence_high) {
            (Some(low), Some(high)) => format!("{low:.3}–{high:.3}"),
            _ => "n/a".into(),
        };
        let _ = writeln!(
            output,
            "| {} | `{}` | {:.3} | {:.3} | {:.3} | {} | {} |",
            metric.experiment,
            metric.metric,
            metric.candidate_median,
            metric.reference_median,
            metric.candidate_reference_ratio,
            interval,
            metric.stable
        );
    }
    if !comparison.warnings.is_empty() {
        output.push_str("\n## Warnings\n\n");
        for warning in &comparison.warnings {
            let _ = writeln!(output, "- {warning}");
        }
    }
    output
}
