use crate::{runner::RunRecord, LabResult};
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::Path};

/// Reports retain censored runs. A timeout is never a sixty-second proof.
#[derive(Debug, Serialize)]
pub struct Summary {
    pub instance: String,
    pub policy: String,
    pub runs: usize,
    pub optimal: usize,
    pub infeasible: usize,
    pub incomplete: usize,
    pub failed: usize,
    pub completed_median_ms: Option<f64>,
    pub completed_min_ms: Option<f64>,
    pub completed_max_ms: Option<f64>,
    pub completion_fractions: Vec<(f64, f64)>,
    pub maximum_observed_rss_bytes: Option<u64>,
}

pub fn summarize(records: &[RunRecord]) -> Vec<Summary> {
    let mut groups: BTreeMap<(String, String), Vec<&RunRecord>> = BTreeMap::new();
    for record in records {
        groups
            .entry((record.instance.clone(), record.policy.clone()))
            .or_default()
            .push(record);
    }
    groups
        .into_iter()
        .map(|((instance, policy), rows)| {
            let mut times: Vec<f64> = rows
                .iter()
                .filter(|row| row.status == "optimal" || row.status == "infeasible")
                .map(|row| row.solve_ms)
                .collect();
            times.sort_by(f64::total_cmp);
            let median = if times.is_empty() {
                None
            } else if times.len().is_multiple_of(2) {
                Some((times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.0)
            } else {
                Some(times[times.len() / 2])
            };
            Summary {
                instance,
                policy,
                runs: rows.len(),
                optimal: rows.iter().filter(|row| row.status == "optimal").count(),
                infeasible: rows.iter().filter(|row| row.status == "infeasible").count(),
                incomplete: rows.iter().filter(|row| row.status == "incomplete").count(),
                failed: rows.iter().filter(|row| row.status == "error").count(),
                completed_median_ms: median,
                completed_min_ms: times.first().copied(),
                completed_max_ms: times.last().copied(),
                completion_fractions: [10.0, 100.0, 1_000.0, 10_000.0, 60_000.0]
                    .into_iter()
                    .map(|limit| {
                        (
                            limit,
                            times.iter().filter(|&&t| t <= limit).count() as f64
                                / rows.len() as f64,
                        )
                    })
                    .collect(),
                maximum_observed_rss_bytes: rows
                    .iter()
                    .filter_map(|row| row.observed_rss_bytes)
                    .max(),
            }
        })
        .collect()
}

pub fn report(input: &Path, output: &Path) -> LabResult<()> {
    fs::create_dir_all(output)?;
    let mut records = Vec::new();
    collect_runs(input, &mut records)?;
    if records.is_empty() {
        return Err("no measured run files found".into());
    }
    let summaries = summarize(&records);
    fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summaries)?,
    )?;
    let mut csv = String::from("instance,policy,runs,optimal,infeasible,incomplete,failed,completed_median_ms,completed_min_ms,completed_max_ms,observed_peak_rss_bytes\n");
    let mut md = String::from("# Solver evaluation\n\nSynthetic objective units do not describe hardware execution time. Times below measure the solver on this machine. Incomplete runs are censored, never counted as completed proofs. RSS is sampled by the watchdog; it is an observed maximum, not a guaranteed allocation peak.\n\n| Instance | Policy | Complete / runs | Median completed ms | Range ms | Incomplete | Errors |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n");
    for s in &summaries {
        let fmt = |v: Option<f64>| v.map_or_else(|| "—".into(), |n| format!("{n:.3}"));
        md.push_str(&format!(
            "| {} | {} | {} / {} | {} | {}–{} | {} | {} |\n",
            s.instance,
            s.policy,
            s.optimal + s.infeasible,
            s.runs,
            fmt(s.completed_median_ms),
            fmt(s.completed_min_ms),
            fmt(s.completed_max_ms),
            s.incomplete,
            s.failed
        ));
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{}\n",
            quote(&s.instance),
            quote(&s.policy),
            s.runs,
            s.optimal,
            s.infeasible,
            s.incomplete,
            s.failed,
            s.completed_median_ms
                .map_or(String::new(), |n| n.to_string()),
            s.completed_min_ms.map_or(String::new(), |n| n.to_string()),
            s.completed_max_ms.map_or(String::new(), |n| n.to_string()),
            s.maximum_observed_rss_bytes
                .map_or(String::new(), |n| n.to_string())
        ));
    }
    md.push_str("\nCompletion fractions at 10 ms, 100 ms, 1 s, 10 s and 60 s are retained in `summary.json`. Each `*.run.json` records settings, process overhead, bounds and solver counters. Compare settings only on identical model identities. No extrapolation to Seismic compilation time is justified by synthetic results alone.\n");
    fs::write(output.join("summary.csv"), csv)?;
    fs::write(output.join("report.md"), md)?;
    Ok(())
}

pub fn collect_runs(path: &Path, out: &mut Vec<RunRecord>) -> LabResult<()> {
    if path.is_file() {
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".run.json"))
        {
            out.push(serde_json::from_slice(&fs::read(path)?)?);
        }
    } else {
        let mut entries: Vec<_> = fs::read_dir(path)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<Result<_, _>>()?;
        entries.sort();
        for entry in entries {
            collect_runs(&entry, out)?;
        }
    }
    Ok(())
}

fn quote(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn censoring_is_not_a_completion_time() {
        let make = |status: &str, solve_ms| RunRecord {
            instance: "x".into(),
            policy: "dfs".into(),
            status: status.into(),
            solve_ms,
            ..RunRecord::default()
        };
        let rows = vec![
            make("optimal", 10.0),
            make("optimal", 20.0),
            make("incomplete", 60_000.0),
        ];
        let summary = summarize(&rows);
        assert_eq!(summary[0].completed_median_ms, Some(15.0));
        assert_eq!(summary[0].incomplete, 1);
        assert_eq!(summary[0].completion_fractions[1], (100.0, 2.0 / 3.0));
    }
}
