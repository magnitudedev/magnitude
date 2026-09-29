use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, bail};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::model::{
    CaseDefinition, CommandSpec, DecoderKind, DeviceInfo, EvidenceRecord, InvocationKind,
    Measurement, Outcome, OutcomeClass, Producer, ProducerRole, StdinKind, WorkDefinition,
};
use crate::process::ProcessOutput;
use crate::provenance::{command_provenance, now_rfc3339};

pub fn decode_json<T: DeserializeOwned>(bytes: &[u8], label: &str) -> anyhow::Result<T> {
    serde_json::from_slice(bytes).with_context(|| format!("invalid JSON from {label}"))
}

pub fn decode_jsonl<T: DeserializeOwned>(bytes: &[u8], label: &str) -> anyhow::Result<Vec<T>> {
    let text =
        std::str::from_utf8(bytes).with_context(|| format!("{label} output is not UTF-8"))?;
    let mut records = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        records.push(
            serde_json::from_str(line).with_context(|| {
                format!("invalid JSONL record from {label} at line {}", index + 1)
            })?,
        );
    }
    if records.is_empty() {
        bail!("{label} produced no JSONL records");
    }
    Ok(records)
}

pub fn encode_case_stdin(
    case: &CaseDefinition,
    kind: StdinKind,
    runtime_input: Option<&Map<String, Value>>,
) -> anyhow::Result<Option<Vec<u8>>> {
    match kind {
        StdinKind::None => Ok(None),
        StdinKind::ProbeJsonl => {
            let request = ProbeRequest {
                schema_version: 1,
                case_id: &case.id,
                operation: &case.operation,
                input: runtime_input.unwrap_or(&case.inputs),
            };
            let mut bytes = serde_json::to_vec(&request)?;
            bytes.push(b'\n');
            Ok(Some(bytes))
        }
        StdinKind::CaseJson => Ok(Some(serde_json::to_vec(case)?)),
        StdinKind::CaseJsonl => {
            let mut bytes = serde_json::to_vec(case)?;
            bytes.push(b'\n');
            Ok(Some(bytes))
        }
    }
}

pub async fn decode_process_evidence(
    run_id: &str,
    case: &CaseDefinition,
    role: ProducerRole,
    invocation: InvocationKind,
    spec: &CommandSpec,
    output: &ProcessOutput,
    extra_components: Vec<crate::model::Component>,
) -> anyhow::Result<EvidenceRecord> {
    let mut warnings = Vec::new();
    if output.stdout_truncated {
        warnings.push("producer stdout exceeded the configured capture limit".to_owned());
    }
    if output.stderr_truncated {
        warnings.push("producer stderr exceeded the configured capture limit".to_owned());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !stderr.is_empty() {
        warnings.push(format!(
            "producer wrote to stderr: {}",
            bounded_message(&stderr)
        ));
    }

    if !output.success() || output.stdout_truncated {
        let message = if output.stdout_truncated {
            "producer stdout exceeded the configured limit; partial structured output was rejected"
                .to_owned()
        } else if output.timed_out {
            format!("producer timed out after {} ms", output.elapsed.as_millis())
        } else {
            format!("producer exited with status {:?}", output.exit_code)
        };
        return make_wrapped_evidence(
            run_id,
            case,
            role,
            invocation,
            spec,
            output,
            Outcome {
                class: OutcomeClass::RuntimeError,
                code: Some(
                    if output.stdout_truncated {
                        "stdout-limit"
                    } else if output.timed_out {
                        "timeout"
                    } else {
                        "nonzero-exit"
                    }
                    .to_owned(),
                ),
                message: Some(message),
            },
            json!({ "exit_code": output.exit_code, "timed_out": output.timed_out }),
            Vec::new(),
            Map::new(),
            warnings,
            extra_components,
        )
        .await;
    }

    match spec.decoder {
        DecoderKind::EvidenceJson => {
            let evidence: EvidenceRecord = decode_json(&output.stdout, "producer")?;
            validate_producer_evidence(evidence, run_id, case, role)
        }
        DecoderKind::EvidenceJsonl => {
            let records: Vec<EvidenceRecord> = decode_jsonl(&output.stdout, "producer")?;
            if records.len() != 1 {
                bail!("producer must emit exactly one evidence JSONL record per case");
            }
            validate_producer_evidence(records.into_iter().next().unwrap(), run_id, case, role)
        }
        DecoderKind::ProbeJsonl => {
            let records: Vec<ProbeResponse> = decode_jsonl(&output.stdout, "primitive producer")?;
            let [response] = records.as_slice() else {
                bail!("primitive producer must emit exactly one response JSONL record per case");
            };
            if response.schema_version != 1
                || response.case_id != case.id
                || response.operation != case.operation
            {
                bail!("primitive response envelope does not match the submitted request");
            }
            let (outcome, evidence) = match response.status {
                ProbeStatus::Ok => {
                    let evidence = response
                        .evidence
                        .clone()
                        .context("ok primitive response is missing evidence")?;
                    (success(), evidence)
                }
                ProbeStatus::Error => {
                    let error = response
                        .error
                        .as_ref()
                        .context("error primitive response is missing error details")?;
                    (
                        Outcome {
                            class: error.class,
                            code: error.code.clone(),
                            message: error.message.clone(),
                        },
                        Value::Null,
                    )
                }
            };
            let structured = if outcome.class == OutcomeClass::Success {
                decode_probe_evidence(evidence)?
            } else {
                ProbeEvidence::legacy(Value::Null)
            };
            let mut record = make_wrapped_evidence(
                run_id,
                case,
                role,
                invocation,
                spec,
                output,
                outcome,
                structured.output,
                structured.measurements,
                structured.effective_configuration,
                warnings,
                extra_components,
            )
            .await?;
            if let Some(work) = structured.work {
                record.work = work;
            }
            if !structured.devices.is_empty() {
                record.provenance.devices = structured.devices;
            }
            record.producer.version = structured.producer_version;
            if case.category == crate::model::Category::Performance
                && record.outcome.class == OutcomeClass::Success
                && record
                    .measurements
                    .iter()
                    .any(|measurement| measurement.samples.len() != 1)
            {
                bail!(
                    "timed primitive probe must emit exactly one sample per measurement invocation"
                );
            }
            Ok(record)
        }
        DecoderKind::ExitStatus => {
            make_wrapped_evidence(
                run_id,
                case,
                role,
                invocation,
                spec,
                output,
                Outcome {
                    class: OutcomeClass::Success,
                    code: Some("exit-0".to_owned()),
                    message: None,
                },
                json!({ "exit_code": 0 }),
                Vec::new(),
                Map::new(),
                warnings,
                extra_components,
            )
            .await
        }
        DecoderKind::Json => {
            let value: Value = decode_json(&output.stdout, "producer")?;
            make_wrapped_evidence(
                run_id,
                case,
                role,
                invocation,
                spec,
                output,
                success(),
                value,
                Vec::new(),
                Map::new(),
                warnings,
                extra_components,
            )
            .await
        }
        DecoderKind::Jsonl => {
            let values: Vec<Value> = decode_jsonl(&output.stdout, "producer")?;
            make_wrapped_evidence(
                run_id,
                case,
                role,
                invocation,
                spec,
                output,
                success(),
                Value::Array(values),
                Vec::new(),
                Map::new(),
                warnings,
                extra_components,
            )
            .await
        }
        DecoderKind::LlamaBenchJson => {
            let (value, measurements, configuration) = decode_llama_bench(&output.stdout, case)?;
            make_wrapped_evidence(
                run_id,
                case,
                role,
                invocation,
                spec,
                output,
                success(),
                value,
                measurements,
                configuration,
                warnings,
                extra_components,
            )
            .await
        }
        DecoderKind::BatchedBenchJsonl => {
            let (value, measurements, configuration) = decode_batched_bench(&output.stdout, case)?;
            make_wrapped_evidence(
                run_id,
                case,
                role,
                invocation,
                spec,
                output,
                success(),
                value,
                measurements,
                configuration,
                warnings,
                extra_components,
            )
            .await
        }
        DecoderKind::PerplexityText => {
            let mut text = output.stdout.clone();
            text.extend_from_slice(b"\n");
            text.extend_from_slice(&output.stderr);
            let value = decode_perplexity_text(&text)?;
            make_wrapped_evidence(
                run_id,
                case,
                role,
                invocation,
                spec,
                output,
                success(),
                value,
                Vec::new(),
                Map::new(),
                warnings,
                extra_components,
            )
            .await
        }
        DecoderKind::BackendOpsSql => {
            let (value, measurements, configuration) =
                decode_backend_ops_sql(&output.stdout, case)?;
            make_wrapped_evidence(
                run_id,
                case,
                role,
                invocation,
                spec,
                output,
                success(),
                value,
                measurements,
                configuration,
                warnings,
                extra_components,
            )
            .await
        }
    }
}

fn decode_batched_bench(
    bytes: &[u8],
    case: &CaseDefinition,
) -> anyhow::Result<(Value, Vec<Measurement>, Map<String, Value>)> {
    let records: Vec<Value> = decode_jsonl(bytes, "llama-batched-bench")?;
    let prompt = case.inputs.get("prompt_tokens").and_then(Value::as_u64);
    let generation = case
        .inputs
        .get("generation_tokens_per_sequence")
        .and_then(Value::as_u64);
    let sequences = case
        .inputs
        .get("parallel_sequences")
        .and_then(Value::as_u64);
    let shared = case.inputs.get("shared_prompt").and_then(Value::as_bool);
    let matches = records
        .iter()
        .filter(|record| {
            let object = record.as_object();
            prompt.is_none_or(|expected| {
                object
                    .and_then(|value| value.get("pp"))
                    .and_then(Value::as_u64)
                    == Some(expected)
            }) && generation.is_none_or(|expected| {
                object
                    .and_then(|value| value.get("tg"))
                    .and_then(Value::as_u64)
                    == Some(expected)
            }) && sequences.is_none_or(|expected| {
                object
                    .and_then(|value| value.get("pl"))
                    .and_then(Value::as_u64)
                    == Some(expected)
            }) && shared.is_none_or(|expected| {
                object
                    .and_then(|value| value.get("is_pp_shared"))
                    .and_then(Value::as_i64)
                    == Some(i64::from(expected))
            })
        })
        .collect::<Vec<_>>();
    let [selected] = matches.as_slice() else {
        bail!("llama-batched-bench must emit exactly one row matching pp/tg/pl/shared_prompt");
    };
    let object = selected.as_object().unwrap();
    let numeric = |name: &str| -> anyhow::Result<f64> {
        object
            .get(name)
            .and_then(Value::as_f64)
            .with_context(|| format!("batched benchmark row is missing {name}"))
    };
    let t_pp = numeric("t_pp")?;
    let t_tg = numeric("t_tg")?;
    let total = numeric("t")?;
    let speed = numeric("speed")?;
    if [t_pp, t_tg, total, speed]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        bail!("batched benchmark timings and throughput must be finite and positive");
    }
    let mut configuration = Map::new();
    for name in [
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
    ] {
        let value = object
            .get(name)
            .with_context(|| format!("batched benchmark row is missing {name}"))?;
        configuration.insert(name.to_owned(), value.clone());
    }
    // Durations and rates are measurements, not semantic output. Keep the
    // exact upstream work/configuration projection separate so performance
    // validity does not require the two implementations to take equal time.
    let semantic_output = Value::Object(configuration.clone());
    let gpu_layers = configuration
        .get("n_gpu_layers")
        .and_then(Value::as_i64)
        .context("batched benchmark n_gpu_layers is invalid")?;
    if gpu_layers == -2 || gpu_layers == i64::from(u32::MAX) {
        configuration.insert("n_gpu_layers".to_owned(), json!("all"));
    } else if gpu_layers == -1 {
        configuration.insert("n_gpu_layers".to_owned(), json!("auto"));
    }
    let expected_kv = case
        .inputs
        .get("expected_kv_tokens")
        .and_then(Value::as_u64)
        .context("batched benchmark case is missing expected_kv_tokens")?;
    let actual_kv = object
        .get("n_kv")
        .and_then(Value::as_u64)
        .context("batched benchmark n_kv is invalid")?;
    let maximum_kv = object
        .get("n_kv_max")
        .and_then(Value::as_u64)
        .context("batched benchmark n_kv_max is invalid")?;
    if actual_kv != expected_kv || maximum_kv < expected_kv {
        bail!(
            "batched benchmark observed KV work {actual_kv}/{maximum_kv}, expected {expected_kv} tokens within capacity"
        );
    }
    configuration.insert(
        "kv_unified".to_owned(),
        case.inputs
            .get("kv_unified")
            .cloned()
            .context("batched benchmark case is missing kv_unified")?,
    );
    Ok((
        semantic_output,
        vec![
            Measurement {
                name: "prompt_duration".to_owned(),
                unit: "s".to_owned(),
                samples: vec![t_pp],
            },
            Measurement {
                name: "generation_duration".to_owned(),
                unit: "s".to_owned(),
                samples: vec![t_tg],
            },
            Measurement {
                name: "duration".to_owned(),
                unit: "s".to_owned(),
                samples: vec![total],
            },
            Measurement {
                name: "tokens_per_second".to_owned(),
                unit: "tokens/s".to_owned(),
                samples: vec![speed],
            },
        ],
        configuration,
    ))
}

fn decode_perplexity_text(bytes: &[u8]) -> anyhow::Result<Value> {
    let text = std::str::from_utf8(bytes).context("llama-perplexity output is not UTF-8")?;
    let estimate =
        regex::Regex::new(r"Final estimate:\s*PPL\s*=\s*([0-9.eE+-]+)\s*\+/-\s*([0-9.eE+-]+)")?;
    let captures = estimate
        .captures(text)
        .context("llama-perplexity output has no final PPL estimate")?;
    let number = |index: usize| -> anyhow::Result<f64> {
        let value = captures
            .get(index)
            .context("missing perplexity capture")?
            .as_str()
            .parse::<f64>()?;
        if !value.is_finite() {
            bail!("perplexity output contains a non-finite value")
        }
        Ok(value)
    };
    let mean_kld = regex::Regex::new(r"Mean\s+KLD:\s*([0-9.eE+-]+)\s*(?:±|\+/-)\s*([0-9.eE+-]+)")?
        .captures(text)
        .map(|captures| -> anyhow::Result<Value> {
            Ok(json!({
                "mean": captures[1].parse::<f64>()?,
                "uncertainty": captures[2].parse::<f64>()?,
            }))
        })
        .transpose()?;
    Ok(json!({
        "perplexity": number(1)?,
        "uncertainty": number(2)?,
        "kld": mean_kld,
        "probability_statistics": Value::Null,
    }))
}

fn parse_csv_bool(value: &str) -> anyhow::Result<bool> {
    match value {
        "1" | "true" => Ok(true),
        "0" | "false" => Ok(false),
        _ => bail!("invalid CSV boolean {value:?}"),
    }
}

fn decode_backend_ops_sql(
    bytes: &[u8],
    case: &CaseDefinition,
) -> anyhow::Result<(Value, Vec<Measurement>, Map<String, Value>)> {
    let text = std::str::from_utf8(bytes).context("backend-ops SQL is not UTF-8")?;
    const FIELDS: [&str; 16] = [
        "test_time",
        "build_commit",
        "backend_name",
        "op_name",
        "op_params",
        "test_mode",
        "supported",
        "passed",
        "error_message",
        "time_us",
        "flops",
        "bandwidth_gb_s",
        "memory_kb",
        "n_runs",
        "device_description",
        "backend_reg_name",
    ];
    let mut rows = Vec::new();
    let mut durations = Vec::new();
    let mut backends = BTreeSet::new();
    for line in text
        .lines()
        .filter(|line| line.starts_with("INSERT INTO test_backend_ops"))
    {
        let values_text = line
            .split_once("VALUES (")
            .and_then(|(_, suffix)| suffix.strip_suffix(");"))
            .context("malformed backend-ops SQL INSERT")?;
        let values = parse_sql_quoted_values(values_text)?;
        if values.len() != FIELDS.len() {
            bail!(
                "backend-ops SQL row has {} values, expected {}",
                values.len(),
                FIELDS.len()
            );
        }
        let fields = FIELDS
            .iter()
            .copied()
            .zip(values.iter().map(String::as_str))
            .collect::<BTreeMap<_, _>>();
        if fields["test_mode"] != "perf" {
            bail!("backend-ops SQL contains non-perf row");
        }
        // The pinned upstream perf path constructs `test_result` without the
        // optional device-description and registry-name arguments, so its SQL
        // rows contain empty strings for those two fields. The exact device is
        // still bound by the `-b` invocation captured in evidence, while the
        // result itself supplies the concrete backend instance name.
        if fields["build_commit"].trim().is_empty()
            || fields["backend_name"].trim().is_empty()
            || matches!(
                fields["backend_name"].to_ascii_lowercase().as_str(),
                "unknown" | "auto"
            )
        {
            bail!("backend-ops SQL row has unresolved tool/backend/device identity");
        }
        let supported = parse_csv_bool(fields["supported"])?;
        let passed = parse_csv_bool(fields["passed"])?;
        let time_us = fields["time_us"].parse::<f64>()?;
        let flops = fields["flops"].parse::<f64>()?;
        let bandwidth = fields["bandwidth_gb_s"].parse::<f64>()?;
        let memory_kb = fields["memory_kb"].parse::<u64>()?;
        let n_runs = fields["n_runs"].parse::<u64>()?;
        if supported && (!passed || !time_us.is_finite() || time_us <= 0.0 || n_runs == 0) {
            bail!("supported backend-ops perf row has invalid result fields");
        }
        if supported {
            durations.push((fields["op_name"].to_owned(), time_us * 1_000.0));
        }
        backends.insert(fields["backend_name"].to_owned());
        rows.push(json!({
            "build_commit": fields["build_commit"],
            "backend_name": fields["backend_name"],
            "backend_reg_name": fields["backend_reg_name"],
            "device_description": fields["device_description"],
            "op_name": fields["op_name"],
            "op_params": fields["op_params"],
            "supported": supported,
            "passed": passed,
            "error_message": fields["error_message"],
            "time_us": time_us,
            "flops": flops,
            "bandwidth_gb_s": bandwidth,
            "memory_kb": memory_kb,
            "n_runs": n_runs,
        }));
    }
    if rows.is_empty() {
        bail!(
            "backend-ops SQL contains no result rows; the exact backend_device filter may not match an upstream device name"
        );
    }
    if durations.is_empty() {
        bail!("backend-ops SQL contains no supported perf result rows");
    }
    let requested = case
        .inputs
        .get("operations")
        .and_then(Value::as_array)
        .context("backend-ops case has no explicit operations")?;
    let expected_backend = case
        .inputs
        .get("backend_device")
        .and_then(Value::as_str)
        .context("backend-ops case has no exact backend_device")?;
    let identities = rows
        .iter()
        .map(|row| {
            (
                row.get("backend_name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                row.get("backend_reg_name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                row.get("device_description")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            )
        })
        .collect::<BTreeSet<_>>();
    if identities.len() != 1 {
        bail!(
            "backend-ops filtered qualification must report exactly one observed backend/device identity"
        );
    }
    let (backend_name, backend_registry, device_description) =
        identities.into_iter().next().unwrap();
    let succeeded = rows
        .iter()
        .filter(|row| {
            row.get("supported").and_then(Value::as_bool) == Some(true)
                && row.get("passed").and_then(Value::as_bool) == Some(true)
        })
        .filter_map(|row| row.get("op_name").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let missing = requested
        .iter()
        .filter_map(Value::as_str)
        .filter(|operation| !succeeded.contains(operation))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "backend-ops perf output has no supported, passing row for requested operations: {}",
            missing.join(", ")
        );
    }
    let measurements = durations
        .into_iter()
        .enumerate()
        .map(|(index, (operation, duration))| {
            let operation = operation.to_ascii_lowercase().replace('_', "-");
            Measurement {
                name: format!("duration.{operation}.{index:04}"),
                unit: "ns".to_owned(),
                samples: vec![duration],
            }
        })
        .collect();
    Ok((
        Value::Array(rows),
        measurements,
        Map::from_iter([
            ("backends".to_owned(), json!(backends)),
            ("backend_device_filter".to_owned(), json!(expected_backend)),
            ("observed_backend_name".to_owned(), json!(backend_name)),
            (
                "observed_backend_registry".to_owned(),
                json!(backend_registry),
            ),
            (
                "observed_device_description".to_owned(),
                json!(device_description),
            ),
        ]),
    ))
}

fn parse_sql_quoted_values(input: &str) -> anyhow::Result<Vec<String>> {
    let mut values = Vec::new();
    let mut chars = input.chars().peekable();
    loop {
        while chars
            .peek()
            .is_some_and(|character| character.is_whitespace() || *character == ',')
        {
            chars.next();
        }
        if chars.peek().is_none() {
            break;
        }
        if chars.next() != Some('\'') {
            bail!("backend-ops SQL values must be single-quoted");
        }
        let mut value = String::new();
        loop {
            match chars.next() {
                Some('\'') if chars.peek() == Some(&'\'') => {
                    chars.next();
                    value.push('\'');
                }
                Some('\'') => break,
                Some(character) => value.push(character),
                None => bail!("unterminated backend-ops SQL value"),
            }
        }
        values.push(value);
    }
    Ok(values)
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ProbeRequest<'a> {
    schema_version: u32,
    case_id: &'a str,
    operation: &'a str,
    input: &'a Map<String, Value>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProbeResponse {
    schema_version: u32,
    case_id: String,
    operation: String,
    status: ProbeStatus,
    #[serde(default)]
    evidence: Option<Value>,
    #[serde(default)]
    error: Option<ProbeError>,
}

#[derive(Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
enum ProbeStatus {
    Ok,
    Error,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeError {
    class: OutcomeClass,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TimedProbeEvidenceV1 {
    schema_version: u32,
    output: Value,
    #[serde(default)]
    measurements: Vec<Measurement>,
    #[serde(default)]
    work: Option<WorkDefinition>,
    #[serde(default)]
    effective_configuration: Map<String, Value>,
    #[serde(default)]
    devices: Vec<DeviceInfo>,
    #[serde(default)]
    producer_version: Option<String>,
}

struct ProbeEvidence {
    output: Value,
    measurements: Vec<Measurement>,
    work: Option<WorkDefinition>,
    effective_configuration: Map<String, Value>,
    devices: Vec<DeviceInfo>,
    producer_version: Option<String>,
}

impl ProbeEvidence {
    fn legacy(output: Value) -> Self {
        Self {
            output,
            measurements: Vec::new(),
            work: None,
            effective_configuration: Map::new(),
            devices: Vec::new(),
            producer_version: None,
        }
    }
}

fn decode_probe_evidence(value: Value) -> anyhow::Result<ProbeEvidence> {
    let is_versioned = value
        .as_object()
        .is_some_and(|object| object.contains_key("schemaVersion"));
    if !is_versioned {
        return Ok(ProbeEvidence::legacy(value));
    }
    let timed: TimedProbeEvidenceV1 = serde_json::from_value(value)
        .context("invalid versioned primitive-probe evidence payload")?;
    if timed.schema_version != 1 {
        bail!(
            "unsupported primitive-probe evidence schema version {}",
            timed.schema_version
        );
    }
    for measurement in &timed.measurements {
        if measurement.samples.is_empty()
            || measurement.samples.iter().any(|sample| !sample.is_finite())
        {
            bail!(
                "primitive-probe measurement {} contains invalid samples",
                measurement.name
            );
        }
    }
    Ok(ProbeEvidence {
        output: timed.output,
        measurements: timed.measurements,
        work: timed.work,
        effective_configuration: timed.effective_configuration,
        devices: timed.devices,
        producer_version: timed.producer_version,
    })
}

fn validate_producer_evidence(
    evidence: EvidenceRecord,
    run_id: &str,
    case: &CaseDefinition,
    role: ProducerRole,
) -> anyhow::Result<EvidenceRecord> {
    if evidence.run_id != run_id {
        bail!("producer evidence run_id does not match the active run");
    }
    evidence.validate_for(case, role)?;
    Ok(evidence)
}

#[allow(clippy::too_many_arguments)]
async fn make_wrapped_evidence(
    run_id: &str,
    case: &CaseDefinition,
    role: ProducerRole,
    invocation: InvocationKind,
    spec: &CommandSpec,
    process: &ProcessOutput,
    outcome: Outcome,
    output: Value,
    measurements: Vec<Measurement>,
    effective_configuration: Map<String, Value>,
    warnings: Vec<String>,
    extra_components: Vec<crate::model::Component>,
) -> anyhow::Result<EvidenceRecord> {
    let provenance = command_provenance(
        invocation,
        spec,
        &process.program,
        effective_configuration,
        extra_components,
    )
    .await?;
    let producer_name = process
        .program
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_owned();
    let binary_sha256 = provenance
        .components
        .first()
        .and_then(|component| component.binary_sha256.clone());
    let timing = case.timing.as_ref();
    Ok(EvidenceRecord {
        schema_version: crate::model::SCHEMA_VERSION.to_owned(),
        run_id: run_id.to_owned(),
        case_id: case.id.clone(),
        category: case.category,
        primitive: case.primitive.clone(),
        operation: case.operation.clone(),
        recorded_at: now_rfc3339(),
        producer: Producer {
            role,
            kind: Some(invocation),
            name: producer_name,
            version: None,
            binary_sha256,
        },
        outcome,
        work: WorkDefinition {
            parameters: case.inputs.clone(),
            included: timing.map_or_else(Vec::new, |timing| timing.included.clone()),
            excluded: timing.map_or_else(Vec::new, |timing| timing.excluded.clone()),
            item_count: None,
            plan_sha256: None,
        },
        output,
        measurements,
        provenance,
        warnings,
    })
}

fn success() -> Outcome {
    Outcome {
        class: OutcomeClass::Success,
        code: None,
        message: None,
    }
}

fn bounded_message(value: &str) -> String {
    const MAX_CHARS: usize = 500;
    let mut chars = value.chars();
    let bounded: String = chars.by_ref().take(MAX_CHARS).collect();
    if chars.next().is_some() {
        format!("{bounded}…")
    } else {
        bounded
    }
}

fn decode_llama_bench(
    bytes: &[u8],
    case: &CaseDefinition,
) -> anyhow::Result<(Value, Vec<Measurement>, Map<String, Value>)> {
    let value: Value = decode_json(bytes, "llama-bench")?;
    let records = match value {
        Value::Array(records) => records,
        record @ Value::Object(_) => vec![record],
        _ => bail!("llama-bench output must be an object or array"),
    };
    let selected = select_benchmark_record(&records, &case.inputs)?;
    let object = selected
        .as_object()
        .context("benchmark record must be an object")?;
    let samples = object
        .get("samples_ns")
        .and_then(Value::as_array)
        .context("benchmark record is missing samples_ns")?
        .iter()
        .map(|sample| {
            sample
                .as_f64()
                .context("samples_ns must contain finite numbers")
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let required_samples = case
        .timing
        .as_ref()
        .map_or(1, |timing| timing.measurement_iterations as usize);
    if samples.len() != required_samples
        || samples
            .iter()
            .any(|sample| !sample.is_finite() || *sample <= 0.0)
    {
        bail!(
            "benchmark samples_ns must contain exactly {required_samples} finite positive samples"
        );
    }
    let configuration = normalize_llama_bench_configuration(object)?;
    let measured_prompt_tokens = object
        .get("n_prompt")
        .and_then(Value::as_u64)
        .context("benchmark n_prompt is invalid")?;
    let measured_generation_tokens = object
        .get("n_gen")
        .and_then(Value::as_u64)
        .context("benchmark n_gen is invalid")?;
    let prepared_context_depth_tokens = object
        .get("n_depth")
        .and_then(Value::as_u64)
        .context("benchmark n_depth is invalid")?;
    let batch_tokens = object
        .get("n_batch")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .context("benchmark n_batch must be positive")?;
    let measured_tokens = measured_prompt_tokens.saturating_add(measured_generation_tokens);
    if measured_tokens == 0 {
        bail!("benchmark record must measure at least one prompt or generation token");
    }
    let prompt_decode_calls = measured_prompt_tokens.div_ceil(batch_tokens);
    let depth_decode_calls = prepared_context_depth_tokens.div_ceil(batch_tokens);
    let measurement_repetitions = samples.len();
    let token_schedule = case
        .inputs
        .get("token_schedule")
        .and_then(Value::as_str)
        .context("llama-bench case is missing token_schedule")?;
    let semantic_output = json!({
        "operation": case.operation,
        "measuredPromptTokens": measured_prompt_tokens,
        "measuredPromptDecodeCalls": prompt_decode_calls,
        "measuredGenerationTokens": measured_generation_tokens,
        "measuredGenerationDecodeCalls": measured_generation_tokens,
        "preparedContextDepthTokens": prepared_context_depth_tokens,
        "preparedContextDepthDecodeCalls": depth_decode_calls,
        "measurementRepetitions": measurement_repetitions,
        "measuredTokens": measured_tokens,
        "tokenSchedule": token_schedule,
    });
    let throughput_samples = samples
        .iter()
        .map(|duration_ns| measured_tokens as f64 * 1_000_000_000.0 / duration_ns)
        .collect();
    Ok((
        semantic_output,
        vec![
            Measurement {
                name: "duration".to_owned(),
                unit: "ns".to_owned(),
                samples,
            },
            Measurement {
                name: "tokens_per_second".to_owned(),
                unit: "tokens/s".to_owned(),
                samples: throughput_samples,
            },
        ],
        configuration,
    ))
}

fn select_benchmark_record<'a>(
    records: &'a [Value],
    inputs: &Map<String, Value>,
) -> anyhow::Result<&'a Value> {
    let expected_prompt = inputs
        .get("n_prompt")
        .or_else(|| inputs.get("prompt_tokens"))
        .and_then(Value::as_u64);
    let expected_gen = inputs
        .get("n_gen")
        .or_else(|| inputs.get("generation_tokens"))
        .and_then(Value::as_u64);
    let matches = records
        .iter()
        .filter(|record| {
            let object = record.as_object();
            let prompt = object
                .and_then(|object| object.get("n_prompt"))
                .and_then(Value::as_u64);
            let generation = object
                .and_then(|object| object.get("n_gen"))
                .and_then(Value::as_u64);
            expected_prompt.is_none_or(|expected| prompt == Some(expected))
                && expected_gen.is_none_or(|expected| generation == Some(expected))
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [record] => Ok(record),
        [] => bail!("benchmark output contains no record matching n_prompt/n_gen"),
        _ => bail!("benchmark output contains multiple records matching n_prompt/n_gen"),
    }
}

fn normalize_llama_bench_configuration(
    object: &Map<String, Value>,
) -> anyhow::Result<Map<String, Value>> {
    let aliases = [
        ("model", "model_filename"),
        ("model_bytes", "model_size"),
        ("threads", "n_threads"),
        ("batch_size", "n_batch"),
        ("ubatch_size", "n_ubatch"),
        ("n_gpu_layers", "n_gpu_layers"),
        ("cpu_strict", "cpu_strict"),
        ("threadpool_poll", "poll"),
        ("kv_type_k", "type_k"),
        ("kv_type_v", "type_v"),
        ("n_prompt", "n_prompt"),
        ("n_gen", "n_gen"),
        ("context_depth", "n_depth"),
    ];
    let mut normalized = Map::new();
    for (normalized_name, source_name) in aliases {
        let value = object.get(source_name).with_context(|| {
            format!("benchmark record is missing same-work field {source_name}")
        })?;
        normalized.insert(normalized_name.to_owned(), value.clone());
    }
    let flash = object
        .get("flash_attn")
        .or_else(|| object.get("flash_attention"));
    if let Some(flash) = flash {
        let normalized_flash = match flash {
            Value::Number(value) if value.as_i64() == Some(-1) => json!("auto"),
            Value::Number(value) if value.as_i64() == Some(0) => json!("off"),
            Value::Number(value) if value.as_i64() == Some(1) => json!("on"),
            Value::String(_) => flash.clone(),
            _ => bail!("benchmark flash attention value is invalid"),
        };
        normalized.insert("flash_attention".to_owned(), normalized_flash);
    } else {
        bail!("benchmark record is missing flash attention configuration");
    }
    let prompt = object
        .get("n_prompt")
        .and_then(Value::as_u64)
        .context("benchmark n_prompt is invalid")?;
    let generation = object
        .get("n_gen")
        .and_then(Value::as_u64)
        .context("benchmark n_gen is invalid")?;
    let depth = object
        .get("n_depth")
        .and_then(Value::as_u64)
        .or_else(|| {
            object
                .get("requested_n_ctx")
                .and_then(Value::as_u64)
                .map(|context| context.saturating_sub(prompt + generation))
        })
        .unwrap_or(0);
    normalized.insert("context_depth".to_owned(), json!(depth));
    normalized.insert(
        "requested_n_ctx".to_owned(),
        json!(prompt.saturating_add(generation).saturating_add(depth)),
    );
    let requested_n_ctx = prompt.saturating_add(generation).saturating_add(depth);
    let requested_batch_size = normalized
        .get("batch_size")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .context("benchmark n_batch must be positive")?;
    let requested_ubatch_size = normalized
        .get("ubatch_size")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .context("benchmark n_ubatch must be positive")?;
    let effective_batch_size = requested_n_ctx.min(requested_batch_size);
    let effective_ubatch_size = effective_batch_size.min(requested_ubatch_size);
    normalized.insert(
        "requested_batch_size".to_owned(),
        json!(requested_batch_size),
    );
    normalized.insert(
        "requested_ubatch_size".to_owned(),
        json!(requested_ubatch_size),
    );
    normalized.insert("batch_size".to_owned(), json!(effective_batch_size));
    normalized.insert("ubatch_size".to_owned(), json!(effective_ubatch_size));
    // llama.cpp pads every context to a 256-token boundary during context
    // construction (llama-context.cpp). llama-bench records only its requested
    // pp + tg + depth size, so the adapter derives the actual native size while
    // preserving the unpadded request separately above.
    let effective_n_ctx = requested_n_ctx
        .checked_add(255)
        .map(|value| value / 256 * 256)
        .context("benchmark requested context overflows native 256-token padding")?;
    normalized.insert("effective_n_ctx".to_owned(), json!(effective_n_ctx));

    for (normalized_name, source_name) in [
        ("split_mode", "split_mode"),
        ("main_gpu", "main_gpu"),
        ("devices", "devices"),
        ("tensor_split", "tensor_split"),
        ("use_mmap", "use_mmap"),
        ("use_direct_io", "use_direct_io"),
        ("embeddings", "embeddings"),
        ("no_host", "no_host"),
    ] {
        normalized.insert(
            normalized_name.to_owned(),
            object
                .get(source_name)
                .with_context(|| format!("benchmark record is missing {source_name}"))?
                .clone(),
        );
    }
    let no_kv_offload = object
        .get("no_kv_offload")
        .and_then(Value::as_bool)
        .context("benchmark no_kv_offload is invalid")?;
    let no_op_offload = object
        .get("no_op_offload")
        .and_then(Value::as_i64)
        .context("benchmark no_op_offload is invalid")?;
    normalized.insert("offload_kqv".to_owned(), json!(!no_kv_offload));
    normalized.insert("operation_offload".to_owned(), json!(no_op_offload == 0));
    normalized.insert("kv_unified".to_owned(), json!(false));
    let gpu = normalized
        .get("n_gpu_layers")
        .and_then(Value::as_i64)
        .context("benchmark n_gpu_layers is invalid")?;
    if gpu == -2 || gpu == i64::from(u32::MAX) {
        normalized.insert("n_gpu_layers".to_owned(), json!("all"));
    } else if gpu == -1 {
        normalized.insert("n_gpu_layers".to_owned(), json!("auto"));
    }
    let backends = object
        .get("backends")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .context("llama-bench record is missing actual backends")?;
    let gpu = object
        .get("gpu_info")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let cpu = object
        .get("cpu_info")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let device = if gpu.is_empty() { cpu } else { gpu };
    if device.is_empty() {
        bail!("llama-bench record is missing actual CPU/GPU device identity");
    }
    normalized.insert("actual_backends".to_owned(), json!(backends));
    normalized.insert("device_identity".to_owned(), json!(device));
    normalized.insert("swa_full".to_owned(), json!(false));
    normalized.insert("memory_clear_data".to_owned(), json!(false));
    normalized.insert("warmup".to_owned(), json!(true));
    normalized.insert(
        "threadpool_contract".to_owned(),
        json!("persistent-per-case"),
    );
    Ok(normalized)
}

pub async fn read_evidence(path: &Path) -> anyhow::Result<EvidenceRecord> {
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("failed to read evidence {}", path.display()))?;
    decode_json(&bytes, &path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_case(relative: &str) -> CaseDefinition {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../parity/cases")
            .join(relative);
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    #[test]
    fn decodes_jsonl_with_blank_lines() {
        let values: Vec<Value> = decode_jsonl(b"{\"a\":1}\n\n{\"a\":2}\n", "fixture").unwrap();
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn normalizes_reference_benchmark_configuration() {
        let mut object = serde_json::from_value::<Map<String, Value>>(json!({
            "model_filename": "m.gguf", "model_size": 1024,
            "n_threads": 4, "n_batch": 8, "n_ubatch": 8,
            "n_gpu_layers": -2, "cpu_strict": false, "poll": 50,
            "type_k": "f16", "type_v": "f16", "n_prompt": 8,
            "n_gen": 0, "n_depth": 0, "flash_attn": 1,
            "split_mode": "layer", "main_gpu": 0, "no_kv_offload": false,
            "devices": "", "tensor_split": "", "use_mmap": true,
            "use_direct_io": false, "embeddings": false, "no_op_offload": 0,
            "no_host": false,
            "backends": "CPU", "cpu_info": "test cpu", "gpu_info": ""
        }))
        .unwrap();
        let normalized = normalize_llama_bench_configuration(&object).unwrap();
        assert_eq!(normalized["model"], "m.gguf");
        assert_eq!(normalized["flash_attention"], "on");
        assert_eq!(normalized["n_gpu_layers"], "all");
        assert_eq!(normalized["requested_n_ctx"], 8);
        assert_eq!(normalized["effective_n_ctx"], 256);

        object.insert("n_gpu_layers".to_owned(), json!(-1));
        let automatic = normalize_llama_bench_configuration(&object).unwrap();
        assert_eq!(automatic["n_gpu_layers"], "auto");
    }

    #[test]
    fn llama_bench_output_projects_declared_work_from_the_official_record() {
        let case = load_case("performance/llama-bench/prompt-generation-128-32.json");
        let record = json!({
            "model_filename": "m.gguf", "model_size": 1024,
            "n_threads": 4, "n_batch": 128, "n_ubatch": 128,
            "n_gpu_layers": -2, "cpu_strict": false, "poll": 50,
            "type_k": "f16", "type_v": "f16", "n_prompt": 128,
            "n_gen": 32, "n_depth": 0, "flash_attn": 1,
            "split_mode": "layer", "main_gpu": 0, "no_kv_offload": false,
            "devices": "", "tensor_split": "", "use_mmap": true,
            "use_direct_io": false, "embeddings": false, "no_op_offload": 0,
            "no_host": false,
            "backends": "Metal", "cpu_info": "test cpu", "gpu_info": "test gpu",
            "samples_ns": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        });

        let (output, _, configuration) =
            decode_llama_bench(&serde_json::to_vec(&record).unwrap(), &case).unwrap();

        assert_eq!(output["measuredPromptTokens"], 128);
        assert_eq!(output["measuredPromptDecodeCalls"], 1);
        assert_eq!(output["measuredGenerationTokens"], 32);
        assert_eq!(output["measuredGenerationDecodeCalls"], 32);
        assert_eq!(output["preparedContextDepthTokens"], 0);
        assert_eq!(output["preparedContextDepthDecodeCalls"], 0);
        assert_eq!(configuration["requested_n_ctx"], 160);
        assert_eq!(configuration["effective_n_ctx"], 256);
        assert_eq!(output["measurementRepetitions"], 10);
        assert_eq!(output["measuredTokens"], 160);
        assert_eq!(output["tokenSchedule"], "pinned-llama-bench-c-rand-default");
    }

    #[test]
    fn versioned_probe_evidence_carries_timing_work_and_configuration() {
        let decoded = decode_probe_evidence(json!({
            "schemaVersion": 1,
            "output": {"ok": true},
            "measurements": [{"name": "duration", "unit": "ns", "samples": [10.0]}],
            "work": {"parameters": {"tokens": 1}, "included": ["decode"], "excluded": ["load"], "item_count": 1},
            "effectiveConfiguration": {"threads": 4},
            "devices": [{"backend": "Metal", "name": "Apple GPU"}],
            "producerVersion": "probe-v1"
        })).unwrap();
        assert_eq!(decoded.measurements[0].samples, vec![10.0]);
        assert_eq!(decoded.work.unwrap().item_count, Some(1));
        assert_eq!(decoded.effective_configuration["threads"], 4);
        assert_eq!(decoded.devices[0].backend, "Metal");
    }

    #[test]
    fn decodes_pinned_batched_bench_jsonl_shape_and_kv_work() {
        let case = load_case("performance/batched-bench/two-sequence-independent-prompts.json");
        let line = b"{\"n_kv_max\":160,\"n_batch\":64,\"n_ubatch\":64,\"flash_attn\":0,\"is_pp_shared\":0,\"n_gpu_layers\":-1,\"n_threads\":4,\"n_threads_batch\":4,\"pp\":64,\"tg\":16,\"pl\":2,\"n_kv\":160,\"t_pp\":0.1,\"t_tg\":0.2,\"t\":0.3,\"speed\":533.333333}\n";
        let (output, measurements, configuration) = decode_batched_bench(line, &case).unwrap();
        assert_eq!(output["n_kv"], 160);
        assert!(output.get("t").is_none());
        assert!(output.get("speed").is_none());
        assert_eq!(configuration["n_kv"], 160);
        assert_eq!(configuration["kv_unified"], false);
        assert_eq!(configuration["n_gpu_layers"], "auto");
        assert_eq!(
            measurements
                .iter()
                .find(|measurement| measurement.name == "tokens_per_second")
                .unwrap()
                .samples
                .len(),
            1
        );
    }

    #[test]
    fn backend_ops_sql_requires_passing_rows_for_every_requested_operation() {
        let mut case = load_case("performance/backend-ops/native-perf-qualification.json");
        case.inputs
            .insert("backend_device".to_owned(), json!("Metal"));
        let rows = ["MUL_MAT", "ROPE", "SOFT_MAX", "FLASH_ATTN_EXT"].into_iter().map(|operation| {
            format!("INSERT INTO test_backend_ops (fields) VALUES ('now', 'commit', 'MTL0', '{operation}', 'shape', 'perf', '1', '1', '', '10.0', '100.0', '2.0', '4', '5', 'Apple GPU', 'Metal');")
        }).collect::<Vec<_>>().join("\n");
        let (_, measurements, configuration) =
            decode_backend_ops_sql(rows.as_bytes(), &case).unwrap();
        assert_eq!(measurements.len(), 4);
        assert_eq!(configuration["backend_device_filter"], "Metal");
        assert_eq!(configuration["observed_backend_name"], "MTL0");

        let incomplete = rows.lines().take(3).collect::<Vec<_>>().join("\n");
        assert!(decode_backend_ops_sql(incomplete.as_bytes(), &case).is_err());
    }

    #[test]
    fn backend_ops_sql_accepts_pinned_perf_rows_without_optional_device_metadata() {
        let mut case = load_case("performance/backend-ops/native-perf-qualification.json");
        case.inputs
            .insert("operations".to_owned(), json!(["SOFT_MAX"]));
        case.inputs
            .insert("backend_device".to_owned(), json!("MTL0"));
        let row = "INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-07-18T06:26:26Z', '9e3b928fd', 'MTL0', 'SOFT_MAX', 'type=f32,ne=[64,64,20,1],mask=0,sinks=0,m_prec=f32,nr23=[1,1],scale=1.000000,max_bias=0.000000,inplace=0', 'perf', '1', '1', '', '222.961055', '0.000000', '2.737648', '640', '8191', '', '');";

        let (decoded, measurements, configuration) =
            decode_backend_ops_sql(row.as_bytes(), &case).unwrap();

        assert_eq!(decoded[0]["device_description"], "");
        assert_eq!(decoded[0]["backend_reg_name"], "");
        assert_eq!(measurements.len(), 1);
        assert_eq!(configuration["backend_device_filter"], "MTL0");
        assert_eq!(configuration["observed_backend_name"], "MTL0");
        assert_eq!(configuration["observed_backend_registry"], "");
        assert_eq!(configuration["observed_device_description"], "");
    }

    #[test]
    fn backend_ops_sql_reports_an_unmatched_exact_device_filter() {
        let mut case = load_case("performance/backend-ops/native-perf-qualification.json");
        case.inputs
            .insert("backend_device".to_owned(), json!("not-a-device"));
        let table_only = b"CREATE TABLE IF NOT EXISTS test_backend_ops (test_time TEXT);\n";

        let error = decode_backend_ops_sql(table_only, &case).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("exact backend_device filter may not match")
        );
    }
}
