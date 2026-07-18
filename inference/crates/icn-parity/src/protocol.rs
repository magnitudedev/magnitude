use std::collections::BTreeSet;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::decode::decode_jsonl;
use crate::model::validate_operation;
use crate::process::ProcessOutput;

pub(crate) const PROBE_PROTOCOL_VERSION: u32 = 1;
pub(crate) const PROBE_TRANSPORT: &str = "jsonl-stdin-stdout";
pub(crate) const PREFLIGHT_CASE_ID: &str = "preflight.protocol-describe";
pub(crate) const DESCRIBE_OPERATION: &str = "protocol.describe";

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ProtocolDescription {
    pub protocol_version: u32,
    pub transport: String,
    pub operations: BTreeSet<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DescribeRequest<'a> {
    schema_version: u32,
    case_id: &'a str,
    operation: &'a str,
    input: serde_json::Map<String, Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DescribeResponse {
    schema_version: u32,
    case_id: String,
    operation: String,
    status: ProbeStatus,
    #[serde(default)]
    evidence: Option<RawProtocolDescription>,
    #[serde(default)]
    error: Option<ProbeError>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum ProbeStatus {
    Ok,
    Error,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeError {
    class: String,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawProtocolDescription {
    protocol_version: u32,
    transport: String,
    operations: Vec<String>,
}

pub(crate) fn describe_request_jsonl() -> anyhow::Result<Vec<u8>> {
    let request = DescribeRequest {
        schema_version: PROBE_PROTOCOL_VERSION,
        case_id: PREFLIGHT_CASE_ID,
        operation: DESCRIBE_OPERATION,
        input: serde_json::Map::new(),
    };
    let mut bytes = serde_json::to_vec(&request)?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub(crate) fn validate_protocol_description(
    label: &str,
    output: &ProcessOutput,
    required_operations: &BTreeSet<String>,
) -> anyhow::Result<ProtocolDescription> {
    validate_process_result(label, output)?;
    let responses: Vec<DescribeResponse> = decode_jsonl(
        &output.stdout,
        &format!("{label} protocol.describe preflight"),
    )?;
    let [response] = responses.as_slice() else {
        bail!(
            "{label} protocol preflight expected exactly one JSONL response, received {}",
            responses.len()
        );
    };
    if response.schema_version != PROBE_PROTOCOL_VERSION {
        bail!(
            "{label} protocol response envelope has schemaVersion {}; expected {}",
            response.schema_version,
            PROBE_PROTOCOL_VERSION
        );
    }
    if response.case_id != PREFLIGHT_CASE_ID || response.operation != DESCRIBE_OPERATION {
        bail!(
            "{label} protocol response envelope does not echo caseId {PREFLIGHT_CASE_ID:?} and operation {DESCRIBE_OPERATION:?}"
        );
    }
    let raw = match response.status {
        ProbeStatus::Ok => {
            if response.error.is_some() {
                bail!("{label} successful protocol response unexpectedly contains an error");
            }
            response
                .evidence
                .as_ref()
                .context("successful protocol response is missing evidence")?
        }
        ProbeStatus::Error => {
            if response.evidence.is_some() {
                bail!("{label} failed protocol response unexpectedly contains evidence");
            }
            let error = response
                .error
                .as_ref()
                .context("failed protocol response is missing error details")?;
            let code = error.code.as_deref().unwrap_or("unspecified");
            let message = error.message.as_deref().unwrap_or("no message");
            bail!(
                "{label} rejected protocol.describe: class={}, code={code}, message={message}",
                error.class
            );
        }
    };
    if raw.protocol_version != PROBE_PROTOCOL_VERSION {
        bail!(
            "{label} advertises protocolVersion {}; expected {}",
            raw.protocol_version,
            PROBE_PROTOCOL_VERSION
        );
    }
    if raw.transport != PROBE_TRANSPORT {
        bail!(
            "{label} advertises transport {:?}; expected {:?}",
            raw.transport,
            PROBE_TRANSPORT
        );
    }
    if raw.operations.is_empty() {
        bail!("{label} advertises no protocol operations");
    }
    let mut operations = BTreeSet::new();
    for operation in &raw.operations {
        validate_operation(operation)
            .with_context(|| format!("{label} advertises invalid operation {operation:?}"))?;
        if !operations.insert(operation.clone()) {
            bail!("{label} advertises duplicate operation {operation:?}");
        }
    }
    if !operations.contains(DESCRIBE_OPERATION) {
        bail!("{label} does not advertise the protocol.describe operation it just served");
    }
    let missing = required_operations
        .difference(&operations)
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "{label} does not advertise operations required by the selected cases: {}",
            missing.join(", ")
        );
    }
    Ok(ProtocolDescription {
        protocol_version: raw.protocol_version,
        transport: raw.transport.clone(),
        operations,
    })
}

fn validate_process_result(label: &str, output: &ProcessOutput) -> anyhow::Result<()> {
    if output.timed_out {
        bail!("{label} protocol preflight timed out");
    }
    if output.stdout_truncated {
        bail!("{label} protocol preflight exceeded its stdout limit");
    }
    if output.stderr_truncated {
        bail!("{label} protocol preflight exceeded its stderr limit");
    }
    if output.exit_code != Some(0) {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        let suffix = if stderr.is_empty() {
            String::new()
        } else {
            format!("; stderr: {}", bounded_text(stderr, 1024))
        };
        bail!(
            "{label} protocol preflight exited with status {:?}{suffix}",
            output.exit_code
        );
    }
    Ok(())
}

fn bounded_text(value: &str, limit: usize) -> String {
    let mut output = value.chars().take(limit).collect::<String>();
    if value.chars().count() > limit {
        output.push_str("...");
    }
    output
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use super::*;

    fn output(stdout: Value) -> ProcessOutput {
        let mut stdout = serde_json::to_vec(&stdout).unwrap();
        stdout.push(b'\n');
        ProcessOutput {
            program: PathBuf::from("producer"),
            exit_code: Some(0),
            stdout,
            stderr: Vec::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            timed_out: false,
            elapsed: Duration::from_millis(1),
        }
    }

    fn response(description: Value) -> Value {
        serde_json::json!({
            "schemaVersion": 1,
            "caseId": PREFLIGHT_CASE_ID,
            "operation": DESCRIBE_OPERATION,
            "status": "ok",
            "evidence": description,
        })
    }

    fn description(operations: &[&str]) -> Value {
        serde_json::json!({
            "protocolVersion": 1,
            "transport": "jsonl-stdin-stdout",
            "operations": operations,
        })
    }

    #[test]
    fn accepts_matching_protocol_and_required_operations() {
        let output = output(response(description(&[
            "protocol.describe",
            "sampler.apply",
        ])));
        let required = BTreeSet::from(["sampler.apply".to_owned()]);

        let actual =
            validate_protocol_description("candidate icn-probe", &output, &required).unwrap();

        assert_eq!(actual.protocol_version, 1);
        assert_eq!(actual.transport, PROBE_TRANSPORT);
        assert_eq!(actual.operations.len(), 2);
    }

    #[test]
    fn rejects_a_missing_runtime_capability() {
        let output = output(response(description(&["protocol.describe"])));
        let required = BTreeSet::from(["sampler.apply".to_owned()]);

        let error =
            validate_protocol_description("candidate icn-probe", &output, &required).unwrap_err();

        assert!(error.to_string().contains("sampler.apply"));
        assert!(error.to_string().contains("selected cases"));
    }

    #[test]
    fn rejects_protocol_or_transport_drift() {
        let wrong_version = output(response(serde_json::json!({
            "protocolVersion": 2,
            "transport": "jsonl-stdin-stdout",
            "operations": ["protocol.describe"],
        })));
        assert!(
            validate_protocol_description("oracle", &wrong_version, &BTreeSet::new())
                .unwrap_err()
                .to_string()
                .contains("protocolVersion 2")
        );

        let wrong_transport = output(response(serde_json::json!({
            "protocolVersion": 1,
            "transport": "argv-json",
            "operations": ["protocol.describe"],
        })));
        assert!(
            validate_protocol_description("oracle", &wrong_transport, &BTreeSet::new())
                .unwrap_err()
                .to_string()
                .contains("argv-json")
        );
    }

    #[test]
    fn rejects_duplicate_or_malformed_advertised_operations() {
        let duplicate = output(response(description(&[
            "protocol.describe",
            "protocol.describe",
        ])));
        assert!(
            validate_protocol_description("oracle", &duplicate, &BTreeSet::new())
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );

        let malformed = output(response(description(&[
            "protocol.describe",
            "Bad Operation",
        ])));
        let error =
            validate_protocol_description("oracle", &malformed, &BTreeSet::new()).unwrap_err();
        assert!(format!("{error:#}").contains("invalid operation"));
    }

    #[test]
    fn reports_typed_protocol_rejection() {
        let output = output(serde_json::json!({
            "schemaVersion": 1,
            "caseId": PREFLIGHT_CASE_ID,
            "operation": DESCRIBE_OPERATION,
            "status": "error",
            "error": {
                "class": "invalid-input",
                "code": "unsupported-operation",
                "message": "unsupported operation: protocol.describe",
            },
        }));

        let error =
            validate_protocol_description("legacy probe", &output, &BTreeSet::new()).unwrap_err();

        let message = error.to_string();
        assert!(message.contains("invalid-input"));
        assert!(message.contains("unsupported-operation"));
    }

    #[test]
    fn rejects_multiple_response_records() {
        let record = serde_json::to_string(&response(description(&["protocol.describe"]))).unwrap();
        let mut process = output(serde_json::Value::Null);
        process.stdout = format!("{record}\n{record}\n").into_bytes();

        let error =
            validate_protocol_description("oracle", &process, &BTreeSet::new()).unwrap_err();

        assert!(error.to_string().contains("exactly one"));
    }

    #[test]
    fn describe_request_matches_schema_v1() {
        let bytes = describe_request_jsonl().unwrap();
        assert!(bytes.ends_with(b"\n"));
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["schemaVersion"], 1);
        assert_eq!(value["caseId"], PREFLIGHT_CASE_ID);
        assert_eq!(value["operation"], DESCRIBE_OPERATION);
        assert_eq!(value["input"], serde_json::json!({}));
    }
}
