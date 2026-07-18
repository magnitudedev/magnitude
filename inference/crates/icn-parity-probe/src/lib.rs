//! Strict JSON Lines adapter for the production ICN primitive parity surface.
//!
//! The probe deliberately owns only transport concerns. Primitive semantics
//! remain in `icn-llamacpp`, so the parity suite exercises the production
//! binding path rather than a second implementation in this binary.

use std::fmt;
use std::io::{self, BufRead, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_INPUT_LINE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    case_id: String,
    operation: String,
    #[serde(default)]
    input: Map<String, Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Response {
    schema_version: u32,
    case_id: String,
    operation: String,
    status: ResponseStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    evidence: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ErrorBody>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum ResponseStatus {
    Ok,
    Error,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    class: String,
    code: String,
    message: String,
}

#[derive(Debug)]
enum RequestError {
    InvalidJson(String),
    InvalidRequest(String),
    UnsupportedSchemaVersion(u32),
}

impl RequestError {
    fn code(&self) -> &'static str {
        match self {
            Self::InvalidJson(_) => "invalid-json",
            Self::InvalidRequest(_) => "invalid-request",
            Self::UnsupportedSchemaVersion(_) => "unsupported-schema-version",
        }
    }
}

impl fmt::Display for RequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(message) | Self::InvalidRequest(message) => {
                formatter.write_str(message)
            }
            Self::UnsupportedSchemaVersion(version) => write!(
                formatter,
                "unsupported schemaVersion {version}; expected {PROTOCOL_VERSION}"
            ),
        }
    }
}

enum BoundedLine {
    Data(Vec<u8>),
    TooLarge,
}

/// Run the schema-v1 parity protocol until EOF.
///
/// Empty lines are ignored. Every other line produces exactly one JSON
/// response followed by a newline. Protocol failures are represented in that
/// response and do not terminate the loop; only input/output failures do.
pub fn run_jsonl<R: BufRead, W: Write>(mut reader: R, mut writer: W) -> io::Result<()> {
    while let Some(line) = read_bounded_line(&mut reader, MAX_INPUT_LINE_BYTES)? {
        let response = match line {
            BoundedLine::Data(line) if line.is_empty() => continue,
            BoundedLine::Data(line) => response_for_line(&line),
            BoundedLine::TooLarge => error_response(
                "",
                "",
                "invalid-input",
                "input-too-large",
                format!("input line exceeds {MAX_INPUT_LINE_BYTES} bytes"),
            ),
        };

        serde_json::to_writer(&mut writer, &response).map_err(io::Error::other)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    Ok(())
}

fn response_for_line(line: &[u8]) -> Response {
    let request = match parse_request(line) {
        Ok(request) => request,
        Err(error) => {
            return error_response("", "", "invalid-input", error.code(), error.to_string());
        }
    };

    match catch_unwind(AssertUnwindSafe(|| {
        icn_llamacpp::parity_probe::execute(&request.operation, &request.input)
    })) {
        Ok(Ok(evidence)) => Response {
            schema_version: PROTOCOL_VERSION,
            case_id: request.case_id,
            operation: request.operation,
            status: ResponseStatus::Ok,
            evidence: Some(evidence),
            error: None,
        },
        Ok(Err(error)) => error_response(
            &request.case_id,
            &request.operation,
            error.class(),
            error.code(),
            error.to_string(),
        ),
        Err(payload) => error_response(
            &request.case_id,
            &request.operation,
            "runtime-error",
            "probe-panic",
            panic_message(payload),
        ),
    }
}

fn parse_request(line: &[u8]) -> Result<Request, RequestError> {
    let request: Request = serde_json::from_slice(line).map_err(|error| {
        let message = error.to_string();
        if error.is_syntax() || error.is_eof() {
            RequestError::InvalidJson(message)
        } else {
            RequestError::InvalidRequest(message)
        }
    })?;
    if request.schema_version != PROTOCOL_VERSION {
        return Err(RequestError::UnsupportedSchemaVersion(
            request.schema_version,
        ));
    }
    Ok(request)
}

fn error_response(
    case_id: &str,
    operation: &str,
    class: &str,
    code: &str,
    message: String,
) -> Response {
    Response {
        schema_version: PROTOCOL_VERSION,
        case_id: case_id.to_owned(),
        operation: operation.to_owned(),
        status: ResponseStatus::Error,
        evidence: None,
        error: Some(ErrorBody {
            class: class.to_owned(),
            code: code.to_owned(),
            message,
        }),
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "parity probe operation panicked with a non-string payload".to_owned()
    }
}

fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    max_bytes: usize,
) -> io::Result<Option<BoundedLine>> {
    let mut line = Vec::new();
    let mut too_large = false;
    let mut saw_input = false;

    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            return if saw_input {
                Ok(Some(if too_large {
                    BoundedLine::TooLarge
                } else {
                    BoundedLine::Data(line)
                }))
            } else {
                Ok(None)
            };
        }

        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let content_bytes = newline.unwrap_or(buffer.len());
        saw_input = true;

        if !too_large {
            if content_bytes <= max_bytes.saturating_sub(line.len()) {
                line.extend_from_slice(&buffer[..content_bytes]);
            } else {
                too_large = true;
                line.clear();
            }
        }

        let consumed = content_bytes + usize::from(newline.is_some());
        reader.consume(consumed);
        if newline.is_some() {
            return Ok(Some(if too_large {
                BoundedLine::TooLarge
            } else {
                BoundedLine::Data(line)
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, Cursor};

    use serde_json::Value;

    use super::{BoundedLine, parse_request, read_bounded_line, run_jsonl};

    #[test]
    fn parses_schema_v1_envelope() {
        let request = parse_request(
            br#"{"schemaVersion":1,"caseId":"case","operation":"protocol.describe","input":{}}"#,
        )
        .unwrap();

        assert_eq!(request.case_id, "case");
        assert_eq!(request.operation, "protocol.describe");
        assert!(request.input.is_empty());
    }

    #[test]
    fn rejects_unknown_request_fields() {
        let error = parse_request(
            br#"{"schemaVersion":1,"caseId":"case","operation":"protocol.describe","input":{},"extra":true}"#,
        )
        .unwrap_err();

        assert_eq!(error.code(), "invalid-request");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn bounded_reader_discards_oversized_line_and_preserves_next_line() {
        let mut reader = BufReader::with_capacity(3, Cursor::new(b"123456789\nok\n"));

        assert!(matches!(
            read_bounded_line(&mut reader, 8).unwrap(),
            Some(BoundedLine::TooLarge)
        ));
        match read_bounded_line(&mut reader, 8).unwrap() {
            Some(BoundedLine::Data(line)) => assert_eq!(line, b"ok"),
            _ => panic!("expected the line after the oversized record"),
        }
    }

    #[test]
    fn emits_one_error_for_each_nonempty_invalid_line() {
        let input = b"\nnot-json\n   \n";
        let mut output = Vec::new();

        run_jsonl(Cursor::new(input), &mut output).unwrap();

        let responses: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(responses.len(), 2);
        assert!(responses.iter().all(|response| {
            response["status"] == "error" && response["error"]["class"] == "invalid-input"
        }));
    }

    #[test]
    fn unsupported_operation_is_a_typed_invalid_input_error() {
        let mut output = Vec::new();
        run_jsonl(
            Cursor::new(
                b"{\"schemaVersion\":1,\"caseId\":\"unknown\",\"operation\":\"does.not.exist\",\"input\":{}}\n",
            ),
            &mut output,
        )
        .unwrap();

        let response: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(response["schemaVersion"], 1);
        assert_eq!(response["caseId"], "unknown");
        assert_eq!(response["operation"], "does.not.exist");
        assert_eq!(response["status"], "error");
        assert_eq!(response["error"]["class"], "invalid-input");
        assert_eq!(response["error"]["code"], "unsupported-operation");
        assert!(
            response["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("does.not.exist"))
        );
    }
}
