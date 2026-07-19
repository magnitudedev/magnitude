use std::collections::BTreeMap;
use std::time::Instant;

use futures_util::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

use crate::model::{
    BenchmarkError, RawStreamEvent, RequestObservation, RunOutcome, TargetConfig, TargetKind,
    TargetProbe, TimingObservation, ToolCallObservation, UsageObservation,
};

#[derive(Debug, Clone)]
pub(crate) struct ChatRequestPlan {
    pub id: String,
    pub messages: Vec<Value>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub max_tokens: u32,
    pub cache_prompt: bool,
    pub ignore_eos: bool,
    pub timings_per_token: bool,
    pub slot_id: Option<u32>,
}

#[derive(Debug, Clone)]
pub(crate) enum ClientProgress {
    FirstSemantic,
    Predicted(u64),
}

#[derive(Clone)]
pub(crate) struct EndpointClient {
    target: TargetConfig,
    client: reqwest::Client,
    timeout: std::time::Duration,
}

impl EndpointClient {
    pub(crate) fn new(target: TargetConfig, timeout_seconds: u64) -> Result<Self, BenchmarkError> {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(64)
            .tcp_nodelay(true)
            .timeout(std::time::Duration::from_secs(timeout_seconds))
            .build()?;
        Ok(Self {
            target,
            client,
            timeout: std::time::Duration::from_secs(timeout_seconds),
        })
    }

    pub(crate) fn target(&self) -> &TargetConfig {
        &self.target
    }

    pub(crate) async fn probe(&self) -> TargetProbe {
        let mut warnings = Vec::new();
        let health = self
            .get_optional(&self.url("health"))
            .await
            .map_err(|error| {
                warnings.push(format!("health probe failed: {error}"));
            })
            .ok();
        let models = self
            .get_optional(&self.url("v1/models"))
            .await
            .map_err(|error| {
                warnings.push(format!("model probe failed: {error}"));
            })
            .ok();
        let properties = match self.target.kind {
            TargetKind::Generic => None,
            TargetKind::Icn => self
                .get_optional(&self.url("v1/props"))
                .await
                .map_err(|error| {
                    warnings.push(format!("ICN properties probe failed: {error}"));
                })
                .ok(),
            TargetKind::LlamaCpp => self
                .get_optional(&self.url("props"))
                .await
                .map_err(|error| {
                    warnings.push(format!("llama.cpp properties probe failed: {error}"));
                })
                .ok(),
        };
        TargetProbe {
            health,
            models,
            properties,
            warnings,
        }
    }

    pub(crate) async fn reset_cache(&self) -> Result<bool, String> {
        match self.target.kind {
            TargetKind::Icn => Ok(true),
            TargetKind::Generic => Ok(false),
            TargetKind::LlamaCpp => {
                let props = self
                    .get_optional(&self.url("props"))
                    .await
                    .map_err(|error| error.to_string())?;
                let slots = props
                    .get("total_slots")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| "llama.cpp /props did not report total_slots".to_owned())?;
                for slot in 0..slots {
                    let url = format!("{}/slots/{slot}?action=erase", self.base_url());
                    let response = self
                        .authorize(self.client.post(url))
                        .send()
                        .await
                        .map_err(|error| error.to_string())?;
                    if !response.status().is_success() {
                        return Err(format!("slot {slot} erase returned {}", response.status()));
                    }
                }
                Ok(true)
            }
        }
    }

    pub(crate) async fn execute(
        &self,
        plan: ChatRequestPlan,
        progress: Option<mpsc::UnboundedSender<ClientProgress>>,
    ) -> RequestObservation {
        match self.execute_inner(&plan, progress).await {
            Ok(observation) => observation,
            Err(error) => RequestObservation {
                id: plan.id,
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
                output_sha256: format!("{:x}", Sha256::digest([])),
                raw_events: Vec::new(),
                error: Some(error.to_string()),
            },
        }
    }

    async fn execute_inner(
        &self,
        plan: &ChatRequestPlan,
        progress: Option<mpsc::UnboundedSender<ClientProgress>>,
    ) -> Result<RequestObservation, BenchmarkError> {
        let mut body = serde_json::Map::new();
        body.insert("model".into(), Value::String(self.target.model.clone()));
        body.insert("messages".into(), Value::Array(plan.messages.clone()));
        body.insert("stream".into(), Value::Bool(true));
        body.insert("stream_options".into(), json!({"include_usage": true}));
        body.insert("max_completion_tokens".into(), json!(plan.max_tokens));
        body.insert("temperature".into(), json!(0.0));
        body.insert("top_p".into(), json!(1.0));
        body.insert("seed".into(), json!(42));
        if !matches!(self.target.kind, TargetKind::Generic) {
            let omit_fixed_work_controls = self
                .target
                .configuration
                .get("omit_fixed_work_controls")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !omit_fixed_work_controls {
                body.insert("cache_prompt".into(), json!(plan.cache_prompt));
                body.insert("ignore_eos".into(), json!(plan.ignore_eos));
            }
            body.insert("timings_per_token".into(), json!(plan.timings_per_token));
            body.insert(
                "chat_template_kwargs".into(),
                json!({"enable_thinking": false}),
            );
        }
        if matches!(self.target.kind, TargetKind::Icn) {
            body.insert("reasoning_effort".into(), Value::String("none".into()));
        }
        if matches!(self.target.kind, TargetKind::LlamaCpp)
            && let Some(slot_id) = plan.slot_id
        {
            body.insert("id_slot".into(), json!(slot_id));
        }
        if let Some(tools) = &plan.tools {
            body.insert("tools".into(), Value::Array(tools.clone()));
        }
        if let Some(tool_choice) = &plan.tool_choice {
            body.insert("tool_choice".into(), tool_choice.clone());
        }

        let started = Instant::now();
        let response = self
            .authorize(
                self.client
                    .post(self.url("v1/chat/completions"))
                    .header(CONTENT_TYPE, "application/json")
                    .json(&Value::Object(body)),
            )
            .send()
            .await?;
        let status = response.status();
        let headers_ms = started.elapsed().as_secs_f64() * 1_000.0;
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Ok(error_observation(
                &plan.id,
                Some(status.as_u16()),
                headers_ms,
                started.elapsed().as_secs_f64() * 1_000.0,
                format!("HTTP {status}: {text}"),
            ));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::<u8>::new();
        let mut raw_events = Vec::new();
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = BTreeMap::<u64, ToolCallObservation>::new();
        let mut usage = None;
        let mut timings = None;
        let mut finish_reason = None;
        let mut first_event_ms = None;
        let mut first_semantic_ms = None;
        let mut saw_done = false;
        let mut stream_error = None;

        while let Some(chunk) = tokio::time::timeout(self.timeout, stream.next())
            .await
            .map_err(|_| {
                BenchmarkError::Endpoint(format!("request {} stream timed out", plan.id))
            })?
        {
            let chunk = chunk?;
            buffer.extend_from_slice(&chunk);
            while let Some((frame, consumed)) = next_sse_frame(&buffer) {
                buffer.drain(..consumed);
                let Some(data) = sse_data(&frame) else {
                    continue;
                };
                if data == "[DONE]" {
                    saw_done = true;
                    continue;
                }
                let payload: Value = serde_json::from_str(&data)?;
                let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
                first_event_ms.get_or_insert(elapsed_ms);
                let semantic = accumulate_payload(
                    &payload,
                    &mut content,
                    &mut reasoning,
                    &mut tool_calls,
                    &mut usage,
                    &mut timings,
                    &mut finish_reason,
                    &mut stream_error,
                );
                if semantic && first_semantic_ms.is_none() {
                    first_semantic_ms = Some(elapsed_ms);
                    if let Some(progress) = &progress {
                        let _ = progress.send(ClientProgress::FirstSemantic);
                    }
                }
                if let Some(predicted) = timings.as_ref().and_then(|value| value.predicted_n)
                    && let Some(progress) = &progress
                {
                    let _ = progress.send(ClientProgress::Predicted(predicted));
                }
                raw_events.push(RawStreamEvent {
                    elapsed_ms,
                    payload,
                });
            }
        }

        let completed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let tool_calls = tool_calls.into_values().collect::<Vec<_>>();
        let output_sha256 = output_digest(&content, &reasoning, &tool_calls);
        let outcome = if stream_error.is_some() || !saw_done {
            RunOutcome::Error
        } else {
            RunOutcome::Valid
        };
        Ok(RequestObservation {
            id: plan.id.clone(),
            outcome,
            status: Some(status.as_u16()),
            headers_ms: Some(headers_ms),
            first_event_ms,
            first_semantic_ms,
            completed_ms,
            finish_reason,
            content,
            reasoning,
            tool_calls,
            usage,
            timings,
            output_sha256,
            raw_events,
            error: stream_error
                .or_else(|| (!saw_done).then_some("stream ended without [DONE]".into())),
        })
    }

    async fn get_optional(&self, url: &str) -> Result<Value, BenchmarkError> {
        let response = self.authorize(self.client.get(url)).send().await?;
        if !response.status().is_success() {
            return Err(BenchmarkError::Endpoint(format!(
                "GET {url} returned {}",
                response.status()
            )));
        }
        Ok(response.json().await?)
    }

    fn authorize(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.target.api_key {
            Some(key) => request.header(AUTHORIZATION, format!("Bearer {key}")),
            None => request,
        }
    }

    fn base_url(&self) -> &str {
        self.target
            .endpoint
            .trim_end_matches('/')
            .trim_end_matches("/v1")
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url(), path.trim_start_matches('/'))
    }
}

fn next_sse_frame(buffer: &[u8]) -> Option<(Vec<u8>, usize)> {
    for index in 0..buffer.len().saturating_sub(1) {
        if buffer[index..].starts_with(b"\n\n") {
            return Some((buffer[..index].to_vec(), index + 2));
        }
        if buffer[index..].starts_with(b"\r\n\r\n") {
            return Some((buffer[..index].to_vec(), index + 4));
        }
    }
    None
}

fn sse_data(frame: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(frame);
    let data = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>();
    (!data.is_empty()).then(|| data.join("\n"))
}

#[allow(clippy::too_many_arguments)]
fn accumulate_payload(
    payload: &Value,
    content: &mut String,
    reasoning: &mut String,
    tool_calls: &mut BTreeMap<u64, ToolCallObservation>,
    usage: &mut Option<UsageObservation>,
    timings: &mut Option<TimingObservation>,
    finish_reason: &mut Option<String>,
    stream_error: &mut Option<String>,
) -> bool {
    if let Some(error) = payload.get("error") {
        *stream_error = Some(error.to_string());
    }
    if let Some(raw_usage) = payload.get("usage").filter(|value| !value.is_null()) {
        *usage = Some(UsageObservation {
            prompt_tokens: number(raw_usage, "prompt_tokens").unwrap_or(0),
            completion_tokens: number(raw_usage, "completion_tokens").unwrap_or(0),
            total_tokens: number(raw_usage, "total_tokens").unwrap_or(0),
        });
    }
    if let Some(raw_timings) = payload.get("timings").filter(|value| !value.is_null()) {
        *timings = Some(TimingObservation {
            cache_n: number(raw_timings, "cache_n"),
            prompt_n: number(raw_timings, "prompt_n"),
            prompt_ms: float(raw_timings, "prompt_ms"),
            prompt_per_second: float(raw_timings, "prompt_per_second"),
            predicted_n: number(raw_timings, "predicted_n"),
            predicted_ms: float(raw_timings, "predicted_ms"),
            predicted_per_second: float(raw_timings, "predicted_per_second"),
            draft_n: number(raw_timings, "draft_n"),
            draft_n_accepted: number(raw_timings, "draft_n_accepted"),
        });
    }

    let mut semantic = false;
    for choice in payload
        .get("choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            *finish_reason = Some(reason.to_owned());
        }
        let Some(delta) = choice.get("delta") else {
            continue;
        };
        if let Some(text) = delta.get("content").and_then(Value::as_str)
            && !text.is_empty()
        {
            content.push_str(text);
            semantic = true;
        }
        if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str)
            && !text.is_empty()
        {
            reasoning.push_str(text);
            semantic = true;
        }
        for raw_call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let index = raw_call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let call = tool_calls
                .entry(index)
                .or_insert_with(|| ToolCallObservation {
                    index,
                    id: String::new(),
                    name: String::new(),
                    arguments: String::new(),
                });
            if let Some(id) = raw_call.get("id").and_then(Value::as_str) {
                call.id.push_str(id);
                semantic = true;
            }
            if let Some(function) = raw_call.get("function") {
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    call.name.push_str(name);
                    semantic = true;
                }
                if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                    call.arguments.push_str(arguments);
                    semantic |= !arguments.is_empty();
                }
            }
        }
    }
    semantic
}

fn number(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

fn float(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

pub(crate) fn output_digest(
    content: &str,
    reasoning: &str,
    tool_calls: &[ToolCallObservation],
) -> String {
    let mut digest = Sha256::new();
    digest.update(content.as_bytes());
    digest.update([0]);
    digest.update(reasoning.as_bytes());
    digest.update([0]);
    let semantic_tools = tool_calls
        .iter()
        .map(|call| {
            json!({
                "index": call.index,
                "name": call.name,
                "arguments": serde_json::from_str::<Value>(&call.arguments)
                    .unwrap_or_else(|_| Value::String(call.arguments.clone())),
            })
        })
        .collect::<Vec<_>>();
    // Tool-call IDs are endpoint-generated transport handles. They must be
    // preserved in raw evidence, but they are not part of semantic equality.
    digest.update(serde_json::to_vec(&semantic_tools).unwrap_or_default());
    format!("{:x}", digest.finalize())
}

fn error_observation(
    id: &str,
    status: Option<u16>,
    headers_ms: f64,
    completed_ms: f64,
    error: String,
) -> RequestObservation {
    RequestObservation {
        id: id.to_owned(),
        outcome: RunOutcome::Error,
        status,
        headers_ms: Some(headers_ms),
        first_event_ms: None,
        first_semantic_ms: None,
        completed_ms,
        finish_reason: None,
        content: String::new(),
        reasoning: String::new(),
        tool_calls: Vec::new(),
        usage: None,
        timings: None,
        output_sha256: format!("{:x}", Sha256::digest([])),
        raw_events: Vec::new(),
        error: Some(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_complete_lf_and_crlf_sse_frames() {
        let buffer = b"data: {\"a\":1}\n\ndata: [DONE]\r\n\r\npartial";
        let (first, consumed) = next_sse_frame(buffer).unwrap();
        assert_eq!(sse_data(&first).as_deref(), Some("{\"a\":1}"));
        let (second, second_consumed) = next_sse_frame(&buffer[consumed..]).unwrap();
        assert_eq!(sse_data(&second).as_deref(), Some("[DONE]"));
        assert_eq!(&buffer[consumed + second_consumed..], b"partial");
    }

    #[test]
    fn accumulates_content_tools_usage_and_timings() {
        let payload = json!({
            "choices": [{"delta": {
                "content": "x",
                "tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "{}"}}]
            }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "timings": {"cache_n": 1, "prompt_n": 2, "predicted_n": 2}
        });
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tools = BTreeMap::new();
        let mut usage = None;
        let mut timings = None;
        let mut finish = None;
        let mut error = None;
        assert!(accumulate_payload(
            &payload,
            &mut content,
            &mut reasoning,
            &mut tools,
            &mut usage,
            &mut timings,
            &mut finish,
            &mut error,
        ));
        assert_eq!(content, "x");
        assert_eq!(tools[&0].name, "f");
        assert_eq!(usage.unwrap().total_tokens, 5);
        assert_eq!(timings.unwrap().cache_n, Some(1));
        assert_eq!(finish.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn semantic_output_digest_ignores_transport_tool_ids_and_json_formatting() {
        let left = ToolCallObservation {
            index: 0,
            id: "provider-a-id".into(),
            name: "replace_text".into(),
            arguments: r#"{"path":"src/example.ts","old_text":"OLD"}"#.into(),
        };
        let right = ToolCallObservation {
            index: 0,
            id: "provider-b-id".into(),
            name: "replace_text".into(),
            arguments: r#"{ "old_text": "OLD", "path": "src/example.ts" }"#.into(),
        };
        assert_eq!(
            output_digest("", "", &[left]),
            output_digest("", "", &[right])
        );
    }
}
