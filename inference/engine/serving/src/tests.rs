//! Protocol parity fixtures, ported from the llama-based service's API tests
//! and run against a scripted model source.
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use futures_util::future::BoxFuture;
use http_body_util::BodyExt;
use magnitude_chat::output::{
    Completion, GenerationTimings, OutputEvent, Progress, Termination, TimingSnapshot, TokenUsage,
};
use magnitude_chat::reasoning::{EffortDomain, EffortMapping, ReasoningProfile};
use magnitude_chat::request::{Entry, OutputFormat, PromptCache, ToolCall, UserPart};
use magnitude_chat::{ChatError, ChatInput, EndOfGeneration, GenerationRequest, ReasoningIntent, ToolChoice};
use magnitude_engine::chat::{AppliedTemplate, ModelProperties};
use magnitude_engine::error::RequestError;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower::ServiceExt;

use crate::chat::{ChatCompletionRequest, adapt_request, validate_request};
use crate::error::{ModelUnavailable, ServingError};
use crate::source::{
    GenerationEvent, GenerationStream, HostChat, LoadProgress, ModelInvocation, ModelLoadProgress,
    ModelLoadStage, ServedModels,
};
use crate::{Serving, anthropic, responses, router};

// ---------------------------------------------------------------------------
// Scripted model source

#[derive(Clone)]
struct Script {
    /// The admitted prompt size, or the refusal before admission.
    admission: Result<u64, ServingError>,
    progress: Vec<Progress>,
    output: Vec<(OutputEvent, Option<TimingSnapshot>)>,
    /// `None` completes normally; `Some` fails after the scripted output.
    failure: Option<ServingError>,
    /// Holds the stream open after output until the consumer goes away.
    hang: bool,
    /// Leading prompt tokens the completion reports as restored from cache.
    cached_input_tokens: u64,
}

impl Script {
    fn text(text: &str) -> Self {
        Self::output(vec![OutputEvent::TextDelta(text.into())])
    }

    fn output(events: Vec<OutputEvent>) -> Self {
        Self::timed(events.into_iter().map(|event| (event, None)).collect())
    }

    fn timed(output: Vec<(OutputEvent, Option<TimingSnapshot>)>) -> Self {
        Self {
            admission: Ok(11),
            progress: Vec::new(),
            output,
            failure: None,
            hang: false,
            cached_input_tokens: 0,
        }
    }

    fn refused(error: ServingError) -> Self {
        Self {
            admission: Err(error),
            ..Self::text("unused")
        }
    }
}

fn completion(
    output: &[(OutputEvent, Option<TimingSnapshot>)],
    cached_input_tokens: u64,
) -> Completion {
    let tools = output
        .iter()
        .any(|(event, _)| matches!(event, OutputEvent::ToolCallFinished { .. }));
    Completion {
        usage: TokenUsage {
            input_tokens: 11,
            cached_input_tokens,
            output_tokens: 7,
            reasoning_output_tokens: 1,
        },
        termination: if tools {
            Termination::ToolCalls
        } else {
            Termination::Natural
        },
        timings: GenerationTimings {
            prompt_ms: 2.0,
            decode_ms: 3.0,
            time_to_first_token_ms: 4.0,
            sampler_ms: 0.5,
            parser_ms: 0.25,
            draft_tokens: 0,
            accepted_draft_tokens: 0,
        },
    }
}

#[derive(Default)]
struct Observed {
    invocations: AtomicUsize,
    requests: Mutex<Vec<(String, GenerationRequest)>>,
    counted: Mutex<Vec<ChatInput>>,
    /// Set when a consumer dropped a live generation.
    cancelled: AtomicBool,
    /// Set when a pending model binding was dropped.
    binding_dropped: AtomicBool,
}

struct Scripted {
    script: Script,
    /// The model name the source serves.
    model: &'static str,
    /// Binding never completes (models a load in progress).
    pending: bool,
    binding_failure: Option<ServingError>,
    observed: Arc<Observed>,
}

impl Scripted {
    fn new(script: Script) -> Self {
        Self {
            script,
            model: "test-model",
            pending: false,
            binding_failure: None,
            observed: Arc::default(),
        }
    }
}

struct ScriptedInvocation {
    script: Script,
    model: String,
    observed: Arc<Observed>,
}

impl ModelInvocation for ScriptedInvocation {
    fn generate(self: Box<Self>, request: GenerationRequest) -> GenerationStream {
        self.observed
            .requests
            .lock()
            .unwrap()
            .push((self.model.clone(), request));
        let (sender, receiver) = mpsc::channel(4);
        let script = self.script;
        let observed = self.observed;
        tokio::spawn(async move {
            let send = |event| {
                let sender = sender.clone();
                async move { sender.send(event).await.is_ok() }
            };
            for progress in &script.progress {
                if !send(GenerationEvent::Progress(*progress)).await {
                    return;
                }
            }
            let prompt_tokens = match script.admission {
                Ok(prompt_tokens) => prompt_tokens,
                Err(error) => {
                    send(GenerationEvent::Failed(error)).await;
                    return;
                }
            };
            if !send(GenerationEvent::Admitted { prompt_tokens }).await {
                return;
            }
            if !matches!(script.output.first(), Some((OutputEvent::Started, _)))
                && !send(GenerationEvent::Output {
                    event: OutputEvent::Started,
                    snapshot: None,
                })
                .await
            {
                return;
            }
            for (event, snapshot) in script.output.iter().cloned() {
                if !send(GenerationEvent::Output { event, snapshot }).await {
                    observed.cancelled.store(true, Ordering::Release);
                    return;
                }
            }
            if script.hang {
                sender.closed().await;
                observed.cancelled.store(true, Ordering::Release);
                return;
            }
            send(match script.failure {
                Some(error) => GenerationEvent::Failed(error),
                None => GenerationEvent::Completed(completion(
                    &script.output,
                    script.cached_input_tokens,
                )),
            })
            .await;
        });
        GenerationStream::new(receiver)
    }
}

/// Sets its flag when a pending binding future is dropped.
struct DropFlag(Arc<Observed>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.binding_dropped.store(true, Ordering::Release);
    }
}

impl ServedModels for Scripted {
    fn invoke(
        &self,
        model: &str,
        progress: Option<LoadProgress>,
    ) -> BoxFuture<'_, Result<Box<dyn ModelInvocation>, ServingError>> {
        let model = model.to_owned();
        Box::pin(async move {
            self.observed.invocations.fetch_add(1, Ordering::Relaxed);
            if model != self.model {
                return Err(ServingError::Model(ModelUnavailable::NotFound(format!(
                    "model `{model}` is not served"
                ))));
            }
            if let Some(progress) = progress {
                progress(ModelLoadProgress {
                    stage: ModelLoadStage::LoadingWeights,
                    fraction: 0.5,
                });
            }
            if self.pending {
                let _flag = DropFlag(self.observed.clone());
                std::future::pending::<()>().await;
            }
            if let Some(error) = &self.binding_failure {
                return Err(error.clone());
            }
            Ok(Box::new(ScriptedInvocation {
                script: self.script.clone(),
                model,
                observed: self.observed.clone(),
            }) as Box<dyn ModelInvocation>)
        })
    }

    fn host(&self, model: &str) -> BoxFuture<'_, Result<Arc<dyn HostChat>, ServingError>> {
        let result = if model == self.model {
            Ok(Arc::new(ScriptedHost {
                observed: self.observed.clone(),
            }) as Arc<dyn HostChat>)
        } else {
            Err(ServingError::Model(ModelUnavailable::NotFound(format!(
                "model `{model}` is not installed"
            ))))
        };
        Box::pin(std::future::ready(result))
    }
}

struct ScriptedHost {
    observed: Arc<Observed>,
}

fn toggle_profile() -> ReasoningProfile {
    ReasoningProfile {
        detector: magnitude_chat::reasoning::DETECTOR.into(),
        template_identity: "fixture".into(),
        option_context: "{}".into(),
        default_effort: Some("high".into()),
        mappings: [("none", false), ("high", true)]
            .into_iter()
            .map(|(effort, enabled)| EffortMapping {
                effort: effort.into(),
                controls: [("enable_thinking".to_owned(), json!(enabled))]
                    .into_iter()
                    .collect(),
                aliases: Vec::new(),
            })
            .collect(),
        baseline_shapes: vec![true],
        effort_domain: EffortDomain::Closed,
        supports_reasoning_output: Some(true),
        supports_preserve_reasoning: false,
    }
}

impl HostChat for ScriptedHost {
    fn count(&self, input: &ChatInput) -> Result<u64, ServingError> {
        self.observed.counted.lock().unwrap().push(input.clone());
        Ok(input.conversation.entries().len() as u64)
    }

    fn apply_template(&self, _input: &ChatInput) -> Result<AppliedTemplate, ServingError> {
        Ok(AppliedTemplate {
            prompt: "prompt".into(),
            generation_prompt: "<think>\n".into(),
            grammar: String::new(),
            preserved_tokens: Vec::new(),
            additional_stops: vec!["<|im_end|>".into()],
            supports_thinking: true,
            thinking_start_tag: Some("<think>".into()),
            thinking_end_tag: Some("</think>".into()),
            template_fingerprint: "fingerprint".into(),
        })
    }

    fn properties(&self) -> Result<ModelProperties, ServingError> {
        Ok(ModelProperties {
            model_path: "/models/test.gguf".into(),
            model_size_bytes: 42,
            name: Some("Test".into()),
            architecture: Some("qwen35".into()),
            context_tokens: 4096,
            training_context_tokens: 262_144,
            vision: false,
            chat_template: "{{ messages }}".into(),
            template_fingerprint: "fingerprint".into(),
            template_capabilities: [
                "supports_string_content",
                "supports_typed_content",
                "supports_tools",
                "supports_tool_calls",
                "supports_parallel_tool_calls",
                "supports_system_role",
                "supports_preserve_reasoning",
                "supports_reasoning_effort",
                "supports_object_arguments",
            ]
            .into_iter()
            .map(|name| (name.to_owned(), true))
            .collect(),
            tools: true,
            structured_output: true,
            reasoning: toggle_profile(),
        })
    }
}

fn app(source: Scripted) -> axum::Router {
    router(Serving::new(Arc::new(source)))
}

struct Reply {
    status: StatusCode,
    headers: axum::http::HeaderMap,
    body: String,
}

async fn send(app: axum::Router, path: &str, headers: &[(&str, &str)], body: Value) -> Reply {
    let mut request = Request::post(path).header("content-type", "application/json");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let response = app
        .oneshot(request.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    Reply {
        status,
        headers,
        body: String::from_utf8(body.to_vec()).unwrap(),
    }
}

async fn post_chat(source: Scripted, request: Value) -> (StatusCode, String) {
    let reply = send(app(source), "/v1/chat/completions", &[], request).await;
    (reply.status, reply.body)
}

const ANTHROPIC: &[(&str, &str)] = &[("anthropic-version", "2023-06-01")];
const PROGRESS: &[(&str, &str)] = &[("Magnitude-Include-Progress", "true")];

fn minimal_request() -> Value {
    json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true
    })
}

fn request_from_json(value: Value) -> ChatCompletionRequest {
    serde_json::from_value(value).expect("request must decode")
}

fn chat_request(value: Value) -> Result<GenerationRequest, crate::ApiError> {
    adapt_request(request_from_json(value)).map(|adapted| adapted.request)
}

fn stream_json(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE data must be JSON"))
        .collect()
}

fn snapshot(generated_tokens: u64, decode_ms: f64) -> TimingSnapshot {
    TimingSnapshot {
        cached_prompt_tokens: 0,
        prompt_tokens: 11,
        generated_tokens,
        timings: GenerationTimings {
            prompt_ms: 2.0,
            decode_ms,
            ..GenerationTimings::default()
        },
    }
}

fn context_overflow() -> ServingError {
    ServingError::Chat(ChatError::ContextLengthExceeded {
        prompt_tokens: 1,
        context_tokens: 1,
    })
}

// ---------------------------------------------------------------------------
// Request adaptation

#[test]
fn equivalent_wire_dialects_construct_equal_canonical_requests() {
    let chat = chat_request(json!({
        "model": "test-model",
        "messages": [
            { "role": "system", "content": "be concise" },
            { "role": "user", "content": "hello" }
        ],
        "max_tokens": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 0
    }))
    .unwrap();
    let responses = responses::adapt(
        serde_json::from_value(json!({
            "model": "test-model",
            "instructions": "be concise",
            "input": "hello",
            "max_output_tokens": 16,
            "temperature": 1.0,
            "top_p": 1.0
        }))
        .unwrap(),
    )
    .unwrap()
    .request;
    let anthropic = anthropic::adapt(
        serde_json::from_value(json!({
            "model": "test-model",
            "system": "be concise",
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 16,
            "temperature": 1.0,
            "top_p": 1.0
        }))
        .unwrap(),
    )
    .unwrap()
    .request;
    assert_eq!(chat, responses);
    assert_eq!(responses, anthropic);
}

#[test]
fn openai_adapters_do_not_invent_output_token_limits() {
    let chat = chat_request(minimal_request()).unwrap();
    assert_eq!(chat.controls.max_output_tokens, None);
    let responses = responses::adapt(
        serde_json::from_value(json!({ "model": "test-model", "input": "hello" })).unwrap(),
    )
    .unwrap()
    .request;
    assert_eq!(responses.controls.max_output_tokens, None);
}

#[test]
fn compatibility_requests_ignore_unknown_object_fields_recursively() {
    assert!(serde_json::from_value::<ChatCompletionRequest>(json!({
        "model": "test-model",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,AA==", "future_image_option": true },
                "future_content_option": true
            }],
            "future_message_option": true
        }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object"},
                "future_function_option": true
            },
            "future_tool_option": true
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": "lookup", "future_choice_name_option": true},
            "future_choice_option": true
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
                "future_schema_option": true
            },
            "future_format_option": true
        },
        "stream_options": {"include_usage": true, "future_stream_option": true},
        "preserve_thinking": true
    }))
    .is_ok());
    assert!(serde_json::from_value::<responses::ResponseCreateRequest>(json!({
        "model": "test-model",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": "hello", "future_content_option": true }],
            "future_message_option": true
        }],
        "tools": [{
            "type": "function",
            "name": "lookup",
            "parameters": {"type": "object"},
            "future_tool_option": true
        }],
        "tool_choice": { "type": "function", "name": "lookup", "future_choice_option": true },
        "reasoning": {"effort": "high", "future_reasoning_option": true},
        "text": {
            "format": {"type": "text", "future_format_option": true},
            "future_text_option": true
        },
        "future_request_option": true
    }))
    .is_ok());
    assert!(serde_json::from_value::<anthropic::MessagesRequest>(json!({
        "model": "test-model",
        "system": [{ "type": "text", "text": "system", "future_system_option": true }],
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "AA==",
                    "future_source_option": true
                },
                "future_content_option": true
            }],
            "future_message_option": true
        }],
        "max_tokens": 16,
        "tools": [{
            "name": "lookup",
            "input_schema": {"type": "object"},
            "future_tool_option": true
        }],
        "tool_choice": {"type": "auto", "future_choice_option": true},
        "thinking": { "type": "enabled", "budget_tokens": 8, "future_thinking_option": true },
        "future_request_option": true
    }))
    .is_ok());
}

#[test]
fn compatibility_requests_keep_known_shapes_strict() {
    assert!(serde_json::from_value::<ChatCompletionRequest>(json!({
        "model": "test-model",
        "messages": [{"role": "future_role", "content": "hello"}]
    }))
    .is_err());
    assert!(serde_json::from_value::<responses::ResponseContentPart>(json!({
        "type": "future_content",
        "text": "hello"
    }))
    .is_err());
    assert!(serde_json::from_value::<anthropic::ContentBlock>(json!({
        "type": "future_content",
        "text": "hello"
    }))
    .is_err());
    assert!(serde_json::from_value::<ChatCompletionRequest>(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream_options": {"include_usage": "yes"}
    }))
    .is_err());
}

#[test]
fn chat_accepts_disabled_store_but_rejects_persistence() {
    assert!(chat_request(json!({
        "model": "test-model",
        "messages": [{ "role": "user", "content": "hello" }],
        "store": false
    }))
    .is_ok());
    let error = chat_request(json!({
        "model": "test-model",
        "messages": [{ "role": "user", "content": "hello" }],
        "store": true
    }))
    .unwrap_err();
    assert_eq!(
        error.body.message,
        "store is not supported by this local runtime"
    );
}

fn assistant(request: &GenerationRequest, index: usize) -> &magnitude_chat::request::AssistantTurn {
    match &request.input.conversation.entries()[index] {
        Entry::Assistant(turn) => turn,
        Entry::User(_) => panic!("entry {index} must be an assistant turn"),
    }
}

fn user(request: &GenerationRequest, index: usize) -> &[UserPart] {
    match &request.input.conversation.entries()[index] {
        Entry::User(parts) => parts,
        Entry::Assistant(_) => panic!("entry {index} must be a user entry"),
    }
}

// OpenCode records a step that failed before producing anything as an empty
// assistant message and replays it on every later turn.
#[test]
fn chat_skips_empty_assistant_messages_in_history() {
    let request = chat_request(json!({
        "model": "test-model",
        "messages": [
            { "role": "user", "content": "read notes.txt" },
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "read the file first",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "read", "arguments": "{\"path\":\"notes.txt\"}" }
                }]
            },
            { "role": "tool", "tool_call_id": "call_1", "content": "secret PERIWINKLE" },
            { "role": "assistant", "content": "" },
            { "role": "assistant", "content": null },
            { "role": "assistant", "content": "The secret word is PERIWINKLE." },
            { "role": "user", "content": "and again?" }
        ]
    }))
    .unwrap();
    let entries = request.input.conversation.entries();
    assert!(entries.iter().all(|entry| match entry {
        Entry::Assistant(turn) => turn.text.is_some() || turn.reasoning.is_some() || !turn.tool_calls.is_empty(),
        Entry::User(_) => true,
    }));
    assert_eq!(
        assistant(&request, 1).tool_calls[0].result,
        vec![magnitude_chat::request::ToolResultPart::Text("secret PERIWINKLE".into())]
    );
}

#[test]
fn chat_accepts_empty_assistant_content_when_tool_calls_are_present() {
    let request = chat_request(json!({
        "model": "test-model",
        "messages": [
            { "role": "user", "content": "look this up" },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "lookup", "arguments": "{\"q\":\"x\"}" }
                }]
            },
            { "role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": "result" },
            { "role": "user", "content": "continue" }
        ]
    }))
    .unwrap();
    let turn = assistant(&request, 1);
    assert!(turn.text.is_none());
    assert_eq!(turn.tool_calls.len(), 1);
}

#[test]
fn chat_normalizes_permitted_empty_content_without_placeholder_text() {
    let request = chat_request(json!({
        "model": "test-model",
        "messages": [
            { "role": "system", "content": "" },
            { "role": "user", "content": "" },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "lookup", "arguments": "{}" }
                }]
            },
            { "role": "tool", "tool_call_id": "call_1", "content": "" }
        ]
    }))
    .unwrap();
    assert!(request.input.conversation.system().is_none());
    assert!(user(&request, 0).is_empty());
    let turn = assistant(&request, 1);
    assert!(turn.text.is_none());
    assert!(turn.reasoning.is_none());
    assert!(turn.tool_calls[0].result.is_empty());
}

#[test]
fn chat_combines_leading_system_and_developer_instructions() {
    let request = chat_request(json!({
        "model": "test-model",
        "messages": [
            { "role": "system", "content": "system" },
            { "role": "developer", "content": "developer" },
            { "role": "user", "content": "hello" }
        ]
    }))
    .unwrap();
    assert_eq!(
        request.input.conversation.system(),
        Some("system\ndeveloper")
    );
}

fn responses_request(value: Value) -> Result<GenerationRequest, crate::ApiError> {
    responses::adapt(serde_json::from_value(value).expect("request must decode"))
        .map(|adapted| adapted.request)
}

fn anthropic_request(value: Value) -> Result<GenerationRequest, crate::ApiError> {
    anthropic::adapt(serde_json::from_value(value).expect("request must decode"))
        .map(|adapted| adapted.request)
}

#[test]
fn responses_normalizes_permitted_empty_content_without_placeholder_text() {
    let request = responses_request(json!({
        "model": "test-model",
        "instructions": "",
        "input": [
            { "type": "message", "role": "user", "content": "" },
            { "type": "message", "role": "assistant", "content": "" },
            { "type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}" },
            { "type": "function_call_output", "id": "out_1", "call_id": "call_1", "output": "" }
        ]
    }))
    .unwrap();
    assert!(request.input.conversation.system().is_none());
    assert!(user(&request, 0).is_empty());
    assert!(assistant(&request, 1).text.is_none());
    assert!(assistant(&request, 1).tool_calls.is_empty());
    assert!(assistant(&request, 2).tool_calls[0].result.is_empty());
}

#[test]
fn responses_accepts_easy_messages_and_replayed_output_messages() {
    let request = responses_request(json!({
        "model": "test-model",
        "prompt_cache_key": "session-1",
        "input": [
            { "role": "developer", "content": "be concise" },
            { "type": "message", "role": "user", "content": "hello" },
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{ "type": "summary_text", "text": "brief thought" }]
            },
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "hi", "annotations": [] }]
            },
            { "role": "user", "content": "again" }
        ]
    }))
    .unwrap();
    assert_eq!(request.input.conversation.system(), Some("be concise"));
    assert_eq!(request.input.conversation.entries().len(), 3);
    assert_eq!(
        assistant(&request, 1).reasoning.as_deref(),
        Some("brief thought")
    );
}

#[test]
fn responses_retains_non_function_tools_without_executing_them() {
    let request = responses_request(json!({
        "model": "test-model",
        "input": "hello",
        "reasoning": { "effort": "medium", "summary": "auto" },
        "include": ["reasoning.encrypted_content"],
        "client_metadata": { "turn_id": "turn-1" },
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "parameters": { "type": "object" },
                "strict": false
            },
            {
                "type": "namespace",
                "name": "mcp__example",
                "description": "Example remote tools",
                "tools": [{ "type": "function", "name": "remote_action", "parameters": { "type": "object" } }]
            },
            { "type": "web_search", "external_web_access": true },
            { "type": "file_search", "vector_store_ids": ["vs_1"] }
        ]
    }))
    .unwrap();
    let definitions = request.input.tools.definitions();
    assert_eq!(definitions.len(), 1);
    assert_eq!(definitions[0].name, "exec_command");
}

#[test]
fn responses_rejects_malformed_function_tools_instead_of_demoting_them() {
    let error = responses_request(json!({
        "model": "test-model",
        "input": "hello",
        "tools": [{ "type": "function", "name": "broken" }]
    }))
    .unwrap_err();
    assert_eq!(error.body.message, "malformed function tool declaration");
}

fn full_output() -> (magnitude_chat::output::Output, Completion) {
    (
        magnitude_chat::output::Output {
            reasoning: Some("brief thought".into()),
            text: Some("calling a tool".into()),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "lookup".into(),
                arguments: serde_json::Map::new(),
            }],
        },
        Completion {
            usage: TokenUsage::default(),
            termination: Termination::ToolCalls,
            timings: GenerationTimings::default(),
        },
    )
}

// Replay closure: everything each protocol's projection emits must parse back
// through that protocol's input path, because clients resend emitted output
// verbatim as later history.
#[test]
fn responses_output_items_replay_as_input() {
    let (output, completion) = full_output();
    let projection = responses::adapt(
        serde_json::from_value(json!({ "model": "test-model", "input": "hi" })).unwrap(),
    )
    .unwrap()
    .projection;
    let emitted = serde_json::to_value(responses::from_result(
        "resp_test",
        1,
        "test-model",
        &projection,
        &output,
        &completion,
    ))
    .unwrap();
    let mut input = emitted["output"].as_array().unwrap().clone();
    input.push(json!({ "type": "function_call_output", "call_id": "call_1", "output": "result" }));
    input.push(json!({ "role": "user", "content": "continue" }));
    let request = responses_request(json!({ "model": "test-model", "input": input })).unwrap();
    assert_eq!(request.input.conversation.entries().len(), 3);
    assert_eq!(
        assistant(&request, 0).reasoning.as_deref(),
        Some("brief thought")
    );
}

// Codex re-serializes our reasoning item with explicit nulls for the fields
// we omit, and adds its own metadata to every item.
#[test]
fn responses_accepts_codex_replayed_items_with_null_fields() {
    let metadata = json!({ "turn_id": "turn-1" });
    let request = responses_request(json!({
        "model": "test-model",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{ "type": "input_text", "text": "be concise" }]
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "user",
                "content": [{ "type": "input_text", "text": "read notes.txt" }],
                "internal_chat_message_metadata_passthrough": metadata
            },
            {
                "type": "reasoning",
                "id": "rs_icn_1",
                "summary": [{ "type": "summary_text", "text": "read the file first" }],
                "content": null,
                "encrypted_content": null,
                "internal_chat_message_metadata_passthrough": metadata
            },
            {
                "type": "function_call",
                "id": "fc_call_1",
                "name": "exec_command",
                "arguments": "{\"cmd\":\"cat notes.txt\"}",
                "call_id": "call_1",
                "internal_chat_message_metadata_passthrough": metadata
            },
            {
                "type": "function_call_output",
                "id": "fco_1",
                "call_id": "call_1",
                "output": "secret PERIWINKLE",
                "internal_chat_message_metadata_passthrough": metadata
            },
            {
                "type": "reasoning",
                "id": "rs_icn_2",
                "summary": null,
                "content": null
            }
        ]
    }))
    .unwrap();
    assert_eq!(
        assistant(&request, 1).reasoning.as_deref(),
        Some("read the file first")
    );
    assert_eq!(
        assistant(&request, 1).tool_calls[0].result,
        vec![magnitude_chat::request::ToolResultPart::Text("secret PERIWINKLE".into())]
    );
}

#[test]
fn anthropic_output_blocks_replay_as_input() {
    let (output, completion) = full_output();
    let response = anthropic::message("msg_test", "test-model", &output, &completion);
    let content = serde_json::to_value(&response.content).unwrap();
    let request = anthropic_request(json!({
        "model": "test-model",
        "max_tokens": 16,
        "messages": [
            { "role": "user", "content": "hello" },
            { "role": "assistant", "content": content },
            { "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": "call_1", "content": "result" }
            ] }
        ]
    }))
    .unwrap();
    assert_eq!(request.input.conversation.entries().len(), 2);
    assert_eq!(
        assistant(&request, 1).reasoning.as_deref(),
        Some("brief thought")
    );
    assert_eq!(assistant(&request, 1).tool_calls.len(), 1);
}

#[test]
fn chat_output_message_replays_as_input() {
    let (output, completion) = full_output();
    let response = serde_json::to_value(crate::chat::chat_completion_response(
        "chatcmpl_test".into(),
        1,
        "test-model".into(),
        output,
        &completion,
    ))
    .unwrap();
    let message = response["choices"][0]["message"].clone();
    let request = chat_request(json!({
        "model": "test-model",
        "messages": [
            { "role": "user", "content": "hello" },
            message,
            { "role": "tool", "tool_call_id": "call_1", "content": "result" },
            { "role": "user", "content": "continue" }
        ]
    }))
    .unwrap();
    assert_eq!(
        assistant(&request, 1).reasoning.as_deref(),
        Some("brief thought")
    );
    assert_eq!(assistant(&request, 1).tool_calls.len(), 1);
}

#[test]
fn anthropic_maps_omitted_tool_result_content_to_an_empty_result() {
    let request = anthropic_request(json!({
        "model": "test-model",
        "max_tokens": 16,
        "output_config": { "effort": "high" },
        "system": [{ "type": "text", "text": "be concise", "cache_control": { "type": "ephemeral" } }],
        "messages": [
            { "role": "user", "content": [{
                "type": "text", "text": "look this up", "cache_control": { "type": "ephemeral" }
            }] },
            { "role": "assistant", "content": [{
                "type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}
            }] },
            { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "toolu_1" }] }
        ]
    }))
    .unwrap();
    assert!(assistant(&request, 1).tool_calls[0].result.is_empty());
}

#[test]
fn anthropic_thinking_maps_to_reasoning_intent_and_hard_budget() {
    let request = |thinking: Value, effort: Option<&str>| {
        let mut value = json!({
            "model": "test-model",
            "max_tokens": 16,
            "thinking": thinking,
            "messages": [{ "role": "user", "content": "hello" }]
        });
        if let Some(effort) = effort {
            value["output_config"] = json!({ "effort": effort });
        }
        anthropic_request(value)
    };
    let enabled = request(json!({"type": "enabled", "budget_tokens": 64}), None).unwrap();
    assert_eq!(enabled.input.reasoning, ReasoningIntent::Enabled);
    assert_eq!(enabled.controls.reasoning_budget.map(|b| b.get()), Some(64));
    let effort = request(json!({"type": "enabled", "budget_tokens": 8}), Some("low")).unwrap();
    assert_eq!(
        effort.input.reasoning,
        ReasoningIntent::Effort {
            effort: "low".into()
        }
    );
    assert_eq!(effort.controls.reasoning_budget.map(|b| b.get()), Some(8));
    let disabled = request(json!({"type": "disabled"}), None).unwrap();
    assert_eq!(disabled.input.reasoning, ReasoningIntent::Disabled);
    let adaptive = request(json!({"type": "adaptive"}), None).unwrap();
    assert_eq!(adaptive.input.reasoning, ReasoningIntent::ModelDefault);
    assert!(adaptive.controls.reasoning_budget.is_none());
    assert!(request(json!({"type": "disabled"}), Some("high")).is_err());
    assert!(request(json!({"type": "adaptive"}), Some("none")).is_err());
    assert!(request(json!({"type": "enabled", "budget_tokens": 0}), None).is_err());
}

#[test]
fn anthropic_maps_system_role_messages_to_positioned_user_entries() {
    let request = anthropic_request(json!({
        "model": "test-model",
        "max_tokens": 16,
        "system": "base instructions",
        "messages": [
            { "role": "user", "content": "hello" },
            { "role": "system", "content": [{ "type": "text", "text": "runtime instructions" }] },
            { "role": "assistant", "content": "hi" }
        ]
    }))
    .unwrap();
    assert_eq!(request.input.conversation.system(), Some("base instructions"));
    assert_eq!(request.input.conversation.entries().len(), 3);
    assert_eq!(
        user(&request, 1),
        [UserPart::Text("runtime instructions".into())]
    );
}

fn anthropic_system(system: Value) -> Option<String> {
    anthropic_request(json!({
        "model": "test-model",
        "max_tokens": 16,
        "system": system,
        "messages": [{ "role": "user", "content": "hello" }]
    }))
    .unwrap()
    .input
    .conversation
    .system()
    .map(str::to_owned)
}

#[test]
fn anthropic_strips_claude_code_attribution_before_model_context() {
    assert_eq!(
        anthropic_system(json!(
            "x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;You are Claude Code."
        ))
        .as_deref(),
        Some("You are Claude Code.")
    );
    assert_eq!(
        anthropic_system(json!([
            { "type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.181; cc_entrypoint=cli; cch=a5145;" },
            { "type": "text", "text": "You are Claude Code." }
        ]))
        .as_deref(),
        Some("You are Claude Code.")
    );
    assert_eq!(
        anthropic_system(json!("x-anthropic-billing-header: cch=a5145;")),
        None
    );
}

#[test]
fn anthropic_attribution_stripping_is_nonce_invariant_and_idempotent() {
    let system = |cch: &str| {
        anthropic_system(json!(format!(
            "x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch={cch};You are Claude Code."
        )))
    };
    let first = system("a5145");
    assert_eq!(first, system("0beef"));
    assert_eq!(anthropic_system(json!(first.clone().unwrap())), first);
}

#[test]
fn anthropic_attribution_recognition_is_strictly_positional() {
    assert_eq!(
        anthropic_system(json!([
            { "type": "text", "text": "Real instructions." },
            { "type": "text", "text": "x-anthropic-billing-header: cch=a5145;" }
        ]))
        .as_deref(),
        Some("Real instructions.\nx-anthropic-billing-header: cch=a5145;")
    );
    let mention = "Mention of x-anthropic-billing-header: cch=a5145; in prose.";
    assert_eq!(anthropic_system(json!(mention)).as_deref(), Some(mention));
    let malformed = "x-anthropic-billing-header: cc_version=2.1.101; no stamp";
    assert_eq!(anthropic_system(json!(malformed)).as_deref(), Some(malformed));
}

// Replay closure: an empty generation is emitted as an empty assistant turn in
// each protocol's native shape, so that shape must replay, as nothing.
#[test]
fn empty_assistant_turns_replay_as_nothing() {
    for content in [Value::Null, json!("")] {
        let request = chat_request(json!({
            "model": "test-model",
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": content },
                { "role": "user", "content": "again" }
            ]
        }))
        .unwrap();
        assert!(request
            .input
            .conversation
            .entries()
            .iter()
            .all(|entry| matches!(entry, Entry::User(_))));
    }
    let request = anthropic_request(json!({
        "model": "test-model",
        "max_tokens": 16,
        "messages": [
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": [] },
            { "role": "user", "content": "again" }
        ]
    }))
    .unwrap();
    assert!(request
        .input
        .conversation
        .entries()
        .iter()
        .all(|entry| matches!(entry, Entry::User(_))));
}

#[test]
fn empty_canonical_output_uses_each_protocols_native_empty_shape() {
    let output = magnitude_chat::output::Output::default();
    let completion = Completion {
        usage: TokenUsage::default(),
        termination: Termination::Natural,
        timings: GenerationTimings::default(),
    };
    let chat = serde_json::to_value(crate::chat::chat_completion_response(
        "chatcmpl_test".into(),
        1,
        "test-model".into(),
        output.clone(),
        &completion,
    ))
    .unwrap();
    assert!(chat["choices"][0]["message"]["content"].is_null());
    assert!(chat["choices"][0]["message"].get("tool_calls").is_none());
    let projection = responses::adapt(
        serde_json::from_value(json!({ "model": "test-model", "input": "hi" })).unwrap(),
    )
    .unwrap()
    .projection;
    let responses = serde_json::to_value(responses::from_result(
        "resp_test",
        1,
        "test-model",
        &projection,
        &output,
        &completion,
    ))
    .unwrap();
    assert_eq!(responses["output"], json!([]));
    let anthropic =
        serde_json::to_value(anthropic::message("msg_test", "test-model", &output, &completion))
            .unwrap();
    assert_eq!(anthropic["content"], json!([]));
}

#[test]
fn maps_the_complete_chat_request_contract() {
    let adapted = adapt_request(request_from_json(json!({
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
            ]},
            {
                "role": "assistant",
                "content": null,
                "reasoning_content": "because",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"}
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "lookup",
                "description": "Look something up",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                "strict": true
            }},
            {"type": "function", "function": { "name": "other", "parameters": {"type": "object"} }}
        ],
        "tool_choice": {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "required",
                "tools": [{"type": "function", "function": {"name": "lookup"}}]
            }
        },
        "parallel_tool_calls": false,
        "reasoning_effort": "high",
        "thinking_budget_tokens": 64,
        "response_format": {
            "type": "json_schema",
            "json_schema": { "name": "answer", "strict": true, "schema": {"type": "object", "required": ["ok"]} }
        },
        "chat_template_kwargs": {"custom": 7},
        "stop": ["END", "STOP"],
        "max_completion_tokens": 99,
        "temperature": 0.25,
        "top_p": 0.75,
        "top_k": 20,
        "seed": 9,
        "stream": true,
        "stream_options": {"include_usage": true},
        "cache_prompt": false,
        "ignore_eos": true,
        "timings_per_token": true
    })))
    .unwrap();
    assert!(adapted.include_usage);
    assert!(adapted.timings_per_token);
    let request = adapted.request;
    assert_eq!(request.input.conversation.system(), Some("system"));
    assert_eq!(request.input.conversation.entries().len(), 2);
    assert!(matches!(
        user(&request, 0),
        [UserPart::Text(text), UserPart::Image(image)]
            if text == "look" && image.media_type == "image/png"
    ));
    let turn = assistant(&request, 1);
    assert_eq!(turn.reasoning.as_deref(), Some("because"));
    assert_eq!(turn.tool_calls[0].call.name, "lookup");
    assert_eq!(turn.tool_calls[0].call.id, "call-1");
    assert_eq!(request.input.tools.definitions().len(), 2);
    assert_eq!(
        request.input.tools.choice(),
        &ToolChoice::Allowed {
            names: vec!["lookup".into()],
            required: true
        }
    );
    assert!(!request.input.tools.parallel());
    assert_eq!(
        request.input.reasoning,
        ReasoningIntent::Effort {
            effort: "high".into()
        }
    );
    assert_eq!(request.controls.reasoning_budget.map(|b| b.get()), Some(64));
    assert_eq!(request.input.template_arguments["custom"], json!(7));
    assert!(matches!(
        &request.input.output,
        OutputFormat::JsonSchema { name, schema }
            if name == "answer" && schema.source()["type"] == "object"
    ));
    assert_eq!(request.controls.stops, ["END", "STOP"]);
    assert_eq!(request.controls.max_output_tokens.map(|n| n.get()), Some(99));
    assert_eq!(request.controls.sampling.temperature, 0.25);
    assert_eq!(request.controls.sampling.top_p, 0.75);
    assert_eq!(request.controls.sampling.top_k, 20);
    assert_eq!(request.controls.sampling.seed, 9);
    assert_eq!(request.controls.prompt_cache, PromptCache::Disabled);
    assert_eq!(request.controls.end_of_generation, EndOfGeneration::Suppress);
}

#[test]
fn preserves_model_defaults_when_optional_controls_are_omitted() {
    let adapted = adapt_request(request_from_json(minimal_request())).unwrap();
    assert!(!adapted.include_usage);
    let request = adapted.request;
    assert_eq!(request.input.tools.choice(), &ToolChoice::Auto);
    assert!(request.input.tools.parallel());
    assert_eq!(request.input.reasoning, ReasoningIntent::ModelDefault);
    assert!(request.controls.reasoning_budget.is_none());
    assert_eq!(request.input.output, OutputFormat::Text);
    assert!(request.input.template_arguments.is_empty());
    assert!(request.controls.stops.is_empty());
    assert_eq!(request.controls.prompt_cache, PromptCache::Allowed);
    assert_eq!(request.controls.end_of_generation, EndOfGeneration::Stop);
    assert_eq!(request.controls.sampling.temperature, 0.8);
    assert_eq!(request.controls.sampling.top_p, 0.95);
    assert_eq!(request.controls.sampling.seed, 42);
}

#[test]
fn rejects_network_image_urls_before_the_executor() {
    let mut request = minimal_request();
    request["messages"] = json!([{
        "role": "user",
        "content": [{ "type": "image_url", "image_url": {"url": "https://example.invalid/image.png"} }]
    }]);
    let error = chat_request(request).unwrap_err();
    assert!(error.body.message.contains("network URLs are not supported"));
}

#[test]
fn rejects_partial_or_separated_tool_exchanges_before_model_admission() {
    let call = json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"}
        }]
    });
    for messages in [
        json!([{"role": "user", "content": "find it"}, call.clone()]),
        json!([
            {"role": "user", "content": "find it"},
            call,
            {"role": "user", "content": "interrupt"},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"}
        ]),
    ] {
        let mut request = minimal_request();
        request["messages"] = messages;
        let error = chat_request(request).unwrap_err();
        assert!(error.body.message.contains("no immediately following result"));
    }
}

#[test]
fn history_preservation_does_not_conflict_with_reasoning_effort() {
    let mut request = minimal_request();
    request["reasoning_effort"] = json!("high");
    request["chat_template_kwargs"] = json!({"preserve_thinking": false});
    let request = chat_request(request).unwrap();
    assert_eq!(
        request.input.reasoning,
        ReasoningIntent::Effort {
            effort: "high".into()
        }
    );
    assert_eq!(
        request.input.template_arguments["preserve_thinking"],
        json!(false)
    );
}

#[test]
fn timing_control_accepts_tolerant_boolean_semantics() {
    for value in [
        Value::Null,
        json!(false),
        json!("true"),
        json!(1),
        json!({"enabled": true}),
    ] {
        let mut request = minimal_request();
        request["timings_per_token"] = value;
        assert!(!validate_request(request_from_json(request)).unwrap().timings_per_token);
    }
    let mut request = minimal_request();
    request["timings_per_token"] = json!(true);
    assert!(validate_request(request_from_json(request)).unwrap().timings_per_token);
}

#[test]
fn maps_grammar_response_format() {
    let mut request = minimal_request();
    request["response_format"] = json!({ "type": "grammar", "grammar": "root ::= \"yes\" | \"no\"" });
    assert_eq!(
        chat_request(request).unwrap().input.output,
        OutputFormat::Grammar("root ::= \"yes\" | \"no\"".into())
    );
}

#[test]
fn rejects_conflicting_or_lossy_request_controls() {
    let error = |patch: Value| {
        let mut request = minimal_request();
        for (key, value) in patch.as_object().unwrap() {
            request[key] = value.clone();
        }
        chat_request(request).unwrap_err().body.message
    };
    assert!(error(json!({"reasoning_effort": "none", "thinking_budget_tokens": 10}))
        .contains("reasoning is disabled"));
    assert!(error(json!({"chat_template_kwargs": {"enable_thinking": false}, "thinking_budget_tokens": 10}))
        .contains("disable reasoning"));
    assert!(error(json!({"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": true}}))
        .contains("conflicts"));
    assert!(error(json!({
        "tools": [{"type": "function", "function": { "name": "known", "parameters": {"type": "object"} }}],
        "tool_choice": { "type": "function", "function": {"name": "missing"} }
    }))
    .contains("undefined tool"));
    assert!(error(json!({"response_format": { "type": "json_schema", "json_schema": {"name": "bad", "schema": 42} }}))
        .contains("JSON Schema"));
    assert!(error(json!({"response_format": {"type": "grammar", "grammar": ""}}))
        .contains("grammar must not be empty"));
    assert!(error(json!({"stop": ["END", "END"]})).contains("duplicate stop"));
    assert!(error(json!({"n": 2})).contains("n = 1"));
    assert!(error(json!({"top_p": 0})).contains("top_p"));
    assert!(error(json!({"reasoning_effort": "ultra"})).contains("reasoning_effort"));
}

#[test]
fn normalizes_disabled_aliases_to_the_none_effort() {
    let mut request = minimal_request();
    request["reasoning_effort"] = json!("off");
    assert_eq!(
        chat_request(request).unwrap().input.reasoning,
        ReasoningIntent::Effort {
            effort: "none".into()
        }
    );
}

// ---------------------------------------------------------------------------
// HTTP behavior

#[tokio::test]
async fn request_bodies_larger_than_the_axum_default_limit_reach_the_handler() {
    let oversized = "x".repeat(3 * 1024 * 1024);
    assert!(oversized.len() < crate::MAX_HTTP_BODY_BYTES);
    let (status, _) = post_chat(
        Scripted::new(Script::text("hello")),
        json!({ "model": "test-model", "messages": [{ "role": "user", "content": oversized }] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn fake_model_serves_openai_compatible_sse() {
    let reply = send(
        app(Scripted::new(Script::text("hello world"))),
        "/v1/chat/completions",
        &[],
        minimal_request(),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(reply.headers["content-type"], "text/event-stream");
    assert!(reply.headers.contains_key("x-request-id"));
    assert!(reply.body.contains("chat.completion.chunk"));
    assert!(reply.body.contains("data: [DONE]"));
}

#[tokio::test]
async fn chat_defaults_to_one_non_streaming_completion() {
    let reply = send(
        app(Scripted::new(Script::text("hello world"))),
        "/v1/chat/completions",
        &[],
        json!({ "model": "test-model", "messages": [{"role": "user", "content": "hi"}] }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(reply.headers["content-type"], "application/json");
    let body: Value = serde_json::from_str(&reply.body).unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 11);
    assert_eq!(body["usage"]["completion_tokens"], 7);
    assert_eq!(body["timings"]["predicted_n"], 7);
}

#[tokio::test]
async fn streams_cumulative_timings_on_group_terminal_deltas() {
    let script = Script::timed(vec![
        (OutputEvent::ReasoningDelta("buffered group prefix".into()), None),
        (
            OutputEvent::TextDelta("first group end".into()),
            Some(snapshot(1, 0.001)),
        ),
        (
            OutputEvent::TextDelta("second group".into()),
            Some(snapshot(2, 4.0)),
        ),
    ]);
    let mut request = minimal_request();
    request["timings_per_token"] = json!(true);
    let (status, body) = post_chat(Scripted::new(script), request).await;
    assert_eq!(status, StatusCode::OK);
    let chunks = stream_json(&body);
    assert_eq!(chunks.len(), 5);
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert!(chunks[0].get("timings").is_none());
    assert_eq!(
        chunks[1]["choices"][0]["delta"]["reasoning_content"],
        "buffered group prefix"
    );
    assert!(chunks[1].get("timings").is_none());
    assert_eq!(chunks[2]["timings"]["predicted_n"], 1);
    assert_eq!(chunks[2]["timings"]["predicted_ms"], 0.001);
    assert_eq!(chunks[3]["timings"]["predicted_n"], 2);
    assert_eq!(chunks[3]["timings"]["predicted_ms"], 4.0);
    let terminal = &chunks[4];
    assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
    assert_eq!(terminal["timings"]["predicted_n"], 7);
    let fields = terminal["timings"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        fields,
        BTreeSet::from([
            "cache_n",
            "parser_ms",
            "predicted_ms",
            "predicted_n",
            "predicted_per_second",
            "predicted_per_token_ms",
            "prompt_ms",
            "prompt_n",
            "prompt_per_second",
            "prompt_per_token_ms",
            "sampler_ms",
            "time_to_first_token_ms",
        ])
    );
    assert_eq!(terminal["timings"]["cache_n"], 0);
    assert_eq!(terminal["timings"]["prompt_n"], 11);
    assert_eq!(terminal["timings"]["prompt_per_second"], 5_500.0);
    assert_eq!(terminal["timings"]["time_to_first_token_ms"], 4.0);
}

#[tokio::test]
async fn first_sample_without_semantic_delta_attaches_timing_to_role() {
    let script = Script::timed(vec![(OutputEvent::Started, Some(snapshot(1, 0.001)))]);
    let mut request = minimal_request();
    request["timings_per_token"] = json!(true);
    request["stream_options"] = json!({"include_usage": true});
    let (status, body) = post_chat(Scripted::new(script), request).await;
    assert_eq!(status, StatusCode::OK);
    let chunks = stream_json(&body);
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert!(chunks[0]["choices"][0]["delta"]["content"].is_null());
    assert_eq!(chunks[0]["timings"]["predicted_n"], 1);
    assert_eq!(chunks[1]["choices"][0]["finish_reason"], "stop");
    assert!(chunks[1].get("timings").is_none());
    assert_eq!(chunks[2]["choices"], json!([]));
    assert_eq!(chunks[2]["timings"]["predicted_n"], 7);
}

#[tokio::test]
async fn start_timing_is_kept_when_per_token_timings_are_off() {
    let script = Script::timed(vec![(OutputEvent::Started, Some(snapshot(1, 0.001)))]);
    let (_, body) = post_chat(Scripted::new(script), minimal_request()).await;
    let chunks = stream_json(&body);
    assert_eq!(chunks[0]["timings"]["predicted_n"], 1);
    assert_eq!(chunks[1]["timings"]["predicted_n"], 7);

    let script = Script::timed(vec![(
        OutputEvent::TextDelta("answer".into()),
        Some(snapshot(1, 0.5)),
    )]);
    let (_, body) = post_chat(Scripted::new(script), minimal_request()).await;
    let chunks = stream_json(&body);
    assert!(chunks[0].get("timings").is_none());
    assert!(chunks[1].get("timings").is_none());
    assert_eq!(chunks[2]["timings"]["predicted_n"], 7);
}

fn tool_call_output() -> Vec<OutputEvent> {
    vec![
        OutputEvent::ReasoningDelta("thought".into()),
        OutputEvent::TextDelta("answer".into()),
        OutputEvent::ToolCallStarted {
            index: 0,
            id: "call-1".into(),
            name: "lookup".into(),
        },
        OutputEvent::ToolInputDelta {
            index: 0,
            fragment: "{}".into(),
        },
        OutputEvent::ToolCallFinished { index: 0 },
    ]
}

#[tokio::test]
async fn streams_reasoning_content_tool_calls_finish_usage_and_timings() {
    let mut request = minimal_request();
    request["stream_options"] = json!({"include_usage": true});
    let (status, body) =
        post_chat(Scripted::new(Script::output(tool_call_output())), request).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("data: [DONE]"));
    let chunks = stream_json(&body);
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[1]["choices"][0]["delta"]["reasoning_content"], "thought");
    assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "answer");
    assert_eq!(chunks[3]["choices"][0]["delta"]["tool_calls"][0]["id"], "call-1");
    assert_eq!(
        chunks[3]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert_eq!(chunks[5]["choices"][0]["finish_reason"], "tool_calls");
    assert!(chunks[5].get("timings").is_none());
    assert_eq!(chunks[6]["choices"], json!([]));
    assert_eq!(chunks[6]["usage"]["prompt_tokens"], 11);
    assert_eq!(chunks[6]["usage"]["completion_tokens"], 7);
    assert_eq!(chunks[6]["usage"]["total_tokens"], 18);
    assert_eq!(chunks[6]["timings"]["prompt_ms"], 2.0);
    assert_eq!(chunks[6]["timings"]["predicted_per_second"], 7_000.0 / 3.0);
}

#[tokio::test]
async fn non_streaming_chat_assembles_tool_calls_from_the_same_events() {
    let (status, body) = post_chat(
        Scripted::new(Script::output(tool_call_output())),
        json!({ "model": "test-model", "messages": [{"role": "user", "content": "hi"}] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body: Value = serde_json::from_str(&body).unwrap();
    let message = &body["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], "thought");
    assert_eq!(message["content"], "answer");
    assert_eq!(message["tool_calls"][0]["function"]["arguments"], "{}");
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn failure_after_commit_is_an_explicit_stream_error_without_success_sentinel() {
    let mut script = Script::text("partial");
    script.failure = Some(ServingError::Request(RequestError::DeviceLost {
        reason: "scripted failure".into(),
    }));
    let (status, body) = post_chat(Scripted::new(script), minimal_request()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("event: error"));
    assert!(!body.contains("data: [DONE]"));
    let error = stream_json(&body).pop().unwrap();
    assert_eq!(error["error"]["type"], "server_error");
    assert_eq!(error["error"]["code"], "device_lost");
    assert!(error["error"]["message"]
        .as_str()
        .unwrap()
        .contains("scripted failure"));
}

#[tokio::test]
async fn streaming_context_overflow_uses_protocol_native_http_errors() {
    let (status, body) = post_chat(Scripted::new(Script::refused(context_overflow())), minimal_request()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap(),
        json!({
            "error": {
                "message": "prompt is too long: 1 tokens leave no generation capacity in a 1-token context",
                "type": "invalid_request_error",
                "param": "messages",
                "code": "context_length_exceeded"
            }
        })
    );

    let reply = send(
        app(Scripted::new(Script::refused(context_overflow()))),
        "/v1/responses",
        &[],
        json!({ "model": "test-model", "input": "hi", "stream": true }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_str(&reply.body).unwrap();
    assert_eq!(body["error"]["code"], "context_length_exceeded");
    assert_eq!(body["error"]["param"], "input");

    let reply = send(
        app(Scripted::new(Script::refused(context_overflow()))),
        "/anthropic/v1/messages",
        ANTHROPIC,
        json!({
            "model": "test-model",
            "max_tokens": 32,
            "stream": true,
            "messages": [{ "role": "user", "content": "hi" }]
        }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_str(&reply.body).unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(
        body["error"]["message"],
        "prompt is too long: 1 tokens leave no generation capacity in a 1-token context"
    );
    assert!(body["request_id"].as_str().is_some());
    assert!(reply.headers.contains_key("request-id"));
}

#[tokio::test]
async fn engine_admission_refusals_map_to_integration_error_codes() {
    for (error, status, code) in [
        (RequestError::Overloaded, StatusCode::SERVICE_UNAVAILABLE, "overloaded"),
        (RequestError::MemoryReclaim, StatusCode::SERVICE_UNAVAILABLE, "memory_pressure"),
        (
            RequestError::InsufficientMemory(magnitude_engine::error::InsufficientMemory {
                required: 2,
                available: 1,
            }),
            StatusCode::SERVICE_UNAVAILABLE,
            "insufficient_memory",
        ),
    ] {
        let (reply_status, body) = post_chat(
            Scripted::new(Script::refused(ServingError::Request(error))),
            minimal_request(),
        )
        .await;
        assert_eq!(reply_status, status);
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["error"]["code"], code);
    }
}

#[tokio::test]
async fn explicit_progress_streams_report_context_overflow_in_stream() {
    let reply = send(
        app(Scripted::new(Script::refused(context_overflow()))),
        "/v1/chat/completions",
        PROGRESS,
        minimal_request(),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    assert!(reply.body.contains("event: error"));
    assert!(reply.body.contains("context_length_exceeded"));
    assert!(!reply.body.contains("data: [DONE]"));

    let reply = send(
        app(Scripted::new(Script::refused(context_overflow()))),
        "/v1/responses",
        PROGRESS,
        json!({ "model": "test-model", "input": "hi", "stream": true }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    assert!(reply.body.contains("response.failed"));
    assert!(reply.body.contains("context_length_exceeded"));
}

#[tokio::test]
async fn chat_progress_is_present_only_when_explicitly_requested() {
    let script = || {
        let mut script = Script::text("hello");
        script.progress = vec![Progress::Preparing, Progress::Queued];
        script
    };
    let (status, body) = post_chat(Scripted::new(script()), minimal_request()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(stream_json(&body)
        .iter()
        .all(|chunk| chunk.get("progress").is_none()));

    let reply = send(
        app(Scripted::new(script())),
        "/v1/chat/completions",
        PROGRESS,
        minimal_request(),
    )
    .await;
    let chunks = stream_json(&reply.body);
    assert!(chunks.iter().any(|chunk| chunk["progress"]
        == json!({ "phase": "model_loading", "stage": "loading_weights", "fraction": 0.5 })));
    assert!(chunks
        .iter()
        .any(|chunk| chunk["progress"]["phase"] == "queued" && chunk["choices"] == json!([])));
}

#[tokio::test]
async fn responses_progress_reports_loading_and_queue_events() {
    let mut script = Script::text("hello");
    script.progress = vec![Progress::Queued];
    let reply = send(
        app(Scripted::new(script)),
        "/v1/responses",
        PROGRESS,
        json!({ "model": "test-model", "input": "hi", "stream": true }),
    )
    .await;
    // The same loading progress as Chat Completions.
    assert!(reply
        .body
        .contains(r#""progress":{"phase":"model_loading","stage":"loading_weights","fraction":0.5}"#));
    assert!(reply.body.contains("\"phase\":\"queued\""));
    assert!(reply.body.contains("event: response.completed"));
}

#[tokio::test]
async fn every_local_protocol_forwards_a_normalized_reasoning_effort() {
    let source = Scripted::new(Script::text("hello"));
    let observed = source.observed.clone();
    let app = app(source);
    let (status, _) = {
        let reply = send(
            app.clone(),
            "/v1/chat/completions",
            &[],
            json!({ "model": "test-model", "reasoning_effort": "extra-high", "messages": [{ "role": "user", "content": "hi" }] }),
        )
        .await;
        (reply.status, reply.body)
    };
    assert_eq!(status, StatusCode::OK);
    let reply = send(
        app.clone(),
        "/v1/responses",
        &[],
        json!({ "model": "test-model", "reasoning": { "effort": "medium" }, "input": "hi" }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    let reply = send(
        app,
        "/anthropic/v1/messages",
        ANTHROPIC,
        json!({
            "model": "test-model",
            "max_tokens": 32,
            "output_config": { "effort": "medium" },
            "messages": [{ "role": "user", "content": "hi" }]
        }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    let requests = observed.requests.lock().unwrap();
    let efforts = requests
        .iter()
        .map(|(_, request)| request.input.reasoning.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        efforts,
        [
            ReasoningIntent::Effort { effort: "xhigh".into() },
            ReasoningIntent::Effort { effort: "medium".into() },
            ReasoningIntent::Effort { effort: "medium".into() },
        ]
    );
}

#[tokio::test]
async fn responses_stream_uses_the_same_model_pipeline() {
    let reply = send(
        app(Scripted::new(Script::text("hello"))),
        "/v1/responses",
        &[],
        json!({ "model": "test-model", "input": "hi", "stream": true }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    assert!(reply.body.contains("event: response.created"));
    assert!(reply.body.contains("event: response.in_progress"));
    assert!(reply.body.contains("event: response.output_text.delta"));
    assert!(reply.body.contains("event: response.completed"));
    assert!(reply.body.contains("\"sequence_number\":0"));
}

#[tokio::test]
async fn responses_supports_typed_non_streaming_requests() {
    let reply = send(
        app(Scripted::new(Script::text("hello"))),
        "/v1/responses",
        &[],
        json!({
            "model": "test-model",
            "instructions": "answer precisely",
            "max_output_tokens": 17,
            "temperature": 0.2,
            "top_p": 0.8,
            "parallel_tool_calls": false,
            "metadata": { "trace": "test" },
            "tools": [{
                "type": "function",
                "name": "lookup",
                "description": "Look up a value",
                "parameters": { "type": "object" },
                "strict": true
            }],
            "tool_choice": "required",
            "input": [{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] }]
        }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    let body: Value = serde_json::from_str(&reply.body).unwrap();
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["output"][0]["content"][0]["text"], "hello");
    assert_eq!(body["usage"]["output_tokens"], 7);
    assert_eq!(body["usage"]["output_tokens_details"]["reasoning_tokens"], 1);
    assert_eq!(body["instructions"], "answer precisely");
    assert_eq!(body["max_output_tokens"], 17);
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["top_p"], 0.8);
    assert_eq!(body["parallel_tool_calls"], false);
    assert_eq!(body["tool_choice"], "required");
    assert_eq!(body["tools"][0]["name"], "lookup");
    assert_eq!(body["metadata"]["trace"], "test");
}

#[tokio::test]
async fn responses_streams_reasoning_and_function_call_items() {
    let reply = send(
        app(Scripted::new(Script::output(tool_call_output()))),
        "/v1/responses",
        &[],
        json!({ "model": "test-model", "input": "hi", "stream": true }),
    )
    .await;
    let events = stream_json(&reply.body);
    let kinds = events
        .iter()
        .map(|event| event["type"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    for expected in [
        "response.reasoning_summary_text.delta",
        "response.output_text.delta",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.completed",
    ] {
        assert!(kinds.iter().any(|kind| kind == expected), "{expected} in {kinds:?}");
    }
    let completed = events.last().unwrap();
    let items = completed["response"]["output"].as_array().unwrap();
    assert_eq!(
        items.iter().map(|item| item["type"].clone()).collect::<Vec<_>>(),
        [json!("reasoning"), json!("message"), json!("function_call")]
    );
    let sequence = events
        .iter()
        .map(|event| event["sequence_number"].as_u64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(sequence, (0..sequence.len() as u64).collect::<Vec<_>>());
}

async fn start_response(script: Script) -> responses::ResponseStream {
    let request = serde_json::from_value(json!({ "model": "test-model", "input": "hi", "stream": true }))
        .expect("request must decode");
    responses::start_response_stream(
        Serving::new(Arc::new(Scripted::new(script))),
        &axum::http::HeaderMap::new(),
        request,
    )
    .await
    .unwrap_or_else(|error| panic!("{}", error.body.message))
}

/// Items conclude in `output_index` order even when a message follows a tool
/// call, and the concluded output is exactly the response's `output`.
#[tokio::test]
async fn responses_conclude_items_in_output_order() {
    let mut stream = start_response(Script::output(vec![
        OutputEvent::ToolCallStarted {
            index: 0,
            id: "call-1".into(),
            name: "lookup".into(),
        },
        OutputEvent::ToolInputDelta {
            index: 0,
            fragment: "{}".into(),
        },
        OutputEvent::ToolCallFinished { index: 0 },
        OutputEvent::TextDelta("after".into()),
    ]))
    .await;
    let mut done = Vec::new();
    let mut output = Value::Null;
    while let Some(event) = stream.events.recv().await {
        match event["type"].as_str().unwrap() {
            "response.output_item.done" => {
                done.push((event["output_index"].clone(), event["item"].clone()));
            }
            "response.completed" => output = event["response"]["output"].clone(),
            _ => {}
        }
    }
    assert_eq!(
        done.iter().map(|(index, _)| index.clone()).collect::<Vec<_>>(),
        [json!(0), json!(1)]
    );
    assert_eq!(
        Value::Array(done.into_iter().map(|(_, item)| item).collect()),
        output
    );
    assert_eq!(output[0]["type"], "function_call");
    assert_eq!(Value::Array(stream.concluded.await.unwrap()), output);
}

#[tokio::test]
async fn a_failed_response_concludes_nothing() {
    let mut script = Script::text("partial");
    script.failure = Some(ServingError::Request(RequestError::DeviceLost {
        reason: "scripted failure".into(),
    }));
    let mut stream = start_response(script).await;
    let mut last = Value::Null;
    while let Some(event) = stream.events.recv().await {
        last = event;
    }
    assert_eq!(last["type"], "response.failed");
    assert!(stream.concluded.await.is_err());
}

#[tokio::test]
async fn anthropic_messages_echoes_the_gateway_alias_without_leaking_it_to_inference() {
    let source = Scripted::new(Script::text("hello"));
    let observed = source.observed.clone();
    let reply = send(
        app(source),
        "/anthropic/v1/messages",
        &[
            ("anthropic-version", "2023-06-01"),
            ("Magnitude-Gateway-Model", "anthropic-local/test-model"),
        ],
        json!({ "model": "test-model", "max_tokens": 32, "messages": [{ "role": "user", "content": "hi" }] }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    let body: Value = serde_json::from_str(&reply.body).unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["model"], "anthropic-local/test-model");
    assert_eq!(body["content"][0]["text"], "hello");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(observed.requests.lock().unwrap()[0].0, "test-model");
}

#[tokio::test]
async fn anthropic_count_tokens_is_host_only() {
    let source = Scripted::new(Script::text("unused"));
    let observed = source.observed.clone();
    let reply = send(
        app(source),
        "/anthropic/v1/messages/count_tokens",
        ANTHROPIC,
        // The endpoint takes no generation controls: no `max_tokens`.
        json!({ "model": "test-model", "messages": [{ "role": "user", "content": "hi" }] }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(
        serde_json::from_str::<Value>(&reply.body).unwrap(),
        json!({ "input_tokens": 1 })
    );
    assert_eq!(observed.invocations.load(Ordering::Relaxed), 0);
    assert_eq!(observed.counted.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn anthropic_requires_the_supported_protocol_version() {
    let reply = send(
        app(Scripted::new(Script::text("hello"))),
        "/anthropic/v1/messages",
        &[("anthropic-version", "2024-01-01")],
        json!({ "model": "test-model", "max_tokens": 32, "messages": [{ "role": "user", "content": "hi" }] }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    assert!(reply.body.contains("unsupported anthropic-version"));
}

#[tokio::test]
async fn anthropic_stream_follows_message_and_content_block_lifecycle() {
    let reply = send(
        app(Scripted::new(Script::output(tool_call_output()))),
        "/anthropic/v1/messages",
        ANTHROPIC,
        json!({ "model": "test-model", "max_tokens": 32, "stream": true, "messages": [{ "role": "user", "content": "hi" }] }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    let body = reply.body;
    let start = body.find("event: message_start").unwrap();
    let block = body.find("event: content_block_start").unwrap();
    let delta = body.find("event: content_block_delta").unwrap();
    let stop = body.find("event: content_block_stop").unwrap();
    let message_stop = body.find("event: message_stop").unwrap();
    assert!(start < block && block < delta && delta < stop && stop < message_stop);
    assert!(body.contains("\"input_tokens\":11"));
    assert!(body.contains("\"thinking_delta\""));
    assert!(body.contains("\"signature_delta\""));
    assert!(body.contains("\"input_json_delta\""));
    assert!(body.contains("\"stop_reason\":\"tool_use\""));
}

fn cache_hit() -> Scripted {
    Scripted::new(Script {
        cached_input_tokens: 8,
        ..Script::text("hello")
    })
}

/// Anthropic prompt usage parts are disjoint: 11 prompt tokens with an
/// 8-token cache hit are 3 uncached input plus 8 cache reads.
fn cache_hit_usage() -> Value {
    json!({
        "input_tokens": 3,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 8,
        "output_tokens": 7
    })
}

#[tokio::test]
async fn anthropic_message_usage_counts_cache_reads_once() {
    let reply = send(
        app(cache_hit()),
        "/anthropic/v1/messages",
        ANTHROPIC,
        json!({ "model": "test-model", "max_tokens": 32, "messages": [{ "role": "user", "content": "hi" }] }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    let body = serde_json::from_str::<Value>(&reply.body).unwrap();
    assert_eq!(body["usage"], cache_hit_usage());
}

#[tokio::test]
async fn anthropic_stream_message_delta_carries_cumulative_usage() {
    let reply = send(
        app(cache_hit()),
        "/anthropic/v1/messages",
        ANTHROPIC,
        json!({ "model": "test-model", "max_tokens": 32, "stream": true, "messages": [{ "role": "user", "content": "hi" }] }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    let events = stream_json(&reply.body);
    let message_delta = events
        .iter()
        .find(|event| event["type"] == "message_delta")
        .unwrap();
    assert_eq!(message_delta["usage"], cache_hit_usage());
}

#[tokio::test]
async fn unknown_models_are_not_found_on_every_protocol() {
    let reply = send(
        app(Scripted::new(Script::text("hello"))),
        "/v1/chat/completions",
        &[],
        json!({ "model": "other", "messages": [{"role": "user", "content": "hi"}] }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::NOT_FOUND);
    assert!(reply.body.contains("model_not_found"));
    let reply = send(
        app(Scripted::new(Script::text("hello"))),
        "/anthropic/v1/messages",
        ANTHROPIC,
        json!({ "model": "other", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}] }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::NOT_FOUND);
    assert!(reply.body.contains("not_found_error"));
}

#[tokio::test]
async fn invalid_chat_is_rejected_before_model_binding() {
    let source = Scripted::new(Script::text("hello"));
    let observed = source.observed.clone();
    let reply = send(
        app(source),
        "/v1/chat/completions",
        &[],
        json!({ "model": "test-model", "messages": [{"role": "assistant", "content": null}], "stream": true }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    assert_eq!(observed.invocations.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn progress_chat_reports_binding_failure_in_stream() {
    let mut source = Scripted::new(Script::text("unused"));
    source.binding_failure = Some(ServingError::Model(ModelUnavailable::InstanceStopped));
    let reply = send(app(source), "/v1/chat/completions", PROGRESS, minimal_request()).await;
    assert_eq!(reply.status, StatusCode::OK);
    let error = stream_json(&reply.body).pop().unwrap();
    assert_eq!(error["error"]["type"], "model_error");
    assert_eq!(error["error"]["code"], "model_instance_stopped");
}

#[tokio::test]
async fn ordinary_stream_waits_for_model_binding_before_opening() {
    let mut source = Scripted::new(Script::text("ready"));
    source.pending = true;
    let observed = source.observed.clone();
    let mut response = Box::pin(app(source).oneshot(
        Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(minimal_request().to_string()))
            .unwrap(),
    ));
    assert!(tokio::time::timeout(Duration::from_millis(10), &mut response)
        .await
        .is_err());
    drop(response);
    tokio::time::timeout(Duration::from_secs(1), async {
        while !observed.binding_dropped.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping the pending HTTP request cancels the model binding");
}

#[tokio::test]
async fn progress_stream_opens_before_pending_model_binding_finishes() {
    let mut source = Scripted::new(Script::text("ready"));
    source.pending = true;
    let observed = source.observed.clone();
    let response = tokio::time::timeout(
        Duration::from_secs(1),
        app(source).oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("Magnitude-Include-Progress", "true")
                .body(Body::from(minimal_request().to_string()))
                .unwrap(),
        ),
    )
    .await
    .expect("progress response opens before the model is bound")
    .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    tokio::task::yield_now().await;
    drop(response);
    tokio::time::timeout(Duration::from_secs(1), async {
        while !observed.binding_dropped.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping the open stream cancels its pending binding");
}

#[tokio::test]
async fn dropping_a_committed_stream_cancels_the_generation() {
    let mut script = Script::text("first");
    script.hang = true;
    let source = Scripted::new(script);
    let observed = source.observed.clone();
    let response = app(source)
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(minimal_request().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    body.frame().await.unwrap().unwrap();
    drop(body);
    tokio::time::timeout(Duration::from_secs(1), async {
        while !observed.cancelled.load(Ordering::Acquire) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("a disconnected client cancels its generation");
}

#[tokio::test]
async fn template_application_and_properties_are_host_only() {
    let source = Scripted::new(Script::text("unused"));
    let observed = source.observed.clone();
    let app = app(source);
    let reply = send(
        app.clone(),
        "/api/v1/chat/templates/apply",
        &[],
        json!({ "model": "test-model", "messages": [{"role": "user", "content": "hi"}] }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    let body: Value = serde_json::from_str(&reply.body).unwrap();
    assert_eq!(body["prompt"], "prompt");
    assert_eq!(body["grammar_lazy"], false);
    assert_eq!(body["thinking_end_tag"], "</think>");
    assert_eq!(body["template_fingerprint"], "fingerprint");

    let reply = send(app, "/api/v1/models/test-model/properties", &[], json!({})).await;
    assert_eq!(reply.status, StatusCode::OK);
    let body: Value = serde_json::from_str(&reply.body).unwrap();
    assert_eq!(body["default_generation_settings"]["n_ctx"], 4096);
    assert_eq!(body["training_context_tokens"], 262_144);
    assert_eq!(body["reasoning"]["reasoning_efforts"], json!(["none", "high"]));
    assert_eq!(body["reasoning"]["default_reasoning_effort"], "high");
    assert_eq!(body["template_capabilities"]["enable_thinking"], true);
    assert!(body.get("execution").is_none());
    assert_eq!(observed.invocations.load(Ordering::Relaxed), 0);
}
