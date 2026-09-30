//! OpenAI Responses over HTTP (JSON and SSE) and WebSocket. The WebSocket
//! keeps per-connection logical request history: `generate: false` warms a
//! request without generating, and `previous_response_id` appends new input to
//! that history.
use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use magnitude_chat::output::{Completion, Output, OutputEvent, Progress, Termination};
use magnitude_chat::request::{
    AssistantTurn, Conversation, Entry, GenerationControls, OutputFormat, PromptCache,
    SamplingControls, ToolCall, ToolDefinition, ToolExchange, ToolResultPart, Tools, UserPart,
};
use magnitude_chat::schema::JsonSchema;
use magnitude_chat::{ChatInput, EndOfGeneration, GenerationRequest, ReasoningIntent, ToolChoice};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use utoipa::openapi::Ref;
use utoipa::openapi::schema::AnyOfBuilder;
use utoipa::{PartialSchema, ToSchema};

use crate::chat::{
    ChatCompletionProgress, ReasoningEffortRequest, admitted, collect, ended_without_outcome,
    progress_value,
};
use crate::error::{ApiError, ApiErrorBody, ErrorResponse, ServingError};
use crate::source::{GenerationEvent, GenerationStream, LoadProgress, ModelLoadProgress};
use crate::{Serving, include_progress, media, unix_timestamp, with_request_id};

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponseCreateRequest {
    pub model: String,
    pub input: ResponseInput,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub tools: Option<Vec<ResponseTool>>,
    #[schema(value_type = Object, nullable = false)]
    pub tool_choice: Option<ResponseToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub reasoning: Option<ResponseReasoning>,
    pub text: Option<ResponseText>,
    #[serde(default)]
    pub stream: bool,
    pub store: Option<bool>,
    pub metadata: Option<Map<String, Value>>,
    pub include: Option<Vec<String>>,
    pub client_metadata: Option<Map<String, Value>>,
    pub previous_response_id: Option<String>,
    pub prompt_cache_key: Option<String>,
    pub truncation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponseStreamEvent {
    pub r#type: String,
    pub sequence_number: u64,
    #[serde(flatten)]
    pub data: BTreeMap<String, Value>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub status: String,
    pub error: Option<ResponseError>,
    pub incomplete_details: Option<IncompleteDetails>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    pub parallel_tool_calls: bool,
    pub previous_response_id: Option<String>,
    pub reasoning: ResponseReasoningResult,
    pub store: bool,
    pub temperature: Option<f32>,
    pub text: ResponseTextResult,
    pub tool_choice: Value,
    pub tools: Vec<Value>,
    pub top_p: Option<f32>,
    pub truncation: String,
    pub usage: ResponseUsage,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseError {
    pub message: String,
    pub r#type: String,
    pub code: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct IncompleteDetails {
    pub reason: &'static str,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputItem {
    Reasoning {
        id: String,
        status: &'static str,
        summary: Vec<ResponseSummaryPart>,
    },
    Message {
        id: String,
        status: &'static str,
        role: &'static str,
        content: Vec<ResponseOutputContent>,
    },
    FunctionCall {
        id: String,
        status: &'static str,
        call_id: String,
        name: String,
        arguments: String,
    },
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseSummaryPart {
    pub r#type: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseOutputContent {
    pub r#type: &'static str,
    pub text: String,
    pub annotations: Vec<Value>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseReasoningResult {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ResponseTextResult {
    pub format: ResponseTextFormatResult,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ResponseTextFormatResult {
    pub r#type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseUsage {
    pub input_tokens: u64,
    pub input_tokens_details: ResponseInputTokenDetails,
    pub output_tokens: u64,
    pub output_tokens_details: ResponseOutputTokenDetails,
    pub total_tokens: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseInputTokenDetails {
    pub cached_tokens: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseOutputTokenDetails {
    pub reasoning_tokens: u64,
}

// One constructor per output item kind, shared by the non-streaming response
// and the streaming projection. Both paths must emit identical item shapes:
// clients replay emitted items verbatim as later input, and the replay
// closure test exercises these exact constructors.
pub(crate) fn reasoning_item(
    id: String,
    status: &'static str,
    text: Option<String>,
) -> ResponseOutputItem {
    ResponseOutputItem::Reasoning {
        id,
        status,
        summary: text
            .map(|text| {
                vec![ResponseSummaryPart {
                    r#type: "summary_text",
                    text,
                }]
            })
            .unwrap_or_default(),
    }
}

pub(crate) fn message_item(
    id: String,
    status: &'static str,
    text: Option<String>,
) -> ResponseOutputItem {
    ResponseOutputItem::Message {
        id,
        status,
        role: "assistant",
        content: text
            .map(|text| {
                vec![ResponseOutputContent {
                    r#type: "output_text",
                    text,
                    annotations: Vec::new(),
                }]
            })
            .unwrap_or_default(),
    }
}

pub(crate) fn function_call_item(
    id: String,
    status: &'static str,
    call_id: String,
    name: String,
    arguments: String,
) -> ResponseOutputItem {
    ResponseOutputItem::FunctionCall {
        id,
        status,
        call_id,
        name,
        arguments,
    }
}

fn arguments(call: &ToolCall) -> String {
    serde_json::to_string(&call.arguments).expect("a JSON object serializes")
}

pub fn from_result(
    id: &str,
    created_at: u64,
    model: &str,
    projection: &ResponseProjection,
    output: &Output,
    completion: &Completion,
) -> ResponseObject {
    let mut items = Vec::new();
    if let Some(reasoning) = &output.reasoning {
        items.push(reasoning_item(
            format!("rs_{}", &id[5..]),
            "completed",
            Some(reasoning.clone()),
        ));
    }
    if let Some(text) = &output.text {
        items.push(message_item(
            format!("msg_{}", &id[5..]),
            "completed",
            Some(text.clone()),
        ));
    }
    items.extend(output.tool_calls.iter().map(|call| {
        function_call_item(
            format!("fc_{}", call.id),
            "completed",
            call.id.clone(),
            call.name.clone(),
            arguments(call),
        )
    }));
    let incomplete = completion.termination == Termination::OutputLimit;
    let usage = completion.usage;
    ResponseObject {
        id: id.to_owned(),
        object: "response",
        created_at,
        status: if incomplete { "incomplete" } else { "completed" }.to_owned(),
        error: None,
        incomplete_details: incomplete.then_some(IncompleteDetails {
            reason: "max_output_tokens",
        }),
        instructions: projection.instructions.clone(),
        max_output_tokens: projection.max_output_tokens,
        model: model.to_owned(),
        output: items,
        parallel_tool_calls: projection.parallel_tool_calls,
        previous_response_id: projection.previous_response_id.clone(),
        reasoning: ResponseReasoningResult {
            effort: projection.reasoning_effort.clone(),
            summary: projection.reasoning_summary.clone(),
        },
        store: projection.store,
        temperature: projection.temperature,
        text: projection.text.clone(),
        tool_choice: projection.tool_choice.clone(),
        tools: projection.tools.clone(),
        top_p: projection.top_p,
        truncation: projection.truncation.clone(),
        usage: ResponseUsage {
            input_tokens: usage.input_tokens,
            input_tokens_details: ResponseInputTokenDetails {
                cached_tokens: usage.cached_input_tokens,
            },
            output_tokens: usage.output_tokens,
            output_tokens_details: ResponseOutputTokenDetails {
                reasoning_tokens: usage.reasoning_output_tokens,
            },
            total_tokens: usage.input_tokens.saturating_add(usage.output_tokens),
        },
        metadata: projection.metadata.clone(),
    }
}

/// Request fields echoed on every response object.
#[derive(Clone)]
pub struct ResponseProjection {
    instructions: Option<String>,
    max_output_tokens: Option<u32>,
    parallel_tool_calls: bool,
    previous_response_id: Option<String>,
    reasoning_effort: Option<String>,
    reasoning_summary: Option<String>,
    store: bool,
    temperature: Option<f32>,
    text: ResponseTextResult,
    tool_choice: Value,
    tools: Vec<Value>,
    top_p: Option<f32>,
    truncation: String,
    metadata: Map<String, Value>,
}

type ToolProjection = (usize, String, String, String, String);

pub(crate) struct StreamProjector {
    id: String,
    message_id: String,
    created_at: u64,
    model: String,
    sender: mpsc::Sender<Value>,
    sequence: Arc<AtomicU64>,
    next_output_index: usize,
    message_output_index: Option<usize>,
    reasoning_output_index: Option<usize>,
    text: String,
    reasoning: String,
    tool_calls: BTreeMap<usize, ToolProjection>,
    projection: ResponseProjection,
}

/// The stream consumer is gone; the generation is dropped (cancelled).
struct Disconnected;

impl StreamProjector {
    pub(crate) fn new(
        id: String,
        created_at: u64,
        model: String,
        sender: mpsc::Sender<Value>,
        sequence: Arc<AtomicU64>,
        projection: ResponseProjection,
    ) -> Self {
        let message_id = format!("msg_{}", &id[5..]);
        Self {
            id,
            message_id,
            created_at,
            model,
            sender,
            sequence,
            next_output_index: 0,
            message_output_index: None,
            reasoning_output_index: None,
            text: String::new(),
            reasoning: String::new(),
            tool_calls: BTreeMap::new(),
            projection,
        }
    }

    fn base(&self, status: &'static str, output: Value) -> Value {
        response_base(
            &self.id,
            self.created_at,
            &self.model,
            &self.projection,
            status,
            output,
        )
    }

    async fn send(&self, event_type: &'static str, value: Value) -> Result<(), Disconnected> {
        let mut data = object(value);
        data.insert("type".into(), Value::String(event_type.into()));
        data.insert(
            "sequence_number".into(),
            Value::from(self.sequence.fetch_add(1, Ordering::Relaxed)),
        );
        self.sender
            .send(Value::Object(data))
            .await
            .map_err(|_| Disconnected)
    }

    async fn created(&self) -> Result<(), Disconnected> {
        self.send(
            "response.created",
            serde_json::json!({ "response": self.base("in_progress", serde_json::json!([])) }),
        )
        .await
    }

    async fn in_progress(&self) -> Result<(), Disconnected> {
        self.send(
            "response.in_progress",
            serde_json::json!({ "response": self.base("in_progress", serde_json::json!([])) }),
        )
        .await
    }

    async fn progress(&self, progress: Progress) -> Result<(), Disconnected> {
        self.send(
            "response.magnitude_progress",
            serde_json::json!({ "response_id": self.id, "progress": progress_value(progress) }),
        )
        .await
    }

    async fn observe(&mut self, event: OutputEvent) -> Result<(), ProjectionError> {
        match event {
            OutputEvent::Started | OutputEvent::ToolCallFinished { .. } => {}
            OutputEvent::ReasoningDelta(text) => {
                let index = match self.reasoning_output_index {
                    Some(index) => index,
                    None => {
                        let index = self.allocate_output();
                        self.reasoning_output_index = Some(index);
                        let item = reasoning_item(self.reasoning_id(), "in_progress", None);
                        self.send(
                            "response.output_item.added",
                            serde_json::json!({ "output_index": index, "item": item }),
                        )
                        .await?;
                        index
                    }
                };
                self.reasoning.push_str(&text);
                self.send(
                    "response.reasoning_summary_text.delta",
                    serde_json::json!({
                        "item_id": self.reasoning_id(), "output_index": index,
                        "summary_index": 0, "delta": text,
                    }),
                )
                .await?;
            }
            OutputEvent::TextDelta(text) => {
                let index = match self.message_output_index {
                    Some(index) => index,
                    None => {
                        let index = self.allocate_output();
                        self.message_output_index = Some(index);
                        let item = message_item(self.message_id.clone(), "in_progress", None);
                        self.send(
                            "response.output_item.added",
                            serde_json::json!({ "output_index": index, "item": item }),
                        )
                        .await?;
                        self.send(
                            "response.content_part.added",
                            serde_json::json!({
                                "item_id": self.message_id, "output_index": index, "content_index": 0,
                                "part": { "type": "output_text", "text": "", "annotations": [] },
                            }),
                        )
                        .await?;
                        index
                    }
                };
                self.text.push_str(&text);
                self.send(
                    "response.output_text.delta",
                    serde_json::json!({
                        "item_id": self.message_id, "output_index": index,
                        "content_index": 0, "delta": text,
                    }),
                )
                .await?;
            }
            OutputEvent::ToolCallStarted { index, id, name } => {
                let output_index = self.allocate_output();
                let item_id = format!("fc_{id}");
                self.tool_calls.insert(
                    index,
                    (
                        output_index,
                        item_id.clone(),
                        id.clone(),
                        name.clone(),
                        String::new(),
                    ),
                );
                let item = function_call_item(item_id, "in_progress", id, name, String::new());
                self.send(
                    "response.output_item.added",
                    serde_json::json!({ "output_index": output_index, "item": item }),
                )
                .await?;
            }
            OutputEvent::ToolInputDelta { index, fragment } => {
                let (output_index, item_id) = {
                    let entry = self.tool_calls.get_mut(&index).ok_or_else(|| {
                        ProjectionError::Failed(ServingError::Internal(
                            "tool input arrived before tool start".into(),
                        ))
                    })?;
                    entry.4.push_str(&fragment);
                    (entry.0, entry.1.clone())
                };
                self.send(
                    "response.function_call_arguments.delta",
                    serde_json::json!({
                        "item_id": item_id, "output_index": output_index, "delta": fragment,
                    }),
                )
                .await?;
            }
        }
        Ok(())
    }

    async fn finish(&self, completion: &Completion) -> Result<(), Disconnected> {
        let mut indexed = Vec::new();
        if let Some(index) = self.reasoning_output_index {
            let item_id = self.reasoning_id();
            let item = serde_json::to_value(reasoning_item(
                item_id.clone(),
                "completed",
                Some(self.reasoning.clone()),
            ))
            .expect("output item is serializable");
            self.send(
                "response.reasoning_summary_text.done",
                serde_json::json!({
                    "item_id": item_id, "output_index": index, "summary_index": 0, "text": self.reasoning,
                }),
            )
            .await?;
            self.send(
                "response.output_item.done",
                serde_json::json!({ "output_index": index, "item": item }),
            )
            .await?;
            indexed.push((index, item));
        }
        if let Some(index) = self.message_output_index {
            let item = serde_json::to_value(message_item(
                self.message_id.clone(),
                "completed",
                Some(self.text.clone()),
            ))
            .expect("output item is serializable");
            let content = item["content"].clone();
            self.send(
                "response.output_text.done",
                serde_json::json!({
                    "item_id": self.message_id, "output_index": index, "content_index": 0, "text": self.text,
                }),
            )
            .await?;
            self.send(
                "response.content_part.done",
                serde_json::json!({
                    "item_id": self.message_id, "output_index": index, "content_index": 0,
                    "part": content[0],
                }),
            )
            .await?;
            self.send(
                "response.output_item.done",
                serde_json::json!({ "output_index": index, "item": item }),
            )
            .await?;
            indexed.push((index, item));
        }
        for (output_index, item_id, call_id, name, arguments) in self.tool_calls.values() {
            let item = serde_json::to_value(function_call_item(
                item_id.clone(),
                "completed",
                call_id.clone(),
                name.clone(),
                arguments.clone(),
            ))
            .expect("output item is serializable");
            self.send(
                "response.function_call_arguments.done",
                serde_json::json!({
                    "item_id": item_id, "output_index": output_index, "arguments": arguments,
                }),
            )
            .await?;
            self.send(
                "response.output_item.done",
                serde_json::json!({ "output_index": output_index, "item": item }),
            )
            .await?;
            indexed.push((*output_index, item));
        }
        indexed.sort_by_key(|(index, _)| *index);
        let output = Value::Array(indexed.into_iter().map(|(_, item)| item).collect());
        let incomplete = completion.termination == Termination::OutputLimit;
        let mut completed = self.base(
            if incomplete { "incomplete" } else { "completed" },
            output,
        );
        if incomplete {
            completed["incomplete_details"] = serde_json::json!({ "reason": "max_output_tokens" });
        }
        completed["usage"] = usage_value(completion);
        self.send(
            if incomplete {
                "response.incomplete"
            } else {
                "response.completed"
            },
            serde_json::json!({ "response": completed }),
        )
        .await
    }

    async fn fail(&self, error: &ApiErrorBody) {
        let mut response = self.base("failed", serde_json::json!([]));
        response["error"] = serde_json::to_value(error).expect("error body serializes");
        let _ = self
            .send("response.failed", serde_json::json!({ "response": response }))
            .await;
    }

    fn allocate_output(&mut self) -> usize {
        let index = self.next_output_index;
        self.next_output_index += 1;
        index
    }

    fn reasoning_id(&self) -> String {
        format!("rs_{}", &self.id[5..])
    }
}

enum ProjectionError {
    Disconnected,
    Failed(ServingError),
}

impl From<Disconnected> for ProjectionError {
    fn from(_: Disconnected) -> Self {
        Self::Disconnected
    }
}

fn send_loading_progress(
    sender: &mpsc::Sender<Value>,
    sequence: &AtomicU64,
    response_id: &str,
    load: ModelLoadProgress,
) {
    let value = serde_json::json!({
        "type": "response.magnitude_progress",
        "sequence_number": sequence.fetch_add(1, Ordering::Relaxed),
        "response_id": response_id,
        "progress": ChatCompletionProgress::model_loading(load),
    });
    let _ = sender.try_send(value);
}

fn object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => unreachable!("response events are JSON objects"),
    }
}

fn usage_value(completion: &Completion) -> Value {
    let usage = completion.usage;
    serde_json::json!({
        "input_tokens": usage.input_tokens,
        "input_tokens_details": { "cached_tokens": usage.cached_input_tokens },
        "output_tokens": usage.output_tokens,
        "output_tokens_details": { "reasoning_tokens": usage.reasoning_output_tokens },
        "total_tokens": usage.input_tokens.saturating_add(usage.output_tokens),
    })
}

fn response_base(
    id: &str,
    created_at: u64,
    model: &str,
    projection: &ResponseProjection,
    status: &str,
    output: Value,
) -> Value {
    serde_json::json!({
        "id": id, "object": "response", "created_at": created_at, "status": status,
        "error": null, "incomplete_details": null, "instructions": projection.instructions,
        "max_output_tokens": projection.max_output_tokens, "model": model, "output": output,
        "parallel_tool_calls": projection.parallel_tool_calls,
        "previous_response_id": projection.previous_response_id,
        "reasoning": { "effort": projection.reasoning_effort, "summary": projection.reasoning_summary },
        "store": projection.store, "temperature": projection.temperature, "text": projection.text,
        "tool_choice": projection.tool_choice, "tools": projection.tools, "top_p": projection.top_p,
        "truncation": projection.truncation, "usage": null, "metadata": projection.metadata,
    })
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Items(Vec<ResponseInputItem>),
}

// Untagged because the protocol's shorthand message form carries no `type`
// discriminator; variants are matched by shape, and each explicit `type`
// field is a single-literal enum so a present tag is still verified.
// Replay closure invariant: every item `ResponseOutputItem` can emit must
// parse here, because clients replay our output verbatim as later input.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponseInputItem {
    Message(ResponseInputMessage),
    Reasoning(ResponseReasoningInput),
    FunctionCall(ResponseFunctionCall),
    FunctionCallOutput(ResponseFunctionCallOutput),
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponseInputMessage {
    #[serde(default)]
    pub r#type: Option<ResponseMessageType>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub phase: Option<String>,
    pub role: ResponseRole,
    pub content: ResponseMessageContent,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponseReasoningInput {
    pub r#type: ResponseReasoningInputType,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub content: Vec<ResponseReasoningInputContent>,
    #[serde(default)]
    pub summary: Vec<ResponseReasoningInputContent>,
    #[serde(default)]
    pub encrypted_content: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseReasoningInputContent {
    ReasoningText { text: String },
    SummaryText { text: String },
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponseFunctionCall {
    pub r#type: ResponseFunctionCallType,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponseFunctionCallOutput {
    pub r#type: ResponseFunctionCallOutputType,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    pub call_id: String,
    pub output: FunctionCallOutput,
}

impl PartialSchema for ResponseInputItem {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        AnyOfBuilder::new()
            .item(Ref::from_schema_name(ResponseInputMessage::name()))
            .item(Ref::from_schema_name(ResponseReasoningInput::name()))
            .item(Ref::from_schema_name(ResponseFunctionCall::name()))
            .item(Ref::from_schema_name(ResponseFunctionCallOutput::name()))
            .into()
    }
}

impl ToSchema for ResponseInputItem {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        for (name, schema) in [
            (ResponseInputMessage::name(), ResponseInputMessage::schema()),
            (
                ResponseReasoningInput::name(),
                ResponseReasoningInput::schema(),
            ),
            (ResponseFunctionCall::name(), ResponseFunctionCall::schema()),
            (
                ResponseFunctionCallOutput::name(),
                ResponseFunctionCallOutput::schema(),
            ),
        ] {
            schemas.push((name.into_owned(), schema));
        }
        ResponseInputMessage::schemas(schemas);
        ResponseReasoningInput::schemas(schemas);
        ResponseFunctionCall::schemas(schemas);
        ResponseFunctionCallOutput::schemas(schemas);
    }
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResponseMessageType {
    Message,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResponseReasoningInputType {
    Reasoning,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFunctionCallType {
    FunctionCall,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFunctionCallOutputType {
    FunctionCallOutput,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum FunctionCallOutput {
    Text(String),
    Parts(Vec<FunctionCallOutputPart>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FunctionCallOutputPart {
    InputText { text: String },
    InputImage { image_url: String },
}

#[derive(Debug, Clone, Copy, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ResponseRole {
    User,
    Assistant,
    System,
    Developer,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ResponseMessageContent {
    Text(String),
    Parts(Vec<ResponseContentPart>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentPart {
    InputText {
        text: String,
    },
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<Value>,
    },
    InputImage {
        image_url: String,
    },
}

// Function declarations are the executable semantic core and stay strictly
// typed. Every other declaration (namespace, web_search, and any future
// hosted tool type) is opaque by policy: never locally executable, retained
// verbatim only for response projection, so its shape is deliberately not
// modeled and new hosted tool types require no adapter change.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseTool {
    Function(ResponseFunctionTool),
    Other(Value),
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResponseFunctionTool {
    pub r#type: ResponseFunctionType,
    pub name: String,
    pub description: Option<String>,
    pub parameters: Map<String, Value>,
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFunctionType {
    Function,
}

impl PartialSchema for ResponseTool {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        AnyOfBuilder::new()
            .item(Ref::from_schema_name(ResponseFunctionTool::name()))
            .item(utoipa::openapi::schema::ObjectBuilder::new())
            .into()
    }
}

impl ToSchema for ResponseTool {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((
            ResponseFunctionTool::name().into_owned(),
            ResponseFunctionTool::schema(),
        ));
        ResponseFunctionTool::schemas(schemas);
    }
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ResponseToolChoice {
    Mode(ResponseToolChoiceMode),
    Function(ResponseFunctionChoice),
}

#[derive(Debug, Clone, Copy, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResponseToolChoiceMode {
    None,
    Auto,
    Required,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFunctionChoice {
    Function { name: String },
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponseReasoning {
    pub effort: Option<ReasoningEffortRequest>,
    pub summary: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponseText {
    pub format: ResponseTextFormat,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTextFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Map<String, Value>,
        #[serde(default)]
        strict: bool,
    },
}

pub struct AdaptedResponseRequest {
    pub model: String,
    pub request: GenerationRequest,
    pub projection: ResponseProjection,
}

fn invalid(error: magnitude_chat::ChatError) -> ApiError {
    ApiError::invalid(error.to_string())
}

pub fn adapt(request: ResponseCreateRequest) -> Result<AdaptedResponseRequest, ApiError> {
    let projection = response_projection(&request);
    if request.model.is_empty() {
        return Err(ApiError::invalid("model is required"));
    }
    if request.store == Some(true) {
        return Err(ApiError::invalid(
            "store is not supported by this local runtime",
        ));
    }
    if request.previous_response_id.is_some() {
        return Err(ApiError::invalid(
            "previous_response_id is not supported; submit the full input history",
        ));
    }
    if request
        .truncation
        .as_deref()
        .is_some_and(|value| value != "disabled")
    {
        return Err(ApiError::invalid("automatic truncation is not supported"));
    }
    let conversation = context(request.instructions, request.input)?;
    let mut definitions = Vec::new();
    for tool in request.tools.unwrap_or_default() {
        match tool {
            ResponseTool::Function(function) => definitions.push(ToolDefinition {
                name: function.name,
                description: function.description,
                parameters: JsonSchema::new(function.parameters).map_err(invalid)?,
            }),
            // Opaque declarations are projection-only. The one guard: a
            // function-typed declaration that failed strict parsing must stay
            // a request error, never a silent demotion to non-executable.
            ResponseTool::Other(value) => match value.get("type").and_then(Value::as_str) {
                Some("function") => {
                    return Err(ApiError::invalid("malformed function tool declaration"));
                }
                Some(_) => {}
                None => return Err(ApiError::invalid("tool declarations require a type")),
            },
        }
    }
    let choice = match request.tool_choice {
        None | Some(ResponseToolChoice::Mode(ResponseToolChoiceMode::Auto)) => ToolChoice::Auto,
        Some(ResponseToolChoice::Mode(ResponseToolChoiceMode::None)) => ToolChoice::None,
        Some(ResponseToolChoice::Mode(ResponseToolChoiceMode::Required)) => ToolChoice::Required,
        Some(ResponseToolChoice::Function(ResponseFunctionChoice::Function { name })) => {
            ToolChoice::Named(name)
        }
    };
    let tools = Tools::new(
        definitions,
        choice,
        request.parallel_tool_calls.unwrap_or(true),
    )
    .map_err(invalid)?;
    let reasoning = match request.reasoning.and_then(|value| value.effort) {
        Some(effort) => ReasoningIntent::Effort {
            effort: effort.normalize()?.into(),
        },
        None => ReasoningIntent::ModelDefault,
    };
    let output = match request.text.map(|value| value.format) {
        None | Some(ResponseTextFormat::Text) => OutputFormat::Text,
        Some(ResponseTextFormat::JsonObject) => OutputFormat::JsonObject,
        // Output is constrained best effort whether or not it is strict.
        Some(ResponseTextFormat::JsonSchema { name, schema, .. }) => OutputFormat::JsonSchema {
            name,
            schema: JsonSchema::new(schema).map_err(invalid)?,
        },
    };
    let max_output_tokens = request
        .max_output_tokens
        .map(|value| {
            NonZeroU32::new(value)
                .ok_or_else(|| ApiError::invalid("max_output_tokens must be positive"))
        })
        .transpose()?;
    let sampling = openai_sampling(request.temperature, request.top_p)?;
    Ok(AdaptedResponseRequest {
        model: request.model,
        request: GenerationRequest {
            input: ChatInput {
                conversation,
                tools,
                reasoning,
                output,
                template_arguments: Map::new(),
            },
            controls: GenerationControls {
                max_output_tokens,
                sampling,
                stops: Vec::new(),
                end_of_generation: EndOfGeneration::Stop,
                reasoning_budget: None,
                prompt_cache: PromptCache::Allowed,
            },
        },
        projection,
    })
}

/// Responses sampling: OpenAI defaults, a fixed seed.
fn openai_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> Result<SamplingControls, ApiError> {
    let temperature = temperature.unwrap_or(0.8);
    let top_p = top_p.unwrap_or(0.95);
    sampling(temperature, top_p)
}

/// Validated temperature and top-p with identity shaping otherwise and seed 0.
pub(crate) fn sampling(temperature: f32, top_p: f32) -> Result<SamplingControls, ApiError> {
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err(ApiError::invalid(
            "temperature must be finite and between 0 and 2",
        ));
    }
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
        return Err(ApiError::invalid(
            "top_p must be finite, greater than 0 and at most 1",
        ));
    }
    Ok(SamplingControls {
        temperature,
        top_p,
        top_k: 0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
    })
}

fn response_projection(request: &ResponseCreateRequest) -> ResponseProjection {
    let tool_choice = match request.tool_choice.as_ref() {
        None | Some(ResponseToolChoice::Mode(ResponseToolChoiceMode::Auto)) => {
            Value::String("auto".into())
        }
        Some(ResponseToolChoice::Mode(ResponseToolChoiceMode::None)) => {
            Value::String("none".into())
        }
        Some(ResponseToolChoice::Mode(ResponseToolChoiceMode::Required)) => {
            Value::String("required".into())
        }
        Some(ResponseToolChoice::Function(ResponseFunctionChoice::Function { name })) => {
            serde_json::json!({ "type": "function", "name": name })
        }
    };
    let tools = request
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|tool| serde_json::to_value(tool).expect("response tool is serializable"))
        .collect();
    let format = match request.text.as_ref().map(|text| &text.format) {
        Some(ResponseTextFormat::JsonObject) => ResponseTextFormatResult {
            r#type: "json_object",
            name: None,
            schema: None,
            strict: None,
        },
        Some(ResponseTextFormat::JsonSchema {
            name,
            schema,
            strict,
        }) => ResponseTextFormatResult {
            r#type: "json_schema",
            name: Some(name.clone()),
            schema: Some(schema.clone()),
            strict: Some(*strict),
        },
        None | Some(ResponseTextFormat::Text) => ResponseTextFormatResult {
            r#type: "text",
            name: None,
            schema: None,
            strict: None,
        },
    };
    ResponseProjection {
        instructions: request.instructions.clone(),
        max_output_tokens: request.max_output_tokens,
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        previous_response_id: request.previous_response_id.clone(),
        reasoning_effort: request
            .reasoning
            .as_ref()
            .and_then(|value| value.effort.as_ref())
            .map(|value| value.0.clone()),
        reasoning_summary: request
            .reasoning
            .as_ref()
            .and_then(|value| value.summary.clone()),
        store: request.store.unwrap_or(false),
        temperature: request.temperature,
        text: ResponseTextResult { format },
        tool_choice,
        tools,
        top_p: request.top_p,
        truncation: request
            .truncation
            .clone()
            .unwrap_or_else(|| "disabled".into()),
        metadata: request.metadata.clone().unwrap_or_default(),
    }
}

fn joined(parts: &mut Vec<String>) -> Option<String> {
    let text = std::mem::take(parts).join("\n");
    (!text.is_empty()).then_some(text)
}

fn context(instructions: Option<String>, input: ResponseInput) -> Result<Conversation, ApiError> {
    let mut system = instructions
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    let items = match input {
        ResponseInput::Text(text) => vec![ResponseInputItem::Message(ResponseInputMessage {
            r#type: None,
            id: None,
            status: None,
            phase: None,
            role: ResponseRole::User,
            content: ResponseMessageContent::Text(text),
        })],
        ResponseInput::Items(items) => items,
    };
    let mut entries = Vec::new();
    let mut pending_reasoning = Vec::new();
    let mut index = 0;
    while index < items.len() {
        match &items[index] {
            ResponseInputItem::Message(ResponseInputMessage { role, content, .. }) => match role {
                ResponseRole::System | ResponseRole::Developer => {
                    if !entries.is_empty() || !pending_reasoning.is_empty() {
                        return Err(ApiError::invalid(
                            "system and developer messages must precede conversation entries",
                        ));
                    }
                    let text = text_content(content)?;
                    if !text.is_empty() {
                        system.push(text);
                    }
                }
                ResponseRole::User => {
                    if !pending_reasoning.is_empty() {
                        entries.push(Entry::Assistant(AssistantTurn {
                            reasoning: joined(&mut pending_reasoning),
                            text: None,
                            tool_calls: Vec::new(),
                        }));
                    }
                    entries.push(Entry::User(user_content(content)?));
                }
                ResponseRole::Assistant => {
                    let text = text_content(content)?;
                    entries.push(Entry::Assistant(AssistantTurn {
                        reasoning: joined(&mut pending_reasoning),
                        text: (!text.is_empty()).then_some(text),
                        tool_calls: Vec::new(),
                    }));
                }
            },
            ResponseInputItem::Reasoning(reasoning) => {
                let text = reasoning_input_text(reasoning);
                if !text.is_empty() {
                    pending_reasoning.push(text);
                }
            }
            ResponseInputItem::FunctionCall(_) => {
                let mut calls = Vec::new();
                while let Some(ResponseInputItem::FunctionCall(ResponseFunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
                })) = items.get(index)
                {
                    let arguments =
                        serde_json::from_str::<Map<String, Value>>(arguments).map_err(|error| {
                            ApiError::invalid(format!(
                                "function_call arguments must be a JSON object: {error}",
                            ))
                        })?;
                    crate::chat::require_non_empty(call_id, "function_call call_id")?;
                    crate::chat::require_non_empty(name, "function_call name")?;
                    calls.push(ToolCall {
                        id: call_id.clone(),
                        name: name.clone(),
                        arguments,
                    });
                    index += 1;
                }
                let mut results = BTreeMap::new();
                while let Some(ResponseInputItem::FunctionCallOutput(
                    ResponseFunctionCallOutput {
                        call_id, output, ..
                    },
                )) = items.get(index)
                {
                    if results
                        .insert(call_id.clone(), function_output(output)?)
                        .is_some()
                    {
                        return Err(ApiError::invalid(format!(
                            "duplicate function_call_output for {call_id}",
                        )));
                    }
                    index += 1;
                }
                let exchanges = calls
                    .into_iter()
                    .map(|call| {
                        let result = results.remove(&call.id).ok_or_else(|| {
                            ApiError::invalid(format!(
                                "function_call {} has no following output",
                                call.id,
                            ))
                        })?;
                        Ok(ToolExchange { call, result })
                    })
                    .collect::<Result<Vec<_>, ApiError>>()?;
                if let Some(id) = results.keys().next() {
                    return Err(ApiError::invalid(format!(
                        "function_call_output {id} has no matching call",
                    )));
                }
                entries.push(Entry::Assistant(AssistantTurn {
                    reasoning: joined(&mut pending_reasoning),
                    text: None,
                    tool_calls: exchanges,
                }));
                continue;
            }
            ResponseInputItem::FunctionCallOutput(ResponseFunctionCallOutput {
                call_id, ..
            }) => {
                return Err(ApiError::invalid(format!(
                    "function_call_output {call_id} has no preceding function_call",
                )));
            }
        }
        index += 1;
    }
    if !pending_reasoning.is_empty() {
        entries.push(Entry::Assistant(AssistantTurn {
            reasoning: joined(&mut pending_reasoning),
            text: None,
            tool_calls: Vec::new(),
        }));
    }
    let system = (!system.is_empty()).then(|| system.join("\n"));
    if entries.is_empty() {
        return Err(ApiError::invalid("input must not be empty"));
    }
    Conversation::new(system, entries).map_err(invalid)
}

fn reasoning_input_text(reasoning: &ResponseReasoningInput) -> String {
    let parts = if reasoning.content.is_empty() {
        &reasoning.summary
    } else {
        &reasoning.content
    };
    parts
        .iter()
        .map(|part| match part {
            ResponseReasoningInputContent::ReasoningText { text }
            | ResponseReasoningInputContent::SummaryText { text } => text.as_str(),
        })
        .collect::<Vec<_>>()
        .concat()
}

fn text_content(content: &ResponseMessageContent) -> Result<String, ApiError> {
    match content {
        ResponseMessageContent::Text(text) => Ok(text.clone()),
        ResponseMessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ResponseContentPart::InputText { text }
                | ResponseContentPart::OutputText { text, .. } => Ok(text.as_str()),
                ResponseContentPart::InputImage { .. } => Err(ApiError::invalid(
                    "images are not valid in system or assistant message content",
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| parts.concat()),
    }
}

fn image(url: &str) -> Result<magnitude_chat::request::ImageInput, ApiError> {
    media::image(url).map_err(|error| ApiError::invalid(error.to_string()))
}

fn user_content(content: &ResponseMessageContent) -> Result<Vec<UserPart>, ApiError> {
    let mut values = Vec::new();
    match content {
        ResponseMessageContent::Text(text) => {
            if !text.is_empty() {
                values.push(UserPart::Text(text.clone()));
            }
        }
        ResponseMessageContent::Parts(parts) => {
            for part in parts {
                match part {
                    ResponseContentPart::InputText { text }
                    | ResponseContentPart::OutputText { text, .. } => {
                        if !text.is_empty() {
                            values.push(UserPart::Text(text.clone()));
                        }
                    }
                    ResponseContentPart::InputImage { image_url } => {
                        values.push(UserPart::Image(image(image_url)?));
                    }
                }
            }
        }
    }
    Ok(values)
}

fn function_output(output: &FunctionCallOutput) -> Result<Vec<ToolResultPart>, ApiError> {
    let mut values = Vec::new();
    match output {
        FunctionCallOutput::Text(text) => {
            if !text.is_empty() {
                values.push(ToolResultPart::Text(text.clone()));
            }
        }
        FunctionCallOutput::Parts(parts) => {
            for part in parts {
                match part {
                    FunctionCallOutputPart::InputText { text } => {
                        if !text.is_empty() {
                            values.push(ToolResultPart::Text(text.clone()));
                        }
                    }
                    FunctionCallOutputPart::InputImage { image_url } => {
                        values.push(ToolResultPart::Image(image(image_url)?));
                    }
                }
            }
        }
    }
    Ok(values)
}

#[utoipa::path(post, path = "/v1/responses", operation_id = "createResponse", tag = "inference",
    request_body(content = ResponseCreateRequest, content_type = "application/json"),
    params(
        ("Magnitude-Include-Progress" = Option<bool>, Header, nullable = false, description = "Include Magnitude loading and inference progress events")
    ),
    responses(
        (status = 200, description = "OpenAI-compatible response or event stream", content(
            (ResponseObject = "application/json"),
            (String = "text/event-stream")
        )),
        (status = 400, description = "Invalid Responses request", body = ErrorResponse),
        (status = 404, description = "Requested model is unavailable", body = ErrorResponse),
        (status = 409, description = "Model unavailable", body = ErrorResponse),
        (status = 422, description = "Runtime target failed validation", body = ErrorResponse),
        (status = 500, description = "Inference failed", body = ErrorResponse),
        (status = 503, description = "Memory or request capacity is temporarily unavailable", body = ErrorResponse)
    )
)]
pub async fn responses(
    State(state): State<Serving>,
    headers: HeaderMap,
    payload: Result<Json<ResponseCreateRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = payload.map_err(|error| ApiError::invalid(error.body_text()))?;
    if request.stream {
        let (request_id, receiver) = start_response_stream(state, &headers, request).await?;
        let receiver = ReceiverStream::new(receiver).map(|value| {
            let event_type = value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("message")
                .to_owned();
            Ok::<_, Infallible>(Event::default().event(event_type).data(value.to_string()))
        });
        let response = Sse::new(receiver)
            .keep_alive(KeepAlive::default())
            .into_response();
        return Ok(with_request_id(response, &request_id));
    }
    let adapted = adapt(request)?;
    let id = state.next_id("resp_icn_");
    let created_at = unix_timestamp();
    let invocation = state
        .models
        .invoke(&adapted.model, None)
        .await
        .map_err(|error| ApiError::from(error).with_param("model"))?;
    let (output, completion) = collect(invocation.generate(adapted.request))
        .await
        .map_err(|error| ApiError::from(error).with_param("input"))?;
    let response = Json(from_result(
        &id,
        created_at,
        &adapted.model,
        &adapted.projection,
        &output,
        &completion,
    ))
    .into_response();
    Ok(with_request_id(response, &id))
}

/// Start a streamed response. Without progress, the stream opens only after
/// the model is bound and the request admitted, so those failures keep their
/// HTTP status; with progress it opens at once and reports them in-stream.
async fn start_response_stream(
    state: Serving,
    headers: &HeaderMap,
    request: ResponseCreateRequest,
) -> Result<(String, mpsc::Receiver<Value>), ApiError> {
    let adapted = adapt(request)?;
    let id = state.next_id("resp_icn_");
    let created_at = unix_timestamp();
    let include_progress = include_progress(headers);
    let (sender, receiver) = mpsc::channel::<Value>(32);
    let sequence = Arc::new(AtomicU64::new(0));
    let projector = StreamProjector::new(
        id.clone(),
        created_at,
        adapted.model.clone(),
        sender.clone(),
        sequence.clone(),
        adapted.projection,
    );
    if include_progress {
        let progress_id = id.clone();
        let progress: LoadProgress = Arc::new(move |load| {
            send_loading_progress(&sender, &sequence, &progress_id, load);
        });
        tokio::spawn(async move {
            if projector.created().await.is_err() {
                return;
            }
            let invocation = tokio::select! {
                result = state.models.invoke(&adapted.model, Some(progress)) => result,
                _ = projector.sender.closed() => return,
            };
            match invocation {
                Ok(invocation) => {
                    let stream = invocation.generate(adapted.request);
                    project(stream, Vec::new(), projector, true).await;
                }
                Err(error) => projector.fail(&ApiError::from(error).body).await,
            }
        });
        return Ok((id, receiver));
    }
    let invocation = state
        .models
        .invoke(&adapted.model, None)
        .await
        .map_err(|error| ApiError::from(error).with_param("model"))?;
    let mut stream = invocation.generate(adapted.request);
    let (_, before) = admitted(&mut stream)
        .await
        .map_err(|error| ApiError::from(error).with_param("input"))?;
    tokio::spawn(async move {
        if projector.created().await.is_ok() {
            project(stream, before, projector, false).await;
        }
    });
    Ok((id, receiver))
}

async fn project(
    mut stream: GenerationStream,
    before: Vec<GenerationEvent>,
    mut projector: StreamProjector,
    include_progress: bool,
) {
    if projector.in_progress().await.is_err() {
        return;
    }
    let mut pending = before.into_iter();
    loop {
        let event = match pending.next() {
            Some(event) => Some(event),
            None => tokio::select! {
                event = stream.next() => event,
                () = projector.sender.closed() => return,
            },
        };
        let result = match event {
            Some(GenerationEvent::Progress(progress)) => {
                if include_progress {
                    projector.progress(progress).await.map_err(ProjectionError::from)
                } else {
                    Ok(())
                }
            }
            Some(GenerationEvent::Admitted { .. }) => Ok(()),
            Some(GenerationEvent::Output { event, .. }) => projector.observe(event).await,
            Some(GenerationEvent::Completed(completion)) => {
                let _ = projector.finish(&completion).await;
                return;
            }
            Some(GenerationEvent::Failed(error)) => Err(ProjectionError::Failed(error)),
            None => Err(ProjectionError::Failed(ended_without_outcome())),
        };
        match result {
            Ok(()) => {}
            Err(ProjectionError::Disconnected) => return,
            Err(ProjectionError::Failed(error)) => {
                projector.fail(&ApiError::from(error).body).await;
                return;
            }
        }
    }
}

pub async fn responses_websocket(
    State(state): State<Serving>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    upgrade.on_upgrade(move |socket| serve_responses_websocket(socket, state, headers))
}

async fn send_websocket_value(socket: &mut WebSocket, value: Value) -> bool {
    socket
        .send(Message::Text(value.to_string().into()))
        .await
        .is_ok()
}

pub(crate) fn websocket_error_event(error: &ApiErrorBody, sequence_number: u64) -> Value {
    serde_json::json!({
        "type": "error",
        "code": error.code,
        "message": error.message,
        "param": error.param,
        "sequence_number": sequence_number,
    })
}

async fn send_websocket_error(
    socket: &mut WebSocket,
    error: &ApiErrorBody,
    sequence: &AtomicU64,
) -> bool {
    let sequence_number = sequence.fetch_add(1, Ordering::Relaxed);
    send_websocket_value(socket, websocket_error_event(error, sequence_number)).await
}

/// Resolve one WebSocket message into its full logical request against the
/// connection's history. `generate: false` requests are recorded, not run.
pub(crate) fn websocket_logical_request(
    mut request: Value,
    history: &HashMap<String, Value>,
) -> Result<(Value, bool), String> {
    let object = request
        .as_object_mut()
        .ok_or_else(|| "WebSocket message must be a JSON object".to_owned())?;
    if object.get("type").and_then(Value::as_str) != Some("response.create") {
        return Err("WebSocket message type must be response.create".to_owned());
    }
    let generate = object.remove("generate").and_then(|value| value.as_bool()) != Some(false);
    object.remove("type");
    if let Some(previous_id) = object
        .get("previous_response_id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
    {
        let previous = history
            .get(&previous_id)
            .ok_or_else(|| format!("Unknown previous_response_id: {previous_id}"))?;
        let previous_input = previous
            .get("input")
            .and_then(Value::as_array)
            .ok_or_else(|| "Previous WebSocket request input was not an array".to_owned())?;
        let incremental_input = object
            .get("input")
            .and_then(Value::as_array)
            .ok_or_else(|| "Incremental WebSocket request input must be an array".to_owned())?;
        let mut input = previous_input.clone();
        input.extend(incremental_input.iter().cloned());
        object.insert("input".to_owned(), Value::Array(input));
        object.remove("previous_response_id");
    }
    object.insert("stream".to_owned(), Value::Bool(true));
    Ok((request, generate))
}

fn warmup_events(id: &str) -> [Value; 2] {
    [
        serde_json::json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": { "id": id, "status": "in_progress" },
        }),
        serde_json::json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": id,
                "status": "completed",
                "usage": {
                    "input_tokens": 0,
                    "input_tokens_details": null,
                    "output_tokens": 0,
                    "output_tokens_details": null,
                    "total_tokens": 0,
                }
            },
        }),
    ]
}

async fn serve_responses_websocket(mut socket: WebSocket, state: Serving, headers: HeaderMap) {
    let mut history = HashMap::<String, Value>::new();
    while let Some(message) = socket.next().await {
        let sequence = AtomicU64::new(0);
        let request = match message {
            Ok(Message::Text(text)) => serde_json::from_str::<Value>(&text),
            Ok(Message::Binary(bytes)) => serde_json::from_slice::<Value>(&bytes),
            Ok(Message::Ping(bytes)) => {
                if socket.send(Message::Pong(bytes)).await.is_err() {
                    return;
                }
                continue;
            }
            Ok(Message::Pong(_)) => continue,
            Ok(Message::Close(_)) | Err(_) => return,
        };
        let request = match request {
            Ok(request) => request,
            Err(error) => {
                let error = ApiError::invalid(format!("Invalid JSON: {error}"));
                if !send_websocket_error(&mut socket, &error.body, &sequence).await {
                    return;
                }
                continue;
            }
        };
        let (logical, generate) = match websocket_logical_request(request, &history) {
            Ok(request) => request,
            Err(error) => {
                let error = ApiError::invalid(error);
                if !send_websocket_error(&mut socket, &error.body, &sequence).await {
                    return;
                }
                continue;
            }
        };
        if !generate {
            let id = state.next_id("resp_icn_");
            for event in warmup_events(&id) {
                if !send_websocket_value(&mut socket, event).await {
                    return;
                }
            }
            history.insert(id, logical);
            continue;
        }
        let request = match serde_json::from_value::<ResponseCreateRequest>(logical.clone()) {
            Ok(request) => request,
            Err(error) => {
                let error = ApiError::invalid(error.to_string());
                if !send_websocket_error(&mut socket, &error.body, &sequence).await {
                    return;
                }
                continue;
            }
        };
        let (id, mut receiver) = match start_response_stream(state.clone(), &headers, request).await
        {
            Ok(stream) => stream,
            Err(error) => {
                if !send_websocket_error(&mut socket, &error.body, &sequence).await {
                    return;
                }
                continue;
            }
        };
        while let Some(event) = receiver.recv().await {
            if !send_websocket_value(&mut socket, event).await {
                return;
            }
        }
        history.insert(id, logical);
    }
}

#[cfg(test)]
mod websocket_tests {
    use super::*;

    #[test]
    fn errors_use_the_standard_top_level_event_shape() {
        let error = ApiError::from(ServingError::Chat(
            magnitude_chat::ChatError::ContextLengthExceeded {
                prompt_tokens: 33,
                context_tokens: 32,
            },
        ))
        .with_param("input");
        assert_eq!(
            websocket_error_event(&error.body, 7),
            serde_json::json!({
                "type": "error",
                "code": "context_length_exceeded",
                "message": "prompt is too long: 33 tokens leave no generation capacity in a 32-token context",
                "param": "input",
                "sequence_number": 7,
            })
        );
    }

    #[test]
    fn warmup_is_retained_for_incremental_generation() {
        let warmup = serde_json::json!({
            "type": "response.create",
            "model": "local-model",
            "input": [{"role": "user", "content": "hello"}],
            "generate": false,
        });
        let (warmup, generate) = websocket_logical_request(warmup, &HashMap::new()).unwrap();
        assert!(!generate);
        let history = HashMap::from([("warm-1".to_owned(), warmup)]);
        let request = serde_json::json!({
            "type": "response.create",
            "model": "local-model",
            "input": [],
            "previous_response_id": "warm-1",
        });
        let (logical, generate) = websocket_logical_request(request, &history).unwrap();
        assert!(generate);
        assert!(logical.get("previous_response_id").is_none());
        assert_eq!(logical["input"].as_array().unwrap().len(), 1);
        assert_eq!(logical["stream"], Value::Bool(true));
    }

    #[test]
    fn incremental_items_append_to_the_previous_logical_request() {
        let history = HashMap::from([(
            "resp-1".to_owned(),
            serde_json::json!({
                "model": "local-model",
                "input": [{"role": "user", "content": "hello"}],
                "stream": true,
            }),
        )]);
        let request = serde_json::json!({
            "type": "response.create",
            "model": "local-model",
            "input": [
                {"type": "message", "role": "assistant", "content": []},
                {"type": "function_call_output", "call_id": "call-1", "output": "ok"}
            ],
            "previous_response_id": "resp-1",
        });
        let (logical, _) = websocket_logical_request(request, &history).unwrap();
        assert_eq!(logical["input"].as_array().unwrap().len(), 3);
    }
}
