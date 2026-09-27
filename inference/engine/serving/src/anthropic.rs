//! Anthropic Messages (stream and non-stream) and `count_tokens`. Counting
//! renders exactly as generation does and never binds or loads a model.
use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::num::NonZeroU32;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use magnitude_chat::output::{Completion, Output, OutputEvent, Termination};
use magnitude_chat::request::{
    AssistantTurn, Conversation, Entry, GenerationControls, OutputFormat, PromptCache, ToolCall,
    ToolDefinition, ToolExchange, ToolResultPart, Tools, UserPart,
};
use magnitude_chat::{ChatInput, EndOfGeneration, GenerationRequest, ReasoningIntent, ToolChoice};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use utoipa::ToSchema;

use crate::chat::{ReasoningEffortRequest, admitted, collect, ended_without_outcome};
use crate::error::{ApiError, ServingError};
use crate::source::{GenerationEvent, GenerationStream};
use crate::{Serving, media};

const LOCAL_THINKING_SIGNATURE: &str = "magnitude-local-v1";

#[derive(Debug, Deserialize, ToSchema)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    #[schema(nullable = false)]
    pub system: Option<SystemPrompt>,
    pub max_tokens: u32,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    #[serde(default)]
    pub tools: Vec<Tool>,
    #[schema(nullable = false)]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[schema(nullable = false)]
    pub thinking: Option<Thinking>,
    pub metadata: Option<Value>,
    pub output_config: Option<OutputConfig>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct OutputConfig {
    #[schema(nullable = false)]
    pub effort: Option<ReasoningEffortRequest>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemBlock {
    Text {
        text: String,
        #[serde(default)]
        #[serde(rename = "cache_control")]
        _cache_control: Option<Value>,
    },
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct Message {
    pub role: Role,
    pub content: Content,
}

#[derive(Debug, Clone, Copy, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
        #[serde(default)]
        #[serde(rename = "cache_control")]
        _cache_control: Option<Value>,
    },
    Thinking {
        thinking: String,
        #[serde(default)]
        #[serde(rename = "signature")]
        _signature: Option<String>,
    },
    Image {
        source: ImageSource,
        #[serde(default)]
        #[serde(rename = "cache_control")]
        _cache_control: Option<Value>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Map<String, Value>,
        #[serde(default)]
        #[serde(rename = "cache_control")]
        _cache_control: Option<Value>,
    },
    ToolResult {
        tool_use_id: String,
        #[schema(nullable = false)]
        content: Option<ToolResultContent>,
        #[serde(default)]
        #[serde(rename = "is_error")]
        _is_error: bool,
        #[serde(default)]
        #[serde(rename = "cache_control")]
        _cache_control: Option<Value>,
    },
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 {
        media_type: String,
        data: String,
    },
    Url {
        #[serde(rename = "url")]
        _url: String,
    },
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    Text {
        text: String,
        #[serde(default)]
        #[serde(rename = "cache_control")]
        _cache_control: Option<Value>,
    },
    Image {
        source: ImageSource,
        #[serde(default)]
        #[serde(rename = "cache_control")]
        _cache_control: Option<Value>,
    },
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Map<String, Value>,
    #[serde(default)]
    #[serde(rename = "cache_control")]
    pub _cache_control: Option<Value>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    Auto {
        #[serde(default)]
        disable_parallel_tool_use: bool,
    },
    Any {
        #[serde(default)]
        disable_parallel_tool_use: bool,
    },
    Tool {
        name: String,
        #[serde(default)]
        disable_parallel_tool_use: bool,
    },
    None,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Thinking {
    /// Enables reasoning with a hard reasoning-token budget.
    Enabled { budget_tokens: u32 },
    Adaptive,
    Disabled,
}

pub struct AdaptedRequest {
    pub model: String,
    pub request: GenerationRequest,
    pub stream: bool,
}

fn invalid(error: magnitude_chat::ChatError) -> ApiError {
    ApiError::invalid(error.to_string())
}

pub fn adapt(request: MessagesRequest) -> Result<AdaptedRequest, ApiError> {
    if request.model.is_empty() {
        return Err(ApiError::invalid("model is required"));
    }
    if request.messages.is_empty() {
        return Err(ApiError::invalid("messages must not be empty"));
    }
    let max_tokens = NonZeroU32::new(request.max_tokens)
        .ok_or_else(|| ApiError::invalid("max_tokens must be greater than zero"))?;
    if request.top_k.is_some() {
        return Err(ApiError::invalid(
            "top_k is not supported by this local runtime",
        ));
    }
    if request
        .metadata
        .as_ref()
        .is_some_and(|value| !value.is_object())
    {
        return Err(ApiError::invalid("metadata must be an object"));
    }
    let conversation = context(request.system, request.messages)?;
    let definitions = request
        .tools
        .into_iter()
        .map(|tool| ToolDefinition {
            name: tool.name,
            description: tool.description,
            parameters: tool.input_schema,
        })
        .collect();
    let (choice, parallel) = match request.tool_choice {
        None => (ToolChoice::Auto, true),
        Some(AnthropicToolChoice::Auto {
            disable_parallel_tool_use,
        }) => (ToolChoice::Auto, !disable_parallel_tool_use),
        Some(AnthropicToolChoice::Any {
            disable_parallel_tool_use,
        }) => (ToolChoice::Required, !disable_parallel_tool_use),
        Some(AnthropicToolChoice::Tool {
            name,
            disable_parallel_tool_use,
        }) => (ToolChoice::Named(name), !disable_parallel_tool_use),
        Some(AnthropicToolChoice::None) => (ToolChoice::None, false),
    };
    let tools = Tools::new(definitions, choice, parallel).map_err(invalid)?;
    let effort = request
        .output_config
        .and_then(|config| config.effort)
        .map(|effort| effort.normalize())
        .transpose()?;
    if effort == Some("none") {
        return Err(ApiError::invalid(
            "output_config.effort must select an enabled reasoning behavior",
        ));
    }
    let budget = match request.thinking {
        Some(Thinking::Enabled { budget_tokens }) => Some(
            NonZeroU32::new(budget_tokens)
                .ok_or_else(|| ApiError::invalid("thinking budget_tokens must be positive"))?,
        ),
        Some(Thinking::Adaptive | Thinking::Disabled) | None => None,
    };
    let reasoning = match (request.thinking, effort) {
        (Some(Thinking::Disabled), Some(_)) => {
            return Err(ApiError::invalid(
                "output_config.effort cannot be used when thinking is disabled",
            ));
        }
        (Some(Thinking::Disabled), None) => ReasoningIntent::Disabled,
        (_, Some(effort)) => ReasoningIntent::Effort {
            effort: effort.into(),
        },
        (Some(Thinking::Enabled { .. }), None) => ReasoningIntent::Enabled,
        (Some(Thinking::Adaptive) | None, None) => ReasoningIntent::ModelDefault,
    };
    for stop in &request.stop_sequences {
        crate::chat::require_non_empty(stop, "stop sequence")?;
    }
    let sampling = crate::responses::sampling(
        request.temperature.unwrap_or(1.0),
        request.top_p.unwrap_or(1.0),
    )?;
    Ok(AdaptedRequest {
        model: request.model,
        request: GenerationRequest {
            input: ChatInput {
                conversation,
                tools,
                reasoning,
                output: OutputFormat::Text,
                template_arguments: Map::new(),
            },
            controls: GenerationControls {
                max_output_tokens: Some(max_tokens),
                sampling,
                stops: request.stop_sequences,
                end_of_generation: EndOfGeneration::Stop,
                reasoning_budget: budget,
                prompt_cache: PromptCache::Allowed,
            },
        },
        stream: request.stream,
    })
}

// Claude Code attribution projection. Claude Code prepends provider-reserved
// billing metadata as the leading system text,
// `x-anthropic-billing-header: …; cch=<stamp>;<optional real prompt>`, which
// api.anthropic.com strips before the model sees it. This adapter is the one
// place wire bytes become model-visible text, so it owns the same stripping:
// the per-request cch stamp would otherwise defeat prompt-prefix reuse and
// the metadata would become model-visible prompt content. Recognition is
// strict and positional: leading block, exact sentinel, `cch=` plus five
// bytes plus `;`. Everything else passes through byte-identical; an
// unrecognized sentinel shape is preserved with a diagnostic, never guessed
// at. Magnitude-launched Claude Code also sets
// CLAUDE_CODE_ATTRIBUTION_HEADER=0, so this projection covers only clients
// Magnitude did not launch.
const ATTRIBUTION_SENTINEL: &str = "x-anthropic-billing-header:";

fn project_system(mut blocks: Vec<String>) -> Vec<String> {
    let Some(first) = blocks.first_mut() else {
        return blocks;
    };
    match project_attribution(std::mem::take(first)) {
        Some(text) => *first = text,
        None => {
            blocks.remove(0);
        }
    }
    blocks
}

fn project_attribution(text: String) -> Option<String> {
    if !text.starts_with(ATTRIBUTION_SENTINEL) {
        return Some(text);
    }
    let Some(found) = text[ATTRIBUTION_SENTINEL.len()..].find("cch=") else {
        tracing::warn!("unrecognized Claude Code attribution shape; preserving system content");
        return Some(text);
    };
    let stamp_end = ATTRIBUTION_SENTINEL.len() + found + "cch=".len() + 5;
    if text.as_bytes().get(stamp_end) != Some(&b';') {
        tracing::warn!("unrecognized Claude Code attribution shape; preserving system content");
        return Some(text);
    }
    let suffix = &text[stamp_end + 1..];
    if suffix.is_empty() {
        None
    } else {
        Some(suffix.to_owned())
    }
}

fn context(system: Option<SystemPrompt>, messages: Vec<Message>) -> Result<Conversation, ApiError> {
    let system = match system {
        None => None,
        Some(system) => {
            let blocks: Vec<String> = match system {
                SystemPrompt::Text(text) => vec![text],
                SystemPrompt::Blocks(blocks) => blocks
                    .into_iter()
                    .map(|SystemBlock::Text { text, .. }| text)
                    .collect(),
            };
            let had_blocks = !blocks.is_empty();
            let blocks = project_system(blocks);
            if had_blocks && blocks.is_empty() {
                // The entire system content was attribution metadata.
                None
            } else {
                let text = blocks.join("\n");
                crate::chat::require_non_empty(&text, "system")?;
                Some(text)
            }
        }
    };
    let mut entries = Vec::new();
    let mut messages = messages.into_iter().peekable();
    while let Some(message) = messages.next() {
        match message.role {
            Role::System => entries.push(Entry::User(system_role_content(message.content)?)),
            Role::User => entries.push(Entry::User(user_content(message.content)?)),
            Role::Assistant => {
                let (reasoning, text, calls) = assistant_content(message.content)?;
                if reasoning.is_none() && text.is_none() && calls.is_empty() {
                    return Err(ApiError::invalid(
                        "assistant message content must not be empty",
                    ));
                }
                let (exchanges, trailing_user_content) = if calls.is_empty() {
                    (Vec::new(), Vec::new())
                } else {
                    let next = messages.next().ok_or_else(|| {
                        ApiError::invalid(
                            "assistant tool_use blocks require a following user tool_result message",
                        )
                    })?;
                    if !matches!(next.role, Role::User) {
                        return Err(ApiError::invalid(
                            "assistant tool_use blocks must be followed by user tool_result blocks",
                        ));
                    }
                    let (mut results, trailing_user_content) = tool_results(next.content)?;
                    let mut exchanges = Vec::with_capacity(calls.len());
                    for call in calls {
                        let result = results.remove(&call.id).ok_or_else(|| {
                            ApiError::invalid(format!(
                                "tool_use {} has no matching tool_result",
                                call.id
                            ))
                        })?;
                        exchanges.push(ToolExchange { call, result });
                    }
                    if let Some(tool_use_id) = results.keys().next() {
                        return Err(ApiError::invalid(format!(
                            "tool_result {tool_use_id} has no matching tool_use",
                        )));
                    }
                    (exchanges, trailing_user_content)
                };
                entries.push(Entry::Assistant(AssistantTurn {
                    reasoning,
                    text,
                    tool_calls: exchanges,
                }));
                if !trailing_user_content.is_empty() {
                    entries.push(Entry::User(trailing_user_content));
                }
            }
        }
    }
    Conversation::new(system, entries).map_err(invalid)
}

fn nonempty_text(text: String, field: &str) -> Result<String, ApiError> {
    crate::chat::require_non_empty(&text, field)?;
    Ok(text)
}

fn nonempty_parts<T>(values: Vec<T>, field: &str) -> Result<Vec<T>, ApiError> {
    if values.is_empty() {
        return Err(ApiError::invalid(format!("{field} must not be empty")));
    }
    Ok(values)
}

// System-role messages are the Anthropic protocol's mid-conversation operator
// channel; Claude Code uses it to surface text the user typed mid-turn. Local
// chat templates have no mid-sequence system turn and canonical context
// carries exactly one leading system prompt, so these become user entries at
// their original position, never part of the leading system prompt.
fn system_role_content(content: Content) -> Result<Vec<UserPart>, ApiError> {
    let mut values = Vec::new();
    for block in blocks(content) {
        match block {
            ContentBlock::Text { text, .. } => {
                values.push(UserPart::Text(nonempty_text(text, "system message text")?))
            }
            _ => {
                return Err(ApiError::invalid(
                    "system message content must contain only text",
                ));
            }
        }
    }
    nonempty_parts(values, "system message content")
}

fn user_content(content: Content) -> Result<Vec<UserPart>, ApiError> {
    let mut values = Vec::new();
    for block in blocks(content) {
        match block {
            ContentBlock::Text { text, .. } => {
                values.push(UserPart::Text(nonempty_text(text, "user text")?))
            }
            ContentBlock::Image { source, .. } => values.push(UserPart::Image(image(source)?)),
            ContentBlock::ToolResult { .. } => {
                return Err(ApiError::invalid(
                    "tool_result blocks require a preceding assistant tool_use message",
                ));
            }
            ContentBlock::Thinking { .. } | ContentBlock::ToolUse { .. } => {
                return Err(ApiError::invalid(
                    "invalid content block for a user message",
                ));
            }
        }
    }
    nonempty_parts(values, "user content")
}

type AssistantContent = (Option<String>, Option<String>, Vec<ToolCall>);

fn assistant_content(content: Content) -> Result<AssistantContent, ApiError> {
    let mut reasoning = Vec::new();
    let mut text = Vec::new();
    let mut calls = Vec::new();
    let mut ids = BTreeSet::new();
    for block in blocks(content) {
        match block {
            ContentBlock::Text { text: value, .. } => text.push(value),
            ContentBlock::Thinking { thinking, .. } => reasoning.push(thinking),
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                crate::chat::require_non_empty(&id, "tool_use id")?;
                crate::chat::require_non_empty(&name, "tool_use name")?;
                if !ids.insert(id.clone()) {
                    return Err(ApiError::invalid(format!("duplicate tool_use id: {id}")));
                }
                calls.push(ToolCall {
                    id,
                    name,
                    arguments: input,
                });
            }
            ContentBlock::Image { .. } | ContentBlock::ToolResult { .. } => {
                return Err(ApiError::invalid(
                    "invalid content block for an assistant message",
                ));
            }
        }
    }
    Ok((
        joined(reasoning, "assistant thinking")?,
        joined(text, "assistant text")?,
        calls,
    ))
}

type ToolResults = (BTreeMap<String, Vec<ToolResultPart>>, Vec<UserPart>);

fn tool_results(content: Content) -> Result<ToolResults, ApiError> {
    let mut results = BTreeMap::new();
    let mut trailing_content = Vec::new();
    for block in blocks(content) {
        let ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } = block
        else {
            match block {
                ContentBlock::Text { text, .. } => trailing_content.push(UserPart::Text(
                    nonempty_text(text, "user text after tool results")?,
                )),
                ContentBlock::Image { source, .. } => {
                    trailing_content.push(UserPart::Image(image(source)?))
                }
                ContentBlock::Thinking { .. } | ContentBlock::ToolUse { .. } => {
                    return Err(ApiError::invalid(
                        "invalid content block after tool_result blocks",
                    ));
                }
                ContentBlock::ToolResult { .. } => unreachable!("variant was matched above"),
            }
            continue;
        };
        if !trailing_content.is_empty() {
            return Err(ApiError::invalid(
                "tool_result blocks must precede other user content",
            ));
        }
        let values = match content {
            None => Vec::new(),
            Some(ToolResultContent::Text(text)) => {
                vec![ToolResultPart::Text(nonempty_text(text, "tool result")?)]
            }
            Some(ToolResultContent::Blocks(blocks)) => blocks
                .into_iter()
                .map(|block| match block {
                    ToolResultBlock::Text { text, .. } => {
                        Ok(ToolResultPart::Text(nonempty_text(text, "tool result text")?))
                    }
                    ToolResultBlock::Image { source, .. } => {
                        Ok(ToolResultPart::Image(image(source)?))
                    }
                })
                .collect::<Result<Vec<_>, ApiError>>()?,
        };
        if results.insert(tool_use_id.clone(), values).is_some() {
            return Err(ApiError::invalid(format!(
                "duplicate tool_result for {tool_use_id}"
            )));
        }
    }
    Ok((results, trailing_content))
}

fn blocks(content: Content) -> Vec<ContentBlock> {
    match content {
        Content::Text(text) => vec![ContentBlock::Text {
            text,
            _cache_control: None,
        }],
        Content::Blocks(blocks) => blocks,
    }
}

fn joined(values: Vec<String>, field: &str) -> Result<Option<String>, ApiError> {
    if values.is_empty() {
        Ok(None)
    } else {
        nonempty_text(values.join(""), field).map(Some)
    }
}

fn image(source: ImageSource) -> Result<magnitude_chat::request::ImageInput, ApiError> {
    match source {
        ImageSource::Base64 { media_type, data } => {
            media::image(&format!("data:{media_type};base64,{data}"))
                .map_err(|error| ApiError::invalid(error.to_string()))
        }
        ImageSource::Url { .. } => Err(ApiError::invalid(
            "network image URLs are not supported; use a base64 image source",
        )),
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorEnvelope {
    pub r#type: &'static str,
    pub error: ErrorBody,
    pub request_id: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorBody {
    pub r#type: &'static str,
    pub message: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct MessageResponse {
    pub id: String,
    pub r#type: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ResponseContentBlock>,
    pub stop_reason: &'static str,
    pub stop_sequence: Option<String>,
    pub usage: UsageResponse,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentBlock {
    Thinking {
        thinking: String,
        signature: &'static str,
    },
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Map<String, Value>,
    },
}

#[derive(Debug, Serialize, ToSchema)]
pub struct UsageResponse {
    pub input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CountTokensResponse {
    pub input_tokens: u64,
}

pub fn message(id: &str, model: &str, output: &Output, completion: &Completion) -> MessageResponse {
    let mut content = Vec::new();
    if let Some(reasoning) = &output.reasoning {
        content.push(ResponseContentBlock::Thinking {
            thinking: reasoning.clone(),
            signature: LOCAL_THINKING_SIGNATURE,
        });
    }
    if let Some(text) = &output.text {
        content.push(ResponseContentBlock::Text { text: text.clone() });
    }
    content.extend(
        output
            .tool_calls
            .iter()
            .map(|call| ResponseContentBlock::ToolUse {
                id: call.id.clone(),
                name: call.name.clone(),
                input: call.arguments.clone(),
            }),
    );
    let (stop_reason, stop_sequence) = stop(&completion.termination);
    MessageResponse {
        id: id.to_owned(),
        r#type: "message",
        role: "assistant",
        model: model.to_owned(),
        content,
        stop_reason,
        stop_sequence: stop_sequence.map(str::to_owned),
        usage: UsageResponse {
            input_tokens: completion.usage.input_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: completion.usage.cached_input_tokens,
            output_tokens: completion.usage.output_tokens,
        },
    }
}

fn stop(termination: &Termination) -> (&'static str, Option<&str>) {
    match termination {
        Termination::Natural => ("end_turn", None),
        Termination::StopSequence(sequence) => ("stop_sequence", Some(sequence.as_str())),
        Termination::OutputLimit => ("max_tokens", None),
        Termination::ToolCalls => ("tool_use", None),
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    MessageStart {
        message: StreamMessage,
    },
    ContentBlockStart {
        index: usize,
        content_block: StreamContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: StreamDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: OutputUsage,
    },
    MessageStop,
    Error {
        error: StreamError,
        request_id: String,
    },
}

#[derive(Debug, Serialize)]
struct StreamMessage {
    id: String,
    r#type: &'static str,
    role: &'static str,
    model: String,
    content: Vec<ResponseContentBlock>,
    stop_reason: Option<&'static str>,
    stop_sequence: Option<String>,
    usage: UsageResponse,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamContentBlock {
    Thinking { thinking: String, signature: String },
    Text { text: String },
    ToolUse {
        id: String,
        name: String,
        input: Map<String, Value>,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamDelta {
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Serialize)]
struct MessageDelta {
    stop_reason: &'static str,
    stop_sequence: Option<String>,
}

#[derive(Debug, Serialize)]
struct OutputUsage {
    output_tokens: u64,
}

#[derive(Debug, Serialize)]
struct StreamError {
    r#type: &'static str,
    message: String,
}

type SseSender = mpsc::Sender<Result<Event, Infallible>>;

/// The consumer disconnected; dropping the generation cancels it.
struct Disconnected;

struct StreamProjector {
    id: String,
    model: String,
    request_id: String,
    sender: SseSender,
    next_index: usize,
    reasoning_index: Option<usize>,
    text_index: Option<usize>,
    tools: BTreeMap<usize, usize>,
}

impl StreamProjector {
    async fn send(&self, event: &'static str, value: &StreamEvent) -> Result<(), Disconnected> {
        let data = serde_json::to_string(value).expect("stream events serialize");
        self.sender
            .send(Ok(Event::default().event(event).data(data)))
            .await
            .map_err(|_| Disconnected)
    }

    async fn start(&self, input_tokens: u64) -> Result<(), Disconnected> {
        self.send(
            "message_start",
            &StreamEvent::MessageStart {
                message: StreamMessage {
                    id: self.id.clone(),
                    r#type: "message",
                    role: "assistant",
                    model: self.model.clone(),
                    content: Vec::new(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: UsageResponse {
                        input_tokens,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                        output_tokens: 0,
                    },
                },
            },
        )
        .await
    }

    async fn begin_block(&mut self, block: StreamContentBlock) -> Result<usize, Disconnected> {
        let index = self.next_index;
        self.next_index += 1;
        self.send(
            "content_block_start",
            &StreamEvent::ContentBlockStart {
                index,
                content_block: block,
            },
        )
        .await?;
        Ok(index)
    }

    async fn delta(&self, index: usize, delta: StreamDelta) -> Result<(), Disconnected> {
        self.send(
            "content_block_delta",
            &StreamEvent::ContentBlockDelta { index, delta },
        )
        .await
    }

    async fn observe(&mut self, event: OutputEvent) -> Result<(), ProjectionError> {
        match event {
            OutputEvent::Started | OutputEvent::ToolCallFinished { .. } => {}
            OutputEvent::ReasoningDelta(thinking) => {
                let index = match self.reasoning_index {
                    Some(index) => index,
                    None => {
                        let index = self
                            .begin_block(StreamContentBlock::Thinking {
                                thinking: String::new(),
                                signature: String::new(),
                            })
                            .await?;
                        self.reasoning_index = Some(index);
                        index
                    }
                };
                self.delta(index, StreamDelta::ThinkingDelta { thinking })
                    .await?;
            }
            OutputEvent::TextDelta(text) => {
                let index = match self.text_index {
                    Some(index) => index,
                    None => {
                        let index = self
                            .begin_block(StreamContentBlock::Text {
                                text: String::new(),
                            })
                            .await?;
                        self.text_index = Some(index);
                        index
                    }
                };
                self.delta(index, StreamDelta::TextDelta { text }).await?;
            }
            OutputEvent::ToolCallStarted { index, id, name } => {
                let output_index = self
                    .begin_block(StreamContentBlock::ToolUse {
                        id,
                        name,
                        input: Map::new(),
                    })
                    .await?;
                self.tools.insert(index, output_index);
            }
            OutputEvent::ToolInputDelta { index, fragment } => {
                let output_index = *self.tools.get(&index).ok_or_else(|| {
                    ProjectionError::Failed(ServingError::Internal(
                        "tool input arrived before tool start".into(),
                    ))
                })?;
                self.delta(
                    output_index,
                    StreamDelta::InputJsonDelta {
                        partial_json: fragment,
                    },
                )
                .await?;
            }
        }
        Ok(())
    }

    async fn finish(&self, completion: &Completion) -> Result<(), Disconnected> {
        if let Some(index) = self.reasoning_index {
            self.delta(
                index,
                StreamDelta::SignatureDelta {
                    signature: LOCAL_THINKING_SIGNATURE.to_owned(),
                },
            )
            .await?;
        }
        for index in 0..self.next_index {
            self.send(
                "content_block_stop",
                &StreamEvent::ContentBlockStop { index },
            )
            .await?;
        }
        let (stop_reason, stop_sequence) = stop(&completion.termination);
        self.send(
            "message_delta",
            &StreamEvent::MessageDelta {
                delta: MessageDelta {
                    stop_reason,
                    stop_sequence: stop_sequence.map(str::to_owned),
                },
                usage: OutputUsage {
                    output_tokens: completion.usage.output_tokens,
                },
            },
        )
        .await?;
        self.send("message_stop", &StreamEvent::MessageStop).await
    }

    async fn fail(&self, error: ServingError) {
        let _ = self
            .send(
                "error",
                &StreamEvent::Error {
                    error: StreamError {
                        r#type: "api_error",
                        message: error.to_string(),
                    },
                    request_id: self.request_id.clone(),
                },
            )
            .await;
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

async fn project(mut stream: GenerationStream, mut projector: StreamProjector) {
    loop {
        let event = tokio::select! {
            event = stream.next() => event,
            () = projector.sender.closed() => return,
        };
        let result = match event {
            Some(GenerationEvent::Progress(_) | GenerationEvent::Admitted { .. }) => Ok(()),
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
                projector.fail(error).await;
                return;
            }
        }
    }
}

fn validate_anthropic_version(headers: &HeaderMap) -> Result<(), ApiError> {
    let version = headers
        .get("anthropic-version")
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| ApiError::invalid("anthropic-version header is required"))?;
    if version != "2023-06-01" {
        return Err(ApiError::invalid(format!(
            "unsupported anthropic-version: {version}"
        )));
    }
    Ok(())
}

fn with_anthropic_request_id(mut response: Response, request_id: &str) -> Response {
    if let Ok(value) = request_id.parse() {
        response.headers_mut().insert("request-id", value);
    }
    response
}

fn anthropic_error_response(request_id: String, error: ApiError) -> Response {
    let error_type = match error.status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
        _ => "api_error",
    };
    let response = (
        error.status,
        Json(ErrorEnvelope {
            r#type: "error",
            error: ErrorBody {
                r#type: error_type,
                message: error.body.message,
            },
            request_id: request_id.clone(),
        }),
    )
        .into_response();
    with_anthropic_request_id(response, &request_id)
}

#[utoipa::path(
    post,
    path = "/anthropic/v1/messages/count_tokens",
    operation_id = "countAnthropicMessageTokens",
    tag = "anthropic",
    request_body = MessagesRequest,
    responses(
        (status = 200, description = "Anthropic-compatible input token count", body = CountTokensResponse),
        (status = 400, description = "Invalid Anthropic request", body = ErrorEnvelope),
        (status = 404, description = "Model is not installed", body = ErrorEnvelope),
        (status = 409, description = "Installed model cannot be resolved for serving", body = ErrorEnvelope),
        (status = 422, description = "Installed model failed integrity validation", body = ErrorEnvelope),
        (status = 500, description = "Token counting failed", body = ErrorEnvelope)
    )
)]
pub async fn anthropic_count_tokens(
    State(state): State<Serving>,
    headers: HeaderMap,
    payload: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let request_id = state.next_id("req_icn_");
    let result = async {
        validate_anthropic_version(&headers)?;
        let Json(request) = payload.map_err(|error| ApiError::invalid(error.body_text()))?;
        let adapted = adapt(request)?;
        let host = state
            .models
            .host(&adapted.model)
            .await
            .map_err(ApiError::from)?;
        let input = adapted.request.input;
        let count = tokio::task::spawn_blocking(move || host.count(&input))
            .await
            .map_err(|error| ApiError::server(format!("token-count task failed: {error}")))?
            .map_err(ApiError::from)?;
        Ok::<_, ApiError>(
            Json(CountTokensResponse {
                input_tokens: count,
            })
            .into_response(),
        )
    }
    .await;
    match result {
        Ok(response) => with_anthropic_request_id(response, &request_id),
        Err(error) => anthropic_error_response(request_id, error),
    }
}

#[utoipa::path(
    post,
    path = "/anthropic/v1/messages",
    operation_id = "createAnthropicMessage",
    tag = "anthropic",
    request_body = MessagesRequest,
    responses(
        (status = 200, description = "Anthropic-compatible message or event stream", body = MessageResponse),
        (status = 400, description = "Invalid Anthropic request", body = ErrorEnvelope),
        (status = 404, description = "Requested model is unavailable", body = ErrorEnvelope),
        (status = 409, description = "Model unavailable", body = ErrorEnvelope),
        (status = 500, description = "Inference failed", body = ErrorEnvelope),
        (status = 503, description = "Memory or request capacity is temporarily unavailable", body = ErrorEnvelope)
    )
)]
pub async fn anthropic_messages(
    State(state): State<Serving>,
    headers: HeaderMap,
    payload: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let request_id = state.next_id("req_icn_");
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return anthropic_error_response(request_id, ApiError::invalid(error.body_text()));
        }
    };
    match messages(state, headers, request, request_id.clone()).await {
        Ok(response) => with_anthropic_request_id(response, &request_id),
        Err(error) => anthropic_error_response(request_id, error),
    }
}

async fn messages(
    state: Serving,
    headers: HeaderMap,
    request: MessagesRequest,
    request_id: String,
) -> Result<Response, ApiError> {
    validate_anthropic_version(&headers)?;
    let adapted = adapt(request)?;
    let response_model = headers
        .get("Magnitude-Gateway-Model")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| adapted.model.clone());
    let invocation = state
        .models
        .invoke(&adapted.model, None)
        .await
        .map_err(ApiError::from)?;
    let id = state.next_id("msg_icn_");
    let mut stream = invocation.generate(adapted.request);
    if !adapted.stream {
        let (output, completion) = collect(stream).await.map_err(ApiError::from)?;
        return Ok(Json(message(&id, &response_model, &output, &completion)).into_response());
    }
    let (input_tokens, _) = admitted(&mut stream).await.map_err(ApiError::from)?;
    let (sender, receiver) = mpsc::channel::<Result<Event, Infallible>>(32);
    let projector = StreamProjector {
        id,
        model: response_model,
        request_id,
        sender,
        next_index: 0,
        reasoning_index: None,
        text_index: None,
        tools: BTreeMap::new(),
    };
    tokio::spawn(async move {
        if projector.start(input_tokens).await.is_ok() {
            project(stream, projector).await;
        }
    });
    Ok(Sse::new(ReceiverStream::new(receiver))
        .keep_alive(KeepAlive::default())
        .into_response())
}
