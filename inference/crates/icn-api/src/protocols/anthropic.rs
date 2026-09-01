use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use utoipa::ToSchema;

use super::super::{
    ApiError, AppState, ImageInput, admit_invocation, domain, domain_error, execute_with_journal,
    non_empty_text, non_empty_vec,
};
use super::chat::ReasoningEffortRequest;

const LOCAL_THINKING_SIGNATURE: &str = "magnitude-local-v1";

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct MessagesRequest {
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
    pub tool_choice: Option<ToolChoice>,
    #[schema(nullable = false)]
    pub thinking: Option<Thinking>,
    pub metadata: Option<Value>,
    pub output_config: Option<OutputConfig>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct OutputConfig {
    #[schema(nullable = false)]
    pub effort: Option<ReasoningEffortRequest>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum SystemBlock {
    Text {
        text: String,
        #[serde(default)]
        #[serde(rename = "cache_control")]
        _cache_control: Option<Value>,
    },
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct Message {
    pub role: Role,
    pub content: Content,
}

#[derive(Debug, Clone, Copy, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum Content {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ContentBlock {
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
        is_error: bool,
        #[serde(default)]
        #[serde(rename = "cache_control")]
        _cache_control: Option<Value>,
    },
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ImageSource {
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
pub(crate) enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ToolResultBlock {
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
pub(crate) struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Map<String, Value>,
    #[serde(default)]
    #[serde(rename = "cache_control")]
    pub _cache_control: Option<Value>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ToolChoice {
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
pub(crate) enum Thinking {
    Enabled { budget_tokens: u32 },
    Adaptive,
    Disabled,
}

pub(crate) struct AdaptedRequest {
    pub(crate) invocation: domain::InferenceInvocation,
    pub(crate) stream: bool,
}

pub(crate) fn adapt(request: MessagesRequest) -> Result<AdaptedRequest, ApiError> {
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
    let context = context(request.system, request.messages)?;
    let definitions = request
        .tools
        .into_iter()
        .map(|tool| {
            Ok(domain::ToolDefinition::new(
                domain::ToolName::try_new(tool.name).map_err(domain_error)?,
                tool.description,
                domain::JsonObject::new(tool.input_schema),
            ))
        })
        .collect::<Result<Vec<_>, ApiError>>()?;
    let (choice, parallelism) = match request.tool_choice {
        None
        | Some(ToolChoice::Auto {
            disable_parallel_tool_use: false,
        }) => (domain::ToolChoice::Auto, domain::ToolParallelism::Parallel),
        Some(ToolChoice::Auto {
            disable_parallel_tool_use: true,
        }) => (
            domain::ToolChoice::Auto,
            domain::ToolParallelism::Sequential,
        ),
        Some(ToolChoice::Any {
            disable_parallel_tool_use,
        }) => (
            domain::ToolChoice::Required,
            parallelism(disable_parallel_tool_use),
        ),
        Some(ToolChoice::Tool {
            name,
            disable_parallel_tool_use,
        }) => (
            domain::ToolChoice::Specific {
                name: domain::ToolName::try_new(name).map_err(domain_error)?,
            },
            parallelism(disable_parallel_tool_use),
        ),
        Some(ToolChoice::None) => (
            domain::ToolChoice::Disabled,
            domain::ToolParallelism::Sequential,
        ),
    };
    let tools = domain::ToolConfiguration::try_new(definitions, choice, parallelism)
        .map_err(domain_error)?;
    let effort = request
        .output_config
        .and_then(|config| config.effort)
        .map(|effort| effort.normalize())
        .transpose()?;
    if effort
        .as_ref()
        .is_some_and(|effort| effort.as_str() == "none")
    {
        return Err(ApiError::invalid(
            "output_config.effort must select an enabled reasoning behavior",
        ));
    }
    let budget = match request.thinking {
        Some(Thinking::Enabled { budget_tokens }) => Some(
            NonZeroU32::new(budget_tokens)
                .ok_or_else(|| ApiError::invalid("thinking budget_tokens must be positive"))?,
        ),
        Some(Thinking::Adaptive) | Some(Thinking::Disabled) | None => None,
    };
    let reasoning = match (request.thinking, effort) {
        (Some(Thinking::Disabled), Some(_)) => {
            return Err(ApiError::invalid(
                "output_config.effort cannot be used when thinking is disabled",
            ));
        }
        (Some(Thinking::Disabled), None) => domain::ReasoningIntent::Disabled {
            template_args: BTreeMap::new(),
        },
        (_, Some(effort)) => domain::ReasoningIntent::Effort {
            effort,
            template_args: BTreeMap::new(),
            budget,
        },
        (Some(Thinking::Enabled { .. }), None) => domain::ReasoningIntent::Enabled {
            template_args: BTreeMap::new(),
            budget,
        },
        (Some(Thinking::Adaptive) | None, None) => domain::ReasoningIntent::ModelDefault {
            template_args: BTreeMap::new(),
            budget: None,
        },
    };
    let stops = request
        .stop_sequences
        .into_iter()
        .map(|value| domain::StopSequence::try_new(value).map_err(domain_error))
        .collect::<Result<Vec<_>, _>>()?;
    let temperature = request.temperature.unwrap_or(1.0);
    let top_p = request.top_p.unwrap_or(1.0);
    let generation = domain::GenerationParameters::new(
        Some(max_tokens),
        domain::SamplingParameters::new(
            domain::Temperature::try_new(temperature).map_err(domain_error)?,
            domain::TopP::try_new(top_p).map_err(domain_error)?,
            0,
        ),
        stops,
        domain::EndOfGenerationPolicy::StopAtModelEnd,
    );
    let model = request.model;
    Ok(AdaptedRequest {
        invocation: domain::InferenceInvocation::new(
            domain::InferenceModelSelector::try_new(model).map_err(domain_error)?,
            domain::InferenceRequest::new(
                context,
                tools,
                reasoning,
                domain::OutputConstraint::Text,
                generation,
                domain::PromptReusePolicy::Allowed,
            ),
        ),
        stream: request.stream,
    })
}

fn parallelism(disabled: bool) -> domain::ToolParallelism {
    if disabled {
        domain::ToolParallelism::Sequential
    } else {
        domain::ToolParallelism::Parallel
    }
}

// Claude Code attribution projection. Claude Code prepends provider-reserved
// billing metadata as the leading system text —
// `x-anthropic-billing-header: …; cch=<stamp>;<optional real prompt>` — which
// api.anthropic.com strips before the model sees it. This adapter is the one
// place wire bytes become model-visible text, so it owns the same stripping:
// the per-request cch stamp would otherwise defeat prompt-prefix reuse and
// the metadata would become model-visible prompt content. Recognition is
// strict and positional — leading block, exact sentinel, `cch=` plus five
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

fn context(
    system: Option<SystemPrompt>,
    messages: Vec<Message>,
) -> Result<domain::InferenceContext, ApiError> {
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
                Some(non_empty_text(blocks.join("\n"), "system")?)
            }
        }
    };
    let mut entries = Vec::new();
    let mut messages = messages.into_iter().peekable();
    while let Some(message) = messages.next() {
        match message.role {
            Role::System => entries.push(domain::ContextEntry::User {
                entry: domain::UserEntry::new(system_role_content(message.content)?),
            }),
            Role::User => entries.push(domain::ContextEntry::User {
                entry: domain::UserEntry::new(user_content(message.content)?),
            }),
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
                        ApiError::invalid("assistant tool_use blocks require a following user tool_result message")
                    })?;
                    if !matches!(next.role, Role::User) {
                        return Err(ApiError::invalid(
                            "assistant tool_use blocks must be followed by user tool_result blocks",
                        ));
                    }
                    let (mut results, trailing_user_content) = tool_results(next.content)?;
                    let mut exchanges = Vec::with_capacity(calls.len());
                    for call in calls {
                        let result = results.remove(call.id().as_str()).ok_or_else(|| {
                            ApiError::invalid(format!(
                                "tool_use {} has no matching tool_result",
                                call.id().as_str()
                            ))
                        })?;
                        exchanges.push(domain::ToolExchange::new(call, result));
                    }
                    if let Some(tool_use_id) = results.keys().next() {
                        return Err(ApiError::invalid(format!(
                            "tool_result {tool_use_id} has no matching tool_use",
                        )));
                    }
                    (exchanges, trailing_user_content)
                };
                entries.push(domain::ContextEntry::Assistant {
                    entry: domain::AssistantEntry::new(reasoning, text, exchanges),
                });
                if !trailing_user_content.is_empty() {
                    entries.push(domain::ContextEntry::User {
                        entry: domain::UserEntry::new(trailing_user_content),
                    });
                }
            }
        }
    }
    Ok(domain::InferenceContext::new(
        system,
        non_empty_vec(entries, "messages")?,
    ))
}

// System-role messages are the Anthropic protocol's mid-conversation operator
// channel; Claude Code uses it to surface text the user typed mid-turn. Local
// chat templates have no mid-sequence system turn and canonical context
// carries exactly one leading system prompt, so these become user entries at
// their original position — never part of the leading system prompt. See
// design/inference/http-protocol-compatibility.md.
fn system_role_content(content: Content) -> Result<Vec<domain::UserContent>, ApiError> {
    let mut values = Vec::new();
    for block in blocks(content) {
        match block {
            ContentBlock::Text { text, .. } => values.push(domain::UserContent::Text {
                text: non_empty_text(text, "system message text")?,
            }),
            _ => {
                return Err(ApiError::invalid(
                    "system message content must contain only text",
                ));
            }
        }
    }
    non_empty_vec(values, "system message content").map(domain::NonEmptyVec::into_vec)
}

fn user_content(content: Content) -> Result<Vec<domain::UserContent>, ApiError> {
    let blocks = blocks(content);
    let mut values = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text, .. } => values.push(domain::UserContent::Text {
                text: non_empty_text(text, "user text")?,
            }),
            ContentBlock::Image { source, .. } => values.push(domain::UserContent::Image {
                image: image(source)?,
            }),
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
    non_empty_vec(values, "user content").map(domain::NonEmptyVec::into_vec)
}

fn assistant_content(
    content: Content,
) -> Result<
    (
        Option<domain::NonEmptyText>,
        Option<domain::NonEmptyText>,
        Vec<domain::ToolCall>,
    ),
    ApiError,
> {
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
                if !ids.insert(id.clone()) {
                    return Err(ApiError::invalid(format!("duplicate tool_use id: {id}")));
                }
                calls.push(domain::ToolCall::new(
                    domain::ToolCallId::try_new(id).map_err(domain_error)?,
                    domain::ToolName::try_new(name).map_err(domain_error)?,
                    domain::JsonObject::new(input),
                ));
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

fn tool_results(
    content: Content,
) -> Result<
    (
        BTreeMap<String, domain::ToolResult>,
        Vec<domain::UserContent>,
    ),
    ApiError,
> {
    let mut results = BTreeMap::new();
    let mut trailing_content = Vec::new();
    for block in blocks(content) {
        let ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            ..
        } = block
        else {
            match block {
                ContentBlock::Text { text, .. } => {
                    trailing_content.push(domain::UserContent::Text {
                        text: non_empty_text(text, "user text after tool results")?,
                    });
                }
                ContentBlock::Image { source, .. } => {
                    trailing_content.push(domain::UserContent::Image {
                        image: image(source)?,
                    });
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
            Some(ToolResultContent::Text(text)) => vec![domain::ToolResultContent::Text {
                text: non_empty_text(text, "tool result")?,
            }],
            Some(ToolResultContent::Blocks(blocks)) => blocks
                .into_iter()
                .map(|block| match block {
                    ToolResultBlock::Text { text, .. } => Ok(domain::ToolResultContent::Text {
                        text: non_empty_text(text, "tool result text")?,
                    }),
                    ToolResultBlock::Image { source, .. } => Ok(domain::ToolResultContent::Image {
                        image: image(source)?,
                    }),
                })
                .collect::<Result<Vec<_>, ApiError>>()?,
        };
        let result = domain::ToolResult::new(
            if is_error {
                domain::ToolOutcome::Error
            } else {
                domain::ToolOutcome::Success
            },
            values,
        );
        if results.insert(tool_use_id.clone(), result).is_some() {
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

fn joined(
    values: Vec<String>,
    field: &'static str,
) -> Result<Option<domain::NonEmptyText>, ApiError> {
    if values.is_empty() {
        Ok(None)
    } else {
        non_empty_text(values.join(""), field).map(Some)
    }
}

fn image(source: ImageSource) -> Result<ImageInput, ApiError> {
    match source {
        ImageSource::Base64 { media_type, data } => {
            let value = format!("data:{media_type};base64,{data}");
            let decoded =
                crate::media::decode_image_data_url(&value, crate::media::MAX_HTTP_IMAGE_BYTES)
                    .map_err(|error| ApiError::invalid(error.to_string()))?;
            Ok(ImageInput::new(decoded.media_type, decoded.bytes))
        }
        ImageSource::Url { .. } => Err(ApiError::invalid(
            "network image URLs are not supported; use a base64 image source",
        )),
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ErrorEnvelope {
    pub(crate) r#type: &'static str,
    pub(crate) error: ErrorBody,
    pub(crate) request_id: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ErrorBody {
    pub(crate) r#type: &'static str,
    pub(crate) message: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct MessageResponse {
    pub(crate) id: String,
    pub(crate) r#type: &'static str,
    pub(crate) role: &'static str,
    pub(crate) model: String,
    pub(crate) content: Vec<ResponseContentBlock>,
    pub(crate) stop_reason: &'static str,
    pub(crate) stop_sequence: Option<String>,
    pub(crate) usage: UsageResponse,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponseContentBlock {
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
pub(crate) struct UsageResponse {
    pub(crate) input_tokens: u64,
    pub(crate) cache_creation_input_tokens: u64,
    pub(crate) cache_read_input_tokens: u64,
    pub(crate) output_tokens: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CountTokensResponse {
    pub(crate) input_tokens: u64,
}

pub(crate) fn message(id: &str, model: &str, result: &domain::InferenceResult) -> MessageResponse {
    let mut content = Vec::new();
    if let Some(reasoning) = result.output().reasoning() {
        content.push(ResponseContentBlock::Thinking {
            thinking: reasoning.as_str().to_owned(),
            signature: LOCAL_THINKING_SIGNATURE,
        });
    }
    if let Some(text) = result.output().text() {
        content.push(ResponseContentBlock::Text {
            text: text.as_str().to_owned(),
        });
    }
    if !result.output().tool_calls().is_empty() {
        content.extend(result.output().tool_calls().iter().map(|call| {
            ResponseContentBlock::ToolUse {
                id: call.id().as_str().to_owned(),
                name: call.name().as_str().to_owned(),
                input: call.input().as_map().clone(),
            }
        }));
    }
    let (stop_reason, stop_sequence) = stop(result.termination());
    MessageResponse {
        id: id.to_owned(),
        r#type: "message",
        role: "assistant",
        model: model.to_owned(),
        content,
        stop_reason,
        stop_sequence: stop_sequence.map(str::to_owned),
        usage: usage(result),
    }
}

fn usage(result: &domain::InferenceResult) -> UsageResponse {
    UsageResponse {
        input_tokens: result.usage().input_tokens(),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: result.usage().cached_input_tokens(),
        output_tokens: result.usage().output_tokens(),
    }
}

fn stop(termination: &domain::Termination) -> (&'static str, Option<&str>) {
    match termination {
        domain::Termination::Natural => ("end_turn", None),
        domain::Termination::StopSequence { sequence } => {
            ("stop_sequence", Some(sequence.as_str()))
        }
        domain::Termination::OutputLimit => ("max_tokens", None),
        domain::Termination::ToolCalls => ("tool_use", None),
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
    Thinking {
        thinking: String,
        signature: String,
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

pub(crate) struct StreamProjector {
    id: String,
    model: String,
    request_id: String,
    sender: mpsc::Sender<Result<Event, Infallible>>,
    input_tokens: u64,
    next_index: usize,
    reasoning_index: Option<usize>,
    text_index: Option<usize>,
    tools: BTreeMap<usize, usize>,
}

impl StreamProjector {
    pub(crate) fn new(
        id: String,
        model: String,
        request_id: String,
        sender: mpsc::Sender<Result<Event, Infallible>>,
        input_tokens: u64,
    ) -> Self {
        Self {
            id,
            model,
            request_id,
            sender,
            input_tokens,
            next_index: 0,
            reasoning_index: None,
            text_index: None,
            tools: BTreeMap::new(),
        }
    }

    fn send(&self, event: &'static str, value: &StreamEvent) -> bool {
        serde_json::to_string(value)
            .ok()
            .and_then(|data| {
                self.sender
                    .blocking_send(Ok(Event::default().event(event).data(data)))
                    .ok()
            })
            .is_some()
    }

    pub(crate) fn start(&self) -> bool {
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
                        input_tokens: self.input_tokens,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                        output_tokens: 0,
                    },
                },
            },
        )
    }

    fn begin_block(&mut self, block: StreamContentBlock) -> Option<usize> {
        let index = self.next_index;
        self.next_index += 1;
        self.send(
            "content_block_start",
            &StreamEvent::ContentBlockStart {
                index,
                content_block: block,
            },
        )
        .then_some(index)
    }

    pub(crate) fn observe(
        &mut self,
        observation: &domain::InferenceObservation,
    ) -> Result<(), icn_contracts::InferenceError> {
        use domain::InferenceOutputEvent as Output;
        let domain::InferenceObservationEvent::Output { event } = observation.event() else {
            return Ok(());
        };
        let sent = match event {
            Output::Started | Output::ToolCallFinished { .. } => true,
            Output::ReasoningDelta { text } => {
                let index = match self.reasoning_index {
                    Some(index) => index,
                    None => {
                        let index = self
                            .begin_block(StreamContentBlock::Thinking {
                                thinking: String::new(),
                                signature: String::new(),
                            })
                            .ok_or_else(disconnected)?;
                        self.reasoning_index = Some(index);
                        index
                    }
                };
                self.send(
                    "content_block_delta",
                    &StreamEvent::ContentBlockDelta {
                        index,
                        delta: StreamDelta::ThinkingDelta {
                            thinking: text.as_str().to_owned(),
                        },
                    },
                )
            }
            Output::TextDelta { text } => {
                let index = match self.text_index {
                    Some(index) => index,
                    None => {
                        let index = self
                            .begin_block(StreamContentBlock::Text {
                                text: String::new(),
                            })
                            .ok_or_else(disconnected)?;
                        self.text_index = Some(index);
                        index
                    }
                };
                self.send(
                    "content_block_delta",
                    &StreamEvent::ContentBlockDelta {
                        index,
                        delta: StreamDelta::TextDelta {
                            text: text.as_str().to_owned(),
                        },
                    },
                )
            }
            Output::ToolCallStarted { index, id, name } => {
                let output_index = self
                    .begin_block(StreamContentBlock::ToolUse {
                        id: id.as_str().to_owned(),
                        name: name.as_str().to_owned(),
                        input: Map::new(),
                    })
                    .ok_or_else(disconnected)?;
                self.tools.insert(*index, output_index);
                true
            }
            Output::ToolInputDelta {
                index,
                json_fragment,
            } => {
                let output_index = self.tools.get(index).ok_or_else(|| {
                    icn_contracts::InferenceError::Backend(
                        "tool input arrived before tool start".into(),
                    )
                })?;
                self.send(
                    "content_block_delta",
                    &StreamEvent::ContentBlockDelta {
                        index: *output_index,
                        delta: StreamDelta::InputJsonDelta {
                            partial_json: json_fragment.as_str().to_owned(),
                        },
                    },
                )
            }
        };
        sent.then_some(()).ok_or_else(disconnected)
    }

    pub(crate) fn finish(&self, result: &domain::InferenceResult) {
        if let Some(index) = self.reasoning_index
            && !self.send(
                "content_block_delta",
                &StreamEvent::ContentBlockDelta {
                    index,
                    delta: StreamDelta::SignatureDelta {
                        signature: LOCAL_THINKING_SIGNATURE.to_owned(),
                    },
                },
            )
        {
            return;
        }
        for index in 0..self.next_index {
            if !self.send(
                "content_block_stop",
                &StreamEvent::ContentBlockStop { index },
            ) {
                return;
            }
        }
        let (stop_reason, stop_sequence) = stop(result.termination());
        if !self.send(
            "message_delta",
            &StreamEvent::MessageDelta {
                delta: MessageDelta {
                    stop_reason,
                    stop_sequence: stop_sequence.map(str::to_owned),
                },
                usage: OutputUsage {
                    output_tokens: result.usage().output_tokens(),
                },
            },
        ) {
            return;
        }
        let _ = self.send("message_stop", &StreamEvent::MessageStop);
    }

    pub(crate) fn fail(&self, error: icn_contracts::InferenceError) {
        let _ = self.send(
            "error",
            &StreamEvent::Error {
                error: StreamError {
                    r#type: "api_error",
                    message: error.to_string(),
                },
                request_id: self.request_id.clone(),
            },
        );
    }
}

fn disconnected() -> icn_contracts::InferenceError {
    icn_contracts::InferenceError::Callback("stream consumer disconnected".into())
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
        (status = 404, description = "Requested model is unavailable", body = ErrorEnvelope),
        (status = 500, description = "Token counting failed", body = ErrorEnvelope)
    )
)]
pub(crate) async fn anthropic_count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let request_id = format!("req_icn_{}", state.next_id.fetch_add(1, Ordering::Relaxed));
    let result = async {
        validate_anthropic_version(&headers)?;
        let Json(request) = payload.map_err(|error| ApiError::invalid(error.body_text()))?;
        let adapted = adapt(request)?;
        let controller = state
            .model_controller
            .as_ref()
            .ok_or_else(|| ApiError::server("model control is not configured"))?;
        let admitted = admit_invocation(controller, adapted.invocation, None).await?;
        let count = tokio::task::spawn_blocking(move || {
            admitted.lease.backend().count_tokens(admitted.request)
        })
        .await
        .map_err(|error| ApiError::server(format!("token-count task failed: {error}")))?
        .map_err(ApiError::from_inference)?;
        Ok::<_, ApiError>(
            Json(CountTokensResponse {
                input_tokens: count,
            })
            .into_response(),
        )
    }
    .await;
    match result {
        Ok(mut response) => {
            if let Ok(value) = request_id.parse() {
                response.headers_mut().insert("request-id", value);
            }
            response
        }
        Err(error) => anthropic_error_response(request_id, error),
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
        (status = 500, description = "Inference failed", body = ErrorEnvelope)
    )
)]
pub(crate) async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let request_id = format!("req_icn_{}", state.next_id.fetch_add(1, Ordering::Relaxed));
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return anthropic_error_response(request_id, ApiError::invalid(error.body_text()));
        }
    };
    match anthropic_messages_inner(state, headers, request, request_id.clone()).await {
        Ok(mut response) => {
            if let Ok(value) = request_id.parse() {
                response.headers_mut().insert("request-id", value);
            }
            response
        }
        Err(error) => anthropic_error_response(request_id, error),
    }
}

async fn anthropic_messages_inner(
    state: AppState,
    headers: HeaderMap,
    request: MessagesRequest,
    request_id: String,
) -> Result<Response, ApiError> {
    validate_anthropic_version(&headers)?;
    let adapted = adapt(request)?;
    let requested_model = adapted.invocation.model().as_str().to_owned();
    let response_model = headers
        .get("Magnitude-Gateway-Model")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| requested_model.clone());
    let controller = state
        .model_controller
        .as_ref()
        .ok_or_else(|| ApiError::server("model control is not configured"))?;
    let stream = adapted.stream;
    let admitted = admit_invocation(controller, adapted.invocation, None).await?;
    let lease = admitted.lease;
    let request = admitted.request;
    let id = format!("msg_icn_{}", state.next_id.fetch_add(1, Ordering::Relaxed));
    if !stream {
        let span = tracing::Span::current();
        let result = tokio::task::spawn_blocking(move || {
            span.in_scope(|| execute_with_journal(lease.backend().as_ref(), request, |_| Ok(())))
        })
        .await
        .map_err(|error| ApiError::server(format!("inference task failed: {error}")))?
        .map_err(ApiError::from_inference)?;
        return Ok(Json(message(&id, &response_model, &result)).into_response());
    }

    let (sender, receiver) = mpsc::channel::<Result<Event, Infallible>>(32);
    let count_backend = Arc::clone(lease.backend());
    let count_request = request.clone();
    let input_tokens =
        tokio::task::spawn_blocking(move || count_backend.count_tokens(count_request))
            .await
            .map_err(|error| ApiError::server(format!("token-count task failed: {error}")))?
            .map_err(ApiError::from_inference)?;
    let span = tracing::Span::current();
    tokio::task::spawn_blocking(move || {
        span.in_scope(|| {
            let mut projector =
                StreamProjector::new(id, response_model, request_id, sender, input_tokens);
            if !projector.start() {
                return;
            }
            let result = execute_with_journal(lease.backend().as_ref(), request, |observation| {
                projector.observe(observation)
            });
            match result {
                Ok(result) => projector.finish(&result),
                Err(error) => projector.fail(error),
            }
        });
    });
    Ok(Sse::new(ReceiverStream::new(receiver))
        .keep_alive(KeepAlive::default())
        .into_response())
}

fn anthropic_error_response(request_id: String, error: ApiError) -> Response {
    let error_type = match error.status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        _ => "api_error",
    };
    let mut response = (
        error.status,
        Json(ErrorEnvelope {
            r#type: "error",
            error: ErrorBody {
                r#type: error_type,
                message: error.body.error.message,
            },
            request_id: request_id.clone(),
        }),
    )
        .into_response();
    if let Ok(value) = request_id.parse() {
        response.headers_mut().insert("request-id", value);
    }
    response
}
