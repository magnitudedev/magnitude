//! Protocol-neutral chat requests. Chat Completions, Responses and Anthropic
//! Messages all adapt into these values; rendering them is the one path by
//! which a conversation becomes template input, so counting, template
//! application and generation see identical prompts.
use super::{ChatError, ChatRequest, ReasoningIntent, ToolChoice};
use serde_json::{json, Map, Value};
use std::{collections::BTreeSet, num::NonZeroU32, sync::Arc};

/// Validated image bytes. Source policy (data URLs, size limits) is applied by
/// the protocol layer before a request reaches the engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageInput {
    pub media_type: String,
    pub bytes: Arc<[u8]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UserPart {
    Text(String),
    Image(ImageInput),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolResultPart {
    Text(String),
    Image(ImageInput),
}

/// An assistant tool call and the result the caller supplied for it.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolExchange {
    pub call: ToolCall,
    pub result: Vec<ToolResultPart>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssistantTurn {
    pub reasoning: Option<String>,
    pub text: Option<String>,
    pub tool_calls: Vec<ToolExchange>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    User(Vec<UserPart>),
    Assistant(AssistantTurn),
}

/// One leading system prompt and a nonempty ordered history.
#[derive(Clone, Debug, PartialEq)]
pub struct Conversation {
    system: Option<String>,
    entries: Vec<Entry>,
}

impl Conversation {
    pub fn new(system: Option<String>, entries: Vec<Entry>) -> Result<Self, ChatError> {
        if entries.is_empty() {
            return Err(ChatError::InvalidRequest(
                "conversation entries must not be empty".into(),
            ));
        }
        if system.as_deref() == Some("") {
            return Err(ChatError::InvalidRequest(
                "system prompt must not be empty".into(),
            ));
        }
        Ok(Self { system, entries })
    }

    pub fn system(&self) -> Option<&str> {
        self.system.as_deref()
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Map<String, Value>,
}

/// Offered tools and how the model may use them. Names are unique and every
/// selection names an offered tool.
#[derive(Clone, Debug, PartialEq)]
pub struct Tools {
    definitions: Vec<ToolDefinition>,
    choice: ToolChoice,
    parallel: bool,
}

impl Tools {
    pub fn new(
        definitions: Vec<ToolDefinition>,
        choice: ToolChoice,
        parallel: bool,
    ) -> Result<Self, ChatError> {
        let invalid = |message: String| Err(ChatError::InvalidRequest(message));
        let mut names = BTreeSet::new();
        for definition in &definitions {
            if definition.name.is_empty() {
                return invalid("tool function name must not be empty".into());
            }
            if !names.insert(definition.name.as_str()) {
                return invalid(format!("duplicate tool name: {}", definition.name));
            }
        }
        match &choice {
            ToolChoice::Required if definitions.is_empty() => {
                return invalid("required tool choice needs at least one tool definition".into());
            }
            ToolChoice::Named(name) if !names.contains(name.as_str()) => {
                return invalid(format!("tool choice references unknown tool: {name}"));
            }
            ToolChoice::Allowed { names: allowed, .. } => {
                if allowed.is_empty() {
                    return invalid("allowed tool choice must contain at least one tool".into());
                }
                let mut seen = BTreeSet::new();
                for name in allowed {
                    if !seen.insert(name.as_str()) {
                        return invalid(format!("duplicate allowed tool name: {name}"));
                    }
                    if !names.contains(name.as_str()) {
                        return invalid(format!("tool choice references unknown tool: {name}"));
                    }
                }
            }
            _ => {}
        }
        Ok(Self {
            definitions,
            choice,
            parallel,
        })
    }

    pub fn none() -> Self {
        Self {
            definitions: Vec::new(),
            choice: ToolChoice::None,
            parallel: false,
        }
    }

    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    pub fn choice(&self) -> &ToolChoice {
        &self.choice
    }

    pub fn parallel(&self) -> bool {
        self.parallel
    }

    /// Whether the model may call a tool for this request.
    pub fn enabled(&self) -> bool {
        !self.definitions.is_empty() && self.choice != ToolChoice::None
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum OutputFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Map<String, Value>,
        strict: bool,
    },
    /// A caller-supplied GBNF grammar.
    Grammar(String),
}

/// Everything that determines the rendered prompt. Counting, template
/// application and generation take exactly this value.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatInput {
    pub conversation: Conversation,
    pub tools: Tools,
    pub reasoning: ReasoningIntent,
    pub output: OutputFormat,
    /// Caller template arguments (`chat_template_kwargs`).
    pub template_arguments: Map<String, Value>,
}

/// Rendered template input and the images its media markers stand for, in
/// prompt order.
pub struct RenderedInput {
    pub request: ChatRequest,
    pub images: Vec<ImageInput>,
}

impl ChatInput {
    /// Render into the template-level request. Images become the model's media
    /// marker; a model without one rejects image input.
    pub fn render(&self, now: i64, media_marker: Option<&str>) -> Result<RenderedInput, ChatError> {
        let mut images = Vec::new();
        let mut media = |image: &ImageInput| -> Result<Value, ChatError> {
            let marker = media_marker.ok_or_else(|| {
                ChatError::InvalidRequest("the loaded model takes no image input".into())
            })?;
            images.push(image.clone());
            Ok(json!({"type": "media_marker", "text": marker}))
        };
        let mut messages = Vec::new();
        if let Some(system) = self.conversation.system() {
            messages.push(json!({"role": "system", "content": system}));
        }
        for entry in self.conversation.entries() {
            match entry {
                Entry::User(parts) => {
                    let parts = parts
                        .iter()
                        .map(|part| match part {
                            UserPart::Text(text) => Ok(json!({"type": "text", "text": text})),
                            UserPart::Image(image) => media(image),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    messages.push(json!({"role": "user", "content": content(parts)}));
                }
                Entry::Assistant(turn) => {
                    let mut message = json!({"role": "assistant", "content": turn.text});
                    if let Some(reasoning) = &turn.reasoning {
                        message["reasoning_content"] = json!(reasoning);
                    }
                    if !turn.tool_calls.is_empty() {
                        message["tool_calls"] = turn
                            .tool_calls
                            .iter()
                            .map(|exchange| {
                                json!({"id": exchange.call.id, "type": "function", "function": {
                                    "name": exchange.call.name,
                                    "arguments": exchange.call.arguments,
                                }})
                            })
                            .collect();
                    }
                    messages.push(message);
                    for exchange in &turn.tool_calls {
                        let parts = exchange
                            .result
                            .iter()
                            .map(|part| match part {
                                ToolResultPart::Text(text) => {
                                    Ok(json!({"type": "text", "text": text}))
                                }
                                ToolResultPart::Image(image) => media(image),
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": exchange.call.id,
                            "content": content(parts),
                        }));
                    }
                }
            }
        }
        let mut request = ChatRequest::new(messages, now);
        request.tools = self
            .tools
            .definitions()
            .iter()
            .map(|tool| {
                let mut function = json!({"name": tool.name, "parameters": tool.parameters});
                if let Some(description) = &tool.description {
                    function["description"] = json!(description);
                }
                json!({"type": "function", "function": function})
            })
            .collect();
        request.tool_choice = self.tools.choice().clone();
        request.parallel_tool_calls = self.tools.parallel();
        request.template_arguments = self.template_arguments.clone();
        request.reasoning = self.reasoning.clone();
        match &self.output {
            OutputFormat::Text => {}
            // JSON mode promises a JSON object. The template treats an empty
            // schema as no output format, so the object shape is explicit.
            OutputFormat::JsonObject => request.json_schema = Some(json!({"type": "object"})),
            OutputFormat::JsonSchema { schema, .. } => {
                request.json_schema = Some(Value::Object(schema.clone()))
            }
            OutputFormat::Grammar(grammar) => request.grammar = Some(grammar.clone()),
        }
        Ok(RenderedInput { request, images })
    }
}

/// Text-only content renders as a string; mixed or media content as parts.
fn content(parts: Vec<Value>) -> Value {
    match parts.as_slice() {
        [] => json!(""),
        [single] if single["type"] == "text" => single["text"].clone(),
        _ => Value::Array(parts),
    }
}

/// Numerical sampling controls; the engine validates their ranges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingControls {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: u64,
}

/// Whether the request's prompt may reuse and publish retained prefix state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptCache {
    Allowed,
    Disabled,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationControls {
    /// `None` generates until the served context is exhausted.
    pub max_output_tokens: Option<NonZeroU32>,
    pub sampling: SamplingControls,
    pub stops: Vec<String>,
    pub end_of_generation: magnitude_generation::EndOfGeneration,
    /// Hard cap on reasoning tokens, enforced by forcing reasoning closure.
    pub reasoning_budget: Option<NonZeroU32>,
    pub prompt_cache: PromptCache,
}

/// A generation request: the rendered input plus how to generate from it.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationRequest {
    pub input: ChatInput,
    pub controls: GenerationControls,
}
