//! Strict chat wire validation. Distribution policy is validated before native
//! preparation; model identities and context bounds remain host-owned.
use super::{ChatRequest, PreparedChat, TemplateBundle, TemplateSelection, ToolChoice};
use crate::{
    generation::{Options, Sampling},
    inputs::ByteBpeTokenizer,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashSet;

fn present<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}
fn one() -> f64 {
    1.0
}
fn yes() -> bool {
    true
}
fn single() -> u32 {
    1
}
#[derive(Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum FunctionKind {
    #[default]
    Function,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Function {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    parameters: Option<Map<String, Value>>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    strict: Option<bool>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Tool {
    #[serde(rename = "type", default)]
    kind: FunctionKind,
    function: Function,
}
#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum Arguments {
    Text(String),
    Object(Map<String, Value>),
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Invocation {
    name: String,
    arguments: Arguments,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ToolCall {
    id: String,
    #[serde(rename = "type", default)]
    kind: FunctionKind,
    function: Invocation,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Detail {
    Auto,
}
fn auto_detail() -> Detail {
    Detail::Auto
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ImageUrl {
    url: String,
    #[serde(default = "auto_detail")]
    detail: Detail,
}
#[derive(Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum TextKind {
    #[default]
    Text,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ImageKind {
    ImageUrl,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TextPart {
    #[serde(rename = "type", default)]
    kind: TextKind,
    text: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ImagePart {
    #[serde(rename = "type")]
    kind: ImageKind,
    image_url: ImageUrl,
}
#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum Part {
    Text(TextPart),
    ImageUrl(ImagePart),
}
#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<Part>),
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    content: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}
#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    #[default]
    Auto,
    Required,
    None,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedFunction {
    name: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedChoice {
    #[serde(rename = "type", default)]
    _kind: FunctionKind,
    function: NamedFunction,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum Choice {
    Mode(Mode),
    Named(NamedChoice),
}
impl Default for Choice {
    fn default() -> Self {
        Self::Mode(Mode::Auto)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SchemaDefinition {
    name: String,
    #[serde(default, rename = "description")]
    _description: Option<String>,
    schema: Map<String, Value>,
    #[serde(default = "yes", rename = "strict")]
    _strict: bool,
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Format {
    Text {},
    JsonObject {},
    JsonSchema { json_schema: SchemaDefinition },
}
impl Default for Format {
    fn default() -> Self {
        Self::Text {}
    }
}
#[derive(Deserialize)]
#[serde(untagged)]
enum Stops {
    One(String),
    Many(Vec<String>),
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Body {
    model: String,
    messages: Vec<Message>,
    #[serde(default)]
    tools: Vec<Tool>,
    #[serde(default)]
    tool_choice: Choice,
    #[serde(default = "yes")]
    parallel_tool_calls: bool,
    #[serde(default)]
    chat_template_kwargs: Map<String, Value>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    response_format: Format,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default = "one")]
    temperature: f64,
    #[serde(default = "one")]
    top_p: f64,
    #[serde(default)]
    top_k: usize,
    #[serde(default)]
    min_p: f64,
    #[serde(default = "one")]
    repetition_penalty: f64,
    #[serde(default)]
    presence_penalty: f64,
    #[serde(default)]
    frequency_penalty: f64,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    stop: Option<Stops>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: StreamOptions,
    #[serde(default = "single")]
    n: u32,
}
#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Unsupported(String),
}
impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) | Self::Unsupported(message) => formatter.write_str(message),
        }
    }
}
impl std::error::Error for Error {}
impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}
impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::Invalid(message.into())
    }
}
impl From<Error> for String {
    fn from(error: Error) -> Self {
        error.to_string()
    }
}
/// Validated policy and typed messages; callers cannot bypass validation by
/// constructing a wire request directly.
pub struct Request {
    body: Body,
    stops: Vec<String>,
    output_limit: usize,
}
pub struct ModelLimits<'a> {
    pub model: &'a str,
    pub context_tokens: usize,
    pub vocabulary: usize,
    pub output_capacity: usize,
    pub forced_quantum: usize,
}
pub struct PreparedGeneration {
    pub chat: PreparedChat,
    pub options: Options,
}
impl Request {
    pub fn parse(bytes: &[u8], max_bytes: usize) -> Result<Self, Error> {
        if bytes.len() > max_bytes {
            return Err("chat request exceeds body byte limit".into());
        }
        let body: Body = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        if body.model.is_empty()
            || body.messages.is_empty()
            || body.messages.len() > 4096
            || body.tools.len() > 512
        {
            return Err("invalid model, message count, or tool count".into());
        }
        if body.n != 1
            || body.temperature < 0.0
            || body.top_p <= 0.0
            || body.top_p > 1.0
            || body.min_p < 0.0
            || body.min_p > 1.0
            || body.repetition_penalty <= 0.0
        {
            return Err("invalid generation option range".into());
        }
        if !matches!(body.temperature, 0.0 | 1.0)
            || body.top_p != 1.0
            || body.top_k != 0
            || body.min_p != 0.0
            || body.repetition_penalty != 1.0
            || body.presence_penalty != 0.0
            || body.frequency_penalty != 0.0
        {
            return Err(Error::Unsupported(
                "only temperature 0 or 1 with unmodified logits is supported".into(),
            ));
        }
        if body
            .max_tokens
            .zip(body.max_completion_tokens)
            .is_some_and(|(a, b)| a != b)
            || body
                .max_tokens
                .into_iter()
                .chain(body.max_completion_tokens)
                .any(|value| value > i32::MAX as u32)
        {
            return Err("completion token limits disagree or exceed the supported range".into());
        }
        if body.reasoning_effort.as_deref().is_some_and(|value| {
            !matches!(
                value,
                "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "adaptive"
            )
        }) {
            return Err("unknown reasoning effort".into());
        }
        let stops = match &body.stop {
            None => vec![],
            Some(Stops::One(stop)) => vec![stop.clone()],
            Some(Stops::Many(stops)) => stops.clone(),
        };
        if stops.len() > 4
            || stops
                .iter()
                .any(|stop| stop.is_empty() || stop.chars().count() > 1024)
        {
            return Err("provide at most four nonempty stop strings up to 1024 characters".into());
        }
        let mut names = HashSet::new();
        for tool in &body.tools {
            if tool.function.name.is_empty() || !names.insert(tool.function.name.as_str()) {
                return Err("tool function names must be nonempty and unique".into());
            }
        }
        match &body.tool_choice {
            Choice::Named(choice) if !names.contains(choice.function.name.as_str()) => {
                return Err(Error::Unsupported("named tool is unavailable".into()))
            }
            Choice::Mode(Mode::Required) if body.tools.is_empty() => {
                return Err(Error::Unsupported(
                    "required tool choice needs a tool".into(),
                ))
            }
            _ => {}
        }
        if !matches!(body.response_format, Format::Text { .. })
            && !body.tools.is_empty()
            && !matches!(body.tool_choice, Choice::Mode(Mode::None))
        {
            return Err(Error::Unsupported(
                "JSON response formats cannot be combined with offered tools".into(),
            ));
        }
        if let Format::JsonSchema { json_schema } = &body.response_format {
            if json_schema.name.is_empty() {
                return Err("schema name must be nonempty".into());
            }
        }
        for message in &body.messages {
            if message
                .tool_calls
                .iter()
                .any(|call| call.function.name.is_empty())
            {
                return Err("historical tool function names must be nonempty".into());
            }
            if let Some(Content::Parts(parts)) = &message.content {
                for part in parts {
                    if let Part::ImageUrl(ImagePart { image_url, .. }) = part {
                        if image_url.url.is_empty() || image_url.url.chars().count() > 24 << 20 {
                            return Err("image URL is empty or exceeds its limit".into());
                        }
                    }
                }
            }
        }
        let output_limit = body
            .max_completion_tokens
            .or(body.max_tokens)
            .unwrap_or(512) as usize;
        Ok(Self {
            body,
            stops,
            output_limit,
        })
    }
    pub fn model(&self) -> &str {
        &self.body.model
    }
    pub fn stream(&self) -> bool {
        self.body.stream
    }
    pub fn include_usage(&self) -> bool {
        self.body.stream_options.include_usage
    }
    pub fn stops(&self) -> &[String] {
        &self.stops
    }
    pub fn output_limit(&self) -> usize {
        self.output_limit
    }
    pub fn prepare(
        &self,
        bundle: &TemplateBundle,
        tokenizer: &ByteBpeTokenizer,
        selection: &TemplateSelection<'_>,
        now: i64,
        limits: &ModelLimits<'_>,
    ) -> Result<PreparedGeneration, String> {
        if self.model() != limits.model {
            return Err("requested model is not loaded".into());
        }
        if limits.context_tokens == 0
            || limits.context_tokens > i32::MAX as usize
            || limits.output_capacity == 0
            || limits.vocabulary < tokenizer.vocabulary()
        {
            return Err("invalid model generation limits".into());
        }
        if self.body.messages.iter().any(|message| matches!(&message.content,
            Some(Content::Parts(parts)) if parts.iter().any(|part| matches!(part, Part::ImageUrl(_))))) {
            return Err("image requests require the model media preparation path, which is not yet connected".into());
        }
        let mut request = ChatRequest::new(
            self.body
                .messages
                .iter()
                .map(serde_json::to_value)
                .collect::<Result<_, _>>()
                .map_err(|error| error.to_string())?,
            now,
        );
        request.tools = self
            .body
            .tools
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<_, _>>()
            .map_err(|error| error.to_string())?;
        request.tool_choice = match &self.body.tool_choice {
            Choice::Mode(Mode::Auto) => ToolChoice::Auto,
            Choice::Mode(Mode::None) => ToolChoice::None,
            Choice::Mode(Mode::Required) => ToolChoice::Required,
            Choice::Named(choice) => ToolChoice::Named(choice.function.name.clone()),
        };
        request.parallel_tool_calls = self.body.parallel_tool_calls;
        request.template_arguments = self.body.chat_template_kwargs.clone();
        request.reasoning_effort = self.body.reasoning_effort.clone();
        request.json_schema = match &self.body.response_format {
            Format::Text { .. } => None,
            Format::JsonObject { .. } => Some(serde_json::json!({"type":"object"})),
            Format::JsonSchema { json_schema } => Some(Value::Object(json_schema.schema.clone())),
        };
        let chat = PreparedChat::prepare(bundle, tokenizer, &request, selection)?;
        if chat.prompt_tokens() > limits.context_tokens {
            return Err("rendered prompt exceeds configured context".into());
        }
        let options = Options {
            max_tokens: self
                .output_limit
                .min(limits.context_tokens - chat.prompt_tokens() + 1),
            output_capacity: limits.output_capacity,
            context_limit: limits.context_tokens,
            vocabulary: limits.vocabulary,
            stop_tokens: tokenizer.stop_tokens().clone(),
            sampling: if self.body.temperature == 0.0 {
                Sampling::Greedy
            } else {
                Sampling::Categorical
            },
            seed: self.body.seed,
            forced_quantum: limits.forced_quantum,
        };
        Ok(PreparedGeneration { chat, options })
    }
}
