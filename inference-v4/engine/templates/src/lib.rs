//! Owned template preparation and semantic parsing, independent of inference.
//! Native code is statically linked; no runtime source tree or Python is required.
mod native;

pub use native::{build_info, OutputStream, PreparedRequest, Template};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidArgument,
    InvalidHandle,
    OutOfMemory,
    Native,
    Incompatible,
}
impl Error {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidArgument,
            message: message.into(),
        }
    }
    pub(crate) fn incompatible(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Incompatible,
            message: message.into(),
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(f)
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BuildInfo {
    pub version: u32,
    pub abi: u32,
    pub extraction: u32,
    pub upstream: String,
    pub build: String,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
}

/// Native request data. Artifact selection, named-tool normalization, tokenization,
/// and validation of host options belong to the engine's chat preparation layer.
/// Omitted template arguments remain omitted, including reasoning controls.
#[derive(Clone, Debug, Serialize)]
pub struct Request {
    pub messages: Vec<Value>,
    pub now: i64,
    pub tools: Vec<Value>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub template_arguments: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<Value>,
}
impl Request {
    pub fn new(messages: Vec<Value>, now: i64) -> Self {
        Self {
            messages,
            now,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: true,
            template_arguments: Map::new(),
            json_schema: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GrammarTrigger {
    pub r#type: i32,
    pub value: String,
    pub token: i32,
}

/// Immutable description of the same native plan used to create output streams.
/// `grammar` is GBNF; it must be converted and bound to the exact tokenizer before
/// constrained generation. Only `grammar_initial_prefix` initializes the matcher.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreparedDescription {
    pub version: u32,
    pub prompt: String,
    pub generation_prefix: String,
    pub parser: String,
    pub format: String,
    pub grammar: String,
    pub grammar_dialect: String,
    pub grammar_initial_prefix: String,
    pub grammar_lazy: bool,
    pub grammar_triggers: Vec<GrammarTrigger>,
    pub preserved_tokens: Vec<String>,
    pub additional_stops: Vec<String>,
    pub supports_thinking: bool,
    pub thinking_start: String,
    pub thinking_ends: Vec<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum TerminalCause {
    Natural = 0,
    Length = 1,
    UserStop = 2,
    Cancelled = 3,
    Failed = 4,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Content {
        text: String,
    },
    Reasoning {
        text: String,
    },
    ToolStart {
        index: u32,
        name: String,
        id: String,
    },
    ToolArguments {
        index: u32,
        text: String,
    },
    ToolComplete {
        index: u32,
    },
    Finish {
        cause: TerminalCause,
    },
}

pub type SpecialTokens = BTreeMap<String, String>;
