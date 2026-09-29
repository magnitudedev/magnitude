//! Host-only chat operations over a model's [`HostArtifacts`] (integration
//! spec §8.5): request preparation, token counting, template application and
//! model properties. None of them leases an instance or loads a model, and all
//! of them render and tokenize exactly as generation does, so counts agree
//! with generation.
use crate::error::RequestError;
use crate::host::HostArtifacts;
use magnitude_chat::{
    reasoning::ReasoningProfile, request::RenderedInput, ChatError, ChatInput, PreparedChat,
    TemplateSelection,
};
use magnitude_family_contracts::PreparedModelInput;
use std::{collections::BTreeMap, time::SystemTime};

/// Which context bound input preparation checks against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputBound {
    /// The served context: requests that will generate.
    Served,
    /// The artifact's declared capability: host-only sizing, which never
    /// enlarges the served bound.
    Declared,
}

/// A request rendered, tokenized and interpreted by the model's family.
pub struct PreparedInput {
    pub chat: PreparedChat,
    pub input: PreparedModelInput,
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs() as i64)
}

fn input_error(error: RequestError) -> ChatError {
    match error {
        RequestError::InvalidRequest { reason } => ChatError::InvalidRequest(reason),
        other => ChatError::Internal(other.to_string()),
    }
}

/// Render `input` for the model and prepare its numerical input.
pub fn prepare(
    host: &HostArtifacts,
    input: &ChatInput,
    bound: InputBound,
) -> Result<PreparedInput, ChatError> {
    let RenderedInput { request, images } = input.render(now(), host.media_placeholder())?;
    let chat = PreparedChat::prepare(
        host.templates(),
        host.tokenizer(),
        &request,
        &TemplateSelection::default(),
    )?;
    if let Some(constraint) = chat.constraint() {
        crate::telemetry::span_grammar(&constraint.report);
    }
    let tokens = chat.input().tokens.clone();
    let input = match bound {
        InputBound::Served => host.prepare_input(tokens, &images),
        InputBound::Declared => host.prepare_count_input(tokens, &images),
    }
    .map_err(input_error)?;
    Ok(PreparedInput { chat, input })
}

/// The model input tokens a request occupies, including expanded media.
pub fn count_tokens(host: &HostArtifacts, input: &ChatInput) -> Result<u64, ChatError> {
    let prepared = prepare(host, input, InputBound::Declared)?;
    Ok(prepared.input.tokens().len() as u64)
}

/// The prompt and output constraints a request prepares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppliedTemplate {
    pub prompt: String,
    pub generation_prompt: String,
    pub grammar: String,
    pub preserved_tokens: Vec<String>,
    pub additional_stops: Vec<String>,
    pub supports_thinking: bool,
    pub thinking_start_tag: Option<String>,
    pub thinking_end_tag: Option<String>,
    pub template_fingerprint: String,
}

pub fn apply_template(host: &HostArtifacts, input: &ChatInput) -> Result<AppliedTemplate, ChatError> {
    let prepared = prepare(host, input, InputBound::Declared)?;
    let description = prepared.chat.native().description();
    let fingerprint = host.template_inspection().fingerprint.clone();
    let nonempty = |value: &str| (!value.is_empty()).then(|| value.to_owned());
    Ok(AppliedTemplate {
        prompt: description.prompt.clone(),
        generation_prompt: description.generation_prefix.clone(),
        grammar: prepared
            .chat
            .constraint()
            .map_or_else(String::new, |source| source.gbnf.clone()),
        preserved_tokens: description.preserved_tokens.clone(),
        additional_stops: description.additional_stops.clone(),
        supports_thinking: description.supports_thinking,
        thinking_start_tag: nonempty(&description.thinking_start),
        thinking_end_tag: description
            .thinking_ends
            .first()
            .and_then(|end| nonempty(end)),
        template_fingerprint: fingerprint,
    })
}

/// Model facts the service reports without loading the model.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelProperties {
    /// The GGUF the model was opened from (the first shard of a split GGUF).
    pub model_path: std::path::PathBuf,
    /// Bytes across every shard.
    pub model_size_bytes: u64,
    pub name: Option<String>,
    pub architecture: Option<String>,
    /// The served context bound.
    pub context_tokens: u64,
    /// The artifact's declared context capability.
    pub training_context_tokens: u64,
    pub vision: bool,
    pub chat_template: String,
    pub template_fingerprint: String,
    /// The native template's capability flags.
    pub template_capabilities: BTreeMap<String, bool>,
    /// A required tool call prepares with tool-call constraints.
    pub tools: bool,
    /// A JSON-schema request prepares with output constraints.
    pub structured_output: bool,
    pub reasoning: ReasoningProfile,
}

pub fn model_properties(host: &HostArtifacts) -> Result<ModelProperties, ChatError> {
    let inspection = host.template_inspection().clone();
    let target = host.package().target();
    let directory = target.directory();
    let text = |key: &str| {
        directory
            .value(key)
            .and_then(|value| value.string())
            .map(str::to_owned)
    };
    let mut sources = target.sources();
    let model_path = sources
        .next()
        .expect("a GGUF artifact has at least one shard")
        .path()
        .to_owned();
    Ok(ModelProperties {
        model_path,
        model_size_bytes: target.sources().map(|source| source.size()).sum(),
        name: text("general.name"),
        architecture: text("general.architecture"),
        context_tokens: host.definition().decoder.context_limit,
        training_context_tokens: host.declared_context_limit(),
        vision: host.media_placeholder().is_some(),
        chat_template: host.templates().default_source().to_owned(),
        template_fingerprint: inspection.fingerprint,
        template_capabilities: host
            .templates()
            .capabilities()
            .map_err(ChatError::Internal)?,
        tools: inspection.tools,
        structured_output: inspection.structured_output,
        reasoning: inspection.reasoning,
    })
}
