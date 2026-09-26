use super::{
    reasoning::ResolvedReasoning, ByteBpeTokenizer, ChatError, ChatRequest, PreparedRequest,
    SpecialTokens, TemplateBundle, TemplateSelection, TokenId,
};
use magnitude_generation::ReasoningBudget;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;

/// Immutable symbolic constraint input. The execution owner must bind it to the
/// exact vocabulary and validate the prefix before admitting generation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConstraintPlan {
    pub artifact_identity: String,
    pub tokenizer_identity: String,
    pub template_identity: String,
    pub converter_identity: String,
    pub gbnf: String,
    pub initial_prefix: String,
}

/// Data that can cross into the execution owner without native parser handles.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedChatInput {
    pub artifact_identity: String,
    pub tokenizer_identity: String,
    pub tokens: Vec<TokenId>,
    pub constraint: Option<ConstraintPlan>,
}

/// Prompt, token input, and parser all derive from one prepared native request.
pub struct PreparedChat {
    native: PreparedRequest,
    input: PreparedChatInput,
    reasoning: ResolvedReasoning,
}

fn internal(message: String) -> ChatError {
    ChatError::Internal(message)
}

impl PreparedChat {
    pub fn prepare(
        bundle: &TemplateBundle,
        tokenizer: &ByteBpeTokenizer,
        request: &ChatRequest,
        selection: &TemplateSelection<'_>,
    ) -> Result<Self, ChatError> {
        let (native, reasoning) = bundle.prepare(request, selection)?;
        let description = native.description();
        let tokens = tokenizer
            .encode(&description.prompt, SpecialTokens::Recognize)
            .map_err(internal)?;
        if tokens.is_empty() {
            return Err(ChatError::InvalidRequest(
                "chat template produced no input tokens".into(),
            ));
        }
        let (gbnf, initial_prefix) = match (&request.grammar, description.grammar.is_empty()) {
            (Some(grammar), true) => (grammar.clone(), String::new()),
            (Some(_), false) => {
                return Err(ChatError::InvalidRequest(
                    "a custom grammar cannot be combined with template output constraints".into(),
                ))
            }
            (None, _) => (
                description.grammar.clone(),
                description.grammar_initial_prefix.clone(),
            ),
        };
        let constraint = if gbnf.is_empty() {
            None
        } else {
            let prefix = tokenizer
                .encode(&initial_prefix, SpecialTokens::Recognize)
                .map_err(internal)?;
            let mut bytes = Vec::new();
            for token in prefix {
                bytes.extend_from_slice(tokenizer.piece(token, false).map_err(internal)?);
            }
            if bytes != initial_prefix.as_bytes() {
                return Err(ChatError::Internal(
                    "grammar initial prefix is not exactly representable by the tokenizer".into(),
                ));
            }
            Some(ConstraintPlan {
                artifact_identity: tokenizer.artifact_identity().into(),
                tokenizer_identity: tokenizer.identity().into(),
                template_identity: native.template_identity().into(),
                converter_identity: magnitude_generation::grammar::CONVERTER_IDENTITY.into(),
                gbnf,
                initial_prefix,
            })
        };
        Ok(Self {
            native,
            input: PreparedChatInput {
                artifact_identity: tokenizer.artifact_identity().into(),
                tokenizer_identity: tokenizer.identity().into(),
                tokens,
                constraint,
            },
            reasoning,
        })
    }
    pub fn prompt(&self) -> &str {
        &self.native.description().prompt
    }
    pub fn input(&self) -> &PreparedChatInput {
        &self.input
    }
    pub fn native(&self) -> &PreparedRequest {
        &self.native
    }
    pub fn prompt_tokens(&self) -> usize {
        self.input.tokens.len()
    }
    pub fn reasoning(&self) -> &ResolvedReasoning {
        &self.reasoning
    }

    /// The hard reasoning cap for this prompt: the template's reasoning tags as
    /// tokens, and whether the prompt already opened reasoning.
    pub fn reasoning_budget(
        &self,
        tokenizer: &ByteBpeTokenizer,
        tokens: NonZeroU32,
    ) -> Result<ReasoningBudget, ChatError> {
        let description = self.native.description();
        let end = description.thinking_ends.first().filter(|end| !end.is_empty());
        let (start, end) = match (description.thinking_start.as_str(), end) {
            (start, Some(end)) if description.supports_thinking && !start.is_empty() => {
                (start, end)
            }
            _ => {
                return Err(ChatError::InvalidRequest(
                    "the model's template does not delimit reasoning, so a reasoning budget \
                     cannot be enforced"
                        .into(),
                ))
            }
        };
        let encode = |text: &str| {
            tokenizer
                .encode(text.trim(), SpecialTokens::Recognize)
                .map_err(internal)
        };
        Ok(ReasoningBudget {
            tokens: tokens.get(),
            start: encode(start)?,
            end: encode(end)?,
            open: description.generation_prefix.trim_end().ends_with(start.trim()),
        })
    }
}
