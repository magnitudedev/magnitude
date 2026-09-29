use super::{
    reasoning::ResolvedReasoning, ByteBpeTokenizer, ChatError, ChatRequest, PreparedRequest,
    SpecialTokens, TemplateBundle, TemplateSelection, TokenId,
};
use magnitude_generation::ReasoningBudget;
use magnitude_grammar::{CompileReport, Grammar};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;

/// A compiled output constraint. The execution owner binds it to its own
/// vocabulary, which must be the one the prefix was tokenized with.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConstraintPlan {
    pub artifact_identity: String,
    pub tokenizer_identity: String,
    pub grammar: Grammar,
    /// The grammar's leading text that the prompt already contains.
    pub prefix: Vec<TokenId>,
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

/// The GBNF an output constraint was compiled from, and how it compiled.
pub struct ConstraintSource {
    pub gbnf: String,
    pub report: CompileReport,
}

/// Prompt, token input, and parser all derive from one prepared native request.
pub struct PreparedChat {
    native: PreparedRequest,
    input: PreparedChatInput,
    constraint: Option<ConstraintSource>,
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
            .encode_sequence(&description.prompt)
            .map_err(internal)?;
        if tokens.is_empty() {
            return Err(ChatError::InvalidRequest(
                "chat template produced no input tokens".into(),
            ));
        }
        let (gbnf, initial_prefix, supplied) =
            match (&request.grammar, description.grammar.is_empty()) {
                (Some(grammar), true) => (grammar.clone(), String::new(), true),
                (Some(_), false) => {
                    return Err(ChatError::InvalidRequest(
                        "a custom grammar cannot be combined with template output constraints"
                            .into(),
                    ))
                }
                (None, _) => (
                    description.grammar.clone(),
                    description.grammar_initial_prefix.clone(),
                    false,
                ),
            };
        let (plan, constraint) = if gbnf.is_empty() {
            (None, None)
        } else {
            let prefix = tokenizer
                .encode(&initial_prefix, SpecialTokens::Recognize)
                .map_err(internal)?;
            let mut bytes = Vec::new();
            for token in &prefix {
                bytes.extend_from_slice(tokenizer.piece(*token, false).map_err(internal)?);
            }
            if bytes != initial_prefix.as_bytes() {
                return Err(ChatError::Internal(
                    "grammar initial prefix is not exactly representable by the tokenizer".into(),
                ));
            }
            // A caller's grammar may be invalid; a template's must compile.
            let compiled = bundle.compile_grammar(&gbnf).map_err(|error| {
                if supplied {
                    ChatError::InvalidRequest(error.to_string())
                } else {
                    ChatError::Internal(error.to_string())
                }
            })?;
            (
                Some(ConstraintPlan {
                    artifact_identity: tokenizer.artifact_identity().into(),
                    tokenizer_identity: tokenizer.identity().into(),
                    grammar: compiled.grammar.clone(),
                    prefix,
                }),
                Some(ConstraintSource {
                    gbnf,
                    report: compiled.report.clone(),
                }),
            )
        };
        Ok(Self {
            native,
            input: PreparedChatInput {
                artifact_identity: tokenizer.artifact_identity().into(),
                tokenizer_identity: tokenizer.identity().into(),
                tokens,
                constraint: plan,
            },
            constraint,
            reasoning,
        })
    }
    /// The output constraint's GBNF and compile report, when constrained.
    pub fn constraint(&self) -> Option<&ConstraintSource> {
        self.constraint.as_ref()
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
        let end = description
            .thinking_ends
            .first()
            .filter(|end| !end.is_empty());
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
            open: description
                .generation_prefix
                .trim_end()
                .ends_with(start.trim()),
        })
    }
}
