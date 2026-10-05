use super::{
    reasoning::ResolvedReasoning, AddedTokenEnd, ByteBpeTokenizer, ChatError, ChatRequest,
    EncodedSequence, PreparedRequest, SpecialTokens, TemplateBundle, TemplateSelection, TokenId,
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
    /// Where the prompt's added tokens end, from its one encoding.
    added_token_ends: Vec<AddedTokenEnd>,
    constraint: Option<ConstraintSource>,
    reasoning: ResolvedReasoning,
}

/// Stands in for the last message's content to locate where it begins. The
/// prompts diverge at or after the content's start; the boundary is the
/// added token before it, so content that happens to begin like the probe
/// finds the same boundary.
const CONTENT_PROBE: &str = "\u{E000}";

fn internal(message: String) -> ChatError {
    ChatError::Internal(message)
}

impl PreparedChat {
    /// Template grammars must compile fully lexical. A template's
    /// parallel-call grammar can fail that where its single-call grammar does
    /// not (Qwen's, for an object of free-form JSON among optional
    /// properties): certification renders lexemes one byte at a time and the
    /// first mask exceeds the matcher's item limit. Such a request is
    /// prepared for one tool call per turn instead.
    pub fn prepare(
        bundle: &TemplateBundle,
        tokenizer: &ByteBpeTokenizer,
        request: &ChatRequest,
        selection: &TemplateSelection<'_>,
    ) -> Result<Self, ChatError> {
        let prepared = Self::prepare_exactly(bundle, tokenizer, request, selection)?;
        let character_lexemes = |prepared: &Self| {
            prepared
                .constraint
                .as_ref()
                .map_or(0, |constraint| constraint.report.character_lexemes)
        };
        if !request.parallel_tool_calls
            || request.grammar.is_some()
            || character_lexemes(&prepared) == 0
        {
            return Ok(prepared);
        }
        let mut single = request.clone();
        single.parallel_tool_calls = false;
        let fallback = Self::prepare_exactly(bundle, tokenizer, &single, selection)?;
        Ok(if character_lexemes(&fallback) < character_lexemes(&prepared) {
            fallback
        } else {
            prepared
        })
    }

    fn prepare_exactly(
        bundle: &TemplateBundle,
        tokenizer: &ByteBpeTokenizer,
        request: &ChatRequest,
        selection: &TemplateSelection<'_>,
    ) -> Result<Self, ChatError> {
        let (native, reasoning) = bundle.prepare(request, selection)?;
        let description = native.description();
        let EncodedSequence {
            tokens,
            added_token_ends,
        } = tokenizer
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
            added_token_ends,
            constraint,
            reasoning,
        })
    }
    /// The prompt token position that opens the last message: the position
    /// after the last added token (the template's turn markup) before that
    /// message's content. Every request differing from `request` only in
    /// that content shares the prompt's tokens up to it, so it is where a
    /// conversation's next variation of its last message diverges.
    ///
    /// `request` is rendered again with the last message's content replaced
    /// by [`CONTENT_PROBE`]; the content begins at the first byte where the
    /// two prompts differ. `None` when the template rejects the probe (it
    /// may validate content) or renders the prompt unchanged, or when no
    /// added token opens the message inside the prompt.
    pub fn last_message_boundary(
        &self,
        bundle: &TemplateBundle,
        request: &ChatRequest,
        selection: &TemplateSelection<'_>,
    ) -> Option<usize> {
        let mut probe = request.clone();
        probe
            .messages
            .last_mut()?
            .as_object_mut()?
            .insert("content".into(), CONTENT_PROBE.into());
        let (native, _) = bundle.prepare(&probe, selection).ok()?;
        let prompt = self.prompt();
        let probed = &native.description().prompt;
        if prompt == probed {
            return None;
        }
        let content = prompt
            .bytes()
            .zip(probed.bytes())
            .take_while(|(left, right)| left == right)
            .count();
        self.added_token_ends
            .iter()
            .rev()
            .find(|end| end.offset <= content)
            .map(|end| end.position)
            .filter(|&position| position < self.input.tokens.len())
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
