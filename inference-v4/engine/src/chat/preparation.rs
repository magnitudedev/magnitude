use super::{ChatRequest, PreparedRequest, TemplateBundle, TemplateSelection};
use crate::inputs::{ByteBpeTokenizer, SpecialTokens, TokenId};
use serde::{Deserialize, Serialize};

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
pub struct PreparedInput {
    pub artifact_identity: String,
    pub tokenizer_identity: String,
    pub tokens: Vec<TokenId>,
    pub constraint: Option<ConstraintPlan>,
}

/// Prompt, token input, and parser all derive from one prepared native request.
pub struct PreparedChat {
    native: PreparedRequest,
    input: PreparedInput,
}
impl PreparedChat {
    pub fn prepare(
        bundle: &TemplateBundle,
        tokenizer: &ByteBpeTokenizer,
        request: &ChatRequest,
        selection: &TemplateSelection<'_>,
    ) -> Result<Self, String> {
        let native = bundle.prepare(request, selection)?;
        let description = native.description();
        let tokens = tokenizer.encode(&description.prompt, SpecialTokens::Recognize)?;
        if tokens.is_empty() {
            return Err("chat template produced no input tokens".into());
        }
        let constraint = if description.grammar.is_empty() {
            None
        } else {
            let prefix = tokenizer.encode(
                &description.grammar_initial_prefix,
                SpecialTokens::Recognize,
            )?;
            let mut bytes = Vec::new();
            for token in prefix {
                bytes.extend_from_slice(tokenizer.piece(token, false)?);
            }
            if bytes != description.grammar_initial_prefix.as_bytes() {
                return Err(
                    "grammar initial prefix is not exactly representable by the tokenizer".into(),
                );
            }
            Some(ConstraintPlan {
                artifact_identity: tokenizer.artifact_identity().into(),
                tokenizer_identity: tokenizer.identity().into(),
                template_identity: native.template_identity().into(),
                converter_identity: crate::generation::grammar::CONVERTER_IDENTITY.into(),
                gbnf: description.grammar.clone(),
                initial_prefix: description.grammar_initial_prefix.clone(),
            })
        };
        Ok(Self {
            native,
            input: PreparedInput {
                artifact_identity: tokenizer.artifact_identity().into(),
                tokenizer_identity: tokenizer.identity().into(),
                tokens,
                constraint,
            },
        })
    }
    pub fn prompt(&self) -> &str {
        &self.native.description().prompt
    }
    pub fn input(&self) -> &PreparedInput {
        &self.input
    }
    pub fn native(&self) -> &PreparedRequest {
        &self.native
    }
    pub fn prompt_tokens(&self) -> usize {
        self.input.tokens.len()
    }
}
