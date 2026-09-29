//! Generation intent bound to the worker's own vocabulary. Options and
//! constraint plans cross the host/worker boundary as descriptions; the
//! worker checks they were prepared for its tokenizer and instantiates the
//! grammar matcher here.
use magnitude_artifacts::InputLayout;
use magnitude_chat::{ByteBpeTokenizer, ConstraintPlan, PieceKind};
use magnitude_generation::{Constraint, GenerationSeed, Options, TokenId};
use magnitude_grammar::{CacheLimits, Vocabulary};
use std::sync::Arc;

pub struct GenerationBinding {
    vocabulary: Vocabulary,
    tokenizer: Arc<ByteBpeTokenizer>,
}

impl GenerationBinding {
    pub fn new(
        tokenizer: Arc<ByteBpeTokenizer>,
        projection: usize,
        cache: CacheLimits,
    ) -> Result<Self, String> {
        let vocabulary = Vocabulary::new(tokenizer.clone(), projection, cache)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            vocabulary,
            tokenizer,
        })
    }

    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Bind prepared tokens, options and constraint to this vocabulary.
    /// Tokens and layout come from the family-interpreted input, which may
    /// expand media placeholders.
    pub fn seed(
        &mut self,
        constraint: Option<&ConstraintPlan>,
        options: Options,
        tokens: &[TokenId],
        layout: &InputLayout,
    ) -> Result<GenerationSeed, String> {
        let tokenizer = &self.tokenizer;
        if options.vocabulary != self.vocabulary.projection()
            || &options.stop_tokens != tokenizer.stop_tokens()
            || &options.suppressed_tokens != tokenizer.suppressed_tokens()
            || tokens.iter().any(|token| {
                token.0 as usize >= tokenizer.vocabulary()
                    || tokenizer.kind(*token).ok() == Some(PieceKind::Unused)
            })
        {
            return Err(
                "prepared input or generation options do not match the execution vocabulary".into(),
            );
        }
        let constraint = constraint
            .map(|plan| -> Result<Box<dyn Constraint>, String> {
                if plan.artifact_identity != tokenizer.artifact_identity()
                    || plan.tokenizer_identity != tokenizer.identity()
                {
                    return Err(
                        "the constraint was prepared for another artifact or tokenizer".into(),
                    );
                }
                let bound = self
                    .vocabulary
                    .bind(&plan.grammar, &plan.prefix)
                    .map_err(|error| error.to_string())?;
                Ok(Box::new(bound))
            })
            .transpose()?;
        GenerationSeed::new(tokens.to_vec(), layout.clone(), options, constraint)
    }
}
