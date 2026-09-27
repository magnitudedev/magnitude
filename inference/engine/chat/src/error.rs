use super::reasoning::ReasoningError;

/// Why a chat request cannot be prepared or counted for a model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatError {
    /// The request is malformed or asks for something the model's template
    /// cannot render or constrain.
    InvalidRequest(String),
    /// The requested reasoning behavior is unavailable for this model.
    Reasoning(ReasoningError),
    /// The prepared input leaves no generation capacity in the served context.
    ContextLengthExceeded {
        prompt_tokens: usize,
        context_tokens: usize,
    },
    /// Artifact-derived chat state is inconsistent with itself.
    Internal(String),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) | Self::Internal(message) => formatter.write_str(message),
            Self::Reasoning(error) => error.fmt(formatter),
            Self::ContextLengthExceeded {
                prompt_tokens,
                context_tokens,
            } => write!(
                formatter,
                "prompt is too long: {prompt_tokens} tokens leave no generation capacity in a \
                 {context_tokens}-token context"
            ),
        }
    }
}

impl std::error::Error for ChatError {}
