//! Family-neutral chat semantics: protocol-neutral requests, template
//! rendering, reasoning resolution, tokenization, constraint descriptions,
//! output parsing and the semantic output stream. Numerical execution and
//! transport ownership are composed above this crate.
pub mod artifacts;
pub mod conformance;
mod error;
pub mod generation;
pub mod output;
mod preparation;
pub mod reasoning;
pub mod request;
pub mod schema;
mod stream;
mod templates;
mod tokenizer;

pub use error::ChatError;
pub use magnitude_generation::{
    DetailedUsage, EndOfGeneration, FinishReason, Options, OutputToken, Sampling, TokenId,
};
pub use magnitude_templates::{
    Event, PreparedDescription, PreparedRequest, Relaxation, RelaxationReason, RelaxationSubject,
    TerminalCause,
};
pub use preparation::{ConstraintPlan, ConstraintSource, PreparedChat, PreparedChatInput};
pub use reasoning::{ReasoningIntent, ResolvedReasoning};
pub use request::{ChatInput, GenerationControls, GenerationRequest};
pub use stream::{ChatStream, StopText, TokenChatStream};
pub use templates::{
    ChatRequest, TemplateBundle, TemplateInspection, TemplateSelection, TemplateVariant,
    ToolChoice, TEMPLATE_FINGERPRINT_VERSION,
};
pub use tokenizer::{
    AddedTokenEnd, BpeConfig, ByteBpeTokenizer, EncodedSequence, Normalization, PieceEncoding,
    PieceKind, SpecialTokens, Split, SplitBehavior, TokenDecoder, TokenizerError,
};

/// Measured physical execution time attributed to a request. Durations are
/// accumulated at completed program boundaries, in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionTimings {
    pub prompt_ns: u64,
    pub predicted_ns: u64,
}
