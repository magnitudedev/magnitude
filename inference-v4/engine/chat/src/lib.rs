//! Family-neutral chat preparation and semantic response handling.
//! Numerical execution and transport-runtime ownership are composed above this crate.
pub mod artifacts;
mod constraints;
mod preparation;
pub mod reasoning;
mod response;
mod stream;
mod templates;
mod tokenizer;
pub mod wire;

pub use constraints::{CacheLimits, ConstraintState, Vocabulary};
pub use magnitude_generation::{
    DetailedUsage, FinishReason, Options, OutputToken, Sampling, TokenId,
};
pub use magnitude_templates::{Event, PreparedDescription, PreparedRequest, TerminalCause};
pub use preparation::{ConstraintPlan, PreparedChat, PreparedChatInput};
pub use response::{CompleteResponse, SseResponse};
pub use stream::{ChatStream, StopText, TokenChatStream};
pub use templates::{ChatRequest, TemplateBundle, TemplateSelection, TemplateVariant, ToolChoice};
pub use tokenizer::{BpeConfig, ByteBpeTokenizer, PieceKind, SpecialTokens, TokenDecoder};

/// Measured physical execution time attributed to a request. Durations are
/// accumulated at completed program boundaries, in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionTimings {
    pub prompt_ns: u64,
    pub predicted_ns: u64,
}

/// A transport-neutral semantic publication produced from accepted output tokens.
pub struct ChatPublication {
    /// Present on terminal publication after the execution owner acknowledges stop.
    pub usage: Option<DetailedUsage>,
    /// Generation method identity used to derive the speculative backend.
    pub method: Option<String>,
    /// Present with terminal usage after the physical owner reconciles work.
    pub timings: Option<ExecutionTimings>,
    pub events: Vec<Event>,
    /// Accepted content may accompany a terminal execution failure.
    pub error: Option<String>,
}
