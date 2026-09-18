//! Host chat preparation and semantic publication, independent of model execution.
mod preparation;
pub mod reasoning;
mod stream;
mod session;
mod response;
mod templates;
pub mod wire;

pub use magnitude_templates::{Event, PreparedDescription, PreparedRequest, TerminalCause};
pub use preparation::{ConstraintPlan, PreparedChat, PreparedInput};
pub use stream::{ChatStream, StopText, TokenChatStream};
pub use session::{ChatPublication, Session};
pub use response::{CompleteResponse, SseResponse, Usage};
pub use templates::{ChatRequest, TemplateBundle, TemplateSelection, TemplateVariant, ToolChoice};
