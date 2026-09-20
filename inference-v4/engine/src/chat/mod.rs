//! Host chat preparation and semantic publication, independent of model execution.
mod preparation;
pub mod reasoning;
mod response;
mod session;
mod stream;
mod templates;
pub mod wire;

pub use magnitude_templates::{Event, PreparedDescription, PreparedRequest, TerminalCause};
pub use preparation::{ConstraintPlan, PreparedChat, PreparedInput};
pub use response::{CompleteResponse, SseResponse, Usage};
pub use session::{ChatPublication, Session};
pub use stream::{ChatStream, StopText, TokenChatStream};
pub use templates::{ChatRequest, TemplateBundle, TemplateSelection, TemplateVariant, ToolChoice};
