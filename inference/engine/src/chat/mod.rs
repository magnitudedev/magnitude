//! Host-side chat over a loaded model: generation sessions against the
//! numerical worker and the host-only operations (count, template apply,
//! properties) that share their rendering.
mod operations;
mod session;

pub use operations::{
    apply_template, count_tokens, model_properties, prepare, AppliedTemplate, InputBound,
    ModelProperties, PreparedInput,
};
pub use session::{generate, SessionError, SessionEvent, SessionLimits};
