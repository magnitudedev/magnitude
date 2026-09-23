//! Service policy observes logical work and resource prices, never tensor layouts.
mod runtime;
pub use magnitude_service::ServiceLimits;
pub use runtime::{EngineClient, EngineRequest, EngineService};
