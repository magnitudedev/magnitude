//! Preparation-time joint candidate and invocation search.
mod evolution;
mod observation;
mod options;
mod statistics;
pub use observation::{
    CaseArgument, ControlledObserver, Observation, ObservationCase, ObservationError,
    ObservationProtocol, ObservationRequest,
};
pub use options::{
    EvaluationMethod, FeedbackOptions, InvocationParameter, InvocationRange, InvocationScope,
    PreparationOptions,
};
mod invocation;

mod confirmation;

mod scheduling;
pub use scheduling::FeedbackCosts;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeedbackError {
    InvalidOptions(String),
    Observation(ObservationError),
}
mod controller;
pub use controller::FeedbackReport;

mod preparation;
pub use preparation::FeedbackPreparation;

mod support;
pub use support::{assess_observation_support, ObservationSupport};
