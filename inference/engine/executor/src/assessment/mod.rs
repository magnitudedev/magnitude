//! Metadata-only model assessment.
//!
//! Every model is assessed analytically from its headers and
//! allocation-free execution plan: memory fit ([`assess`]), and decode speed
//! at each requested depth ([`estimate`]) from its per-call decode demand
//! ([`demand`], each call's launches from its entry's declaration,
//! [`launches`]) over its executor's measured kernel costs
//! ([`executor_costs`]). No model is loaded and nothing runs on a device:
//! the costs derive from measurements the measurement job recorded.

pub mod assess;
#[cfg(test)]
mod catalog;
pub mod demand;
pub mod estimate;
pub mod executor_costs;
#[cfg(test)]
pub(crate) mod fixtures;
pub mod launches;

pub use assess::{
    assess_execution_fit, decode_pricing, estimate_execution, prepare_execution_assessment,
    AssessmentRequest, DomainFit, ExecutionAssessment, ExecutionFit, ExecutionShortfall,
    PreparedExecutionAssessment,
};
pub use demand::{Bytes, Call, DecodeDemand, Experts, Flops, HistoryRead, Work};
pub use estimate::{
    estimate_decode, performance_depths, step_seconds, CallSeconds, PerformanceEstimate, Pricing,
    StepSeconds,
};
pub use executor_costs::{
    ExecutorCosts, ExecutorCostsError, Figure, HistoryCosts, MatrixCosts, MeasuredCost,
    Measurement, Placement, Placements, StoredCosts, Unmeasured, WeightRate, HISTORY_LAYOUTS,
    SAMPLING_SECONDS,
};
pub use launches::{Declarations, Kernel, Launches};

use crate::GraphError;
use std::fmt;

/// An assessment failure: it produces no result. A kernel domain violation
/// (`Graph(GraphError::KernelDomain)`) is a property of the model on the
/// backend, which the engine classifies as unsupported; every other failure
/// is operational, and the service drops the target.
#[derive(Clone, Debug, PartialEq)]
pub enum AssessmentError {
    /// The model's plans could not be derived from its headers.
    Plan(String),
    /// Header-derived demand is inconsistent with the plans.
    Demand(String),
    /// The model's graphs could not be built on the planned backend.
    Graph(GraphError),
    /// A memory observation or fit bound could not be established.
    Memory(String),
    /// A figure the estimate needs (a weight format's throughput, a history
    /// layout's attention cost) is not measured on the executor: the
    /// estimate is unavailable.
    Unmeasured(String),
}

impl fmt::Display for AssessmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(message) => write!(formatter, "assessment planning failed: {message}"),
            Self::Demand(message) => write!(formatter, "assessment demand failed: {message}"),
            Self::Graph(error) => {
                write!(formatter, "assessment graph construction failed: {error}")
            }
            Self::Memory(message) => write!(formatter, "assessment memory bound failed: {message}"),
            Self::Unmeasured(message) => write!(formatter, "not measured: {message}"),
        }
    }
}

impl std::error::Error for AssessmentError {}
