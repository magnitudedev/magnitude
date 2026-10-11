//! Metadata-only model assessment.
//!
//! Every model is assessed analytically from its headers and
//! allocation-free execution plan: memory fit ([`assess`]), and decode speed
//! at each requested depth from its decode demand ([`demand`]) over the
//! device's memory bandwidth ([`bandwidth`], [`costs`], [`estimate`]). No
//! model is loaded and nothing runs on the device.

pub mod assess;
pub mod bandwidth;
#[cfg(test)]
mod catalog;
pub mod costs;
pub mod demand;
pub mod estimate;
#[cfg(test)]
pub(crate) mod fixtures;

pub use assess::{
    assess_execution, finish_execution_assessment, prepare_execution_assessment, AssessmentRequest,
    DomainFit, ExecutionAssessment, PreparedExecutionAssessment,
};
pub use bandwidth::{normalize_name, resolve_bandwidth, BandwidthSource, DeviceBandwidth, DeviceClass};
pub use costs::{DecodeCosts, DECODE_COSTS};
pub use demand::{DecodeDemand, HistoryRead};
pub use estimate::{
    estimate_performance, performance_depths, step_seconds, PerformanceEstimate, StepSeconds,
};

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
        }
    }
}

impl std::error::Error for AssessmentError {}
