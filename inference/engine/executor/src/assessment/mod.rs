//! Metadata-only model assessment.
//!
//! A fixed, model-free measurement basis is taken once per device
//! ([`plan`], [`basis`], [`measure`]). Every model is then assessed
//! analytically from its headers and allocation-free execution plan: decode
//! demand ([`demand`]), speed at each requested depth ([`estimate`]), and
//! memory fit ([`assess`]). No model is loaded, benchmarked or tuned.

pub mod assess;
pub mod basis;
#[cfg(test)]
mod catalog;
pub mod demand;
pub mod estimate;
pub mod measure;
pub mod persist;
pub mod plan;

pub use assess::{
    assess_execution, finish_execution_assessment, prepare_execution_assessment, AssessmentRequest,
    DecodeSpeed, DomainFit, ExecutionAssessment, PreparedExecutionAssessment,
};
pub use basis::{
    BasisIdentity, ClassCost, ClassMeasurement, CostModel, HeadGeometry, HistoryCost,
    MeasuredPoint, MeasurementBasis, MeasurementKey, OperationClass, PointShape, ProjectionCost,
    SecondsBand, MEASUREMENT_PROTOCOL_VERSION,
};
pub use demand::{DecodeDemand, DemandTerm, TermShape};
pub use estimate::{
    estimate_performance, performance_depths, term_seconds, PerformanceConfidence,
    PerformanceEstimate, HIGH_CONFIDENCE_RANGE, MODERATE_CONFIDENCE_RANGE,
};
pub use measure::{
    complete_basis, measure_basis, measure_entry, ClassProfile, MeasurementError,
    MeasurementFailure,
};
pub use persist::{basis_file_name, basis_json, load_basis, parse_basis, store_basis};
pub use plan::measurement_plan;

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
    /// A measured cost produced a non-finite or nonpositive time.
    Estimate(String),
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
            Self::Estimate(message) => write!(formatter, "assessment estimate failed: {message}"),
        }
    }
}

impl std::error::Error for AssessmentError {}
