//! Metadata-only model assessment.
//!
//! A fixed measurement basis is taken once per device ([`basis`],
//! [`measure`]). Every model is then assessed analytically from its headers
//! and allocation-free execution plan: decode demand ([`demand`]), speed at
//! each requested depth ([`estimate`]), and memory fit and compatibility
//! ([`assess`]). No model is loaded, benchmarked or tuned.

pub mod assess;
pub mod basis;
pub mod demand;
pub mod estimate;
pub mod measure;
pub mod persist;
pub mod plan;

pub use assess::{
    assess_execution, AssessmentRequest, DomainFit, ExecutionAssessment, IncompatibleReason,
};
pub use basis::{
    BasisIdentity, ClassCost, ClassMeasurement, CostModel, MeasuredPoint, MeasurementBasis,
    MeasurementKey, OperationClass, SecondsBand, MEASUREMENT_PROTOCOL_VERSION,
};
pub use demand::{DecodeDemand, DemandTerm};
pub use estimate::{
    estimate_performance, performance_depths, PerformanceConfidence, PerformanceEstimate,
    HIGH_CONFIDENCE_RANGE, MODERATE_CONFIDENCE_RANGE,
};
pub use measure::{
    measure_basis, measure_basis_observed, measure_class, ClassProfile, MeasurementError,
};
pub use persist::{basis_file_name, basis_json, load_basis, parse_basis, store_basis};
pub use plan::{measurement_plan, DeclaredConfiguration, QWEN35_CONFIGURATIONS};

use std::fmt;

/// An operational assessment failure. It produces no result: the service
/// drops the target. Incompatibility is a result, not an error.
#[derive(Clone, Debug, PartialEq)]
pub enum AssessmentError {
    /// The model's plans could not be derived from its headers.
    Plan(String),
    /// Header-derived demand is inconsistent with the plans.
    Demand(String),
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
            Self::Memory(message) => write!(formatter, "assessment memory bound failed: {message}"),
            Self::Estimate(message) => write!(formatter, "assessment estimate failed: {message}"),
        }
    }
}

impl std::error::Error for AssessmentError {}
