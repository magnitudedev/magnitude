//! Metadata-only model assessment.
//!
//! Every model is assessed analytically from its headers and
//! allocation-free execution plan: memory fit ([`assess`]), and decode speed
//! at each requested depth ([`estimate`]) from its per-call decode demand
//! ([`demand`], each call's launches from its entry's declaration,
//! [`launches`]) over its executor's measured kernel costs
//! ([`executor_costs`]). No model is loaded and nothing runs on a device:
//! the costs derive from measurements the measurement job recorded. A
//! streamed decode is priced by the one cost function of where a missed
//! expert is computed ([`cost`]) over a load's miss record ([`misses`]).
//! What the measurement job measures is what consumers request of it
//! ([`requests`]), in the order of who waits on it.

pub mod assess;
#[cfg(test)]
mod catalog;
pub mod cost;
pub mod demand;
pub mod estimate;
pub mod executor_costs;
#[cfg(test)]
pub(crate) mod fixtures;
pub mod launches;
pub mod misses;
pub mod priced;
pub mod requests;
pub mod step;
pub mod streaming;
pub mod streaming_costs;

pub use assess::{
    assess_execution_fit, decode_demand, decode_pricing, estimate_execution, execution_memory_charge,
    prepare_execution_assessment, AssessmentRequest, ColdSource, DomainFit, ExecutionAssessment,
    ExecutionFit, ExecutionShortfall, ExpertShares, FileOnly, PreparedExecutionAssessment, Residency,
};
pub use demand::{Bytes, Call, DecodeDemand, Experts, Flops, HistoryRead, Work};
pub use estimate::{
    estimate_decode, performance_depths, step_seconds, CallSeconds, ExpertPass, PerformanceEstimate, Pricing,
    StepSeconds,
};
pub use executor_costs::{
    ChainCost, Derived, ExecutorCosts, ExecutorCostsError, FormatRates, Held, HistoryCosts, HistoryLine,
    MatrixCosts, MeasuredCost, Measurement, OperandStorage, Placement, Placements, Required, RouteCosts, StoredCosts,
    WeightRate, HISTORY_LAYOUTS, MEASUREMENT_SECONDS,
};
pub use launches::{Declarations, Kernel, Launches};
pub use cost::{
    BankLines, CpuLines, CpuLinesError, StoredCpuSite, ExpertLines, Exposed, GpuLines, Hits, HostLines, HostLinesError, LineError, Miss, Nanoseconds, Pass,
    PassLine, PricingError, RowCost, RowLine, Sample, ServedMisses, Site, SiteLines,
};
pub use priced::Priced;
pub(crate) use streaming_costs::CpuForm;
pub use requests::{PointRequest, PointRequests, SiteRequest, Waiter};
pub use step::{LayerLoad, Lead, ReadAhead, StepInputs, StepTime};
pub use misses::{
    Anticipated, HeldSlots, HintCounts, LaunchResiduals, LaunchSample, LayerStep, MissMergeError, MissRecord, MissRecordError, RecordFeed,
    Residuals, StoredMisses, MeasuredLayer, StreamedStep, TableCurve, MISS_RATE_ERROR,
};
pub use streaming::{
    BankSlots, DecodeWaiting, GroupLayerTerms, GroupReads, GroupUnion, Notice, PlaceGrant, Places, Stages,
};
pub use streaming_costs::{
    CeilingError, ConvertCosts, ConvertForm, FileLine, MemoryCosts, MemoryError, MoveCosts, MoveError, MoveForm, RecordLine,
    CONVERTED_RECORDS,
    CpuCosts, GraphMemory, MoveLines, ReadLine, ReadLineError, RequestReads, TierCopyCosts, TierCopyError, VolumeCosts,
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

impl From<crate::PlanError> for AssessmentError {
    fn from(error: crate::PlanError) -> Self {
        Self::Plan(error.to_string())
    }
}

impl std::error::Error for AssessmentError {}
