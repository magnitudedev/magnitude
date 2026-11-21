//! Metadata-only model assessment.
//!
//! Every model is assessed analytically from its headers and
//! allocation-free execution plan: memory fit ([`assess`]), and decode speed
//! at each requested depth ([`estimate`]) from its per-call decode demand
//! ([`demand`], each call's launches from its entry's declaration,
//! [`launches`]) over its executor's measured kernel costs
//! ([`executor_costs`]). No model is loaded and nothing runs on a device:
//! the costs derive from measurements the measurement job recorded. A
//! streamed decode is priced from the plan's structure through the one step
//! model ([`streamed`]), on the machine's stored figures and the one generic
//! miss model; a load prices its own bank on the same step, beside what its
//! decode read this session ([`DecodePrice`]). What the
//! measurement job measures is assessment's standing requests, fixed per
//! backend, then what loads request of it at runtime ([`requests`]).

pub mod assess;
#[cfg(test)]
mod catalog;
#[cfg(test)]
mod draft_traces;
pub mod cost;
pub mod demand;
pub mod estimate;
pub mod executor_costs;
#[cfg(test)]
pub(crate) mod fixtures;
pub mod launches;
#[cfg(feature = "machine-line")]
pub mod line;
pub mod misses;
#[cfg(test)]
mod parity;
#[cfg(test)]
mod target_outlines;
pub mod pool;
pub mod priced;
pub mod requests;
pub mod stage_timer;
pub mod streamed;
pub mod streaming;
pub mod streaming_costs;
#[cfg(test)]
mod vision_traces;
#[cfg(any(test, feature = "fixtures"))]
pub mod synthetic;

pub use assess::{
    assess_execution_fit, assess_execution_fit_with_priced_bank, decode_demand, estimate_execution,
    launch_prices, lookahead, execution_memory_charge, PricedBankContext, StructuralCharge,
    TargetLayoutCache, prepare_execution_assessment, AssessmentRequest, ColdSource, DomainFit,
    ExecutionAssessment, ExecutionFit, ExecutionShortfall, ExpertShares, FileOnly, MemoryBreakdown,
    ModelPlacement, PreparedExecutionAssessment, Residency,
};
pub use crate::programs::graph::sizing::SizingScope;
pub use demand::{format_of, layout_of, Call, DecodeDemand};
pub use estimate::{estimate_decode, expert_pass, performance_depths, PerformanceEstimate};
pub use executor_costs::{
    ChainCost, CostWalk, Derived, ExecutorCosts, ExecutorCostsError, FormatRates, Held, HistoryCosts,
    MatrixCosts, MeasuredCost, Measurement, Placement, Placements, Required, RouteCosts, StoredCosts,
    WeightRate, Whose, HISTORY_LAYOUTS, MEASUREMENT_SECONDS,
};
pub use launches::{Declarations, Kernel, Launches, Words};
#[cfg(feature = "machine-line")]
pub use line::{Line, MachineLine, MachineLineError, Operand, Peak, Rate, Rows, Stream, Sweep};
pub use cost::{copy_hits, copy_nanoseconds, expert_lines, host_lines, BankLines, Beside, HostLinesError, StoredCpuSite, POLICY_LAYOUT};
pub use priced::Priced;
pub(crate) use streaming_costs::CpuForm;
#[cfg(feature = "machine-line")]
pub use requests::line_outstanding;
pub use requests::{MeasurementPhase, Outstanding, PendingPrices, PointRequest, PointRequests, SiteRequest, Waiter};
pub use streamed::{streamed_layers, StreamedError, StreamedFigures};
pub use misses::{Anticipated, HintCounts, LayerStep, TableCurve, TableRange};
pub use streaming::{
    GroupLayerTerms, GroupUnion, Notice, Stages,
};
pub use streaming_costs::{
    CeilingError, ConvertCosts, ConvertForm, MemoryCosts, MemoryError,
    CONVERTED_RECORDS,
    CpuCosts, GraphMemory, RequestReads, StorageCosts, StorageError, TierCopyCosts, TierCopyError, TierCopyPoint, StoredVolume, VolumeCosts,
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
    /// The load's kernel inventory could not be formed on the backend.
    Catalog(crate::CatalogError),
    /// The backend has no native implementation of an entry the load
    /// prepares.
    Unimplemented {
        entry: &'static str,
        backend: seismic::BackendName,
    },
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
            Self::Catalog(error) => write!(formatter, "assessment demand failed: {error}"),
            Self::Unimplemented { entry, backend } => write!(
                formatter,
                "assessment demand failed: `{entry}` has no native implementation for {}",
                backend.as_str()
            ),
        }
    }
}

impl From<crate::PlanError> for AssessmentError {
    fn from(error: crate::PlanError) -> Self {
        Self::Plan(error.to_string())
    }
}

impl std::error::Error for AssessmentError {}

/// A price's refusal as a demand failure, its reason as the cost model
/// states it.
impl From<seismic::measure::pricing::PriceError> for AssessmentError {
    fn from(error: seismic::measure::pricing::PriceError) -> Self {
        Self::Demand(error.to_string())
    }
}
