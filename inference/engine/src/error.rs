//! Typed outcomes of every public engine operation (integration spec §4.4).
//!
//! Lower crates keep their own precise errors; this module is where they are
//! classified into the distinctions a host must act on. Every value here is
//! serializable, so a cause and its retry meaning survive the worker boundary.

use magnitude_executor::platform::{
    DeviceRequest, MemoryPolicyError, PlatformError, SelectionError,
};
use magnitude_executor::{CatalogError, CatalogFailure, GraphError, PlanError};
pub use magnitude_artifacts::{UnsupportedReason, WeightTypeCount};
use magnitude_family_contracts::FamilyError;
use seismic::ResolveError as SeismicResolveError;
use seismic::StoreIdentity;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Why a loaded model stopped serving.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnloadCause {
    /// Headroom stayed at or below the planning reserve after every release
    /// (engine memory policy): the engine unloaded itself.
    MemoryPressure,
    /// The host asked the worker to shut down.
    Shutdown,
    /// The device or its driver failed during execution.
    DeviceLost { reason: String },
    /// An engine invariant failed; the worker cannot continue.
    Internal { reason: String },
}

impl fmt::Display for UnloadCause {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryPressure => formatter.write_str("memory pressure"),
            Self::Shutdown => formatter.write_str("shutdown"),
            Self::DeviceLost { reason } => write!(formatter, "device lost: {reason}"),
            Self::Internal { reason } => write!(formatter, "internal failure: {reason}"),
        }
    }
}

/// Model material that cannot be read or is corrupt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactError {
    /// A package component is absent or unreadable.
    Missing { path: String, reason: String },
    /// A component was read but is not valid model material.
    Invalid { reason: String },
}

impl ArtifactError {
    pub(crate) fn from_artifacts(error: magnitude_artifacts::Error, path: &std::path::Path) -> Self {
        match error {
            magnitude_artifacts::Error::Io(error) => Self::Missing {
                path: path.display().to_string(),
                reason: error.to_string(),
            },
            other => Self::Invalid {
                reason: other.to_string(),
            },
        }
    }
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { path, reason } => write!(formatter, "artifact {path} is missing: {reason}"),
            Self::Invalid { reason } => write!(formatter, "artifact is invalid: {reason}"),
        }
    }
}

/// Material this engine cannot execute here, with every cause the engine
/// names and the developer text describing the refusal. `reasons` is never
/// empty. Never retryable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnsupportedModel {
    pub reasons: Vec<UnsupportedReason>,
    pub detail: String,
}

impl UnsupportedModel {
    pub fn new(reasons: Vec<UnsupportedReason>, detail: impl Into<String>) -> Self {
        let detail = detail.into();
        debug_assert!(!reasons.is_empty(), "a refusal names no reason: {detail}");
        Self { reasons, detail }
    }

    pub fn reason(reason: UnsupportedReason, detail: impl Into<String>) -> Self {
        Self::new(vec![reason], detail)
    }

    /// Material whose header does not describe a model the engine can
    /// build.
    pub fn unreadable(detail: impl Into<String>) -> Self {
        Self::reason(UnsupportedReason::Unreadable, detail)
    }

    /// A drafter or projector that disagrees with the target it is bound
    /// to.
    pub fn mismatch(detail: impl Into<String>) -> Self {
        let detail = detail.into();
        Self::reason(
            UnsupportedReason::TargetMismatch {
                detail: detail.clone(),
            },
            detail,
        )
    }

    /// The refusals of independent checks as one: every distinct reason in
    /// order, and every detail. `None` when nothing was refused.
    pub fn union(refusals: Vec<Self>) -> Option<Self> {
        if refusals.is_empty() {
            return None;
        }
        let mut reasons = Vec::new();
        let mut details = Vec::with_capacity(refusals.len());
        for refusal in refusals {
            for reason in refusal.reasons {
                if !reasons.contains(&reason) {
                    reasons.push(reason);
                }
            }
            details.push(refusal.detail);
        }
        Some(Self::new(reasons, details.join("; ")))
    }
}

impl UnsupportedModel {
    /// Chat templates the engine cannot read or render.
    pub fn chat_template(detail: impl Into<String>) -> Self {
        Self::reason(UnsupportedReason::ChatTemplate, detail)
    }
}

impl From<magnitude_chat::TokenizerError> for UnsupportedModel {
    fn from(error: magnitude_chat::TokenizerError) -> Self {
        let detail = error.to_string();
        match error {
            magnitude_chat::TokenizerError::Unsupported(reasons) => Self::new(reasons, detail),
            magnitude_chat::TokenizerError::Invalid(_) => Self::unreadable(detail),
        }
    }
}

impl From<FamilyError> for UnsupportedModel {
    fn from(error: FamilyError) -> Self {
        Self::new(error.reasons, error.detail)
    }
}

impl fmt::Display for UnsupportedModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

/// A device request that no longer resolves to exactly one usable device.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceSelectionError {
    /// No usable device matches the request; `evidence` lists why each
    /// candidate (and each backend without candidates) is unusable.
    Unavailable {
        request: DeviceRequest,
        evidence: Vec<String>,
    },
}

impl fmt::Display for DeviceSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable { request, evidence } => write!(
                formatter,
                "no usable device matches `{request}`: {}",
                evidence.join("; ")
            ),
        }
    }
}

impl From<SelectionError> for DeviceSelectionError {
    fn from(error: SelectionError) -> Self {
        match error {
            SelectionError::NoCandidate { request, evidence, .. } => {
                Self::Unavailable { request, evidence }
            }
        }
    }
}

impl From<SeismicResolveError> for DeviceSelectionError {
    fn from(error: SeismicResolveError) -> Self {
        match error {
            SeismicResolveError::Missing(selector) => Self::Unavailable {
                request: DeviceRequest::Selector(selector),
                evidence: vec![format!("device {selector} is not present")],
            },
        }
    }
}

/// A memory claim that could not be satisfied after the release order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InsufficientMemory {
    pub required: u64,
    pub available: u64,
}

impl fmt::Display for InsufficientMemory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} bytes are required but {} are available above the planning reserve",
            self.required, self.available
        )
    }
}

/// `EngineConfiguration::resolve`: device-free host resolution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolveError {
    /// A configuration control is malformed or outside the model's domain
    /// (served context, method, proposal width, service limits).
    InvalidConfiguration { reason: String },
    Artifact(ArtifactError),
    Unsupported(UnsupportedModel),
}

impl fmt::Display for ResolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { reason } => {
                write!(formatter, "invalid engine configuration: {reason}")
            }
            Self::Artifact(error) => error.fmt(formatter),
            Self::Unsupported(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ResolveError {}

/// `ResolvedEngineConfiguration::preview`: read-only planning.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreviewError {
    Unsupported(UnsupportedModel),
    Device(DeviceSelectionError),
    /// A required memory observation failed (Blind). Retryable.
    MemoryObservationUnavailable { reason: String },
    Internal { reason: String },
}

impl fmt::Display for PreviewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(error) => error.fmt(formatter),
            Self::Device(error) => error.fmt(formatter),
            Self::MemoryObservationUnavailable { reason } => {
                write!(formatter, "memory observation unavailable: {reason}")
            }
            Self::Internal { reason } => write!(formatter, "internal preview failure: {reason}"),
        }
    }
}

impl std::error::Error for PreviewError {}

/// Constructing a ready engine: worker startup, device resolution, planning,
/// program preparation, allocation and weight import.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadError {
    Artifact(ArtifactError),
    Unsupported(UnsupportedModel),
    Device(DeviceSelectionError),
    /// A startup claim could not be satisfied in `store`, one of the stores
    /// the executor's placement charges. Retryable once memory frees.
    InsufficientMemory {
        purpose: String,
        store: StoreIdentity,
        memory: InsufficientMemory,
    },
    /// `store` was in the Reclaim band when the load claimed from it or
    /// while it imported weights: its headroom was at or below the planning
    /// reserve, or its host was in distress. No byte count describes it.
    /// Retryable once the store recovers.
    MemoryPressure {
        purpose: String,
        store: StoreIdentity,
    },
    MemoryObservationUnavailable { reason: String },
    DeviceLost { reason: String },
    /// The worker ended, or broke the protocol, before it was ready.
    WorkerLost { reason: String },
    Internal { reason: String },
}

impl LoadError {
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            Self::InsufficientMemory { .. }
                | Self::MemoryPressure { .. }
                | Self::MemoryObservationUnavailable { .. }
                | Self::DeviceLost { .. }
                | Self::WorkerLost { .. }
        )
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(error) => error.fmt(formatter),
            Self::Unsupported(error) => error.fmt(formatter),
            Self::Device(error) => error.fmt(formatter),
            Self::InsufficientMemory {
                purpose,
                store,
                memory,
            } => write!(formatter, "{purpose} in {store}: {memory}"),
            Self::MemoryPressure { purpose, store } => {
                write!(formatter, "{purpose} in {store}: memory pressure")
            }
            Self::MemoryObservationUnavailable { reason } => {
                write!(formatter, "memory observation unavailable: {reason}")
            }
            Self::DeviceLost { reason } => write!(formatter, "device lost: {reason}"),
            Self::WorkerLost { reason } => write!(formatter, "worker lost during load: {reason}"),
            Self::Internal { reason } => write!(formatter, "internal load failure: {reason}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<ResolveError> for LoadError {
    fn from(error: ResolveError) -> Self {
        match error {
            ResolveError::InvalidConfiguration { reason } => Self::Internal { reason },
            ResolveError::Artifact(error) => Self::Artifact(error),
            ResolveError::Unsupported(error) => Self::Unsupported(error),
        }
    }
}

/// One request's outcome other than normal completion. Exactly one terminal
/// outcome reaches the host per admitted request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestError {
    /// Malformed or unsupported request controls.
    InvalidRequest { reason: String },
    /// The prepared input exceeds the loaded context.
    ContextLengthExceeded { required: u64, limit: u64 },
    /// A claim could not be satisfied after the release order. Retryable.
    InsufficientMemory(InsufficientMemory),
    /// A required memory observation failed (Blind). Retryable.
    MemoryObservationUnavailable { reason: String },
    /// The engine is in the Reclaim band and pauses admission
    /// (`memory_pressure`). Retryable.
    MemoryReclaim,
    /// The model is no longer loaded.
    ModelUnloaded { cause: UnloadCause },
    /// The device cannot provision another request's numerical state. Retryable.
    Overloaded,
    /// The device or driver failed during execution.
    DeviceLost { reason: String },
    /// Weights the request's steps read from the model file during
    /// execution (a streamed table's rows) could not be read. Other
    /// requests continue. Retryable.
    WeightReadFailed { reason: String },
    /// The caller cancelled.
    Cancelled,
    /// The worker ended before the request's terminal outcome.
    WorkerLost { reason: String },
    /// An engine invariant failed.
    Internal { reason: String },
}

impl RequestError {
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            Self::InsufficientMemory(_)
                | Self::MemoryObservationUnavailable { .. }
                | Self::MemoryReclaim
                | Self::ModelUnloaded {
                    cause: UnloadCause::MemoryPressure
                }
                | Self::Overloaded
                | Self::DeviceLost { .. }
                | Self::WeightReadFailed { .. }
                | Self::WorkerLost { .. }
        )
    }

    pub(crate) fn internal(reason: impl Into<String>) -> Self {
        Self::Internal {
            reason: reason.into(),
        }
    }
}

impl fmt::Display for RequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest { reason } => write!(formatter, "invalid request: {reason}"),
            Self::ContextLengthExceeded { required, limit } => write!(
                formatter,
                "the input requires {required} tokens but the loaded context is {limit}"
            ),
            Self::InsufficientMemory(memory) => memory.fmt(formatter),
            Self::MemoryObservationUnavailable { reason } => {
                write!(formatter, "memory observation unavailable: {reason}")
            }
            Self::MemoryReclaim => formatter.write_str(
                "memory headroom is at or below the planning reserve; reclaiming memory",
            ),
            Self::ModelUnloaded { cause } => write!(formatter, "model unloaded: {cause}"),
            Self::Overloaded => formatter.write_str("the engine's request capacity is exhausted"),
            Self::DeviceLost { reason } => write!(formatter, "device lost: {reason}"),
            Self::WeightReadFailed { reason } => {
                write!(formatter, "model weights could not be read: {reason}")
            }
            Self::Cancelled => formatter.write_str("request cancelled"),
            Self::WorkerLost { reason } => write!(formatter, "worker lost: {reason}"),
            Self::Internal { reason } => write!(formatter, "internal engine failure: {reason}"),
        }
    }
}

impl std::error::Error for RequestError {}

impl From<magnitude_scheduler::owner::AdmissionError> for RequestError {
    fn from(error: magnitude_scheduler::owner::AdmissionError) -> Self {
        use magnitude_scheduler::{owner::AdmissionError, publication::ModelUnloadCause};
        match error {
            AdmissionError::MemoryObservationUnavailable(reason) => {
                Self::MemoryObservationUnavailable { reason }
            }
            AdmissionError::MemoryReclaim => Self::MemoryReclaim,
            AdmissionError::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure,
            } => Self::ModelUnloaded {
                cause: UnloadCause::MemoryPressure,
            },
            AdmissionError::Refused(error) => error.into(),
        }
    }
}

impl From<magnitude_scheduler::publication::RequestError> for RequestError {
    fn from(error: magnitude_scheduler::publication::RequestError) -> Self {
        use magnitude_executor::DeviceError;
        use magnitude_scheduler::publication::{
            CapacityResource, ModelUnloadCause, RequestError as Scheduled,
        };
        match error {
            Scheduled::Input(reason) => Self::InvalidRequest { reason },
            Scheduled::State(magnitude_state::Error::Capacity {
                required,
                available_bytes,
            }) => Self::InsufficientMemory(InsufficientMemory {
                required,
                available: available_bytes,
            }),
            Scheduled::State(error) => Self::internal(error.to_string()),
            Scheduled::Capacity(capacity) => match capacity.resource {
                CapacityResource::Execution(_) => Self::InsufficientMemory(InsufficientMemory {
                    required: capacity.required,
                    available: capacity.available,
                }),
                CapacityResource::LiveSequence
                | CapacityResource::SubmittedSuccessor
                | CapacityResource::BranchCheckpoint
                | CapacityResource::RetainedPrefix
                | CapacityResource::RetainedMethodFeature
                | CapacityResource::RetainedMediaFeature => Self::Overloaded,
            },
            Scheduled::Device(DeviceError::Lost(reason)) => Self::DeviceLost { reason },
            Scheduled::Device(error) => Self::DeviceLost {
                reason: error.to_string(),
            },
            Scheduled::Invariant(error) => Self::internal(error.to_string()),
            Scheduled::TableRead(error) => Self::WeightReadFailed {
                reason: error.detail,
            },
            Scheduled::WorkerClosed => Self::WorkerLost {
                reason: "the execution owner closed the request's stream".into(),
            },
            Scheduled::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure,
            } => Self::ModelUnloaded {
                cause: UnloadCause::MemoryPressure,
            },
        }
    }
}

/// Why an execution owner stopped: the classified failure it terminated
/// every live request with. A device failure keeps its meaning (as it does
/// for those requests); anything else is an engine invariant.
impl From<&magnitude_scheduler::publication::RequestError> for UnloadCause {
    fn from(error: &magnitude_scheduler::publication::RequestError) -> Self {
        use magnitude_executor::DeviceError;
        use magnitude_scheduler::publication::RequestError as Scheduled;
        match error {
            Scheduled::Device(DeviceError::Lost(reason)) => Self::DeviceLost {
                reason: reason.clone(),
            },
            Scheduled::Device(error) => Self::DeviceLost {
                reason: error.to_string(),
            },
            other => Self::Internal {
                reason: other.to_string(),
            },
        }
    }
}

/// An execution owner that panicked stopped without a classified cause.
impl From<magnitude_scheduler::worker::Panicked> for UnloadCause {
    fn from(_: magnitude_scheduler::worker::Panicked) -> Self {
        Self::Internal {
            reason: "execution owner panicked".into(),
        }
    }
}

/// Planner rejections of a recognized, validated definition are properties
/// of the material on this execution path: a form, operator, scale or
/// layout the backend's planner does not execute. The others are arithmetic
/// or resource failures of the engine itself.
pub(crate) fn classify_plan(error: PlanError, backend: seismic::BackendName) -> PlanOutcome {
    let unsupported = |detail: &dyn fmt::Display| {
        PlanOutcome::Unsupported(UnsupportedModel::reason(
            UnsupportedReason::Backend {
                backend: backend.as_str().to_owned(),
            },
            format!("unsupported on the {} backend: {detail}", backend.as_str()),
        ))
    };
    match error {
        PlanError::Unsupported(feature) => unsupported(&feature),
        error @ (PlanError::InvalidDefinition(_)
        | PlanError::Topology(_)
        | PlanError::UnsupportedOperator { .. }
        | PlanError::Deferred(_)
        | PlanError::UnportedScale(_)) => unsupported(&error),
        error @ (PlanError::Arithmetic(_)
        | PlanError::ResourcePlanning(_)
        | PlanError::Resource(_)) => PlanOutcome::Internal(error.to_string()),
    }
}

/// A graph construction failure: a kernel call outside its kernel's domain
/// is a property of the model on this backend; any other failure is an
/// engine defect.
pub(crate) fn classify_graph(error: GraphError) -> PlanOutcome {
    match error {
        GraphError::KernelDomain(violation) => PlanOutcome::Unsupported(kernel_domain(
            violation.backend.as_str(),
            violation.entry,
            violation.statics,
        )),
        GraphError::Invalid(reason) => PlanOutcome::Internal(reason),
    }
}

/// A program preparation failure: the load-side twin of [`classify_graph`],
/// classifying a kernel call outside its kernel's domain as the same value.
pub(crate) fn classify_catalog(error: CatalogError) -> PlanOutcome {
    match error.failure {
        CatalogFailure::KernelDomain { entry, statics } => {
            PlanOutcome::Unsupported(kernel_domain(error.backend.as_str(), entry, statics))
        }
        // A load that stops unprepared is taken as its own outcome before
        // any failure is classified; one that reaches here was not asked for.
        CatalogFailure::Preparation { .. }
        | CatalogFailure::Tuning { .. }
        | CatalogFailure::Qualification { .. }
        | CatalogFailure::Unprepared { .. } => PlanOutcome::Internal(error.to_string()),
    }
}

/// A kernel call of the model's program with statics outside the kernel's
/// domain on the backend: no configuration executes them.
fn kernel_domain(backend: &str, entry: &str, statics: Vec<(String, u64)>) -> UnsupportedModel {
    let reason = UnsupportedReason::KernelDomain {
        backend: backend.to_owned(),
        entry: entry.to_owned(),
        statics,
    };
    let detail = format!("no admissible configuration for {reason}");
    UnsupportedModel::reason(reason, detail)
}

pub(crate) enum PlanOutcome {
    Unsupported(UnsupportedModel),
    Internal(String),
}

/// Classify a platform failure during device selection or opening.
pub(crate) enum PlatformOutcome {
    Device(DeviceSelectionError),
    MemoryObservationUnavailable(String),
    Unsupported(UnsupportedModel),
    Internal(String),
}

pub(crate) fn classify_platform(error: PlatformError) -> PlatformOutcome {
    match error {
        PlatformError::Selection(error) => PlatformOutcome::Device(error.into()),
        PlatformError::Resolve(error) => PlatformOutcome::Device(error.into()),
        PlatformError::Memory(MemoryPolicyError::Observation(error)) => {
            PlatformOutcome::MemoryObservationUnavailable(error.to_string())
        }
        PlatformError::Policy { path, outcome } => {
            PlatformOutcome::Unsupported(UnsupportedModel::reason(
                UnsupportedReason::Backend {
                    backend: path.to_string(),
                },
                format!("unsupported on the {path} backend: {outcome}"),
            ))
        }
        error @ (PlatformError::Memory(_)
        | PlatformError::Open(_)
        | PlatformError::Qualification(_)) => PlatformOutcome::Internal(error.to_string()),
    }
}

impl From<PlatformOutcome> for PreviewError {
    fn from(outcome: PlatformOutcome) -> Self {
        match outcome {
            PlatformOutcome::Device(error) => Self::Device(error),
            PlatformOutcome::MemoryObservationUnavailable(reason) => {
                Self::MemoryObservationUnavailable { reason }
            }
            PlatformOutcome::Unsupported(error) => Self::Unsupported(error),
            PlatformOutcome::Internal(reason) => Self::Internal { reason },
        }
    }
}

impl From<PlatformOutcome> for LoadError {
    fn from(outcome: PlatformOutcome) -> Self {
        match outcome {
            PlatformOutcome::Device(error) => Self::Device(error),
            PlatformOutcome::MemoryObservationUnavailable(reason) => {
                Self::MemoryObservationUnavailable { reason }
            }
            PlatformOutcome::Unsupported(error) => Self::Unsupported(error),
            PlatformOutcome::Internal(reason) => Self::Internal { reason },
        }
    }
}

impl From<PlanOutcome> for PreviewError {
    fn from(outcome: PlanOutcome) -> Self {
        match outcome {
            PlanOutcome::Unsupported(error) => Self::Unsupported(error),
            PlanOutcome::Internal(reason) => Self::Internal { reason },
        }
    }
}

impl From<PlanOutcome> for LoadError {
    fn from(outcome: PlanOutcome) -> Self {
        match outcome {
            PlanOutcome::Unsupported(error) => Self::Unsupported(error),
            PlanOutcome::Internal(reason) => Self::Internal { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_executor::{DeviceError, InvariantError};
    use magnitude_scheduler::{owner::AdmissionError, publication::RequestError as Scheduled};

    /// A failed execution owner keeps its classified failure: a device loss
    /// behind it unloads the model and refuses later admissions as device
    /// loss, not as an engine invariant.
    #[test]
    fn owner_failures_keep_device_loss() {
        let lost = Scheduled::Device(DeviceError::Lost("command buffer error".into()));
        assert_eq!(
            UnloadCause::from(&lost),
            UnloadCause::DeviceLost {
                reason: "command buffer error".into()
            }
        );
        assert_eq!(
            RequestError::from(AdmissionError::Refused(lost)),
            RequestError::DeviceLost {
                reason: "command buffer error".into()
            }
        );
        let execution = Scheduled::Device(DeviceError::Execution("fault".into()));
        assert!(matches!(
            UnloadCause::from(&execution),
            UnloadCause::DeviceLost { .. }
        ));
        let invariant = Scheduled::Invariant(InvariantError {
            context: "service worker",
            detail: "broken".into(),
        });
        assert_eq!(
            UnloadCause::from(&invariant),
            UnloadCause::Internal {
                reason: "service worker: broken".into()
            }
        );
    }
}
