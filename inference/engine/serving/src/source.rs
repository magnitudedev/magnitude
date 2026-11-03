//! What protocol handlers need from the models they serve (integration spec
//! §8.2). Handlers are router-independent: they name a model and receive
//! its resolution, whose host chat semantics never lease or load and whose
//! binding, of the kind the model serves, may load the model and holds its
//! release guard.
//! The CLI serves one in-process engine; the service resolves canonical models,
//! ensures instances and leases them. Tests serve scripted models.
use crate::error::ServingError;
use futures_util::future::BoxFuture;
use magnitude_chat::{
    output::{Completion, OutputEvent, Progress, TimingSnapshot},
    ChatInput, GenerationRequest,
};
use magnitude_decision::{DecisionAnswers, DecisionRequest};
use magnitude_engine::chat::{AppliedTemplate, ModelProperties};
use serde::Serialize;
use std::sync::Arc;
use utoipa::ToSchema;
use tokio::sync::mpsc;

/// Where a model load is. The service's model instance status names the same stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadStage {
    /// Waiting for memory admission.
    Queued,
    /// Resolving, planning, opening the device and preparing programs.
    Preparing,
    /// Tuning kernels for the device; only a load that tunes enters it.
    Optimizing,
    LoadingWeights,
    /// State allocation, warm-up and readiness verification.
    Finalizing,
}

/// Selected optional tuning work, not a fixed sequence of global phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum KernelTuningOperation {
    Discovery,
    Form,
    Measure,
    Promote,
    Confirm,
}

#[derive(Debug, Clone, PartialEq, Serialize, ToSchema)]
pub struct KernelTuningWork {
    pub entry: String,
    pub operation: KernelTuningOperation,
}

/// Exact stage observations; budget time is not configuration completion.
#[derive(Debug, Clone, PartialEq, Serialize, ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelLoadDetails {
    Tuning {
        elapsed_ms: u64,
        budget_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[schema(nullable = false)]
        work: Option<KernelTuningWork>,
        measured_configurations: u64,
        formed_programs: u64,
    },
    Weights {
        completed_bytes: u64,
        total_bytes: u64,
    },
}

/// A model load's stage and completed fraction in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelLoadProgress {
    pub stage: ModelLoadStage,
    pub fraction: f32,
    pub details: Option<ModelLoadDetails>,
}

/// Observes model loading.
pub type LoadProgress = Arc<dyn Fn(ModelLoadProgress) + Send + Sync>;

/// The models a router serves.
pub trait ServedModels: Send + Sync + 'static {
    /// Resolve the model `model` names once, without leasing or loading it:
    /// its host chat semantics and, by what it serves, the binding of
    /// exactly that resolution. A route of the other kind refuses it from
    /// the resolution, so a wrong-kind request never loads the model.
    fn resolve(&self, model: &str) -> BoxFuture<'_, Result<ResolvedModel<'_>, ServingError>>;
}

/// One resolution of a served model.
pub struct ResolvedModel<'a> {
    /// Operations that never lease or load: counting, template application,
    /// properties.
    pub host: Arc<dyn HostChat>,
    pub target: ServedTarget<'a>,
}

/// A resolved model's binding, by what it serves.
pub enum ServedTarget<'a> {
    Chat(Box<dyn ChatTarget<'a> + 'a>),
    Decisions(Box<dyn DecisionTarget<'a> + 'a>),
}

/// A resolved chat model.
pub trait ChatTarget<'a>: Send {
    /// Bind a generation to the resolved model, loading it if the source
    /// manages residency. The binding releases its guard when the returned
    /// invocation (and the generation it started) ends.
    fn invoke(
        self: Box<Self>,
        progress: Option<LoadProgress>,
    ) -> BoxFuture<'a, Result<Box<dyn ChatInvocation>, ServingError>>;
}

/// A resolved decision model.
pub trait DecisionTarget<'a>: Send {
    /// Bind a decision to the resolved model, loading it if the source
    /// manages residency.
    fn invoke(self: Box<Self>) -> BoxFuture<'a, Result<Box<dyn DecisionInvocation>, ServingError>>;
}

/// One bound chat model generation.
pub trait ChatInvocation: Send {
    /// Start generating. Dropping the returned stream cancels the request.
    fn generate(self: Box<Self>, request: GenerationRequest) -> GenerationStream;
}

/// One bound decision model generation.
pub trait DecisionInvocation: Send {
    /// Answer a decision request. Dropping the future cancels it.
    fn decide(
        self: Box<Self>,
        request: DecisionRequest,
    ) -> BoxFuture<'static, Result<DecisionAnswers, ServingError>>;
}

/// Chat semantics that run on the host alone, with the same rendering and
/// tokenization as generation.
pub trait HostChat: Send + Sync {
    /// Input tokens the rendered request occupies, including expanded media.
    fn count(&self, input: &ChatInput) -> Result<u64, ServingError>;
    fn apply_template(&self, input: &ChatInput) -> Result<AppliedTemplate, ServingError>;
    fn properties(&self) -> Result<ModelProperties, ServingError>;
}

/// A request's lifecycle as the engine reports it. `Admitted` precedes all
/// output; exactly one of `Completed` and `Failed` ends the stream.
#[derive(Clone, Debug)]
pub enum GenerationEvent {
    Progress(Progress),
    Admitted {
        prompt_tokens: u64,
    },
    Output {
        event: OutputEvent,
        snapshot: Option<TimingSnapshot>,
    },
    Completed(Completion),
    Failed(ServingError),
}

/// The receiving end of one generation. Dropping it cancels the request.
pub struct GenerationStream {
    events: mpsc::Receiver<GenerationEvent>,
}

impl GenerationStream {
    pub fn new(events: mpsc::Receiver<GenerationEvent>) -> Self {
        Self { events }
    }

    /// The next event; `None` only after a terminal event.
    pub async fn next(&mut self) -> Option<GenerationEvent> {
        self.events.recv().await
    }
}
