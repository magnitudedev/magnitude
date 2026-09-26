//! What protocol handlers need from the models they serve (integration spec
//! §8.2). Handlers are router-independent: they name a model and receive
//! either a generation binding (which may load the model and holds its
//! release guard) or its host chat semantics (which never lease or load).
//! The CLI serves one in-process engine; the service resolves canonical models,
//! ensures instances and leases them. Tests serve scripted models.
use crate::error::ServingError;
use futures_util::future::BoxFuture;
use magnitude_chat::{
    output::{Completion, OutputEvent, Progress, TimingSnapshot},
    ChatInput, GenerationRequest,
};
use magnitude_engine::chat::{AppliedTemplate, ModelProperties};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Observes model loading as a completed fraction in `[0, 1]`.
pub type LoadProgress = Arc<dyn Fn(f32) + Send + Sync>;

/// The models a router serves.
pub trait ServedModels: Send + Sync + 'static {
    /// Bind a generation to the model `model` names, loading it if the source
    /// manages residency. The binding is to one exact model generation and
    /// releases its guard when the returned invocation (and every generation
    /// it started) ends.
    fn invoke(
        &self,
        model: &str,
        progress: Option<LoadProgress>,
    ) -> BoxFuture<'_, Result<Box<dyn ModelInvocation>, ServingError>>;

    /// The model's host chat semantics for operations that never lease or
    /// load: counting, template application.
    fn host(&self, model: &str) -> BoxFuture<'_, Result<Arc<dyn HostChat>, ServingError>>;
}

/// One bound model generation.
pub trait ModelInvocation: Send {
    /// Start generating. Dropping the returned stream cancels the request.
    fn generate(self: Box<Self>, request: GenerationRequest) -> GenerationStream;
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
