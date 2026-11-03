//! Serving over the engine library. A model's host artifacts are resolved
//! once, and what the model serves chooses its binding: a chat binding
//! generates through the host chat session, and a decision binding decides
//! through the model's decision description and a readout plan. Model
//! sources (the standalone engine, the service) resolve their models into
//! this and supply only how a resolution is acquired.
use crate::error::{ModelUnavailable, ServingError};
use crate::source::{
    ChatInvocation, ChatTarget, DecisionInvocation, DecisionTarget, GenerationEvent,
    GenerationStream, HostChat, LoadProgress, ResolvedModel, ServedTarget,
};
use futures_util::future::BoxFuture;
use magnitude_chat::{ChatInput, GenerationRequest};
use magnitude_decision::plan::ReadoutPlanError;
use magnitude_decision::{DecisionAnswers, DecisionRequest, Decider};
use magnitude_engine::chat::{
    self, AppliedTemplate, ModelProperties, SessionError, SessionEvent, SessionLimits,
};
use magnitude_engine::error::RequestError;
use magnitude_engine::host::HostArtifacts;
use magnitude_engine::invocation::{Invocation, InvocationError};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Event buffering between a session and its protocol framing.
const EVENT_CAPACITY: usize = 32;

impl From<SessionError> for ServingError {
    fn from(error: SessionError) -> Self {
        match error {
            SessionError::Chat(error) => Self::Chat(error),
            SessionError::Request(error) => Self::Request(error),
        }
    }
}

impl From<InvocationError> for ServingError {
    fn from(error: InvocationError) -> Self {
        match error {
            InvocationError::UnknownModel { .. } | InvocationError::NotInstalled { .. } => {
                Self::Model(ModelUnavailable::NotFound(error.to_string()))
            }
            InvocationError::Load(error) => Self::Load(error),
            InvocationError::Unavailable(error) => Self::Request(error),
        }
    }
}

/// How a source acquires the one model it resolved: the invocation bound to
/// exactly that resolution, loading the model if the source manages
/// residency.
pub type Acquire<'a> = Box<
    dyn FnOnce(Option<LoadProgress>) -> BoxFuture<'a, Result<Invocation, ServingError>>
        + Send
        + 'a,
>;

/// The resolution of a model whose host artifacts are `host`: a chat model
/// generates within `limits`, a decision model decides through its type.
pub fn resolved<'a>(
    host: Arc<HostArtifacts>,
    limits: SessionLimits,
    acquire: Acquire<'a>,
) -> ResolvedModel<'a> {
    let target = match host.decider() {
        Some(decider) => ServedTarget::Decisions(Box::new(DecisionModel {
            decider: Arc::clone(decider),
            acquire,
        })),
        None => ServedTarget::Chat(Box::new(ChatModel { limits, acquire })),
    };
    ResolvedModel {
        host: Arc::new(EngineHost { host }),
        target,
    }
}

struct ChatModel<'a> {
    limits: SessionLimits,
    acquire: Acquire<'a>,
}

impl<'a> ChatTarget<'a> for ChatModel<'a> {
    fn invoke(
        self: Box<Self>,
        progress: Option<LoadProgress>,
    ) -> BoxFuture<'a, Result<Box<dyn ChatInvocation>, ServingError>> {
        let Self { limits, acquire } = *self;
        Box::pin(async move {
            let invocation = acquire(progress).await?;
            Ok(Box::new(ChatBinding { invocation, limits }) as Box<dyn ChatInvocation>)
        })
    }
}

struct DecisionModel<'a> {
    decider: Arc<Decider>,
    acquire: Acquire<'a>,
}

impl<'a> DecisionTarget<'a> for DecisionModel<'a> {
    fn invoke(self: Box<Self>) -> BoxFuture<'a, Result<Box<dyn DecisionInvocation>, ServingError>> {
        let Self { decider, acquire } = *self;
        Box::pin(async move {
            let invocation = acquire(None).await?;
            Ok(Box::new(DecisionBinding {
                decider,
                invocation,
            }) as Box<dyn DecisionInvocation>)
        })
    }
}

/// A chat model's engine invocation: its host artifacts, worker client and
/// release guard. The guard lives until the generation's session ends.
struct ChatBinding {
    invocation: Invocation,
    limits: SessionLimits,
}

impl ChatInvocation for ChatBinding {
    fn generate(self: Box<Self>, request: GenerationRequest) -> GenerationStream {
        let Invocation {
            host,
            engine,
            release,
        } = self.invocation;
        let (session_sender, mut session_events) = mpsc::channel(EVENT_CAPACITY);
        let (sender, receiver) = mpsc::channel(EVENT_CAPACITY);
        tokio::spawn(chat::generate(host, engine, request, self.limits, session_sender));
        tokio::spawn(async move {
            // The release guard is held until the session publishes its
            // terminal event or the consumer goes away.
            let _release = release;
            loop {
                let event = tokio::select! {
                    event = session_events.recv() => event,
                    () = sender.closed() => return,
                };
                let Some(event) = event else { return };
                let event = match event {
                    SessionEvent::Progress(progress) => GenerationEvent::Progress(progress),
                    SessionEvent::Admitted { prompt_tokens } => {
                        GenerationEvent::Admitted { prompt_tokens }
                    }
                    SessionEvent::Output { event, snapshot } => {
                        GenerationEvent::Output { event, snapshot }
                    }
                    SessionEvent::Completed(completion) => GenerationEvent::Completed(completion),
                    SessionEvent::Failed(error) => GenerationEvent::Failed(error.into()),
                };
                if sender.send(event).await.is_err() {
                    return;
                }
            }
        });
        GenerationStream::new(receiver)
    }
}

/// A decision model's engine invocation and its decider.
struct DecisionBinding {
    decider: Arc<Decider>,
    invocation: Invocation,
}

impl DecisionInvocation for DecisionBinding {
    fn decide(
        self: Box<Self>,
        request: DecisionRequest,
    ) -> BoxFuture<'static, Result<DecisionAnswers, ServingError>> {
        let Self {
            decider,
            invocation: Invocation {
                host,
                engine,
                release,
            },
        } = *self;
        Box::pin(async move {
            // The release guard is held until the answers are read.
            let _release = release;
            // Rendering, tokenizing and image preparation run off the
            // runtime, as a chat session's preparation does.
            let (prepared, plan) = blocking("decision preparation", move || {
                let prepared = decider.prepare(&request).map_err(ServingError::Decision)?;
                let limit = host.definition().decoder.context_limit;
                let plan = prepared
                    .readout_plan(&request, |tokens, images| {
                        let input = host.prepare_input(tokens, images)?;
                        let required = input.tokens().len() as u64;
                        if required > limit {
                            return Err(RequestError::ContextLengthExceeded { required, limit });
                        }
                        Ok(input)
                    })
                    .map_err(|error| match error {
                        ReadoutPlanError::Input(error) => ServingError::Request(error),
                        ReadoutPlanError::Plan(error) => ServingError::Internal(error.to_string()),
                    })?;
                Ok((prepared, plan))
            })
            .await?;
            let values = engine.readout(plan).await?;
            blocking("decision reading", move || {
                prepared.answers(&values).map_err(ServingError::Internal)
            })
            .await
        })
    }
}

/// Run host-side request work on the blocking pool.
async fn blocking<T: Send + 'static>(
    work: &'static str,
    run: impl FnOnce() -> Result<T, ServingError> + Send + 'static,
) -> Result<T, ServingError> {
    tokio::task::spawn_blocking(run)
        .await
        .map_err(|error| ServingError::Internal(format!("{work} failed: {error}")))?
}

/// A model's host artifacts serving the host-only operations.
struct EngineHost {
    host: Arc<HostArtifacts>,
}

impl HostChat for EngineHost {
    fn count(&self, input: &ChatInput) -> Result<u64, ServingError> {
        Ok(chat::count_tokens(&self.host, input)?)
    }

    fn apply_template(&self, input: &ChatInput) -> Result<AppliedTemplate, ServingError> {
        Ok(chat::apply_template(&self.host, input)?)
    }

    fn properties(&self) -> Result<ModelProperties, ServingError> {
        Ok(chat::model_properties(&self.host)?)
    }
}
