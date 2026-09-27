//! Serving over the engine library: a bound engine invocation generates
//! through the host chat session, and a model's host artifacts answer the
//! host-only operations. Model sources (the standalone engine, the service)
//! compose these into their [`ServedModels`](crate::ServedModels).
use crate::error::{ModelUnavailable, ServingError};
use crate::source::{GenerationEvent, GenerationStream, HostChat, ModelInvocation};
use magnitude_chat::{ChatInput, GenerationRequest};
use magnitude_engine::chat::{
    self, AppliedTemplate, ModelProperties, SessionError, SessionEvent, SessionLimits,
};
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

/// One engine invocation: its host artifacts, worker client and release
/// guard. The guard lives until the generation's session ends.
pub struct EngineInvocation {
    invocation: Invocation,
    limits: SessionLimits,
}

impl EngineInvocation {
    pub fn new(invocation: Invocation, limits: SessionLimits) -> Self {
        Self { invocation, limits }
    }
}

impl ModelInvocation for EngineInvocation {
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

/// A model's host artifacts serving the host-only operations.
pub struct EngineHost {
    host: Arc<HostArtifacts>,
}

impl EngineHost {
    pub fn new(host: Arc<HostArtifacts>) -> Self {
        Self { host }
    }
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
