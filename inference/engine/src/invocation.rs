//! How protocol handlers reach a served model (integration spec §8.2).
//!
//! Handlers are router-independent and never load models or choose catalog
//! entries: they ask an [`InvocationSource`] for the model a request names and
//! receive its host artifacts (chat semantics), a handle to its numerical
//! worker and a guard that keeps the binding alive for the request's lifetime.
//! The CLI's source returns its single in-process engine; the service's source
//! resolves the canonical model, ensures the instance and returns its lease as
//! the guard.

use crate::error::{LoadError, RequestError};
use crate::host::HostArtifacts;
use crate::worker::EngineClient;
use serde::{Deserialize, Serialize};
use std::{fmt, future::Future, sync::Arc};

/// Released when the invocation (and every request it admitted) is dropped:
/// a service instance lease, or nothing for a standalone engine.
pub struct ReleaseGuard {
    _held: Option<Box<dyn Send + Sync>>,
}

impl ReleaseGuard {
    pub fn new(guard: impl Send + Sync + 'static) -> Self {
        Self {
            _held: Some(Box::new(guard)),
        }
    }

    /// A binding that owns nothing beyond the engine itself.
    pub const fn none() -> Self {
        Self { _held: None }
    }
}

/// One request's binding to exactly one loaded model generation.
pub struct Invocation {
    pub host: Arc<HostArtifacts>,
    pub engine: EngineClient,
    pub release: ReleaseGuard,
}

/// Why a model name did not produce an invocation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvocationError {
    /// No served or catalog model has this name.
    UnknownModel { model: String },
    /// The model is known but its material is not installed.
    NotInstalled { model: String },
    /// Loading the model failed.
    Load(LoadError),
    /// The model cannot take requests now (for example memory pressure
    /// gating, a stopped instance, or an exhausted engine).
    Unavailable(RequestError),
}

impl fmt::Display for InvocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownModel { model } => write!(formatter, "model `{model}` is not served"),
            Self::NotInstalled { model } => write!(formatter, "model `{model}` is not installed"),
            Self::Load(error) => error.fmt(formatter),
            Self::Unavailable(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for InvocationError {}

/// Resolves the model a request names to its invocation.
pub trait InvocationSource: Send + Sync + 'static {
    fn resolve(
        &self,
        model: &str,
    ) -> impl Future<Output = Result<Invocation, InvocationError>> + Send;
}

/// The standalone engine's source: exactly one served name, one ready engine.
pub struct SingleEngine {
    name: String,
    host: Arc<HostArtifacts>,
    engine: EngineClient,
}

impl SingleEngine {
    pub fn new(name: String, host: Arc<HostArtifacts>, engine: EngineClient) -> Self {
        Self { name, host, engine }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl InvocationSource for SingleEngine {
    fn resolve(
        &self,
        model: &str,
    ) -> impl Future<Output = Result<Invocation, InvocationError>> + Send {
        let result = if model == self.name {
            Ok(Invocation {
                host: self.host.clone(),
                engine: self.engine.clone(),
                release: ReleaseGuard::none(),
            })
        } else {
            Err(InvocationError::UnknownModel {
                model: model.to_owned(),
            })
        };
        std::future::ready(result)
    }
}
