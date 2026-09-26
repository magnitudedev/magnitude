//! The service's models for the protocol handlers (integration spec §8.2, §8.5). A generation
//! resolves the canonical model, ensures its instance and holds the instance lease until the
//! generation ends; host-only operations read the shared resolved configuration and never lease
//! or load.

use std::sync::Arc;

use futures_util::future::BoxFuture;
use magnitude_engine::chat::SessionLimits;
use magnitude_engine::invocation::{Invocation, ReleaseGuard};
use magnitude_service_contracts::InventoryError;
use magnitude_serving::engine::{EngineHost, EngineInvocation};
use magnitude_serving::{
    HostChat, LoadProgress, ModelInvocation, ModelUnavailable, ServedModels, ServingError,
};

use crate::residency::controller::ModelInstances;

/// Output batches buffered per request on each side of the worker, and the parsed output bound.
const SESSION_LIMITS: SessionLimits = SessionLimits {
    output_capacity: 256,
    max_output_bytes: 8 << 20,
};

/// The service's model instances as protocol handlers see them.
pub struct ServiceModels {
    instances: ModelInstances,
}

impl ServiceModels {
    pub fn new(instances: ModelInstances) -> Self {
        Self { instances }
    }
}

impl ServedModels for ServiceModels {
    fn invoke(
        &self,
        model: &str,
        progress: Option<LoadProgress>,
    ) -> BoxFuture<'_, Result<Box<dyn ModelInvocation>, ServingError>> {
        let model = model.to_owned();
        Box::pin(async move {
            let lease = self
                .instances
                .acquire_for_inference(&model, progress)
                .await
                .map_err(unavailable)?;
            let (resident, release) = lease.into_parts();
            let invocation = Invocation {
                host: resident.host,
                engine: resident.client,
                release: ReleaseGuard::new(release),
            };
            Ok(Box::new(EngineInvocation::new(invocation, SESSION_LIMITS)) as Box<dyn ModelInvocation>)
        })
    }

    fn host(&self, model: &str) -> BoxFuture<'_, Result<Arc<dyn HostChat>, ServingError>> {
        let model = model.to_owned();
        Box::pin(async move {
            let host = self.instances.host(&model).await.map_err(unavailable)?;
            Ok(Arc::new(EngineHost::new(host)) as Arc<dyn HostChat>)
        })
    }
}

/// Why the service could not bind a model, in the protocol layer's terms.
fn unavailable(error: InventoryError) -> ServingError {
    ServingError::Model(match error {
        InventoryError::InvalidId(message) | InventoryError::InvalidRequest(message) => {
            ModelUnavailable::InvalidModel(message)
        }
        InventoryError::NotFound(message) => ModelUnavailable::NotFound(message),
        InventoryError::NotReady(message) => ModelUnavailable::NotReady(message),
        InventoryError::Busy(message) | InventoryError::Loaded(message) => {
            ModelUnavailable::Busy(message)
        }
        InventoryError::Unsupported(message) => ModelUnavailable::Unsupported(message),
        InventoryError::Integrity(message) => ModelUnavailable::Integrity(message),
        InventoryError::ModelOperation { code, .. } if code == "model_instance_stopped" => {
            ModelUnavailable::InstanceStopped
        }
        InventoryError::ModelOperation {
            code,
            message,
            retryable,
        } => ModelUnavailable::Operation {
            code,
            message,
            retryable,
        },
        error @ (InventoryError::DeletionUnsafe(_)
        | InventoryError::Io(_)
        | InventoryError::Upstream(_)
        | InventoryError::ConcurrentMutation(_)
        | InventoryError::Internal(_)) => ModelUnavailable::Internal(error.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopped_instances_and_typed_operations_keep_their_meaning() {
        assert_eq!(
            unavailable(InventoryError::ModelOperation {
                code: "model_instance_stopped".to_owned(),
                message: "stopped".to_owned(),
                retryable: false,
            }),
            ServingError::Model(ModelUnavailable::InstanceStopped)
        );
        assert_eq!(
            unavailable(InventoryError::ModelOperation {
                code: "low_memory".to_owned(),
                message: "short".to_owned(),
                retryable: true,
            }),
            ServingError::Model(ModelUnavailable::Operation {
                code: "low_memory".to_owned(),
                message: "short".to_owned(),
                retryable: true,
            })
        );
        assert_eq!(
            unavailable(InventoryError::NotFound("absent".to_owned())),
            ServingError::Model(ModelUnavailable::NotFound("absent".to_owned()))
        );
    }
}
