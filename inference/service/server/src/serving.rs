//! The service's models for the protocol handlers (integration spec §8.2, §8.5). A request
//! resolves the canonical model once, from the shared resolved configuration, without leasing or
//! loading it; a bound request ensures exactly that resolution's instance and holds the instance
//! lease until it ends.

use std::sync::Arc;

use futures_util::future::BoxFuture;
use magnitude_engine::chat::SessionLimits;
use magnitude_engine::invocation::{Invocation, ReleaseGuard};
use magnitude_service_contracts::InventoryError;
use magnitude_service_contracts::models::ModelLoadStage;
use magnitude_serving::engine::{Acquire, resolved};
use magnitude_serving::{
    ModelLoadProgress, ModelLoadStage as ProtocolLoadStage, ModelUnavailable, ResolvedModel,
    ServedModels, ServingError,
};

use crate::residency::LoadingObserver;
use crate::residency::controller::{InferenceTarget, ModelInstances};

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
    /// From the model's resolved configuration, which a request then acquires exactly: the
    /// instance lease is held until the bound request ends.
    fn resolve(&self, model: &str) -> BoxFuture<'_, Result<ResolvedModel<'_>, ServingError>> {
        let model = model.to_owned();
        Box::pin(async move {
            let InferenceTarget { target, host } = self
                .instances
                .resolve_for_inference(&model)
                .await
                .map_err(unavailable)?;
            let acquire: Acquire<'_> = Box::new(move |progress| {
                Box::pin(async move {
                    let observer = progress.map(|progress| -> LoadingObserver {
                        Arc::new(move |stage, fraction, details| {
                            progress(ModelLoadProgress {
                                stage: protocol_stage(stage),
                                fraction,
                                details,
                            })
                        })
                    });
                    let lease = self
                        .instances
                        .acquire_for_inference(target, observer)
                        .await
                        .map_err(unavailable)?;
                    let (resident, release) = lease.into_parts();
                    Ok(Invocation {
                        host: resident.host,
                        engine: resident.client,
                        release: ReleaseGuard::new(release),
                    })
                })
            });
            Ok(resolved(host, SESSION_LIMITS, acquire))
        })
    }
}

/// An instance's load stage as the protocols report it: the same stage under the same name.
fn protocol_stage(stage: ModelLoadStage) -> ProtocolLoadStage {
    match stage {
        ModelLoadStage::Queued => ProtocolLoadStage::Queued,
        ModelLoadStage::Preparing => ProtocolLoadStage::Preparing,
        ModelLoadStage::Optimizing => ProtocolLoadStage::Optimizing,
        ModelLoadStage::LoadingWeights => ProtocolLoadStage::LoadingWeights,
        ModelLoadStage::Finalizing => ProtocolLoadStage::Finalizing,
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
        InventoryError::DoesNotFit(message) => ModelUnavailable::DoesNotFit(message),
        InventoryError::ModelUnsupported { message, .. } => ModelUnavailable::Operation {
            code: "model_unsupported".to_owned(),
            message,
            retryable: false,
        },
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
                code: "memory_shortage".to_owned(),
                message: "short".to_owned(),
                retryable: true,
            }),
            ServingError::Model(ModelUnavailable::Operation {
                code: "memory_shortage".to_owned(),
                message: "short".to_owned(),
                retryable: true,
            })
        );
        assert_eq!(
            unavailable(InventoryError::NotFound("absent".to_owned())),
            ServingError::Model(ModelUnavailable::NotFound("absent".to_owned()))
        );
    }

    #[test]
    fn protocol_progress_names_each_load_stage_as_instance_status_does() {
        for stage in [
            ModelLoadStage::Queued,
            ModelLoadStage::Preparing,
            ModelLoadStage::Optimizing,
            ModelLoadStage::LoadingWeights,
            ModelLoadStage::Finalizing,
        ] {
            assert_eq!(
                serde_json::to_value(stage).unwrap(),
                serde_json::to_value(protocol_stage(stage)).unwrap()
            );
        }
    }
}
