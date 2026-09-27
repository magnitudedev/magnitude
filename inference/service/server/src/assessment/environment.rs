//! The assessment environment: the selected execution device, its measurement basis and the
//! engine configuration every model is assessed with, and the identity that keys every cached
//! assessment made in it.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use magnitude_engine::assessment::{
    AssessmentEnvironment as EngineEnvironment, AssessmentSetup, ModelAssessmentError,
};
use magnitude_engine::options::{ModelPolicy, standard_service_limits};
use magnitude_executor::assessment::{MeasurementBasis, basis_json};
use magnitude_executor::platform::{DeviceRequest, MemoryReserves};
use magnitude_service_contracts::models::AssessmentEnvironmentId;
use seismic::{DeviceCatalog, DeviceMemory, DeviceSelector, DeviceTopology, HostMemoryStatus};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::measurement::{BasisSource, MeasurementJob, MeasurementJobError};

/// Requests one conversation per batch: the engine's standard batch width. Loads use the same
/// limits, so an assessed plan is the plan a load prepares.

/// The engine configuration the service loads and assesses every model with.
pub fn serving_policy() -> ModelPolicy {
    ModelPolicy::default()
}

pub struct AssessmentEnvironment {
    pub id: AssessmentEnvironmentId,
    pub engine: EngineEnvironment,
}

#[derive(Debug)]
pub enum EnvironmentError {
    Measurement(MeasurementJobError),
    Environment(ModelAssessmentError),
    Task(String),
}

impl fmt::Display for EnvironmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Measurement(error) => error.fmt(formatter),
            Self::Environment(error) => write!(formatter, "assessment environment: {error}"),
            Self::Task(error) => write!(formatter, "assessment environment task: {error}"),
        }
    }
}

impl std::error::Error for EnvironmentError {}

impl AssessmentEnvironment {
    /// Select once, before the measurement and model-preparation branches diverge.
    pub async fn select(
        catalog: Arc<DeviceCatalog>,
    ) -> Result<Arc<AssessmentSetup>, EnvironmentError> {
        crate::spawn_blocking_traced(move || {
            AssessmentSetup::discover(
                &catalog,
                DeviceRequest::Automatic,
                MemoryReserves::standard(),
                serving_policy(),
                standard_service_limits(),
            )
            .map(Arc::new)
            .map_err(EnvironmentError::Environment)
        })
        .await
        .map_err(|error| EnvironmentError::Task(error.to_string()))?
    }

    /// Add the measured basis to the one selected setup and derive its cache identity.
    pub async fn establish(
        setup: Arc<AssessmentSetup>,
        measurement: &MeasurementJob,
    ) -> Result<Self, EnvironmentError> {
        let device = setup.selected.info.selector;
        let established = measurement
            .establish(device)
            .await
            .map_err(EnvironmentError::Measurement)?;
        log_basis(
            device,
            established.source,
            established.elapsed,
            &established.basis,
        );
        crate::spawn_blocking_traced(move || {
            let engine = setup
                .with_basis(established.basis)
                .map_err(EnvironmentError::Environment)?;
            Ok(Self {
                id: environment_id(&engine),
                engine,
            })
        })
        .await
        .map_err(|error| EnvironmentError::Task(error.to_string()))?
    }
}

fn log_basis(
    device: DeviceSelector,
    source: BasisSource,
    elapsed: Duration,
    basis: &MeasurementBasis,
) {
    let (source, measured_seconds) = match source {
        BasisSource::Cached => ("cached", None),
        BasisSource::Measured { seconds } => ("measured", Some(seconds)),
    };
    tracing::info!(
        device = %device,
        basis.source = source,
        basis.measured_seconds = measured_seconds,
        basis.classes = basis.classes.len(),
        job.seconds = elapsed.as_secs_f64(),
        "assessment measurement basis established"
    );
}

/// The identity of everything an assessment result depends on besides the model: engine build
/// (with its kernel bundle), backend and toolchain, device, the stable topology and process
/// limits that bound fit capacity, the measurement basis itself, the reserve policy and the
/// serving configuration. The engine build covers everything the engine computes, including model
/// families and the fit workload. Live free memory is not an input.
fn environment_id(engine: &EngineEnvironment) -> AssessmentEnvironmentId {
    let material = json!({
        "engine_build": engine.basis.identity.engine_build,
        "backend": engine.basis.identity.backend,
        "toolchain": engine.basis.identity.device,
        "measurement_protocol": engine.basis.identity.protocol_version,
        "device": engine.setup.selected.info.selector,
        "topology": normalized_topology(&engine.setup.topology),
        "process_limits": process_limits(&engine.setup.host),
        "basis": format!("{:x}", Sha256::digest(basis_json(&engine.basis).to_string().as_bytes())),
        "reserves": format!("{:?}", engine.setup.reserves),
        "policy": format!("{:?}", engine.setup.policy),
        "service": format!("{:?}", engine.setup.service),
    });
    AssessmentEnvironmentId(format!(
        "environment_{:x}",
        Sha256::digest(material.to_string().as_bytes())
    ))
}

/// Devices and memory pools without the per-process revision: identity, kind, backend,
/// availability, memory relationships and capacities.
fn normalized_topology(topology: &DeviceTopology) -> serde_json::Value {
    let pools = topology.pools();
    let pool_index = |id| pools.iter().position(|pool| pool.id == id);
    json!({
        "devices": topology.devices().iter().map(|device| json!({
            "selector": device.selector,
            "name": device.name,
            "kind": format!("{:?}", device.kind),
            "backend": device.backend.as_str(),
            "availability": format!("{:?}", device.availability),
            "memory": match &device.memory {
                DeviceMemory::Established(memory) => json!({
                    "allocation_pool": pool_index(memory.allocation_pool),
                    "host_pool": pool_index(memory.host_pool),
                    "max_allocation_bytes": memory.max_allocation_bytes,
                }),
                DeviceMemory::Unsupported { reason } => json!({ "unsupported": reason }),
            },
            "recommended_working_set_bytes": device.recommended_working_set_bytes(),
        })).collect::<Vec<_>>(),
        "pools": pools.iter().map(|pool| json!({
            "kind": format!("{:?}", pool.kind),
            "capacity_bytes": pool.capacity_bytes,
            "basis": format!("{:?}", pool.basis),
        })).collect::<Vec<_>>(),
    })
}

fn process_limits(host: &HostMemoryStatus) -> serde_json::Value {
    json!({
        "limits": host.limits.iter().map(|limit| json!({
            "kind": format!("{:?}", limit.kind),
            "limit_bytes": limit.limit_bytes,
        })).collect::<Vec<_>>(),
        "visibility": format!("{:?}", host.limit_visibility),
    })
}
