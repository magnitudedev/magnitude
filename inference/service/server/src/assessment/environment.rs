//! The execution environment models are assessed in: its structure, established once from
//! discovery before assessment, and what the measurement job has stored of this machine.

use magnitude_engine::assessment::{AssessmentSetup, AssessmentStructure, Measurements};
use magnitude_engine::options::{ModelPolicy, standard_service_limits};
use magnitude_engine::worker::protocol::EngineBuild;
use magnitude_executor::platform::{DeviceRequest, MemoryReserves};
use magnitude_service_contracts::models::AssessmentEnvironmentId;
use seismic::Devices;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// The engine configuration the service loads and assesses every model with.
pub fn serving_policy() -> ModelPolicy {
    ModelPolicy::default()
}

/// The device the service loads, assesses and reports every model on: the engine's automatic
/// selection.
pub const SERVING_DEVICE: DeviceRequest = DeviceRequest::Automatic;

/// The structure every model is assessed on: the [`SERVING_DEVICE`]. Opens no device
/// and measures nothing.
pub fn assessment_structure(devices: &Devices) -> anyhow::Result<AssessmentStructure> {
    let structure = AssessmentStructure::discover(
        devices,
        SERVING_DEVICE,
        MemoryReserves::standard(),
        serving_policy(),
        standard_service_limits(),
    )?;
    tracing::info!(
        device = %structure.selected.info.selector(),
        device.name = %structure.selected.info.name(),
        backend = structure.selected.info.backend().as_str(),
        "assessment structure established"
    );
    Ok(structure)
}

/// One structure under what is measured of the machine, with the identity that keys every
/// cached assessment made in it.
pub struct AssessmentEnvironment {
    pub id: AssessmentEnvironmentId,
    pub setup: AssessmentSetup,
    /// What of the structure enters every identity on it, derived once with it.
    structure_identity: Arc<serde_json::Value>,
}

impl AssessmentEnvironment {
    /// The environment of `structure` under `measurements`. Fails only when the structure's
    /// host sample did: the process limits it read enter every identity on it.
    pub fn new(
        structure: Arc<AssessmentStructure>,
        measurements: Measurements,
    ) -> anyhow::Result<Self> {
        let structure_identity = Arc::new(structure_identity(&structure)?);
        Ok(Self::of(structure, structure_identity, measurements))
    }

    /// This environment's structure under `measurements`.
    pub fn measured(&self, measurements: Measurements) -> Self {
        Self::of(
            Arc::clone(&self.setup.structure),
            Arc::clone(&self.structure_identity),
            measurements,
        )
    }

    /// Only stable evidence enters identity: the structure's, and the executor costs once
    /// derived (that they are pending, or unavailable and why, otherwise), not current usage or
    /// measurement timestamps. The costs enter as the stored records they were derived from (the
    /// structure's identity includes the engine build, which derives costs from records).
    fn of(
        structure: Arc<AssessmentStructure>,
        structure_identity: Arc<serde_json::Value>,
        measurements: Measurements,
    ) -> Self {
        let measured = match &measurements {
            Measurements::Pending => json!("pending"),
            Measurements::Unavailable { reason } => json!({ "unavailable": reason }),
            Measurements::Measured { records, .. } => json!({ "records": records.as_ref() }),
        };
        let material = json!({
            "structure": &*structure_identity,
            "measurements": measured,
        });
        Self {
            id: AssessmentEnvironmentId(format!(
                "environment_{:x}",
                Sha256::digest(material.to_string().as_bytes())
            )),
            setup: AssessmentSetup {
                structure,
                measurements,
            },
            structure_identity,
        }
    }
}

/// What of `structure` enters identity: the engine build, the selected device, the topology's
/// normalized structure, the process limits and their visibility, the reserves and the
/// configuration models are assessed with.
fn structure_identity(structure: &AssessmentStructure) -> anyhow::Result<serde_json::Value> {
    let host = structure
        .status
        .sample
        .host_ram
        .as_ref()
        .map_err(Clone::clone)?;
    let process_limits = host
        .limits
        .iter()
        .map(|limit| json!({ "kind": format!("{:?}", limit.kind), "bytes": limit.limit_bytes }))
        .collect::<Vec<_>>();
    Ok(json!({
        "engine_build": EngineBuild::current().0,
        "device": structure.selected.info.selector().to_string(),
        "topology": structure.topology.normalized(),
        "process_limits": process_limits,
        "limit_visibility": format!("{:?}", host.limit_visibility),
        "reserves": structure.reserves.identity(),
        "policy": format!("{:?}", structure.policy),
        "service": format!("{:?}", structure.service),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::BackendName;

    fn cpu_structure() -> AssessmentStructure {
        AssessmentStructure::discover(
            &seismic::Devices::discover(&[]).expect("device catalog"),
            DeviceRequest::Backend(BackendName::Cpu),
            MemoryReserves::standard(),
            serving_policy(),
            standard_service_limits(),
        )
        .expect("the CPU backend is selectable")
    }

    fn environment_id(
        structure: AssessmentStructure,
        measurements: Measurements,
    ) -> AssessmentEnvironmentId {
        AssessmentEnvironment::new(Arc::new(structure), measurements)
            .unwrap()
            .id
    }

    #[test]
    fn a_different_backend_hardware_or_measurement_is_a_different_environment() {
        let id = environment_id(cpu_structure(), Measurements::Pending);
        assert_eq!(
            environment_id(cpu_structure(), Measurements::Pending),
            id,
            "the identity is stable"
        );

        // Any other executor of the machine (another backend, another device).
        let mut other = cpu_structure();
        if let Some(executor) = other
            .topology
            .executors()
            .iter()
            .find(|executor| executor.backend() != BackendName::Cpu)
            .cloned()
        {
            other.selected.info = executor;
            assert_ne!(environment_id(other, Measurements::Pending), id);
        }

        let unavailable = Measurements::Unavailable {
            reason: "no record".to_owned(),
        };
        assert_ne!(environment_id(cpu_structure(), unavailable), id);
    }
}
