//! The execution environment models are assessed in: its structure, established once from
//! discovery before assessment, and what the measurement job has stored of this machine.

use magnitude_engine::assessment::{AssessmentSetup, AssessmentStructure, Measurements};
use magnitude_engine::options::{ModelPolicy, standard_service_limits};
use magnitude_engine::worker::protocol::EngineBuild;
use magnitude_executor::ExecutionPath;
use magnitude_executor::platform::{DeviceRequest, MemoryReserves, SelectionError, select};
use magnitude_service_contracts::models::AssessmentEnvironmentId;
use seismic::{Devices, ExecutorSelector};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// The engine configuration the service loads and assesses every model with.
pub fn serving_policy() -> ModelPolicy {
    ModelPolicy::default()
}

/// The request the service's device is selected by: the engine's automatic selection.
const SERVING_REQUEST: DeviceRequest = DeviceRequest::Automatic;

/// The device the service assesses, loads and reports every model on: [`SERVING_REQUEST`]
/// selected once over the service's topology. The service discovers its device inventory once
/// and never refreshes it, so the topology has one revision for the service's lifetime and the
/// serving device is decided once, at startup. Assessment, every load preview (and so every
/// load) and the hardware report read this value; none of them selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServingDevice(ExecutorSelector);

impl ServingDevice {
    /// Select the serving device over `devices`' topology.
    pub fn select(devices: &Devices) -> Result<Self, SelectionError> {
        let executor = select(&devices.topology(), ExecutionPath::Native, SERVING_REQUEST)?;
        tracing::info!(
            device = %executor.selector(),
            device.name = %executor.name(),
            backend = executor.backend().as_str(),
            "serving device selected"
        );
        Ok(Self(executor.selector()))
    }

    /// The serving device named by `selector`, for tests that serve on a chosen device.
    #[cfg(test)]
    pub(crate) fn of(selector: ExecutorSelector) -> Self {
        Self(selector)
    }

    pub fn selector(self) -> ExecutorSelector {
        self.0
    }

    /// The engine request that opens exactly this device or fails.
    pub fn request(self) -> DeviceRequest {
        DeviceRequest::Selector(self.0)
    }
}

/// The structure every model is assessed on: the [`ServingDevice`], its graphs priced as a load
/// prices them from the kernel cache at `kernels`. Opens no device and measures nothing.
pub fn assessment_structure(
    devices: &Devices,
    serving: ServingDevice,
    kernels: Option<&std::path::Path>,
) -> anyhow::Result<AssessmentStructure> {
    let structure = AssessmentStructure::discover(
        devices,
        serving.request(),
        kernels,
        MemoryReserves::of_machine(kernels.zip(Some(&*devices.topology()))),
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
    /// The identity of the structure alone, derived once with it: what a model's preparation,
    /// which reads only the structure, is keyed by.
    pub structure_id: Arc<str>,
}

impl AssessmentEnvironment {
    /// The environment of `structure` under `measurements`. Fails only when the structure's
    /// host sample did: the process limits it read enter every identity on it.
    pub fn new(
        structure: Arc<AssessmentStructure>,
        measurements: Measurements,
    ) -> anyhow::Result<Self> {
        let structure_id = structure_id(&structure)?;
        Ok(Self::of(structure, structure_id, measurements))
    }

    /// The environment of what the stored records give assessment now (`stored`,
    /// [`AssessmentStructure::stored`]): this one's structure identity while its structure is
    /// this one's, a new one when the records priced an input of it anew.
    pub fn stored(&self, stored: AssessmentSetup) -> Self {
        let structure_id = if Arc::ptr_eq(&stored.structure, &self.setup.structure) {
            Arc::clone(&self.structure_id)
        } else {
            // The records price the same structure anew: its host sample is the one this
            // identity was derived from.
            structure_id(&stored.structure).expect("a stored structure keeps its host sample")
        };
        Self::of(stored.structure, structure_id, stored.measurements)
    }

    /// Only stable evidence enters identity: the structure's, and the executor costs once
    /// derived (that they are pending, or unavailable and why, otherwise), not current usage or
    /// measurement timestamps. The costs enter as the stored records they were derived from (the
    /// structure's identity includes the engine build, which derives costs from records). No
    /// model's own measurement enters it. The identity keys every assessment's work, cache entry and contract
    /// identity, the candidates' cache, and the snapshot's `environment_id`, and a new one is
    /// what moves the pool to new records ([`super::assessor::ModelAssessor::adopt`]).
    fn of(
        structure: Arc<AssessmentStructure>,
        structure_id: Arc<str>,
        measurements: Measurements,
    ) -> Self {
        let measured = match &measurements {
            Measurements::Pending => json!("pending"),
            Measurements::Unavailable { reason } => json!({ "unavailable": reason }),
            Measurements::Measured { records, .. } => json!({ "records": records.as_ref() }),
        };
        let material = json!({
            "structure": &*structure_id,
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
            structure_id,
        }
    }
}

/// The identity of `structure`'s [`structure_identity`].
fn structure_id(structure: &AssessmentStructure) -> anyhow::Result<Arc<str>> {
    let identity = structure_identity(structure)?;
    Ok(format!("structure_{:x}", Sha256::digest(identity.to_string().as_bytes())).into())
}

/// What of `structure` enters identity: the engine build, the selected device and what its graphs
/// are priced at, the topology's normalized structure, the process limits and their visibility,
/// the reserves and the configuration models are assessed with.
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
        .map(|limit| json!({ "kind": limit.kind, "bytes": limit.limit_bytes }))
        .collect::<Vec<_>>();
    Ok(json!({
        "engine_build": EngineBuild::current().0,
        "device": structure.selected.info.selector().to_string(),
        "graph_cost": structure.selected.graph_cost,
        "topology": structure.topology.normalized(),
        "process_limits": process_limits,
        "limit_visibility": host.limit_visibility,
        "reserves": structure.reserves.identity(),
        "policy": structure.policy,
        "service": structure.service,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::BackendName;

    fn cpu_structure() -> AssessmentStructure {
        assessment_structure(
            &seismic::Devices::discover(&[]).expect("device catalog"),
            ServingDevice::of(ExecutorSelector::HostCpu),
            None,
        )
        .expect("the host CPU is usable")
    }

    /// The serving device is the automatic selection, and the structure is assessed on exactly
    /// the serving device it is given, whatever the automatic selection is.
    #[test]
    fn the_structure_is_assessed_on_the_serving_device() {
        let devices = seismic::Devices::discover(&[]).expect("device catalog");
        let automatic = select(&devices.topology(), ExecutionPath::Native, SERVING_REQUEST)
            .expect("the host CPU is usable");
        let serving = ServingDevice::select(&devices).unwrap();
        assert_eq!(serving.selector(), automatic.selector());
        for executor in devices
            .topology()
            .executors()
            .iter()
            .filter(|executor| magnitude_executor::platform::unusable_reason(executor).is_none())
        {
            let serving = ServingDevice::of(executor.selector());
            let structure = assessment_structure(&devices, serving, None).unwrap();
            assert_eq!(structure.selected.info.selector(), serving.selector());
            assert_eq!(structure.device, serving.request());
        }
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
