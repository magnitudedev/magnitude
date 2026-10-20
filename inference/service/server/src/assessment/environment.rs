//! The assessment environment: the selected execution device, its memory bandwidth and the
//! engine configuration every model is assessed with, and the identity that keys every cached
//! assessment made in it. Established once, during service start.

use magnitude_engine::assessment::AssessmentSetup;
use magnitude_engine::options::{ModelPolicy, standard_service_limits};
use magnitude_engine::worker::protocol::EngineBuild;
use magnitude_executor::assessment::{BandwidthSource, DeviceClass};
use magnitude_executor::platform::{DeviceRequest, MemoryReserves};
use magnitude_service_contracts::models::AssessmentEnvironmentId;
use seismic::Devices;
use serde_json::json;
use sha2::{Digest, Sha256};

/// The engine configuration the service loads and assesses every model with.
pub fn serving_policy() -> ModelPolicy {
    ModelPolicy::default()
}

/// The device the service loads, assesses and reports every model on: the engine's automatic
/// selection.
pub const SERVING_DEVICE: DeviceRequest = DeviceRequest::Automatic;

pub struct AssessmentEnvironment {
    pub id: AssessmentEnvironmentId,
    pub setup: AssessmentSetup,
}

impl AssessmentEnvironment {
    /// Select the device a native load selects, resolve its bandwidth, take one status sample
    /// and derive the environment's identity. Opens no device.
    pub fn establish(catalog: &Devices) -> anyhow::Result<Self> {
        let setup = AssessmentSetup::discover(
            catalog,
            SERVING_DEVICE,
            MemoryReserves::standard(),
            serving_policy(),
            standard_service_limits(),
        )?;
        let bandwidth = setup.bandwidth;
        tracing::info!(
            device = %setup.selected.info.selector(),
            device.name = %setup.selected.info.name(),
            backend = setup.selected.info.backend().as_str(),
            bandwidth.bytes_per_second = bandwidth.bytes_per_second,
            bandwidth.source = bandwidth_source(bandwidth.source),
            "assessment environment established"
        );
        Ok(Self {
            id: environment_id(&setup)?,
            setup,
        })
    }
}

fn bandwidth_source(source: BandwidthSource) -> &'static str {
    match source {
        BandwidthSource::Reported => "reported",
        BandwidthSource::Published => "published",
        BandwidthSource::Assumed(DeviceClass::DedicatedGpu) => "assumed_dedicated_gpu",
        BandwidthSource::Assumed(DeviceClass::IntegratedGpu) => "assumed_integrated_gpu",
        BandwidthSource::Assumed(DeviceClass::Cpu) => "assumed_cpu",
    }
}

/// The identity of everything an assessment result depends on besides the model: engine build
/// (covering model families, the fit workload, decode costs and the bandwidth table), device and
/// its resolved bandwidth, the topology's normalized structure and the process limits that bound
/// fit capacity, the reserve policy and the serving configuration. Live free memory is not an
/// input. Fails only when the host sample did: its process limits enter the identity.
fn environment_id(setup: &AssessmentSetup) -> anyhow::Result<AssessmentEnvironmentId> {
    let host = setup
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
    let material = json!({
        "engine_build": EngineBuild::current().0,
        "device": setup.selected.info.selector().to_string(),
        "device_name": setup.selected.info.name(),
        "bandwidth": {
            "bytes_per_second": setup.bandwidth.bytes_per_second,
            "source": bandwidth_source(setup.bandwidth.source),
        },
        "topology": setup.topology.normalized(),
        "process_limits": process_limits,
        "limit_visibility": format!("{:?}", host.limit_visibility),
        "reserves": setup.reserves.identity(),
        "policy": format!("{:?}", setup.policy),
        "service": format!("{:?}", setup.service),
    });
    Ok(AssessmentEnvironmentId(format!(
        "environment_{:x}",
        Sha256::digest(material.to_string().as_bytes())
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::BackendName;

    fn cpu_setup() -> AssessmentSetup {
        AssessmentSetup::discover(
            &seismic::Devices::discover(&[]).expect("device catalog"),
            DeviceRequest::Backend(BackendName::Cpu),
            MemoryReserves::standard(),
            serving_policy(),
            standard_service_limits(),
        )
        .expect("the CPU backend is selectable")
    }

    #[test]
    fn a_different_backend_or_hardware_is_a_different_environment() {
        let setup = cpu_setup();
        let id = environment_id(&setup).unwrap();
        assert_eq!(
            environment_id(&setup).unwrap(),
            id,
            "the identity is stable"
        );

        // Any other executor of the machine (another backend, another device).
        let mut other = cpu_setup();
        if let Some(executor) = other
            .topology
            .executors()
            .iter()
            .find(|executor| executor.backend() != BackendName::Cpu)
            .cloned()
        {
            other.selected.info = executor;
            assert_ne!(environment_id(&other).unwrap(), id);
        }

        let mut setup = cpu_setup();
        setup.bandwidth.bytes_per_second += 1;
        assert_ne!(environment_id(&setup).unwrap(), id);
    }
}
