//! The assessment environment: the selected execution device, its memory bandwidth and the
//! engine configuration every model is assessed with, and the identity that keys every cached
//! assessment made in it. Established once, during service start.

use magnitude_engine::assessment::{AssessmentSetup, ModelAssessmentError};
use magnitude_engine::options::{ModelPolicy, standard_service_limits};
use magnitude_engine::worker::protocol::EngineBuild;
use magnitude_executor::assessment::{BandwidthSource, DeviceClass};
use magnitude_executor::platform::{DeviceRequest, MemoryReserves};
use magnitude_service_contracts::models::AssessmentEnvironmentId;
use seismic::{DeviceCatalog, DeviceMemory, DeviceTopology, HostMemoryStatus};
use serde_json::json;
use sha2::{Digest, Sha256};

/// The engine configuration the service loads and assesses every model with.
pub fn serving_policy() -> ModelPolicy {
    ModelPolicy::default()
}

pub struct AssessmentEnvironment {
    pub id: AssessmentEnvironmentId,
    pub setup: AssessmentSetup,
}

impl AssessmentEnvironment {
    /// Select the device a native load selects, resolve its bandwidth, observe the host and
    /// derive the environment's identity. Opens no device.
    pub fn establish(catalog: &DeviceCatalog) -> Result<Self, ModelAssessmentError> {
        let setup = AssessmentSetup::discover(
            catalog,
            DeviceRequest::Automatic,
            MemoryReserves::standard(),
            serving_policy(),
            standard_service_limits(),
        )?;
        let bandwidth = setup.bandwidth;
        tracing::info!(
            device = %setup.selected.info.selector,
            device.name = %setup.selected.info.name,
            backend = setup.selected.info.backend.as_str(),
            bandwidth.bytes_per_second = bandwidth.bytes_per_second,
            bandwidth.source = bandwidth_source(bandwidth.source),
            "assessment environment established"
        );
        Ok(Self {
            id: environment_id(&setup),
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
/// its resolved bandwidth, the stable topology and process limits that bound fit capacity, the
/// reserve policy and the serving configuration. Live free memory is not an input.
fn environment_id(setup: &AssessmentSetup) -> AssessmentEnvironmentId {
    let material = json!({
        "engine_build": EngineBuild::current().0,
        "backend": setup.selected.info.backend.as_str(),
        "device": setup.selected.info.selector,
        "device_name": setup.selected.info.name,
        "bandwidth": {
            "bytes_per_second": setup.bandwidth.bytes_per_second,
            "source": bandwidth_source(setup.bandwidth.source),
        },
        "topology": normalized_topology(&setup.topology),
        "process_limits": process_limits(&setup.host),
        "reserves": format!("{:?}", setup.reserves),
        "policy": format!("{:?}", setup.policy),
        "service": format!("{:?}", setup.service),
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
