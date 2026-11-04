//! Discovery (spec §2.3): what the backend crates and the OS report of the
//! machine, read into the reports `seismic_topology::build` turns into a
//! topology revision.

use crate::host;
use seismic_topology::{Availability, CpuReport, Reports, VolumeReport};
use std::fmt;
use std::path::PathBuf;

/// Why no topology could be discovered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiscoveryError {
    /// Host RAM capacity could not be established; no topology is complete
    /// without it.
    HostMemory(String),
    /// The OS build, the CPU's and Metal's driver, could not be read.
    OsBuild(String),
    /// The CPU as the OS reports it could not be read.
    CpuIdentity(String),
    /// A cache the OS reports for the CPU, part of its identity, could not
    /// be read.
    CpuCache(String),
    /// A served path's volume could not be identified or sized.
    Volume { path: PathBuf, message: String },
}

impl fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostMemory(message) => write!(f, "host memory discovery failed: {message}"),
            Self::OsBuild(message) => write!(f, "the OS build could not be read: {message}"),
            Self::CpuIdentity(message) => {
                write!(f, "the CPU identity could not be read: {message}")
            }
            Self::CpuCache(message) => write!(f, "the CPU's caches could not be read: {message}"),
            Self::Volume { path, message } => {
                write!(
                    f,
                    "the volume of {} could not be read: {message}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for DiscoveryError {}

/// What discovery reads, with the native handles status sampling reads
/// again.
pub(crate) struct Gathered {
    pub(crate) reports: Reports,
    pub(crate) natives: Natives,
}

/// Unopened native handles a sample reads.
#[derive(Default)]
pub(crate) struct Natives {
    #[cfg(target_os = "macos")]
    pub(crate) metal: Vec<seismic_metal::DeviceHandle>,
    /// Each Vulkan device's handle, by UUID.
    #[cfg(not(target_os = "macos"))]
    pub(crate) vulkan: Vec<([u8; 16], seismic_vulkan::DeviceHandle)>,
}

/// Read the backend crates and the OS. Only cheap native queries run, except
/// what the backends themselves need to establish availability, grants and
/// capability: a Metal device's language and tensor-operation probes
/// (compiles), and a discrete CUDA device's usable memory, which needs its
/// context (once per process).
pub(crate) fn gather(paths: &[PathBuf]) -> Result<Gathered, DiscoveryError> {
    let host = host::capacity().map_err(DiscoveryError::HostMemory)?;
    let os_build = host::os_build().map_err(DiscoveryError::OsBuild)?;
    let mut diagnostics = Vec::new();

    let cache = host::last_level_cache().map_err(DiscoveryError::CpuCache)?;
    let cpu = CpuReport {
        name: format!("Host CPU ({})", std::env::consts::ARCH),
        identity: host::cpu_identity(cache.map(|cache| cache.value))
            .map_err(DiscoveryError::CpuIdentity)?,
        cache,
        availability: cpu_availability(seismic_native_cpu::Tier::detected()),
        max_allocation: seismic_cpu::MAX_ALLOCATION_BYTES,
    };

    #[cfg(target_os = "macos")]
    let (metal, natives) = {
        let (metal, handles) = seismic_metal::topology::describe(&mut diagnostics)
            .into_iter()
            .unzip();
        (metal, Natives { metal: handles })
    };
    #[cfg(not(target_os = "macos"))]
    let mut natives = Natives::default();
    #[cfg(not(target_os = "macos"))]
    let metal = Vec::new();

    let cuda = seismic_cuda::topology::describe(&mut diagnostics);

    #[cfg(not(target_os = "macos"))]
    let vulkan = {
        let described = seismic_vulkan::topology::describe(&mut diagnostics);
        natives.vulkan = described
            .iter()
            .map(|(report, handle)| (report.uuid, *handle))
            .collect();
        described.into_iter().map(|(report, _)| report).collect()
    };
    // Vulkan is not built on macOS (MoltenVK is not supported).
    #[cfg(target_os = "macos")]
    let vulkan = {
        diagnostics.push(seismic_topology::Diagnostic {
            source: seismic_topology::DiagnosticSource::Backend(
                seismic_topology::BackendName::Vulkan,
            ),
            message: "this build has no Vulkan runtime".into(),
        });
        Vec::new()
    };

    #[cfg(windows)]
    let adapters = match crate::adapters::adapters() {
        Ok(adapters) => adapters,
        Err(message) => {
            diagnostics.push(seismic_topology::Diagnostic {
                source: seismic_topology::DiagnosticSource::Adapters,
                message,
            });
            Vec::new()
        }
    };
    #[cfg(not(windows))]
    let adapters = Vec::new();

    let volumes = paths
        .iter()
        .map(|path| {
            let failed = |error: std::io::Error| DiscoveryError::Volume {
                path: path.clone(),
                message: error.to_string(),
            };
            Ok(VolumeReport {
                path: path.clone(),
                id: crate::volume::identify(path).map_err(failed)?,
                size: crate::volume::size(path).map_err(failed)?,
            })
        })
        .collect::<Result<_, DiscoveryError>>()?;

    Ok(Gathered {
        reports: Reports {
            host,
            os_build,
            cpu,
            metal,
            cuda,
            vulkan,
            adapters,
            volumes,
            diagnostics,
        },
        natives,
    })
}

/// Discovery must enforce the same minimum instruction set as native
/// specialization. Otherwise assessment promises a CPU that cannot form even
/// the first import kernel (for example QEMU's default virtual CPU).
pub(crate) fn cpu_availability(tier: Option<seismic_native_cpu::Tier>) -> Availability {
    match tier {
        Some(_) => Availability::Available,
        None => Availability::Unavailable {
            reason: format!(
                "CPU execution requires {} ({}) or higher; \
                 expose these CPU features to the virtual machine",
                seismic_native_cpu::Tier::all()[0].name(),
                seismic_native_cpu::Tier::all()[0].features(),
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::cpu_availability;
    use seismic_topology::Availability;

    #[test]
    fn cpu_discovery_enforces_the_native_instruction_set_floor() {
        let Availability::Unavailable { reason } = cpu_availability(None) else {
            panic!("a CPU with no native instruction-set tier was admitted");
        };
        assert!(reason.contains(seismic_native_cpu::Tier::all()[0].features()));
        for tier in seismic_native_cpu::Tier::all() {
            assert_eq!(cpu_availability(Some(*tier)), Availability::Available);
        }
    }
}
