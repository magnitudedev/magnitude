use seismic::{BackendName, Device, DeviceCatalog, DeviceId};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceFacts {
    pub memory_bytes: u64,
    pub unified_memory: bool,
}

#[derive(Clone, Debug)]
pub struct Endpoint {
    pub backend: BackendName,
    pub ordinal: u32,
    pub name: String,
    pub facts: DeviceFacts,
    pub unavailable: Vec<String>,
    pub(crate) id: Option<DeviceId>,
}

impl Endpoint {
    pub fn is_available(&self) -> bool {
        self.unavailable.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct Topology {
    pub endpoints: Vec<Endpoint>,
    pub host_memory_bytes: u64,
}

pub struct Discovery {
    catalog: DeviceCatalog,
    topology: Topology,
}

impl Discovery {
    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    pub fn open(&self, endpoint: &Endpoint) -> Result<Device, DiscoveryError> {
        let id = endpoint.id.ok_or_else(|| DiscoveryError::Open {
            backend: endpoint.backend,
            ordinal: endpoint.ordinal,
            outcome: endpoint
                .unavailable
                .join("; ")
                .if_empty("device was not discovered"),
        })?;
        self.catalog.open(id).map_err(|error| DiscoveryError::Open {
            backend: endpoint.backend,
            ordinal: endpoint.ordinal,
            outcome: error.to_string(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiscoveryError {
    Catalog(String),
    HostMemory(String),
    Open {
        backend: BackendName,
        ordinal: u32,
        outcome: String,
    },
}

impl fmt::Display for DiscoveryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Catalog(outcome) => write!(formatter, "device discovery failed: {outcome}"),
            Self::HostMemory(outcome) => {
                write!(formatter, "host memory discovery failed: {outcome}")
            }
            Self::Open {
                backend,
                ordinal,
                outcome,
            } => write!(
                formatter,
                "failed to open {} device {ordinal}: {outcome}",
                backend.as_str()
            ),
        }
    }
}

impl std::error::Error for DiscoveryError {}

pub fn discover() -> Result<Discovery, DiscoveryError> {
    let catalog =
        DeviceCatalog::discover().map_err(|error| DiscoveryError::Catalog(error.to_string()))?;
    let host_memory_bytes = host_memory_bytes()?;
    let mut endpoints = Vec::new();
    let backends = [BackendName::Metal, BackendName::Cuda, BackendName::Cpu];
    for backend in backends {
        #[cfg(not(target_os = "macos"))]
        if backend == BackendName::Metal {
            continue;
        }
        let devices = catalog
            .devices()
            .iter()
            .filter(|device| device.backend == backend)
            .collect::<Vec<_>>();
        if devices.is_empty() {
            endpoints.push(Endpoint {
                backend,
                ordinal: 0,
                name: format!("{}:0", backend.as_str()),
                facts: DeviceFacts {
                    memory_bytes: 0,
                    unified_memory: backend != BackendName::Cuda,
                },
                unavailable: vec!["no device was discovered by Seismic".into()],
                id: None,
            });
            continue;
        }
        for (ordinal, descriptor) in devices.into_iter().enumerate() {
            let unavailable = catalog
                .open(descriptor.id)
                .err()
                .map(|error| vec![error.to_string()])
                .unwrap_or_default();
            endpoints.push(Endpoint {
                backend,
                ordinal: ordinal as u32,
                name: descriptor.name.clone(),
                facts: DeviceFacts {
                    memory_bytes: descriptor.memory_bytes,
                    unified_memory: backend != BackendName::Cuda,
                },
                unavailable,
                id: Some(descriptor.id),
            });
        }
    }
    Ok(Discovery {
        catalog,
        topology: Topology {
            endpoints,
            host_memory_bytes,
        },
    })
}

#[cfg(target_os = "macos")]
fn host_memory_bytes() -> Result<u64, DiscoveryError> {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let mut bytes = 0_u64;
    let mut length = std::mem::size_of::<u64>();
    let status = unsafe {
        sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if status == 0 && length == std::mem::size_of::<u64>() && bytes > 0 {
        Ok(bytes)
    } else {
        Err(DiscoveryError::HostMemory(format!(
            "sysctl hw.memsize returned status {status} and {length} bytes"
        )))
    }
}

#[cfg(target_os = "linux")]
fn host_memory_bytes() -> Result<u64, DiscoveryError> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| DiscoveryError::HostMemory(error.to_string()))?;
    let kib = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| DiscoveryError::HostMemory("MemTotal is absent or invalid".into()))?;
    kib.checked_mul(1024)
        .ok_or_else(|| DiscoveryError::HostMemory("MemTotal overflow".into()))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn host_memory_bytes() -> Result<u64, DiscoveryError> {
    Err(DiscoveryError::HostMemory(
        "host memory discovery is unsupported on this operating system".into(),
    ))
}

trait IfEmpty {
    fn if_empty(self, fallback: &str) -> String;
}

impl IfEmpty for String {
    fn if_empty(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.into()
        } else {
            self
        }
    }
}
