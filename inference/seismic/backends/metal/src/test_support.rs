//! The Metal device every backend test runs on, the store its probe
//! answers are kept in, and the programs its kernels form through.
use crate::toolchain::MetalToolchain;
use crate::{DeviceHandle, Metal, MetalDevice, MetalPrograms};
use seismic_native_target::{
    ArtifactKey, Compile, DeviceDescription, Formed, MemoryStore, NativeCompilationError,
    ProgramSource, Programs, Toolchain,
};
use std::sync::{Arc, Once};

/// The process's probe-answer store: each device's probes run once a test process.
pub(crate) fn probes() -> &'static MemoryStore {
    static STORE: std::sync::OnceLock<MemoryStore> = std::sync::OnceLock::new();
    STORE.get_or_init(MemoryStore::default)
}

static VALIDATION: Once = Once::new();

/// Opens the system default device with the Metal API validation layer
/// enabled. Metal reads `MTL_DEBUG_LAYER` when the process makes its first
/// Metal call, so the variable is set once before any test touches Metal;
/// an encoding the API forbids then aborts the test process.
pub(crate) fn metal_device() -> MetalDevice {
    VALIDATION.call_once(|| std::env::set_var("MTL_DEBUG_LAYER", "1"));
    MetalDevice::open(DeviceHandle::system_default().unwrap(), store()).unwrap()
}

/// The store a backend test's device opens with: a tally of host RAM, the
/// store Metal's shared storage lies in, admitting everything.
pub(crate) fn store() -> std::sync::Arc<dyn seismic_memory::entry::Store> {
    std::sync::Arc::new(seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam))
}

/// The ledger entry a backend test's allocations are charged to: a tally
/// of its own, as [`store`]'s.
pub(crate) fn entry() -> std::sync::Arc<dyn seismic_memory::entry::Entry> {
    std::sync::Arc::new(seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam))
}

/// The programs of `device` in this crate's tests, which sit below the
/// runtime's program former: each source compiled by the device's
/// toolchain, so a test of a kernel's rendering and execution runs the
/// compile the former would.
pub(crate) fn programs(device: &MetalDevice, description: &DeviceDescription<Metal>) -> MetalPrograms {
    Arc::new(Compiled(MetalToolchain::new(device.clone(), description.device_info())))
}

struct Compiled(MetalToolchain);

impl Programs<MetalToolchain> for Compiled {
    fn form(
        &self,
        sources: &[ProgramSource],
    ) -> Result<Vec<Formed<<MetalToolchain as Toolchain>::Program>>, NativeCompilationError> {
        sources
            .iter()
            .map(|source| {
                let (_, program) = self.0.compile(source)?;
                Ok(Formed::new(
                    ArtifactKey::of(&[source.text.as_bytes()]),
                    vec![Arc::new(Compile::new(source.entries.clone(), program))],
                ))
            })
            .collect()
    }
}
