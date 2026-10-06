//! Device selection, the engine memory policy, budget admission and the
//! phase-one readiness gate over the one Seismic device API. Seismic owns
//! discovery, identity, capacity, observations and enforcement; this module
//! owns only model-facing policy. Platform readiness qualifies the available
//! kernel pack; it does not claim that a complete target, head, or encoder
//! executor exists.

mod policy;
mod qualification;
mod selection;

pub use policy::{
    band_of, fit_capacities, host_distress, observe_domains, refresh_device_ceiling,
    DomainReading, DomainRole, DomainThresholds, FitCapacity, HostDistress, MemoryBand,
    MemoryConstraint, MemoryPolicyError, MemoryReserves,
};
pub use qualification::{
    open_selected, select_device, OpenedPlatform, PlatformConfig, PlatformError, SelectedDevice,
};
pub use selection::{
    select, unusable_reason, DeviceRequest, DeviceRequestParseError, SelectionError,
    AUTOMATIC_BACKEND_ORDER,
};

/// Relaxes the macOS GPU watchdog for this process. M1 and M2 GPUs cannot
/// preempt a long prefill attention launch, so while the window server waits
/// for the GPU, macOS kills the command buffer ("Impacting Interactivity")
/// and the model is lost. `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` is read by Apple's
/// GPU driver; llama.cpp sets it for the same failure. Call first in `main`,
/// before any thread or Metal device exists; child processes inherit it, and
/// an explicit setting is kept.
pub fn relax_gpu_watchdog() {
    const RELAX: &str = "AGX_RELAX_CDM_CTXSTORE_TIMEOUT";
    if cfg!(target_os = "macos") && std::env::var_os(RELAX).is_none() {
        std::env::set_var(RELAX, "1");
    }
}
