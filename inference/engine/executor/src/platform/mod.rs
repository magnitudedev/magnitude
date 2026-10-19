//! Device selection, the engine memory policy, budget admission and the
//! phase-one readiness gate over the one Seismic device API. Seismic owns
//! discovery, identity, capacity, observations and enforcement; this module
//! owns only model-facing policy. Platform readiness qualifies the available
//! kernel pack; it does not claim that a complete target, head, or encoder
//! executor exists.

mod policy;
mod qualification;
mod selection;

pub(crate) use policy::refresh_device_ceiling;
pub use policy::{
    DemandFit, DomainReading, DomainRole, DomainThresholds, Fit, FitMode, FitRefusal, HostDistress,
    MemoryBand, MemoryConstraint, MemoryPolicyError, MemoryReserves, PlacementFit, StorageDemand,
    StoreFit, observe_domains,
};
pub use qualification::{
    OpenedPlatform, PlatformConfig, PlatformError, SelectedDevice, open_selected, select_device,
};
pub use selection::{
    AUTOMATIC_BACKEND_ORDER, DeviceRequest, DeviceRequestParseError, SelectionError, select,
    unusable_reason,
};

/// The space a download may take on the volume holding `path`: each call
/// reads what the OS lets this process still write there, and takes fit's
/// volume ceiling of it ([`MemoryReserves::volume_thresholds`], the
/// thresholds [`Fit::store`] holds every volume to).
pub fn volume_space(
    path: std::path::PathBuf,
    reserves: MemoryReserves,
) -> impl Fn() -> Result<u64, seismic::ObservationError> + Send + Sync + 'static {
    move || {
        let available = seismic::volume_available(&path)?.value;
        Ok(reserves.volume_thresholds().ceiling(available))
    }
}

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
