//! Device selection, the engine memory policy, budget admission and the
//! phase-one readiness gate over the one Seismic device API. Seismic owns
//! discovery, identity, capacity, observations and enforcement; this module
//! owns only model-facing policy. Platform readiness qualifies the available
//! kernel pack; it does not claim that a complete target, head, or encoder
//! executor exists.

mod policy;
mod qualification;
mod selection;
mod volume;

pub(crate) use policy::refresh_device_ceiling;
pub(crate) use policy::ShownBasis;
pub use policy::{
    DemandFit, DisplacementBaseline, DomainReading, DomainRole, DomainThresholds, Fit, FitMode, FitRefusal, HostDistress, HostEmergency,
    StoreMotion, HostRoom, ReleaseRate, Releasable, Releasing, LoadStage, MemoryBand, MemoryConstraint, MemoryPolicyError, MemoryReserves, MotionMeter, PageProbe, PlacementFit,
    READING_INTERVAL, RECOVERY_STABLE_TIME, Recovery, Serving, Shown, StorageDemand, StoreFit, observe_domains,
};
pub use volume::{VolumeReserve, VolumeReserves};
pub use qualification::{
    FitReading, GraphPricing, OpenedPlatform, PlatformConfig, PlatformError, SelectedDevice, graph_pricing, host_tier, open_selected, select_device, select_recorded,
    selected,
};
pub use selection::{
    AUTOMATIC_BACKEND_ORDER, DeviceRequest, DeviceRequestParseError, SelectionError, select,
    unusable_reason,
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
