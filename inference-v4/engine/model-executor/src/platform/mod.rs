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
    band_of, fit_capacities, observe_domains, refresh_device_ceiling, DomainReading, DomainRole,
    DomainThresholds, FitCapacity, MemoryBand, MemoryConstraint, MemoryPolicyError,
    MemoryReserves,
};
pub use qualification::{
    open_selected, select_device, OpenedPlatform, PlatformConfig, PlatformError, SelectedDevice,
};
pub use selection::{select, DeviceRequest, DeviceRequestParseError, SelectionError};
