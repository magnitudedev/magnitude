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

pub(crate) use policy::domain_readings;
pub use policy::{
    DemandFit, DomainReading, DomainRole, Fit, FitMode, FitRefusal, MemoryConstraint, MemoryPolicyError,
    PlacementFit, ReleaseRate, StorageDemand, StoreFit, observe_domains,
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
