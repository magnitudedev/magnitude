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
    assessment_capacity, growth_availability, refresh_allocation_ceiling, GrowthAvailability,
    MemoryConstraint, MemoryPolicyError,
};
pub use qualification::{
    open_selected, select_device, OpenedPlatform, PlatformConfig, PlatformError, SelectedDevice,
};
pub use selection::{select, DeviceRequest, DeviceRequestParseError, SelectionError};
