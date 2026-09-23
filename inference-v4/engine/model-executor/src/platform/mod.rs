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
    admit_growth, assessment_capacity, host_planning_reserve, MemoryConstraint,
    MemoryPolicyError, DEDICATED_PLANNING_RESERVE,
};
pub use qualification::{
    open_selected, select_device, BudgetError, MemoryRequirements, OpenedPlatform,
    PlatformConfig, PlatformError, QualifiedPlatform, SelectedDevice,
};
pub use selection::{select, SelectionError};
