//! Device discovery, selection, budget planning, and the phase-one readiness
//! gate. Platform readiness qualifies the available kernel pack; it does not
//! claim that a complete target, head, or encoder executor exists.

mod discovery;
mod qualification;
mod selection;

pub use discovery::{discover, DeviceFacts, Discovery, DiscoveryError, Endpoint, Topology};
pub use qualification::{
    open_selected, prepare, prepare_selected, OpenedPlatform, PlatformConfig, PlatformError,
    QualifiedPlatform,
};
pub use selection::{budget, select, BudgetError, MemoryRequirements, Plan, SelectionError};
