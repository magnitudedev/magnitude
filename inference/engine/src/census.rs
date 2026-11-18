//! The allocation census a loaded engine reports (integration spec §9.2):
//! each store's charge by class, as its ledger charged it: every byte the
//! model holds, its host tables' row caches among them, is charged to the
//! ledger once. Because memory is elastic these are the
//! standing at the observation, not a fixed reservation. Beside it, how the
//! owner serves at the observation (`ServingCensus`): its prompt groups'
//! places and where its routed experts are.

use magnitude_executor::StoreAllocation;
pub use magnitude_executor::{BankHolding, BankResize, ExpertCover, ExpertResidency, HostResidency, TableResidency};
pub use magnitude_scheduler::serving::ServingCensus;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationCensus {
    pub stores: Vec<StoreAllocation>,
    pub serving: ServingCensus,
}
