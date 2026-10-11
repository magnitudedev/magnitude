//! The native kernels a load prepares, derived from its plan without a
//! device.

use crate::native::{kernel_requests, TuningLimits};
use crate::{BackendPlan, CatalogError};
use magnitude_family_contracts::ModelDefinition;
use seismic::KernelRequest;

/// Every native kernel a load of `plan`'s model prepares on its backend,
/// each once: the walk preparation makes, listing each entry instead of
/// preparing it.
pub fn kernel_inventory(
    definition: &ModelDefinition,
    plan: &BackendPlan,
) -> Result<Vec<KernelRequest>, CatalogError> {
    kernel_requests(
        plan.backend(),
        plan.programs(),
        definition,
        plan.load(),
        TuningLimits::of(plan.policy().limits(), definition),
    )
    .map_err(|failure| CatalogError::native(plan.backend(), failure))
}
