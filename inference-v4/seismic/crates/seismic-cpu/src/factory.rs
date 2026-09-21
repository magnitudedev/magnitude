//! CPU structural implementation registrations.
//!
//! CPU supports the complete standard operation set through the compiler's
//! universal portable implementation. It advertises no backend capability
//! intrinsics and has no additional structural algorithm family: an empty
//! list is therefore the exhaustive CPU-specific factory set.

use crate::Cpu;
use seismic_compiler::implementation::ImplementationFactory;

pub(crate) fn structural_factories() -> Vec<Box<dyn ImplementationFactory<Cpu>>> {
    Vec::new()
}
