//! CUDA structural implementation registrations.
//!
//! The standard library's portable implementation factory constructs the
//! full operation set. Backend structural policies are registered here only
//! through the compiler's shared semantic walker; CUDA never owns a second
//! traversal or a raw partial builder. This remains empty until that sealed
//! policy surface is available.

use crate::Cuda;
use seismic_compiler::implementation::ImplementationFactory;

pub(crate) fn structural_factories() -> Vec<Box<dyn ImplementationFactory<Cuda>>> {
    Vec::new()
}
