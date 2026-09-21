//! Metal structural implementation alternatives.
//!
//! Authored portable and Metal lowering/helper bodies are constructed by
//! the core semantic factory. Metal currently adds no independent structural
//! rewrite beyond those authored candidates.

use crate::Metal;
use seismic_compiler::implementation::ImplementationFactory;

pub(crate) fn structural_factories() -> Vec<Box<dyn ImplementationFactory<Metal>>> {
    Vec::new()
}
