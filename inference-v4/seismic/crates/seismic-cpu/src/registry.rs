//! The sealed CPU capability/factory registry.

use crate::{factory, Cpu};
use seismic_compiler::target::CapabilityRegistry;
use std::sync::OnceLock;

pub fn registry() -> &'static CapabilityRegistry<Cpu> {
    static REGISTRY: OnceLock<CapabilityRegistry<Cpu>> = OnceLock::new();
    REGISTRY
        .get_or_init(|| CapabilityRegistry::assemble(Vec::new(), factory::structural_factories()))
}
