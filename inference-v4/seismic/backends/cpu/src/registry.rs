//! The sealed CPU capability registry.

use crate::Cpu;
use seismic_compiler::target::{CompilerRegistry, CompilerRegistryParts};
use std::sync::OnceLock;

fn native_launch_constraints(
    _: &seismic_native_target::DeviceDescription<Cpu>,
    _: &mut seismic_lang::expr::ExprArena,
    _: &seismic_ir::schedule::Launch<Cpu>,
    _: &seismic_ir::storage::LaunchLocalLayout,
    _: &seismic_ir::kernel::Kernel<Cpu>,
    _: &seismic_native_target::NativeKernelDescription<Cpu>,
) -> Vec<seismic_lang::expr::BoolExpr> {
    Vec::new()
}

fn addressable_resources(
    _: &<Cpu as seismic_ir::physical_target::PhysicalDialect>::Facts,
) -> Vec<seismic_ir::physical_target::AddressableResourceClass> {
    Vec::new()
}

pub fn registry() -> &'static CompilerRegistry<Cpu> {
    static REGISTRY: OnceLock<CompilerRegistry<Cpu>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        CompilerRegistry::assemble(CompilerRegistryParts {
            capabilities: Vec::new(),

            native_launch_constraints,
            addressable_resources,
            emitted_intrinsics: Default::default(),
        })
    })
}
