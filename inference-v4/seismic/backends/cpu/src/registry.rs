//! The sealed CPU capability/factory registry.

use crate::{factory, Cpu, CpuLaunchMode};
use seismic_compiler::target::{CompilerRegistry, CompilerRegistryParts};
use std::sync::OnceLock;

fn cooperative_launch_mode(
    _: &<Cpu as seismic_ir::target::KernelDialect>::Facts,
) -> Option<CpuLaunchMode> {
    None
}

fn native_launch_constraints(
    _: &seismic_target::DeviceDescription<Cpu>,
    _: &mut seismic_lang::expr::ExprArena,
    _: &seismic_ir::schedule::Launch,
    _: &seismic_ir::storage::LaunchLocalLayout,
    _: &seismic_ir::kernel::Kernel<Cpu>,
    _: &seismic_target::NativeKernelDescription<Cpu>,
) -> Vec<seismic_lang::expr::BoolExpr> {
    Vec::new()
}

fn addressable_resources(
    _: &<Cpu as seismic_ir::target::KernelDialect>::Facts,
) -> Vec<seismic_ir::target::AddressableResourceClass> {
    Vec::new()
}

pub fn registry() -> &'static CompilerRegistry<Cpu> {
    static REGISTRY: OnceLock<CompilerRegistry<Cpu>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        CompilerRegistry::assemble(CompilerRegistryParts {
            capabilities: Vec::new(),
            structural_factories: factory::structural_factories(),
            independent_launch_mode: CpuLaunchMode,
            cooperative_launch_mode,
            native_launch_constraints,
            addressable_resources,
            emitted_intrinsics: Default::default(),
        })
    })
}
