//! One CPU native kernel. Schedule and emission layout remain core-owned.

use crate::workers::LaunchEntry;
use cranelift_jit::JITModule;
use seismic_ir::target::KernelEmissionLayout;
use std::sync::Arc;

/// The JIT memory of one variant, freed when its last kernel drops.
pub(crate) struct JitMemory {
    module: Option<JITModule>,
}

// Finalized code memory is immutable and freed once after the last kernel.
unsafe impl Send for JitMemory {}
unsafe impl Sync for JitMemory {}

impl JitMemory {
    pub(crate) fn new(module: JITModule) -> Self {
        Self {
            module: Some(module),
        }
    }
}

impl Drop for JitMemory {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // Every function pointer into this memory is retained by the same
            // CompiledKernel Arc, so none can outlive the allocation.
            unsafe { module.free_memory() };
        }
    }
}

pub struct CompiledKernel {
    pub(crate) entry: LaunchEntry,
    pub(crate) memory: Arc<JitMemory>,
    pub(crate) layout: KernelEmissionLayout,
}

impl std::fmt::Debug for CompiledKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledKernel")
            .field("words", &self.layout.words.total)
            .finish()
    }
}
