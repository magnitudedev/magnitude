//! One CUDA native kernel and the exact launch-frame ABI. Schedule and static
//! emission layout are compiler-owned.

use crate::driver::{Handle, Module};
use seismic_compiler::target::KernelEmissionLayout;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct LaunchFrame {
    pub buffers: u64,
    pub words: u64,
    pub results: u64,
    pub participant_scratch: u64,
    pub register_scratch: u64,
}

pub struct CompiledKernel {
    pub(crate) module: Arc<Module>,
    pub(crate) function: Handle,
    pub(crate) layout: KernelEmissionLayout,
    pub(crate) occupancy: OccupancyRelation,
}

#[derive(Clone, Debug)]
pub(crate) struct OccupancyRelation {
    pub(crate) blocks: Vec<BlockOccupancy>,
}

#[derive(Clone, Debug)]
pub(crate) struct BlockOccupancy {
    pub(crate) threads: u32,
    /// Inclusive maximum dynamic-shared byte value and its exact active
    /// block count. Entries are ordered and the final threshold is the
    /// reflected function's maximum dynamic-shared domain.
    pub(crate) regimes: Vec<(u64, u32)>,
}

unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

impl std::fmt::Debug for CompiledKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledKernel")
            .field("words", &self.layout.words.total)
            .finish()
    }
}
