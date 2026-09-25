//! One CUDA native kernel and the exact launch-frame ABI. Schedule and static
//! emission layout are compiler-owned.

use crate::driver::{Handle, Module};
use seismic_ir::physical_target::KernelEmissionLayout;
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
    /// Exact cubin submitted to the loader that created `module`.
    pub(crate) image: Box<[u8]>,
    pub(crate) entry: Box<str>,
    /// Exact value submitted to the post-load function configuration setter.
    pub(crate) configured_dynamic_shared_bytes: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OccupancyRelation {
    pub(crate) blocks: Vec<BlockOccupancy>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockOccupancy {
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
            .field("image_bytes", &self.image.len())
            .field("entry", &self.entry)
            .field(
                "configured_dynamic_shared_bytes",
                &self.configured_dynamic_shared_bytes,
            )
            .finish()
    }
}
