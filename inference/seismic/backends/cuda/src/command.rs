//! One CUDA native kernel and the exact launch-frame ABI. Schedule and static
//! emission layout are compiler-owned.

use crate::direct::DirectModule;
use seismic_ir::physical_target::KernelEmissionLayout;
use seismic_native_target::Compile;
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
    /// The live compile that holds the kernel, kept so the device's
    /// programs serve it to every later formation of its source.
    pub(crate) program: Arc<Compile<DirectModule>>,
    /// The kernel's position in `program`.
    pub(crate) function: usize,
    pub(crate) layout: KernelEmissionLayout,
    /// Exact value submitted to the post-load function configuration setter.
    pub(crate) configured_dynamic_shared_bytes: i32,
}

impl CompiledKernel {
    pub(crate) fn module(&self) -> &DirectModule {
        self.program.program()
    }
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

impl std::fmt::Debug for CompiledKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledKernel")
            .field("words", &self.layout.words.total)
            .field("image_bytes", &self.module().image_bytes())
            .field("entry", &self.program.entries()[self.function].symbol)
            .field(
                "configured_dynamic_shared_bytes",
                &self.configured_dynamic_shared_bytes,
            )
            .finish()
    }
}
