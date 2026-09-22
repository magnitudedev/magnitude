//! One compiled Metal kernel and the minimal native ABI metadata its executor
//! consumes.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;
pub struct Pipeline {
    pub(crate) state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// The minimal native ABI metadata retained by the executable. Geometry,
    /// representations, storage topology, and resource facts remain owned by
    /// the consumed core plan and are not mirrored here.
    pub(crate) words: seismic_ir::target::KernelWordLayout,
    pub(crate) result_slots: Vec<(
        seismic_ir::schedule::AnyScalarSlot,
        seismic_lang::types::DType,
    )>,
}

// MTLComputePipelineState is immutable and documented as thread-safe.
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("words", &self.words.total)
            .field("results", &self.result_slots.len())
            .finish()
    }
}
