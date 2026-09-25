//! One compiled Metal kernel, its formation input, and the minimal native ABI
//! metadata its executor consumes.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;
pub struct Pipeline {
    pub(crate) state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Exact MSL text and entry submitted to form this retained pipeline.
    /// Metal does not expose the resulting native instruction image here.
    pub(crate) source: Box<str>,
    pub(crate) entry: Box<str>,
    /// The minimal native ABI metadata retained by the executable. Geometry,
    /// representations, storage topology, and resource facts remain owned by
    /// the consumed core plan and are not mirrored here.
    pub(crate) words: seismic_ir::physical_target::KernelWordLayout,
}

// MTLComputePipelineState is immutable and documented as thread-safe.
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("words", &self.words.total)
            .field("source_bytes", &self.source.len())
            .field("entry", &self.entry)
            .finish()
    }
}
