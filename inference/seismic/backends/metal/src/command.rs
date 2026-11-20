//! One formed Seismic-IR kernel and the minimal native ABI metadata its
//! executor consumes.

use crate::DirectPipeline;
use seismic_native_target::Compile;
use std::sync::Arc;

pub struct Pipeline {
    /// The compile that holds the kernel's pipeline, kept alive with it, and
    /// the pipeline's position in it.
    program: Arc<Compile<Vec<DirectPipeline>>>,
    position: usize,
    /// The minimal native ABI metadata retained by the executable. Geometry,
    /// representations, storage topology, and resource limits remain owned by
    /// the consumed core plan and are not mirrored here.
    pub(crate) words: seismic_ir::physical_target::KernelWordLayout,
}

impl Pipeline {
    pub(crate) fn new(
        program: Arc<Compile<Vec<DirectPipeline>>>,
        position: usize,
        words: seismic_ir::physical_target::KernelWordLayout,
    ) -> Self {
        Self { program, position, words }
    }

    pub(crate) fn pipeline(&self) -> &DirectPipeline {
        &self.program.program()[self.position]
    }
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("words", &self.words.total)
            .field("entry", &self.program.entries()[self.position].symbol)
            .finish()
    }
}
