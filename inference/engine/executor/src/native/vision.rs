use crate::{CatalogFailure, VisionEntry, VisionKernel};
use magnitude_kernels::{
    post_norm_residual, vision_attention, vision_clamp, vision_linear, vision_norm,
    vision_patch_stem, vision_pool, vision_position,
};
use seismic::{Element, NativeKernel};
use std::collections::HashMap;

/// The element bound to `name` in a vision kernel's bindings.
pub(super) fn element(
    entry: &'static str,
    kernel: &VisionKernel,
    name: &str,
) -> Result<Element, CatalogFailure> {
    kernel
        .elements
        .iter()
        .find(|(bound, _)| *bound == name)
        .map(|(_, element)| *element)
        .ok_or_else(|| CatalogFailure::Preparation {
            entry,
            bindings: format!("{kernel:?}"),
            outcome: format!("the vision kernel binds no element {name}"),
        })
}

/// Immutable vision-stage specializations, one per distinct kernel of the
/// projector's vision program, prepared before any projector weight is
/// imported.
#[derive(Clone, Debug, Default)]
pub struct VisionKernels {
    pub(crate) patch_stem: HashMap<VisionKernel, NativeKernel<vision_patch_stem::Entry>>,
    pub(crate) norm: HashMap<VisionKernel, NativeKernel<vision_norm::Entry>>,
    pub(crate) linear: HashMap<VisionKernel, NativeKernel<vision_linear::Entry>>,
    pub(crate) clamp: HashMap<VisionKernel, NativeKernel<vision_clamp::Entry>>,
    pub(crate) attention: HashMap<VisionKernel, NativeKernel<vision_attention::Entry>>,
    pub(crate) pool: HashMap<VisionKernel, NativeKernel<vision_pool::Entry>>,
    pub(crate) position: HashMap<VisionKernel, NativeKernel<vision_position::Entry>>,
    pub(crate) post_norm: HashMap<VisionKernel, NativeKernel<post_norm_residual::Entry>>,
}

impl VisionKernels {
    /// Whether `kernel` has been specialized.
    pub(crate) fn contains(&self, kernel: &VisionKernel) -> bool {
        match kernel.entry {
            VisionEntry::PatchStem => self.patch_stem.contains_key(kernel),
            VisionEntry::Norm => self.norm.contains_key(kernel),
            VisionEntry::Linear => self.linear.contains_key(kernel),
            VisionEntry::Clamp => self.clamp.contains_key(kernel),
            VisionEntry::Attention => self.attention.contains_key(kernel),
            VisionEntry::Pool => self.pool.contains_key(kernel),
            VisionEntry::Position => self.position.contains_key(kernel),
            VisionEntry::PostNormResidual => self.post_norm.contains_key(kernel),
        }
    }
}
