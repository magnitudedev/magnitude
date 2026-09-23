use crate::{VisionBlockBinding, VisionMergerBinding, VisionPatchBinding};
use magnitude_model_kernels::{
    qwen_vision_block, qwen_vision_feature_output, qwen_vision_merger, qwen_vision_stem,
};
use seismic::NativeKernel;
use std::collections::HashMap;

/// Immutable vision-stage specializations prepared before any projector
/// weight is imported.
#[derive(Debug, Default)]
pub struct VisionKernels {
    pub(super) stem: HashMap<VisionPatchBinding, NativeKernel<qwen_vision_stem::Entry>>,
    pub(super) blocks: HashMap<VisionBlockBinding, NativeKernel<qwen_vision_block::Entry>>,
    pub(super) merger: HashMap<VisionMergerBinding, NativeKernel<qwen_vision_merger::Entry>>,
    pub(super) output:
        HashMap<VisionMergerBinding, NativeKernel<qwen_vision_feature_output::Entry>>,
}
