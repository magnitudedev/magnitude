use crate::{VisionBlockBinding, VisionMergerBinding, VisionPatchBinding};
use magnitude_model_kernels::{qwen_vision_block, qwen_vision_merger, qwen_vision_stem};
use seismic::NativeKernel;
use std::collections::HashMap;

/// The static dimensions a vision entry's implementations may declare, from
/// the projector geometry (each backend declares the ones it specializes on).
#[derive(Clone, Copy, Debug)]
pub(super) struct VisionStatics {
    pub stem: [(&'static str, u64); 3],
    pub block: [(&'static str, u64); 3],
    pub merger: [(&'static str, u64); 3],
}

impl VisionStatics {
    pub fn of(geometry: &magnitude_model_contracts::VisionGeometry) -> Self {
        let hidden = geometry.hidden;
        Self {
            stem: [
                ("C", geometry.channels),
                ("P", geometry.patch),
                ("H", hidden),
            ],
            block: [
                ("H", geometry.heads),
                ("P", hidden / geometry.heads / 4),
                ("F", geometry.intermediate),
            ],
            merger: [
                ("G", geometry.merge * geometry.merge),
                ("H", hidden),
                ("D", geometry.output_hidden),
            ],
        }
    }
}

/// Immutable vision-stage specializations prepared before any projector
/// weight is imported.
#[derive(Debug, Default)]
pub struct VisionKernels {
    pub(super) stem: HashMap<VisionPatchBinding, NativeKernel<qwen_vision_stem::Entry>>,
    pub(super) blocks: HashMap<VisionBlockBinding, NativeKernel<qwen_vision_block::Entry>>,
    pub(super) merger: HashMap<VisionMergerBinding, NativeKernel<qwen_vision_merger::Entry>>,
}
