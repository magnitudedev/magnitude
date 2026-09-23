use super::{AttentionKernels, DenseKernels};
use crate::HeadBinding;
use magnitude_model_kernels::{head_logits_rows, qwen_features_rows, qwen_head_rows};
use seismic::NativeKernel;
use std::collections::HashMap;

/// Immutable draft-head specializations. A single semantic head key owns all
/// native entries needed by that head block.
#[derive(Debug, Default)]
pub struct HeadKernels {
    pub(super) input: HashMap<HeadBinding, NativeKernel<qwen_head_rows::Entry>>,
    pub(super) attention: HashMap<HeadBinding, AttentionKernels>,
    pub(super) dense: HashMap<HeadBinding, DenseKernels>,
    pub(super) features: HashMap<HeadBinding, NativeKernel<qwen_features_rows::Entry>>,
    pub(super) logits: HashMap<HeadBinding, NativeKernel<head_logits_rows::Entry>>,
}
