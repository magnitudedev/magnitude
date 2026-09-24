use super::{AttentionKernels, DenseKernels};
use crate::HeadBinding;
use magnitude_model_kernels::{
    head_logits_rows, qwen_draft_rows, readout_features_rows, sample_rows, shape_rows,
};
use seismic::NativeKernel;
use std::collections::HashMap;

/// Proposals are drawn from the leading token ids only: byte-pair
/// vocabularies number their pieces in merge order, so the leading ids are
/// the frequent ones, and the draft's vocabulary projection (the largest
/// part of a draft step) streams a fraction of the output weight. Keeping
/// ids equal to indices keeps the draft's position-keyed selection coupled
/// to the target's. Verification selects over the whole vocabulary, so this
/// bounds only which tokens can be proposed, never which are emitted.
/// Qwen3.5 English prose: 84% of emitted tokens lie below 32768, 96% below
/// 65536, 99% below 131072.
const DRAFT_VOCABULARY: u64 = 65_536;

/// The vocabulary the head's `head_logits_rows` projects onto.
pub(crate) fn draft_vocabulary(vocabulary: u64) -> u64 {
    vocabulary.min(DRAFT_VOCABULARY)
}

/// Immutable draft-head specializations. A single semantic head key owns all
/// native entries needed by that head block.
#[derive(Debug, Default)]
pub struct HeadKernels {
    pub(super) input: HashMap<HeadBinding, NativeKernel<qwen_draft_rows::Entry>>,
    pub(super) attention: HashMap<HeadBinding, AttentionKernels>,
    pub(super) dense: HashMap<HeadBinding, DenseKernels>,
    pub(super) features: HashMap<HeadBinding, NativeKernel<readout_features_rows::Entry>>,
    pub(super) logits: HashMap<HeadBinding, NativeKernel<head_logits_rows::Entry>>,
    /// Token selection over the draft vocabulary, shared by every block.
    pub(super) shape: Option<NativeKernel<shape_rows::Entry>>,
    pub(super) sample: Option<NativeKernel<sample_rows::Entry>>,
}
