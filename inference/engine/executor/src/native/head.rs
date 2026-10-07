use super::{AttentionKernels, DenseKernels, ProgressiveReadoutKernels, RoutedKernels};
use crate::HeadBinding;
use magnitude_kernels::{
    attention_project, draft_rows, head_logits_rows, readout_features_rows, sample_rows, shape_rows,
};
use seismic::NativeKernel;
use std::collections::HashMap;

/// Immutable draft-head specializations. A single semantic head key owns all
/// native entries needed by that head block.
#[derive(Debug, Default)]
pub struct HeadKernels {
    pub(super) input: HashMap<HeadBinding, NativeKernel<draft_rows::Entry>>,
    pub(super) attention: HashMap<HeadBinding, AttentionKernels>,
    pub(super) priming_project: HashMap<HeadBinding, NativeKernel<attention_project::Entry>>,
    pub(super) dense: HashMap<HeadBinding, DenseKernels>,
    pub(super) routed: HashMap<HeadBinding, RoutedKernels>,
    pub(super) features: HashMap<HeadBinding, NativeKernel<readout_features_rows::Entry>>,
    pub(super) logits: HashMap<HeadBinding, HeadLogitsKernels>,
    /// Token selection over the draft vocabulary, shared by every block.
    pub(super) shape: Option<NativeKernel<shape_rows::Entry>>,
    pub(super) sample: Option<NativeKernel<sample_rows::Entry>>,
}

/// A head block's projection onto the draft vocabulary (`HeadProjection`).
#[derive(Clone, Debug)]
pub enum HeadLogitsKernels {
    /// `head_logits_rows` over the packed output projection.
    Packed(NativeKernel<head_logits_rows::Entry>),
    /// Every vocabulary row of the progressive planes: a certified
    /// selection's levels for drafting slots up to the backend's certified
    /// bound, the full exact pass beyond.
    Progressive(ProgressiveReadoutKernels),
}
