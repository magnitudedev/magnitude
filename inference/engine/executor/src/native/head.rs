use super::{AttentionKernels, DenseKernels, GeneralRoutedKernels, ProjectionKernels, RoutedKernels};
use crate::HeadBinding;
use magnitude_kernels::{
    attention_project, draft_rows, draft_top_k, guess_chain, place_logits_rows,
    readout_features_rows, readout_project_rows, sample_rows, shape_rows, shortlist_token_rows,
};
use seismic::NativeKernel;
use std::collections::HashMap;

/// Narrow coarse-to-exact shortlist readout kernels for a draft head block:
/// `coarse` projects features onto the narrow 2-bit table (`I = 0`),
/// `selection` (`draft_top_k`) picks the top-K candidates, `rows`
/// (`shortlist_token_rows`) extracts canonical token IDs for gathered exact
/// refinement (`logits` with `I = 1, S = K`), and `placement`
/// (`place_logits_rows`) scatters the exact logits back into `[O, V]`.
#[derive(Clone, Debug)]
pub(crate) struct ShortlistReadoutKernels {
    pub coarse: super::RangedKernels<readout_project_rows::Entry>,
    pub selection: NativeKernel<draft_top_k::Entry>,
    pub rows: NativeKernel<shortlist_token_rows::Entry>,
    pub placement: NativeKernel<place_logits_rows::Entry>,
}

/// Immutable draft-head specializations. A single semantic head key owns all
/// native entries needed by that head block.
#[derive(Debug, Default)]
pub struct HeadKernels {
    pub(super) streams: HashMap<HeadBinding, super::StreamsKernels>,
    pub(super) copy: HashMap<HeadBinding, NativeKernel<magnitude_kernels::feature_copy::Entry>>,
    pub(super) gather: HashMap<HeadBinding, NativeKernel<magnitude_kernels::feature_rows::Entry>>,
    pub(super) input: HashMap<HeadBinding, super::RangedKernels<draft_rows::Entry>>,
    pub(super) attention: HashMap<HeadBinding, AttentionKernels>,
    pub(super) priming_project: HashMap<HeadBinding, super::RangedKernels<attention_project::Entry>>,
    pub(super) dense: HashMap<HeadBinding, DenseKernels>,
    pub(super) routed: HashMap<HeadBinding, RoutedKernels>,
    pub(super) general_routed: HashMap<HeadBinding, GeneralRoutedKernels>,
    pub(super) features: HashMap<HeadBinding, NativeKernel<readout_features_rows::Entry>>,
    pub(super) logits: HashMap<HeadBinding, ProjectionKernels>,
    pub(super) shortlist: HashMap<HeadBinding, ShortlistReadoutKernels>,
    /// Token selection over the draft vocabulary, shared by every block.
    pub(super) shape: Option<NativeKernel<shape_rows::Entry>>,
    pub(super) sample: Option<NativeKernel<sample_rows::Entry>>,
    /// Two-row decode's chain at a front launch's tail, of a load that
    /// fronts (`ProgramPlan::fronts`).
    pub(super) chain: Option<NativeKernel<guess_chain::Entry>>,
}
