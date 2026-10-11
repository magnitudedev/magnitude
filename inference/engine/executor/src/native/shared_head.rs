use magnitude_kernels::{
    cluster_token_rows, draft_top_k, place_logits_rows, project_rows, readout_features_rows,
    readout_project_rows, tap_rows,
};
use seismic::NativeKernel;

/// The specializations of a draft head reading target history apart from its
/// decoder blocks: joining the embedding and hidden rows, the projection
/// into the head's width, its normalized output, its own vocabulary table
/// and the projection back to the target's width. Its blocks' attention and
/// dense entries are decoder block entries, prepared by binding beside the
/// target's; token selection is the target's.
#[derive(Debug, Default)]
pub struct SharedHeadKernels {
    pub(super) join: Option<NativeKernel<tap_rows::Entry>>,
    pub(super) input: Option<NativeKernel<project_rows::Entry>>,
    pub(super) features: Option<NativeKernel<readout_features_rows::Entry>>,
    pub(super) projection: Option<super::ProjectionKernels>,
    pub(super) centroids: Option<NativeKernel<readout_project_rows::Entry>>,
    pub(super) cluster_selection: Option<NativeKernel<draft_top_k::Entry>>,
    pub(super) cluster_rows: Option<NativeKernel<cluster_token_rows::Entry>>,
    pub(super) placement: Option<NativeKernel<place_logits_rows::Entry>>,
    pub(super) chain: Option<NativeKernel<project_rows::Entry>>,
}
