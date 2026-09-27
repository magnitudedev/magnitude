use super::{AttentionKernels, DenseKernels};
use crate::{AttentionBinding, DenseBinding};
use magnitude_kernels::{
    dense_output, draft_confidence, embedding_rows, readout_features_rows, readout_head_rows,
};
use seismic::NativeKernel;
use std::collections::HashMap;

/// Immutable specializations of a separate draft (DFlash, DSpark), one per
/// distinct binding. Token selection reuses the target's entries (the draft
/// selects over the whole vocabulary through the target's projection).
#[derive(Debug, Default)]
pub struct DraftKernels {
    /// The layers' block attention and their context injection (the same
    /// attention with the fusion norm as its input norm).
    pub(super) attention: HashMap<AttentionBinding, AttentionKernels>,
    pub(super) dense: HashMap<DenseBinding, DenseKernels>,
    /// Embeds the block's tokens (raw rows).
    pub(super) embedding: Option<NativeKernel<embedding_rows::Entry>>,
    /// The output norm and the target's vocabulary projection of the
    /// proposing rows.
    pub(super) head: Option<NativeKernel<readout_head_rows::Entry>>,
    pub(super) markov: Option<MarkovKernels>,
}

/// One draft layer's entries.
#[derive(Clone, Debug)]
pub struct DraftBlockKernels {
    pub attention: AttentionKernels,
    pub injection: AttentionKernels,
    pub dense: DenseKernels,
}

/// DSpark's chain: the Markov memory of the previous token, its projection
/// onto the slot logits, and the slot's confidence over its output-normed
/// row.
#[derive(Clone, Debug)]
pub struct MarkovKernels {
    pub embedding: NativeKernel<embedding_rows::Entry>,
    pub projection: NativeKernel<dense_output::Entry>,
    pub features: NativeKernel<readout_features_rows::Entry>,
    pub confidence: NativeKernel<draft_confidence::Entry>,
}
