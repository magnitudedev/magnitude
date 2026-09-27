use crate::{
    AttentionBinding, DenseBinding, EmbeddingBinding, FeaturesBinding, ReadoutBinding,
    RecurrentBinding, RoutedBinding,
};
use magnitude_model_kernels::{
    attention_output, dense_expand, dense_output, embedding_rows, gated_attention_decode,
    gated_attention_decode_k8v4, gated_attention_prefill, gated_attention_prefill_k8v4,
    gated_attention_project, gated_delta_chunk, gated_delta_output, gated_delta_project,
    gated_delta_step, readout_features_rows, readout_head_rows, readout_selected_rows,
    routed_combine, routed_expand, routed_experts, routed_group, routed_output, routed_route,
};
use seismic::NativeKernel;
use std::collections::HashMap;

/// Immutable decoder-stage specializations keyed by semantic bindings.
#[derive(Debug, Default)]
pub struct TargetKernels {
    pub(super) embedding: HashMap<EmbeddingBinding, NativeKernel<embedding_rows::Entry>>,
    pub(super) attention: HashMap<AttentionBinding, AttentionKernels>,
    pub(super) recurrent: HashMap<RecurrentBinding, RecurrentKernels>,
    pub(super) dense: HashMap<DenseBinding, DenseKernels>,
    pub(super) routed: HashMap<RoutedBinding, RoutedKernels>,
    pub(super) readout: HashMap<ReadoutBinding, ReadoutKernels>,
    pub(super) features: HashMap<FeaturesBinding, NativeKernel<readout_features_rows::Entry>>,
    pub(super) selected: HashMap<ReadoutBinding, NativeKernel<readout_selected_rows::Entry>>,
}

/// The target readout: final-norm features, and the head projection that
/// normalizes its own rows.
#[derive(Clone, Debug)]
pub struct ReadoutKernels {
    pub features: NativeKernel<readout_features_rows::Entry>,
    pub head: NativeKernel<readout_head_rows::Entry>,
}

#[derive(Clone, Debug)]
pub struct RoutedKernels {
    pub route: NativeKernel<routed_route::Entry>,
    /// Decode form (row classes up to the GEMV bound).
    pub expand: NativeKernel<routed_expand::Entry>,
    pub output: NativeKernel<routed_output::Entry>,
    /// Grouped form (larger row classes).
    pub group: NativeKernel<routed_group::Entry>,
    pub experts: NativeKernel<routed_experts::Entry>,
    pub combine: NativeKernel<routed_combine::Entry>,
}

/// An attention block: normed Q/K/V projection, the fused attention entry
/// over the block's history codec, output projection plus residual.
#[derive(Clone, Debug)]
pub struct AttentionKernels {
    pub project: NativeKernel<gated_attention_project::Entry>,
    pub history: AttentionHistoryKernels,
    pub output: NativeKernel<attention_output::Entry>,
}

/// The fused attention entries of one history codec: `decode` for decode row
/// classes, `prefill` for the rest. The codec fixes the history planes the
/// entries take, in the order of the codec's plane descriptors.
#[derive(Clone, Debug)]
pub enum AttentionHistoryKernels {
    Dense {
        decode: NativeKernel<gated_attention_decode::Entry>,
        prefill: NativeKernel<gated_attention_prefill::Entry>,
    },
    AffineK8V4 {
        decode: NativeKernel<gated_attention_decode_k8v4::Entry>,
        prefill: NativeKernel<gated_attention_prefill_k8v4::Entry>,
    },
}

impl AttentionHistoryKernels {
    pub fn invocation_workspace_bytes(&self) -> u64 {
        match self {
            Self::Dense { decode, prefill } => {
                decode.invocation_workspace_bytes() + prefill.invocation_workspace_bytes()
            }
            Self::AffineK8V4 { decode, prefill } => {
                decode.invocation_workspace_bytes() + prefill.invocation_workspace_bytes()
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct DenseKernels {
    pub expand: NativeKernel<dense_expand::Entry>,
    pub output: NativeKernel<dense_output::Entry>,
}

/// A recurrent block: normed projection, the state advance (row-sequential
/// `step` for small row classes, chunked for the rest), gated output.
#[derive(Clone, Debug)]
pub struct RecurrentKernels {
    pub project: NativeKernel<gated_delta_project::Entry>,
    pub step: NativeKernel<gated_delta_step::Entry>,
    pub chunk: NativeKernel<gated_delta_chunk::Entry>,
    pub output: NativeKernel<gated_delta_output::Entry>,
}
