use crate::{
    AttentionBinding, DenseBinding, EmbeddingBinding, FeaturesBinding, ReadoutBinding,
    RecurrentBinding, RoutedBinding,
};
use magnitude_model_kernels::{
    qwen_attention_decode, qwen_attention_decode_k8v4, attention_output,
    qwen_attention_prefill, qwen_attention_prefill_k8v4, qwen_attention_project, qwen_dense_expand, qwen_dense_output, embedding_rows,
    readout_features_rows, readout_head_rows,
    qwen_recurrent_chunk, qwen_recurrent_output, qwen_recurrent_project, qwen_recurrent_step,
    qwen_routed_combine, qwen_routed_expand,
    qwen_routed_experts, qwen_routed_group, qwen_routed_output, qwen_routed_route, readout_selected_rows,
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
    pub route: NativeKernel<qwen_routed_route::Entry>,
    /// Decode form (row classes up to the GEMV bound).
    pub expand: NativeKernel<qwen_routed_expand::Entry>,
    pub output: NativeKernel<qwen_routed_output::Entry>,
    /// Grouped form (larger row classes).
    pub group: NativeKernel<qwen_routed_group::Entry>,
    pub experts: NativeKernel<qwen_routed_experts::Entry>,
    pub combine: NativeKernel<qwen_routed_combine::Entry>,
}

/// An attention block: normed Q/K/V projection, the fused attention entry
/// over the block's history codec, output projection plus residual.
#[derive(Clone, Debug)]
pub struct AttentionKernels {
    pub project: NativeKernel<qwen_attention_project::Entry>,
    pub history: AttentionHistoryKernels,
    pub output: NativeKernel<attention_output::Entry>,
}

/// The fused attention entries of one history codec: `decode` for decode row
/// classes, `prefill` for the rest. The codec fixes the history planes the
/// entries take, in the order of the codec's plane descriptors.
#[derive(Clone, Debug)]
pub enum AttentionHistoryKernels {
    Dense {
        decode: NativeKernel<qwen_attention_decode::Entry>,
        prefill: NativeKernel<qwen_attention_prefill::Entry>,
    },
    AffineK8V4 {
        decode: NativeKernel<qwen_attention_decode_k8v4::Entry>,
        prefill: NativeKernel<qwen_attention_prefill_k8v4::Entry>,
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
    pub expand: NativeKernel<qwen_dense_expand::Entry>,
    pub output: NativeKernel<qwen_dense_output::Entry>,
}

/// A recurrent block: normed projection, the state advance (row-sequential
/// `step` for small row classes, chunked for the rest), gated output.
#[derive(Clone, Debug)]
pub struct RecurrentKernels {
    pub project: NativeKernel<qwen_recurrent_project::Entry>,
    pub step: NativeKernel<qwen_recurrent_step::Entry>,
    pub chunk: NativeKernel<qwen_recurrent_chunk::Entry>,
    pub output: NativeKernel<qwen_recurrent_output::Entry>,
}
