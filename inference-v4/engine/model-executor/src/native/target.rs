use crate::{
    AttentionBinding, DenseBinding, EmbeddingBinding, FeaturesBinding, ReadoutBinding,
    RecurrentBinding, RoutedBinding,
};
use magnitude_model_kernels::{
    head_logits_rows, qwen_attention_attend, qwen_attention_normalize, qwen_attention_output,
    qwen_attention_prepare, qwen_attention_project, qwen_dense_expand, qwen_dense_expand_demanded,
    qwen_dense_output, qwen_dense_output_demanded, qwen_embedding_rows, qwen_features_rows,
    qwen_recurrent_mix, qwen_recurrent_normalize, qwen_recurrent_output, qwen_recurrent_prepare,
    qwen_recurrent_project, qwen_recurrent_scan, qwen_routed_expand, qwen_routed_logits,
    qwen_routed_normalize, qwen_routed_output, qwen_routed_select, qwen_selected_rows,
};
use seismic::NativeKernel;
use std::collections::HashMap;

/// Immutable decoder-stage specializations keyed by semantic bindings.
#[derive(Debug, Default)]
pub struct TargetKernels {
    pub(super) embedding: HashMap<EmbeddingBinding, NativeKernel<qwen_embedding_rows::Entry>>,
    pub(super) attention: HashMap<AttentionBinding, AttentionKernels>,
    pub(super) recurrent: HashMap<RecurrentBinding, RecurrentKernels>,
    pub(super) dense: HashMap<DenseBinding, DenseKernels>,
    pub(super) routed: HashMap<RoutedBinding, RoutedKernels>,
    pub(super) readout: HashMap<ReadoutBinding, ReadoutKernels>,
    pub(super) features: HashMap<FeaturesBinding, NativeKernel<qwen_features_rows::Entry>>,
    pub(super) selected: HashMap<ReadoutBinding, NativeKernel<qwen_selected_rows::Entry>>,
}

#[derive(Clone, Debug)]
pub struct ReadoutKernels {
    pub features: NativeKernel<qwen_features_rows::Entry>,
    pub logits: NativeKernel<head_logits_rows::Entry>,
}

#[derive(Clone, Debug)]
pub struct RoutedKernels {
    pub normalize: NativeKernel<qwen_routed_normalize::Entry>,
    pub logits: NativeKernel<qwen_routed_logits::Entry>,
    pub select: NativeKernel<qwen_routed_select::Entry>,
    pub expand: NativeKernel<qwen_routed_expand::Entry>,
    pub output: NativeKernel<qwen_routed_output::Entry>,
}

#[derive(Clone, Debug)]
pub struct AttentionKernels {
    pub normalize: NativeKernel<qwen_attention_normalize::Entry>,
    pub project: NativeKernel<qwen_attention_project::Entry>,
    pub prepare: NativeKernel<qwen_attention_prepare::Entry>,
    pub attend: NativeKernel<qwen_attention_attend::Entry>,
    pub output: NativeKernel<qwen_attention_output::Entry>,
}

#[derive(Clone, Debug)]
pub struct DenseKernels {
    pub expand: NativeKernel<qwen_dense_expand::Entry>,
    pub output: NativeKernel<qwen_dense_output::Entry>,
    pub expand_demanded: NativeKernel<qwen_dense_expand_demanded::Entry>,
    pub output_demanded: NativeKernel<qwen_dense_output_demanded::Entry>,
}

#[derive(Clone, Debug)]
pub struct RecurrentKernels {
    pub normalize: NativeKernel<qwen_recurrent_normalize::Entry>,
    pub project: NativeKernel<qwen_recurrent_project::Entry>,
    pub prepare: NativeKernel<qwen_recurrent_prepare::Entry>,
    pub scan: NativeKernel<qwen_recurrent_scan::Entry>,
    pub mix: NativeKernel<qwen_recurrent_mix::Entry>,
    pub output: NativeKernel<qwen_recurrent_output::Entry>,
}
