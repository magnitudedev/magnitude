use crate::{
    AttentionBinding, DenseBinding, EmbeddingBinding, FeaturesBinding, GeneralRoutedBinding,
    ReadoutBinding, RecurrentBinding, RoutedBinding, ShortConvBinding, StateSpaceBinding,
};
use magnitude_kernels::{
    attention_decode, attention_decode_k8v4, attention_output, attention_prefill,
    attention_prefill_k8v4, attention_project, dense_expand, dense_output, embedding_rows,
    gated_delta_chunk, gated_delta_output, gated_delta_project, gated_delta_step,
    post_norm_residual, project_rows, readout_features_rows, readout_head_rows,
    readout_selected_rows, routed_combine, routed_expand, routed_experts, routed_group,
    routed_output, routed_route,
};
use magnitude_kernels::{
    dense_up, feature_rows, routed_down, routed_experts_up, routed_gate_up, routed_scatter,
    routed_select, routed_up, state_space_chunk, state_space_gate, state_space_step, tap_rows,
};
use magnitude_kernels::{short_conv_project, short_conv_rows};
use seismic::NativeKernel;
use std::collections::HashMap;

/// Immutable decoder-stage specializations keyed by semantic bindings.
#[derive(Debug, Default)]
pub struct TargetKernels {
    pub(super) embedding: HashMap<EmbeddingBinding, NativeKernel<embedding_rows::Entry>>,
    pub(super) attention: HashMap<AttentionBinding, AttentionKernels>,
    pub(super) recurrent: HashMap<RecurrentBinding, RecurrentKernels>,
    pub(super) state_space: HashMap<StateSpaceBinding, StateSpaceKernels>,
    pub(super) short_conv: HashMap<ShortConvBinding, ShortConvKernels>,
    pub(super) general_routed: HashMap<GeneralRoutedBinding, GeneralRoutedKernels>,
    pub(super) dense: HashMap<DenseBinding, DenseKernels>,
    pub(super) routed: HashMap<RoutedBinding, RoutedKernels>,
    pub(super) readout: HashMap<ReadoutBinding, ReadoutKernels>,
    pub(super) features: HashMap<FeaturesBinding, NativeKernel<readout_features_rows::Entry>>,
    pub(super) selected: HashMap<ReadoutBinding, NativeKernel<readout_selected_rows::Entry>>,
    pub(super) taps: Option<TapKernels>,
    pub(super) per_layer_entry: HashMap<crate::PerLayerEntryBinding, PerLayerEntryKernels>,
    pub(super) per_layer: HashMap<crate::PerLayerBinding, PerLayerKernels>,
    pub(super) parallel: HashMap<crate::ParallelBinding, ParallelKernels>,
}

/// A dense branch beside a general routed branch: the dense branch's
/// expansion and its down projection into F32, the routed branch summed onto
/// zeros, and `moe_tail`, which joins both through the sublayer's post-norm
/// tail.
#[derive(Clone, Debug)]
pub struct ParallelKernels {
    pub expand: NativeKernel<dense_expand::Entry>,
    pub down: NativeKernel<magnitude_kernels::project_rows::Entry>,
    pub routed: GeneralRoutedKernels,
    pub tail: NativeKernel<magnitude_kernels::moe_tail::Entry>,
}

/// The per-layer entry (Gemma PLE), one graph after the embedding and its
/// media overlays:
/// - the embedded rows rounded to activations (`import_dense` F32 → A) and
///   projected to every layer's channels (`project_rows` into F32);
/// - the batch rows' uploaded host-table rows converted to the table's
///   resident representation (`table`);
/// - their combination (`per_layer_inputs`), copied into the program's
///   per-layer rows every block reads (`conditioning_overlay`, an F32 row
///   copy).
#[derive(Clone, Debug)]
pub struct PerLayerEntryKernels {
    pub round: NativeKernel<magnitude_kernels::import_dense::Entry>,
    pub project: NativeKernel<magnitude_kernels::project_rows::Entry>,
    pub table: TableConversion,
    pub inputs: NativeKernel<magnitude_kernels::per_layer_inputs::Entry>,
    pub copy: NativeKernel<magnitude_kernels::conditioning_overlay::Entry>,
}

/// The conversion of uploaded host-table rows into their resident form.
#[derive(Clone, Debug)]
pub enum TableConversion {
    Dense(NativeKernel<magnitude_kernels::import_dense::Entry>),
    Repack(NativeKernel<magnitude_kernels::repack_weight::Entry>),
}

/// A per-layer input sublayer: the gate over the layer's slice of the
/// per-layer inputs, and the post-norm tail of its projection.
#[derive(Clone, Debug)]
pub struct PerLayerKernels {
    pub gate: NativeKernel<magnitude_kernels::per_layer_gate::Entry>,
    pub output: PostNormKernels,
}

/// A separate draft's target taps: the tap of a tapped block's (or the
/// readout's exit) residual rows, and the readout's fusion of the taps into
/// the draft's conditioning features.
#[derive(Clone, Debug)]
pub struct TapKernels {
    pub tap: NativeKernel<tap_rows::Entry>,
    pub fusion: NativeKernel<project_rows::Entry>,
    pub features: NativeKernel<feature_rows::Entry>,
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

/// An attention block: normed query/gate/key/value projection, the fused
/// attention entry over the block's history codec, output projection plus
/// residual.
#[derive(Clone, Debug)]
pub struct AttentionKernels {
    pub project: NativeKernel<attention_project::Entry>,
    pub history: AttentionHistoryKernels,
    pub output: SublayerOutput<NativeKernel<attention_output::Entry>>,
}

/// A sublayer's output entries by its `SublayerTail`: the operator's own
/// output projection plus residual (`K`), or the F32 projection and the
/// post-norm row op.
#[derive(Clone, Debug)]
pub enum SublayerOutput<K> {
    Residual(K),
    PostNorm(PostNormKernels),
}

#[derive(Clone, Debug)]
pub struct PostNormKernels {
    pub project: NativeKernel<project_rows::Entry>,
    pub residual: NativeKernel<post_norm_residual::Entry>,
}

impl<E: seismic::Entry> SublayerOutput<NativeKernel<E>> {
    pub fn invocation_workspace_bytes(&self) -> u64 {
        match self {
            Self::Residual(kernel) => kernel.invocation_workspace_bytes(),
            Self::PostNorm(kernels) => {
                kernels.project.invocation_workspace_bytes()
                    + kernels.residual.invocation_workspace_bytes()
            }
        }
    }
}

/// The fused attention entries of one history codec: `decode` for decode row
/// classes, `prefill` for the rest. The codec fixes the history planes the
/// entries take, in the order of the codec's plane descriptors.
#[derive(Clone, Debug)]
pub enum AttentionHistoryKernels {
    Dense {
        decode: NativeKernel<attention_decode::Entry>,
        /// The multi-row decode (draft blocks, verification), tuned on its
        /// own row classes so it can read each history tile once for all
        /// rows.
        verify: Option<NativeKernel<attention_decode::Entry>>,
        prefill: NativeKernel<attention_prefill::Entry>,
    },
    AffineK8V4 {
        decode: NativeKernel<attention_decode_k8v4::Entry>,
        verify: Option<NativeKernel<attention_decode_k8v4::Entry>>,
        verify_four: Option<NativeKernel<attention_decode_k8v4::Entry>>,
        verify_eight: Option<NativeKernel<attention_decode_k8v4::Entry>>,
        prefill: NativeKernel<attention_prefill_k8v4::Entry>,
    },
}

impl AttentionHistoryKernels {
    pub fn invocation_workspace_bytes(&self) -> u64 {
        match self {
            Self::Dense {
                decode,
                verify,
                prefill,
            } => {
                decode
                    .invocation_workspace_bytes()
                    .max(verify.as_ref().map_or(0, |kernel| kernel.invocation_workspace_bytes()))
                    + prefill.invocation_workspace_bytes()
            }
            Self::AffineK8V4 {
                decode,
                verify,
                verify_four,
                verify_eight,
                prefill,
            } => {
                decode
                    .invocation_workspace_bytes()
                    .max(
                        verify
                            .as_ref()
                            .map_or(0, NativeKernel::invocation_workspace_bytes),
                    )
                    .max(
                        verify_four
                            .as_ref()
                            .map_or(0, NativeKernel::invocation_workspace_bytes),
                    )
                    .max(
                        verify_eight
                            .as_ref()
                            .map_or(0, NativeKernel::invocation_workspace_bytes),
                    )
                    + prefill.invocation_workspace_bytes()
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct DenseKernels {
    pub expand: NativeKernel<dense_expand::Entry>,
    pub output: SublayerOutput<NativeKernel<dense_output::Entry>>,
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

/// A general routed feed-forward (`operators::routed`).
#[derive(Clone, Debug)]
pub struct GeneralRoutedKernels {
    pub select: NativeKernel<routed_select::Entry>,
    pub experts: ExpertKernels,
    /// Decode rows: the weighted down projections onto the base.
    pub down: NativeKernel<routed_down::Entry>,
    /// Grouped rows: tiles, and the unpermuted weighted sum onto the base.
    pub group: NativeKernel<routed_group::Entry>,
    pub scatter: NativeKernel<routed_scatter::Entry>,
    /// The shared expert: expansion, then its down projection onto the
    /// residual.
    pub shared: Option<(DenseExpansionKernel, NativeKernel<dense_output::Entry>)>,
    /// A latent operator's down projection (into activations) and up
    /// projection onto the base.
    pub latent: Option<(
        NativeKernel<project_rows::Entry>,
        NativeKernel<dense_output::Entry>,
    )>,
}

/// The routed experts' expansion entries: decode, then grouped.
#[derive(Clone, Debug)]
pub enum ExpertKernels {
    Gated {
        decode: NativeKernel<routed_gate_up::Entry>,
        grouped: NativeKernel<routed_experts::Entry>,
    },
    Plain {
        decode: NativeKernel<routed_up::Entry>,
        grouped: NativeKernel<routed_experts_up::Entry>,
    },
}

/// A dense expansion: gated (`dense_expand`) or up-only (`dense_up`).
#[derive(Clone, Debug)]
pub enum DenseExpansionKernel {
    Gated(NativeKernel<dense_expand::Entry>),
    Plain(NativeKernel<dense_up::Entry>),
}

/// A state-space block (`operators::state_space`): the normed projection
/// row, the state advance (`step` for small row classes, chunked for the
/// rest), the gated group norm, and the output projection plus residual.
#[derive(Clone, Debug)]
pub struct StateSpaceKernels {
    pub project: NativeKernel<attention_project::Entry>,
    pub step: NativeKernel<state_space_step::Entry>,
    pub chunk: NativeKernel<state_space_chunk::Entry>,
    pub gate: NativeKernel<state_space_gate::Entry>,
    pub output: NativeKernel<attention_output::Entry>,
}

/// A short-convolution block (`operators::short_conv`): the normed
/// `u | C` projection, the gated convolution with its window publication,
/// and the output projection plus residual.
#[derive(Clone, Debug)]
pub struct ShortConvKernels {
    pub project: NativeKernel<short_conv_project::Entry>,
    pub rows: NativeKernel<short_conv_rows::Entry>,
    pub output: NativeKernel<attention_output::Entry>,
}
