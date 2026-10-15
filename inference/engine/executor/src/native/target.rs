use crate::operators::streams::StreamsBinding;
use crate::{
    AttentionBinding, DenseBinding, EmbeddingBinding, FeaturesBinding, GeneralRoutedBinding,
    ReadoutBinding, RecurrentBinding, RoutedBinding, ShortConvBinding, StateSpaceBinding,
};
use magnitude_kernels::{
    attention_append_dense, attention_append_k8v4, attention_decode, attention_decode_k8v4,
    attention_output, attention_prefill, attention_prefill_k8v4, attention_project, dense_expand,
    dense_output, embedding_rows, gated_delta_chunk, gated_delta_project,
    gated_delta_project_convolved, gated_delta_step, gated_delta_step_convolved, indexer_select,
    post_norm_residual, project_rows, readout_exact_rows, readout_features_rows, readout_head_rows,
    readout_planes_rows, readout_refine_rows, readout_selected_rows, readout_top_rows,
    routed_combine, routed_expand, routed_experts, routed_group, routed_output, routed_route,
    routed_route_shared,
};
use magnitude_kernels::{
    dense_up, feature_rows, routed_down, routed_experts_up, routed_gate_up, routed_scatter,
    routed_select, routed_up, state_space_chunk, state_space_gate, state_space_step, tap_rows,
};
use magnitude_kernels::{short_conv_project, short_conv_rows};
use magnitude_kernels::{
    routed_bank_fill, routed_convert_wide, routed_copy, routed_route_streamed, routed_scratch_request,
    routed_scratch_wait, routed_step,
};
use seismic::NativeKernel;
use std::collections::HashMap;

/// Immutable decoder-stage specializations keyed by semantic bindings.
#[derive(Debug, Default)]
pub struct TargetKernels {
    pub(super) residual_features: Option<NativeKernel<feature_rows::Entry>>,
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
    pub(super) taps: Option<TapKernels>,
    pub(super) per_layer_entry: HashMap<crate::PerLayerEntryBinding, PerLayerEntryKernels>,
    pub(super) per_layer: HashMap<crate::PerLayerBinding, PerLayerKernels>,
    pub(super) parallel: HashMap<crate::ParallelBinding, ParallelKernels>,
    pub(super) streams: HashMap<StreamsBinding, StreamsKernels>,
    pub(super) ngram: HashMap<crate::NGramBinding, NGramKernels>,
}

/// An n-gram input sublayer (`operators::ngram`): the table rows of each
/// position (`rows`), read from the streamed table and decoded to
/// activations (`stored`), their key and value projections into F32, the
/// gate and the row op.
#[derive(Clone, Debug)]
pub struct NGramKernels {
    pub rows: NativeKernel<magnitude_kernels::ngram_rows::Entry>,
    pub stored: NativeKernel<magnitude_kernels::stored_rows::Entry>,
    pub key: NativeKernel<project_rows::Entry>,
    pub value: NativeKernel<project_rows::Entry>,
    pub gate: NativeKernel<magnitude_kernels::ngram_gate::Entry>,
    pub inject: NativeKernel<magnitude_kernels::ngram_inject::Entry>,
    /// The copy of a step's selected tokens where the host reads them
    /// early, in the entry graph of a step that reads selected tokens: the
    /// table's filler learns the rows it names from them.
    pub notice: NativeKernel<magnitude_kernels::selection_notice::Entry>,
}

/// A decoder's hyper-connection streams (`operators::streams`): the
/// sublayers' mix sites (`down`, with inject values) and the exit site
/// (`exit`, without), their second stage (`up`), and the widening of the
/// embedded rows (`repeat`). Every site of a decoder shares one binding.
#[derive(Clone, Debug)]
pub struct StreamsKernels {
    pub down: NativeKernel<magnitude_kernels::stream_mix_down::Entry>,
    pub exit: NativeKernel<magnitude_kernels::stream_mix_down::Entry>,
    pub up: NativeKernel<magnitude_kernels::stream_mix_up::Entry>,
    pub repeat: NativeKernel<magnitude_kernels::repeat_rows::Entry>,
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

/// The target readout: final-norm features, and the vocabulary projection
/// that normalizes its own rows, by the plan's head placement.
#[derive(Clone, Debug)]
pub struct ReadoutKernels {
    pub features: NativeKernel<readout_features_rows::Entry>,
    pub head: ReadoutHeadKernels,
}

/// The vocabulary projection's entries (`ReadoutHead`).
#[derive(Clone, Debug)]
pub enum ReadoutHeadKernels {
    /// The head GEMV, and its projection of selected vocabulary rows.
    Packed {
        head: NativeKernel<readout_head_rows::Entry>,
        selected: NativeKernel<readout_selected_rows::Entry>,
    },
    Qualified {
        packed: Box<ReadoutHeadKernels>,
        progressive: ProgressiveReadoutKernels,
    },
}

impl ReadoutHeadKernels {
    pub(crate) fn invocation_workspace_bytes(&self) -> u128 {
        match self {
            Self::Packed { head, selected } => {
                u128::from(head.invocation_workspace_bytes())
                    + u128::from(selected.invocation_workspace_bytes())
            }
            Self::Qualified {
                packed,
                progressive,
                ..
            } => packed.invocation_workspace_bytes() + progressive_workspace_bytes(progressive),
        }
    }
}

fn progressive_workspace_bytes(kernels: &ProgressiveReadoutKernels) -> u128 {
    u128::from(kernels.top.invocation_workspace_bytes())
        + u128::from(kernels.refine.invocation_workspace_bytes())
        + u128::from(kernels.exact.invocation_workspace_bytes())
        + u128::from(kernels.planes.invocation_workspace_bytes())
}

/// A progressive head's certified levels and its full exact pass.
#[derive(Clone, Debug)]
pub struct ProgressiveReadoutKernels {
    pub top: NativeKernel<readout_top_rows::Entry>,
    pub refine: NativeKernel<readout_refine_rows::Entry>,
    pub exact: NativeKernel<readout_exact_rows::Entry>,
    pub planes: NativeKernel<readout_planes_rows::Entry>,
}

#[derive(Clone, Debug)]
pub struct RoutedKernels {
    /// Every row class of the expand form; the grouped row classes of the
    /// shared-route form.
    pub route: NativeKernel<routed_route::Entry>,
    /// Decode form (row classes up to the GEMV bound).
    pub decode: RoutedDecodeKernels,
    pub output: NativeKernel<routed_output::Entry>,
    /// Grouped form (larger row classes).
    pub group: NativeKernel<routed_group::Entry>,
    pub experts: NativeKernel<routed_experts::Entry>,
    pub combine: NativeKernel<routed_combine::Entry>,
    /// A streamed layer's bank-served decode entries.
    pub streamed: Option<StreamedRoutedKernels>,
}

/// A streamed layer's entries beside the resident ones (spec 4.6, 4.7,
/// 4.9). Its decode rows run the resident shared-route form's `choices` and
/// `output` over the bank and the bound slots, so every value is the
/// resident load's; these entries route with resolve and hints, fill the
/// absent experts, and fill the scratch banks of larger row classes.
#[derive(Clone, Debug)]
pub struct StreamedRoutedKernels {
    pub step: NativeKernel<routed_step::Entry>,
    /// The routing with resolve and the hints of the layers ahead, at the
    /// resident shared route's parameters.
    pub route: NativeKernel<routed_route_streamed::Entry>,
    /// The park and conversion of the absent experts into their slots.
    pub fill: NativeKernel<routed_bank_fill::Entry>,
    /// The scratch-served classes' fill (spec 4.9): the request, a round's
    /// park, and the copy of bank slots per routed tensor.
    pub scratch_request: NativeKernel<routed_scratch_request::Entry>,
    pub scratch_wait: NativeKernel<routed_scratch_wait::Entry>,
    pub copy: [NativeKernel<routed_copy::Entry>; 3],
    /// A round's convert, per routed tensor: every record of the round at
    /// once.
    pub convert_wide: [NativeKernel<routed_convert_wide::Entry>; 3],
}

/// The decode rows' routing and expansions, by the backend's
/// `DecodeForm`.
#[derive(Clone, Debug)]
pub enum RoutedDecodeKernels {
    /// `routed_route`, then the choices' and the shared expert's gate/up.
    Expand(NativeKernel<routed_expand::Entry>),
    /// The routing with the shared expert's gate/up, then the choices'.
    SharedRoute {
        route: NativeKernel<routed_route_shared::Entry>,
        choices: NativeKernel<routed_gate_up::Entry>,
    },
}

/// An attention block: normed query/gate/key/value projection, the fused
/// attention entry over the block's history codec, output projection plus
/// residual.
#[derive(Clone, Debug)]
pub struct AttentionKernels {
    pub codec: magnitude_state::CodecSpec,
    pub project: NativeKernel<attention_project::Entry>,
    pub history: AttentionHistoryKernels,
    pub output: SublayerOutput<NativeKernel<attention_output::Entry>>,
    /// A selected read's indexer and selected prefill
    /// (`AttentionBinding::selection`).
    pub selection: Option<SelectionKernels>,
}

/// A selected read's entries: the indexer's normed query and raw-key
/// projection (`attention_project`: query rows `heads · width`, key rows
/// `width`), `indexer_select`, and the codec's prefill in its WorkList form
/// (`B` words) for the classes that score.
#[derive(Clone, Debug)]
pub struct SelectionKernels {
    pub project: NativeKernel<attention_project::Entry>,
    pub select: NativeKernel<indexer_select::Entry>,
    pub prefill: SelectedPrefillKernel,
}

/// The WorkList prefill of a selected read's history codec.
#[derive(Clone, Debug)]
pub enum SelectedPrefillKernel {
    Dense(NativeKernel<attention_prefill::Entry>),
    AffineK8V4(NativeKernel<attention_prefill_k8v4::Entry>),
}

impl SelectionKernels {
    pub fn invocation_workspace_bytes(&self) -> u64 {
        self.project.invocation_workspace_bytes()
            + self.select.invocation_workspace_bytes()
            + match &self.prefill {
                SelectedPrefillKernel::Dense(kernel) => kernel.invocation_workspace_bytes(),
                SelectedPrefillKernel::AffineK8V4(kernel) => kernel.invocation_workspace_bytes(),
            }
    }
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
        append: NativeKernel<attention_append_dense::Entry>,
        decode: NativeKernel<attention_decode::Entry>,
        /// The multi-row decode (draft blocks, verification), tuned on its
        /// own row classes so it can read each history tile once for all
        /// rows.
        verify: Option<NativeKernel<attention_decode::Entry>>,
        prefill: NativeKernel<attention_prefill::Entry>,
    },
    AffineK8V4 {
        append: NativeKernel<attention_append_k8v4::Entry>,
        decode: NativeKernel<attention_decode_k8v4::Entry>,
        verify: Option<NativeKernel<attention_decode_k8v4::Entry>>,
        verify_four: Option<NativeKernel<attention_decode_k8v4::Entry>>,
        verify_eight: Option<NativeKernel<attention_decode_k8v4::Entry>>,
        /// The prefill of a launch that lists no history row tiles: it reads
        /// the history in place.
        prefill: NativeKernel<attention_prefill_k8v4::Entry>,
        /// The prefill of a launch that lists the history row tiles its rows
        /// see, where the entry's forms differ by it
        /// (`StateResourcePlan::lists_history_tiles`).
        prefill_listed: Option<NativeKernel<attention_prefill_k8v4::Entry>>,
    },
}

impl AttentionHistoryKernels {
    pub fn invocation_workspace_bytes(&self) -> u64 {
        match self {
            Self::Dense {
                append,
                decode,
                verify,
                prefill,
            } => {
                append.invocation_workspace_bytes()
                    + decode.invocation_workspace_bytes().max(
                        verify
                            .as_ref()
                            .map_or(0, |kernel| kernel.invocation_workspace_bytes()),
                    )
                    + prefill.invocation_workspace_bytes()
            }
            Self::AffineK8V4 {
                append,
                decode,
                verify,
                verify_four,
                verify_eight,
                prefill,
                prefill_listed,
            } => {
                append.invocation_workspace_bytes()
                    + decode
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
                    + prefill.invocation_workspace_bytes().max(
                        prefill_listed
                            .as_ref()
                            .map_or(0, NativeKernel::invocation_workspace_bytes),
                    )
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct DenseKernels {
    pub expand: NativeKernel<dense_expand::Entry>,
    pub output: SublayerOutput<NativeKernel<dense_output::Entry>>,
}

/// A recurrent block: normed projection, the state advance publishing the
/// gated rows (row-sequential `step` for small row classes, chunked for the
/// rest), and the plain residual output projection.
#[derive(Clone, Debug)]
pub struct RecurrentKernels {
    /// The chunked row classes' projection, and every row class's in the
    /// step form.
    pub project: NativeKernel<gated_delta_project::Entry>,
    /// The step row classes, by the backend's `StepForm`.
    pub step: RecurrentStepKernels,
    pub chunk: NativeKernel<gated_delta_chunk::Entry>,
    pub output: NativeKernel<attention_output::Entry>,
}

/// The step row classes' entries, by the backend's `StepForm`.
#[derive(Clone, Debug)]
pub enum RecurrentStepKernels {
    /// `gated_delta_project`, then the step that convolves.
    Step(NativeKernel<gated_delta_step::Entry>),
    /// The projection that also convolves, then the step over its channels.
    Convolved {
        project: NativeKernel<gated_delta_project_convolved::Entry>,
        step: NativeKernel<gated_delta_step_convolved::Entry>,
    },
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
