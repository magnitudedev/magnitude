//! A decoder block's sublayers as graph fragments: the one dispatch from a
//! block's admitted operators (`PairedBlock`) and their kernel entries
//! (prepared kernels or checked bindings) to each operator's own fragment
//! (`operators::<op>::graph`). The target graph builds a block by calling
//! the mixer, the feed-forward and the per-layer input in order.

use super::attention::graph::{
    attention, attention_weights, AttentionBlock, AttentionGraphEntries, CheckedAttentionEntries,
};
use super::dense_ffn::graph::{dense, CheckedDenseEntries, DenseGraphEntries};
use super::gated_delta::graph::{
    recurrent, CheckedRecurrentEntries, RecurrentBlock, RecurrentControlPorts,
    RecurrentGraphEntries, RecurrentStatePorts,
};
use super::parallel::graph::{parallel, CheckedParallelEntries, ParallelGraphEntries};
use super::per_layer::graph::{
    per_layer as per_layer_graph, CheckedPerLayerEntries, PerLayerEntries, PerLayerWeights,
};
use super::routed::fused_graph::{
    self as routed_graph, CheckedRoutedEntries, ExpertShape, RoutedGraphEntries,
};
use super::routed::graph::{
    class_constants, general_routed, CheckedGeneralRoutedEntries, GeneralRoutedGraphEntries,
    RoutedSum,
};
use super::short_conv::graph::{
    short_conv, CheckedShortConvEntries, ShortConvBlock, ShortConvGraphEntries,
};
use super::state_space::graph::{
    state_space, CheckedStateSpaceEntries, StateSpaceBlock, StateSpaceGraphEntries,
};
use super::{
    attention as attention_operator, dense_ffn, post_norm_epsilon, FeedForward, Mixer, PairedBlock,
};
use crate::native::{AttestedFeedForward, AttestedMixer, OutputScales};
use crate::programs::graph::{draft::GraphDraft, GraphError};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{activation, weight, WeightPort};
use crate::{
    FeedForwardProgramSlot, GeneralRoutedShape, MixerProgramSlot, ModelLoadPlan, PerLayerBinding,
    ShortConvShape, StateResourcePlan, StateSpaceShape,
};
use magnitude_family_contracts::{
    Decoder, RecurrentHeadMapping, SublayerIndex, WeightKind, WeightScope,
};
use magnitude_state::LayerRef;
use seismic::{
    Element, NativeGraph, NativeGraphClassSlice, NativeGraphMetadata, NativePort, WorkflowTensor,
};

/// A block's state ports, by the state its mixer keeps.
#[derive(Clone)]
pub(crate) enum BlockStatePorts {
    /// The layer's history planes in the codec's plane-descriptor order.
    Attention(Vec<NativePort>),
    /// The layer's recurrent bank components, consecutive in the store's
    /// component order (`operators::Mixer::bank_components`).
    Recurrent(Vec<NativePort>),
}

impl From<RecurrentStatePorts> for BlockStatePorts {
    fn from(ports: RecurrentStatePorts) -> Self {
        Self::Recurrent(vec![ports.window, ports.delta, ports.tape])
    }
}

/// A block's per-launch control ports, by the state its mixer keeps.
#[derive(Clone)]
pub(crate) enum BlockControlPorts {
    Attention {
        coordinates: NativePort,
        visible: NativePort,
        fresh: NativePort,
        destinations: NativePort,
    },
    Recurrent(RecurrentControlPorts),
}

/// The kernel entries of a block's mixer: prepared kernels or checked
/// bindings.
pub(crate) enum MixerEntries<'a, G: GraphDraft + 'a> {
    Attention(AttentionGraphEntries<'a, G>),
    Recurrent(RecurrentGraphEntries<'a, G>),
    StateSpace(StateSpaceGraphEntries<'a, G>),
    ShortConv(ShortConvGraphEntries<'a, G>),
}

impl<'a> From<&'a AttestedMixer> for MixerEntries<'a, NativeGraph> {
    fn from(kernels: &'a AttestedMixer) -> Self {
        match kernels {
            AttestedMixer::Attention(kernels) => Self::Attention(kernels.into()),
            AttestedMixer::Recurrent(kernels) => Self::Recurrent(kernels.into()),
            AttestedMixer::StateSpace(kernels) => Self::StateSpace(kernels.into()),
            AttestedMixer::ShortConv(kernels) => Self::ShortConv(kernels.into()),
        }
    }
}

/// The kernel entries of a block's feed-forward.
pub(crate) enum FeedForwardEntries<'a, G: GraphDraft + 'a> {
    Dense(DenseGraphEntries<'a, G>),
    Routed(RoutedGraphEntries<'a, G>),
    GeneralRouted(GeneralRoutedGraphEntries<'a, G>),
    Parallel(ParallelGraphEntries<'a, G>),
}

impl<'a> From<&'a AttestedFeedForward> for FeedForwardEntries<'a, NativeGraph> {
    fn from(kernels: &'a AttestedFeedForward) -> Self {
        match kernels {
            AttestedFeedForward::Dense(kernels) => Self::Dense(kernels.into()),
            AttestedFeedForward::Routed(kernels) => Self::Routed(kernels.into()),
            AttestedFeedForward::GeneralRouted(kernels) => Self::GeneralRouted(kernels.into()),
            AttestedFeedForward::Parallel(kernels) => Self::Parallel(kernels.into()),
        }
    }
}

/// A mixer's checked entry bindings, from its program slot.
pub(crate) enum CheckedMixerEntries {
    Attention(CheckedAttentionEntries),
    Recurrent(CheckedRecurrentEntries),
    StateSpace(CheckedStateSpaceEntries),
    ShortConv(CheckedShortConvEntries),
}

impl CheckedMixerEntries {
    pub(crate) fn new(slot: MixerProgramSlot) -> Self {
        match slot {
            MixerProgramSlot::Attention(binding) => {
                Self::Attention(CheckedAttentionEntries::new(binding))
            }
            MixerProgramSlot::Recurrent(binding) => {
                Self::Recurrent(CheckedRecurrentEntries::new(binding))
            }
            MixerProgramSlot::StateSpace(binding) => {
                Self::StateSpace(CheckedStateSpaceEntries::new(binding))
            }
            MixerProgramSlot::ShortConv(binding) => {
                Self::ShortConv(CheckedShortConvEntries::new(binding))
            }
        }
    }

    pub(crate) fn entries(&self) -> Result<MixerEntries<'_, NativeGraphMetadata>, String> {
        Ok(match self {
            Self::Attention(checked) => MixerEntries::Attention(checked.entries()?),
            Self::Recurrent(checked) => MixerEntries::Recurrent(checked.entries()),
            Self::StateSpace(checked) => MixerEntries::StateSpace(checked.entries()),
            Self::ShortConv(checked) => MixerEntries::ShortConv(checked.entries()),
        })
    }
}

/// A feed-forward's checked entry bindings, from its program slot.
pub(crate) enum CheckedFeedForwardEntries {
    Dense(CheckedDenseEntries),
    Routed(CheckedRoutedEntries),
    GeneralRouted(CheckedGeneralRoutedEntries),
    Parallel(CheckedParallelEntries),
}

impl CheckedFeedForwardEntries {
    pub(crate) fn new(slot: FeedForwardProgramSlot) -> Self {
        match slot {
            FeedForwardProgramSlot::Dense(binding) => {
                Self::Dense(CheckedDenseEntries::new(binding))
            }
            FeedForwardProgramSlot::Routed(binding) => {
                Self::Routed(CheckedRoutedEntries::new(binding))
            }
            FeedForwardProgramSlot::GeneralRouted(binding) => {
                Self::GeneralRouted(CheckedGeneralRoutedEntries::new(binding))
            }
            FeedForwardProgramSlot::Parallel(binding) => {
                Self::Parallel(CheckedParallelEntries::new(binding))
            }
        }
    }

    pub(crate) fn entries(&self) -> FeedForwardEntries<'_, NativeGraphMetadata> {
        match self {
            Self::Dense(checked) => FeedForwardEntries::Dense(checked.entries()),
            Self::Routed(checked) => FeedForwardEntries::Routed(checked.entries()),
            Self::GeneralRouted(checked) => FeedForwardEntries::GeneralRouted(checked.entries()),
            Self::Parallel(checked) => FeedForwardEntries::Parallel(checked.entries()),
        }
    }
}

/// The per-layer input sublayer's entries and binding, prepared or checked.
pub(crate) type PerLayerParts<'a, G> = (PerLayerEntries<'a, G>, PerLayerBinding);

/// A per-layer sublayer's checked entry bindings, from its binding.
pub(crate) fn checked_per_layer(
    binding: Option<PerLayerBinding>,
) -> Result<Option<(CheckedPerLayerEntries, PerLayerBinding)>, String> {
    binding
        .map(|binding| Ok((CheckedPerLayerEntries::new(binding)?, binding)))
        .transpose()
}

/// Everything a block's sublayers are built from apart from their entries.
pub(crate) struct BlockSublayers<'a> {
    pub paired: &'a PairedBlock<'a>,
    pub load: &'a ModelLoadPlan,
    pub geometry: &'a Decoder,
    pub state: &'a StateResourcePlan,
    pub block_index: usize,
    pub component_index: usize,
    pub rows: u64,
    pub segments: u64,
    pub slots: u64,
    pub output_scales: OutputScales,
}

impl BlockSublayers<'_> {
    fn scope(&self, sublayer: u32) -> Result<WeightScope, String> {
        Ok(WeightScope::TargetSublayer(self.index(sublayer)?))
    }

    fn index(&self, sublayer: u32) -> Result<SublayerIndex, String> {
        Ok(SublayerIndex {
            block: u32::try_from(self.block_index).map_err(|_| "block index exceeds u32")?,
            sublayer,
        })
    }

    /// The block's mixer over `hidden`: the new residual, and the block's
    /// state and control ports.
    pub(crate) fn mixer<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        entries: MixerEntries<'e, G>,
        hidden: &WorkflowTensor,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(WorkflowTensor, BlockStatePorts, BlockControlPorts), GraphError> {
        let Self {
            paired,
            load,
            geometry,
            state,
            block_index: _,
            component_index,
            rows,
            segments,
            slots,
            output_scales,
        } = *self;
        let scope = self.scope(0)?;
        let epsilon = paired.epsilon() as f32;
        Ok(match (paired.mixer, entries) {
            (Mixer::Attention(operator), MixerEntries::Attention(entries)) => {
                let shape = attention_operator::shape(geometry.hidden, operator)
                    .map_err(|error| error.to_string())?;
                let post_norm_epsilon = post_norm_epsilon(paired.mixer_output);
                let attention_weights = attention_weights(
                    &shape,
                    operator,
                    true,
                    post_norm_epsilon.is_some(),
                    |kind| weight(graph, load, scope, kind, weights),
                )?;
                // The block reads its history domain's slab tensor, a Shared
                // layer its source's.
                let history = state
                    .target_state()
                    .layer_history(LayerRef::Target(self.index(0)?.block))
                    .ok_or("attention block has no history domain")?
                    .store;
                let (mixed, ports, controls) = attention(
                    graph,
                    entries,
                    &attention_weights,
                    constants,
                    hidden,
                    AttentionBlock {
                        rows,
                        segments,
                        history_rows: u64::try_from(history.rows)
                            .map_err(|_| "history rows exceed u64")?,
                        slab_rows: history.slab_rows,
                        shape,
                        operator,
                        epsilon,
                        head_epsilon: attention_operator::head_norm_epsilon(
                            operator,
                            paired.epsilon(),
                        )
                        .map_err(|error| error.to_string())?
                            as f32,
                        post_norm_epsilon: post_norm_epsilon.unwrap_or_default() as f32,
                        post_norm_scale: output_scales.mixer,
                        activation: activation(geometry),
                        inject_only: false,
                    },
                )?;
                (
                    mixed,
                    BlockStatePorts::Attention(ports.planes),
                    BlockControlPorts::Attention {
                        coordinates: controls.coordinates,
                        visible: controls.visible,
                        fresh: controls.fresh,
                        destinations: controls.destinations,
                    },
                )
            }
            (Mixer::GatedDelta(shape), MixerEntries::Recurrent(entries)) => {
                let (mixed, ports, controls) = recurrent(
                    graph,
                    entries,
                    state,
                    load,
                    scope,
                    weights,
                    hidden,
                    RecurrentBlock {
                        rows,
                        hidden: geometry.hidden,
                        slots,
                        slab_banks: state.target_state().bank_slab_banks()?,
                        key_heads: shape.key_heads,
                        value_heads: shape.value_heads,
                        width: shape.width,
                        convolution_width: shape.convolution_width,
                        grouped: matches!(shape.head_mapping, RecurrentHeadMapping::Grouped),
                        epsilon,
                        component_index,
                    },
                )?;
                (mixed, ports.into(), BlockControlPorts::Recurrent(controls))
            }
            (Mixer::StateSpace(operator), MixerEntries::StateSpace(entries)) => {
                let (mixed, ports, controls) = state_space(
                    graph,
                    entries,
                    state,
                    load,
                    scope,
                    weights,
                    hidden,
                    StateSpaceBlock {
                        rows,
                        slots,
                        slab_banks: state.target_state().bank_slab_banks()?,
                        shape: StateSpaceShape::of(geometry.hidden, operator)
                            .map_err(|error| error.to_string())?,
                        epsilon,
                        norm_epsilon: operator.norm.epsilon as f32,
                        component_index,
                    },
                )?;
                (mixed, ports.into(), BlockControlPorts::Recurrent(controls))
            }
            (Mixer::ShortConv(operator), MixerEntries::ShortConv(entries)) => {
                let absent_scale = GraphConstant::absent_scale(graph, constants)?;
                let (mixed, window, controls) = short_conv(
                    graph,
                    entries,
                    state,
                    load,
                    scope,
                    weights,
                    &absent_scale,
                    hidden,
                    ShortConvBlock {
                        rows,
                        slots,
                        slab_banks: state.target_state().bank_slab_banks()?,
                        shape: ShortConvShape::of(geometry.hidden, operator)
                            .map_err(|error| error.to_string())?,
                        epsilon,
                        component_index,
                    },
                )?;
                (
                    mixed,
                    BlockStatePorts::Recurrent(vec![window]),
                    BlockControlPorts::Recurrent(controls),
                )
            }
            _ => return Err("target block mixer and its kernel entries disagree".into()),
        })
    }

    /// The block's feed-forward over the mixer's residual `mixed`; a lone
    /// mixer block's output is `mixed`.
    pub(crate) fn feed_forward<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        entries: Option<FeedForwardEntries<'e, G>>,
        mixed: WorkflowTensor,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<WorkflowTensor, GraphError> {
        let Self {
            paired,
            load,
            geometry,
            rows,
            output_scales,
            ..
        } = *self;
        let scope = self.scope(1)?;
        let epsilon = paired.epsilon() as f32;
        Ok(match (paired.feed_forward, entries) {
            (None, None) => mixed,
            (Some(sublayer), Some(entries)) => match (sublayer.op, entries) {
                (FeedForward::Dense(operator), FeedForwardEntries::Dense(entries)) => dense(
                    graph,
                    entries,
                    load,
                    scope,
                    weights,
                    constants,
                    &mixed,
                    rows,
                    epsilon,
                    dense_ffn::activation_code(operator.up.activation()),
                    post_norm_epsilon(sublayer.output).unwrap_or_default() as f32,
                    output_scales.feed_forward,
                )?,
                (FeedForward::Routed(shape), FeedForwardEntries::Routed(entries)) => {
                    routed_graph::routed(
                        graph,
                        entries,
                        load,
                        scope,
                        weights,
                        &mixed,
                        rows,
                        geometry.hidden,
                        &ExpertShape::of_operator(shape)?,
                        epsilon,
                    )?
                }
                (
                    FeedForward::GeneralRouted(operator),
                    FeedForwardEntries::GeneralRouted(entries),
                ) => general_routed(
                    graph,
                    entries,
                    load,
                    scope,
                    weights,
                    constants,
                    &mixed,
                    RoutedSum::Residual,
                    rows,
                    &GeneralRoutedShape::of(geometry.hidden, operator)
                        .map_err(|error| error.to_string())?,
                    epsilon,
                )?,
                (FeedForward::Parallel(branches), FeedForwardEntries::Parallel(entries)) => {
                    parallel(
                        graph,
                        entries,
                        load,
                        self.index(1)?,
                        weights,
                        constants,
                        &mixed,
                        rows,
                        &GeneralRoutedShape::of(geometry.hidden, branches.routed)
                            .map_err(|error| error.to_string())?,
                        epsilon,
                        dense_ffn::activation_code(branches.dense.up.activation()),
                        output_scales.feed_forward,
                    )?
                }
                _ => return Err("target block feed-forward and its kernel entries disagree".into()),
            },
            _ => return Err("target block feed-forward and its kernel entries disagree".into()),
        })
    }

    /// The block's per-layer input sublayer over `residual`: the new
    /// residual and the per-layer rows port it reads.
    pub(crate) fn per_layer<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        parts: Option<PerLayerParts<'e, G>>,
        residual: WorkflowTensor,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(WorkflowTensor, Option<NativePort>), GraphError> {
        let Self {
            paired,
            load,
            rows,
            output_scales,
            ..
        } = *self;
        match (paired.per_layer, parts) {
            (None, None) => Ok((residual, None)),
            (Some(sublayer), Some((entries, binding))) => {
                let scope = self.scope(2)?;
                let per_layer_weights = PerLayerWeights {
                    gate: weight(graph, load, scope, WeightKind::PerLayerGate, weights)?,
                    projection: weight(
                        graph,
                        load,
                        scope,
                        WeightKind::PerLayerProjection,
                        weights,
                    )?,
                    post_norm: weight(graph, load, scope, WeightKind::PostNorm, weights)?,
                };
                let rows_port = graph.port_with_class_extent(
                    Element::f32(),
                    &[rows, binding.layers * binding.width],
                    0,
                    "M",
                )?;
                let out_rows = GraphConstant::identity_for_class(graph, rows, Some("M"))?;
                let absent_scale = GraphConstant::absent_scale(graph, constants)?;
                let advanced = per_layer_graph(
                    graph,
                    entries,
                    &per_layer_weights,
                    &absent_scale,
                    &residual,
                    rows_port.tensor(),
                    out_rows.port().tensor(),
                    binding,
                    i32::try_from(sublayer.op.layer).map_err(|_| "per-layer index exceeds i32")?,
                    dense_ffn::activation_code(sublayer.op.activation),
                    rows,
                    post_norm_epsilon(sublayer.output)
                        .ok_or("a per-layer input sublayer ends in a post-norm tail")?
                        as f32,
                    output_scales.per_layer,
                )?;
                constants.push(out_rows);
                Ok((advanced, Some(rows_port)))
            }
            _ => Err("target block per-layer sublayer and its kernel entries disagree".into()),
        }
    }
}

/// A feed-forward's class dimensions for a `rows`-row launch: a routed
/// feed-forward's grouped entries size their tile table from the row count
/// alone.
pub(crate) fn feed_forward_class_slice(
    paired: &PairedBlock,
    rows: u64,
    mut slice: NativeGraphClassSlice,
) -> Result<NativeGraphClassSlice, String> {
    let grouped = match paired.feed_forward.map(|sublayer| sublayer.op) {
        Some(FeedForward::Routed(shape)) => Some((
            shape,
            &["routed_group", "routed_experts", "routed_combine"][..],
        )),
        Some(
            FeedForward::GeneralRouted(shape)
            | FeedForward::Parallel(super::parallel::DenseBesideRouted { routed: shape, .. }),
        ) => Some((
            shape,
            &[
                "routed_group",
                "routed_experts",
                "routed_experts_up",
                "routed_scatter",
            ][..],
        )),
        Some(FeedForward::Dense(_)) | None => None,
    };
    if let Some((shape, entries)) = grouped {
        if !routed_graph::decodes(rows) {
            let blocks = routed_graph::grouped_blocks(rows, shape.experts, shape.selected)?;
            for entry in entries {
                slice = slice.scoped(entry, "B", [blocks]);
            }
        }
    }
    Ok(slice)
}

/// The row-class constants a block's sublayers bind at `rows` rows (for
/// resource accounting): the row table of a dense or general routed
/// feed-forward and of a per-layer input tail, a branch's zero root and a
/// latent sum's zero base.
pub(crate) fn class_constants_of(
    paired: &PairedBlock,
    hidden: u64,
    rows: u64,
) -> Result<Vec<GraphConstant>, String> {
    let mut constants = match paired.feed_forward.map(|sublayer| sublayer.op) {
        Some(FeedForward::Dense(_)) => vec![GraphConstant::identity_value(rows)?],
        Some(FeedForward::GeneralRouted(operator)) => class_constants(
            &GeneralRoutedShape::of(hidden, operator).map_err(|error| error.to_string())?,
            RoutedSum::Residual,
            rows,
        )?,
        // The dense branch's row table, and the routed branch's.
        Some(FeedForward::Parallel(branches)) => {
            let mut constants = vec![GraphConstant::identity_value(rows)?];
            constants.extend(class_constants(
                &GeneralRoutedShape::of(hidden, branches.routed)
                    .map_err(|error| error.to_string())?,
                RoutedSum::Branch,
                rows,
            )?);
            constants
        }
        Some(FeedForward::Routed(_)) | None => Vec::new(),
    };
    if paired.per_layer.is_some() {
        constants.push(GraphConstant::identity_value(rows)?);
    }
    Ok(constants)
}
