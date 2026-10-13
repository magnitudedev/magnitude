//! A decoder block's sublayers as graph fragments: the one dispatch from a
//! block's admitted operators (`PairedBlock`) and their kernel entries
//! (prepared kernels or checked bindings) to each operator's own fragment
//! (`operators::<op>::graph`). The target graph builds a block by calling
//! the mixer, the feed-forward and the per-layer input in order.

use super::attention::graph::{
    AttentionBlock, AttentionBounds, AttentionGraphEntries, CheckedAttentionEntries,
    HistoryTilesPort, IndexerControlPorts, IndexerStatePorts, SelectionBlock, attention,
    attention_weights,
};
use super::dense_ffn::graph::{CheckedDenseEntries, DenseGraphEntries, dense};
use super::gated_delta::graph::{
    CheckedRecurrentEntries, RecurrentBlock, RecurrentControlPorts, RecurrentGraphEntries,
    RecurrentStatePorts, recurrent,
};
use super::ngram::graph::{NGramBlock, NGramEntries, NGramPorts, ngram as ngram_graph};
use super::parallel::graph::{CheckedParallelEntries, ParallelGraphEntries, parallel};
use super::per_layer::graph::{
    CheckedPerLayerEntries, PerLayerEntries, PerLayerWeights, per_layer as per_layer_graph,
};
use super::routed::fused_graph::{
    self as routed_graph, CheckedRoutedEntries, ExpertShape, RoutedGraphEntries,
};
use super::routed::graph::{
    CheckedGeneralRoutedEntries, GeneralRoutedGraphEntries, RoutedSum, class_constants,
    general_routed,
};
use super::short_conv::graph::{
    CheckedShortConvEntries, ShortConvBlock, ShortConvGraphEntries, short_conv,
};
use super::state_space::graph::{
    CheckedStateSpaceEntries, StateSpaceBlock, StateSpaceGraphEntries, state_space,
};
use super::streams::StreamsGeometry;
use super::streams::graph::{StreamsEntries, mix_site};
use super::{
    FeedForward, InputForm, Join, Mixer, PairedBlock, SublayerInput, SublayerSource,
    attention as attention_operator, bank_component_index, dense_ffn, gated_delta,
    post_norm_epsilon,
};
use crate::native::{AttestedFeedForward, AttestedMixer, OutputScales};
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{PackingBound, WeightPort, activation, weight};
use crate::{
    FeedForwardProgramSlot, GeneralRoutedShape, MixerProgramSlot, ModelLoadPlan, NGramBinding,
    PerLayerBinding, ShortConvShape, StateResourcePlan, StateSpaceShape,
};
use magnitude_family_contracts::{
    Decoder, HistoryReads, RecurrentHeadMapping, SublayerIndex, WeightKind, WeightScope,
};
use magnitude_state::LayerRef;
use seismic::{
    BackendName, Element, NativeGraph, NativeGraphClassSlice, NativeGraphMetadata, NativePort,
    WorkflowTensor,
};

/// A block's state ports, by the state its mixer keeps.
#[derive(Clone)]
pub(crate) enum BlockStatePorts {
    /// The layer's history planes in the codec's plane-descriptor order,
    /// and a selected read's indexer planes.
    Attention(Vec<NativePort>, Option<IndexerStatePorts>),
    /// The layer's recurrent bank components, consecutive in the store's
    /// component order (`operators::Mixer::bank_components`).
    Recurrent(Vec<NativePort>),
}

impl BlockStatePorts {
    /// The ports of the layer's history or bank, whichever state they are
    /// of.
    pub(crate) fn ports(&self) -> &Vec<NativePort> {
        match self {
            Self::Attention(ports, _) | Self::Recurrent(ports) => ports,
        }
    }

    /// Leave the ports' vector holding exactly them, so a prepared graph's
    /// host heap follows from its topology.
    pub(crate) fn shrink_to_fit(&mut self) {
        match self {
            Self::Attention(ports, _) | Self::Recurrent(ports) => ports.shrink_to_fit(),
        }
    }
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
        history_tiles: Option<HistoryTilesPort>,
        /// A selected read's indexer tables.
        indexer: Option<IndexerControlPorts>,
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
    /// The entries of `slot` on `backend`; `lists` when its attention
    /// graphs have classes that list their launch's history row tiles
    /// (`StateResourcePlan::lists_history_tiles`).
    pub(crate) fn new(
        slot: MixerProgramSlot,
        backend: BackendName,
        lists: bool,
    ) -> Result<Self, String> {
        Ok(match slot {
            MixerProgramSlot::Attention(binding) => {
                Self::Attention(CheckedAttentionEntries::new(binding, lists))
            }
            MixerProgramSlot::Recurrent(binding) => {
                Self::Recurrent(CheckedRecurrentEntries::new(binding, backend)?)
            }
            MixerProgramSlot::StateSpace(binding) => {
                Self::StateSpace(CheckedStateSpaceEntries::new(binding))
            }
            MixerProgramSlot::ShortConv(binding) => {
                Self::ShortConv(CheckedShortConvEntries::new(binding))
            }
        })
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
    /// The entries of `slot` on `backend`.
    pub(crate) fn new(slot: FeedForwardProgramSlot, backend: BackendName) -> Result<Self, String> {
        Ok(match slot {
            FeedForwardProgramSlot::Dense(binding) => {
                Self::Dense(CheckedDenseEntries::new(binding))
            }
            FeedForwardProgramSlot::Routed(binding) => {
                Self::Routed(CheckedRoutedEntries::new(binding, backend)?)
            }
            FeedForwardProgramSlot::GeneralRouted(binding) => {
                Self::GeneralRouted(CheckedGeneralRoutedEntries::new(binding))
            }
            FeedForwardProgramSlot::Parallel(binding) => {
                Self::Parallel(CheckedParallelEntries::new(binding))
            }
        })
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

/// The n-gram input's entries and binding, prepared or checked.
pub(crate) type NGramParts<'a, G> = (NGramEntries<'a, G>, NGramBinding);

/// A per-layer sublayer's checked entry bindings, from its binding.
pub(crate) fn checked_per_layer(
    binding: Option<PerLayerBinding>,
) -> Result<Option<(CheckedPerLayerEntries, PerLayerBinding)>, String> {
    binding
        .map(|binding| Ok((CheckedPerLayerEntries::new(binding)?, binding)))
        .transpose()
}

/// Reject a residual-streams input for a mixer whose kernels read only one
/// normalized row.
fn single_stream(form: InputForm) -> Result<(), GraphError> {
    if form.parts != 1 {
        return Err("this mixer over residual streams".into());
    }
    Ok(())
}

/// Whose block a graph fragment builds:the scope of its weights, and the
/// target layer whose history its attention mixer reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockLane {
    /// A target decoder block: its own weights, its own history.
    Target,
    /// A block of a draft head reading the history of target block `source`
    /// (`operators::SharedHead`).
    Head { source: u32 },
}

/// Everything a block's sublayers are built from apart from their entries.
/// `geometry` is the decoder the block belongs to: the target's, or a draft
/// head's blocks at the head's width.
pub(crate) struct BlockSublayers<'a> {
    pub lane: BlockLane,
    pub paired: &'a PairedBlock<'a>,
    pub load: &'a ModelLoadPlan,
    pub geometry: &'a Decoder,
    pub state: &'a StateResourcePlan,
    pub block_index: usize,
    pub rows: u64,
    pub segments: u64,
    pub slots: u64,
    /// Whether an attention block's class lists the history row tiles its
    /// launch's rows see.
    pub listed: u64,
    /// Whether a selected attention read's class scores (some row of its
    /// launch is past the selection threshold).
    pub scored: bool,
    pub output_scales: OutputScales,
}

impl BlockSublayers<'_> {
    fn scope(&self, sublayer: u32) -> Result<WeightScope, String> {
        let index = self.index(sublayer)?;
        Ok(match self.lane {
            BlockLane::Target => WeightScope::TargetSublayer(index),
            BlockLane::Head { .. } => WeightScope::HeadSublayer(index),
        })
    }

    /// The target layer whose history the block's attention reads.
    fn history_layer(&self) -> Result<LayerRef, String> {
        Ok(LayerRef::Target(match self.lane {
            BlockLane::Target => self.index(0)?.block,
            BlockLane::Head { source } => source,
        }))
    }

    fn index(&self, sublayer: u32) -> Result<SublayerIndex, String> {
        Ok(SublayerIndex {
            block: u32::try_from(self.block_index).map_err(|_| "block index exceeds u32")?,
            sublayer,
        })
    }

    /// What sublayer `sublayer` reads of the residual `hidden` (its
    /// `source`), and how its output joins it: the residual itself under its
    /// input norm, joined as one stream; or its mix site's partials over the
    /// streams, joined by the scatter with the site's inject values.
    #[allow(clippy::too_many_arguments)]
    fn read<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        source: SublayerSource<'_>,
        streams: Option<StreamsEntries<'e, G>>,
        sublayer: u32,
        hidden: &WorkflowTensor,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(WorkflowTensor, InputForm, Join), GraphError> {
        match source {
            SublayerSource::Norm(_) => {
                Ok((hidden.clone(), InputForm::SINGLE, Join::single(Some("M"))))
            }
            SublayerSource::Mix(_) => {
                let geometry = StreamsGeometry::of(self.geometry)
                    .map_err(|error| error.to_string())?
                    .ok_or("a mix site without residual streams")?;
                let entries = streams.ok_or("a mix site without its streams entries")?;
                let site = mix_site(
                    graph,
                    entries,
                    self.load,
                    self.scope(sublayer)?,
                    geometry,
                    hidden,
                    self.rows,
                    source.epsilon() as f32,
                    true,
                    weights,
                )?;
                let inject = site
                    .inject
                    .ok_or("a sublayer's mix site without inject values")?;
                Ok((
                    site.partials,
                    source.form(geometry.streams),
                    Join {
                        values: Some(inject),
                        streams: geometry.streams,
                        injected: true,
                        class: Some("M"),
                    },
                ))
            }
        }
    }

    /// The block's mixer over `hidden`: the new residual, and the block's
    /// state and control ports.
    pub(crate) fn mixer<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        entries: MixerEntries<'e, G>,
        streams: Option<StreamsEntries<'e, G>>,
        hidden: &WorkflowTensor,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(WorkflowTensor, BlockStatePorts, BlockControlPorts), GraphError> {
        let Self {
            paired,
            load,
            geometry,
            state,
            block_index,
            rows,
            segments,
            slots,
            listed,
            scored,
            output_scales,
            ..
        } = *self;
        let scope = self.scope(0)?;
        let epsilon = paired.epsilon() as f32;
        let component_index = bank_component_index(&geometry.blocks, block_index)
            .map_err(|error| error.to_string())?;
        let (input_rows, form, join) = self.read(
            graph,
            paired.mixer_input,
            streams,
            0,
            hidden,
            weights,
            constants,
        )?;
        let input = SublayerInput {
            rows: &input_rows,
            form,
        };
        Ok(match (paired.mixer, entries) {
            (Mixer::Attention(operator), MixerEntries::Attention(entries)) => {
                let shape = attention_operator::shape(geometry.hidden, operator)
                    .map_err(|error| error.to_string())?;
                let post_norm_epsilon = post_norm_epsilon(paired.mixer_output);
                let bounds = AttentionBounds::planned(
                    graph, load, scope, operator, &shape, true, weights, constants,
                )?;
                let attention_weights = attention_weights(
                    &shape,
                    operator,
                    form,
                    true,
                    post_norm_epsilon.is_some(),
                    bounds,
                    |kind| weight(graph, load, scope, kind, weights),
                )?;
                // The block reads its history domain's slab tensor, a Shared
                // layer its source's.
                let history = state
                    .target_state()
                    .layer_history(self.history_layer()?)
                    .ok_or("attention block has no history domain")?
                    .store;
                let (mixed, ports, controls) = attention(
                    graph,
                    entries,
                    &attention_weights,
                    constants,
                    input,
                    hidden,
                    &join,
                    AttentionBlock {
                        rows,
                        segments,
                        history_rows: u64::try_from(history.rows)
                            .map_err(|_| "history rows exceed u64")?,
                        slab_rows: history.slab_rows,
                        history_tiles: listed,
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
                        // A head's rows read target history and append
                        // nowhere.
                        causal_cache: self.lane == BlockLane::Target,
                        selection: attention_operator::selection_shape(
                            &operator.reads,
                            geometry.context_limit,
                        )
                        .map(|shape| {
                            let (window, blocks) = state
                                .target_state()
                                .indexer_history(self.index(0)?.block)
                                .ok_or("a selected attention block has no indexer history")?;
                            Ok::<_, GraphError>(SelectionBlock {
                                shape,
                                scored,
                                window_rows: u64::try_from(window.rows)
                                    .map_err(|_| "indexer rows exceed u64")?,
                                window_slab_rows: window.slab_rows,
                                block_rows: u64::try_from(blocks.rows)
                                    .map_err(|_| "indexer rows exceed u64")?,
                                block_slab_rows: blocks.slab_rows,
                            })
                        })
                        .transpose()?,
                    },
                )?;
                (
                    mixed,
                    BlockStatePorts::Attention(ports.planes, ports.indexer),
                    BlockControlPorts::Attention {
                        coordinates: controls.coordinates,
                        visible: controls.visible.ok_or("attention visibility is absent")?,
                        fresh: controls.fresh.ok_or("attention fresh rows are absent")?,
                        destinations: controls.destinations,
                        history_tiles: controls.history_tiles,
                        indexer: controls.indexer,
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
                    constants,
                    input,
                    hidden,
                    &join,
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
                        gate_function: gated_delta::gate_function(shape),
                        epsilon,
                        component_index,
                    },
                )?;
                (mixed, ports.into(), BlockControlPorts::Recurrent(controls))
            }
            (Mixer::StateSpace(operator), MixerEntries::StateSpace(entries)) => {
                single_stream(input.form)?;
                let (mixed, ports, controls) = state_space(
                    graph,
                    entries,
                    state,
                    load,
                    scope,
                    weights,
                    constants,
                    hidden,
                    &join,
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
                single_stream(input.form)?;
                let (mixed, window, controls) = short_conv(
                    graph,
                    entries,
                    state,
                    load,
                    scope,
                    weights,
                    constants,
                    hidden,
                    &join,
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
        streams: Option<StreamsEntries<'e, G>>,
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
        // A mix site's sublayer reads its partials; the others read `mixed`.
        let read = |graph: &mut G,
                    weights: &mut Vec<(WeightPort, NativePort)>,
                    constants: &mut Vec<GraphConstant>,
                    source: Option<SublayerSource<'_>>| {
            let source = source.ok_or("a routed feed-forward without its input source")?;
            self.read(graph, source, streams, 1, &mixed, weights, constants)
        };
        Ok(match (paired.feed_forward, entries) {
            (None, None) => mixed,
            (Some(sublayer), Some(entries)) => match (sublayer.op, entries) {
                (FeedForward::Dense(operator), FeedForwardEntries::Dense(entries)) => {
                    if matches!(sublayer.input, Some(SublayerSource::Mix(_))) {
                        return Err("a dense feed-forward over residual streams".into());
                    }
                    dense(
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
                    )?
                }
                (FeedForward::Routed(shape), FeedForwardEntries::Routed(entries)) => {
                    let (input_rows, form, join) = read(graph, weights, constants, sublayer.input)?;
                    routed_graph::routed(
                        graph,
                        entries,
                        load,
                        scope,
                        weights,
                        constants,
                        SublayerInput {
                            rows: &input_rows,
                            form,
                        },
                        &mixed,
                        &join,
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
                    projection_bound: PackingBound::planned(
                        graph,
                        load,
                        magnitude_family_contracts::WeightRole {
                            scope,
                            kind: WeightKind::PerLayerProjection,
                        },
                        weights,
                        constants,
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
                let absent_scale = GraphConstant::absent(graph, constants)?.tensor().clone();
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

    /// The block's n-gram input over the streams `residual`, after its
    /// mixer's bank components and under its recurrent `controls`: the new
    /// streams, its windows, its token rows and its table read's ports.
    pub(crate) fn ngram<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        parts: Option<NGramParts<'e, G>>,
        controls: &BlockControlPorts,
        residual: WorkflowTensor,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(WorkflowTensor, Option<NGramPorts>), GraphError> {
        let Self {
            paired,
            load,
            geometry,
            state,
            block_index,
            rows,
            slots,
            ..
        } = *self;
        let (sublayer, (entries, binding)) = match (paired.ngram, parts) {
            (None, None) => return Ok((residual, None)),
            (Some(sublayer), Some(parts)) => (sublayer, parts),
            _ => return Err("target block n-gram input and its kernel entries disagree".into()),
        };
        let BlockControlPorts::Recurrent(controls) = controls else {
            return Err("an n-gram input in a block without recurrent controls".into());
        };
        let component_index = bank_component_index(&geometry.blocks, block_index)
            .map_err(|error| error.to_string())?
            + paired.mixer.bank_components();
        let (streams, ports) = ngram_graph(
            graph,
            entries,
            state,
            load,
            self.scope(2)?,
            weights,
            constants,
            &residual,
            controls,
            NGramBlock {
                rows,
                slots,
                slab_banks: state.target_state().bank_slab_banks()?,
                binding,
                hash: &sublayer.op.hash,
                epsilon: sublayer.op.key_norm.epsilon as f32,
                component_index,
            },
        )?;
        Ok((streams, Some(ports)))
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
/// feed-forward and of a per-layer input tail, a branch's zero root, a
/// latent sum's zero base and an attention prefill's empty bitmap.
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
    // A prefill's empty WorkList bitmap (a Visible read's, and a selected
    // read's in its classes without scoring); a selected multi-row decode's
    // disabled fused append (`attention::graph`).
    if let Mixer::Attention(operator) = paired.mixer {
        if !super::attention::graph::decodes(rows) {
            constants.push(GraphConstant::u32_zeros_value(rows, 0)?);
        } else if rows > 1 && matches!(operator.reads, HistoryReads::Selected { .. }) {
            constants.push(GraphConstant::i32_value(&vec![-1; rows as usize]));
        }
    }
    // The empty inject of an output entry joining a single stream
    // (`Join::inject`): a mixer's output projection (an attention mixer's
    // unless it ends in a post-norm tail), and the fused routed tail.
    let mixer_joins = matches!(paired.mixer_input, SublayerSource::Norm(_))
        && (!matches!(paired.mixer, Mixer::Attention(_))
            || post_norm_epsilon(paired.mixer_output).is_none());
    let feed_forward_joins = paired.feed_forward.is_some_and(|sublayer| {
        matches!(sublayer.op, FeedForward::Routed(_))
            && matches!(sublayer.input, Some(SublayerSource::Norm(_)))
    });
    if mixer_joins || feed_forward_joins {
        constants.push(GraphConstant::zeros_value(rows, 0)?);
    }
    Ok(constants)
}
