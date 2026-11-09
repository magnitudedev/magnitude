//! A decoder block's sublayers as graph fragments: the one dispatch from a
//! block's admitted operators (`PairedBlock`) and their kernel entries
//! (prepared kernels or checked bindings) to each operator's own fragment
//! (`operators::<op>::graph`). The target graph builds a block by calling
//! the mixer, the feed-forward and the per-layer input in order.

use super::attention::graph::{
    AttentionBlock, AttentionGraphEntries, AttentionOperands, CheckedAttentionEntries,
    HistoryTilesPort, IndexerControlPorts, IndexerStatePorts, SelectionBlock, attention,
    attention_weights,
};
use super::dense_ffn::graph::{CheckedDenseEntries, DenseGraphEntries, dense};
use super::gated_delta::graph::{
    CheckedRecurrentEntries, RecurrentBlock, RecurrentControlPorts, RecurrentGraphEntries,
    RecurrentStatePorts, recurrent,
};
use super::ngram::graph::{
    NGramBlock, NGramEntries, NGramPorts, NGramStepPorts, ngram as ngram_graph,
};
use super::parallel::DenseBesideRouted;
use super::parallel::graph::{CheckedParallelEntries, ParallelGraphEntries, parallel};
use super::per_layer::graph::{
    CheckedPerLayerEntries, PerLayerEntries, PerLayerWeights, per_layer as per_layer_graph,
};
use super::routed::fused_graph::{
    self as routed_graph, CheckedRoutedEntries, ExpertShape, RoutedGraphEntries,
};
use super::routed::graph::{
    CheckedGeneralRoutedEntries, GeneralRoutedGraphEntries, RoutedSum,
    general_pass, general_routed,
};
use super::routed::Expansion;
use super::routed::split::{PassPorts, Split};
use super::routed::streamed::{LayerBank, scratch_fill};
use super::short_conv::graph::{
    CheckedShortConvEntries, ShortConvBlock, ShortConvGraphEntries, short_conv,
};
use super::state_space::graph::{
    CheckedStateSpaceEntries, StateSpaceBlock, StateSpaceGraphEntries, state_space,
};
use super::streams::graph::{SiteEntry, StreamsEntries, join as stream_join, mix_site, sinkhorn_site};
use super::streams::{ResidualCarry, StreamsGeometry};
use super::{
    FeedForward, InputForm, Join, Mixer, PairedBlock, SublayerInput, SublayerSource,
    attention as attention_operator, bank_component_index, dense_ffn, gated_delta,
    post_norm_epsilon,
};
use crate::native::{AttestedFeedForward, AttestedMixer, OutputScales};
use crate::programs::graph::{
    GraphError,
    draft::{ControlSource, GraphDraft},
};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{
    WeightPort, activation, packed_weight, scaled_weight, weight,
};
use crate::{
    FeedForwardProgramSlot, GeneralRoutedShape, MixerProgramSlot, ModelLoadPlan, NGramBinding,
    PerLayerBinding, ShortConvShape, StateResourcePlan, StateSpaceShape,
};
use magnitude_family_contracts::{
    Decoder, RecurrentHeadMapping, SublayerIndex, WeightKind, WeightScope,
};
use magnitude_state::LayerRef;
use seismic::{
    BackendName, Element, NativeGraph, NativeGraphClassSlice, NativeGraphMetadata, NativePort,
    WorkflowTensor,
};

/// A block's state ports and their per-launch controls: its attention
/// mixer's history, and the recurrent bank its mixer or its n-gram input
/// keeps (an n-gram input's windows join a recurrent mixer's bank, or are
/// the block's bank beside an attention history).
#[derive(Clone, Default)]
pub(crate) struct BlockState {
    pub history: Option<HistoryPorts>,
    pub bank: Option<BankPorts>,
}

/// An attention layer's history planes and its launch's controls.
#[derive(Clone)]
pub(crate) struct HistoryPorts {
    /// The planes in the codec's plane-descriptor order.
    pub planes: Vec<NativePort>,
    /// A selected read's indexer planes.
    pub indexer_planes: Option<IndexerStatePorts>,
    pub coordinates: NativePort,
    pub visible: NativePort,
    pub fresh: NativePort,
    pub destinations: NativePort,
    pub history_tiles: Option<HistoryTilesPort>,
    /// A selected read's indexer tables.
    pub indexer: Option<IndexerControlPorts>,
}

/// A block's recurrent bank: its components, consecutive in the store's
/// component order (`operators::PairedBlock::bank_components`), and its
/// launch's slot tables.
#[derive(Clone)]
pub(crate) struct BankPorts {
    pub components: Vec<NativePort>,
    pub controls: RecurrentControlPorts,
}

impl BlockState {
    fn bank(components: Vec<NativePort>, controls: RecurrentControlPorts) -> Self {
        Self {
            history: None,
            bank: Some(BankPorts { components, controls }),
        }
    }

    /// The capacities of its history planes' and bank components' vectors.
    pub(crate) fn capacities(&self) -> [usize; 2] {
        [
            self.history.as_ref().map_or(0, |history| history.planes.capacity()),
            self.bank.as_ref().map_or(0, |bank| bank.components.capacity()),
        ]
    }

    /// Leave each vector holding exactly its ports, so a prepared graph's
    /// host heap follows from its topology.
    pub(crate) fn shrink_to_fit(&mut self) {
        if let Some(history) = &mut self.history {
            history.planes.shrink_to_fit();
        }
        if let Some(bank) = &mut self.bank {
            bank.components.shrink_to_fit();
        }
    }

    /// Its kind, as a block's timing names it.
    pub(crate) fn kind(&self) -> &'static str {
        match (&self.history, &self.bank) {
            (Some(_), _) => "attention",
            (None, Some(_)) => "recurrent",
            (None, None) => "back",
        }
    }

    /// Join an n-gram input's bank components to the block's: its mixer's
    /// bank, under its controls, or the block's own under the slot tables
    /// the input took (`controls`), beside a history or alone (a back).
    fn join_ngram(
        &mut self,
        components: Vec<NativePort>,
        controls: Option<RecurrentControlPorts>,
    ) -> Result<(), GraphError> {
        match (&mut self.bank, controls) {
            (Some(bank), None) => bank.components.extend(components),
            (None, Some(controls)) => self.bank = Some(BankPorts { components, controls }),
            _ => return Err("an n-gram input's slot tables disagree with its block".into()),
        }
        Ok(())
    }
}

impl From<(RecurrentStatePorts, RecurrentControlPorts)> for BlockState {
    fn from((ports, controls): (RecurrentStatePorts, RecurrentControlPorts)) -> Self {
        Self::bank(vec![ports.window, ports.delta, ports.tape], controls)
    }
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

/// A block's feed-forward as its graph builds it: its entries (none of a
/// lone mixer block), and how a routed feed-forward reads its experts
/// (`LayerBank`).
pub(crate) struct FeedForwardGraph<'p, 'a, G: GraphDraft + 'a> {
    pub entries: Option<FeedForwardEntries<'a, G>>,
    pub bank: LayerBank<'p, 'a, G>,
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
                Self::Attention(CheckedAttentionEntries::new(binding, backend, lists)?)
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

/// What a sublayer reads of the residual, and how its output joins it
/// (`BlockSublayers::read`).
struct SublayerRead {
    input: WorkflowTensor,
    form: InputForm,
    join: Join,
    /// A Sinkhorn site's pre, which the next site reads.
    pre: Option<WorkflowTensor>,
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
    /// How the block's graph takes its launch's controls: a target block
    /// is one of the launch's graphs, one a block, so it binds the launch's
    /// one upload of them; a draft head's blocks are one graph that
    /// uploads them itself.
    fn controls(&self) -> ControlSource {
        match self.lane {
            BlockLane::Target => ControlSource::Bound,
            BlockLane::Head { .. } => ControlSource::Uploaded,
        }
    }

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

    /// The decoder's residual streams; an error for a single stream.
    fn streams_geometry(&self) -> Result<StreamsGeometry, GraphError> {
        Ok(StreamsGeometry::of(self.geometry)
            .map_err(|error| error.to_string())?
            .ok_or("a mix site without residual streams")?)
    }

    /// What sublayer `sublayer` reads of the residual `hidden` (its
    /// `source`), and how its output joins it: the residual itself under its
    /// input norm, joined as one stream; a low-rank mix site's partials over
    /// the streams, joined by the scatter with the site's inject values; or
    /// a Sinkhorn site's input row under the sublayer's norm, joined by the
    /// scatter with the site's post | comb, the site's pre carried on.
    #[allow(clippy::too_many_arguments)]
    fn read<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        source: SublayerSource<'_>,
        streams: Option<StreamsEntries<'e, G>>,
        sublayer: u32,
        hidden: &ResidualCarry<WorkflowTensor>,
        weights: &mut Vec<(WeightPort, NativePort)>,
    ) -> Result<SublayerRead, GraphError> {
        let site = streams.and_then(|entries| entries.site(sublayer));
        match source {
            SublayerSource::Norm(_) => Ok(SublayerRead {
                input: hidden.rows.clone(),
                form: InputForm::SINGLE,
                join: Join::single(Some("M")),
                pre: hidden.pre.clone(),
            }),
            SublayerSource::Mix(_) => {
                let geometry = self.streams_geometry()?;
                let (Some(low_rank), Some(SiteEntry::LowRank(entries))) = (geometry.low_rank(), site) else {
                    return Err("a low-rank mix site without low-rank streams and their entries".into());
                };
                let site = mix_site(
                    graph,
                    entries,
                    self.load,
                    self.scope(sublayer)?,
                    low_rank,
                    &hidden.rows,
                    self.rows,
                    source.epsilon() as f32,
                    true,
                    weights,
                )?;
                let inject = site
                    .inject
                    .ok_or("a sublayer's mix site without inject values")?;
                Ok(SublayerRead {
                    input: site.partials,
                    form: source.form(geometry.streams),
                    join: Join::streams(geometry, inject, Some("M")),
                    pre: None,
                })
            }
            SublayerSource::Sinkhorn(_) => {
                let geometry = self.streams_geometry()?;
                let (Some(mixing), Some(SiteEntry::Sinkhorn(entry))) = (geometry.sinkhorn(), site) else {
                    return Err("a Sinkhorn mix site without Sinkhorn streams and their entries".into());
                };
                let site = sinkhorn_site(
                    graph,
                    entry,
                    self.load,
                    self.scope(sublayer)?,
                    geometry,
                    mixing,
                    hidden,
                    self.rows,
                    weights,
                )?;
                Ok(SublayerRead {
                    input: site.input,
                    form: source.form(geometry.streams),
                    join: Join::streams(geometry, site.join, Some("M")),
                    pre: Some(site.pre),
                })
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
        carry: &ResidualCarry<WorkflowTensor>,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(ResidualCarry<WorkflowTensor>, BlockState), GraphError> {
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
        let SublayerRead { input: input_rows, form, join, pre } =
            self.read(graph, paired.mixer_input, streams, 0, carry, weights)?;
        let hidden = &carry.rows;
        let input = SublayerInput {
            rows: &input_rows,
            form,
        };
        let (mixed, block_state) = match (paired.mixer, entries) {
            (Mixer::Attention(operator), MixerEntries::Attention(entries)) => {
                let shape = attention_operator::shape(geometry.hidden, operator)
                    .map_err(|error| error.to_string())?;
                let post_norm_epsilon = post_norm_epsilon(paired.mixer_output);
                let operands = AttentionOperands::planned(
                    graph, load, scope, operator, &shape, true, weights, constants,
                )?;
                let attention_weights = attention_weights(
                    &shape,
                    operator,
                    form,
                    true,
                    post_norm_epsilon.is_some(),
                    operands,
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
                        controls: self.controls(),
                    },
                )?;
                (
                    mixed,
                    BlockState {
                        history: Some(HistoryPorts {
                            planes: ports.planes,
                            indexer_planes: ports.indexer,
                            coordinates: controls.coordinates,
                            visible: controls.visible.ok_or("attention visibility is absent")?,
                            fresh: controls.fresh.ok_or("attention fresh rows are absent")?,
                            destinations: controls.destinations,
                            history_tiles: controls.history_tiles,
                            indexer: controls.indexer,
                        }),
                        bank: None,
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
                        controls: self.controls(),
                    },
                )?;
                (mixed, (ports, controls).into())
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
                        controls: self.controls(),
                    },
                )?;
                (mixed, (ports, controls).into())
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
                        post_norm_epsilon: post_norm_epsilon(paired.mixer_output)
                            .unwrap_or_default() as f32,
                        post_norm_scale: output_scales.mixer,
                        component_index,
                        controls: self.controls(),
                    },
                )?;
                (mixed, BlockState::bank(vec![window], controls))
            }
            _ => return Err("target block mixer and its kernel entries disagree".into()),
        };
        Ok((ResidualCarry { rows: mixed, pre }, block_state))
    }

    /// A scratch-served step's fill of the block's streamed routed layer
    /// (`scratch_fill`: every routed form's fill is its group's).
    pub(crate) fn fill<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        feed_forward: FeedForwardGraph<'_, 'e, G>,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<WorkflowTensor, GraphError> {
        let scope = match self.paired.feed_forward.map(|sublayer| sublayer.op) {
            Some(FeedForward::Routed(_) | FeedForward::GeneralRouted(_)) => self.scope(1)?,
            // A parallel block's routed branch's.
            Some(FeedForward::Parallel(_)) => DenseBesideRouted::scopes(self.index(1)?)[1],
            _ => return Err("a fill of a block without a streamed routed layer".into()),
        };
        scratch_fill(graph, feed_forward.bank, self.load, scope, weights, constants)
    }

    /// A merge set's pass over the block's routed layer (`group_pass`,
    /// `general_pass`) over its `ports`, from its crossed rows into its
    /// merged ones.
    pub(crate) fn pass<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        feed_forward: FeedForwardGraph<'_, 'e, G>,
        ports: &mut PassPorts,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<WorkflowTensor, GraphError> {
        let FeedForwardGraph { entries, bank } = feed_forward;
        let (load, rows, hidden) = (self.load, self.rows, self.geometry.hidden);
        let general = |operator| GeneralRoutedShape::of(hidden, operator).map_err(|error| error.to_string());
        match (self.paired.feed_forward.map(|sublayer| sublayer.op), entries) {
            (Some(FeedForward::Routed(shape)), Some(FeedForwardEntries::Routed(entries))) => routed_graph::group_pass(
                graph,
                entries,
                bank,
                load,
                self.scope(1)?,
                weights,
                constants,
                ports,
                rows,
                hidden,
                &ExpertShape::of_operator(shape)?,
            ),
            (Some(FeedForward::GeneralRouted(operator)), Some(FeedForwardEntries::GeneralRouted(entries))) => {
                general_pass(
                    graph,
                    entries,
                    bank,
                    load,
                    self.scope(1)?,
                    weights,
                    constants,
                    ports,
                    rows,
                    &general(operator)?,
                )
            }
            (Some(FeedForward::Parallel(branches)), Some(FeedForwardEntries::Parallel(entries))) => general_pass(
                graph,
                entries.routed,
                bank,
                load,
                DenseBesideRouted::scopes(self.index(1)?)[1],
                weights,
                constants,
                ports,
                rows,
                &general(branches.routed)?,
            ),
            _ => Err("a pass of a block without a routed layer".into()),
        }
    }

    /// The block's feed-forward over the mixer's residual `carry`; a lone
    /// mixer block's output is `carry`. A routed feed-forward reads its
    /// experts as `FeedForwardGraph::bank` says, and returns the sequence
    /// input a run writes when it reads its sequence; a split one is built
    /// as its front, which exports the carry it read, or its back, which
    /// reads its site again from that carry (`split`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn feed_forward<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        feed_forward: FeedForwardGraph<'_, 'e, G>,
        streams: Option<StreamsEntries<'e, G>>,
        carry: ResidualCarry<WorkflowTensor>,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
        split: Option<Split<'_>>,
    ) -> Result<(ResidualCarry<WorkflowTensor>, Option<NativePort>), GraphError> {
        let FeedForwardGraph { entries, bank } = feed_forward;
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
        let mixed = &carry.rows;
        // A mix site's sublayer reads its site; the others read `mixed`.
        let read = |graph: &mut G,
                    weights: &mut Vec<(WeightPort, NativePort)>,
                    source: Option<SublayerSource<'_>>| {
            let source = source.ok_or("a routed feed-forward without its input source")?;
            self.read(graph, source, streams, 1, &carry, weights)
        };
        // A split layer's front exports the carry it read; the others carry
        // their output rows on with the site's pre.
        let front = matches!(split, Some(Split::Front(_)));
        let carried = |output: WorkflowTensor, pre: Option<WorkflowTensor>| match front {
            true => carry.clone(),
            false => ResidualCarry { rows: output, pre },
        };
        let single = |source: Option<SublayerSource<'_>>, what: &str| -> Result<(), GraphError> {
            match source {
                None | Some(SublayerSource::Norm(_)) => Ok(()),
                Some(_) => Err(format!("a {what} feed-forward over residual streams").into()),
            }
        };
        let output = match (paired.feed_forward, entries) {
            (None, None) => carry.rows.clone(),
            (Some(sublayer), Some(entries)) => match (sublayer.op, entries) {
                (FeedForward::Dense(operator), FeedForwardEntries::Dense(entries)) => {
                    single(sublayer.input, "dense")?;
                    dense(
                        graph,
                        entries,
                        load,
                        scope,
                        weights,
                        constants,
                        mixed,
                        rows,
                        epsilon,
                        Expansion::of(&operator.up),
                        post_norm_epsilon(sublayer.output).unwrap_or_default() as f32,
                        output_scales.feed_forward,
                    )?
                }
                (FeedForward::Routed(shape), FeedForwardEntries::Routed(entries)) => {
                    let SublayerRead { input: input_rows, form, join, pre } =
                        read(graph, weights, sublayer.input)?;
                    let (output, sequence) = routed_graph::routed(
                        graph,
                        entries,
                        bank,
                        load,
                        scope,
                        weights,
                        constants,
                        SublayerInput {
                            rows: &input_rows,
                            form,
                        },
                        mixed,
                        &join,
                        rows,
                        geometry.hidden,
                        &ExpertShape::of_operator(shape)?,
                        epsilon,
                        split,
                    )?;
                    return Ok((carried(output, pre), sequence));
                }
                (
                    FeedForward::GeneralRouted(operator),
                    FeedForwardEntries::GeneralRouted(entries),
                ) => {
                    let shape = GeneralRoutedShape::of(geometry.hidden, operator).map_err(|error| error.to_string())?;
                    let routed = |graph: &mut G,
                                  weights: &mut Vec<(WeightPort, NativePort)>,
                                  constants: &mut Vec<GraphConstant>,
                                  input: &WorkflowTensor,
                                  sum| {
                        general_routed(
                            graph, entries, bank, load, scope, weights, constants, input, sum, rows, &shape, epsilon,
                            split,
                        )
                    };
                    return match sublayer.input {
                        None | Some(SublayerSource::Norm(_)) => {
                            let (output, sequence) = routed(graph, weights, constants, mixed, RoutedSum::Residual)?;
                            Ok((carried(output, carry.pre.clone()), sequence))
                        }
                        // Over Sinkhorn streams the routed branch sums onto
                        // zeros, and `stream_join` joins it into the streams.
                        Some(source @ SublayerSource::Sinkhorn(_)) => {
                            let read = self.read(graph, source, streams, 1, &carry, weights)?;
                            let (branch, sequence) =
                                routed(graph, weights, constants, &read.input, RoutedSum::Branch)?;
                            if front {
                                return Ok((carry.clone(), sequence));
                            }
                            let entry = streams
                                .and_then(|streams| streams.sinkhorn())
                                .ok_or("a Sinkhorn join without its entry")?
                                .join;
                            let joined = stream_join(
                                graph,
                                entry,
                                self.streams_geometry()?,
                                mixed,
                                &branch,
                                read.join.values.as_ref().ok_or("a Sinkhorn site without its join values")?,
                                rows,
                            )?;
                            Ok((ResidualCarry { rows: joined, pre: read.pre }, sequence))
                        }
                        Some(SublayerSource::Mix(_)) => {
                            Err("a general routed feed-forward over low-rank streams".into())
                        }
                    };
                }
                (FeedForward::Parallel(branches), FeedForwardEntries::Parallel(entries)) => {
                    single(sublayer.input, "parallel")?;
                    let (output, sequence) = parallel(
                        graph,
                        entries,
                        bank,
                        load,
                        self.index(1)?,
                        weights,
                        constants,
                        mixed,
                        rows,
                        &GeneralRoutedShape::of(geometry.hidden, branches.routed)
                            .map_err(|error| error.to_string())?,
                        epsilon,
                        Expansion::of(&branches.dense.up),
                        output_scales.feed_forward,
                        split,
                    )?;
                    return Ok((carried(output, carry.pre.clone()), sequence));
                }
                _ => return Err("target block feed-forward and its kernel entries disagree".into()),
            },
            _ => return Err("target block feed-forward and its kernel entries disagree".into()),
        };
        Ok((ResidualCarry { rows: output, pre: carry.pre.clone() }, None))
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
                    gate: scaled_weight(
                        graph,
                        load,
                        scope,
                        WeightKind::PerLayerGate,
                        weights,
                        constants,
                    )?,
                    projection: packed_weight(
                        graph,
                        load,
                        scope,
                        WeightKind::PerLayerProjection,
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
                let out_rows = GraphConstant::row_identity(graph, rows, Some("M"))?;
                let advanced = per_layer_graph(
                    graph,
                    entries,
                    &per_layer_weights,
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
    /// mixer's bank components: the new streams and what each step binds.
    /// Its windows join `block`'s bank under its controls, or, in a block
    /// whose mixer keeps none (an attention block, or a scratch-served
    /// block's back), are the block's bank under the slot tables it takes.
    pub(crate) fn ngram<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        parts: Option<NGramParts<'e, G>>,
        block: &mut BlockState,
        residual: WorkflowTensor,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(WorkflowTensor, Option<NGramStepPorts>), GraphError> {
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
            block.bank.as_ref().map(|bank| &bank.controls),
            NGramBlock {
                rows,
                slots,
                slab_banks: state.target_state().bank_slab_banks()?,
                binding,
                hash: &sublayer.op.hash,
                epsilon: sublayer.op.epsilon as f32,
                component_index,
                controls: self.controls(),
            },
        )?;
        let NGramPorts { bank, step, controls } = ports;
        block.join_ngram(bank, controls)?;
        Ok((streams, Some(step)))
    }
}

/// A feed-forward's class dimensions for a `rows`-row launch: a routed
/// feed-forward's grouped entries size their tile table from the row count
/// alone, its combine (the fused combine, the general scatter) from the
/// rows its experts were grouped over, `grouped` (the launch's, or the
/// chain's after its group's pass).
pub(crate) fn feed_forward_class_slice(
    paired: &PairedBlock,
    rows: u64,
    grouped_rows: u64,
    mut slice: NativeGraphClassSlice,
) -> Result<NativeGraphClassSlice, String> {
    let grouped = match paired.feed_forward.map(|sublayer| sublayer.op) {
        Some(FeedForward::Routed(shape)) => Some((shape, &["routed_group", "routed_experts"][..], "routed_combine")),
        Some(
            FeedForward::GeneralRouted(shape)
            | FeedForward::Parallel(DenseBesideRouted { routed: shape, .. }),
        ) => Some((shape, &["routed_group", "routed_experts", "routed_experts_up"][..], "routed_scatter")),
        Some(FeedForward::Dense(_)) | None => None,
    };
    if let Some((shape, entries, combine)) = grouped {
        if !routed_graph::decodes(rows) {
            let blocks = routed_graph::grouped_blocks(rows, shape.experts, shape.selected)?;
            for entry in entries {
                slice = slice.scoped(entry, "B", [blocks]);
            }
            let blocks = routed_graph::grouped_blocks(grouped_rows, shape.experts, shape.selected)?;
            slice = slice.scoped(combine, "B", [blocks]);
        }
    }
    Ok(slice)
}

