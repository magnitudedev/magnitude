//! A decoder block's sublayers as graph fragments: the one dispatch from a
//! block's admitted operators (`PairedBlock`) and their kernel entries
//! (prepared kernels or checked bindings) to each operator's own fragment
//! (`operators::<op>::graph`). The target graph builds a block by calling
//! the mixer, the feed-forward and the per-layer input in order.

use super::attention::graph::{
    AttentionGraphEntries, AttentionOperands, CheckedAttentionEntries, HistoryTilesPort,
    IndexerControlPorts, IndexerStatePorts, attention, attention_weights,
};
use super::attention::split::{
    AttentionBlock, AttentionSite, AttentionTopology, SelectionSite, split as attention_split,
};
use crate::programs::graph::facts::Numbers;
use seismic::{Argument, Table};
use super::attention::pooled::{
    IndexBlock, OwnBlock, PickedBlock, PickedPlane, PickedPorts, PooledSource, PooledWindow,
};
use super::attention::PickedSource;
use super::dense_ffn::graph::{CheckedDenseEntries, DenseGraphEntries, DenseNumbers, dense};
use super::gated_delta::graph::{
    CheckedRecurrentEntries, RecurrentBlock, RecurrentControlPorts, RecurrentGraphEntries,
    RecurrentStatePorts, RecurrentTopology, recurrent,
};
use super::ngram::graph::{
    NGramEntries, NGramNumbers, NGramPorts, NGramStepPorts, NGramTopology, ngram as ngram_graph,
};
use super::parallel::DenseBesideRouted;
use super::parallel::graph::{
    CheckedParallelEntries, ParallelGraphEntries, ParallelNumbers, ParallelTopology, parallel,
};
use super::per_layer::graph::{
    PerLayerEntries, PerLayerNumbers, PerLayerWeights, per_layer as per_layer_graph,
};
use super::routed::fused_graph::{
    self as routed_graph, CheckedRoutedEntries, ExpertShape, RoutedGraphEntries, RoutedTopology,
};
use super::routed::graph::{
    CheckedGeneralRoutedEntries, GeneralRoutedGraphEntries, GeneralRoutedLayer, GeneralRoutedTopology,
    RoutedSum, general_pass, general_routed, split as general_split,
};
use super::routed::split::{PassPorts, Split};
use super::routed::streamed::{BankForm, BankNumbers, LayerBank, scratch_fill};
use super::routed::ExpansionForm;
use super::output::PostNorm;
use super::short_conv::graph::{
    CheckedShortConvEntries, ShortConvBlock, ShortConvGraphEntries, ShortConvTopology, short_conv,
};
use super::state_space::graph::{
    CheckedStateSpaceEntries, StateSpaceBlock, StateSpaceGraphEntries, StateSpaceTopology, state_space,
};
use super::streams::graph::{SiteEntry, StreamsEntries, join as stream_join, mix_site, sinkhorn_site};
use super::streams::{ResidualCarry, SinkhornTopology, StreamsGeometry};
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
    PlannedWeights, WeightPort, activation, packed_weight, scaled_weight, weight,
};
use crate::{
    FeedForwardProgramSlot, GeneralRoutedShape, MixerProgramSlot, ModelLoadPlan, NGramBinding,
    PerLayerBinding, StateResourcePlan,
};
use magnitude_family_contracts::{
    Attention, Decoder, OutputForm, SublayerIndex, WeightKind, WeightScope,
};
use magnitude_state::LayerRef;
use seismic::{
    BackendName, Dim, Element, NativeGraph, NativeGraphMetadata, NativePort,
    WorkflowTensor,
};

/// The post-norm tail of a sublayer ending in `output`, at the layer's
/// output `scale`; none of a residual tail.
fn post_norm(output: &OutputForm, scale: f32) -> Option<PostNorm<f64>> {
    post_norm_epsilon(output).map(|epsilon| PostNorm { epsilon, scale: f64::from(scale) })
}

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
    /// An index-list source's planes and tables.
    pub picked: Option<PickedPorts>,
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

/// What a sublayer reads of the residual: the residual under its input
/// norm, joined as one stream; a low-rank mix site's partials at its norm's
/// epsilon; or the input row of a Sinkhorn site of the decoder's numerics
/// (`SublayerGraph::read`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ReadTopology<R = Argument<f64>> {
    Norm,
    Mix { epsilon: R },
    Sinkhorn(SinkhornTopology<R>),
}

impl<R> Numbers<R> for ReadTopology<R> {
    type Of<E> = ReadTopology<E>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> ReadTopology<E> {
        match self {
            Self::Norm => ReadTopology::Norm,
            Self::Mix { epsilon } => ReadTopology::Mix { epsilon: f(epsilon) },
            Self::Sinkhorn(mixing) => ReadTopology::Sinkhorn(mixing.map(&mut *f)),
        }
    }
}

impl ReadTopology<f64> {
    /// The read of `source` over the decoder's `streams`.
    fn of(source: SublayerSource<'_>, streams: Option<&StreamsGeometry>) -> Result<Self, GraphError> {
        Ok(match source {
            SublayerSource::Norm(_) => Self::Norm,
            SublayerSource::Mix(_) => Self::Mix { epsilon: source.epsilon() },
            SublayerSource::Sinkhorn(_) => Self::Sinkhorn(
                streams
                    .and_then(StreamsGeometry::sinkhorn)
                    .ok_or("a Sinkhorn sublayer source without Sinkhorn streams")?
                    .topology(),
            ),
        })
    }
}

impl<R> ReadTopology<R> {
    /// How the sublayer's consumer entry forms its input over `streams`
    /// residual streams (`InputForm::read`).
    fn form(&self, streams: u64) -> InputForm {
        InputForm::read(matches!(self, Self::Mix { .. }), streams)
    }
}

/// A block's mixer as its graph branches on it.
/// Its reals `R` and its model-sourced tables `T` (attention's): the
/// model's, a key's indices, or its builder's arguments and tables.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MixerTopology<R = Argument<f64>, T = Table> {
    Attention(AttentionTopology<R, T>),
    Recurrent(RecurrentTopology<R>),
    StateSpace(StateSpaceTopology<R>),
    ShortConv(ShortConvTopology<R>),
}

impl<R, T> MixerTopology<R, T> {
    /// It with its tables `f` of its own.
    pub(crate) fn tables<U>(self, f: impl FnMut(T) -> U) -> MixerTopology<R, U> {
        match self {
            Self::Attention(topology) => MixerTopology::Attention(topology.tables(f)),
            Self::Recurrent(topology) => MixerTopology::Recurrent(topology),
            Self::StateSpace(topology) => MixerTopology::StateSpace(topology),
            Self::ShortConv(topology) => MixerTopology::ShortConv(topology),
        }
    }
}

impl<R, T> Numbers<R> for MixerTopology<R, T> {
    type Of<E> = MixerTopology<E, T>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> MixerTopology<E, T> {
        match self {
            Self::Attention(topology) => MixerTopology::Attention(topology.map(f)),
            Self::Recurrent(topology) => MixerTopology::Recurrent(topology.map(f)),
            Self::StateSpace(topology) => MixerTopology::StateSpace(topology.map(f)),
            Self::ShortConv(topology) => MixerTopology::ShortConv(topology.map(f)),
        }
    }
}

/// A block's mixer's numbers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MixerNumbers<D = Dim> {
    Attention(AttentionBlock<D>),
    Recurrent(RecurrentBlock<D>),
    StateSpace(StateSpaceBlock<D>),
    ShortConv(ShortConvBlock<D>),
}

impl<D> Numbers<D> for MixerNumbers<D> {
    type Of<E> = MixerNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> MixerNumbers<E> {
        match self {
            Self::Attention(numbers) => MixerNumbers::Attention(numbers.map(f)),
            Self::Recurrent(numbers) => MixerNumbers::Recurrent(numbers.map(f)),
            Self::StateSpace(numbers) => MixerNumbers::StateSpace(numbers.map(f)),
            Self::ShortConv(numbers) => MixerNumbers::ShortConv(numbers.map(f)),
        }
    }
}

/// A block's feed-forward as its graph branches on it: a dense one with
/// its post-norm tail's reals (none without one), or a routed form.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FeedForwardTopology<R = Argument<f64>> {
    Dense { expansion: ExpansionForm, post_norm: Option<PostNorm<R>> },
    Routed(RoutedTopology),
    GeneralRouted(GeneralRoutedTopology<R>),
    Parallel(ParallelTopology<R>),
}

impl<R> Numbers<R> for FeedForwardTopology<R> {
    type Of<E> = FeedForwardTopology<E>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> FeedForwardTopology<E> {
        match self {
            Self::Dense { expansion, post_norm } => {
                FeedForwardTopology::Dense { expansion, post_norm: post_norm.map(|post_norm| post_norm.map(&mut *f)) }
            }
            Self::Routed(topology) => FeedForwardTopology::Routed(topology),
            Self::GeneralRouted(topology) => FeedForwardTopology::GeneralRouted(topology.map(f)),
            Self::Parallel(topology) => FeedForwardTopology::Parallel(topology.map(f)),
        }
    }
}

/// A block's feed-forward's numbers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FeedForwardNumbers<D = Dim> {
    Dense(DenseNumbers<D>),
    Routed(ExpertShape<D>),
    GeneralRouted(GeneralRoutedLayer<D>),
    Parallel(ParallelNumbers<D>),
}

impl<D> Numbers<D> for FeedForwardNumbers<D> {
    type Of<E> = FeedForwardNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> FeedForwardNumbers<E> {
        match self {
            Self::Dense(numbers) => FeedForwardNumbers::Dense(numbers.map(f)),
            Self::Routed(numbers) => FeedForwardNumbers::Routed(numbers.map(f)),
            Self::GeneralRouted(numbers) => FeedForwardNumbers::GeneralRouted(numbers.map(f)),
            Self::Parallel(numbers) => FeedForwardNumbers::Parallel(numbers.map(f)),
        }
    }
}

/// Everything a block's sublayers branch on (`BlockSublayers::split`): its
/// lane, its norms' epsilon, and each sublayer's read of the residual and
/// operator topology (each with its own output scale). No model number.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SublayersTopology<R = Argument<f64>, T = Table> {
    pub lane: BlockLane,
    pub epsilon: R,
    pub mixer: (ReadTopology<R>, MixerTopology<R, T>),
    pub feed_forward: Option<(Option<ReadTopology<R>>, FeedForwardTopology<R>)>,
    /// A per-layer input sublayer's post-norm tail, which it always ends
    /// in (its layer and activation code are numbers,
    /// `SublayersNumbers::per_layer`).
    pub per_layer: Option<PostNorm<R>>,
    pub ngram: Option<NGramTopology<R, T>>,
}

impl<R, T> SublayersTopology<R, T> {
    /// It with its tables `f` of its own (`MixerTopology::tables`).
    pub(crate) fn tables<U>(self, mut f: impl FnMut(T) -> U) -> SublayersTopology<R, U> {
        let (read, mixer) = self.mixer;
        SublayersTopology {
            lane: self.lane,
            epsilon: self.epsilon,
            mixer: (read, mixer.tables(&mut f)),
            feed_forward: self.feed_forward,
            per_layer: self.per_layer,
            ngram: self.ngram.map(|topology| topology.tables(&mut f)),
        }
    }
}

impl<R, T> Numbers<R> for SublayersTopology<R, T> {
    type Of<E> = SublayersTopology<E, T>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> SublayersTopology<E, T> {
        let (read, mixer) = self.mixer;
        SublayersTopology {
            lane: self.lane,
            epsilon: f(self.epsilon),
            mixer: (read.map(&mut *f), mixer.map(&mut *f)),
            feed_forward: self
                .feed_forward
                .map(|(read, topology)| (read.map(|read| read.map(&mut *f)), topology.map(&mut *f))),
            per_layer: self.per_layer.map(|post_norm| post_norm.map(&mut *f)),
            ngram: self.ngram.map(|topology| topology.map(&mut *f)),
        }
    }
}

/// Every number a block's sublayers are sized by: the class's rows, slots,
/// segments and listed history tiles, the decoder's width and streams, and
/// each sublayer's own. Each operator's class numbers are the block's
/// (`share_classes`), so one fact varies each class dimension.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SublayersNumbers<D = Dim> {
    pub rows: D,
    pub slots: D,
    pub segments: D,
    pub listed: D,
    pub hidden: D,
    pub streams: Option<StreamsGeometry<D, D>>,
    pub mixer: MixerNumbers<D>,
    pub feed_forward: Option<FeedForwardNumbers<D>>,
    /// A per-layer input's numbers, its layer and its activation code.
    pub per_layer: Option<(PerLayerNumbers<D>, D, D)>,
    pub ngram: Option<NGramNumbers<D>>,
}

impl<D> Numbers<D> for SublayersNumbers<D> {
    type Of<E> = SublayersNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> SublayersNumbers<E> {
        SublayersNumbers {
            rows: f(self.rows),
            slots: f(self.slots),
            segments: f(self.segments),
            listed: f(self.listed),
            hidden: f(self.hidden),
            streams: self.streams.map(|streams| streams.map(&mut *f)),
            mixer: self.mixer.map(f),
            feed_forward: self.feed_forward.map(|numbers| numbers.map(&mut *f)),
            per_layer: self.per_layer.map(|(numbers, layer, activation)| (numbers.map(&mut *f), f(layer), f(activation))),
            ngram: self.ngram.map(|numbers| numbers.map(&mut *f)),
        }
    }
}

impl<D: Clone> SublayersNumbers<D> {
    /// The operators' class numbers made the block's own: a key indexed
    /// from it holds one fact for each class dimension, which its classes
    /// vary (`seismic::ClassFact`).
    pub(crate) fn share_classes(mut self) -> Self {
        let (rows, slots, segments) = (&self.rows, &self.slots, &self.segments);
        match &mut self.mixer {
            MixerNumbers::Attention(numbers) => numbers.share_classes(rows, segments, &self.listed),
            MixerNumbers::Recurrent(numbers) => (numbers.rows, numbers.slots) = (rows.clone(), slots.clone()),
            MixerNumbers::StateSpace(numbers) => (numbers.rows, numbers.slots) = (rows.clone(), slots.clone()),
            MixerNumbers::ShortConv(numbers) => (numbers.rows, numbers.slots) = (rows.clone(), slots.clone()),
        }
        if let Some(numbers) = &mut self.ngram {
            (numbers.rows, numbers.slots) = (rows.clone(), slots.clone());
        }
        self
    }
}

/// What a sublayer reads of the residual, and how its output joins it
/// (`SublayerGraph::read`).
struct SublayerRead {
    input: WorkflowTensor,
    form: InputForm,
    join: Join,
    /// The pre the next site reads: a Sinkhorn site's own when sites mix
    /// by the previous site's, or the one the residual carried here.
    pre: Option<WorkflowTensor>,
}

/// Whose block a graph fragment builds:the scope of its weights, and the
/// target layer whose history its attention mixer reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BlockLane {
    /// A target decoder block: its own weights, its own history.
    Target,
    /// A block of a draft head reading the history of target block `source`
    /// (`operators::SharedHead`).
    Head { source: u32 },
}

/// The model a block's sublayers are split from (`BlockSublayers::split`).
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
        Ok(scope_of(self.lane, self.index(sublayer)?))
    }

    /// The decoder site of the block's attention, whose other layers its
    /// index-list source resolves: a target block's, in the target decoder
    /// (a draft head's blocks read no index-list source).
    fn site(&self) -> Result<Option<attention_operator::DecoderSite<'_>>, String> {
        Ok(match self.lane {
            BlockLane::Target => Some(attention_operator::DecoderSite {
                decoder: self.geometry,
                index: self.index(0)?,
            }),
            BlockLane::Head { .. } => None,
        })
    }

    /// The rows of the index-list source `operator` reads at this class: an
    /// indexing source's indexer over its Block read, with an own source's
    /// domains (no Window at rate 1), or a reused source's plane. Each list
    /// and window read takes the class's `segments` spans.
    fn picked<'o>(
        &self,
        operator: &'o Attention,
        segments: u64,
    ) -> Result<Option<PickedBlock<'o>>, GraphError> {
        let rows = |rows: usize| u64::try_from(rows).map_err(|_| "pooled rows exceed u64");
        let state = self.state.target_state();
        let missing = "an index-list source has no pooled history";
        let plane = |PooledSource { block, part }| {
            let (window, plane) = state.pooled_history(block, part).ok_or(missing)?;
            Ok::<_, GraphError>((
                window,
                PickedPlane {
                    rows: rows(plane.rows)?,
                    slab_rows: plane.slab_rows,
                },
            ))
        };
        let plan = |error: crate::PlanError| GraphError::from(error.to_string());
        // An indexer's list over the Block read: its spans, the rows it lists
        // at the store's served context and the candidates it scores.
        let index = |part: u32, indexer: &'o magnitude_family_contracts::Indexer, rate: u64| {
            let (query_rounding, key_rounding) = attention_operator::indexer_roundings(indexer);
            Ok::<_, GraphError>(IndexBlock {
                indexer,
                spans: segments,
                scores: attention_operator::list_scores(rows(state.context_rows)?, rate),
                candidates: attention_operator::candidates(self.site()?, u64::from(part), indexer)
                    .map_err(plan)?,
                query_rounding,
                key_rounding,
            })
        };
        let Some(picked) = attention_operator::picked_source(operator).map_err(plan)? else {
            return Ok(None);
        };
        let (window, plane) = plane(picked.pooled(self.index(0)?.block))?;
        match picked {
            PickedSource::Reused { .. } => Ok(Some(PickedBlock::Reused { plane })),
            PickedSource::Reindex { source, part, indexer } => {
                let rate = attention_operator::shared_rows(self.geometry, source, part)
                    .map_err(plan)?
                    .rate;
                Ok(Some(PickedBlock::Indexed {
                    plane,
                    index: index(part, indexer, rate)?,
                    own: None,
                }))
            }
            PickedSource::Own { part, rows: source, indexer } => {
                // The incomplete group's tokens wait in the Window domain:
                // none at rate 1.
                let window = window
                    .map(|window| {
                        Ok::<_, GraphError>(PooledWindow {
                            rows: rows(window.rows)?,
                            slab_rows: window.slab_rows,
                        })
                    })
                    .transpose()?;
                Ok(Some(PickedBlock::Indexed {
                    plane,
                    index: index(part, indexer, source.rate)?,
                    own: Some(OwnBlock { rows: source, window }),
                }))
            }
        }
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

    /// The block's sublayers as their graphs branch on them and the numbers
    /// they are sized by, its routed layer reading its experts as `bank`
    /// says, its per-layer input and n-gram input bound by `per_layer` and
    /// `ngram`: the one place a block's model numbers are read.
    pub(crate) fn split<G: GraphDraft>(
        &self,
        bank: &LayerBank<'_, '_, G>,
        per_layer: Option<PerLayerBinding>,
        ngram: Option<NGramBinding>,
    ) -> Result<(SublayersTopology<f64, std::sync::Arc<[u8]>>, SublayersNumbers<u64>), GraphError> {
        let Self { paired, load, geometry, state, block_index, rows, segments, slots, listed, scored, output_scales, .. } =
            *self;
        let hidden = geometry.hidden;
        let epsilon = paired.epsilon() as f32;
        let component_index = bank_component_index(&geometry.blocks, block_index).map_err(|error| error.to_string())?;
        let streams = StreamsGeometry::of(geometry).map_err(|error| error.to_string())?;
        let (mixer, mixer_numbers) = match paired.mixer {
            Mixer::Attention(operator) => {
                let shape = attention_operator::shape(hidden, operator, self.site()?).map_err(|error| error.to_string())?;
                // The block reads its history domain's slab tensor, a Shared
                // layer its source's.
                let history = state
                    .target_state()
                    .layer_history(self.history_layer()?)
                    .ok_or("attention block has no history domain")?
                    .store;
                let (topology, numbers) = attention_split(&AttentionSite {
                    rows,
                    segments,
                    history_rows: u64::try_from(history.rows).map_err(|_| "history rows exceed u64")?,
                    slab_rows: history.slab_rows,
                    history_tiles: listed,
                    shape,
                    operator,
                    epsilon,
                    head_epsilon: attention_operator::head_norm_epsilon(operator, paired.epsilon())
                        .map_err(|error| error.to_string())? as f32,
                    post_norm: post_norm(paired.mixer_output, output_scales.mixer),
                    activation: activation(geometry),
                    inject_only: false,
                    // A head's rows read target history and append
                    // nowhere.
                    causal_cache: self.lane == BlockLane::Target,
                    selection: attention_operator::selection_shape(operator, geometry.context_limit)
                        .map(|shape| {
                            let (window, blocks) = state
                                .target_state()
                                .indexer_history(self.index(0)?.block)
                                .ok_or("a selected attention block has no indexer history")?;
                            Ok::<_, GraphError>(SelectionSite {
                                shape,
                                scored,
                                window_rows: u64::try_from(window.rows).map_err(|_| "indexer rows exceed u64")?,
                                window_slab_rows: window.slab_rows,
                                block_rows: u64::try_from(blocks.rows).map_err(|_| "indexer rows exceed u64")?,
                                block_slab_rows: blocks.slab_rows,
                            })
                        })
                        .transpose()?,
                    picked: self.picked(operator, segments)?,
                    controls: self.controls(),
                })?;
                (
                    MixerTopology::Attention(topology),
                    MixerNumbers::Attention(numbers),
                )
            }
            Mixer::GatedDelta(shape) => {
                let (topology, numbers) = gated_delta::graph::split(
                    shape,
                    hidden,
                    rows,
                    slots,
                    epsilon,
                    state,
                    component_index,
                    self.controls(),
                )?;
                (MixerTopology::Recurrent(topology), MixerNumbers::Recurrent(numbers))
            }
            Mixer::StateSpace(operator) => {
                let (topology, numbers) = super::state_space::graph::split(
                    operator,
                    hidden,
                    rows,
                    slots,
                    epsilon,
                    state,
                    component_index,
                    self.controls(),
                )?;
                (MixerTopology::StateSpace(topology), MixerNumbers::StateSpace(numbers))
            }
            Mixer::ShortConv(operator) => {
                let (topology, numbers) = super::short_conv::graph::split(
                    operator,
                    hidden,
                    rows,
                    slots,
                    epsilon,
                    post_norm(paired.mixer_output, output_scales.mixer),
                    state,
                    component_index,
                    self.controls(),
                )?;
                (MixerTopology::ShortConv(topology), MixerNumbers::ShortConv(numbers))
            }
        };
        let single = |source: Option<SublayerSource<'_>>, what: &str| -> Result<(), GraphError> {
            match source {
                None | Some(SublayerSource::Norm(_)) => Ok(()),
                Some(_) => Err(format!("a {what} feed-forward over residual streams").into()),
            }
        };
        let scope = self.scope(1)?;
        let (feed_forward, feed_forward_numbers) = match paired.feed_forward {
            None => (None, None),
            Some(sublayer) => {
                let (topology, numbers) = match sublayer.op {
                    FeedForward::Dense(operator) => {
                        if let Some(SublayerSource::Mix(_)) = sublayer.input {
                            return Err("a dense feed-forward over low-rank streams".into());
                        }
                        let (expansion, numbers) = dense_ffn::graph::split(load, scope, &operator.up)?;
                        let post_norm = post_norm(sublayer.output, output_scales.feed_forward);
                        (FeedForwardTopology::Dense { expansion, post_norm }, FeedForwardNumbers::Dense(numbers))
                    }
                    FeedForward::Routed(shape) => {
                        if sublayer.input.is_none() {
                            return Err("a routed feed-forward without its input source".into());
                        }
                        let (topology, numbers) =
                            routed_graph::split(load, scope, routed_graph::RoutedSource::Operator { routed: shape, hidden }, bank)?;
                        (FeedForwardTopology::Routed(topology), FeedForwardNumbers::Routed(numbers))
                    }
                    FeedForward::GeneralRouted(operator) => {
                        if let Some(SublayerSource::Mix(_)) = sublayer.input {
                            return Err("a general routed feed-forward over low-rank streams".into());
                        }
                        let shape =
                            GeneralRoutedShape::of(hidden, operator).map_err(|error| error.to_string())?;
                        let (topology, numbers) = general_split(load, scope, &shape, bank)?;
                        (FeedForwardTopology::GeneralRouted(topology), FeedForwardNumbers::GeneralRouted(numbers))
                    }
                    FeedForward::Parallel(branches) => {
                        single(sublayer.input, "parallel")?;
                        let (topology, numbers) =
                            super::parallel::graph::split(load, self.index(1)?, branches, hidden, output_scales.feed_forward, bank)?;
                        (FeedForwardTopology::Parallel(topology), FeedForwardNumbers::Parallel(numbers))
                    }
                };
                let read = sublayer.input.map(|source| ReadTopology::of(source, streams.as_ref())).transpose()?;
                (Some((read, topology)), Some(numbers))
            }
        };
        let (per_layer, per_layer_numbers) = match (paired.per_layer, per_layer) {
            (None, None) => (None, None),
            (Some(sublayer), Some(binding)) => (
                Some(
                    post_norm(sublayer.output, output_scales.per_layer)
                        .ok_or("a per-layer input sublayer ends in a post-norm tail")?,
                ),
                Some((
                    PerLayerNumbers::of(&binding),
                    u64::from(sublayer.op.layer),
                    super::code(dense_ffn::activation_code(sublayer.op.activation)),
                )),
            ),
            _ => return Err("target block per-layer sublayer and its binding disagree".into()),
        };
        let (ngram, ngram_numbers) = match (paired.ngram, ngram) {
            (None, None) => (None, None),
            (Some(sublayer), Some(binding)) => {
                let (topology, numbers) = super::ngram::graph::split(
                    binding,
                    &sublayer.op.hash,
                    rows,
                    slots,
                    sublayer.op.epsilon as f32,
                    state,
                    component_index + paired.mixer.bank_components(),
                    self.controls(),
                )?;
                (Some(topology), Some(numbers))
            }
            _ => return Err("target block n-gram input and its binding disagree".into()),
        };
        Ok((
            SublayersTopology {
                lane: self.lane,
                epsilon: f64::from(epsilon),
                mixer: (ReadTopology::of(paired.mixer_input, streams.as_ref())?, mixer),
                feed_forward,
                per_layer,
                ngram,
            },
            SublayersNumbers {
                rows,
                slots,
                segments,
                listed,
                hidden,
                streams: streams.map(StreamsGeometry::numbers),
                mixer: mixer_numbers,
                feed_forward: feed_forward_numbers,
                per_layer: per_layer_numbers,
                ngram: ngram_numbers,
            },
        ))
    }
}

/// The weight scope of sublayer `index` of a block of `lane`.
fn scope_of(lane: BlockLane, index: SublayerIndex) -> WeightScope {
    match lane {
        BlockLane::Target => WeightScope::TargetSublayer(index),
        BlockLane::Head { .. } => WeightScope::HeadSublayer(index),
    }
}

/// A block's sublayers' graph fragments, built from their split
/// (`BlockSublayers::split`) alone: at a recording's facts, or at the
/// model's own numbers (`facts::constant`), over the weights `weights`,
/// naming its sublayers' scopes as block `block`'s: a target block's graph
/// its own as block 0 (`relative_scope`), a head's its index in the head.
pub(crate) struct SublayerGraph<'s, W: ?Sized> {
    pub block: u32,
    pub topology: &'s SublayersTopology,
    pub numbers: &'s SublayersNumbers<Dim>,
    pub weights: &'s W,
}

impl<W: PlannedWeights + ?Sized> SublayerGraph<'_, W> {
    fn index(&self, sublayer: u32) -> SublayerIndex {
        SublayerIndex { block: self.block, sublayer }
    }

    fn scope(&self, sublayer: u32) -> WeightScope {
        scope_of(self.topology.lane, self.index(sublayer))
    }

    /// The decoder's residual streams; an error for a single stream.
    fn streams_geometry(&self) -> Result<&StreamsGeometry<Dim, Dim>, GraphError> {
        Ok(self.numbers.streams.as_ref().ok_or("a mix site without residual streams")?)
    }

    /// What sublayer `sublayer` reads of the residual `hidden` (`read`),
    /// and how its output joins it: the residual itself under its input
    /// norm, joined as one stream; a low-rank mix site's partials over the
    /// streams, joined by the scatter with the site's inject values; or a
    /// Sinkhorn site's input row under the sublayer's norm, joined by the
    /// scatter with the site's post | comb, the site's pre carried on when
    /// sites mix by the previous site's.
    #[allow(clippy::too_many_arguments)]
    fn read<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        read: ReadTopology,
        streams: Option<StreamsEntries<'e, G>>,
        sublayer: u32,
        hidden: &ResidualCarry<WorkflowTensor>,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<SublayerRead, GraphError> {
        let site = streams.and_then(|entries| entries.site(sublayer));
        let rows = &self.numbers.rows;
        match read {
            ReadTopology::Norm => Ok(SublayerRead {
                input: hidden.rows.clone(),
                form: InputForm::SINGLE,
                join: Join::single(Some("M")),
                pre: hidden.pre.clone(),
            }),
            ReadTopology::Mix { epsilon } => {
                let geometry = self.streams_geometry()?;
                let (Some(low_rank), Some(SiteEntry::LowRank(entries))) = (geometry.low_rank(), site) else {
                    return Err("a low-rank mix site without low-rank streams and their entries".into());
                };
                let site = mix_site(
                    graph,
                    entries,
                    self.weights,
                    self.scope(sublayer),
                    low_rank,
                    &hidden.rows,
                    rows.clone(),
                    epsilon.f32(),
                    true,
                    weights,
                )?;
                let inject = site.inject.ok_or("a sublayer's mix site without inject values")?;
                Ok(SublayerRead {
                    input: site.partials,
                    form: read.form(geometry.streams),
                    join: Join::streams(geometry, inject, Some("M")),
                    pre: None,
                })
            }
            ReadTopology::Sinkhorn(mixing) => {
                let geometry = self.streams_geometry()?;
                let Some(SiteEntry::Sinkhorn(entry)) = site else {
                    return Err("a Sinkhorn mix site without Sinkhorn streams and their entries".into());
                };
                let site = sinkhorn_site(
                    graph,
                    entry,
                    self.weights,
                    self.scope(sublayer),
                    geometry.clone(),
                    mixing,
                    hidden,
                    rows.clone(),
                    weights,
                    constants,
                )?;
                Ok(SublayerRead {
                    input: site.input,
                    form: read.form(geometry.streams),
                    join: Join::streams(geometry, site.join, Some("M")),
                    pre: site.pre,
                })
            }
        }
    }

    /// The block's mixer over `carry`: the new residual, and the block's
    /// state and control ports.
    /// `mixer`'s class scope where it keeps a bank: its request slots
    /// (`B`), which no other entry of the block reads by that name.
    pub(crate) fn mixer<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        entries: MixerEntries<'e, G>,
        streams: Option<StreamsEntries<'e, G>>,
        carry: &ResidualCarry<WorkflowTensor>,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(ResidualCarry<WorkflowTensor>, BlockState), GraphError> {
        let scope = self.scope(0);
        let planned = self.weights;
        let (read, ref mixer) = self.topology.mixer;
        let SublayerRead { input: input_rows, form, join, pre } = self.read(graph, read, streams, 0, carry, weights, constants)?;
        let hidden = &carry.rows;
        let input = SublayerInput { rows: &input_rows, form };
        let (mixed, block_state) = match (mixer, &self.numbers.mixer, entries) {
            (
                MixerTopology::Attention(topology),
                MixerNumbers::Attention(numbers),
                MixerEntries::Attention(entries),
            ) => {
                let operands =
                    AttentionOperands::planned(graph, planned, scope, topology, numbers, true, weights, constants)?;
                let attention_weights = attention_weights(topology, numbers, form, true, operands, |kind| {
                    weight(graph, planned, scope, kind, weights)
                })?;
                let (mixed, ports, controls) =
                    attention(graph, entries, topology, numbers, &attention_weights, constants, input, hidden, &join)?;
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
                            picked: ports.picked,
                        }),
                        bank: None,
                    },
                )
            }
            (MixerTopology::Recurrent(topology), MixerNumbers::Recurrent(numbers), MixerEntries::Recurrent(entries)) => {
                graph.set_class_scope(Some(RECURRENT_SLOTS));
                let (mixed, ports, controls) = recurrent(
                    graph, entries, topology, numbers, planned, scope, weights, constants, input, hidden, &join,
                )?;
                graph.set_class_scope(None);
                (mixed, (ports, controls).into())
            }
            (
                MixerTopology::StateSpace(topology),
                MixerNumbers::StateSpace(numbers),
                MixerEntries::StateSpace(entries),
            ) => {
                graph.set_class_scope(Some(RECURRENT_SLOTS));
                let (mixed, ports, controls) =
                    state_space(graph, entries, topology, numbers, planned, scope, weights, constants, hidden, &join)?;
                graph.set_class_scope(None);
                (mixed, (ports, controls).into())
            }
            (MixerTopology::ShortConv(topology), MixerNumbers::ShortConv(numbers), MixerEntries::ShortConv(entries)) => {
                graph.set_class_scope(Some(RECURRENT_SLOTS));
                let (mixed, window, controls) =
                    short_conv(graph, entries, topology, numbers, planned, scope, weights, constants, hidden, &join)?;
                graph.set_class_scope(None);
                (mixed, BlockState::bank(vec![window], controls))
            }
            _ => return Err("target block mixer and its kernel entries disagree".into()),
        };
        Ok((ResidualCarry { rows: mixed, pre }, block_state))
    }

    /// The scope of the block's streamed routed layer and how it reads its
    /// experts: a parallel block's routed branch's.
    fn routed_layer(&self) -> Result<(WeightScope, &BankForm, Option<&BankNumbers>), GraphError> {
        match (&self.topology.feed_forward, &self.numbers.feed_forward) {
            (Some((_, FeedForwardTopology::Routed(topology))), Some(FeedForwardNumbers::Routed(numbers))) => {
                Ok((self.scope(1), &topology.bank, numbers.bank.as_ref()))
            }
            (
                Some((_, FeedForwardTopology::GeneralRouted(topology))),
                Some(FeedForwardNumbers::GeneralRouted(numbers)),
            ) => Ok((self.scope(1), &topology.bank, numbers.bank.as_ref())),
            (Some((_, FeedForwardTopology::Parallel(topology))), Some(FeedForwardNumbers::Parallel(numbers))) => Ok((
                DenseBesideRouted::scopes(self.index(1))[1],
                &topology.routed.bank,
                numbers.routed.bank.as_ref(),
            )),
            _ => Err("a block without a routed layer".into()),
        }
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
        let (scope, form, numbers) = self.routed_layer()?;
        scratch_fill(graph, feed_forward.bank, form, numbers, self.weights, scope, weights, constants)
    }

    /// A merge set's pass over the block's routed layer (`group_pass`,
    /// `general_pass`) over its `ports`, from its crossed rows into its
    /// merged ones.
    pub(crate) fn pass<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        feed_forward: FeedForwardGraph<'_, 'e, G>,
        ports: &mut PassPorts,
        blocks: &Dim,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<WorkflowTensor, GraphError> {
        let FeedForwardGraph { entries, bank } = feed_forward;
        let (planned, rows, scope) = (self.weights, &self.numbers.rows, self.scope(1));
        let topology = self.topology.feed_forward.as_ref().map(|(_, topology)| topology);
        match (topology, &self.numbers.feed_forward, entries) {
            (
                Some(FeedForwardTopology::Routed(topology)),
                Some(FeedForwardNumbers::Routed(numbers)),
                Some(FeedForwardEntries::Routed(entries)),
            ) => routed_graph::group_pass(
                graph, entries, bank, topology, numbers, planned, scope, weights, constants, ports, blocks, rows,
            ),
            (
                Some(FeedForwardTopology::GeneralRouted(topology)),
                Some(FeedForwardNumbers::GeneralRouted(numbers)),
                Some(FeedForwardEntries::GeneralRouted(entries)),
            ) => general_pass(
                graph, entries, bank, topology, numbers, planned, scope, weights, constants, ports, blocks, rows,
            ),
            (
                Some(FeedForwardTopology::Parallel(topology)),
                Some(FeedForwardNumbers::Parallel(numbers)),
                Some(FeedForwardEntries::Parallel(entries)),
            ) => general_pass(
                graph,
                entries.routed,
                bank,
                &topology.routed,
                &numbers.routed,
                planned,
                DenseBesideRouted::scopes(self.index(1))[1],
                weights,
                constants,
                ports,
                blocks,
                rows,
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
        let (planned, rows, scope) = (self.weights, &self.numbers.rows, self.scope(1));
        let epsilon = self.topology.epsilon.f32();
        let mixed = &carry.rows;
        // A split layer's front exports the carry it read; the others carry
        // their output rows on with the site's pre.
        let front = matches!(split, Some(Split::Front(_)));
        let carried = |output: WorkflowTensor, pre: Option<WorkflowTensor>| match front {
            true => carry.clone(),
            false => ResidualCarry { rows: output, pre },
        };
        let (read, topology, entries) = match (&self.topology.feed_forward, entries) {
            (None, None) => return Ok((carry.clone(), None)),
            (Some((read, topology)), Some(entries)) => (*read, topology, entries),
            _ => return Err("target block feed-forward and its kernel entries disagree".into()),
        };
        let output = match (topology, &self.numbers.feed_forward, entries) {
            (
                FeedForwardTopology::Dense { expansion, post_norm },
                Some(FeedForwardNumbers::Dense(numbers)),
                FeedForwardEntries::Dense(entries),
            ) => {
                let dense = |graph: &mut G,
                             weights: &mut Vec<(WeightPort, NativePort)>,
                             constants: &mut Vec<GraphConstant>,
                             input: &WorkflowTensor,
                             sum: &WorkflowTensor| {
                    dense(
                        graph, entries, *expansion, numbers, planned, scope, weights, constants, input, sum, rows, None,
                        epsilon, post_norm.as_ref(),
                    )
                };
                match read {
                    None | Some(ReadTopology::Norm) => dense(graph, weights, constants, mixed, mixed)?,
                    // Over Sinkhorn streams the dense branch sums onto zeros.
                    Some(read @ ReadTopology::Sinkhorn(_)) => {
                        let (joined, ()) = self.sinkhorn_branch(
                            graph,
                            read,
                            streams,
                            &carry,
                            weights,
                            constants,
                            false,
                            |graph, weights, constants, input| {
                                let zeros = GraphConstant::row_zeros(
                                    graph,
                                    Element::f32(),
                                    rows.clone(),
                                    &numbers.hidden,
                                    Some("M"),
                                )?;
                                let branch = dense(graph, weights, constants, input, zeros.port().tensor())?;
                                constants.push(zeros);
                                Ok((branch, ()))
                            },
                        )?;
                        return Ok((joined, None));
                    }
                    Some(ReadTopology::Mix { .. }) => return Err("a dense feed-forward over low-rank streams".into()),
                }
            }
            (
                FeedForwardTopology::Routed(topology),
                Some(FeedForwardNumbers::Routed(numbers)),
                FeedForwardEntries::Routed(entries),
            ) => {
                let read = read.ok_or("a routed feed-forward without its input source")?;
                let SublayerRead { input: input_rows, form, join, pre } =
                    self.read(graph, read, streams, 1, &carry, weights, constants)?;
                let (output, sequence) = routed_graph::routed(
                    graph,
                    entries,
                    bank,
                    topology,
                    numbers,
                    planned,
                    scope,
                    weights,
                    constants,
                    SublayerInput { rows: &input_rows, form },
                    mixed,
                    &join,
                    rows,
                    epsilon,
                    split,
                )?;
                return Ok((carried(output, pre), sequence));
            }
            (
                FeedForwardTopology::GeneralRouted(topology),
                Some(FeedForwardNumbers::GeneralRouted(numbers)),
                FeedForwardEntries::GeneralRouted(entries),
            ) => {
                let routed = |graph: &mut G,
                              weights: &mut Vec<(WeightPort, NativePort)>,
                              constants: &mut Vec<GraphConstant>,
                              input: &WorkflowTensor,
                              sum| {
                    general_routed(
                        graph, entries, bank, topology, numbers, planned, scope, weights, constants, input, sum, rows,
                        epsilon, split,
                    )
                };
                return match read {
                    None | Some(ReadTopology::Norm) => {
                        let (output, sequence) = routed(graph, weights, constants, mixed, RoutedSum::Residual)?;
                        Ok((carried(output, carry.pre.clone()), sequence))
                    }
                    // Over Sinkhorn streams the routed branch sums onto
                    // zeros (`RoutedSum::Branch`).
                    Some(read @ ReadTopology::Sinkhorn(_)) => self.sinkhorn_branch(
                        graph,
                        read,
                        streams,
                        &carry,
                        weights,
                        constants,
                        front,
                        |graph, weights, constants, input| routed(graph, weights, constants, input, RoutedSum::Branch),
                    ),
                    Some(ReadTopology::Mix { .. }) => Err("a general routed feed-forward over low-rank streams".into()),
                };
            }
            (
                FeedForwardTopology::Parallel(topology),
                Some(FeedForwardNumbers::Parallel(numbers)),
                FeedForwardEntries::Parallel(entries),
            ) => {
                let (output, sequence) = parallel(
                    graph,
                    entries,
                    bank,
                    topology,
                    numbers,
                    planned,
                    self.index(1),
                    weights,
                    constants,
                    mixed,
                    rows,
                    epsilon,
                    split,
                )?;
                return Ok((carried(output, carry.pre.clone()), sequence));
            }
            _ => return Err("target block feed-forward and its kernel entries disagree".into()),
        };
        Ok((ResidualCarry { rows: output, pre: carry.pre.clone() }, None))
    }

    /// A feed-forward branch behind its Sinkhorn site: `branch` runs over
    /// the site's input row and returns its F32 output `[rows, H]`, summed
    /// onto zeros, which `stream_join` joins into the streams `carry` holds
    /// by the site's post | comb. A split layer's front returns the carry it
    /// read, its back joining the branch.
    #[allow(clippy::too_many_arguments)]
    fn sinkhorn_branch<'e, G: GraphDraft + 'e, T>(
        &self,
        graph: &mut G,
        read: ReadTopology,
        streams: Option<StreamsEntries<'e, G>>,
        carry: &ResidualCarry<WorkflowTensor>,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
        front: bool,
        branch: impl FnOnce(
            &mut G,
            &mut Vec<(WeightPort, NativePort)>,
            &mut Vec<GraphConstant>,
            &WorkflowTensor,
        ) -> Result<(WorkflowTensor, T), GraphError>,
    ) -> Result<(ResidualCarry<WorkflowTensor>, T), GraphError> {
        let read = self.read(graph, read, streams, 1, carry, weights, constants)?;
        let (output, value) = branch(graph, weights, constants, &read.input)?;
        if front {
            return Ok((carry.clone(), value));
        }
        let entry = streams
            .and_then(|streams| streams.sinkhorn())
            .ok_or("a Sinkhorn join without its entry")?
            .join;
        let joined = stream_join(
            graph,
            entry,
            self.streams_geometry()?.clone(),
            &carry.rows,
            &output,
            read.join.values.as_ref().ok_or("a Sinkhorn site without its join values")?,
            self.numbers.rows.clone(),
        )?;
        Ok((ResidualCarry { rows: joined, pre: read.pre }, value))
    }

    /// The block's per-layer input sublayer over `residual`: the new
    /// residual and the per-layer rows port it reads.
    pub(crate) fn per_layer<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        entries: Option<PerLayerEntries<'e, G>>,
        residual: WorkflowTensor,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(WorkflowTensor, Option<NativePort>), GraphError> {
        let (tail, numbers, layer, activation, entries) = match (self.topology.per_layer, &self.numbers.per_layer, entries) {
            (None, None, None) => return Ok((residual, None)),
            (Some(tail), Some((numbers, layer, activation)), Some(entries)) => (tail, numbers, layer, activation, entries),
            _ => return Err("target block per-layer sublayer and its kernel entries disagree".into()),
        };
        let (planned, rows, scope) = (self.weights, &self.numbers.rows, self.scope(2));
        let per_layer_weights = PerLayerWeights {
            gate: scaled_weight(graph, planned, scope, WeightKind::PerLayerGate, weights, constants)?,
            projection: packed_weight(graph, planned, scope, WeightKind::PerLayerProjection, weights, constants)?,
            post_norm: weight(graph, planned, scope, WeightKind::PostNorm, weights)?,
        };
        let rows_port =
            graph.port_with_class_extent(Element::f32(), &[rows.clone(), &numbers.layers * &numbers.width], 0, "M")?;
        let out_rows = GraphConstant::row_identity(graph, rows.clone(), Some("M"))?;
        let advanced = per_layer_graph(
            graph,
            entries,
            &per_layer_weights,
            &residual,
            rows_port.tensor(),
            out_rows.port().tensor(),
            numbers,
            layer.word().ok_or("a per-layer index exceeds i32")?,
            super::scalar::<i32>(activation, "a per-layer activation code")?,
            rows,
            &tail,
        )?;
        constants.push(out_rows);
        Ok((advanced, Some(rows_port)))
    }

    /// The block's n-gram input over the streams `residual`, after its
    /// mixer's bank components: the new streams and what each step binds.
    /// Its windows join `block`'s bank under its controls, or, in a block
    /// whose mixer keeps none (an attention block, or a scratch-served
    /// block's back), are the block's bank under the slot tables it takes.
    pub(crate) fn ngram<'e, G: GraphDraft + 'e>(
        &self,
        graph: &mut G,
        entries: Option<NGramEntries<'e, G>>,
        block: &mut BlockState,
        residual: WorkflowTensor,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<(WorkflowTensor, Option<NGramStepPorts>), GraphError> {
        let (topology, numbers, entries) = match (&self.topology.ngram, &self.numbers.ngram, entries) {
            (None, None, None) => return Ok((residual, None)),
            (Some(topology), Some(numbers), Some(entries)) => (topology, numbers, entries),
            _ => return Err("target block n-gram input and its kernel entries disagree".into()),
        };
        let (streams, ports) = ngram_graph(
            graph,
            entries,
            topology,
            numbers,
            self.weights,
            self.scope(2),
            weights,
            constants,
            &residual,
            block.bank.as_ref().map(|bank| &bank.controls),
        )?;
        let NGramPorts { bank, step, controls } = ports;
        block.join_ngram(bank, controls)?;
        Ok((streams, Some(step)))
    }
}

/// The class scope of a recurrent mixer's nodes and ports, whose `B` is the
/// block's request slots: a block's other entries call other dimensions `B`
/// (a routed feed-forward's expert tiles).
pub(crate) const RECURRENT_SLOTS: &str = "recurrent_slots";
