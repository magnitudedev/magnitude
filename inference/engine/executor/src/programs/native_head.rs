//! The native draft-head program. One sealed graph per head class runs a
//! head transaction in a single device submission: the entry pass over the
//! committed rows, then one chained pass per further proposal. A pass is the
//! head block (`draft_rows`, the decoder's attention and feed-forward entries,
//! the final norm) and, when drafting, the vocabulary projection and the
//! position-keyed selection. A chained pass embeds the previous pass's
//! selection (`sample_rows` result rows are `draft_rows` token rows) and
//! conditions on its output feature, so no proposal returns to the host
//! before the chain ends. Each drafting pass publishes, beside its
//! selections, their probabilities under the logits it selected them from
//! (`readout::select`), so the host verifies the chain by the head's
//! confidence in each proposal. A causal-only transaction (priming,
//! catch-up) drafts nothing: its entry pass ends once the attention has
//! appended the rows' K/V history, the only thing a later draft reads of
//! them. A front launch runs behind a selecting target step of two-row
//! decode: its entry pass embeds the step's selections and is conditioned
//! by its features on the device, selects on every entry row, and at its
//! tail makes the next pass's chain from the step's token rows and
//! selections and its own (`guess_chain`).

use super::{DeviceSubmission, FrontOutput, HeadOutput, HeadProgram, HeadSelected, HeadSelections};
use crate::device_table::SelectedRead;
use crate::operators::streams::{
    BlockStreams, Sites, StreamsGeometry,
    graph::{CheckedStreamsEntries, MixSite, SiteEntries, StreamsEntries, mix_site},
};
use crate::operators::{
    self, CausalHeadBlock, InputForm, Join, Mixer, SublayerInput, paired_block,
};
use crate::{
    DeviceError, FeedForwardProgramSlot, HeadBinding, HeadLaunchCore,
    InvariantError, ModelLoadPlan, NativeGraphOutputLease, NativeGraphWorkspaceLease, ResidentHead,
    ResidentOutput, ResidentWeight, ResourceLimits, SubmitError, ValidatedHeadLaunch,
    completion::CompletionWaiter,
    native::{AttestedFeedForward, AttestedHead, AttestedHeadBlock, AttestedLanding},
    operators::attention::graph::{
        self as attention_graph, AttentionControlPorts, AttentionGraphEntries, AttentionWeights,
        CheckedAttentionEntries, IndexerHistoryPorts,
    },
    operators::attention::split::{AttentionBlock, AttentionSite, AttentionTopology, SelectionSite, split},
    planning::{DomainRows, HeadHistoryGeometry},
    operators::dense_ffn::graph::{
        CheckedDenseEntries, DenseGraphEntries, DenseNumbers, split as dense_split,
    },
    operators::routed::ExpansionForm,
    operators::streams::LowRankGeometry,
    programs::graph::facts::{Numbers, constant, constant_reals},
    operators::output::TailEntries,
    operators::routed::fused_graph::{self as routed, CheckedRoutedEntries, RoutedGraphEntries},
    operators::routed::graph::{
        CheckedGeneralRoutedEntries, GeneralRoutedGraphEntries, GeneralRoutedLayer, GeneralRoutedTopology,
        RoutedSum, general_routed, split as general_split,
    },
    operators::routed::streamed::LayerBank,
    programs::{
        graph::RowForm,
        graph::readout::{self, SelectionPorts, shapes},
        graph::{
            GraphError,
            draft::{Certified, ElementBoundDraft, GraphDraft},
            sizing::SizingOutline,
        },
        native_constants::{
            CheckedGraphFamilyResources, CheckedGraphResources, ConstantTensors, GraphConstant,
            distinct_storage_bytes,
        },
        native_target_graph::{
            complete, selection_scores,
            CheckedLandingEntries, GatheredTable, GraphWeights, LandedReadPorts, LandingEntries,
            TARGET_TABLE, WeightPort, weight,
        },
    },
};
use magnitude_batching::{HistoryTables, TargetBatchUpload, row_class};
use magnitude_family_contracts::{
    ActivationDType, Decoder, ProgressivePlane, SublayerIndex, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::feature_rows;
use magnitude_kernels::constants::{
    CHAINING_FRONT, CHAINING_ROW, CHAINING_WORDS, CONTINUING_DEPTH, CONTINUING_SLOT, CONTINUING_WORDS,
    VERIFY_ROWS,
};
use magnitude_kernels::{
    attention_project, draft_rows, draft_top_k, feature_copy, guess_chain, place_logits_rows,
    readout_features_rows, readout_project_rows, sample_rows, shape_rows, shortlist_token_rows,
    token_confidence,
};
use magnitude_state::LayerRef;
use seismic::{
    Argument, BackendName, BoundNativeGraphPlan, Device, Dim, Element, NativeGraph,
    NativeGraphFamily, NativeGraphLayout, NativeGraphMetadata,
    NativeGraphPlan, NativeGraphStorageBytes, NativePort, Table, Tensor, WorkflowTensor,
};
use std::{collections::BTreeMap, rc::Rc, sync::Arc};

fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native head program",
        detail: detail.into(),
    })
}
fn device(error: impl ToString) -> SubmitError {
    SubmitError::Device(DeviceError::Execution(error.to_string()))
}
fn i32_bytes(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn activation(dtype: ActivationDType) -> Element {
    match dtype {
        ActivationDType::F16 => Element::f16(),
        ActivationDType::BF16 => Element::bf16(),
    }
}

/// One head graph: the entry pass's row class, the head's state and what
/// its launch runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct HeadGraphClass {
    pub entry_rows: u64,
    pub history: HeadHistoryGeometry,
    pub launch: HeadLaunchClass,
}

/// What a head graph's launch runs over its entry rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum HeadLaunchClass {
    /// A transaction of the head's own: the slot class of its outputs and
    /// of every chained pass, and the number of selections per slot, 0 for
    /// a causal-only transaction (priming, catch-up), which selects nothing.
    Transaction {
        slots: u64,
        steps: u64,
        /// Whether selection is shaped first.
        shaped: bool,
        /// Whether the head's indexer scores: some row of some pass has
        /// more complete units than it picks (`SelectionSite::scored`).
        /// Only a drafting graph of a head that selects takes both.
        scored: bool,
        /// Whether a causal-only entry of one slot reads its conditioning
        /// from a target feature output on the device (a prompt chunk's
        /// entry run behind the chunk) rather than host-written rows.
        device: bool,
        prefix: bool,
    },
    /// A front launch behind a selecting target step of two-row decode:
    /// its entry tokens are the step's selections on the device, each
    /// entry row selects once (its slots are its entry rows), and its tail
    /// makes the next pass's chain (`guess_chain`). A decode pass's has at
    /// most the decode form's rows (VERIFY_ROWS) and is conditioned by the
    /// step's feature rows; the prompt's closing chunk's (`prefix`) is one
    /// entry row conditioned by the chunk's boundary feature.
    Front { shaped: bool, scored: bool, prefix: bool },
}

impl HeadGraphClass {
    /// The slot class of its outputs and of every chained pass.
    pub(crate) fn slots(&self) -> u64 {
        match self.launch {
            HeadLaunchClass::Transaction { slots, .. } => slots,
            HeadLaunchClass::Front { .. } => self.entry_rows,
        }
    }

    /// Selections per slot.
    pub(crate) fn steps(&self) -> u64 {
        match self.launch {
            HeadLaunchClass::Transaction { steps, .. } => steps,
            HeadLaunchClass::Front { .. } => 1,
        }
    }

    pub(crate) fn scored(&self) -> bool {
        match self.launch {
            HeadLaunchClass::Transaction { scored, .. } | HeadLaunchClass::Front { scored, .. } => scored,
        }
    }

    /// The span class every head graph is sealed with: the most spans any
    /// history of the head's store presents to a launch, to a power of two.
    pub(crate) fn segments(&self) -> Result<u64, String> {
        Self::segments_of(&self.history)
    }

    pub(crate) fn segments_of(history: &HeadHistoryGeometry) -> Result<u64, String> {
        u64::try_from(
            history
                .span_limit
                .checked_next_power_of_two()
                .ok_or("head segment class overflows")?,
        )
        .map_err(|_| "head segment class exceeds u64".into())
    }
}

/// The admitted head classes used by both prepared formation and metadata
/// assessment. A causal head has one pass, conditioned by host rows or, for
/// one slot, by the target's features on the device. A drafting head chains
/// up to the method's proposal bound. A load that `fronts`
/// (`ModelLoadPlan::fronts`) runs both of a pass's forms (§7.1): its head
/// selects in front launches behind front passes, and drafts in
/// transactions the proposals a verify pass runs on.
pub(crate) fn head_graph_classes(
    limits: ResourceLimits,
    history: HeadHistoryGeometry,
    proposals: usize,
    fronts: bool,
) -> Result<Vec<HeadGraphClass>, String> {
    let row_classes = magnitude_batching::row_classes(limits.max_launch_rows)
        .into_iter()
        .map(|rows| rows as u64)
        .collect::<Vec<_>>();
    let max_rows = *row_classes.last().ok_or_else(|| {
        format!(
            "launch row bound {} has no row class",
            limits.max_launch_rows
        )
    })?;
    let slot_bound = limits.max_launch_slots.min(limits.max_launch_rows);
    let slot_classes = magnitude_batching::row_classes(slot_bound)
        .into_iter()
        .map(|slots| slots as u64)
        .collect::<Vec<_>>();
    let mut classes = Vec::new();
    let transaction = |entry_rows, slots, steps, (shaped, scored), device, prefix| HeadGraphClass {
        entry_rows,
        history,
        launch: HeadLaunchClass::Transaction { slots, steps, shaped, scored, device, prefix },
    };
    for &entry_rows in &row_classes {
        for &slots in slot_classes.iter().filter(|slots| **slots <= entry_rows) {
            classes.push(transaction(entry_rows, slots, 0, (false, false), false, false));
        }
        for prefix in [false, true] {
            classes.push(transaction(entry_rows, 1, 0, (false, false), true, prefix));
        }
    }
    // A head that selects does with and without its indexer scoring
    // (`HeadLaunchClass::Transaction::scored`), and shaped or not.
    let selecting = [false, true]
        .into_iter()
        .flat_map(|shaped| {
            [false, true]
                .into_iter()
                .take(1 + usize::from(history.indexer.is_some()))
                .map(move |scored| (shaped, scored))
        })
        .collect::<Vec<_>>();
    if fronts {
        // A decode pass's front launch, its entry rows the pass's rows (at
        // most the decode form's), and the prompt's closing chunk's, one row.
        for &(shaped, scored) in &selecting {
            let front = |entry_rows, prefix| HeadGraphClass {
                entry_rows,
                history,
                launch: HeadLaunchClass::Front { shaped, scored, prefix },
            };
            classes.extend(row_classes.iter().filter(|rows| **rows <= VERIFY_ROWS).map(|&rows| front(rows, false)));
            classes.push(front(1, true));
        }
    }
    // Drafting passes project the draft vocabulary per slot: their slots are
    // bounded by the selection bound, not the launch's request slots.
    for &slots in slot_classes
        .iter()
        .filter(|slots| **slots <= limits.max_drafting_slots as u64)
    {
        let entry_bound = row_class(
            (slots as usize)
                .saturating_mul(proposals + 1)
                .min(max_rows as usize),
        )
        .ok_or("head entry row bound has no class")? as u64;
        for &entry_rows in row_classes
            .iter()
            .filter(|rows| **rows >= slots && **rows <= entry_bound)
        {
            for steps in 1..=(proposals as u64) {
                for &selection in &selecting {
                    classes.push(transaction(entry_rows, slots, steps, selection, false, false));
                    if slots == 1 {
                        classes.push(transaction(entry_rows, 1, steps, selection, true, false));
                    }
                }
            }
        }
    }
    Ok(classes)
}

/// The per-run inputs of one pass.
struct PassPorts {
    controls: AttentionControlPorts,
    /// The head's history planes, in plane-descriptor order.
    planes: Vec<NativePort>,
    /// A selected read's indexer planes and tables.
    indexer: Option<IndexerHistoryPorts>,
    selection: Option<SelectionPorts>,
    /// A chained pass's landed read of the tokens it embeds, of a table
    /// with landing slots.
    landed: Option<LandedReadPorts>,
}

struct HeadGraphEntries<'a, G: GraphDraft + 'a> {
    /// The landing entries of the token table's landed reads, when it has
    /// landing slots.
    landing: Option<LandingEntries<'a, G>>,
    streams: Option<StreamsEntries<'a, G>>,
    gather: G::Binding<'a, feature_rows::Entry>,
    /// The head's input rows and its priming projection, by row range.
    input: G::Ranged<'a, draft_rows::Entry>,
    copy: G::Binding<'a, feature_copy::Entry>,
    attention: AttentionGraphEntries<'a, G>,
    priming_project: G::Ranged<'a, attention_project::Entry>,
    feed_forward: HeadFeedForwardEntries<'a, G>,
    features: G::Binding<'a, readout_features_rows::Entry>,
    logits: readout::ProjectionEntries<'a, G>,
    shortlist: Option<ShortlistEntries<'a, G>>,
    shape: G::Binding<'a, shape_rows::Entry>,
    sample: G::Binding<'a, sample_rows::Entry>,
    /// A drafting graph's proposals' confidences.
    confidence: G::Binding<'a, token_confidence::Entry>,
    /// Of a load that fronts, the chain at its front graphs' tail.
    chain: Option<G::Binding<'a, guess_chain::Entry>>,
}

struct ShortlistEntries<'a, G: GraphDraft + 'a> {
    coarse: G::Ranged<'a, readout_project_rows::Entry>,
    selection: G::Binding<'a, draft_top_k::Entry>,
    rows: G::Binding<'a, shortlist_token_rows::Entry>,
    placement: G::Binding<'a, place_logits_rows::Entry>,
}

enum HeadFeedForwardEntries<'a, G: GraphDraft + 'a> {
    Dense(DenseGraphEntries<'a, G>),
    Routed(RoutedGraphEntries<'a, G>),
    GeneralRouted(GeneralRoutedGraphEntries<'a, G>),
}

impl<'a> HeadGraphEntries<'a, NativeGraph> {
    fn prepared(
        block: &'a AttestedHeadBlock,
        head: &'a AttestedHead,
        landing: Option<&'a AttestedLanding>,
    ) -> Result<Self, String> {
        Ok(Self {
            landing: landing.map(LandingEntries::native),
            streams: block.streams.as_ref().map(Into::into),
            gather: &block.gather,
            input: &block.input,
            copy: &block.copy,
            attention: (&block.attention).into(),
            priming_project: &block.priming_project,
            feed_forward: match &block.feed_forward {
                AttestedFeedForward::Dense(handles) => {
                    HeadFeedForwardEntries::Dense(handles.into())
                }
                AttestedFeedForward::Routed(handles) => {
                    HeadFeedForwardEntries::Routed(handles.into())
                }
                AttestedFeedForward::GeneralRouted(handles) => {
                    HeadFeedForwardEntries::GeneralRouted(handles.into())
                }
                // `operators::admit` refuses a draft head's parallel form.
                AttestedFeedForward::Parallel(_) => {
                    return Err("draft head parallel feed-forward form".into());
                }
            },
            features: &block.features,
            logits: readout::ProjectionEntries::prepared(&block.logits),
            shortlist: block.shortlist.as_ref().map(|shortlist| ShortlistEntries {
                coarse: &shortlist.coarse,
                selection: &shortlist.selection,
                rows: &shortlist.rows,
                placement: &shortlist.placement,
            }),
            shape: &head.shape,
            sample: &head.sample,
            confidence: &head.confidence,
            chain: head.chain.as_ref(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedHeadEntries {
    landing: Option<CheckedLandingEntries>,
    streams: Option<CheckedStreamsEntries>,
    gather: [(&'static str, Element); 1],
    copy: [(&'static str, Element); 1],
    input: [(&'static str, Element); 6],
    attention: CheckedAttentionEntries,
    feed_forward: CheckedHeadFeedForwardEntries,
    features: [(&'static str, Element); 2],
    logits: readout::CheckedProjectionEntries,
    shortlist: Option<[(&'static str, Element); 2]>,
    selection: [(&'static str, Element); 0],
    /// Whether its front graphs' selections feed the chain
    /// (`ModelLoadPlan::fronts`), beside its transactions' proposals.
    fronts: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CheckedHeadFeedForwardEntries {
    Dense(CheckedDenseEntries),
    Routed(CheckedRoutedEntries),
    GeneralRouted(CheckedGeneralRoutedEntries),
}

impl CheckedHeadEntries {
    fn new(binding: HeadBinding, backend: BackendName, load: &ModelLoadPlan) -> Result<Self, String> {
        Ok(Self {
            landing: CheckedLandingEntries::of(load, TARGET_TABLE).map_err(|error| error.to_string())?,
            streams: binding.streams.map(|streams| CheckedStreamsEntries::new(BlockStreams::LowRank(streams))),
            gather: [("A", binding.conditioning)],
            copy: [("A", binding.conditioning)],
            input: [
                ("C", binding.conditioning),
                ("EW", binding.embedding_table),
                ("A", binding.activation),
                ("EN", binding.embedding_norm),
                ("HN", binding.hidden_norm),
                ("CW", binding.combine),
            ],
            attention: CheckedAttentionEntries::new(binding.attention, backend, false)?,
            feed_forward: match binding.feed_forward {
                FeedForwardProgramSlot::Dense(binding) => {
                    CheckedHeadFeedForwardEntries::Dense(CheckedDenseEntries::new(binding))
                }
                FeedForwardProgramSlot::Routed(binding) => CheckedHeadFeedForwardEntries::Routed(
                    CheckedRoutedEntries::new(binding, backend)?,
                ),
                FeedForwardProgramSlot::GeneralRouted(binding) => {
                    CheckedHeadFeedForwardEntries::GeneralRouted(CheckedGeneralRoutedEntries::new(binding))
                }
                // `operators::admit` refuses a draft head's parallel form.
                FeedForwardProgramSlot::Parallel(_) => {
                    return Err("draft head parallel feed-forward form".into());
                }
            },
            features: [("NW", binding.output_norm), ("A", binding.activation)],
            logits: readout::CheckedProjectionEntries::new(
                binding.projection,
                binding.output_norm,
                binding.activation,
                true,
            ),
            shortlist: binding
                .shortlist
                .map(|coarse| [("OW", coarse), ("A", binding.activation)]),
            selection: [],
            fronts: load.fronts(),
        })
    }

    fn entries<G: ElementBoundDraft + 'static>(&self) -> Result<HeadGraphEntries<'_, G>, String> {
        Ok(HeadGraphEntries {
            landing: self.landing.as_ref().map(CheckedLandingEntries::entries::<G>),
            streams: self.streams.as_ref().map(CheckedStreamsEntries::entries::<G>),
            gather: G::bind(&self.gather),
            input: G::bind_ranged(&self.input),
            copy: G::bind(&self.copy),
            attention: self.attention.entries()?,
            priming_project: self.attention.entries::<G>()?.project,
            feed_forward: match &self.feed_forward {
                CheckedHeadFeedForwardEntries::Dense(entries) => {
                    HeadFeedForwardEntries::Dense(entries.entries())
                }
                CheckedHeadFeedForwardEntries::Routed(entries) => {
                    HeadFeedForwardEntries::Routed(entries.entries())
                }
                CheckedHeadFeedForwardEntries::GeneralRouted(entries) => {
                    HeadFeedForwardEntries::GeneralRouted(entries.entries())
                }
            },
            features: G::bind(&self.features),
            logits: self.logits.entries(),
            shortlist: self.shortlist.as_ref().map(|coarse| ShortlistEntries {
                coarse: G::bind_ranged(coarse),
                selection: G::bind(&self.selection),
                rows: G::bind(&self.selection),
                placement: G::bind(&self.selection),
            }),
            shape: G::bind(&self.selection),
            sample: G::bind(&self.selection),
            confidence: G::bind(&self.selection),
            chain: self.fronts.then(|| G::bind(&self.selection)),
        })
    }
}

struct HeadGraphParts<P> {
    plan: P,
    /// The entry pass's tokens: the table rows the host writes or, of a
    /// front graph, the target step's selections the device binds.
    tokens: NativePort,
    conditioning: Option<NativePort>,
    previous: Option<NativePort>,
    /// Each slot's last entry row, read by the entry pass's features; absent
    /// from a causal-only graph, which reads no features.
    out_rows: Option<NativePort>,
    passes: Vec<PassPorts>,
    constants: Vec<GraphConstant>,
    weights: Vec<(WeightPort, NativePort)>,
    /// The complete output head, when drafting.
    projection: Option<readout::ProjectionPorts>,
    /// Selections `[steps * slots, 2]` when selecting; otherwise the entry
    /// rows' block input, exported so the graph has a result.
    output: WorkflowTensor,
    selected: SelectedPorts,
}

/// What a head graph's selections feed.
enum SelectedPorts {
    /// Nothing: a causal-only graph selects nothing.
    Causal,
    /// A drafting graph's proposals, with each selection's probability
    /// under the logits it was selected from, `[steps * slots]` in the
    /// selections' order.
    Proposals { confidence: WorkflowTensor },
    /// A front graph's chain.
    Chain(FrontPorts),
}

/// A front graph's chain (`guess_chain`): the target step's token rows the
/// device binds, the host's chaining and continuing words, and its exported
/// outputs ([`crate::programs::FrontOutput`]).
struct FrontPorts {
    tokens: NativePort,
    chaining: NativePort,
    continuing: NativePort,
    guesses: WorkflowTensor,
    next_selected: WorkflowTensor,
    next_guesses: WorkflowTensor,
}

type PreparedHeadGraph = HeadGraphParts<NativeGraphPlan>;

impl<P> HeadGraphParts<P> {
    fn map_plan<Q>(
        self,
        map: impl FnOnce(P) -> Result<Q, GraphError>,
    ) -> Result<HeadGraphParts<Q>, GraphError> {
        Ok(HeadGraphParts {
            plan: map(self.plan)?,
            tokens: self.tokens,
            conditioning: self.conditioning,
            previous: self.previous,
            out_rows: self.out_rows,
            passes: self.passes,
            constants: self.constants,
            weights: self.weights,
            projection: self.projection,
            output: self.output,
            selected: self.selected,
        })
    }
}

pub struct PreparedHeadGraphs {
    conditioning_bytes: usize,
    /// The units its attention's indexer picks, if it selects: a launch
    /// scores where some row has more complete units.
    selection_top: Option<u64>,
    classes: BTreeMap<HeadGraphClass, PreparedHeadGraph>,
    family: NativeGraphFamily,
}

pub(crate) struct BoundHeadGraphs {
    prepared: Rc<PreparedHeadGraphs>,
    bound: BTreeMap<HeadGraphClass, BoundNativeGraphPlan>,
    constants: Vec<Tensor>,
    /// The token embedding as the passes gather it: the entry pass by the
    /// host's tokens, a chained pass by the device's selections.
    embedding: crate::device_table::ResidentTable,
}

/// How a head graph's entry pass runs: what its class fixes beside its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct HeadPasses {
    shaped: bool,
    device: bool,
    prefix: bool,
    front: bool,
}

impl HeadPasses {
    fn of(launch: HeadLaunchClass) -> Self {
        match launch {
            HeadLaunchClass::Transaction { shaped, device, prefix, .. } => {
                Self { shaped, device, prefix, front: false }
            }
            // A front graph's entry is conditioned on the device.
            HeadLaunchClass::Front { shaped, prefix, .. } => Self { shaped, device: true, prefix, front: true },
        }
    }
}

/// A drafting head graph's passes: how many it chains and the projection
/// they select through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct HeadDrafting {
    steps: std::num::NonZeroU64,
    /// The projection as its scope is stored.
    projection: readout::StoredHead,
    /// Whether its passes select through the certified levels.
    certified: bool,
    /// Whether its passes run narrow coarse-to-exact shortlist readout.
    shortlist: bool,
}

/// A head's feed-forward as its builder branches on it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum HeadFeedForward<R = Argument<f64>> {
    Dense(ExpansionForm),
    Routed(routed::RoutedTopology),
    GeneralRouted(GeneralRoutedTopology<R>),
}

impl<R> Numbers<R> for HeadFeedForward<R> {
    type Of<E> = HeadFeedForward<E>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> HeadFeedForward<E> {
        match self {
            Self::Dense(form) => HeadFeedForward::Dense(form),
            Self::Routed(topology) => HeadFeedForward::Routed(topology),
            Self::GeneralRouted(topology) => HeadFeedForward::GeneralRouted(topology.map(f)),
        }
    }
}

/// A head feed-forward's numbers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum HeadFeedForwardNumbers<D> {
    Dense(DenseNumbers<D>),
    Routed(routed::ExpertShape<D>),
    GeneralRouted(GeneralRoutedLayer<D>),
}

impl<D> Numbers<D> for HeadFeedForwardNumbers<D> {
    type Of<E> = HeadFeedForwardNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> HeadFeedForwardNumbers<E> {
        match self {
            Self::Dense(numbers) => HeadFeedForwardNumbers::Dense(numbers.map(f)),
            Self::Routed(shape) => HeadFeedForwardNumbers::Routed(shape.map(f)),
            Self::GeneralRouted(layer) => HeadFeedForwardNumbers::GeneralRouted(layer.map(f)),
        }
    }
}

/// Everything structural a head graph's builder branches on: its passes,
/// chain, elements, its sublayers' topologies and its projection's stored
/// form. It holds no model number ([`HeadNumbers`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct HeadForm<R = Argument<f64>, T = Table> {
    passes: HeadPasses,
    /// Whether a chained pass is conditioned by the residual (F32) rather
    /// than the features.
    residual_chain: bool,
    activation: Element,
    /// Its norms' epsilon: a real ([`Numbers`] over `R`).
    epsilon: R,
    attention: AttentionTopology<R, T>,
    feed_forward: HeadFeedForward<R>,
    /// Its drafting passes; none for a causal-only graph, which only
    /// publishes its entry rows' K/V history.
    drafting: Option<HeadDrafting>,
}

impl<R, T> Numbers<R> for HeadForm<R, T> {
    type Of<E> = HeadForm<E, T>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> HeadForm<E, T> {
        HeadForm {
            passes: self.passes,
            residual_chain: self.residual_chain,
            activation: self.activation,
            epsilon: f(self.epsilon),
            attention: self.attention.map(&mut *f),
            feed_forward: self.feed_forward.map(f),
            drafting: self.drafting,
        }
    }
}

impl<R, T> HeadForm<R, T> {
    /// It with each of its tables `f` of its own.
    fn tables<U>(self, f: impl FnMut(T) -> U) -> HeadForm<R, U> {
        HeadForm {
            passes: self.passes,
            residual_chain: self.residual_chain,
            activation: self.activation,
            epsilon: self.epsilon,
            attention: self.attention.tables(f),
            feed_forward: self.feed_forward,
            drafting: self.drafting,
        }
    }
}

/// Every model number a head graph is sized by.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct HeadNumbers<D = Dim> {
    /// Its attention at the entry pass, whose `rows` are the class's entry
    /// rows: the one fact its classes vary.
    attention: AttentionBlock<D>,
    slots: D,
    hidden: D,
    vocabulary: D,
    /// Its low-rank streams; none for a single residual stream.
    streams: Option<LowRankGeometry<D>>,
    feed_forward: HeadFeedForwardNumbers<D>,
    weights: GraphWeights<D>,
}

impl<D> Numbers<D> for HeadNumbers<D> {
    type Of<E> = HeadNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> HeadNumbers<E> {
        HeadNumbers {
            attention: self.attention.map(&mut *f),
            slots: f(self.slots),
            hidden: f(self.hidden),
            vocabulary: f(self.vocabulary),
            streams: self.streams.map(|streams| LowRankGeometry {
                hidden: f(streams.hidden),
                streams: streams.streams,
                rank: f(streams.rank),
            }),
            feed_forward: self.feed_forward.map(&mut *f),
            weights: self.weights.map(&mut *f),
        }
    }
}

/// The weights a head graph reads: its block's and sublayers', and the
/// target's token table and output projection.
fn head_reads(role: WeightRole) -> bool {
    match role.scope {
        WeightScope::HeadBlock(0) | WeightScope::HeadSublayer(SublayerIndex { block: 0, .. }) => true,
        WeightScope::Target => matches!(
            role.kind,
            WeightKind::Embedding | WeightKind::Output | WeightKind::OutputPlane(_)
        ),
        _ => false,
    }
}

/// A head planned for a load, split once into what every class's graph is
/// built of ([`Self::class`]): the only reading of the model behind a head
/// graph.
struct HeadSplit<'h> {
    attention: &'h magnitude_family_contracts::Attention,
    attention_shape: crate::AttentionShape,
    /// The static axes of its attention's selected read, if it selects, and
    /// the units its indexer picks.
    selection: Option<(crate::SelectionShape, u64)>,
    /// The block's norms' epsilon, and its attention heads' norms'.
    epsilon: f64,
    head_epsilon: f64,
    activation: Element,
    residual_chain: bool,
    hidden: u64,
    vocabulary: u64,
    streams: Option<LowRankGeometry<u64>>,
    feed_forward: (HeadFeedForward<f64>, HeadFeedForwardNumbers<u64>),
    /// The target projection as the head's drafting passes read it.
    plans: readout::HeadPlans<'h>,
    shortlist: bool,
    weights: GraphWeights<u64>,
}

impl<'h> HeadSplit<'h> {
    fn of(
        load: &'h ModelLoadPlan,
        geometry: &Decoder,
        head: CausalHeadBlock<'h>,
        feed_forward: FeedForwardProgramSlot,
    ) -> Result<Self, GraphError> {
        let hidden = geometry.hidden;
        // A head's streams are low-rank (`operators::SINKHORN_HEAD`).
        let streams = StreamsGeometry::from_residual(hidden, head.residual)
            .map_err(|error| error.to_string())?
            .map(|streams| streams.low_rank().ok_or(crate::operators::SINKHORN_HEAD))
            .transpose()?;
        let paired = paired_block(head.block).map_err(|error| error.to_string())?;
        let Mixer::Attention(attention) = paired.mixer else {
            return Err("draft head block must attend".into());
        };
        let attention_shape =
            operators::attention::shape(hidden, attention, None).map_err(|error| error.to_string())?;
        let head_epsilon = operators::attention::head_norm_epsilon(attention, paired.epsilon())
            .map_err(|error| error.to_string())?;
        let selection = operators::attention::selection_shape(attention, geometry.context_limit)
            .zip(operators::attention::admitted_selection(attention).map(|read| read.indexer.top));
        let feed_forward_scope = WeightScope::HeadSublayer(SublayerIndex { block: 0, sublayer: 1 });
        let feed_forward = match feed_forward {
            FeedForwardProgramSlot::Dense(_) => {
                let Some(crate::operators::FeedForward::Dense(dense)) =
                    paired.feed_forward.map(|sublayer| sublayer.op)
                else {
                    return Err("draft head dense entries without a dense feed-forward".into());
                };
                let (expansion, numbers) = dense_split(load, feed_forward_scope, &dense.up)?;
                (HeadFeedForward::Dense(expansion), HeadFeedForwardNumbers::Dense(numbers))
            }
            FeedForwardProgramSlot::Routed(binding) => {
                let (topology, shape) = routed::split::<NativeGraphMetadata>(
                    load,
                    feed_forward_scope,
                    routed::RoutedSource::Binding(&binding),
                    &crate::operators::routed::streamed::LayerBank::Held,
                )?;
                (HeadFeedForward::Routed(topology), HeadFeedForwardNumbers::Routed(shape))
            }
            FeedForwardProgramSlot::GeneralRouted(_) => {
                let Some(crate::operators::FeedForward::GeneralRouted(operator)) =
                    paired.feed_forward.map(|sublayer| sublayer.op)
                else {
                    return Err("draft head general routed entries without a general routed feed-forward".into());
                };
                let shape =
                    crate::GeneralRoutedShape::of(hidden, operator).map_err(|error| error.to_string())?;
                // A head's routed layers hold their experts whole.
                let (topology, layer) = general_split::<NativeGraphMetadata>(load, feed_forward_scope, &shape, &LayerBank::Held)?;
                (HeadFeedForward::GeneralRouted(topology), HeadFeedForwardNumbers::GeneralRouted(layer))
            }
            // `operators::admit` refuses a draft head's parallel form.
            FeedForwardProgramSlot::Parallel(_) => {
                return Err("draft head parallel feed-forward form".into());
            }
        };
        Ok(Self {
            attention,
            attention_shape,
            selection,
            epsilon: paired.epsilon(),
            head_epsilon,
            activation: activation(geometry.activation_dtype),
            residual_chain: matches!(head.chain, magnitude_family_contracts::HeadChain::Residual),
            hidden,
            vocabulary: geometry.vocabulary,
            streams,
            feed_forward,
            plans: readout::HeadPlans::for_recipe(load, WeightScope::Target, WeightScope::HeadBlock(0), true)?,
            shortlist: load
                .weight_by_role(WeightRole {
                    scope: WeightScope::HeadBlock(0),
                    kind: WeightKind::DraftOutput,
                })
                .is_some(),
            weights: GraphWeights::of(load, head_reads),
        })
    }

    /// `class`'s graph as its builder takes it: its topology and its numbers.
    fn class(&self, class: HeadGraphClass) -> Result<(HeadForm<f64, Arc<[u8]>>, HeadNumbers<u64>), GraphError> {
        let consistent = match class.launch {
            HeadLaunchClass::Transaction { slots, steps, shaped, scored, device, prefix } => {
                slots != 0
                    && slots <= class.entry_rows
                    && (steps != 0 || !(shaped || scored))
                    && (!device || (slots == 1 && (!prefix || steps == 0)))
            }
            HeadLaunchClass::Front { prefix, .. } => {
                class.entry_rows <= VERIFY_ROWS && (!prefix || class.entry_rows == 1)
            }
        };
        if !consistent
            || class.entry_rows == 0
            || class.history.tokens.rows == 0
            || (class.scored() && class.history.indexer.is_none())
        {
            return Err(format!("head graph class {class:?} is inconsistent").into());
        }
        let drafting = std::num::NonZeroU64::new(class.steps()).map(|steps| HeadDrafting {
            steps,
            projection: self.plans.selected(),
            certified: self.plans.certifies(class.slots()),
            shortlist: self.shortlist,
        });
        // A causal-only transaction only publishes its entry rows' K/V
        // history: nothing reads their attention output, so its pass ends at
        // the append and binds none of the weights after it. Its classes
        // list no history row tiles.
        let selection = match (self.selection, class.history.indexer) {
            (Some((shape, _)), Some([window, blocks])) => Some(SelectionSite {
                shape,
                scored: class.scored(),
                window_rows: window.rows,
                window_slab_rows: window.slab_rows,
                block_rows: blocks.rows,
                block_slab_rows: blocks.slab_rows,
            }),
            (None, None) => None,
            _ => return Err(format!("head graph class {class:?} and its attention disagree on selecting").into()),
        };
        let (attention_topology, attention_numbers) = split(&AttentionSite {
            rows: class.entry_rows,
            segments: class.segments()?,
            history_rows: class.history.tokens.rows,
            slab_rows: class.history.tokens.slab_rows,
            history_tiles: 0,
            shape: self.attention_shape,
            operator: self.attention,
            epsilon: self.epsilon as f32,
            head_epsilon: self.head_epsilon as f32,
            post_norm: None,
            activation: self.activation,
            inject_only: drafting.is_none(),
            // A selected read's rows read their launch's earlier rows as
            // stored history (`domain::cached_causal_domain`).
            causal_cache: selection.is_some(),
            selection,
            picked: None,
            controls: crate::programs::graph::draft::ControlSource::Uploaded,
        })?;
        let (feed_forward, feed_forward_numbers) = self.feed_forward.clone();
        let form = HeadForm {
            passes: HeadPasses::of(class.launch),
            residual_chain: self.residual_chain,
            activation: self.activation,
            epsilon: self.epsilon,
            attention: attention_topology,
            feed_forward,
            drafting,
        };
        let numbers = HeadNumbers {
            attention: attention_numbers,
            slots: class.slots(),
            hidden: self.hidden,
            vocabulary: self.vocabulary,
            streams: self.streams.clone(),
            feed_forward: feed_forward_numbers,
            weights: self.weights.clone(),
        };
        Ok((form, numbers))
    }
}

fn head_graph_topology<'a, G: GraphDraft + 'a>(
    graph: G,
    entries: HeadGraphEntries<'a, G>,
    topology: &HeadForm,
    numbers: &HeadNumbers,
) -> Result<HeadGraphParts<G::Plan>, GraphError> {
    head_graph_draft(graph, entries, topology, numbers)?.map_plan(|graph| graph.seal())
}

/// A head graph's passes. A progressive projection certifies the selections
/// of the slot classes its reader qualified (`HeadPlans::certifies`; a
/// proposal is only ever verified, so a certified selection serves every
/// slot it fits) and projects every draft-vocabulary row beyond them.
fn head_graph_draft<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entries: HeadGraphEntries<'a, G>,
    topology: &HeadForm,
    numbers: &HeadNumbers,
) -> Result<HeadGraphParts<G>, GraphError> {
    let planned = &numbers.weights;
    let HeadPasses { device, prefix, front, .. } = topology.passes;
    let steps = topology.drafting.map_or(0, |drafting| drafting.steps.get());
    let (entry_rows, slots, hidden, vocabulary) =
        (&numbers.attention.rows, &numbers.slots, &numbers.hidden, &numbers.vocabulary);
    let streams = &numbers.streams;
    let sites = entries
        .streams
        .map(|entries| entries.low_rank().ok_or(crate::operators::SINKHORN_HEAD))
        .transpose()?;
    let stream_count = streams.as_ref().map_or(1, |streams| streams.streams);
    let conditioning_width = hidden * stream_count;
    let conditioning_element = if topology.residual_chain {
        Element::f32()
    } else {
        topology.activation
    };
    let form = InputForm {
        parts: stream_count,
        normalized: streams.is_none(),
    };
    let epsilon = topology.epsilon.f32();
    let mut weights = Vec::new();
    macro_rules! weight {
        ($scope:expr, $kind:expr) => {
            weight(&mut graph, planned, $scope, $kind, &mut weights)?
        };
    }
    // The one supported head block: its own weights, then its attention and
    // feed-forward sublayers ([`head_sublayer`]).
    let head = WeightScope::HeadBlock(0);
    // The entry pass embeds the host's tokens, a chained pass the tokens
    // the device selected.
    let mut table = GatheredTable::of(
        &mut graph,
        planned,
        WeightScope::Target,
        WeightKind::Embedding,
        &mut weights,
    )?;
    let embedding_norm = weight!(head, WeightKind::HeadEmbeddingNorm);
    let hidden_norm = weight!(head, WeightKind::HeadHiddenNorm);
    let combine = weight!(head, WeightKind::HeadCombine);
    let inject_only = topology.drafting.is_none();
    let mut constants = Vec::new();
    // Draft head sublayers add their outputs to the residual (admission).
    let attention_weights = AttentionWeights::planned(
        &mut graph,
        planned,
        head_sublayer(0),
        (&topology.attention, &numbers.attention),
        (form, !inject_only),
        None,
        (&mut weights, &mut constants),
    )?;
    let output_norm = if inject_only {
        None
    } else if streams.is_some() {
        Some(GraphConstant::empty(&mut graph, topology.activation, &mut constants)?)
    } else {
        Some(weight!(head, WeightKind::OutputNorm))
    };
    let projection = topology
        .drafting
        .map(|drafting| {
            drafting.projection.ports(&mut graph, drafting.certified, &mut readout::PlannedPorts(planned))
        })
        .transpose()?;
    let shortlist_projection = if topology.drafting.is_some_and(|drafting| drafting.shortlist) {
        let role = WeightRole {
            scope: head,
            kind: WeightKind::DraftOutput,
        };
        let port = crate::programs::native_target_graph::weight_port(
            &mut graph,
            planned,
            role,
            crate::programs::native_target_graph::WeightPart::Values,
            &mut weights,
        )?;
        let empty_scale = GraphConstant::f32(&mut graph, &[])?;
        let scale = empty_scale.port().clone();
        constants.push(empty_scale);
        Some(readout::ProjectionPorts::Packed {
            weight: port,
            scale,
            scale_extent: Dim::from(0),
        })
    } else {
        None
    };

    let entry_dims = [
        ("M", entry_rows.clone()),
        ("V", table.rows.clone()),
        ("D", hidden.clone()),
        ("S", Dim::from(stream_count)),
        ("R", Dim::from(0)),
        ("F", Dim::from(1)),
    ];
    let tokens = if front {
        graph.port(Element::i32(), &[entry_rows.clone(), Dim::from(2)])?
    } else {
        graph.input_for(G::first(entries.input), "tokens", &entry_dims)?
    };
    // The host writes the entry rows' conditioning, or a device-conditioned
    // class binds the leading rows of its chunk's target features.
    let conditioning = if prefix && entry_rows.equals(1u64) {
        None
    } else if device {
        Some(graph.port(
            conditioning_element,
            &[entry_rows.clone(), conditioning_width.clone()],
        )?)
    } else {
        Some(graph.input_for(G::first(entries.input), "conditioning", &entry_dims)?)
    };
    let prefix_conditioning = if prefix {
        Some(graph.port(conditioning_element, &[Dim::from(1), conditioning_width.clone()])?)
    } else {
        None
    };
    let assembled = if let Some(previous) = &prefix_conditioning {
        let mut rows = graph.local_for(
            entries.copy,
            "out",
            &[("M", entry_rows.clone()), ("D", conditioning_width.clone())],
        )?;
        let mut first = rows.tensor_mut().slice_leading(0, 1);
        graph.enqueue(
            entries.copy,
            &[("M", Dim::from(1)), ("D", conditioning_width.clone())],
            feature_copy::WorkflowArgs {
                input: previous.tensor().into(),
                out: (&mut first).into(),
            },
        )?;
        if entry_rows.greater_than(1u64) {
            let source = conditioning
                .as_ref()
                .unwrap()
                .tensor()
                .slice_leading(0u64, entry_rows - 1u64);
            let mut tail = rows.tensor_mut().slice_leading(1u64, entry_rows);
            graph.enqueue(
                entries.copy,
                &[("M", entry_rows - 1u64), ("D", conditioning_width.clone())],
                feature_copy::WorkflowArgs {
                    input: (&source).into(),
                    out: (&mut tail).into(),
                },
            )?;
        }
        Some(rows)
    } else {
        None
    };
    let out_rows = if inject_only {
        None
    } else {
        Some(graph.input_for(
            entries.features,
            "out_rows",
            &[
                ("M", entry_rows.clone()),
                ("O", slots.clone()),
                ("D", hidden.clone()),
                ("PT", Dim::from(1)),
                ("NM", hidden.clone()),
            ],
        )?)
    };
    let selections = (steps > 0)
        .then(|| {
            graph.local_for(
                entries.sample,
                "result",
                &[("M", slots * steps), ("V", vocabulary.clone())],
            )
        })
        .transpose()?;
    // A drafting graph's selections' confidences, beside them step-major;
    // a pass's selection `s` was drawn from its logits row `s`. A front
    // graph's selections feed its chain alone.
    let selecting = match (steps, front) {
        (0, _) => Selecting::<G>::Causal,
        (_, false) => Selecting::Proposals {
            entry: entries.confidence,
            confidence: graph
                .local_for(
                    entries.confidence,
                    "confidence",
                    &[
                        ("M", slots * steps),
                        ("N", slots.clone()),
                        ("V", vocabulary.clone()),
                    ],
                )?
                .tensor()
                .clone(),
            rows: GraphConstant::row_identity(&mut graph, slots.clone(), None)?,
        },
        (_, true) => Selecting::Chain(entries.chain.ok_or_else(|| "a front head graph on a load that does not front".to_owned())?),
    };
    let context = HeadPassContext {
        entries: &entries,
        topology,
        numbers,
        sites,
        stream_count,
        conditioning_width: &conditioning_width,
        form,
        epsilon,
        attention_weights: &attention_weights,
        embedding_norm: &embedding_norm,
        hidden_norm: &hidden_norm,
        combine: &combine,
        tokens: &tokens,
        entry_conditioning: assembled.as_ref().or(conditioning.as_ref()),
        output_norm: output_norm.as_ref(),
        out_rows: out_rows.as_ref(),
        projection: projection.as_ref(),
        shortlist: shortlist_projection.as_ref(),
        selections: selections.as_ref(),
        selecting: &selecting,
    };
    let mut passes = Vec::new();
    let mut entry_features = None;
    let mut previous: Option<WorkflowTensor> = None;
    for pass in 0..steps.max(1) {
        match head_pass(&mut graph, &context, pass, previous.as_ref(), &mut table, &mut weights, &mut constants)? {
            HeadPass::Causal(ports, attended) => {
                return Ok(HeadGraphParts {
                    plan: graph,
                    tokens,
                    conditioning,
                    previous: prefix_conditioning,
                    out_rows: None,
                    passes: vec![ports],
                    constants,
                    weights,
                    projection: None,
                    output: attended,
                    selected: SelectedPorts::Causal,
                });
            }
            HeadPass::Drafted { ports, features, chained } => {
                passes.push(ports);
                if pass == 0 {
                    entry_features = Some(features);
                }
                previous = Some(chained);
            }
        }
    }
    let output = match &selections {
        Some(selections) => selections.tensor().clone(),
        None => entry_features.ok_or_else(|| "head graph has no entry pass".to_owned())?,
    };
    graph.export(&output)?;
    let selected = match selecting {
        Selecting::Causal => SelectedPorts::Causal,
        Selecting::Proposals { confidence, rows, .. } => {
            graph.export(&confidence)?;
            constants.push(rows);
            SelectedPorts::Proposals { confidence }
        }
        // A front graph's chain, from the target step's token rows and
        // selections and its own selections on the same rows.
        Selecting::Chain(entry) => SelectedPorts::Chain({
            let dimensions = [
                ("S", entry_rows.clone()),
                ("M", entry_rows.clone()),
                ("N", Dim::from(VERIFY_ROWS)),
            ];
            let tokens_rows =
                graph.port(Element::i32(), &[entry_rows.clone(), Dim::from(2)])?;
            let chaining = graph.input_for(entry, "chaining", &dimensions)?;
            let continuing = graph.input_for(entry, "continuing", &dimensions)?;
            let mut guesses = graph.local_for(entry, "guesses", &dimensions)?;
            let mut next_selected = graph.local_for(entry, "next_selected", &dimensions)?;
            let mut next_guesses = graph.local_for(entry, "next_guesses", &dimensions)?;
            graph.enqueue(
                entry,
                &dimensions,
                guess_chain::WorkflowArgs {
                    tokens: tokens_rows.tensor().into(),
                    chaining: chaining.tensor().into(),
                    selected: tokens.tensor().into(),
                    drafted: (&output).into(),
                    continuing: continuing.tensor().into(),
                    guesses: guesses.tensor_mut().into(),
                    next_selected: next_selected.tensor_mut().into(),
                    next_guesses: next_guesses.tensor_mut().into(),
                },
            )?;
            let [guesses, next_selected, next_guesses] =
                [guesses, next_selected, next_guesses].map(|port| port.tensor().clone());
            for output in [&guesses, &next_selected, &next_guesses] {
                graph.export(output)?;
            }
            FrontPorts {
                tokens: tokens_rows,
                chaining,
                continuing,
                guesses,
                next_selected,
                next_guesses,
            }
        }),
    };
    Ok(HeadGraphParts {
        plan: graph,
        tokens,
        conditioning,
        previous: prefix_conditioning,
        out_rows: Some(out_rows.ok_or_else(|| "head graph has no entry pass".to_owned())?),
        passes,
        constants,
        weights,
        projection,
        output,
        selected,
    })
}

/// The scope of sublayer `sublayer` of the one supported head block: its
/// attention (0) or its feed-forward (1).
fn head_sublayer(sublayer: u32) -> WeightScope {
    WeightScope::HeadSublayer(SublayerIndex { block: 0, sublayer })
}

/// What every pass of a head graph reads beside the graph, its token
/// table, its weights and its constants ([`head_pass`]).
struct HeadPassContext<'r, 'a, G: GraphDraft + 'a> {
    entries: &'r HeadGraphEntries<'a, G>,
    topology: &'r HeadForm,
    numbers: &'r HeadNumbers,
    /// Its low-rank streams' mix sites' entries; none for a single residual
    /// stream.
    sites: Option<Sites<SiteEntries<'a, G>>>,
    stream_count: u64,
    /// The width of a pass's conditioning rows: every stream's.
    conditioning_width: &'r Dim,
    form: InputForm,
    epsilon: Argument<f32>,
    attention_weights: &'r AttentionWeights,
    embedding_norm: &'r WorkflowTensor,
    hidden_norm: &'r WorkflowTensor,
    combine: &'r WorkflowTensor,
    /// The entry pass's tokens.
    tokens: &'r NativePort,
    /// The entry pass's conditioning rows: a prefix class's assembled rows,
    /// or the bound or host-written rows.
    entry_conditioning: Option<&'r NativePort>,
    /// Its output norm and each slot's last entry row, of a drafting graph.
    output_norm: Option<&'r WorkflowTensor>,
    out_rows: Option<&'r NativePort>,
    /// Its selecting passes' projection and selections, and what the
    /// selections feed ([`Selecting`]).
    projection: Option<&'r readout::ProjectionPorts>,
    shortlist: Option<&'r readout::ProjectionPorts>,
    selections: Option<&'r NativePort>,
    selecting: &'r Selecting<'a, G>,
}

/// One pass of a head graph ([`head_pass`]).
enum HeadPass {
    /// A causal-only graph's one pass: its ports and its attended rows, the
    /// graph's exported result.
    Causal(PassPorts, WorkflowTensor),
    /// A drafting pass: its ports, its features and the rows the next pass
    /// is conditioned by.
    Drafted { ports: PassPorts, features: WorkflowTensor, chained: WorkflowTensor },
}

/// Pass `pass` of a head graph, the next after `previous`'s rows (none for
/// the entry pass): its input, attention and, of a drafting graph,
/// feed-forward, features and selection.
fn head_pass<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    context: &HeadPassContext<'_, 'a, G>,
    pass: u64,
    previous: Option<&WorkflowTensor>,
    table: &mut GatheredTable,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<HeadPass, GraphError> {
    let (entries, topology, numbers) = (context.entries, context.topology, context.numbers);
    let planned = &numbers.weights;
    let (slots, hidden, streams) = (&numbers.slots, &numbers.hidden, &numbers.streams);
    let (form, epsilon, stream_count) = (context.form, context.epsilon, context.stream_count);
    let rows = if pass == 0 { numbers.attention.rows.clone() } else { slots.clone() };
    let (input, landed) = pass_input(graph, context, (pass, &rows), previous, table, weights, constants)?;
    let attention_site = match (streams, &context.sites) {
        (Some(streams), Some(sites)) => Some(mix_site(
            graph,
            sites.mixer,
            planned,
            head_sublayer(0),
            streams.clone(),
            &input,
            rows.clone(),
            epsilon,
            true,
            weights,
        )?),
        (None, None) => None,
        _ => return Err("head streams entries disagree with its residual form".into()),
    };
    let join = Join {
        values: attention_site.as_ref().and_then(|site| site.inject.clone()),
        streams: Dim::from(stream_count),
        width: Dim::from(streams.as_ref().map_or(0, |streams| streams.streams)),
        class: (pass == 0).then_some("head_entry_rows"),
    };
    let (attended, state, controls) = attention_graph::attention(
        graph,
        AttentionGraphEntries {
            project: if topology.drafting.is_none() {
                entries.priming_project
            } else {
                entries.attention.project
            },
            ..entries.attention
        },
        &topology.attention,
        &numbers.attention.at_rows(rows.clone()),
        context.attention_weights,
        constants,
        SublayerInput {
            rows: attention_site
                .as_ref()
                .map_or(&input, |site| &site.partials),
            form,
        },
        &input,
        &join,
    )?;
    let (Some(output_norm), Some(out_rows)) = (context.output_norm, context.out_rows) else {
        graph.export(&attended)?;
        return Ok(HeadPass::Causal(
            PassPorts {
                controls,
                planes: state.planes,
                indexer: state.indexer,
                selection: None,
                landed,
            },
            attended,
        ));
    };
    let feed_forward_site = match (streams, &context.sites) {
        (Some(streams), Some(sites)) => Some(mix_site(
            graph,
            sites
                .feed_forward
                .ok_or("head streams entries lack the feed-forward site")?,
            planned,
            head_sublayer(1),
            streams.clone(),
            &attended,
            rows.clone(),
            epsilon,
            true,
            weights,
        )?),
        _ => None,
    };
    let advanced = head_feed_forward(
        graph,
        context,
        (pass, &rows),
        (&attended, feed_forward_site.as_ref()),
        weights,
        constants,
    )?;
    // The entry pass reads each slot's last entry row; a chained
    // pass has one row per slot.
    let chained_rows = (pass > 0)
        .then(|| GraphConstant::row_identity(graph, slots.clone(), None))
        .transpose()?;
    let mixed = match (streams, &context.sites) {
        (Some(streams), Some(sites)) => {
            mix_site(
                graph,
                sites
                    .exit
                    .ok_or("head streams entries lack the exit site")?,
                planned,
                WeightScope::HeadBlock(0),
                streams.clone(),
                &advanced,
                rows.clone(),
                epsilon,
                false,
                weights,
            )?
            .partials
        }
        _ => advanced.clone(),
    };
    let features = graph
        .enqueue(
            entries.features,
            &[
                ("M", rows.clone()),
                ("O", slots.clone()),
                ("D", hidden.clone()),
                ("PT", Dim::from(form.parts)),
                ("NM", form.norm_extent(hidden.clone())),
            ],
            readout_features_rows::WorkflowArgs {
                hidden: (&mixed.reshape(&[rows.clone(), Dim::from(form.parts), hidden.clone()])).into(),
                norm: output_norm.into(),
                out_rows: chained_rows
                    .as_ref()
                    .map_or(out_rows.tensor(), |rows| rows.port().tensor())
                    .into(),
                epsilon,
            },
        )?
        .value;
    let pass_rows = chained_rows
        .as_ref()
        .map_or(out_rows.tensor(), |rows| rows.port().tensor())
        .clone();
    let chained_features = if topology.residual_chain {
        graph
            .enqueue(
                entries.gather,
                &[("M", rows.clone()), ("O", slots.clone()), ("D", context.conditioning_width.clone())],
                feature_rows::WorkflowArgs {
                    fused: (&advanced).into(),
                    out_rows: (&pass_rows).into(),
                },
            )?
            .value
    } else {
        features.clone()
    };
    constants.extend(chained_rows);
    let selection = pass_selection(
        graph,
        context,
        (pass, &rows),
        &features,
        (&mixed, output_norm, &pass_rows),
        constants,
    )?;
    Ok(HeadPass::Drafted {
        ports: PassPorts {
            controls,
            planes: state.planes,
            indexer: state.indexer,
            selection,
            landed,
        },
        features,
        chained: chained_features,
    })
}

/// The `rows` input rows of pass `pass`: the entry pass embeds the host's
/// tokens (a front graph's, the target step's selections, as a chained pass
/// embeds its own) and combines them with its conditioning rows; a chained pass
/// embeds the tokens the previous pass selected, read from `table` (landed,
/// of a table with landing slots), and combines them with the `previous`
/// pass's rows. Returns them and the landed read.
fn pass_input<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    context: &HeadPassContext<'_, 'a, G>,
    (pass, rows): (u64, &Dim),
    previous: Option<&WorkflowTensor>,
    table: &mut GatheredTable,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<(WorkflowTensor, Option<LandedReadPorts>), GraphError> {
    let (entries, numbers) = (context.entries, context.numbers);
    let (slots, hidden) = (&numbers.slots, &numbers.hidden);
    let (stream_count, epsilon) = (context.stream_count, context.epsilon);
    // The entry pass's conditioning: the host's rows, the target's
    // features, or a prefix assembled from both.
    let entry_conditioning = || {
        context
            .entry_conditioning
            .map(|rows| rows.tensor())
            .ok_or_else(|| "a head graph's entry pass has no conditioning".to_owned())
    };
    Ok(match previous {
        None if context.topology.passes.front => {
            let (gathered, table_rows) = (table.table.clone(), table.rows.clone());
            let selected = context.tokens.tensor();
            let conditioned = entry_conditioning()?;
            table.read_selected(
                graph,
                &numbers.weights,
                weights,
                constants,
                entries.landing,
                selected.into(),
                rows,
                Some("head_entry_rows"),
                |graph, directory| {
                    Ok(graph
                        .enqueue(
                            G::pick(entries.input, rows),
                            &[
                                ("M", rows.clone()),
                                ("V", table_rows),
                                ("D", hidden.clone()),
                                ("S", Dim::from(stream_count)),
                                ("R", directory.shape[0].clone()),
                                ("F", directory.shape[1].clone()),
                            ],
                            draft_rows::WorkflowArgs {
                                tokens: selected.into(),
                                directory: (&directory.tensor).into(),
                                table: (&gathered).into(),
                                conditioning: conditioned.into(),
                                embedding_norm: context.embedding_norm.into(),
                                hidden_norm: context.hidden_norm.into(),
                                combine: context.combine.into(),
                                epsilon,
                            },
                        )?
                        .value)
                },
            )?
        }
        None => {
            let directory = table.by_rows(graph, constants)?;
            let input = graph
                .enqueue(
                    G::pick(entries.input, rows),
                    &[
                        ("M", rows.clone()),
                        ("V", table.rows.clone()),
                        ("D", hidden.clone()),
                        ("S", Dim::from(stream_count)),
                        ("R", directory.shape[0].clone()),
                        ("F", directory.shape[1].clone()),
                    ],
                    draft_rows::WorkflowArgs {
                        tokens: context.tokens.tensor().into(),
                        directory: (&directory.tensor).into(),
                        table: (&table.table).into(),
                        conditioning: entry_conditioning()?.into(),
                        embedding_norm: context.embedding_norm.into(),
                        hidden_norm: context.hidden_norm.into(),
                        combine: context.combine.into(),
                        epsilon,
                    },
                )?
                .value;
            (input, None)
        }
        Some(features) => {
            let selected = context
                .selections
                .ok_or_else(|| "chained pass without selections".to_owned())?
                .tensor()
                .slice_leading(slots * (pass - 1), slots * pass);
            let (gathered, table_rows) = (table.table.clone(), table.rows.clone());
            table.read_selected(
                graph,
                &numbers.weights,
                weights,
                constants,
                entries.landing,
                (&selected).into(),
                rows,
                None,
                |graph, directory| {
                    Ok(graph
                        .enqueue(
                            G::pick(entries.input, rows),
                            &[
                                ("M", rows.clone()),
                                ("V", table_rows),
                                ("D", hidden.clone()),
                                ("S", Dim::from(stream_count)),
                                ("R", directory.shape[0].clone()),
                                ("F", directory.shape[1].clone()),
                            ],
                            draft_rows::WorkflowArgs {
                                tokens: (&selected).into(),
                                directory: (&directory.tensor).into(),
                                table: (&gathered).into(),
                                conditioning: features.into(),
                                embedding_norm: context.embedding_norm.into(),
                                hidden_norm: context.hidden_norm.into(),
                                combine: context.combine.into(),
                                epsilon,
                            },
                        )?
                        .value)
                },
            )?
        }
    })
}

/// Pass `pass`'s feed-forward over its `rows` `attended` rows (through its
/// mix site `site`, of low-rank streams), which adds to its residual.
fn head_feed_forward<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    context: &HeadPassContext<'_, 'a, G>,
    (pass, rows): (u64, &Dim),
    (attended, site): (&WorkflowTensor, Option<&MixSite>),
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<WorkflowTensor, GraphError> {
    let (topology, numbers) = (context.topology, context.numbers);
    let (planned, feed_forward_scope) = (&numbers.weights, head_sublayer(1));
    let (form, epsilon) = (context.form, context.epsilon);
    Ok(match (&context.entries.feed_forward, &topology.feed_forward, &numbers.feed_forward) {
        (
            HeadFeedForwardEntries::Dense(entries),
            HeadFeedForward::Dense(expansion),
            HeadFeedForwardNumbers::Dense(dense_numbers),
        ) => {
            // A head's feed-forward adds to its residual: it has no
            // post-norm tail.
            let TailEntries::Residual(output) = entries.output else {
                return Err("draft head feed-forward has a post-norm tail".into());
            };
            crate::operators::dense_ffn::graph::dense(
                graph,
                DenseGraphEntries { expand: entries.expand, output: TailEntries::Residual(output) },
                *expansion,
                dense_numbers,
                planned,
                feed_forward_scope,
                weights,
                constants,
                attended,
                attended,
                rows,
                (pass == 0).then_some("head_entry_rows"),
                epsilon,
                None,
            )?
        }
        (
            HeadFeedForwardEntries::Routed(entries),
            HeadFeedForward::Routed(routed_topology),
            HeadFeedForwardNumbers::Routed(shape),
        ) => {
            let join = Join {
                values: site.and_then(|site| site.inject.clone()),
                streams: Dim::from(context.stream_count),
                width: Dim::from(numbers.streams.as_ref().map_or(0, |streams| streams.streams)),
                class: (pass == 0).then_some("head_entry_rows"),
            };
            // A head's routed layers hold their experts whole: a
            // streamed one is refused (`routed`: its bank entries
            // disagree with its plan).
            let (output, _) = routed::routed(
                graph,
                RoutedGraphEntries {
                    route: entries.route,
                    decode: entries.decode,
                    output: entries.output,
                    operand_norm: entries.operand_norm,
                    group: entries.group,
                    experts: entries.experts,
                    combine: entries.combine,
                },
                crate::operators::routed::streamed::LayerBank::Held,
                routed_topology,
                shape,
                planned,
                feed_forward_scope,
                weights,
                constants,
                SublayerInput {
                    rows: site.map_or(attended, |site| &site.partials),
                    form,
                },
                attended,
                &join,
                rows,
                epsilon,
                None,
                None,
            )?;
            output
        }
        (
            HeadFeedForwardEntries::GeneralRouted(entries),
            HeadFeedForward::GeneralRouted(general_topology),
            HeadFeedForwardNumbers::GeneralRouted(layer),
        ) => {
            // The body's general routed lowering over the head's single
            // residual (`operators::admit` refuses low-rank streams),
            // its experts held whole.
            if site.is_some() {
                return Err("a general routed draft head over low-rank streams".into());
            }
            let (output, _) = general_routed(
                graph,
                *entries,
                LayerBank::Held,
                general_topology,
                layer,
                planned,
                feed_forward_scope,
                weights,
                constants,
                attended,
                RoutedSum::Residual,
                rows,
                epsilon,
                None,
            )?;
            output
        }
        _ => return Err("head feed-forward entries and plan disagree".into()),
    })
}

/// Pass `pass`'s selection of a selecting graph, from the logits of its
/// `rows` `features` (a progressive projection's from its `raw` hidden rows,
/// output norm and out rows); none of a graph that does not select.
fn pass_selection<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    context: &HeadPassContext<'_, 'a, G>,
    (pass, rows): (u64, &Dim),
    features: &WorkflowTensor,
    (mixed, output_norm, pass_rows): (&WorkflowTensor, &WorkflowTensor, &WorkflowTensor),
    constants: &mut Vec<GraphConstant>,
) -> Result<Option<SelectionPorts>, GraphError> {
    let (Some(projection), Some(selections)) = (context.projection, context.selections) else {
        return Ok(None);
    };
    let (entries, numbers) = (context.entries, context.numbers);
    let (slots, hidden, vocabulary) = (&numbers.slots, &numbers.hidden, &numbers.vocabulary);
    let projection_entries = entries.logits.borrowed();
    let readout::Readout {
        mut logits,
        certified,
    } = if let (Some(shortlist_entries), Some(shortlist_ports)) =
        (&entries.shortlist, context.shortlist)
    {
        let coarse_logits = readout::project_readout(
            graph,
            readout::ProjectionEntries::PreparedPacked(shortlist_entries.coarse),
            shortlist_ports,
            readout::ReadoutInput::PreparedFeatures {
                features,
                canonical_rows: None,
            },
            readout::ProjectionGeometry {
                rows: rows.clone(),
                projected: slots.clone(),
                selected: Dim::from(0),
                vocabulary: vocabulary.clone(),
                hidden: hidden.clone(),
                columns: vocabulary.clone(),
            },
            Argument::from(0.0),
            None,
            constants,
        )?
        .logits;
        let k = Dim::from(crate::shortlist::SHORTLIST_K);
        let top_dims = [
            ("M", slots.clone()),
            ("V", vocabulary.clone()),
            ("K", k.clone()),
        ];
        let candidates_port = graph.local_for(shortlist_entries.selection, "candidates", &top_dims)?;
        let unary_port = graph.local_for(shortlist_entries.selection, "unary", &top_dims)?;
        let mut candidates = candidates_port.tensor().clone();
        let mut unary = unary_port.tensor().clone();
        graph.enqueue(
            shortlist_entries.selection,
            &top_dims,
            draft_top_k::WorkflowArgs {
                logits: (&coarse_logits).into(),
                candidates: (&mut candidates).into(),
                unary: (&mut unary).into(),
            },
        )?;
        let token_rows = graph
            .enqueue(
                shortlist_entries.rows,
                &[("M", slots.clone()), ("K", k.clone())],
                shortlist_token_rows::WorkflowArgs {
                    selected: (&candidates.reshape(&[slots.clone(), k.clone(), Dim::from(2)])).into(),
                },
            )?
            .value;
        let compact = readout::project_readout(
            graph,
            projection_entries,
            projection,
            readout::ReadoutInput::PreparedFeatures {
                features,
                canonical_rows: Some(&token_rows),
            },
            readout::ProjectionGeometry {
                rows: rows.clone(),
                projected: slots.clone(),
                selected: Dim::from(0),
                vocabulary: vocabulary.clone(),
                hidden: hidden.clone(),
                columns: k.clone(),
            },
            Argument::from(0.0),
            None,
            constants,
        )?
        .logits;
        let placement_dims = [
            ("M", slots.clone()),
            ("V", vocabulary.clone()),
            ("S", k),
        ];
        let logits_port = graph.local_for(shortlist_entries.placement, "logits", &placement_dims)?;
        let mut placed = logits_port.tensor().clone();
        graph.enqueue(
            shortlist_entries.placement,
            &placement_dims,
            place_logits_rows::WorkflowArgs {
                scores: (&compact).into(),
                token_rows: (&token_rows).into(),
                logits: (&mut placed).into(),
            },
        )?;
        readout::Readout {
            logits: placed,
            certified: None,
        }
    } else {
        let input = match projection {
            readout::ProjectionPorts::Packed { .. } => {
                readout::ReadoutInput::PreparedFeatures {
                    features,
                    canonical_rows: None,
                }
            }
            readout::ProjectionPorts::Progressive { .. } => readout::ReadoutInput::RawRms {
                hidden: mixed,
                norm: output_norm,
                form: context.form,
                out_rows: pass_rows,
                epsilon: context.epsilon,
            },
        };
        readout::project_readout(
            graph,
            projection_entries,
            projection,
            input,
            readout::ProjectionGeometry {
                rows: rows.clone(),
                projected: slots.clone(),
                selected: slots.clone(),
                vocabulary: vocabulary.clone(),
                hidden: hidden.clone(),
                columns: vocabulary.clone(),
            },
            Argument::from(0.0),
            Some(entries.sample),
            constants,
        )?
    };
    let mut result = selections
        .tensor()
        .slice_leading(slots * pass, slots * (pass + 1));
    let (inputs, view) = match certified {
        Some(certified) => (certified.selecting, Some(certified.view)),
        None => (
            readout::selection_inputs(
                graph,
                entries.sample,
                slots,
                vocabulary,
            )?,
            None,
        ),
    };
    let shaped = context.topology.passes.shaped;
    Ok(Some(match context.selecting {
        Selecting::Proposals { entry, confidence, rows } => {
            let mut confidence =
                confidence.slice_leading(slots * pass, slots * (pass + 1));
            readout::select(
                graph,
                readout::SelectionEntries {
                    shape: entries.shape,
                    sample: entries.sample,
                    confidence: *entry,
                },
                vocabulary,
                &mut logits,
                view.as_ref(),
                rows.port().tensor(),
                slots,
                shaped,
                inputs,
                &mut result,
                &mut confidence,
            )?
        }
        Selecting::Chain(_) | Selecting::Causal => readout::sample(
            graph,
            entries.shape,
            entries.sample,
            vocabulary,
            &mut logits,
            slots,
            shaped,
            inputs,
            (&mut result).into(),
        )?,
    }))
}

/// What a head graph's selections feed, as its builder holds them.
enum Selecting<'a, G: GraphDraft + 'a> {
    Causal,
    /// Each pass's selections' confidences and the identity rows they are
    /// read through.
    Proposals {
        entry: G::Binding<'a, token_confidence::Entry>,
        confidence: WorkflowTensor,
        rows: GraphConstant,
    },
    Chain(G::Binding<'a, guess_chain::Entry>),
}

#[cfg(test)]
pub(crate) fn checked_head_family_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    head: CausalHeadBlock<'_>,
    binding: HeadBinding,
    classes: impl IntoIterator<Item = HeadGraphClass>,
) -> Result<CheckedGraphResources, GraphError> {
    checked_head_family_with_kernels(backend, load, geometry, head, binding, classes)
        .map(|(resources, _, _)| resources)
}

pub(crate) fn checked_head_family_with_kernels(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    head: CausalHeadBlock<'_>,
    binding: HeadBinding,
    classes: impl IntoIterator<Item = HeadGraphClass>,
) -> Result<
    (
        CheckedGraphResources,
        std::collections::HashSet<seismic::KernelRequest>,
        crate::programs::graph::sizing::DraftFamilySizingCensus,
    ),
    GraphError,
> {
    let classes = classes.into_iter().collect::<Vec<_>>();
    HeadFamilyComposition::evaluate(backend, load, geometry, head, binding, &classes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct HeadGroupKey {
    front: bool,
    steps: u64,
    shaped: bool,
    scored: bool,
    device: bool,
    prefix: bool,
    prefix_tail: bool,
    history: HeadHistoryGeometry,
    entry_form: RowForm,
    certified: bool,
    slot_form: Option<RowForm>,
}

impl HeadGroupKey {
    fn of(class: HeadGraphClass, split: &HeadSplit<'_>) -> Self {
        let passes = HeadPasses::of(class.launch);
        let steps = class.steps();
        let slots = class.slots();
        Self {
            front: passes.front,
            steps,
            shaped: passes.shaped,
            scored: class.scored(),
            device: passes.device,
            prefix: passes.prefix,
            prefix_tail: passes.prefix && class.entry_rows > 1,
            history: class.history,
            entry_form: RowForm::of(class.entry_rows),
            certified: steps > 0 && split.plans.certifies(slots),
            slot_form: (steps > 1).then(|| RowForm::of(slots)),
        }
    }
}

/// Single-owner partitioner of head graph classes into topology-identical
/// certification groups, shared by [`certify_head_family`] and
/// [`HeadFamilyComposition`].
fn partition_head_groups(
    classes: &[HeadGraphClass],
    split: &HeadSplit<'_>,
) -> Result<Vec<Vec<HeadGraphClass>>, GraphError> {
    let mut groups: BTreeMap<HeadGroupKey, Vec<HeadGraphClass>> = BTreeMap::new();
    for &class in classes {
        let key = HeadGroupKey::of(class, split);
        let group = groups.entry(key).or_default();
        if group.contains(&class) {
            return Err("head graph class is duplicated".into());
        }
        group.push(class);
    }
    Ok(groups.into_values().collect())
}

fn certify_head_group<L: Certified>(
    backend: BackendName,
    entries: &CheckedHeadEntries,
    split: &HeadSplit<'_>,
    group: &[HeadGraphClass],
) -> Result<(L, Vec<GraphConstant>), GraphError> {
    let (layout, all_constants) = L::certify_group(backend, group, |draft, class| {
        let (form, numbers) = split.class(class)?;
        let form = crate::programs::graph::facts::constant_reals(form).tables(Table::from);
        let numbers = crate::programs::graph::facts::constant(numbers);
        let parts = head_graph_draft(draft, entries.entries()?, &form, &numbers)
            .map_err(|error| error.context(format!("head class {class:?}")))?;
        Ok((parts.plan, parts.constants))
    })?;
    let constants = all_constants.into_iter().flatten().collect();
    Ok((layout, constants))
}

struct HeadGroupOut {
    storage: NativeGraphStorageBytes,
    constants: Vec<GraphConstant>,
    plan_heap: seismic::HostHeap,
    topology_heap: seismic::HostHeap,
    plan_count: usize,
    kernels: Vec<seismic::KernelRequest>,
    draft_width: Option<crate::programs::graph::sizing::DraftWidthSizingCensus>,
}

/// Single-owner composition of a head graph family's resources and kernel set,
/// used by both assessment and load reservation.
pub(crate) struct HeadFamilyComposition;

impl HeadFamilyComposition {
    pub(crate) fn evaluate(
        backend: BackendName,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        head: CausalHeadBlock<'_>,
        binding: HeadBinding,
        classes: &[HeadGraphClass],
    ) -> Result<
        (
            CheckedGraphResources,
            std::collections::HashSet<seismic::KernelRequest>,
            crate::programs::graph::sizing::DraftFamilySizingCensus,
        ),
        GraphError,
    > {
        let entries = CheckedHeadEntries::new(binding, backend, load)?;
        let split = HeadSplit::of(load, geometry, head, binding.feed_forward)?;
        let groups = partition_head_groups(classes, &split)?;
        crate::assessment::stage_timer::record_stage(|t| {
            t.native_head_groups += groups.len() as u64;
        });

        let group_outputs: Vec<HeadGroupOut> =
            crate::assessment::pool::par_for_results(&groups, |group| {
                let (layout, constants) =
                    certify_head_group::<SizingOutline>(backend, &entries, &split, group)?;
                let draft_width = group.first().and_then(|class| {
                    let key = HeadGroupKey::of(*class, &split);
                    let proposals = u8::try_from(key.steps).ok()?;
                    let proposing = !key.front
                        && proposals > 0
                        && !key.shaped
                        && !key.scored
                        && !key.device
                        && !key.prefix
                        && !key.prefix_tail
                        && key.entry_form == RowForm::of(1);
                    proposing.then(|| crate::programs::graph::sizing::DraftWidthSizingCensus {
                        proposals,
                        rows_per_sequence: 1,
                        calls: layout.launches().to_vec(),
                        timelines: layout.timeline_count(),
                    })
                });
                Ok::<_, GraphError>(HeadGroupOut {
                    storage: layout.storage_bytes(),
                    constants,
                    plan_heap: layout.plan_heap(),
                    topology_heap: layout.topology_heap(),
                    plan_count: group.len(),
                    kernels: layout.kernels().to_vec(),
                    draft_width,
                })
            })?;

        let mut family = CheckedGraphFamilyResources::new();
        let mut kernels = std::collections::HashSet::new();
        let mut widths = std::collections::BTreeMap::new();
        for out in group_outputs {
            family.include(out.storage, out.constants);
            // A head launch runs on its own lane, never in a decode step's.
            family.include_graph(out.storage.recorded, false);
            if out.plan_count > 0 {
                family.include_plan(out.topology_heap)?;
            }
            for _ in 0..out.plan_count {
                family.include_plan(out.plan_heap)?;
            }
            kernels.extend(out.kernels);
            if let Some(width) = out.draft_width {
                widths.entry(width.proposals).or_insert(width);
            }
        }

        let census = crate::programs::graph::sizing::DraftFamilySizingCensus {
            widths: widths.into_values().collect(),
        };
        Ok((family.finish()?, kernels, census))
    }
}

/// Direct contract sizing records zero traces.
#[cfg(test)]
pub(crate) fn head_traces_recorded() -> usize {
    0
}

pub(crate) fn certify_head_family<L: Certified>(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    head: CausalHeadBlock<'_>,
    binding: HeadBinding,
    classes: &[HeadGraphClass],
) -> Result<(CheckedGraphResources, BTreeMap<HeadGraphClass, L>), GraphError> {
    let entries = CheckedHeadEntries::new(binding, backend, load)?;
    let split = HeadSplit::of(load, geometry, head, binding.feed_forward)?;
    let groups = partition_head_groups(classes, &split)?;
    crate::assessment::stage_timer::record_stage(|t| t.native_head_groups += groups.len() as u64);
    let mut family = CheckedGraphFamilyResources::new();
    let mut layouts = BTreeMap::new();
    for group in groups {
        let (layout, constants) = certify_head_group::<L>(backend, &entries, &split, &group)?;
        family.include(layout.storage_bytes(), constants);
        // A head launch runs on its own lane, never in a decode step's.
        family.include_graph(layout.storage_bytes().recorded, false);
        if !group.is_empty() {
            family.include_plan(layout.topology_heap())?;
        }
        for class in group {
            family.include_plan(layout.plan_heap())?;
            layouts.insert(class, layout.clone());
        }
    }
    Ok((family.finish()?, layouts))
}

#[cfg(test)]
pub(crate) fn verify_head_family_certificates(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    head: CausalHeadBlock<'_>,
    binding: HeadBinding,
    classes: &[HeadGraphClass],
) -> Result<(), GraphError> {
    let (_, layouts) = certify_head_family::<NativeGraphLayout>(backend, load, geometry, head, binding, classes)?;
    let checked = CheckedHeadEntries::new(binding, backend, load)?;
    let split = HeadSplit::of(load, geometry, head, binding.feed_forward)?;
    for &class in classes {
        let (form, numbers) = split.class(class)?;
        let exact = head_graph_draft(
            NativeGraphMetadata::new(backend),
            checked.entries()?,
            &constant_reals(form).tables(Table::from),
            &constant(numbers),
        )?;
        let charged = exact
            .plan
            .seal_with_layout(&layouts[&class])
            .map_err(|error| GraphError::from(error).context(format!("head class {class:?}")))?;
        if !charged.within(layouts[&class].storage_bytes()) {
            return Err(format!("head class {class:?} charged outside its certified layout").into());
        }
    }
    Ok(())
}

impl PreparedHeadGraphs {
    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        distinct_storage_bytes(
            self.classes
                .values()
                .flat_map(|graph| graph.constants.iter()),
        )
    }

    /// `landing`: the landing entries of the token table, when it has
    /// landing slots.
    pub(crate) fn prepare(
        target_device: &Device,
        kernels: &AttestedHead,
        landing: Option<&AttestedLanding>,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        head: CausalHeadBlock<'_>,
        classes: impl IntoIterator<Item = HeadGraphClass>,
    ) -> Result<Self, SubmitError> {
        let _sizing = super::graph::sizing::SizingScope::enter_with_device(Some(
            *target_device.properties(),
        ));
        let classes = classes.into_iter().collect::<Vec<_>>();
        let binding = kernels
            .blocks
            .first()
            .ok_or_else(|| invalid("attested head block is absent"))?
            .binding;
        let (resources, layouts) = certify_head_family(
            target_device.backend(),
            load,
            geometry,
            head,
            binding,
            &classes,
        )
        .map_err(|error| invalid(error.to_string()))?;
        let split = HeadSplit::of(load, geometry, head, binding.feed_forward)
            .map_err(|error| invalid(error.to_string()))?;
        let mut prepared = BTreeMap::new();
        for class in classes {
            if prepared.contains_key(&class) {
                return Err(invalid("head graph class is duplicated"));
            }
            let graph = Self::prepare_class(target_device, kernels, landing, &split, class, &layouts[&class])?;
            prepared.insert(class, graph);
        }
        if prepared.is_empty() {
            return Err(invalid("head graph family has no classes"));
        }
        let plans = prepared.values().map(|graph| (false, graph.plan.clone()));
        let family = super::graph::serial_family("head", plans, &resources.graphs).map_err(device)?;
        super::graph::assert_storage_eq("head", &family, resources.storage);
        Ok(Self {
            selection_top: split.selection.map(|(_, top)| top),
            conditioning_bytes: load.residual_conditioning().map_or(
                geometry.hidden as usize * geometry.activation_dtype.bytes(),
                |width| width as usize * 4,
            ),
            classes: prepared,
            family,
        })
    }

    fn prepare_class(
        target_device: &Device,
        kernels: &AttestedHead,
        landing: Option<&AttestedLanding>,
        split: &HeadSplit<'_>,
        class: HeadGraphClass,
        layout: &NativeGraphLayout,
    ) -> Result<PreparedHeadGraph, SubmitError> {
        let block = kernels
            .blocks
            .first()
            .ok_or_else(|| invalid("attested head block is absent"))?;
        let (form, numbers) = split.class(class).map_err(|error| invalid(error.to_string()))?;
        head_graph_topology(
            target_device.native_graph_with_layout(layout),
            HeadGraphEntries::prepared(block, kernels, landing).map_err(invalid)?,
            &constant_reals(form).tables(Table::from),
            &constant(numbers),
        )
        .map_err(|error| invalid(format!("head class {class:?}: {error}")))
    }

    pub fn family(&self) -> &NativeGraphFamily {
        &self.family
    }

    pub fn workspace_bytes_max(&self) -> u64 {
        self.family.workspace_bytes()
    }

    pub fn output_bytes_max(&self) -> u64 {
        self.family.output_bytes()
    }

    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    pub(crate) fn bind_weights(
        self: &Rc<Self>,
        charges: &seismic::Charges,
        resident: &ResidentHead,
    ) -> Result<BoundHeadGraphs, SubmitError> {
        let mut uploaded = ConstantTensors::new(charges);
        let absent_scale =
            Tensor::zeros(charges, Element::f32(), &[0])
                .map_err(device)?;
        let projection = match &resident.output {
            ResidentOutput::Packed(weight) => vec![(None, weight.tensor())],
            ResidentOutput::Progressive(progressive) => ProgressivePlane::ALL
                .into_iter()
                .map(|plane| (Some(plane), progressive.plane(plane).tensor()))
                .collect(),
        };
        let projected = |plane: Option<ProgressivePlane>| -> Result<&Tensor, SubmitError> {
            projection
                .iter()
                .find(|(placed, _)| *placed == plane)
                .map(|(_, tensor)| *tensor)
                .ok_or_else(|| invalid("the head projection and the resident output head disagree"))
        };
        let mut bound = BTreeMap::new();
        for (class, graph) in &self.classes {
            let constants = graph
                .constants
                .iter()
                .map(|constant| Ok((constant.port(), uploaded.tensor(constant).map_err(device)?)))
                .collect::<Result<Vec<_>, SubmitError>>()?;
            let fixed = graph
                .weights
                .iter()
                .map(|(weight, port)| {
                    let resident = resident_head_weight(resident, weight.role)?;
                    Ok((port, weight.part.of(resident).map_err(invalid)?))
                })
                .chain(constants.iter().map(|(port, tensor)| Ok((*port, tensor))))
                .chain(match &graph.projection {
                    None => Vec::new(),
                    Some(readout::ProjectionPorts::Packed { weight, scale, .. }) => {
                        let ResidentOutput::Packed(output) = &resident.output else {
                            return Err(invalid(
                                "the head projection and the resident output head disagree",
                            ));
                        };
                        vec![
                            projected(None).map(|tensor| (weight, tensor)),
                            Ok((scale, output.scale().unwrap_or(&absent_scale))),
                        ]
                    }
                    Some(readout::ProjectionPorts::Progressive { planes: ports, .. }) => ports
                        .iter()
                        .map(|(plane, port)| projected(Some(*plane)).map(|tensor| (port, tensor)))
                        .collect(),
                })
                .collect::<Result<Vec<_>, SubmitError>>()?;
            bound.insert(*class, graph.plan.bind_static(charges, &fixed).map_err(device)?);
        }
        Ok(BoundHeadGraphs {
            prepared: self.clone(),
            bound,
            constants: uploaded.into_tensors(),
            embedding: resident.embedding.table().clone(),
        })
    }
}

impl BoundHeadGraphs {
    /// The state every head graph was sealed for (the head store's, see
    /// [`head_graph_classes`]).
    fn history(&self) -> Result<HeadHistoryGeometry, SubmitError> {
        self.prepared
            .classes
            .keys()
            .next()
            .map(|class| class.history)
            .ok_or_else(|| invalid("head graph family has no classes"))
    }

    fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.constants.iter().try_fold(0u64, |bytes, tensor| {
            bytes
                .checked_add(tensor.storage_bytes())
                .ok_or("head graph constant charge overflows")
        })
    }

    fn class(
        &self,
        class: HeadGraphClass,
    ) -> Result<(&PreparedHeadGraph, &BoundNativeGraphPlan), SubmitError> {
        let graph = self
            .prepared
            .classes
            .get(&class)
            .ok_or_else(|| invalid(format!("head graph class {class:?} was not prepared")))?;
        let bound = self
            .bound
            .get(&class)
            .ok_or_else(|| invalid(format!("head graph class {class:?} was not bound")))?;
        Ok((graph, bound))
    }
}

fn resident_head_weight(
    resident: &ResidentHead,
    role: WeightRole,
) -> Result<&ResidentWeight, SubmitError> {
    match (role.scope, role.kind) {
        (WeightScope::Target, WeightKind::Embedding) => Ok(&resident.embedding),
        _ => resident.weights.get(role).map_err(invalid),
    }
}

/// One history read's controls of a pass, padded to `rows` rows of the
/// graph's span class spans: padding rows attend to nothing and append
/// nowhere.
struct ReadControls {
    visible: Vec<u8>,
    fresh: Vec<u8>,
    destinations: Vec<u8>,
    /// Each actual row's visible rows in total.
    seen: Vec<i32>,
}

impl ReadControls {
    fn new(
        history: &HistoryTables,
        actual: usize,
        rows: usize,
        segments: usize,
    ) -> Result<Self, SubmitError> {
        let mut visible = vec![0_i32; rows * segments * 2];
        for (row, ranges) in history.visible[..actual].iter().enumerate() {
            let used = ranges
                .iter()
                .rposition(|range| range[1] > range[0])
                .map_or(0, |last| last + 1);
            if used > segments {
                return Err(invalid(format!(
                    "head row attends {used} history spans; the head admits {segments}"
                )));
            }
            for (span, [start, end]) in ranges[..used].iter().enumerate() {
                let at = (row * segments + span) * 2;
                visible[at] = *start;
                visible[at + 1] = *end;
            }
        }
        let mut seen = complete(&history.visible[..actual]);
        seen.resize(rows, 0);
        let padded = |values: &[[i32; 2]], fill: [i32; 2]| {
            i32_bytes((0..rows).flat_map(|row| {
                values
                    .get(row)
                    .filter(|_| row < actual)
                    .copied()
                    .unwrap_or(fill)
            }))
        };
        Ok(Self {
            visible: i32_bytes(visible),
            fresh: padded(&history.fresh, [0, 0]),
            destinations: i32_bytes((0..rows).map(|row| {
                if row < actual {
                    history.destinations[row]
                } else {
                    -1
                }
            })),
            seen,
        })
    }
}

/// One pass's controls: its rows' coordinates and the controls of the history
/// reads its graph binds.
struct PassControls {
    coordinates: Vec<u8>,
    tokens: ReadControls,
    /// A selected read's keys: the pending tokens' Window read and the
    /// units' Block read.
    indexer: Option<[ReadControls; 2]>,
}

impl PassControls {
    fn new(
        pass: &TargetBatchUpload<'_>,
        rows: usize,
        segments: usize,
        geometry: &HeadHistoryGeometry,
    ) -> Result<Self, SubmitError> {
        let actual = pass.actual_rows;
        if actual > rows {
            return Err(invalid("head pass has more rows than its graph class"));
        }
        let read = |read: usize| {
            pass.histories
                .get(read)
                .ok_or_else(|| invalid("head pass lacks a history read of its store"))
                .and_then(|history| ReadControls::new(history, actual, rows, segments))
        };
        Ok(Self {
            coordinates: i32_bytes((0..rows).flat_map(|row| {
                pass.coordinates
                    .get(row)
                    .filter(|_| row < actual)
                    .copied()
                    .unwrap_or([0; 4])
            })),
            tokens: read(geometry.tokens.read)?,
            indexer: geometry
                .indexer
                .map(|[window, blocks]| Ok::<_, SubmitError>([read(window.read)?, read(blocks.read)?]))
                .transpose()?,
        })
    }

    /// Whether the pass's indexer scores: some row sees more complete units
    /// than it picks (`selection_scores`).
    fn scores(&self, top: u64) -> bool {
        let Some([_, blocks]) = &self.indexer else { return false };
        selection_scores(
            blocks.seen.iter().zip(blocks.destinations.chunks_exact(4)).map(|(&seen, destination)| {
                (seen.into(), i32::from_le_bytes(destination.try_into().expect("four bytes")) >= 0)
            }),
            top,
        )
    }
}

pub struct NativeHeadProgram {
    geometry: Decoder,
    graphs: BoundHeadGraphs,
    /// What a step binds its tensors through: charges that allocate
    /// nothing, as every storage a step binds is a load's, which formed any
    /// record table it needs (`seismic::NativeGraphBindings::set`).
    unallocating: seismic::Charges,
    waiter: CompletionWaiter,
}

impl NativeHeadProgram {
    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.graphs.constant_bytes()
    }

    /// `unallocating`: charges of its device that allocate nothing
    /// (`seismic::Charges::none`).
    pub(crate) fn new(geometry: Decoder, graphs: BoundHeadGraphs, unallocating: seismic::Charges) -> Result<Self, SubmitError> {
        let waiter = CompletionWaiter::spawn().map_err(device)?;
        Ok(Self {
            geometry,
            graphs,
            unallocating,
            waiter,
        })
    }

    /// Queue one head launch. Returns its completion and what its
    /// selections make: a drafting transaction's proposals, a front
    /// launch's chain.
    fn queue(
        &self,
        core: &HeadLaunchCore,
        workspace: &mut NativeGraphWorkspaceLease,
        output: &mut NativeGraphOutputLease,
    ) -> Result<(seismic::NativeGraphCompletion, HeadOutput), SubmitError> {
        let batch = core.batch();
        let entry = batch.upload();
        let chain = batch.chain().collect::<Vec<_>>();
        let steps = batch.steps();
        let entry_rows = entry.class.rows();
        let shaped = std::iter::once(&entry)
            .chain(&chain)
            .any(|pass| pass.shaping[..pass.select_rows.len()].iter().any(shapes));
        let geometry = self.graphs.history()?;
        let crate::HeadState::Owned {
            advances,
            conditioning,
            ..
        } = core.state()
        else {
            return Err(invalid("head batch has no state advance"));
        };
        let bound_planes = advances
            .first()
            .ok_or_else(|| invalid("head batch has no state advance"))?
            .bindings()
            .history;
        // A domain's planes of one layer, in plane-descriptor order, of the
        // rows the graphs were sealed for.
        let planes_of = |domain: DomainRows, layer: LayerRef| -> Result<Vec<Tensor>, SubmitError> {
            let planes = bound_planes
                .iter()
                .filter(|plane| plane.domain.0 == domain.read && plane.layer == layer)
                .collect::<Vec<_>>();
            if planes.is_empty()
                || planes.iter().any(|plane| {
                    plane.slab_rows != domain.slab_rows
                        || plane.buffer.extents().first().copied() != Some(domain.rows)
                })
            {
                return Err(invalid(format!("the head's {layer:?} planes differ from its graphs' domain {domain:?}")));
            }
            Ok(planes.into_iter().map(|plane| plane.buffer.clone()).collect())
        };
        let planes = planes_of(geometry.tokens, LayerRef::Head(0))?;
        let keys = geometry
            .indexer
            .map(|[window, blocks]| {
                let layer = LayerRef::HeadIndexKeys { block: 0, part: 0 };
                Ok::<_, SubmitError>([planes_of(window, layer)?, planes_of(blocks, layer)?])
            })
            .transpose()?;
        let segments = HeadGraphClass::segments_of(&geometry).map_err(invalid)?;
        // A chained pass's rows are one per slot, of the slot class.
        let controls = std::iter::once(Ok((&entry, entry_rows)))
            .chain(chain.iter().map(|pass| {
                row_class(pass.actual_rows)
                    .map(|rows| (pass, rows))
                    .ok_or_else(|| invalid("head slots have no class"))
            }))
            .map(|pass| pass.and_then(|(pass, rows)| PassControls::new(pass, rows, segments as usize, &geometry)))
            .collect::<Result<Vec<_>, _>>()?;
        let scored = match self.graphs.prepared.selection_top {
            Some(top) => controls.iter().any(|pass| pass.scores(top)),
            None => false,
        };
        let scored = scored && steps > 0;
        let launch = match conditioning {
            crate::HeadConditioning::Front(reads) => HeadLaunchClass::Front {
                shaped,
                scored,
                prefix: matches!(reads.features, crate::FrontFeatures::Boundary(_)),
            },
            conditioning => HeadLaunchClass::Transaction {
                slots: row_class(batch.actual_slots()).ok_or_else(|| invalid("head slots have no class"))? as u64,
                steps: steps as u64,
                shaped,
                scored,
                device: matches!(conditioning, crate::HeadConditioning::Features { .. }),
                prefix: matches!(conditioning, crate::HeadConditioning::Features { previous: Some(_), .. }),
            },
        };
        let class = HeadGraphClass { entry_rows: entry_rows as u64, history: geometry, launch };
        let slots = class.slots() as usize;
        let (graph, bound) = self.graphs.class(class)?;
        if graph.passes.len() != chain.len() + 1 {
            return Err(invalid("head graph passes differ from the batch's steps"));
        }
        let mut bindings = bound.bindings();
        for pass in &graph.passes {
            if pass.planes.len() != planes.len() {
                return Err(invalid(
                    "head history planes differ from the attention entry",
                ));
            }
            for (port, plane) in pass.planes.iter().zip(&planes) {
                bindings.set(&self.unallocating, port, plane).map_err(device)?;
            }
            match (&pass.indexer, &keys) {
                (Some(ports), Some([window, blocks])) => {
                    let ([window_keys, window_coordinates], [block_keys]) = (&window[..], &blocks[..]) else {
                        return Err(invalid("a head's indexer keys are not two pending-key planes and one unit-key plane"));
                    };
                    for (port, plane) in [
                        (&ports.planes.window_keys, window_keys),
                        (&ports.planes.window_coordinates, window_coordinates),
                        (&ports.planes.block_keys, block_keys),
                    ] {
                        bindings.set(&self.unallocating, port, plane).map_err(device)?;
                    }
                }
                (None, None) => {}
                _ => return Err(invalid("a head's indexer planes differ from its graph's")),
            }
        }
        let mut active = workspace.slot_mut().activate(&graph.plan).map_err(device)?;
        match conditioning {
            // Host rows in entry-row order (the launch checked each slot's
            // rows against the activation width), padded with zeros.
            crate::HeadConditioning::Rows(rows) => {
                let row_bytes = self.graphs.prepared.conditioning_bytes;
                let mut conditioning = Vec::with_capacity(entry_rows * row_bytes);
                for rows in rows {
                    conditioning.extend_from_slice(rows.bytes());
                }
                if conditioning.len() != entry.actual_rows * row_bytes {
                    return Err(invalid(
                        "head conditioning bytes differ from the entry rows",
                    ));
                }
                conditioning.resize(entry_rows * row_bytes, 0);
                active
                    .write_input(
                        graph
                            .conditioning
                            .as_ref()
                            .ok_or_else(|| invalid("host conditioning port is absent"))?,
                        &conditioning,
                    )
                    .map_err(device)?;
            }
            // The target features' leading class rows; rows past the entry
            // rows condition padding rows, which append nowhere.
            crate::HeadConditioning::Features {
                current: features,
                previous,
            } => {
                if let (Some(port), Some(previous)) = (&graph.previous, previous) {
                    bindings
                        .set(&self.unallocating, port, previous.allocation().tensor().map_err(device)?)
                        .map_err(device)?;
                }
                if let Some(port) = &graph.conditioning {
                    let features = features
                        .tensor()
                        .slice_leading(0, entry_rows as u64)
                        .map_err(|error| invalid(format!("head conditioning features: {error}")))?;
                    bindings.set(&self.unallocating, port, &features).map_err(device)?;
                }
            }
            // The target step's leading class rows: entry row `r` reads its
            // selection, feature and token row `r`; rows past the entry
            // rows are padding rows', which append nowhere and whose chain
            // rows chain nothing.
            crate::HeadConditioning::Front(reads) => {
                let leading = |tensor: &crate::GraphOutputTensor, what: &str| {
                    tensor
                        .tensor()
                        .slice_leading(0, entry_rows as u64)
                        .map_err(|error| invalid(format!("front head {what}: {error}")))
                };
                match (&reads.features, &graph.conditioning, &graph.previous) {
                    (crate::FrontFeatures::Rows(features), Some(port), None) => {
                        bindings.set(&self.unallocating, port, &leading(features, "features")?).map_err(device)?;
                    }
                    (crate::FrontFeatures::Boundary(boundary), None, Some(port)) => {
                        bindings
                            .set(&self.unallocating, port, boundary.allocation().tensor().map_err(device)?)
                            .map_err(device)?;
                    }
                    _ => return Err(invalid("a front launch's conditioning differs from its graph class")),
                }
                let SelectedPorts::Chain(chain) = &graph.selected else {
                    return Err(invalid("a front head graph has no chain"));
                };
                bindings.set(&self.unallocating, &graph.tokens, &leading(&reads.selected, "selections")?).map_err(device)?;
                bindings.set(&self.unallocating, &chain.tokens, &leading(&reads.tokens, "token rows")?).map_err(device)?;
            }
        }
        // A chained pass embeds the device's selection, a landed read of a
        // table with landing slots.
        let landed = graph
            .passes
            .iter()
            .filter_map(|pass| pass.landed.as_ref())
            .map(|ports| ports.queue(&self.graphs.embedding, &self.unallocating, &mut bindings))
            .collect::<Result<Vec<_>, _>>()
            .map_err(invalid)?;
        match (&graph.selected, batch.continuing()) {
            // The host's tokens, as the table's rows.
            (SelectedPorts::Causal | SelectedPorts::Proposals { .. }, None) => {
                let rows = self
                    .graphs
                    .embedding
                    .indices(entry.tokens)
                    .map_err(invalid)?;
                active
                    .write_input(
                        &graph.tokens,
                        &i32_bytes(rows.iter().flat_map(|row| [*row, 0])),
                    )
                    .map_err(device)?;
            }
            (SelectedPorts::Chain(chain), Some(continuing)) => {
                let [chaining, continuing] = front_words(batch, continuing, entry_rows)?;
                active.write_input(&chain.chaining, &chaining).map_err(device)?;
                active.write_input(&chain.continuing, &continuing).map_err(device)?;
            }
            _ => return Err(invalid("a head graph's chain differs from its batch's")),
        }
        if let Some(out_rows) = &graph.out_rows {
            active
                .write_input(
                    out_rows,
                    &i32_bytes(
                        (0..slots).map(|slot| entry.out_rows.get(slot).copied().unwrap_or(0)),
                    ),
                )
                .map_err(device)?;
        }
        for ((ports, pass), controls) in graph.passes.iter().zip(std::iter::once(&entry).chain(&chain)).zip(&controls) {
            active
                .write_input(&ports.controls.coordinates, &controls.coordinates)
                .map_err(device)?;
            if let Some(read) = &ports.controls.read {
                active
                    .write_input(&read.visible, &controls.tokens.visible)
                    .map_err(device)?;
                active
                    .write_input(&read.fresh, &controls.tokens.fresh)
                    .map_err(device)?;
            }
            active
                .write_input(&ports.controls.destinations, &controls.tokens.destinations)
                .map_err(device)?;
            match (&ports.indexer, &controls.indexer) {
                (Some(ports), Some([window, blocks])) => {
                    let tables = &ports.tables;
                    active.write_input(&tables.window_visible, &window.visible).map_err(device)?;
                    active.write_input(&tables.window_destinations, &window.destinations).map_err(device)?;
                    active.write_input(&tables.block_destinations, &blocks.destinations).map_err(device)?;
                    // A class that scores lists each row's complete units.
                    match (&tables.index, class.scored()) {
                        (Some(index), true) => {
                            active.write_input(&index.block_visible, &blocks.visible).map_err(device)?;
                            active.write_input(&index.complete, &i32_bytes(blocks.seen.iter().copied())).map_err(device)?;
                            active
                                .write_input(
                                    &index.units,
                                    &i32_bytes(blocks.seen.iter().flat_map(|&complete| [0, complete])),
                                )
                                .map_err(device)?;
                        }
                        (None, false) => {}
                        _ => return Err(invalid("a head graph's scoring differs from its launch's")),
                    }
                }
                (None, None) => {}
                _ => return Err(invalid("a head pass's indexer controls differ from its graph's")),
            }
            if let Some(selection) = &ports.selection {
                let words = self.geometry.vocabulary.div_ceil(32) as usize;
                readout::write_selection(pass, &mut active, selection, slots, words)?;
            }
        }
        let active_output = output
            .activate(&graph.plan)
            .map_err(SubmitError::Invariant)?;
        let (active_output, completion) = active_output
            .run(|outputs| active.attach(bindings, outputs).and_then(|ready| ready.submit()))
            .map_err(device)?;
        let landed_reads = landed.into_iter().map(SelectedRead::submitted).collect();
        let owner = active_output.publish();
        let exported = |tensor: &WorkflowTensor, what: &str| {
            owner
                .tensor(tensor)
                .ok_or_else(|| invalid(format!("head graph omitted its {what}")))
        };
        let selected = match &graph.selected {
            SelectedPorts::Causal => HeadSelected::Causal,
            SelectedPorts::Proposals { confidence } => HeadSelected::Proposals(HeadSelections {
                tokens: exported(&graph.output, "selections")?,
                confidence: Some(exported(confidence, "confidence")?),
            }),
            SelectedPorts::Chain(chain) => HeadSelected::Chain(FrontOutput {
                guesses: exported(&chain.guesses, "chain's guesses")?,
                next_selected: exported(&chain.next_selected, "chain's next selections")?,
                next_guesses: exported(&chain.next_guesses, "chain's next guesses")?,
            }),
        };
        Ok((completion, HeadOutput { selected, landed_reads }))
    }
}

/// A front launch's chain words (`guess_chain`): per entry row of its
/// class, its `chaining` row, which of a slot's first row is that row's
/// token row (the batch's row: the launch binds the step's token rows from
/// the first one's) and the slot's front rows, and none of every other row;
/// per row of the decode form, its `continuing` row, the first entry row of
/// the slot its `queued` row continues and its depth there, or none.
fn front_words(
    batch: &magnitude_batching::ValidatedHeadBatch,
    queued: &[Option<magnitude_batching::FrontContinuation>],
    entry_rows: usize,
) -> Result<[Vec<u8>; 2], SubmitError> {
    let word = |count: u64, fields: [(u64, usize); 2]| -> Result<Vec<i32>, SubmitError> {
        let mut row = vec![0; count as usize];
        for (field, value) in fields {
            row[field as usize] = i32::try_from(value).map_err(|_| invalid("a front chain word exceeds i32"))?;
        }
        Ok(row)
    };
    let none = |count: u64, field: u64| {
        let mut row = vec![0; count as usize];
        row[field as usize] = -1;
        row
    };
    let firsts = batch.slots().map(|slot| slot.span()).collect::<Vec<_>>();
    let mut chaining = vec![none(CHAINING_WORDS, CHAINING_ROW); entry_rows];
    for span in &firsts {
        chaining[span.start] = word(CHAINING_WORDS, [(CHAINING_ROW, span.start), (CHAINING_FRONT, span.len() - 1)])?;
    }
    if queued.len() > VERIFY_ROWS as usize {
        return Err(invalid("a front launch continues more rows than the decode form's"));
    }
    let continuing = (0..VERIFY_ROWS as usize)
        .map(|row| match queued.get(row).copied().flatten() {
            Some(continuation) => {
                let slot = firsts
                    .get(continuation.slot)
                    .ok_or_else(|| invalid("a queued row continues no slot of its front launch"))?;
                word(CONTINUING_WORDS, [(CONTINUING_SLOT, slot.start), (CONTINUING_DEPTH, continuation.depth)])
            }
            None => Ok(none(CONTINUING_WORDS, CONTINUING_SLOT)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok([chaining, continuing].map(|rows| i32_bytes(rows.into_iter().flatten())))
}

impl HeadProgram for NativeHeadProgram {
    type Submission =
        DeviceSubmission<HeadLaunchCore, NativeGraphWorkspaceLease, HeadOutput>;

    fn submit(
        &mut self,
        mut launch: ValidatedHeadLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedHeadLaunch)> {
        let queued = {
            let (core, workspace, output) = launch.execution_parts_mut();
            self.queue(core, workspace, output)
        };
        let (completion, output) = match queued {
            Ok(queued) => queued,
            Err(error) => return Err((error, launch)),
        };
        let (core, workspace, _) = launch.into_submission_parts();
        Ok(DeviceSubmission::new(
            self.waiter.completion(vec![completion]),
            core,
            workspace,
            output,
        ))
    }
}

#[cfg(test)]
mod composition_tests {
    use super::*;

    #[test]
    fn head_family_composition_matches_unrolled_bit_for_bit() {
        let hidden = crate::planning::fixtures::mix_site_hidden();
        let definition = crate::planning::fixtures::with_mtp_head(
            crate::planning::fixtures::fixture_definition_of(hidden),
        );
        let manifest = crate::planning::fixtures::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: true,
                vision: false,
            },
            seismic::Layout::Rows16,
            crate::PlannedMethod::Plain,
            crate::planning::fixtures::LIMITS,
        )
        .unwrap();
        let method = crate::PlannedMethod::Mtp {
            greedy_proposals: 4,
            sampled_proposals: 4,
            widths: crate::DraftWidths::Adaptive,
        };
        let limits = ResourceLimits {
            max_launch_rows: 256,
            max_launch_slots: 16,
            max_selected_rows: 16,
            max_drafting_slots: 16,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead: false,
            chain_capacity: std::num::NonZeroUsize::MIN,
            conversations: std::num::NonZeroUsize::MIN,
        };
        let state = crate::ResourcePlanner::state_plan(
            &definition,
            &load,
            method,
            magnitude_state::KvCodec::Dense,
            limits,
            crate::ResourceCapacity {
                domain_bytes: 512 * 1024 * 1024,
                tensor_operations: crate::TensorOperations::Absent,
                allocations: seismic::NativeAllocations::EXACT,
            },
        )
        .unwrap();
        let plan = load
            .program_plan(&definition, magnitude_state::KvCodec::Dense)
            .unwrap();
        let head = definition.head.clone().unwrap();
        let head_block = CausalHeadBlock::of(&head, &head.blocks[0]).unwrap();
        let geometry = state.head_state().unwrap().head_geometry().unwrap();
        let classes =
            head_graph_classes(limits, geometry, method.draft_rows(), load.fronts()).unwrap();
        assert!(classes
            .iter()
            .any(|c| c.steps() == 4 && HeadPasses::of(c.launch).shaped));
        assert!(classes
            .iter()
            .any(|c| c.steps() == 2 && !HeadPasses::of(c.launch).shaped));
        let binding = plan.head().unwrap().blocks()[0];

        let (composed_resources, composed_kernels, _census) = HeadFamilyComposition::evaluate(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            head_block,
            binding,
            &classes,
        )
        .unwrap();

        let (unrolled_resources, unrolled_outlines) = certify_head_family::<SizingOutline>(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            head_block,
            binding,
            &classes,
        )
        .unwrap();
        let mut unrolled_kernels = std::collections::HashSet::new();
        for outline in unrolled_outlines.values() {
            unrolled_kernels.extend(outline.kernels().iter().cloned());
        }

        assert_eq!(composed_resources.storage, unrolled_resources.storage);
        assert_eq!(composed_resources.host_heap, unrolled_resources.host_heap);
        assert_eq!(
            composed_resources.binding_constant_bytes,
            unrolled_resources.binding_constant_bytes
        );
        assert_eq!(composed_resources.graphs, unrolled_resources.graphs);
        assert_eq!(composed_kernels, unrolled_kernels);

        // Monotonicity witness: no smaller `steps` (including `steps == 1`) or
        // `shaped = false` drafting class ever needs more storage than its
        // `(max_steps, shaped = true)` representative, and no device-conditioned
        // causal class (`prefix = false`) ever needs more storage than its
        // host-conditioned representative.
        let max_steps = method.draft_rows() as u64;
        for &class in &classes {
            let s = unrolled_outlines[&class].storage_bytes();
            match class.launch {
                HeadLaunchClass::Transaction {
                    slots,
                    steps,
                    scored,
                    device: false,
                    prefix: false,
                    ..
                } if steps >= 1 => {
                    let rep_steps = if steps == 1 { 1 } else { max_steps };
                    let rep_class = HeadGraphClass {
                        entry_rows: class.entry_rows,
                        history: class.history,
                        launch: HeadLaunchClass::Transaction {
                            slots,
                            steps: rep_steps,
                            shaped: true,
                            scored,
                            device: false,
                            prefix: false,
                        },
                    };
                    let rep = unrolled_outlines[&rep_class].storage_bytes();
                    assert!(
                        s.workspace <= rep.workspace
                            && s.sync <= rep.sync
                            && s.output <= rep.output
                            && s.upload <= rep.upload,
                        "drafting class {class:?} storage {s:?} exceeds representative {rep_class:?} storage {rep:?}"
                    );
                }
                HeadLaunchClass::Transaction {
                    steps: 0,
                    device: true,
                    prefix: false,
                    ..
                } => {
                    let rep_class = HeadGraphClass {
                        entry_rows: class.entry_rows,
                        history: class.history,
                        launch: HeadLaunchClass::Transaction {
                            slots: 1,
                            steps: 0,
                            shaped: false,
                            scored: false,
                            device: false,
                            prefix: false,
                        },
                    };
                    let rep = unrolled_outlines[&rep_class].storage_bytes();
                    assert!(
                        s.workspace <= rep.workspace
                            && s.sync <= rep.sync
                            && s.output <= rep.output
                            && s.upload <= rep.upload,
                        "causal device class {class:?} storage {s:?} exceeds host representative {rep_class:?} storage {rep:?}"
                    );
                }
                _ => {}
            }
        }

        // Independent equality witness: real sealed NativeGraphLayouts.
        let (laid_resources, _) = certify_head_family::<NativeGraphLayout>(
            BackendName::Cpu,
            &load,
            &definition.decoder,
            head_block,
            binding,
            &classes,
        )
        .unwrap();
        assert_eq!(composed_resources, laid_resources);
        // Verify single-slot device-conditioned drafting classes (`device: true,
        // prefix: false, steps: 1..=proposals`) are sealed and certified.
        for steps in 1..=max_steps {
            let device_draft = HeadGraphClass {
                entry_rows: 4,
                history: geometry,
                launch: HeadLaunchClass::Transaction {
                    slots: 1,
                    steps,
                    shaped: false,
                    scored: false,
                    device: true,
                    prefix: false,
                },
            };
            assert!(
                classes.contains(&device_draft),
                "expected device-conditioned drafting class {device_draft:?} to be sealed"
            );
        }
    }
}
