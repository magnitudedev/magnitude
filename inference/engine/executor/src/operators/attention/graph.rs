//! Attention block graph: the normed query | gate | key | value projection
//! (`attention_project`), the fused attention entry of the history codec
//! (head norms, rotary, K/V append at the rows' destinations, attention over
//! the visible history spans and fresh rows, output gate) and the output
//! projection plus residual (or its post-norm tail). Decode row classes use the codec's decode entry
//! (`attention_decode`, `attention_decode_k8v4`), or a dense history's prefill
//! entry where the decode entry holds no form of the shape ([`dense_decodes`]);
//! larger classes its prefill entry. Both share one contract, so their ports
//! have the same geometry.
//!
//! Every optional part of the operator is an axis of extent 0 or 1
//! ([`AttentionShape`]): an absent projection segment binds a zero-row view
//! of the query weight, an absent head norm a zero-row view of the block's
//! unit norm row. The rotary table (the coordinate axis, frequency and
//! amplitude of every rotated pair) and the unit row are graph constants,
//! bound once with the weights.

use crate::operators::output::{CheckedPostNormEntries, PostNormShape, TailEntries, post_norm};
use crate::operators::{InputForm, Join, SublayerInput, scalar};
use crate::programs::graph::{
    GraphError,
    draft::GraphDraft,
};
use super::pooled::{
    Picked, PickedEntries, PickedInputs, PickedPorts, PickedWeights, PoolEntry, PooledEntries,
    PooledWeights, Pooling,
};
use super::selection::{
    self, CandidateEntries, IndexerClass, IndexerElements, IndexerEntries, IndexerInputs, Projection,
    ProjectionInput,
};
use super::split::{AttentionBlock, AttentionPair, AttentionTopology, PickedPart, Plane, SelectionPart};
use crate::planning::{GateSpec, KeyValueSpec, RoundingArguments, ValueSpec, axis};
use crate::programs::native_target_graph::{
    PackedWeight, PlannedWeights, WeightOperands, WeightPort, packed_weight, weight,
};
use crate::{AttentionBinding, AttentionShape, SublayerTail};
use crate::{
    native::{
        AttentionHistoryKernels, AttentionKernels, DenseDecodeKernels, PoolKernel,
        SelectedPrefillKernel,
    },
    programs::native_constants::GraphConstant,
};
use magnitude_family_contracts::{Rotary, WeightKind, WeightScope};
use magnitude_kernels::{
    attention_append_dense, attention_append_k8v4, attention_decode, attention_decode_k8v4,
    attention_output, attention_prefill, attention_prefill_k8v4, attention_project, candidate_rows,
    pool_rows, project_rows, unit_reads, widen_rows,
};
use magnitude_state::{CodecSpec, HistoryEncoding, HistoryValue};
use seismic::{
    Argument, Dim, Element, NativeGraph, NativePort, WorkflowTensor, WorkflowTensorView,
};

/// An attention block's entries; each projection's by row range.
pub(crate) struct AttentionGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Ranged<'a, attention_project::Entry>,
    pub history: AttentionHistoryEntries<'a, G>,
    pub output: TailEntries<'a, G, attention_output::Entry>,
    /// A grouped output's first product (`project_rows` over its groups).
    pub grouped: Option<G::Ranged<'a, project_rows::Entry>>,
    /// An absorbed query's per-head products (`project_rows` over its heads).
    pub absorbed: Option<G::Ranged<'a, project_rows::Entry>>,
    /// A selected read's indexer entries.
    pub selection: Option<SelectionGraphEntries<'a, G>>,
    /// A low-rank query's up projection.
    pub low_rank_query: Option<LowRankQueryEntries<'a, G>>,
    /// An indexing index-list source's rows (an own source's) and indexer.
    pub picked: Option<PickedEntries<'a, G>>,
}

impl<'a, G: GraphDraft + 'a> Copy for AttentionGraphEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for AttentionGraphEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A low-rank query's entries after the attention projection: its `qr`
/// rows widened to F32, then normalized and projected to the query heads.
pub(crate) struct LowRankQueryEntries<'a, G: GraphDraft + 'a> {
    pub widen: G::Binding<'a, widen_rows::Entry>,
    pub project: G::Ranged<'a, attention_project::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for LowRankQueryEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for LowRankQueryEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A selected read's entries ([`select`]): its keys' projection (`None` when
/// fused into the main 6-segment `attention_project`) and pooling, its
/// indexer, its reads and the codec's WorkList prefill.
pub(crate) struct SelectionGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: Option<G::Ranged<'a, attention_project::Entry>>,
    pub pool: G::Binding<'a, pool_rows::Entry>,
    pub indexer: IndexerEntries<'a, G>,
    pub reads: G::Binding<'a, unit_reads::Entry>,
    pub rows: G::Binding<'a, candidate_rows::Entry>,
    pub prefill: SelectedPrefillEntries<'a, G>,
}

impl<'a, G: GraphDraft + 'a> Copy for SelectionGraphEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for SelectionGraphEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// The WorkList prefill of a selected read's history codec.
pub(crate) enum SelectedPrefillEntries<'a, G: GraphDraft + 'a> {
    Dense(G::Binding<'a, attention_prefill::Entry>),
    AffineK8V4(G::Binding<'a, attention_prefill_k8v4::Entry>),
}

impl<'a, G: GraphDraft + 'a> Copy for SelectedPrefillEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for SelectedPrefillEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

pub(crate) enum AttentionHistoryEntries<'a, G: GraphDraft + 'a> {
    Dense {
        codec: CodecSpec,
        append: G::Binding<'a, attention_append_dense::Entry>,
        decode: DenseDecodeEntries<'a, G>,
        prefill: G::Binding<'a, attention_prefill::Entry>,
    },
    AffineK8V4 {
        codec: CodecSpec,
        append: G::Binding<'a, attention_append_k8v4::Entry>,
        decode: G::Binding<'a, attention_decode_k8v4::Entry>,
        prefill: G::Binding<'a, attention_prefill_k8v4::Entry>,
        /// The prefill of a class that lists its launch's history row tiles
        /// (`StateResourcePlan::lists_history_tiles`).
        prefill_listed: Option<G::Binding<'a, attention_prefill_k8v4::Entry>>,
    },
}

impl<'a, G: GraphDraft + 'a> Copy for AttentionHistoryEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for AttentionHistoryEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// What attends a dense history's decode row classes ([`dense_decodes`]).
pub(crate) enum DenseDecodeEntries<'a, G: GraphDraft + 'a> {
    Decode(G::Binding<'a, attention_decode::Entry>),
    Prefill(G::Binding<'a, attention_prefill::Entry>),
}

impl<'a, G: GraphDraft + 'a> Copy for DenseDecodeEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for DenseDecodeEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Whether a dense history's decode row classes attend with the decode
/// entry on `backend`. A shape of the base form
/// (`AttentionShape::extends_base_form`) always does, so the decode entry's
/// refusal of it stands. Where the decode declaration holds no form of an
/// extended shape (CUDA's, which declares none of the extended forms), the
/// prefill entry, tuned on those classes, attends them, a selected read's
/// rows through the prefill entry's WorkList bitmap. An absorbed latent read
/// (`AB`) has no F16-staged form in either entry: the declarations refuse
/// F16-staged matrix forms for it (precision.md, "Absorbed latent
/// attention"), so the catalog decides. Metal's matrix decode stages in the
/// activation element and holds GLM's 64 queries of 512 (shared-tile form),
/// Vulkan's vector decode holds it, and CUDA's decode holds no stored key so
/// CUDA sends it to prefill.
pub(crate) fn dense_decodes(
    backend: seismic::BackendName,
    binding: AttentionBinding,
) -> Result<bool, crate::native::CatalogFailure> {
    if !binding.shape.extends_base_form() {
        return Ok(true);
    }
    let elements = [("A", binding.activation)];
    let statics = binding.shape.dense_mix_statics();
    match crate::programs::graph::sizing::eval_with_device(backend, |dev| {
        <attention_decode::Entry as seismic::EntryContract>::contract_scratch_configs(
            backend,
            Some(dev),
            &elements,
            &statics,
        )
    }) {
        Ok(_) => Ok(true),
        Err(seismic::ContractError::DomainViolation) => Ok(false),
        Err(error) => Err(crate::native::CatalogFailure::Preparation {
            entry: <attention_decode::Entry as seismic::Entry>::NAME,
            bindings: format!("{:?}", binding.shape),
            outcome: error.to_string(),
        }),
    }
}

impl<'a> From<&'a AttentionKernels> for AttentionGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a AttentionKernels) -> Self {
        let history = match &kernels.history {
            AttentionHistoryKernels::Dense {
                append,
                decode,
                prefill,
            } => AttentionHistoryEntries::Dense {
                codec: kernels.codec,
                append,
                decode: match decode {
                    DenseDecodeKernels::Decode(kernel) => DenseDecodeEntries::Decode(kernel),
                    DenseDecodeKernels::Prefill(kernel) => DenseDecodeEntries::Prefill(kernel),
                },
                prefill,
            },
            AttentionHistoryKernels::AffineK8V4 {
                append,
                decode,
                prefill,
                prefill_listed,
            } => AttentionHistoryEntries::AffineK8V4 {
                codec: kernels.codec,
                append,
                decode,
                prefill,
                prefill_listed: prefill_listed.as_ref(),
            },
        };
        Self {
            project: &kernels.project,
            history,
            output: (&kernels.output).into(),
            grouped: kernels.grouped.as_ref(),
            absorbed: kernels.absorbed.as_ref(),
            selection: kernels
                .selection
                .as_ref()
                .map(|selection| SelectionGraphEntries {
                    project: selection.keys.project.as_ref(),
                    pool: &selection.keys.pool,
                    indexer: (&selection.indexer).into(),
                    reads: &selection.reads,
                    rows: &selection.rows,
                    prefill: match &selection.prefill {
                        SelectedPrefillKernel::Dense(kernel) => {
                            SelectedPrefillEntries::Dense(kernel)
                        }
                        SelectedPrefillKernel::AffineK8V4(kernel) => {
                            SelectedPrefillEntries::AffineK8V4(kernel)
                        }
                    },
                }),
            low_rank_query: kernels
                .low_rank_query
                .as_ref()
                .map(|low_rank| LowRankQueryEntries {
                    widen: &low_rank.widen,
                    project: &low_rank.project,
                }),
            picked: kernels.picked.as_ref().map(|picked| PickedEntries {
                rows: picked.rows.as_ref().map(|rows| PooledEntries {
                    project: &rows.project,
                    widen: rows.widen.as_ref(),
                    pool: match &rows.pool {
                        PoolKernel::Grouped(kernel) => PoolEntry::Grouped(kernel),
                        PoolKernel::Identity(kernel) => PoolEntry::Identity(kernel),
                    },
                    key_project: &rows.key_project,
                    keys: &rows.keys,
                }),
                indexer: (&picked.indexer).into(),
            }),
        }
    }
}

impl<'a> From<&'a crate::native::IndexerKernels> for IndexerEntries<'a, NativeGraph> {
    fn from(indexer: &'a crate::native::IndexerKernels) -> Self {
        IndexerEntries {
            query_project: indexer.query_project.as_ref(),
            weights_project: indexer.weights_project.as_ref(),
            score: &indexer.score,
            topk: &indexer.topk,
            candidates: indexer.candidates.as_ref().map(|candidates| CandidateEntries {
                blocks: &candidates.blocks,
                rows: &candidates.rows,
            }),
        }
    }
}

type CheckedLowRankQueryEntries = ([(&'static str, Element); 1], [(&'static str, Element); 10]);

/// Entry element assignments from the same program binding used by native
/// preparation. The checked graph obtains all shapes from the entries.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedAttentionEntries {
    project: [(&'static str, Element); 10],
    mix: [(&'static str, Element); 1],
    output: [(&'static str, Element); 2],
    grouped: Option<[(&'static str, Element); 3]>,
    absorbed: Option<[(&'static str, Element); 3]>,
    post_norm: Option<CheckedPostNormEntries>,
    history: CodecSpec,
    /// Whether the graphs may list their launch's history row tiles
    /// (`StateResourcePlan::lists_history_tiles`, for a program whose
    /// classes have the listing axis).
    lists: bool,
    /// Whether a dense history's decode row classes attend with the decode
    /// entry ([`dense_decodes`]).
    decodes: bool,
    /// A selected read's keys' projection and indexer's elements.
    selection: Option<CheckedSelectionEntries>,
    /// A low-rank query's widening and up projection.
    low_rank_query: Option<CheckedLowRankQueryEntries>,
    /// An indexing index-list source's entries.
    picked: Option<CheckedPickedEntries>,
}

/// An indexing index-list source's entry elements: an own source's `u | g`
/// and index-key projections, the indexer's ([`CheckedIndexerEntries`]),
/// and the activation element its rows' other entries read (`widen_rows`,
/// at rate 1 `attention_append_dense`, `indexer_keys`) and `pool_rows`'
/// (its rows stored in A).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedPickedEntries {
    elements: super::pooled::PickedElements,
    indexer: CheckedIndexerEntries,
    activation: [(&'static str, Element); 1],
    pool: [(&'static str, Element); 2],
}

impl CheckedPickedEntries {
    fn new(binding: AttentionBinding) -> Option<Self> {
        let elements = super::pooled::PickedElements::of(binding)?;
        Some(Self {
            elements,
            indexer: CheckedIndexerEntries::new(elements.indexer, binding.activation),
            activation: [("A", binding.activation)],
            pool: [("A", binding.activation), ("Y", binding.activation)],
        })
    }

    fn entries<G: crate::programs::graph::draft::ElementBoundDraft>(
        &self,
    ) -> PickedEntries<'_, G> {
        PickedEntries {
            rows: self.elements.rows.as_ref().map(|rows| PooledEntries {
                project: G::bind_ranged(&rows.project[..]),
                widen: matches!(rows.pooling, Pooling::Identity).then(|| G::bind(&self.activation[..])),
                pool: match rows.pooling {
                    Pooling::Identity => PoolEntry::Identity(G::bind(&self.activation[..])),
                    Pooling::Grouped => PoolEntry::Grouped(G::bind(&self.pool[..])),
                },
                key_project: G::bind_ranged(&rows.key_project[..]),
                keys: G::bind(&self.activation[..]),
            }),
            indexer: self.indexer.entries(),
        }
    }
}

/// An indexer's entry elements: its projections' ([`IndexerElements`]) and
/// the activation element `indexer_score` reads (`rows_topk`,
/// `candidate_blocks` and `candidate_rows` read none).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedIndexerEntries {
    elements: IndexerElements,
    activation: [(&'static str, Element); 1],
}

impl CheckedIndexerEntries {
    fn new(elements: IndexerElements, activation: Element) -> Self {
        Self { elements, activation: [("A", activation)] }
    }

    fn entries<G: crate::programs::graph::draft::ElementBoundDraft>(
        &self,
    ) -> IndexerEntries<'_, G> {
        IndexerEntries {
            query_project: self
                .elements
                .query_project
                .as_ref()
                .map(|elements| G::bind_ranged(&elements[..])),
            weights_project: self
                .elements
                .weights_project
                .as_ref()
                .map(|elements| G::bind_ranged(&elements[..])),
            score: G::bind(&self.activation[..]),
            topk: G::bind(&[]),
            candidates: self.elements.produces().then_some(CandidateEntries {
                blocks: G::bind(&[]),
                rows: G::bind(&[]),
            }),
        }
    }
}

/// A selected read's entry elements: its keys' `u | g` projection
/// (`pooled::tokens_project`, published directly in F32; `None` when fused
/// into the main 6-segment `attention_project`), `pool_rows`' (its keys stored
/// in F32), and its indexer's (`unit_reads` and `candidate_rows` read none).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedSelectionEntries {
    project: Option<[(&'static str, Element); 10]>,
    pool: [(&'static str, Element); 2],
    indexer: CheckedIndexerEntries,
}

impl CheckedSelectionEntries {
    fn new(binding: AttentionBinding, selection: crate::SelectionBinding) -> Self {
        let activation = binding.activation;
        let fused = binding.fuses_selection_projections();
        Self {
            project: (!fused).then(|| {
                super::pooled::tokens_project(
                    activation,
                    binding.norm,
                    selection.keys.projection,
                    selection.keys.gate,
                    Pooling::Grouped,
                )
            }),
            pool: [("A", activation), ("Y", Element::f32())],
            indexer: CheckedIndexerEntries::new(
                IndexerElements::of(binding, selection.indexer, fused && !binding.key_value_only),
                activation,
            ),
        }
    }

    fn entries<'a, G: crate::programs::graph::draft::ElementBoundDraft + 'a>(
        &'a self,
        prefill: SelectedPrefillEntries<'a, G>,
    ) -> SelectionGraphEntries<'a, G> {
        SelectionGraphEntries {
            project: self
                .project
                .as_ref()
                .map(|elements| G::bind_ranged(&elements[..])),
            pool: G::bind(&self.pool[..]),
            indexer: self.indexer.entries(),
            reads: G::bind(&[]),
            rows: G::bind(&[]),
            prefill,
        }
    }
}

/// Refuses by name, on a backend whose entries round a row over 32 lanes
/// (Metal's and CUDA's `round_lanes`), any rounding of `binding`'s rows whose
/// runs they cannot hold ([`crate::planning::RoundingArgs::rounds_in_lanes`]):
/// its keys' and pooled rows' at the head width, its indexer's query's and
/// written keys' at the indexer's width. The CPU's entries round any run;
/// Vulkan's declare no rounding.
fn admit_lane_roundings(binding: AttentionBinding, backend: seismic::BackendName) -> Result<(), String> {
    use seismic::BackendName;
    match backend {
        BackendName::Metal | BackendName::Cuda => {}
        BackendName::Cpu | BackendName::Vulkan => return Ok(()),
    }
    let width = binding.shape.widths.width;
    let pooled = match binding.picked {
        Some(crate::PickedBinding::Own(pooled)) => Some(pooled.rounding),
        Some(crate::PickedBinding::Reindex { .. } | crate::PickedBinding::Reused) | None => None,
    };
    let indexer = binding.picked.and_then(|picked| picked.indexer());
    let rows = [
        Some(("attention keys", width, binding.shape.key_rounding)),
        pooled.map(|rounding| ("pooled rows", width, rounding)),
        indexer.map(|indexer| ("indexer query", indexer.width, indexer.query_rounding)),
        indexer.map(|indexer| ("indexer keys", indexer.width, indexer.key_rounding)),
    ];
    match rows
        .into_iter()
        .flatten()
        .find(|(_, width, rounding)| !rounding.rounds_in_lanes(*width))
    {
        None => Ok(()),
        Some((rows, width, rounding)) => Err(format!(
            "{backend:?} rounds {rows} {width} wide over 32 lanes, which runs of {} values \
             cannot follow: width / block must be a multiple of 32 or a power of two",
            rounding.block
        )),
    }
}

impl CheckedAttentionEntries {
    pub(crate) fn new(
        binding: AttentionBinding,
        backend: seismic::BackendName,
        lists: bool,
    ) -> Result<Self, String> {
        admit_lane_roundings(binding, backend)?;
        Ok(Self {
            picked: CheckedPickedEntries::new(binding),
            lists: lists && lists_history_tiles(binding),
            decodes: binding.history.encoding() != HistoryEncoding::Dense
                || dense_decodes(backend, binding).map_err(|error| format!("{error:?}"))?,
            // The up projection normalizes the F32 `qr` rows by their norm
            // and reads no gate, key or value rows.
            low_rank_query: binding.low_rank_query.map(|low_rank| {
                (
                    [("A", binding.activation)],
                    [
                        ("NW", low_rank.norm),
                        ("QW", binding.query),
                        ("GW", binding.query),
                        ("KW", binding.query),
                        ("VW", binding.query),
                        ("IQW", binding.query),
                        ("IKW", binding.query),
                        ("A", binding.activation),
                        ("Y", binding.activation),
                        ("IY", binding.activation),
                    ],
                )
            }),
            selection: binding
                .selection
                .map(|selection| CheckedSelectionEntries::new(binding, selection)),
            project: [
                ("NW", binding.norm),
                ("QW", binding.query_segment()),
                ("GW", binding.gate_segment()),
                ("KW", binding.key_segment()),
                ("VW", binding.value_segment()),
                ("IQW", binding.indexer_query_segment()),
                ("IKW", binding.indexer_key_segment()),
                ("A", binding.activation),
                ("Y", binding.activation),
                ("IY", binding.indexer_key_output()),
            ],
            mix: [("A", binding.activation)],
            output: [("OW", binding.output), ("A", binding.activation)],
            // The grouped products feed the output projection in A.
            grouped: binding.grouped.map(|grouped| {
                [
                    ("A", binding.activation),
                    ("W", grouped.element),
                    ("Y", binding.activation),
                ]
            }),
            // The absorbed queries are the attention entries' queries, in A.
            absorbed: binding.absorbed().map(|absorbed| {
                [
                    ("A", binding.activation),
                    ("W", absorbed.element),
                    ("Y", binding.activation),
                ]
            }),
            post_norm: match binding.tail {
                SublayerTail::Residual => None,
                SublayerTail::PostNorm { norm, .. } => Some(CheckedPostNormEntries::new(
                    binding.output,
                    binding.activation,
                    norm,
                )),
            },
            history: binding.history,
        })
    }

    pub(crate) fn entries<G: crate::programs::graph::draft::ElementBoundDraft>(
        &self,
    ) -> Result<AttentionGraphEntries<'_, G>, String> {
        let history = match self.history.encoding() {
            HistoryEncoding::Dense => AttentionHistoryEntries::Dense {
                codec: self.history,
                append: G::bind(&self.mix[..]),
                decode: if self.decodes {
                    DenseDecodeEntries::Decode(G::bind(&self.mix[..]))
                } else {
                    DenseDecodeEntries::Prefill(G::bind(&self.mix[..]))
                },
                prefill: G::bind(&self.mix[..]),
            },
            HistoryEncoding::Affine => AttentionHistoryEntries::AffineK8V4 {
                codec: self.history,
                append: G::bind(&self.mix[..]),
                decode: G::bind(&self.mix[..]),
                prefill: G::bind(&self.mix[..]),
                prefill_listed: self.lists.then(|| G::bind(&self.mix[..])),
            },
            HistoryEncoding::Unsupported => {
                return Err("rotated K4/V4 has no native attention entry".into());
            }
        };
        Ok(AttentionGraphEntries {
            project: G::bind_ranged(&self.project),
            history,
            output: match &self.post_norm {
                None => TailEntries::Residual(G::bind_ranged(&self.output[..])),
                Some(post_norm) => TailEntries::PostNorm(post_norm.entries()),
            },
            grouped: self.grouped.as_ref().map(|grouped| G::bind_ranged(&grouped[..])),
            absorbed: self.absorbed.as_ref().map(|absorbed| G::bind_ranged(&absorbed[..])),
            selection: self.selection.as_ref().map(|selection| {
                selection.entries(match self.history.encoding() {
                    HistoryEncoding::Affine => SelectedPrefillEntries::AffineK8V4(G::bind(&self.mix[..])),
                    _ => SelectedPrefillEntries::Dense(G::bind(&self.mix[..])),
                })
            }),
            low_rank_query: self
                .low_rank_query
                .as_ref()
                .map(|(widen, project)| LowRankQueryEntries {
                    widen: G::bind(&widen[..]),
                    project: G::bind_ranged(&project[..]),
                }),
            picked: self.picked.as_ref().map(CheckedPickedEntries::entries),
        })
    }
}

/// Whether `binding`'s prefill graphs may list their launch's history row
/// tiles where its device reads decoded history: an affine history read
/// whole (a selected read's WorkList prefill lists none).
pub(crate) fn lists_history_tiles(binding: AttentionBinding) -> bool {
    binding.history.encoding() == HistoryEncoding::Affine && binding.selection.is_none()
}

/// A layer's attention form: the static axes its history codec's fused
/// entries specialize on, the shared mix statics then the dense entries' key
/// rounding and picks or the affine planes' widths. The preparation's cases
/// and the demand's calls read it alike; a launch variant (a prefill's tile
/// listing) is the case's own beside it.
pub(crate) fn form_statics<T, A>(
    shape: &AttentionShape<T, A>,
    codec: CodecSpec,
) -> Result<Vec<(&'static str, u64)>, String> {
    match codec.encoding() {
        HistoryEncoding::Dense => Ok(shape.dense_mix_statics()),
        HistoryEncoding::Affine => Ok(shape
            .mix_statics()
            .into_iter()
            .chain(affine_dimensions(codec)?)
            .collect()),
        HistoryEncoding::Unsupported => {
            Err("rotated K4/V4 history has no native attention entry".into())
        }
    }
}

/// The cache-publishing entries' form: [`form_statics`] without the axes
/// they do not read.
pub(crate) fn append_form_statics<T, A>(
    shape: &AttentionShape<T, A>,
    codec: CodecSpec,
) -> Result<Vec<(&'static str, u64)>, String> {
    Ok(form_statics(shape, codec)?
        .into_iter()
        .filter(|(name, _)| !AttentionShape::unread_by_appends(name))
        .collect())
}

/// One translation from the load-owned plane descriptor to the affine
/// entry's static dimensions; append, tuning and sealed graphs share it.
pub(crate) fn affine_dimensions(codec: CodecSpec) -> Result<Vec<(&'static str, u64)>, String> {
    let geometry = codec.affine_geometry().map_err(|e| e.to_string())?;
    Ok(vec![
        ("KC", geometry.key_codes as u64),
        ("KP", geometry.key_coefficients as u64),
        ("VC", geometry.value_codes as u64),
        ("VP", geometry.value_coefficients as u64),
    ])
}

/// Whether a `rows`-row class attends with the decode entry: row classes up
/// to a draft verification's (`VERIFY_ROWS`); larger classes use the prefill
/// entry.
pub(crate) fn decodes(rows: impl Into<Dim>) -> bool {
    rows.into().at_most(magnitude_kernels::constants::VERIFY_ROWS)
}

/// The block's weight tensors and projection operands, as ports of the graph
/// being built. The operator's form decides which segments are present.
#[derive(Clone)]
pub(crate) struct AttentionWeights {
    /// The input norm; absent for a mix site's partials.
    pub input_norm: Option<WorkflowTensor>,
    /// The projection's query segment: the query projection (with
    /// interleaved gate rows when gated so), or a low-rank query's `down`.
    pub query: PackedWeight,
    /// A low-rank query's `qr` norm and `up`, where the block attends.
    pub low_rank_query: Option<LowRankQueryWeights>,
    /// An absorbed query's per-head key projections, where the block
    /// attends.
    pub absorbed: Option<PackedWeight>,
    /// The gate projection segment, when gated separately.
    pub gate: GateSpec<PackedWeight>,
    /// The key and value projection segments, when the layer projects its own
    /// rows.
    pub key_value: KeyValueSpec<PackedWeight>,
    pub query_norm: Option<WorkflowTensor>,
    pub key_norm: Option<WorkflowTensor>,
    /// The per-head sink logits (`[heads]`), when the operator has sinks.
    pub sinks: Option<WorkflowTensor>,
    pub output: Option<PackedWeight>,
    /// A grouped output's first projection (`a`), when it is grouped.
    pub grouped: Option<PackedWeight>,
    /// The sublayer's post-norm weight, when its tail is a post-norm.
    pub post_norm: Option<WorkflowTensor>,
    /// A selected read's weights, held apart: only an operator that selects
    /// has them.
    pub selection: Option<Box<SelectionWeights>>,
    /// An indexing index-list source's weights, held apart: only an
    /// operator with its own indexer has them.
    pub picked: Option<Box<PickedWeights>>,
}

/// A query segment's scale and bound dimensions alone, in
/// `attention_project`'s order, from its weight's `operands`: a low-rank
/// query's up projection, or an indexer's projection.
pub(crate) fn query_operand_dimensions(operands: &WeightOperands) -> [(&'static str, Dim); 8] {
    [
        ("QS", operands.scale.extent.clone()),
        ("GS", Dim::from(0)),
        ("KS", Dim::from(0)),
        ("VS", Dim::from(0)),
        ("QB", operands.bound.extent.clone()),
        ("GB", Dim::from(0)),
        ("KB", Dim::from(0)),
        ("VB", Dim::from(0)),
    ]
}

/// A low-rank query's weights after its `down`: the norm of its `qr` row
/// (`[rank]`) and its `up` projection (`[Q, rank]`).
#[derive(Clone)]
pub(crate) struct LowRankQueryWeights {
    pub norm: WorkflowTensor,
    pub up: PackedWeight,
}

/// An indexer's weights: its query, the query heads' own norm (if it has
/// one) and its head weights (if projected).
#[derive(Clone)]
pub(crate) struct IndexerWeights {
    pub query: PackedWeight,
    pub query_norm: Option<WorkflowTensor>,
    pub head_weights: Option<PackedWeight>,
}

/// A selected read's weights: its keys' `u` and (gated) `g` projections,
/// their norm (`[W]`, its bias per token), the gate's per-place term
/// (`[rate, W]`), and its indexer's where its class scores.
#[derive(Clone)]
pub(crate) struct SelectionWeights {
    pub projection: PackedWeight,
    pub gate: Option<PackedWeight>,
    pub norm: WorkflowTensor,
    pub bias: Option<WorkflowTensor>,
    pub position: Option<WorkflowTensor>,
    pub indexer: Option<IndexerWeights>,
}

/// An indexer's weights at `scope`, its query heads normed where `query_norm`
/// and head weights projected where `head_weights`.
fn indexer_weights<G: GraphDraft>(
    graph: &mut G,
    planned: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    head_weights: bool,
    query_norm: bool,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<IndexerWeights, GraphError> {
    Ok(IndexerWeights {
        query: packed_weight(graph, planned, scope, WeightKind::IndexerQuery, ports, constants)?,
        query_norm: query_norm
            .then(|| weight(graph, planned, scope, WeightKind::IndexerQueryNorm, ports))
            .transpose()?,
        head_weights: head_weights
            .then(|| packed_weight(graph, planned, scope, WeightKind::IndexerHeadWeights, ports, constants))
            .transpose()?,
    })
}

/// The segments a decode row of a read picking `top` units reads of its
/// `segments` visible spans (`candidate_rows` over its units, no fill):
/// its picked units and its tail form at most `top + 1` runs of keys, split
/// further only where a visible span ends.
pub(crate) fn selected_segments<T, D>(top: T, segments: D) -> D
where
    T: std::ops::Add<u64, Output = T>,
    D: std::ops::Add<T, Output = D>,
{
    segments + (top + 1)
}

/// A query segment's projection alone at `rows` rows (`attention_project`):
/// its `inputs`-wide rows to `outputs` rows, no gate, key or value. A
/// low-rank query's up projection (its `rank`-wide `qr` rows to the query
/// rows), and each of an indexer's projections.
pub(crate) fn query_projection_dimensions<D: From<u64>>(
    rows: D,
    inputs: D,
    outputs: D,
) -> [(&'static str, D); 8] {
    [
        ("M", rows),
        ("D", inputs),
        ("Q", outputs),
        ("GR", D::from(0)),
        ("K", D::from(0)),
        ("V", D::from(0)),
        ("IQ", D::from(0)),
        ("IK", D::from(0)),
    ]
}


/// A selected read's inputs at one graph class: its entries and weights,
/// the layer's normed input `x` and its input norm, the low-rank query's
/// F32 `qr` rows under their norm (for an indexer over them), and its
/// numbers and reals.
pub(crate) struct SelectionInputs<'n, 'v, 'a, G: GraphDraft + 'a> {
    pub entries: SelectionGraphEntries<'a, G>,
    pub weights: &'v SelectionWeights,
    pub x: ProjectionInput<'v>,
    pub input_norm: &'v WorkflowTensorView,
    pub epsilon: Argument<f32>,
    pub qr: Option<(ProjectionInput<'v>, &'v WorkflowTensorView, Argument<f32>)>,
    /// Pre-projected indexer key (`u`, F32) and query (`A`) from the fused
    /// 6-segment `attention_project` when `entries.project` is `None`.
    pub preprojected_key: Option<WorkflowTensor>,
    pub preprojected_query: Option<WorkflowTensor>,
    pub selection: SelectionPart<'n>,
}

/// What a selected read's attention reads: a prefill row's WorkList bitmap
/// `[rows, B]` and, for the decode entry, a decode row's segments
/// `[rows, top + 1 + R, 2]`.
struct Selected {
    bitmap: WorkflowTensor,
    segments: Option<WorkflowTensor>,
}

/// A selected read over the class's rows (`SelectedRead`): its keys
/// (`attention_project`'s `u | g` over `x`, F32, pooled into the Window and
/// the Block plane by `pool_rows`, `PoolNorm`'s norm, the gate's per-place
/// term), and where the class scores, its indexer over the Block read's
/// visible spans (`selection::indexer`, picking units), and its reads
/// (`unit_reads`: the WorkList bitmap, and the units `candidate_rows` maps
/// through the token read's visible spans to a decode row's segments, no
/// fill, where the class attends with the decode entry: `segments`). Its
/// planes and per-run tables are ports of the graph; `rotary` is the
/// attention's table.
#[allow(clippy::too_many_arguments)]
fn select<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    selection: &SelectionInputs<'_, '_, 'a, G>,
    topology: &AttentionTopology,
    numbers: &AttentionBlock,
    controls: &AttentionControlPorts,
    segments: bool,
    rotary: [&WorkflowTensor; 3],
    constants: &mut Vec<GraphConstant>,
) -> Result<(Option<Selected>, IndexerHistoryPorts), GraphError> {
    let SelectionInputs { entries, weights, selection: part, .. } = selection;
    let read = part.read;
    let (form, rows, pairs) = (&topology.form, &numbers.rows, &numbers.widths.rotary_pairs);
    let source = &read.source;
    let width = &source.width;
    let gt = axis(read.keys.gated);
    let gate_rows = gt * *width;
    // The tokens' `u` and `g` under the input norm, then in F32. When fused
    // into the main 6-segment `attention_project`, `preprojected_key` holds `u`
    // directly and no separate `attention_project` launch is emitted.
    let x = &selection.x;
    let (u, g) = match &selection.preprojected_key {
        Some(u) => {
            if weights.gate.is_some() {
                return Err("fused selected-read projection with a gated pool".into());
            }
            (u.clone(), u.slice_leading(0, 0).reshape(&[*rows, Dim::from(0)]))
        }
        None => {
            let project_entry = entries
                .project
                .ok_or("a selected read without its key projection entry or preprojected key")?;
            let empty = weights.projection.weight.slice_leading(0, 0);
            let absent = GraphConstant::absent(graph, constants)?.tensor().clone();
            let gate = weights.gate.as_ref();
            let (projection, gate_operands) =
                (&weights.projection.operands, gate.map(|gate| &gate.operands));
            let operand = |extent: Option<&Dim>| extent.copied().unwrap_or(Dim::from(0));
            let projected = graph.enqueue(
                G::pick(project_entry, rows),
                &[
                    [
                        ("M", *rows),
                        ("D", x.width),
                        ("Q", *width),
                        ("GR", gate_rows),
                        ("K", Dim::from(0)),
                        ("V", Dim::from(0)),
                        ("IQ", Dim::from(0)),
                        ("IK", Dim::from(0)),
                        ("QS", projection.scale.extent),
                        ("GS", operand(gate_operands.map(|operands| &operands.scale.extent))),
                        ("KS", Dim::from(0)),
                        ("VS", Dim::from(0)),
                        ("QB", projection.bound.extent),
                        ("GB", operand(gate_operands.map(|operands| &operands.bound.extent))),
                        ("KB", Dim::from(0)),
                        ("VB", Dim::from(0)),
                    ]
                    .as_slice(),
                    &x.form.dimensions(x.width),
                ]
                .concat(),
                attention_project::WorkflowArgs {
                    hidden: x.rows.into(),
                    input_norm: selection.input_norm.into(),
                    query_weight: (&weights.projection.weight).into(),
                    gate_weight: match gate {
                        Some(gate) => (&gate.weight).into(),
                        None => (&empty).into(),
                    },
                    key_weight: (&empty).into(),
                    value_weight: (&empty).into(),
                    indexer_query_weight: (&empty).into(),
                    indexer_key_weight: (&empty).into(),
                    epsilon: selection.epsilon,
                    project_mode: 0.into(),
                    query_scale: (&projection.scale.tensor).into(),
                    gate_scale: match gate_operands {
                        Some(operands) => (&operands.scale.tensor).into(),
                        None => (&absent).into(),
                    },
                    key_scale: (&absent).into(),
                    value_scale: (&absent).into(),
                    query_bound: (&projection.bound.tensor).into(),
                    gate_bound: match gate_operands {
                        Some(operands) => (&operands.bound.tensor).into(),
                        None => (&absent).into(),
                    },
                    key_bound: (&absent).into(),
                    value_bound: (&absent).into(),
                },
            )?;
            // Grouped pool projections are published directly in F32
            // (`pooled::tokens_project`).
            let u = projected.r0.clone();
            let g = match gate {
                Some(_) => projected.r1.reshape(&[*rows, gate_rows]),
                // No gate scores: zero columns of the F32 rows.
                None => u.slice_leading(0, 0).reshape(&[*rows, Dim::from(0)]),
            };
            (u, g)
        }
    };
    // The keys: each pending token's in the Window, each unit's in the Block.
    let place = if read.keys.place { read.block.clone() } else { Dim::from(0) };
    let token_norm = axis(read.keys.token_norm);
    let pool_dimensions = [
        ("M", rows.clone()),
        ("P", pairs.clone()),
        ("S", width - pairs * 2),
        ("RA", axis(form.rotary_adjacent)),
        ("RT", axis(form.rotary_trailing)),
        ("TB", source.plane.rows.clone()),
        ("TW", read.window.rows.clone()),
        ("GT", gt.clone()),
        ("PR", place.clone()),
        ("TN", token_norm.clone()),
        ("R", numbers.segments.clone()),
        ("RATE", read.block.clone()),
    ];
    let pool_control =
        |graph: &mut G, name: &str| graph.control_for(topology.controls, entries.pool, name, &pool_dimensions);
    let window_visible = pool_control(graph, "window_visible")?;
    let window_destinations = pool_control(graph, "window_destinations")?;
    let block_destinations = pool_control(graph, "block_destinations")?;
    let mut planes = IndexerStatePorts {
        window_keys: graph.port(Element::f32(), &[read.window.rows.clone(), width + &gate_rows])?,
        window_coordinates: graph.port(Element::i32(), &[read.window.rows.clone(), Dim::from(4)])?,
        block_keys: graph.port(Element::f32(), &[source.plane.rows.clone(), width.clone()])?,
    };
    let norm = weights.norm.reshape(&[Dim::from(1), width.clone()]);
    let none = norm.slice_leading(0, 0);
    let bias = match &weights.bias {
        Some(bias) => bias.reshape(&[token_norm, width.clone()]),
        None => none.clone(),
    };
    let position = match &weights.position {
        Some(position) => position.reshape(&[place, width.clone()]),
        None => none,
    };
    let rounding = RoundingArguments::of(None)?;
    let [components, frequencies, amplitudes] = rotary;
    let table = controls.destinations.tensor().slice_leading(0, 0);
    let pool_form = table.reshape(&[
        Dim::from(0),
        axis(form.rotary_adjacent),
        axis(form.rotary_trailing),
        read.block.clone(),
        gt,
    ]);
    let slab_rows = |plane: &Plane<Dim>| scalar::<u32>(&plane.slab_rows, "slab rows");
    graph.enqueue(
        entries.pool,
        &pool_dimensions,
        pool_rows::WorkflowArgs {
            u: (&u).into(),
            g: (&g).into(),
            position: (&position).into(),
            norm: (&norm).into(),
            bias: (&bias).into(),
            rotary_components: components.into(),
            rotary_frequencies: frequencies.into(),
            rotary_amplitudes: amplitudes.into(),
            coordinates: controls.coordinates.tensor().into(),
            window_visible: (&window_visible
                .tensor()
                .reshape(&[rows.clone(), numbers.segments.clone(), Dim::from(2)]))
                .into(),
            window_destinations: window_destinations.tensor().into(),
            block_destinations: block_destinations.tensor().into(),
            window_rows: planes.window_keys.tensor_mut().into(),
            window_coordinates: planes.window_coordinates.tensor_mut().into(),
            block_rows: planes.block_keys.tensor_mut().into(),
            form: (&pool_form).into(),
            epsilon: part.key_epsilon.f32(),
            window_slab_rows: slab_rows(&read.window)?,
            block_slab_rows: slab_rows(&source.plane)?,
            block: rounding.block,
            element: rounding.element,
            scale: rounding.scale,
            amax_floor: rounding.amax_floor,
        },
    )?;
    // A class without scoring only appends.
    let (Some(indexer_weights), true) = (&weights.indexer, read.scored) else {
        let tables = IndexerControlPorts {
            window_visible,
            window_destinations,
            block_destinations,
            index: None,
        };
        return Ok((None, IndexerHistoryPorts { planes, tables }));
    };
    // The indexer over the Block read's visible spans: each row's picked
    // units, over its unit list `[0, c)`.
    let class = IndexerClass {
        indexed: part.indexer,
        rows: rows.clone(),
        spans: source.spans.clone(),
        scores: source.scores.clone(),
        rotary_pairs: pairs.clone(),
    };
    let top = &source.top;
    let list_dimensions = [
        ("M", rows.clone()),
        ("N", source.scores.clone()),
        ("R", Dim::from(1)),
        ("K", top.clone()),
    ];
    let index = IndexControlPorts {
        block_visible: graph.control_for(
            topology.controls,
            entries.indexer.score,
            "spans",
            &class.score_dimensions(
                indexer_weights.query_norm.is_some(),
                indexer_weights.head_weights.is_some(),
                form.rotary_adjacent,
                form.rotary_trailing,
            ),
        )?,
        complete: graph.control_for(topology.controls, entries.indexer.topk, "complete", &list_dimensions)?,
        units: graph.control_for(topology.controls, entries.indexer.topk, "spans", &list_dimensions)?,
    };
    let spans = index
        .block_visible
        .tensor()
        .reshape(&[rows.clone(), source.spans.clone(), Dim::from(2)]);
    let complete = index.complete.tensor().reshape(&[rows.clone()]);
    let units = index.units.tensor().reshape(&[rows.clone(), Dim::from(1), Dim::from(2)]);
    let coordinates = controls.coordinates.tensor().reshape(&[rows.clone(), Dim::from(4)]);
    let query_norm = indexer_weights
        .query_norm
        .as_ref()
        .map(|norm| norm.reshape(&[Dim::from(1), width.clone()]));
    let (query_input, query_norm_input, query_epsilon) = match &selection.qr {
        Some((qr, norm, epsilon)) => (qr, *norm, *epsilon),
        None => (x, selection.input_norm, selection.epsilon),
    };
    let head_weights = indexer_weights.head_weights.as_ref().map(|head_weights| {
        (
            ProjectionInput { rows: x.rows, width: x.width.clone(), form: x.form },
            Projection {
                weight: &head_weights.weight,
                operands: &head_weights.operands,
                norm: selection.input_norm,
                epsilon: selection.epsilon,
            },
        )
    });
    let indexed = selection::indexer(
        graph,
        &class,
        entries.indexer,
        IndexerInputs {
            query_input: ProjectionInput {
                rows: query_input.rows,
                width: query_input.width.clone(),
                form: query_input.form,
            },
            query: Projection {
                weight: &indexer_weights.query.weight,
                operands: &indexer_weights.query.operands,
                norm: query_norm_input,
                epsilon: query_epsilon,
            },
            preprojected_query: selection.preprojected_query.clone(),
            query_norm: query_norm.as_ref(),
            head_weights,
            write: None,
            coordinates: &coordinates,
            rotary,
            rotary_adjacent: form.rotary_adjacent,
            rotary_trailing: form.rotary_trailing,
            keys: &mut planes.block_keys,
            spans: &spans,
            complete: &complete,
            picks: selection::Picks::Units(&units),
        },
        constants,
    )?;
    // Its reads: the bitmap and the units, then a decode row's segments.
    let words = &read.words;
    let bitmap_words = GraphConstant::u32_zeros(graph, 0, words.clone())?;
    let reads = graph.enqueue(
        entries.reads,
        &[("M", rows.clone()), ("K", top.clone()), ("B", words.clone())],
        unit_reads::WorkflowArgs {
            picks: (&indexed.picks).into(),
            complete: (&complete).into(),
            bitmap_words: bitmap_words.port().tensor().into(),
        },
    )?;
    constants.push(bitmap_words);
    let segments = match segments {
        false => None,
        true => {
            let visible = &controls
                .read
                .as_ref()
                .ok_or("attention read controls are absent")?
                .visible;
            let picked = top.clone() + 1;
            Some(
                graph
                    .enqueue(
                        entries.rows,
                        &[
                            ("M", rows.clone()),
                            ("R", numbers.segments.clone()),
                            ("B", picked),
                            ("BU", read.block.clone()),
                            ("K", Dim::from(0)),
                        ],
                        candidate_rows::WorkflowArgs {
                            blocks: (&reads.r1).into(),
                            spans: (&visible
                                .tensor()
                                .reshape(&[rows.clone(), numbers.segments.clone(), Dim::from(2)]))
                                .into(),
                            form: (&table.reshape(&[Dim::from(0), read.block.clone(), Dim::from(0)])).into(),
                        },
                    )?
                    .value,
            )
        }
    };
    let tables = IndexerControlPorts {
        window_visible,
        window_destinations,
        block_destinations,
        index: Some(index),
    };
    Ok((
        Some(Selected { bitmap: reads.r0, segments }),
        IndexerHistoryPorts { planes, tables },
    ))
}

/// The layer's history planes, bound to the state store's planes per run, in
/// the order of the history codec's plane descriptors (the key's planes,
/// then the value's): dense key and value, or affine key codes, key
/// coefficients, value codes and value coefficients.
#[derive(Clone)]
pub(crate) struct AttentionStatePorts {
    pub planes: Vec<NativePort>,
    /// A selected read's indexer planes and tables.
    pub indexer: Option<IndexerHistoryPorts>,
    /// An index-list source's planes and tables.
    pub picked: Option<PickedPorts>,
}

/// A selected read's indexer planes and per-run tables, present together by
/// construction.
#[derive(Clone)]
pub(crate) struct IndexerHistoryPorts {
    pub planes: IndexerStatePorts,
    pub tables: IndexerControlPorts,
}

/// A selected read's key planes, bound to its two history domains' planes
/// per run: the pending tokens' keys (and gate scores) and their
/// coordinates (Window), and the units' keys (Block).
#[derive(Clone)]
pub(crate) struct IndexerStatePorts {
    pub window_keys: NativePort,
    pub window_coordinates: NativePort,
    pub block_keys: NativePort,
}

/// A selected read's per-run tables: each row's spans and destination in
/// its Window read, its destination in its Block read, and where its class
/// scores, its indexer's ([`IndexControlPorts`]).
#[derive(Clone)]
pub(crate) struct IndexerControlPorts {
    pub window_visible: NativePort,
    pub window_destinations: NativePort,
    pub block_destinations: NativePort,
    pub index: Option<IndexControlPorts>,
}

/// A scoring class's indexer tables over the Block read: each row's
/// visible spans (its list), its complete units and its unit list `[0, c)`.
#[derive(Clone)]
pub(crate) struct IndexControlPorts {
    pub block_visible: NativePort,
    pub complete: NativePort,
    pub units: NativePort,
}

/// Per-run read tables of an attending (non-`inject_only`) attention graph:
/// visible history spans, fresh spans over the batch rows, and (for an
/// affine prefill that lists them) the history row tiles the launch's rows see.
#[derive(Clone)]
pub(crate) struct AttentionReadControls {
    pub visible: NativePort,
    pub fresh: NativePort,
    /// The history row tiles the launch's rows see, for an entry that takes
    /// them (the affine prefill).
    pub history_tiles: Option<HistoryTilesPort>,
}

/// Per-run row tables: rotary coordinates, the history row each row's K/V
/// appends to, and (where the graph attends) its read tables.
#[derive(Clone)]
pub(crate) struct AttentionControlPorts {
    pub coordinates: NativePort,
    pub destinations: NativePort,
    pub read: Option<AttentionReadControls>,
}

/// The `history_tiles` input of a graph class that lists the history row
/// tiles its launch's rows see.
#[derive(Clone)]
pub(crate) struct HistoryTilesPort {
    pub port: NativePort,
}

/// The history row tiles the rows of a launch see, from its encoded
/// `visible` table (little-endian `[start, end)` pairs): distinct and
/// ascending, then -1 up to `tiles` entries, as little-endian bytes. `None`
/// when they see more than `tiles`: the launch takes the class that lists
/// none.
pub(crate) fn history_tile_bytes(visible: &[u8], tiles: usize) -> Option<Vec<u8>> {
    let spans = visible.chunks_exact(8).map(|span| {
        [
            i32::from_le_bytes([span[0], span[1], span[2], span[3]]),
            i32::from_le_bytes([span[4], span[5], span[6], span[7]]),
        ]
    });
    magnitude_batching::history_tiles(spans, tiles)
        .map(|tiles| tiles.iter().flat_map(|tile| tile.to_le_bytes()).collect())
}

impl AttentionWeights {
    /// The weights and projection operands at `scope` of the block `topology`
    /// at `numbers` that its graph over `form` reads, its output's only where
    /// it attends (`attends`), with `input_norm` replacing the input norm port
    /// where given.
    pub(crate) fn planned<G: GraphDraft>(
        graph: &mut G,
        planned: &(impl PlannedWeights + ?Sized),
        scope: WeightScope,
        (topology, numbers): (&AttentionTopology, &AttentionBlock),
        (form, attends): (InputForm, bool),
        input_norm: Option<&WorkflowTensor>,
        (ports, constants): (&mut Vec<(WeightPort, NativePort)>, &mut Vec<GraphConstant>),
    ) -> Result<Self, GraphError> {
        let block = AttentionPair::of(topology, numbers);
        let layer = &topology.form;
        let query = if attends {
            packed_weight(graph, planned, scope, topology.query_segment_kind, ports, constants)?
        } else {
            PackedWeight {
                weight: weight(graph, planned, scope, topology.query_segment_kind, ports)?,
                operands: WeightOperands::absent(graph, constants)?,
            }
        };
        let gate = layer.gate.try_map(|()| {
            if attends {
                packed_weight(graph, planned, scope, WeightKind::AttentionGate, ports, constants)
            } else {
                Ok(PackedWeight {
                    weight: weight(graph, planned, scope, WeightKind::AttentionGate, ports)?,
                    operands: WeightOperands::absent(graph, constants)?,
                })
            }
        })?;
        let key_value = match layer.key_value {
            KeyValueSpec::Shared => KeyValueSpec::Shared,
            KeyValueSpec::Fresh { key: (), value } => KeyValueSpec::Fresh {
                key: packed_weight(graph, planned, scope, WeightKind::Key, ports, constants)?,
                value: value.try_map(|()| {
                    packed_weight(graph, planned, scope, WeightKind::Value, ports, constants)
                })?,
            },
        };
        Ok(Self {
            input_norm: form
                .normalized
                .then(|| match input_norm {
                    Some(norm) => Ok(norm.clone()),
                    None => weight(graph, planned, scope, WeightKind::InputNorm, ports),
                })
                .transpose()?,
            query,
            low_rank_query: (attends && block.low_rank_query.is_some())
                .then(|| {
                    Ok::<_, GraphError>(LowRankQueryWeights {
                        norm: weight(graph, planned, scope, WeightKind::QueryDownNorm, ports)?,
                        up: packed_weight(graph, planned, scope, topology.query_kind, ports, constants)?,
                    })
                })
                .transpose()?,
            absorbed: (attends && numbers.absorbed.is_some())
                .then(|| packed_weight(graph, planned, scope, WeightKind::QueryKeys, ports, constants))
                .transpose()?,
            gate,
            key_value,
            query_norm: (attends && layer.query_norm)
                .then(|| weight(graph, planned, scope, WeightKind::QueryNorm, ports))
                .transpose()?,
            // A Shared layer projects no keys, so it has no key norm.
            key_norm: (layer.key_norm && layer.is_fresh())
                .then(|| weight(graph, planned, scope, WeightKind::KeyNorm, ports))
                .transpose()?,
            sinks: (attends && layer.sinks)
                .then(|| weight(graph, planned, scope, WeightKind::AttentionSinks, ports))
                .transpose()?,
            output: attends
                .then(|| packed_weight(graph, planned, scope, WeightKind::AttentionOutput, ports, constants))
                .transpose()?,
            grouped: (attends && numbers.grouped.is_some())
                .then(|| packed_weight(graph, planned, scope, WeightKind::AttentionOutputGrouped, ports, constants))
                .transpose()?,
            post_norm: topology
                .post_norm
                .is_some()
                .then(|| weight(graph, planned, scope, WeightKind::PostNorm, ports))
                .transpose()?,
            picked: match block.picked {
                Some(PickedPart::Indexed(indexed)) if attends => {
                    let index = indexed.source;
                    Some(Box::new(PickedWeights {
                        rows: match indexed.own {
                            Some(own) => Some(PooledWeights {
                                projection: packed_weight(
                                    graph,
                                    planned,
                                    scope,
                                    WeightKind::PooledProjection,
                                    ports,
                                    constants,
                                )?,
                                gate: own
                                    .rows
                                    .gated
                                    .then(|| {
                                        packed_weight(
                                            graph,
                                            planned,
                                            scope,
                                            WeightKind::PooledGate,
                                            ports,
                                            constants,
                                        )
                                    })
                                    .transpose()?,
                                norm: weight(graph, planned, scope, WeightKind::PooledNorm, ports)?,
                                key: packed_weight(
                                    graph,
                                    planned,
                                    scope,
                                    WeightKind::IndexerKey,
                                    ports,
                                    constants,
                                )?,
                                key_norm: weight(
                                    graph,
                                    planned,
                                    scope,
                                    WeightKind::IndexerKeyNorm,
                                    ports,
                                )?,
                            }),
                            None => None,
                        },
                        indexer: indexer_weights(
                            graph,
                            planned,
                            scope,
                            index.head_weights,
                            indexed.query_norm.is_some(),
                            ports,
                            constants,
                        )?,
                    }))
                }
                _ => None,
            },
            selection: block
                .selection
                .map(|selection| {
                    let read = selection.read;
                    Ok::<_, GraphError>(Box::new(SelectionWeights {
                        projection: packed_weight(
                            graph,
                            planned,
                            scope,
                            WeightKind::IndexerKey,
                            ports,
                            constants,
                        )?,
                        gate: read
                            .keys
                            .gated
                            .then(|| {
                                packed_weight(
                                    graph,
                                    planned,
                                    scope,
                                    WeightKind::IndexerKeyGate,
                                    ports,
                                    constants,
                                )
                            })
                            .transpose()?,
                        norm: weight(graph, planned, scope, WeightKind::IndexerKeyNorm, ports)?,
                        bias: read
                            .keys
                            .token_norm
                            .then(|| {
                                weight(graph, planned, scope, WeightKind::IndexerKeyNormBias, ports)
                            })
                            .transpose()?,
                        position: read
                            .keys
                            .place
                            .then(|| {
                                weight(
                                    graph,
                                    planned,
                                    scope,
                                    WeightKind::IndexerKeyGatePosition,
                                    ports,
                                )
                            })
                            .transpose()?,
                        indexer: (attends && (read.scored || (!read.keys.gated && !read.query_rank)))
                            .then(|| {
                                indexer_weights(
                                    graph,
                                    planned,
                                    scope,
                                    read.scored && read.source.head_weights,
                                    read.scored && selection.indexer.query_norm.is_some(),
                                    ports,
                                    constants,
                                )
                            })
                            .transpose()?,
                    }))
                })
                .transpose()?,
        })
    }

    /// These weights as another pass of the same layer reads them under its
    /// own `input_norm`: every projection and its operands, with no post-norm
    /// tail and no selected read or index-list source of the pass's own.
    pub(crate) fn for_block(&self, input_norm: WorkflowTensor) -> Self {
        Self {
            input_norm: Some(input_norm),
            post_norm: None,
            selection: None,
            picked: None,
            ..self.clone()
        }
    }

    /// The projection's scale and bound dimensions, in `attention_project`'s
    /// order: a history injection (`inject_only`) projects only keys and
    /// values.
    pub(crate) fn project_dimensions(&self, inject_only: bool) -> [(&'static str, Dim); 8] {
        let attended = |dim: Dim| if inject_only { Dim::from(0) } else { dim };
        let scale = |w: Option<&PackedWeight>| w.map_or(Dim::from(0), |w| w.operands.scale.extent.clone());
        let bound = |w: Option<&PackedWeight>| w.map_or(Dim::from(0), |w| w.operands.bound.extent.clone());
        [
            ("QS", attended(self.query.operands.scale.extent.clone())),
            ("GS", attended(scale(self.gate.separate()))),
            ("KS", scale(self.key_value.key())),
            ("VS", scale(self.key_value.value())),
            ("QB", attended(self.query.operands.bound.extent.clone())),
            ("GB", attended(bound(self.gate.separate()))),
            ("KB", bound(self.key_value.key())),
            ("VB", bound(self.key_value.value())),
        ]
    }
}

/// The attention block `topology` at `numbers` over `input`: its
/// projection, mix and output, the new residual, and its state and control
/// ports.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    kernels: AttentionGraphEntries<'a, G>,
    topology: &AttentionTopology,
    numbers: &AttentionBlock,
    weights: &AttentionWeights,
    constants: &mut Vec<GraphConstant>,
    input: SublayerInput<'_>,
    residual: &WorkflowTensor,
    join: &Join,
) -> Result<(WorkflowTensor, AttentionStatePorts, AttentionControlPorts), GraphError> {
    let block = AttentionPair::of(topology, numbers);
    let (form, widths) = (&topology.form, &numbers.widths);
    let (rows, width, heads) = (&numbers.rows, &widths.width, widths.heads());
    let hidden = &widths.hidden;
    let inject_only = topology.inject_only;
    let input_norm = match &weights.input_norm {
        Some(norm) => norm.clone(),
        None => GraphConstant::absent(graph, constants)?.tensor().clone(),
    };
    // An absent projection segment reads zero rows of the query weight (or the
    // gate weight for an injected separate-gate layer, preserving `GW`).
    let empty_weight = weights.query.weight.slice_leading(0, 0);
    let empty_gate_weight = weights
        .gate
        .separate()
        .map(|gate| gate.weight.slice_leading(0, 0))
        .unwrap_or_else(|| empty_weight.clone());
    // A history injection reads neither the query's nor the gate's operands.
    let active_query = (!inject_only).then_some(&weights.query);
    let active_gate = (!inject_only).then(|| weights.gate.separate()).flatten();
    let active_key = weights.key_value.key();
    let active_value = weights.key_value.value();
    let fuse_selection = kernels.selection.is_some_and(|sel| sel.project.is_none());
    let (active_indexer_query, active_indexer_key, empty_indexer_weight, indexer_dims) =
        if fuse_selection {
            let (Some(sel_weights), Some(sel_part)) = (weights.selection.as_deref(), block.selection)
            else {
                return Err("fused selection entries without selection weights or block".into());
            };
            let active_iq = (!inject_only)
                .then(|| sel_weights.indexer.as_ref().map(|i| &i.query))
                .flatten();
            let active_ik = Some(&sel_weights.projection);
            if let Some(iq) = active_iq {
                assert_eq!(
                    iq.operands.scale.extent,
                    Dim::from(0),
                    "fused indexer_query_weight must not carry a second-level scale"
                );
            }
            if let Some(ik) = active_ik {
                assert_eq!(
                    ik.operands.scale.extent,
                    Dim::from(0),
                    "fused indexer_key_weight must not carry a second-level scale"
                );
            }
            let iq_rows = if active_iq.is_some() {
                sel_part.read.source.heads * sel_part.read.source.width
            } else {
                Dim::from(0)
            };
            let ik_rows = sel_part.read.source.width;
            (active_iq, active_ik, empty_weight.clone(), [("IQ", iq_rows), ("IK", ik_rows)])
        } else {
            (
                None,
                None,
                empty_weight.clone(),
                [("IQ", Dim::from(0)), ("IK", Dim::from(0))],
            )
        };
    let absent = (active_query.is_none()
        || active_gate.is_none()
        || active_key.is_none()
        || active_value.is_none())
    .then(|| GraphConstant::absent(graph, constants).map(|port| port.tensor().clone()))
    .transpose()?;
    fn opt_weight<'a>(
        w: Option<&'a PackedWeight>,
        empty: &'a WorkflowTensorView,
    ) -> seismic::WorkflowTensorRef<'a> {
        match w {
            Some(w) => (&w.weight).into(),
            None => empty.into(),
        }
    }
    fn opt_scale<'a>(
        w: Option<&'a PackedWeight>,
        absent: Option<&'a WorkflowTensor>,
    ) -> &'a WorkflowTensor {
        w.map_or_else(
            || absent.expect("absent constant for missing segment"),
            |w| &w.operands.scale.tensor,
        )
    }
    fn opt_bound<'a>(
        w: Option<&'a PackedWeight>,
        absent: Option<&'a WorkflowTensor>,
    ) -> &'a WorkflowTensor {
        w.map_or_else(
            || absent.expect("absent constant for missing segment"),
            |w| &w.operands.bound.tensor,
        )
    }
    // The query projection yields the attended query rows, or an absorbed
    // query's heads at their own width; a low-rank query's projection
    // segment projects its `rank` rows of `qr` (a history injection projects
    // no query).
    let query_rows = numbers.query_projection_rows(form);
    let projected = graph.enqueue(
        G::pick(kernels.project, &rows),
        &[
            if inject_only {
                widths.key_value_dimensions(form, rows.clone())
            } else {
                let segment = block.low_rank_query.map_or(&query_rows, |low_rank| low_rank.rank);
                widths.project_dimensions(form, rows.clone(), segment.clone())
            }
            .as_slice(),
            weights.project_dimensions(inject_only).as_slice(),
            &indexer_dims,
            &input.form.dimensions(hidden.clone()),
        ]
        .concat(),
        attention_project::WorkflowArgs {
            hidden: input.rows.into(),
            input_norm: (&input_norm).into(),
            query_weight: opt_weight(active_query, &empty_weight),
            gate_weight: opt_weight(active_gate, &empty_gate_weight),
            key_weight: opt_weight(active_key, &empty_weight),
            value_weight: opt_weight(active_value, &empty_weight),
            indexer_query_weight: opt_weight(active_indexer_query, &empty_indexer_weight),
            indexer_key_weight: opt_weight(active_indexer_key, &empty_weight),
            epsilon: topology.epsilon.f32(),
            project_mode: 0.into(),
            query_scale: opt_scale(active_query, absent.as_ref()).into(),
            gate_scale: opt_scale(active_gate, absent.as_ref()).into(),
            key_scale: opt_scale(active_key, absent.as_ref()).into(),
            value_scale: opt_scale(active_value, absent.as_ref()).into(),
            query_bound: opt_bound(active_query, absent.as_ref()).into(),
            gate_bound: opt_bound(active_gate, absent.as_ref()).into(),
            key_bound: opt_bound(active_key, absent.as_ref()).into(),
            value_bound: opt_bound(active_value, absent.as_ref()).into(),
        },
    )?;
    let preprojected_indexer_query = active_indexer_query.map(|_| projected.r4.clone());
    let preprojected_indexer_key = active_indexer_key.map(|_| projected.r5.clone());
    // A low-rank query's `qr` rows (`[rows, rank]`, the query segment),
    // widened to F32: the intermediate later stages read. The up projection
    // normalizes them by their norm in its prologue and projects the query
    // heads.
    let (query, qr) = match (
        inject_only,
        block.low_rank_query,
        kernels.low_rank_query,
        &weights.low_rank_query,
    ) {
        (true, ..) | (false, None, None, None) => (projected.r0, None),
        (false, Some(low_rank), Some(entries), Some(up)) => {
            let rank = low_rank.rank;
            let qr = graph
                .enqueue(
                    entries.widen,
                    &[("M", rows.clone()), ("D", rank.clone())],
                    widen_rows::WorkflowArgs {
                        rows: (&projected.r0).into(),
                    },
                )?
                .value;
            let empty = up.up.weight.slice_leading(0, 0);
            let absent = GraphConstant::absent(graph, constants)?.tensor().clone();
            let up_operands = &up.up.operands;
            let query = graph
                .enqueue(
                    G::pick(entries.project, &rows),
                    &[
                        query_projection_dimensions(rows.clone(), rank.clone(), query_rows.clone()).as_slice(),
                        query_operand_dimensions(up_operands).as_slice(),
                        &InputForm::SINGLE.dimensions(rank.clone()),
                    ]
                    .concat(),
                    attention_project::WorkflowArgs {
                        hidden: (&qr).into(),
                        input_norm: (&up.norm).into(),
                        query_weight: (&up.up.weight).into(),
                        gate_weight: (&empty).into(),
                        key_weight: (&empty).into(),
                        value_weight: (&empty).into(),
                        indexer_query_weight: (&empty).into(),
                        indexer_key_weight: (&empty).into(),
                        epsilon: low_rank.epsilon.f32(),
                        project_mode: 0.into(),
                        query_scale: (&up_operands.scale.tensor).into(),
                        gate_scale: (&absent).into(),
                        key_scale: (&absent).into(),
                        value_scale: (&absent).into(),
                        query_bound: (&up_operands.bound.tensor).into(),
                        gate_bound: (&absent).into(),
                        key_bound: (&absent).into(),
                        value_bound: (&absent).into(),
                    },
                )?
                .r0;
            (query, Some((qr, rank.clone(), low_rank.epsilon.f32(), &up.norm)))
        }
        _ => {
            return Err(
                "attention low-rank query entries and weights disagree with its operator".into(),
            );
        }
    };
    // An absorbed query: each head's projected row through its own block of
    // the key projections (`project_rows` over the heads, F32 sums published
    // in A), the `W`-wide query the history entry attends with.
    let query = match (inject_only, &numbers.absorbed, kernels.absorbed, &weights.absorbed) {
        (true, ..) | (false, None, None, None) => query,
        (false, Some(absorbed), Some(entry), Some(packed)) => {
            graph
                .enqueue(
                    G::pick(entry, &rows),
                    &[
                        ("M", rows.clone()),
                        ("G", absorbed.heads.clone()),
                        ("K", absorbed.width.clone()),
                        ("N", absorbed.keys.clone()),
                        ("WS", packed.operands.scale.extent.clone()),
                        ("WB", packed.operands.bound.extent.clone()),
                    ],
                    project_rows::WorkflowArgs {
                        source: (&query.reshape(&[rows.clone(), query_rows])).into(),
                        weight: (&packed.weight).into(),
                        weight_scale: (&packed.operands.scale.tensor).into(),
                        weight_bound: (&packed.operands.bound.tensor).into(),
                    },
                )?
                .value
        }
        _ => return Err("absorbed query entries, weights and form disagree".into()),
    };
    // The mix entries read the projection per head.
    let attended_heads = if inject_only { Dim::from(0) } else { heads.clone() };
    let fresh = axis(form.is_fresh());
    let key_width = widths.key_width();
    let projected = ProjectedRows {
        query: query.reshape(&[
            rows.clone(),
            attended_heads.clone(),
            width + widths.interleaved_gate(form),
        ]),
        gate: projected
            .r1
            .reshape(&[rows.clone(), attended_heads, widths.separate_gate(form)]),
        key: projected
            .r2
            .reshape(&[fresh.clone(), rows.clone(), key_width.clone()]),
        value: match form.key_value {
            KeyValueSpec::Shared | KeyValueSpec::Fresh { value: ValueSpec::Projected(()), .. } => {
                projected.r3.reshape(&[fresh, rows.clone(), key_width])
            }
            KeyValueSpec::Fresh { value: ValueSpec::RawKey, .. } => {
                projected.r2.reshape(&[fresh, rows.clone(), key_width])
            }
            // The entries take no value of their own (`VK`): zero layers.
            KeyValueSpec::Fresh { value: ValueSpec::StoredKey, .. } => projected
                .r2
                .slice_leading(0, 0)
                .reshape(&[Dim::from(0), rows.clone(), key_width]),
        },
    };
    // An own index-list source projects the layer's normed input and its
    // indexer the low-rank query's `qr` rows.
    let x = input.rows.slice_leading(0, rows.clone());
    let x_norm = input_norm.reshape(&[if weights.input_norm.is_some() {
        hidden.clone()
    } else {
        Dim::from(0)
    }]);
    let qr_rows = qr.as_ref().map(|(qr, rank, epsilon, norm)| {
        (qr.slice_leading(0, rows.clone()), rank.clone(), *epsilon, norm.reshape(&[rank.clone()]))
    });
    let x_input = || ProjectionInput { rows: &x, width: hidden.clone(), form: input.form };
    // A selected read projects its keys from `x` and its query from `x` or
    // the low-rank query's `qr`.
    let selection = match (
        kernels.selection,
        weights.selection.as_deref(),
        block.selection,
    ) {
        (None, None, None) => None,
        (Some(entries), Some(weights), Some(selection)) => Some(SelectionInputs {
            entries,
            weights,
            x: x_input(),
            input_norm: &x_norm,
            epsilon: topology.epsilon.f32(),
            // Only a graph that runs the indexer reads them.
            qr: match (selection.read.query_rank && weights.indexer.is_some(), &qr_rows) {
                (false, _) => None,
                (true, Some((qr, rank, qr_epsilon, qr_norm))) => Some((
                    ProjectionInput { rows: qr, width: rank.clone(), form: InputForm::SINGLE },
                    qr_norm,
                    *qr_epsilon,
                )),
                (true, None) => return Err("a selected read's query over the absent `qr` rows".into()),
            },
            preprojected_key: preprojected_indexer_key,
            preprojected_query: preprojected_indexer_query,
            selection,
        }),
        _ => return Err("attention selection entries, weights and block disagree".into()),
    };
    let picked = match block.picked {
        None => None,
        Some(PickedPart::Reused { plane }) => Some(Picked::Reused { plane }),
        Some(PickedPart::Indexed(indexed)) => {
            let (Some(entries), Some(weights), Some((qr, rank, qr_epsilon, qr_norm))) =
                (kernels.picked, weights.picked.as_deref(), &qr_rows)
            else {
                return Err(
                    "an indexing index-list source without its entries, weights or low-rank query"
                        .into(),
                );
            };
            Some(Picked::Indexed {
                indexed,
                inputs: PickedInputs {
                    entries,
                    weights,
                    x: ProjectionInput {
                        rows: &x,
                        width: hidden.clone(),
                        form: input.form,
                    },
                    input_norm: &x_norm,
                    epsilon: topology.epsilon.f32(),
                    qr: ProjectionInput {
                        rows: qr,
                        width: rank.clone(),
                        form: InputForm::SINGLE,
                    },
                    qr_norm,
                    qr_epsilon: *qr_epsilon,
                },
            })
        }
    };
    let (attended, state, controls) = mix(
        graph,
        kernels.history,
        (&weights.query_norm, &weights.key_norm),
        &weights.sinks,
        constants,
        &projected,
        topology,
        numbers,
        selection,
        picked,
        join.class,
    )?;
    if inject_only {
        return Ok((residual.clone(), state, controls));
    }
    let attended = attended.ok_or("attention result is absent")?;
    let output_weight = weights
        .output
        .as_ref()
        .ok_or("attention output weight is absent")?;
    // A grouped output first projects each group's run of the attended row
    // through its own block of `a` (`project_rows` over the groups, in A);
    // the output projection reads those products as its `(Q, W)` rows.
    let (attended, (q, w)) = match (&numbers.grouped, kernels.grouped, &weights.grouped) {
        (None, None, None) => (
            attended.reshape(&[rows.clone(), heads.clone(), width.clone()]),
            (heads.clone(), width.clone()),
        ),
        (Some(grouped), Some(entry), Some(packed)) => {
            let products = graph
                .enqueue(
                    G::pick(entry, &rows),
                    &[
                        ("M", rows.clone()),
                        ("G", grouped.groups.clone()),
                        ("K", grouped.inputs.clone()),
                        ("N", grouped.rank.clone()),
                        ("WS", packed.operands.scale.extent.clone()),
                        ("WB", packed.operands.bound.extent.clone()),
                    ],
                    project_rows::WorkflowArgs {
                        source: (&attended.reshape(&[rows.clone(), &heads * width])).into(),
                        weight: (&packed.weight).into(),
                        weight_scale: (&packed.operands.scale.tensor).into(),
                        weight_bound: (&packed.operands.bound.tensor).into(),
                    },
                )?
                .value;
            (
                products.reshape(&[rows.clone(), grouped.groups.clone(), grouped.rank.clone()]),
                (grouped.groups.clone(), grouped.rank.clone()),
            )
        }
        _ => return Err("grouped attention output entries, weights and form disagree".into()),
    };
    let mixed = match (kernels.output, &weights.post_norm, &topology.post_norm) {
        (TailEntries::Residual(output), None, None) => {
            let [streams, injected] = join.dimensions();
            let inject = join.inject(graph, rows.clone(), constants)?;
            graph
                .enqueue(
                    G::pick(output, &rows),
                    &[
                        ("M", rows.clone()),
                        ("D", hidden.clone()),
                        streams,
                        injected,
                        ("Q", q),
                        ("W", w),
                        ("OS", output_weight.operands.scale.extent.clone()),
                        ("OB", output_weight.operands.bound.extent.clone()),
                    ],
                    attention_output::WorkflowArgs {
                        hidden: residual.into(),
                        inject: (&inject).into(),
                        gated: (&attended).into(),
                        output_weight: (&output_weight.weight).into(),
                        output_scale: (&output_weight.operands.scale.tensor).into(),
                        output_bound: (&output_weight.operands.bound.tensor).into(),
                    },
                )?
                .value
        }
        (TailEntries::PostNorm(entries), Some(norm), Some(tail)) => {
            let out_rows = GraphConstant::row_identity(graph, rows.clone(), Some("M"))?;
            let inputs = q * w;
            let mixed = post_norm(
                graph,
                entries,
                residual,
                (&attended.reshape(&[rows.clone(), inputs.clone()])).into(),
                output_weight,
                norm,
                out_rows.port().tensor(),
                PostNormShape {
                    rows: rows.clone(),
                    out: rows.clone(),
                    inputs,
                    outputs: hidden.clone(),
                },
                tail,
            )?;
            constants.push(out_rows);
            mixed
        }
        _ => return Err("attention output entries disagree with its post-norm weight".into()),
    };
    Ok((mixed, state, controls))
}

/// What the dense entries read of a layer's index-list source
/// (`HistorySource` 1): each launch row's picks (`[rows, KS]` i32, plane
/// rows, -1 for none) in the source's pooled plane (`[TS, KV, W]` in A, of
/// `slab_rows` per slab).
#[derive(Clone)]
pub(crate) struct PickedReads {
    pub rows: WorkflowTensorView,
    pub plane: WorkflowTensorView,
    pub slab_rows: Argument<u32>,
}

/// The per-head rows the history codec's attention entry reads: the query
/// (with its interleaved gate), the separate gate, and the fresh keys and
/// values.
#[derive(Clone)]
pub(crate) struct ProjectedRows {
    /// `[rows, heads, width + interleaved gate]`.
    pub query: WorkflowTensorView,
    /// `[rows, heads, separate gate]`.
    pub gate: WorkflowTensorView,
    /// `[fresh, rows, kv heads · width]`.
    pub key: WorkflowTensorView,
    pub value: WorkflowTensorView,
}

/// The history codec's fused attention entry over projected rows: head
/// norms, rotary, K/V append at the rows' destinations and attention over
/// the visible spans and fresh rows, with the operator's per-head `sinks`.
/// Its rotary table and unit norm row join
/// `constants`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mix<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    history: AttentionHistoryEntries<'a, G>,
    (query_norm, key_norm): (&Option<WorkflowTensor>, &Option<WorkflowTensor>),
    sinks: &Option<WorkflowTensor>,
    constants: &mut Vec<GraphConstant>,
    projected: &ProjectedRows,
    topology: &AttentionTopology,
    numbers: &AttentionBlock,
    selection: Option<SelectionInputs<'_, '_, 'a, G>>,
    picked: Option<Picked<'_, '_, 'a, G>>,
    row_class: Option<&'static str>,
) -> Result<
    (
        Option<WorkflowTensor>,
        AttentionStatePorts,
        AttentionControlPorts,
    ),
    GraphError,
> {
    let (form, widths) = (&topology.form, &numbers.widths);
    let (rows, width, kv_heads) = (&numbers.rows, &widths.width, &widths.kv_heads);
    let inject_only = topology.inject_only;
    // A selected read's class that scores reads through the indexer's
    // selection: its prefill through a bitmap of `B` words of `BR`-key
    // blocks.
    let scoring = selection.as_ref().filter(|selection| selection.selection.read.scored);
    let words = scoring.map_or(Dim::from(0), |selection| selection.selection.read.words.clone());
    let block_keys = scoring.map_or(Dim::from(1), |selection| selection.selection.read.block.clone());
    let selected_prefill = scoring.map(|selection| selection.entries.prefill);
    let dense = matches!(&history, AttentionHistoryEntries::Dense { .. });
    let key_rounding = AttentionPair::of(topology, numbers).key_rounding;
    if !dense && (key_rounding.is_some() || picked.is_some()) {
        unreachable!("the binding admits a rounded or index-list source beside a dense history alone");
    }
    // Priming publishes history rows alone, and an index-list source's
    // pooled rows and keys only its attending graph makes.
    if inject_only && picked.is_some() {
        return Err("a history injection of an attention with an index-list source".into());
    }
    // The dense entries also take source 0's key rounding (`KR`) and the
    // index-list source's picks (`KS`) of its plane's `TS` rows.
    let mut dimensions = widths
        .mix_dimensions(
            form,
            rows.clone(),
            numbers.history_rows.clone(),
            numbers.segments.clone(),
        )
        .to_vec();
    match &history {
        AttentionHistoryEntries::Dense { .. } => {
            dimensions.extend(widths.dense_statics(key_rounding.is_some()));
            dimensions.push((
                "TS",
                picked.as_ref().map_or(Dim::from(0), |picked| picked.plane().rows.clone()),
            ));
        }
        AttentionHistoryEntries::AffineK8V4 { codec, .. } => dimensions.extend(
            affine_dimensions(*codec)?
                .into_iter()
                .map(|(name, extent)| (name, Dim::from(extent))),
        ),
    }
    let dimensions = dimensions;
    let heads = widths.heads();
    // The sink rows: the operator's, or none (an injection attends nothing
    // and binds none).
    let sinks = match sinks {
        Some(sinks) => sinks.reshape(&[axis(form.sinks), heads.clone()]),
        None if !form.sinks || inject_only => GraphConstant::absent(graph, constants)?
            .tensor()
            .reshape(&[Dim::from(0), heads.clone()]),
        None => return Err("attention sink weights are absent".into()),
    };
    let input = |graph: &mut G, name: &str| match &history {
        AttentionHistoryEntries::Dense {
            decode: DenseDecodeEntries::Decode(kernel),
            ..
        } => graph.control_for(topology.controls, *kernel, name, &dimensions),
        AttentionHistoryEntries::Dense {
            decode: DenseDecodeEntries::Prefill(prefill),
            ..
        } => graph.control_for(
            topology.controls,
            *prefill,
            name,
            &prefill_dimensions(&dimensions, &words, &block_keys),
        ),
        AttentionHistoryEntries::AffineK8V4 { decode, .. } => {
            graph.control_for(topology.controls, *decode, name, &dimensions)
        }
    };
    // A class the prefill entry attends is a prefill class here, decode
    // rows among them (`dense_decodes`).
    let decode_rows = decodes(rows);
    let decode = decode_rows
        && !matches!(
            &history,
            AttentionHistoryEntries::Dense {
                decode: DenseDecodeEntries::Prefill(_),
                ..
            }
        );
    // The affine prefill of a class that lists its launch's history row
    // tiles sizes its per-call history storage by them, never by the store's
    // reservation; a class that lists none (and an injection-only route,
    // which attends to nothing) reads the history in place.
    let listed = match &history {
        AttentionHistoryEntries::AffineK8V4 {
            prefill_listed: Some(kernel),
            ..
        } if !decode && !inject_only && !numbers.history_tiles.is_zero() => Some(*kernel),
        _ => None,
    };
    let tile_dimensions = dimensions
        .iter()
        .cloned()
        .chain([
            ("L", axis(listed.is_some())),
            (
                "HT",
                listed.map_or(Dim::from(1), |_| numbers.history_tiles.clone()),
            ),
            ("B", words.clone()),
            ("BR", block_keys.clone()),
        ])
        .collect::<Vec<_>>();
    let history_tiles = listed
        .map(|kernel| {
            Ok::<_, GraphError>(HistoryTilesPort {
                port: graph.input_for(kernel, "history_tiles", &tile_dimensions)?,
            })
        })
        .transpose()?;
    let coordinates = input(graph, "coordinates")?;
    let read = (!inject_only)
        .then(|| {
            Ok::<_, GraphError>(AttentionReadControls {
                visible: input(graph, "visible")?,
                fresh: input(graph, "fresh")?,
                history_tiles,
            })
        })
        .transpose()?;
    let controls = AttentionControlPorts {
        coordinates,
        destinations: input(graph, "destinations")?,
        read,
    };
    let pairs = [numbers.widths.rotary_pairs.clone()];
    let rotary = &topology.rotary;
    let components = GraphConstant::table(graph, Element::i32(), &pairs, &rotary.components)?;
    let frequencies = GraphConstant::table(graph, Element::f32(), &pairs, &rotary.frequencies)?;
    let amplitudes = GraphConstant::table(graph, Element::f32(), &pairs, &rotary.amplitudes)?;
    // The unit norm row: the unweighted value norm, and (zero rows of it)
    // every absent head norm.
    let unit = GraphConstant::f32_filled(graph, &[Dim::from(1), width.clone()], 1.0)?;
    let absent_norm = unit.port().tensor().slice_leading(0, 0);
    // The rotary layout (RA, RT), output rotary (RO) and stored-key values
    // (VK), and for the dense entries the key rounding (KR) and the
    // absorbed latent read (AB), carried by the
    // extents of empty tables; the appends take no output rotary.
    let empty_table = controls.destinations.tensor().slice_leading(0, 0);
    let rounded = dense.then(|| axis(key_rounding.is_some()));
    let form_table = empty_table.reshape(
        &[
            Dim::from(0),
            axis(form.rotary_adjacent),
            axis(form.rotary_trailing),
            axis(form.output_rotary),
            axis(form.stored_key()),
        ]
        .into_iter()
        .chain(rounded.clone())
        .chain(dense.then(|| axis(form.is_absorbed())))
        .collect::<Vec<_>>(),
    );
    let append_form = empty_table.reshape(
        &[
            Dim::from(0),
            axis(form.rotary_adjacent),
            axis(form.rotary_trailing),
            axis(form.stored_key()),
        ]
        .into_iter()
        .chain(rounded)
        .collect::<Vec<_>>(),
    );
    // The dense entries' index-list source: each row's picks in the
    // source's pooled plane (made or bound here first), or (KS = TS = 0)
    // zero picks of zero rows.
    let picked = match picked {
        None => None,
        Some(picked) => Some(super::pooled::picked(
            graph,
            topology,
            numbers,
            picked,
            &controls.coordinates,
            [
                components.port().tensor(),
                frequencies.port().tensor(),
                amplitudes.port().tensor(),
            ],
            constants,
        )?),
    };
    let (picked, picked_state) = picked.unzip();
    let picked = picked.as_ref();
    // An absent plane `[0, KV, W]` in the activation element, which its
    // entry binds and reads nothing of.
    let empty_plane = |graph: &mut G, constants: &mut Vec<GraphConstant>| {
        GraphConstant::empty_port(
            graph,
            topology.activation,
            &[Dim::from(0), kv_heads.clone(), width.clone()],
            constants,
        )
    };
    // Only an attending graph reads the picks: a history injection's append
    // reads none, and a graph port no entry reads is unbound.
    let unpicked_plane = match picked {
        Some(_) => None,
        None if dense && !inject_only => Some(empty_plane(graph, constants)?),
        None => None,
    };
    // The affine entries read none.
    let picks = match (picked, &unpicked_plane) {
        (Some(picked), _) => Some((picked.rows.clone(), picked.plane.clone(), picked.slab_rows)),
        (None, Some(plane)) => Some((
            empty_table.reshape(&[rows.clone(), Dim::from(0)]),
            plane
                .tensor()
                .reshape(&[Dim::from(0), kv_heads.clone(), width.clone()]),
            Argument::from(0),
        )),
        (None, None) => None,
    };
    let key_rounding = RoundingArguments::of(key_rounding)?;
    let head_norm = |weight: &Option<WorkflowTensor>| match weight {
        Some(weight) => weight.reshape(&[Dim::from(1), width.clone()]),
        None => absent_norm.clone(),
    };
    let query_norm = head_norm(query_norm);
    // A Shared layer's unread key norm port takes its query norm.
    let key_norm = if form.is_fresh() {
        head_norm(key_norm)
    } else {
        query_norm.clone()
    };
    let value_norm = unit
        .port()
        .tensor()
        .slice_leading(0, u64::from(form.value_norm));
    let ProjectedRows {
        query,
        gate,
        key,
        value,
    } = projected.clone();
    // The state ports: the history codec's planes, in its descriptor order.
    let codec = match &history {
        AttentionHistoryEntries::Dense { codec, .. }
        | AttentionHistoryEntries::AffineK8V4 { codec, .. } => *codec,
    };
    let mut planes = codec
        .planes()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|descriptor| {
            graph.port(
                Element::from(descriptor.dtype),
                &[
                    numbers.history_rows.clone(),
                    kv_heads.clone(),
                    Dim::from(descriptor.row_elements as u64),
                ],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    // A history without a value plane (stored-key values) binds the dense
    // entries' `history_value` to zero rows of one, which they never touch.
    let mut absent_values = match codec.value {
        HistoryValue::Key => Some(empty_plane(graph, constants)?),
        HistoryValue::Own { .. } | HistoryValue::None => None,
    };
    let scale = topology.scale.f32();
    let gate_function = scalar::<i32>(&numbers.gate_function, "an attention gate code")?;
    let slab_rows = scalar::<u32>(&numbers.slab_rows, "slab rows")?;
    let mut destinations = controls.destinations.tensor().clone();
    // State-only entries bind no query, gate, visibility or selection values.
    macro_rules! append {
        ($kernel:expr, $module:ident, {$($extra:ident: $value:expr),*}, $($plane:ident),*) => {{
            let [$($plane),*] = entry_planes(&mut planes, absent_values.as_mut())?;
            let append_dimensions = dimensions.iter().cloned()
                .filter(|(name, _)| !AttentionShape::unread_by_appends(name)).collect::<Vec<_>>();
            graph.enqueue(*$kernel, &append_dimensions, $module::WorkflowArgs {
                key: (&key).into(), value: (&value).into(),
                key_norm: (&key_norm).into(), value_norm: (&value_norm).into(),
                rotary_components: components.port().tensor().into(),
                rotary_frequencies: frequencies.port().tensor().into(),
                rotary_amplitudes: amplitudes.port().tensor().into(),
                coordinates: controls.coordinates.tensor().into(),
                destinations: (&destinations).into(),
                $($extra: $value,)*
                $($plane: $plane.tensor_mut().into(),)*
                form: (&append_form).into(),
                epsilon: topology.head_epsilon.f32(), slab_rows,
            })?;
        }};
    }
    // The dense entries also take source 0's key rounding, and the attending
    // ones the index-list source's picks and plane.
    macro_rules! dense_append {
        ($kernel:expr) => {
            append!($kernel, attention_append_dense, {
                key_block: key_rounding.block,
                key_element: key_rounding.element,
                key_scale: key_rounding.scale,
                key_amax_floor: key_rounding.amax_floor
            }, history_key, history_value)
        };
    }
    // Priming only publishes context K/V, and a selected read's keys: the
    // indexer that picks among them runs only where the layer attends.
    // Select the state-only entry at graph formation to omit
    // attention/merge launches and their scratch buffers, even for a large
    // prompt chunk.
    if inject_only {
        let mut indexer = None;
        if let Some(selection) = &selection {
            let (selected, indexer_ports) = select(
                graph,
                selection,
                topology,
                numbers,
                &controls,
                decode,
                [
                    components.port().tensor(),
                    frequencies.port().tensor(),
                    amplitudes.port().tensor(),
                ],
                constants,
            )?;
            if selected.is_some() {
                return Err("a history injection scores a selected read".into());
            }
            indexer = Some(indexer_ports);
        }
        match &history {
            AttentionHistoryEntries::Dense { append, .. } => {
                dense_append!(append);
            }
            AttentionHistoryEntries::AffineK8V4 { append, .. } => {
                append!(
                    append,
                    attention_append_k8v4,
                    {},
                    history_key_codes,
                    history_key_coefficients,
                    history_value_codes,
                    history_value_coefficients
                );
            }
        }
        constants.extend([components, frequencies, amplitudes, unit]);
        return Ok((
            None,
            AttentionStatePorts {
                planes,
                indexer,
                picked: None,
            },
            controls,
        ));
    }
    // What the entry reads: the host's visible spans and fresh rows, or for
    // a selected read's decode rows the indexer's segments.
    let read = controls
        .read
        .as_ref()
        .expect("attending attention graph has read controls");
    let mut visible = read
        .visible
        .tensor()
        .reshape(&[rows.clone(), numbers.segments.clone(), Dim::from(2)]);
    let fresh = read.fresh.tensor().reshape(&[rows.clone(), Dim::from(2)]);
    let mut attend_dimensions = dimensions.clone();
    // Every entry takes the same arguments but its history planes.
    macro_rules! mix {
        ($kernel:expr, $module:ident, $($plane:ident),*) => {
            mix!(@enqueue $kernel, $module, &attend_dimensions, {}, $($plane),*)
        };
        (@enqueue $kernel:expr, $module:ident, $dimensions:expr, {$($extra:ident: $value:expr),*},
            $($plane:ident),*) => {
            mix!(@phase $kernel, $module, $dimensions, {$($extra: $value),*}, $($plane),*).value
        };
        (@phase $kernel:expr, $module:ident, $dimensions:expr, {$($extra:ident: $value:expr),*},
            $($plane:ident),*) => {{
            let [$($plane),*] = entry_planes(&mut planes, absent_values.as_mut())?;
            graph
                .enqueue(
                    *$kernel,
                    $dimensions,
                    $module::WorkflowArgs {
                        query: (&query).into(),
                        gate: (&gate).into(),
                        key: (&key).into(),
                        value: (&value).into(),
                        query_norm: (&query_norm).into(),
                        key_norm: (&key_norm).into(),
                        value_norm: (&value_norm).into(),
                        sinks: (&sinks).into(),
                        rotary_components: components.port().tensor().into(),
                        rotary_frequencies: frequencies.port().tensor().into(),
                        rotary_amplitudes: amplitudes.port().tensor().into(),
                        coordinates: controls.coordinates.tensor().into(),
                        visible: (&visible).into(),
                        fresh: (&fresh).into(),
                        destinations: (&destinations).into(),
                        $($extra: $value,)*
                        $($plane: $plane.tensor_mut().into(),)*
                        form: (&form_table).into(),
                        epsilon: topology.head_epsilon.f32(),
                        scale,
                        gate_function,
                        slab_rows,
                    },
                )?

        }};
    }
    macro_rules! dense_mix {
        ($kernel:expr, $module:ident, $dimensions:expr $(, $extra:ident: $value:expr)*) => {{
            let (picked_rows, picked_plane, picked_slab_rows) = picks
                .as_ref()
                .ok_or("a dense attention's index-list arguments are absent")?;
            mix!(@enqueue $kernel, $module, $dimensions, {
                selected_rows: picked_rows.into(),
                selected: picked_plane.into(),
                selected_slab_rows: *picked_slab_rows,
                key_block: key_rounding.block,
                key_element: key_rounding.element,
                key_scale: key_rounding.scale,
                key_amax_floor: key_rounding.amax_floor
                $(, $extra: $value)*
            }, history_key, history_value)
        }};
    }
    // A selected read's indexer appends and selects first.
    let mut indexer = None;
    let selected = match &selection {
        None => None,
        Some(selection) => {
            let (selected, indexer_ports) = select(
                graph,
                selection,
                topology,
                numbers,
                &controls,
                decode,
                [
                    components.port().tensor(),
                    frequencies.port().tensor(),
                    amplitudes.port().tensor(),
                ],
                constants,
            )?;
            indexer = Some(indexer_ports);
            selected
        }
    };
    // A selected decode row reads its units' segments of its visible spans
    // (its launch's earlier rows among them, `cached_causal_domain`) and its
    // own row fresh, as a dense decode row does.
    if let Some(Selected { segments: Some(read_segments), .. }) = &selected {
        let top = &selection
            .as_ref()
            .ok_or("selected segments without a selection")?
            .selection
            .read
            .source
            .top;
        let segments = selected_segments(top.clone(), numbers.segments.clone());
        visible = read_segments.reshape(&[rows.clone(), segments.clone(), Dim::from(2)]);
        for (name, value) in &mut attend_dimensions {
            if *name == "R" {
                *value = segments.clone();
            }
        }
    }
    // Earlier causal rows are already in the visible cache spans (a
    // compressed history's, or a selected read's). Decode's fused append
    // cannot publish rows other threadgroups read in the same launch, so
    // publish once first and suppress its later append. Prefill already has
    // an ordered prepare launch and needs no additional phase.
    if topology.causal_cache && decode && rows.greater_than(1) && form.is_fresh() {
        let published = match &history {
            AttentionHistoryEntries::AffineK8V4 { append, .. } => {
                append!(
                    append,
                    attention_append_k8v4,
                    {},
                    history_key_codes,
                    history_key_coefficients,
                    history_value_codes,
                    history_value_coefficients
                );
                true
            }
            // A selected read's domain is causally cached whether or not
            // its class scores.
            AttentionHistoryEntries::Dense { append, .. } if selection.is_some() => {
                dense_append!(append);
                true
            }
            AttentionHistoryEntries::Dense { .. } => false,
        };
        if published {
            let disabled = GraphConstant::row_filled(graph, rows.clone(), -1)?;
            destinations = disabled.port().tensor().clone();
            constants.push(disabled);
        }
    }
    // A prefill's WorkList block keys (`BR`), carried by an extent alone.
    let block_slots = controls
        .destinations
        .tensor()
        .slice_leading(0, 0)
        .reshape(&[Dim::from(0), block_keys.clone()]);
    let attended = match &history {
        AttentionHistoryEntries::Dense {
            decode: DenseDecodeEntries::Decode(kernel),
            ..
        } if decode => {
            dense_mix!(kernel, attention_decode, &attend_dimensions)
        }
        AttentionHistoryEntries::Dense {
            decode: dense_decode,
            prefill,
            ..
        } => {
            let kernel = match (&selected_prefill, dense_decode) {
                (Some(SelectedPrefillEntries::Dense(selected)), _) => selected,
                (None, DenseDecodeEntries::Prefill(decode_prefill)) if decode_rows => {
                    decode_prefill
                }
                (None, _) => prefill,
                (Some(_), _) => return Err("a selected prefill of another codec".into()),
            };
            let bitmap = prefill_selection(graph, rows, &selected, row_class, constants)?;
            dense_mix!(kernel, attention_prefill,
                &prefill_dimensions(&dimensions, &words, &block_keys),
                selection: (&bitmap).into(),
                block_slots: (&block_slots).into())
        }
        AttentionHistoryEntries::AffineK8V4 { decode: kernel, .. } if decode => mix!(
            kernel,
            attention_decode_k8v4,
            history_key_codes,
            history_key_coefficients,
            history_value_codes,
            history_value_coefficients
        ),
        AttentionHistoryEntries::AffineK8V4 { prefill, .. } => {
            // No list: zero rows of an I32 table.
            let unlisted = controls
                .destinations
                .tensor()
                .slice_leading(0, 0)
                .reshape(&[Dim::from(0), Dim::from(1)]);
            let kernel = match &selected_prefill {
                Some(SelectedPrefillEntries::AffineK8V4(selected)) => selected,
                None => listed.as_ref().unwrap_or(prefill),
                Some(_) => return Err("a selected prefill of another codec".into()),
            };
            let bitmap = prefill_selection(graph, rows, &selected, row_class, constants)?;
            mix!(
            @enqueue kernel,
            attention_prefill_k8v4,
            &tile_dimensions,
            {
                selection: (&bitmap).into(),
                block_slots: (&block_slots).into(),
                history_tiles: match &read.history_tiles {
                    Some(tiles) => tiles.port.tensor().into(),
                    None => (&unlisted).into(),
                }
            },
            history_key_codes,
            history_key_coefficients,
            history_value_codes,
            history_value_coefficients
            )
        }
    };
    constants.extend([components, frequencies, amplitudes, unit]);
    Ok((
        Some(attended),
        AttentionStatePorts {
            planes,
            indexer,
            picked: picked_state,
        },
        controls,
    ))
}

/// An entry's history plane arguments: the state ports, then the zero-row
/// value plane of a history whose values are its keys (`absent`).
fn entry_planes<'p, const N: usize>(
    planes: &'p mut [NativePort],
    absent: Option<&'p mut NativePort>,
) -> Result<[&'p mut NativePort; N], GraphError> {
    planes
        .iter_mut()
        .chain(absent)
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "attention history planes disagree with the entry".into())
}

/// A prefill's WorkList bitmap: a scoring selected read's (`unit_reads`),
/// or none (`no_selection`, every visible row).
fn prefill_selection<G: GraphDraft>(
    graph: &mut G,
    rows: &Dim,
    selected: &Option<Selected>,
    row_class: Option<&'static str>,
    constants: &mut Vec<GraphConstant>,
) -> Result<WorkflowTensor, GraphError> {
    match selected {
        Some(selected) => Ok(selected.bitmap.clone()),
        None => no_selection(graph, rows, row_class, constants),
    }
}

/// The prefill entries' dimensions: the fused entries', `B`, the words of a
/// row's WorkList bitmap (0 in the Visible form), and `BR`, the keys of its
/// blocks (1 in the Visible form).
fn prefill_dimensions(
    dimensions: &[(&'static str, Dim)],
    words: &Dim,
    block: &Dim,
) -> Vec<(&'static str, Dim)> {
    dimensions
        .iter()
        .cloned()
        .chain([("B", words.clone()), ("BR", block.clone())])
        .collect()
}

/// A Visible prefill's absent WorkList bitmap: `[rows, 0]`, its rows the
/// class's.
fn no_selection<G: GraphDraft>(
    graph: &mut G,
    rows: &Dim,
    row_class: Option<&'static str>,
    constants: &mut Vec<GraphConstant>,
) -> Result<WorkflowTensor, GraphError> {
    let selection = GraphConstant::row_zeros(graph, Element::u32(), rows.clone(), 0, row_class)?;
    let tensor = selection.port().tensor().clone();
    constants.push(selection);
    Ok(tensor)
}

/// Per rotary pair, the coordinate axis that drives it: the interleaved
/// multi-axis layout assigns pairs to axes round-robin until each axis's
/// section is exhausted; a table rotates by the text position, axis 0.
pub(crate) fn rotary_components(rotary: &Rotary) -> Result<Vec<i32>, String> {
    let (width, sections, axis_pattern) = match rotary {
        Rotary::None => return Ok(Vec::new()),
        Rotary::Table { pairs, .. } => return Ok(vec![0; pairs.len()]),
        Rotary::Interleaved {
            width,
            sections,
            axis_pattern,
            ..
        } => (width, sections, axis_pattern),
    };
    let pairs = usize::try_from(width / 2).map_err(|_| "rotary width exceeds host domain")?;
    let first = *axis_pattern.first().ok_or("rotary axis pattern is empty")?;
    if axis_pattern.len() == 1 {
        return Ok(vec![i32::from(first); pairs]);
    }
    if axis_pattern.len() != 3
        || sections.len() < 3
        || sections[3..].iter().any(|section| *section != 0)
    {
        return Err("unsupported rotary axis and section mapping".into());
    }
    let axis_one_end = sections[1]
        .checked_mul(3)
        .ok_or("rotary section exceeds host domain")?;
    let axis_two_end = sections[2]
        .checked_mul(3)
        .ok_or("rotary section exceeds host domain")?;
    Ok((0..pairs)
        .map(|index| {
            let index = index as u64;
            let axis = if index % 3 == 1 && index < axis_one_end {
                1
            } else if index % 3 == 2 && index < axis_two_end {
                2
            } else {
                0
            };
            i32::from(axis_pattern[axis])
        })
        .collect())
}

/// Per rotary pair `p` of `P`, its angular frequency: `base^(-2p / 2P)` for
/// the interleaved layout, evaluated in f64 and rounded once to f32, or the
/// table's own frequency.
pub(crate) fn rotary_frequencies(rotary: &Rotary) -> Vec<f32> {
    match rotary {
        Rotary::None => Vec::new(),
        Rotary::Interleaved { width, base, .. } => {
            let pairs = width / 2;
            (0..pairs)
                .map(|pair| base.powf(-((2 * pair) as f64) / (2 * pairs) as f64) as f32)
                .collect()
        }
        Rotary::Table { pairs, .. } => pairs.iter().map(|pair| pair.frequency as f32).collect(),
    }
}

/// Per rotary pair, the amplitude its rotation is scaled by (1 is exact
/// identity scaling).
pub(crate) fn rotary_amplitudes(rotary: &Rotary) -> Vec<f32> {
    match rotary {
        Rotary::None => Vec::new(),
        Rotary::Interleaved { width, .. } => vec![1.0; (width / 2) as usize],
        Rotary::Table { pairs, .. } => pairs.iter().map(|pair| pair.amplitude as f32).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planning::{AttentionForm, AttentionWidths, GateSpec, KeyValueSpec, ValueSpec};

    /// A launch takes the class that lists its history row tiles exactly
    /// when its rows see at most that class's tiles: one request with a long
    /// history does, however many rows and spans it has; several requests
    /// whose histories together exceed it take the class that lists none.
    #[test]
    fn a_launch_lists_its_history_tiles_when_they_fit_one_request() {
        let visible = |spans: &[[i32; 2]]| {
            spans
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let listed = |bytes: Vec<u8>| {
            bytes
                .chunks_exact(4)
                .map(|tile| i32::from_le_bytes([tile[0], tile[1], tile[2], tile[3]]))
                .collect::<Vec<_>>()
        };
        // One request of 64 rows over a 16,384-row history in two spans, the
        // second at a lower address: 64 tiles of 256 rows.
        let request = [[40_960, 49_152], [8_192, 16_384]];
        let rows = std::iter::repeat(request)
            .take(64)
            .flatten()
            .collect::<Vec<_>>();
        let tiles = listed(history_tile_bytes(&visible(&rows), 66).unwrap());
        assert_eq!(tiles[..64], (32..64).chain(160..192).collect::<Vec<_>>());
        assert_eq!(tiles[64..], [-1, -1]);
        // A second request with as long a history of its own no longer fits.
        let both = rows
            .iter()
            .copied()
            .chain([[65_536, 73_728], [90_112, 98_304]])
            .collect::<Vec<_>>();
        assert_eq!(history_tile_bytes(&visible(&both), 66), None);
        // A fork sharing the first request's rows adds no tile.
        let fork = rows
            .iter()
            .copied()
            .chain([[40_960, 45_056], [0, 0]])
            .collect::<Vec<_>>();
        assert_eq!(
            listed(history_tile_bytes(&visible(&fork), 66).unwrap()),
            tiles
        );
        // Rows that see no history list nothing.
        assert_eq!(
            listed(history_tile_bytes(&visible(&[[0, 0]]), 2).unwrap()),
            [-1, -1]
        );
    }

    #[test]
    fn rotary_components_interleave_axes_with_section_cutoffs() {
        let rotary = Rotary::Interleaved {
            width: 14,
            base: 10_000.0,
            sections: vec![4, 2, 1, 0],
            axis_pattern: vec![0, 1, 2],
        };
        assert_eq!(rotary_components(&rotary).unwrap(), [0, 1, 2, 0, 1, 0, 0]);
    }

    #[test]
    fn rotary_frequencies_fall_geometrically_from_one() {
        let rotary = Rotary::Interleaved {
            width: 4,
            base: 10_000.0,
            sections: vec![2, 0, 0, 0],
            axis_pattern: vec![0],
        };
        assert_eq!(rotary_frequencies(&rotary), [1.0, 0.01]);
        assert_eq!(rotary_amplitudes(&rotary), [1.0, 1.0]);
    }

    /// DeepSeek V4.1's window attention (64 query heads over one 512-wide
    /// head, adjacent trailing pairs turned back) decodes through the decode
    /// entry on Metal (its matrix form, in head groups of 8), on Vulkan (its
    /// matrix form, the queries from its prepare launch) and on CPU, as a
    /// Qwen-shaped layer does; CUDA's decode entry holds none of the new
    /// forms, so CUDA attends them with its prefill entry. The routing reads
    /// the shape alone: a selected read's rows go where an unselected
    /// read's do.
    #[test]
    fn decode_classes_attend_with_the_decode_entry_where_it_holds_them() {
        use seismic::{BackendName, DType};
        let binding = |shape: AttentionShape| {
            AttentionBinding {
                shape: AttentionShape {
                    form: AttentionForm {
                        gate: shape.form.gate.try_map(|()| Ok::<_, ()>(Element::bf16())).unwrap(),
                        key_value: shape
                            .form
                            .key_value
                            .try_map(|()| Ok::<_, ()>(Element::bf16()), |()| Ok::<_, ()>(Element::bf16()))
                            .unwrap(),
                        query_norm: shape.form.query_norm,
                        key_norm: shape.form.key_norm,
                        value_norm: shape.form.value_norm,
                        sinks: shape.form.sinks,
                        rotary_adjacent: shape.form.rotary_adjacent,
                        rotary_trailing: shape.form.rotary_trailing,
                        output_rotary: shape.form.output_rotary,
                        absorbed: None,
                    },
                    widths: shape.widths,
                    key_rounding: shape.key_rounding,
                },
                key_value_only: false,
                norm: Element::bf16(),
                query: Element::bf16(),
                low_rank_query: None,
                output: Element::bf16(),
                activation: Element::bf16(),
                history: match shape.form.stored_key() {
                    true => CodecSpec::stored_key(DType::BF16, shape.widths.width as usize),
                    false => CodecSpec::dense(
                        DType::BF16,
                        shape.widths.width as usize,
                        shape.widths.width as usize,
                    ),
                },
                tail: SublayerTail::Residual,
                selection: None,
                grouped: None,
                picked: None,
            }
        };
        let v41 = AttentionShape {
            form: AttentionForm {
                gate: GateSpec::None,
                key_value: KeyValueSpec::Fresh {
                    key: (),
                    value: ValueSpec::StoredKey,
                },
                query_norm: false,
                key_norm: true,
                value_norm: false,
                sinks: true,
                rotary_adjacent: true,
                rotary_trailing: true,
                output_rotary: true,
                absorbed: None,
            },
            key_rounding: crate::RoundingArgs::NONE,
            widths: AttentionWidths {
                hidden: 4096,
                kv_heads: 1,
                group: 64,
                rotary_pairs: 32,
                width: 512,
                picks: 0,
            },
        };
        let qwen = AttentionShape {
            form: AttentionForm {
                gate: GateSpec::Interleaved,
                key_value: KeyValueSpec::Fresh {
                    key: (),
                    value: ValueSpec::Projected(()),
                },
                query_norm: true,
                sinks: false,
                rotary_adjacent: false,
                rotary_trailing: false,
                output_rotary: false,
                ..v41.form
            },
            widths: AttentionWidths {
                kv_heads: 2,
                group: 8,
                width: 256,
                ..v41.widths
            },
            key_rounding: crate::RoundingArgs::NONE,
        };
        let decodes = |backend, shape| dense_decodes(backend, binding(shape)).unwrap();
        assert!(decodes(BackendName::Metal, v41));
        assert!(decodes(BackendName::Cpu, v41));
        assert!(decodes(BackendName::Metal, qwen));
        assert!(decodes(BackendName::Vulkan, v41));
        assert!(!decodes(BackendName::Cuda, v41));
        assert!(decodes(BackendName::Cuda, qwen));
        // Each new form alone (a key norm without a query norm, adjacent
        // pairs, trailing pairs, the output rotary, V4.1's own): CUDA's
        // decode entry declares none of them and its prefill entry all of
        // them; Vulkan's and CPU's entries declare all of them.
        let holds = |backend, shape: AttentionShape| {
            // The prefill's block and listing axes at one block, unlisted.
            let mut statics = shape.dense_mix_statics();
            statics.extend([("B", 1), ("L", 0)]);
            let held = |result: Result<_, crate::native::CatalogFailure>| match result {
                Ok(_) => true,
                Err(crate::native::CatalogFailure::KernelDomain { .. }) => false,
                Err(error) => panic!("{error:?}"),
            };
            let bindings = format!("{shape:?}");
            let decode = held(crate::native::entry_statics(
                backend,
                &attention_decode::native_entry_with(attention_decode::Elements {
                    A: Element::bf16(),
                }),
                &bindings,
                &statics,
            ));
            let prefill = held(crate::native::entry_statics(
                backend,
                &attention_prefill::native_entry_with(attention_prefill::Elements {
                    A: Element::bf16(),
                }),
                &bindings,
                &statics,
            ));
            (decode, prefill)
        };
        let forms = [
            AttentionShape { form: AttentionForm { query_norm: false, ..qwen.form }, ..qwen },
            AttentionShape { form: AttentionForm { rotary_adjacent: true, ..qwen.form }, ..qwen },
            AttentionShape { form: AttentionForm { rotary_trailing: true, ..qwen.form }, ..qwen },
            AttentionShape {
                form: AttentionForm { gate: GateSpec::None, output_rotary: true, ..qwen.form },
                ..qwen
            },
            v41,
        ];
        for backend in [BackendName::Cuda, BackendName::Vulkan] {
            assert_eq!(holds(backend, qwen), (true, true), "{backend:?}");
        }
        for form in forms {
            assert_eq!(holds(BackendName::Cuda, form), (false, true), "{form:?}");
            assert_eq!(holds(BackendName::Vulkan, form), (true, true), "{form:?}");
            assert_eq!(holds(BackendName::Cpu, form), (true, true), "{form:?}");
        }
    }
}
