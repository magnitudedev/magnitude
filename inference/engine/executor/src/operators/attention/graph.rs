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
use super::split::{AttentionBlock, AttentionTopology, PickedPart, Plane, SelectedRead, SelectionReals, both, picked as picked_part};
use crate::planning::{RoundingArguments, axis};
use crate::programs::native_target_graph::{PackedWeight, PlannedWeights, WeightOperands, WeightPort};
use crate::{AttentionBinding, AttentionShape, AttentionValues, SublayerTail};
use crate::{
    native::{
        AttentionHistoryKernels, AttentionKernels, DenseDecodeKernels, PoolKernel,
        SelectedPrefillKernel,
    },
    programs::native_constants::GraphConstant,
};
use magnitude_family_contracts::{Rotary, WeightKind, WeightRole, WeightScope};
use magnitude_kernels::{
    attention_append_dense, attention_append_k8v4, attention_decode, attention_decode_k8v4,
    attention_output, attention_prefill, attention_prefill_k8v4, attention_project, candidate_rows,
    pool_rows, project_rows, unit_reads, widen_rows,
};
use magnitude_state::{CodecSpec, HistoryEncoding, HistoryValue};
use seismic::{
    Argument, Dim, Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor,
    WorkflowTensorRef, WorkflowTensorView,
};

pub(crate) struct AttentionGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, attention_project::Entry>,
    pub history: AttentionHistoryEntries<'a, G>,
    pub output: TailEntries<'a, G, attention_output::Entry>,
    /// A grouped output's first product (`project_rows` over its groups).
    pub grouped: Option<G::Binding<'a, project_rows::Entry>>,
    /// An absorbed query's per-head products (`project_rows` over its heads).
    pub absorbed: Option<G::Binding<'a, project_rows::Entry>>,
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
    pub project: G::Binding<'a, attention_project::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for LowRankQueryEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for LowRankQueryEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A selected read's entries ([`select`]): its keys' projection, widening
/// and pooling, its indexer, its reads and the codec's WorkList prefill.
pub(crate) struct SelectionGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, attention_project::Entry>,
    pub widen: G::Binding<'a, widen_rows::Entry>,
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
        verify: Option<G::Binding<'a, attention_decode_k8v4::Entry>>,
        verify_four: Option<G::Binding<'a, attention_decode_k8v4::Entry>>,
        verify_eight: Option<G::Binding<'a, attention_decode_k8v4::Entry>>,
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
    Decode {
        single: G::Binding<'a, attention_decode::Entry>,
        verify: Option<G::Binding<'a, attention_decode::Entry>>,
    },
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
/// their matrix forms for it (precision.md, "Absorbed latent attention"), so
/// the catalog decides. Metal's vector decode holds no group as wide as
/// GLM's (64 queries of 512 exceed its threadgroup memory) and CUDA's decode
/// holds no stored key, so both send it to prefill; Vulkan's vector decode
/// holds it.
pub(crate) fn dense_decodes(
    backend: seismic::BackendName,
    binding: AttentionBinding,
) -> Result<bool, crate::native::CatalogFailure> {
    if !binding.shape.extends_base_form() {
        return Ok(true);
    }
    let entry = attention_decode::native_entry_with(attention_decode::Elements {
        A: binding.activation,
    });
    match crate::native::entry_statics(
        backend,
        &entry,
        &format!("{:?}", binding.shape),
        &binding.shape.dense_mix_statics(),
    ) {
        Ok(_) => Ok(true),
        Err(crate::native::CatalogFailure::KernelDomain { .. }) => Ok(false),
        Err(error) => Err(error),
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
                    DenseDecodeKernels::Decode { single, verify } => DenseDecodeEntries::Decode {
                        single,
                        verify: verify.as_ref(),
                    },
                    DenseDecodeKernels::Prefill(kernel) => DenseDecodeEntries::Prefill(kernel),
                },
                prefill,
            },
            AttentionHistoryKernels::AffineK8V4 {
                append,
                decode,
                verify,
                verify_four,
                verify_eight,
                prefill,
                prefill_listed,
            } => AttentionHistoryEntries::AffineK8V4 {
                codec: kernels.codec,
                append,
                decode,
                verify: verify.as_ref(),
                verify_four: verify_four.as_ref(),
                verify_eight: verify_eight.as_ref(),
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
                    project: &selection.keys.project,
                    widen: &selection.keys.widen,
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
                    widen: &rows.widen,
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
            query_project: &indexer.query_project,
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

/// Entry element assignments from the same program binding used by native
/// preparation. The checked graph obtains all shapes from the entries.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedAttentionEntries {
    project: [(&'static str, Element); 7],
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
    low_rank_query: Option<([(&'static str, Element); 1], [(&'static str, Element); 7])>,
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

    fn entries(&self) -> PickedEntries<'_, NativeGraphMetadata> {
        PickedEntries {
            rows: self.elements.rows.as_ref().map(|rows| PooledEntries {
                project: &rows.project[..],
                widen: &self.activation[..],
                pool: match rows.pooling {
                    Pooling::Identity => PoolEntry::Identity(&self.activation[..]),
                    Pooling::Grouped => PoolEntry::Grouped(&self.pool[..]),
                },
                key_project: &rows.key_project[..],
                keys: &self.activation[..],
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

    fn entries(&self) -> IndexerEntries<'_, NativeGraphMetadata> {
        IndexerEntries {
            query_project: &self.elements.query_project[..],
            weights_project: self.elements.weights_project.as_ref().map(|elements| &elements[..]),
            score: &self.activation[..],
            topk: &[],
            candidates: self.elements.produces().then_some(CandidateEntries { blocks: &[], rows: &[] }),
        }
    }
}

/// A selected read's entry elements: its keys' `u | g` projection
/// (`pooled::tokens_project`), the activation element `widen_rows` reads,
/// `pool_rows`' (its keys stored in F32), and its indexer's (`unit_reads`
/// and `candidate_rows` read none).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CheckedSelectionEntries {
    project: [(&'static str, Element); 7],
    activation: [(&'static str, Element); 1],
    pool: [(&'static str, Element); 2],
    indexer: CheckedIndexerEntries,
}

impl CheckedSelectionEntries {
    fn new(binding: AttentionBinding, selection: crate::SelectionBinding) -> Self {
        let activation = binding.activation;
        Self {
            project: super::pooled::tokens_project(
                activation,
                binding.norm,
                selection.keys.projection,
                selection.keys.gate,
            ),
            activation: [("A", activation)],
            pool: [("A", activation), ("Y", Element::f32())],
            indexer: CheckedIndexerEntries::new(IndexerElements::of(binding, selection.indexer), activation),
        }
    }

    fn entries<'a>(&'a self, prefill: SelectedPrefillEntries<'a, NativeGraphMetadata>) -> SelectionGraphEntries<'a, NativeGraphMetadata> {
        SelectionGraphEntries {
            project: &self.project[..],
            widen: &self.activation[..],
            pool: &self.pool[..],
            indexer: self.indexer.entries(),
            reads: &[],
            rows: &[],
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
                        ("A", binding.activation),
                        ("Y", binding.activation),
                    ],
                )
            }),
            selection: binding
                .selection
                .map(|selection| CheckedSelectionEntries::new(binding, selection)),
            project: [
                ("NW", binding.norm),
                ("QW", binding.query_segment()),
                ("GW", binding.gate),
                ("KW", binding.key),
                ("VW", binding.value),
                ("A", binding.activation),
                ("Y", binding.activation),
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
            absorbed: binding.absorbed.map(|absorbed| {
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

    pub(crate) fn entries(&self) -> Result<AttentionGraphEntries<'_, NativeGraphMetadata>, String> {
        let history = match self.history.encoding() {
            HistoryEncoding::Dense => AttentionHistoryEntries::Dense {
                codec: self.history,
                append: &self.mix[..],
                decode: if self.decodes {
                    DenseDecodeEntries::Decode {
                        single: &self.mix[..],
                        verify: None,
                    }
                } else {
                    DenseDecodeEntries::Prefill(&self.mix[..])
                },
                prefill: &self.mix[..],
            },
            HistoryEncoding::Affine => AttentionHistoryEntries::AffineK8V4 {
                codec: self.history,
                append: &self.mix[..],
                decode: &self.mix[..],
                verify: None,
                verify_four: None,
                verify_eight: None,
                prefill: &self.mix[..],
                prefill_listed: self.lists.then_some(&self.mix[..]),
            },
            HistoryEncoding::Unsupported => {
                return Err("rotated K4/V4 has no native attention entry".into());
            }
        };
        Ok(AttentionGraphEntries {
            project: &self.project,
            history,
            output: match &self.post_norm {
                None => TailEntries::Residual(&self.output[..]),
                Some(post_norm) => TailEntries::PostNorm(post_norm.entries()),
            },
            grouped: self.grouped.as_ref().map(|grouped| &grouped[..]),
            absorbed: self.absorbed.as_ref().map(|absorbed| &absorbed[..]),
            selection: self.selection.as_ref().map(|selection| {
                selection.entries(match self.history.encoding() {
                    HistoryEncoding::Affine => SelectedPrefillEntries::AffineK8V4(&self.mix[..]),
                    _ => SelectedPrefillEntries::Dense(&self.mix[..]),
                })
            }),
            low_rank_query: self
                .low_rank_query
                .as_ref()
                .map(|(widen, project)| LowRankQueryEntries {
                    widen: &widen[..],
                    project: &project[..],
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
pub(crate) fn form_statics(
    shape: &AttentionShape,
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
pub(crate) fn append_form_statics(
    shape: &AttentionShape,
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

/// Row classes up to this size attend with the decode entry; larger classes
/// use the prefill entry.
pub(crate) const DECODE_ROWS: u64 = 8;

/// Whether a `rows`-row class attends with the decode entry.
pub(crate) fn decodes(rows: impl Into<Dim>) -> bool {
    rows.into().at_most(DECODE_ROWS)
}

/// The block's weight tensors, as ports of the graph being built. The
/// operator's form decides which are present.
#[derive(Clone)]
pub(crate) struct AttentionWeights {
    /// The input norm; absent for a mix site's partials.
    pub input_norm: Option<WorkflowTensor>,
    /// The projection's query segment: the query projection (with
    /// interleaved gate rows when gated so), or a low-rank query's `down`.
    pub query: WorkflowTensor,
    /// A low-rank query's `qr` norm and `up`, where the block attends.
    pub low_rank_query: Option<LowRankQueryWeights>,
    /// An absorbed query's per-head key projections, where the block
    /// attends.
    pub absorbed: Option<WorkflowTensor>,
    pub gate: Option<WorkflowTensor>,
    pub key: Option<WorkflowTensor>,
    pub value: Option<WorkflowTensor>,
    pub query_norm: Option<WorkflowTensor>,
    pub key_norm: Option<WorkflowTensor>,
    /// The per-head sink logits (`[heads]`), when the operator has sinks.
    pub sinks: Option<WorkflowTensor>,
    pub output: Option<WorkflowTensor>,
    /// A grouped output's first projection (`a`), when it is grouped.
    pub grouped: Option<WorkflowTensor>,
    /// The sublayer's post-norm weight, when its tail is a post-norm.
    pub post_norm: Option<WorkflowTensor>,
    /// The projections' scales and packing bounds.
    pub operands: AttentionOperands,
    /// A selected read's weights.
    pub selection: Option<SelectionWeights>,
    /// An indexing index-list source's weights.
    pub picked: Option<PickedWeights>,
}

/// The scales and packing bounds of an attention block's projections, in
/// the order of `attention_project`'s and `attention_output`'s ports.
#[derive(Clone)]
pub(crate) struct AttentionOperands {
    /// The query segment's weight's: the query's, or a low-rank query's
    /// `down`.
    pub query: WeightOperands,
    /// A low-rank query's `up`, where the block attends.
    pub low_rank_query: Option<WeightOperands>,
    pub gate: WeightOperands,
    pub key: WeightOperands,
    pub value: WeightOperands,
    pub output: WeightOperands,
    pub grouped: WeightOperands,
    pub absorbed: WeightOperands,
    /// A selected read's projections'.
    pub selection: Option<SelectionOperands>,
    /// An indexing index-list source's projections', where the block
    /// attends.
    pub picked: Option<PickedOperands>,
}

/// An indexing index-list source's projections' operands: an own source's
/// rows' and its indexer's.
#[derive(Clone)]
pub(crate) struct PickedOperands {
    pub rows: Option<PooledOperands>,
    pub indexer: IndexerOperands,
}

/// An indexer's projections' operands: its query's and its head weights'.
#[derive(Clone)]
pub(crate) struct IndexerOperands {
    pub query: WeightOperands,
    pub head_weights: Option<WeightOperands>,
}

/// A selected read's projections' operands: its keys' `u` and (gated) `g`
/// projections', and its indexer's where its class scores.
#[derive(Clone)]
pub(crate) struct SelectionOperands {
    pub projection: WeightOperands,
    pub gate: Option<WeightOperands>,
    pub indexer: Option<IndexerOperands>,
}

/// An own source's rows' projections' operands: its `u` and `g`
/// projections' and its index keys'.
#[derive(Clone)]
pub(crate) struct PooledOperands {
    pub projection: WeightOperands,
    pub gate: Option<WeightOperands>,
    pub key: WeightOperands,
}

impl AttentionOperands {
    /// The operands of the block's weights at `scope` its graph reads
    /// (`attention_weights`): the query, gate and output's only where it
    /// attends (`attends`), and none of an absent segment's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn planned<G: GraphDraft>(
        graph: &mut G,
        weights: &(impl PlannedWeights + ?Sized),
        scope: WeightScope,
        topology: &AttentionTopology,
        numbers: &AttentionBlock,
        attends: bool,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        let mut operands = |graph: &mut G, read: bool, kind| match read {
            true => {
                WeightOperands::planned(graph, weights, WeightRole { scope, kind }, ports, constants)
            }
            false => WeightOperands::absent(graph, constants),
        };
        let form = &topology.form;
        // An indexer's query and (projected) head weights.
        macro_rules! indexer {
            ($head_weights:expr) => {
                IndexerOperands {
                    query: operands(graph, true, WeightKind::IndexerQuery)?,
                    head_weights: match $head_weights {
                        true => Some(operands(graph, true, WeightKind::IndexerHeadWeights)?),
                        false => None,
                    },
                }
            };
        }
        // A selected read's keys are appended by every graph of the block,
        // its indexer run by those that attend and score.
        let selection = match &numbers.selection {
            None => None,
            Some(read) => Some(SelectionOperands {
                projection: operands(graph, true, WeightKind::IndexerKey)?,
                gate: match read.keys.gated {
                    true => Some(operands(graph, true, WeightKind::IndexerKeyGate)?),
                    false => None,
                },
                indexer: match attends && read.scored {
                    true => Some(indexer!(read.source.head_weights)),
                    false => None,
                },
            }),
        };
        let picked = match picked_part(topology, numbers) {
            Some(PickedPart::Indexed { source: index, .. }) if attends => Some(PickedOperands {
                rows: match &index.own {
                    Some(own) => Some(PooledOperands {
                        projection: operands(graph, true, WeightKind::PooledProjection)?,
                        gate: match own.gated {
                            true => Some(operands(graph, true, WeightKind::PooledGate)?),
                            false => None,
                        },
                        key: operands(graph, true, WeightKind::IndexerKey)?,
                    }),
                    None => None,
                },
                indexer: indexer!(index.head_weights),
            }),
            _ => None,
        };
        Ok(Self {
            picked,
            selection,
            query: operands(graph, attends, topology.query_segment_kind)?,
            low_rank_query: match attends && numbers.low_rank_query.is_some() {
                true => Some(operands(graph, true, topology.query_kind)?),
                false => None,
            },
            gate: operands(
                graph,
                attends && form.gated_separately(),
                WeightKind::AttentionGate,
            )?,
            key: operands(graph, form.fresh, WeightKind::Key)?,
            value: operands(graph, form.projects_values(), WeightKind::Value)?,
            output: operands(graph, attends, WeightKind::AttentionOutput)?,
            grouped: operands(
                graph,
                attends && numbers.grouped.is_some(),
                WeightKind::AttentionOutputGrouped,
            )?,
            absorbed: operands(graph, attends && numbers.absorbed.is_some(), WeightKind::QueryKeys)?,
        })
    }

    /// The projection's scale and bound dimensions, in `attention_project`'s
    /// order: a history injection (`inject_only`) projects only keys and
    /// values.
    pub(crate) fn project_dimensions(&self, inject_only: bool) -> [(&'static str, Dim); 8] {
        let attended = |dim: Dim| if inject_only { Dim::from(0) } else { dim };
        [
            ("QS", attended(self.query.scale.extent.clone())),
            ("GS", attended(self.gate.scale.extent.clone())),
            ("KS", self.key.scale.extent.clone()),
            ("VS", self.value.scale.extent.clone()),
            ("QB", attended(self.query.bound.extent.clone())),
            ("GB", attended(self.gate.bound.extent.clone())),
            ("KB", self.key.bound.extent.clone()),
            ("VB", self.value.bound.extent.clone()),
        ]
    }

    /// A query segment's scale and bound dimensions alone, in
    /// `attention_project`'s order, from its weight's `operands`: a
    /// low-rank query's up projection, or an indexer's projection.
    pub(crate) fn query_dimensions(operands: &WeightOperands) -> [(&'static str, Dim); 8] {
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

    /// The output projection's scale and bound dimensions.
    pub(crate) fn output_dimensions(&self) -> [(&'static str, Dim); 2] {
        [
            ("OS", self.output.scale.extent.clone()),
            ("OB", self.output.bound.extent.clone()),
        ]
    }
}

/// A low-rank query's weights after its `down`: the norm of its `qr` row
/// (`[rank]`) and its `up` projection (`[Q, rank]`).
#[derive(Clone)]
pub(crate) struct LowRankQueryWeights {
    pub norm: WorkflowTensor,
    pub up: WorkflowTensor,
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

/// An indexer's weights through `operands`, its query heads normed where
/// `query_norm`.
fn indexer_weights(
    operands: &IndexerOperands,
    query_norm: bool,
    weight: &mut impl FnMut(WeightKind) -> Result<WorkflowTensor, GraphError>,
) -> Result<IndexerWeights, GraphError> {
    let packed = |weight: WorkflowTensor, operands: &WeightOperands| PackedWeight {
        weight,
        operands: operands.clone(),
    };
    Ok(IndexerWeights {
        query: packed(weight(WeightKind::IndexerQuery)?, &operands.query),
        query_norm: match query_norm {
            true => Some(weight(WeightKind::IndexerQueryNorm)?),
            false => None,
        },
        head_weights: match &operands.head_weights {
            Some(head_weights) => Some(packed(weight(WeightKind::IndexerHeadWeights)?, head_weights)),
            None => None,
        },
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
) -> [(&'static str, D); 6] {
    [
        ("M", rows),
        ("D", inputs),
        ("Q", outputs),
        ("GR", D::from(0)),
        ("K", D::from(0)),
        ("V", D::from(0)),
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
    pub read: &'n SelectedRead<Dim>,
    pub reals: &'n SelectionReals<Argument<f64>>,
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
) -> Result<(Option<Selected>, IndexerStatePorts, IndexerControlPorts), GraphError> {
    let SelectionInputs { entries, weights, read, reals, .. } = selection;
    let (form, rows, pairs) = (&topology.form, &numbers.rows, &numbers.widths.rotary_pairs);
    let source = &read.source;
    let width = &source.width;
    let gt = axis(read.keys.gated);
    let gate_rows = &gt * width;
    // The tokens' `u` and `g` under the input norm, then in F32.
    let empty = weights.projection.weight.slice_leading(0, 0);
    let absent = GraphConstant::absent(graph, constants)?.tensor().clone();
    let gate = weights.gate.as_ref();
    let (projection, gate_operands) = (&weights.projection.operands, gate.map(|gate| &gate.operands));
    let operand = |extent: Option<&Dim>| extent.cloned().unwrap_or(Dim::from(0));
    let x = &selection.x;
    let projected = graph.enqueue(
        entries.project,
        &[
            [
                ("M", rows.clone()),
                ("D", x.width.clone()),
                ("Q", width.clone()),
                ("GR", gate_rows.clone()),
                ("K", Dim::from(0)),
                ("V", Dim::from(0)),
                ("QS", projection.scale.extent.clone()),
                ("GS", operand(gate_operands.map(|operands| &operands.scale.extent))),
                ("KS", Dim::from(0)),
                ("VS", Dim::from(0)),
                ("QB", projection.bound.extent.clone()),
                ("GB", operand(gate_operands.map(|operands| &operands.bound.extent))),
                ("KB", Dim::from(0)),
                ("VB", Dim::from(0)),
            ]
            .as_slice(),
            &x.form.dimensions(x.width.clone()),
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
    // A softmax-gated pool's projections are published in F32
    // (`pooled::tokens_project`); a mean's `u` is widened from A.
    let (u, g) = match gate {
        Some(_) => (projected.r0.clone(), projected.r1.reshape(&[rows.clone(), gate_rows.clone()])),
        None => {
            let u = graph
                .enqueue(
                    entries.widen,
                    &[("M", rows.clone()), ("D", width.clone())],
                    widen_rows::WorkflowArgs { rows: (&projected.r0).into() },
                )?
                .value;
            // No gate scores: zero columns of the F32 rows.
            let g = u.slice_leading(0, 0).reshape(&[rows.clone(), Dim::from(0)]);
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
            epsilon: reals.key_epsilon.f32(),
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
        return Ok((None, planes, tables));
    };
    // The indexer over the Block read's visible spans: each row's picked
    // units, over its unit list `[0, c)`.
    let class = IndexerClass {
        source,
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
        &reals.indexer,
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
            let visible = controls.visible.as_ref().ok_or("attention visibility is absent")?;
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
    Ok((Some(Selected { bitmap: reads.r0, segments }), planes, tables))
}

/// The layer's history planes, bound to the state store's planes per run, in
/// the order of the history codec's plane descriptors (the key's planes,
/// then the value's): dense key and value, or affine key codes, key
/// coefficients, value codes and value coefficients.
#[derive(Clone)]
pub(crate) struct AttentionStatePorts {
    pub planes: Vec<NativePort>,
    /// A selected read's indexer planes.
    pub indexer: Option<IndexerStatePorts>,
    /// An index-list source's planes and tables.
    pub picked: Option<PickedPorts>,
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

/// Per-run row tables: rotary coordinates, visible history spans, fresh
/// spans over the batch rows and the history row each row's K/V appends to.
#[derive(Clone)]
pub(crate) struct AttentionControlPorts {
    pub coordinates: NativePort,
    pub visible: Option<NativePort>,
    pub fresh: Option<NativePort>,
    pub destinations: NativePort,
    /// The history row tiles the launch's rows see, for an entry that takes
    /// them (the affine prefill).
    pub history_tiles: Option<HistoryTilesPort>,
    /// A selected read's indexer tables.
    pub indexer: Option<IndexerControlPorts>,
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

/// The block's attention weights in the order and presence its topology,
/// input form, sublayer tail and projection operands define.
pub(crate) fn attention_weights(
    topology: &AttentionTopology,
    numbers: &AttentionBlock,
    form: InputForm,
    needs_output: bool,
    operands: AttentionOperands,
    mut weight: impl FnMut(WeightKind) -> Result<WorkflowTensor, GraphError>,
) -> Result<AttentionWeights, GraphError> {
    let layer = &topology.form;
    let mut present = |present: bool, kind| present.then(|| weight(kind)).transpose();
    Ok(AttentionWeights {
        input_norm: present(form.normalized, WeightKind::InputNorm)?,
        query: present(true, topology.query_segment_kind)?.ok_or("query")?,
        low_rank_query: {
            let low_rank = needs_output && numbers.low_rank_query.is_some();
            present(low_rank, WeightKind::QueryDownNorm)?
                .zip(present(low_rank, topology.query_kind)?)
                .map(|(norm, up)| LowRankQueryWeights { norm, up })
        },
        absorbed: present(needs_output && numbers.absorbed.is_some(), WeightKind::QueryKeys)?,
        gate: present(layer.gated_separately(), WeightKind::AttentionGate)?,
        key: present(layer.fresh, WeightKind::Key)?,
        value: present(layer.projects_values(), WeightKind::Value)?,
        query_norm: present(needs_output && layer.query_norm, WeightKind::QueryNorm)?,
        // A Shared layer projects no keys, so it has no key norm.
        key_norm: present(layer.key_norm && layer.fresh, WeightKind::KeyNorm)?,
        sinks: present(needs_output && layer.sinks, WeightKind::AttentionSinks)?,
        output: present(needs_output, WeightKind::AttentionOutput)?,
        grouped: present(
            needs_output && numbers.grouped.is_some(),
            WeightKind::AttentionOutputGrouped,
        )?,
        post_norm: present(topology.post_norm.is_some(), WeightKind::PostNorm)?,
        picked: match (&operands.picked, picked_part(topology, numbers)) {
            (Some(picked), Some(PickedPart::Indexed { reals: index, .. })) => {
                let packed = |weight: WorkflowTensor, operands: &WeightOperands| PackedWeight {
                    weight,
                    operands: operands.clone(),
                };
                Some(PickedWeights {
                    rows: match &picked.rows {
                        Some(rows) => Some(PooledWeights {
                            projection: packed(weight(WeightKind::PooledProjection)?, &rows.projection),
                            gate: match &rows.gate {
                                Some(gate) => Some(packed(weight(WeightKind::PooledGate)?, gate)),
                                None => None,
                            },
                            norm: weight(WeightKind::PooledNorm)?,
                            key: packed(weight(WeightKind::IndexerKey)?, &rows.key),
                            key_norm: weight(WeightKind::IndexerKeyNorm)?,
                        }),
                        None => None,
                    },
                    indexer: indexer_weights(&picked.indexer, index.query_norm.is_some(), &mut weight)?,
                })
            }
            (None, _) => None,
            (Some(_), _) => {
                return Err("index-list operands of an operator without its own indexer".into());
            }
        },
        selection: match (&operands.selection, both(&numbers.selection, &topology.selection, "selected read")) {
            (Some(selection), Some((read, reals))) => {
                let packed = |weight: WorkflowTensor, operands: &WeightOperands| PackedWeight {
                    weight,
                    operands: operands.clone(),
                };
                Some(SelectionWeights {
                    projection: packed(weight(WeightKind::IndexerKey)?, &selection.projection),
                    gate: match &selection.gate {
                        Some(gate) => Some(packed(weight(WeightKind::IndexerKeyGate)?, gate)),
                        None => None,
                    },
                    norm: weight(WeightKind::IndexerKeyNorm)?,
                    bias: match read.keys.token_norm {
                        true => Some(weight(WeightKind::IndexerKeyNormBias)?),
                        false => None,
                    },
                    position: match read.keys.place {
                        true => Some(weight(WeightKind::IndexerKeyGatePosition)?),
                        false => None,
                    },
                    indexer: match &selection.indexer {
                        Some(indexer) => Some(indexer_weights(
                            indexer,
                            reals.indexer.query_norm.is_some(),
                            &mut weight,
                        )?),
                        None => None,
                    },
                })
            }
            (None, None) => None,
            _ => return Err("a selected read's operands and block disagree".into()),
        },
        operands,
    })
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
    let (form, widths) = (&topology.form, &numbers.widths);
    let (rows, width, heads) = (&numbers.rows, &widths.width, widths.heads());
    let hidden = &widths.hidden;
    let inject_only = topology.inject_only;
    let input_norm = match &weights.input_norm {
        Some(norm) => norm.clone(),
        None => GraphConstant::absent(graph, constants)?.tensor().clone(),
    };
    // An absent projection segment reads zero rows of the query weight.
    let empty_weight = weights.query.slice_leading(0, 0);
    let empty_gate_weight = weights
        .gate
        .as_ref()
        .map(|gate| gate.slice_leading(0, 0))
        .unwrap_or_else(|| empty_weight.clone());
    let operands = &weights.operands;
    // A history injection reads neither the query's nor the gate's operands.
    let (query, gate) = match inject_only {
        true => {
            let absent = WeightOperands::absent(graph, constants)?;
            (absent.clone(), absent)
        }
        false => (operands.query.clone(), operands.gate.clone()),
    };
    // The query projection yields the attended query rows, or an absorbed
    // query's heads at their own width; a low-rank query's projection
    // segment projects its `rank` rows of `qr` (a history injection projects
    // no query).
    let query_rows = numbers.query_projection_rows(form);
    let projected = graph.enqueue(
        kernels.project,
        &[
            if inject_only {
                widths.key_value_dimensions(form, rows.clone())
            } else {
                let segment = numbers.low_rank_query.as_ref().map_or(&query_rows, |low_rank| &low_rank.rank);
                widths.project_dimensions(form, rows.clone(), segment.clone())
            }
            .as_slice(),
            operands.project_dimensions(inject_only).as_slice(),
            &input.form.dimensions(hidden.clone()),
        ]
        .concat(),
        attention_project::WorkflowArgs {
            hidden: input.rows.into(),
            input_norm: (&input_norm).into(),
            query_weight: if inject_only {
                (&empty_weight).into()
            } else {
                (&weights.query).into()
            },
            gate_weight: if inject_only {
                (&empty_gate_weight).into()
            } else {
                segment(&weights.gate, &empty_weight)
            },
            key_weight: segment(&weights.key, &empty_weight),
            value_weight: segment(&weights.value, &empty_weight),
            epsilon: topology.epsilon.f32(),
            project_mode: 0.into(),
            query_scale: (&query.scale.tensor).into(),
            gate_scale: (&gate.scale.tensor).into(),
            key_scale: (&operands.key.scale.tensor).into(),
            value_scale: (&operands.value.scale.tensor).into(),
            query_bound: (&query.bound.tensor).into(),
            gate_bound: (&gate.bound.tensor).into(),
            key_bound: (&operands.key.bound.tensor).into(),
            value_bound: (&operands.value.bound.tensor).into(),
        },
    )?;
    // A low-rank query's `qr` rows (`[rows, rank]`, the query segment),
    // widened to F32: the intermediate later stages read. The up projection
    // normalizes them by their norm in its prologue and projects the query
    // heads.
    let (query, qr) = match (
        inject_only,
        both(&numbers.low_rank_query, &topology.low_rank_query, "low-rank query"),
        kernels.low_rank_query,
        &weights.low_rank_query,
        &operands.low_rank_query,
    ) {
        (true, ..) | (false, None, None, None, None) => (projected.r0, None),
        (false, Some((low_rank, low_rank_reals)), Some(entries), Some(up), Some(up_operands)) => {
            let rank = &low_rank.rank;
            let qr = graph
                .enqueue(
                    entries.widen,
                    &[("M", rows.clone()), ("D", rank.clone())],
                    widen_rows::WorkflowArgs {
                        rows: (&projected.r0).into(),
                    },
                )?
                .value;
            let empty = up.up.slice_leading(0, 0);
            let absent = GraphConstant::absent(graph, constants)?.tensor().clone();
            let query = graph
                .enqueue(
                    entries.project,
                    &[
                        query_projection_dimensions(rows.clone(), rank.clone(), query_rows.clone()).as_slice(),
                        AttentionOperands::query_dimensions(up_operands).as_slice(),
                        &InputForm::SINGLE.dimensions(rank.clone()),
                    ]
                    .concat(),
                    attention_project::WorkflowArgs {
                        hidden: (&qr).into(),
                        input_norm: (&up.norm).into(),
                        query_weight: (&up.up).into(),
                        gate_weight: (&empty).into(),
                        key_weight: (&empty).into(),
                        value_weight: (&empty).into(),
                        epsilon: low_rank_reals.epsilon.f32(),
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
            (query, Some((qr, rank.clone(), low_rank_reals.epsilon.f32(), &up.norm)))
        }
        _ => {
            return Err(
                "attention low-rank query entries, weights and operands disagree with its operator"
                    .into(),
            );
        }
    };
    // An absorbed query: each head's projected row through its own block of
    // the key projections (`project_rows` over the heads, F32 sums published
    // in A), the `W`-wide query the history entry attends with.
    let query = match (inject_only, &numbers.absorbed, kernels.absorbed, &weights.absorbed) {
        (true, ..) | (false, None, None, None) => query,
        (false, Some(absorbed), Some(entry), Some(weight)) => {
            graph
                .enqueue(
                    entry,
                    &[
                        ("M", rows.clone()),
                        ("G", absorbed.heads.clone()),
                        ("K", absorbed.width.clone()),
                        ("N", absorbed.keys.clone()),
                        ("WS", operands.absorbed.scale.extent.clone()),
                        ("WB", operands.absorbed.bound.extent.clone()),
                    ],
                    project_rows::WorkflowArgs {
                        source: (&query.reshape(&[rows.clone(), query_rows])).into(),
                        weight: weight.into(),
                        weight_scale: (&operands.absorbed.scale.tensor).into(),
                        weight_bound: (&operands.absorbed.bound.tensor).into(),
                    },
                )?
                .value
        }
        _ => return Err("absorbed query entries, weights and form disagree".into()),
    };
    // The mix entries read the projection per head.
    let attended_heads = if inject_only { Dim::from(0) } else { heads.clone() };
    let fresh = axis(form.fresh);
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
        value: match form.values {
            AttentionValues::Projected => projected
                .r3
                .reshape(&[fresh, rows.clone(), key_width]),
            AttentionValues::RawKey => projected.r2.reshape(&[fresh, rows.clone(), key_width]),
            // The entries take no value of their own (`VK`): zero layers.
            AttentionValues::StoredKey => projected
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
        &weights.selection,
        both(&numbers.selection, &topology.selection, "selected read"),
    ) {
        (None, None, None) => None,
        (Some(entries), Some(weights), Some((read, reals))) => Some(SelectionInputs {
            entries,
            weights,
            x: x_input(),
            input_norm: &x_norm,
            epsilon: topology.epsilon.f32(),
            // Only a graph that runs the indexer reads them.
            qr: match (read.query_rank && weights.indexer.is_some(), &qr_rows) {
                (false, _) => None,
                (true, Some((qr, rank, qr_epsilon, qr_norm))) => Some((
                    ProjectionInput { rows: qr, width: rank.clone(), form: InputForm::SINGLE },
                    qr_norm,
                    *qr_epsilon,
                )),
                (true, None) => return Err("a selected read's query over the absent `qr` rows".into()),
            },
            read,
            reals,
        }),
        _ => return Err("attention selection entries, weights and block disagree".into()),
    };
    let picked = match picked_part(topology, numbers) {
        None => None,
        Some(PickedPart::Reused { plane }) => Some(Picked::Reused { plane }),
        Some(PickedPart::Indexed { source, reals }) => {
            let (Some(entries), Some(weights), Some((qr, rank, qr_epsilon, qr_norm))) =
                (kernels.picked, &weights.picked, &qr_rows)
            else {
                return Err(
                    "an indexing index-list source without its entries, weights or low-rank query"
                        .into(),
                );
            };
            Some(Picked::Indexed {
                source,
                reals,
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
        (Some(grouped), Some(entry), Some(weight)) => {
            let products = graph
                .enqueue(
                    entry,
                    &[
                        ("M", rows.clone()),
                        ("G", grouped.groups.clone()),
                        ("K", grouped.inputs.clone()),
                        ("N", grouped.rank.clone()),
                        ("WS", operands.grouped.scale.extent.clone()),
                        ("WB", operands.grouped.bound.extent.clone()),
                    ],
                    project_rows::WorkflowArgs {
                        source: (&attended.reshape(&[rows.clone(), &heads * width])).into(),
                        weight: weight.into(),
                        weight_scale: (&operands.grouped.scale.tensor).into(),
                        weight_bound: (&operands.grouped.bound.tensor).into(),
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
                    output,
                    &[
                        [
                            ("M", rows.clone()),
                            ("D", hidden.clone()),
                            streams,
                            injected,
                            ("Q", q),
                            ("W", w),
                        ]
                        .as_slice(),
                        operands.output_dimensions().as_slice(),
                    ]
                    .concat(),
                    attention_output::WorkflowArgs {
                        hidden: residual.into(),
                        inject: (&inject).into(),
                        gated: (&attended).into(),
                        output_weight: output_weight.into(),
                        output_scale: (&operands.output.scale.tensor).into(),
                        output_bound: (&operands.output.bound.tensor).into(),
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
                &PackedWeight {
                    weight: output_weight.clone(),
                    operands: operands.output.clone(),
                },
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
    let scoring = selection.as_ref().filter(|selection| selection.read.scored);
    let words = scoring.map_or(Dim::from(0), |selection| selection.read.words.clone());
    let block_keys = scoring.map_or(Dim::from(1), |selection| selection.read.block.clone());
    let selected_prefill = scoring.map(|selection| selection.entries.prefill);
    let dense = matches!(&history, AttentionHistoryEntries::Dense { .. });
    let key_rounding = both(&numbers.key_rounding, &topology.key_rounding, "key rounding");
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
            decode: DenseDecodeEntries::Decode { single, .. },
            ..
        } => graph.control_for(topology.controls, *single, name, &dimensions),
        AttentionHistoryEntries::Dense {
            decode: DenseDecodeEntries::Prefill(kernel),
            ..
        } => graph.control_for(
            topology.controls,
            *kernel,
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
    let mut controls = AttentionControlPorts {
        coordinates: input(graph, "coordinates")?,
        visible: if inject_only {
            None
        } else {
            Some(input(graph, "visible")?)
        },
        fresh: if inject_only {
            None
        } else {
            Some(input(graph, "fresh")?)
        },
        destinations: input(graph, "destinations")?,
        history_tiles,
        indexer: None,
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
        .chain(dense.then(|| axis(form.absorbed_query)))
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
    let key_norm = if form.fresh {
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
    // What the entry reads: the host's visible spans and fresh rows, or for
    // a selected read's decode rows the indexer's segments.
    let mut visible = controls
        .visible
        .as_ref()
        .map(|port| port.tensor().reshape(&[rows.clone(), numbers.segments.clone(), Dim::from(2)]));
    let fresh = controls
        .fresh
        .as_ref()
        .map(|port| port.tensor().reshape(&[rows.clone(), Dim::from(2)]));
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
                        visible: visible.as_ref().ok_or("attention visibility is absent")?.into(),
                        fresh: fresh.as_ref().ok_or("attention fresh rows are absent")?.into(),
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
    // Priming only publishes context K/V, and a selected read's keys: the
    // indexer that picks among them runs only where the layer attends.
    // Select the state-only entry at graph formation to omit
    // attention/merge launches and their scratch buffers, even for a large
    // prompt chunk.
    if inject_only {
        let mut indexer = None;
        if let Some(selection) = &selection {
            let (selected, planes, tables) = select(
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
            controls.indexer = Some(tables);
            indexer = Some(planes);
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
    // A selected read's indexer appends and selects first.
    let mut indexer_planes = None;
    let selected = match &selection {
        None => None,
        Some(selection) => {
            let (selected, planes, tables) = select(
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
            controls.indexer = Some(tables);
            indexer_planes = Some(planes);
            selected
        }
    };
    // A selected decode row reads its units' segments of its visible spans
    // (its launch's earlier rows among them, `cached_causal_domain`) and its
    // own row fresh, as a dense decode row does.
    if let Some(Selected { segments: Some(read), .. }) = &selected {
        let top = &selection.as_ref().ok_or("selected segments without a selection")?.read.source.top;
        let segments = selected_segments(top.clone(), numbers.segments.clone());
        visible = Some(read.reshape(&[rows.clone(), segments.clone(), Dim::from(2)]));
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
    if topology.causal_cache && decode && rows.greater_than(1) && form.fresh {
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
            decode:
                DenseDecodeEntries::Decode {
                    verify: Some(kernel),
                    ..
                },
            ..
        } if decode && rows.greater_than(1) => {
            dense_mix!(kernel, attention_decode, &attend_dimensions)
        }
        AttentionHistoryEntries::Dense {
            decode: DenseDecodeEntries::Decode { single: kernel, .. },
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
                (None, DenseDecodeEntries::Prefill(kernel)) if decode_rows => kernel,
                (None, _) => prefill,
                (Some(_), _) => return Err("a selected prefill of another codec".into()),
            };
            let bitmap = prefill_selection(graph, rows, &selected, row_class, constants)?;
            dense_mix!(kernel, attention_prefill,
                &prefill_dimensions(&dimensions, &words, &block_keys),
                selection: (&bitmap).into(),
                block_slots: (&block_slots).into())
        }
        AttentionHistoryEntries::AffineK8V4 {
            verify_four: Some(kernel),
            ..
        } if decode && rows.equals(4) => mix!(
            kernel,
            attention_decode_k8v4,
            history_key_codes,
            history_key_coefficients,
            history_value_codes,
            history_value_coefficients
        ),
        AttentionHistoryEntries::AffineK8V4 {
            verify_eight: Some(kernel),
            ..
        } if decode && rows.greater_than(4) => mix!(
            kernel,
            attention_decode_k8v4,
            history_key_codes,
            history_key_coefficients,
            history_value_codes,
            history_value_coefficients
        ),
        AttentionHistoryEntries::AffineK8V4 {
            verify: Some(kernel),
            ..
        } if decode && rows.greater_than(1) => mix!(
            kernel,
            attention_decode_k8v4,
            history_key_codes,
            history_key_coefficients,
            history_value_codes,
            history_value_coefficients
        ),
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
                history_tiles: match &controls.history_tiles {
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
            indexer: indexer_planes,
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

/// A projection segment's weight, or zero rows of the query weight when the
/// form has no such segment.
fn segment<'t>(
    weight: &'t Option<WorkflowTensor>,
    empty: &'t WorkflowTensorView,
) -> WorkflowTensorRef<'t> {
    match weight {
        Some(weight) => weight.into(),
        None => empty.into(),
    }
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
    use crate::planning::{AttentionForm, AttentionWidths, GateForm};

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
        let binding = |shape| AttentionBinding {
            shape,
            key_value_only: false,
            norm: Element::bf16(),
            query: Element::bf16(),
            low_rank_query: None,
            gate: Element::bf16(),
            key: Element::bf16(),
            value: Element::bf16(),
            output: Element::bf16(),
            activation: Element::bf16(),
            history: match shape.form.values {
                AttentionValues::StoredKey => {
                    CodecSpec::stored_key(DType::BF16, shape.widths.width as usize)
                }
                AttentionValues::Projected | AttentionValues::RawKey => {
                    CodecSpec::dense(DType::BF16, shape.widths.width as usize, shape.widths.width as usize)
                }
            },
            tail: SublayerTail::Residual,
            selection: None,
            grouped: None,
            absorbed: None,
            picked: None,
        };
        let v41 = AttentionShape {
            form: AttentionForm {
                gate: GateForm::None,
                fresh: true,
                query_norm: false,
                key_norm: true,
                value_norm: false,
                values: AttentionValues::StoredKey,
                sinks: true,
                rotary_adjacent: true,
                rotary_trailing: true,
                output_rotary: true,
                absorbed_query: false,
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
                gate: GateForm::Interleaved,
                query_norm: true,
                values: AttentionValues::Projected,
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
                form: AttentionForm { gate: GateForm::None, output_rotary: true, ..qwen.form },
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
