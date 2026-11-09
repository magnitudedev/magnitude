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
use crate::operators::{self, InputForm, Join, SublayerInput};
use crate::programs::graph::{
    GraphError,
    draft::{ControlSource, GraphDraft},
};
use super::pooled::{
    Picked, PickedBlock, PickedEntries, PickedInputs, PickedPorts, PickedWeights, PoolEntry,
    PooledEntries, PooledWeights, Pooling,
};
use super::selection::{CandidateEntries, IndexerEntries, ProjectionInput};
use crate::programs::native_target_graph::{PackedWeight, WeightOperands, WeightPort};
use crate::{AttentionBinding, AttentionShape, AttentionValues, SelectionShape, SublayerTail};
use crate::{
    native::{
        AttentionHistoryKernels, AttentionKernels, DenseDecodeKernels, PoolKernel,
        SelectedPrefillKernel,
    },
    programs::native_constants::GraphConstant,
};
use magnitude_family_contracts::{
    Attention, HeadNorm, HeadWeights, Rotary, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    attention_append_dense, attention_append_k8v4, attention_decode, attention_decode_k8v4,
    attention_output, attention_prefill, attention_prefill_k8v4, attention_project, indexer_select,
    project_rows, widen_rows,
};
use magnitude_state::{CodecSpec, HistoryEncoding, HistoryValue};
use seismic::{
    Element, NativeGraph, NativeGraphClassSlice, NativeGraphMetadata, NativePort, WorkflowTensor,
    WorkflowTensorRef, WorkflowTensorView,
};

pub(crate) struct AttentionGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, attention_project::Entry>,
    pub history: AttentionHistoryEntries<'a, G>,
    pub output: TailEntries<'a, G, attention_output::Entry>,
    /// A grouped output's first product (`project_rows` over its groups).
    pub grouped: Option<G::Binding<'a, project_rows::Entry>>,
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

/// A selected read's entries: the indexer's projection, `indexer_select`
/// and the codec's WorkList prefill.
pub(crate) struct SelectionGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, attention_project::Entry>,
    pub select: G::Binding<'a, indexer_select::Entry>,
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
/// extended shape (CUDA's and Vulkan's, which declare neither rounded keys
/// nor index-list picks: only `KR = KS = 0`), the prefill entry, tuned on
/// those classes, attends them. A selected read's decode rows read the
/// indexer's segments, which only the decode entry takes, so its refusal
/// stands.
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
        Err(crate::native::CatalogFailure::KernelDomain { .. }) if binding.selection.is_none() => {
            Ok(false)
        }
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
            selection: kernels
                .selection
                .as_ref()
                .map(|selection| SelectionGraphEntries {
                    project: &selection.project,
                    select: &selection.select,
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
                indexer: IndexerEntries {
                    query_project: &picked.indexer.query_project,
                    weights_project: picked.indexer.weights_project.as_ref(),
                    score: &picked.indexer.score,
                    topk: &picked.indexer.topk,
                    candidates: picked.indexer.candidates.as_ref().map(|candidates| {
                        CandidateEntries {
                            blocks: &candidates.blocks,
                            rows: &candidates.rows,
                        }
                    }),
                },
            }),
        }
    }
}

/// Entry element assignments from the same program binding used by native
/// preparation. The checked graph obtains all shapes from the entries.
pub(crate) struct CheckedAttentionEntries {
    project: [(&'static str, Element); 7],
    mix: [(&'static str, Element); 1],
    output: [(&'static str, Element); 2],
    grouped: Option<[(&'static str, Element); 3]>,
    post_norm: Option<CheckedPostNormEntries>,
    history: CodecSpec,
    /// Whether the graphs may list their launch's history row tiles
    /// (`StateResourcePlan::lists_history_tiles`, for a program whose
    /// classes have the listing axis).
    lists: bool,
    /// Whether a dense history's decode row classes attend with the decode
    /// entry ([`dense_decodes`]).
    decodes: bool,
    /// A selected read's indexer projection and selection.
    indexer_project: Option<[(&'static str, Element); 7]>,
    indexer_select: [(&'static str, Element); 1],
    /// A low-rank query's widening and up projection.
    low_rank_query: Option<([(&'static str, Element); 1], [(&'static str, Element); 7])>,
    /// An indexing index-list source's entries.
    picked: Option<CheckedPickedEntries>,
}

/// An indexing index-list source's entry elements: an own source's `u | g`
/// and index-key projections, the indexer's query and head-weight
/// projections, and the activation element every other entry of it reads
/// (`widen_rows`, `pool_rows` or at rate 1 `attention_append_dense`,
/// `indexer_keys`, `indexer_score`;
/// `rows_topk`, `candidate_blocks` and `candidate_rows` read none).
struct CheckedPickedEntries {
    elements: super::pooled::PickedElements,
    activation: [(&'static str, Element); 1],
}

impl CheckedPickedEntries {
    fn new(binding: AttentionBinding) -> Option<Self> {
        Some(Self {
            elements: super::pooled::PickedElements::of(binding)?,
            activation: [("A", binding.activation)],
        })
    }

    fn entries(&self) -> PickedEntries<'_, NativeGraphMetadata> {
        PickedEntries {
            rows: self.elements.rows.as_ref().map(|rows| PooledEntries {
                project: &rows.project[..],
                widen: &self.activation[..],
                pool: match rows.pooling {
                    Pooling::Identity => PoolEntry::Identity(&self.activation[..]),
                    Pooling::Grouped => PoolEntry::Grouped(&self.activation[..]),
                },
                key_project: &rows.key_project[..],
                keys: &self.activation[..],
            }),
            indexer: IndexerEntries {
                query_project: &self.elements.query_project[..],
                weights_project: self.elements.weights_project.as_ref().map(|elements| &elements[..]),
                score: &self.activation[..],
                topk: &[],
                candidates: self.elements.produces().then_some(CandidateEntries {
                    blocks: &[],
                    rows: &[],
                }),
            },
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
    let width = binding.shape.width;
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
            indexer_project: binding.selection.map(|selection| {
                [
                    ("NW", binding.norm),
                    ("QW", selection.query),
                    ("GW", selection.query),
                    ("KW", selection.key),
                    ("VW", selection.query),
                    ("A", binding.activation),
                    ("Y", binding.activation),
                ]
            }),
            // The indexer reads its projections in the activation element
            // and keeps everything else in F32 (`indexer_select`).
            indexer_select: [("A", binding.activation)],
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
            selection: self
                .indexer_project
                .as_ref()
                .map(|project| SelectionGraphEntries {
                    project: &project[..],
                    select: &self.indexer_select[..],
                    prefill: match self.history.encoding() {
                        HistoryEncoding::Affine => {
                            SelectedPrefillEntries::AffineK8V4(&self.mix[..])
                        }
                        _ => SelectedPrefillEntries::Dense(&self.mix[..]),
                    },
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
pub(crate) fn decodes(rows: u64) -> bool {
    rows <= DECODE_ROWS
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
    pub indexer: Option<IndexerWeights>,
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
    pub indexer_query: WeightOperands,
    pub indexer_key: WeightOperands,
    /// An indexing index-list source's projections', where the block
    /// attends.
    pub picked: Option<PickedOperands>,
}

/// An indexing index-list source's projections' operands: an own source's
/// rows' and its indexer's query and head weights'.
#[derive(Clone)]
pub(crate) struct PickedOperands {
    pub rows: Option<PooledOperands>,
    pub query: WeightOperands,
    pub head_weights: Option<WeightOperands>,
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
        load: &crate::ModelLoadPlan,
        scope: WeightScope,
        operator: &Attention,
        shape: &AttentionShape,
        attends: bool,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        let mut operands = |graph: &mut G, read: bool, kind| match read {
            true => {
                WeightOperands::planned(graph, load, WeightRole { scope, kind }, ports, constants)
            }
            false => WeightOperands::absent(graph, constants),
        };
        let selected =
            attends && operators::attention::admitted_selection(operator).is_some();
        let (rows, indexer) = match operators::attention::picked_source(operator)
            .map_err(|error| error.to_string())?
        {
            Some(operators::attention::PickedSource::Own { rows, indexer, .. }) => {
                (Some(rows), Some(indexer))
            }
            Some(operators::attention::PickedSource::Reindex { indexer, .. }) => (None, Some(indexer)),
            Some(operators::attention::PickedSource::Reused { .. }) | None => (None, None),
        };
        let picked = match indexer {
            Some(indexer) if attends => Some(PickedOperands {
                rows: match rows {
                    Some(rows) => Some(PooledOperands {
                        projection: operands(graph, true, WeightKind::PooledProjection)?,
                        gate: match super::pooled::gated(rows) {
                            true => Some(operands(graph, true, WeightKind::PooledGate)?),
                            false => None,
                        },
                        key: operands(graph, true, WeightKind::IndexerKey)?,
                    }),
                    None => None,
                },
                query: operands(graph, true, WeightKind::IndexerQuery)?,
                head_weights: match indexer.head_weights {
                    HeadWeights::Projected(_) => {
                        Some(operands(graph, true, WeightKind::IndexerHeadWeights)?)
                    }
                    HeadWeights::Uniform => None,
                },
            }),
            _ => None,
        };
        Ok(Self {
            picked,
            indexer_query: operands(graph, selected, WeightKind::IndexerQuery)?,
            indexer_key: operands(graph, selected, WeightKind::IndexerKey)?,
            query: operands(
                graph,
                attends,
                operators::attention::query_segment_kind(operator),
            )?,
            low_rank_query: match attends
                && operators::attention::low_rank_query(operator).is_some()
            {
                true => Some(operands(
                    graph,
                    true,
                    operators::attention::query_kind(operator),
                )?),
                false => None,
            },
            gate: operands(
                graph,
                attends && shape.gate_rows() > 0,
                WeightKind::AttentionGate,
            )?,
            key: operands(graph, shape.key_rows() > 0, WeightKind::Key)?,
            value: operands(graph, shape.value_rows() > 0, WeightKind::Value)?,
            output: operands(graph, attends, WeightKind::AttentionOutput)?,
            grouped: operands(
                graph,
                attends && operators::attention::grouped_output(operator).is_some(),
                WeightKind::AttentionOutputGrouped,
            )?,
        })
    }

    /// The projection's scale and bound dimensions, in `attention_project`'s
    /// order: a history injection (`inject_only`) projects only keys and
    /// values.
    pub(crate) fn project_dimensions(&self, inject_only: bool) -> [(&'static str, u64); 8] {
        let attended = |extent: u64| if inject_only { 0 } else { extent };
        [
            ("QS", attended(self.query.scale.extent)),
            ("GS", attended(self.gate.scale.extent)),
            ("KS", self.key.scale.extent),
            ("VS", self.value.scale.extent),
            ("QB", attended(self.query.bound.extent)),
            ("GB", attended(self.gate.bound.extent)),
            ("KB", self.key.bound.extent),
            ("VB", self.value.bound.extent),
        ]
    }

    /// A low-rank query's up projection's scale and bound dimensions, in
    /// `attention_project`'s order (a query segment alone), from its
    /// `up`'s `operands`.
    pub(crate) fn low_rank_query_dimensions(operands: &WeightOperands) -> [(&'static str, u64); 8] {
        [
            ("QS", operands.scale.extent),
            ("GS", 0),
            ("KS", 0),
            ("VS", 0),
            ("QB", operands.bound.extent),
            ("GB", 0),
            ("KB", 0),
            ("VB", 0),
        ]
    }

    /// The indexer projection's scale and bound dimensions, in
    /// `attention_project`'s order (no gate or value segments).
    pub(crate) fn indexer_dimensions(&self) -> [(&'static str, u64); 8] {
        [
            ("QS", self.indexer_query.scale.extent),
            ("GS", 0),
            ("KS", self.indexer_key.scale.extent),
            ("VS", 0),
            ("QB", self.indexer_query.bound.extent),
            ("GB", 0),
            ("KB", self.indexer_key.bound.extent),
            ("VB", 0),
        ]
    }

    /// The output projection's scale and bound dimensions.
    pub(crate) fn output_dimensions(&self) -> [(&'static str, u64); 2] {
        [
            ("OS", self.output.scale.extent),
            ("OB", self.output.bound.extent),
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

#[derive(Clone)]
/// A selected read's indexer weights: its query and key projections and
/// their norms.
pub(crate) struct IndexerWeights {
    pub query: WorkflowTensor,
    pub key: WorkflowTensor,
    pub query_norm: WorkflowTensor,
    pub key_norm: WorkflowTensor,
}

/// One attention block at one graph class.
pub(crate) struct AttentionBlock<'a> {
    pub rows: u64,
    pub segments: u64,
    pub history_rows: u64,
    pub slab_rows: u32,
    /// The history row tiles one launch of the class can see
    /// (`HistoryStorePlan::launch_tiles` of its request slots).
    pub history_tiles: u64,
    pub shape: AttentionShape,
    pub operator: &'a Attention,
    /// The input norm's epsilon.
    pub epsilon: f32,
    /// The head norms' one epsilon.
    pub head_epsilon: f32,
    /// The post-norm's epsilon, read only by a post-norm tail.
    pub post_norm_epsilon: f32,
    /// The post-norm's output scale: 1, or the layer's output scale.
    pub post_norm_scale: f32,
    pub activation: Element,
    /// Draft priming only publishes K/V history; its attention output is unused.
    pub inject_only: bool,
    /// Causal target rows consume earlier rows through the cache codec.
    pub causal_cache: bool,
    /// A selected read's indexer and its history domains.
    pub selection: Option<SelectionBlock>,
    /// An index-list source's rows (`HistorySource` 1).
    pub picked: Option<PickedBlock<'a>>,
    /// How the graph takes its launch's controls.
    pub controls: ControlSource,
}

/// A selected read's indexer at one graph class: its static axes and its
/// raw keys' (Window) and block keys' (Block) history rows and slab rows.
#[derive(Clone, Copy)]
pub(crate) struct SelectionBlock {
    pub shape: SelectionShape,
    /// Whether the class scores: some row of its launch has more than
    /// `blocks` complete blocks. A class without scoring appends the
    /// indexer's keys alone and reads every visible row.
    pub scored: bool,
    pub window_rows: u64,
    pub window_slab_rows: u32,
    pub block_rows: u64,
    pub block_slab_rows: u32,
}

/// The segments `C` a decode row of a read selecting `blocks` blocks reads
/// (`indexer_select`'s Segments form): every run of its selected blocks and
/// of its tail in its visible spans, split at most where one of those spans
/// (at most the class of `MAX_HISTORY_SPANS`) ends. One capacity for every
/// class, so a load's graphs agree on it: the decode entries' `R` in a
/// selected block's classes (`decode_class_segments`).
pub(crate) fn selected_segments(blocks: u64) -> u64 {
    blocks + 1 + magnitude_state::MAX_HISTORY_SPANS.next_power_of_two() as u64
}

/// The class slice of a block whose attention is `operator`: a selected
/// read's decode entries read its `C` segments whatever the class's history
/// segments (`selected_segments`).
pub(crate) fn decode_class_segments(
    slice: NativeGraphClassSlice,
    operator: &Attention,
) -> NativeGraphClassSlice {
    match operators::attention::admitted_selection(operator) {
        None => slice,
        Some(read) => {
            let segments = selected_segments(read.blocks);
            slice.scoped("attention_decode", "R", [segments]).scoped(
                "attention_decode_k8v4",
                "R",
                [segments],
            )
        }
    }
}

/// The attention projection's dimensions at `rows` rows: the query
/// segment projects the query rows, or a low-rank query's `rank` rows of
/// `qr` (`query_rank`).
pub(crate) fn project_dimensions(
    shape: &AttentionShape,
    query_rank: Option<u64>,
    rows: u64,
) -> [(&'static str, u64); 6] {
    let mut dimensions = shape.project_dimensions(rows);
    if let Some(rank) = query_rank {
        dimensions[2].1 = rank;
    }
    dimensions
}

/// A low-rank query's up projection's dimensions at `rows` rows
/// (`attention_project`): its `rank`-wide `qr` rows to the query rows, no
/// gate, key or value.
pub(crate) fn low_rank_query_dimensions(
    shape: &AttentionShape,
    rank: u64,
    rows: u64,
) -> [(&'static str, u64); 6] {
    [
        ("M", rows),
        ("D", rank),
        ("Q", shape.query_rows()),
        ("GR", 0),
        ("K", 0),
        ("V", 0),
    ]
}

/// The indexer projection's dimensions at `rows` rows over a `hidden`-wide
/// input (`attention_project`): its query heads and raw key, no gate or
/// value.
pub(crate) fn indexer_project_dimensions(
    hidden: u64,
    selection: SelectionShape,
    rows: u64,
) -> [(&'static str, u64); 6] {
    [
        ("M", rows),
        ("D", hidden),
        ("Q", selection.heads * selection.width),
        ("GR", 0),
        ("K", selection.width),
        ("V", 0),
    ]
}

/// A selected read's indexer inputs at one graph class: its selection
/// entry, the projected query heads `[rows, heads, width]` and raw keys
/// `[rows, width]`, its norm rows and its block.
pub(crate) struct SelectionInputs<'a, G: GraphDraft + 'a> {
    pub select: G::Binding<'a, indexer_select::Entry>,
    pub prefill: SelectedPrefillEntries<'a, G>,
    pub query: WorkflowTensorView,
    pub key: WorkflowTensorView,
    pub query_norm: WorkflowTensorView,
    pub key_norm: WorkflowTensorView,
    pub block: SelectionBlock,
}

/// What a selected read's attention reads, from `indexer_select`: a decode
/// row's segments `[rows, C, 2]` and a prefill row's bitmap `[rows, B]`.
struct Selected {
    segments: WorkflowTensor,
    bitmap: WorkflowTensor,
}

/// The indexer of a selected read over the class's rows: appends the rows'
/// raw keys and finished blocks to its planes and selects what each row
/// reads (`indexer_select`), the Segments form for decode rows, the
/// WorkList bitmap for prefill rows. Its planes and per-run tables are
/// ports of the graph; `rotary` is the attention's table.
fn select<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    selection: &SelectionInputs<'a, G>,
    block: &AttentionBlock<'_>,
    controls: &AttentionControlPorts,
    rotary: [&WorkflowTensor; 3],
    constants: &mut Vec<GraphConstant>,
) -> Result<(Selected, IndexerStatePorts, IndexerControlPorts), GraphError> {
    let binding = selection.block.shape;
    let pairs = block.shape.rotary_pairs;
    let rows = block.rows;
    // A class without scoring only appends (B = C = 0).
    let scored = selection.block.scored;
    let segments = if scored && decodes(rows) {
        selected_segments(binding.blocks)
    } else {
        0
    };
    let words = if scored { binding.words } else { 0 };
    let dimensions = [
        ("M", rows),
        ("TW", selection.block.window_rows),
        ("TB", selection.block.block_rows),
        ("H", binding.heads),
        ("P", pairs),
        ("S", binding.width - 2 * pairs),
        ("SB", binding.block),
        ("R", block.segments),
        ("C", segments),
        ("B", words),
    ];
    let mut input =
        |name: &str| graph.control_for(block.controls, selection.select, name, &dimensions);
    let tables = IndexerControlPorts {
        window_visible: input("window_visible")?,
        window_destinations: input("window_destinations")?,
        block_visible: input("block_visible")?,
        block_destinations: input("block_destinations")?,
    };
    let mut planes = IndexerStatePorts {
        window_keys: graph.port(
            Element::f32(),
            &[selection.block.window_rows, binding.width],
        )?,
        window_coordinates: graph.port(Element::i32(), &[selection.block.window_rows, 4])?,
        block_keys: graph.port(Element::f32(), &[selection.block.block_rows, binding.width])?,
    };
    let Some(indexer) = operators::attention::admitted_selection(block.operator) else {
        return Err("a selected block's operator reads every visible row".into());
    };
    // Extent-only arguments: zero rows of a row table, and an empty bitmap.
    let none = controls.destinations.tensor().slice_leading(0, 0);
    let block_slots = none.reshape(&[0, binding.block]);
    let segment_slots = none.reshape(&[0, segments]);
    let bitmap_words = GraphConstant::u32_zeros(graph, 0, words)?;
    let visible = controls
        .visible
        .as_ref()
        .ok_or("attention visibility is absent")?;
    let results = graph.enqueue(
        selection.select,
        &dimensions,
        indexer_select::WorkflowArgs {
            query: (&selection.query).into(),
            key: (&selection.key).into(),
            query_norm: (&selection.query_norm).into(),
            key_norm: (&selection.key_norm).into(),
            rotary_components: rotary[0].into(),
            rotary_frequencies: rotary[1].into(),
            rotary_amplitudes: rotary[2].into(),
            coordinates: controls.coordinates.tensor().into(),
            window_visible: tables.window_visible.tensor().into(),
            window_destinations: tables.window_destinations.tensor().into(),
            block_visible: tables.block_visible.tensor().into(),
            block_destinations: tables.block_destinations.tensor().into(),
            visible: visible.tensor().into(),
            block_slots: (&block_slots).into(),
            segment_slots: (&segment_slots).into(),
            bitmap_words: bitmap_words.port().tensor().into(),
            window_keys: planes.window_keys.tensor_mut().into(),
            window_coordinates: planes.window_coordinates.tensor_mut().into(),
            block_keys: planes.block_keys.tensor_mut().into(),
            selected: i32::try_from(binding.blocks).map_err(|_| "selected blocks exceed i32")?,
            epsilon: indexer.query_norm.epsilon as f32,
            slab_rows: block.slab_rows,
            window_slab_rows: selection.block.window_slab_rows,
            block_slab_rows: selection.block.block_slab_rows,
        },
    )?;
    constants.push(bitmap_words);
    Ok((
        Selected {
            segments: results.r0,
            bitmap: results.r1,
        },
        planes,
        tables,
    ))
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

/// A selected read's indexer planes, bound to its two history domains'
/// planes per run: the raw keys and their coordinates (Window), and the
/// finished block keys (Block).
#[derive(Clone)]
pub(crate) struct IndexerStatePorts {
    pub window_keys: NativePort,
    pub window_coordinates: NativePort,
    pub block_keys: NativePort,
}

/// A selected read's per-run indexer tables: each row's spans and
/// destination in its raw keys' and its block keys' history reads.
#[derive(Clone)]
pub(crate) struct IndexerControlPorts {
    pub window_visible: NativePort,
    pub window_destinations: NativePort,
    pub block_visible: NativePort,
    pub block_destinations: NativePort,
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
/// tiles its launch's rows see, and the tiles it lists.
#[derive(Clone)]
pub(crate) struct HistoryTilesPort {
    pub port: NativePort,
    pub tiles: usize,
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

/// The block's attention weights in the order and presence its operator's
/// form, input form, sublayer tail and projection operands define.
pub(crate) fn attention_weights(
    shape: &AttentionShape,
    operator: &Attention,
    form: InputForm,
    needs_output: bool,
    post_norm: bool,
    operands: AttentionOperands,
    mut weight: impl FnMut(WeightKind) -> Result<WorkflowTensor, GraphError>,
) -> Result<AttentionWeights, GraphError> {
    let mut present = |present: bool, kind| present.then(|| weight(kind)).transpose();
    Ok(AttentionWeights {
        input_norm: present(form.normalized, WeightKind::InputNorm)?,
        query: present(true, operators::attention::query_segment_kind(operator))?
            .ok_or("query")?,
        low_rank_query: {
            let low_rank = needs_output && operators::attention::low_rank_query(operator).is_some();
            present(low_rank, WeightKind::QueryDownNorm)?
                .zip(present(low_rank, operators::attention::query_kind(operator))?)
                .map(|(norm, up)| LowRankQueryWeights { norm, up })
        },
        gate: present(shape.gate_rows() > 0, WeightKind::AttentionGate)?,
        key: present(shape.key_rows() > 0, WeightKind::Key)?,
        value: present(shape.value_rows() > 0, WeightKind::Value)?,
        query_norm: present(needs_output && shape.query_norm > 0, WeightKind::QueryNorm)?,
        // A Shared layer projects no keys, so it has no key norm.
        key_norm: present(shape.key_norm > 0 && shape.fresh > 0, WeightKind::KeyNorm)?,
        sinks: present(needs_output && shape.sinks > 0, WeightKind::AttentionSinks)?,
        output: present(needs_output, WeightKind::AttentionOutput)?,
        grouped: present(
            needs_output && operators::attention::grouped_output(operator).is_some(),
            WeightKind::AttentionOutputGrouped,
        )?,
        post_norm: present(post_norm, WeightKind::PostNorm)?,
        picked: match (
            &operands.picked,
            operators::attention::picked_source(operator).map_err(|error| error.to_string())?,
        ) {
            (
                Some(picked),
                Some(
                    operators::attention::PickedSource::Own { indexer, .. }
                    | operators::attention::PickedSource::Reindex { indexer, .. },
                ),
            ) => {
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
                    query: packed(weight(WeightKind::IndexerQuery)?, &picked.query),
                    query_norm: match indexer.query.norm {
                        HeadNorm::Rms(_) => Some(weight(WeightKind::IndexerQueryNorm)?),
                        HeadNorm::None => None,
                    },
                    head_weights: match &picked.head_weights {
                        Some(head_weights) => {
                            Some(packed(weight(WeightKind::IndexerHeadWeights)?, head_weights))
                        }
                        None => None,
                    },
                })
            }
            (None, _) => None,
            (Some(_), _) => {
                return Err("index-list operands of an operator without its own indexer".into());
            }
        },
        operands,
        indexer: match operators::attention::admitted_selection(operator) {
            None => None,
            Some(_) => Some(IndexerWeights {
                query: weight(WeightKind::IndexerQuery)?,
                key: weight(WeightKind::IndexerKey)?,
                query_norm: weight(WeightKind::IndexerQueryNorm)?,
                key_norm: weight(WeightKind::IndexerKeyNorm)?,
            }),
        },
    })
}

pub(crate) fn attention<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    kernels: AttentionGraphEntries<'a, G>,
    weights: &AttentionWeights,
    constants: &mut Vec<GraphConstant>,
    input: SublayerInput<'_>,
    residual: &WorkflowTensor,
    join: &Join,
    block: AttentionBlock<'_>,
) -> Result<(WorkflowTensor, AttentionStatePorts, AttentionControlPorts), GraphError> {
    let shape = block.shape;
    let (rows, width, heads) = (block.rows, shape.width, shape.heads());
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
    // A low-rank query's projection segment projects its `rank` rows of
    // `qr` (a history injection projects no query).
    let low_rank = operators::attention::low_rank_query(block.operator);
    // A history injection reads neither the query's nor the gate's operands.
    let (query, gate) = match block.inject_only {
        true => {
            let absent = WeightOperands::absent(graph, constants)?;
            (absent.clone(), absent)
        }
        false => (operands.query.clone(), operands.gate.clone()),
    };
    let projected = graph.enqueue(
        kernels.project,
        &[
            if block.inject_only {
                shape.key_value_dimensions(rows)
            } else {
                project_dimensions(&shape, low_rank.map(|(rank, _)| rank), rows)
            }
            .as_slice(),
            operands.project_dimensions(block.inject_only).as_slice(),
            &input.form.dimensions(shape.hidden),
        ]
        .concat(),
        attention_project::WorkflowArgs {
            hidden: input.rows.into(),
            input_norm: (&input_norm).into(),
            query_weight: if block.inject_only {
                (&empty_weight).into()
            } else {
                (&weights.query).into()
            },
            gate_weight: if block.inject_only {
                (&empty_gate_weight).into()
            } else {
                segment(&weights.gate, &empty_weight)
            },
            key_weight: segment(&weights.key, &empty_weight),
            value_weight: segment(&weights.value, &empty_weight),
            epsilon: block.epsilon,
            project_mode: 0,
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
        block.inject_only,
        low_rank,
        kernels.low_rank_query,
        &weights.low_rank_query,
        &operands.low_rank_query,
    ) {
        (true, ..) | (false, None, None, None, None) => (projected.r0, None),
        (false, Some((rank, epsilon)), Some(entries), Some(up), Some(up_operands)) => {
            let qr = graph
                .enqueue(
                    entries.widen,
                    &[("M", rows), ("D", rank)],
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
                        low_rank_query_dimensions(&shape, rank, rows).as_slice(),
                        AttentionOperands::low_rank_query_dimensions(up_operands).as_slice(),
                        &InputForm::SINGLE.dimensions(rank),
                    ]
                    .concat(),
                    attention_project::WorkflowArgs {
                        hidden: (&qr).into(),
                        input_norm: (&up.norm).into(),
                        query_weight: (&up.up).into(),
                        gate_weight: (&empty).into(),
                        key_weight: (&empty).into(),
                        value_weight: (&empty).into(),
                        epsilon: epsilon as f32,
                        project_mode: 0,
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
            (query, Some((qr, rank, epsilon as f32, &up.norm)))
        }
        _ => {
            return Err(
                "attention low-rank query entries, weights and operands disagree with its operator"
                    .into(),
            );
        }
    };
    // The mix entries read the projection per head.
    let projected = ProjectedRows {
        query: query.reshape(&[
            rows,
            if block.inject_only { 0 } else { heads },
            width + shape.interleaved_gate,
        ]),
        gate: projected.r1.reshape(&[
            rows,
            if block.inject_only { 0 } else { heads },
            shape.separate_gate,
        ]),
        key: projected
            .r2
            .reshape(&[shape.fresh, rows, shape.kv_heads * width]),
        value: match shape.values {
            AttentionValues::Projected => projected
                .r3
                .reshape(&[shape.fresh, rows, shape.kv_heads * width]),
            AttentionValues::RawKey => projected
                .r2
                .reshape(&[shape.fresh, rows, shape.kv_heads * width]),
            // The entries take no value of their own (`VK`): zero layers.
            AttentionValues::StoredKey => projected
                .r2
                .slice_leading(0, 0)
                .reshape(&[0, rows, shape.kv_heads * width]),
        },
    };
    // A selected read's indexer projects its query heads and raw key from
    // the same input under the same norm.
    let selection = match (kernels.selection, &weights.indexer, block.selection) {
        (None, None, None) => None,
        (Some(entries), Some(indexer), Some(selection)) => {
            let binding = selection.shape;
            let empty = indexer.query.slice_leading(0, 0);
            let absent = GraphConstant::absent(graph, constants)?.tensor().clone();
            let projected = graph.enqueue(
                entries.project,
                &[
                    indexer_project_dimensions(shape.hidden, binding, rows).as_slice(),
                    operands.indexer_dimensions().as_slice(),
                    &input.form.dimensions(shape.hidden),
                ]
                .concat(),
                attention_project::WorkflowArgs {
                    hidden: input.rows.into(),
                    input_norm: (&input_norm).into(),
                    query_weight: (&indexer.query).into(),
                    query_scale: (&operands.indexer_query.scale.tensor).into(),
                    gate_scale: (&absent).into(),
                    key_scale: (&operands.indexer_key.scale.tensor).into(),
                    value_scale: (&absent).into(),
                    query_bound: (&operands.indexer_query.bound.tensor).into(),
                    gate_bound: (&absent).into(),
                    key_bound: (&operands.indexer_key.bound.tensor).into(),
                    value_bound: (&absent).into(),
                    gate_weight: (&empty).into(),
                    key_weight: (&indexer.key).into(),
                    value_weight: (&empty).into(),
                    epsilon: block.epsilon,
                    project_mode: 0,
                },
            )?;
            Some(SelectionInputs {
                select: entries.select,
                prefill: entries.prefill,
                query: projected.r0.reshape(&[rows, binding.heads, binding.width]),
                key: projected.r2.reshape(&[rows, binding.width]),
                query_norm: indexer.query_norm.reshape(&[1, binding.width]),
                key_norm: indexer.key_norm.reshape(&[1, binding.width]),
                block: selection,
            })
        }
        _ => return Err("attention selection entries, weights and block disagree".into()),
    };
    // An own index-list source projects the layer's normed input and its
    // indexer the low-rank query's `qr` rows.
    let x = input.rows.slice_leading(0, rows);
    let x_norm = input_norm.reshape(&[if weights.input_norm.is_some() { shape.hidden } else { 0 }]);
    let qr_rows = qr.as_ref().map(|(qr, rank, epsilon, norm)| {
        (qr.slice_leading(0, rows), *rank, *epsilon, norm.reshape(&[*rank]))
    });
    let picked = match block.picked {
        None => None,
        Some(PickedBlock::Reused { plane }) => Some(Picked::Reused { plane }),
        Some(PickedBlock::Indexed { plane, index, own }) => {
            let (Some(entries), Some(picked), Some((qr, rank, qr_epsilon, qr_norm))) =
                (kernels.picked, &weights.picked, &qr_rows)
            else {
                return Err(
                    "an indexing index-list source without its entries, weights or low-rank query"
                        .into(),
                );
            };
            Some(Picked::Indexed {
                plane,
                index,
                own,
                inputs: PickedInputs {
                    entries,
                    weights: picked,
                    x: ProjectionInput {
                        rows: &x,
                        width: shape.hidden,
                        form: input.form,
                    },
                    input_norm: &x_norm,
                    epsilon: block.epsilon,
                    qr: ProjectionInput {
                        rows: qr,
                        width: *rank,
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
        &block,
        selection,
        picked,
        join.class,
    )?;
    if block.inject_only {
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
    let (attended, (q, w)) = match (
        operators::attention::grouped_output(block.operator),
        kernels.grouped,
        &weights.grouped,
    ) {
        (None, None, None) => (attended.reshape(&[rows, heads, width]), (heads, width)),
        (Some(grouped), Some(entry), Some(weight)) => {
            let products = graph
                .enqueue(
                    entry,
                    &[
                        ("M", rows),
                        ("G", grouped.groups),
                        ("K", grouped.inputs),
                        ("N", grouped.rank),
                        ("WS", operands.grouped.scale.extent),
                        ("WB", operands.grouped.bound.extent),
                    ],
                    project_rows::WorkflowArgs {
                        source: (&attended.reshape(&[rows, heads * width])).into(),
                        weight: weight.into(),
                        weight_scale: (&operands.grouped.scale.tensor).into(),
                        weight_bound: (&operands.grouped.bound.tensor).into(),
                    },
                )?
                .value;
            (
                products.reshape(&[rows, grouped.groups, grouped.rank]),
                (grouped.groups, grouped.rank),
            )
        }
        _ => return Err("grouped attention output entries, weights and form disagree".into()),
    };
    let mixed = match (kernels.output, &weights.post_norm) {
        (TailEntries::Residual(output), None) => {
            let [streams, injected] = join.dimensions();
            let inject = join.inject(graph, rows, constants)?;
            graph
                .enqueue(
                    output,
                    &[
                        [
                            ("M", rows),
                            ("D", shape.hidden),
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
        (TailEntries::PostNorm(entries), Some(norm)) => {
            let out_rows = GraphConstant::row_identity(graph, rows, Some("M"))?;
            let mixed = post_norm(
                graph,
                entries,
                residual,
                (&attended.reshape(&[rows, q * w])).into(),
                &PackedWeight {
                    weight: output_weight.clone(),
                    operands: operands.output.clone(),
                },
                norm,
                out_rows.port().tensor(),
                PostNormShape {
                    rows,
                    out: rows,
                    inputs: q * w,
                    outputs: shape.hidden,
                },
                block.post_norm_epsilon,
                block.post_norm_scale,
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
    pub slab_rows: u32,
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
    block: &AttentionBlock<'_>,
    selection: Option<SelectionInputs<'a, G>>,
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
    let shape = block.shape;
    let (rows, width) = (block.rows, shape.width);
    // A selected read's class that scores reads through the indexer's
    // selection: its prefill through a bitmap of `B` words of `BR`-key
    // blocks.
    let scoring = selection
        .as_ref()
        .filter(|selection| selection.block.scored);
    let words = scoring.map_or(0, |selection| selection.block.shape.words);
    let block_keys = scoring.map_or(1, |selection| selection.block.shape.block);
    let selected_prefill = scoring.map(|selection| selection.prefill);
    let dense = matches!(&history, AttentionHistoryEntries::Dense { .. });
    if !dense && (shape.key_rounding.rounds() || shape.picks > 0 || picked.is_some()) {
        unreachable!("the binding admits a rounded or index-list source beside a dense history alone");
    }
    // Priming publishes history rows alone, and an index-list source's
    // pooled rows and keys only its attending graph makes.
    if block.inject_only && picked.is_some() {
        return Err("a history injection of an attention with an index-list source".into());
    }
    // The dense entries also take source 0's key rounding (`KR`) and the
    // index-list source's picks (`KS`) of its plane's `TS` rows.
    let dimensions = match &history {
        AttentionHistoryEntries::Dense { .. } => shape.dense_mix_dimensions(
            rows,
            block.history_rows,
            block.segments,
            picked.as_ref().map_or(0, |picked| picked.plane().rows),
        ),
        AttentionHistoryEntries::AffineK8V4 { codec, .. } => shape
            .mix_dimensions(rows, block.history_rows, block.segments)
            .into_iter()
            .chain(affine_dimensions(*codec)?)
            .collect(),
    };
    // The sink rows: the operator's, or none (an injection attends nothing
    // and binds none).
    let sinks = match sinks {
        Some(sinks) => sinks.reshape(&[shape.sinks, shape.heads()]),
        None if shape.sinks == 0 || block.inject_only => GraphConstant::absent(graph, constants)?
            .tensor()
            .reshape(&[0, shape.heads()]),
        None => return Err("attention sink weights are absent".into()),
    };
    let input = |graph: &mut G, name: &str| match &history {
        AttentionHistoryEntries::Dense {
            decode: DenseDecodeEntries::Decode { single, .. },
            ..
        } => graph.control_for(block.controls, *single, name, &dimensions),
        AttentionHistoryEntries::Dense {
            decode: DenseDecodeEntries::Prefill(kernel),
            ..
        } => graph.control_for(
            block.controls,
            *kernel,
            name,
            &prefill_dimensions(&dimensions, words, block_keys),
        ),
        AttentionHistoryEntries::AffineK8V4 { decode, .. } => {
            graph.control_for(block.controls, *decode, name, &dimensions)
        }
    };
    // A class the prefill entry attends is a prefill class here, decode
    // rows among them (`dense_decodes`).
    let decode = decodes(rows)
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
        } if !decode && !block.inject_only && block.history_tiles > 0 => Some(*kernel),
        _ => None,
    };
    let tile_dimensions = dimensions
        .iter()
        .copied()
        .chain([
            ("L", u64::from(listed.is_some())),
            ("HT", listed.map_or(1, |_| block.history_tiles)),
            ("B", words),
            ("BR", block_keys),
        ])
        .collect::<Vec<_>>();
    let history_tiles = listed
        .map(|kernel| {
            Ok::<_, GraphError>(HistoryTilesPort {
                port: graph.input_for(kernel, "history_tiles", &tile_dimensions)?,
                tiles: usize::try_from(block.history_tiles)
                    .map_err(|_| "history tile count exceeds host domain")?,
            })
        })
        .transpose()?;
    let mut controls = AttentionControlPorts {
        coordinates: input(graph, "coordinates")?,
        visible: if block.inject_only {
            None
        } else {
            Some(input(graph, "visible")?)
        },
        fresh: if block.inject_only {
            None
        } else {
            Some(input(graph, "fresh")?)
        },
        destinations: input(graph, "destinations")?,
        history_tiles,
        indexer: None,
    };
    let rotary = &block.operator.rotary;
    let components = GraphConstant::i32(graph, &rotary_components(rotary)?)?;
    let frequencies = GraphConstant::f32(graph, &rotary_frequencies(rotary))?;
    let amplitudes = GraphConstant::f32(graph, &rotary_amplitudes(rotary))?;
    // The unit norm row: the unweighted value norm, and (zero rows of it)
    // every absent head norm.
    let unit = GraphConstant::f32_shaped(graph, &[1, width], &vec![1.0; width as usize])?;
    let absent_norm = unit.port().tensor().slice_leading(0, 0);
    // The rotary layout (RA, RT), output rotary (RO) and stored-key values
    // (VK), and for the dense entries the key rounding (KR), carried by the
    // extents of empty tables; the appends take no output rotary.
    let empty_table = controls.destinations.tensor().slice_leading(0, 0);
    let rounded = dense.then_some(shape.rounds_keys());
    let form = empty_table.reshape(
        &[
            0,
            shape.rotary_adjacent,
            shape.rotary_trailing,
            shape.output_rotary,
            shape.stored_key(),
        ]
        .into_iter()
        .chain(rounded)
        .collect::<Vec<_>>(),
    );
    let append_form = empty_table.reshape(
        &[0, shape.rotary_adjacent, shape.rotary_trailing, shape.stored_key()]
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
            block,
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
    // Only an attending graph reads the picks: a history injection's append
    // reads none, and a graph port no entry reads is unbound.
    let unpicked_plane = match picked {
        Some(_) => None,
        None if dense && !block.inject_only => Some(GraphConstant::empty_port(
            graph,
            block.activation,
            &[0, shape.kv_heads, width],
            constants,
        )?),
        None => None,
    };
    // The affine entries read none.
    let picks = match (picked, &unpicked_plane) {
        (Some(picked), _) => Some((picked.rows.clone(), picked.plane.clone(), picked.slab_rows)),
        (None, Some(plane)) => Some((
            empty_table.reshape(&[rows, 0]),
            plane.tensor().reshape(&[0, shape.kv_heads, width]),
            0,
        )),
        (None, None) => None,
    };
    let key_rounding = shape.key_rounding;
    let head_norm = |weight: &Option<WorkflowTensor>| match weight {
        Some(weight) => weight.reshape(&[1, width]),
        None => absent_norm.clone(),
    };
    let query_norm = head_norm(query_norm);
    // A Shared layer's unread key norm port takes its query norm.
    let key_norm = if shape.fresh == 0 {
        query_norm.clone()
    } else {
        head_norm(key_norm)
    };
    let value_norm = unit.port().tensor().slice_leading(0, shape.value_norm);
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
                &[block.history_rows, shape.kv_heads, descriptor.row_elements as u64],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    // A history without a value plane (stored-key values) binds the dense
    // entries' `history_value` to zero rows of one, which they never touch.
    let mut absent_values = match codec.value {
        HistoryValue::Key => Some(GraphConstant::empty_port(
            graph,
            block.activation,
            &[0, shape.kv_heads, width],
            constants,
        )?),
        HistoryValue::Own { .. } | HistoryValue::None => None,
    };
    let scale = block.operator.scale as f32;
    let gate_function = operators::attention::gate_function(block.operator);
    let mut destinations = controls.destinations.tensor().clone();
    // What the entry reads: the host's visible spans and fresh rows, or for
    // a selected read's decode rows the indexer's segments.
    let mut visible = controls
        .visible
        .as_ref()
        .map(|port| port.tensor().reshape(&[rows, block.segments, 2]));
    let fresh = controls
        .fresh
        .as_ref()
        .map(|port| port.tensor().reshape(&[rows, 2]));
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
                        form: (&form).into(),
                        epsilon: block.head_epsilon,
                        scale,
                        gate_function,
                        slab_rows: block.slab_rows,
                    },
                )?

        }};
    }
    // State-only entries bind no query, gate, visibility or selection values.
    macro_rules! append {
        ($kernel:expr, $module:ident, {$($extra:ident: $value:expr),*}, $($plane:ident),*) => {{
            let [$($plane),*] = entry_planes(&mut planes, absent_values.as_mut())?;
            let append_dimensions = dimensions.iter().copied()
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
                epsilon: block.head_epsilon, slab_rows: block.slab_rows,
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
                key_amax_floor: key_rounding.amax_floor()
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
                key_amax_floor: key_rounding.amax_floor()
                $(, $extra: $value)*
            }, history_key, history_value)
        }};
    }
    // Priming only publishes context K/V. Select the state-only entry at
    // graph formation to omit attention/merge launches and their scratch
    // buffers, even for a large prompt chunk.
    if block.inject_only {
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
                indexer: None,
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
                block,
                &controls,
                [
                    components.port().tensor(),
                    frequencies.port().tensor(),
                    amplitudes.port().tensor(),
                ],
                constants,
            )?;
            controls.indexer = Some(tables);
            indexer_planes = Some(planes);
            selection
                .block
                .scored
                .then_some((selected, selection.block.shape))
        }
    };
    // A selected decode row reads the indexer's segments of its visible
    // spans (its launch's earlier rows among them, `cached_causal_domain`)
    // and its own row fresh, as a dense decode row does.
    if let (Some((selected, binding)), true) = (&selected, decode) {
        let segments = selected_segments(binding.blocks);
        visible = Some(selected.segments.reshape(&[rows, segments, 2]));
        for (name, value) in &mut attend_dimensions {
            if *name == "R" {
                *value = segments;
            }
        }
    }
    // Earlier causal rows are already in the visible cache spans (a
    // compressed history's, or a selected read's). Decode's fused append
    // cannot publish rows other threadgroups read in the same launch, so
    // publish once first and suppress its later append. Prefill already has
    // an ordered prepare launch and needs no additional phase.
    if block.causal_cache && decode && rows > 1 && shape.fresh > 0 {
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
            let disabled = GraphConstant::row_filled(graph, rows, -1)?;
            destinations = disabled.port().tensor().clone();
            constants.push(disabled);
        }
    }
    // A prefill's WorkList block keys (`BR`), carried by an extent alone.
    let block_slots = controls
        .destinations
        .tensor()
        .slice_leading(0, 0)
        .reshape(&[0, block_keys]);
    let attended = match &history {
        AttentionHistoryEntries::Dense {
            decode:
                DenseDecodeEntries::Decode {
                    verify: Some(kernel),
                    ..
                },
            ..
        } if decode && rows > 1 => {
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
                (None, DenseDecodeEntries::Prefill(kernel)) if decodes(rows) => kernel,
                (None, _) => prefill,
                (Some(_), _) => return Err("a selected prefill of another codec".into()),
            };
            let bitmap = prefill_selection(graph, rows, &selected, row_class, constants)?;
            dense_mix!(kernel, attention_prefill,
                &prefill_dimensions(&dimensions, words, block_keys),
                selection: (&bitmap).into(),
                block_slots: (&block_slots).into())
        }
        AttentionHistoryEntries::AffineK8V4 {
            verify_four: Some(kernel),
            ..
        } if decode && rows == 4 => mix!(
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
        } if decode && rows > 4 => mix!(
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
        } if decode && rows > 1 => mix!(
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
                .reshape(&[0, 1]);
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

/// A prefill's WorkList bitmap: a selected read's
/// (`indexer_select`), or none (`no_selection`, every visible row).
fn prefill_selection<G: GraphDraft>(
    graph: &mut G,
    rows: u64,
    selected: &Option<(Selected, SelectionShape)>,
    row_class: Option<&'static str>,
    constants: &mut Vec<GraphConstant>,
) -> Result<WorkflowTensor, GraphError> {
    match selected {
        Some((selected, _)) => Ok(selected.bitmap.clone()),
        None => no_selection(graph, rows, row_class, constants),
    }
}

/// The prefill entries' dimensions: the fused entries', `B`, the words of a
/// row's WorkList bitmap (0 in the Visible form), and `BR`, the keys of its
/// blocks (1 in the Visible form).
fn prefill_dimensions(
    dimensions: &[(&'static str, u64)],
    words: u64,
    block: u64,
) -> Vec<(&'static str, u64)> {
    dimensions
        .iter()
        .copied()
        .chain([("B", words), ("BR", block)])
        .collect()
}

/// A Visible prefill's absent WorkList bitmap: `[rows, 0]`, its rows the
/// class's.
fn no_selection<G: GraphDraft>(
    graph: &mut G,
    rows: u64,
    row_class: Option<&'static str>,
    constants: &mut Vec<GraphConstant>,
) -> Result<WorkflowTensor, GraphError> {
    let selection = GraphConstant::row_zeros(graph, Element::u32(), rows, 0, row_class)?;
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
    /// entry on Metal (its matrix form, in head groups of 8) and on CPU, as
    /// a Qwen-shaped layer does; CUDA's and Vulkan's decode entries hold
    /// none of the new forms, so CUDA attends them with its prefill entry.
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
            history: match shape.values {
                AttentionValues::StoredKey => {
                    CodecSpec::stored_key(DType::BF16, shape.width as usize)
                }
                AttentionValues::Projected | AttentionValues::RawKey => {
                    CodecSpec::dense(DType::BF16, shape.width as usize, shape.width as usize)
                }
            },
            tail: SublayerTail::Residual,
            selection: None,
            grouped: None,
            picked: None,
        };
        let v41 = AttentionShape {
            hidden: 4096,
            kv_heads: 1,
            group: 64,
            rotary_pairs: 32,
            width: 512,
            interleaved_gate: 0,
            separate_gate: 0,
            fresh: 1,
            query_norm: 0,
            key_norm: 1,
            value_norm: 0,
            values: AttentionValues::StoredKey,
            sinks: 1,
            rotary_adjacent: 1,
            rotary_trailing: 1,
            output_rotary: 1,
            key_rounding: crate::RoundingArgs::NONE,
            picks: 0,
        };
        let qwen = AttentionShape {
            kv_heads: 2,
            group: 8,
            width: 256,
            interleaved_gate: 256,
            query_norm: 1,
            values: AttentionValues::Projected,
            sinks: 0,
            rotary_adjacent: 0,
            rotary_trailing: 0,
            output_rotary: 0,
            ..v41
        };
        let decodes = |backend, shape| dense_decodes(backend, binding(shape)).unwrap();
        assert!(decodes(BackendName::Metal, v41));
        assert!(decodes(BackendName::Cpu, v41));
        assert!(decodes(BackendName::Metal, qwen));
        assert!(!decodes(BackendName::Cuda, v41));
        assert!(decodes(BackendName::Cuda, qwen));
        // Each new form alone (a query norm without a key norm, adjacent
        // pairs, trailing pairs, the output rotary, V4.1's own): CUDA's
        // decode entry declares none of them and its prefill entry all of
        // them; Vulkan's mix entries declare none.
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
            AttentionShape { query_norm: 0, ..qwen },
            AttentionShape { rotary_adjacent: 1, ..qwen },
            AttentionShape { rotary_trailing: 1, ..qwen },
            AttentionShape { interleaved_gate: 0, output_rotary: 1, ..qwen },
            v41,
        ];
        for backend in [BackendName::Cuda, BackendName::Vulkan] {
            assert_eq!(holds(backend, qwen), (true, true), "{backend:?}");
        }
        for form in forms {
            assert_eq!(holds(BackendName::Cuda, form), (false, true), "{form:?}");
            assert_eq!(holds(BackendName::Vulkan, form), (false, false), "{form:?}");
            assert_eq!(holds(BackendName::Cpu, form), (true, true), "{form:?}");
        }
    }
}
