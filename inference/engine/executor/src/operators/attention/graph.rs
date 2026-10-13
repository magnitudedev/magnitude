//! Attention block graph: the normed query | gate | key | value projection
//! (`attention_project`), the fused attention entry of the history codec
//! (head norms, rotary, K/V append at the rows' destinations, attention over
//! the visible history spans and fresh rows, output gate) and the output
//! projection plus residual (or its post-norm tail). Decode row classes use the codec's decode entry
//! (`attention_decode`, `attention_decode_k8v4`); larger classes its prefill
//! entry. Both share one contract, so their ports have the same geometry.
//!
//! Every optional part of the operator is an axis of extent 0 or 1
//! ([`AttentionShape`]): an absent projection segment binds a zero-row view
//! of the query weight, an absent head norm a zero-row view of the block's
//! unit norm row. The rotary table (the coordinate axis, frequency and
//! amplitude of every rotated pair) and the unit row are graph constants,
//! bound once with the weights.

use crate::operators::output::{CheckedPostNormEntries, PostNormShape, TailEntries, post_norm};
use crate::operators::{self, InputForm, Join, SublayerInput};
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_target_graph::ScaledWeight;
use crate::{AttentionBinding, AttentionShape, SelectionShape, SublayerTail};
use crate::{
    native::{AttentionHistoryKernels, AttentionKernels, SelectedPrefillKernel},
    programs::native_constants::GraphConstant,
};
use magnitude_family_contracts::{Attention, HistoryReads, Rotary, WeightKind};
use magnitude_kernels::{
    attention_append_dense, attention_append_k8v4, attention_decode, attention_decode_k8v4,
    attention_output, attention_prefill, attention_prefill_k8v4, attention_project, indexer_select,
};
use magnitude_state::{CodecSpec, HistoryEncoding};
use seismic::{
    Element, NativeGraph, NativeGraphClassSlice, NativeGraphMetadata, NativePort, WorkflowTensor,
    WorkflowTensorRef, WorkflowTensorView,
};

pub(crate) struct AttentionGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, attention_project::Entry>,
    pub history: AttentionHistoryEntries<'a, G>,
    pub output: TailEntries<'a, G, attention_output::Entry>,
    /// A selected read's indexer entries.
    pub selection: Option<SelectionGraphEntries<'a, G>>,
}

impl<'a, G: GraphDraft + 'a> Copy for AttentionGraphEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for AttentionGraphEntries<'a, G> {
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
        append: G::Binding<'a, attention_append_dense::Entry>,
        decode: G::Binding<'a, attention_decode::Entry>,
        verify: Option<G::Binding<'a, attention_decode::Entry>>,
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

impl<'a> From<&'a AttentionKernels> for AttentionGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a AttentionKernels) -> Self {
        let history = match &kernels.history {
            AttentionHistoryKernels::Dense {
                append,
                decode,
                verify,
                prefill,
            } => AttentionHistoryEntries::Dense {
                append,
                decode,
                verify: verify.as_ref(),
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
        }
    }
}

/// Entry element assignments from the same program binding used by native
/// preparation. The checked graph obtains all shapes from the entries.
pub(crate) struct CheckedAttentionEntries {
    project: [(&'static str, Element); 6],
    mix: [(&'static str, Element); 1],
    output: [(&'static str, Element); 2],
    post_norm: Option<CheckedPostNormEntries>,
    history: CodecSpec,
    /// Whether the graphs may list their launch's history row tiles
    /// (`StateResourcePlan::lists_history_tiles`, for a program whose
    /// classes have the listing axis).
    lists: bool,
    /// A selected read's indexer projection and selection.
    indexer_project: Option<[(&'static str, Element); 6]>,
    indexer_select: [(&'static str, Element); 1],
}

impl CheckedAttentionEntries {
    pub(crate) fn new(binding: AttentionBinding, lists: bool) -> Self {
        Self {
            lists: lists && lists_history_tiles(binding),
            indexer_project: binding.selection.map(|selection| {
                [
                    ("NW", binding.norm),
                    ("QW", selection.query),
                    ("GW", selection.query),
                    ("KW", selection.key),
                    ("VW", selection.query),
                    ("A", binding.activation),
                ]
            }),
            // The indexer reads its projections in the activation element
            // and keeps everything else in F32 (`indexer_select`).
            indexer_select: [("A", binding.activation)],
            project: [
                ("NW", binding.norm),
                ("QW", binding.query),
                ("GW", binding.gate),
                ("KW", binding.key),
                ("VW", binding.value),
                ("A", binding.activation),
            ],
            mix: [("A", binding.activation)],
            output: [("OW", binding.output), ("A", binding.activation)],
            post_norm: match binding.tail {
                SublayerTail::Residual => None,
                SublayerTail::PostNorm { norm, .. } => Some(CheckedPostNormEntries::new(
                    binding.output,
                    binding.activation,
                    norm,
                )),
            },
            history: binding.history,
        }
    }

    pub(crate) fn entries(&self) -> Result<AttentionGraphEntries<'_, NativeGraphMetadata>, String> {
        let history = match self.history.encoding() {
            HistoryEncoding::Dense => AttentionHistoryEntries::Dense {
                append: &self.mix[..],
                decode: &self.mix[..],
                verify: None,
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
        })
    }
}

/// Whether `binding`'s prefill graphs may list their launch's history row
/// tiles where its device reads decoded history: an affine history read
/// whole (a selected read's WorkList prefill lists none).
pub(crate) fn lists_history_tiles(binding: AttentionBinding) -> bool {
    binding.history.encoding() == HistoryEncoding::Affine && binding.selection.is_none()
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
pub(crate) struct AttentionWeights {
    /// The input norm; absent for a mix site's partials.
    pub input_norm: Option<WorkflowTensor>,
    /// The query projection (with interleaved gate rows when gated so).
    pub query: WorkflowTensor,
    pub gate: Option<WorkflowTensor>,
    pub key: Option<WorkflowTensor>,
    pub value: Option<WorkflowTensor>,
    pub query_norm: Option<WorkflowTensor>,
    pub key_norm: Option<WorkflowTensor>,
    pub output: Option<WorkflowTensor>,
    /// The sublayer's post-norm weight, when its tail is a post-norm.
    pub post_norm: Option<WorkflowTensor>,
    /// A selected read's indexer weights.
    pub indexer: Option<IndexerWeights>,
}

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

/// The class slice of a block whose attention reads `reads`: a selected
/// read's decode entries read its `C` segments whatever the class's history
/// segments (`selected_segments`).
pub(crate) fn decode_class_segments(
    slice: NativeGraphClassSlice,
    reads: &HistoryReads,
) -> NativeGraphClassSlice {
    match reads {
        HistoryReads::Visible => slice,
        HistoryReads::Selected { blocks, .. } => {
            let segments = selected_segments(*blocks);
            slice.scoped("attention_decode", "R", [segments]).scoped(
                "attention_decode_k8v4",
                "R",
                [segments],
            )
        }
    }
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
    let mut input = |name: &str| graph.input_for(selection.select, name, &dimensions);
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
    let HistoryReads::Selected { indexer, .. } = &block.operator.reads else {
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
/// form, its input `form` and its sublayer tail define, from a lookup of a
/// role's port.
pub(crate) fn attention_weights(
    shape: &AttentionShape,
    operator: &Attention,
    form: InputForm,
    needs_output: bool,
    post_norm: bool,
    mut weight: impl FnMut(WeightKind) -> Result<WorkflowTensor, GraphError>,
) -> Result<AttentionWeights, GraphError> {
    let mut present = |present: bool, kind| present.then(|| weight(kind)).transpose();
    Ok(AttentionWeights {
        input_norm: present(form.normalized, WeightKind::InputNorm)?,
        query: present(true, operators::attention::query_kind(operator))?.ok_or("query")?,
        gate: present(shape.gate_rows() > 0, WeightKind::AttentionGate)?,
        key: present(shape.key_rows() > 0, WeightKind::Key)?,
        value: present(shape.value_rows() > 0, WeightKind::Value)?,
        query_norm: present(needs_output && shape.head_norm > 0, WeightKind::QueryNorm)?,
        // A Shared layer projects no keys, so it has no key norm.
        key_norm: present(shape.head_norm > 0 && shape.fresh > 0, WeightKind::KeyNorm)?,
        output: present(needs_output, WeightKind::AttentionOutput)?,
        post_norm: present(post_norm, WeightKind::PostNorm)?,
        indexer: match operator.reads {
            HistoryReads::Visible => None,
            HistoryReads::Selected { .. } => Some(IndexerWeights {
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
        None => GraphConstant::absent_scale(graph, constants)?,
    };
    // An absent projection segment reads zero rows of the query weight.
    let empty_weight = weights.query.slice_leading(0, 0);
    let empty_gate_weight = weights
        .gate
        .as_ref()
        .map(|gate| gate.slice_leading(0, 0))
        .unwrap_or_else(|| empty_weight.clone());
    let projected = graph.enqueue(
        kernels.project,
        &[
            if block.inject_only {
                shape.key_value_dimensions(rows)
            } else {
                shape.project_dimensions(rows)
            }
            .as_slice(),
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
        },
    )?;
    // The mix entries read the projection per head.
    let projected = ProjectedRows {
        query: projected.r0.reshape(&[
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
        value: if shape.projected_value {
            projected
                .r3
                .reshape(&[shape.fresh, rows, shape.kv_heads * width])
        } else {
            projected
                .r2
                .reshape(&[shape.fresh, rows, shape.kv_heads * width])
        },
    };
    // A selected read's indexer projects its query heads and raw key from
    // the same input under the same norm.
    let selection = match (kernels.selection, &weights.indexer, block.selection) {
        (None, None, None) => None,
        (Some(entries), Some(indexer), Some(selection)) => {
            let binding = selection.shape;
            let empty = indexer.query.slice_leading(0, 0);
            let projected = graph.enqueue(
                entries.project,
                &[
                    indexer_project_dimensions(shape.hidden, binding, rows).as_slice(),
                    &input.form.dimensions(shape.hidden),
                ]
                .concat(),
                attention_project::WorkflowArgs {
                    hidden: input.rows.into(),
                    input_norm: (&input_norm).into(),
                    query_weight: (&indexer.query).into(),
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
    let (attended, state, controls) = mix(
        graph,
        kernels.history,
        (&weights.query_norm, &weights.key_norm),
        constants,
        &projected,
        &block,
        selection,
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
    let mixed = match (kernels.output, &weights.post_norm) {
        (TailEntries::Residual(output), None) => {
            let [streams, injected] = join.dimensions();
            let inject = join.inject(graph, rows, constants)?;
            graph
                .enqueue(
                    output,
                    &[
                        ("M", rows),
                        ("D", shape.hidden),
                        ("Q", heads),
                        ("W", width),
                        streams,
                        injected,
                    ],
                    attention_output::WorkflowArgs {
                        hidden: residual.into(),
                        inject: (&inject).into(),
                        gated: (&attended).into(),
                        output_weight: output_weight.into(),
                    },
                )?
                .value
        }
        (TailEntries::PostNorm(entries), Some(norm)) => {
            let out_rows = GraphConstant::identity_for_class(graph, rows, Some("M"))?;
            let absent_scale = GraphConstant::absent_scale(graph, constants)?;
            let mixed = post_norm(
                graph,
                entries,
                residual,
                (&attended.reshape(&[rows, heads * width])).into(),
                &ScaledWeight::unscaled(output_weight.clone(), absent_scale),
                norm,
                out_rows.port().tensor(),
                PostNormShape {
                    rows,
                    out: rows,
                    inputs: heads * width,
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
/// the visible spans and fresh rows. Its rotary table and unit norm row join
/// `constants`.
pub(crate) fn mix<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    history: AttentionHistoryEntries<'a, G>,
    (query_norm, key_norm): (&Option<WorkflowTensor>, &Option<WorkflowTensor>),
    constants: &mut Vec<GraphConstant>,
    projected: &ProjectedRows,
    block: &AttentionBlock<'_>,
    selection: Option<SelectionInputs<'a, G>>,
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
    let mut dimensions = shape
        .mix_dimensions(rows, block.history_rows, block.segments)
        .to_vec();
    if let AttentionHistoryEntries::AffineK8V4 { codec, .. } = &history {
        dimensions.extend(affine_dimensions(*codec)?);
    }
    let input = |graph: &mut G, name: &str| match &history {
        AttentionHistoryEntries::Dense { decode, .. } => {
            graph.input_for(*decode, name, &dimensions)
        }
        AttentionHistoryEntries::AffineK8V4 { decode, .. } => {
            graph.input_for(*decode, name, &dimensions)
        }
    };
    let decode = decodes(rows);
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
    let head_norm = |weight: &Option<WorkflowTensor>| match weight {
        Some(weight) => weight.reshape(&[1, width]),
        None => absent_norm.clone(),
    };
    let query_norm = head_norm(query_norm);
    // The entries take equal query and key norm rows; a Shared layer's
    // unread key norm port takes its query norm.
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
    let mut plane = |element: Element, elements: u64| {
        graph.port(element, &[block.history_rows, shape.kv_heads, elements])
    };
    let mut planes = match &history {
        AttentionHistoryEntries::Dense { .. } => vec![
            plane(block.activation, width)?,
            plane(block.activation, width)?,
        ],
        AttentionHistoryEntries::AffineK8V4 { codec, .. } => codec
            .planes()
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|descriptor| {
                plane(
                    Element::from(descriptor.dtype),
                    descriptor.row_elements as u64,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
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
            let [$($plane),*] = planes.as_mut_slice() else {
                return Err("attention history planes disagree with the entry".into());
            };
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
                        rotary_components: components.port().tensor().into(),
                        rotary_frequencies: frequencies.port().tensor().into(),
                        rotary_amplitudes: amplitudes.port().tensor().into(),
                        coordinates: controls.coordinates.tensor().into(),
                        visible: visible.as_ref().ok_or("attention visibility is absent")?.into(),
                        fresh: fresh.as_ref().ok_or("attention fresh rows are absent")?.into(),
                        destinations: (&destinations).into(),
                        $($extra: $value,)*
                        $($plane: $plane.tensor_mut().into(),)*
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
        ($kernel:expr, $module:ident, $($plane:ident),*) => {{
            let [$($plane),*] = planes.as_mut_slice() else {
                return Err("attention history planes disagree with the entry".into());
            };
            let append_dimensions = dimensions.iter().copied()
                .filter(|(name, _)| !matches!(*name, "R" | "G" | "I" | "U")).collect::<Vec<_>>();
            graph.enqueue(*$kernel, &append_dimensions, $module::WorkflowArgs {
                key: (&key).into(), value: (&value).into(),
                key_norm: (&key_norm).into(), value_norm: (&value_norm).into(),
                rotary_components: components.port().tensor().into(),
                rotary_frequencies: frequencies.port().tensor().into(),
                rotary_amplitudes: amplitudes.port().tensor().into(),
                coordinates: controls.coordinates.tensor().into(),
                destinations: (&destinations).into(),
                $($plane: $plane.tensor_mut().into(),)*
                epsilon: block.head_epsilon, slab_rows: block.slab_rows,
            })?;
        }};
    }
    // Priming only publishes context K/V. Select the state-only entry at
    // graph formation to omit attention/merge launches and their scratch
    // buffers, even for a large prompt chunk.
    if block.inject_only {
        match &history {
            AttentionHistoryEntries::Dense { append, .. } => {
                append!(append, attention_append_dense, history_key, history_value);
            }
            AttentionHistoryEntries::AffineK8V4 { append, .. } => {
                append!(
                    append,
                    attention_append_k8v4,
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
                append!(append, attention_append_dense, history_key, history_value);
                true
            }
            AttentionHistoryEntries::Dense { .. } => false,
        };
        if published {
            let disabled = GraphConstant::i32(graph, &vec![-1; rows as usize])?;
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
            verify: Some(kernel),
            ..
        } if decode && rows > 1 => {
            mix!(kernel, attention_decode, history_key, history_value)
        }
        AttentionHistoryEntries::Dense { decode: kernel, .. } if decode => {
            mix!(kernel, attention_decode, history_key, history_value)
        }
        AttentionHistoryEntries::Dense {
            prefill: kernel, ..
        } => {
            let kernel = match &selected_prefill {
                Some(SelectedPrefillEntries::Dense(selected)) => selected,
                None => kernel,
                Some(_) => return Err("a selected prefill of another codec".into()),
            };
            let bitmap = prefill_selection(graph, rows, &selected, row_class, constants)?;
            mix!(@enqueue kernel, attention_prefill,
                &prefill_dimensions(&dimensions, words, block_keys), {
                selection: (&bitmap).into(),
                block_slots: (&block_slots).into()
            }, history_key, history_value)
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
        },
        controls,
    ))
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
/// class's (a row-class constant, `block::class_constants_of`).
fn no_selection<G: GraphDraft>(
    graph: &mut G,
    rows: u64,
    row_class: Option<&'static str>,
    constants: &mut Vec<GraphConstant>,
) -> Result<WorkflowTensor, GraphError> {
    let selection = match row_class {
        Some(class) => GraphConstant::u32_zeros_for_class(graph, rows, 0, class)?,
        None => GraphConstant::u32_zeros(graph, rows, 0)?,
    };
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
}
