//! The graph of a layer's own indexer (`Selection::Own(Indexer)`) over a
//! pooled source whose key plane shares the source's `Block` domain: the
//! keys of the rows the producer completes (`indexer_keys`, from the key
//! projection of the producer's pre-rotary latent; none for keys shared
//! with the indexer that writes them), each row's scores over its row list
//! (`indexer_score`, from the query projection of `qr` or `x` and, when
//! projected, the head weights of `x`), and its picks (`rows_topk`): the
//! `selected_rows` port attention reads, `[rows, top]` plane rows ascending
//! by position, -1 padded, indexed by class row.
//!
//! A row's list is its visible spans of the Block read, or the candidate
//! pool an earlier indexer over the same rows produced from its scores
//! ([`candidate_pool`]: `candidate_blocks`, `rows_topk` over the blocks,
//! `candidate_rows`).
//!
//! Every projection is `attention_project`'s first segment over its own
//! input under its own norm (the query's `norm`, the input norm for `x`,
//! none for the latent, the producer's for rate-1 tokens), published in the activation element, from which
//! the indexer entries compute in F32.

use crate::CandidatePool;
use crate::operators::InputForm;
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::WeightOperands;
use crate::planning::RoundingArgs;
use magnitude_family_contracts::{HeadNorm, Indexer, IndexerKeys};
use magnitude_kernels::{
    attention_project, candidate_blocks, candidate_rows, indexer_keys, indexer_score, rows_topk,
};
use seismic::{NativePort, WorkflowTensor, WorkflowTensorView};

/// The indexer's entries at one graph class.
pub(crate) struct IndexerEntries<'a, G: GraphDraft + 'a> {
    /// Its projections, each bound to its own weight's and input norm's
    /// elements: the query's and, for projected head weights, the head
    /// weights'.
    pub query_project: G::Binding<'a, attention_project::Entry>,
    pub weights_project: Option<G::Binding<'a, attention_project::Entry>>,
    pub score: G::Binding<'a, indexer_score::Entry>,
    pub topk: G::Binding<'a, rows_topk::Entry>,
    /// A producing indexer's candidate pool ([`candidate_pool`]).
    pub candidates: Option<CandidateEntries<'a, G>>,
}

impl<'a, G: GraphDraft + 'a> Copy for IndexerEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for IndexerEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// The candidate pool's entries: its block scores and its rows (the
/// blocks are picked by the indexer's `rows_topk`).
pub(crate) struct CandidateEntries<'a, G: GraphDraft + 'a> {
    pub blocks: G::Binding<'a, candidate_blocks::Entry>,
    pub rows: G::Binding<'a, candidate_rows::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for CandidateEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for CandidateEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A projection's weight, its operands and the norm of its input rows
/// (`[width]`, or a zero-row view for none).
pub(crate) struct Projection<'v> {
    pub weight: &'v WorkflowTensor,
    pub operands: &'v WeightOperands,
    pub norm: &'v WorkflowTensorView,
    pub epsilon: f32,
}

/// One projection input: F32 rows `[rows, parts · width]` in `form`.
pub(crate) struct ProjectionInput<'v> {
    pub rows: &'v WorkflowTensorView,
    pub width: u64,
    pub form: InputForm,
}

/// The write of the keys of the rows a producer completes, by the indexer
/// keeping its own (`IndexerKeys::Latent`).
pub(crate) struct KeyWrite<'v, 'a, G: GraphDraft + 'a> {
    /// The keys' trained rounding (`IndexBlock::key_rounding`).
    pub rounding: RoundingArgs,
    /// The key projection, bound to its weight's element, and the write.
    pub project: G::Binding<'a, attention_project::Entry>,
    pub write: G::Binding<'a, indexer_keys::Entry>,
    /// The F32 rows `[rows, W]` the key projection forms the producer's
    /// pre-rotary latent from (the latent itself, or at rate 1 the tokens'
    /// rows, whose input norm is the producer's), the key-plane row each
    /// row completes (`[rows]`, -1 for none) and that unit's first token's
    /// coordinates (`[rows, 4]`).
    pub latent: ProjectionInput<'v>,
    pub latent_rows: &'v WorkflowTensorView,
    pub latent_coordinates: &'v WorkflowTensorView,
    /// The key projection (under the latent's input norm, if any) and the
    /// keys' norm `[1, width]`.
    pub key: Projection<'v>,
    pub key_norm: &'v WorkflowTensorView,
}

/// The indexer's inputs at one graph class.
pub(crate) struct IndexerInputs<'v, 'a, G: GraphDraft + 'a> {
    /// The query projection's input (`qr` or `x`, as the indexer's
    /// `IndexerInput` names) and its projection under the query's norm.
    pub query_input: ProjectionInput<'v>,
    pub query: Projection<'v>,
    /// The query heads' own norm `[1, width]` (`IndexerQuery::norm`), if any.
    pub query_norm: Option<&'v WorkflowTensorView>,
    /// The query rows' trained rounding (`IndexBlock::query_rounding`).
    pub query_rounding: RoundingArgs,
    /// The layer's input `x` and the head weights' projection under the
    /// input norm, for `HeadWeights::Projected`.
    pub head_weights: Option<(ProjectionInput<'v>, Projection<'v>)>,
    /// The keys' write; none for keys the indexer shares.
    pub write: Option<KeyWrite<'v, 'a, G>>,
    /// The query rows' coordinates `[rows, 4]` and the attention's rotary
    /// table (components, frequencies, amplitudes) in its layout.
    pub coordinates: &'v WorkflowTensorView,
    pub rotary: [&'v WorkflowTensor; 3],
    pub rotary_adjacent: u64,
    pub rotary_trailing: u64,
    /// The key plane, the pooled source's Block-domain component.
    pub keys: &'v mut NativePort,
    /// Each row's list over the key plane (`[rows, R, 2]` spans: the
    /// source's visible rows, or a candidate pool's) and its complete units
    /// (`[rows]`).
    pub spans: &'v WorkflowTensorView,
    pub complete: &'v WorkflowTensorView,
}

/// The extents of one class of the indexer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IndexerClass {
    pub rows: u64,
    /// Spans per row of its lists (`R`).
    pub spans: u64,
    /// Scores per row (`N`): at least the longest list a row of the class
    /// can have.
    pub scores: u64,
    /// The key plane's rows and slab rows.
    pub plane_rows: u64,
    pub slab_rows: u32,
    /// The rotary pairs of the attention's head width.
    pub rotary_pairs: u64,
}

/// What the indexer of one class computes: each row's scores over its list
/// (`[rows, N]` F32), and its picks (`[rows, top]` i32).
pub(crate) struct Indexed {
    pub scores: WorkflowTensor,
    pub picks: WorkflowTensor,
}

/// One projection of `input` through `attention_project`'s query segment:
/// `[rows, out]` in the activation element.
fn project<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entry: G::Binding<'a, attention_project::Entry>,
    rows: u64,
    out: u64,
    input: &ProjectionInput<'_>,
    projection: &Projection<'_>,
    constants: &mut Vec<GraphConstant>,
) -> Result<WorkflowTensor, GraphError> {
    let empty = projection.weight.slice_leading(0, 0);
    let absent = GraphConstant::absent(graph, constants)?.tensor().clone();
    let operands = projection.operands;
    let dimensions = [
        ("M", rows),
        ("D", input.width),
        ("Q", out),
        ("GR", 0),
        ("K", 0),
        ("V", 0),
        ("QS", operands.scale.extent),
        ("GS", 0),
        ("KS", 0),
        ("VS", 0),
        ("QB", operands.bound.extent),
        ("GB", 0),
        ("KB", 0),
        ("VB", 0),
    ];
    Ok(graph
        .enqueue(
            entry,
            &[dimensions.as_slice(), &input.form.dimensions(input.width)].concat(),
            attention_project::WorkflowArgs {
                hidden: input.rows.into(),
                input_norm: projection.norm.into(),
                query_weight: projection.weight.into(),
                gate_weight: (&empty).into(),
                key_weight: (&empty).into(),
                value_weight: (&empty).into(),
                epsilon: projection.epsilon,
                project_mode: 0,
                query_scale: (&operands.scale.tensor).into(),
                gate_scale: (&absent).into(),
                key_scale: (&absent).into(),
                value_scale: (&absent).into(),
                query_bound: (&operands.bound.tensor).into(),
                gate_bound: (&absent).into(),
                key_bound: (&absent).into(),
                value_bound: (&absent).into(),
            },
        )?
        .r0)
}

/// The indexer `indexer` of one class: writes the keys of the rows the
/// producer completes (an indexer keeping its own), scores each row's list
/// and picks from it, the `selected_rows` attention reads (`[rows, top]`
/// i32).
pub(crate) fn indexer<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    class: IndexerClass,
    indexer: &Indexer,
    entries: IndexerEntries<'a, G>,
    inputs: IndexerInputs<'_, 'a, G>,
    constants: &mut Vec<GraphConstant>,
) -> Result<Indexed, GraphError> {
    let (rows, heads, width) = (class.rows, indexer.heads, indexer.width);
    let pairs = class.rotary_pairs;
    let rotary = [
        ("P", pairs),
        ("S", width - 2 * pairs),
        ("RA", inputs.rotary_adjacent),
        ("RT", inputs.rotary_trailing),
    ];
    // Extent-only tensors: zero rows of the complete units' table.
    let none = inputs.complete.slice_leading(0, 0);
    let form = none.reshape(&[0, inputs.rotary_adjacent, inputs.rotary_trailing]);
    let [components, frequencies, amplitudes] = inputs.rotary;
    // The query heads' own norm (`[1, width]`) and its epsilon, or none
    // (zero rows of the absent F32 operand).
    let (query_norm, query_epsilon) = match (&indexer.query.norm, inputs.query_norm) {
        (HeadNorm::Rms(norm), Some(weight)) => (weight.clone(), norm.epsilon as f32),
        (HeadNorm::None, None) => (
            GraphConstant::absent(graph, constants)?.tensor().reshape(&[0, width]),
            0.0,
        ),
        _ => return Err("an indexer's query norm and its weight disagree".into()),
    };
    let normed = u64::from(matches!(indexer.query.norm, HeadNorm::Rms(_)));
    // The completed rows' keys, before any row scores them: an indexer
    // keeping its own writes them, one sharing them reads its owner's.
    match (&indexer.keys, &inputs.write) {
        (IndexerKeys::Latent { norm, .. }, Some(write)) => {
            let rounding = write.rounding;
            let key = project(graph, write.project, rows, width, &write.latent, &write.key, constants)?;
            graph.enqueue(
                write.write,
                &[[("M", rows), ("TB", class.plane_rows)].as_slice(), &rotary].concat(),
                indexer_keys::WorkflowArgs {
                    key: (&key).into(),
                    key_norm: write.key_norm.into(),
                    latent_rows: write.latent_rows.into(),
                    rotary_components: components.into(),
                    rotary_frequencies: frequencies.into(),
                    rotary_amplitudes: amplitudes.into(),
                    coordinates: write.latent_coordinates.into(),
                    keys: inputs.keys.tensor_mut().into(),
                    form: (&form).into(),
                    epsilon: norm.epsilon as f32,
                    block: rounding.block,
                    element: rounding.element,
                    scale: rounding.scale,
                    amax_floor: rounding.amax_floor(),
                    slab_rows: class.slab_rows,
                },
            )?;
        }
        (IndexerKeys::Shared { .. }, None) => {}
        (IndexerKeys::Latent { .. } | IndexerKeys::Shared { .. }, _) => {
            return Err("an indexer's keys and their write disagree".into());
        }
        (IndexerKeys::Pooled(_), _) => {
            unreachable!("picked_source refuses pooled indexer keys over a pooled source")
        }
    }
    let query = project(graph, entries.query_project, rows, heads * width, &inputs.query_input, &inputs.query, constants)?
        .reshape(&[rows, heads, width]);
    let head_weights = match (&inputs.head_weights, entries.weights_project) {
        (Some((input, projection)), Some(entry)) => {
            project(graph, entry, rows, heads, input, projection, constants)?.reshape(&[rows, heads])
        }
        (None, None) => query.slice_leading(0, 0).reshape(&[rows, 0]),
        _ => return Err("an indexer's head-weight projection and its inputs disagree".into()),
    };
    let query_rounding = inputs.query_rounding;
    let scores = class.scores;
    let score_slots = none.reshape(&[0, scores]);
    let score = graph
        .enqueue(
            entries.score,
            &[
                [
                    ("M", rows),
                    ("H", heads),
                    ("NQ", normed),
                    ("HW", if inputs.head_weights.is_some() { heads } else { 0 }),
                    ("TB", class.plane_rows),
                    ("R", class.spans),
                    ("N", scores),
                ]
                .as_slice(),
                &rotary,
            ]
            .concat(),
            indexer_score::WorkflowArgs {
                query: (&query).into(),
                query_norm: (&query_norm).into(),
                head_weights: (&head_weights).into(),
                rotary_components: components.into(),
                rotary_frequencies: frequencies.into(),
                rotary_amplitudes: amplitudes.into(),
                coordinates: inputs.coordinates.into(),
                keys: inputs.keys.tensor().into(),
                spans: inputs.spans.into(),
                score_slots: (&score_slots).into(),
                form: (&form).into(),
                scale: indexer.scale as f32,
                epsilon: query_epsilon,
                block: query_rounding.block,
                element: query_rounding.element,
                rounding_scale: query_rounding.scale,
                amax_floor: query_rounding.amax_floor(),
                slab_rows: class.slab_rows,
            },
        )?
        .value;
    let top_slots = none.reshape(&[0, indexer.top]);
    let picks = graph
        .enqueue(
            entries.topk,
            &[("M", rows), ("N", scores), ("R", class.spans), ("K", indexer.top)],
            rows_topk::WorkflowArgs {
                scores: (&score).into(),
                complete: inputs.complete.into(),
                spans: inputs.spans.into(),
                top_slots: (&top_slots).into(),
            },
        )?
        .value;
    Ok(Indexed { scores: score, picks })
}

/// The candidate pool of a producing indexer's class from its `scores`
/// over each row's list `spans` (`[rows, R, 2]`, its `complete` units in
/// order): the rows' blocks of `block_units` scored (`candidate_blocks`),
/// the pool's `blocks` best of them picked by the indexer's `topk` over each
/// row's block list, and the pool's spans of the list, filled to the
/// restricted indexers' `top` (`candidate_rows`): `[rows, blocks + R, 2]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn candidate_pool<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    class: IndexerClass,
    pool: CandidatePool,
    topk: G::Binding<'a, rows_topk::Entry>,
    entries: CandidateEntries<'a, G>,
    scores: &WorkflowTensor,
    spans: &WorkflowTensorView,
    complete: &WorkflowTensorView,
) -> Result<WorkflowTensor, GraphError> {
    let rows = class.rows;
    // The blocks a row's list of at most N units spans.
    let listed = class.scores.div_ceil(pool.block_units);
    // Extent-only tensors: zero rows of the complete units' table.
    let none = complete.slice_leading(0, 0);
    let block_slots = none.reshape(&[0, listed]);
    let block_form = none.reshape(&[0, pool.block_units]);
    let scored = graph.enqueue(
        entries.blocks,
        &[("M", rows), ("N", class.scores), ("NB", listed), ("BU", pool.block_units)],
        candidate_blocks::WorkflowArgs {
            scores: scores.into(),
            complete: complete.into(),
            block_slots: (&block_slots).into(),
            form: (&block_form).into(),
        },
    )?;
    let top_slots = none.reshape(&[0, pool.blocks]);
    let picked = graph
        .enqueue(
            topk,
            &[("M", rows), ("N", listed), ("R", 1), ("K", pool.blocks)],
            rows_topk::WorkflowArgs {
                scores: (&scored.r0).into(),
                complete: complete.into(),
                spans: (&scored.r1).into(),
                top_slots: (&top_slots).into(),
            },
        )?
        .value;
    let rows_form = none.reshape(&[0, pool.block_units, pool.top]);
    Ok(graph
        .enqueue(
            entries.rows,
            &[
                ("M", rows),
                ("R", class.spans),
                ("B", pool.blocks),
                ("BU", pool.block_units),
                ("K", pool.top),
            ],
            candidate_rows::WorkflowArgs {
                blocks: (&picked).into(),
                spans: spans.into(),
                form: (&rows_form).into(),
            },
        )?
        .value)
}
