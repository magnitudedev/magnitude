//! The graph of an attention's index-list source (its `HistorySource` 1),
//! which the dense entries read through `selected_rows` and `selected`
//! (`graph::PickedReads`).
//!
//! An own source (`PickedSource::Own`, a Full layer) projects each token's
//! `u = projection · x` and, softmax-gated, `g = gate · x`
//! (`attention_project`'s query and gate segments under the input norm, in
//! F32: published so for a softmax-gated pool, else widened from A), pools
//! each complete group of `rate` tokens into its
//! Block plane (`pool_rows`, the incomplete group's tokens in its Window
//! domain; at rate 1 each token is its own group, there is no Window, and
//! `attention_append_dense` stores it: [`Pooling`]), and picks each row's
//! rows of that plane with its own indexer (`selection::indexer`, over the
//! low-rank query's `qr` and the latent `pool_rows` returns, or at rate 1
//! `u` under the pooled norm, whose index keys it writes).
//!
//! A Reindex source (`PickedSource::Reindex`) makes no rows or keys: it
//! reads part `part` of its source layer's Block plane and index keys
//! (Shared state ports) and picks with its own indexer's query and head
//! weights over the same Block read's spans.
//!
//! An indexer stores its picks to the launch's picks table of the Block
//! read, which the later layers of the launch reusing them bind: a Reindex
//! layer stores over the picks of the layer before it, so each reader binds
//! the latest. An indexer producing a candidate pool stores it to the
//! launch's pool table of the read, which a restricted indexer lists in
//! place of the read's spans (`selection::candidate_pool`).
//!
//! A reused source (`PickedSource::Reused`, a Reuse layer) makes and picks
//! nothing: it reads part `part` of its source layer's plane (a Shared state
//! port) through the read's picks table, bound as an input. Block graphs
//! are separate graphs, so the picks and the pool cross between them as the
//! launch-scoped tables they bind, as the launch's bound controls do.

use super::graph::{IndexerWeights, PickedReads};
use crate::operators::scalar;
use super::split::{AttentionBlock, AttentionTopology, Candidates, IndexedReals, IndexedSource, OwnReals, OwnRows, Plane, both};
use crate::planning::{RoundingArguments, axis};
use super::selection::{
    self, IndexerClass, IndexerElements, IndexerEntries, IndexerInputs, KeyWrite, Projection, ProjectionInput,
    segment,
};
use crate::operators::InputForm;
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::PackedWeight;
use crate::{CandidatesBinding, PickedBinding};
use magnitude_family_contracts::{Indexer, Pool, PooledRows};
use magnitude_kernels::{attention_append_dense, attention_project, indexer_keys, pool_rows, widen_rows};
use seismic::{Argument, Dim, Element, NativePort, WorkflowTensor, WorkflowTensorView};

/// An own source's rows' entries at one graph class: the `u` and `g`
/// projections (the query and gate segments), their widening to the F32
/// rows the pooling reads where they are published in A (an ungated pool),
/// the pooling, and its index keys' projection and write.
pub(crate) struct PooledEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, attention_project::Entry>,
    pub widen: G::Binding<'a, widen_rows::Entry>,
    pub pool: PoolEntry<'a, G>,
    pub key_project: G::Binding<'a, attention_project::Entry>,
    pub keys: G::Binding<'a, indexer_keys::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for PooledEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for PooledEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// How an own source pools its tokens: in groups of its rate, the
/// incomplete group's tokens in its Window domain (`pool_rows`), or at rate
/// 1, each token its own row with no Window, which needs no pooling entry
/// of its own: `attention_append_dense` stores it as a window history
/// stores its keys (normed, rotated and rounded at the row's own
/// coordinates), and its index keys' projection forms the latent in its
/// input norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Pooling {
    Grouped,
    Identity,
}

impl Pooling {
    /// The pooling of a source of `rate` tokens a row: the one choice
    /// preparation, the checked graphs and the graph share.
    pub(crate) fn of(rate: u64) -> Self {
        match rate {
            1 => Self::Identity,
            _ => Self::Grouped,
        }
    }

    /// `attention_append_dense`'s form at rate 1 beside the row's `P`, `S`
    /// and layout: one head (`KV`) of one layer's rows (`F`), its key normed
    /// (`N`) and rounded as the pooled rows are (`KR`, where `rounds`; a
    /// rounded key is held in the activation dtype after its norm and its
    /// rotation, as the trained model's are), and no value plane (`VK`,
    /// `NV`).
    pub(crate) fn identity_append(rounds: bool) -> [(&'static str, u64); 6] {
        [("KV", 1), ("F", 1), ("N", 1), ("NV", 0), ("VK", 1), ("KR", u64::from(rounds))]
    }
}

/// An own source's pooling entry at one graph class ([`Pooling`]).
pub(crate) enum PoolEntry<'a, G: GraphDraft + 'a> {
    Grouped(G::Binding<'a, pool_rows::Entry>),
    Identity(G::Binding<'a, attention_append_dense::Entry>),
}

impl<'a, G: GraphDraft + 'a> Copy for PoolEntry<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for PoolEntry<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// An indexing source's entries at one graph class: its own rows' (none
/// for a Reindex source) and its indexer's.
pub(crate) struct PickedEntries<'a, G: GraphDraft + 'a> {
    pub rows: Option<PooledEntries<'a, G>>,
    pub indexer: IndexerEntries<'a, G>,
}

impl<'a, G: GraphDraft + 'a> Copy for PickedEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for PickedEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// The elements an indexing source's projections are specialized at, the
/// one assignment preparation and the checked graphs share: an own
/// source's rows' (the `u | g` projection over `x` under the input norm,
/// its index keys' over the F32 latent: no norm, a zero-row view of the key
/// norm; at rate 1 over `u` under the pooled rows' norm), and its indexer's
/// ([`IndexerElements`]: admission holds an index-list source's indexer to
/// the query rank). The `u | g` projection is published in F32 for a
/// softmax-gated pool (whose pooling is F32 from its projections on) and in
/// A otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PickedElements {
    pub indexer: IndexerElements,
    pub rows: Option<PooledElements>,
}

/// An own source's rows' projections' elements (`PickedElements::rows`),
/// and its pooling, which names its pooling entry ([`PoolEntry`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PooledElements {
    pub project: [(&'static str, Element); 7],
    pub key_project: [(&'static str, Element); 7],
    pub pooling: Pooling,
}

impl PickedElements {
    /// `None` for a binding without an indexer of its own (no index-list
    /// source, or a reused one).
    pub(crate) fn of(binding: crate::AttentionBinding) -> Option<Self> {
        let picked = binding.picked?;
        let indexer = picked.indexer()?;
        let activation = binding.activation;
        Some(Self {
            indexer: IndexerElements::of(binding, indexer),
            rows: match picked {
                // A mean or identity pool's absent gate segment binds a
                // zero-row view of the projection, at its element (as
                // `attention::binding`'s absent segments bind the query's).
                PickedBinding::Own(pooled) => Some(PooledElements {
                    project: tokens_project(activation, binding.norm, pooled.projection, pooled.gate),
                    key_project: match Pooling::of(pooled.rate) {
                        Pooling::Grouped => segment(activation, pooled.key_norm, pooled.key, activation),
                        Pooling::Identity => segment(activation, pooled.norm, pooled.key, activation),
                    },
                    pooling: Pooling::of(pooled.rate),
                }),
                PickedBinding::Reindex { .. } | PickedBinding::Reused => None,
            },
        })
    }
}

/// The `u | g` projection's elements over `x` under the input norm `norm`:
/// `u` through `projection` and, gated, `g` through `gate` (a mean pool's
/// absent gate a zero-row view of the projection), published in F32 for a
/// softmax-gated pool and in A otherwise.
pub(crate) fn tokens_project(
    activation: Element,
    norm: Element,
    projection: Element,
    gate: Option<Element>,
) -> [(&'static str, Element); 7] {
    [
        ("NW", norm),
        ("QW", projection),
        ("GW", gate.unwrap_or(projection)),
        ("KW", projection),
        ("VW", projection),
        ("A", activation),
        ("Y", if gate.is_some() { Element::f32() } else { activation }),
    ]
}

/// An own source's rows' weights: the pooled rows' projection, gate
/// (gated) and norm, and its index keys' projection and norm.
#[derive(Clone)]
pub(crate) struct PooledWeights {
    pub projection: PackedWeight,
    pub gate: Option<PackedWeight>,
    /// `[W]`.
    pub norm: WorkflowTensor,
    pub key: PackedWeight,
    pub key_norm: WorkflowTensor,
}

/// An indexing source's weights: its own rows' (none for a Reindex
/// source), and its indexer's.
#[derive(Clone)]
pub(crate) struct PickedWeights {
    pub rows: Option<PooledWeights>,
    pub indexer: IndexerWeights,
}

/// The Block plane an index-list source's picks address at one graph
/// class: the plane's rows and slab rows. It names no layer: blocks of one
/// shape share the graph, and each binds its own source's plane
/// ([`PooledSource`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PickedPlane {
    pub rows: u64,
    pub slab_rows: u32,
}

/// A block's index-list source at one graph class, built once from the
/// operator's `PickedSource`: an indexing source's indexer and (an own
/// source's) rows with its plane, or a reused source's plane alone.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PickedBlock<'o> {
    Indexed {
        plane: PickedPlane,
        index: IndexBlock<'o>,
        /// An own source's rows; none for a Reindex source.
        own: Option<OwnBlock<'o>>,
    },
    Reused {
        plane: PickedPlane,
    },
}

/// An indexer over a Block read at one graph class: the operator's
/// indexer, the spans (`R`) and scores (`N`) of a row's list over the read
/// (`attention::list_scores`), the candidates it scores, and its query's
/// and written keys' roundings (`attention::indexer_roundings`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct IndexBlock<'o> {
    pub indexer: &'o Indexer,
    pub spans: u64,
    pub scores: u64,
    pub candidates: CandidatesBinding,
    pub query_rounding: crate::planning::RoundingArgs,
    pub key_rounding: crate::planning::RoundingArgs,
}

/// An own source at one graph class: the operator's pooled rows and its
/// Window domain's (none at rate 1, whose tokens are complete groups).
#[derive(Clone, Copy, Debug)]
pub(crate) struct OwnBlock<'o> {
    pub rows: &'o PooledRows,
    pub window: Option<PooledWindow>,
}

/// An own source's Window domain at one graph class: the incomplete
/// group's tokens' rows and slab rows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PooledWindow {
    pub rows: u64,
    pub slab_rows: u32,
}

/// A block's index-list source as its graph makes it: an indexing source
/// with what it takes of the attention's input, or a reused source's plane.
pub(crate) enum Picked<'o, 'v, 'a, G: GraphDraft + 'a> {
    Indexed {
        source: &'o IndexedSource<Dim>,
        reals: &'o IndexedReals<Argument<f64>>,
        inputs: PickedInputs<'v, 'a, G>,
    },
    Reused {
        plane: &'o Plane<Dim>,
    },
}

/// A block's pooled source (`LayerRef::Pooled { block, part }`): the planes
/// its index-list source reads (`PickedSource::pooled`), kept per block
/// beside the graph its shape may share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PooledSource {
    pub block: u32,
    pub part: u32,
}

/// What the graph takes of the attention's input for an indexing source:
/// its entries and weights, the layer's normed input `x` and its input
/// norm, and the low-rank query's F32 `qr` rows under their norm.
pub(crate) struct PickedInputs<'v, 'a, G: GraphDraft + 'a> {
    pub entries: PickedEntries<'a, G>,
    pub weights: &'v PickedWeights,
    pub x: ProjectionInput<'v>,
    pub input_norm: &'v WorkflowTensorView,
    pub epsilon: Argument<f32>,
    pub qr: ProjectionInput<'v>,
    pub qr_norm: &'v WorkflowTensorView,
    pub qr_epsilon: Argument<f32>,
}

impl<'a, G: GraphDraft + 'a> Picked<'_, '_, 'a, G> {
    pub(crate) fn plane(&self) -> &Plane<Dim> {
        match self {
            Self::Indexed { source, .. } => &source.plane,
            Self::Reused { plane } => plane,
        }
    }
}

/// The source's ports, bound per run to the running block's own source
/// ([`PooledSource`]): the Block plane the dense entries
/// read (its own, or part `part` of its source layer's) and the launch's
/// picks table of that read (stored by an indexing source, read by a
/// reused one), and an indexing source's other ports.
#[derive(Clone)]
pub(crate) struct PickedPorts {
    pub plane: NativePort,
    pub picks: NativePort,
    pub index: Option<IndexPorts>,
}

/// An indexing source's ports: the read's index keys (state), each row's
/// complete units and its list (tables), and an own source's rows' ports.
#[derive(Clone)]
pub(crate) struct IndexPorts {
    pub index_keys: NativePort,
    pub complete: NativePort,
    pub lists: ListPorts,
    pub own: Option<OwnPorts>,
}

/// The table of an indexer's lists.
#[derive(Clone)]
pub(crate) enum ListPorts {
    /// Each row's spans of the Block read (its visible spans), and the
    /// launch's pool table a producing indexer stores.
    Visible {
        spans: NativePort,
        pool: Option<NativePort>,
    },
    /// The launch's pool table a restricted indexer lists.
    Pool(NativePort),
}

/// An own source's rows' ports: its Block destinations (a table) and its
/// Window ports (none at rate 1).
#[derive(Clone)]
pub(crate) struct OwnPorts {
    pub block_destinations: NativePort,
    pub window: Option<WindowPorts>,
}

/// An own source's Window ports: its rows and coordinates (state), and its
/// spans and destinations (tables).
#[derive(Clone)]
pub(crate) struct WindowPorts {
    pub rows: NativePort,
    pub coordinates: NativePort,
    pub visible: NativePort,
    pub destinations: NativePort,
}

/// Whether `source`'s pool is softmax-gated (`GT`, and its `g` rows): the
/// one test of the operator, which the binding records as its gate
/// ([`crate::PooledBinding::gt`]), and the operands and graph read.
pub(crate) fn gated(source: &PooledRows) -> bool {
    match source.pool {
        Pool::SoftmaxGated { .. } => true,
        Pool::Identity | Pool::Mean => false,
    }
}

/// What an own source's rows give its indexer's key write: the F32 rows its
/// keys' projection reads and their norm (`pool_rows`' latent under none;
/// at rate 1 the tokens' `u` under the pooled norm and its epsilon, which
/// forms the latent), the key-plane row each row completes and its
/// coordinates.
struct Pooled {
    keys: WorkflowTensorView,
    keys_norm: Option<(WorkflowTensorView, Argument<f32>)>,
    latent_rows: WorkflowTensorView,
    latent_coordinates: WorkflowTensorView,
}

/// An own source's rows over `coordinates` and the attention's rotary
/// table: the `u | g` projection, `pool_rows` into `plane` and its Window,
/// or at rate 1 `attention_append_dense` into `plane` alone, and its ports.
#[allow(clippy::too_many_arguments)]
fn own_rows<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    topology: &AttentionTopology,
    numbers: &AttentionBlock,
    own: &OwnRows<Dim>,
    own_reals: &OwnReals<Argument<f64>>,
    plane_numbers: &Plane<Dim>,
    plane: &mut NativePort,
    entries: PooledEntries<'a, G>,
    weights: &PooledWeights,
    inputs: &PickedInputs<'_, 'a, G>,
    coordinates: &NativePort,
    rotary: [&WorkflowTensor; 3],
    constants: &mut Vec<GraphConstant>,
) -> Result<(Pooled, OwnPorts), GraphError> {
    let form = &topology.form;
    let (rows, width) = (&numbers.rows, &numbers.widths.width);
    let pairs = &numbers.widths.rotary_pairs;
    let pooled_epsilon = own_reals.epsilon.f32();
    let slab_rows = |plane: &Plane<Dim>| scalar::<u32>(&plane.slab_rows, "slab rows");
    let gt = axis(own.gated);
    let gate_rows = &gt * width;
    // The tokens' `u` and `g` under the input norm, then in F32.
    let empty = weights.projection.weight.slice_leading(0, 0);
    let absent = GraphConstant::absent(graph, constants)?.tensor().clone();
    let gate = weights.gate.as_ref();
    let (projection, gate_operands) = (
        &weights.projection.operands,
        gate.map(|gate| &gate.operands),
    );
    let operand = |extent: Option<&Dim>| extent.cloned().unwrap_or(Dim::from(0));
    let projected = graph.enqueue(
        entries.project,
        &[
            [
                ("M", rows.clone()),
                ("D", inputs.x.width.clone()),
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
            &inputs.x.form.dimensions(inputs.x.width.clone()),
        ]
        .concat(),
        attention_project::WorkflowArgs {
            hidden: inputs.x.rows.into(),
            input_norm: inputs.input_norm.into(),
            query_weight: (&weights.projection.weight).into(),
            gate_weight: match gate {
                Some(gate) => (&gate.weight).into(),
                None => (&empty).into(),
            },
            key_weight: (&empty).into(),
            value_weight: (&empty).into(),
            epsilon: inputs.epsilon,
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
    let widen = |graph: &mut G, narrow: &WorkflowTensor, columns: &Dim| {
        Ok::<_, GraphError>(
            graph
                .enqueue(
                    entries.widen,
                    &[("M", rows.clone()), ("D", columns.clone())],
                    widen_rows::WorkflowArgs {
                        rows: narrow.into(),
                    },
                )?
                .value,
        )
    };
    // A softmax-gated pool's projections are published in F32
    // (`PooledElements`); the others' are widened from A.
    let u = match gate {
        Some(_) => projected.r0.clone(),
        None => widen(graph, &projected.r0, width)?,
    };
    let rounding = RoundingArguments::of(both(&own.rounding, &own_reals.rounding, "own rows' rounding"))?;
    let [components, frequencies, amplitudes] = rotary;
    let norm = weights.norm.reshape(&[Dim::from(1), width.clone()]);
    // The entry's `form`: the row's rotary layout (RA, RT), then `rest`.
    let form_table = |rest: &[Dim]| {
        let extents = [
            &[
                Dim::from(0),
                axis(form.rotary_adjacent),
                axis(form.rotary_trailing),
            ][..],
            rest,
        ]
        .concat();
        coordinates.tensor().slice_leading(0, 0).reshape(&extents)
    };
    // Both entries' dimensions: the rows and the row's layout.
    let layout = [
        ("M", rows.clone()),
        ("P", pairs.clone()),
        ("S", width - pairs * 2),
        ("RA", axis(form.rotary_adjacent)),
        ("RT", axis(form.rotary_trailing)),
    ];
    let pooling = (entries.pool, own.window.as_ref());
    let (pooled, block_destinations, window) = match pooling {
        // Groups of the rate, the incomplete group's tokens in the Window
        // domain, read through its spans.
        (PoolEntry::Grouped(entry), Some(window)) => {
            let g = match gate {
                Some(_) => projected.r1.reshape(&[rows.clone(), gate_rows.clone()]),
                // No gate scores: zero columns of the F32 rows.
                None => u.slice_leading(0, 0).reshape(&[rows.clone(), Dim::from(0)]),
            };
            let dimensions = [
                &layout[..],
                &[
                    ("TB", plane_numbers.rows.clone()),
                    ("TW", window.rows.clone()),
                    ("GT", gt.clone()),
                    ("PR", Dim::from(0)),
                    ("TN", Dim::from(0)),
                    ("R", numbers.segments.clone()),
                    ("RATE", own.rate.clone()),
                ],
            ]
            .concat();
            let control = |graph: &mut G, name: &str| {
                graph.control_for(topology.controls, entry, name, &dimensions)
            };
            let block_destinations = control(graph, "block_destinations")?;
            let destinations = control(graph, "window_destinations")?;
            let visible = control(graph, "window_visible")?;
            let mut rows_port =
                graph.port(Element::f32(), &[window.rows.clone(), width + &gate_rows])?;
            let mut coordinates_port =
                graph.port(Element::i32(), &[window.rows.clone(), Dim::from(4)])?;
            let spans = visible
                .tensor()
                .reshape(&[rows.clone(), numbers.segments.clone(), Dim::from(2)]);
            // It writes the plane's rows `[TB, W]`.
            let mut block_rows = plane
                .tensor_mut()
                .reshape(&[plane_numbers.rows.clone(), width.clone()]);
            let form = form_table(&[own.rate.clone(), gt]);
            let pooled = graph.enqueue(
                entry,
                &dimensions,
                pool_rows::WorkflowArgs {
                    u: (&u).into(),
                    g: (&g).into(),
                    position: (&norm.slice_leading(0, 0)).into(),
                    norm: (&norm).into(),
                    bias: (&norm.slice_leading(0, 0)).into(),
                    rotary_components: components.into(),
                    rotary_frequencies: frequencies.into(),
                    rotary_amplitudes: amplitudes.into(),
                    coordinates: coordinates.tensor().into(),
                    window_visible: (&spans).into(),
                    window_destinations: destinations.tensor().into(),
                    block_destinations: block_destinations.tensor().into(),
                    window_rows: rows_port.tensor_mut().into(),
                    window_coordinates: coordinates_port.tensor_mut().into(),
                    block_rows: (&mut block_rows).into(),
                    form: (&form).into(),
                    epsilon: pooled_epsilon,
                    window_slab_rows: slab_rows(window)?,
                    block_slab_rows: slab_rows(plane_numbers)?,
                    block: rounding.block,
                    element: rounding.element,
                    scale: rounding.scale,
                    amax_floor: rounding.amax_floor,
                },
            )?;
            let window = WindowPorts {
                rows: rows_port,
                coordinates: coordinates_port,
                visible,
                destinations,
            };
            let pooled = Pooled {
                keys: pooled.r0.slice_leading(0, rows.clone()),
                keys_norm: None,
                latent_rows: pooled.r1.reshape(&[rows.clone()]),
                latent_coordinates: pooled.r2.reshape(&[rows.clone(), Dim::from(4)]),
            };
            (pooled, block_destinations, Some(window))
        }
        // Rate 1: each token its own row, with no Window, stored as a window
        // history stores its keys: one head of the projected row, normed,
        // rotated at the row's coordinates (held after each where rounded),
        // rounded, with no value plane.
        (PoolEntry::Identity(entry), None) => {
            let append = Pooling::identity_append(own.rounding.is_some()).map(|(name, extent)| (name, Dim::from(extent)));
            let dimensions = [&layout[..], &[("T", plane_numbers.rows.clone())], &append].concat();
            let block_destinations =
                graph.control_for(topology.controls, entry, "destinations", &dimensions)?;
            let key = projected.r0.reshape(&[Dim::from(1), rows.clone(), width.clone()]);
            let mut history_key = plane
                .tensor_mut()
                .reshape(&[plane_numbers.rows.clone(), Dim::from(1), width.clone()]);
            let mut history_value = GraphConstant::empty_port(
                graph,
                topology.activation,
                &[Dim::from(0), Dim::from(1), width.clone()],
                constants,
            )?;
            let [.., (_, vk), (_, kr)] = append;
            let form = form_table(&[vk, kr]);
            graph.enqueue(
                entry,
                &dimensions,
                attention_append_dense::WorkflowArgs {
                    key: (&key).into(),
                    value: (&projected
                        .r0
                        .slice_leading(0, 0)
                        .reshape(&[Dim::from(0), rows.clone(), width.clone()]))
                        .into(),
                    key_norm: (&norm).into(),
                    value_norm: (&weights
                        .norm
                        .slice_leading(0, 0)
                        .reshape(&[Dim::from(0), width.clone()]))
                        .into(),
                    rotary_components: components.into(),
                    rotary_frequencies: frequencies.into(),
                    rotary_amplitudes: amplitudes.into(),
                    coordinates: coordinates.tensor().into(),
                    destinations: block_destinations.tensor().into(),
                    history_key: (&mut history_key).into(),
                    history_value: history_value.tensor_mut().into(),
                    form: (&form).into(),
                    epsilon: pooled_epsilon,
                    slab_rows: slab_rows(plane_numbers)?,
                    key_block: rounding.block,
                    key_element: rounding.element,
                    key_scale: rounding.scale,
                    key_amax_floor: rounding.amax_floor,
                },
            )?;
            // Its latent is `u` under the pooled norm, which the keys'
            // projection forms; each stored row is its own, at its own
            // coordinates.
            let pooled = Pooled {
                keys: u.slice_leading(0, rows.clone()),
                keys_norm: Some((weights.norm.reshape(&[width.clone()]), pooled_epsilon)),
                latent_rows: block_destinations.tensor().reshape(&[rows.clone()]),
                latent_coordinates: coordinates.tensor().reshape(&[rows.clone(), Dim::from(4)]),
            };
            (pooled, block_destinations, None)
        }
        _ => return Err("a pooled source keeps a Window domain exactly above rate 1".into()),
    };
    Ok((
        pooled,
        OwnPorts {
            block_destinations,
            window,
        },
    ))
}

/// The block's index-list source over `coordinates` and the attention's
/// rotary table `rotary`: what the dense entries read, and its ports.
pub(crate) fn picked<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    topology: &AttentionTopology,
    numbers: &AttentionBlock,
    picked: Picked<'_, '_, 'a, G>,
    coordinates: &NativePort,
    rotary: [&WorkflowTensor; 3],
    constants: &mut Vec<GraphConstant>,
) -> Result<(PickedReads, PickedPorts), GraphError> {
    let form = &topology.form;
    let widths = &numbers.widths;
    let (rows, width, kv_heads) = (&numbers.rows, &widths.width, &widths.kv_heads);
    let (index, reals, inputs) = match picked {
        Picked::Indexed { source, reals, inputs } => (source, reals, inputs),
        Picked::Reused { plane: plane_numbers } => {
            let plane = plane_port(graph, topology.activation, plane_numbers, width)?;
            let picks = graph.port(Element::i32(), &[rows.clone(), widths.picks.clone()])?;
            let read = picked_reads(
                picks.tensor().reshape(&[rows.clone(), widths.picks.clone()]),
                &plane,
                plane_numbers,
                kv_heads,
                width,
            )?;
            return Ok((
                read,
                PickedPorts {
                    plane,
                    picks,
                    index: None,
                },
            ));
        }
    };
    let IndexedSource {
        plane: plane_numbers,
        width: index_width,
        top,
        spans,
        scores,
        ..
    } = index;
    let mut plane = plane_port(graph, topology.activation, plane_numbers, width)?;
    let weights = inputs.weights;
    // An own source makes its rows, whose keys its indexer writes.
    let pooled = match (both(&index.own, &reals.own, "own rows"), inputs.entries.rows, &weights.rows) {
        (Some((own, own_reals)), Some(entries), Some(rows_weights)) => {
            let (pooled, ports) = own_rows(
                graph,
                topology,
                numbers,
                own,
                own_reals,
                plane_numbers,
                &mut plane,
                entries,
                rows_weights,
                &inputs,
                coordinates,
                rotary,
                constants,
            )?;
            Some((pooled, ports, entries, rows_weights))
        }
        (None, None, None) => None,
        _ => return Err("an index-list source's rows, entries and weights disagree".into()),
    };
    // The indexer over the read's index keys: each row's picks.
    let mut index_keys =
        graph.port(Element::f32(), &[plane_numbers.rows.clone(), index_width.clone()])?;
    // A row lists the read's visible spans, or a restricted indexer the
    // candidate pool of `blocks` blocks and those spans' fill.
    let (list_spans, list_scores) = match &index.candidates {
        Candidates::All | Candidates::Produce(_) => (spans.clone(), scores.clone()),
        Candidates::Restrict(pool) => (
            &pool.blocks + spans,
            pool.units().min(scores),
        ),
    };
    let pairs = &widths.rotary_pairs;
    let class = IndexerClass {
        source: index,
        rows: rows.clone(),
        spans: list_spans.clone(),
        scores: list_scores.clone(),
        rotary_pairs: pairs.clone(),
    };
    let score_dimensions = class.score_dimensions(
        weights.indexer.query_norm.is_some(),
        weights.indexer.head_weights.is_some(),
        form.rotary_adjacent,
        form.rotary_trailing,
    );
    let entries = inputs.entries.indexer;
    // The lists: the read's visible spans (a launch control), or the pool
    // table a restricted indexer binds. A pool table's spans take the
    // pool's blocks beside the class's segments.
    let lists = match &index.candidates {
        Candidates::All => ListPorts::Visible {
            spans: graph.control_for(topology.controls, entries.score, "spans", &score_dimensions)?,
            pool: None,
        },
        Candidates::Produce(pool) => {
            let mut pool_dimensions = score_dimensions.clone();
            for (name, value) in &mut pool_dimensions {
                if *name == "R" {
                    *value = &pool.blocks + &list_spans;
                }
            }
            let visible = graph.control_for(
                topology.controls,
                entries.score,
                "spans",
                &score_dimensions,
            )?;
            ListPorts::Visible {
                spans: visible,
                pool: Some(graph.port_for(entries.score, "spans", &pool_dimensions)?),
            }
        }
        Candidates::Restrict(_) => ListPorts::Pool(graph.port_for(entries.score, "spans", &score_dimensions)?),
    };
    let spans_port = match &lists {
        ListPorts::Visible { spans, .. } | ListPorts::Pool(spans) => spans,
    };
    let complete = graph.control_for(
        topology.controls,
        entries.topk,
        "complete",
        &[
            ("M", rows.clone()),
            ("N", list_scores.clone()),
            ("R", list_spans.clone()),
            ("K", top.clone()),
        ],
    )?;
    let query_norm = weights
        .indexer
        .query_norm
        .as_ref()
        .map(|norm| norm.reshape(&[Dim::from(1), index_width.clone()]));
    let coordinates_view = coordinates.tensor().reshape(&[rows.clone(), Dim::from(4)]);
    let spans_view = spans_port
        .tensor()
        .reshape(&[rows.clone(), list_spans, Dim::from(2)]);
    let complete_view = complete.tensor().reshape(&[rows.clone()]);
    let head_weights = match (index.head_weights, &weights.indexer.head_weights) {
        (true, Some(weight)) => Some((
            ProjectionInput {
                rows: inputs.x.rows,
                width: inputs.x.width.clone(),
                form: inputs.x.form,
            },
            Projection {
                weight: &weight.weight,
                operands: &weight.operands,
                norm: inputs.input_norm,
                epsilon: inputs.epsilon,
            },
        )),
        (true, None) => {
            return Err("an indexer's projected head weights are absent".into());
        }
        (false, None) => None,
        (false, Some(_)) => {
            return Err("head weights of an indexer that projects none".into());
        }
    };
    // An own source writes its rows' keys. Their projection reads the
    // latent under no input norm (zero rows of an F32 norm), or at rate 1
    // the tokens' `u` under the pooled norm, which forms the latent.
    let key_norms = pooled.as_ref().map(|(_, _, _, rows_weights)| {
        (
            rows_weights.key_norm.slice_leading(0, 0),
            rows_weights
                .key_norm
                .reshape(&[Dim::from(1), index_width.clone()]),
        )
    });
    let write = pooled.as_ref().zip(key_norms.as_ref()).map(
        |((pooled, _, entries, rows_weights), (none, key_norm))| {
            let (form, norm, epsilon) = match &pooled.keys_norm {
                None => (InputForm::PLAIN, none, Argument::from(0.0)),
                Some((norm, epsilon)) => (InputForm::SINGLE, norm, *epsilon),
            };
            KeyWrite {
                project: entries.key_project,
                write: entries.keys,
                latent: ProjectionInput {
                    rows: &pooled.keys,
                    width: width.clone(),
                    form,
                },
                latent_rows: &pooled.latent_rows,
                latent_coordinates: &pooled.latent_coordinates,
                key: Projection {
                    weight: &rows_weights.key.weight,
                    operands: &rows_weights.key.operands,
                    norm,
                    epsilon,
                },
                key_norm,
            }
        },
    );
    let indexed = selection::indexer(
        graph,
        &class,
        reals,
        entries,
        IndexerInputs {
            query_input: ProjectionInput {
                rows: inputs.qr.rows,
                width: inputs.qr.width.clone(),
                form: inputs.qr.form,
            },
            query: Projection {
                weight: &weights.indexer.query.weight,
                operands: &weights.indexer.query.operands,
                norm: inputs.qr_norm,
                epsilon: inputs.qr_epsilon,
            },
            query_norm: query_norm.as_ref(),
            head_weights,
            write,
            coordinates: &coordinates_view,
            rotary,
            rotary_adjacent: form.rotary_adjacent,
            rotary_trailing: form.rotary_trailing,
            keys: &mut index_keys,
            spans: &spans_view,
            complete: &complete_view,
            picks: selection::Picks::Rows,
        },
        constants,
    )?;
    // A producing indexer's candidate pool, from its scores over the
    // read's spans, for the restricted indexers after it.
    match (&index.candidates, &lists, entries.candidates) {
        (
            Candidates::Produce(pool),
            ListPorts::Visible { pool: Some(stored), .. },
            Some(candidates),
        ) => {
            let made = selection::candidate_pool(
                graph,
                &class,
                pool,
                entries.topk,
                candidates,
                &indexed.scores,
                &spans_view,
                &complete_view,
            )?;
            graph.store(&made, stored)?;
        }
        (Candidates::All | Candidates::Restrict(_), _, None) => {}
        _ => return Err("an indexer's candidate pool, its table and its entries disagree".into()),
    }
    // The launch's picks table of the read, for the later layers reusing
    // them.
    let stored = graph.port(Element::i32(), &[rows.clone(), top.clone()])?;
    graph.store(&indexed.picks, &stored)?;
    Ok((
        picked_reads(
            indexed.picks.reshape(&[rows.clone(), top.clone()]),
            &plane,
            plane_numbers,
            kv_heads,
            width,
        )?,
        PickedPorts {
            plane,
            picks: stored,
            index: Some(IndexPorts {
                index_keys,
                complete,
                lists,
                own: pooled.map(|(_, ports, ..)| ports),
            }),
        },
    ))
}

/// A pooled source's Block plane, as its state stores it
/// (`magnitude_state`'s pooled component): its rows of one `width` head.
fn plane_port<G: GraphDraft>(
    graph: &mut G,
    element: Element,
    plane: &Plane<Dim>,
    width: &Dim,
) -> Result<NativePort, GraphError> {
    Ok(graph.port(element, &[plane.rows.clone(), Dim::from(1), width.clone()])?)
}

/// What the dense entries read of the source: each row's `picks` in its
/// `plane`, viewed `[TS, KV, W]`.
fn picked_reads(
    picks: WorkflowTensorView,
    plane: &NativePort,
    numbers: &Plane<Dim>,
    kv_heads: &Dim,
    width: &Dim,
) -> Result<PickedReads, GraphError> {
    Ok(PickedReads {
        rows: picks,
        plane: plane
            .tensor()
            .reshape(&[numbers.rows.clone(), kv_heads.clone(), width.clone()]),
        slab_rows: scalar(&numbers.slab_rows, "slab rows")?,
    })
}
