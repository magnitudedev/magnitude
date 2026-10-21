//! The n-gram input's graph fragment over the streams: each batch row's
//! table rows from its token row and the token window (`ngram_rows`), read
//! from the streamed table and decoded to activations (`stored_rows`,
//! through `stored_table_rows`), the key `[S * H]` and value `[H]`
//! projections into F32, the gate (`ngram_gate`) and the row op over the
//! layer's window (`ngram_inject`). Slots and versions follow the recurrent
//! contract: the fragment binds its block's recurrent control ports, and
//! both windows are components of its block's bank.

use crate::NGramBinding;
use crate::native::NGramKernels;
use crate::operators::gated_delta::graph::RecurrentControlPorts;
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{
    GatheredTable, StoredReadPorts, WeightPort, packed_weight, stored_table_rows, weight,
};
use crate::{ModelLoadPlan, StateResourcePlan};
use magnitude_family_contracts::{NGramHash, WeightKind, WeightScope};
use magnitude_kernels::{ngram_gate, ngram_inject, ngram_rows, project_rows, stored_rows};
use seismic::{Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

pub(crate) struct NGramEntries<'a, G: GraphDraft + 'a> {
    pub rows: G::Binding<'a, ngram_rows::Entry>,
    pub stored: G::Binding<'a, stored_rows::Entry>,
    pub key: G::Binding<'a, project_rows::Entry>,
    pub value: G::Binding<'a, project_rows::Entry>,
    pub gate: G::Binding<'a, ngram_gate::Entry>,
    pub inject: G::Binding<'a, ngram_inject::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for NGramEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for NGramEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a> From<&'a NGramKernels> for NGramEntries<'a, NativeGraph> {
    fn from(kernels: &'a NGramKernels) -> Self {
        Self {
            rows: &kernels.rows,
            stored: &kernels.stored,
            key: &kernels.key,
            value: &kernels.value,
            gate: &kernels.gate,
            inject: &kernels.inject,
        }
    }
}

/// Entry element assignments from the binding.
pub(crate) struct CheckedNGramEntries {
    stored: [(&'static str, Element); 3],
    key: [(&'static str, Element); 3],
    value: [(&'static str, Element); 3],
    norms: [(&'static str, Element); 1],
}

impl CheckedNGramEntries {
    pub(crate) fn new(binding: NGramBinding) -> Self {
        Self {
            stored: [
                ("E", binding.table_source),
                ("U", binding.table),
                ("A", binding.activation),
            ],
            key: [
                ("A", binding.activation),
                ("W", binding.key),
                ("Y", Element::f32()),
            ],
            value: [
                ("A", binding.activation),
                ("W", binding.value),
                ("Y", Element::f32()),
            ],
            norms: [("NW", binding.norm)],
        }
    }

    pub(crate) fn entries(&self) -> NGramEntries<'_, NativeGraphMetadata> {
        NGramEntries {
            rows: &[],
            stored: &self.stored,
            key: &self.key,
            value: &self.value,
            gate: &self.norms,
            inject: &self.norms,
        }
    }
}

/// One n-gram input at one graph class.
pub(crate) struct NGramBlock<'h> {
    pub rows: u64,
    pub slots: u64,
    pub slab_banks: u32,
    pub binding: NGramBinding,
    /// The row rule `ngram_rows` evaluates.
    pub hash: &'h NGramHash,
    pub epsilon: f32,
    /// Index of this layer's window in the store's recurrent components;
    /// its token window follows it.
    pub component_index: usize,
}

/// The fragment's state ports (the window, then the token window) and the
/// ports each step binds.
pub(crate) struct NGramPorts {
    pub window: NativePort,
    pub token_window: NativePort,
    pub step: NGramStepPorts,
}

/// What each step binds of an n-gram input: its token rows (the entry's kept
/// copy of the host's tokens, or the previous step's selection, in the
/// `sample_rows` layout) and its table read's ports.
#[derive(Clone)]
pub(crate) struct NGramStepPorts {
    pub tokens: NativePort,
    pub table: StoredReadPorts,
}

/// The n-gram input over the `[rows, S * H]` streams `residual`: the new
/// streams and its ports.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ngram<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: NGramEntries<'a, G>,
    state: &StateResourcePlan,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    residual: &WorkflowTensor,
    controls: &RecurrentControlPorts,
    block: NGramBlock<'_>,
) -> Result<(WorkflowTensor, NGramPorts), GraphError> {
    let binding = block.binding;
    let rows = block.rows;
    let (s, h) = (binding.streams, binding.hidden);
    let heads = binding.heads();
    let embedding = binding.embedding();
    let hash = block.hash;
    if hash.context != binding.context || hash.heads_per_order != binding.heads_per_order {
        return Err("the n-gram row rule disagrees with its binding".into());
    }
    let banks = bank_count(state)?;
    let (window_element, window_extents) = bank_component(state, banks, block.component_index)?;
    let (token_element, token_extents) = bank_component(state, banks, block.component_index + 1)?;
    let [_, window_rows, channels] = window_extents[..] else {
        return Err("the n-gram window is not a bank of rows".into());
    };
    if channels != s * h || window_rows <= binding.dilation * (binding.kernel - 1) {
        return Err("the n-gram window disagrees with its binding".into());
    }
    let [_, token_rows, 1] = token_extents[..] else {
        return Err("the n-gram token window is not a bank of single tokens".into());
    };
    // Both windows record the same tape rows.
    let tape = token_rows
        .checked_sub(binding.context)
        .filter(|tape| *tape == window_rows - binding.dilation * (binding.kernel - 1))
        .ok_or("the n-gram windows disagree on their tape rows")?;
    if token_element != Element::u32() {
        return Err("the n-gram token window does not hold u32 tokens".into());
    }
    let mut window_port = graph.port(window_element, &window_extents)?;
    let mut token_window_port = graph.port(token_element, &token_extents)?;

    // The table rows of each batch row, from the step's token rows: a
    // device tensor whatever their source (the entry's kept copy of the
    // host's tokens, or the previous step's selection), bound per run.
    let tokens = graph.port_with_class_extent(Element::i32(), &[rows, 2], 0, "M")?;
    let rule = [
        GraphConstant::u64(graph, &hash.multipliers)?,
        GraphConstant::u64(graph, &hash.sizes)?,
        GraphConstant::u64(graph, &hash.offsets)?,
    ];
    let reset = i32::try_from(hash.reset).map_err(|_| "the n-gram reset token exceeds i32")?;
    let table_rows = graph
        .enqueue(
            entries.rows,
            &[
                ("M", rows),
                ("B", block.slots),
                ("N", banks),
                ("C", binding.context),
                ("P", binding.heads_per_order),
                ("T", tape),
            ],
            ngram_rows::WorkflowArgs {
                tokens: tokens.tensor().into(),
                multipliers: rule[0].port().tensor().into(),
                sizes: rule[1].port().tensor().into(),
                offsets: rule[2].port().tensor().into(),
                segments: controls.segments.tensor().into(),
                stop: controls.stop.tensor().into(),
                previous_bank: controls.previous_bank.tensor().into(),
                previous_tape: controls.previous_tape.tensor().into(),
                following_bank: controls.following_bank.tensor().into(),
                window: token_window_port.tensor_mut().into(),
                reset,
                slab_banks: block.slab_banks,
            },
        )?
        .value;
    constants.extend(rule);

    // Read and decode them: one table row of `width` per head, a batch
    // row's heads one after another.
    let mut table = GatheredTable::of(graph, load, scope, WeightKind::NGramTable, weights)?;
    let (decoded, table_ports) = stored_table_rows(
        graph,
        entries.stored,
        &mut table,
        load,
        weights,
        constants,
        &table_rows,
        rows * heads,
    )?;
    let decoded = decoded.reshape(&[rows, heads * binding.width]);

    let mut weight = |graph: &mut G, kind| weight(graph, load, scope, kind, weights);
    let key_norm = weight(graph, WeightKind::NGramKeyNorm)?.reshape(&[s, h]);
    let query_norm = weight(graph, WeightKind::NGramQueryNorm)?.reshape(&[s, h]);
    let convolution_norm = weight(graph, WeightKind::NGramConvolutionNorm)?.reshape(&[s, h]);
    let convolution = weight(graph, WeightKind::NGramConvolution)?;
    let mut project = |graph: &mut G,
                       entry: G::Binding<'a, project_rows::Entry>,
                       kind: WeightKind,
                       outputs: u64,
                       constants: &mut Vec<GraphConstant>| {
        let projection = packed_weight(graph, load, scope, kind, weights, constants)?;
        graph
            .enqueue(
                entry,
                &[
                    ("M", rows),
                    ("K", embedding),
                    ("N", outputs),
                    ("WS", projection.operands.scale.extent),
                    ("WB", projection.operands.bound.extent),
                ],
                project_rows::WorkflowArgs {
                    source: (&decoded).into(),
                    weight: (&projection.weight).into(),
                    weight_bound: (&projection.operands.bound.tensor).into(),
                    weight_scale: (&projection.operands.scale.tensor).into(),
                },
            )
            .map(|result| result.value)
    };
    let key = project(graph, entries.key, WeightKind::NGramKey, s * h, constants)?;
    let value = project(graph, entries.value, WeightKind::NGramValue, h, constants)?;
    let scales = graph
        .enqueue(
            entries.gate,
            &[("M", rows), ("H", h), ("S", s)],
            ngram_gate::WorkflowArgs {
                hidden: (&residual.reshape(&[rows, s, h])).into(),
                key: (&key.reshape(&[rows, s, h])).into(),
                value: (&value).into(),
                key_norm: (&key_norm).into(),
                query_norm: (&query_norm).into(),
                epsilon: block.epsilon,
            },
        )?
        .value;
    let streams = graph
        .enqueue(
            entries.inject,
            &[
                ("M", rows),
                ("B", block.slots),
                ("N", banks),
                ("R", window_rows),
                ("S", s),
                ("H", h),
                ("K", binding.kernel),
            ],
            ngram_inject::WorkflowArgs {
                hidden: residual.into(),
                scales: (&scales).into(),
                value: (&value).into(),
                norm: (&convolution_norm).into(),
                convolution: (&convolution).into(),
                segments: controls.segments.tensor().into(),
                stop: controls.stop.tensor().into(),
                previous_bank: controls.previous_bank.tensor().into(),
                previous_tape: controls.previous_tape.tensor().into(),
                following_bank: controls.following_bank.tensor().into(),
                window: window_port.tensor_mut().into(),
                dilation: i32::try_from(binding.dilation)
                    .map_err(|_| "n-gram dilation exceeds i32")?,
                slab_banks: block.slab_banks,
            },
        )?
        .value;
    Ok((
        streams,
        NGramPorts {
            window: window_port,
            token_window: token_window_port,
            step: NGramStepPorts {
                tokens,
                table: table_ports,
            },
        },
    ))
}

/// The banks of the target store's recurrent arenas.
fn bank_count(state: &StateResourcePlan) -> Result<u64, GraphError> {
    let store = state.target_state();
    Ok(u64::try_from(
        store
            .bank_capacity
            .storage_total()
            .map_err(|error| error.to_string())?,
    )
    .map_err(|_| "recurrent bank count exceeds u64")?)
}

/// The element and `[banks, ..]` extents of recurrent component `index`.
fn bank_component(
    state: &StateResourcePlan,
    banks: u64,
    index: usize,
) -> Result<(Element, Vec<u64>), GraphError> {
    let component = state
        .target_state()
        .recurrent_components
        .get(index)
        .ok_or("an n-gram window component is absent")?;
    let extents = std::iter::once(Ok(banks))
        .chain(component.shape.iter().map(|extent| {
            u64::try_from(*extent).map_err(|_| "recurrent state extent exceeds u64".to_owned())
        }))
        .collect::<Result<Vec<_>, String>>()?;
    Ok((Element::dense(component.dtype), extents))
}
