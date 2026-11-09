//! The n-gram input's graph fragment over the streams: each batch row's
//! table rows from its token row and the token window (`ngram_rows`), read
//! from the streamed table and decoded to activations (`stored_rows`,
//! through `stored_table_rows`), the key `[S * H]` and value `[H]`
//! projections into F32, the gate (`ngram_gate`) and the row op: the gated
//! value added (`ngram_add`), or added with its convolution over the
//! layer's window (`ngram_inject`). Slots and versions follow the recurrent
//! contract: the fragment binds its block's recurrent control ports, and
//! its windows are components of its block's bank.

use crate::NGramBinding;
use crate::native::{NGramKernels, NGramOutputKernel};
use crate::operators::gated_delta::graph::RecurrentControlPorts;
use crate::programs::graph::{
    GraphError,
    draft::{ControlSource, GraphDraft},
};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{
    GatheredTable, StoredReadPorts, WeightPort, packed_weight, stored_table_rows, weight,
};
use crate::{ModelLoadPlan, NGramGateForm, StateResourcePlan};
use magnitude_family_contracts::{NGramHash, WeightKind, WeightScope};
use magnitude_kernels::{
    ngram_add, ngram_gate, ngram_inject, ngram_rows, project_rows, stored_rows,
};
use seismic::{Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

pub(crate) struct NGramEntries<'a, G: GraphDraft + 'a> {
    pub rows: G::Binding<'a, ngram_rows::Entry>,
    pub stored: G::Binding<'a, stored_rows::Entry>,
    pub key: G::Binding<'a, project_rows::Entry>,
    pub value: G::Binding<'a, project_rows::Entry>,
    pub gate: G::Binding<'a, ngram_gate::Entry>,
    pub output: NGramOutputEntry<'a, G>,
}

impl<'a, G: GraphDraft + 'a> Copy for NGramEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for NGramEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// The row op's entry: `ngram_add` without a convolution, `ngram_inject`
/// with one.
pub(crate) enum NGramOutputEntry<'a, G: GraphDraft + 'a> {
    Add(G::Binding<'a, ngram_add::Entry>),
    Inject(G::Binding<'a, ngram_inject::Entry>),
}

impl<'a, G: GraphDraft + 'a> Copy for NGramOutputEntry<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for NGramOutputEntry<'a, G> {
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
            output: match &kernels.output {
                NGramOutputKernel::Add(kernel) => NGramOutputEntry::Add(kernel),
                NGramOutputKernel::Inject(kernel) => NGramOutputEntry::Inject(kernel),
            },
        }
    }
}

/// Entry element assignments from the binding.
pub(crate) struct CheckedNGramEntries {
    stored: [(&'static str, Element); 2],
    key: [(&'static str, Element); 3],
    value: [(&'static str, Element); 3],
    norms: [(&'static str, Element); 1],
    convolution: bool,
}

impl CheckedNGramEntries {
    pub(crate) fn new(binding: NGramBinding) -> Self {
        Self {
            stored: [("E", binding.table), ("A", binding.activation)],
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
            convolution: binding.convolution.is_some(),
        }
    }

    pub(crate) fn entries(&self) -> NGramEntries<'_, NativeGraphMetadata> {
        NGramEntries {
            rows: &[],
            stored: &self.stored,
            key: &self.key,
            value: &self.value,
            gate: &self.norms,
            output: match self.convolution {
                false => NGramOutputEntry::Add(&[][..]),
                true => NGramOutputEntry::Inject(&self.norms[..]),
            },
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
    /// Index of this layer's first component in the store's recurrent
    /// components: its convolution's window, when it has one, then its
    /// token window.
    pub component_index: usize,
    /// How the graph takes its launch's slot tables, where the block's
    /// bank has no other owner to take them.
    pub controls: ControlSource,
}

/// The fragment's state ports (its bank components, in component order)
/// and the ports each step binds.
pub(crate) struct NGramPorts {
    pub bank: Vec<NativePort>,
    pub step: NGramStepPorts,
    /// The slot tables it takes as its own inputs, in a block whose bank it
    /// alone keeps.
    pub controls: Option<RecurrentControlPorts>,
}

/// What each step binds of an n-gram input: its token rows (the entry's kept
/// copy of the host's tokens, or the previous step's selection, in the
/// `sample_rows` layout) and its table read's ports.
#[derive(Clone)]
pub(crate) struct NGramStepPorts {
    pub tokens: NativePort,
    pub table: StoredReadPorts,
}

/// The class scope of an n-gram input's nodes, whose `B` is its bank's
/// request slots: in an attention block, the class's slots, beside the
/// attention kernels' own static `B`.
pub(crate) const NGRAM_SLOTS: &str = "ngram_slots";

/// The n-gram input over the `[rows, S * H]` streams `residual`: the new
/// streams and its ports, its nodes in [`NGRAM_SLOTS`].
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
    controls: Option<&RecurrentControlPorts>,
    block: NGramBlock<'_>,
) -> Result<(WorkflowTensor, NGramPorts), GraphError> {
    graph.set_class_scope(Some(NGRAM_SLOTS));
    let binding = block.binding;
    let rows = block.rows;
    let (s, h) = (binding.streams, binding.hidden);
    let heads = binding.heads();
    let embedding = binding.embedding();
    let hash = block.hash;
    if crate::planning::NGramRowRule::of(hash) != binding.rule {
        return Err("the n-gram row rule disagrees with its binding".into());
    }
    let banks = bank_count(state)?;
    let token_index = block.component_index + usize::from(binding.convolution.is_some());
    let (token_element, token_extents) = bank_component(state, banks, token_index)?;
    let [_, token_rows, 1] = token_extents[..] else {
        return Err("the n-gram token window is not a bank of single tokens".into());
    };
    if token_element != Element::u32() {
        return Err("the n-gram token window does not hold u32 tokens".into());
    }
    let tape = token_rows
        .checked_sub(binding.rule.context)
        .ok_or("the n-gram token window is shorter than its context")?;
    // A convolution's window records the token window's tape rows.
    let mut window = match binding.convolution {
        None => None,
        Some(taps) => {
            let (element, extents) = bank_component(state, banks, block.component_index)?;
            let [_, window_rows, channels] = extents[..] else {
                return Err("the n-gram window is not a bank of rows".into());
            };
            if channels != s * h || window_rows <= taps.reach() || window_rows != taps.reach() + tape {
                return Err("the n-gram windows disagree on their tape rows".into());
            }
            Some((graph.port(element, &extents)?, window_rows, taps))
        }
    };
    let mut token_window_port = graph.port(token_element, &token_extents)?;

    // The table rows of each batch row, from the step's token rows: a
    // device tensor whatever their source (the entry's kept copy of the
    // host's tokens, or the previous step's selection), bound per run.
    let tokens = graph.port_with_class_extent(Element::i32(), &[rows, 2], 0, "M")?;
    let map = hash.token_map.as_deref().unwrap_or(&[]);
    let boundary = hash
        .boundary
        .map(|boundary| i32::try_from(boundary).map_err(|_| "the n-gram boundary exceeds i32"))
        .transpose()?;
    let rule = [
        GraphConstant::u64(graph, &hash.multipliers)?,
        GraphConstant::u64(graph, &hash.sizes)?,
        GraphConstant::u64(graph, &hash.offsets)?,
        GraphConstant::u32(graph, map)?,
        GraphConstant::i32(graph, boundary.as_slice())?,
    ];
    let fill = i32::try_from(hash.fill).map_err(|_| "the n-gram fill exceeds i32")?;
    let table_dimensions = [
        ("M", rows),
        ("B", block.slots),
        ("N", banks),
        ("T", tape),
    ]
    .into_iter()
    .chain(binding.rule.statics())
    .collect::<Vec<_>>();
    let mut made = None;
    let controls = match controls {
        Some(controls) => controls,
        None => {
            let mut table =
                |name| graph.control_for(block.controls, entries.rows, name, &table_dimensions);
            made.insert(RecurrentControlPorts {
                segments: table("segments")?,
                stop: table("stop")?,
                previous_bank: table("previous_bank")?,
                previous_tape: table("previous_tape")?,
                following_bank: table("following_bank")?,
            })
        }
    };
    let table_rows = graph
        .enqueue(
            entries.rows,
            &table_dimensions,
            ngram_rows::WorkflowArgs {
                tokens: tokens.tensor().into(),
                multipliers: rule[0].port().tensor().into(),
                sizes: rule[1].port().tensor().into(),
                offsets: rule[2].port().tensor().into(),
                map: rule[3].port().tensor().into(),
                boundary: rule[4].port().tensor().into(),
                segments: controls.segments.tensor().into(),
                stop: controls.stop.tensor().into(),
                previous_bank: controls.previous_bank.tensor().into(),
                previous_tape: controls.previous_tape.tensor().into(),
                following_bank: controls.following_bank.tensor().into(),
                window: token_window_port.tensor_mut().into(),
                fill,
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
        &table_rows,
        rows * heads,
    )?;
    let decoded = decoded.reshape(&[rows, heads * binding.width]);

    let mut weight = |graph: &mut G, kind| weight(graph, load, scope, kind, weights);
    // The gate's weights: the two norms, or the stored product beside zero
    // rows of a query norm.
    let (key_norm, query_norm) = match binding.gate {
        NGramGateForm::Norms => (
            weight(graph, WeightKind::NGramKeyNorm)?.reshape(&[s, h]),
            weight(graph, WeightKind::NGramQueryNorm)?.reshape(&[s, h]),
        ),
        NGramGateForm::Product => (
            weight(graph, WeightKind::NGramGate)?.reshape(&[s, h]),
            GraphConstant::empty_of(graph, binding.norm, &[0, h], constants)?.reshape(&[0, h]),
        ),
    };
    let gate_form = binding.gate.product();
    let form = GraphConstant::empty_of(graph, Element::i32(), &[0, gate_form], constants)?;
    let convolution = match window {
        None => None,
        Some(_) => Some((
            weight(graph, WeightKind::NGramConvolutionNorm)?.reshape(&[s, h]),
            weight(graph, WeightKind::NGramConvolution)?,
        )),
    };
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
                    ("G", 1),
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
            &[("M", rows), ("H", h), ("S", s), ("PR", gate_form)],
            ngram_gate::WorkflowArgs {
                hidden: (&residual.reshape(&[rows, s, h])).into(),
                key: (&key.reshape(&[rows, s, h])).into(),
                value: (&value).into(),
                key_norm: (&key_norm).into(),
                query_norm: (&query_norm).into(),
                form: (&form).into(),
                epsilon: block.epsilon,
            },
        )?
        .value;
    let streams = match (entries.output, window.as_mut().zip(convolution.as_ref())) {
        (NGramOutputEntry::Add(entry), None) => {
            graph
                .enqueue(
                    entry,
                    &[("M", rows), ("S", s), ("H", h)],
                    ngram_add::WorkflowArgs {
                        hidden: residual.into(),
                        scales: (&scales).into(),
                        value: (&value).into(),
                    },
                )?
                .value
        }
        (
            NGramOutputEntry::Inject(entry),
            Some(((window_port, window_rows, taps), (norm, convolution))),
        ) => {
            graph
                .enqueue(
                    entry,
                    &[
                        ("M", rows),
                        ("B", block.slots),
                        ("N", banks),
                        ("R", *window_rows),
                        ("S", s),
                        ("H", h),
                        ("K", taps.kernel),
                    ],
                    ngram_inject::WorkflowArgs {
                        hidden: residual.into(),
                        scales: (&scales).into(),
                        value: (&value).into(),
                        norm: norm.into(),
                        convolution: convolution.into(),
                        segments: controls.segments.tensor().into(),
                        stop: controls.stop.tensor().into(),
                        previous_bank: controls.previous_bank.tensor().into(),
                        previous_tape: controls.previous_tape.tensor().into(),
                        following_bank: controls.following_bank.tensor().into(),
                        window: window_port.tensor_mut().into(),
                        dilation: i32::try_from(taps.dilation)
                            .map_err(|_| "n-gram dilation exceeds i32")?,
                        slab_banks: block.slab_banks,
                    },
                )?
                .value
        }
        _ => return Err("the n-gram row op disagrees with its window".into()),
    };
    let bank = window
        .map(|(port, _, _)| port)
        .into_iter()
        .chain([token_window_port])
        .collect();
    graph.set_class_scope(None);
    Ok((
        streams,
        NGramPorts {
            bank,
            step: NGramStepPorts {
                tokens,
                table: table_ports,
            },
            controls: made,
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
