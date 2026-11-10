//! The n-gram input's graph fragment over the streams: each batch row's
//! table rows from its token row and the token window (`ngram_rows`), read
//! from the streamed table and decoded to activations (`stored_rows`,
//! through `stored_table_rows`), the key `[S * H]` and value `[H]`
//! projections into F32, the gate (`ngram_gate`) and the row op: the gated
//! value added (`ngram_add`), or added with its convolution over the
//! layer's window (`ngram_inject`). Slots and versions follow the recurrent
//! contract: the fragment binds its block's recurrent control ports, and
//! its windows are components of its block's bank.

use crate::native::{NGramKernels, NGramOutputKernel};
use crate::operators::gated_delta::graph::RecurrentControlPorts;
use crate::operators::split_bank;
use crate::programs::graph::facts::Numbers;
use crate::programs::graph::{
    GraphError,
    draft::{ControlSource, GraphDraft},
};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{
    GatheredTable, PlannedWeights, StoredReadPorts, WeightPort, packed_weight, stored_table_rows, weight,
};
use crate::{NGramBinding, NGramGateForm, StateResourcePlan};
use magnitude_family_contracts::{NGramHash, WeightKind, WeightScope};
use magnitude_kernels::{
    ngram_add, ngram_gate, ngram_inject, ngram_rows, project_rows, stored_rows,
};
use seismic::{Argument, Dim, Element, NativeGraph, NativeGraphMetadata, NativePort, Table, WorkflowTensor};
use std::sync::Arc;

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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

/// What decides an n-gram input's graph beside its numbers
/// ([`NGramNumbers`]): its binding's forms and elements, its row rule, its
/// hash's fill and tables `T` (its graph's constants, `NGramTables`), its
/// gate's epsilon, how it takes its slot tables, and its windows' elements.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NGramTopology<R = Argument<f64>, T = Table> {
    pub streams: u64,
    pub rule: crate::planning::NGramRowRule,
    pub gate: NGramGateForm,
    pub convolution: Option<crate::planning::NGramTaps>,
    pub norm: Element,
    /// The hashed id a missing or cut predecessor reads.
    pub fill: i32,
    pub tables: NGramTables<T>,
    pub epsilon: R,
    pub controls: ControlSource,
    /// The token window's element, and the convolution window's.
    pub windows: (Element, Option<Element>),
}

/// The n-gram hash's tables, as `ngram_rows` reads them, each sized by the
/// row rule: its multipliers (`[C + 1]` u64), its heads' sizes and offsets
/// (`[C * P]` u64 each), its token map (`[V]` u32) and its boundary token
/// (`[NB]` i32).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NGramTables<T> {
    pub multipliers: T,
    pub sizes: T,
    pub offsets: T,
    pub map: T,
    pub boundary: T,
}

impl<R, T> NGramTopology<R, T> {
    /// It with its tables `f` of its own (`AttentionTopology::tables`).
    pub(crate) fn tables<U>(self, mut f: impl FnMut(T) -> U) -> NGramTopology<R, U> {
        let NGramTables { multipliers, sizes, offsets, map, boundary } = self.tables;
        NGramTopology {
            streams: self.streams,
            rule: self.rule,
            gate: self.gate,
            convolution: self.convolution,
            norm: self.norm,
            fill: self.fill,
            tables: NGramTables {
                multipliers: f(multipliers),
                sizes: f(sizes),
                offsets: f(offsets),
                map: f(map),
                boundary: f(boundary),
            },
            epsilon: self.epsilon,
            controls: self.controls,
            windows: self.windows,
        }
    }
}

impl<R, T> Numbers<R> for NGramTopology<R, T> {
    type Of<E> = NGramTopology<E, T>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> NGramTopology<E, T> {
        NGramTopology {
            streams: self.streams,
            rule: self.rule,
            gate: self.gate,
            convolution: self.convolution,
            norm: self.norm,
            fill: self.fill,
            tables: self.tables,
            epsilon: f(self.epsilon),
            controls: self.controls,
            windows: self.windows,
        }
    }
}

/// An n-gram input's numbers at one graph class: its rows and slots, the
/// streams' width, the width of a table row, and its bank: the banks, those
/// of a slab, and the token window's (and the convolution window's) extents
/// past the bank axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NGramNumbers<D = Dim> {
    pub rows: D,
    pub slots: D,
    pub hidden: D,
    pub width: D,
    pub banks: D,
    pub slab_banks: D,
    pub token: Vec<D>,
    pub window: Option<Vec<D>>,
}

impl<D> Numbers<D> for NGramNumbers<D> {
    type Of<E> = NGramNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> NGramNumbers<E> {
        NGramNumbers {
            rows: f(self.rows),
            slots: f(self.slots),
            hidden: f(self.hidden),
            width: f(self.width),
            banks: f(self.banks),
            slab_banks: f(self.slab_banks),
            token: self.token.map(&mut *f),
            window: self.window.map(|window| window.map(&mut *f)),
        }
    }
}

/// The n-gram input of `binding` and `hash` at a class of `rows` rows and
/// `slots` slots, its windows the store's recurrent components from
/// `component_index` on (the convolution's window, when it has one, then
/// the token window).
#[allow(clippy::too_many_arguments)]
pub(crate) fn split(
    binding: NGramBinding,
    hash: &NGramHash,
    rows: u64,
    slots: u64,
    epsilon: f32,
    state: &StateResourcePlan,
    component_index: usize,
    controls: ControlSource,
) -> Result<(NGramTopology<f64, Arc<[u8]>>, NGramNumbers<u64>), GraphError> {
    if crate::planning::NGramRowRule::of(hash) != binding.rule {
        return Err("the n-gram row rule disagrees with its binding".into());
    }
    let (elements, numbers) = match binding.convolution {
        None => {
            let ([token], bank) = split_bank::<1>(state, component_index)?;
            let [token_extents] = bank.components;
            ((token, None), (bank.banks, bank.slab_banks, token_extents, None))
        }
        Some(_) => {
            let ([window, token], bank) = split_bank::<2>(state, component_index)?;
            let [window_extents, token_extents] = bank.components;
            ((token, Some(window)), (bank.banks, bank.slab_banks, token_extents, Some(window_extents)))
        }
    };
    let (banks, slab_banks, token, window) = numbers;
    let boundary = hash
        .boundary
        .map(|boundary| i32::try_from(boundary).map_err(|_| "the n-gram boundary exceeds i32"))
        .transpose()?;
    Ok((
        NGramTopology {
            streams: binding.streams,
            rule: binding.rule,
            gate: binding.gate,
            convolution: binding.convolution,
            norm: binding.norm,
            fill: i32::try_from(hash.fill).map_err(|_| "the n-gram fill exceeds i32")?,
            tables: NGramTables {
                multipliers: bytes(&hash.multipliers, u64::to_le_bytes),
                sizes: bytes(&hash.sizes, u64::to_le_bytes),
                offsets: bytes(&hash.offsets, u64::to_le_bytes),
                map: bytes(hash.token_map.as_deref().unwrap_or(&[]), u32::to_le_bytes),
                boundary: bytes(boundary.as_slice(), i32::to_le_bytes),
            },
            epsilon: f64::from(epsilon),
            controls,
            windows: elements,
        },
        NGramNumbers { rows, slots, hidden: binding.hidden, width: binding.width, banks, slab_banks, token, window },
    ))
}

/// `values` as a table's bytes, each value's `word`.
fn bytes<V: Copy, const N: usize>(values: &[V], word: fn(V) -> [u8; N]) -> Arc<[u8]> {
    values.iter().copied().flat_map(word).collect()
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
    topology: &NGramTopology,
    numbers: &NGramNumbers,
    load: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    residual: &WorkflowTensor,
    controls: Option<&RecurrentControlPorts>,
) -> Result<(WorkflowTensor, NGramPorts), GraphError> {
    graph.set_class_scope(Some(NGRAM_SLOTS));
    let rows = &numbers.rows;
    let (s, h) = (Dim::from(topology.streams), &numbers.hidden);
    let rule = topology.rule;
    let heads = rule.context * rule.heads_per_order;
    let embedding = &numbers.width * heads;
    let banks = &numbers.banks;
    let slab_banks = numbers.slab_banks.word::<u32>().ok_or("bank slab count exceeds u32")?;
    let (token_element, window_element) = topology.windows;
    let [token_rows, unit] = &numbers.token[..] else {
        return Err("the n-gram token window is not a bank of single tokens".into());
    };
    if !unit.equals(1) || token_element != Element::u32() {
        return Err("the n-gram token window does not hold u32 tokens".into());
    }
    let token_extents = [banks, token_rows, unit];
    let tape = token_rows
        .checked_sub(topology.rule.context)
        .ok_or("the n-gram token window is shorter than its context")?;
    // A convolution's window records the token window's tape rows.
    let mut window = match (topology.convolution, &numbers.window, window_element) {
        (None, None, None) => None,
        (Some(taps), Some(extents), Some(element)) => {
            let [window_rows, channels] = &extents[..] else {
                return Err("the n-gram window is not a bank of rows".into());
            };
            if !channels.equals(&s * h)
                || window_rows.at_most(taps.reach())
                || !window_rows.equals(&tape + taps.reach())
            {
                return Err("the n-gram windows disagree on their tape rows".into());
            }
            Some((graph.port(element, &[banks, window_rows, channels])?, window_rows.clone(), taps))
        }
        _ => return Err("the n-gram convolution disagrees with its windows".into()),
    };
    let mut token_window_port = graph.port(token_element, &token_extents)?;

    // The table rows of each batch row, from the step's token rows: a
    // device tensor whatever their source (the entry's kept copy of the
    // host's tokens, or the previous step's selection), bound per run.
    let tokens = graph.port_with_class_extent(Element::i32(), &[rows.clone(), Dim::from(2)], 0, "M")?;
    let tables = &topology.tables;
    let table = |graph: &mut G, element, extent: u64, table| GraphConstant::table(graph, element, &[Dim::from(extent)], table);
    let rule_tables = [
        table(graph, Element::u64(), rule.context + 1, &tables.multipliers)?,
        table(graph, Element::u64(), heads, &tables.sizes)?,
        table(graph, Element::u64(), heads, &tables.offsets)?,
        table(graph, Element::u32(), rule.mapped, &tables.map)?,
        table(graph, Element::i32(), rule.boundaries, &tables.boundary)?,
    ];
    let table_dimensions = [
        ("M", rows.clone()),
        ("B", numbers.slots.clone()),
        ("N", banks.clone()),
        ("T", tape),
    ]
    .into_iter()
    .chain(rule.statics().map(|(name, value)| (name, Dim::from(value))))
    .collect::<Vec<_>>();
    let mut made = None;
    let controls = match controls {
        Some(controls) => controls,
        None => {
            let mut table =
                |name| graph.control_for(topology.controls, entries.rows, name, &table_dimensions);
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
                multipliers: rule_tables[0].port().tensor().into(),
                sizes: rule_tables[1].port().tensor().into(),
                offsets: rule_tables[2].port().tensor().into(),
                map: rule_tables[3].port().tensor().into(),
                boundary: rule_tables[4].port().tensor().into(),
                segments: controls.segments.tensor().into(),
                stop: controls.stop.tensor().into(),
                previous_bank: controls.previous_bank.tensor().into(),
                previous_tape: controls.previous_tape.tensor().into(),
                following_bank: controls.following_bank.tensor().into(),
                window: token_window_port.tensor_mut().into(),
                fill: topology.fill.into(),
                slab_banks,
            },
        )?
        .value;
    constants.extend(rule_tables);

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
        &(rows * heads),
    )?;
    let decoded = decoded.reshape(&[rows.clone(), embedding.clone()]);

    let mut weight = |graph: &mut G, kind| weight(graph, load, scope, kind, weights);
    // The gate's weights: the two norms, or the stored product beside zero
    // rows of a query norm.
    let (key_norm, query_norm) = match topology.gate {
        NGramGateForm::Norms => (
            weight(graph, WeightKind::NGramKeyNorm)?.reshape(&[&s, h]),
            weight(graph, WeightKind::NGramQueryNorm)?.reshape(&[&s, h]),
        ),
        NGramGateForm::Product => (
            weight(graph, WeightKind::NGramGate)?.reshape(&[&s, h]),
            GraphConstant::empty_of(graph, topology.norm, &[&Dim::from(0), h], constants)?
                .reshape(&[&Dim::from(0), h]),
        ),
    };
    let gate_form = topology.gate.product();
    let form = GraphConstant::empty_of(graph, Element::i32(), &[0, gate_form], constants)?;
    let convolution = match window {
        None => None,
        Some(_) => Some((
            weight(graph, WeightKind::NGramConvolutionNorm)?.reshape(&[&s, h]),
            weight(graph, WeightKind::NGramConvolution)?,
        )),
    };
    let mut project = |graph: &mut G,
                       entry: G::Binding<'a, project_rows::Entry>,
                       kind: WeightKind,
                       outputs: Dim,
                       constants: &mut Vec<GraphConstant>| {
        let projection = packed_weight(graph, load, scope, kind, weights, constants)?;
        graph
            .enqueue(
                entry,
                &[
                    ("M", rows.clone()),
                    ("G", Dim::from(1)),
                    ("K", embedding.clone()),
                    ("N", outputs),
                    ("WS", projection.operands.scale.extent.clone()),
                    ("WB", projection.operands.bound.extent.clone()),
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
    let key = project(graph, entries.key, WeightKind::NGramKey, &s * h, constants)?;
    let value = project(graph, entries.value, WeightKind::NGramValue, h.clone(), constants)?;
    let scales = graph
        .enqueue(
            entries.gate,
            &[("M", rows), ("H", h), ("S", &s), ("PR", &Dim::from(gate_form))],
            ngram_gate::WorkflowArgs {
                hidden: (&residual.reshape(&[rows, &s, h])).into(),
                key: (&key.reshape(&[rows, &s, h])).into(),
                value: (&value).into(),
                key_norm: (&key_norm).into(),
                query_norm: (&query_norm).into(),
                form: (&form).into(),
                epsilon: topology.epsilon.f32(),
            },
        )?
        .value;
    let streams = match (entries.output, window.as_mut().zip(convolution.as_ref())) {
        (NGramOutputEntry::Add(entry), None) => {
            graph
                .enqueue(
                    entry,
                    &[("M", rows), ("S", &s), ("H", h)],
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
                        ("B", &numbers.slots),
                        ("N", banks),
                        ("R", &*window_rows),
                        ("S", &s),
                        ("H", h),
                        ("K", &Dim::from(taps.kernel)),
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
                            .map_err(|_| "n-gram dilation exceeds i32")?
                            .into(),
                        slab_banks,
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
