//! Per-layer inputs (Gemma PLE, model-family plan §3.7).
//!
//! The per-layer entry graph runs once per step, after the embedding and its
//! media overlays: it rounds the embedded rows to activations and projects
//! them to every layer's channels, converts the batch rows' uploaded
//! host-table rows to the table's resident representation, combines both
//! (`per_layer_inputs`), and copies the result into the program's per-layer
//! rows. Each block's per-layer input sublayer then gates its layer's slice
//! of those rows (`per_layer_gate`) and adds its projection through its
//! post-norm tail.

use crate::native::{PerLayerEntryKernels, PerLayerKernels, TableConversion};
use crate::operators::output::{CheckedPostNormEntries, PostNorm, PostNormEntries, PostNormShape, post_norm};
use crate::programs::graph::facts::Numbers;
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_target_graph::{PackedWeight, ScaledWeight, WeightView};
use crate::{PerLayerBinding, PerLayerEntryBinding, SublayerTail};
use magnitude_family_contracts::PerLayerEntry;
use magnitude_kernels::{
    conditioning_overlay, import_dense, per_layer_gate, per_layer_inputs, project_rows,
    repack_weight,
};
use seismic::{Argument, Dim, Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

/// The numbers of the per-layer rows: the residual's width, the layers and
/// each layer's width.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PerLayerNumbers<D = Dim> {
    pub hidden: D,
    pub layers: D,
    pub width: D,
}

impl<D> Numbers<D> for PerLayerNumbers<D> {
    type Of<E> = PerLayerNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> PerLayerNumbers<E> {
        PerLayerNumbers { hidden: f(self.hidden), layers: f(self.layers), width: f(self.width) }
    }
}

impl PerLayerNumbers<u64> {
    /// The numbers of a per-layer input sublayer's `binding`.
    pub(crate) fn of(binding: &PerLayerBinding) -> Self {
        Self { hidden: binding.hidden, layers: binding.layers, width: binding.width }
    }
}

/// What decides the per-layer entry graph beside its numbers and its
/// weights: the reals its combine takes (its projection norm's epsilon and
/// its scales), `R` (`RecurrentTopology`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PerLayerEntryTopology<R = Argument<f64>> {
    pub epsilon: R,
    pub gathered_scale: R,
    pub projected_scale: R,
    pub scale: R,
}

impl<R> Numbers<R> for PerLayerEntryTopology<R> {
    type Of<E> = PerLayerEntryTopology<E>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> PerLayerEntryTopology<E> {
        PerLayerEntryTopology {
            epsilon: f(self.epsilon),
            gathered_scale: f(self.gathered_scale),
            projected_scale: f(self.projected_scale),
            scale: f(self.scale),
        }
    }
}

/// The topology and numbers of the per-layer entry of `binding` and
/// `entry`.
pub(crate) fn split_entry(
    binding: &PerLayerEntryBinding,
    entry: &PerLayerEntry,
) -> (PerLayerEntryTopology<f64>, PerLayerNumbers<u64>) {
    (
        PerLayerEntryTopology {
            epsilon: entry.projection_norm.epsilon,
            gathered_scale: entry.table_scale,
            projected_scale: entry.projection_scale,
            scale: entry.combine_scale,
        },
        PerLayerNumbers { hidden: binding.hidden, layers: binding.layers, width: binding.width },
    )
}

/// How the uploaded table rows become resident rows.
pub(crate) enum TableEntry<'a, G: GraphDraft + 'a> {
    Dense(G::Binding<'a, import_dense::Entry>),
    Repack(G::Binding<'a, repack_weight::Entry>),
}

impl<'a, G: GraphDraft + 'a> Copy for TableEntry<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for TableEntry<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

pub(crate) struct PerLayerEntryEntries<'a, G: GraphDraft + 'a> {
    pub round: G::Binding<'a, import_dense::Entry>,
    pub project: G::Binding<'a, project_rows::Entry>,
    pub table: TableEntry<'a, G>,
    pub inputs: G::Binding<'a, per_layer_inputs::Entry>,
    pub copy: G::Binding<'a, conditioning_overlay::Entry>,
}

impl<'a> From<&'a PerLayerEntryKernels> for PerLayerEntryEntries<'a, NativeGraph> {
    fn from(kernels: &'a PerLayerEntryKernels) -> Self {
        Self {
            round: &kernels.round,
            project: &kernels.project,
            table: match &kernels.table {
                TableConversion::Dense(kernel) => TableEntry::Dense(kernel),
                TableConversion::Repack(kernel) => TableEntry::Repack(kernel),
            },
            inputs: &kernels.inputs,
            copy: &kernels.copy,
        }
    }
}

/// Entry element assignments of the per-layer entry graph from its binding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedPerLayerEntryEntries {
    round: [(&'static str, Element); 2],
    project: [(&'static str, Element); 3],
    table: [(&'static str, Element); 2],
    dense_table: bool,
    inputs: [(&'static str, Element); 2],
    copy: [(&'static str, Element); 0],
}

impl CheckedPerLayerEntryEntries {
    pub(crate) fn new(binding: PerLayerEntryBinding) -> Self {
        Self {
            round: [("E", Element::f32()), ("U", binding.activation)],
            project: [
                ("A", binding.activation),
                ("W", binding.projection),
                ("Y", Element::f32()),
            ],
            table: [("E", binding.table_source), ("U", binding.table)],
            dense_table: binding.table_source.dtype().is_some() && binding.table.dtype().is_some(),
            inputs: [("TW", binding.table), ("NW", binding.norm)],
            copy: [],
        }
    }

    pub(crate) fn entries(&self) -> PerLayerEntryEntries<'_, NativeGraphMetadata> {
        PerLayerEntryEntries {
            round: &self.round,
            project: &self.project,
            table: if self.dense_table {
                TableEntry::Dense(&self.table[..])
            } else {
                TableEntry::Repack(&self.table[..])
            },
            inputs: &self.inputs,
            copy: &self.copy,
        }
    }
}

/// The per-layer entry graph's ports: the embedded (and overlaid) hidden
/// rows, the uploaded table rows, the per-layer rows it writes, and its
/// projection and norm weights.
#[derive(Clone)]
pub(crate) struct PerLayerEntryPorts {
    pub hidden: NativePort,
    pub gathered: NativePort,
    pub rows: NativePort,
    pub projection: NativePort,
    pub norm: NativePort,
    /// The operand an absent scale or bound port binds.
    pub absent: NativePort,
    /// The projection's second-level scale (`WeightPlan::scale_extent`),
    /// where it has one.
    pub projection_scale: Option<NativePort>,
    /// The projection's packing bound (`WeightPlan::bound`), where it has
    /// one.
    pub projection_bound: Option<NativePort>,
}

/// The class slice of the per-layer entry over each of `rows`: every row
/// axis is the batch row class. The entry imports name their row axis `N`.
pub(crate) fn per_layer_entry_class_slice(
    rows: impl IntoIterator<Item = u64> + Clone,
) -> seismic::NativeGraphClassSlice {
    seismic::NativeGraphClassSlice::new()
        .dimension("M", rows.clone())
        .scoped("import_dense", "N", rows.clone())
        .scoped("repack_weight", "N", rows)
}

/// The per-layer entry over `rows` batch rows, its `projection` and `norm`
/// weights' views.
pub(crate) fn per_layer_entry<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: PerLayerEntryEntries<'a, G>,
    topology: &PerLayerEntryTopology,
    numbers: &PerLayerNumbers,
    projection: &WeightView,
    norm: &WeightView,
    rows: &Dim,
) -> Result<PerLayerEntryPorts, GraphError> {
    let PerLayerNumbers { hidden: width, layers, width: layer_width } = numbers;
    let channels = layers * layer_width;
    // The projection's scale and packing bound where it has them; the
    // absent operand binds their ports where it has none.
    let mut operand = |extent: &Option<Dim>| {
        extent
            .as_ref()
            .map(|extent| graph.port(Element::f32(), std::slice::from_ref(extent)))
            .transpose()
    };
    let projection_scale = operand(&projection.scale)?;
    let projection_bound = operand(&projection.bound)?;
    let none = || Dim::from(0);
    let (scale_extent, bound_extent) =
        (projection.scale.clone().unwrap_or_else(none), projection.bound.clone().unwrap_or_else(none));
    let projection = graph.port(projection.resident, &projection.shape)?;
    let norm = graph.port(norm.resident, &norm.shape)?;
    let absent = graph.port(Element::f32(), &[0])?;
    let hidden = graph.port_with_class_extent(Element::f32(), &[rows.clone(), width.clone()], 0, "M")?;
    let table_dimensions = [("B", Dim::from(1)), ("N", rows.clone()), ("K", channels.clone())];
    let gathered = match entries.table {
        TableEntry::Dense(kernel) => graph.input_for(kernel, "source", &table_dimensions)?,
        TableEntry::Repack(kernel) => graph.input_for(kernel, "source", &table_dimensions)?,
    };
    let mut destination =
        graph.port_with_class_extent(Element::f32(), &[rows.clone(), channels.clone()], 0, "M")?;
    let rounded = graph
        .enqueue(
            entries.round,
            &[("B", Dim::from(1)), ("N", rows.clone()), ("K", width.clone())],
            import_dense::WorkflowArgs {
                source: (&hidden.tensor().reshape(&[Dim::from(1), rows.clone(), width.clone()])).into(),
            },
        )?
        .value;
    let projected = graph
        .enqueue(
            entries.project,
            &[
                ("M", rows.clone()),
                ("G", Dim::from(1)),
                ("K", width.clone()),
                ("N", channels.clone()),
                ("WS", scale_extent),
                ("WB", bound_extent),
            ],
            project_rows::WorkflowArgs {
                source: (&rounded.reshape(&[rows.clone(), width.clone()])).into(),
                weight: projection.tensor().into(),
                weight_scale: projection_scale.as_ref().unwrap_or(&absent).tensor().into(),
                weight_bound: projection_bound.as_ref().unwrap_or(&absent).tensor().into(),
            },
        )?
        .value;
    let table = match entries.table {
        TableEntry::Dense(kernel) => {
            graph
                .enqueue(
                    kernel,
                    &table_dimensions,
                    import_dense::WorkflowArgs {
                        source: gathered.tensor().into(),
                    },                )?
                .value
        }
        TableEntry::Repack(kernel) => {
            graph
                .enqueue(
                    kernel,
                    &table_dimensions,
                    repack_weight::WorkflowArgs {
                        source: gathered.tensor().into(),
                    },                )?
                .value
        }
    };
    let combined = graph
        .enqueue(
            entries.inputs,
            &[("M", rows.clone()), ("L", layers.clone()), ("P", layer_width.clone())],
            per_layer_inputs::WorkflowArgs {
                gathered: (&table.reshape(&[rows.clone(), channels.clone()])).into(),
                projected: (&projected).into(),
                norm: norm.tensor().into(),
                epsilon: topology.epsilon.f32(),
                gathered_scale: topology.gathered_scale.f32(),
                projected_scale: topology.projected_scale.f32(),
                scale: topology.scale.f32(),
            },
        )?
        .value;
    graph.enqueue(
        entries.copy,
        &[("M", rows.clone()), ("D", channels)],
        conditioning_overlay::WorkflowArgs {
            input: (&combined).into(),
            out: destination.tensor_mut().into(),
        },    )?;
    Ok(PerLayerEntryPorts {
        hidden,
        gathered,
        rows: destination,
        projection,
        norm,
        absent,
        projection_scale,
        projection_bound,
    })
}

pub(crate) struct PerLayerEntries<'a, G: GraphDraft + 'a> {
    pub gate: G::Binding<'a, per_layer_gate::Entry>,
    pub output: PostNormEntries<'a, G>,
}

impl<'a> From<&'a PerLayerKernels> for PerLayerEntries<'a, NativeGraph> {
    fn from(kernels: &'a PerLayerKernels) -> Self {
        Self {
            gate: &kernels.gate,
            output: PostNormEntries {
                project: &kernels.output.project,
                residual: &kernels.output.residual,
            },
        }
    }
}

/// Entry element assignments of a per-layer input sublayer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedPerLayerEntries {
    gate: [(&'static str, Element); 2],
    output: CheckedPostNormEntries,
}

impl CheckedPerLayerEntries {
    pub(crate) fn new(binding: PerLayerBinding) -> Result<Self, String> {
        let SublayerTail::PostNorm { norm, .. } = binding.tail else {
            return Err("a per-layer input sublayer ends in a post-norm tail".into());
        };
        Ok(Self {
            gate: [("GW", binding.gate), ("A", binding.activation)],
            output: CheckedPostNormEntries::new(binding.projection, binding.activation, norm),
        })
    }

    pub(crate) fn entries(&self) -> PerLayerEntries<'_, NativeGraphMetadata> {
        PerLayerEntries {
            gate: &self.gate,
            output: self.output.entries(),
        }
    }
}

/// The weight ports of a per-layer input sublayer.
pub(crate) struct PerLayerWeights {
    pub gate: ScaledWeight,
    pub projection: PackedWeight,
    pub post_norm: WorkflowTensor,
}

/// A per-layer input sublayer over `residual` (`[rows, D]` F32): its layer's
/// gate over `inputs` (the per-layer rows, `[rows, L·P]` F32), the
/// projection back to D, and the post-norm tail; returns the new residual.
#[allow(clippy::too_many_arguments)]
pub(crate) fn per_layer<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: PerLayerEntries<'a, G>,
    weights: &PerLayerWeights,
    residual: &WorkflowTensor,
    inputs: &WorkflowTensor,
    out_rows: &WorkflowTensor,
    numbers: &PerLayerNumbers,
    layer: Argument<i32>,
    activation: Argument<i32>,
    rows: &Dim,
    tail: &PostNorm,
) -> Result<WorkflowTensor, GraphError> {
    let PerLayerNumbers { hidden, layers, width } = numbers;
    let gated = graph
        .enqueue(
            entries.gate,
            &[
                ("M", rows.clone()),
                ("D", hidden.clone()),
                ("L", layers.clone()),
                ("P", width.clone()),
                ("GS", weights.gate.scale.extent.clone()),
            ],
            per_layer_gate::WorkflowArgs {
                hidden: residual.into(),
                gate_weight: (&weights.gate.weight).into(),
                inputs: (&inputs.reshape(&[rows.clone(), layers.clone(), width.clone()])).into(),
                layer,
                activation,
                gate_scale: (&weights.gate.scale.tensor).into(),
            },        )?
        .value;
    post_norm(
        graph,
        entries.output,
        residual,
        (&gated).into(),
        &weights.projection,
        &weights.post_norm,
        out_rows,
        PostNormShape {
            rows: rows.clone(),
            out: rows.clone(),
            inputs: width.clone(),
            outputs: hidden.clone(),
        },
        tail,
    )
}
