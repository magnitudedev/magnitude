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

use crate::operators::output::{
    post_norm, CheckedPostNormEntries, PostNormEntries, PostNormShape,
};
use crate::programs::graph::draft::GraphDraft;
use crate::native::{PerLayerEntryKernels, PerLayerKernels, TableConversion};
use crate::{PerLayerBinding, PerLayerEntryBinding, SublayerTail, WeightPlan};
use magnitude_family_contracts::PerLayerEntry;
use magnitude_kernels::{
    conditioning_overlay, import_dense, per_layer_gate, per_layer_inputs, project_rows,
    repack_weight,
};
use seismic::{Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

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
    pub absent_scale: NativePort,
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

/// The per-layer entry over `rows` batch rows.
pub(crate) fn per_layer_entry<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: PerLayerEntryEntries<'a, G>,
    projection: &WeightPlan,
    norm: &WeightPlan,
    binding: PerLayerEntryBinding,
    entry: &PerLayerEntry,
    rows: u64,
) -> Result<PerLayerEntryPorts, String> {
    let channels = binding.layers * binding.width;
    let projection = graph.port(projection.resident, &projection.shape)?;
    let norm = graph.port(norm.resident, &norm.shape)?;
    let absent_scale = graph.port(Element::f32(), &[0])?;
    let hidden = graph.port_with_class_extent(Element::f32(), &[rows, binding.hidden], 0, "M")?;
    let table_dimensions = [("B", 1), ("N", rows), ("K", channels)];
    let gathered = match entries.table {
        TableEntry::Dense(kernel) => graph.input_for(kernel, "source", &table_dimensions)?,
        TableEntry::Repack(kernel) => graph.input_for(kernel, "source", &table_dimensions)?,
    };
    let mut destination =
        graph.port_with_class_extent(Element::f32(), &[rows, channels], 0, "M")?;
    let rounded = graph
        .enqueue(
            entries.round,
            &[("B", 1), ("N", rows), ("K", binding.hidden)],
            import_dense::WorkflowArgs {
                source: (&hidden.tensor().reshape(&[1, rows, binding.hidden])).into(),
            },
        )?
        .value;
    let projected = graph
        .enqueue(
            entries.project,
            &[("M", rows), ("K", binding.hidden), ("N", channels), ("WS", 0)],
            project_rows::WorkflowArgs {
                source: (&rounded.reshape(&[rows, binding.hidden])).into(),
                weight: projection.tensor().into(),
                weight_scale: absent_scale.tensor().into(),
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
                    },
                )?
                .value
        }
        TableEntry::Repack(kernel) => {
            graph
                .enqueue(
                    kernel,
                    &table_dimensions,
                    repack_weight::WorkflowArgs {
                        source: gathered.tensor().into(),
                    },
                )?
                .value
        }
    };
    let combined = graph
        .enqueue(
            entries.inputs,
            &[("M", rows), ("L", binding.layers), ("P", binding.width)],
            per_layer_inputs::WorkflowArgs {
                gathered: (&table.reshape(&[rows, channels])).into(),
                projected: (&projected).into(),
                norm: norm.tensor().into(),
                epsilon: entry.projection_norm.epsilon as f32,
                gathered_scale: entry.table_scale as f32,
                projected_scale: entry.projection_scale as f32,
                scale: entry.combine_scale as f32,
            },
        )?
        .value;
    graph.enqueue(
        entries.copy,
        &[("M", rows), ("D", channels)],
        conditioning_overlay::WorkflowArgs {
            input: (&combined).into(),
            out: destination.tensor_mut().into(),
        },
    )?;
    Ok(PerLayerEntryPorts {
        hidden,
        gathered,
        rows: destination,
        projection,
        norm,
        absent_scale,
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
    pub gate: WorkflowTensor,
    pub projection: WorkflowTensor,
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
    weight_scale: &WorkflowTensor,
    residual: &WorkflowTensor,
    inputs: &WorkflowTensor,
    out_rows: &WorkflowTensor,
    binding: PerLayerBinding,
    layer: i32,
    activation: i32,
    rows: u64,
    epsilon: f32,
    scale: f32,
) -> Result<WorkflowTensor, String> {
    let gated = graph
        .enqueue(
            entries.gate,
            &[
                ("M", rows),
                ("D", binding.hidden),
                ("L", binding.layers),
                ("P", binding.width),
                ("GS", 0),
            ],
            per_layer_gate::WorkflowArgs {
                hidden: residual.into(),
                gate_weight: (&weights.gate).into(),
                inputs: (&inputs.reshape(&[rows, binding.layers, binding.width])).into(),
                layer,
                activation,
                gate_scale: weight_scale.into(),
            },
        )?
        .value;
    post_norm(
        graph,
        entries.output,
        residual,
        (&gated).into(),
        &weights.projection,
        weight_scale,
        &weights.post_norm,
        out_rows,
        PostNormShape {
            rows,
            out: rows,
            inputs: binding.width,
            outputs: binding.hidden,
        },
        epsilon,
        scale,
    )
}
