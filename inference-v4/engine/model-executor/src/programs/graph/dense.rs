//! Dense feed-forward block graph: the RMS-normed paired gate/up projection
//! with SiLU·mul, then the down projection plus the residual. Both entries
//! gather their rows through `out_rows`; a block that advances every row binds
//! the identity table as a graph constant, so no run uploads it.

use super::super::native_constants::GraphConstant;
use super::super::native_target_graph::weight;
use super::draft::GraphDraft;
use crate::{native::DenseKernels, DenseBinding, ModelLoadPlan};
use magnitude_model_contracts::{WeightKind, WeightRole, WeightScope};
use magnitude_model_kernels::{dense_expand, dense_output};
use seismic::{Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

/// The same dense topology accepts prepared entries or checked metadata
/// bindings. The latter carries only element assignments, never shapes.
pub(crate) struct DenseGraphEntries<'a, G: GraphDraft + 'a> {
    pub expand: G::Binding<'a, dense_expand::Entry>,
    pub output: G::Binding<'a, dense_output::Entry>,
}

impl<'a> From<&'a DenseKernels> for DenseGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a DenseKernels) -> Self {
        Self {
            expand: &kernels.expand,
            output: &kernels.output,
        }
    }
}

pub(crate) struct CheckedDenseEntries {
    expand: [(&'static str, Element); 4],
    output: [(&'static str, Element); 2],
}

impl CheckedDenseEntries {
    pub(crate) fn new(binding: DenseBinding) -> Self {
        Self {
            expand: [
                ("NW", binding.norm),
                ("GW", binding.gate),
                ("UW", binding.up),
                ("A", binding.activation),
            ],
            output: [("DW", binding.down), ("A", binding.activation)],
        }
    }

    pub(crate) fn entries(&self) -> DenseGraphEntries<'_, NativeGraphMetadata> {
        DenseGraphEntries {
            expand: &self.expand,
            output: &self.output,
        }
    }
}

/// Dimensions come from the resident weight plan used for both routes.
pub(crate) fn dimensions(
    load: &ModelLoadPlan,
    scope: WeightScope,
    rows: u64,
) -> Result<[(&'static str, u64); 4], String> {
    let role = WeightRole {
        scope,
        kind: WeightKind::DenseGate,
    };
    let plan = load
        .weights()
        .find(|weight| weight.role == role)
        .ok_or_else(|| format!("missing planned weight {role:?}"))?;
    let [features, hidden] = plan.shape.as_slice() else {
        return Err("dense gate weight is not rank two".into());
    };
    Ok([("M", rows), ("O", rows), ("H", *hidden), ("F", *features)])
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dense<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    kernels: DenseGraphEntries<'a, G>,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    residual: &WorkflowTensor,
    rows: u64,
    epsilon: f32,
) -> Result<WorkflowTensor, String> {
    let dimensions = dimensions(load, scope, rows)?;
    let norm = weight(graph, load, scope, WeightKind::FeedForwardNorm, weights)?;
    let gate = weight(graph, load, scope, WeightKind::DenseGate, weights)?;
    let up = weight(graph, load, scope, WeightKind::DenseUp, weights)?;
    let down = weight(graph, load, scope, WeightKind::DenseDown, weights)?;
    let out_rows = GraphConstant::identity(graph, rows)?;
    let product = graph
        .enqueue(
            kernels.expand,
            &dimensions,
            dense_expand::WorkflowArgs {
                residual: residual.into(),
                norm: (&norm).into(),
                gate_weight: (&gate).into(),
                up_weight: (&up).into(),
                out_rows: out_rows.port().tensor().into(),
                eps: epsilon,
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    let output = graph
        .enqueue(
            kernels.output,
            &dimensions,
            dense_output::WorkflowArgs {
                residual: residual.into(),
                product: (&product).into(),
                down_weight: (&down).into(),
                out_rows: out_rows.port().tensor().into(),
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    constants.push(out_rows);
    Ok(output)
}
