//! Dense feed-forward block graph: the RMS-normed paired gate/up projection
//! with SiLU·mul, then the down projection plus the residual. Both entries
//! gather their rows through `out_rows`; a block that advances every row binds
//! the identity table as a graph constant, so no run uploads it.

use super::super::native_constants::GraphConstant;
use super::super::native_target_graph::weight;
use crate::{ModelLoadPlan, native::DenseKernels};
use magnitude_model_contracts::{WeightKind, WeightRole, WeightScope};
use magnitude_model_kernels::{qwen_dense_expand, qwen_dense_output};
use seismic::{NativeGraph, NativePort, WorkflowTensor};

#[allow(clippy::too_many_arguments)]
pub(crate) fn dense(
    graph: &mut NativeGraph,
    kernels: &DenseKernels,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    residual: &WorkflowTensor,
    rows: u64,
    epsilon: f32,
) -> Result<WorkflowTensor, String> {
    let norm = weight(graph, load, scope, WeightKind::FeedForwardNorm, weights)?;
    let gate = weight(graph, load, scope, WeightKind::DenseGate, weights)?;
    let up = weight(graph, load, scope, WeightKind::DenseUp, weights)?;
    let down = weight(graph, load, scope, WeightKind::DenseDown, weights)?;
    let out_rows = GraphConstant::identity(graph, rows)?;
    let product = graph
        .enqueue(
            &kernels.expand,
            qwen_dense_expand::WorkflowArgs {
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
            &kernels.output,
            qwen_dense_output::WorkflowArgs {
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
