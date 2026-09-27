//! Dense feed-forward block graph: the RMS-normed paired gate/up projection
//! with SiLU·mul, then the down projection plus the residual (or, for a
//! post-norm sublayer, the down projection into F32 rows normalized into the
//! residual). Every entry gathers its rows through `out_rows`; a block that
//! advances every row binds the identity table as a graph constant, so no run
//! uploads it.

use crate::operators::output::{post_norm, CheckedPostNormEntries, PostNormShape, TailEntries};
use crate::programs::graph::draft::GraphDraft;
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::weight;
use crate::{native::DenseKernels, DenseBinding, ModelLoadPlan, SublayerTail};
use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};
use magnitude_kernels::{dense_expand, dense_output};
use seismic::{Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

/// The same dense topology accepts prepared entries or checked metadata
/// bindings. The latter carries only element assignments, never shapes.
pub(crate) struct DenseGraphEntries<'a, G: GraphDraft + 'a> {
    pub expand: G::Binding<'a, dense_expand::Entry>,
    pub output: TailEntries<'a, G, dense_output::Entry>,
}

impl<'a> From<&'a DenseKernels> for DenseGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a DenseKernels) -> Self {
        Self {
            expand: &kernels.expand,
            output: (&kernels.output).into(),
        }
    }
}

pub(crate) struct CheckedDenseEntries {
    expand: [(&'static str, Element); 4],
    output: [(&'static str, Element); 2],
    post_norm: Option<CheckedPostNormEntries>,
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
            post_norm: match binding.tail {
                SublayerTail::Residual => None,
                SublayerTail::PostNorm { norm, .. } => Some(CheckedPostNormEntries::new(
                    binding.down,
                    binding.activation,
                    norm,
                )),
            },
        }
    }

    pub(crate) fn entries(&self) -> DenseGraphEntries<'_, NativeGraphMetadata> {
        DenseGraphEntries {
            expand: &self.expand,
            output: match &self.post_norm {
                None => TailEntries::Residual(&self.output[..]),
                Some(post_norm) => TailEntries::PostNorm(post_norm.entries()),
            },
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

/// `post_norm_epsilon` is the sublayer's post-norm epsilon, read only by a
/// post-norm tail.
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
    activation: i32,
    post_norm_epsilon: f32,
    post_norm_scale: f32,
) -> Result<WorkflowTensor, String> {
    let dimensions = dimensions(load, scope, rows)?;
    let norm = weight(graph, load, scope, WeightKind::InputNorm, weights)?;
    let gate = weight(graph, load, scope, WeightKind::DenseGate, weights)?;
    let up = weight(graph, load, scope, WeightKind::DenseUp, weights)?;
    let down = weight(graph, load, scope, WeightKind::DenseDown, weights)?;
    let out_rows = GraphConstant::identity_for_class(graph, rows, Some("M"))?;
    let absent_scale = GraphConstant::absent_scale(graph, constants)?;
    let expand_dimensions = [dimensions.as_slice(), &[("GS", 0), ("US", 0)]].concat();
    let output_dimensions = [dimensions.as_slice(), &[("DS", 0)]].concat();
    let product = graph
        .enqueue(
            kernels.expand,
            &expand_dimensions,
            dense_expand::WorkflowArgs {
                residual: residual.into(),
                norm: (&norm).into(),
                gate_weight: (&gate).into(),
                up_weight: (&up).into(),
                out_rows: out_rows.port().tensor().into(),
                eps: epsilon,
                activation,
                gate_scale: (&absent_scale).into(),
                up_scale: (&absent_scale).into(),
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    let output = match kernels.output {
        TailEntries::Residual(output) => graph
            .enqueue(
                output,
                &output_dimensions,
                dense_output::WorkflowArgs {
                    residual: residual.into(),
                    product: (&product).into(),
                    down_weight: (&down).into(),
                    out_rows: out_rows.port().tensor().into(),
                    down_scale: (&absent_scale).into(),
                },
            )
            .map_err(|error| error.to_string())?
            .value,
        TailEntries::PostNorm(entries) => {
            let post_norm_weight = weight(graph, load, scope, WeightKind::PostNorm, weights)?;
            let [_, _, (_, hidden), (_, features)] = dimensions;
            post_norm(
                graph,
                entries,
                residual,
                (&product).into(),
                &down,
                &absent_scale,
                &post_norm_weight,
                out_rows.port().tensor(),
                PostNormShape {
                    rows,
                    out: rows,
                    inputs: features,
                    outputs: hidden,
                },
                post_norm_epsilon,
                post_norm_scale,
            )?
        }
    };
    constants.push(out_rows);
    Ok(output)
}
