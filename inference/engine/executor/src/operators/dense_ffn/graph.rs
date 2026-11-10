//! Dense feed-forward block graph: the RMS-normed paired gate/up projection
//! with SiLU·mul, then the down projection plus the residual (or, for a
//! post-norm sublayer, the down projection into F32 rows normalized into the
//! residual). Every entry gathers its rows through `out_rows`; a block that
//! advances every row binds the identity table as a graph constant, so no run
//! uploads it.

use crate::operators::output::{CheckedPostNormEntries, PostNorm, PostNormShape, TailEntries, post_norm};
use crate::operators::routed::{Expansion, ExpansionForm};
use crate::operators::scalar;
use crate::programs::graph::facts::Numbers;
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{PlannedWeights, WeightPort, packed_weight, weight};
use crate::{DenseBinding, ModelLoadPlan, SublayerTail, native::DenseKernels};
use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};
use magnitude_kernels::{dense_expand, dense_output};
use seismic::{Argument, Dim, Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

/// The numbers of a dense feed-forward: its residual's width, its features
/// and its expansion's activation code (`dense_expand`'s `activation`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DenseNumbers<D = Dim> {
    pub hidden: D,
    pub features: D,
    pub activation: D,
}

impl<D> Numbers<D> for DenseNumbers<D> {
    type Of<E> = DenseNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> DenseNumbers<E> {
        DenseNumbers { hidden: f(self.hidden), features: f(self.features), activation: f(self.activation) }
    }
}

impl DenseNumbers {
    /// The expansion and output entries' dimensions over `rows` rows.
    pub(crate) fn dimensions(&self, rows: &Dim) -> [(&'static str, Dim); 4] {
        [("M", rows.clone()), ("O", rows.clone()), ("H", self.hidden.clone()), ("F", self.features.clone())]
    }
}

/// The topology (its expansion, `up`'s) and numbers of the dense
/// feed-forward of `scope`, the numbers from its resident gate weight's
/// `[features, hidden]` plan (both routes' plan): the only reading of the
/// model they make.
pub(crate) fn split(
    load: &ModelLoadPlan,
    scope: WeightScope,
    up: &magnitude_family_contracts::FeedForwardUp,
) -> Result<(ExpansionForm, DenseNumbers<u64>), GraphError> {
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
    let expansion = Expansion::of(up);
    Ok((
        expansion.form(),
        DenseNumbers { hidden: *hidden, features: *features, activation: expansion.activation_number() },
    ))
}

/// `tail` is the sublayer's post-norm tail's reals, none for a
/// residual tail. Its row identity follows the class dimension `rows_class`
/// (`GraphConstant::row_identity`), or is fixed without one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    kernels: DenseGraphEntries<'a, G>,
    expansion: ExpansionForm,
    numbers: &DenseNumbers,
    planned: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    residual: &WorkflowTensor,
    rows: &Dim,
    rows_class: Option<&'static str>,
    epsilon: Argument<f32>,
    tail: Option<&PostNorm>,
) -> Result<WorkflowTensor, GraphError> {
    let dimensions = numbers.dimensions(rows);
    let norm = weight(graph, planned, scope, WeightKind::InputNorm, weights)?;
    let mut packed =
        |graph: &mut G, kind| packed_weight(graph, planned, scope, kind, weights, constants);
    let gate = packed(graph, WeightKind::DenseGate)?;
    let up = packed(graph, WeightKind::DenseUp)?;
    let down = packed(graph, WeightKind::DenseDown)?;
    let out_rows = GraphConstant::row_identity(graph, rows, rows_class)?;
    let expand_dimensions = [
        dimensions.as_slice(),
        &[
            ("GS", gate.operands.scale.extent.clone()),
            ("US", up.operands.scale.extent.clone()),
            ("GB", gate.operands.bound.extent.clone()),
            ("UB", up.operands.bound.extent.clone()),
            ("LB", Dim::from(magnitude_family_contracts::GluLimit::word(expansion.limit))),
        ],
    ]
    .concat();
    let output_dimensions = [
        dimensions.as_slice(),
        &[("DS", down.operands.scale.extent.clone()), ("DB", down.operands.bound.extent.clone())],
    ]
    .concat();
    let product = graph
        .enqueue(
            kernels.expand,
            &expand_dimensions,
            dense_expand::WorkflowArgs {
                residual: residual.into(),
                norm: (&norm).into(),
                gate_weight: (&gate.weight).into(),
                up_weight: (&up.weight).into(),
                out_rows: out_rows.port().tensor().into(),
                eps: epsilon.into(),
                activation: scalar(&numbers.activation, "a dense activation code")?,
                gate_scale: (&gate.operands.scale.tensor).into(),
                up_scale: (&up.operands.scale.tensor).into(),
                gate_bound: (&gate.operands.bound.tensor).into(),
                up_bound: (&up.operands.bound.tensor).into(),
                glu_limit: (&crate::operators::glu_limit(out_rows.port().tensor(), expansion.limit)).into(),
            },
        )?
        .value;
    let output = match (kernels.output, tail) {
        (TailEntries::Residual(output), None) => {
            graph
                .enqueue(
                    output,
                    &output_dimensions,
                    dense_output::WorkflowArgs {
                        residual: residual.into(),
                        product: (&product).into(),
                        down_weight: (&down.weight).into(),
                        out_rows: out_rows.port().tensor().into(),
                        down_scale: (&down.operands.scale.tensor).into(),
                        down_bound: (&down.operands.bound.tensor).into(),
                    },
                )?
                .value
        }
        (TailEntries::PostNorm(entries), Some(tail)) => {
            let post_norm_weight = weight(graph, planned, scope, WeightKind::PostNorm, weights)?;
            post_norm(
                graph,
                entries,
                residual,
                (&product).into(),
                &down,
                &post_norm_weight,
                out_rows.port().tensor(),
                PostNormShape {
                    rows: rows.clone(),
                    out: rows.clone(),
                    inputs: numbers.features.clone(),
                    outputs: numbers.hidden.clone(),
                },
                tail,
            )?
        }
        _ => return Err("dense output entries disagree with its post-norm tail".into()),
    };
    constants.push(out_rows);
    Ok(output)
}
