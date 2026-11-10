//! A dense branch beside a general routed branch (Gemma 4 26B,
//! [`DenseBesideRouted`]). Both branches read the same residual
//! through their own input norms: the dense branch's gated expansion is
//! projected down into F32 rows, and the routed branch sums its experts onto
//! zeros. `moe_tail` normalizes each branch with its output norm, normalizes
//! their sum with the sublayer's post-norm, and adds it to the residual
//! (times the layer scale of a scaled tail).

use super::DenseBesideRouted;
use crate::native::ParallelKernels;
use crate::operators::dense_ffn::graph::DenseNumbers;
use crate::operators::routed::graph::{
    CheckedGeneralRoutedEntries, GeneralRoutedGraphEntries, GeneralRoutedLayer, GeneralRoutedTopology, RoutedSum,
    general_routed,
};
use crate::operators::routed::{ExpansionForm, split::Split, streamed::LayerBank};
use crate::operators::scalar;
use crate::programs::graph::facts::Numbers;
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{PlannedWeights, WeightPort, packed_weight, weight};
use crate::{GeneralRoutedShape, ModelLoadPlan, ParallelBinding};
use magnitude_family_contracts::{SublayerIndex, WeightKind, WeightScope};
use magnitude_kernels::{dense_expand, moe_tail, project_rows};
use seismic::{Argument, Dim, Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

/// What decides a parallel sublayer's graph beside its numbers: its dense
/// branch's expansion and its routed branch's topology.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ParallelTopology<R = seismic::Argument<f64>> {
    pub dense: ExpansionForm,
    pub routed: GeneralRoutedTopology<R>,
    /// The tail's layer scale.
    pub scale: R,
}

impl<R> Numbers<R> for ParallelTopology<R> {
    type Of<E> = ParallelTopology<E>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> ParallelTopology<E> {
        ParallelTopology { dense: self.dense, routed: self.routed.map(&mut *f), scale: f(self.scale) }
    }
}

/// The numbers of a parallel sublayer: its dense branch's and its routed
/// branch's.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ParallelNumbers<D = Dim> {
    pub dense: DenseNumbers<D>,
    pub routed: GeneralRoutedLayer<D>,
}

impl<D> Numbers<D> for ParallelNumbers<D> {
    type Of<E> = ParallelNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> ParallelNumbers<E> {
        ParallelNumbers { dense: self.dense.map(&mut *f), routed: self.routed.map(f) }
    }
}

/// The topology and numbers of the parallel sublayer `sublayer` of
/// `branches` over a `hidden`-wide residual, whose tail's layer scale is
/// `scale` and whose routed branch reads its experts as `bank` does.
pub(crate) fn split<G: GraphDraft>(
    load: &ModelLoadPlan,
    sublayer: SublayerIndex,
    branches: DenseBesideRouted<'_>,
    hidden: u64,
    scale: f32,
    bank: &LayerBank<'_, '_, G>,
) -> Result<(ParallelTopology<f64>, ParallelNumbers<u64>), GraphError> {
    let [dense_scope, routed_scope] = DenseBesideRouted::scopes(sublayer);
    let (expansion, dense) = crate::operators::dense_ffn::graph::split(load, dense_scope, &branches.dense.up)?;
    let shape = GeneralRoutedShape::of(hidden, branches.routed).map_err(|error| error.to_string())?;
    let (routed_topology, routed) = crate::operators::routed::graph::split(load, routed_scope, &shape, bank)?;
    Ok((
        ParallelTopology { dense: expansion, routed: routed_topology, scale: f64::from(scale) },
        ParallelNumbers { dense, routed },
    ))
}

pub(crate) struct ParallelGraphEntries<'a, G: GraphDraft + 'a> {
    pub expand: G::Binding<'a, dense_expand::Entry>,
    pub down: G::Binding<'a, project_rows::Entry>,
    pub routed: GeneralRoutedGraphEntries<'a, G>,
    pub tail: G::Binding<'a, moe_tail::Entry>,
}

impl<'a> From<&'a ParallelKernels> for ParallelGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a ParallelKernels) -> Self {
        Self {
            expand: &kernels.expand,
            down: &kernels.down,
            routed: (&kernels.routed).into(),
            tail: &kernels.tail,
        }
    }
}

/// Entry element assignments of a parallel sublayer from its binding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedParallelEntries {
    expand: [(&'static str, Element); 4],
    down: [(&'static str, Element); 3],
    routed: CheckedGeneralRoutedEntries,
    tail: [(&'static str, Element); 1],
}

impl CheckedParallelEntries {
    pub(crate) fn new(binding: ParallelBinding) -> Self {
        let dense = binding.dense;
        Self {
            expand: [
                ("NW", dense.norm),
                ("GW", dense.gate),
                ("UW", dense.up),
                ("A", dense.activation),
            ],
            down: [
                ("A", dense.activation),
                ("W", dense.down),
                ("Y", Element::f32()),
            ],
            routed: CheckedGeneralRoutedEntries::new(binding.routed),
            tail: [("NW", binding.norm)],
        }
    }

    pub(crate) fn entries(&self) -> ParallelGraphEntries<'_, NativeGraphMetadata> {
        ParallelGraphEntries {
            expand: &self.expand,
            down: &self.down,
            routed: self.routed.entries(),
            tail: &self.tail,
        }
    }
}

/// The parallel sublayer `sublayer` over `residual` (`[rows, D]` F32): the
/// new residual, and the sequence input a run of a streamed routed branch
/// writes (its `bank`, `general_routed`). A split sublayer's front is its
/// routed branch's, and its back the rest.
#[allow(clippy::too_many_arguments)]
pub(crate) fn parallel<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: ParallelGraphEntries<'a, G>,
    bank: LayerBank<'_, 'a, G>,
    topology: &ParallelTopology,
    numbers: &ParallelNumbers,
    planned: &(impl PlannedWeights + ?Sized),
    sublayer: SublayerIndex,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    residual: &WorkflowTensor,
    rows: &Dim,
    epsilon: Argument<f32>,
    split: Option<Split<'_>>,
) -> Result<(WorkflowTensor, Option<NativePort>), GraphError> {
    let [dense_scope, routed_scope] = DenseBesideRouted::scopes(sublayer);
    let front = matches!(split, Some(Split::Front(_)));
    let (routed, sequence) = general_routed(
        graph,
        entries.routed,
        bank,
        &topology.routed,
        &numbers.routed,
        planned,
        routed_scope,
        weights,
        constants,
        residual,
        RoutedSum::Branch,
        rows,
        epsilon,
        split,
    )?;
    if front {
        return Ok((routed, sequence));
    }
    let (expansion, dense_numbers) = (topology.dense, &numbers.dense);
    let dimensions = dense_numbers.dimensions(rows);
    let (hidden, features) = (&dense_numbers.hidden, &dense_numbers.features);
    let norm = weight(graph, planned, dense_scope, WeightKind::InputNorm, weights)?;
    let dense_norm = weight(graph, planned, dense_scope, WeightKind::PostNorm, weights)?;
    let routed_norm = weight(graph, planned, routed_scope, WeightKind::PostNorm, weights)?;
    let tail_norm = weight(
        graph,
        planned,
        WeightScope::TargetSublayer(sublayer),
        WeightKind::PostNorm,
        weights,
    )?;
    let out_rows = GraphConstant::row_identity(graph, rows, Some("M"))?;
    let mut packed =
        |graph: &mut G, kind| packed_weight(graph, planned, dense_scope, kind, weights, constants);
    let gate = packed(graph, WeightKind::DenseGate)?;
    let up = packed(graph, WeightKind::DenseUp)?;
    let down = packed(graph, WeightKind::DenseDown)?;
    let product = graph
        .enqueue(
            entries.expand,
            &[
                dimensions.as_slice(),
                &[
                    ("GS", gate.operands.scale.extent.clone()),
                    ("US", up.operands.scale.extent.clone()),
                    ("GB", gate.operands.bound.extent.clone()),
                    ("UB", up.operands.bound.extent.clone()),
                    ("LB", Dim::from(magnitude_family_contracts::GluLimit::word(expansion.limit))),
                ],
            ]
            .concat(),
            dense_expand::WorkflowArgs {
                residual: residual.into(),
                norm: (&norm).into(),
                gate_weight: (&gate.weight).into(),
                up_weight: (&up.weight).into(),
                out_rows: out_rows.port().tensor().into(),
                eps: epsilon.into(),
                activation: scalar(&dense_numbers.activation, "a dense activation code")?,
                gate_scale: (&gate.operands.scale.tensor).into(),
                up_scale: (&up.operands.scale.tensor).into(),
                gate_bound: (&gate.operands.bound.tensor).into(),
                up_bound: (&up.operands.bound.tensor).into(),
                glu_limit: (&crate::operators::glu_limit(out_rows.port().tensor(), expansion.limit)).into(),
            },
        )?
        .value;
    let dense = graph
        .enqueue(
            entries.down,
            &[
                ("M", rows.clone()),
                ("G", Dim::from(1)),
                ("K", features.clone()),
                ("N", hidden.clone()),
                ("WS", down.operands.scale.extent.clone()),
                ("WB", down.operands.bound.extent.clone()),
            ],
            project_rows::WorkflowArgs {
                source: (&product).into(),
                weight: (&down.weight).into(),
                weight_scale: (&down.operands.scale.tensor).into(),
                weight_bound: (&down.operands.bound.tensor).into(),
            },
        )?
        .value;
    constants.push(out_rows);
    let output = graph
        .enqueue(
            entries.tail,
            &[("M", rows.clone()), ("D", hidden.clone())],
            moe_tail::WorkflowArgs {
                residual: residual.into(),
                dense: (&dense).into(),
                routed: (&routed).into(),
                dense_norm: (&dense_norm).into(),
                routed_norm: (&routed_norm).into(),
                norm: (&tail_norm).into(),
                epsilon,
                scale: topology.scale.f32(),
            },
        )?
        .value;
    Ok((output, sequence))
}
