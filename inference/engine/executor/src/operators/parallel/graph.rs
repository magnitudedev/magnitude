//! A dense branch beside a general routed branch (Gemma 4 26B,
//! [`DenseBesideRouted`]). Both branches read the same residual
//! through their own input norms: the dense branch's gated expansion is
//! projected down into F32 rows, and the routed branch sums its experts onto
//! zeros. `moe_tail` normalizes each branch with its output norm, normalizes
//! their sum with the sublayer's post-norm, and adds it to the residual
//! (times the layer scale of a scaled tail).

use super::DenseBesideRouted;
use crate::native::ParallelKernels;
use crate::operators::dense_ffn::graph::dimensions as dense_dimensions;
use crate::operators::routed::graph::{
    CheckedGeneralRoutedEntries, GeneralRoutedGraphEntries, RoutedSum, general_routed,
};
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{WeightPort, packed_weight, weight};
use crate::{GeneralRoutedShape, ModelLoadPlan, ParallelBinding};
use magnitude_family_contracts::{SublayerIndex, WeightKind, WeightScope};
use magnitude_kernels::{dense_expand, moe_tail, project_rows};
use seismic::{Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

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

/// The parallel sublayer `sublayer` over `residual` (`[rows, D]` F32);
/// returns the new residual. `activation` is the dense branch's activation
/// code, `scale` the tail's layer scale.
#[allow(clippy::too_many_arguments)]
pub(crate) fn parallel<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: ParallelGraphEntries<'a, G>,
    load: &ModelLoadPlan,
    sublayer: SublayerIndex,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    residual: &WorkflowTensor,
    rows: u64,
    routed_shape: &GeneralRoutedShape,
    epsilon: f32,
    activation: i32,
    scale: f32,
) -> Result<WorkflowTensor, GraphError> {
    let [dense_scope, routed_scope] = DenseBesideRouted::scopes(sublayer);
    let dimensions = dense_dimensions(load, dense_scope, rows)?;
    let [_, _, (_, hidden), (_, features)] = dimensions;
    let norm = weight(graph, load, dense_scope, WeightKind::InputNorm, weights)?;
    let dense_norm = weight(graph, load, dense_scope, WeightKind::PostNorm, weights)?;
    let routed_norm = weight(graph, load, routed_scope, WeightKind::PostNorm, weights)?;
    let tail_norm = weight(
        graph,
        load,
        WeightScope::TargetSublayer(sublayer),
        WeightKind::PostNorm,
        weights,
    )?;
    let out_rows = GraphConstant::identity_for_class(graph, rows, Some("M"))?;
    let mut packed =
        |graph: &mut G, kind| packed_weight(graph, load, dense_scope, kind, weights, constants);
    let gate = packed(graph, WeightKind::DenseGate)?;
    let up = packed(graph, WeightKind::DenseUp)?;
    let down = packed(graph, WeightKind::DenseDown)?;
    let product = graph
        .enqueue(
            entries.expand,
            &[
                dimensions.as_slice(),
                &[
                    ("GS", gate.operands.scale.extent),
                    ("US", up.operands.scale.extent),
                    ("GB", gate.operands.bound.extent),
                    ("UB", up.operands.bound.extent),
                ],
            ]
            .concat(),
            dense_expand::WorkflowArgs {
                residual: residual.into(),
                norm: (&norm).into(),
                gate_weight: (&gate.weight).into(),
                up_weight: (&up.weight).into(),
                out_rows: out_rows.port().tensor().into(),
                eps: epsilon,
                activation,
                gate_scale: (&gate.operands.scale.tensor).into(),
                up_scale: (&up.operands.scale.tensor).into(),
                gate_bound: (&gate.operands.bound.tensor).into(),
                up_bound: (&up.operands.bound.tensor).into(),
            },
        )?
        .value;
    let dense = graph
        .enqueue(
            entries.down,
            &[
                ("M", rows),
                ("K", features),
                ("N", hidden),
                ("WS", down.operands.scale.extent),
                ("WB", down.operands.bound.extent),
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
    let routed = general_routed(
        graph,
        entries.routed,
        load,
        routed_scope,
        weights,
        constants,
        residual,
        RoutedSum::Branch,
        rows,
        routed_shape,
        epsilon,
    )?;
    Ok(graph
        .enqueue(
            entries.tail,
            &[("M", rows), ("D", hidden)],
            moe_tail::WorkflowArgs {
                residual: residual.into(),
                dense: (&dense).into(),
                routed: (&routed).into(),
                dense_norm: (&dense_norm).into(),
                routed_norm: (&routed_norm).into(),
                norm: (&tail_norm).into(),
                epsilon,
                scale,
            },        )?
        .value)
}
