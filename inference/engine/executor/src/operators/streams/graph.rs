//! The streams' graph fragments: the widening of the embedded rows into the
//! streams (block 0), and a mix site over the streams (each sublayer's, and
//! the decoder's exit).

use super::{StreamsBinding, StreamsGeometry};
use crate::ModelLoadPlan;
use crate::native::StreamsKernels;
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{WeightPort, weight};
use magnitude_family_contracts::{WeightKind, WeightScope};
use magnitude_kernels::{repeat_rows, stream_mix_down, stream_mix_up};
use seismic::{Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

/// The kernel entries of a decoder's streams: the sublayers' mix sites
/// (`down`, with inject), the exit site (`exit`, without), their second stage
/// (`up`), and the widening (`repeat`).
pub(crate) struct StreamsEntries<'a, G: GraphDraft + 'a> {
    pub down: G::Binding<'a, stream_mix_down::Entry>,
    pub exit: G::Binding<'a, stream_mix_down::Entry>,
    pub up: G::Binding<'a, stream_mix_up::Entry>,
    pub repeat: G::Binding<'a, repeat_rows::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for StreamsEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for StreamsEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a> From<&'a StreamsKernels> for StreamsEntries<'a, NativeGraph> {
    fn from(kernels: &'a StreamsKernels) -> Self {
        Self {
            down: &kernels.down,
            exit: &kernels.exit,
            up: &kernels.up,
            repeat: &kernels.repeat,
        }
    }
}

/// The element assignments of a decoder's streams entries from its binding.
/// The exit site binds its absent inject weight at the down weight's element
/// (a zero-row view of it).
pub(crate) struct CheckedStreamsEntries {
    down: [(&'static str, Element); 4],
    exit: [(&'static str, Element); 4],
    up: [(&'static str, Element); 3],
    repeat: [(&'static str, Element); 0],
}

impl CheckedStreamsEntries {
    pub(crate) fn new(binding: StreamsBinding) -> Self {
        Self {
            down: down_elements(binding, binding.inject),
            exit: down_elements(binding, binding.down),
            up: [
                ("UW", binding.up),
                ("NW", binding.norm),
                ("A", binding.activation),
            ],
            repeat: [],
        }
    }

    pub(crate) fn entries(&self) -> StreamsEntries<'_, NativeGraphMetadata> {
        StreamsEntries {
            down: &self.down,
            exit: &self.exit,
            up: &self.up,
            repeat: &self.repeat,
        }
    }
}

/// `stream_mix_down`'s element assignments with inject weights of `inject`.
pub(crate) fn down_elements(
    binding: StreamsBinding,
    inject: Element,
) -> [(&'static str, Element); 4] {
    [
        ("NW", binding.norm),
        ("DW", binding.down),
        ("IW", inject),
        ("A", binding.activation),
    ]
}

/// The streams of `rows` embedded rows `[rows, H]`: S copies of each.
pub(crate) fn widen<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entry: G::Binding<'a, repeat_rows::Entry>,
    geometry: StreamsGeometry,
    embedded: &WorkflowTensor,
    rows: u64,
    constants: &mut Vec<GraphConstant>,
) -> Result<WorkflowTensor, GraphError> {
    let copies = GraphConstant::f32(graph, &vec![1.0; geometry.streams as usize])?;
    let scales = copies.port().tensor().clone();
    constants.push(copies);
    Ok(graph
        .enqueue(
            entry,
            &[("M", rows), ("H", geometry.hidden), ("S", geometry.streams)],
            repeat_rows::WorkflowArgs {
                rows: embedded.into(),
                scales: (&scales).into(),
            },
        )?
        .value)
}

/// A mix site's results: the per-stream partials its consumer sums, and the
/// inject values its sublayer's output scatters with (none at the exit).
pub(crate) struct MixSite {
    pub partials: WorkflowTensor,
    pub inject: Option<WorkflowTensor>,
}

/// The mix site of `scope` over the `[rows, S * H]` streams `residual`: its
/// norm, down and up weights (and, `injected`, its inject weights) are the
/// scope's hyper-connection roles.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mix_site<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: StreamsEntries<'a, G>,
    load: &ModelLoadPlan,
    scope: WeightScope,
    geometry: StreamsGeometry,
    residual: &WorkflowTensor,
    rows: u64,
    epsilon: f32,
    injected: bool,
    weights: &mut Vec<(WeightPort, NativePort)>,
) -> Result<MixSite, GraphError> {
    let (h, s, r) = (geometry.hidden, geometry.streams, geometry.rank);
    // The mix entries read the streams and their norm as `[.., S, H]`.
    let streams = residual.reshape(&[rows, s, h]);
    let norm =
        weight(graph, load, scope, WeightKind::HyperConnectionNorm, weights)?.reshape(&[s, h]);
    let down = weight(graph, load, scope, WeightKind::HyperConnectionDown, weights)?;
    let up = weight(graph, load, scope, WeightKind::HyperConnectionUp, weights)?;
    let inject_weight = if injected {
        weight(
            graph,
            load,
            scope,
            WeightKind::HyperConnectionInject,
            weights,
        )?
        .reshape(&[s, s * h])
    } else {
        down.slice_leading(0, 0)
    };
    let mixed = graph.enqueue(
        if injected { entries.down } else { entries.exit },
        &[
            ("M", rows),
            ("H", h),
            ("S", s),
            ("LR", r),
            ("C", u64::from(injected)),
        ],
        stream_mix_down::WorkflowArgs {
            hidden: (&streams).into(),
            norm: (&norm).into(),
            down_weight: (&down).into(),
            inject_weight: (&inject_weight).into(),
            epsilon,
        },
    )?;
    let partials = graph
        .enqueue(
            entries.up,
            &[("M", rows), ("H", h), ("S", s), ("LR", r)],
            stream_mix_up::WorkflowArgs {
                low: (&mixed.r0).into(),
                up_weight: (&up).into(),
                hidden: (&streams).into(),
                inverse: (&mixed.r2).into(),
                norm: (&norm).into(),
            },
        )?
        .value;
    Ok(MixSite {
        partials,
        inject: injected.then_some(mixed.r1),
    })
}
