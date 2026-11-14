//! The streams' graph fragments: the widening of the embedded rows into the
//! streams (block 0), a mix site over the streams (each sublayer's, and a
//! low-rank decoder's exit), and under Sinkhorn mixing the exit's collapse
//! (by the last site's pre, or into the streams' mean) and the join of a
//! sublayer whose output joins nothing itself.

use super::{
    BlockStreams, LowRankGeometry, MixDownBinding, MixUpBinding, ResidualCarry, SinkhornExit,
    SinkhornTopology, Sites, StreamsGeometry, StreamsMixing, carried,
};
use crate::native::{MixingKernels, StreamsKernels};
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{PlannedWeights, WeightPort, weight};
use magnitude_family_contracts::{SinkhornInput, WeightKind, WeightScope};
use magnitude_kernels::{
    repeat_rows, stream_collapse, stream_join, stream_mix_down, stream_mix_sinkhorn, stream_mix_up,
};
use seismic::{Argument, Dim, Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

/// The kernel entries of one mix site: `stream_mix_down` (with inject
/// values at a sublayer's site, without at the exit) and `stream_mix_up`.
pub(crate) struct SiteEntries<'a, G: GraphDraft + 'a> {
    pub down: G::Binding<'a, stream_mix_down::Entry>,
    pub up: G::Binding<'a, stream_mix_up::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for SiteEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for SiteEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// The kernel entries of a block's Sinkhorn streams: every site's
/// (`stream_mix_sinkhorn`), the exit's collapse and the join of an output
/// that joins nothing itself.
pub(crate) struct SinkhornEntries<'a, G: GraphDraft + 'a> {
    pub site: G::Binding<'a, stream_mix_sinkhorn::Entry>,
    pub collapse: G::Binding<'a, stream_collapse::Entry>,
    pub join: G::Binding<'a, stream_join::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for SinkhornEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for SinkhornEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// The kernel entries of a block's mix sites, by its streams' mixing.
pub(crate) enum MixingEntries<'a, G: GraphDraft + 'a> {
    LowRank(Sites<SiteEntries<'a, G>>),
    Sinkhorn(SinkhornEntries<'a, G>),
}

impl<'a, G: GraphDraft + 'a> Copy for MixingEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for MixingEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// The entries of the mix site a sublayer reads through.
pub(crate) enum SiteEntry<'a, G: GraphDraft + 'a> {
    LowRank(SiteEntries<'a, G>),
    Sinkhorn(G::Binding<'a, stream_mix_sinkhorn::Entry>),
}

impl<'a, G: GraphDraft + 'a> Copy for SiteEntry<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for SiteEntry<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// The kernel entries of a block's streams: its mix sites', and the
/// widening (`repeat`).
pub(crate) struct StreamsEntries<'a, G: GraphDraft + 'a> {
    pub mixing: MixingEntries<'a, G>,
    pub repeat: G::Binding<'a, repeat_rows::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for StreamsEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for StreamsEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, G: GraphDraft + 'a> StreamsEntries<'a, G> {
    /// The entries of the site sublayer `sublayer` of the block reads
    /// through: the mixer's (0) or the feed-forward's (1). Every Sinkhorn
    /// site is the one entry.
    pub(crate) fn site(&self, sublayer: u32) -> Option<SiteEntry<'a, G>> {
        match self.mixing {
            MixingEntries::LowRank(sites) => match sublayer {
                0 => Some(SiteEntry::LowRank(sites.mixer)),
                1 => sites.feed_forward.map(SiteEntry::LowRank),
                _ => None,
            },
            MixingEntries::Sinkhorn(entries) => Some(SiteEntry::Sinkhorn(entries.site)),
        }
    }

    /// The low-rank sites' entries, of low-rank mixing.
    pub(crate) fn low_rank(&self) -> Option<Sites<SiteEntries<'a, G>>> {
        match self.mixing {
            MixingEntries::LowRank(sites) => Some(sites),
            MixingEntries::Sinkhorn(_) => None,
        }
    }

    /// The Sinkhorn streams' entries, of Sinkhorn mixing.
    pub(crate) fn sinkhorn(&self) -> Option<SinkhornEntries<'a, G>> {
        match self.mixing {
            MixingEntries::Sinkhorn(entries) => Some(entries),
            MixingEntries::LowRank(_) => None,
        }
    }
}

impl<'a> From<&'a StreamsKernels> for StreamsEntries<'a, NativeGraph> {
    fn from(kernels: &'a StreamsKernels) -> Self {
        Self {
            mixing: match &kernels.mixing {
                MixingKernels::LowRank(sites) => MixingEntries::LowRank(sites.as_ref().map(|site| SiteEntries {
                    down: &site.down,
                    up: &site.up,
                })),
                MixingKernels::Sinkhorn(sinkhorn) => MixingEntries::Sinkhorn(SinkhornEntries {
                    site: &sinkhorn.site,
                    collapse: &sinkhorn.collapse,
                    join: &sinkhorn.join,
                }),
            },
            repeat: &kernels.repeat,
        }
    }
}

/// The element assignments of a mix site's entries.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedSiteEntries {
    down: [(&'static str, Element); 4],
    up: [(&'static str, Element); 3],
}

/// The element assignments of a block's mix sites' entries: the low-rank
/// sites', or the Sinkhorn entries', which bind none.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CheckedMixing {
    LowRank(Sites<CheckedSiteEntries>),
    Sinkhorn([(&'static str, Element); 0]),
}

/// The element assignments of a block's streams entries from its binding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedStreamsEntries {
    mixing: CheckedMixing,
    repeat: [(&'static str, Element); 0],
}

impl CheckedStreamsEntries {
    pub(crate) fn new(binding: BlockStreams) -> Self {
        Self {
            mixing: match binding {
                BlockStreams::LowRank(binding) => CheckedMixing::LowRank(binding.entries().map(|site| {
                    CheckedSiteEntries {
                        down: down_elements(site.down),
                        up: up_elements(site.up),
                    }
                })),
                BlockStreams::Sinkhorn(_) => CheckedMixing::Sinkhorn([]),
            },
            repeat: [],
        }
    }

    pub(crate) fn entries(&self) -> StreamsEntries<'_, NativeGraphMetadata> {
        StreamsEntries {
            mixing: match &self.mixing {
                CheckedMixing::LowRank(sites) => MixingEntries::LowRank(sites.as_ref().map(|site| SiteEntries {
                    down: &site.down[..],
                    up: &site.up[..],
                })),
                CheckedMixing::Sinkhorn(none) => MixingEntries::Sinkhorn(SinkhornEntries {
                    site: &none[..],
                    collapse: &none[..],
                    join: &none[..],
                }),
            },
            repeat: &self.repeat,
        }
    }
}

/// `stream_mix_down`'s element assignments.
pub(crate) fn down_elements(binding: MixDownBinding) -> [(&'static str, Element); 4] {
    [
        ("NW", binding.norm),
        ("DW", binding.down),
        ("IW", binding.inject_element()),
        ("A", binding.activation),
    ]
}

/// `stream_mix_up`'s element assignments.
pub(crate) fn up_elements(binding: MixUpBinding) -> [(&'static str, Element); 3] {
    [
        ("UW", binding.up),
        ("NW", binding.norm),
        ("A", binding.activation),
    ]
}

/// The streams of `rows` embedded rows `[rows, H]`: S copies of each.
pub(crate) fn widen<'a, G: GraphDraft + 'a, D: Clone>(
    graph: &mut G,
    entry: G::Binding<'a, repeat_rows::Entry>,
    geometry: StreamsGeometry<D, D>,
    embedded: &WorkflowTensor,
    rows: impl Into<Dim>,
    constants: &mut Vec<GraphConstant>,
) -> Result<ResidualCarry<WorkflowTensor>, GraphError>
where
    D: Into<Dim>,
{
    let (geometry, rows) = (geometry.dims(), rows.into());
    let copies = GraphConstant::f32(graph, &vec![1.0; geometry.streams as usize])?;
    let scales = copies.port().tensor().clone();
    constants.push(copies);
    let streams = graph
        .enqueue(
            entry,
            &[("M", rows.clone()), ("H", geometry.hidden.clone()), ("S", Dim::from(geometry.streams))],
            repeat_rows::WorkflowArgs {
                rows: embedded.into(),
                scales: (&scales).into(),
            },
        )?
        .value;
    // Under Sinkhorn mixing by the previous site's pre the first site reads
    // the one-hot pre `[1, 0, .., 0]` (`SinkhornInput::PreviousSite`).
    let pre = geometry
        .carried_pre()
        .map(|streams| {
            let one_hot = GraphConstant::row_one_hot(graph, rows.clone(), streams, Some("M"))?;
            let pre = one_hot.port().tensor().clone();
            constants.push(one_hot);
            Ok::<_, GraphError>(pre)
        })
        .transpose()?;
    Ok(ResidualCarry { rows: streams, pre })
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
pub(crate) fn mix_site<'a, G: GraphDraft + 'a, D: Clone>(
    graph: &mut G,
    entries: SiteEntries<'a, G>,
    load: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    geometry: LowRankGeometry<D>,
    residual: &WorkflowTensor,
    rows: impl Into<Dim>,
    epsilon: Argument<f32>,
    injected: bool,
    weights: &mut Vec<(WeightPort, NativePort)>,
) -> Result<MixSite, GraphError>
where
    D: Into<Dim>,
{
    let geometry = geometry.dims();
    let rows = rows.into();
    let (h, s, r) = (&geometry.hidden, Dim::from(geometry.streams), &geometry.rank);
    // The mix entries read the streams and their norm as `[.., S, H]`.
    let streams = residual.reshape(&[&rows, &s, h]);
    let norm =
        weight(graph, load, scope, WeightKind::HyperConnectionNorm, weights)?.reshape(&[&s, h]);
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
        .reshape(&[s.clone(), &s * h])
    } else {
        down.slice_leading(0, 0)
    };
    let mixed = graph.enqueue(
        entries.down,
        &[
            ("M", &rows),
            ("H", h),
            ("S", &s),
            ("LR", r),
            ("C", &Dim::from(u64::from(injected))),
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
            &[("M", &rows), ("H", h), ("S", &s), ("LR", r)],
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

/// A Sinkhorn site's results: its sublayer's input `[rows, H]` (which the
/// sublayer's input norm normalizes), its `pre` `[rows, S]` (the next
/// site's previous, when sites mix by the previous site's) and its join row
/// post | comb `[rows, S + S * S]`.
pub(crate) struct SinkhornSite {
    pub input: WorkflowTensor,
    pub pre: Option<WorkflowTensor>,
    pub join: WorkflowTensor,
}

/// The Sinkhorn site of `scope` over `carry`'s `[rows, S * H]` streams,
/// mixing by the previous site's pre `carry` holds or by its own (the
/// streams' `SinkhornInput`): its function, base and scale weights are the
/// scope's hyper-connection roles.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sinkhorn_site<'a, G: GraphDraft + 'a, D: Clone>(
    graph: &mut G,
    entry: G::Binding<'a, stream_mix_sinkhorn::Entry>,
    load: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    geometry: StreamsGeometry<D, D>,
    mixing: SinkhornTopology,
    carry: &ResidualCarry<WorkflowTensor>,
    rows: impl Into<Dim>,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<SinkhornSite, GraphError>
where
    D: Into<Dim>,
{
    let geometry = geometry.dims();
    let rows = rows.into();
    let (h, s) = (&geometry.hidden, Dim::from(geometry.streams));
    let StreamsMixing::Sinkhorn { input, numerics: iterations } = &geometry.mixing else {
        return Err("a Sinkhorn site over low-rank streams".into());
    };
    let (input, iterations) = (*input, iterations.word::<u32>().ok_or("Sinkhorn iterations exceed u32")?);
    // A site mixing by its own pre reads none: `[rows, 0]`.
    let previous = match (input, &carry.pre) {
        (SinkhornInput::PreviousSite, Some(pre)) => pre.clone(),
        (SinkhornInput::OwnSite, None) => GraphConstant::absent_rows(graph, rows.clone(), Some("M"), constants)?,
        _ => return Err("a Sinkhorn site's carried pre disagrees with its mixing".into()),
    };
    let streams = carry.rows.reshape(&[&rows, &s, h]);
    let mut weight = |kind| weight(graph, load, scope, kind, weights);
    let (function, base, scale) = (
        weight(WeightKind::HyperConnectionFunction)?,
        weight(WeightKind::HyperConnectionBase)?,
        weight(WeightKind::HyperConnectionScale)?,
    );
    let site = graph.enqueue(
        entry,
        &[("M", &rows), ("H", h), ("S", &s), ("P", &Dim::from(carried(input)))],
        stream_mix_sinkhorn::WorkflowArgs {
            hidden: (&streams).into(),
            previous: (&previous).into(),
            function: (&function).into(),
            base: (&base).into(),
            scale: (&scale).into(),
            epsilon: mixing.epsilon.f32(),
            norm_epsilon: mixing.norm_epsilon.f32(),
            iterations,
        },
    )?;
    Ok(SinkhornSite {
        input: site.r0,
        pre: (input == SinkhornInput::PreviousSite).then_some(site.r1),
        join: site.r2,
    })
}

/// The decoder's exit over `carry`'s Sinkhorn streams: their collapse
/// `[rows, H]` as `exit` names it, which the readout normalizes.
pub(crate) fn collapse<'a, G: GraphDraft + 'a, D: Clone>(
    graph: &mut G,
    entry: G::Binding<'a, stream_collapse::Entry>,
    geometry: StreamsGeometry<D, D>,
    carry: &ResidualCarry<WorkflowTensor>,
    exit: SinkhornExit,
    rows: impl Into<Dim>,
    constants: &mut Vec<GraphConstant>,
) -> Result<WorkflowTensor, GraphError>
where
    D: Into<Dim>,
{
    let geometry = geometry.dims();
    let rows = rows.into();
    let (h, s) = (&geometry.hidden, Dim::from(geometry.streams));
    // The mean reads no pre: `[rows, 0]`.
    let pre = match exit {
        SinkhornExit::LastPre => carry.pre.clone().ok_or("the exit of Sinkhorn streams without the last site's pre")?,
        SinkhornExit::Mean => GraphConstant::absent_rows(graph, rows.clone(), Some("M"), constants)?,
    };
    Ok(graph
        .enqueue(
            entry,
            &[("M", &rows), ("H", h), ("S", &s), ("P", &Dim::from(exit.previous()))],
            stream_collapse::WorkflowArgs {
                hidden: (&carry.rows.reshape(&[&rows, &s, h])).into(),
                pre: (&pre).into(),
            },
        )?
        .value)
}

/// The F32 output `y` `[rows, H]` of a sublayer whose output entry joins
/// nothing itself, joined into the `[rows, S * H]` streams `residual` with
/// its site's join values `values` (`stream_join`).
pub(crate) fn join<'a, G: GraphDraft + 'a, D: Clone>(
    graph: &mut G,
    entry: G::Binding<'a, stream_join::Entry>,
    geometry: StreamsGeometry<D, D>,
    residual: &WorkflowTensor,
    y: &WorkflowTensor,
    values: &WorkflowTensor,
    rows: impl Into<Dim>,
) -> Result<WorkflowTensor, GraphError>
where
    D: Into<Dim>,
{
    let geometry = geometry.dims();
    Ok(graph
        .enqueue(
            entry,
            &[
                ("M", rows.into()),
                ("H", geometry.hidden.clone()),
                ("SN", Dim::from(geometry.streams)),
                ("IN", Dim::from(geometry.join_width())),
            ],
            stream_join::WorkflowArgs {
                residual: residual.into(),
                y: y.into(),
                inject: values.into(),
            },
        )?
        .value)
}
