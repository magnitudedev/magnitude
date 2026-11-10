//! An attention block as its graph builders take it: its topology
//! ([`AttentionTopology`]: its structure, every real and every table) and
//! its numbers ([`AttentionBlock`]: every model number its graph is sized
//! by). [`split`] is the only reading of the model behind them.
//!
//! The topology is generic over what a real is (`R`: the model's `f64`, a
//! key's `RealIndex`, a builder's `Argument<f64>`), mapped as its
//! [`Numbers`], and over what a table is (`T`: its bytes, a key's
//! `TableIndex`, a builder's `Table`), mapped by
//! [`AttentionTopology::tables`].
//!
//! An optional part is an `Option` on each side: its reals in the topology,
//! its numbers in the block. [`split`] is their only constructor (the part
//! fields are private to the attention operator) and every map keeps an
//! `Option`'s shape, so the two agree; a builder takes them together
//! ([`both`]), and a disagreement is an invariant violation.

use super::graph::{rotary_amplitudes, rotary_components, rotary_frequencies};
use super::pooled::{PickedBlock, PickedPlane};
use crate::operators::attention as operator;
use crate::operators::output::PostNorm;
use crate::planning::{AttentionForm, AttentionWidths, RoundingWords};
use crate::programs::graph::facts::Numbers;
use crate::programs::graph::{GraphError, draft::ControlSource};
use crate::{AttentionShape, CandidatePool, CandidatesBinding};
use magnitude_family_contracts::{Attention, HeadNorm, HeadWeights, IndexerKeys, WeightKind};
use seismic::{Argument, Dim, Element, Table};
use std::sync::Arc;

/// One attention block at one graph class as its frame reads it from the
/// model: what [`split`] takes.
pub(crate) struct AttentionSite<'a> {
    pub rows: u64,
    pub segments: u64,
    pub history_rows: u64,
    pub slab_rows: u32,
    /// The history row tiles one launch of the class can see
    /// (`HistoryStorePlan::launch_tiles` of its request slots).
    pub history_tiles: u64,
    pub shape: AttentionShape,
    pub operator: &'a Attention,
    /// The input norm's epsilon.
    pub epsilon: f32,
    /// The head norms' one epsilon.
    pub head_epsilon: f32,
    /// The post-norm tail's reals; none for a residual tail.
    pub post_norm: Option<PostNorm<f64>>,
    pub activation: Element,
    /// Draft priming only publishes K/V history; its attention output is unused.
    pub inject_only: bool,
    /// Causal target rows consume earlier rows through the cache codec.
    pub causal_cache: bool,
    /// A selected read's indexer and its history domains.
    pub selection: Option<SelectionSite>,
    /// An index-list source's rows (`HistorySource` 1).
    pub picked: Option<PickedBlock<'a>>,
    /// How the graph takes its launch's controls.
    pub controls: ControlSource,
}

/// A selected read's indexer at one graph class: its static axes and its
/// raw keys' (Window) and block keys' (Block) history rows and slab rows.
#[derive(Clone, Copy)]
pub(crate) struct SelectionSite {
    pub shape: crate::SelectionShape,
    /// Whether the class scores: some row of its launch has more than
    /// `blocks` complete blocks. A class without scoring appends the
    /// indexer's keys alone and reads every visible row.
    pub scored: bool,
    pub window_rows: u64,
    pub window_slab_rows: u32,
    pub block_rows: u64,
    pub block_slab_rows: u32,
}

/// Everything structural an attention block's builders branch on but its
/// optional parts' own structure (which [`AttentionBlock`] holds with their
/// numbers), every real and every table its graph binds. It holds no model
/// number.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AttentionTopology<R = Argument<f64>, T = Table> {
    /// The flags and kinds of its shape.
    pub form: AttentionForm,
    /// The weight kinds of the query and of the projection's query segment
    /// (`operators::attention::query_kind`, `query_segment_kind`).
    pub query_kind: WeightKind,
    pub query_segment_kind: WeightKind,
    pub activation: Element,
    pub inject_only: bool,
    pub causal_cache: bool,
    pub controls: ControlSource,
    /// The input norm's epsilon.
    pub epsilon: R,
    /// The head norms' one epsilon.
    pub head_epsilon: R,
    /// The post-norm tail's reals; none for a residual tail.
    pub post_norm: Option<PostNorm<R>>,
    /// The attention's score scale.
    pub scale: R,
    /// The rotary table, one entry per rotated pair (`P`).
    pub rotary: RotaryTables<T>,
    /// The optional parts' reals, each present exactly where
    /// [`AttentionBlock`] holds the part.
    /// Source 0's key rounding's floor (`KR`).
    pub(in crate::operators::attention) key_rounding: Option<R>,
    pub(in crate::operators::attention) low_rank_query: Option<LowRankReals<R>>,
    pub(in crate::operators::attention) selection: Option<SelectionReals<R>>,
    pub(in crate::operators::attention) picked: Option<PickedReals<R>>,
}

/// The rotary table's three columns: per rotated pair, its coordinate axis
/// (`i32`), frequency and amplitude (`f32`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RotaryTables<T> {
    pub components: T,
    pub frequencies: T,
    pub amplitudes: T,
}

/// A low-rank query's `qr` norm's epsilon.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LowRankReals<R> {
    pub epsilon: R,
}

/// A selected read's query norm's epsilon.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SelectionReals<R> {
    pub epsilon: R,
}

/// An index-list source's reals: none for a reused one's.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PickedReals<R> {
    Reused,
    Indexed(IndexedReals<R>),
}

/// An indexing source's reals.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct IndexedReals<R> {
    /// Its query heads' own norm's epsilon, if they have one
    /// (`IndexerQuery::norm`).
    pub query_norm: Option<R>,
    /// The epsilon of the norm of the keys it writes
    /// (`IndexerKeys::Latent`); none for keys it shares with their owner.
    pub key_norm: Option<R>,
    /// Its score scale.
    pub scale: R,
    /// Its query's and written keys' roundings' floors, each present
    /// exactly where [`IndexedSource`] holds the rounding.
    pub(in crate::operators::attention) query_rounding: Option<R>,
    pub(in crate::operators::attention) key_rounding: Option<R>,
    /// An own source's rows' reals, present exactly where
    /// [`IndexedSource::own`] is.
    pub(in crate::operators::attention) own: Option<OwnReals<R>>,
}

/// An own source's rows' norm's epsilon and its rounding's floor, present
/// exactly where [`OwnRows::rounding`] is.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct OwnReals<R> {
    pub epsilon: R,
    pub(in crate::operators::attention) rounding: Option<R>,
}

impl<R, T> Numbers<R> for AttentionTopology<R, T> {
    type Of<E> = AttentionTopology<E, T>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> AttentionTopology<E, T> {
        AttentionTopology {
            form: self.form,
            query_kind: self.query_kind,
            query_segment_kind: self.query_segment_kind,
            activation: self.activation,
            inject_only: self.inject_only,
            causal_cache: self.causal_cache,
            controls: self.controls,
            epsilon: f(self.epsilon),
            head_epsilon: f(self.head_epsilon),
            post_norm: self.post_norm.map(|post_norm| post_norm.map(&mut *f)),
            scale: f(self.scale),
            rotary: self.rotary,
            key_rounding: self.key_rounding.map(&mut *f),
            low_rank_query: self.low_rank_query.map(|part| LowRankReals { epsilon: f(part.epsilon) }),
            selection: self.selection.map(|part| SelectionReals { epsilon: f(part.epsilon) }),
            picked: self.picked.map(|picked| match picked {
                PickedReals::Reused => PickedReals::Reused,
                PickedReals::Indexed(reals) => PickedReals::Indexed(IndexedReals {
                    query_norm: reals.query_norm.map(&mut *f),
                    key_norm: reals.key_norm.map(&mut *f),
                    scale: f(reals.scale),
                    query_rounding: reals.query_rounding.map(&mut *f),
                    key_rounding: reals.key_rounding.map(&mut *f),
                    own: reals.own.map(|own| OwnReals {
                        epsilon: f(own.epsilon),
                        rounding: own.rounding.map(&mut *f),
                    }),
                }),
            }),
        }
    }
}

impl<R, T> AttentionTopology<R, T> {
    /// Its tables mapped by `f`, in field order: a split's bytes to a key's
    /// indices (`FactSink::table`), a key's to a recording's (`Facts::table`),
    /// or bytes to a load's constants (`Table::from`).
    pub(crate) fn tables<U>(self, mut f: impl FnMut(T) -> U) -> AttentionTopology<R, U> {
        let RotaryTables { components, frequencies, amplitudes } = self.rotary;
        AttentionTopology {
            form: self.form,
            query_kind: self.query_kind,
            query_segment_kind: self.query_segment_kind,
            activation: self.activation,
            inject_only: self.inject_only,
            causal_cache: self.causal_cache,
            controls: self.controls,
            epsilon: self.epsilon,
            head_epsilon: self.head_epsilon,
            post_norm: self.post_norm,
            scale: self.scale,
            rotary: RotaryTables {
                components: f(components),
                frequencies: f(frequencies),
                amplitudes: f(amplitudes),
            },
            key_rounding: self.key_rounding,
            low_rank_query: self.low_rank_query,
            selection: self.selection,
            picked: self.picked,
        }
    }
}

/// An optional part's two sides together: its numbers and its reals, which
/// [`split`] makes present together.
///
/// # Panics
///
/// Where one side holds the part and the other does not: no split makes
/// such a pair.
pub(in crate::operators::attention) fn both<'n, 'r, N, R>(
    numbers: &'n Option<N>,
    reals: &'r Option<R>,
    part: &str,
) -> Option<(&'n N, &'r R)> {
    match (numbers, reals) {
        (Some(numbers), Some(reals)) => Some((numbers, reals)),
        (None, None) => None,
        _ => panic!("an attention block's {part} numbers and reals disagree"),
    }
}

/// An index-list source's two sides together ([`both`]).
pub(in crate::operators::attention) enum PickedPart<'n, D, R> {
    Reused { plane: &'n Plane<D> },
    Indexed { source: &'n IndexedSource<D>, reals: &'n IndexedReals<R> },
}

/// A block's index-list source with its reals, if it has one.
///
/// # Panics
///
/// Where the two sides disagree: no split makes such a pair.
pub(in crate::operators::attention) fn picked<'n, D, R, T>(
    topology: &'n AttentionTopology<R, T>,
    numbers: &'n AttentionBlock<D>,
) -> Option<PickedPart<'n, D, R>> {
    match both(&numbers.picked, &topology.picked, "index-list source")? {
        (PickedSource::Reused { plane }, PickedReals::Reused) => Some(PickedPart::Reused { plane }),
        (PickedSource::Indexed(source), PickedReals::Indexed(reals)) => {
            Some(PickedPart::Indexed { source, reals })
        }
        _ => panic!("an attention block's index-list source numbers and reals disagree"),
    }
}

/// An attention block's numbers, and each optional part's numbers and
/// structure, present exactly where the block has the part.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AttentionBlock<D = Dim> {
    /// `M`, `R`: the class's rows and their visible spans.
    pub rows: D,
    pub segments: D,
    /// `T`: the history plane's rows, and its slab rows.
    pub history_rows: D,
    pub slab_rows: D,
    /// The history row tiles one launch of the class can see: 0 for a
    /// class that lists none.
    pub history_tiles: D,
    /// The shape's numbers.
    pub widths: AttentionWidths<D>,
    /// The entries' `gate_function` code.
    pub gate_function: D,
    /// Source 0's key rounding's scalars (`KR`).
    pub(in crate::operators::attention) key_rounding: Option<RoundingWords<D>>,
    pub(in crate::operators::attention) low_rank_query: Option<LowRankQuery<D>>,
    pub(in crate::operators::attention) grouped: Option<GroupedOutput<D>>,
    pub(in crate::operators::attention) selection: Option<SelectedRead<D>>,
    pub(in crate::operators::attention) picked: Option<PickedSource<D>>,
}

impl<D: Clone> AttentionBlock<D> {
    /// The same block at `rows` rows.
    pub(crate) fn at_rows(&self, rows: D) -> Self {
        Self { rows, ..self.clone() }
    }

    /// The block at its class's own `rows`, `segments` and listed history
    /// tiles: the dimensions its classes vary, which an indexing source's
    /// list spans follow (a row's list spans its visible segments).
    pub(crate) fn share_classes(&mut self, rows: &D, segments: &D, history_tiles: &D) {
        self.rows = rows.clone();
        self.segments = segments.clone();
        self.history_tiles = history_tiles.clone();
        if let Some(PickedSource::Indexed(source)) = &mut self.picked {
            source.spans = segments.clone();
        }
    }
}

/// A low-rank query's rank.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LowRankQuery<D = Dim> {
    pub rank: D,
}

/// A grouped output's `project_rows` axes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GroupedOutput<D = Dim> {
    pub groups: D,
    pub inputs: D,
    pub rank: D,
}

/// A selected read at one class: whether the class scores, its axes and
/// its history domains' planes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SelectedRead<D = Dim> {
    pub scored: bool,
    pub heads: D,
    pub width: D,
    pub block: D,
    pub blocks: D,
    pub words: D,
    pub window: Plane<D>,
    pub blocks_plane: Plane<D>,
}

/// A history plane's rows and slab rows.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Plane<D = Dim> {
    pub rows: D,
    pub slab_rows: D,
}

/// An index-list source: another layer's picks over a plane, or its own
/// indexer's.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PickedSource<D = Dim> {
    Reused { plane: Plane<D> },
    Indexed(IndexedSource<D>),
}

/// An indexing source: the Block plane it picks from and its indexer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct IndexedSource<D = Dim> {
    pub plane: Plane<D>,
    /// The indexer's heads, their width and its picks.
    pub heads: D,
    pub width: D,
    pub top: D,
    /// The spans (`R`) and scores (`N`) of a row's list over the read.
    pub spans: D,
    pub scores: D,
    /// The candidates it scores.
    pub candidates: Candidates<D>,
    /// Whether its head weights are projected (`HeadWeights::Projected`).
    pub head_weights: bool,
    /// Its query's and written keys' roundings' scalars.
    pub(in crate::operators::attention) query_rounding: Option<RoundingWords<D>>,
    pub(in crate::operators::attention) key_rounding: Option<RoundingWords<D>>,
    /// An own source's rows; none for a Reindex source.
    pub(in crate::operators::attention) own: Option<OwnRows<D>>,
}

/// The candidates an indexer scores: every complete unit, every unit while
/// it produces a candidate pool, or a pool's.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Candidates<D = Dim> {
    All,
    Produce(CandidatePool<D>),
    Restrict(CandidatePool<D>),
}

/// An own source's rows: its pool's gate (`GT`) and rounding, its rate and
/// its Window domain (none at rate 1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct OwnRows<D = Dim> {
    pub gated: bool,
    pub(in crate::operators::attention) rounding: Option<RoundingWords<D>>,
    pub rate: D,
    pub window: Option<Plane<D>>,
}

impl<D> Numbers<D> for Plane<D> {
    type Of<E> = Plane<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> Plane<E> {
        Plane { rows: f(self.rows), slab_rows: f(self.slab_rows) }
    }
}

impl<D> Numbers<D> for IndexedSource<D> {
    type Of<E> = IndexedSource<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> IndexedSource<E> {
        IndexedSource {
            plane: self.plane.map(&mut *f),
            heads: f(self.heads),
            width: f(self.width),
            top: f(self.top),
            spans: f(self.spans),
            scores: f(self.scores),
            candidates: match self.candidates {
                Candidates::All => Candidates::All,
                Candidates::Produce(pool) => Candidates::Produce(pool.map(&mut *f)),
                Candidates::Restrict(pool) => Candidates::Restrict(pool.map(&mut *f)),
            },
            head_weights: self.head_weights,
            query_rounding: self.query_rounding.map(|rounding| rounding.map(&mut *f)),
            key_rounding: self.key_rounding.map(|rounding| rounding.map(&mut *f)),
            own: self.own.map(|own| OwnRows {
                gated: own.gated,
                rounding: own.rounding.map(|rounding| rounding.map(&mut *f)),
                rate: f(own.rate),
                window: own.window.map(|window| window.map(&mut *f)),
            }),
        }
    }
}

impl<D> Numbers<D> for AttentionBlock<D> {
    type Of<E> = AttentionBlock<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> AttentionBlock<E> {
        AttentionBlock {
            rows: f(self.rows),
            segments: f(self.segments),
            history_rows: f(self.history_rows),
            slab_rows: f(self.slab_rows),
            history_tiles: f(self.history_tiles),
            widths: self.widths.map(&mut *f),
            gate_function: f(self.gate_function),
            key_rounding: self.key_rounding.map(|rounding| rounding.map(&mut *f)),
            low_rank_query: self.low_rank_query.map(|low_rank| LowRankQuery { rank: f(low_rank.rank) }),
            grouped: self.grouped.map(|grouped| GroupedOutput {
                groups: f(grouped.groups),
                inputs: f(grouped.inputs),
                rank: f(grouped.rank),
            }),
            selection: self.selection.map(|read| SelectedRead {
                scored: read.scored,
                heads: f(read.heads),
                width: f(read.width),
                block: f(read.block),
                blocks: f(read.blocks),
                words: f(read.words),
                window: read.window.map(&mut *f),
                blocks_plane: read.blocks_plane.map(&mut *f),
            }),
            picked: self.picked.map(|picked| match picked {
                PickedSource::Reused { plane } => PickedSource::Reused { plane: plane.map(&mut *f) },
                PickedSource::Indexed(source) => PickedSource::Indexed(source.map(&mut *f)),
            }),
        }
    }
}

/// `site` as its builders take it: its topology, with the model's own reals
/// and tables, and its numbers. The only reading of the model behind an
/// attention block's graph.
pub(crate) fn split(
    site: &AttentionSite<'_>,
) -> Result<(AttentionTopology<f64, Arc<[u8]>>, AttentionBlock<u64>), GraphError> {
    let attention = site.operator;
    let selected = match (site.selection, operator::admitted_selection(attention)) {
        (None, _) => None,
        (Some(selection), Some(read)) => Some((selection, read)),
        (Some(_), None) => {
            return Err("a selected attention block's operator reads every visible row".into());
        }
    };
    let low_rank_query = operator::low_rank_query(attention);
    let picked = site.picked.map(split_picked);
    let key_rounding = site.shape.key_rounding.part().unzip();
    let topology = AttentionTopology {
        form: site.shape.form,
        query_kind: operator::query_kind(attention),
        query_segment_kind: operator::query_segment_kind(attention),
        activation: site.activation,
        inject_only: site.inject_only,
        causal_cache: site.causal_cache,
        controls: site.controls,
        epsilon: f64::from(site.epsilon),
        head_epsilon: f64::from(site.head_epsilon),
        post_norm: site.post_norm,
        scale: attention.scale,
        rotary: RotaryTables {
            components: bytes(rotary_components(&attention.rotary)?, i32::to_le_bytes),
            frequencies: bytes(rotary_frequencies(&attention.rotary), f32::to_le_bytes),
            amplitudes: bytes(rotary_amplitudes(&attention.rotary), f32::to_le_bytes),
        },
        key_rounding: key_rounding.1,
        low_rank_query: low_rank_query.map(|(_, epsilon)| LowRankReals { epsilon }),
        selection: selected.map(|(_, read)| SelectionReals { epsilon: read.query_norm.epsilon }),
        picked: picked.as_ref().map(|(_, reals)| reals.clone()),
    };
    let numbers = AttentionBlock {
        rows: site.rows,
        segments: site.segments,
        history_rows: site.history_rows,
        slab_rows: u64::from(site.slab_rows),
        history_tiles: site.history_tiles,
        widths: site.shape.widths,
        gate_function: crate::operators::code(operator::gate_function(attention)),
        key_rounding: key_rounding.0,
        low_rank_query: low_rank_query.map(|(rank, _)| LowRankQuery { rank }),
        grouped: operator::grouped_output(attention).map(|grouped| GroupedOutput {
            groups: grouped.groups,
            inputs: grouped.inputs,
            rank: grouped.rank,
        }),
        selection: selected.map(|(selection, _)| SelectedRead {
            scored: selection.scored,
            heads: selection.shape.heads,
            width: selection.shape.width,
            block: selection.shape.block,
            blocks: selection.shape.blocks,
            words: selection.shape.words,
            window: Plane {
                rows: selection.window_rows,
                slab_rows: u64::from(selection.window_slab_rows),
            },
            blocks_plane: Plane {
                rows: selection.block_rows,
                slab_rows: u64::from(selection.block_slab_rows),
            },
        }),
        picked: picked.map(|(numbers, _)| numbers),
    };
    Ok((topology, numbers))
}

/// A table column's bytes: each value's little-endian bytes in order.
fn bytes<V, const N: usize>(values: Vec<V>, of: fn(V) -> [u8; N]) -> Arc<[u8]> {
    values.into_iter().flat_map(of).collect()
}

fn plane(plane: PickedPlane) -> Plane<u64> {
    Plane { rows: plane.rows, slab_rows: u64::from(plane.slab_rows) }
}

/// [`split`] of an index-list source: its numbers and its reals.
fn split_picked(picked: PickedBlock<'_>) -> (PickedSource<u64>, PickedReals<f64>) {
    let (picked_plane, index, own) = match picked {
        PickedBlock::Reused { plane: picked_plane } => {
            return (PickedSource::Reused { plane: plane(picked_plane) }, PickedReals::Reused);
        }
        PickedBlock::Indexed { plane, index, own } => (plane, index, own),
    };
    let indexer = index.indexer;
    let (query_rounding, query_floor) = index.query_rounding.part().unzip();
    let (key_rounding, key_floor) = index.key_rounding.part().unzip();
    let own_rounding = own.as_ref().map(|own| crate::planning::rounding_args(own.rows.rounding).part().unzip());
    let numbers = IndexedSource {
        plane: plane(picked_plane),
        heads: indexer.heads,
        width: indexer.width,
        top: indexer.top,
        spans: index.spans,
        scores: index.scores,
        candidates: match index.candidates {
            CandidatesBinding::All => Candidates::All,
            CandidatesBinding::Produce(candidates) => Candidates::Produce(candidates),
            CandidatesBinding::Restrict(candidates) => Candidates::Restrict(candidates),
        },
        head_weights: matches!(indexer.head_weights, HeadWeights::Projected(_)),
        query_rounding,
        key_rounding,
        own: own.as_ref().zip(own_rounding).map(|(own, (rounding, _))| OwnRows {
            gated: super::pooled::gated(own.rows),
            rounding,
            rate: own.rows.rate,
            window: own.window.map(|window| Plane {
                rows: window.rows,
                slab_rows: u64::from(window.slab_rows),
            }),
        }),
    };
    let reals = IndexedReals {
        query_norm: match &indexer.query.norm {
            HeadNorm::Rms(norm) => Some(norm.epsilon),
            HeadNorm::None => None,
        },
        key_norm: match &indexer.keys {
            IndexerKeys::Latent { norm, .. } => Some(norm.epsilon),
            IndexerKeys::Shared { .. } => None,
            IndexerKeys::Pooled(_) => {
                unreachable!("picked_source refuses pooled indexer keys over a pooled source")
            }
        },
        scale: indexer.scale,
        query_rounding: query_floor,
        key_rounding: key_floor,
        own: own.zip(own_rounding).map(|(own, (_, rounding))| OwnReals { epsilon: own.rows.norm.epsilon, rounding }),
    };
    (PickedSource::Indexed(numbers), PickedReals::Indexed(reals))
}
