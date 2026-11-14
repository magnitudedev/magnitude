//! Hyper-connection residual streams (`streams.seismic`). Under
//! `ResidualForm::HyperConnections` the residual between block graphs is the
//! S streams as one `[M, S * H]` F32 tensor: block 0 widens the embedded rows
//! to S copies (`repeat_rows`), each sublayer reads its input through a mix
//! site (`stream_mix_down`, `stream_mix_up`: the per-stream partials its
//! consumer sums) and scatters its output into the streams with the site's
//! inject values (the output entries' join), and the last block ends with
//! the decoder's exit site, whose partials the readout sums.
//!
//! Under Sinkhorn mixing each site is one entry (`stream_mix_sinkhorn`): it
//! reads the streams and gives its sublayer one input row (normalized by the
//! sublayer's input norm), mixed by the previous site's `pre` or its own
//! (`SinkhornInput`), and the join row post | comb its sublayer's output
//! joins with. Mixing by the previous site's, a site also gives its own
//! `pre`, which crosses every boundary between graphs beside the streams
//! ([`ResidualCarry`]); block 0 starts it one-hot. The decoder's exit
//! collapses the streams (`stream_collapse`) by the last site's `pre` or
//! into their mean, which the readout normalizes.

pub(crate) mod graph;

use crate::error::PlanError;
use crate::operators::ExitMix;
use magnitude_family_contracts::{
    Decoder, ExitNorm, HyperConnectionMixing, ResidualForm, SinkhornInput, WeightKind, WeightScope,
};
use seismic::Element;

/// The geometry of a decoder's streams: S streams of `hidden`, and how its
/// mix sites form their coefficients. The stream count is topology (the
/// graphs repeat over it); the width and rank are numbers `D`. Sinkhorn
/// mixing names whose `pre` its sites mix by, which is topology, and
/// carries `M`: the model's numerics (`SinkhornMixing`), or in a graph's
/// numbers (`numbers`) its iteration count, a number the kernel loops over,
/// whose topology holds the epsilons as reals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamsGeometry<D = u64, M = SinkhornMixing> {
    pub hidden: D,
    pub streams: u64,
    pub mixing: StreamsMixing<D, M>,
}

/// How every mix site of a decoder forms its coefficients
/// (`HyperConnectionMixing`): through a low-rank bottleneck, or by one
/// Sinkhorn projection of numerics `M` mixing by the `pre` `input` names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StreamsMixing<D = u64, M = SinkhornMixing> {
    LowRank { rank: D },
    Sinkhorn { input: SinkhornInput, numerics: M },
}

/// `stream_mix_sinkhorn`'s `P` under `input`: 1 when a site mixes by the
/// previous site's pre and publishes its own for the next, 0 when it mixes
/// by its own.
pub(crate) fn carried(input: SinkhornInput) -> u64 {
    match input {
        SinkhornInput::PreviousSite => 1,
        SinkhornInput::OwnSite => 0,
    }
}

/// How the decoder's exit collapses Sinkhorn streams (`stream_collapse`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SinkhornExit {
    /// By the last site's pre, which the streams carry.
    LastPre,
    /// Into the streams' mean `Σ_s X[s] / S` bit for bit, for any S: every
    /// stream summed in stream order, then divided by S, reading no pre.
    Mean,
}

impl SinkhornExit {
    /// The collapse the decoder's exit `mix` ends Sinkhorn streams with;
    /// none for a low-rank site of its own.
    pub(crate) fn of(mix: ExitMix<'_>) -> Option<Self> {
        match mix {
            ExitMix::Previous(_) => Some(Self::LastPre),
            ExitMix::Mean(_) => Some(Self::Mean),
            ExitMix::Own(_) => None,
        }
    }

    /// The collapse `decoder`'s exit ends its Sinkhorn streams with.
    pub(crate) fn of_decoder(decoder: &Decoder) -> Result<Self, PlanError> {
        let ExitNorm::HyperConnectionHead(head) = &decoder.exit.norm else {
            return Err(PlanError::Topology("residual streams without their exit"));
        };
        Self::of(ExitMix::of(head)?)
            .ok_or(PlanError::Topology("Sinkhorn streams ending in a low-rank exit site"))
    }

    /// `stream_collapse`'s `P`: 1 collapsing by the last site's pre, 0 into
    /// the mean.
    pub fn previous(self) -> u64 {
        match self {
            Self::LastPre => 1,
            Self::Mean => 0,
        }
    }
}

/// The Sinkhorn mixing's numerics, as `stream_mix_sinkhorn` takes them: its
/// iteration count, the epsilon of `pre`, the comb and its normalizations,
/// and its projection's RMS epsilon.
#[derive(Clone, Copy, Debug)]
pub struct SinkhornMixing {
    pub iterations: u32,
    pub epsilon: f32,
    pub norm_epsilon: f32,
}

impl SinkhornMixing {
    fn key(&self) -> (u32, u32, u32) {
        (self.iterations, self.epsilon.to_bits(), self.norm_epsilon.to_bits())
    }
}

impl PartialEq for SinkhornMixing {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for SinkhornMixing {}

impl std::hash::Hash for SinkhornMixing {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state)
    }
}

impl SinkhornMixing {
    /// Its epsilons as a graph's topology holds them.
    pub(crate) fn topology(self) -> SinkhornTopology<f64> {
        SinkhornTopology {
            epsilon: f64::from(self.epsilon),
            norm_epsilon: f64::from(self.norm_epsilon),
        }
    }
}

/// The Sinkhorn mixing's epsilons as a graph's topology holds them, as
/// reals `R` (`SinkhornMixing`); its iteration count is a number of the
/// graph's streams geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SinkhornTopology<R = seismic::Argument<f64>> {
    pub epsilon: R,
    pub norm_epsilon: R,
}

impl<R> crate::programs::graph::facts::Numbers<R> for SinkhornTopology<R> {
    type Of<E> = SinkhornTopology<E>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> SinkhornTopology<E> {
        SinkhornTopology {
            epsilon: f(self.epsilon),
            norm_epsilon: f(self.norm_epsilon),
        }
    }
}

impl<D> crate::programs::graph::facts::Numbers<D> for StreamsGeometry<D, D> {
    type Of<E> = StreamsGeometry<E, E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> StreamsGeometry<E, E> {
        StreamsGeometry {
            hidden: f(self.hidden),
            streams: self.streams,
            mixing: match self.mixing {
                StreamsMixing::LowRank { rank } => StreamsMixing::LowRank { rank: f(rank) },
                StreamsMixing::Sinkhorn { input, numerics: iterations } => {
                    StreamsMixing::Sinkhorn { input, numerics: f(iterations) }
                }
            },
        }
    }
}

impl<D: Clone, M> StreamsGeometry<D, M> {
    /// The width of a row of the streams, `S * H`.
    pub fn width(&self) -> D
    where
        D: std::ops::Mul<u64, Output = D>,
    {
        self.hidden.clone() * self.streams
    }

    /// The low-rank sites' geometry, of low-rank mixing.
    pub fn low_rank(&self) -> Option<LowRankGeometry<D>> {
        match &self.mixing {
            StreamsMixing::LowRank { rank } => Some(LowRankGeometry {
                hidden: self.hidden.clone(),
                streams: self.streams,
                rank: rank.clone(),
            }),
            StreamsMixing::Sinkhorn { .. } => None,
        }
    }

    /// The Sinkhorn numerics, of Sinkhorn mixing.
    pub fn sinkhorn(&self) -> Option<&M> {
        match &self.mixing {
            StreamsMixing::Sinkhorn { numerics, .. } => Some(numerics),
            StreamsMixing::LowRank { .. } => None,
        }
    }

    /// The width of the `pre` a row of the streams carries from site to
    /// site: S under Sinkhorn mixing by the previous site's, none else.
    pub fn carried_pre(&self) -> Option<u64> {
        match self.mixing {
            StreamsMixing::Sinkhorn { input: SinkhornInput::PreviousSite, .. } => Some(self.streams),
            StreamsMixing::Sinkhorn { input: SinkhornInput::OwnSite, .. } | StreamsMixing::LowRank { .. } => None,
        }
    }

    /// The width of a row of the streams' join values `IN`: an inject value
    /// per stream, and under Sinkhorn mixing the comb after them.
    pub fn join_width(&self) -> u64 {
        match self.mixing {
            StreamsMixing::LowRank { .. } => self.streams,
            StreamsMixing::Sinkhorn { .. } => self.streams + self.streams * self.streams,
        }
    }
}

impl<D> StreamsGeometry<D, D> {
    /// Its numbers as a graph builder takes them.
    pub(crate) fn dims(self) -> StreamsGeometry<seismic::Dim, seismic::Dim>
    where
        D: Into<seismic::Dim>,
    {
        crate::programs::graph::facts::Numbers::map(self, &mut Into::into)
    }
}

impl StreamsGeometry {
    /// Its numbers as a graph's numbers hold them: the Sinkhorn iteration
    /// count in place of the mixing's numerics, whose epsilons the graph's
    /// topology holds (`SinkhornMixing::topology`).
    pub(crate) fn numbers(self) -> StreamsGeometry<u64, u64> {
        StreamsGeometry {
            hidden: self.hidden,
            streams: self.streams,
            mixing: match self.mixing {
                StreamsMixing::LowRank { rank } => StreamsMixing::LowRank { rank },
                StreamsMixing::Sinkhorn { input, numerics } => {
                    StreamsMixing::Sinkhorn { input, numerics: u64::from(numerics.iterations) }
                }
            },
        }
    }

    /// The streams of `decoder`: none for a single residual stream.
    pub(crate) fn of(decoder: &Decoder) -> Result<Option<Self>, PlanError> {
        Self::from_residual(decoder.hidden, &decoder.residual)
    }

    pub(crate) fn from_residual(hidden: u64, residual: &ResidualForm) -> Result<Option<Self>, PlanError> {
        let ResidualForm::HyperConnections(form) = residual else {
            return Ok(None);
        };
        let mixing = match form.mixing {
            HyperConnectionMixing::LowRankGated { rank } => StreamsMixing::LowRank { rank },
            HyperConnectionMixing::Sinkhorn {
                iterations,
                epsilon,
                norm_epsilon,
                input,
            } => StreamsMixing::Sinkhorn {
                input,
                numerics: SinkhornMixing {
                    iterations: u32::try_from(iterations)
                        .map_err(|_| PlanError::Arithmetic("Sinkhorn iterations exceed u32"))?,
                    epsilon: epsilon as f32,
                    norm_epsilon: norm_epsilon as f32,
                },
            },
        };
        Ok(Some(Self {
            hidden,
            streams: form.streams,
            mixing,
        }))
    }

}

/// S low-rank streams of `hidden`, and the rank of their mix sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LowRankGeometry<D = u64> {
    pub hidden: D,
    pub streams: u64,
    pub rank: D,
}

impl<D: Clone> LowRankGeometry<D> {
    /// The width of a row of the streams, `S * H`.
    pub fn width(&self) -> D
    where
        D: std::ops::Mul<u64, Output = D>,
    {
        self.hidden.clone() * self.streams
    }

    /// Its numbers as a graph builder takes them.
    pub(crate) fn dims(self) -> LowRankGeometry<seismic::Dim>
    where
        D: Into<seismic::Dim>,
    {
        LowRankGeometry { hidden: self.hidden.into(), streams: self.streams, rank: self.rank.into() }
    }
}

/// What crosses every boundary between graphs of the residual: its rows (the
/// `[rows, S * H]` streams, or one `[rows, H]` stream), and under Sinkhorn
/// mixing by the previous site's pre the `[rows, S]` F32 `pre` of the last
/// site run over them, which the next site reads
/// ([`StreamsGeometry::carried_pre`]). A block's graph reads and leaves one,
/// as its input and output ports; a split block's front leaves it to its
/// back; a run binds it from graph to graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResidualCarry<T> {
    pub rows: T,
    pub pre: Option<T>,
}

impl<T> ResidualCarry<T> {
    /// The rows of a residual that carries no `pre`.
    pub fn rows(rows: T) -> Self {
        Self { rows, pre: None }
    }

    pub fn as_ref(&self) -> ResidualCarry<&T> {
        ResidualCarry {
            rows: &self.rows,
            pre: self.pre.as_ref(),
        }
    }

    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> ResidualCarry<U> {
        ResidualCarry {
            rows: map(self.rows),
            pre: self.pre.map(map),
        }
    }

    pub fn try_map<U, E>(self, mut map: impl FnMut(T) -> Result<U, E>) -> Result<ResidualCarry<U>, E> {
        Ok(ResidualCarry {
            rows: map(self.rows)?,
            pre: self.pre.map(map).transpose()?,
        })
    }
}

/// The planned elements of a mix site's norm, down and up weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MixSite {
    pub norm: Element,
    pub down: Element,
    pub up: Element,
}

/// A sublayer's mix site, which also scatters the sublayer's output into
/// the streams with its inject weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InjectingSite {
    pub site: MixSite,
    pub inject: Element,
}

/// One value for each mix site a block's graph runs, in graph order: its
/// mixer's, its feed-forward's (absent for a lone mixer block), and the
/// exit site it ends with (present for the decoder's last block and for
/// every draft head block).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sites<T> {
    pub mixer: T,
    pub feed_forward: Option<T>,
    pub exit: Option<T>,
}

impl<T> Sites<T> {
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> Sites<U> {
        Sites {
            mixer: map(self.mixer),
            feed_forward: self.feed_forward.map(&mut map),
            exit: self.exit.map(map),
        }
    }

    pub fn as_ref(&self) -> Sites<&T> {
        Sites {
            mixer: &self.mixer,
            feed_forward: self.feed_forward.as_ref(),
            exit: self.exit.as_ref(),
        }
    }

    /// Each present site, in graph order.
    pub fn present(self) -> impl Iterator<Item = T> {
        std::iter::once(self.mixer)
            .chain(self.feed_forward)
            .chain(self.exit)
    }

    /// Each present site with its scope among `[mixer, feed_forward, exit]`.
    pub fn scoped(self, scopes: [WeightScope; 3]) -> impl Iterator<Item = (T, WeightScope)> {
        let [mixer, feed_forward, exit] = scopes;
        std::iter::once((self.mixer, mixer))
            .chain(self.feed_forward.map(|site| (site, feed_forward)))
            .chain(self.exit.map(|site| (site, exit)))
    }
}

impl<T> Sites<Option<T>> {
    pub fn flatten(self) -> Option<Sites<T>> {
        Some(Sites {
            mixer: self.mixer?,
            feed_forward: match self.feed_forward {
                Some(site) => Some(site?),
                None => None,
            },
            exit: match self.exit {
                Some(site) => Some(site?),
                None => None,
            },
        })
    }
}

/// The streams a block's graph reads and writes, as its kernels bind them:
/// low-rank sites, each at its own weights' elements, or Sinkhorn sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockStreams {
    LowRank(StreamsBinding),
    Sinkhorn(SinkhornBinding),
}

impl BlockStreams {
    /// The block's streams of `geometry`, whose low-rank sites `low_rank`
    /// binds and whose Sinkhorn streams end in `sinkhorn_exit`'s collapse.
    pub(crate) fn of(
        geometry: StreamsGeometry,
        low_rank: impl FnOnce(LowRankGeometry) -> Result<StreamsBinding, PlanError>,
        sinkhorn_exit: impl FnOnce() -> Result<SinkhornExit, PlanError>,
    ) -> Result<Self, PlanError> {
        Ok(match geometry.mixing {
            StreamsMixing::LowRank { rank } => Self::LowRank(low_rank(LowRankGeometry {
                hidden: geometry.hidden,
                streams: geometry.streams,
                rank,
            })?),
            StreamsMixing::Sinkhorn { input, numerics } => Self::Sinkhorn(SinkhornBinding {
                hidden: geometry.hidden,
                streams: geometry.streams,
                input,
                mixing: numerics,
                exit: sinkhorn_exit()?,
            }),
        })
    }
}

/// A block's Sinkhorn streams: S streams of `hidden` and their mixing by
/// the `pre` `input` names, whose sites' weights are F32
/// (`is_fixed_dense_role`), so their entries bind no element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SinkhornBinding {
    pub hidden: u64,
    pub streams: u64,
    pub input: SinkhornInput,
    pub mixing: SinkhornMixing,
    /// The collapse the decoder's exit ends the streams with.
    pub exit: SinkhornExit,
}

impl SinkhornBinding {
    /// Its sites' `P` (`carried`).
    pub fn carried(&self) -> u64 {
        carried(self.input)
    }
}

/// The low-rank streams a block's graph reads and writes, and the elements
/// of each mix site it runs over them, as that site's own weights planned
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamsBinding {
    pub geometry: LowRankGeometry,
    pub mixer: InjectingSite,
    pub feed_forward: Option<InjectingSite>,
    pub exit: Option<MixSite>,
    pub activation: Element,
}

/// `stream_mix_down`'s elements at a site: its norm and down weights, its
/// inject weights (none at an exit site, which injects into nothing and
/// binds its absent inject weights at the down weight's element), and the
/// activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MixDownBinding {
    pub norm: Element,
    pub down: Element,
    pub inject: Option<Element>,
    pub activation: Element,
}

impl MixDownBinding {
    pub fn injected(self) -> bool {
        self.inject.is_some()
    }

    /// The element its inject weight port binds.
    pub fn inject_element(self) -> Element {
        self.inject.unwrap_or(self.down)
    }
}

/// `stream_mix_up`'s elements at a site: its up weights, its norm and the
/// activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MixUpBinding {
    pub up: Element,
    pub norm: Element,
    pub activation: Element,
}

/// The entry bindings of one mix site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SiteEntryBindings {
    pub down: MixDownBinding,
    pub up: MixUpBinding,
}

impl StreamsBinding {
    /// The site sublayer `sublayer` of the block reads through: the
    /// mixer's (0) or the feed-forward's (1).
    pub fn sublayer(&self, sublayer: u32) -> Option<InjectingSite> {
        match sublayer {
            0 => Some(self.mixer),
            1 => self.feed_forward,
            _ => None,
        }
    }

    /// The entry bindings of each of its sites.
    pub fn entries(&self) -> Sites<SiteEntryBindings> {
        let activation = self.activation;
        let entries = |site: MixSite, inject| SiteEntryBindings {
            down: MixDownBinding {
                norm: site.norm,
                down: site.down,
                inject,
                activation,
            },
            up: MixUpBinding {
                up: site.up,
                norm: site.norm,
                activation,
            },
        };
        let injecting = |site: InjectingSite| entries(site.site, Some(site.inject));
        Sites {
            mixer: injecting(self.mixer),
            feed_forward: self.feed_forward.map(injecting),
            exit: self.exit.map(|site| entries(site, None)),
        }
    }
}

/// The streams binding of a block whose mix sites are the mixer at `mixer`,
/// the feed-forward at `feed_forward` (absent for a lone mixer block) and
/// the exit site at `exit` (when the block ends with one). `lookup` resolves
/// the planned element of a role in a scope; each site binds its own.
pub(crate) fn binding(
    geometry: LowRankGeometry,
    mixer: WeightScope,
    feed_forward: Option<WeightScope>,
    exit: Option<WeightScope>,
    lookup: impl Fn(WeightScope, WeightKind) -> Result<Element, PlanError>,
    activation: Element,
) -> Result<StreamsBinding, PlanError> {
    let site = |scope| -> Result<MixSite, PlanError> {
        Ok(MixSite {
            norm: lookup(scope, WeightKind::HyperConnectionNorm)?,
            down: lookup(scope, WeightKind::HyperConnectionDown)?,
            up: lookup(scope, WeightKind::HyperConnectionUp)?,
        })
    };
    let injecting = |scope| -> Result<InjectingSite, PlanError> {
        Ok(InjectingSite {
            site: site(scope)?,
            inject: lookup(scope, WeightKind::HyperConnectionInject)?,
        })
    };
    Ok(StreamsBinding {
        geometry,
        mixer: injecting(mixer)?,
        feed_forward: feed_forward.map(injecting).transpose()?,
        exit: exit.map(site).transpose()?,
        activation,
    })
}

#[cfg(test)]
mod tests {
    use crate::error::PlanError;
    use crate::operators::admit;
    use crate::planning::fixtures::{
        qwen4exp_fixture_definition, sinkhorn_fixture_definition_of,
        sinkhorn_fixture_definition_over,
    };
    use magnitude_family_contracts::{ExitSite, SinkhornInput};

    /// A mean exit is admitted over Sinkhorn streams of any count, a power
    /// of two or not (its collapse sums then divides by S, exact at either),
    /// and refused by name over low-rank streams, which have no collapse.
    #[test]
    fn a_mean_exit_is_admitted_over_sinkhorn_streams_of_any_count() {
        let mean = sinkhorn_fixture_definition_of(2, SinkhornInput::OwnSite, ExitSite::Mean);
        assert_eq!(admit(&mean, false), Ok(()));
        let three = sinkhorn_fixture_definition_over(3, 2, SinkhornInput::OwnSite, ExitSite::Mean);
        assert_eq!(admit(&three, false), Ok(()));
        let mut low_rank = qwen4exp_fixture_definition();
        low_rank.decoder.exit.norm = mean.decoder.exit.norm.clone();
        assert_eq!(
            admit(&low_rank, false),
            Err(PlanError::Unsupported("a mean exit over low-rank hyper-connection streams"))
        );
    }
}
