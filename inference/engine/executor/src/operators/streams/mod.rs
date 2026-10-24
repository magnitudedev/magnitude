//! Hyper-connection residual streams (`streams.seismic`). Under
//! `ResidualForm::HyperConnections` the residual between block graphs is the
//! S streams as one `[M, S * H]` F32 tensor: block 0 widens the embedded rows
//! to S copies (`repeat_rows`), each sublayer reads its input through a mix
//! site (`stream_mix_down`, `stream_mix_up`: the per-stream partials its
//! consumer sums) and scatters its output into the streams with the site's
//! inject values (the output entries' join), and the last block ends with
//! the decoder's exit site, whose partials the readout sums.

pub(crate) mod graph;

use crate::error::PlanError;
use magnitude_family_contracts::{
    Decoder, HyperConnectionMixing, ResidualForm, WeightKind, WeightScope,
};
use seismic::Element;

/// The geometry of a decoder's streams: S streams of `hidden`, and the low
/// rank of its mix sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamsGeometry {
    pub hidden: u64,
    pub streams: u64,
    pub rank: u64,
}

impl StreamsGeometry {
    /// The streams of `decoder`: none for a single residual stream.
    pub(crate) fn of(decoder: &Decoder) -> Result<Option<Self>, PlanError> {
        Self::from_residual(decoder.hidden, &decoder.residual)
    }

    pub(crate) fn from_residual(hidden: u64, residual: &ResidualForm) -> Result<Option<Self>, PlanError> {
        let ResidualForm::HyperConnections(form) = residual else {
            return Ok(None);
        };
        let HyperConnectionMixing::LowRankGated { rank } = form.mixing else {
            return Err(PlanError::Unsupported("hyper-connection mixing"));
        };
        Ok(Some(Self {
            hidden,
            streams: form.streams,
            rank,
        }))
    }

    /// The width of a row of the streams, `S * H`.
    pub fn width(&self) -> u64 {
        self.streams * self.hidden
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

/// The streams a block's graph reads and writes, and the elements of each
/// mix site it runs over them, as that site's own weights planned them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamsBinding {
    pub geometry: StreamsGeometry,
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
    geometry: StreamsGeometry,
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
