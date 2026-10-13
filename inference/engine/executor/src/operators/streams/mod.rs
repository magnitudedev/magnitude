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
    Decoder, HyperConnectionMixing, InputNorm, ResidualForm, WeightKind, WeightScope,
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

/// The geometry and element bindings of a decoder's mix sites, which every
/// site shares: the elements of the norm, down, inject and up weights and
/// the activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamsBinding {
    pub geometry: StreamsGeometry,
    pub norm: Element,
    pub down: Element,
    pub inject: Element,
    pub up: Element,
    pub activation: Element,
}

/// The binding of `decoder`'s mix sites: none for a single residual stream.
/// `lookup` resolves the planned element of a role in a scope; every site
/// (the sublayers' and the exit's) must plan the same elements.
pub(crate) fn binding(
    decoder: &Decoder,
    lookup: impl Fn(WeightScope, WeightKind) -> Result<Element, PlanError>,
    activation: Element,
) -> Result<Option<StreamsBinding>, PlanError> {
    let Some(geometry) = StreamsGeometry::of(decoder)? else {
        return Ok(None);
    };
    binding_sites(geometry, WeightScope::Target, decoder.sublayers().filter_map(|(index, sublayer)| {
        matches!(sublayer.input, InputNorm::HyperConnectionMix(_)).then_some(WeightScope::TargetSublayer(index))
    }), lookup, activation).map(Some)
}

pub(crate) fn binding_sites(
    geometry: StreamsGeometry,
    exit: WeightScope,
    sites: impl IntoIterator<Item = WeightScope>,
    lookup: impl Fn(WeightScope, WeightKind) -> Result<Element, PlanError>,
    activation: Element,
) -> Result<StreamsBinding, PlanError> {
    let site = |scope| -> Result<[Element; 3], PlanError> {
        Ok([
            lookup(scope, WeightKind::HyperConnectionNorm)?,
            lookup(scope, WeightKind::HyperConnectionDown)?,
            lookup(scope, WeightKind::HyperConnectionUp)?,
        ])
    };
    let [norm, down, up] = site(exit)?;
    let mut inject = None;
    for scope in sites {
        let elements = lookup(scope, WeightKind::HyperConnectionInject)?;
        if site(scope)? != [norm, down, up] || inject.is_some_and(|inject| inject != elements) {
            return Err(PlanError::Unsupported(
                "hyper-connection sites with differing weight representations",
            ));
        }
        inject = Some(elements);
    }
    Ok(StreamsBinding {
        geometry,
        norm,
        down,
        inject: inject.ok_or(PlanError::Topology("hyper-connections without a mix site"))?,
        up,
        activation,
    })
}
