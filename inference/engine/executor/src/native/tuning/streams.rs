//! Tuning cases of a decoder's hyper-connection mix sites
//! (`operators::streams`): `stream_mix_down` at the sublayers' sites (with
//! inject values) or the exit site (without), and `stream_mix_up` at every
//! site. The widening (`repeat_rows`) has no tuning parameters.

use super::{EntryTuning, ModelInputs, PointShape, TuningInputs, TuningLimits, row_points};
use crate::operators::streams::{MixDownBinding, MixUpBinding, StreamsGeometry};
use magnitude_family_contracts::{WeightKind, WeightScope};
use magnitude_kernels::{stream_mix_down, stream_mix_up};
use seismic::{BuildError, Element, Tensor};

/// The streams' geometry of the tuned model.
fn geometry(inputs: &ModelInputs<'_>) -> Result<StreamsGeometry, String> {
    StreamsGeometry::of(&inputs.definition.decoder)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "the decoder has a single residual stream".into())
}

/// The streams `[rows, S, H]` and a site's norm `[S, H]`.
fn streams_and_norm(
    inputs: &mut TuningInputs<'_, '_>,
    scope: WeightScope,
    geometry: StreamsGeometry,
    rows: u64,
    seed: u64,
) -> Result<(Tensor, Tensor), BuildError> {
    let (s, h) = (geometry.streams, geometry.hidden);
    let streams = inputs.activation(Element::f32(), &[rows, s, h], seed)?;
    let norm = inputs
        .weight(scope, WeightKind::HyperConnectionNorm)?
        .reshape(&[s, h])?;
    Ok((streams, norm))
}

/// `stream_mix_down` at the mix sites `scopes`: sublayers' sites, with
/// inject values, or exit sites, without.
#[derive(Clone)]
pub(crate) struct StreamMixDownTuning {
    pub binding: MixDownBinding,
    /// Every site prepared with this specialization.
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

pub(crate) struct StreamMixDownCase {
    hidden: Tensor,
    norm: Tensor,
    down: Tensor,
    inject: Tensor,
    epsilon: f32,
}

impl StreamMixDownTuning {
    fn elements(&self) -> stream_mix_down::Elements {
        stream_mix_down::Elements {
            NW: self.binding.norm,
            DW: self.binding.down,
            IW: self.binding.inject_element(),
            A: self.binding.activation,
        }
    }
}

impl EntryTuning for StreamMixDownTuning {
    type Entry = stream_mix_down::Entry;
    type Case = StreamMixDownCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        let elements = self.elements();
        format!(
            "NW={},DW={},IW={},A={},C={}",
            elements.NW.name(),
            elements.DW.name(),
            elements.IW.name(),
            elements.A.name(),
            u64::from(self.binding.injected())
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let geometry = geometry(inputs)?;
        Ok(vec![
            ("H", geometry.hidden),
            ("S", geometry.streams),
            ("LR", geometry.rank),
            ("C", u64::from(self.binding.injected())),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let geometry = geometry(inputs)?;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let (hidden, norm) =
                    streams_and_norm(inputs, scope, geometry, point.rows, index as u64 + 1)?;
                let down = inputs.weight(scope, WeightKind::HyperConnectionDown)?;
                let inject = if self.binding.injected() {
                    inputs
                        .weight(scope, WeightKind::HyperConnectionInject)?
                        .reshape(&[geometry.streams, geometry.width()])
                        .map_err(|error| error.to_string())?
                } else {
                    down.slice_leading(0, 0)
                        .map_err(|error| error.to_string())?
                };
                Ok(StreamMixDownCase {
                    hidden,
                    norm,
                    down,
                    inject,
                    epsilon: self.epsilon,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> stream_mix_down::Args<'a> {
        stream_mix_down::Args {
            hidden: &case.hidden,
            norm: &case.norm,
            down_weight: &case.down,
            inject_weight: &case.inject,
            epsilon: case.epsilon,
        }
    }

    generated_entry!(stream_mix_down, this => this.elements());
}

/// `stream_mix_up` at every mix site `scopes`.
#[derive(Clone)]
pub(crate) struct StreamMixUpTuning {
    pub binding: MixUpBinding,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct StreamMixUpCase {
    low: Tensor,
    up: Tensor,
    hidden: Tensor,
    inverse: Tensor,
    norm: Tensor,
}

impl StreamMixUpTuning {
    fn elements(&self) -> stream_mix_up::Elements {
        stream_mix_up::Elements {
            UW: self.binding.up,
            NW: self.binding.norm,
            A: self.binding.activation,
        }
    }
}

impl EntryTuning for StreamMixUpTuning {
    type Entry = stream_mix_up::Entry;
    type Case = StreamMixUpCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "UW={},NW={},A={}",
            self.binding.up.name(),
            self.binding.norm.name(),
            self.binding.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let geometry = geometry(inputs)?;
        Ok(vec![
            ("H", geometry.hidden),
            ("S", geometry.streams),
            ("LR", geometry.rank),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let geometry = geometry(inputs)?;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 3 * index as u64;
                let (hidden, norm) =
                    streams_and_norm(inputs, scope, geometry, point.rows, seed + 1)?;
                Ok(StreamMixUpCase {
                    low: inputs.activation(
                        self.binding.activation,
                        &[point.rows, geometry.rank],
                        seed + 2,
                    )?,
                    up: inputs.weight(scope, WeightKind::HyperConnectionUp)?,
                    hidden,
                    inverse: inputs.activation(
                        Element::f32(),
                        &[point.rows, geometry.streams],
                        seed + 3,
                    )?,
                    norm,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> stream_mix_up::Args<'a> {
        stream_mix_up::Args {
            low: &case.low,
            up_weight: &case.up,
            hidden: &case.hidden,
            inverse: &case.inverse,
            norm: &case.norm,
        }
    }

    generated_entry!(stream_mix_up, this => this.elements());
}
