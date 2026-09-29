//! Tuning cases of the dense feed-forward entries, and what every projection
//! case shares. Each case derives its static values from resident weight
//! shapes and model geometry, and builds every argument set from real
//! resident weights of distinct layers plus case-owned activations.

use super::{
    cpu_projection_screening, row_points, EntryTuning, PointShape, TuningInputs, TuningLimits,
};
use magnitude_family_contracts::{WeightKind, WeightScope};
use magnitude_kernels::{dense_expand, dense_output, dense_up};
use seismic::{Device, Element, ScreeningPoint, Tensor};

/// The static dimensions `[rows, columns]` a projection weight fixes, checked
/// to agree across every layer that shares the specialization.
pub(crate) fn projection_shape(
    inputs: &TuningInputs<'_, '_>,
    scopes: &[WeightScope],
    kind: WeightKind,
) -> Result<(u64, u64), String> {
    let mut shape = None;
    for &scope in scopes {
        let [rows, columns] = inputs.weight_shape(scope, kind)?[..] else {
            return Err(format!("{kind:?} of {scope:?} is not a matrix"));
        };
        match shape {
            None => shape = Some((rows, columns)),
            Some(expected) if expected != (rows, columns) => {
                return Err(format!(
                    "layers sharing one specialization disagree on {kind:?}: {expected:?} vs {:?}",
                    (rows, columns)
                ))
            }
            Some(_) => {}
        }
    }
    shape.ok_or_else(|| "a tuning case needs at least one layer".to_owned())
}

/// The extent of a projection weight's accumulator-scale port, checked to
/// agree across every layer that shares the specialization.
pub(crate) fn scale_extent(
    inputs: &TuningInputs<'_, '_>,
    scopes: &[WeightScope],
    kind: WeightKind,
) -> Result<u64, String> {
    let mut extents = scopes
        .iter()
        .map(|&scope| inputs.scale_extent(scope, kind))
        .collect::<Result<Vec<_>, _>>()?;
    extents.dedup();
    match extents[..] {
        [extent] => Ok(extent),
        [] => Err("a tuning case needs at least one layer".into()),
        _ => Err(format!(
            "layers sharing one specialization disagree on {kind:?}'s scale: {extents:?}"
        )),
    }
}

/// `dense_expand`: RMS prologue, paired gate/up projection, act·mul, over
/// the `gate_kind`/`up_kind` weights (a dense feed-forward's or a routed
/// operator's shared expert).
pub(crate) struct DenseExpandTuning {
    pub norm: Element,
    pub gate: Element,
    pub up: Element,
    pub activation: Element,
    pub gate_kind: WeightKind,
    pub up_kind: WeightKind,
    /// The activation code (`functions.seismic`).
    pub function: i32,
    /// Every layer prepared with this specialization.
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

pub(crate) struct DenseExpandCase {
    residual: Tensor,
    out_rows: Tensor,
    norm: Tensor,
    gate: Tensor,
    up: Tensor,
    gate_scale: Tensor,
    up_scale: Tensor,
    epsilon: f32,
    function: i32,
}

impl DenseExpandTuning {
    fn elements(&self) -> dense_expand::Elements {
        dense_expand::Elements {
            NW: self.norm,
            GW: self.gate,
            UW: self.up,
            A: self.activation,
        }
    }
}

impl EntryTuning for DenseExpandTuning {
    type Entry = dense_expand::Entry;
    type Case = DenseExpandCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn weight_scopes(&self) -> &[WeightScope] {
        &self.scopes
    }

    fn bindings(&self) -> String {
        format!(
            "NW={},GW={},UW={},A={}",
            self.norm.name(),
            self.gate.name(),
            self.up.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (features, hidden) = projection_shape(inputs, &self.scopes, self.gate_kind)?;
        if projection_shape(inputs, &self.scopes, self.up_kind)? != (features, hidden) {
            return Err("dense gate and up shapes differ".into());
        }
        Ok(vec![
            ("H", hidden),
            ("F", features),
            ("GS", scale_extent(inputs, &self.scopes, self.gate_kind)?),
            ("US", scale_extent(inputs, &self.scopes, self.up_kind)?),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn screening(&self, device: &Device, points: &[PointShape]) -> Vec<ScreeningPoint> {
        cpu_projection_screening(device, points)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let gate = inputs.weight(scope, self.gate_kind)?;
                let hidden = gate.extents()[1];
                Ok(DenseExpandCase {
                    residual: inputs.activation(
                        Element::f32(),
                        &[point.rows, hidden],
                        index as u64 + 1,
                    )?,
                    out_rows: inputs.every_row(point.rows)?,
                    norm: inputs.weight(scope, WeightKind::InputNorm)?,
                    up: inputs.weight(scope, self.up_kind)?,
                    gate_scale: inputs.unit_scale(inputs.scale_extent(scope, self.gate_kind)?)?,
                    up_scale: inputs.unit_scale(inputs.scale_extent(scope, self.up_kind)?)?,
                    gate,
                    epsilon: self.epsilon,
                    function: self.function,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> dense_expand::Args<'a> {
        dense_expand::Args {
            residual: &case.residual,
            norm: &case.norm,
            gate_weight: &case.gate,
            up_weight: &case.up,
            out_rows: &case.out_rows,
            eps: case.epsilon,
            activation: case.function,
            gate_scale: &case.gate_scale,
            up_scale: &case.up_scale,
        }
    }

    generated_entry!(dense_expand, this => this.elements());
}

/// `dense_up`: RMS prologue, up-only projection, act (a routed operator's
/// up-only shared expert, or an up-only dense feed-forward).
pub(crate) struct DenseUpTuning {
    pub norm: Element,
    pub up: Element,
    pub activation: Element,
    pub up_kind: WeightKind,
    /// The activation code (`functions.seismic`).
    pub function: i32,
    /// Every layer prepared with this specialization.
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

pub(crate) struct DenseUpCase {
    residual: Tensor,
    out_rows: Tensor,
    norm: Tensor,
    up: Tensor,
    up_scale: Tensor,
    epsilon: f32,
    function: i32,
}

impl EntryTuning for DenseUpTuning {
    type Entry = dense_up::Entry;
    type Case = DenseUpCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn weight_scopes(&self) -> &[WeightScope] {
        &self.scopes
    }

    fn bindings(&self) -> String {
        format!(
            "NW={},UW={},A={}",
            self.norm.name(),
            self.up.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (features, hidden) = projection_shape(inputs, &self.scopes, self.up_kind)?;
        Ok(vec![
            ("H", hidden),
            ("F", features),
            ("US", scale_extent(inputs, &self.scopes, self.up_kind)?),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn screening(&self, device: &Device, points: &[PointShape]) -> Vec<ScreeningPoint> {
        cpu_projection_screening(device, points)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let up = inputs.weight(scope, self.up_kind)?;
                let hidden = up.extents()[1];
                Ok(DenseUpCase {
                    residual: inputs.activation(
                        Element::f32(),
                        &[point.rows, hidden],
                        index as u64 + 1,
                    )?,
                    out_rows: inputs.every_row(point.rows)?,
                    norm: inputs.weight(scope, WeightKind::InputNorm)?,
                    up,
                    up_scale: inputs.unit_scale(inputs.scale_extent(scope, self.up_kind)?)?,
                    epsilon: self.epsilon,
                    function: self.function,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> dense_up::Args<'a> {
        dense_up::Args {
            residual: &case.residual,
            norm: &case.norm,
            up_weight: &case.up,
            out_rows: &case.out_rows,
            eps: case.epsilon,
            activation: case.function,
            up_scale: &case.up_scale,
        }
    }

    generated_entry!(dense_up, this => dense_up::Elements {
        NW: this.norm,
        UW: this.up,
        A: this.activation,
    });
}

/// `dense_output`: the `down_kind` projection plus residual (a dense or
/// shared expert's down projection, or a latent operator's up projection).
pub(crate) struct DenseOutputTuning {
    pub down: Element,
    pub activation: Element,
    pub down_kind: WeightKind,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct DenseOutputCase {
    residual: Tensor,
    product: Tensor,
    down: Tensor,
    out_rows: Tensor,
    down_scale: Tensor,
}

impl DenseOutputTuning {
    fn elements(&self) -> dense_output::Elements {
        dense_output::Elements {
            DW: self.down,
            A: self.activation,
        }
    }
}

impl EntryTuning for DenseOutputTuning {
    type Entry = dense_output::Entry;
    type Case = DenseOutputCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn weight_scopes(&self) -> &[WeightScope] {
        &self.scopes
    }

    fn bindings(&self) -> String {
        format!("DW={},A={}", self.down.name(), self.activation.name())
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (hidden, features) = projection_shape(inputs, &self.scopes, self.down_kind)?;
        Ok(vec![
            ("H", hidden),
            ("F", features),
            ("DS", scale_extent(inputs, &self.scopes, self.down_kind)?),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn screening(&self, device: &Device, points: &[PointShape]) -> Vec<ScreeningPoint> {
        cpu_projection_screening(device, points)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let down = inputs.weight(scope, self.down_kind)?;
                let (hidden, features) = (down.extents()[0], down.extents()[1]);
                Ok(DenseOutputCase {
                    residual: inputs.activation(
                        Element::f32(),
                        &[point.rows, hidden],
                        2 * index as u64 + 1,
                    )?,
                    product: inputs.activation(
                        self.activation,
                        &[point.rows, features],
                        2 * index as u64 + 2,
                    )?,
                    out_rows: inputs.every_row(point.rows)?,
                    down_scale: inputs.unit_scale(inputs.scale_extent(scope, self.down_kind)?)?,
                    down,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> dense_output::Args<'a> {
        dense_output::Args {
            residual: &case.residual,
            product: &case.product,
            down_weight: &case.down,
            out_rows: &case.out_rows,
            down_scale: &case.down_scale,
        }
    }

    generated_entry!(dense_output, this => this.elements());
}
