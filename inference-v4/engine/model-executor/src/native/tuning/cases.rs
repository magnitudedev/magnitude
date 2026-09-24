//! Tuning cases of the dense feed-forward entries, and what every projection
//! case shares. Each case derives its static values from resident weight
//! shapes and model geometry, and builds every argument set from real
//! resident weights of distinct layers plus case-owned activations.

use super::{row_points, EntryTuning, PointShape, TuningInputs, TuningLimits};
use magnitude_model_contracts::{WeightKind, WeightScope};
use magnitude_model_kernels::{qwen_dense_expand, qwen_dense_output};
use seismic::{Element, Tensor};

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

/// `qwen_dense_expand`: RMS prologue, paired gate/up projection, SiLU·mul.
pub(crate) struct DenseExpandTuning {
    pub norm: Element,
    pub gate: Element,
    pub up: Element,
    pub activation: Element,
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
    epsilon: f32,
}

impl DenseExpandTuning {
    fn elements(&self) -> qwen_dense_expand::Elements {
        qwen_dense_expand::Elements {
            NW: self.norm,
            GW: self.gate,
            UW: self.up,
            A: self.activation,
        }
    }
}

impl EntryTuning for DenseExpandTuning {
    type Entry = qwen_dense_expand::Entry;
    type Case = DenseExpandCase;

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
        let (features, hidden) = projection_shape(inputs, &self.scopes, WeightKind::DenseGate)?;
        if projection_shape(inputs, &self.scopes, WeightKind::DenseUp)? != (features, hidden) {
            return Err("dense gate and up shapes differ".into());
        }
        Ok(vec![("H", hidden), ("F", features)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
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
                let gate = inputs.weight(scope, WeightKind::DenseGate)?;
                let hidden = gate.extents()[1];
                Ok(DenseExpandCase {
                    residual: inputs.activation(
                        Element::f32(),
                        &[point.rows, hidden],
                        index as u64 + 1,
                    )?,
                    out_rows: inputs.every_row(point.rows)?,
                    norm: inputs.weight(scope, WeightKind::FeedForwardNorm)?,
                    up: inputs.weight(scope, WeightKind::DenseUp)?,
                    gate,
                    epsilon: self.epsilon,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> qwen_dense_expand::Args<'a> {
        qwen_dense_expand::Args {
            residual: &case.residual,
            norm: &case.norm,
            gate_weight: &case.gate,
            up_weight: &case.up,
            out_rows: &case.out_rows,
            eps: case.epsilon,
        }
    }

    generated_entry!(qwen_dense_expand, this => this.elements());
}

/// `qwen_dense_output`: down projection plus residual.
pub(crate) struct DenseOutputTuning {
    pub down: Element,
    pub activation: Element,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct DenseOutputCase {
    residual: Tensor,
    product: Tensor,
    down: Tensor,
    out_rows: Tensor,
}

impl DenseOutputTuning {
    fn elements(&self) -> qwen_dense_output::Elements {
        qwen_dense_output::Elements {
            DW: self.down,
            A: self.activation,
        }
    }
}

impl EntryTuning for DenseOutputTuning {
    type Entry = qwen_dense_output::Entry;
    type Case = DenseOutputCase;

    fn bindings(&self) -> String {
        format!("DW={},A={}", self.down.name(), self.activation.name())
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (hidden, features) = projection_shape(inputs, &self.scopes, WeightKind::DenseDown)?;
        Ok(vec![("H", hidden), ("F", features)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
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
                let down = inputs.weight(scope, WeightKind::DenseDown)?;
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
                    down,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> qwen_dense_output::Args<'a> {
        qwen_dense_output::Args {
            residual: &case.residual,
            product: &case.product,
            down_weight: &case.down,
            out_rows: &case.out_rows,
        }
    }

    generated_entry!(qwen_dense_output, this => this.elements());
}
