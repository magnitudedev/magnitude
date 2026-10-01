//! Tuning case of the per-layer input gate (`per_layer_gate`, Gemma PLE): the
//! gate projection of the block's residual rows, its activation, and the
//! product with the layer's slice of the per-layer inputs.

use super::cases::projection_shape;
use super::{row_points, EntryTuning, PointShape, TuningInputs, TuningLimits};
use crate::PerLayerBinding;
use magnitude_family_contracts::{Operator, WeightKind, WeightScope};
use magnitude_kernels::per_layer_gate;
use seismic::{Element, Tensor};

/// `per_layer_gate` of the per-layer sublayers of one binding.
pub(crate) struct PerLayerGateTuning {
    pub binding: PerLayerBinding,
    /// Every layer prepared with this specialization.
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct PerLayerGateCase {
    hidden: Tensor,
    gate: Tensor,
    inputs: Tensor,
    layer: i32,
    activation: i32,
    absent_scale: Tensor,
}

impl PerLayerGateTuning {
    fn elements(&self) -> per_layer_gate::Elements {
        per_layer_gate::Elements {
            GW: self.binding.gate,
            A: self.binding.activation,
        }
    }
}

impl EntryTuning for PerLayerGateTuning {
    type Entry = per_layer_gate::Entry;
    type Case = PerLayerGateCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "GW={},A={}",
            self.binding.gate.name(),
            self.binding.activation.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let b = self.binding;
        if projection_shape(inputs, &self.scopes, WeightKind::PerLayerGate)? != (b.width, b.hidden)
        {
            return Err("the per-layer gate disagrees with the binding".into());
        }
        Ok(vec![
            ("D", b.hidden),
            ("L", b.layers),
            ("P", b.width),
            ("GS", 0),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let b = self.binding;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let Operator::PerLayerInput(op) = inputs.operator(&[scope])? else {
                    return Err("a per-layer gate case binds a per-layer input sublayer".into());
                };
                let layer = i32::try_from(op.layer).map_err(|_| "per-layer index exceeds i32")?;
                let activation = crate::operators::dense_ffn::activation_code(op.activation);
                let seed = 2 * index as u64;
                Ok(PerLayerGateCase {
                    hidden: inputs.activation(Element::f32(), &[point.rows, b.hidden], seed + 1)?,
                    inputs: inputs.activation(
                        Element::f32(),
                        &[point.rows, b.layers, b.width],
                        seed + 2,
                    )?,
                    gate: inputs.weight(scope, WeightKind::PerLayerGate)?,
                    layer,
                    activation,
                    absent_scale: inputs.activation(Element::f32(), &[0], 0)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> per_layer_gate::Args<'a> {
        per_layer_gate::Args {
            hidden: &case.hidden,
            gate_weight: &case.gate,
            inputs: &case.inputs,
            layer: case.layer,
            activation: case.activation,
            gate_scale: &case.absent_scale,
        }
    }

    generated_entry!(per_layer_gate, this => this.elements());
}
