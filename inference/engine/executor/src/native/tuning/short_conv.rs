//! Tuning cases of the short-convolution block (`operators::short_conv`):
//! the normed `u | C` projection (`short_conv_project`) and the output
//! projection (`attention_output` over one head of `channels`).
//! `short_conv_rows` declares no parameters.

use super::cases::projection_shape;
use super::{
    cpu_projection_screening, row_points, EntryTuning, PointShape, TuningInputs, TuningLimits,
};
use crate::ShortConvBinding;
use magnitude_family_contracts::{WeightKind, WeightScope};
use magnitude_kernels::{attention_output, short_conv_project};
use seismic::{Device, Element, ScreeningPoint, Tensor};

/// `short_conv_project`: the input RMS and the `B`, `C`, `X` projections.
pub(crate) struct ShortConvProjectTuning {
    pub binding: ShortConvBinding,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

pub(crate) struct ShortConvProjectCase {
    residual: Tensor,
    norm: Tensor,
    input_gate: Tensor,
    output_gate: Tensor,
    value: Tensor,
    epsilon: f32,
    absent_scale: Tensor,
}

impl ShortConvProjectTuning {
    fn elements(&self) -> short_conv_project::Elements {
        let b = self.binding;
        short_conv_project::Elements {
            NW: b.norm,
            BW: b.input_gate,
            CW: b.output_gate,
            XW: b.value,
            A: b.activation,
        }
    }
}

impl EntryTuning for ShortConvProjectTuning {
    type Entry = short_conv_project::Entry;
    type Case = ShortConvProjectCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        let b = self.binding;
        format!(
            "NW={},BW={},CW={},XW={},A={}",
            b.norm.name(),
            b.input_gate.name(),
            b.output_gate.name(),
            b.value.name(),
            b.activation.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.binding.shape;
        for kind in [
            WeightKind::ShortConvInputGate,
            WeightKind::ShortConvOutputGate,
            WeightKind::ShortConvValue,
        ] {
            if projection_shape(inputs, &self.scopes, kind)? != (shape.channels, shape.hidden) {
                return Err(format!(
                    "the short convolution {kind:?} projection disagrees with the binding"
                ));
            }
        }
        let [_, statics @ ..] = shape.project_dimensions(0);
        let mut statics = statics.to_vec();
        statics.extend([("BS", 0), ("CS", 0), ("XS", 0)]);
        Ok(statics)
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
        let hidden = self.binding.shape.hidden;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                Ok(ShortConvProjectCase {
                    residual: inputs.activation(
                        Element::f32(),
                        &[point.rows, hidden],
                        index as u64 + 1,
                    )?,
                    norm: inputs.weight(scope, WeightKind::InputNorm)?,
                    input_gate: inputs.weight(scope, WeightKind::ShortConvInputGate)?,
                    output_gate: inputs.weight(scope, WeightKind::ShortConvOutputGate)?,
                    value: inputs.weight(scope, WeightKind::ShortConvValue)?,
                    epsilon: self.epsilon,
                    absent_scale: inputs.activation(Element::f32(), &[0], 0)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> short_conv_project::Args<'a> {
        short_conv_project::Args {
            residual: &case.residual,
            norm: &case.norm,
            b_weight: &case.input_gate,
            c_weight: &case.output_gate,
            x_weight: &case.value,
            eps: case.epsilon,
            b_scale: &case.absent_scale,
            c_scale: &case.absent_scale,
            x_scale: &case.absent_scale,
        }
    }

    generated_entry!(short_conv_project, this => this.elements());
}

/// `attention_output` as the short-convolution output projection plus
/// residual.
pub(crate) struct ShortConvOutputTuning {
    pub binding: ShortConvBinding,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct ShortConvOutputCase {
    hidden: Tensor,
    gated: Tensor,
    output: Tensor,
}

impl EntryTuning for ShortConvOutputTuning {
    type Entry = attention_output::Entry;
    type Case = ShortConvOutputCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "OW={},A={}",
            self.binding.output.name(),
            self.binding.activation.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.binding.shape;
        if projection_shape(inputs, &self.scopes, WeightKind::RecurrentOutput)?
            != (shape.hidden, shape.channels)
        {
            return Err(
                "the short convolution output projection disagrees with the binding".into(),
            );
        }
        let [_, statics @ ..] = shape.output_dimensions(0);
        Ok(statics.to_vec())
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
        let shape = self.binding.shape;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 2 * index as u64;
                Ok(ShortConvOutputCase {
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, shape.hidden],
                        seed + 1,
                    )?,
                    gated: inputs.activation(
                        self.binding.activation,
                        &[point.rows, 1, shape.channels],
                        seed + 2,
                    )?,
                    output: inputs.weight(scope, WeightKind::RecurrentOutput)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> attention_output::Args<'a> {
        attention_output::Args {
            hidden: &case.hidden,
            gated: &case.gated,
            output_weight: &case.output,
        }
    }

    generated_entry!(attention_output, this => attention_output::Elements {
        OW: this.binding.output,
        A: this.binding.activation,
    });
}
