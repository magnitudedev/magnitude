//! Tuning case of a post-norm sublayer's output projection (`project_rows`
//! publishing F32 for the `post_norm_residual` row op). The row op itself has
//! no tuning parameters.

use super::cases::{projection_shape, scale_extent};
use super::{
    cpu_projection_screening, row_points, EntryTuning, PointShape, TuningInputs, TuningLimits,
};
use magnitude_family_contracts::{WeightKind, WeightScope};
use magnitude_kernels::project_rows;
use seismic::{Device, Element, ScreeningPoint, Tensor};

/// `project_rows` of one sublayer output projection (`kind`: the attention
/// output or the dense down projection), activation rows in, F32 rows out.
pub(crate) struct ProjectRowsTuning {
    pub weight: Element,
    pub activation: Element,
    pub kind: WeightKind,
    /// The published element: F32 for a post-norm row op, activations for a
    /// latent down projection.
    pub output: Element,
    /// Every layer prepared with this specialization.
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct ProjectRowsCase {
    source: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
}

impl ProjectRowsTuning {
    fn elements(&self) -> project_rows::Elements {
        project_rows::Elements {
            A: self.activation,
            W: self.weight,
            Y: self.output,
        }
    }
}

impl EntryTuning for ProjectRowsTuning {
    type Entry = project_rows::Entry;
    type Case = ProjectRowsCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "A={},W={},Y={}",
            self.activation.name(),
            self.weight.name(),
            self.output.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (outputs, inputs_width) = projection_shape(inputs, &self.scopes, self.kind)?;
        Ok(vec![
            ("K", inputs_width),
            ("N", outputs),
            ("WS", scale_extent(inputs, &self.scopes, self.kind)?),
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
                let weight = inputs.weight(scope, self.kind)?;
                let width = weight.extents()[1];
                Ok(ProjectRowsCase {
                    source: inputs.activation(
                        self.activation,
                        &[point.rows, width],
                        index as u64 + 1,
                    )?,
                    weight,
                    weight_scale: inputs.unit_scale(inputs.scale_extent(scope, self.kind)?)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> project_rows::Args<'a> {
        project_rows::Args {
            source: &case.source,
            weight: &case.weight,
            weight_scale: &case.weight_scale,
        }
    }

    generated_entry!(project_rows, this => this.elements());
}
