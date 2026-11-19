//! Tuning cases of the vision tower's projections: the patch stem
//! (`vision_patch_stem`) and the linears (`vision_linear`), each of one
//! kernel specialization of the projector's program over the matrices its
//! invocations project through, at the rows they take in an image of the
//! load's largest class. The epilogue operands (bias rows, the position
//! table, residual and gate rows, clamp bounds) are generated: they do not
//! bear on a configuration's timing or agreement.

use super::{EntryTuning, ModelInputs, PointShape, TuningInputs, TuningLimits, points_at};
use crate::native::vision::element;
use crate::operators::vision::{VisionGeometry, VisionProjection, VisionRows};
use crate::{CatalogFailure, VisionKernel};
use magnitude_family_contracts::WeightRole;
use magnitude_kernels::{vision_linear, vision_patch_stem};
use seismic::{BuildError, Element, Tensor};

/// The points of a case whose invocations take `rows`: each at those rows of
/// an image of the load's largest class. A load that encodes no images (a
/// listing of its programs alone) still prepares its vision program, whose
/// kernels it witnesses at the smallest image.
fn image_points(
    geometry: VisionGeometry,
    rows: impl IntoIterator<Item = VisionRows>,
    limits: TuningLimits,
) -> Vec<PointShape> {
    let cells = limits.image_cells.max(1);
    let mut rows = rows
        .into_iter()
        .map(|rows| geometry.rows(rows, cells))
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows.dedup();
    points_at(rows)
}

/// The static dimension `name` of `kernel`, which its plan fixes.
fn dimension(kernel: &VisionKernel, name: &str) -> Result<u64, String> {
    kernel
        .dimension(name)
        .ok_or_else(|| format!("the vision kernel fixes no dimension {name}"))
}

/// `vision_patch_stem` of one kernel specialization.
#[derive(Clone)]
pub(crate) struct VisionPatchStemTuning {
    kernel: VisionKernel,
    first: Element,
    next: Element,
    bias: Element,
    table: Element,
    geometry: VisionGeometry,
    /// The frames each invocation projects its patch rows through: the
    /// first frame's weight, and the further frame's where the stem has one.
    frames: Vec<(WeightRole, Option<WeightRole>)>,
}

pub(crate) struct VisionPatchStemCase {
    pixels: Tensor,
    frame: Tensor,
    next_frame: Tensor,
    bias: Tensor,
    table: Tensor,
    indices: Tensor,
    coefficients: Tensor,
}

/// Rows of a stem case's position table: the four corners each patch row
/// blends.
const TABLE_ROWS: u64 = 4;

impl VisionPatchStemTuning {
    /// The case of `kernel`, invoked as `projections` in a tower of
    /// `geometry` ([`crate::VisionProgramPlan`]).
    pub fn new(
        kernel: &VisionKernel,
        projections: &[VisionProjection],
        geometry: VisionGeometry,
    ) -> Result<Self, CatalogFailure> {
        let name = <vision_patch_stem::Entry as seismic::Entry>::NAME;
        Ok(Self {
            kernel: kernel.clone(),
            first: element(name, kernel, "W0")?,
            next: element(name, kernel, "W1")?,
            bias: element(name, kernel, "B")?,
            table: element(name, kernel, "PE")?,
            geometry,
            frames: projections
                .iter()
                .map(|projection| match (projection.rows, projection.weights.as_slice()) {
                    (VisionRows::Patches, [frame]) => (*frame, None),
                    (VisionRows::Patches, [frame, next]) => (*frame, Some(*next)),
                    other => panic!("a patch stem projects patch rows through one or two frames, not {other:?}"),
                })
                .collect(),
        })
    }

    fn elements(&self) -> vision_patch_stem::Elements {
        vision_patch_stem::Elements {
            W0: self.first,
            W1: self.next,
            B: self.bias,
            PE: self.table,
        }
    }
}

impl EntryTuning for VisionPatchStemTuning {
    type Entry = vision_patch_stem::Entry;
    type Case = VisionPatchStemCase;

    fn launches(&self) -> usize {
        self.frames.len()
    }

    fn bindings(&self) -> String {
        format!(
            "W0={},W1={},B={},PE={}",
            self.first.name(),
            self.next.name(),
            self.bias.name(),
            self.table.name()
        )
    }

    fn statics(&self, _inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(self.kernel.statics.clone())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        image_points(self.geometry, [VisionRows::Patches], limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let [c, s, p, h, nb, nt] =
            ["C", "S", "P", "H", "NB", "NT"].map(|name| dimension(&self.kernel, name));
        let (c, s, p, h, nb, nt) = (c?, s?, p?, h?, nb?, nt?);
        let m = point.rows;
        let corners =
            usize::try_from(nt * m * 4).map_err(|_| "vision stem corners exceed usize")?;
        TuningInputs::rotation_scopes(&self.frames, point)
            .into_iter()
            .enumerate()
            .map(|(index, (frame, next_frame))| {
                let frame = inputs
                    .weight(frame.scope, frame.kind)?
                    .reshape(&[h, c, p, p])?;
                let next_frame = match next_frame {
                    Some(next) => inputs
                        .weight(next.scope, next.kind)?
                        .reshape(&[s, h, c, p, p])?,
                    None => inputs.scratch(self.next, &[0, h, c, p, p])?,
                };
                let seed = 3 * index as u64;
                Ok(VisionPatchStemCase {
                    pixels: inputs.activation(Element::f32(), &[m, c, 1 + s, p, p], seed + 1)?,
                    frame,
                    next_frame,
                    bias: inputs.activation(self.bias, &[nb, h], seed + 2)?,
                    table: inputs.activation(self.table, &[nt * TABLE_ROWS, h], seed + 3)?,
                    indices: inputs.i32s(
                        &[nt, m, 4],
                        &(0..corners)
                            .map(|corner| (corner % 4) as i32)
                            .collect::<Vec<_>>(),
                    )?,
                    coefficients: inputs.f32s(&[nt, m, 4], &vec![0.25; corners])?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> vision_patch_stem::Args<'a> {
        vision_patch_stem::Args {
            pixels: &case.pixels,
            frame_weight: &case.frame,
            next_frame_weight: &case.next_frame,
            bias: &case.bias,
            table: &case.table,
            indices: &case.indices,
            coefficients: &case.coefficients,
        }
    }

    generated_entry!(vision_patch_stem, this => this.elements());
}

/// `vision_linear` of one kernel specialization.
#[derive(Clone)]
pub(crate) struct VisionLinearTuning {
    kernel: VisionKernel,
    activation: Element,
    weight: Element,
    bias: Element,
    output: Element,
    geometry: VisionGeometry,
    /// The rows each invocation projects and the weight it projects them
    /// through.
    invocations: Vec<(VisionRows, WeightRole)>,
}

pub(crate) struct VisionLinearCase {
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    residual: Tensor,
    gate: Tensor,
    minimum: Tensor,
    maximum: Tensor,
}

impl VisionLinearTuning {
    /// The case of `kernel`, invoked as `projections` in a tower of
    /// `geometry` ([`crate::VisionProgramPlan`]).
    pub fn new(
        kernel: &VisionKernel,
        projections: &[VisionProjection],
        geometry: VisionGeometry,
    ) -> Result<Self, CatalogFailure> {
        let name = <vision_linear::Entry as seismic::Entry>::NAME;
        let invocations = projections
            .iter()
            .map(|projection| match projection.weights.as_slice() {
                [weight] => (projection.rows, *weight),
                other => panic!("a vision linear projects through one weight, not {other:?}"),
            })
            .collect();
        Ok(Self {
            kernel: kernel.clone(),
            activation: element(name, kernel, "A")?,
            weight: element(name, kernel, "W")?,
            bias: element(name, kernel, "B")?,
            output: element(name, kernel, "Y")?,
            geometry,
            invocations,
        })
    }

    fn elements(&self) -> vision_linear::Elements {
        vision_linear::Elements {
            A: self.activation,
            W: self.weight,
            B: self.bias,
            Y: self.output,
        }
    }
}

impl EntryTuning for VisionLinearTuning {
    type Entry = vision_linear::Entry;
    type Case = VisionLinearCase;

    fn launches(&self) -> usize {
        self.invocations.len()
    }

    fn bindings(&self) -> String {
        format!(
            "A={},W={},B={},Y={}",
            self.activation.name(),
            self.weight.name(),
            self.bias.name(),
            self.output.name()
        )
    }

    fn statics(&self, _inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(self.kernel.statics.clone())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        image_points(
            self.geometry,
            self.invocations.iter().map(|(rows, _)| *rows),
            limits,
        )
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let [n, k, nb, nr, ng, nc] =
            ["N", "K", "NB", "NR", "NG", "NC"].map(|name| dimension(&self.kernel, name));
        let (n, k, nb, nr, ng, nc) = (n?, k?, nb?, nr?, ng?, nc?);
        let m = point.rows;
        let bounds = usize::try_from(nc).map_err(|_| "vision clamp bounds exceed usize")?;
        TuningInputs::rotation_scopes(&self.invocations, point)
            .into_iter()
            .enumerate()
            .map(|(index, (_, weight))| {
                let seed = 4 * index as u64;
                Ok(VisionLinearCase {
                    x: inputs.activation(self.activation, &[m, k], seed + 1)?,
                    weight: inputs.weight(weight.scope, weight.kind)?.reshape(&[n, k])?,
                    bias: inputs.activation(self.bias, &[nb, n], seed + 2)?,
                    residual: inputs.activation(Element::f32(), &[nr, m, n], seed + 3)?,
                    gate: inputs.activation(self.activation, &[ng, m, n], seed + 4)?,
                    // Bounds holding every value: the clamp runs and changes
                    // nothing.
                    minimum: inputs.f32s(&[nc], &vec![f32::MIN; bounds])?,
                    maximum: inputs.f32s(&[nc], &vec![f32::MAX; bounds])?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> vision_linear::Args<'a> {
        vision_linear::Args {
            x: &case.x,
            weight: &case.weight,
            bias: &case.bias,
            residual: &case.residual,
            gate: &case.gate,
            minimum: &case.minimum,
            maximum: &case.maximum,
            activation: 0,
        }
    }

    generated_entry!(vision_linear, this => this.elements());
}
