//! The vision tower as a program of vision operators (model-family plan
//! §3.11).
//!
//! A projector description becomes one ordered list of operator invocations,
//! each naming its kernel specialization (entry, element bindings, static
//! dimensions), its weights by role and its operands by the invocations that
//! produce them. Planning admits the projector and derives this program once;
//! preparation specializes each distinct kernel of it; the graph builder
//! enqueues it. Forms the vision entries do not run are refused here with a
//! typed plan error.

use crate::error::PlanError;
use magnitude_family_contracts::{
    CellReduction, LinearPart, NormPart, VisionActivation, VisionAttentionScale,
    VisionAttentionSpan, VisionBlock, VisionDescription, VisionLinear, VisionLinearSite,
    VisionNorm, VisionNormSite, VisionStem, VisionUp, VisionWeight, WeightKind, WeightRole,
    WeightScope,
};
use seismic::Element;

/// The entries a vision program invokes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VisionEntry {
    PatchStem,
    Norm,
    Linear,
    Clamp,
    Attention,
    Pool,
    Position,
    /// The decoder's sandwich-norm tail (`post_norm_residual`).
    PostNormResidual,
}

/// One specialization of a vision entry: its element bindings and the static
/// dimensions a backend may specialize on.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VisionKernel {
    pub entry: VisionEntry,
    pub elements: Vec<(&'static str, Element)>,
    pub statics: Vec<(&'static str, u64)>,
}

/// How many rows an operand has: one per patch, or one per merged cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionRows {
    Patches,
    Cells,
}

/// A value of the tower: a graph input, or the result of an earlier
/// invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionValue {
    /// The prepared patch rows, `[M, frames · C · P · P]` F32.
    Pixels,
    Result(usize),
}

/// The operands of one invocation. Absent optional weights bind an empty
/// F32 constant of the argument's shape.
#[derive(Clone, Debug, PartialEq)]
pub enum VisionOperands {
    PatchStem {
        frame: WeightRole,
        next_frame: Option<WeightRole>,
        bias: Option<WeightRole>,
        table: WeightRole,
    },
    Norm {
        source: VisionValue,
        /// The rows of the source (and of the result's cells).
        rows: VisionRows,
        weight: Option<WeightRole>,
        bias: Option<WeightRole>,
        /// The result gathers the source rows through the merge order input.
        gathered: bool,
        epsilon: f32,
        centered: bool,
        interleave: bool,
    },
    Linear {
        x: VisionValue,
        rows: VisionRows,
        weight: WeightRole,
        bias: Option<WeightRole>,
        residual: Option<VisionValue>,
        gate: Option<VisionValue>,
        /// Output clamp bounds.
        bounds: Option<(WeightRole, WeightRole)>,
        activation: i32,
    },
    Clamp {
        x: VisionValue,
        rows: VisionRows,
        minimum: WeightRole,
        maximum: WeightRole,
    },
    Attention {
        query: VisionValue,
        key: VisionValue,
        value: VisionValue,
        query_norm: Option<WeightRole>,
        key_norm: Option<WeightRole>,
        value_norm: bool,
        windowed: bool,
        log_base: f32,
        epsilon: f32,
        unit_scale: bool,
    },
    Pool {
        source: VisionValue,
        standard: Option<(WeightRole, WeightRole)>,
        weight: f32,
        scale: f32,
    },
    Position {
        source: VisionValue,
        table: WeightRole,
    },
    PostNormResidual {
        residual: VisionValue,
        projected: VisionValue,
        norm: WeightRole,
        epsilon: f32,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionOp {
    pub kernel: VisionKernel,
    /// Width of the result's rows (per patch or per cell).
    pub width: u64,
    pub operands: VisionOperands,
}

/// A projector's tower in operator order. The last invocation's result is
/// the image features.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionProgram {
    pub ops: Vec<VisionOp>,
    /// Patch rows per merged cell.
    pub cell_rows: u64,
    /// The width of one prepared patch row.
    pub pixel_width: u64,
    /// Whether the graph reads rotary coordinates, window spans and the
    /// merge order.
    pub coordinates: bool,
    pub windows: bool,
}

impl VisionProgram {
    /// The distinct kernels of the program, in first-use order.
    pub fn kernels(&self) -> Vec<&VisionKernel> {
        let mut kernels: Vec<&VisionKernel> = Vec::new();
        for op in &self.ops {
            if !kernels.contains(&&op.kernel) {
                kernels.push(&op.kernel);
            }
        }
        kernels
    }
}

fn unsupported(what: &'static str) -> PlanError {
    PlanError::Unsupported(what)
}

fn activation_code(activation: VisionActivation) -> i32 {
    match activation {
        VisionActivation::GeluTanh => 1,
        VisionActivation::GeluErf => 2,
        VisionActivation::GeluQuick => 3,
    }
}

fn role(scope: WeightScope, weight: VisionWeight) -> WeightRole {
    WeightRole {
        scope,
        kind: WeightKind::Vision(weight),
    }
}

fn block_scope(index: usize) -> Result<WeightScope, PlanError> {
    Ok(WeightScope::VisionBlock(
        u32::try_from(index).map_err(|_| PlanError::Arithmetic("vision block index exceeds u32"))?,
    ))
}

struct Builder<'a> {
    element: &'a dyn Fn(WeightRole) -> Result<Element, PlanError>,
    activation: Element,
    ops: Vec<VisionOp>,
}

impl Builder<'_> {
    fn push(&mut self, entry: VisionEntry, elements: Vec<(&'static str, Element)>,
        statics: Vec<(&'static str, u64)>, width: u64, operands: VisionOperands) -> VisionValue {
        self.ops.push(VisionOp {
            kernel: VisionKernel {
                entry,
                elements,
                statics,
            },
            width,
            operands,
        });
        VisionValue::Result(self.ops.len() - 1)
    }

    fn element(&self, role: Option<WeightRole>) -> Result<Element, PlanError> {
        role.map_or(Ok(Element::f32()), |role| (self.element)(role))
    }

    /// A norm of `source` (`cells` rows of `members` rows of `width`),
    /// published to `output`.
    #[allow(clippy::too_many_arguments)]
    fn norm(&mut self, source: VisionValue, rows: VisionRows, members: u64, width: u64,
        norm: &VisionNorm, scope: WeightScope, site: VisionNormSite, output: Element,
        gathered: bool, interleave: bool) -> Result<VisionValue, PlanError> {
        let part = |part| role(scope, VisionWeight::Norm { site, part });
        let (weight, bias, centered) = match norm {
            VisionNorm::Layer { .. } => (Some(part(NormPart::Weight)), Some(part(NormPart::Bias)), true),
            VisionNorm::Rms { weight, .. } => (weight.as_ref().map(|_| part(NormPart::Weight)), None, false),
        };
        let elements = vec![
            ("NWE", self.element(weight)?),
            ("NBE", self.element(bias)?),
            ("Y", output),
        ];
        let statics = vec![
            ("G", members),
            ("H", width),
            ("NW", u64::from(weight.is_some())),
            ("NB", u64::from(bias.is_some())),
            ("NO", u64::from(gathered)),
        ];
        Ok(self.push(VisionEntry::Norm, elements, statics, members * width,
            VisionOperands::Norm {
                source,
                rows,
                weight,
                bias,
                gathered,
                epsilon: norm.epsilon() as f32,
                centered,
                interleave,
            }))
    }

    /// `linear` over `x` (`inputs` wide), its input clamped first when it
    /// is clippable.
    #[allow(clippy::too_many_arguments)]
    fn linear(&mut self, x: VisionValue, rows: VisionRows, inputs: u64, linear: &VisionLinear,
        scope: WeightScope, site: VisionLinearSite, output: Element, residual: Option<VisionValue>,
        gate: Option<VisionValue>, activation: i32) -> Result<VisionValue, PlanError> {
        let part = |part| role(scope, VisionWeight::Linear { site, part });
        let x = match &linear.clamp {
            None => x,
            Some(_) => {
                let (minimum, maximum) = (part(LinearPart::InputMinimum), part(LinearPart::InputMaximum));
                let elements = vec![("A", self.activation)];
                self.push(VisionEntry::Clamp, elements, Vec::new(), inputs,
                    VisionOperands::Clamp { x, rows, minimum, maximum })
            }
        };
        let weight = part(LinearPart::Weight);
        let bias = linear.bias.as_ref().map(|_| part(LinearPart::Bias));
        let bounds = linear
            .clamp
            .as_ref()
            .map(|_| (part(LinearPart::OutputMinimum), part(LinearPart::OutputMaximum)));
        let outputs = linear.outputs();
        let elements = vec![
            ("A", self.activation),
            ("W", self.element(Some(weight))?),
            ("B", self.element(bias)?),
            ("Y", output),
        ];
        let statics = vec![
            ("N", outputs),
            ("K", inputs),
            ("NB", u64::from(bias.is_some())),
            ("NR", u64::from(residual.is_some())),
            ("NG", u64::from(gate.is_some())),
            ("NC", u64::from(bounds.is_some())),
        ];
        Ok(self.push(VisionEntry::Linear, elements, statics, outputs,
            VisionOperands::Linear { x, rows, weight, bias, residual, gate, bounds, activation }))
    }

    /// `x + post_norm(projected)`: a sandwich post-norm (a weighted RMS).
    fn post_norm(&mut self, residual: VisionValue, projected: VisionValue, width: u64,
        norm: &VisionNorm, scope: WeightScope, site: VisionNormSite) -> Result<VisionValue, PlanError> {
        let VisionNorm::Rms { weight: Some(_), epsilon } = norm else {
            return Err(unsupported("vision post-norm other than a weighted RMS"));
        };
        let norm_role = role(scope, VisionWeight::Norm { site, part: NormPart::Weight });
        let elements = vec![("NW", self.element(Some(norm_role))?)];
        Ok(self.push(VisionEntry::PostNormResidual, elements, Vec::new(), width,
            VisionOperands::PostNormResidual {
                residual,
                projected,
                norm: norm_role,
                epsilon: *epsilon as f32,
            }))
    }

    fn block(&mut self, h: VisionValue, hidden: u64, index: usize, block: &VisionBlock)
        -> Result<VisionValue, PlanError> {
        let scope = block_scope(index)?;
        let (a, f32e) = (self.activation, Element::f32());
        let patches = VisionRows::Patches;
        let attention = &block.attention;
        let normalized = self.norm(h, patches, 1, hidden, &block.attention_norm, scope,
            VisionNormSite::AttentionInput, a, false, false)?;
        let inner = attention.heads * attention.width;
        let query = self.linear(normalized, patches, hidden, &attention.query, scope,
            VisionLinearSite::Query, a, None, None, 0)?;
        let key = self.linear(normalized, patches, hidden, &attention.key, scope,
            VisionLinearSite::Key, a, None, None, 0)?;
        let value = self.linear(normalized, patches, hidden, &attention.value, scope,
            VisionLinearSite::Value, a, None, None, 0)?;
        let epsilons = [
            attention.query_norm.as_ref().map(|norm| norm.epsilon),
            attention.key_norm.as_ref().map(|norm| norm.epsilon),
            attention.value_norm.as_ref().map(|norm| norm.epsilon),
        ];
        let mut present = epsilons.iter().flatten();
        let epsilon = present.next().copied();
        if present.any(|other| Some(*other) != epsilon) {
            return Err(unsupported("vision head norms with different epsilons"));
        }
        let query_norm = attention.query_norm.as_ref().map(|_| role(scope, VisionWeight::QueryNorm));
        let key_norm = attention.key_norm.as_ref().map(|_| role(scope, VisionWeight::KeyNorm));
        let windowed = attention.span == VisionAttentionSpan::Window;
        let elements = vec![("A", a)];
        let statics = vec![
            ("H", attention.heads),
            ("P", attention.width / 4),
            ("NQ", u64::from(query_norm.is_some())),
            ("NV", u64::from(attention.value_norm.is_some())),
            ("WS", u64::from(windowed)),
        ];
        let attended = self.push(VisionEntry::Attention, elements, statics, inner,
            VisionOperands::Attention {
                query,
                key,
                value,
                query_norm,
                key_norm,
                value_norm: attention.value_norm.is_some(),
                windowed,
                log_base: attention.rotary_base.ln() as f32,
                // Unread without head norms.
                epsilon: epsilon.unwrap_or(0.0) as f32,
                unit_scale: attention.scale == VisionAttentionScale::Unit,
            });
        let h = match &block.attention_post_norm {
            None => self.linear(attended, patches, inner, &attention.output, scope,
                VisionLinearSite::AttentionOutput, f32e, Some(h), None, 0)?,
            Some(norm) => {
                let projected = self.linear(attended, patches, inner, &attention.output, scope,
                    VisionLinearSite::AttentionOutput, f32e, None, None, 0)?;
                self.post_norm(h, projected, hidden, norm, scope, VisionNormSite::AttentionOutput)?
            }
        };
        let feedforward = &block.feedforward;
        let intermediate = feedforward.intermediate();
        let normalized = self.norm(h, patches, 1, hidden, &block.feedforward_norm, scope,
            VisionNormSite::FeedForwardInput, a, false, false)?;
        let code = activation_code(feedforward.activation);
        let up = match &feedforward.up {
            VisionUp::Plain(up) => self.linear(normalized, patches, hidden, up, scope,
                VisionLinearSite::Up, a, None, None, code)?,
            VisionUp::Gated { gate, up } => {
                let gate = self.linear(normalized, patches, hidden, gate, scope,
                    VisionLinearSite::Gate, a, None, None, code)?;
                self.linear(normalized, patches, hidden, up, scope, VisionLinearSite::Up, a,
                    None, Some(gate), 0)?
            }
        };
        Ok(match &block.feedforward_post_norm {
            None => self.linear(up, patches, intermediate, &feedforward.down, scope,
                VisionLinearSite::Down, f32e, Some(h), None, 0)?,
            Some(norm) => {
                let projected = self.linear(up, patches, intermediate, &feedforward.down, scope,
                    VisionLinearSite::Down, f32e, None, None, 0)?;
                self.post_norm(h, projected, hidden, norm, scope, VisionNormSite::FeedForwardOutput)?
            }
        })
    }
}

/// The program of `description`, with the resident element of every
/// weight role from `element`.
pub(crate) fn vision_program(
    description: &VisionDescription,
    element: &dyn Fn(WeightRole) -> Result<Element, PlanError>,
) -> Result<VisionProgram, PlanError> {
    let activation = Element::dense(crate::planning::activation_dtype(description.activation_dtype));
    let mut b = Builder {
        element,
        activation,
        ops: Vec::new(),
    };
    let f32e = Element::f32();
    let (hidden, vision) = (description.hidden, WeightScope::Vision);
    let preprocessing = &description.preprocessing;
    let pixel_width = description
        .patch_row_width()
        .map_err(|_| PlanError::Arithmetic("vision patch row width overflows"))?;
    let table = role(vision, VisionWeight::Position);
    let mut h = match &description.stem {
        VisionStem::Patch { frames, bias, positions: _, norm } => {
            let frame = role(WeightScope::VisionPatch(0), VisionWeight::PatchFrame);
            let next_frame = match frames.len() {
                1 => None,
                2 => Some(role(WeightScope::VisionPatch(1), VisionWeight::PatchFrame)),
                _ => return Err(unsupported("vision patch stem frame count")),
            };
            let bias = bias.as_ref().map(|_| role(vision, VisionWeight::PatchBias));
            let elements = vec![
                ("W0", b.element(Some(frame))?),
                ("W1", b.element(next_frame)?),
                ("B", b.element(bias)?),
                ("PE", b.element(Some(table))?),
            ];
            let statics = vec![
                ("C", preprocessing.channels),
                ("S", u64::from(next_frame.is_some())),
                ("P", preprocessing.patch),
                ("H", hidden),
                ("NB", u64::from(bias.is_some())),
            ];
            let h = b.push(VisionEntry::PatchStem, elements, statics, hidden,
                VisionOperands::PatchStem { frame, next_frame, bias, table });
            match norm {
                None => h,
                Some(norm) => b.norm(h, VisionRows::Patches, 1, hidden, norm, vision,
                    VisionNormSite::Stem, f32e, false, false)?,
            }
        }
        VisionStem::NormalizedPatch { input_norm, projection, projection_norm, positions: _, position_norm } => {
            let normalized = b.norm(VisionValue::Pixels, VisionRows::Patches, 1, pixel_width,
                input_norm, vision, VisionNormSite::Stem, activation, false, false)?;
            let projected = b.linear(normalized, VisionRows::Patches, pixel_width, projection,
                vision, VisionLinearSite::Patch, f32e, None, None, 0)?;
            let projected = b.norm(projected, VisionRows::Patches, 1, hidden, projection_norm,
                vision, VisionNormSite::StemProjection, f32e, false, false)?;
            let elements = vec![("PE", b.element(Some(table))?)];
            let positioned = b.push(VisionEntry::Position, elements, vec![("H", hidden)], hidden,
                VisionOperands::Position { source: projected, table });
            b.norm(positioned, VisionRows::Patches, 1, hidden, position_norm, vision,
                VisionNormSite::StemPosition, f32e, false, false)?
        }
    };
    for (index, block) in description.blocks.iter().enumerate() {
        h = b.block(h, hidden, index, block)?;
    }
    let merger = &description.merger;
    let cell_rows = description.cell_rows();
    let windows = description.window.is_some();
    if merger.standardize.is_some() && !matches!(merger.reduction, CellReduction::Average { .. }) {
        return Err(unsupported("vision standardization without a cell average"));
    }
    let mut x = match merger.reduction {
        CellReduction::Concatenate | CellReduction::Interleave => {
            let interleave = merger.reduction == CellReduction::Interleave;
            let (norm, site) = match (&merger.norm, &merger.projection_norm) {
                (Some(norm), None) => (norm, VisionNormSite::Merger),
                (None, Some(norm)) if cell_rows == 1 => (norm, VisionNormSite::MergerProjection),
                _ => return Err(unsupported("vision merger norms around a cell concatenation")),
            };
            b.norm(h, VisionRows::Cells, cell_rows, hidden, norm, vision, site, activation,
                windows, interleave)?
        }
        CellReduction::Average { scale } => {
            if merger.norm.is_some() || windows {
                return Err(unsupported("vision merger norm or windows before a cell average"));
            }
            let Some(norm) = &merger.projection_norm else {
                return Err(unsupported("vision cell average without a projection norm"));
            };
            let standard = merger.standardize.as_ref().map(|_| {
                (role(vision, VisionWeight::StandardizeBias), role(vision, VisionWeight::StandardizeScale))
            });
            let elements = vec![
                ("SB", b.element(standard.map(|(bias, _)| bias))?),
                ("SS", b.element(standard.map(|(_, scale)| scale))?),
            ];
            let statics = vec![("G", cell_rows), ("H", hidden), ("NS", u64::from(standard.is_some()))];
            let pooled = b.push(VisionEntry::Pool, elements, statics, hidden,
                VisionOperands::Pool {
                    source: h,
                    standard,
                    weight: (1.0 / cell_rows as f64) as f32,
                    scale: scale as f32,
                });
            b.norm(pooled, VisionRows::Cells, 1, hidden, norm, vision,
                VisionNormSite::MergerProjection, activation, false, false)?
        }
    };
    let mut width = b.ops[b.ops.len() - 1].width;
    let last = merger.stages.len() - 1;
    for (stage_index, stage) in merger.stages.iter().enumerate() {
        let scope = WeightScope::VisionMerger(
            u32::try_from(stage_index).map_err(|_| PlanError::Arithmetic("vision merger stage index"))?,
        );
        let output = if stage_index == last { f32e } else { activation };
        if (stage_index == last) == stage.activation.is_some() {
            return Err(unsupported("vision merger activation placement"));
        }
        x = b.linear(x, VisionRows::Cells, width, &stage.linear, scope, VisionLinearSite::Merger,
            output, None, None, stage.activation.map_or(0, activation_code))?;
        width = stage.linear.outputs();
    }
    if let Some(norm) = &merger.output_norm {
        b.norm(x, VisionRows::Cells, 1, width, norm, vision, VisionNormSite::Output, f32e,
            false, false)?;
    }
    Ok(VisionProgram {
        ops: b.ops,
        cell_rows,
        pixel_width,
        coordinates: !description.blocks.is_empty(),
        windows,
    })
}
