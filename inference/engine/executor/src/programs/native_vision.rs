//! Ordered native vision tower: the projector's vision program
//! (`operators::vision`) enqueued over one validated image, one graph per
//! exact patch class.

use super::graph::draft::GraphDraft;
use super::native_constants::{
    CheckedGraphFamilyResources, CheckedGraphResources, ConstantTensors, GraphConstant,
};
use super::{ReadySubmission, VisionProgram};
use crate::error::PlanError;
use crate::native::{AttestedVision, VisionKernels};
use crate::operators::vision::{
    vision_program, VisionKernel, VisionOperands, VisionProgram as VisionOps, VisionRows,
    VisionValue,
};
use crate::{
    DeviceError, GraphOutputTensor, InvariantError, ModelLoadPlan, NativeGraphOutputLease,
    NativeGraphWorkspaceLease, ResidentVision, SubmitError, ValidatedVisionLaunch,
    VisionLaunchCore, VisionProgramPlan, WeightPlan,
};
use magnitude_family_contracts::{VisionDescription, WeightRole};
use magnitude_kernels::{
    post_norm_residual, vision_attention, vision_clamp, vision_linear, vision_norm,
    vision_patch_stem, vision_pool, vision_position,
};
use seismic::{
    BackendName, BoundNativeGraphPlan, Device, Element, NativeGraph, NativeGraphClassSlice,
    NativeGraphFamily, NativeGraphFamilySlot, NativeGraphLayout, NativeGraphMetadata,
    NativeGraphPlan, NativeGraphStorageBytes, NativeKernel, NativePort, Tensor, WorkflowTensor,
    WorkflowTensorView,
};
use std::collections::HashMap;
use std::rc::Rc;

fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native vision program",
        detail: detail.into(),
    })
}

fn device(error: impl ToString) -> SubmitError {
    SubmitError::Device(DeviceError::Execution(error.to_string()))
}

/// The class scope of the invocations whose rows are merged cells; the
/// others' rows are patches.
const CELLS: &str = "vision_cells";

fn planned_weight(load: &ModelLoadPlan, role: WeightRole) -> Result<&WeightPlan, SubmitError> {
    load.weights()
        .find(|plan| plan.role == role)
        .ok_or_else(|| invalid(format!("planned vision weight {role:?} is absent")))
}

/// The projector's vision program over the planned resident elements.
fn planned_program(
    description: &VisionDescription,
    load: &ModelLoadPlan,
) -> Result<VisionOps, SubmitError> {
    vision_program(description, &|role| {
        load.weights()
            .find(|plan| plan.role == role)
            .map(|plan| plan.resident)
            .ok_or(PlanError::Topology("a vision weight is not planned"))
    })
    .map_err(|error| invalid(error.to_string()))
}

/// The kernel a graph binds for each vision invocation: the prepared native
/// kernel, or the element bindings of a metadata graph.
trait VisionBindings<G: GraphDraft> {
    fn patch_stem(
        &self,
        kernel: &VisionKernel,
    ) -> Result<G::Binding<'_, vision_patch_stem::Entry>, SubmitError>;
    fn norm(&self, kernel: &VisionKernel)
        -> Result<G::Binding<'_, vision_norm::Entry>, SubmitError>;
    fn linear(
        &self,
        kernel: &VisionKernel,
    ) -> Result<G::Binding<'_, vision_linear::Entry>, SubmitError>;
    fn clamp(
        &self,
        kernel: &VisionKernel,
    ) -> Result<G::Binding<'_, vision_clamp::Entry>, SubmitError>;
    fn attention(
        &self,
        kernel: &VisionKernel,
    ) -> Result<G::Binding<'_, vision_attention::Entry>, SubmitError>;
    fn pool(&self, kernel: &VisionKernel)
        -> Result<G::Binding<'_, vision_pool::Entry>, SubmitError>;
    fn position(
        &self,
        kernel: &VisionKernel,
    ) -> Result<G::Binding<'_, vision_position::Entry>, SubmitError>;
    fn post_norm(
        &self,
        kernel: &VisionKernel,
    ) -> Result<G::Binding<'_, post_norm_residual::Entry>, SubmitError>;
}

fn prepared<'a, E: seismic::Entry>(
    kernels: &'a HashMap<VisionKernel, NativeKernel<E>>,
    kernel: &VisionKernel,
) -> Result<&'a NativeKernel<E>, SubmitError> {
    kernels
        .get(kernel)
        .ok_or_else(|| invalid(format!("vision kernel {kernel:?} was not prepared")))
}

impl VisionBindings<NativeGraph> for VisionKernels {
    fn patch_stem(
        &self,
        kernel: &VisionKernel,
    ) -> Result<&NativeKernel<vision_patch_stem::Entry>, SubmitError> {
        prepared(&self.patch_stem, kernel)
    }
    fn norm(&self, kernel: &VisionKernel) -> Result<&NativeKernel<vision_norm::Entry>, SubmitError> {
        prepared(&self.norm, kernel)
    }
    fn linear(
        &self,
        kernel: &VisionKernel,
    ) -> Result<&NativeKernel<vision_linear::Entry>, SubmitError> {
        prepared(&self.linear, kernel)
    }
    fn clamp(
        &self,
        kernel: &VisionKernel,
    ) -> Result<&NativeKernel<vision_clamp::Entry>, SubmitError> {
        prepared(&self.clamp, kernel)
    }
    fn attention(
        &self,
        kernel: &VisionKernel,
    ) -> Result<&NativeKernel<vision_attention::Entry>, SubmitError> {
        prepared(&self.attention, kernel)
    }
    fn pool(&self, kernel: &VisionKernel) -> Result<&NativeKernel<vision_pool::Entry>, SubmitError> {
        prepared(&self.pool, kernel)
    }
    fn position(
        &self,
        kernel: &VisionKernel,
    ) -> Result<&NativeKernel<vision_position::Entry>, SubmitError> {
        prepared(&self.position, kernel)
    }
    fn post_norm(
        &self,
        kernel: &VisionKernel,
    ) -> Result<&NativeKernel<post_norm_residual::Entry>, SubmitError> {
        prepared(&self.post_norm, kernel)
    }
}

/// The element bindings of a planned kernel.
fn planned_elements<'a>(
    plan: &'a VisionProgramPlan,
    kernel: &VisionKernel,
) -> Result<&'a [(&'static str, Element)], SubmitError> {
    plan.kernels()
        .iter()
        .find(|planned| *planned == kernel)
        .map(|planned| planned.elements.as_slice())
        .ok_or_else(|| invalid(format!("vision kernel {kernel:?} is not planned")))
}

impl VisionBindings<NativeGraphMetadata> for VisionProgramPlan {
    fn patch_stem(&self, kernel: &VisionKernel) -> Result<&[(&str, Element)], SubmitError> {
        planned_elements(self, kernel)
    }
    fn norm(&self, kernel: &VisionKernel) -> Result<&[(&str, Element)], SubmitError> {
        planned_elements(self, kernel)
    }
    fn linear(&self, kernel: &VisionKernel) -> Result<&[(&str, Element)], SubmitError> {
        planned_elements(self, kernel)
    }
    fn clamp(&self, kernel: &VisionKernel) -> Result<&[(&str, Element)], SubmitError> {
        planned_elements(self, kernel)
    }
    fn attention(&self, kernel: &VisionKernel) -> Result<&[(&str, Element)], SubmitError> {
        planned_elements(self, kernel)
    }
    fn pool(&self, kernel: &VisionKernel) -> Result<&[(&str, Element)], SubmitError> {
        planned_elements(self, kernel)
    }
    fn position(&self, kernel: &VisionKernel) -> Result<&[(&str, Element)], SubmitError> {
        planned_elements(self, kernel)
    }
    fn post_norm(&self, kernel: &VisionKernel) -> Result<&[(&str, Element)], SubmitError> {
        planned_elements(self, kernel)
    }
}

/// The static dimension `name` of a kernel.
fn static_value(kernel: &VisionKernel, name: &str) -> Result<u64, SubmitError> {
    kernel
        .statics
        .iter()
        .find(|(bound, _)| *bound == name)
        .map(|(_, value)| *value)
        .ok_or_else(|| {
            invalid(format!(
                "vision kernel {:?} has no dimension {name}",
                kernel.entry
            ))
        })
}

/// A constant of a vision graph, uploaded once at binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Constant {
    /// A one-element F32 tensor whose zero-row views are the absent F32
    /// operands.
    EmptyF32,
    /// Likewise for absent i32 operands.
    EmptyI32,
    /// The unit rows of a weightless value norm of one head width.
    Unit(u64),
    /// The row identity of the post-norms' output rows.
    Identity,
}

impl Constant {
    /// The constants `program` binds, the empty operands first.
    fn of(program: &VisionOps) -> Result<Vec<Self>, SubmitError> {
        let mut constants = vec![Self::EmptyF32, Self::EmptyI32];
        for op in &program.ops {
            let constant = match op.operands {
                VisionOperands::Attention {
                    value_norm: true, ..
                } => Self::Unit(4 * static_value(&op.kernel, "P")?),
                VisionOperands::PostNormResidual { .. } => Self::Identity,
                _ => continue,
            };
            if !constants.contains(&constant) {
                constants.push(constant);
            }
        }
        Ok(constants)
    }

    /// Its value at `rows` patch rows, without a graph port.
    fn value(self, rows: u64) -> Result<GraphConstant, String> {
        match self {
            Self::EmptyF32 => GraphConstant::f32_value(&[1], &[0.0]),
            Self::EmptyI32 => Ok(GraphConstant::i32_value(&[0])),
            Self::Unit(width) => GraphConstant::f32_value(&[1, width], &vec![1.0; width as usize]),
            Self::Identity => GraphConstant::identity_value(rows),
        }
    }

    /// The same value as an external port of `graph`.
    fn draw<G: GraphDraft>(self, graph: &mut G, rows: u64) -> Result<GraphConstant, String> {
        match self {
            Self::EmptyF32 => GraphConstant::f32_shaped(graph, &[1], &[0.0]),
            Self::EmptyI32 => GraphConstant::i32(graph, &[0]),
            Self::Unit(width) => {
                GraphConstant::f32_shaped(graph, &[1, width], &vec![1.0; width as usize])
            }
            Self::Identity => GraphConstant::identity_for_class(graph, rows, Some("O")),
        }
    }
}

/// The graph inputs of one image, the weight ports and the constants.
struct VisionGraphPorts {
    pixels: NativePort,
    indices: NativePort,
    coefficients: NativePort,
    coordinates: Option<NativePort>,
    spans: Option<NativePort>,
    order: Option<NativePort>,
    weights: Vec<(WeightRole, NativePort)>,
    constants: Vec<GraphConstant>,
    features: WorkflowTensor,
}

struct Draft<'b, G: GraphDraft, B: VisionBindings<G>> {
    graph: G,
    bindings: &'b B,
    load: &'b ModelLoadPlan,
    program: &'b VisionOps,
    rows: u64,
    cells: u64,
    values: Vec<WorkflowTensor>,
    pixels: Option<NativePort>,
    indices: Option<(NativePort, NativePort)>,
    coordinates: Option<NativePort>,
    spans: Option<NativePort>,
    order: Option<NativePort>,
    weights: Vec<(WeightRole, NativePort)>,
    constants: Vec<(Constant, GraphConstant)>,
}

impl<G: GraphDraft, B: VisionBindings<G>> Draft<'_, G, B> {
    fn count(&self, rows: VisionRows) -> u64 {
        match rows {
            VisionRows::Patches => self.rows,
            VisionRows::Cells => self.cells,
        }
    }

    /// Scopes the next invocation's class dimensions to its rows.
    fn scope(&mut self, rows: VisionRows) {
        self.graph.set_class_scope(match rows {
            VisionRows::Patches => None,
            VisionRows::Cells => Some(CELLS),
        });
    }

    fn weight(
        &mut self,
        role: WeightRole,
        extents: &[u64],
    ) -> Result<WorkflowTensorView, SubmitError> {
        let plan = planned_weight(self.load, role)?;
        let port = self
            .graph
            .port(plan.resident, &plan.shape)
            .map_err(device)?;
        let view = port.tensor().reshape(extents);
        self.weights.push((role, port));
        Ok(view)
    }

    fn constant(&self, which: Constant) -> Result<&WorkflowTensor, SubmitError> {
        self.constants
            .iter()
            .find(|(constant, _)| *constant == which)
            .map(|(_, constant)| constant.port().tensor())
            .ok_or_else(|| invalid(format!("vision graph constant {which:?} is not drawn")))
    }

    /// Zero rows `[0, ...extents]` of an empty constant.
    fn empty(&self, which: Constant, extents: &[u64]) -> Result<WorkflowTensorView, SubmitError> {
        let shape = std::iter::once(0)
            .chain(extents.iter().copied())
            .collect::<Vec<_>>();
        Ok(self.constant(which)?.slice_leading(0, 0).reshape(&shape))
    }

    /// An optional weight as one row `[1, ...extents]`, or zero rows.
    fn rows_of(
        &mut self,
        role: Option<WeightRole>,
        extents: &[u64],
    ) -> Result<WorkflowTensorView, SubmitError> {
        match role {
            Some(role) => {
                let shape = std::iter::once(1)
                    .chain(extents.iter().copied())
                    .collect::<Vec<_>>();
                self.weight(role, &shape)
            }
            None => self.empty(Constant::EmptyF32, extents),
        }
    }

    fn value(&self, value: VisionValue, extents: &[u64]) -> Result<WorkflowTensorView, SubmitError> {
        match value {
            VisionValue::Result(index) => self
                .values
                .get(index)
                .map(|tensor| tensor.reshape(extents))
                .ok_or_else(|| invalid("a vision operand precedes its producer")),
            VisionValue::Pixels => input_view(&self.pixels, extents),
        }
    }

    /// The row width of a value.
    fn width(&self, value: VisionValue) -> Result<u64, SubmitError> {
        match value {
            VisionValue::Result(index) => self
                .program
                .ops
                .get(index)
                .map(|op| op.width)
                .ok_or_else(|| invalid("a vision operand precedes its producer")),
            VisionValue::Pixels => Ok(self.program.pixel_width),
        }
    }

    /// Rows `L` of the position table `[L, width]`.
    fn table_rows(&self, table: WeightRole, width: u64) -> Result<u64, SubmitError> {
        let values = planned_weight(self.load, table)?
            .shape
            .iter()
            .product::<u64>();
        if width == 0 || values % width != 0 {
            return Err(invalid("the vision position table is not whole rows"));
        }
        Ok(values / width)
    }

    fn enqueue(
        &mut self,
        kernel: &VisionKernel,
        operands: &VisionOperands,
    ) -> Result<WorkflowTensor, SubmitError> {
        let rows = self.rows;
        let statics = |name| static_value(kernel, name);
        let result = match operands {
            VisionOperands::PatchStem {
                frame,
                next_frame,
                bias,
                table,
            } => {
                let (c, s, p, h) = (statics("C")?, statics("S")?, statics("P")?, statics("H")?);
                let l = self.table_rows(*table, h)?;
                let dims = [
                    ("M", rows),
                    ("C", c),
                    ("S", s),
                    ("P", p),
                    ("H", h),
                    ("L", l),
                    ("NB", statics("NB")?),
                ];
                let binding = self.bindings.patch_stem(kernel)?;
                let pixels = self
                    .graph
                    .input_for(binding, "pixels", &dims)
                    .map_err(device)?;
                let indices = self
                    .graph
                    .input_for(binding, "indices", &dims)
                    .map_err(device)?;
                let coefficients = self
                    .graph
                    .input_for(binding, "coefficients", &dims)
                    .map_err(device)?;
                let frame = self.weight(*frame, &[h, c, p, p])?;
                let next_frame = match next_frame {
                    Some(role) => self.weight(*role, &[1, h, c, p, p])?,
                    None => self.empty(Constant::EmptyF32, &[h, c, p, p])?,
                };
                let bias = self.rows_of(*bias, &[h])?;
                let table = self.weight(*table, &[l, h])?;
                let value = self
                    .graph
                    .enqueue::<vision_patch_stem::Entry>(
                        binding,
                        &dims,
                        vision_patch_stem::WorkflowArgs {
                            pixels: pixels.tensor().into(),
                            frame_weight: (&frame).into(),
                            next_frame_weight: (&next_frame).into(),
                            bias: (&bias).into(),
                            table: (&table).into(),
                            indices: indices.tensor().into(),
                            coefficients: coefficients.tensor().into(),
                        },
                    )
                    .map_err(device)?
                    .value;
                self.pixels = Some(pixels);
                self.indices = Some((indices, coefficients));
                value
            }
            VisionOperands::Norm {
                source,
                rows: row_kind,
                weight,
                bias,
                gathered,
                epsilon,
                centered,
                interleave,
            } => {
                let (g, h) = (statics("G")?, statics("H")?);
                // A norm of G > 1 member rows per cell reads patch rows and
                // publishes cells.
                let cells = if g == 1 { self.count(*row_kind) } else { self.cells };
                let dims = [
                    ("C", cells),
                    ("G", g),
                    ("H", h),
                    ("NW", statics("NW")?),
                    ("NB", statics("NB")?),
                    ("NO", statics("NO")?),
                ];
                let binding = self.bindings.norm(kernel)?;
                self.scope(if g == 1 { *row_kind } else { VisionRows::Cells });
                if *source == VisionValue::Pixels {
                    self.pixels = Some(
                        self.graph
                            .input_for(binding, "source", &dims)
                            .map_err(device)?,
                    );
                }
                let source = self.value(*source, &[cells, g, h])?;
                let weight = self.rows_of(*weight, &[h])?;
                let bias = self.rows_of(*bias, &[h])?;
                let order = if *gathered {
                    let port = self
                        .graph
                        .input_for(binding, "order", &dims)
                        .map_err(device)?;
                    let view = port.tensor().reshape(&[1, cells * g]);
                    self.order = Some(port);
                    view
                } else {
                    self.empty(Constant::EmptyI32, &[cells * g])?
                };
                self.graph
                    .enqueue::<vision_norm::Entry>(
                        binding,
                        &dims,
                        vision_norm::WorkflowArgs {
                            source: (&source).into(),
                            weight: (&weight).into(),
                            bias: (&bias).into(),
                            order: (&order).into(),
                            epsilon: *epsilon,
                            centered: i32::from(*centered),
                            interleave: i32::from(*interleave),
                        },
                    )
                    .map_err(device)?
                    .value
            }
            VisionOperands::Linear {
                x,
                rows: row_kind,
                weight,
                bias,
                residual,
                gate,
                bounds,
                activation,
            } => {
                let (n, k) = (statics("N")?, statics("K")?);
                let m = self.count(*row_kind);
                let dims = [
                    ("M", m),
                    ("N", n),
                    ("K", k),
                    ("NB", statics("NB")?),
                    ("NR", statics("NR")?),
                    ("NG", statics("NG")?),
                    ("NC", statics("NC")?),
                ];
                let binding = self.bindings.linear(kernel)?;
                self.scope(*row_kind);
                let x = self.value(*x, &[m, k])?;
                let weight = self.weight(*weight, &[n, k])?;
                let bias = self.rows_of(*bias, &[n])?;
                let residual = match residual {
                    Some(value) => self.value(*value, &[1, m, n])?,
                    None => self.empty(Constant::EmptyF32, &[m, n])?,
                };
                let gate = match gate {
                    Some(value) => self.value(*value, &[1, m, n])?,
                    None => x.slice_leading(0, 0).reshape(&[0, m, n]),
                };
                let (minimum, maximum) = match bounds {
                    Some((minimum, maximum)) => {
                        (self.weight(*minimum, &[1])?, self.weight(*maximum, &[1])?)
                    }
                    None => (
                        self.empty(Constant::EmptyF32, &[])?,
                        self.empty(Constant::EmptyF32, &[])?,
                    ),
                };
                self.graph
                    .enqueue::<vision_linear::Entry>(
                        binding,
                        &dims,
                        vision_linear::WorkflowArgs {
                            x: (&x).into(),
                            weight: (&weight).into(),
                            bias: (&bias).into(),
                            residual: (&residual).into(),
                            gate: (&gate).into(),
                            minimum: (&minimum).into(),
                            maximum: (&maximum).into(),
                            activation: *activation,
                        },
                    )
                    .map_err(device)?
                    .value
            }
            VisionOperands::Clamp {
                x,
                rows: row_kind,
                minimum,
                maximum,
            } => {
                let m = self.count(*row_kind);
                let n = self.width(*x)?;
                let dims = [("M", m), ("N", n)];
                let binding = self.bindings.clamp(kernel)?;
                self.scope(*row_kind);
                let x = self.value(*x, &[m, n])?;
                let minimum = self.weight(*minimum, &[1])?;
                let maximum = self.weight(*maximum, &[1])?;
                self.graph
                    .enqueue::<vision_clamp::Entry>(
                        binding,
                        &dims,
                        vision_clamp::WorkflowArgs {
                            x: (&x).into(),
                            minimum: (&minimum).into(),
                            maximum: (&maximum).into(),
                        },
                    )
                    .map_err(device)?
                    .value
            }
            VisionOperands::Attention {
                query,
                key,
                value,
                query_norm,
                key_norm,
                value_norm,
                windowed,
                log_base,
                epsilon,
                unit_scale,
            } => {
                let (heads, quarter) = (statics("H")?, statics("P")?);
                let width = 4 * quarter;
                let dims = [
                    ("M", rows),
                    ("H", heads),
                    ("P", quarter),
                    ("NQ", statics("NQ")?),
                    ("NV", statics("NV")?),
                    ("WS", statics("WS")?),
                ];
                let binding = self.bindings.attention(kernel)?;
                if self.coordinates.is_none() {
                    self.coordinates = Some(
                        self.graph
                            .input_for(binding, "coordinates", &dims)
                            .map_err(device)?,
                    );
                }
                if *windowed && self.spans.is_none() {
                    self.spans = Some(
                        self.graph
                            .input_for(binding, "spans", &dims)
                            .map_err(device)?,
                    );
                }
                let shape = [rows, heads, width];
                let query = self.value(*query, &shape)?;
                let key = self.value(*key, &shape)?;
                let value = self.value(*value, &shape)?;
                let query_norm = self.rows_of(*query_norm, &[width])?;
                let key_norm = self.rows_of(*key_norm, &[width])?;
                let value_norm = if *value_norm {
                    self.constant(Constant::Unit(width))?.reshape(&[1, width])
                } else {
                    self.empty(Constant::EmptyF32, &[width])?
                };
                let coordinates = input_view(&self.coordinates, &[rows, 2])?;
                let spans = if *windowed {
                    input_view(&self.spans, &[1, rows, 2])?
                } else {
                    self.empty(Constant::EmptyI32, &[rows, 2])?
                };
                self.graph
                    .enqueue::<vision_attention::Entry>(
                        binding,
                        &dims,
                        vision_attention::WorkflowArgs {
                            query: (&query).into(),
                            key: (&key).into(),
                            value: (&value).into(),
                            query_norm: (&query_norm).into(),
                            key_norm: (&key_norm).into(),
                            value_norm: (&value_norm).into(),
                            coordinates: (&coordinates).into(),
                            spans: (&spans).into(),
                            log_base: *log_base,
                            epsilon: *epsilon,
                            unit_scale: i32::from(*unit_scale),
                        },
                    )
                    .map_err(device)?
                    .value
            }
            VisionOperands::Pool {
                source,
                standard,
                weight,
                scale,
            } => {
                let (g, h) = (statics("G")?, statics("H")?);
                let cells = self.cells;
                let dims = [("M", cells), ("G", g), ("H", h), ("NS", statics("NS")?)];
                let binding = self.bindings.pool(kernel)?;
                self.scope(VisionRows::Cells);
                let source = self.value(*source, &[cells, g, h])?;
                let standard_bias = self.rows_of(standard.map(|(bias, _)| bias), &[h])?;
                let standard_scale = self.rows_of(standard.map(|(_, scale)| scale), &[h])?;
                self.graph
                    .enqueue::<vision_pool::Entry>(
                        binding,
                        &dims,
                        vision_pool::WorkflowArgs {
                            source: (&source).into(),
                            standard_bias: (&standard_bias).into(),
                            standard_scale: (&standard_scale).into(),
                            weight: *weight,
                            scale: *scale,
                        },
                    )
                    .map_err(device)?
                    .value
            }
            VisionOperands::Position { source, table } => {
                let h = statics("H")?;
                let l = self.table_rows(*table, h)?;
                let dims = [("M", rows), ("H", h), ("L", l)];
                let binding = self.bindings.position(kernel)?;
                let indices = self
                    .graph
                    .input_for(binding, "indices", &dims)
                    .map_err(device)?;
                let coefficients = self
                    .graph
                    .input_for(binding, "coefficients", &dims)
                    .map_err(device)?;
                let source = self.value(*source, &[rows, h])?;
                let table = self.weight(*table, &[l, h])?;
                let value = self
                    .graph
                    .enqueue::<vision_position::Entry>(
                        binding,
                        &dims,
                        vision_position::WorkflowArgs {
                            source: (&source).into(),
                            table: (&table).into(),
                            indices: indices.tensor().into(),
                            coefficients: coefficients.tensor().into(),
                        },
                    )
                    .map_err(device)?
                    .value;
                self.indices = Some((indices, coefficients));
                value
            }
            VisionOperands::PostNormResidual {
                residual,
                projected,
                norm,
                epsilon,
            } => {
                let width = self.width(*projected)?;
                let dims = [("M", rows), ("O", rows), ("D", width)];
                let binding = self.bindings.post_norm(kernel)?;
                let residual = self.value(*residual, &[rows, width])?;
                let projected = self.value(*projected, &[rows, width])?;
                let norm = self.weight(*norm, &[width])?;
                let out_rows = self.constant(Constant::Identity)?.clone();
                self.graph
                    .enqueue::<post_norm_residual::Entry>(
                        binding,
                        &dims,
                        post_norm_residual::WorkflowArgs {
                            residual: (&residual).into(),
                            projected: (&projected).into(),
                            norm: (&norm).into(),
                            out_rows: (&out_rows).into(),
                            epsilon: *epsilon,
                            scale: 1.0,
                        },
                    )
                    .map_err(device)?
                    .value
            }
        };
        self.graph.set_class_scope(None);
        Ok(result)
    }
}

/// A graph input viewed with `extents`.
fn input_view(port: &Option<NativePort>, extents: &[u64]) -> Result<WorkflowTensorView, SubmitError> {
    port.as_ref()
        .map(|port| port.tensor().reshape(extents))
        .ok_or_else(|| invalid("a vision graph input is read before it is created"))
}

fn vision_graph_draft<G: GraphDraft, B: VisionBindings<G>>(
    mut graph: G,
    bindings: &B,
    load: &ModelLoadPlan,
    program: &VisionOps,
    rows: u64,
) -> Result<(G, VisionGraphPorts), SubmitError> {
    if rows == 0 || rows % program.cell_rows != 0 {
        return Err(invalid(
            "the vision patch class is not a positive cell multiple",
        ));
    }
    let constants = Constant::of(program)?
        .into_iter()
        .map(|constant| Ok((constant, constant.draw(&mut graph, rows).map_err(invalid)?)))
        .collect::<Result<Vec<_>, SubmitError>>()?;
    let mut draft = Draft {
        graph,
        bindings,
        load,
        program,
        rows,
        cells: rows / program.cell_rows,
        values: Vec::with_capacity(program.ops.len()),
        pixels: None,
        indices: None,
        coordinates: None,
        spans: None,
        order: None,
        weights: Vec::new(),
        constants,
    };
    for op in &program.ops {
        let result = draft.enqueue(&op.kernel, &op.operands)?;
        draft.values.push(result);
    }
    let features = draft
        .values
        .last()
        .cloned()
        .ok_or_else(|| invalid("the vision program has no invocation"))?;
    draft.graph.export(&features).map_err(device)?;
    let (pixels, (indices, coefficients)) = draft
        .pixels
        .zip(draft.indices)
        .ok_or_else(|| invalid("the vision program reads no pixels or positions"))?;
    if draft.coordinates.is_some() != program.coordinates
        || draft.spans.is_some() != program.windows
        || draft.order.is_some() != program.windows
    {
        return Err(invalid(
            "the vision program's inputs disagree with its invocations",
        ));
    }
    Ok((
        draft.graph,
        VisionGraphPorts {
            pixels,
            indices,
            coefficients,
            coordinates: draft.coordinates,
            spans: draft.spans,
            order: draft.order,
            weights: draft.weights,
            constants: draft
                .constants
                .into_iter()
                .map(|(_, constant)| constant)
                .collect(),
            features,
        },
    ))
}

/// The graph family's storage and the distinct constants its binding
/// uploads, for assessment.
pub(crate) fn checked_vision_family_resources(
    backend: BackendName,
    load: &ModelLoadPlan,
    description: &VisionDescription,
    plan: &VisionProgramPlan,
    patch_rows: impl IntoIterator<Item = u64>,
) -> Result<CheckedGraphResources, String> {
    let patch_rows = patch_rows.into_iter().collect::<Vec<_>>();
    let program = planned_program(description, load).map_err(|error| error.to_string())?;
    let (storage, _) = certify_vision_family(backend, load, &program, plan, &patch_rows)?;
    let constants = Constant::of(&program).map_err(|error| error.to_string())?;
    let mut resources = CheckedGraphFamilyResources::new();
    for &rows in &patch_rows {
        resources.include(
            storage,
            constants
                .iter()
                .map(|constant| constant.value(rows))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    resources.finish()
}

fn certify_vision_family(
    backend: BackendName,
    load: &ModelLoadPlan,
    program: &VisionOps,
    plan: &VisionProgramPlan,
    patch_rows: &[u64],
) -> Result<(NativeGraphStorageBytes, Vec<NativeGraphLayout>), String> {
    let cell = program.cell_rows;
    let &largest = patch_rows
        .iter()
        .max()
        .ok_or("the vision graph family has no exact patch class")?;
    if patch_rows.iter().any(|&rows| rows == 0 || rows % cell != 0)
        || patch_rows
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != patch_rows.len()
    {
        return Err("the vision graph patch classes are invalid or duplicated".into());
    }
    let (graph, _) = vision_graph_draft(
        NativeGraphMetadata::new_template(backend),
        plan,
        load,
        program,
        largest,
    )
    .map_err(|error| error.to_string())?;
    // Patch-row invocations read the class's patch rows; cell-row ones its
    // cells. The norm's rows are its `C` (the stem's `C` is its channels).
    let slices = patch_rows
        .iter()
        .map(|&rows| {
            NativeGraphClassSlice::new()
                .dimension("M", [rows])
                .dimension("O", [rows])
                .scoped(<vision_norm::Entry as seismic::Entry>::NAME, "C", [rows])
                .scoped(CELLS, "M", [rows / cell])
                .scoped(CELLS, "C", [rows / cell])
        })
        .collect::<Vec<_>>();
    let layout = graph
        .seal_template()
        .and_then(|template| template.certify(&slices))
        .map_err(|error| error.to_string())?;
    Ok((layout.storage_bytes(), vec![layout; patch_rows.len()]))
}

#[cfg(test)]
pub(crate) fn verify_vision_family_certificates(
    backend: BackendName,
    load: &ModelLoadPlan,
    description: &VisionDescription,
    plan: &VisionProgramPlan,
    patch_rows: &[u64],
) -> Result<(), String> {
    let program = planned_program(description, load).map_err(|error| error.to_string())?;
    let (_, layouts) = certify_vision_family(backend, load, &program, plan, patch_rows)?;
    for (&rows, layout) in patch_rows.iter().zip(&layouts) {
        let (graph, _) = vision_graph_draft(
            NativeGraphMetadata::new(backend),
            plan,
            load,
            &program,
            rows,
        )
        .map_err(|error| error.to_string())?;
        let charged = graph
            .seal_with_layout(layout)
            .map_err(|error| format!("vision class {rows}: {error}"))?;
        if charged != layout.storage_bytes() {
            return Err(format!("vision class {rows} charged a different layout"));
        }
    }
    Ok(())
}

pub struct NativeVisionProgram {
    graphs: BoundVisionGraphs,
}

impl NativeVisionProgram {
    pub(crate) fn new(graphs: BoundVisionGraphs) -> Self {
        Self { graphs }
    }

    fn execute_graph(
        &self,
        core: &VisionLaunchCore,
        workspace: &mut NativeGraphWorkspaceLease,
        output: &mut Option<NativeGraphOutputLease>,
    ) -> Result<GraphOutputTensor, SubmitError> {
        let input = core.batch().input();
        let rows = u64::try_from(core.batch().patch_rows())
            .map_err(|_| invalid("vision patch rows exceed u64"))?;
        let spatial = input.spatial();
        let indices = spatial.interpolation_indices();
        let indices = (0..spatial.rows())
            .flat_map(|row| indices.iter().map(move |plane| plane[row]))
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let coefficients = spatial.interpolation_coefficients();
        let coefficients = (0..spatial.rows())
            .flat_map(|row| coefficients.iter().map(move |plane| plane[row]))
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let pairs = |pairs: &[[i32; 2]]| {
            pairs
                .iter()
                .flatten()
                .copied()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>()
        };
        let coordinates = pairs(spatial.attention_coordinates());
        let spans = pairs(spatial.window_ranges());
        // Result row q of the merger reads the tower row that holds the
        // processor's patch row q.
        let mut order = vec![0i32; spatial.rows()];
        for (tower_row, &source) in spatial.patch_order().iter().enumerate() {
            order[source] =
                i32::try_from(tower_row).map_err(|_| invalid("vision rows exceed i32"))?;
        }
        let order = order
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let (prepared, bound) = self.graphs.class(rows)?;
        let output = output
            .take()
            .ok_or_else(|| invalid("vision graph output lease is absent"))?;
        Ok(prepared
            .run(
                rows,
                bound,
                workspace.slot_mut(),
                output,
                VisionGraphUploads {
                    pixels: input.pixels().data(),
                    indices: &indices,
                    coefficients: &coefficients,
                    coordinates: &coordinates,
                    spans: &spans,
                    order: &order,
                },
            )?
            .features)
    }
}

struct PreparedVisionGraph {
    patch_rows: u64,
    plan: NativeGraphPlan,
    ports: VisionGraphPorts,
}

pub struct PreparedVisionGraphs {
    device: Device,
    variants: Vec<PreparedVisionGraph>,
    family: NativeGraphFamily,
}

pub(crate) struct BoundVisionGraphs {
    prepared: Rc<PreparedVisionGraphs>,
    variants: Vec<BoundNativeGraphPlan>,
    /// The uploaded graph constants, held with the plans that bind them.
    _constants: Vec<Tensor>,
}

pub(crate) struct VisionGraphUploads<'a> {
    pub pixels: &'a [u8],
    pub indices: &'a [u8],
    pub coefficients: &'a [u8],
    pub coordinates: &'a [u8],
    pub spans: &'a [u8],
    pub order: &'a [u8],
}

pub(crate) struct VisionGraphResult {
    pub features: GraphOutputTensor,
}

impl PreparedVisionGraphs {
    pub(crate) fn prepare_exact_classes(
        target_device: &Device,
        handles: &AttestedVision,
        load: &ModelLoadPlan,
        description: &VisionDescription,
        patch_rows: impl IntoIterator<Item = u64>,
    ) -> Result<Self, SubmitError> {
        let patch_rows = patch_rows.into_iter().collect::<Vec<_>>();
        let program = planned_program(description, load)?;
        let plan = VisionProgramPlan::new(program.kernels().into_iter().cloned().collect());
        let (_, layouts) = certify_vision_family(
            target_device.backend(),
            load,
            &program,
            &plan,
            &patch_rows,
        )
        .map_err(invalid)?;
        let mut variants = Vec::new();
        for (rows, layout) in patch_rows.into_iter().zip(layouts) {
            if variants
                .iter()
                .any(|variant: &PreparedVisionGraph| variant.patch_rows == rows)
            {
                return Err(invalid("vision graph patch class is duplicated"));
            }
            let (graph, ports) = vision_graph_draft(
                target_device.native_graph_with_layout(&layout),
                &handles.kernels,
                load,
                &program,
                rows,
            )?;
            variants.push(PreparedVisionGraph {
                patch_rows: rows,
                plan: graph.seal().map_err(device)?,
                ports,
            });
        }
        if variants.is_empty() {
            return Err(invalid("vision graph family has no exact patch class"));
        }
        let plans = variants
            .iter()
            .map(|variant| variant.plan.clone())
            .collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).map_err(device)?;
        Ok(Self {
            device: target_device.clone(),
            variants,
            family,
        })
    }

    pub fn workspace_bytes_max(&self) -> u64 {
        self.family.workspace_bytes()
    }

    pub fn output_bytes_max(&self) -> u64 {
        self.family.output_bytes()
    }

    pub fn family(&self) -> &NativeGraphFamily {
        &self.family
    }

    pub(crate) fn bind_weights(
        self: &Rc<Self>,
        resident: &ResidentVision,
    ) -> Result<BoundVisionGraphs, SubmitError> {
        let mut uploaded = ConstantTensors::new(self.device.clone());
        let mut variants = Vec::with_capacity(self.variants.len());
        for variant in &self.variants {
            let constants = variant
                .ports
                .constants
                .iter()
                .map(|constant| Ok((constant.port(), uploaded.tensor(constant).map_err(invalid)?)))
                .collect::<Result<Vec<_>, SubmitError>>()?;
            let mut fixed = Vec::with_capacity(variant.ports.weights.len() + constants.len());
            for (role, port) in &variant.ports.weights {
                fixed.push((port, resident.weights.get(*role).map_err(invalid)?.tensor()));
            }
            fixed.extend(constants.iter().map(|(port, tensor)| (*port, tensor)));
            variants.push(variant.plan.bind_static(&fixed).map_err(device)?);
        }
        Ok(BoundVisionGraphs {
            prepared: self.clone(),
            variants,
            _constants: uploaded.into_tensors(),
        })
    }

    pub(crate) fn run(
        &self,
        patch_rows: u64,
        bound: &BoundNativeGraphPlan,
        slot: &mut NativeGraphFamilySlot,
        mut output: NativeGraphOutputLease,
        uploads: VisionGraphUploads<'_>,
    ) -> Result<VisionGraphResult, SubmitError> {
        let variant = self
            .variants
            .iter()
            .find(|variant| variant.patch_rows == patch_rows)
            .ok_or_else(|| invalid("vision graph patch class was not prepared"))?;
        let ports = &variant.ports;
        let bindings = bound.bindings();
        let mut active = slot.activate(&variant.plan).map_err(device)?;
        for (port, bytes) in [
            (Some(&ports.pixels), uploads.pixels),
            (Some(&ports.indices), uploads.indices),
            (Some(&ports.coefficients), uploads.coefficients),
            (ports.coordinates.as_ref(), uploads.coordinates),
            (ports.spans.as_ref(), uploads.spans),
            (ports.order.as_ref(), uploads.order),
        ] {
            if let Some(port) = port {
                active.write_input(port, bytes).map_err(device)?;
            }
        }
        let outputs = output
            .activate(&variant.plan)
            .map_err(SubmitError::Invariant)?;
        let outputs = active
            .attach(bindings, outputs)
            .and_then(super::run_graph)
            .map_err(device)?;
        let owner = output.publish(outputs);
        let features = owner
            .tensor(&ports.features)
            .ok_or_else(|| invalid("vision graph omitted retained features"))?;
        Ok(VisionGraphResult { features })
    }
}

impl BoundVisionGraphs {
    pub(crate) fn class(
        &self,
        patch_rows: u64,
    ) -> Result<(&PreparedVisionGraphs, &BoundNativeGraphPlan), SubmitError> {
        let index = self
            .prepared
            .variants
            .iter()
            .position(|variant| variant.patch_rows == patch_rows)
            .ok_or_else(|| invalid("vision graph patch class was not prepared"))?;
        Ok((&self.prepared, &self.variants[index]))
    }
}

impl VisionProgram for NativeVisionProgram {
    type Submission =
        ReadySubmission<VisionLaunchCore, NativeGraphWorkspaceLease, GraphOutputTensor>;
    fn submit(
        &mut self,
        mut launch: ValidatedVisionLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedVisionLaunch)> {
        let result = {
            let (core, workspace, output) = launch.execution_parts_mut();
            self.execute_graph(core, workspace, output)
        };
        let features = match result {
            Ok(features) => features,
            Err(error) => return Err((error, launch)),
        };
        let (core, workspace, _) = launch.into_submission_parts();
        Ok(ReadySubmission::new(core, workspace, features))
    }
}
