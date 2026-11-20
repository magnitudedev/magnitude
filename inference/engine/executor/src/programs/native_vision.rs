//! Ordered native vision tower: the projector's vision program
//! (`operators::vision`) enqueued over one validated image, one graph per
//! exact patch class.
//!
//! A tower's graph is recorded over its topology ([`VisionTopology`]: the
//! program's invocations, operands, elements and inputs) at its numbers
//! ([`VisionNumbers`]: its class's rows and cells, widths and statics) and
//! its weights (`GraphWeights`), as one trace per topology
//! (`seismic::trace`): assessment evaluates it at every later model of the
//! same topology, and a load certifies the same recorder at its own numbers.

use super::graph::facts::{Numbers, constant, constant_reals};
use super::graph::{
    GraphError,
    draft::{Certified, ElementBoundDraft, GraphDraft},
    sizing::SizingOutline,
};
use super::native_target_graph::{GraphWeights, PlannedWeights};
use super::native_constants::{
    CheckedGraphFamilyResources, CheckedGraphResources, ConstantTensors, GraphConstant,
    distinct_storage_bytes,
};
use super::{ReadySubmission, VisionProgram};
use crate::error::PlanError;
use crate::native::{AttestedVision, VisionKernels};
use crate::operators::vision::{
    LinearBounds, VisionEntry, VisionKernel, VisionOp, VisionOperands, VisionProgram as VisionOps,
    VisionRows, VisionValue, vision_program,
};
use crate::{
    DeviceError, GraphOutputTensor, InvariantError, ModelLoadPlan, NativeGraphOutputLease,
    NativeGraphWorkspaceLease, ResidentVision, SubmitError, ValidatedVisionLaunch,
    VisionLaunchCore, VisionProgramPlan,
};
use magnitude_family_contracts::{
    VisionAttentionSpan, VisionDescription, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    post_norm_residual, vision_attention, vision_clamp, vision_gated_convolution, vision_linear,
    vision_norm, vision_patch_stem, vision_pool, vision_position,
};
#[cfg(test)]
use seismic::NativeGraphMetadata;
use seismic::{
    Argument, BackendName, BoundNativeGraphPlan, ContractNodeShape, Device, Dim, Element,
    EntryContract, GraphCensus, NativeGraph, NativeGraphFamily,
    NativeGraphFamilySlot, NativeGraphLayout, NativeGraphPlan, NativeGraphStorageBytes,
    NativeKernel, NativePort, Tensor, WorkflowTensor, WorkflowTensorView,
};
use std::collections::{HashMap, HashSet};
use std::ops::Range;
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

/// The projector's vision program over the planned resident elements.
fn planned_program(
    description: &VisionDescription,
    load: &ModelLoadPlan,
) -> Result<VisionOps, GraphError> {
    let vision_weights: HashMap<WeightRole, Element> = load
        .weights()
        .filter(|plan| matches!(plan.role.kind, WeightKind::Vision(_)))
        .map(|plan| (plan.role, plan.resident))
        .collect();
    vision_program(description, &|role| {
        vision_weights
            .get(&role)
            .copied()
            .ok_or(PlanError::Topology("a vision weight is not planned"))
    })
    .map_err(|error| error.to_string().into())
}

/// What decides a vision graph's topology: its backend, each invocation's
/// entry, elements, static names and operands, and which inputs the tower
/// reads. Every number of it is one of its [`VisionNumbers`]; its weights
/// are its [`GraphWeights`]. Its real scalar arguments are `R`s ([`Numbers`]
/// over them): the model's in a split, their indexes in a key, recorded
/// arguments in a builder.
#[derive(Clone, PartialEq, Eq, Hash)]
struct VisionTopology<R = Argument<f64>> {
    backend: BackendName,
    ops: Vec<OpTopology<R>>,
    coordinates: bool,
    windows: bool,
    convolution: bool,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct OpTopology<R> {
    entry: VisionEntry,
    elements: Vec<(&'static str, Element)>,
    /// The names of its kernel's static dimensions, in its
    /// [`OpNumbers::statics`] order.
    statics: Vec<&'static str>,
    operands: VisionOperands<R>,
}

impl<R> Numbers<R> for VisionTopology<R> {
    type Of<E> = VisionTopology<E>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> VisionTopology<E> {
        VisionTopology {
            backend: self.backend,
            ops: self
                .ops
                .into_iter()
                .map(|op| OpTopology {
                    entry: op.entry,
                    elements: op.elements,
                    statics: op.statics,
                    operands: op.operands.map(&mut *f),
                })
                .collect(),
            coordinates: self.coordinates,
            windows: self.windows,
            convolution: self.convolution,
        }
    }
}

/// The numbers of a vision graph of one [`VisionTopology`]: its class's
/// rows and the program's numbers.
#[derive(Clone, PartialEq, Eq, Hash)]
struct VisionNumbers<D> {
    /// The class's tower rows: the register rows, then its patch rows (the
    /// slices' unscoped `M`).
    rows: D,
    /// The class's merged cells (the slices' `M` of [`CELLS`]).
    cells: D,
    /// Patch rows per merged cell.
    cell_rows: D,
    /// Register rows ahead of the patch rows.
    registers: D,
    /// The width of one prepared patch row.
    pixel_width: D,
    ops: Vec<OpNumbers<D>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct OpNumbers<D> {
    /// The width of its result's rows.
    width: D,
    statics: Vec<D>,
}

impl<D> Numbers<D> for VisionNumbers<D> {
    type Of<E> = VisionNumbers<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> VisionNumbers<E> {
        VisionNumbers {
            rows: f(self.rows),
            cells: f(self.cells),
            cell_rows: f(self.cell_rows),
            registers: f(self.registers),
            pixel_width: f(self.pixel_width),
            ops: self
                .ops
                .into_iter()
                .map(|op| OpNumbers { width: f(op.width), statics: op.statics.map(&mut *f) })
                .collect(),
        }
    }
}

/// The topology and numbers of `program`'s graph on `backend` at a class of
/// `patches` patch rows, and the weights of `load` it reads.
fn split(
    backend: BackendName,
    program: &VisionOps,
    load: &ModelLoadPlan,
    patches: u64,
) -> Result<(VisionTopology<f64>, VisionNumbers<u64>, GraphWeights<u64>), GraphError> {
    let roles = program
        .ops
        .iter()
        .flat_map(|op| op.operands.weights())
        .collect::<std::collections::HashSet<_>>();
    let weights = GraphWeights::of(load, |role| roles.contains(&role));
    let topology = VisionTopology {
        backend,
        ops: program
            .ops
            .iter()
            .map(|op| OpTopology {
                entry: op.kernel.entry,
                elements: op.kernel.elements.clone(),
                statics: op.kernel.statics.iter().map(|(name, _)| *name).collect(),
                operands: op.operands.clone().map(&mut |real: f32| f64::from(real)),
            })
            .collect(),
        coordinates: program.coordinates,
        windows: program.windows,
        convolution: program.convolution.is_some(),
    };
    let (rows, cells) = class_rows(program.registers, program.cell_rows, patches)?;
    let numbers = VisionNumbers {
        rows,
        cells,
        cell_rows: program.cell_rows,
        registers: program.registers,
        pixel_width: program.pixel_width,
        ops: program
            .ops
            .iter()
            .map(|op| OpNumbers {
                width: op.width,
                statics: op.kernel.statics.iter().map(|(_, value)| *value).collect(),
            })
            .collect(),
    };
    Ok((topology, numbers, weights))
}

/// Direct contract sizing records zero traces.
#[cfg(test)]
pub(crate) fn vision_traces_recorded() -> usize {
    0
}

/// The kernel a graph binds for invocation `index`: the prepared native
/// kernel, or the element bindings of a metadata graph.
trait VisionBindings<G: GraphDraft> {
    /// The projecting entries bind the kernel of the range `rows` lie in.
    fn patch_stem(
        &self,
        index: usize,
        rows: &Dim,
    ) -> Result<G::Binding<'_, vision_patch_stem::Entry>, GraphError>;
    fn norm(&self, index: usize) -> Result<G::Binding<'_, vision_norm::Entry>, GraphError>;
    fn linear(
        &self,
        index: usize,
        rows: &Dim,
    ) -> Result<G::Binding<'_, vision_linear::Entry>, GraphError>;
    fn clamp(&self, index: usize) -> Result<G::Binding<'_, vision_clamp::Entry>, GraphError>;
    fn attention(&self, index: usize)
    -> Result<G::Binding<'_, vision_attention::Entry>, GraphError>;
    fn gated_convolution(
        &self,
        index: usize,
    ) -> Result<G::Binding<'_, vision_gated_convolution::Entry>, GraphError>;
    fn pool(&self, index: usize) -> Result<G::Binding<'_, vision_pool::Entry>, GraphError>;
    fn position(&self, index: usize)
    -> Result<G::Binding<'_, vision_position::Entry>, GraphError>;
    fn post_norm(&self, index: usize)
    -> Result<G::Binding<'_, post_norm_residual::Entry>, GraphError>;
}

/// The prepared kernels of a program's invocations.
struct PreparedBindings<'a> {
    kernels: &'a VisionKernels,
    program: &'a VisionOps,
}

impl PreparedBindings<'_> {
    fn kernel(&self, index: usize) -> Result<&VisionKernel, GraphError> {
        self.program
            .ops
            .get(index)
            .map(|op| &op.kernel)
            .ok_or_else(|| "a vision invocation is out of its program".into())
    }
}

fn prepared<'a, K>(
    kernels: &'a HashMap<VisionKernel, K>,
    kernel: &VisionKernel,
) -> Result<&'a K, GraphError> {
    kernels
        .get(kernel)
        .ok_or_else(|| format!("vision kernel {kernel:?} was not prepared").into())
}

impl VisionBindings<NativeGraph> for PreparedBindings<'_> {
    fn patch_stem(
        &self,
        index: usize,
        rows: &Dim,
    ) -> Result<&NativeKernel<vision_patch_stem::Entry>, GraphError> {
        Ok(prepared(&self.kernels.patch_stem, self.kernel(index)?)?.for_rows(rows))
    }
    fn norm(&self, index: usize) -> Result<&NativeKernel<vision_norm::Entry>, GraphError> {
        prepared(&self.kernels.norm, self.kernel(index)?)
    }
    fn linear(&self, index: usize, rows: &Dim) -> Result<&NativeKernel<vision_linear::Entry>, GraphError> {
        Ok(prepared(&self.kernels.linear, self.kernel(index)?)?.for_rows(rows))
    }
    fn clamp(&self, index: usize) -> Result<&NativeKernel<vision_clamp::Entry>, GraphError> {
        prepared(&self.kernels.clamp, self.kernel(index)?)
    }
    fn attention(&self, index: usize) -> Result<&NativeKernel<vision_attention::Entry>, GraphError> {
        prepared(&self.kernels.attention, self.kernel(index)?)
    }
    fn gated_convolution(
        &self,
        index: usize,
    ) -> Result<&NativeKernel<vision_gated_convolution::Entry>, GraphError> {
        prepared(&self.kernels.gated_convolution, self.kernel(index)?)
    }
    fn pool(&self, index: usize) -> Result<&NativeKernel<vision_pool::Entry>, GraphError> {
        prepared(&self.kernels.pool, self.kernel(index)?)
    }
    fn position(&self, index: usize) -> Result<&NativeKernel<vision_position::Entry>, GraphError> {
        prepared(&self.kernels.position, self.kernel(index)?)
    }
    fn post_norm(&self, index: usize) -> Result<&NativeKernel<post_norm_residual::Entry>, GraphError> {
        prepared(&self.kernels.post_norm, self.kernel(index)?)
    }
}

impl<R> VisionTopology<R> {
    /// The element bindings of invocation `index`.
    fn elements(&self, index: usize) -> Result<&[(&'static str, Element)], GraphError> {
        self.ops
            .get(index)
            .map(|op| op.elements.as_slice())
            .ok_or_else(|| "a vision invocation is out of its program".into())
    }
}

impl<G: ElementBoundDraft, R> VisionBindings<G> for VisionTopology<R> {
    fn patch_stem(&self, index: usize, _rows: &Dim) -> Result<G::Binding<'_, vision_patch_stem::Entry>, GraphError> {
        Ok(G::bind(self.elements(index)?))
    }
    fn norm(&self, index: usize) -> Result<G::Binding<'_, vision_norm::Entry>, GraphError> {
        Ok(G::bind(self.elements(index)?))
    }
    fn linear(&self, index: usize, _rows: &Dim) -> Result<G::Binding<'_, vision_linear::Entry>, GraphError> {
        Ok(G::bind(self.elements(index)?))
    }
    fn clamp(&self, index: usize) -> Result<G::Binding<'_, vision_clamp::Entry>, GraphError> {
        Ok(G::bind(self.elements(index)?))
    }
    fn attention(&self, index: usize) -> Result<G::Binding<'_, vision_attention::Entry>, GraphError> {
        Ok(G::bind(self.elements(index)?))
    }
    fn gated_convolution(&self, index: usize) -> Result<G::Binding<'_, vision_gated_convolution::Entry>, GraphError> {
        Ok(G::bind(self.elements(index)?))
    }
    fn pool(&self, index: usize) -> Result<G::Binding<'_, vision_pool::Entry>, GraphError> {
        Ok(G::bind(self.elements(index)?))
    }
    fn position(&self, index: usize) -> Result<G::Binding<'_, vision_position::Entry>, GraphError> {
        Ok(G::bind(self.elements(index)?))
    }
    fn post_norm(&self, index: usize) -> Result<G::Binding<'_, post_norm_residual::Entry>, GraphError> {
        Ok(G::bind(self.elements(index)?))
    }
}

/// A constant of a vision graph, uploaded once at binding.
#[derive(Clone, Debug)]
enum Constant {
    /// A one-element F32 tensor whose zero-row views are the absent F32
    /// operands.
    EmptyF32,
    /// Likewise for absent i32 operands.
    EmptyI32,
    /// The unit rows of a weightless value norm of one head width.
    Unit(Dim),
    /// The row identity of the post-norms' output rows.
    Identity,
    /// One F32 value (a feed-forward limit's bound), as its bits.
    Bound(u32),
}

impl Constant {
    /// Whether it is `other`: a unit row of the same width (a recorded
    /// comparison).
    fn is(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::EmptyF32, Self::EmptyF32)
            | (Self::EmptyI32, Self::EmptyI32)
            | (Self::Identity, Self::Identity) => true,
            (Self::Unit(width), Self::Unit(other)) => width.equals(other),
            (Self::Bound(bits), Self::Bound(other)) => bits == other,
            _ => false,
        }
    }

    /// The constants a graph of `topology` at `numbers` binds, the empty
    /// operands first.
    fn of(topology: &VisionTopology, numbers: &VisionNumbers<Dim>) -> Result<Vec<Self>, GraphError> {
        let mut constants = vec![Self::EmptyF32, Self::EmptyI32];
        for (index, op) in topology.ops.iter().enumerate() {
            let needed = match op.operands {
                VisionOperands::Attention { value_norm: true, .. } => {
                    vec![Self::Unit(static_value(topology, numbers, index, "P")? * 4u64)]
                }
                VisionOperands::PostNormResidual { .. } => vec![Self::Identity],
                VisionOperands::Linear {
                    bounds: Some(LinearBounds::Fixed { minimum, maximum }),
                    ..
                } => vec![Self::Bound(minimum), Self::Bound(maximum)],
                _ => continue,
            };
            for constant in needed {
                if !constants.iter().any(|known| known.is(&constant)) {
                    constants.push(constant);
                }
            }
        }
        Ok(constants)
    }

    /// The same value as an external port of `graph`, at `rows` tower rows.
    fn draw<G: GraphDraft>(&self, graph: &mut G, rows: &Dim) -> Result<GraphConstant, GraphError> {
        match self {
            Self::EmptyF32 => GraphConstant::f32_shaped(graph, &[1], &[0.0]),
            Self::EmptyI32 => GraphConstant::i32(graph, &[0]),
            Self::Unit(width) => GraphConstant::f32_filled(graph, &[Dim::from(1u64), width.clone()], 1.0),
            Self::Identity => GraphConstant::row_identity(graph, rows.clone(), Some("O")),
            Self::Bound(bits) => GraphConstant::f32_shaped(graph, &[1], &[f32::from_bits(*bits)]),
        }
    }
}

/// Static dimension `name` of invocation `index`.
fn static_value<R>(
    topology: &VisionTopology<R>,
    numbers: &VisionNumbers<Dim>,
    index: usize,
    name: &str,
) -> Result<Dim, GraphError> {
    let op = &topology.ops[index];
    op.statics
        .iter()
        .position(|bound| *bound == name)
        .map(|at| numbers.ops[index].statics[at].clone())
        .ok_or_else(|| format!("vision kernel {:?} has no dimension {name}", op.entry).into())
}

/// The graph inputs of one image, the weight ports and the constants.
struct VisionGraphPorts {
    pixels: NativePort,
    indices: NativePort,
    coefficients: NativePort,
    /// The register rows' blend of the register table.
    register_blend: Option<(NativePort, NativePort)>,
    coordinates: Option<NativePort>,
    window_spans: Option<NativePort>,
    image_spans: Option<NativePort>,
    order: Option<NativePort>,
    neighbors: Option<NativePort>,
    weights: Vec<(WeightRole, NativePort)>,
    constants: Vec<GraphConstant>,
    features: WorkflowTensor,
}

struct Draft<'b, G: GraphDraft, B: VisionBindings<G>, W: PlannedWeights + ?Sized> {
    graph: G,
    bindings: &'b B,
    topology: &'b VisionTopology,
    numbers: &'b VisionNumbers<Dim>,
    planned: &'b W,
    /// Tower rows: the register rows, then the class's patch rows.
    rows: Dim,
    cells: Dim,
    values: Vec<WorkflowTensor>,
    pixels: Option<NativePort>,
    indices: Option<(NativePort, NativePort)>,
    register_blend: Option<(NativePort, NativePort)>,
    coordinates: Option<NativePort>,
    window_spans: Option<NativePort>,
    image_spans: Option<NativePort>,
    order: Option<NativePort>,
    neighbors: Option<NativePort>,
    weights: Vec<(WeightRole, NativePort)>,
    constants: Vec<(Constant, GraphConstant)>,
}

impl<G: GraphDraft, B: VisionBindings<G>, W: PlannedWeights + ?Sized> Draft<'_, G, B, W> {
    fn count(&self, rows: VisionRows) -> Dim {
        match rows {
            VisionRows::Patches => self.rows.clone(),
            VisionRows::Cells => self.cells.clone(),
        }
    }

    fn weight(&mut self, role: WeightRole, extents: &[Dim]) -> Result<WorkflowTensorView, GraphError> {
        let planned = self.planned.weight(role)?;
        let port = self.graph.port(planned.resident, &planned.shape)?;
        let view = port.tensor().reshape(extents);
        self.weights.push((role, port));
        Ok(view)
    }

    fn constant(&self, which: &Constant) -> Result<&WorkflowTensor, GraphError> {
        self.constants
            .iter()
            .find(|(constant, _)| constant.is(which))
            .map(|(_, constant)| constant.port().tensor())
            .ok_or_else(|| format!("vision graph constant {which:?} is not drawn").into())
    }

    /// Zero rows `[0, ...extents]` of an empty constant.
    fn empty(&self, which: Constant, extents: &[Dim]) -> Result<WorkflowTensorView, GraphError> {
        let shape = std::iter::once(Dim::from(0u64))
            .chain(extents.iter().cloned())
            .collect::<Vec<_>>();
        Ok(self.constant(&which)?.slice_leading(0u64, 0u64).reshape(&shape))
    }

    /// An optional weight as one row `[1, ...extents]`, or zero rows.
    fn rows_of(
        &mut self,
        role: Option<WeightRole>,
        extents: &[Dim],
    ) -> Result<WorkflowTensorView, GraphError> {
        match role {
            Some(role) => {
                let shape = std::iter::once(Dim::from(1u64))
                    .chain(extents.iter().cloned())
                    .collect::<Vec<_>>();
                self.weight(role, &shape)
            }
            None => self.empty(Constant::EmptyF32, extents),
        }
    }

    fn value(&self, value: VisionValue, extents: &[Dim]) -> Result<WorkflowTensorView, GraphError> {
        let result = |index: usize| {
            self.values
                .get(index)
                .ok_or_else(|| GraphError::from("a vision operand precedes its producer"))
        };
        match value {
            VisionValue::Result(index) => Ok(result(index)?.reshape(extents)),
            VisionValue::PatchRows(index) => {
                Ok(result(index)?
                    .slice_leading(self.numbers.registers.clone(), self.rows.clone())
                    .reshape(extents))
            }
            VisionValue::Pixels => input_view(&self.pixels, extents),
        }
    }

    /// The row width of a value.
    fn width(&self, value: VisionValue) -> Result<Dim, GraphError> {
        match value {
            VisionValue::Result(index) | VisionValue::PatchRows(index) => self
                .numbers
                .ops
                .get(index)
                .map(|op| op.width.clone())
                .ok_or_else(|| "a vision operand precedes its producer".into()),
            VisionValue::Pixels => Ok(self.numbers.pixel_width.clone()),
        }
    }

    /// Rows `L` of the position table `[L, width]`.
    fn table_rows(&self, table: WeightRole, width: &Dim) -> Result<Dim, GraphError> {
        let values = self
            .planned
            .weight(table)?
            .shape
            .iter()
            .fold(Dim::from(1u64), |values, extent| values * extent);
        if width.is_zero() || !values.rem(width).is_zero() {
            return Err("the vision position table is not whole rows".into());
        }
        Ok(values / width)
    }

    /// `source` rows plus their blend of `table` rows (`vision_position`),
    /// and the graph inputs of the blend's indices and coefficients.
    fn blend(
        &mut self,
        index: usize,
        source: VisionValue,
        table: WeightRole,
    ) -> Result<(WorkflowTensor, (NativePort, NativePort)), GraphError> {
        let (rows, h) = (self.rows.clone(), self.statics(index, "H")?);
        let l = self.table_rows(table, &h)?;
        let dims = [("M", rows.clone()), ("H", h.clone()), ("L", l.clone())];
        let binding = self.bindings.position(index)?;
        let indices = self.graph.input_for(binding, "indices", &dims)?;
        let coefficients = self.graph.input_for(binding, "coefficients", &dims)?;
        let source = self.value(source, &[rows, h.clone()])?;
        let table = self.weight(table, &[l, h])?;
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
            )?
            .value;
        Ok((value, (indices, coefficients)))
    }

    fn statics(&self, index: usize, name: &str) -> Result<Dim, GraphError> {
        static_value(self.topology, self.numbers, index, name)
    }

    /// Invocation `index` of the program.
    fn enqueue(&mut self, index: usize) -> Result<WorkflowTensor, GraphError> {
        let (topology, numbers) = (self.topology, self.numbers);
        let rows = self.rows.clone();
        let statics = |name| static_value(topology, numbers, index, name);
        let result = match &topology.ops[index].operands {
            VisionOperands::PatchStem {
                frame,
                next_frame,
                bias,
                table,
            } => {
                let (c, s, p, h) = (statics("C")?, statics("S")?, statics("P")?, statics("H")?);
                let l = match table {
                    Some(table) => self.table_rows(*table, &h)?,
                    None => Dim::from(0u64),
                };
                let dims = [
                    ("M", rows.clone()),
                    ("C", c.clone()),
                    ("S", s),
                    ("P", p.clone()),
                    ("H", h.clone()),
                    ("L", l.clone()),
                    ("NB", statics("NB")?),
                    ("NT", statics("NT")?),
                ];
                let binding = self.bindings.patch_stem(index, &rows)?;
                let pixels = self.graph.input_for(binding, "pixels", &dims)?;
                let indices = self.graph.input_for(binding, "indices", &dims)?;
                let coefficients = self.graph.input_for(binding, "coefficients", &dims)?;
                let frame_extents = [h.clone(), c, p.clone(), p];
                let frame = self.weight(*frame, &frame_extents)?;
                let next_frame = match next_frame {
                    Some(role) => {
                        let extents = std::iter::once(Dim::from(1u64))
                            .chain(frame_extents.iter().cloned())
                            .collect::<Vec<_>>();
                        self.weight(*role, &extents)?
                    }
                    None => self.empty(Constant::EmptyF32, &frame_extents)?,
                };
                let bias = self.rows_of(*bias, &[h.clone()])?;
                let table = match table {
                    Some(table) => self.weight(*table, &[l, h])?,
                    None => self.empty(Constant::EmptyF32, &[h])?,
                };
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
                    )?
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
                activation,
            } => {
                let (g, h) = (statics("G")?, statics("H")?);
                // A norm of G > 1 member rows per cell reads patch rows and
                // publishes cells.
                let kind = if g.equals(1u64) { *row_kind } else { VisionRows::Cells };
                let cells = self.count(kind);
                let dims = [
                    ("C", cells.clone()),
                    ("G", g.clone()),
                    ("H", h.clone()),
                    ("NW", statics("NW")?),
                    ("NB", statics("NB")?),
                    ("NO", statics("NO")?),
                ];
                let binding = self.bindings.norm(index)?;
                if *source == VisionValue::Pixels {
                    self.pixels = Some(self.graph.input_for(binding, "source", &dims)?);
                }
                let source = self.value(*source, &[cells.clone(), g.clone(), h.clone()])?;
                let weight = self.rows_of(*weight, &[h.clone()])?;
                let bias = self.rows_of(*bias, &[h])?;
                let members = &cells * &g;
                let order = if *gathered {
                    let port = self.graph.input_for(binding, "order", &dims)?;
                    let view = port.tensor().reshape(&[Dim::from(1u64), members]);
                    self.order = Some(port);
                    view
                } else {
                    self.empty(Constant::EmptyI32, &[members])?
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
                            epsilon: epsilon.f32(),
                            centered: i32::from(*centered).into(),
                            interleave: i32::from(*interleave).into(),
                            activation: (*activation).into(),
                        },
                    )?
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
                    ("M", m.clone()),
                    ("N", n.clone()),
                    ("K", k.clone()),
                    ("NB", statics("NB")?),
                    ("NR", statics("NR")?),
                    ("NG", statics("NG")?),
                    ("NC", statics("NC")?),
                ];
                let binding = self.bindings.linear(index, &m)?;
                let x = self.value(*x, &[m.clone(), k.clone()])?;
                let weight = self.weight(*weight, &[n.clone(), k])?;
                let bias = self.rows_of(*bias, &[n.clone()])?;
                let rows_by_width = [Dim::from(1u64), m.clone(), n.clone()];
                let residual = match residual {
                    Some(value) => self.value(*value, &rows_by_width)?,
                    None => self.empty(Constant::EmptyF32, &[m.clone(), n.clone()])?,
                };
                let gate = match gate {
                    Some(value) => self.value(*value, &rows_by_width)?,
                    None => x.slice_leading(0u64, 0u64).reshape(&[Dim::from(0u64), m, n]),
                };
                let unit = [Dim::from(1u64)];
                let (minimum, maximum) = match bounds {
                    Some(LinearBounds::Stored { minimum, maximum }) => (
                        self.weight(*minimum, &unit)?,
                        self.weight(*maximum, &unit)?,
                    ),
                    Some(LinearBounds::Fixed { minimum, maximum }) => (
                        self.constant(&Constant::Bound(*minimum))?.reshape(&unit),
                        self.constant(&Constant::Bound(*maximum))?.reshape(&unit),
                    ),
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
                            activation: (*activation).into(),
                        },
                    )?
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
                let dims = [("M", m.clone()), ("N", n.clone())];
                let binding = self.bindings.clamp(index)?;
                let x = self.value(*x, &[m, n])?;
                let minimum = self.weight(*minimum, &[Dim::from(1u64)])?;
                let maximum = self.weight(*maximum, &[Dim::from(1u64)])?;
                self.graph
                    .enqueue::<vision_clamp::Entry>(
                        binding,
                        &dims,
                        vision_clamp::WorkflowArgs {
                            x: (&x).into(),
                            minimum: (&minimum).into(),
                            maximum: (&maximum).into(),
                        },
                    )?
                    .value
            }
            VisionOperands::Attention {
                query,
                key,
                value,
                query_norm,
                key_norm,
                value_norm,
                span,
                log_base,
                epsilon,
                unit_scale,
            } => {
                let (kv_heads, group, quarter) = (statics("KV")?, statics("G")?, statics("P")?);
                let (heads, width) = (&kv_heads * &group, &quarter * 4u64);
                let dims = [
                    ("M", rows.clone()),
                    ("KV", kv_heads.clone()),
                    ("G", group),
                    ("P", quarter),
                    ("NQ", statics("NQ")?),
                    ("NV", statics("NV")?),
                    ("WS", statics("WS")?),
                ];
                let binding = self.bindings.attention(index)?;
                if self.coordinates.is_none() {
                    self.coordinates = Some(self.graph.input_for(binding, "coordinates", &dims)?);
                }
                // A row reads the keys of its span: its window's rows, or
                // its image's, which excludes a larger class's padding rows.
                let spans = match span {
                    VisionAttentionSpan::Window => &mut self.window_spans,
                    VisionAttentionSpan::Full => &mut self.image_spans,
                };
                if spans.is_none() {
                    *spans = Some(self.graph.input_for(binding, "spans", &dims)?);
                }
                let spans = input_view(spans, &[Dim::from(1u64), rows.clone(), Dim::from(2u64)])?;
                let query = self.value(*query, &[rows.clone(), heads, width.clone()])?;
                let key = self.value(*key, &[rows.clone(), kv_heads.clone(), width.clone()])?;
                let value = self.value(*value, &[rows.clone(), kv_heads, width.clone()])?;
                let query_norm = self.rows_of(*query_norm, &[width.clone()])?;
                let key_norm = self.rows_of(*key_norm, &[width.clone()])?;
                let value_norm = if *value_norm {
                    self.constant(&Constant::Unit(width.clone()))?
                        .reshape(&[Dim::from(1u64), width])
                } else {
                    self.empty(Constant::EmptyF32, &[width])?
                };
                let coordinates = input_view(&self.coordinates, &[rows, Dim::from(2u64)])?;
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
                            log_base: log_base.f32(),
                            epsilon: epsilon.f32(),
                            unit_scale: i32::from(*unit_scale).into(),
                        },
                    )?
                    .value
            }
            VisionOperands::GatedConvolution {
                value,
                gate,
                kernel: weights,
            } => {
                let (h, taps) = (statics("H")?, statics("T")?);
                let dims = [("M", rows.clone()), ("H", h.clone()), ("T", taps.clone())];
                let binding = self.bindings.gated_convolution(index)?;
                if self.neighbors.is_none() {
                    self.neighbors = Some(self.graph.input_for(binding, "neighbors", &dims)?);
                }
                let neighbors = input_view(&self.neighbors, &[rows.clone(), taps.clone()])?;
                let value = self.value(*value, &[rows.clone(), h.clone()])?;
                let gate = self.value(*gate, &[rows, h.clone()])?;
                let weights = self.weight(*weights, &[h, taps])?;
                self.graph
                    .enqueue::<vision_gated_convolution::Entry>(
                        binding,
                        &dims,
                        vision_gated_convolution::WorkflowArgs {
                            value: (&value).into(),
                            gate: (&gate).into(),
                            weight: (&weights).into(),
                            neighbors: (&neighbors).into(),
                        },
                    )?
                    .value
            }
            VisionOperands::Pool {
                source,
                standard,
                weight,
                scale,
            } => {
                let (g, h) = (statics("G")?, statics("H")?);
                let cells = self.cells.clone();
                let dims = [
                    ("M", cells.clone()),
                    ("G", g.clone()),
                    ("H", h.clone()),
                    ("NS", statics("NS")?),
                ];
                let binding = self.bindings.pool(index)?;
                let source = self.value(*source, &[cells, g, h.clone()])?;
                let standard_bias = self.rows_of(standard.map(|(bias, _)| bias), &[h.clone()])?;
                let standard_scale = self.rows_of(standard.map(|(_, scale)| scale), &[h])?;
                self.graph
                    .enqueue::<vision_pool::Entry>(
                        binding,
                        &dims,
                        vision_pool::WorkflowArgs {
                            source: (&source).into(),
                            standard_bias: (&standard_bias).into(),
                            standard_scale: (&standard_scale).into(),
                            weight: weight.f32(),
                            scale: scale.f32(),
                        },
                    )?
                    .value
            }
            VisionOperands::Position { source, table } => {
                let (value, blend) = self.blend(index, *source, *table)?;
                self.indices = Some(blend);
                value
            }
            VisionOperands::Registers { source, table } => {
                let (value, blend) = self.blend(index, *source, *table)?;
                self.register_blend = Some(blend);
                value
            }
            VisionOperands::PostNormResidual {
                residual,
                projected,
                norm,
                epsilon,
            } => {
                let width = self.width(*projected)?;
                let dims = [("M", rows.clone()), ("O", rows.clone()), ("D", width.clone())];
                let binding = self.bindings.post_norm(index)?;
                let residual = self.value(*residual, &[rows.clone(), width.clone()])?;
                let projected = self.value(*projected, &[rows, width.clone()])?;
                let norm = self.weight(*norm, &[width])?;
                let out_rows = self.constant(&Constant::Identity)?.clone();
                self.graph
                    .enqueue::<post_norm_residual::Entry>(
                        binding,
                        &dims,
                        post_norm_residual::WorkflowArgs {
                            residual: (&residual).into(),
                            projected: (&projected).into(),
                            norm: (&norm).into(),
                            out_rows: (&out_rows).into(),
                            epsilon: epsilon.f32(),
                            scale: 1.0f32.into(),
                        },
                    )?
                    .value
            }
        };
        Ok(result)
    }
}

/// The patch rows of every class a load prepares vision graphs for: the
/// cell classes of images of at most `max_image_cells` cells, in patch rows.
pub(crate) fn image_patch_classes(
    max_image_cells: usize,
    description: &VisionDescription,
) -> Result<Vec<u64>, String> {
    let cell = description.cell_rows();
    let classes = magnitude_batching::image_cell_classes(max_image_cells)
        .into_iter()
        .map(|cells| {
            (cells as u64)
                .checked_mul(cell)
                .ok_or_else(|| "vision patch class overflows".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if classes.is_empty() {
        return Err("the load admits no image cells".into());
    }
    Ok(classes)
}

/// A graph input viewed with `extents`.
fn input_view(port: &Option<NativePort>, extents: &[Dim]) -> Result<WorkflowTensorView, GraphError> {
    port.as_ref()
        .map(|port| port.tensor().reshape(extents))
        .ok_or_else(|| "a vision graph input is read before it is created".into())
}

/// The tower rows (the register rows, then the patches) and merged cells of
/// a class of `patches` patch rows.
fn class_rows(registers: u64, cell_rows: u64, patches: u64) -> Result<(u64, u64), GraphError> {
    let rows = patches
        .checked_add(registers)
        .ok_or("vision tower rows overflow")?;
    Ok((rows, patches / cell_rows))
}

/// The graph of `topology` at `numbers` over the weights `planned`, its
/// kernels bound by `bindings`.
fn vision_graph_draft<G: GraphDraft, B: VisionBindings<G>, W: PlannedWeights + ?Sized>(
    mut graph: G,
    bindings: &B,
    topology: &VisionTopology,
    numbers: &VisionNumbers<Dim>,
    planned: &W,
) -> Result<(G, VisionGraphPorts), GraphError> {
    let rows = numbers.rows.clone();
    let cells = numbers.cells.clone();
    let patches = rows
        .checked_sub(&numbers.registers)
        .ok_or("the vision class has fewer rows than registers")?;
    if patches.is_zero() || !(&cells * &numbers.cell_rows).equals(&patches) {
        return Err("the vision patch class is not a positive cell multiple".into());
    }
    let constants = Constant::of(topology, numbers)?
        .into_iter()
        .map(|constant| {
            let drawn = constant.draw(&mut graph, &rows)?;
            Ok((constant, drawn))
        })
        .collect::<Result<Vec<_>, GraphError>>()?;
    let mut draft = Draft {
        graph,
        bindings,
        topology,
        numbers,
        planned,
        cells,
        rows,
        values: Vec::with_capacity(topology.ops.len()),
        pixels: None,
        indices: None,
        register_blend: None,
        coordinates: None,
        window_spans: None,
        image_spans: None,
        order: None,
        neighbors: None,
        weights: Vec::new(),
        constants,
    };
    for index in 0..topology.ops.len() {
        let result = draft.enqueue(index)?;
        draft.values.push(result);
    }
    let features = draft
        .values
        .last()
        .cloned()
        .ok_or("the vision program has no invocation")?;
    draft.graph.export(&features)?;
    let (pixels, (indices, coefficients)) = draft
        .pixels
        .zip(draft.indices)
        .ok_or("the vision program reads no pixels or positions")?;
    if draft.coordinates.is_some() != topology.coordinates
        || draft.window_spans.is_some() != topology.windows
        || draft.order.is_some() != topology.windows
        || draft.register_blend.is_some() != !numbers.registers.is_zero()
        || draft.neighbors.is_some() != topology.convolution
    {
        return Err("the vision program's inputs disagree with its invocations".into());
    }
    Ok((
        draft.graph,
        VisionGraphPorts {
            pixels,
            indices,
            coefficients,
            register_blend: draft.register_blend,
            coordinates: draft.coordinates,
            window_spans: draft.window_spans,
            image_spans: draft.image_spans,
            order: draft.order,
            neighbors: draft.neighbors,
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

/// Single-owner per-kind composition for a vision encoder family.
///
/// A vision tower is a sequential pipeline `stem -> blocks[0..L] -> merger`.
/// Identical consecutive blocks (and periodic block patterns) are collapsed so
/// each distinct stage/block kind is drafted and sized once while preserving
/// exact inter-layer carry lifetimes, R_b colouring parity, upload/output/sync
/// storage, full-depth `GraphCensus`, binding constants, host heap, and kernel
/// requests.
pub(crate) struct VisionFamilyComposition {
    pub(crate) resources: CheckedGraphResources,
    pub(crate) kernels: HashSet<seismic::KernelRequest>,
    /// Per-patch-class storage bytes, in `patch_rows` order.
    pub(crate) class_storage: Vec<NativeGraphStorageBytes>,
}

struct VisionStages {
    stem: Range<usize>,
    blocks: Vec<Range<usize>>,
    merger: Range<usize>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum RelativeValue {
    Pixels,
    Carry,
    Local(usize),
    PatchCarry,
    PatchLocal(usize),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum RelativeBounds {
    Stored,
    Fixed { minimum: u32, maximum: u32 },
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum RelativeOperands {
    PatchStem {
        has_next_frame: bool,
        has_bias: bool,
        has_table: bool,
    },
    Norm {
        source: RelativeValue,
        rows: VisionRows,
        has_weight: bool,
        has_bias: bool,
        gathered: bool,
        epsilon_bits: u32,
        centered: bool,
        interleave: bool,
        activation: i32,
    },
    Linear {
        x: RelativeValue,
        rows: VisionRows,
        has_bias: bool,
        residual: Option<RelativeValue>,
        gate: Option<RelativeValue>,
        bounds: Option<RelativeBounds>,
        activation: i32,
    },
    Clamp {
        x: RelativeValue,
        rows: VisionRows,
    },
    Attention {
        query: RelativeValue,
        key: RelativeValue,
        value: RelativeValue,
        has_query_norm: bool,
        has_key_norm: bool,
        value_norm: bool,
        span: VisionAttentionSpan,
        log_base_bits: u32,
        epsilon_bits: u32,
        unit_scale: bool,
    },
    GatedConvolution {
        value: RelativeValue,
        gate: RelativeValue,
    },
    Pool {
        source: RelativeValue,
        has_standard: bool,
        weight_bits: u32,
        scale_bits: u32,
    },
    Position {
        source: RelativeValue,
    },
    Registers {
        source: RelativeValue,
    },
    PostNormResidual {
        residual: RelativeValue,
        projected: RelativeValue,
        epsilon_bits: u32,
    },
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct BlockOpSignature {
    kernel: VisionKernel,
    width: u64,
    operands: RelativeOperands,
    weight_specs: Vec<(Element, Vec<u64>)>,
}

fn for_each_operand_value(operands: &VisionOperands, mut f: impl FnMut(VisionValue)) {
    match operands {
        VisionOperands::PatchStem { .. } => {}
        VisionOperands::Norm { source, .. }
        | VisionOperands::Pool { source, .. }
        | VisionOperands::Position { source, .. }
        | VisionOperands::Registers { source, .. } => f(*source),
        VisionOperands::Linear { x, residual, gate, .. } => {
            f(*x);
            if let Some(residual) = residual {
                f(*residual);
            }
            if let Some(gate) = gate {
                f(*gate);
            }
        }
        VisionOperands::Clamp { x, .. } => f(*x),
        VisionOperands::Attention { query, key, value, .. } => {
            f(*query);
            f(*key);
            f(*value);
        }
        VisionOperands::GatedConvolution { value, gate, .. } => {
            f(*value);
            f(*gate);
        }
        VisionOperands::PostNormResidual { residual, projected, .. } => {
            f(*residual);
            f(*projected);
        }
    }
}

fn remap_operand_values(
    operands: &VisionOperands,
    mut remap: impl FnMut(VisionValue) -> VisionValue,
) -> VisionOperands {
    match operands.clone() {
        VisionOperands::PatchStem { frame, next_frame, bias, table } => {
            VisionOperands::PatchStem { frame, next_frame, bias, table }
        }
        VisionOperands::Norm {
            source,
            rows,
            weight,
            bias,
            gathered,
            epsilon,
            centered,
            interleave,
            activation,
        } => VisionOperands::Norm {
            source: remap(source),
            rows,
            weight,
            bias,
            gathered,
            epsilon,
            centered,
            interleave,
            activation,
        },
        VisionOperands::Linear {
            x,
            rows,
            weight,
            bias,
            residual,
            gate,
            bounds,
            activation,
        } => VisionOperands::Linear {
            x: remap(x),
            rows,
            weight,
            bias,
            residual: residual.map(&mut remap),
            gate: gate.map(&mut remap),
            bounds,
            activation,
        },
        VisionOperands::Clamp { x, rows, minimum, maximum } => {
            VisionOperands::Clamp { x: remap(x), rows, minimum, maximum }
        }
        VisionOperands::Attention {
            query,
            key,
            value,
            query_norm,
            key_norm,
            value_norm,
            span,
            log_base,
            epsilon,
            unit_scale,
        } => VisionOperands::Attention {
            query: remap(query),
            key: remap(key),
            value: remap(value),
            query_norm,
            key_norm,
            value_norm,
            span,
            log_base,
            epsilon,
            unit_scale,
        },
        VisionOperands::GatedConvolution { value, gate, kernel } => {
            VisionOperands::GatedConvolution { value: remap(value), gate: remap(gate), kernel }
        }
        VisionOperands::Pool { source, standard, weight, scale } => {
            VisionOperands::Pool { source: remap(source), standard, weight, scale }
        }
        VisionOperands::Position { source, table } => {
            VisionOperands::Position { source: remap(source), table }
        }
        VisionOperands::Registers { source, table } => {
            VisionOperands::Registers { source: remap(source), table }
        }
        VisionOperands::PostNormResidual { residual, projected, norm, epsilon } => {
            VisionOperands::PostNormResidual {
                residual: remap(residual),
                projected: remap(projected),
                norm,
                epsilon,
            }
        }
    }
}

fn op_output_element(op: &VisionOp) -> Element {
    match op.kernel.entry {
        VisionEntry::Norm | VisionEntry::Linear => op
            .kernel
            .elements
            .iter()
            .find(|(name, _)| *name == "Y")
            .map(|(_, el)| *el)
            .unwrap_or_else(Element::f32),
        VisionEntry::Clamp | VisionEntry::Attention | VisionEntry::GatedConvolution => op
            .kernel
            .elements
            .iter()
            .find(|(name, _)| *name == "A")
            .map(|(_, el)| *el)
            .unwrap_or_else(Element::f32),
        VisionEntry::PatchStem
        | VisionEntry::Pool
        | VisionEntry::Position
        | VisionEntry::PostNormResidual => Element::f32(),
    }
}

/// Partition `program.ops` into `stem`, `blocks[0..L]`, and `merger`, verifying
/// that every stage boundary carries only the single hidden tensor `h`.
fn partition_vision_stages(program: &VisionOps) -> Option<VisionStages> {
    let mut last_op: Vec<Option<usize>> = Vec::new();
    for (idx, op) in program.ops.iter().enumerate() {
        for role in op.operands.weights() {
            if let WeightScope::VisionBlock(block) = role.scope {
                let block = block as usize;
                if block >= last_op.len() {
                    last_op.resize(block + 1, None);
                }
                last_op[block] = Some(idx);
            }
        }
    }
    if last_op.is_empty() {
        return None;
    }
    let last_ops: Vec<usize> = last_op.into_iter().collect::<Option<Vec<_>>>()?;
    if last_ops.windows(2).any(|pair| pair[0] >= pair[1]) {
        return None;
    }
    // Block 0's operator output reads the stem's output `h` as its residual.
    let first_block0_residual = program
        .ops
        .iter()
        .take(last_ops[0] + 1)
        .enumerate()
        .find_map(|(idx, op)| {
            let is_block0 = op
                .operands
                .weights()
                .into_iter()
                .any(|role| role.scope == WeightScope::VisionBlock(0));
            if !is_block0 {
                return None;
            }
            match op.operands {
                VisionOperands::Linear { residual: Some(VisionValue::Result(r)), .. }
                | VisionOperands::PostNormResidual { residual: VisionValue::Result(r), .. }
                    if r < idx =>
                {
                    Some(r)
                }
                _ => None,
            }
        })?;
    let stem_end = first_block0_residual + 1;
    if stem_end == 0 || stem_end > last_ops[0] {
        return None;
    }
    let stem = 0..stem_end;
    let mut blocks = Vec::with_capacity(last_ops.len());
    let mut start = stem_end;
    for &end_inclusive in &last_ops {
        let end = end_inclusive + 1;
        if start >= end {
            return None;
        }
        blocks.push(start..end);
        start = end;
    }
    let merger = start..program.ops.len();
    if merger.is_empty() {
        return None;
    }
    // Verify that every stage boundary carries only the single hidden tensor `h`
    // of uniform width and element.
    let carry_op = &program.ops[stem.end - 1];
    let carry_width = carry_op.width;
    let carry_element = op_output_element(carry_op);
    for range in &blocks {
        let out_op = &program.ops[range.end - 1];
        if out_op.width != carry_width || op_output_element(out_op) != carry_element {
            return None;
        }
        for idx in range.clone() {
            let mut valid = true;
            for_each_operand_value(&program.ops[idx].operands, |value| match value {
                VisionValue::Result(src) | VisionValue::PatchRows(src) => {
                    if src != range.start - 1 && !(range.start <= src && src < idx) {
                        valid = false;
                    }
                }
                VisionValue::Pixels => valid = false,
            });
            if !valid {
                return None;
            }
        }
    }
    for idx in merger.clone() {
        let mut valid = true;
        for_each_operand_value(&program.ops[idx].operands, |value| match value {
            VisionValue::Result(src) | VisionValue::PatchRows(src) => {
                if src != merger.start - 1 && !(merger.start <= src && src < idx) {
                    valid = false;
                }
            }
            VisionValue::Pixels => valid = false,
        });
        if !valid {
            return None;
        }
    }
    Some(VisionStages { stem, blocks, merger })
}

fn relative_value(value: VisionValue, stage_start: usize) -> Option<RelativeValue> {
    Some(match value {
        VisionValue::Pixels => RelativeValue::Pixels,
        VisionValue::Result(idx) if idx + 1 == stage_start => RelativeValue::Carry,
        VisionValue::Result(idx) if idx >= stage_start => RelativeValue::Local(idx - stage_start),
        VisionValue::PatchRows(idx) if idx + 1 == stage_start => RelativeValue::PatchCarry,
        VisionValue::PatchRows(idx) if idx >= stage_start => {
            RelativeValue::PatchLocal(idx - stage_start)
        }
        _ => return None,
    })
}

fn block_signature(
    ops: &[VisionOp],
    stage_start: usize,
    weights: &HashMap<WeightRole, (Element, &[u64])>,
) -> Option<Vec<BlockOpSignature>> {
    let mut sig = Vec::with_capacity(ops.len());
    for op in ops {
        let rel = |v| relative_value(v, stage_start);
        let operands = match &op.operands {
            VisionOperands::PatchStem { next_frame, bias, table, .. } => {
                RelativeOperands::PatchStem {
                    has_next_frame: next_frame.is_some(),
                    has_bias: bias.is_some(),
                    has_table: table.is_some(),
                }
            }
            VisionOperands::Norm {
                source,
                rows,
                weight,
                bias,
                gathered,
                epsilon,
                centered,
                interleave,
                activation,
            } => RelativeOperands::Norm {
                source: rel(*source)?,
                rows: *rows,
                has_weight: weight.is_some(),
                has_bias: bias.is_some(),
                gathered: *gathered,
                epsilon_bits: epsilon.to_bits(),
                centered: *centered,
                interleave: *interleave,
                activation: *activation,
            },
            VisionOperands::Linear {
                x,
                rows,
                bias,
                residual,
                gate,
                bounds,
                activation,
                ..
            } => RelativeOperands::Linear {
                x: rel(*x)?,
                rows: *rows,
                has_bias: bias.is_some(),
                residual: match residual {
                    Some(v) => Some(rel(*v)?),
                    None => None,
                },
                gate: match gate {
                    Some(v) => Some(rel(*v)?),
                    None => None,
                },
                bounds: bounds.map(|b| match b {
                    LinearBounds::Stored { .. } => RelativeBounds::Stored,
                    LinearBounds::Fixed { minimum, maximum } => {
                        RelativeBounds::Fixed { minimum, maximum }
                    }
                }),
                activation: *activation,
            },
            VisionOperands::Clamp { x, rows, .. } => {
                RelativeOperands::Clamp { x: rel(*x)?, rows: *rows }
            }
            VisionOperands::Attention {
                query,
                key,
                value,
                query_norm,
                key_norm,
                value_norm,
                span,
                log_base,
                epsilon,
                unit_scale,
            } => RelativeOperands::Attention {
                query: rel(*query)?,
                key: rel(*key)?,
                value: rel(*value)?,
                has_query_norm: query_norm.is_some(),
                has_key_norm: key_norm.is_some(),
                value_norm: *value_norm,
                span: *span,
                log_base_bits: log_base.to_bits(),
                epsilon_bits: epsilon.to_bits(),
                unit_scale: *unit_scale,
            },
            VisionOperands::GatedConvolution { value, gate, .. } => {
                RelativeOperands::GatedConvolution { value: rel(*value)?, gate: rel(*gate)? }
            }
            VisionOperands::Pool { source, standard, weight, scale } => RelativeOperands::Pool {
                source: rel(*source)?,
                has_standard: standard.is_some(),
                weight_bits: weight.to_bits(),
                scale_bits: scale.to_bits(),
            },
            VisionOperands::Position { source, .. } => {
                RelativeOperands::Position { source: rel(*source)? }
            }
            VisionOperands::Registers { source, .. } => {
                RelativeOperands::Registers { source: rel(*source)? }
            }
            VisionOperands::PostNormResidual { residual, projected, epsilon, .. } => {
                RelativeOperands::PostNormResidual {
                    residual: rel(*residual)?,
                    projected: rel(*projected)?,
                    epsilon_bits: epsilon.to_bits(),
                }
            }
        };
        let mut weight_specs = Vec::new();
        for role in op.operands.weights() {
            let &(element, shape) = weights.get(&role)?;
            weight_specs.push((element, shape.to_vec()));
        }
        sig.push(BlockOpSignature {
            kernel: op.kernel.clone(),
            width: op.width,
            operands,
            weight_specs,
        });
    }
    Some(sig)
}

/// Whether `block_ops` preserves R_b two-ended stack parity between its
/// incoming carry `h_in` and its outgoing carry `h_out`.
fn block_even_crossing_parity(block_ops: &[VisionOp], block_start: usize) -> bool {
    let n = block_ops.len();
    if n == 0 {
        return true;
    }
    // Index 0 is `h_in`, indices 1..=n are `block_ops[0..n]`.
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(n + 1);
    spans.push((1, 1));
    for k in 0..n {
        let start = (k + 1) * 2 + 1;
        spans.push((start, start));
    }
    for (k, op) in block_ops.iter().enumerate() {
        let consumer_end = (k + 1) * 2 + 1;
        for_each_operand_value(&op.operands, |value| match value {
            VisionValue::Result(src) | VisionValue::PatchRows(src) => {
                if src + 1 == block_start {
                    spans[0].1 = spans[0].1.max(consumer_end);
                } else if src >= block_start && src - block_start < n {
                    let slot = (src - block_start) + 1;
                    spans[slot].1 = spans[slot].1.max(consumer_end);
                }
            }
            VisionValue::Pixels => {}
        });
    }
    // `h_out` (index `n`) lives into the following block or merger.
    spans[n].1 = (n + 1) * 2 + 1;

    let crosses = |a: (usize, usize), b: (usize, usize)| {
        a.0 <= b.1
            && b.0 <= a.1
            && !(a.0 <= b.0 && b.1 <= a.1)
            && !(b.0 <= a.0 && a.1 <= b.1)
    };
    let mut colour: Vec<Option<u8>> = vec![None; n + 1];
    let mut queue = std::collections::VecDeque::new();
    colour[0] = Some(0);
    queue.push_back(0usize);
    while let Some(u) = queue.pop_front() {
        let cu = colour[u].unwrap();
        for v in 0..=n {
            if u != v && crosses(spans[u], spans[v]) {
                match colour[v] {
                    None => {
                        colour[v] = Some(cu ^ 1);
                        queue.push_back(v);
                    }
                    Some(cv) if cv == cu => return false,
                    Some(_) => {}
                }
            }
        }
    }
    !matches!(colour[n], Some(1))
}

/// Collapse runs of identical consecutive blocks (and periodic repetitions of
/// even-parity block patterns) so each distinct stage/block kind is drafted
/// and sized once. Returns `None` when no collapsing reduces the block count,
/// or `Some((collapsed_program, kept_block_count))`.
fn collapse_vision_program(
    program: &VisionOps,
    load: &ModelLoadPlan,
) -> Option<(VisionOps, usize, Vec<usize>)> {
    let stages = partition_vision_stages(program)?;
    if stages.blocks.len() <= 1 {
        return None;
    }
    let weights: HashMap<WeightRole, (Element, &[u64])> = load
        .weights()
        .filter(|plan| matches!(plan.role.kind, WeightKind::Vision(_)))
        .map(|plan| (plan.role, (plan.resident, plan.shape.as_slice())))
        .collect();

    let mut kind_ids: HashMap<Vec<BlockOpSignature>, usize> = HashMap::new();
    let mut kind_even_parity: Vec<bool> = Vec::new();
    let mut block_kinds: Vec<usize> = Vec::with_capacity(stages.blocks.len());
    for range in &stages.blocks {
        let sig = block_signature(&program.ops[range.clone()], range.start, &weights)?;
        let kind = match kind_ids.get(&sig) {
            Some(&id) => id,
            None => {
                let id = kind_even_parity.len();
                let even = block_even_crossing_parity(&program.ops[range.clone()], range.start);
                kind_even_parity.push(even);
                kind_ids.insert(sig, id);
                id
            }
        };
        block_kinds.push(kind);
    }

    // Group consecutive identical blocks into runs and collapse each run while
    // preserving R_b crossing parity.
    let mut collapsed: Vec<(usize, usize)> = Vec::new(); // (kind, representative_block_idx)
    let mut i = 0;
    while i < block_kinds.len() {
        let kind = block_kinds[i];
        let start = i;
        while i < block_kinds.len() && block_kinds[i] == kind {
            i += 1;
        }
        let count = i - start;
        let kept = if count == 1 {
            1
        } else if kind_even_parity[kind] {
            // Keep 2 blocks for any run of length >= 2 so `prev -> B`,
            // `B -> B`, and `B -> next` transitions are all preserved (e.g. a
            // sandwich post-norm block whose carry uses `"O"` rather than the
            // stem's sliced `"M"`).
            2
        } else if count % 2 == 0 {
            2
        } else {
            3.min(count)
        };
        for offset in 0..kept {
            collapsed.push((kind, start + offset));
        }
    }

    // Collapse periodic repetitions of an even-parity multi-block pattern,
    // keeping 2 periods so the wrap-around transition `last -> first` is also
    // preserved alongside `stem -> first` and `last -> merger`.
    let m = collapsed.len();
    for period in 2..=(m / 2) {
        if m % period != 0 || m <= period * 2 {
            continue;
        }
        let repeats = (0..m).all(|idx| collapsed[idx].0 == collapsed[idx % period].0);
        if repeats && collapsed[..period].iter().all(|&(k, _)| kind_even_parity[k]) {
            collapsed.truncate(period * 2);
            break;
        }
    }

    if collapsed.len() >= stages.blocks.len() {
        return None;
    }

    let mut ops = Vec::with_capacity(
        stages.stem.len()
            + collapsed.iter().map(|&(_, b)| stages.blocks[b].len()).sum::<usize>()
            + stages.merger.len(),
    );
    ops.extend(program.ops[stages.stem.clone()].iter().cloned());
    let mut prev_out = ops.len() - 1;
    let remap = |value: VisionValue, old_start: usize, new_start: usize, prev_out: usize| match value {
        VisionValue::Pixels => VisionValue::Pixels,
        VisionValue::Result(idx) if idx + 1 == old_start => VisionValue::Result(prev_out),
        VisionValue::Result(idx) => VisionValue::Result(new_start + (idx - old_start)),
        VisionValue::PatchRows(idx) if idx + 1 == old_start => VisionValue::PatchRows(prev_out),
        VisionValue::PatchRows(idx) => VisionValue::PatchRows(new_start + (idx - old_start)),
    };
    let mut kind_first_start: Vec<Option<usize>> = vec![None; kind_ids.len()];
    for &(kind, block_idx) in &collapsed {
        let range = &stages.blocks[block_idx];
        let new_start = ops.len();
        if kind_first_start[kind].is_none() {
            kind_first_start[kind] = Some(new_start);
        }
        for op in &program.ops[range.clone()] {
            let mut cloned = op.clone();
            cloned.operands = remap_operand_values(&op.operands, |v| {
                remap(v, range.start, new_start, prev_out)
            });
            ops.push(cloned);
        }
        prev_out = ops.len() - 1;
    }
    let new_merger_start = ops.len();
    for op in &program.ops[stages.merger.clone()] {
        let mut cloned = op.clone();
        cloned.operands = remap_operand_values(&op.operands, |v| {
            remap(v, stages.merger.start, new_merger_start, prev_out)
        });
        ops.push(cloned);
    }
    let mut op_to_collapsed = Vec::with_capacity(program.ops.len());
    op_to_collapsed.extend(0..stages.stem.len());
    for (block_idx, range) in stages.blocks.iter().enumerate() {
        let kind = block_kinds[block_idx];
        let start = kind_first_start[kind].expect("every block kind is kept in collapsed");
        op_to_collapsed.extend(start..start + range.len());
    }
    op_to_collapsed.extend(new_merger_start..new_merger_start + stages.merger.len());
    debug_assert_eq!(op_to_collapsed.len(), program.ops.len());
    Some((VisionOps { ops, ..*program }, collapsed.len(), op_to_collapsed))
}

fn vision_entry_node_shape(
    entry: VisionEntry,
    backend: BackendName,
) -> Result<ContractNodeShape, GraphError> {
    let result = match entry {
        VisionEntry::PatchStem => vision_patch_stem::Entry::contract_node_shape(backend),
        VisionEntry::Norm => vision_norm::Entry::contract_node_shape(backend),
        VisionEntry::Linear => vision_linear::Entry::contract_node_shape(backend),
        VisionEntry::Clamp => vision_clamp::Entry::contract_node_shape(backend),
        VisionEntry::Attention => vision_attention::Entry::contract_node_shape(backend),
        VisionEntry::GatedConvolution => {
            vision_gated_convolution::Entry::contract_node_shape(backend)
        }
        VisionEntry::Pool => vision_pool::Entry::contract_node_shape(backend),
        VisionEntry::Position => vision_position::Entry::contract_node_shape(backend),
        VisionEntry::PostNormResidual => post_norm_residual::Entry::contract_node_shape(backend),
    };
    result.map_err(|error| GraphError::Invalid(format!("vision entry {entry:?}: {error}")))
}

/// Exact `GraphCensus` over every op of the full unrolled vision program,
/// composed through `outline.compose_recorded` so active CUDA launches and
/// static non-CUDA launches share one backend-agnostic path.
fn full_recorded_census(
    backend: BackendName,
    outline: &SizingOutline,
    program: &VisionOps,
    op_to_collapsed: &[usize],
) -> Result<GraphCensus, GraphError> {
    let mut shapes: HashMap<VisionEntry, (usize, usize, usize)> = HashMap::new();
    let mut census_nodes = Vec::with_capacity(program.ops.len());
    for (op, &collapsed_idx) in program.ops.iter().zip(op_to_collapsed) {
        let entry = op.kernel.entry;
        let triple = match shapes.get(&entry) {
            Some(&triple) => triple,
            None => {
                let shape = vision_entry_node_shape(entry, backend)?;
                let triple = (shape.arguments, shape.words, shape.launches);
                shapes.insert(entry, triple);
                triple
            }
        };
        census_nodes.push((collapsed_idx, triple));
    }
    Ok(outline.compose_recorded(backend, &census_nodes, &[]))
}

impl VisionFamilyComposition {
    pub(crate) fn certify(
        backend: BackendName,
        load: &ModelLoadPlan,
        program: &VisionOps,
        patch_rows: &[u64],
    ) -> Result<Self, GraphError> {
        let collapsed = collapse_vision_program(program, load);
        let sized_program = collapsed.as_ref().map_or(program, |(c, _, _)| c);
        let family = VisionFamily::of(backend, load, sized_program, patch_rows)?;
        let (collapsed_storage, classes) = family.certify::<SizingOutline>()?;
        // Sync scratch is allocated disjointly per node across all layers; if
        // any vision kernel ever requests sync scratch while collapsed, size
        // the full unrolled program so `sync` remains exact.
        let (classes, op_to_collapsed) = if collapsed.is_some() && collapsed_storage.sync != 0 {
            let full_family = VisionFamily::of(backend, load, program, patch_rows)?;
            let (_, full_classes) = full_family.certify::<SizingOutline>()?;
            (full_classes, None)
        } else {
            (classes, collapsed.map(|(_, _, map)| map))
        };

        let mut class_storage = Vec::with_capacity(classes.len());
        for (outline, _) in &classes {
            let mut storage = outline.storage_bytes();
            if let Some(map) = &op_to_collapsed {
                storage.recorded = full_recorded_census(backend, outline, program, map)?;
            }
            class_storage.push(storage);
        }
        let storage = class_storage
            .iter()
            .copied()
            .fold(NativeGraphStorageBytes::default(), NativeGraphStorageBytes::most);

        let mut resources = CheckedGraphFamilyResources::new();
        let mut kernels = HashSet::new();
        // Each class's layout, a graph of its own; an encode runs on its own
        // lane, never in a decode step's launch.
        for &class_s in &class_storage {
            resources.include_graph(class_s.recorded, false);
        }
        if let Some((first, _)) = classes.first() {
            kernels.extend(first.kernels().iter().cloned());
        }
        for (&patches, (_, constants)) in patch_rows.iter().zip(&classes) {
            let (rows, _) = class_rows(program.registers, program.cell_rows, patches)?;
            resources.include(
                storage,
                constants
                    .iter()
                    .map(|constant| constant.at_rows(rows))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        Ok(Self {
            resources: resources.finish()?,
            kernels,
            class_storage,
        })
    }
}

/// Whether every kernel of `program` is one `plan` planned.
fn planned_by(program: &VisionOps, plan: &VisionProgramPlan) -> Result<(), GraphError> {
    match program.kernels().into_iter().find(|kernel| !plan.kernels().contains(kernel)) {
        Some(kernel) => Err(format!("vision kernel {kernel:?} is not planned").into()),
        None => Ok(()),
    }
}

/// The graph family's storage and the distinct constants its binding
/// uploads, for assessment.
#[cfg(test)]
pub(crate) fn checked_vision_family_resources(
    backend: BackendName,
    load: &ModelLoadPlan,
    description: &VisionDescription,
    plan: &VisionProgramPlan,
    patch_rows: impl IntoIterator<Item = u64>,
) -> Result<CheckedGraphResources, GraphError> {
    checked_vision_family_with_kernels(backend, load, description, plan, patch_rows)
        .map(|(resources, _)| resources)
}

pub(crate) fn checked_vision_family_with_kernels(
    backend: BackendName,
    load: &ModelLoadPlan,
    description: &VisionDescription,
    plan: &VisionProgramPlan,
    patch_rows: impl IntoIterator<Item = u64>,
) -> Result<(CheckedGraphResources, HashSet<seismic::KernelRequest>), GraphError> {
    let patch_rows = patch_rows.into_iter().collect::<Vec<_>>();
    let program = planned_program(description, load)?;
    planned_by(&program, plan)?;
    let composition = VisionFamilyComposition::certify(backend, load, &program, &patch_rows)?;
    Ok((composition.resources, composition.kernels))
}

/// A vision graph family: its trace key and the split it indexes (its
/// topology and reals, its numbers at its largest class and the weights it
/// reads), its topology over its reals, its numbers, its weights and its
/// classes' patch rows: what its recordings and its load's graphs alike are
/// built of.
struct VisionFamily {
    topology: VisionTopology<f64>,
    numbers: VisionNumbers<u64>,
    weights: GraphWeights<u64>,
    patch_rows: Vec<u64>,
}

impl VisionFamily {
    fn of(
        backend: BackendName,
        load: &ModelLoadPlan,
        program: &VisionOps,
        patch_rows: &[u64],
    ) -> Result<Self, GraphError> {
        let cell = program.cell_rows;
        let &largest = patch_rows
            .iter()
            .max()
            .ok_or("the vision graph family has no patch class")?;
        if patch_rows.iter().any(|&rows| rows == 0 || rows % cell != 0)
            || patch_rows
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != patch_rows.len()
        {
            return Err("the vision graph patch classes are invalid or duplicated".into());
        }
        let (topology, numbers, weights) = split(backend, program, load, largest)?;
        Ok(Self { topology, numbers, weights, patch_rows: patch_rows.to_vec() })
    }

    fn certify<L: Certified>(
        &self,
    ) -> Result<(NativeGraphStorageBytes, Vec<(L, Vec<GraphConstant>)>), GraphError> {
        let topology = constant_reals(self.topology.clone());
        let weights = constant(self.weights.clone());
        let mut layouts = Vec::with_capacity(self.patch_rows.len());
        for &patches in &self.patch_rows {
            let numbers = self.class(patches)?;
            let (layout, mut all_constants) = L::certify_group(self.topology.backend, &[patches], |draft, _| {
                let (draft, ports) = vision_graph_draft(
                    draft,
                    &topology,
                    &topology,
                    &numbers,
                    &weights,
                )?;
                Ok((draft, ports.constants))
            })?;
            let constants = all_constants.pop().unwrap_or_default();
            layouts.push((layout, constants));
        }
        let storage = layouts
            .iter()
            .map(|(layout, _)| layout.storage_bytes())
            .fold(NativeGraphStorageBytes::default(), NativeGraphStorageBytes::most);
        Ok((storage, layouts))
    }

    /// The numbers of its class of `patches` patch rows, as a prepared
    /// graph's constants.
    fn class(&self, patches: u64) -> Result<VisionNumbers<Dim>, GraphError> {
        let (rows, cells) = class_rows(self.numbers.registers, self.numbers.cell_rows, patches)?;
        Ok(constant(VisionNumbers { rows, cells, ..self.numbers.clone() }))
    }
}

/// What `description`'s graph family on `backend` certifies to over
/// `patch_rows`: its storage and each class's layout or outline with the
/// constants its binding uploads.
#[cfg(test)]
#[allow(clippy::type_complexity)]
pub(crate) fn certified_vision_family<L: Certified>(
    backend: BackendName,
    load: &ModelLoadPlan,
    description: &VisionDescription,
    patch_rows: &[u64],
) -> Result<(NativeGraphStorageBytes, Vec<(L, Vec<GraphConstant>)>), GraphError> {
    let program = planned_program(description, load)?;
    VisionFamily::of(backend, load, &program, patch_rows)?.certify()
}

#[cfg(test)]
pub(crate) fn verify_vision_family_certificates(
    backend: BackendName,
    load: &ModelLoadPlan,
    description: &VisionDescription,
    plan: &VisionProgramPlan,
    patch_rows: &[u64],
) -> Result<(), GraphError> {
    let program = planned_program(description, load)?;
    planned_by(&program, plan)?;
    let family = VisionFamily::of(backend, load, &program, patch_rows)?;
    let (_, layouts) = family.certify::<NativeGraphLayout>()?;
    let topology = constant_reals(family.topology.clone());
    let weights = constant(family.weights.clone());
    for (&rows, (layout, _)) in patch_rows.iter().zip(&layouts) {
        let (graph, _) = vision_graph_draft(
            NativeGraphMetadata::new(backend),
            &topology,
            &topology,
            &family.class(rows)?,
            &weights,
        )?;
        let charged = graph
            .seal_with_layout(layout)
            .map_err(|error| GraphError::from(error).context(format!("vision class {rows}")))?;
        if charged != layout.storage_bytes() {
            return Err(format!("vision class {rows} charged a different layout").into());
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

    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.graphs.constant_bytes()
    }

    fn execute_graph(
        &self,
        core: &VisionLaunchCore,
        workspace: &mut NativeGraphWorkspaceLease,
        output: &mut Option<NativeGraphOutputLease>,
    ) -> Result<GraphOutputTensor, SubmitError> {
        let input = core.batch().input();
        let class = core.batch().class_patch_rows();
        let rows = u64::try_from(class).map_err(|_| invalid("vision patch rows exceed u64"))?;
        let spatial = input.spatial();
        let real = spatial.rows();
        let image_rows = i32::try_from(real).map_err(|_| invalid("vision rows exceed i32"))?;
        let (prepared, bound) = self.graphs.class(rows)?;
        // The tower's register rows lead: zero pixels, no position blend,
        // and one register table row each.
        let registers = usize::try_from(prepared.registers)
            .map_err(|_| invalid("vision register rows exceed usize"))?;
        let attended = i32::try_from(registers + real)
            .map_err(|_| invalid("vision rows exceed i32"))?;
        // The class's rows past the image's are padding: zero inputs, read
        // by no image row (every span lies within the image), and dropped
        // from the published features.
        let padding = class
            .checked_sub(real)
            .ok_or_else(|| invalid("vision image exceeds its class"))?;
        // A tower without a position table blends nothing: its blend inputs
        // have no rows.
        let (indices, coefficients) = match spatial.position_blend() {
            Some(blend) => (
                std::iter::repeat_n(0, registers * blend.indices.len())
                    .chain(
                        (0..real)
                            .flat_map(|row| blend.indices.iter().map(move |plane| plane[row])),
                    )
                    .chain(std::iter::repeat_n(0, padding * blend.indices.len()))
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>(),
                std::iter::repeat_n(0.0, registers * blend.coefficients.len())
                    .chain(
                        (0..real)
                            .flat_map(|row| blend.coefficients.iter().map(move |plane| plane[row])),
                    )
                    .chain(std::iter::repeat_n(0.0, padding * blend.coefficients.len()))
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
            None => (Vec::new(), Vec::new()),
        };
        // `attended` bounds the register rows.
        let register_indices = (0..registers as i32)
            .flat_map(|row| [row, 0, 0, 0])
            .chain(std::iter::repeat_n(0, class * 4))
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let register_coefficients = std::iter::repeat_n([1.0f32, 0.0, 0.0, 0.0], registers)
            .flatten()
            .chain(std::iter::repeat_n(0.0, class * 4))
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let pairs = |pairs: &[[i32; 2]], pad: [i32; 2]| {
            std::iter::repeat_n(&[0, 0], registers)
                .chain(pairs)
                .chain(std::iter::repeat_n(&pad, padding))
                .flatten()
                .copied()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>()
        };
        let coordinates = pairs(spatial.attention_coordinates(), [0, 0]);
        // A padding row reads the last image row's window, so a tile's key
        // walk stays within the image. (Windows take no register rows.)
        let windows = spatial.window_ranges();
        let window_spans = pairs(windows, windows.last().copied().unwrap_or([0, image_rows]));
        let image_spans = std::iter::repeat_n([0, attended], registers + class)
            .flatten()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        // Result row q of the merger reads the tower row that holds the
        // processor's patch row q; padding rows read themselves.
        let mut order = (0..class)
            .map(|row| i32::try_from(row).map_err(|_| invalid("vision rows exceed i32")))
            .collect::<Result<Vec<_>, _>>()?;
        for (tower_row, &source) in spatial.patch_order().iter().enumerate() {
            order[source] =
                i32::try_from(tower_row).map_err(|_| invalid("vision rows exceed i32"))?;
        }
        let order = order
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let neighbors = match prepared.convolution {
            Some(side) => {
                let [_, height, width] = input.grid();
                convolution_neighbors(
                    spatial.attention_coordinates(),
                    [height, width],
                    side,
                    class,
                )?
            }
            None => Vec::new(),
        };
        let pixels = input.pixels().data();
        let pixel_row = pixels.len() / real.max(1);
        let pixels = std::iter::repeat_n(0, registers * pixel_row)
            .chain(pixels.iter().copied())
            .chain(std::iter::repeat_n(0, padding * pixel_row))
            .collect::<Vec<_>>();
        let output = output
            .as_mut()
            .ok_or_else(|| invalid("vision graph output lease is absent"))?;
        Ok(prepared
            .run(
                rows,
                bound,
                workspace.slot_mut(),
                output,
                VisionGraphUploads {
                    pixels: &pixels,
                    indices: &indices,
                    coefficients: &coefficients,
                    register_indices: &register_indices,
                    register_coefficients: &register_coefficients,
                    coordinates: &coordinates,
                    window_spans: &window_spans,
                    image_spans: &image_spans,
                    order: &order,
                    neighbors: &neighbors,
                },
            )?
            .features)
    }
}

/// The tower row holding each of the `side`² taps (raster order of the
/// window centred on the row) of every row of a class of `class` rows, or
/// -1 past the image's edges and for padding rows. Image rows are placed on
/// the `[height, width]` patch grid by their rotary coordinates.
fn convolution_neighbors(
    coordinates: &[[i32; 2]],
    [height, width]: [usize; 2],
    side: u64,
    class: usize,
) -> Result<Vec<u8>, SubmitError> {
    let side = usize::try_from(side).map_err(|_| invalid("vision convolution side exceeds usize"))?;
    let reach = side / 2;
    let place = |[y, x]: [i32; 2]| {
        let (y, x) = (usize::try_from(y).ok()?, usize::try_from(x).ok()?);
        (y < height && x < width).then_some((y, x))
    };
    let mut grid = vec![-1i32; height * width];
    for (row, &coordinate) in coordinates.iter().enumerate() {
        let (y, x) = place(coordinate).ok_or_else(|| invalid("vision row lies off its grid"))?;
        grid[y * width + x] =
            i32::try_from(row).map_err(|_| invalid("vision rows exceed i32"))?;
    }
    let mut neighbors = Vec::with_capacity(class * side * side);
    for &coordinate in coordinates {
        let (y, x) = place(coordinate).ok_or_else(|| invalid("vision row lies off its grid"))?;
        for dy in 0..side {
            for dx in 0..side {
                let (ny, nx) = ((y + dy).checked_sub(reach), (x + dx).checked_sub(reach));
                neighbors.push(match (ny, nx) {
                    (Some(ny), Some(nx)) if ny < height && nx < width => grid[ny * width + nx],
                    _ => -1,
                });
            }
        }
    }
    neighbors.resize(class * side * side, -1);
    Ok(neighbors.into_iter().flat_map(i32::to_le_bytes).collect())
}

struct PreparedVisionGraph {
    patch_rows: u64,
    plan: NativeGraphPlan,
    ports: VisionGraphPorts,
    outputs: NativeGraphFamily,
}

pub struct PreparedVisionGraphs {
    variants: Vec<PreparedVisionGraph>,
    family: NativeGraphFamily,
    /// Register rows ahead of each image's patch rows.
    registers: u64,
    /// The kernel side of the tower's gated convolutions.
    convolution: Option<u64>,
}

pub(crate) struct BoundVisionGraphs {
    prepared: Rc<PreparedVisionGraphs>,
    variants: Vec<BoundNativeGraphPlan>,
    /// The uploaded graph constants, held with the plans that bind them.
    constants: Vec<Tensor>,
}

pub(crate) struct VisionGraphUploads<'a> {
    pub pixels: &'a [u8],
    pub indices: &'a [u8],
    pub coefficients: &'a [u8],
    pub register_indices: &'a [u8],
    pub register_coefficients: &'a [u8],
    pub coordinates: &'a [u8],
    pub window_spans: &'a [u8],
    pub image_spans: &'a [u8],
    pub order: &'a [u8],
    pub neighbors: &'a [u8],
}

pub(crate) struct VisionGraphResult {
    pub features: GraphOutputTensor,
}

impl PreparedVisionGraphs {
    pub(crate) fn prepare_classes(
        target_device: &Device,
        handles: &AttestedVision,
        load: &ModelLoadPlan,
        description: &VisionDescription,
        patch_rows: impl IntoIterator<Item = u64>,
    ) -> Result<Self, SubmitError> {
        let graph_error = |error: GraphError| invalid(error.to_string());
        let patch_rows = patch_rows.into_iter().collect::<Vec<_>>();
        let program = planned_program(description, load).map_err(graph_error)?;
        // Single-owner reservation and census via per-kind composition; the
        // sealed full-depth plans below are checked against it as an
        // independent witness.
        let composition = VisionFamilyComposition::certify(
            target_device.backend(),
            load,
            &program,
            &patch_rows,
        )
        .map_err(graph_error)?;
        let family = VisionFamily::of(target_device.backend(), load, &program, &patch_rows)
            .map_err(graph_error)?;
        let (_, layouts) = family.certify::<NativeGraphLayout>().map_err(graph_error)?;
        let layouts = layouts.into_iter().map(|(layout, _)| layout).collect::<Vec<_>>();
        let bindings = PreparedBindings { kernels: &handles.kernels, program: &program };
        let topology = constant_reals(family.topology.clone());
        let weights = constant(family.weights.clone());
        let mut variants = Vec::new();
        for ((rows, layout), &class_storage) in patch_rows
            .into_iter()
            .zip(&layouts)
            .zip(&composition.class_storage)
        {
            if variants
                .iter()
                .any(|variant: &PreparedVisionGraph| variant.patch_rows == rows)
            {
                return Err(invalid("vision graph patch class is duplicated"));
            }
            let (graph, ports) = vision_graph_draft(
                target_device.native_graph_with_layout(layout),
                &bindings,
                &topology,
                &family.class(rows).map_err(graph_error)?,
                &weights,
            )
            .map_err(graph_error)?;
            let plan = graph.seal().map_err(device)?;
            let class_graphs = seismic::LaunchGraphs::of(
                [seismic::LaunchedGraph {
                    graph: class_storage.recorded,
                    pinned_runs: 0,
                    runs: 1,
                }],
                Vec::new(),
            )
            .ok_or_else(|| invalid("vision graph census overflows"))?;
            let outputs = super::graph::serial_family(
                "vision output",
                [(false, plan.clone())],
                &class_graphs,
            )
            .map_err(device)?;
            super::graph::assert_storage_eq("vision output", &outputs, class_storage);
            variants.push(PreparedVisionGraph {
                patch_rows: rows,
                outputs,
                plan,
                ports,
            });
        }
        if variants.is_empty() {
            return Err(invalid("vision graph family has no patch class"));
        }
        let plans = variants.iter().map(|variant| (false, variant.plan.clone()));
        let family = super::graph::serial_family("vision", plans, &composition.resources.graphs)
            .map_err(device)?;
        super::graph::assert_storage_eq("vision", &family, composition.resources.storage);
        Ok(Self {
            variants,
            family,
            registers: program.registers,
            convolution: program.convolution,
        })
    }

    pub fn workspace_bytes_max(&self) -> u64 {
        self.family.workspace_bytes()
    }

    /// The bytes of the distinct graph constants a binding uploads
    /// ([`Self::bind_weights`]).
    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        distinct_storage_bytes(self.variants.iter().flat_map(|variant| variant.ports.constants.iter()))
    }

    pub fn output_bytes_max(&self) -> u64 {
        self.family.output_bytes()
    }

    pub fn family(&self) -> &NativeGraphFamily {
        &self.family
    }

    /// Each class's patch rows and its one-plan family: an encode's
    /// workspace and its image's output are of its own class.
    pub fn class_families(&self) -> impl Iterator<Item = (u64, &NativeGraphFamily)> {
        self.variants
            .iter()
            .map(|variant| (variant.patch_rows, &variant.outputs))
    }

    pub(crate) fn bind_weights(
        self: &Rc<Self>,
        charges: &seismic::Charges,
        resident: &ResidentVision,
    ) -> Result<BoundVisionGraphs, SubmitError> {
        let mut uploaded = ConstantTensors::new(charges);
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
            variants.push(variant.plan.bind_static(charges, &fixed).map_err(device)?);
        }
        Ok(BoundVisionGraphs {
            prepared: self.clone(),
            variants,
            constants: uploaded.into_tensors(),
        })
    }

    pub(crate) fn run(
        &self,
        patch_rows: u64,
        bound: &BoundNativeGraphPlan,
        slot: &mut NativeGraphFamilySlot,
        output: &mut NativeGraphOutputLease,
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
        let register_blend = ports.register_blend.as_ref();
        for (port, bytes) in [
            (Some(&ports.pixels), uploads.pixels),
            (Some(&ports.indices), uploads.indices),
            (Some(&ports.coefficients), uploads.coefficients),
            (register_blend.map(|(indices, _)| indices), uploads.register_indices),
            (
                register_blend.map(|(_, coefficients)| coefficients),
                uploads.register_coefficients,
            ),
            (ports.coordinates.as_ref(), uploads.coordinates),
            (ports.window_spans.as_ref(), uploads.window_spans),
            (ports.image_spans.as_ref(), uploads.image_spans),
            (ports.order.as_ref(), uploads.order),
            (ports.neighbors.as_ref(), uploads.neighbors),
        ] {
            if let Some(port) = port {
                active.write_input(port, bytes).map_err(device)?;
            }
        }
        let active_output = output
            .activate(&variant.plan)
            .map_err(SubmitError::Invariant)?;
        let active_output = active_output
            .run_outputs(|outputs| active.attach(bindings, outputs).and_then(super::run_graph))
            .map_err(device)?;
        let owner = active_output.publish();
        let features = owner
            .tensor(&ports.features)
            .ok_or_else(|| invalid("vision graph omitted retained features"))?;
        Ok(VisionGraphResult { features })
    }
}

impl BoundVisionGraphs {
    fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.constants.iter().try_fold(0u64, |bytes, tensor| {
            bytes
                .checked_add(tensor.storage_bytes())
                .ok_or("vision graph constant charge overflows")
        })
    }

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

#[cfg(test)]
mod tests {
    use super::convolution_neighbors;

    fn decode(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    /// A 4 × 4 patch grid held in 2 × 2 cell order: each row's 3 × 3 window
    /// (raster order) names the tower rows of its grid neighbours, -1 past
    /// the edges, and padding rows read nothing.
    #[test]
    fn neighbours_follow_the_grid_through_cell_order() {
        let mut coordinates = Vec::new();
        for cell_y in 0..2 {
            for cell_x in 0..2 {
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    coordinates.push([2 * cell_y + dy, 2 * cell_x + dx]);
                }
            }
        }
        let class = coordinates.len() + 2;
        let neighbors = decode(&convolution_neighbors(&coordinates, [4, 4], 3, class).unwrap());
        assert_eq!(neighbors.len(), class * 9);
        let window = |row: usize| &neighbors[row * 9..][..9];
        // (0, 0): only the lower-right 2 × 2 of its window is on the grid.
        assert_eq!(window(0), [-1, -1, -1, -1, 0, 1, -1, 2, 3]);
        // (1, 1): its lower-right neighbour (2, 2) opens the fourth cell.
        assert_eq!(window(3), [0, 1, 4, 2, 3, 6, 8, 9, 12]);
        assert!(neighbors[16 * 9..].iter().all(|tap| *tap == -1));
        assert!(convolution_neighbors(&[[4, 0]], [4, 4], 3, 1).is_err());
    }

    #[test]
    fn vision_family_composition_matches_unrolled_multi_layer_graphs() {
        use super::{
            VisionFamily, VisionFamilyComposition, class_rows, collapse_vision_program,
            planned_program,
        };
        use crate::programs::graph::sizing::SizingOutline;
        use crate::programs::native_constants::CheckedGraphFamilyResources;
        use crate::{ArtifactComponent, ArtifactComponentKind, ModelLoadPlan, WeightPlan};
        use magnitude_artifacts::ArtifactIdentity;
        use magnitude_family_contracts::{
            ActivationDType, CellReduction, MergerStage, PositionSampling, VisionActivation,
            VisionAttention, VisionAttentionScale, VisionAttentionSpan, VisionBlock,
            VisionDescription, VisionFeedForward, VisionLinear, VisionMerger, VisionNorm,
            VisionOperator, VisionPositions, VisionPreprocessing, VisionResampling, VisionResize,
            VisionStem, VisionUp, WeightDescriptor,
        };
        use seismic::{BackendName, Element, NativeGraphLayout};
        use std::collections::HashSet;

        let f32 = Element::f32();
        let stored = |name: &str, shape: &[u64]| WeightDescriptor::stored(name, shape);
        let linear = |name: &str, outputs: u64, inputs: u64| VisionLinear {
            weight: stored(&format!("{name}.weight"), &[outputs, inputs]),
            bias: Some(stored(&format!("{name}.bias"), &[outputs])),
            clamp: None,
        };
        let layer_norm = |name: &str, width: u64| VisionNorm::Layer {
            weight: stored(&format!("{name}.weight"), &[width]),
            bias: stored(&format!("{name}.bias"), &[width]),
            epsilon: 1e-6,
        };
        let rms_norm = |name: &str, width: u64| VisionNorm::Rms {
            weight: Some(stored(&format!("{name}.weight"), &[width])),
            epsilon: 1e-6,
        };
        let make_block = |prefix: &str, span: VisionAttentionSpan, sandwich: bool, gated: bool| {
            VisionBlock {
                operator_norm: layer_norm(&format!("{prefix}.ln1"), 128),
                operator: VisionOperator::Attention(VisionAttention {
                    heads: 1,
                    kv_heads: 1,
                    width: 128,
                    query: linear(&format!("{prefix}.q"), 128, 128),
                    key: linear(&format!("{prefix}.k"), 128, 128),
                    value: linear(&format!("{prefix}.v"), 128, 128),
                    query_norm: None,
                    key_norm: None,
                    value_norm: None,
                    rotary_base: Some(10000.0),
                    scale: VisionAttentionScale::InverseSqrtWidth,
                    span,
                    output: linear(&format!("{prefix}.o"), 128, 128),
                }),
                operator_post_norm: sandwich
                    .then(|| rms_norm(&format!("{prefix}.op_post"), 128)),
                feedforward_norm: layer_norm(&format!("{prefix}.ln2"), 128),
                feedforward: VisionFeedForward {
                    up: if gated {
                        VisionUp::Gated {
                            gate: linear(&format!("{prefix}.gate"), 256, 128),
                            up: linear(&format!("{prefix}.up"), 256, 128),
                            limit: Some(7.0),
                        }
                    } else {
                        VisionUp::Plain(linear(&format!("{prefix}.up"), 256, 128))
                    },
                    activation: VisionActivation::GeluTanh,
                    down: linear(&format!("{prefix}.down"), 128, 256),
                },
                feedforward_post_norm: sandwich
                    .then(|| rms_norm(&format!("{prefix}.ff_post"), 128)),
            }
        };

        let base_preprocessing = VisionPreprocessing {
            resize: VisionResize::PixelBounds { min_pixels: 16, max_pixels: 256 },
            resampling: VisionResampling::Bicubic,
            mean: [0.5; 3],
            std: [0.5; 3],
            channels: 3,
            patch: 2,
            merge: 2,
        };
        let base_stem = VisionStem::Patch {
            frames: vec![
                stored("patch.0", &[128, 3, 2, 2]),
                stored("patch.1", &[128, 3, 2, 2]),
            ],
            bias: Some(stored("patch.bias", &[128])),
            positions: Some(VisionPositions {
                table: stored("position", &[16, 128]),
                sampling: PositionSampling::AlignedCorners { side: 4 },
            }),
            norm: None,
            registers: None,
        };
        let base_merger = VisionMerger {
            norm: Some(layer_norm("post", 128)),
            reduction: CellReduction::Concatenate,
            standardize: None,
            projection_norm: None,
            stages: vec![
                MergerStage {
                    norm: None,
                    linear: linear("mm.0", 512, 512),
                    activation: Some(VisionActivation::GeluErf),
                },
                MergerStage {
                    norm: None,
                    linear: linear("mm.2", 128, 512),
                    activation: None,
                },
            ],
            feed_forward: None,
            output_norm: None,
        };

        // 1. 27-layer uniform vision tower (Qwen3.5-shaped): collapses 27 -> 2 blocks.
        let uniform_27 = VisionDescription {
            activation_dtype: ActivationDType::BF16,
            hidden: 128,
            output_hidden: 128,
            preprocessing: base_preprocessing.clone(),
            stem: base_stem.clone(),
            window: None,
            blocks: (0..27)
                .map(|i| make_block(&format!("b{i}"), VisionAttentionSpan::Full, false, false))
                .collect(),
            merger: base_merger.clone(),
        };

        // 2. 16-layer mixed Window + Full attention tower (Qwen2.5-VL-shaped):
        //    collapses 16 -> 6 blocks (`[Window, Window, Full] * 2`).
        let mixed_16 = VisionDescription {
            activation_dtype: ActivationDType::BF16,
            hidden: 128,
            output_hidden: 128,
            preprocessing: base_preprocessing.clone(),
            stem: base_stem.clone(),
            window: Some(4),
            blocks: (0..16)
                .map(|i| {
                    let span = if (i + 1) % 4 == 0 {
                        VisionAttentionSpan::Full
                    } else {
                        VisionAttentionSpan::Window
                    };
                    make_block(&format!("m{i}"), span, false, false)
                })
                .collect(),
            merger: base_merger.clone(),
        };

        // 3. 12-layer sandwich post-norm + register rows + gated FFN with fixed bounds:
        //    exercises row-dependent `Identity` constant, `Bound` constants, `register_blend`,
        //    and `PostNormResidual`. Collapses 12 -> 2 blocks.
        let sandwich_12 = VisionDescription {
            activation_dtype: ActivationDType::BF16,
            hidden: 128,
            output_hidden: 128,
            preprocessing: base_preprocessing,
            stem: VisionStem::Patch {
                frames: vec![stored("patch.0", &[128, 3, 2, 2])],
                bias: None,
                positions: Some(VisionPositions {
                    table: stored("position", &[16, 128]),
                    sampling: PositionSampling::CroppedSquare { side: 4 },
                }),
                norm: None,
                registers: Some(stored("registers", &[3, 128])),
            },
            window: None,
            blocks: (0..12)
                .map(|i| make_block(&format!("s{i}"), VisionAttentionSpan::Full, true, true))
                .collect(),
            merger: base_merger,
        };

        let patch_rows = [4u64, 8, 16];
        let cases = [
            ("uniform_27", uniform_27, 2usize),
            ("mixed_16", mixed_16, 6usize),
            ("sandwich_12", sandwich_12, 2usize),
        ];

        for (label, description, expected_collapsed_blocks) in cases {
            description.validate().unwrap_or_else(|e| panic!("{label}: invalid fixture: {e}"));
            let component = ArtifactComponent {
                kind: ArtifactComponentKind::Projector,
                identity: ArtifactIdentity([9; 32]),
            };
            let weights = description
                .weights()
                .into_iter()
                .map(|(role, descriptor)| {
                    let bytes = descriptor.shape.iter().product::<u64>() * 4;
                    WeightPlan {
                        role,
                        component,
                        source: f32,
                        upload: f32,
                        resident: f32,
                        shape: descriptor.shape.clone(),
                        descriptor: descriptor.clone(),
                        source_bytes: bytes,
                        resident_bytes: bytes,
                        scale: None,
                        bound: 0,
                        holding: crate::Holding::Whole,
                    }
                })
                .collect();
            let load = ModelLoadPlan::from_vision(weights);
            let program = planned_program(&description, &load).unwrap();

            // Verify collapsing actually reduced the block count as expected.
            let (_, kept_blocks, _) = collapse_vision_program(&program, &load)
                .unwrap_or_else(|| panic!("{label}: expected vision program to collapse"));
            assert_eq!(
                kept_blocks, expected_collapsed_blocks,
                "{label}: collapsed block count"
            );

            let composed =
                VisionFamilyComposition::certify(BackendName::Cpu, &load, &program, &patch_rows)
                    .unwrap();

            // Reference 1: full unrolled `SizingOutline`.
            let full_family = VisionFamily::of(BackendName::Cpu, &load, &program, &patch_rows).unwrap();
            let (ref_storage, ref_outlines) = full_family.certify::<SizingOutline>().unwrap();
            let mut ref_resources = CheckedGraphFamilyResources::new();
            let mut ref_kernels = HashSet::new();
            for (outline, _) in &ref_outlines {
                ref_resources.include_graph(outline.storage_bytes().recorded, false);
                ref_kernels.extend(outline.kernels().iter().cloned());
            }
            for (&patches, (_, constants)) in patch_rows.iter().zip(&ref_outlines) {
                let (rows, _) = class_rows(program.registers, program.cell_rows, patches).unwrap();
                ref_resources.include(
                    ref_storage,
                    constants
                        .iter()
                        .map(|constant| constant.at_rows(rows))
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap(),
                );
            }
            let ref_resources = ref_resources.finish().unwrap();

            assert_eq!(
                composed.resources, ref_resources,
                "{label}: VisionFamilyComposition resources must match full unrolled SizingOutline"
            );
            assert_eq!(
                composed.kernels, ref_kernels,
                "{label}: VisionFamilyComposition kernels must match full unrolled SizingOutline"
            );
            for (idx, (outline, _)) in ref_outlines.iter().enumerate() {
                assert_eq!(
                    composed.class_storage[idx],
                    outline.storage_bytes(),
                    "{label}: class {idx} storage must match full unrolled SizingOutline"
                );
            }

            // Reference 2: full unrolled `NativeGraphLayout`.
            let (layout_storage, ref_layouts) =
                full_family.certify::<NativeGraphLayout>().unwrap();
            assert_eq!(
                composed.resources.storage, layout_storage,
                "{label}: VisionFamilyComposition storage must match full unrolled NativeGraphLayout"
            );
            for (idx, (layout, _)) in ref_layouts.iter().enumerate() {
                assert_eq!(
                    composed.class_storage[idx],
                    layout.storage_bytes(),
                    "{label}: class {idx} storage must match full unrolled NativeGraphLayout"
                );
            }
        }
    }
}
