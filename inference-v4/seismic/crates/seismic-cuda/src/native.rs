//! Encoding boundary for one fully resolved CUDA launch.

use crate::physical::{CudaDialect, CudaResolvedLayout, CudaResolvedView, CudaValueType};
use seismic_compiler::pipeline::{EncodedPlan, EncodedScheduleItem};
use seismic_compiler::terminal::{
    ConditionalCase, ResolvedScalarInstruction, ResolvedScalarInstructionKind as I, ScalarIndex,
    ScalarLiteral, ScalarValueId, StorageOperation,
};
use seismic_lang::{
    intrinsics::Operation,
    sir::Math,
    syntax::ast::{AssignOp, BinaryOp, UnaryOp},
    types::DType,
};
use seismic_realization::executable::{
    AbiRole, AccessMode, BarrierScope, ResolvedAxisMap, ResolvedBindingId, ResolvedGridMapping,
    ResolvedKernelStep, ResolvedLaunch, ResolvedLaunchId, ResolvedOperandTransport,
    ResolvedStorageId, ResolvedValueTransport,
};
use std::collections::BTreeMap;

pub const MAX_KERNEL_PARAMETER_BYTES: usize = 32_764;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaAbiAllocation {
    pub id: ResolvedStorageId,
    pub role: Option<AbiRole>,
    pub bytes: u64,
    pub alignment: u64,
    pub layout: CudaResolvedLayout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaBinding {
    pub id: ResolvedBindingId,
    pub slot: u32,
    pub allocation: ResolvedStorageId,
    pub access: AccessMode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaLaunch {
    pub id: ResolvedLaunchId,
    pub name: String,
    pub grid: [u64; 3],
    pub block: [u64; 3],
    pub bindings: Vec<CudaBinding>,
    pub target: crate::target::PtxTarget,
    pub launch_bound: [u64; 3],
    pub ptx: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Emitted {
    pub name: String,
    pub launches: Vec<CudaLaunch>,
    pub abi: Vec<CudaAbiAllocation>,
}

pub fn encode_launch(
    launch: &ResolvedLaunch<CudaDialect>,
    target: &crate::target::TargetProfile,
) -> Result<CudaLaunch, String> {
    let target = target
        .plan(crate::target::TargetRequirement::ScalarBaseline)
        .map_err(|error| error.to_string())?;
    let name = format!("seismic_launch_{}", launch.id.0);
    let block = launch.geometry.participants_per_workgroup;
    let grid = launch.geometry.workgroups;
    let bindings = launch
        .binding_groups
        .iter()
        .flat_map(|group| {
            group.members.iter().map(move |binding| CudaBinding {
                id: binding.id,
                slot: group.slot,
                allocation: binding.storage,
                access: binding.access,
            })
        })
        .collect::<Vec<_>>();
    if bindings.len() * std::mem::size_of::<u64>() > MAX_KERNEL_PARAMETER_BYTES {
        return Err("resolved CUDA launch exceeds kernel parameter ABI".into());
    }
    let ptx = encode_ptx(&name, target, block, &bindings, launch)?;
    Ok(CudaLaunch {
        id: launch.id,
        name,
        grid,
        block,
        bindings,
        target,
        launch_bound: block,
        ptx,
    })
}

fn encode_ptx(
    name: &str,
    target: crate::target::PtxTarget,
    block: [u64; 3],
    bindings: &[CudaBinding],
    launch: &ResolvedLaunch<CudaDialect>,
) -> Result<String, String> {
    let parameters = bindings
        .iter()
        .enumerate()
        .map(|(slot, _)| format!(".param .u64 __storage_{slot}"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut builder = Ptx::default();
    let mut storage = BTreeMap::new();
    for (slot, binding) in bindings.iter().enumerate() {
        let pointer = builder.r64();
        builder.push(format!("ld.param.u64 {pointer}, [__storage_{slot}];"));
        storage.insert(binding.allocation, pointer);
    }
    for step in launch.kernel.steps.iter() {
        match step {
            ResolvedKernelStep::MappedTask {
                mapping,
                bindings,
                instructions,
                ..
            } => {
                builder.task(mapping, bindings, instructions, launch, &storage)?;
            }
            ResolvedKernelStep::Barrier { scope, .. } => match scope {
                BarrierScope::Workgroup => {
                    builder.push("membar.cta;");
                    builder.push("bar.sync 0;");
                }
                BarrierScope::Subgroup => {
                    builder.push("membar.cta;");
                    builder.push("bar.warp.sync 0xffffffff;");
                }
            },
            ResolvedKernelStep::Publish { .. } => builder.push("membar.gl;"),
        }
    }
    let exact_math = builder
        .exact_math
        .then(|| {
            format!(
                "{}\n{}\n",
                include_str!("math/exp.ptx"),
                include_str!("math/portable_math.ptx")
            )
        })
        .unwrap_or_default();
    Ok(format!(
        ".version {}.{}\n.target {}\n.address_size 64\n{exact_math}.visible .entry {name}({parameters}) .maxntid {}, {}, {} {{\n{}\n{}\n  ret;\n}}\n",
        target.isa.major, target.isa.minor, target.architecture_spelling(),
        block[0], block[1], block[2], builder.declarations(), builder.body.join("\n")
    ))
}

#[derive(Clone)]
enum Value {
    Void,
    Scalar {
        register: String,
        dtype: DType,
    },
    Index(String),
    Pointer {
        register: String,
        dtype: DType,
    },
    Tensor {
        register: String,
        dtype: DType,
        shape: Vec<u64>,
        strides: Vec<u64>,
    },
    Tuple(Vec<Value>),
    Packed {
        shape: Vec<u64>,
        planes: Vec<(String, Value)>,
    },
    Range(String, String),
}

#[derive(Default)]
struct Ptx {
    r32: usize,
    r64: usize,
    f32: usize,
    pred: usize,
    labels: usize,
    params: Vec<String>,
    exact_math: bool,
    grid: [u64; 3],
    block: [u64; 3],
    body: Vec<String>,
    values: BTreeMap<ScalarValueId, Value>,
}

impl Ptx {
    fn r32(&mut self) -> String {
        let v = format!("%r{}", self.r32);
        self.r32 += 1;
        v
    }
    fn r64(&mut self) -> String {
        let v = format!("%rd{}", self.r64);
        self.r64 += 1;
        v
    }
    fn f32(&mut self) -> String {
        let v = format!("%f{}", self.f32);
        self.f32 += 1;
        v
    }
    fn pred(&mut self) -> String {
        let v = format!("%p{}", self.pred);
        self.pred += 1;
        v
    }
    fn label(&mut self) -> String {
        let v = format!("L{}", self.labels);
        self.labels += 1;
        v
    }
    fn push(&mut self, line: impl Into<String>) {
        self.body.push(format!("  {}", line.into()));
    }
    fn declarations(&self) -> String {
        format!("  .reg .b32 %r<{}>;\n  .reg .b64 %rd<{}>;\n  .reg .f32 %f<{}>;\n  .reg .pred %p<{}>;\n{}", self.r32.max(1), self.r64.max(1), self.f32.max(1), self.pred.max(1), self.params.join("\n"))
    }

    fn exact_math_call(&mut self, function: &str, argument: String) -> String {
        self.exact_math = true;
        let ordinal = self.params.len() / 2;
        let input = format!("__math_arg_{ordinal}");
        let output = format!("__math_result_{ordinal}");
        self.params.push(format!("  .param .b32 {input};"));
        self.params.push(format!("  .param .b32 {output};"));
        let result = self.f32();
        self.push(format!("st.param.f32 [{input}], {argument};"));
        self.push(format!("call.uni ({output}), {function}, ({input});"));
        self.push(format!("ld.param.f32 {result}, [{output}];"));
        result
    }

    fn task(
        &mut self,
        mapping: &seismic_realization::executable::ResolvedParticipantMap,
        bindings: &[ResolvedOperandTransport],
        instructions: &seismic_realization::executable::NonEmpty<
            crate::physical::CudaResolvedInstruction,
        >,
        launch: &ResolvedLaunch<CudaDialect>,
        pointers: &BTreeMap<ResolvedStorageId, String>,
    ) -> Result<(), String> {
        self.values.clear();
        self.grid = launch.geometry.workgroups;
        self.block = launch.geometry.participants_per_workgroup;
        let first = instructions.iter().next().expect("nonempty");
        for binding in bindings.iter() {
            let Some(value) = first.operand_values.get(&binding.operand).copied() else {
                continue;
            };
            let transported = self.transport_value(
                &binding.transport,
                &first.value_types[&value],
                launch,
                pointers,
                first.result_values.values().any(|result| *result == value),
            )?;
            self.values.insert(value, transported);
        }
        let mut active = Vec::new();
        let mut serial = Vec::new();
        let mut grid_stride = Vec::new();
        for axis in &mapping.axes {
            match axis {
                ResolvedAxisMap::Grid {
                    logical_axis,
                    workgroup_axis,
                    participant_axis,
                    mode,
                } => {
                    let group = self.r32();
                    let lane = self.r32();
                    let width = self.r32();
                    let index = self.r64();
                    self.push(format!(
                        "mov.u32 {group}, %ctaid.{};",
                        axis_name(*workgroup_axis)?
                    ));
                    self.push(format!(
                        "mov.u32 {lane}, %tid.{};",
                        axis_name(*participant_axis)?
                    ));
                    self.push(format!(
                        "mov.u32 {width}, %ntid.{};",
                        axis_name(*participant_axis)?
                    ));
                    let group64 = self.r64();
                    let lane64 = self.r64();
                    let width64 = self.r64();
                    self.push(format!("cvt.u64.u32 {group64}, {group};"));
                    self.push(format!("cvt.u64.u32 {lane64}, {lane};"));
                    self.push(format!("cvt.u64.u32 {width64}, {width};"));
                    self.push(format!(
                        "mad.lo.u64 {index}, {group64}, {width64}, {lane64};"
                    ));
                    if let ResolvedGridMapping::GridStride { stride } = mode {
                        let base = self.r64();
                        self.push(format!("mov.u64 {base}, {index};"));
                        grid_stride.push((index.clone(), base, *stride, *logical_axis as usize));
                    }
                    for binder in &first.axis_binders[*logical_axis as usize] {
                        self.values.insert(*binder, Value::Index(index.clone()));
                    }
                    if mapping.masks_inactive_participants {
                        let p = self.pred();
                        self.push(format!(
                            "setp.ge.u64 {p}, {index}, {};",
                            mapping.logical_extents[*logical_axis as usize]
                        ));
                        active.push(p);
                    }
                }
                ResolvedAxisMap::SubgroupLane { logical_axis } => {
                    let lane = self.r32();
                    self.push(format!("mov.u32 {lane}, %laneid;"));
                    let index = self.r64();
                    self.push(format!("cvt.u64.u32 {index}, {lane};"));
                    for binder in &first.axis_binders[*logical_axis as usize] {
                        self.values.insert(*binder, Value::Index(index.clone()));
                    }
                }
                ResolvedAxisMap::Serial { logical_axis } => serial.push(*logical_axis as usize),
            }
        }
        let done = self.label();
        for predicate in active {
            self.push(format!("@{predicate} bra {done};"));
        }
        let mut grid_loops = Vec::new();
        for (loop_index, (index, _, stride, axis)) in grid_stride.iter().enumerate() {
            let top = self.label();
            self.body.push(format!("{top}:"));
            for (inner, base, _, _) in grid_stride.iter().skip(loop_index + 1) {
                self.push(format!("mov.u64 {inner}, {base};"));
            }
            grid_loops.push((top, index.clone(), *stride, mapping.logical_extents[*axis]));
        }
        let mut loops = Vec::new();
        for axis in serial {
            let index = self.r64();
            self.push(format!("mov.u64 {index}, 0;"));
            for binder in &first.axis_binders[axis] {
                self.values.insert(*binder, Value::Index(index.clone()));
            }
            let top = self.label();
            self.body.push(format!("{top}:"));
            loops.push((top, index, mapping.logical_extents[axis]));
        }
        for instruction in instructions.iter() {
            self.instruction(
                &instruction.scalar,
                &instruction.value_types,
                &instruction.result_values,
                &instruction.views,
                launch,
                pointers,
            )?;
        }
        for (top, index, extent) in loops.into_iter().rev() {
            self.push(format!("add.u64 {index}, {index}, 1;"));
            let p = self.pred();
            self.push(format!("setp.lt.u64 {p}, {index}, {extent};"));
            self.push(format!("@{p} bra {top};"));
        }
        for (top, index, stride, extent) in grid_loops.into_iter().rev() {
            self.push(format!("add.u64 {index}, {index}, {stride};"));
            let p = self.pred();
            self.push(format!("setp.lt.u64 {p}, {index}, {extent};"));
            self.push(format!("@{p} bra {top};"));
        }
        self.body.push(format!("{done}:"));
        Ok(())
    }

    fn transport_value(
        &mut self,
        transport: &ResolvedValueTransport,
        ty: &CudaValueType,
        launch: &ResolvedLaunch<CudaDialect>,
        pointers: &BTreeMap<ResolvedStorageId, String>,
        output: bool,
    ) -> Result<Value, String> {
        match (transport, ty) {
            (ResolvedValueTransport::Void, CudaValueType::Void) => Ok(Value::Void),
            (ResolvedValueTransport::Storage(ids), CudaValueType::Packed { shape, planes, .. })
                if ids.len() == planes.len() =>
            {
                Ok(Value::Packed {
                    shape: shape.clone(),
                    planes: ids
                        .iter()
                        .zip(planes)
                        .map(|(id, (name, dtype, elements))| {
                            let register = pointers
                                .get(id)
                                .cloned()
                                .ok_or_else(|| format!("packed storage#{} is not bound", id.0))?;
                            Ok((
                                name.clone(),
                                Value::Tensor {
                                    register,
                                    dtype: *dtype,
                                    shape: vec![*elements],
                                    strides: vec![u64::from(dtype.bytes())],
                                },
                            ))
                        })
                        .collect::<Result<_, String>>()?,
                })
            }
            (ResolvedValueTransport::Storage(ids), CudaValueType::Tensor { dtype, shape }) => {
                let id = *ids.iter().next().unwrap();
                let pointer = pointers
                    .get(&id)
                    .cloned()
                    .ok_or_else(|| format!("storage#{} is not bound", id.0))?;
                let descriptor = launch
                    .storage
                    .values()
                    .find(|value| value.id == id)
                    .ok_or("bound storage has no descriptor")?;
                let strides = dense_strides(descriptor, shape, *dtype)?;
                Ok(Value::Tensor {
                    register: pointer,
                    dtype: *dtype,
                    shape: shape.clone(),
                    strides,
                })
            }
            (ResolvedValueTransport::Storage(ids), CudaValueType::Scalar(dtype)) => {
                let id = *ids.iter().next().unwrap();
                let pointer = pointers
                    .get(&id)
                    .cloned()
                    .ok_or("scalar storage is not bound")?;
                if output {
                    Ok(Value::Pointer {
                        register: pointer,
                        dtype: *dtype,
                    })
                } else {
                    self.load(pointer, *dtype)
                }
            }
            (ResolvedValueTransport::Storage(ids), CudaValueType::Index) => {
                let id = *ids.iter().next().unwrap();
                let pointer = pointers
                    .get(&id)
                    .cloned()
                    .ok_or("index storage is not bound")?;
                let value = self.r64();
                self.push(format!("ld.global.u64 {value}, [{pointer}];"));
                Ok(Value::Index(value))
            }
            (ResolvedValueTransport::Tuple(fields), CudaValueType::Tuple(types))
                if fields.len() == types.len() =>
            {
                Ok(Value::Tuple(
                    fields
                        .iter()
                        .zip(types)
                        .map(|(field, ty)| {
                            self.transport_value(field, ty, launch, pointers, output)
                        })
                        .collect::<Result<_, _>>()?,
                ))
            }
            (ResolvedValueTransport::Kernel(_), _) => {
                Err("CUDA launch received unresolved kernel-local input".into())
            }
            _ => Err("CUDA transport disagrees with resolved scalar type".into()),
        }
    }

    fn instruction(
        &mut self,
        instruction: &ResolvedScalarInstruction,
        types: &BTreeMap<ScalarValueId, CudaValueType>,
        result_values: &BTreeMap<u32, ScalarValueId>,
        views: &BTreeMap<seismic_lang::logical::LocalViewId, CudaResolvedView>,
        launch: &ResolvedLaunch<CudaDialect>,
        pointers: &BTreeMap<ResolvedStorageId, String>,
    ) -> Result<(), String> {
        let result = instruction.result;
        let value = match &instruction.kind {
            I::NoOp => None,
            I::Literal(value) => Some(self.literal(value, result.and_then(|id| types.get(&id)))?),
            I::Shape(value) => {
                let r = self.r64();
                self.push(format!("mov.u64 {r}, {value};"));
                Some(Value::Index(r))
            }
            I::Tuple(values) => Some(Value::Tuple(
                values
                    .iter()
                    .map(|id| self.get(*id))
                    .collect::<Result<_, _>>()?,
            )),
            I::Range(lo, hi) => Some(Value::Range(self.index(*lo)?, self.index(*hi)?)),
            I::Field { value, index } => match self.get(*value)? {
                Value::Tuple(values) => {
                    Some(values.get(*index).cloned().ok_or("tuple field is absent")?)
                }
                _ => return Err("field base is not tuple".into()),
            },
            I::Storage {
                operation,
                bindings,
                inputs,
                fill_bits,
            } => {
                let mut values = Vec::new();
                for (binding_index, binding) in bindings.iter().enumerate() {
                    let pointer = pointers
                        .get(&binding.storage)
                        .cloned()
                        .ok_or("storage operation allocation is not launch-bound")?;
                    let descriptor = launch
                        .storage
                        .values()
                        .find(|value| value.id == binding.storage)
                        .ok_or("storage descriptor is absent")?;
                    let dtype = descriptor
                        .layout
                        .dtype
                        .ok_or("storage operation layout has no element dtype")?;
                    if matches!(operation, StorageOperation::Fill) {
                        self.fill(
                            pointer.clone(),
                            dtype,
                            descriptor.layout.elements,
                            fill_bits.unwrap_or(0),
                            launch.geometry.workgroups,
                            launch.geometry.participants_per_workgroup,
                        )?;
                    }
                    if matches!(
                        operation,
                        StorageOperation::Snapshot
                            | StorageOperation::Decode
                            | StorageOperation::Materialize
                    ) && !inputs.is_empty()
                    {
                        let input = *inputs.get(binding_index).unwrap_or(&inputs[0]);
                        let source = match self.get(input)? {
                            Value::Packed { planes, .. } => planes
                                .into_iter()
                                .find(|(plane, _)| plane == &binding.plane.join("."))
                                .map(|(_, value)| value)
                                .ok_or("storage operation packed input plane is absent")?,
                            value => value,
                        };
                        self.copy_value(pointer.clone(), source, Some(dtype))?;
                    }
                    values.push((
                        binding.plane.join("."),
                        Value::Tensor {
                            register: pointer,
                            dtype,
                            shape: vec![descriptor.layout.elements],
                            strides: vec![u64::from(dtype.bytes())],
                        },
                    ));
                }
                if values.len() == 1 && values[0].0.is_empty() {
                    Some(values.pop().unwrap().1)
                } else {
                    let shape = result
                        .and_then(|id| types.get(&id))
                        .and_then(|ty| match ty {
                            CudaValueType::Packed { shape, .. } => Some(shape.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    Some(Value::Packed {
                        shape,
                        planes: values,
                    })
                }
            }
            I::View { view, base } => {
                let mut value = self.get(*base)?;
                match views
                    .get(view)
                    .ok_or_else(|| format!("CUDA view#{} has no resolved transform", view.0))?
                {
                    CudaResolvedView::Identity => {}
                    CudaResolvedView::Reshape { shape } => match &mut value {
                        Value::Tensor {
                            dtype,
                            shape: current,
                            strides,
                            ..
                        } => {
                            *current = shape.clone();
                            *strides = contiguous_strides(shape, *dtype)?;
                        }
                        Value::Packed { shape: current, .. } => *current = shape.clone(),
                        _ => return Err("reshape base is not tensor storage".into()),
                    },
                    CudaResolvedView::Transpose { permutation } => match &mut value {
                        Value::Tensor { shape, strides, .. } => {
                            if permutation.len() != shape.len() {
                                return Err("transpose rank disagrees with tensor".into());
                            }
                            let source_shape = shape.clone();
                            let source_strides = strides.clone();
                            *shape = permutation
                                .iter()
                                .map(|axis| source_shape[*axis as usize])
                                .collect();
                            *strides = permutation
                                .iter()
                                .map(|axis| source_strides[*axis as usize])
                                .collect();
                        }
                        Value::Packed { shape, .. } => {
                            let source = shape.clone();
                            *shape = permutation
                                .iter()
                                .map(|axis| source[*axis as usize])
                                .collect();
                        }
                        _ => return Err("transpose base is not tensor storage".into()),
                    },
                }
                Some(value)
            }
            I::Index { base, indices } => Some(self.index_value(self.get(*base)?, indices)?),
            I::Cast { dtype, value } => Some(self.cast(self.get(*value)?, *dtype)?),
            I::Unary { op, value } => Some(self.unary(*op, self.get(*value)?)?),
            I::Binary { op, lhs, rhs } => {
                Some(self.binary(*op, self.get(*lhs)?, self.get(*rhs)?)?)
            }
            I::Math { op, arguments } => Some(self.math(*op, arguments)?),
            I::Select {
                condition,
                then_value,
                else_value,
            } => Some(self.select(
                self.get(*condition)?,
                self.get(*then_value)?,
                self.get(*else_value)?,
            )?),
            I::Extent { base, axis } => {
                let extent = match self.get(*base)? {
                    Value::Tensor { shape, .. } => {
                        *shape.get(*axis).ok_or("tensor extent axis is absent")?
                    }
                    Value::Packed { shape, .. } => *shape
                        .get(*axis)
                        .ok_or("packed tensor extent axis is absent")?,
                    _ => return Err("extent base is not tensor".into()),
                };
                let r = self.r64();
                self.push(format!("mov.u64 {r}, {extent};"));
                Some(Value::Index(r))
            }
            I::Intrinsic {
                operation,
                arguments,
            } => Some(self.intrinsic(*operation, arguments)?),
            I::Accessor { base, name } => match self.get(*base)? {
                Value::Packed { planes, .. } => {
                    let mut value = planes
                        .into_iter()
                        .find(|(plane, _)| plane == name)
                        .map(|(_, value)| value)
                        .ok_or_else(|| format!("packed plane `{name}` is absent"))?;
                    if let (
                        Value::Tensor {
                            dtype,
                            shape,
                            strides,
                            ..
                        },
                        Some(CudaValueType::Tensor {
                            dtype: result_dtype,
                            shape: result_shape,
                        }),
                    ) = (&mut value, result.and_then(|id| types.get(&id)))
                    {
                        *dtype = *result_dtype;
                        *shape = result_shape.clone();
                        *strides = contiguous_strides(result_shape, *result_dtype)?;
                    }
                    Some(value)
                }
                value => Some(value),
            },
            I::Geometry { base, axis, valid } => {
                let value = self.get(*base)?;
                let extent = self.geometry(value, *axis)?;
                Some(if *valid {
                    let Value::Index(index) = extent else {
                        unreachable!()
                    };
                    let predicate = self.pred();
                    let register = self.r32();
                    self.push(format!("setp.ne.u64 {predicate}, {index}, 0;"));
                    self.push(format!("selp.u32 {register}, 1, 0, {predicate};"));
                    Value::Scalar {
                        register,
                        dtype: DType::Bool,
                    }
                } else {
                    extent
                })
            }
            I::Atomic { op, place, value } => {
                self.atomic(*op, self.get(*place)?, self.get(*value)?)?;
                Some(self.get(*value)?)
            }
            I::Assign { target, op, value } => {
                self.assign(*op, self.get(*target)?, self.get(*value)?)?;
                None
            }
            I::Publish { value, destination } => {
                self.assign(AssignOp::Assign, self.get(*destination)?, self.get(*value)?)?;
                self.push("membar.gl;");
                None
            }
            I::Conditional {
                condition,
                then_body,
                else_body,
            } => {
                self.conditional(
                    self.get(*condition)?,
                    then_body,
                    else_body,
                    types,
                    result_values,
                    views,
                    launch,
                    pointers,
                )?;
                None
            }
            I::ConditionalMerge { cases } => Some(self.merge(cases)?),
            I::Yield(_) => None,
            I::Return {
                port, path, value, ..
            } => {
                let destination = result_values
                    .get(port)
                    .copied()
                    .ok_or_else(|| format!("result port#{port} has no scalar binding"))?;
                let mut destination = self.get(destination)?;
                for field in path {
                    destination = match destination {
                        Value::Tuple(fields) => fields
                            .get(*field as usize)
                            .cloned()
                            .ok_or_else(|| format!("result port#{port} path is absent"))?,
                        _ => return Err(format!("result port#{port} path enters non-tuple")),
                    };
                }
                self.assign(AssignOp::Assign, destination, self.get(*value)?)?;
                None
            }
        };
        if let (Some(id), Some(value)) = (result, value) {
            self.values.insert(id, value);
        }
        Ok(())
    }

    fn get(&self, id: ScalarValueId) -> Result<Value, String> {
        self.values
            .get(&id)
            .cloned()
            .ok_or_else(|| format!("scalar value#{} is undefined", id.0))
    }
    fn scalar(&mut self, value: Value) -> Result<Value, String> {
        match value {
            Value::Pointer { register, dtype } => self.load(register, dtype),
            Value::Scalar { .. } | Value::Index(_) => Ok(value),
            _ => Err("value is not scalar".into()),
        }
    }
    fn index(&mut self, id: ScalarValueId) -> Result<String, String> {
        match self.get(id)? {
            Value::Index(v) => Ok(v),
            Value::Scalar {
                register,
                dtype: dtype @ (DType::U32 | DType::I32),
            } => {
                let index = self.r64();
                self.push(format!(
                    "cvt.{}.{} {index}, {register};",
                    if dtype == DType::I32 { "s64" } else { "u64" },
                    if dtype == DType::I32 { "s32" } else { "u32" }
                ));
                Ok(index)
            }
            _ => Err("value is not an index".into()),
        }
    }
    fn bool_const(&mut self, value: bool) -> String {
        let r = self.r32();
        self.push(format!("mov.u32 {r}, {};", u8::from(value)));
        r
    }

    fn literal(
        &mut self,
        value: &ScalarLiteral,
        ty: Option<&CudaValueType>,
    ) -> Result<Value, String> {
        Ok(match value {
            ScalarLiteral::Int(value) => match ty {
                Some(CudaValueType::Index) => {
                    let r = self.r64();
                    self.push(format!("mov.u64 {r}, {value};"));
                    Value::Index(r)
                }
                _ => {
                    let r = self.r32();
                    self.push(format!("mov.u32 {r}, {value};"));
                    Value::Scalar {
                        register: r,
                        dtype: match ty {
                            Some(CudaValueType::Scalar(dtype)) => *dtype,
                            _ => DType::I32,
                        },
                    }
                }
            },
            ScalarLiteral::Float(bits) => {
                let r = self.f32();
                let bits = (f64::from_bits(*bits) as f32).to_bits();
                self.push(format!("mov.b32 {r}, 0x{bits:08x};"));
                Value::Scalar {
                    register: r,
                    dtype: DType::F32,
                }
            }
            ScalarLiteral::Bool(value) => Value::Scalar {
                register: self.bool_const(*value),
                dtype: DType::Bool,
            },
        })
    }

    fn load(&mut self, pointer: String, dtype: DType) -> Result<Value, String> {
        let register = if dtype == DType::F32 {
            let r = self.f32();
            self.push(format!("ld.global.f32 {r}, [{pointer}];"));
            r
        } else {
            let r = self.r32();
            self.push(format!(
                "ld.global.{} {r}, [{pointer}];",
                storage_type(dtype)
            ));
            if matches!(dtype, DType::F16 | DType::BF16) {
                let f = self.f32();
                self.push(format!(
                    "cvt.f32.{} {f}, {r};",
                    if dtype == DType::F16 { "f16" } else { "bf16" }
                ));
                return Ok(Value::Scalar { register: f, dtype });
            }
            r
        };
        Ok(Value::Scalar { register, dtype })
    }
    fn store(&mut self, pointer: String, value: Value, dtype: DType) -> Result<(), String> {
        let Value::Scalar {
            mut register,
            dtype: source,
        } = value
        else {
            return Err("store value is not scalar".into());
        };
        if matches!(dtype, DType::F16 | DType::BF16) && source == DType::F32 {
            let r = self.r32();
            self.push(format!(
                "cvt.rn.{}.f32 {r}, {register};",
                if dtype == DType::F16 { "f16" } else { "bf16" }
            ));
            register = r;
        }
        self.push(format!(
            "st.global.{} [{pointer}], {register};",
            storage_type(dtype)
        ));
        Ok(())
    }
    fn copy_value(
        &mut self,
        pointer: String,
        value: Value,
        dtype: Option<DType>,
    ) -> Result<(), String> {
        match value {
            Value::Scalar { .. } => {
                self.store(pointer, value, dtype.ok_or("copy destination lacks dtype")?)
            }
            Value::Pointer { register, dtype } => {
                let loaded = self.load(register, dtype)?;
                self.store(pointer, loaded, dtype)
            }
            Value::Tensor {
                register,
                dtype,
                shape,
                ..
            } => self.copy_tensor(pointer, register, dtype, &shape),
            _ => Err("aggregate copy requires explicit element task".into()),
        }
    }

    fn copy_tensor(
        &mut self,
        destination: String,
        source: String,
        dtype: DType,
        shape: &[u64],
    ) -> Result<(), String> {
        let elements = shape
            .iter()
            .try_fold(1_u64, |a, b| a.checked_mul(*b))
            .ok_or("CUDA tensor copy extent overflow")?;
        let (index, stride) = self.global_thread_index()?;
        let top = self.label();
        let done = self.label();
        let invalid = self.pred();
        self.body.push(format!("{top}:"));
        self.push(format!("setp.ge.u64 {invalid}, {index}, {elements};"));
        self.push(format!("@{invalid} bra {done};"));
        let offset = self.r64();
        let input = self.r64();
        let output = self.r64();
        self.push(format!("mul.lo.u64 {offset}, {index}, {};", dtype.bytes()));
        self.push(format!("add.u64 {input}, {source}, {offset};"));
        self.push(format!("add.u64 {output}, {destination}, {offset};"));
        let value = self.load(input, dtype)?;
        self.store(output, value, dtype)?;
        self.push(format!("add.u64 {index}, {index}, {stride};"));
        self.push(format!("bra {top};"));
        self.body.push(format!("{done}:"));
        Ok(())
    }

    fn global_thread_index(&mut self) -> Result<(String, u64), String> {
        let tx = self.r32();
        let ty = self.r32();
        let tz = self.r32();
        let bx = self.r32();
        let by = self.r32();
        let bz = self.r32();
        self.push(format!("mov.u32 {tx}, %tid.x;"));
        self.push(format!("mov.u32 {ty}, %tid.y;"));
        self.push(format!("mov.u32 {tz}, %tid.z;"));
        self.push(format!("mov.u32 {bx}, %ctaid.x;"));
        self.push(format!("mov.u32 {by}, %ctaid.y;"));
        self.push(format!("mov.u32 {bz}, %ctaid.z;"));
        let local = self.r32();
        let block_id = self.r32();
        self.push(format!(
            "mad.lo.u32 {local}, {tz}, {}, {ty};",
            self.block[1]
        ));
        self.push(format!(
            "mad.lo.u32 {local}, {local}, {}, {tx};",
            self.block[0]
        ));
        self.push(format!(
            "mad.lo.u32 {block_id}, {bz}, {}, {by};",
            self.grid[1]
        ));
        self.push(format!(
            "mad.lo.u32 {block_id}, {block_id}, {}, {bx};",
            self.grid[0]
        ));
        let index = self.r64();
        let block64 = self.r64();
        let local64 = self.r64();
        self.push(format!("cvt.u64.u32 {block64}, {block_id};"));
        self.push(format!("cvt.u64.u32 {local64}, {local};"));
        let threads = self
            .block
            .iter()
            .try_fold(1_u64, |a, b| a.checked_mul(*b))
            .ok_or("CUDA block size overflow")?;
        let stride = self
            .grid
            .iter()
            .chain(self.block.iter())
            .try_fold(1_u64, |a, b| a.checked_mul(*b))
            .ok_or("CUDA launch size overflow")?;
        self.push(format!(
            "mad.lo.u64 {index}, {block64}, {threads}, {local64};"
        ));
        Ok((index, stride))
    }

    fn fill(
        &mut self,
        pointer: String,
        dtype: DType,
        elements: u64,
        bits: u64,
        grid: [u64; 3],
        block: [u64; 3],
    ) -> Result<(), String> {
        let tx = self.r32();
        let ty = self.r32();
        let tz = self.r32();
        let bx = self.r32();
        let by = self.r32();
        let bz = self.r32();
        self.push(format!("mov.u32 {tx}, %tid.x;"));
        self.push(format!("mov.u32 {ty}, %tid.y;"));
        self.push(format!("mov.u32 {tz}, %tid.z;"));
        self.push(format!("mov.u32 {bx}, %ctaid.x;"));
        self.push(format!("mov.u32 {by}, %ctaid.y;"));
        self.push(format!("mov.u32 {bz}, %ctaid.z;"));
        let local = self.r32();
        let block_id = self.r32();
        self.push(format!("mad.lo.u32 {local}, {tz}, {}, {ty};", block[1]));
        self.push(format!("mad.lo.u32 {local}, {local}, {}, {tx};", block[0]));
        self.push(format!("mad.lo.u32 {block_id}, {bz}, {}, {by};", grid[1]));
        self.push(format!(
            "mad.lo.u32 {block_id}, {block_id}, {}, {bx};",
            grid[0]
        ));
        let index = self.r64();
        let block64 = self.r64();
        let local64 = self.r64();
        self.push(format!("cvt.u64.u32 {block64}, {block_id};"));
        self.push(format!("cvt.u64.u32 {local64}, {local};"));
        let threads = block
            .iter()
            .try_fold(1_u64, |a, b| a.checked_mul(*b))
            .ok_or("CUDA fill block size overflow")?;
        let stride = grid
            .iter()
            .chain(block.iter())
            .try_fold(1_u64, |a, b| a.checked_mul(*b))
            .ok_or("CUDA fill launch size overflow")?;
        self.push(format!(
            "mad.lo.u64 {index}, {block64}, {threads}, {local64};"
        ));
        let top = self.label();
        let done = self.label();
        let valid = self.pred();
        self.body.push(format!("{top}:"));
        self.push(format!("setp.ge.u64 {valid}, {index}, {elements};"));
        self.push(format!("@{valid} bra {done};"));
        let offset = self.r64();
        let address = self.r64();
        self.push(format!("mul.lo.u64 {offset}, {index}, {};", dtype.bytes()));
        self.push(format!("add.u64 {address}, {pointer}, {offset};"));
        self.push(format!(
            "st.global.{} [{address}], 0x{:x};",
            storage_type(dtype),
            bits
        ));
        self.push(format!("add.u64 {index}, {index}, {stride};"));
        self.push(format!("bra {top};"));
        self.body.push(format!("{done}:"));
        Ok(())
    }

    fn index_value(&mut self, base: Value, indices: &[ScalarIndex]) -> Result<Value, String> {
        let (pointer, dtype, shape, strides) = match base {
            Value::Tensor {
                register,
                dtype,
                shape,
                strides,
            } => (register, dtype, shape, strides),
            Value::Pointer { register, dtype } => {
                (register, dtype, vec![], vec![u64::from(dtype.bytes())])
            }
            _ => return Err("index base is not storage".into()),
        };
        let offset = self.r64();
        self.push(format!("mov.u64 {offset}, 0;"));
        let mut remaining_shape = Vec::new();
        let mut remaining_strides = Vec::new();
        for (axis, index) in indices.iter().enumerate() {
            match index {
                ScalarIndex::Point(id) | ScalarIndex::Coordinate(id) => {
                    let i = self.index(*id)?;
                    let term = self.r64();
                    self.push(format!(
                        "mul.lo.u64 {term}, {i}, {};",
                        strides
                            .get(axis)
                            .copied()
                            .unwrap_or(u64::from(dtype.bytes()))
                    ));
                    self.push(format!("add.u64 {offset}, {offset}, {term};"));
                }
                ScalarIndex::Slice(_) => {
                    remaining_shape.push(*shape.get(axis).unwrap_or(&1));
                    remaining_strides.push(*strides.get(axis).unwrap_or(&u64::from(dtype.bytes())));
                }
                ScalarIndex::Range { start, end } => {
                    if let Some(start) = start {
                        let i = self.index(*start)?;
                        let term = self.r64();
                        self.push(format!("mul.lo.u64 {term}, {i}, {};", strides[axis]));
                        self.push(format!("add.u64 {offset}, {offset}, {term};"));
                    }
                    let end = end.map(|id| self.index(id)).transpose()?;
                    let _ = end;
                    remaining_shape.push(*shape.get(axis).unwrap_or(&1));
                    remaining_strides.push(strides[axis]);
                }
            }
        }
        let address = self.r64();
        self.push(format!("add.u64 {address}, {pointer}, {offset};"));
        if remaining_shape.is_empty() {
            Ok(Value::Pointer {
                register: address,
                dtype,
            })
        } else {
            Ok(Value::Tensor {
                register: address,
                dtype,
                shape: remaining_shape,
                strides: remaining_strides,
            })
        }
    }

    fn cast(&mut self, value: Value, dtype: DType) -> Result<Value, String> {
        let value = self.scalar(value)?;
        let Value::Scalar {
            register,
            dtype: source,
        } = value
        else {
            return Err("cast source is not scalar".into());
        };
        if source == dtype {
            return Ok(Value::Scalar { register, dtype });
        }
        if source.is_float() && dtype.is_float() {
            let out = self.f32();
            self.push(format!("mov.b32 {out}, {register};"));
            return Ok(Value::Scalar {
                register: out,
                dtype,
            });
        }
        let out = if dtype == DType::F32 {
            self.f32()
        } else {
            self.r32()
        };
        self.push(format!(
            "cvt{}.{}.{} {out}, {register};",
            if dtype == DType::F32 { ".rn" } else { "" },
            ptx_type(dtype),
            ptx_type(source)
        ));
        Ok(Value::Scalar {
            register: out,
            dtype,
        })
    }
    fn unary(&mut self, op: UnaryOp, value: Value) -> Result<Value, String> {
        let value = self.scalar(value)?;
        let Value::Scalar { register, dtype } = value else {
            return Err("unary source is not scalar".into());
        };
        let out = if dtype == DType::F32 {
            self.f32()
        } else {
            self.r32()
        };
        let code = match op {
            UnaryOp::Neg => {
                if dtype.is_float() {
                    "neg.f32"
                } else {
                    "neg.s32"
                }
            }
            UnaryOp::Not => "xor.b32",
            UnaryOp::BitNot => "not.b32",
        };
        if op == UnaryOp::Not {
            self.push(format!("{code} {out}, {register}, 1;"))
        } else {
            self.push(format!("{code} {out}, {register};"))
        };
        Ok(Value::Scalar {
            register: out,
            dtype,
        })
    }

    fn binary(&mut self, op: BinaryOp, lhs: Value, rhs: Value) -> Result<Value, String> {
        let lhs = self.scalar(lhs)?;
        let rhs = self.scalar(rhs)?;
        let (
            Value::Scalar { register: a, dtype },
            Value::Scalar {
                register: b,
                dtype: bd,
            },
        ) = (lhs, rhs)
        else {
            return Err("binary operands are not scalar".into());
        };
        if dtype != bd {
            return Err("binary operand dtype mismatch".into());
        }
        let comparison = matches!(
            op,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        );
        if comparison {
            let p = self.pred();
            let out = self.r32();
            self.push(format!(
                "setp.{}.{} {p}, {a}, {b};",
                cmp(op),
                ptx_type(dtype)
            ));
            self.push(format!("selp.u32 {out}, 1, 0, {p};"));
            return Ok(Value::Scalar {
                register: out,
                dtype: DType::Bool,
            });
        }
        let out = if dtype.is_float() {
            self.f32()
        } else {
            self.r32()
        };
        self.push(format!("{} {out}, {a}, {b};", bin(op, dtype)?));
        Ok(Value::Scalar {
            register: out,
            dtype,
        })
    }

    fn math(&mut self, op: Math, arguments: &[ScalarValueId]) -> Result<Value, String> {
        let raw_values = arguments
            .iter()
            .map(|id| self.get(*id))
            .collect::<Result<Vec<_>, _>>()?;
        let values = raw_values
            .into_iter()
            .map(|value| self.scalar(value))
            .collect::<Result<Vec<_>, _>>()?;
        let scalar = |v: &Value| match v {
            Value::Scalar { register, dtype } => Ok((register.clone(), *dtype)),
            _ => Err("math argument is not scalar".to_string()),
        };
        let (a, dtype) = scalar(&values[0])?;
        let mut out = self.f32();
        match op {
            Math::Fma => {
                let (b, _) = scalar(&values[1])?;
                let (c, _) = scalar(&values[2])?;
                self.push(format!("fma.rn.f32 {out}, {a}, {b}, {c};"))
            }
            Math::Exp => out = self.exact_math_call("seismic_exp", a),
            Math::ExpFast => {
                let t = self.f32();
                self.push(format!("mul.f32 {t}, {a}, 0f3fb8aa3b;"));
                self.push(format!("ex2.approx.f32 {out}, {t};"))
            }
            Math::Log => out = self.exact_math_call("seismic_log", a),
            Math::Sin => out = self.exact_math_call("seismic_sin", a),
            Math::Cos => out = self.exact_math_call("seismic_cos", a),
            Math::Sqrt => self.push(format!("sqrt.rn.f32 {out}, {a};")),
            Math::Rsqrt => self.push(format!("rsqrt.approx.f32 {out}, {a};")),
            Math::Abs => self.push(format!("abs.f32 {out}, {a};")),
            Math::Max | Math::Min => {
                let (b, _) = scalar(&values[1])?;
                self.push(format!(
                    "{}.f32 {out}, {a}, {b};",
                    if op == Math::Max { "max" } else { "min" }
                ))
            }
        }
        Ok(Value::Scalar {
            register: out,
            dtype: if dtype.is_float() { dtype } else { DType::F32 },
        })
    }
    fn select(
        &mut self,
        condition: Value,
        then_value: Value,
        else_value: Value,
    ) -> Result<Value, String> {
        let condition = self.scalar(condition)?;
        let then_value = self.scalar(then_value)?;
        let else_value = self.scalar(else_value)?;
        let Value::Scalar { register: cond, .. } = condition else {
            return Err("select condition is not scalar".into());
        };
        let (
            Value::Scalar { register: a, dtype },
            Value::Scalar {
                register: b,
                dtype: bd,
            },
        ) = (then_value, else_value)
        else {
            return Err("aggregate select requires conditional merge".into());
        };
        if dtype != bd {
            return Err("select dtype mismatch".into());
        }
        let p = self.pred();
        self.push(format!("setp.ne.u32 {p}, {cond}, 0;"));
        let out = if dtype.is_float() {
            self.f32()
        } else {
            self.r32()
        };
        self.push(format!("selp.{} {out}, {a}, {b}, {p};", ptx_type(dtype)));
        Ok(Value::Scalar {
            register: out,
            dtype,
        })
    }
    fn intrinsic(
        &mut self,
        operation: Operation,
        arguments: &[ScalarValueId],
    ) -> Result<Value, String> {
        match operation {
            Operation::LaneIndex => {
                let r = self.r32();
                self.push(format!("mov.u32 {r}, %laneid;"));
                Ok(Value::Scalar {
                    register: r,
                    dtype: DType::U32,
                })
            }
            Operation::ShuffleIndex => {
                let value = self.scalar(self.get(arguments[0])?)?;
                let index = self.scalar(self.get(arguments[1])?)?;
                let (
                    Value::Scalar {
                        register: value,
                        dtype,
                    },
                    Value::Scalar {
                        register: index, ..
                    },
                ) = (value, index)
                else {
                    return Err("shuffle arguments are not scalar".into());
                };
                let out = if dtype.is_float() {
                    self.f32()
                } else {
                    self.r32()
                };
                self.push(format!(
                    "shfl.sync.idx.b32 {out}, {value}, {index}, 31, 0xffffffff;"
                ));
                Ok(Value::Scalar {
                    register: out,
                    dtype,
                })
            }
            Operation::SimdSum | Operation::SimdMax | Operation::SimdMin => {
                let argument = self.get(arguments[0])?;
                let mut value = self.scalar(argument)?;
                for offset in [16, 8, 4, 2, 1] {
                    let Value::Scalar { register, dtype } = value.clone() else {
                        return Err("subgroup reduction value is not scalar".into());
                    };
                    let shuffled = if dtype.is_float() {
                        self.f32()
                    } else {
                        self.r32()
                    };
                    self.push(format!(
                        "shfl.sync.down.b32 {shuffled}, {register}, {offset}, 31, 0xffffffff;"
                    ));
                    value = if operation == Operation::SimdSum {
                        self.binary(
                            BinaryOp::Add,
                            value,
                            Value::Scalar {
                                register: shuffled,
                                dtype,
                            },
                        )?
                    } else {
                        let out = if dtype.is_float() {
                            self.f32()
                        } else {
                            self.r32()
                        };
                        self.push(format!(
                            "{}.{} {out}, {register}, {shuffled};",
                            if operation == Operation::SimdMax {
                                "max"
                            } else {
                                "min"
                            },
                            ptx_type(dtype)
                        ));
                        Value::Scalar {
                            register: out,
                            dtype,
                        }
                    };
                }
                Ok(value)
            }
            _ => Err(format!(
                "CUDA matrix intrinsic `{}` has no admitted scalar PTX encoding",
                operation.name()
            )),
        }
    }
    fn geometry(&mut self, value: Value, axis: usize) -> Result<Value, String> {
        match value {
            Value::Tensor { shape, .. } => {
                let r = self.r64();
                self.push(format!(
                    "mov.u64 {r}, {};",
                    shape.get(axis).copied().ok_or("geometry axis absent")?
                ));
                Ok(Value::Index(r))
            }
            Value::Packed { shape, .. } => {
                let r = self.r64();
                self.push(format!(
                    "mov.u64 {r}, {};",
                    shape.get(axis).copied().ok_or("geometry axis absent")?
                ));
                Ok(Value::Index(r))
            }
            _ => Err("geometry base is not tensor".into()),
        }
    }
    fn atomic(&mut self, op: BinaryOp, place: Value, value: Value) -> Result<(), String> {
        let Value::Pointer {
            register: pointer,
            dtype,
        } = place
        else {
            return Err("atomic target is not a scalar place".into());
        };
        let value = self.scalar(value)?;
        let Value::Scalar {
            register: value, ..
        } = value
        else {
            return Err("atomic value is not scalar".into());
        };
        let operation = match op {
            BinaryOp::Add => "add",
            BinaryOp::BitAnd => "and",
            BinaryOp::BitOr => "or",
            BinaryOp::BitXor => "xor",
            _ => return Err("unsupported CUDA atomic operation".into()),
        };
        let out = if dtype.is_float() {
            self.f32()
        } else {
            self.r32()
        };
        self.push(format!(
            "atom.global.{operation}.{} {out}, [{pointer}], {value};",
            ptx_type(dtype)
        ));
        Ok(())
    }
    fn assign(&mut self, op: AssignOp, target: Value, value: Value) -> Result<(), String> {
        match target {
            Value::Pointer { register, dtype } => {
                let value = if op == AssignOp::Assign {
                    value
                } else {
                    let old = self.load(register.clone(), dtype)?;
                    self.binary(
                        match op {
                            AssignOp::Add => BinaryOp::Add,
                            AssignOp::Sub => BinaryOp::Sub,
                            AssignOp::Mul => BinaryOp::Mul,
                            AssignOp::Assign => unreachable!(),
                        },
                        old,
                        value,
                    )?
                };
                self.store(register, value, dtype)
            }
            Value::Tensor {
                register, dtype, ..
            } => self.copy_value(register, value, Some(dtype)),
            _ => Err("assignment target is not storage".into()),
        }
    }
    fn conditional(
        &mut self,
        condition: Value,
        then_body: &[ResolvedScalarInstruction],
        else_body: &[ResolvedScalarInstruction],
        types: &BTreeMap<ScalarValueId, CudaValueType>,
        result_values: &BTreeMap<u32, ScalarValueId>,
        views: &BTreeMap<seismic_lang::logical::LocalViewId, CudaResolvedView>,
        launch: &ResolvedLaunch<CudaDialect>,
        pointers: &BTreeMap<ResolvedStorageId, String>,
    ) -> Result<(), String> {
        let condition = self.scalar(condition)?;
        let Value::Scalar { register, .. } = condition else {
            return Err("conditional condition is not scalar".into());
        };
        let p = self.pred();
        let els = self.label();
        let done = self.label();
        self.push(format!("setp.eq.u32 {p}, {register}, 0;"));
        self.push(format!("@{p} bra {els};"));
        for instruction in then_body {
            self.instruction(instruction, types, result_values, views, launch, pointers)?
        }
        self.push(format!("bra {done};"));
        self.body.push(format!("{els}:"));
        for instruction in else_body {
            self.instruction(instruction, types, result_values, views, launch, pointers)?
        }
        self.body.push(format!("{done}:"));
        Ok(())
    }
    fn merge(&mut self, cases: &[ConditionalCase]) -> Result<Value, String> {
        let first = cases.first().ok_or("conditional merge has no cases")?;
        let prototype = self.get(first.value)?;
        let (output, dtype, is_index) = match prototype {
            Value::Scalar { dtype, .. } => (
                if dtype.is_float() {
                    self.f32()
                } else {
                    self.r32()
                },
                dtype,
                false,
            ),
            Value::Index(_) => (self.r64(), DType::U32, true),
            _ => return Err("aggregate conditional merge is unsupported".into()),
        };
        for case in cases {
            let mut predicates = Vec::new();
            for (id, expected) in &case.predicates {
                let predicate_value = self.get(*id)?;
                let Value::Scalar { register, .. } = self.scalar(predicate_value)? else {
                    return Err("merge predicate is not scalar".into());
                };
                let p = self.pred();
                self.push(format!(
                    "setp.{}.u32 {p}, {register}, 0;",
                    if *expected { "ne" } else { "eq" }
                ));
                predicates.push(p)
            }
            let selected = if predicates.is_empty() {
                let p = self.pred();
                self.push(format!("setp.eq.u32 {p}, 0, 0;"));
                p
            } else {
                let mut selected = predicates[0].clone();
                for predicate in predicates.iter().skip(1) {
                    let both = self.pred();
                    self.push(format!("and.pred {both}, {selected}, {predicate};"));
                    selected = both;
                }
                selected
            };
            match self.get(case.value)? {
                Value::Scalar {
                    register,
                    dtype: case_dtype,
                } if !is_index && case_dtype == dtype => {
                    self.push(format!(
                        "@{selected} mov.{} {output}, {register};",
                        ptx_type(dtype)
                    ));
                }
                Value::Index(register) if is_index => {
                    self.push(format!("@{selected} mov.u64 {output}, {register};"));
                }
                _ => return Err("conditional merge case type mismatch".into()),
            }
        }
        if is_index {
            Ok(Value::Index(output))
        } else {
            Ok(Value::Scalar {
                register: output,
                dtype,
            })
        }
    }
}

fn axis_name(axis: u8) -> Result<&'static str, String> {
    match axis {
        0 => Ok("x"),
        1 => Ok("y"),
        2 => Ok("z"),
        _ => Err("CUDA axis is outside x/y/z".into()),
    }
}
fn storage_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 | DType::BF16 => "b16",
        DType::I32 => "s32",
        DType::U32 => "u32",
        DType::Bool => "u8",
    }
}
fn ptx_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => "f32",
        DType::I32 => "s32",
        DType::U32 | DType::Bool => "u32",
    }
}
fn cmp(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Eq => "eq",
        BinaryOp::Ne => "ne",
        BinaryOp::Lt => "lt",
        BinaryOp::Le => "le",
        BinaryOp::Gt => "gt",
        BinaryOp::Ge => "ge",
        _ => "eq",
    }
}
fn bin(op: BinaryOp, dtype: DType) -> Result<&'static str, String> {
    Ok(match op {
        BinaryOp::Or | BinaryOp::BitOr => "or.b32",
        BinaryOp::And | BinaryOp::BitAnd => "and.b32",
        BinaryOp::BitXor => "xor.b32",
        BinaryOp::Shl => "shl.b32",
        BinaryOp::Shr => {
            if dtype == DType::I32 {
                "shr.s32"
            } else {
                "shr.u32"
            }
        }
        BinaryOp::Add => {
            if dtype.is_float() {
                "add.rn.f32"
            } else {
                "add.u32"
            }
        }
        BinaryOp::Sub => {
            if dtype.is_float() {
                "sub.rn.f32"
            } else {
                "sub.u32"
            }
        }
        BinaryOp::Mul => {
            if dtype.is_float() {
                "mul.rn.f32"
            } else {
                "mul.lo.u32"
            }
        }
        BinaryOp::Div => {
            if dtype.is_float() {
                "div.rn.f32"
            } else if dtype == DType::I32 {
                "div.s32"
            } else {
                "div.u32"
            }
        }
        BinaryOp::Rem => {
            if dtype.is_float() {
                return Err("PTX has no direct floating remainder".into());
            } else if dtype == DType::I32 {
                "rem.s32"
            } else {
                "rem.u32"
            }
        }
        _ => return Err("comparison requested as arithmetic".into()),
    })
}
fn dense_strides(
    descriptor: &seismic_realization::executable::ResolvedStorage<CudaDialect>,
    shape: &[u64],
    dtype: DType,
) -> Result<Vec<u64>, String> {
    if let seismic_realization::executable::ResolvedPhysicalAddress::DenseAffine {
        byte_strides,
        ..
    } = &descriptor.provenance.address
    {
        if byte_strides.len() == shape.len() {
            return Ok(byte_strides.clone());
        }
    }
    contiguous_strides(shape, dtype)
}

fn contiguous_strides(shape: &[u64], dtype: DType) -> Result<Vec<u64>, String> {
    let mut values = vec![0; shape.len()];
    let mut stride = u64::from(dtype.bytes());
    for (axis, extent) in shape.iter().enumerate().rev() {
        values[axis] = stride;
        stride = stride.checked_mul(*extent).ok_or("CUDA stride overflow")?
    }
    Ok(values)
}

pub fn assemble(encoded: EncodedPlan<CudaDialect, CudaLaunch>) -> Result<Emitted, String> {
    fn collect(
        encoded: EncodedPlan<CudaDialect, CudaLaunch>,
        launches: &mut Vec<CudaLaunch>,
        abi: &mut BTreeMap<ResolvedStorageId, CudaAbiAllocation>,
    ) -> Result<(), String> {
        for storage in &encoded.resolved.device_storage().allocations {
            let value = CudaAbiAllocation {
                id: storage.id,
                role: storage.provenance.abi.clone(),
                bytes: storage.bytes,
                alignment: storage.alignment,
                layout: storage.layout.clone(),
            };
            if abi.insert(storage.id, value).is_some() {
                return Err("resolved CUDA hierarchy repeats storage identity".into());
            }
        }
        for item in encoded.items {
            match item {
                EncodedScheduleItem::Phase(phase) => launches.extend(phase.launches),
                EncodedScheduleItem::Subplan(subplan) => collect(*subplan.encoded, launches, abi)?,
            }
        }
        Ok(())
    }
    let name = encoded.resolved.identity().entry.clone();
    let mut launches = Vec::new();
    let mut abi = BTreeMap::new();
    collect(encoded, &mut launches, &mut abi)?;
    if launches.is_empty() {
        return Err("resolved CUDA plan contains no launch".into());
    }
    Ok(Emitted {
        name,
        launches,
        abi: abi.into_values().collect(),
    })
}
