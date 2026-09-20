//! Mechanical MSL encoding of resolved executable launches.

use crate::physical::{MetalDialect, ResolvedMetalInstruction, ResolvedMetalLayout};
use seismic_compiler::{
    pipeline::{EncodedPlan, EncodedScheduleItem},
    terminal::{
        ResolvedScalarInstructionKind as I, ScalarIndex, ScalarLiteral, ScalarValueId,
        StorageOperation,
    },
};
use seismic_lang::{
    abi::ScalarParameter,
    logical::{LocalViewTransform, Type},
    repr,
    syntax::ast::{AssignOp, BinaryOp, UnaryOp},
    types::DType,
};
use seismic_realization::{
    dispatch::{GroupDispatch, TileDeclaration},
    executable::{
        AbiRole, AccessMode, BindingGroupKind, ResolvedAxisMap, ResolvedBindingGroup,
        ResolvedKernelStep, ResolvedLaunch, ResolvedOperandTransport, ResolvedPhysicalAddress,
        ResolvedStorage, ResolvedStorageId, ResolvedValueTransport, StorageScope,
    },
    BufferRole, BufferSpec,
};
use std::collections::{BTreeMap, BTreeSet};

pub const MAX_KERNEL_BUFFERS: usize = 31;

#[derive(Clone, Debug, PartialEq)]
pub struct Emitted {
    pub source: String,
    pub launches: Vec<Launch>,
    pub execution: Vec<ExecutionItem>,
    pub buffers: Vec<BufferSpec>,
    /// Exact resolved storage identity for each entry in `buffers`.
    pub buffer_ids: Vec<ResolvedStorageId>,
    pub scalars: Vec<ScalarParameter>,
    pub scratch: Vec<usize>,
    pub scratch_bindings: Vec<BufferSpec>,
    pub status_slot: Option<usize>,
    pub alias_pairs: Vec<(usize, usize, bool)>,
}

impl Emitted {
    pub fn scalar_layout(&self) -> Result<seismic_lang::abi::ScalarLayout, String> {
        seismic_lang::abi::ScalarLayout::natural(&self.scalars)
    }

    pub fn encode_scalars(&self, values: &[f64]) -> Result<Vec<u8>, String> {
        self.scalar_layout()?.encode(values)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExecutionItem {
    Phase(Vec<usize>),
    Subplan(Vec<ExecutionItem>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub kernel: String,
    pub threadgroups: u64,
    pub threads_per_threadgroup: u64,
    pub after_barrier: bool,
    pub dispatch: Option<GroupDispatch>,
    pub tiles: Vec<TileDeclaration>,
    pub declared_threadgroup_bytes: u64,
    pub bindings: Vec<Binding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Binding {
    Buffer(usize),
    Scratch(usize),
    Scalars(usize),
    Status,
    ArgumentTable(Vec<(u32, Binding)>),
}

#[derive(Clone, Debug)]
enum SymbolicBinding {
    Storage(ResolvedStorageId),
    ArgumentTable(Vec<(u32, ResolvedStorageId)>),
}

#[derive(Clone, Debug)]
pub struct EncodedLaunch {
    source: String,
    kernel: String,
    threadgroups: u64,
    threads_per_threadgroup: u64,
    declared_threadgroup_bytes: u64,
    bindings: Vec<SymbolicBinding>,
    storage: BTreeMap<ResolvedStorageId, ResolvedStorage<MetalDialect>>,
}

pub fn encode_launch(launch: &ResolvedLaunch<MetalDialect>) -> Result<EncodedLaunch, String> {
    let kernel = format!("seismic_metal_{}", launch.id.0);
    let mut source = String::new();
    let mut declarations = Vec::new();
    let mut symbolic = Vec::new();
    for group in &launch.binding_groups {
        encode_binding_group(
            group,
            launch,
            &kernel,
            &mut source,
            &mut declarations,
            &mut symbolic,
        )?;
    }
    declarations.push("uint3 tg_pos [[threadgroup_position_in_grid]]".into());
    declarations.push("uint3 tid [[thread_position_in_threadgroup]]".into());
    source.push_str(&format!(
        "kernel void {kernel}(\n    {}\n) {{\n",
        declarations.join(",\n    ")
    ));
    for group in &launch.binding_groups {
        if group.kind == BindingGroupKind::ArgumentTable {
            for binding in group.members.iter() {
                source.push_str(&format!(
                    "  auto s{} = seismic_arguments_{}.s{};\n",
                    binding.storage.0, group.id.0, binding.storage.0
                ));
            }
        }
    }
    let mut renderer = Renderer::new(launch);
    for step in launch.kernel.steps.iter() {
        renderer.step(step)?;
    }
    for line in renderer.lines {
        source.push_str("  ");
        source.push_str(&line);
        source.push('\n');
    }
    source.push_str("}\n\n");
    let threadgroups = launch.geometry.workgroups.iter().try_fold(1u64, |a, b| {
        a.checked_mul(*b).ok_or("Metal workgroup count overflow")
    })?;
    let threads_per_threadgroup = launch
        .geometry
        .participants_per_workgroup
        .iter()
        .try_fold(1u64, |a, b| {
            a.checked_mul(*b).ok_or("Metal participant count overflow")
        })?;
    Ok(EncodedLaunch {
        source,
        kernel,
        threadgroups,
        threads_per_threadgroup,
        declared_threadgroup_bytes: launch.kernel.resources.workgroup_bytes,
        bindings: symbolic,
        storage: launch
            .storage
            .values()
            .cloned()
            .map(|value| (value.id, value))
            .collect(),
    })
}

fn encode_binding_group(
    group: &ResolvedBindingGroup,
    launch: &ResolvedLaunch<MetalDialect>,
    kernel: &str,
    source: &mut String,
    declarations: &mut Vec<String>,
    bindings: &mut Vec<SymbolicBinding>,
) -> Result<(), String> {
    match group.kind {
        BindingGroupKind::Direct => {
            let binding = group.members.iter().next().unwrap();
            let storage = storage(launch, binding.storage)?;
            declarations.push(resource_declaration(storage, binding.access, group.slot));
            bindings.push(SymbolicBinding::Storage(storage.id));
        }
        BindingGroupKind::ArgumentTable => {
            let table = format!("SeismicArguments_{}_{}", kernel, group.id.0);
            source.push_str(&format!("struct {table} {{\n"));
            let mut members = Vec::new();
            for (member, binding) in group.members.iter().enumerate() {
                let storage = storage(launch, binding.storage)?;
                source.push_str(&format!(
                    "  {} {}* s{} [[id({member})]];\n",
                    address_space(binding.access),
                    storage_pointer_type(&storage.layout),
                    storage.id.0,
                ));
                members.push((member as u32, storage.id));
            }
            source.push_str("};\n\n");
            declarations.push(format!(
                "constant {table}& seismic_arguments_{} [[buffer({})]]",
                group.id.0, group.slot
            ));
            bindings.push(SymbolicBinding::ArgumentTable(members));
        }
    }
    Ok(())
}

fn storage(
    launch: &ResolvedLaunch<MetalDialect>,
    id: ResolvedStorageId,
) -> Result<&ResolvedStorage<MetalDialect>, String> {
    launch
        .storage
        .values()
        .find(|storage| storage.id == id)
        .ok_or_else(|| format!("resolved launch omits storage#{}", id.0))
}

fn address_space(access: AccessMode) -> &'static str {
    match access {
        AccessMode::Read => "const device",
        AccessMode::Write | AccessMode::ReadWrite | AccessMode::Atomic => "device",
    }
}

fn resource_declaration(
    storage: &ResolvedStorage<MetalDialect>,
    access: AccessMode,
    slot: u32,
) -> String {
    let address = if matches!(storage.provenance.abi, Some(AbiRole::Parameter { .. }))
        && matches!(storage.layout, ResolvedMetalLayout::Scalar { .. })
    {
        "constant"
    } else {
        address_space(access)
    };
    format!(
        "{} {}* s{} [[buffer({slot})]]",
        address,
        storage_pointer_type(&storage.layout),
        storage.id.0,
    )
}

fn storage_pointer_type(layout: &ResolvedMetalLayout) -> &'static str {
    match layout {
        ResolvedMetalLayout::Scalar { .. } => "uchar",
        _ => scalar_type(layout),
    }
}

fn scalar_type(layout: &ResolvedMetalLayout) -> &'static str {
    match layout {
        ResolvedMetalLayout::Dense { dtype, .. }
        | ResolvedMetalLayout::PackedPlane { dtype, .. } => dtype_name(*dtype),
        ResolvedMetalLayout::Scalar { parameters, .. } => parameters
            .first()
            .map(|parameter| dtype_name(parameter.dtype))
            .unwrap_or("uchar"),
    }
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::I32 => "int",
        DType::U32 => "uint",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        DType::F32 => "float",
    }
}

struct Renderer<'a> {
    launch: &'a ResolvedLaunch<MetalDialect>,
    lines: Vec<String>,
    indent: usize,
    values: BTreeMap<ScalarValueId, String>,
    tuples: BTreeMap<ScalarValueId, Vec<ScalarValueId>>,
    tensors: BTreeMap<ScalarValueId, TensorValue>,
    places: BTreeMap<ScalarValueId, String>,
    declared: BTreeSet<ScalarValueId>,
    history: Vec<String>,
}

#[derive(Clone)]
struct TensorValue {
    planes: Vec<TensorPlane>,
    accessor: Option<PacketAccessor>,
}

#[derive(Clone)]
struct PacketAccessor {
    representation: String,
    name: String,
}

#[derive(Clone)]
struct TensorPlane {
    storage: ResolvedStorageId,
    offset: String,
    strides: Vec<u64>,
}

impl<'a> Renderer<'a> {
    fn new(launch: &'a ResolvedLaunch<MetalDialect>) -> Self {
        Self {
            launch,
            lines: Vec::new(),
            indent: 0,
            values: BTreeMap::new(),
            tuples: BTreeMap::new(),
            tensors: BTreeMap::new(),
            places: BTreeMap::new(),
            declared: BTreeSet::new(),
            history: Vec::new(),
        }
    }

    fn line(&mut self, line: impl Into<String>) {
        self.lines
            .push(format!("{}{}", "  ".repeat(self.indent), line.into()));
    }

    fn step(&mut self, step: &ResolvedKernelStep<MetalDialect>) -> Result<(), String> {
        match step {
            ResolvedKernelStep::MappedTask {
                mapping,
                bindings,
                instructions,
                ..
            } => {
                let first = instructions.iter().next().unwrap();
                for (axis, map) in mapping.axes.iter().enumerate() {
                    let values = first.axis_values.get(axis).cloned().unwrap_or_default();
                    let components = first
                        .axis_component_extents
                        .get(axis)
                        .cloned()
                        .unwrap_or_default();
                    let extent = mapping.logical_extents[axis];
                    match map {
                        ResolvedAxisMap::Grid {
                            workgroup_axis,
                            participant_axis,
                            mode,
                            ..
                        } => {
                            let coordinate = format!(
                                "(long(tg_pos[{workgroup_axis}]) * long({}) + long(tid[{participant_axis}]))",
                                self.launch.geometry.participants_per_workgroup[*participant_axis as usize]
                            );
                            match mode {
                                seismic_realization::executable::ResolvedGridMapping::OnePass => {
                                    self.bind_axis(&values, &components, &coordinate)?;
                                    self.line(format!("if ({coordinate} < {extent}l) {{"));
                                }
                                seismic_realization::executable::ResolvedGridMapping::GridStride { stride } => {
                                    let grid = format!("grid_{}_{}", first.task.0, axis);
                                    self.line(format!("for (long {grid} = {coordinate}; {grid} < {extent}l; {grid} += {stride}l) {{"));
                                    self.bind_axis(&values, &components, &grid)?;
                                }
                            }
                            self.indent += 1;
                        }
                        ResolvedAxisMap::Serial { .. } => {
                            let coordinate = format!("axis_{}_{}", first.task.0, axis);
                            self.line(format!(
                                "for (long {coordinate} = 0; {coordinate} < {extent}l; ++{coordinate}) {{"
                            ));
                            self.indent += 1;
                            self.bind_axis(&values, &components, &coordinate)?;
                        }
                        ResolvedAxisMap::SubgroupLane { .. } => {
                            self.bind_axis(&values, &components, "long(tid.x & 31u)")?;
                        }
                    }
                }
                for (value, bindings) in &first.value_bindings {
                    self.tensors.insert(
                        *value,
                        self.tensor_from_ids(bindings.iter().map(|binding| binding.storage))?,
                    );
                }
                self.bind_inputs(first, bindings)?;
                self.bind_tensor_outputs(first, bindings)?;
                for instruction in instructions.iter() {
                    self.instruction(instruction)?;
                }
                self.store_outputs(first, bindings)?;
                for map in mapping.axes.iter().rev() {
                    if matches!(
                        map,
                        ResolvedAxisMap::Grid { .. } | ResolvedAxisMap::Serial { .. }
                    ) {
                        self.indent = self.indent.saturating_sub(1);
                        self.line("}");
                    }
                }
            }
            ResolvedKernelStep::Barrier { scope, .. } => self.line(match scope {
                seismic_realization::executable::BarrierScope::Subgroup => {
                    "simdgroup_barrier(mem_flags::mem_device);"
                }
                seismic_realization::executable::BarrierScope::Workgroup => {
                    "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);"
                }
            }),
            ResolvedKernelStep::Publish { .. } => {}
        }
        Ok(())
    }

    fn instruction(&mut self, instruction: &ResolvedMetalInstruction) -> Result<(), String> {
        self.history.push(format!("{:?}", instruction.scalar.kind));
        let result = instruction.scalar.result;
        let expression = match &instruction.scalar.kind {
            I::NoOp => return Ok(()),
            I::Literal(ScalarLiteral::Int(value)) => Some(format!("{value}l")),
            I::Literal(ScalarLiteral::Float(bits)) => {
                Some(format!("double({:?})", f64::from_bits(*bits)))
            }
            I::Literal(ScalarLiteral::Bool(value)) => Some(value.to_string()),
            I::Shape(value) => Some(format!("{value}l")),
            I::Tuple(values) => {
                self.tuples.insert(
                    result.ok_or("tuple instruction has no result")?,
                    values.clone(),
                );
                None
            }
            I::Range(lo, hi) => {
                self.tuples.insert(
                    result.ok_or("range instruction has no result")?,
                    vec![*lo, *hi],
                );
                None
            }
            I::Field { value, index } => Some(self.field(*value, *index)?),
            I::Storage {
                operation,
                bindings,
                inputs,
                fill_bits,
            } => {
                let result = result.ok_or("storage instruction has no result")?;
                let tensor =
                    self.tensor_from_ids(bindings.iter().map(|binding| binding.storage))?;
                match operation {
                    StorageOperation::Construct => {}
                    StorageOperation::Fill => {
                        let bits = fill_bits.ok_or("fill operation has no value bits")?;
                        for plane in &tensor.planes {
                            let storage = self.storage(plane.storage)?;
                            let elements = layout_elements(&storage.layout);
                            self.line(format!(
                                "for (ulong fill = 0; fill < {elements}ul; ++fill) s{}[fill] = {}({:?});",
                                plane.storage.0,
                                scalar_type(&storage.layout),
                                f64::from_bits(bits)
                            ));
                        }
                    }
                    StorageOperation::Snapshot | StorageOperation::Materialize => {
                        let source = self
                            .tensors
                            .get(inputs.first().ok_or("storage copy has no source")?)
                            .cloned()
                            .ok_or("storage copy source is not a tensor")?;
                        self.copy_tensor(&tensor, &source)?;
                    }
                    StorageOperation::Decode => {
                        let source = self
                            .tensors
                            .get(inputs.first().ok_or("packed decode has no source")?)
                            .cloned()
                            .ok_or("packed decode source is not a tensor")?;
                        let destination = tensor
                            .planes
                            .first()
                            .ok_or("packed decode destination has no plane")?;
                        let elements = layout_elements(&self.storage(destination.storage)?.layout);
                        let decoded = self.decode_element(&source, "decode")?;
                        self.line(format!(
                            "for (ulong decode = 0; decode < {elements}ul; ++decode) s{}[decode] = {decoded};",
                            destination.storage.0
                        ));
                    }
                }
                self.tensors.insert(result, tensor);
                None
            }
            I::View { view, base } => {
                let logical = instruction
                    .views
                    .get(view)
                    .ok_or("Metal view descriptor is absent")?;
                let mut tensor = match self.tensors.get(base).cloned() {
                    Some(tensor) => tensor,
                    None => self.tensor_from_ids(
                        instruction
                            .view_bindings
                            .get(view)
                            .ok_or("Metal view has no resolved storage binding")?
                            .iter()
                            .map(|binding| binding.storage),
                    )?,
                };
                match &logical.transform {
                    LocalViewTransform::Identity => {}
                    LocalViewTransform::Transpose { permutation } => {
                        for plane in &mut tensor.planes {
                            plane.strides = permutation
                                .iter()
                                .map(|axis| {
                                    plane
                                        .strides
                                        .get(*axis as usize)
                                        .copied()
                                        .ok_or("Metal transpose axis is out of bounds")
                                })
                                .collect::<Result<_, _>>()?;
                        }
                    }
                    LocalViewTransform::Reshape { .. } => {
                        let Type::Tensor(result_ty) = instruction
                            .value_types
                            .get(&result.ok_or("view has no result")?)
                            .ok_or("view result type is absent")?
                        else {
                            return Err("reshape view result is not a tensor".into());
                        };
                        for plane in &mut tensor.planes {
                            let storage = self.storage(plane.storage)?;
                            plane.strides = compact_strides(&result_ty.shape, storage)?;
                        }
                    }
                }
                self.tensors
                    .insert(result.ok_or("view has no result")?, tensor);
                None
            }
            I::Index { base, indices } => {
                let mut tensor = match self.tensors.get(base).cloned() {
                    Some(tensor) => tensor,
                    None if !instruction.access_bindings.is_empty() => self
                        .tensor_from_ids(instruction.access_bindings.iter().copied())?,
                    None => return Err(
                        format!(
                            "Metal task#{} index base value#{} is not a tensor; known tensors {:?}; type {:?}; inputs {:?}; operand values {:?}; history {:?}",
                            instruction.task.0,
                            base.0,
                            self.tensors.keys().collect::<Vec<_>>(),
                            instruction.value_types.get(base),
                            instruction.input_operands,
                            instruction.operand_values,
                            self.history,
                        )
                    ),
                };
                for plane in &mut tensor.planes {
                    let mut offset = plane.offset.clone();
                    let mut retained = Vec::new();
                    for (axis, stride) in plane.strides.iter().copied().enumerate() {
                        match indices.get(axis) {
                            Some(ScalarIndex::Point(value))
                            | Some(ScalarIndex::Coordinate(value)) => {
                                offset = format!(
                                    "({offset} + ulong({}) * {stride}ul)",
                                    self.value(*value)
                                );
                            }
                            Some(ScalarIndex::Range { start, .. }) => {
                                if let Some(start) = start {
                                    offset = format!(
                                        "({offset} + ulong({}) * {stride}ul)",
                                        self.value(*start)
                                    );
                                }
                                retained.push(stride);
                            }
                            Some(ScalarIndex::Slice(_)) | None => retained.push(stride),
                        }
                    }
                    plane.offset = offset;
                    plane.strides = retained;
                }
                let result = result.ok_or("index has no result")?;
                if matches!(instruction.value_types.get(&result), Some(Type::Tensor(_))) {
                    self.tensors.insert(result, tensor);
                    None
                } else {
                    let plane = tensor.planes.first().ok_or("indexed tensor has no plane")?;
                    let expression = if tensor.accessor.is_some() {
                        self.accessor_element(&tensor, &plane.offset)?
                    } else {
                        format!("s{}[{}]", plane.storage.0, plane.offset)
                    };
                    if tensor.accessor.is_none() {
                        self.places.insert(result, expression.clone());
                    }
                    Some(expression)
                }
            }
            I::Cast { dtype, value } => {
                Some(format!("{}({})", dtype_name(*dtype), self.value(*value)))
            }
            I::Unary { op, value } => Some(format!("({}{})", unary(*op), self.value(*value))),
            I::Binary { op, lhs, rhs } => Some(format!(
                "({} {} {})",
                self.value(*lhs),
                binary(*op),
                self.value(*rhs)
            )),
            I::Math { op, arguments } => Some(render_math(*op, arguments, &self.values)?),
            I::Select {
                condition,
                then_value,
                else_value,
            } => Some(format!(
                "({} ? {} : {})",
                self.value(*condition),
                self.value(*then_value),
                self.value(*else_value)
            )),
            I::Extent { base, axis } => {
                let Type::Tensor(tensor) = instruction
                    .value_types
                    .get(base)
                    .ok_or("extent base type is absent")?
                else {
                    return Err("extent base is not a tensor".into());
                };
                let extent = tensor
                    .shape
                    .get(*axis)
                    .and_then(|value| value.as_constant())
                    .ok_or("resolved Metal extent remained symbolic")?;
                Some(format!("{extent}l"))
            }
            I::Intrinsic {
                operation,
                arguments,
            } => self.render_intrinsic(*operation, arguments, instruction, result)?,
            I::Accessor { base, name } => {
                let mut tensor = self
                    .tensors
                    .get(base)
                    .cloned()
                    .ok_or("packet accessor base is not a tensor")?;
                let Type::Tensor(base_ty) = instruction
                    .value_types
                    .get(base)
                    .ok_or("packet accessor base type is absent")?
                else {
                    return Err("packet accessor base is not tensor-typed".into());
                };
                let seismic_lang::types::Elem::Repr(representation) = &base_ty.elem else {
                    return Err("packet accessor requires represented storage".into());
                };
                tensor.accessor = Some(PacketAccessor {
                    representation: representation.clone(),
                    name: name.clone(),
                });
                if let Some(Type::Tensor(result_ty)) =
                    result.and_then(|id| instruction.value_types.get(&id))
                {
                    let strides = compact_shape_strides(&result_ty.shape)?;
                    for plane in &mut tensor.planes {
                        plane.strides = strides.clone();
                    }
                }
                self.tensors
                    .insert(result.ok_or("packet accessor has no result")?, tensor);
                None
            }
            I::Geometry {
                base,
                axis,
                valid: _,
            } => {
                let Type::Tensor(tensor) = instruction
                    .value_types
                    .get(base)
                    .ok_or("geometry base type is absent")?
                else {
                    return Err("geometry base is not a tensor".into());
                };
                let extent = tensor
                    .shape
                    .get(*axis)
                    .and_then(|value| value.as_constant())
                    .ok_or("resolved Metal geometry remained symbolic")?;
                Some(format!("{extent}l"))
            }
            I::Atomic { op, place, value } => {
                let target = self
                    .places
                    .get(place)
                    .cloned()
                    .ok_or("atomic target is not an indexed place")?;
                let function = match op {
                    BinaryOp::Add => "atomic_fetch_add_explicit",
                    BinaryOp::BitAnd => "atomic_fetch_and_explicit",
                    BinaryOp::BitOr => "atomic_fetch_or_explicit",
                    BinaryOp::BitXor => "atomic_fetch_xor_explicit",
                    _ => return Err(format!("unsupported Metal atomic operator `{}`", op.text())),
                };
                let dtype = match instruction.value_types.get(place) {
                    Some(Type::Scalar(dtype)) => *dtype,
                    _ => return Err("atomic place has no scalar dtype".into()),
                };
                if dtype.is_float() && *op != BinaryOp::Add {
                    return Err("Metal floating atomics support addition only".into());
                }
                let atomic = match dtype {
                    DType::I32 => "atomic_int",
                    DType::U32 => "atomic_uint",
                    DType::F32 => "atomic_float",
                    _ => return Err(format!("Metal has no atomic {} storage", dtype.name())),
                };
                self.line(format!(
                    "{function}(reinterpret_cast<device {atomic}*>(&({target})), {}({}), memory_order_relaxed);",
                    dtype_name(dtype),
                    self.value(*value)
                ));
                None
            }
            I::Assign { target, op, value } => {
                let target = self
                    .places
                    .get(target)
                    .cloned()
                    .ok_or("assignment target is not an indexed place")?;
                self.line(format!("{target} {} {};", assign(*op), self.value(*value)));
                None
            }
            I::Publish { value, destination } => {
                let destination = self
                    .places
                    .get(destination)
                    .cloned()
                    .ok_or("publication destination is not an indexed place")?;
                self.line(format!("{destination} = {};", self.value(*value)));
                None
            }
            I::Conditional {
                condition,
                then_body,
                else_body,
            } => {
                for nested in then_body.iter().chain(else_body) {
                    if let Some(value) = nested.result {
                        if self.declared.contains(&value) {
                            continue;
                        }
                        if let Some(ty @ (Type::Scalar(_) | Type::Index { .. })) =
                            instruction.value_types.get(&value)
                        {
                            self.line(format!("{} v{};", type_name(ty)?, value.0));
                            self.values.insert(value, format!("v{}", value.0));
                            self.declared.insert(value);
                        }
                    }
                }
                self.line(format!("if ({}) {{", self.value(*condition)));
                self.indent += 1;
                for nested in then_body {
                    self.resolved_scalar(nested, instruction)?;
                }
                self.indent -= 1;
                self.line("} else {");
                self.indent += 1;
                for nested in else_body {
                    self.resolved_scalar(nested, instruction)?;
                }
                self.indent -= 1;
                self.line("}");
                None
            }
            I::ConditionalMerge { cases } => Some(self.render_merge(cases)?),
            I::Yield(_) | I::Return { .. } => None,
        };
        if let (Some(result), Some(expression)) = (result, expression) {
            let ty = instruction
                .value_types
                .get(&result)
                .ok_or_else(|| format!("value#{} has no type", result.0))?;
            let name = format!("v{}", result.0);
            if self.declared.contains(&result) {
                self.line(format!("{name} = {expression};"));
            } else {
                self.line(format!("{} {name} = {expression};", type_name(ty)?));
                self.declared.insert(result);
            }
            self.values.insert(result, name);
        }
        Ok(())
    }

    fn bind_inputs(
        &mut self,
        instruction: &ResolvedMetalInstruction,
        bindings: &[ResolvedOperandTransport],
    ) -> Result<(), String> {
        for binding in bindings.iter() {
            if !instruction.input_operands.contains(&binding.operand) {
                continue;
            }
            let value = instruction
                .operand_values
                .get(&binding.operand)
                .copied()
                .ok_or_else(|| {
                    format!("input operand#{} has no scalar value", binding.operand.0)
                })?;
            let ty = instruction
                .value_types
                .get(&value)
                .ok_or("input operand value has no type")?;
            self.bind_transport_value(value, ty, &binding.transport)?;
        }
        Ok(())
    }

    fn store_outputs(
        &mut self,
        instruction: &ResolvedMetalInstruction,
        bindings: &[ResolvedOperandTransport],
    ) -> Result<(), String> {
        for binding in bindings.iter() {
            if !instruction.output_operands.contains(&binding.operand) {
                continue;
            }
            let value = instruction
                .operand_values
                .get(&binding.operand)
                .copied()
                .ok_or_else(|| {
                    format!("output operand#{} has no scalar value", binding.operand.0)
                })?;
            let ty = instruction
                .value_types
                .get(&value)
                .ok_or("output operand value has no type")?;
            self.store_transport_value(value, ty, &binding.transport)?;
        }
        Ok(())
    }

    fn bind_tensor_outputs(
        &mut self,
        instruction: &ResolvedMetalInstruction,
        bindings: &[ResolvedOperandTransport],
    ) -> Result<(), String> {
        for binding in bindings.iter() {
            if !instruction.output_operands.contains(&binding.operand) {
                continue;
            }
            let Some(value) = instruction.operand_values.get(&binding.operand).copied() else {
                continue;
            };
            if !matches!(instruction.value_types.get(&value), Some(Type::Tensor(_))) {
                continue;
            }
            if let ResolvedValueTransport::Storage(ids) = &binding.transport {
                self.tensors
                    .insert(value, self.tensor_from_ids(ids.iter().copied())?);
            }
        }
        Ok(())
    }

    fn bind_transport_value(
        &mut self,
        value: ScalarValueId,
        ty: &Type,
        transport: &ResolvedValueTransport,
    ) -> Result<(), String> {
        match (ty, transport) {
            (Type::Void, ResolvedValueTransport::Void) => {}
            (Type::Tuple(types), ResolvedValueTransport::Tuple(fields))
                if types.len() == fields.len() =>
            {
                let synthetic = types
                    .iter()
                    .enumerate()
                    .map(|(index, ty)| {
                        let field = ScalarValueId(value.0.wrapping_add(0x4000_0000 + index as u32));
                        self.bind_transport_value(field, ty, fields.iter().nth(index).unwrap())?;
                        Ok(field)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                self.tuples.insert(value, synthetic);
            }
            (Type::Tensor(_), ResolvedValueTransport::Storage(ids)) => {
                self.tensors
                    .insert(value, self.tensor_from_ids(ids.iter().copied())?);
            }
            (Type::Range { .. }, ResolvedValueTransport::Storage(ids)) if ids.len() == 1 => {
                let storage = *ids.iter().next().unwrap();
                let start = ScalarValueId(value.0.wrapping_add(0x6000_0000));
                let end = ScalarValueId(value.0.wrapping_add(0x6000_0001));
                self.values
                    .insert(start, self.scalar_storage_field(storage, 0)?);
                self.values
                    .insert(end, self.scalar_storage_field(storage, 1)?);
                self.tuples.insert(value, vec![start, end]);
            }
            (_, ResolvedValueTransport::Storage(ids)) if ids.len() == 1 => {
                let storage = *ids.iter().next().unwrap();
                self.values
                    .insert(value, self.scalar_storage_field(storage, 0)?);
            }
            (_, ResolvedValueTransport::Kernel(_)) => {
                return Err("Metal kernel-value input transport has no local producer".into())
            }
            _ => return Err("Metal operand transport disagrees with its value type".into()),
        }
        Ok(())
    }

    fn store_transport_value(
        &mut self,
        value: ScalarValueId,
        ty: &Type,
        transport: &ResolvedValueTransport,
    ) -> Result<(), String> {
        match (ty, transport) {
            (Type::Void, ResolvedValueTransport::Void) => {}
            (Type::Tuple(types), ResolvedValueTransport::Tuple(fields))
                if types.len() == fields.len() =>
            {
                let values = self
                    .tuples
                    .get(&value)
                    .cloned()
                    .ok_or("tuple output has no fields")?;
                for ((ty, transport), value) in types.iter().zip(fields.iter()).zip(values) {
                    self.store_transport_value(value, ty, transport)?;
                }
            }
            (Type::Tensor(tensor_ty), ResolvedValueTransport::Storage(ids)) => {
                let source = self
                    .tensors
                    .get(&value)
                    .cloned()
                    .ok_or("tensor output has no physical value")?;
                let destination = self.tensor_from_ids(ids.iter().copied())?;
                let aliases = source.planes.len() == destination.planes.len()
                    && source.planes.iter().zip(&destination.planes).all(
                        |(source, destination)| {
                            source.storage == destination.storage
                                && source.offset == destination.offset
                        },
                    );
                if !aliases {
                    self.copy_tensor_elements(
                        &destination,
                        &source,
                        resolved_elements(&tensor_ty.shape)?,
                    )?;
                }
            }
            (Type::Range { .. }, ResolvedValueTransport::Storage(ids)) if ids.len() == 1 => {
                let storage = ids.iter().next().unwrap();
                let fields = self
                    .tuples
                    .get(&value)
                    .cloned()
                    .ok_or("range output has no endpoints")?;
                if fields.len() != 2 {
                    return Err("range output must have two endpoints".into());
                }
                for (index, field) in fields.into_iter().enumerate() {
                    let place = self.scalar_storage_place(*storage, index)?;
                    self.line(format!("{place} = {};", self.value(field)));
                }
            }
            (_, ResolvedValueTransport::Storage(ids)) if ids.len() == 1 => {
                let storage = *ids.iter().next().unwrap();
                let place = self.scalar_storage_place(storage, 0)?;
                self.line(format!("{place} = {};", self.value(value)));
            }
            (_, ResolvedValueTransport::Kernel(_)) => {}
            _ => return Err("Metal output transport disagrees with its value type".into()),
        }
        Ok(())
    }

    fn tensor_from_ids(
        &self,
        ids: impl IntoIterator<Item = ResolvedStorageId>,
    ) -> Result<TensorValue, String> {
        let planes = ids
            .into_iter()
            .map(|id| {
                let storage = self.storage(id)?;
                let (byte_offset, strides) = match &storage.provenance.address {
                    ResolvedPhysicalAddress::DenseAffine {
                        byte_offset,
                        byte_strides,
                    } => {
                        let bytes = layout_element_bytes(&storage.layout);
                        (
                            byte_offset / bytes,
                            byte_strides.iter().map(|stride| stride / bytes).collect(),
                        )
                    }
                    ResolvedPhysicalAddress::Representation {
                        logical_strides, ..
                    } => (0, logical_strides.clone()),
                };
                Ok(TensorPlane {
                    storage: id,
                    offset: format!("{byte_offset}ul"),
                    strides,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(TensorValue {
            planes,
            accessor: None,
        })
    }

    fn storage(&self, id: ResolvedStorageId) -> Result<&ResolvedStorage<MetalDialect>, String> {
        self.launch
            .storage
            .values()
            .find(|storage| storage.id == id)
            .ok_or_else(|| format!("Metal instruction names absent storage#{}", id.0))
    }

    fn render_intrinsic(
        &mut self,
        operation: seismic_lang::intrinsics::Operation,
        arguments: &[ScalarValueId],
        instruction: &ResolvedMetalInstruction,
        result: Option<ScalarValueId>,
    ) -> Result<Option<String>, String> {
        use seismic_lang::intrinsics::Operation;
        let argument = |index: usize| {
            arguments
                .get(index)
                .map(|value| self.value(*value))
                .ok_or_else(|| format!("{} intrinsic argument {index} is absent", operation.name()))
        };
        Ok(match operation {
            Operation::LaneIndex => Some("int(tid.x & 31u)".into()),
            Operation::ShuffleIndex => Some(format!(
                "simd_shuffle({}, uint({}))",
                argument(0)?,
                argument(1)?
            )),
            Operation::SimdSum => Some(format!("simd_sum({})", argument(0)?)),
            Operation::SimdMax => Some(format!("simd_max({})", argument(0)?)),
            Operation::SimdMin => Some(format!("simd_min({})", argument(0)?)),
            Operation::Matrix => {
                let result = result.ok_or("matrix declaration has no result")?;
                let ty = instruction
                    .value_types
                    .get(&result)
                    .ok_or("matrix declaration result type is absent")?;
                let name = format!("v{}", result.0);
                self.line(format!("{} {name};", type_name(ty)?));
                self.values.insert(result, name);
                None
            }
            Operation::MatrixLoad | Operation::MatrixLoadTranspose => {
                let fragment = argument(0)?;
                let tensor = self
                    .tensors
                    .get(arguments.get(1).ok_or("matrix load has no tensor")?)
                    .ok_or("matrix load operand is not a tensor")?;
                let plane = tensor
                    .planes
                    .first()
                    .ok_or("matrix load tensor has no plane")?;
                let leading = plane.strides.first().copied().unwrap_or(1);
                let row = argument(2)?;
                let column = argument(3)?;
                if operation == Operation::MatrixLoadTranspose {
                    self.line(format!("simdgroup_load({fragment}, s{} + {}, {leading}ul, ulong2(ulong({column}), ulong({row})), true);", plane.storage.0, plane.offset));
                } else {
                    self.line(format!("simdgroup_load({fragment}, s{} + {}, {leading}ul, ulong2(ulong({column}), ulong({row})));", plane.storage.0, plane.offset));
                }
                None
            }
            Operation::MatrixStore => {
                let fragment = argument(0)?;
                let tensor = self
                    .tensors
                    .get(arguments.get(1).ok_or("matrix store has no tensor")?)
                    .ok_or("matrix store operand is not a tensor")?;
                let plane = tensor
                    .planes
                    .first()
                    .ok_or("matrix store tensor has no plane")?;
                let leading = plane.strides.first().copied().unwrap_or(1);
                self.line(format!("simdgroup_store({fragment}, s{} + {}, {leading}ul, ulong2(ulong({}), ulong({})));", plane.storage.0, plane.offset, argument(3)?, argument(2)?));
                None
            }
            Operation::MatrixMultiplyAccumulate => {
                self.line(format!(
                    "simdgroup_multiply_accumulate({}, {}, {}, {});",
                    argument(0)?,
                    argument(1)?,
                    argument(2)?,
                    argument(3)?
                ));
                None
            }
            Operation::MatrixMatmul | Operation::MatrixMatmulAdd => {
                let result = result.ok_or("logical matrix intrinsic has no result")?;
                let output = self
                    .tensors
                    .get(&result)
                    .cloned()
                    .ok_or("logical matrix result has no bound output storage")?;
                let left = self
                    .tensors
                    .get(arguments.first().ok_or("matmul has no left operand")?)
                    .cloned()
                    .ok_or("matmul left operand is not a tensor")?;
                let right = self
                    .tensors
                    .get(arguments.get(1).ok_or("matmul has no right operand")?)
                    .cloned()
                    .ok_or("matmul right operand is not a tensor")?;
                let Type::Tensor(output_ty) = instruction
                    .value_types
                    .get(&result)
                    .ok_or("matmul result type is absent")?
                else {
                    return Err("matmul result is not tensor-typed".into());
                };
                let Type::Tensor(left_ty) = instruction
                    .value_types
                    .get(&arguments[0])
                    .ok_or("matmul left type is absent")?
                else {
                    return Err("matmul left operand is not tensor-typed".into());
                };
                let extent = |shape: &[seismic_lang::sym::Sym], axis: usize| {
                    shape
                        .get(axis)
                        .and_then(|extent| extent.as_constant())
                        .and_then(|extent| u64::try_from(extent).ok())
                        .ok_or("resolved matmul extent remained symbolic")
                };
                let (m, n, k) = (
                    extent(&output_ty.shape, 0)?,
                    extent(&output_ty.shape, 1)?,
                    extent(&left_ty.shape, 1)?,
                );
                let (a, b, c) = (
                    left.planes
                        .first()
                        .ok_or("matmul left tensor has no plane")?,
                    right
                        .planes
                        .first()
                        .ok_or("matmul right tensor has no plane")?,
                    output
                        .planes
                        .first()
                        .ok_or("matmul result tensor has no plane")?,
                );
                if a.strides.len() != 2 || b.strides.len() != 2 || c.strides.len() != 2 {
                    return Err("logical matrix intrinsic requires rank-two operands".into());
                }
                self.line(format!("for (ulong mm_m = 0; mm_m < {m}ul; ++mm_m) {{"));
                self.indent += 1;
                self.line(format!("for (ulong mm_n = 0; mm_n < {n}ul; ++mm_n) {{"));
                self.indent += 1;
                let initial = if operation == Operation::MatrixMatmulAdd {
                    format!(
                        "float(s{}[{} + mm_m * {}ul + mm_n * {}ul])",
                        c.storage.0, c.offset, c.strides[0], c.strides[1]
                    )
                } else {
                    "0.0f".into()
                };
                self.line(format!("float mm_acc = {initial};"));
                self.line(format!("for (ulong mm_k = 0; mm_k < {k}ul; ++mm_k) mm_acc = fma(float(s{}[{} + mm_m * {}ul + mm_k * {}ul]), float(s{}[{} + mm_n * {}ul + mm_k * {}ul]), mm_acc);", a.storage.0, a.offset, a.strides[0], a.strides[1], b.storage.0, b.offset, b.strides[0], b.strides[1]));
                self.line(format!(
                    "s{}[{} + mm_m * {}ul + mm_n * {}ul] = mm_acc;",
                    c.storage.0, c.offset, c.strides[0], c.strides[1]
                ));
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
                None
            }
        })
    }

    fn render_merge(
        &self,
        cases: &[seismic_compiler::terminal::ConditionalCase],
    ) -> Result<String, String> {
        let mut cases = cases.iter().rev();
        let last = cases.next().ok_or("conditional merge has no cases")?;
        let mut expression = self.value(last.value);
        for case in cases {
            let condition = if case.predicates.is_empty() {
                "true".into()
            } else {
                case.predicates
                    .iter()
                    .map(|(value, expected)| {
                        if *expected {
                            format!("bool({})", self.value(*value))
                        } else {
                            format!("!bool({})", self.value(*value))
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" && ")
            };
            expression = format!(
                "(({condition}) ? {} : {expression})",
                self.value(case.value)
            );
        }
        Ok(expression)
    }

    fn accessor_element(&self, tensor: &TensorValue, index: &str) -> Result<String, String> {
        let accessor = tensor
            .accessor
            .as_ref()
            .ok_or("tensor has no packet accessor")?;
        let representation = repr::lookup(&accessor.representation)
            .ok_or_else(|| format!("unknown representation `{}`", accessor.representation))?;
        if accessor.name == "scale" || accessor.name == "bias" {
            let coefficient = representation
                .coefficient(accessor.name == "bias")
                .ok_or_else(|| format!("representation has no `{}` coefficient", accessor.name))?;
            self.coefficient(
                tensor,
                &format!("(({index}) * {}ul)", representation.group),
                &coefficient,
            )
        } else {
            let plane = representation
                .plane(&accessor.name)
                .ok_or_else(|| format!("unknown physical plane `{}`", accessor.name))?;
            self.raw_plane(tensor, plane.name, index)
        }
    }

    fn decode_element(&self, tensor: &TensorValue, index: &str) -> Result<String, String> {
        let representation_name = tensor
            .planes
            .iter()
            .find_map(|plane| match &self.storage(plane.storage).ok()?.layout {
                ResolvedMetalLayout::PackedPlane { representation, .. } => Some(representation),
                _ => None,
            })
            .ok_or("packed decode source has no representation")?;
        let representation = repr::lookup(representation_name)
            .ok_or_else(|| format!("unknown representation `{representation_name}`"))?;
        let logical = format!(
            "({} + ulong({index}))",
            tensor
                .planes
                .first()
                .ok_or("packed tensor has no planes")?
                .offset
        );
        let code = self.packed_plane(
            tensor,
            "words",
            &logical,
            representation.bits,
            &representation.code,
        )?;
        let scale = self.coefficient(
            tensor,
            &logical,
            &representation
                .coefficient(false)
                .ok_or("representation has no scale")?,
        )?;
        let bias = representation
            .coefficient(true)
            .map(|coefficient| self.coefficient(tensor, &logical, &coefficient))
            .transpose()?
            .unwrap_or_else(|| "0.0f".into());
        Ok(format!("fma(float({code}), float({scale}), float({bias}))"))
    }

    fn coefficient(
        &self,
        tensor: &TensorValue,
        logical: &str,
        coefficient: &repr::Coefficient,
    ) -> Result<String, String> {
        match coefficient {
            repr::Coefficient::Direct { plane } => self.raw_plane(
                tensor,
                plane.name,
                &format!("(ulong({logical}) / {}ul)", plane.group),
            ),
            repr::Coefficient::Product {
                factor,
                coefficients,
                field,
                sign,
            } => {
                let factor_value = self.raw_plane(
                    tensor,
                    factor.name,
                    &format!("(ulong({logical}) / {}ul)", factor.group),
                )?;
                let entry = format!(
                    "((ulong({logical}) / {}ul) * {}ul + {}ul)",
                    coefficients.group, coefficients.fields, field
                );
                let repr::PlaneEncoding::Packed {
                    bits,
                    interpretation,
                } = &coefficients.encoding
                else {
                    return Err("hierarchical coefficient plane is not packed".into());
                };
                let value =
                    self.packed_plane(tensor, coefficients.name, &entry, *bits, interpretation)?;
                Ok(format!(
                    "(float({factor_value}) * float({value}) * {}.0f)",
                    sign
                ))
            }
        }
    }

    fn raw_plane(&self, tensor: &TensorValue, name: &str, index: &str) -> Result<String, String> {
        let plane = tensor.planes.iter().find(|plane| {
            matches!(&self.storage(plane.storage).ok().map(|s| &s.layout), Some(ResolvedMetalLayout::PackedPlane { plane: candidate, .. }) if candidate == name)
        }).ok_or_else(|| format!("packed tensor has no `{name}` plane"))?;
        Ok(format!("s{}[ulong({index})]", plane.storage.0))
    }

    fn packed_plane(
        &self,
        tensor: &TensorValue,
        name: &str,
        index: &str,
        bits: u32,
        interpretation: &repr::CodeInterpretation,
    ) -> Result<String, String> {
        let bit = format!("(ulong({index}) * {bits}ul)");
        let low = self.raw_plane(tensor, name, &format!("({bit} / 32ul)"))?;
        let high = self.raw_plane(tensor, name, &format!("({bit} / 32ul + 1ul)"))?;
        let mask = if bits == 32 {
            u32::MAX
        } else {
            (1u32 << bits) - 1
        };
        let shift = format!("uint({bit} % 32ul)");
        let raw = format!("uint((({shift} + {bits}u <= 32u) ? ({low} >> {shift}) : (({low} >> {shift}) | ({high} << (32u - {shift})))) & {mask}u)");
        Ok(decode_code(&raw, bits, interpretation))
    }

    fn scalar_storage_place(
        &self,
        id: ResolvedStorageId,
        field_index: usize,
    ) -> Result<String, String> {
        let storage = self.storage(id)?;
        let ResolvedMetalLayout::Scalar { parameters, .. } = &storage.layout else {
            return Err(format!("storage#{} is not scalar storage", id.0));
        };
        let layout = seismic_lang::abi::ScalarLayout::natural(parameters)?;
        let field = layout
            .fields
            .get(field_index)
            .ok_or_else(|| format!("storage#{} has no scalar field {field_index}", id.0))?;
        let qualifier = if matches!(storage.provenance.abi, Some(AbiRole::Parameter { .. })) {
            "constant"
        } else {
            "device"
        };
        Ok(format!(
            "(*reinterpret_cast<{qualifier} {}*>(s{} + {}ul))",
            dtype_name(field.parameter.dtype),
            id.0,
            field.offset
        ))
    }

    fn scalar_storage_field(
        &self,
        id: ResolvedStorageId,
        field_index: usize,
    ) -> Result<String, String> {
        self.scalar_storage_place(id, field_index)
    }

    fn copy_tensor(
        &mut self,
        destination: &TensorValue,
        source: &TensorValue,
    ) -> Result<(), String> {
        if destination.planes.len() != source.planes.len() {
            return Err("Metal tensor copy representation planes differ".into());
        }
        for (destination, source) in destination.planes.iter().zip(&source.planes) {
            let elements = layout_elements(&self.storage(destination.storage)?.layout);
            self.line(format!(
                "for (ulong copy = 0; copy < {elements}ul; ++copy) s{}[{} + copy] = s{}[{} + copy];",
                destination.storage.0, destination.offset, source.storage.0, source.offset
            ));
        }
        Ok(())
    }

    fn copy_tensor_elements(
        &mut self,
        destination: &TensorValue,
        source: &TensorValue,
        elements: u64,
    ) -> Result<(), String> {
        if destination.planes.len() != source.planes.len() {
            return Err("Metal tensor copy representation planes differ".into());
        }
        for (destination, source) in destination.planes.iter().zip(&source.planes) {
            self.line(format!(
                "for (ulong copy = 0; copy < {elements}ul; ++copy) s{}[{} + copy] = s{}[{} + copy];",
                destination.storage.0, destination.offset, source.storage.0, source.offset
            ));
        }
        Ok(())
    }

    fn resolved_scalar(
        &mut self,
        scalar: &seismic_compiler::terminal::ResolvedScalarInstruction,
        parent: &ResolvedMetalInstruction,
    ) -> Result<(), String> {
        let mut wrapped = parent.clone();
        wrapped.scalar = scalar.clone();
        self.instruction(&wrapped)
    }

    fn bind_axis(
        &mut self,
        values: &[ScalarValueId],
        components: &[u64],
        coordinate: &str,
    ) -> Result<(), String> {
        if values.len() != components.len() {
            return Err("Metal logical-axis binders disagree with component extents".into());
        }
        for (index, (value, extent)) in values.iter().zip(components).enumerate() {
            let trailing = components[index + 1..]
                .iter()
                .try_fold(1u64, |product, extent| product.checked_mul(*extent))
                .ok_or("Metal coordinate component stride overflow")?;
            let expression = if trailing == 1 {
                format!("(({coordinate}) % {extent}ul)")
            } else {
                format!("((({coordinate}) / {trailing}ul) % {extent}ul)")
            };
            self.values.insert(*value, format!("v{}", value.0));
            self.line(format!("long v{} = long({expression});", value.0));
        }
        Ok(())
    }

    fn value(&self, value: ScalarValueId) -> String {
        self.values
            .get(&value)
            .cloned()
            .unwrap_or_else(|| format!("v{}", value.0))
    }

    fn field(&self, value: ScalarValueId, index: usize) -> Result<String, String> {
        let fields = self
            .tuples
            .get(&value)
            .ok_or_else(|| format!("value#{} is not an aggregate", value.0))?;
        fields
            .get(index)
            .map(|value| self.value(*value))
            .ok_or_else(|| "aggregate field is out of bounds".into())
    }
}

fn type_name(ty: &Type) -> Result<&'static str, String> {
    match ty {
        Type::Scalar(dtype) => Ok(dtype_name(*dtype)),
        Type::Index { .. } => Ok("long"),
        Type::CapabilityValue { name, elem, .. } if name.contains("simdgroup_matrix") => elem
            .as_ref()
            .and_then(|elem| elem.read_dtype())
            .map(|dtype| match dtype {
                DType::F16 => "simdgroup_matrix<half, 8, 8>",
                DType::BF16 => "simdgroup_matrix<bfloat, 8, 8>",
                DType::F32 => "simdgroup_matrix<float, 8, 8>",
                _ => "simdgroup_matrix<float, 8, 8>",
            })
            .ok_or("Metal matrix capability value has no element dtype".into()),
        _ => Err(format!(
            "Metal value declaration has non-scalar type {ty:?}"
        )),
    }
}

fn layout_elements(layout: &ResolvedMetalLayout) -> u64 {
    match layout {
        ResolvedMetalLayout::Dense { elements, .. }
        | ResolvedMetalLayout::PackedPlane { elements, .. } => *elements,
        ResolvedMetalLayout::Scalar { .. } => 1,
    }
}

fn layout_element_bytes(layout: &ResolvedMetalLayout) -> u64 {
    match layout {
        ResolvedMetalLayout::Dense { dtype, .. }
        | ResolvedMetalLayout::PackedPlane { dtype, .. } => u64::from(dtype.bytes()),
        ResolvedMetalLayout::Scalar { bytes, .. } => *bytes,
    }
}

fn compact_strides(
    shape: &[seismic_lang::sym::Sym],
    storage: &ResolvedStorage<MetalDialect>,
) -> Result<Vec<u64>, String> {
    let mut stride = 1u64;
    let mut result = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = stride;
        let extent = shape[axis]
            .as_constant()
            .and_then(|value| u64::try_from(value).ok())
            .ok_or("resolved Metal reshape has a symbolic extent")?;
        stride = stride
            .checked_mul(extent)
            .ok_or("Metal reshape stride overflow")?;
    }
    if stride > layout_elements(&storage.layout) {
        return Err("Metal reshape exceeds its physical storage".into());
    }
    Ok(result)
}

fn compact_shape_strides(shape: &[seismic_lang::sym::Sym]) -> Result<Vec<u64>, String> {
    let mut stride = 1u64;
    let mut result = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = stride;
        let extent = shape[axis]
            .as_constant()
            .and_then(|value| u64::try_from(value).ok())
            .ok_or("resolved Metal tensor shape remained symbolic")?;
        stride = stride
            .checked_mul(extent)
            .ok_or("Metal tensor stride overflow")?;
    }
    Ok(result)
}

fn resolved_elements(shape: &[seismic_lang::sym::Sym]) -> Result<u64, String> {
    Ok(shape.iter().try_fold(1u64, |elements, extent| {
        let extent = extent
            .as_constant()
            .and_then(|value| u64::try_from(value).ok())
            .ok_or("resolved Metal tensor shape remained symbolic")?;
        elements
            .checked_mul(extent)
            .ok_or("Metal tensor element count overflow")
    })?)
}

fn decode_code(raw: &str, bits: u32, interpretation: &repr::CodeInterpretation) -> String {
    match interpretation {
        repr::CodeInterpretation::Unsigned => raw.into(),
        repr::CodeInterpretation::Offset(zero) => format!("(int({raw}) - {zero})"),
        repr::CodeInterpretation::TwosComplement => {
            format!("(int(uint({raw}) << {}u) >> {}u)", 32 - bits, 32 - bits)
        }
        repr::CodeInterpretation::Table(table) => {
            let mut expression = table.last().copied().unwrap_or(0).to_string();
            for (index, value) in table.iter().copied().enumerate().rev().skip(1) {
                expression = format!("(uint({raw}) == {index}u ? {value} : {expression})");
            }
            expression
        }
    }
}

fn assign(op: AssignOp) -> &'static str {
    op.text()
}

fn unary(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "-",
        UnaryOp::Not => "!",
        UnaryOp::BitNot => "~",
    }
}

fn binary(op: BinaryOp) -> &'static str {
    op.text()
}

fn render_math(
    op: seismic_lang::sir::Math,
    arguments: &[ScalarValueId],
    values: &BTreeMap<ScalarValueId, String>,
) -> Result<String, String> {
    let value = |index: usize| {
        arguments
            .get(index)
            .map(|id| {
                values
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| format!("v{}", id.0))
            })
            .ok_or("math argument is absent")
    };
    use seismic_lang::sir::Math;
    Ok(match op {
        Math::Fma => format!("fma({}, {}, {})", value(0)?, value(1)?, value(2)?),
        Math::Max => format!("max({}, {})", value(0)?, value(1)?),
        Math::Min => format!("min({}, {})", value(0)?, value(1)?),
        Math::Exp => format!("exp({})", value(0)?),
        Math::ExpFast => format!("fast::exp({})", value(0)?),
        Math::Rsqrt => format!("rsqrt({})", value(0)?),
        Math::Sqrt => format!("sqrt({})", value(0)?),
        Math::Log => format!("log({})", value(0)?),
        Math::Sin => format!("sin({})", value(0)?),
        Math::Cos => format!("cos({})", value(0)?),
        Math::Abs => format!("abs({})", value(0)?),
    })
}

pub fn assemble(encoded: EncodedPlan<MetalDialect, EncodedLaunch>) -> Result<Emitted, String> {
    let mut sources = String::from("#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n#pragma clang fp contract(off)\nusing namespace metal;\n\n");
    let mut launches = Vec::new();
    let mut storage = BTreeMap::new();
    collect_storage_and_source(&encoded, &mut storage, &mut sources);
    let mut aliases = Aliases::default();
    collect_aliases(&encoded, &mut aliases)?;
    let (
        buffers,
        buffer_ids,
        buffer_slots,
        scratch_bindings,
        scratch_slots,
        scalars,
        scalar_storage,
    ) = allocation_tables(&storage, &mut aliases)?;
    let execution = flatten_plan(
        encoded,
        &mut launches,
        &buffer_slots,
        &scratch_slots,
        &scalar_storage,
    )?;
    Ok(Emitted {
        source: sources,
        launches,
        execution,
        buffers,
        buffer_ids,
        scalars,
        scratch: scratch_bindings
            .iter()
            .map(|binding| binding.bytes)
            .collect(),
        scratch_bindings,
        status_slot: None,
        alias_pairs: vec![],
    })
}

fn collect_storage_and_source(
    plan: &EncodedPlan<MetalDialect, EncodedLaunch>,
    storage: &mut BTreeMap<ResolvedStorageId, ResolvedStorage<MetalDialect>>,
    source: &mut String,
) {
    for item in &plan.items {
        match item {
            EncodedScheduleItem::Phase(phase) => {
                for launch in &phase.launches {
                    storage.extend(launch.storage.clone());
                    source.push_str(&launch.source);
                }
            }
            EncodedScheduleItem::Subplan(subplan) => {
                collect_storage_and_source(&subplan.encoded, storage, source)
            }
        }
    }
}

#[derive(Default)]
struct Aliases(BTreeMap<ResolvedStorageId, ResolvedStorageId>);

impl Aliases {
    fn find(&mut self, value: ResolvedStorageId) -> ResolvedStorageId {
        let parent = self.0.get(&value).copied().unwrap_or(value);
        if parent == value {
            value
        } else {
            let root = self.find(parent);
            self.0.insert(value, root);
            root
        }
    }
    fn union(&mut self, a: ResolvedStorageId, b: ResolvedStorageId) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            self.0.insert(hi, lo);
        }
    }
}

fn collect_aliases(
    plan: &EncodedPlan<MetalDialect, EncodedLaunch>,
    aliases: &mut Aliases,
) -> Result<(), String> {
    for item in &plan.items {
        if let EncodedScheduleItem::Subplan(subplan) = item {
            for binding in subplan
                .resolved
                .inputs
                .iter()
                .chain(&subplan.resolved.results)
            {
                alias_transport(&binding.caller, &binding.callee, aliases)?;
            }
            collect_aliases(&subplan.encoded, aliases)?;
        }
    }
    Ok(())
}

fn alias_transport(
    a: &ResolvedValueTransport,
    b: &ResolvedValueTransport,
    aliases: &mut Aliases,
) -> Result<(), String> {
    match (a, b) {
        (ResolvedValueTransport::Void, ResolvedValueTransport::Void) => {}
        (ResolvedValueTransport::Storage(a), ResolvedValueTransport::Storage(b))
            if a.len() == b.len() =>
        {
            for (a, b) in a.iter().zip(b.iter()) {
                aliases.union(*a, *b);
            }
        }
        (ResolvedValueTransport::Tuple(a), ResolvedValueTransport::Tuple(b))
            if a.len() == b.len() =>
        {
            for (a, b) in a.iter().zip(b.iter()) {
                alias_transport(a, b, aliases)?;
            }
        }
        _ => return Err("call boundary transport shapes differ".into()),
    }
    Ok(())
}

type Slots = BTreeMap<ResolvedStorageId, usize>;

fn allocation_tables(
    storage: &BTreeMap<ResolvedStorageId, ResolvedStorage<MetalDialect>>,
    aliases: &mut Aliases,
) -> Result<
    (
        Vec<BufferSpec>,
        Vec<ResolvedStorageId>,
        Slots,
        Vec<BufferSpec>,
        Slots,
        Vec<ScalarParameter>,
        BTreeMap<ResolvedStorageId, usize>,
    ),
    String,
> {
    let mut roots = BTreeMap::new();
    for (id, value) in storage {
        roots
            .entry(aliases.find(*id))
            .or_insert_with(|| value.clone());
    }
    let mut scalar_roots = roots
        .iter()
        .filter(|(_, value)| {
            matches!(value.provenance.abi, Some(AbiRole::Parameter { .. }))
                && matches!(value.layout, ResolvedMetalLayout::Scalar { .. })
        })
        .collect::<Vec<_>>();
    scalar_roots.sort_by_key(|(_, value)| abi_key(value));
    let mut scalars = Vec::new();
    let mut scalar_field_indices = BTreeMap::new();
    for (root, value) in scalar_roots {
        let ResolvedMetalLayout::Scalar { parameters, .. } = &value.layout else {
            unreachable!()
        };
        scalar_field_indices.insert(*root, scalars.len());
        scalars.extend(parameters.iter().cloned());
    }
    let scalar_layout = seismic_lang::abi::ScalarLayout::natural(&scalars)?;
    let mut scalar_storage = scalar_field_indices
        .into_iter()
        .map(|(root, field)| (root, scalar_layout.fields[field].offset))
        .collect::<BTreeMap<_, _>>();
    let mut external = roots
        .iter()
        .filter(|(_, value)| {
            value.scope == StorageScope::External
                && !(matches!(value.provenance.abi, Some(AbiRole::Parameter { .. }))
                    && matches!(value.layout, ResolvedMetalLayout::Scalar { .. }))
        })
        .collect::<Vec<_>>();
    external.sort_by_key(|(_, value)| abi_key(value));
    let mut buffers = Vec::new();
    let mut buffer_ids = Vec::new();
    let mut buffer_slots = BTreeMap::new();
    for (slot, (root, value)) in external.into_iter().enumerate() {
        buffers.push(buffer_spec(value)?);
        buffer_ids.push(*root);
        buffer_slots.insert(*root, slot);
    }
    let mut scratch = Vec::new();
    let mut scratch_slots = BTreeMap::new();
    for (root, value) in roots
        .iter()
        .filter(|(_, value)| value.scope == StorageScope::Device)
    {
        let slot = scratch.len();
        scratch.push(BufferSpec {
            parameter: format!("internal_{}", root.0),
            plane: plane(value),
            role: BufferRole::Internal,
            bytes: value.bytes as usize,
            alignment: value.alignment as usize,
        });
        scratch_slots.insert(*root, slot);
    }
    let ids = storage.keys().copied().collect::<Vec<_>>();
    for id in ids {
        let root = aliases.find(id);
        if let Some(slot) = buffer_slots.get(&root).copied() {
            buffer_slots.insert(id, slot);
        }
        if let Some(slot) = scratch_slots.get(&root).copied() {
            scratch_slots.insert(id, slot);
        }
        if let Some(offset) = scalar_storage.get(&root).copied() {
            scalar_storage.insert(id, offset);
        }
    }
    Ok((
        buffers,
        buffer_ids,
        buffer_slots,
        scratch,
        scratch_slots,
        scalars,
        scalar_storage,
    ))
}

fn abi_key(storage: &ResolvedStorage<MetalDialect>) -> (u8, u32, Vec<u32>, String) {
    match &storage.provenance.abi {
        Some(AbiRole::Parameter {
            ordinal,
            path,
            representation_plane,
        }) => (
            0,
            *ordinal,
            path.clone(),
            representation_plane.clone().unwrap_or_default(),
        ),
        Some(AbiRole::Result {
            ordinal,
            path,
            representation_plane,
        }) => (
            1,
            *ordinal,
            path.clone(),
            representation_plane.clone().unwrap_or_default(),
        ),
        _ => (2, 0, vec![], String::new()),
    }
}

fn plane(storage: &ResolvedStorage<MetalDialect>) -> String {
    match &storage.layout {
        ResolvedMetalLayout::PackedPlane { plane, .. } => plane.clone(),
        _ => String::new(),
    }
}

fn buffer_spec(storage: &ResolvedStorage<MetalDialect>) -> Result<BufferSpec, String> {
    let (parameter, role) = match &storage.provenance.abi {
        Some(AbiRole::Parameter { ordinal, .. }) => {
            (format!("parameter_{ordinal}"), BufferRole::Parameter)
        }
        Some(AbiRole::Result { ordinal, path, .. }) => (
            format!("result_{ordinal}"),
            BufferRole::Result { path: path.clone() },
        ),
        _ => return Err("external Metal storage has no parameter/result ABI role".into()),
    };
    Ok(BufferSpec {
        parameter,
        plane: plane(storage),
        role,
        bytes: storage.bytes as usize,
        alignment: storage.alignment as usize,
    })
}

fn flatten_plan(
    plan: EncodedPlan<MetalDialect, EncodedLaunch>,
    launches: &mut Vec<Launch>,
    buffers: &Slots,
    scratch: &Slots,
    scalar_storage: &BTreeMap<ResolvedStorageId, usize>,
) -> Result<Vec<ExecutionItem>, String> {
    let mut execution = Vec::new();
    for item in plan.items {
        match item {
            EncodedScheduleItem::Phase(phase) => {
                let mut indices = Vec::new();
                for launch in phase.launches {
                    let index = launches.len();
                    indices.push(index);
                    let bindings = launch
                        .bindings
                        .into_iter()
                        .map(|binding| runtime_binding(binding, buffers, scratch, scalar_storage))
                        .collect::<Result<Vec<_>, _>>()?;
                    launches.push(Launch {
                        kernel: launch.kernel,
                        threadgroups: launch.threadgroups,
                        threads_per_threadgroup: launch.threads_per_threadgroup,
                        after_barrier: index != 0,
                        dispatch: None,
                        tiles: vec![],
                        declared_threadgroup_bytes: launch.declared_threadgroup_bytes,
                        bindings,
                    });
                }
                execution.push(ExecutionItem::Phase(indices));
            }
            EncodedScheduleItem::Subplan(subplan) => execution.push(ExecutionItem::Subplan(
                flatten_plan(*subplan.encoded, launches, buffers, scratch, scalar_storage)?,
            )),
        }
    }
    Ok(execution)
}

fn runtime_binding(
    binding: SymbolicBinding,
    buffers: &Slots,
    scratch: &Slots,
    scalar_storage: &BTreeMap<ResolvedStorageId, usize>,
) -> Result<Binding, String> {
    let one = |id| {
        if let Some(offset) = scalar_storage.get(&id).copied() {
            return Ok(Binding::Scalars(offset));
        }
        buffers
            .get(&id)
            .copied()
            .map(Binding::Buffer)
            .or_else(|| scratch.get(&id).copied().map(Binding::Scratch))
            .ok_or_else(|| format!("storage#{} has no runtime allocation", id.0))
    };
    match binding {
        SymbolicBinding::Storage(id) => one(id),
        SymbolicBinding::ArgumentTable(values) => Ok(Binding::ArgumentTable(
            values
                .into_iter()
                .map(|(member, id)| Ok((member, one(id)?)))
                .collect::<Result<_, String>>()?,
        )),
    }
}
