//! CPU construction of the executable physical contract.
//!
//! Every logical task becomes an explicit native launch. Values that cross a
//! launch or call boundary use retained storage transports; no native stage is
//! allowed to invent placement, scheduling, or call bindings.

use seismic_compiler::{
    pipeline::{Backend, EncodedPlan},
    terminal::{
        self, ResolvedScalarInstruction, ScalarBindings, ScalarCapabilitySet, ScalarInstruction,
        ScalarStorageBinding, ScalarValueId, ValueBindingKey,
    },
};
use seismic_lang::{
    logical::{
        LogicalDependencyKind, LogicalEndpoint, LogicalProgram, LogicalTaskGraph, OperandId,
        StorageRef, TensorType, Type, ValueRef,
    },
    repr,
    sym::Sym,
    types::{DType, Elem},
};
use seismic_realization::executable::{
    self as exec, AbiRole, AccessMode, AxisMap, BindingGroupKind, BindingGroupTemplate, BindingId,
    BindingTemplate, DependencyTransport, DispatchTemplate, ExecutableDialect,
    ExecutableTargetLimits, ExecutableTargetProfile, KernelStep, KernelTemplate, KernelValueId,
    LaunchId, LaunchTemplate, NonEmpty, OperandTransportTemplate, ParticipantMap, PhaseId,
    PhaseTemplate, PhysicalAddressTemplate, PhysicalStorageProvenance, PhysicalSubrangeTemplate,
    PlanFamily, PlanFamilyBuilder, Replication, ResolvedStorageId, ScheduleBuilder, StorageId,
    StorageScope, StorageTemplate, ValueTransportTemplate,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuDialect;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpuTemplateLayout {
    TensorPlane {
        dtype: DType,
        shape: Vec<Sym>,
        representation: Option<String>,
        plane: Vec<String>,
    },
    ScalarSlot {
        dtype: DType,
        words: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpuResolvedLayout {
    TensorPlane {
        dtype: DType,
        shape: Vec<u64>,
        representation: Option<String>,
        plane: Vec<String>,
    },
    ScalarSlot {
        dtype: DType,
        words: u32,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum CpuTemplateInstruction {
    Operand {
        value: ScalarValueId,
        operand: OperandId,
        ty: Type,
        transport: ValueTransportTemplate,
    },
    Coordinate {
        value: ScalarValueId,
        axis: u32,
        component: u32,
        components: Vec<Sym>,
    },
    Scalar {
        scalar: ScalarInstruction,
        value_types: BTreeMap<ScalarValueId, Type>,
        views: BTreeMap<seismic_lang::logical::LocalViewId, seismic_lang::logical::LocalView>,
    },
    Output {
        value: ScalarValueId,
        operand: OperandId,
        ty: Type,
        transport: ValueTransportTemplate,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum CpuResolvedInstruction {
    Operand {
        value: ScalarValueId,
        operand: OperandId,
        ty: Type,
        transport: exec::ResolvedValueTransport,
    },
    Coordinate {
        value: ScalarValueId,
        axis: u32,
        component: u32,
        components: Vec<u64>,
    },
    Scalar {
        scalar: ResolvedScalarInstruction,
        value_types: BTreeMap<ScalarValueId, Type>,
        views: BTreeMap<seismic_lang::logical::LocalViewId, seismic_lang::logical::LocalView>,
    },
    Output {
        value: ScalarValueId,
        operand: OperandId,
        ty: Type,
        transport: exec::ResolvedValueTransport,
    },
}

impl ExecutableDialect for CpuDialect {
    type TemplateInstruction = CpuTemplateInstruction;
    type ResolvedInstruction = CpuResolvedInstruction;
    type TemplateLayout = CpuTemplateLayout;
    type ResolvedLayout = CpuResolvedLayout;
    type Capability = ScalarCapabilitySet;

    fn consequences(
        instruction: &Self::TemplateInstruction,
    ) -> exec::InstructionConsequences<Self::Capability> {
        match instruction {
            CpuTemplateInstruction::Scalar { scalar, .. } => scalar.consequences().clone(),
            CpuTemplateInstruction::Operand { transport, .. } => {
                transport_consequences(transport, AccessMode::Read)
            }
            CpuTemplateInstruction::Output { transport, .. } => {
                transport_consequences(transport, AccessMode::Write)
            }
            CpuTemplateInstruction::Coordinate { .. } => exec::InstructionConsequences {
                accesses: vec![],
                capability: None,
                numerical: vec![],
                resources: exec::InstructionResources::default(),
            },
        }
    }

    fn resolve_instruction(
        instruction: &Self::TemplateInstruction,
        symbols: &BTreeMap<String, i64>,
        storage: &BTreeMap<StorageId, ResolvedStorageId>,
    ) -> Result<Self::ResolvedInstruction, String> {
        Ok(match instruction {
            CpuTemplateInstruction::Operand {
                value,
                operand,
                ty,
                transport,
            } => CpuResolvedInstruction::Operand {
                value: *value,
                operand: *operand,
                ty: ty.clone(),
                transport: resolve_transport(transport, storage)?,
            },
            CpuTemplateInstruction::Coordinate {
                value,
                axis,
                component,
                components,
            } => CpuResolvedInstruction::Coordinate {
                value: *value,
                axis: *axis,
                component: *component,
                components: components
                    .iter()
                    .map(|extent| {
                        u64::try_from(extent.eval(&|name| symbols.get(name).copied()).ok_or_else(
                            || format!("unresolved CPU coordinate extent `{extent}`"),
                        )?)
                        .map_err(|_| "CPU coordinate extent is negative".to_string())
                    })
                    .collect::<Result<_, _>>()?,
            },
            CpuTemplateInstruction::Scalar {
                scalar,
                value_types,
                views,
            } => CpuResolvedInstruction::Scalar {
                scalar: terminal::resolve_scalar_instruction(scalar, symbols, storage)?,
                value_types: value_types.clone(),
                views: views.clone(),
            },
            CpuTemplateInstruction::Output {
                value,
                operand,
                ty,
                transport,
            } => CpuResolvedInstruction::Output {
                value: *value,
                operand: *operand,
                ty: ty.clone(),
                transport: resolve_transport(transport, storage)?,
            },
        })
    }

    fn resolve_layout(
        layout: &Self::TemplateLayout,
        symbols: &BTreeMap<String, i64>,
    ) -> Result<Self::ResolvedLayout, String> {
        Ok(match layout {
            CpuTemplateLayout::TensorPlane {
                dtype,
                shape,
                representation,
                plane,
            } => {
                CpuResolvedLayout::TensorPlane {
                    dtype: *dtype,
                    shape: shape
                        .iter()
                        .map(|extent| {
                            u64::try_from(
                                extent.eval(&|name| symbols.get(name).copied()).ok_or_else(
                                    || format!("unresolved CPU layout extent `{extent}`"),
                                )?,
                            )
                            .map_err(|_| "CPU layout extent is negative".to_string())
                        })
                        .collect::<Result<_, _>>()?,
                    representation: representation.clone(),
                    plane: plane.clone(),
                }
            }
            CpuTemplateLayout::ScalarSlot { dtype, words } => CpuResolvedLayout::ScalarSlot {
                dtype: *dtype,
                words: *words,
            },
        })
    }
}

fn transport_consequences(
    transport: &ValueTransportTemplate,
    mode: AccessMode,
) -> exec::InstructionConsequences<ScalarCapabilitySet> {
    let mut accesses = Vec::new();
    fn collect(
        value: &ValueTransportTemplate,
        mode: AccessMode,
        out: &mut Vec<exec::PhysicalAccess>,
    ) {
        match value {
            ValueTransportTemplate::Void => {}
            ValueTransportTemplate::Kernel(_) => {}
            ValueTransportTemplate::Storage(values) => {
                out.extend(values.iter().map(|storage| exec::PhysicalAccess {
                    storage: *storage,
                    mode,
                }))
            }
            ValueTransportTemplate::Tuple(values) => {
                for value in values.iter() {
                    collect(value, mode, out);
                }
            }
        }
    }
    collect(transport, mode, &mut accesses);
    exec::InstructionConsequences {
        accesses,
        capability: None,
        numerical: vec![],
        resources: exec::InstructionResources::default(),
    }
}

fn resolve_transport(
    transport: &ValueTransportTemplate,
    storage: &BTreeMap<StorageId, ResolvedStorageId>,
) -> Result<exec::ResolvedValueTransport, String> {
    Ok(match transport {
        ValueTransportTemplate::Void => exec::ResolvedValueTransport::Void,
        ValueTransportTemplate::Kernel(value) => {
            return Err(format!(
                "CPU instruction directly references unresolved kernel value#{}",
                value.0
            ));
        }
        ValueTransportTemplate::Storage(values) => {
            exec::ResolvedValueTransport::Storage(map_nonempty(values, |id| {
                storage
                    .get(id)
                    .copied()
                    .ok_or_else(|| format!("CPU transport names absent storage#{}", id.0))
            })?)
        }
        ValueTransportTemplate::Tuple(values) => {
            exec::ResolvedValueTransport::Tuple(map_nonempty(values, |value| {
                resolve_transport(value, storage).map(Box::new)
            })?)
        }
    })
}

fn map_nonempty<T, U>(
    values: &NonEmpty<T>,
    mut map: impl FnMut(&T) -> Result<U, String>,
) -> Result<NonEmpty<U>, String> {
    let mut values = values.iter();
    let mut result = NonEmpty::new(map(values.next().expect("NonEmpty invariant"))?);
    for value in values {
        result.push(map(value)?);
    }
    Ok(result)
}

pub fn capability_fingerprint(limits: &super::mapping::Limits) -> String {
    format!(
        "seismic-cpu-executable-v2:{}:workers={}:scratch={}",
        std::env::consts::ARCH,
        limits.workers,
        limits.max_scratch_bytes
    )
}

pub fn target_profile(
    limits: &super::mapping::Limits,
) -> ExecutableTargetProfile<ScalarCapabilitySet> {
    ExecutableTargetProfile {
        target: super::mapping::TARGET.into(),
        capability_fingerprint: capability_fingerprint(limits),
        toolchain_fingerprint: format!(
            "cranelift-{}-{}",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::ARCH
        ),
        capabilities: BTreeSet::from([ScalarCapabilitySet::default()]),
        limits: ExecutableTargetLimits {
            max_allocation_bytes: u64::MAX,
            max_device_bytes: u64::MAX,
            max_workgroup_bytes: 0,
            max_private_bytes_per_participant: limits.max_scratch_bytes,
            max_bindings_per_launch: u64::MAX,
            max_registers_per_kernel: u64::MAX,
            max_workgroups: [u64::MAX, 1, 1],
            max_participants_per_workgroup: 1,
        },
    }
}

struct GraphStorage<'a> {
    program: &'a LogicalProgram,
    graph: &'a LogicalTaskGraph,
    next_storage: u32,
    next_kernel_value: u32,
    templates: Vec<(Option<OperandId>, StorageTemplate<CpuDialect>)>,
    storage: BTreeMap<StorageRef, ValueTransportTemplate>,
    operands: BTreeMap<OperandId, ValueTransportTemplate>,
}

impl<'a> GraphStorage<'a> {
    fn new(program: &'a LogicalProgram, graph: &'a LogicalTaskGraph) -> Self {
        Self {
            program,
            graph,
            next_storage: 0,
            next_kernel_value: 0,
            templates: vec![],
            storage: BTreeMap::new(),
            operands: BTreeMap::new(),
        }
    }

    fn build(mut self) -> Result<Self, String> {
        for (port, input) in self.graph.inputs.iter().enumerate() {
            if let Type::Tensor(tensor) = &input.ty {
                let storage = StorageRef::Input {
                    port: port as u32,
                    path: Vec::new(),
                };
                let transport = self.allocate_tensor(tensor, Some(storage.clone()), None)?;
                self.storage.insert(storage, transport);
            }
        }
        for (port, result) in self.graph.results.iter().enumerate() {
            if let Type::Tensor(tensor) = &result.ty {
                let storage = StorageRef::Result {
                    port: port as u32,
                    path: Vec::new(),
                };
                let transport = self.allocate_tensor(tensor, Some(storage.clone()), None)?;
                self.storage.insert(storage, transport);
            }
        }
        let aggregate_results = self
            .graph
            .calls
            .iter()
            .filter(|call| !call.outputs.contains(&call.result))
            .map(|call| call.result)
            .collect::<BTreeSet<_>>();
        for operand in &self.graph.operands {
            if aggregate_results.contains(&operand.id) {
                continue;
            }
            let inferred_storage =
                operand
                    .storage
                    .clone()
                    .or_else(|| match (&operand.value, &operand.ty) {
                        (ValueRef::Input(port), Type::Tensor(_)) => self
                            .graph
                            .inputs
                            .get(*port as usize)
                            .map(|_| StorageRef::Input {
                                port: *port,
                                path: Vec::new(),
                            }),
                        (ValueRef::Result(port), Type::Tensor(_)) => self
                            .graph
                            .results
                            .get(*port as usize)
                            .map(|_| StorageRef::Result {
                                port: *port,
                                path: Vec::new(),
                            }),
                        _ => None,
                    });
            let transport = if let Some(storage) = &inferred_storage {
                if let Some(existing) = self.storage.get(storage) {
                    let existing = existing.clone();
                    existing
                } else {
                    let transport = self.allocate_type(
                        &operand.ty,
                        Some(storage.clone()),
                        Some(operand.id),
                        &[],
                    )?;
                    self.storage.insert(storage.clone(), transport.clone());
                    transport
                }
            } else {
                self.allocate_type(&operand.ty, None, Some(operand.id), &[])?
            };
            self.operands.insert(operand.id, transport);
        }
        for call in &self.graph.calls {
            if !aggregate_results.contains(&call.result) {
                continue;
            }
            let ty = &self
                .graph
                .operand(call.result)
                .ok_or("call aggregate operand is absent")?
                .ty;
            let mut leaves = call.outputs.iter();
            let transport = aggregate_transport(ty, &mut leaves, &self.operands)?;
            if leaves.next().is_some() {
                return Err("call result has more physical leaves than its aggregate type".into());
            }
            self.operands.insert(call.result, transport);
        }
        for (ordinal, local) in self.graph.storage.iter().enumerate() {
            let storage = StorageRef::Local(seismic_lang::logical::LocalStorageId(ordinal as u32));
            if !self.storage.contains_key(&storage) {
                let transport = self.allocate_tensor(&local.ty, Some(storage.clone()), None)?;
                self.storage.insert(storage, transport);
            }
        }
        Ok(self)
    }

    fn allocate_type(
        &mut self,
        ty: &Type,
        logical: Option<StorageRef>,
        operand: Option<OperandId>,
        path: &[u32],
    ) -> Result<ValueTransportTemplate, String> {
        match ty {
            Type::Tuple(fields) => {
                let mut values = fields
                    .iter()
                    .enumerate()
                    .map(|(index, field)| {
                        let mut child = path.to_vec();
                        child.push(index as u32);
                        self.allocate_type(field, None, operand, &child)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(ValueTransportTemplate::Tuple(nonempty(
                    values.drain(..).map(Box::new),
                )?))
            }
            Type::Tensor(tensor) => self.allocate_tensor(tensor, logical, operand),
            Type::Scalar(dtype) => self.allocate_scalar(*dtype, 1, operand, path),
            Type::Index { .. } => self.allocate_scalar(DType::I32, 1, operand, path),
            Type::Range { .. } => self.allocate_scalar(DType::I32, 2, operand, path),
            Type::CapabilityValue { .. } => {
                let value = KernelValueId(self.next_kernel_value);
                self.next_kernel_value += 1;
                Ok(ValueTransportTemplate::Kernel(value))
            }
            Type::Void => Ok(ValueTransportTemplate::Void),
        }
    }

    fn allocate_scalar(
        &mut self,
        dtype: DType,
        words: u32,
        operand: Option<OperandId>,
        path: &[u32],
    ) -> Result<ValueTransportTemplate, String> {
        let id = self.fresh_storage();
        let bytes = u64::from(words) * 8;
        let abi = operand.and_then(|operand| {
            self.graph
                .operand(operand)
                .and_then(|operand| match operand.value {
                    ValueRef::Input(ordinal) => Some(AbiRole::Parameter {
                        ordinal,
                        path: path.to_vec(),
                        representation_plane: None,
                    }),
                    ValueRef::Result(ordinal) => Some(AbiRole::Result {
                        ordinal,
                        path: path.to_vec(),
                        representation_plane: None,
                    }),
                    ValueRef::Local(_) => None,
                })
        });
        let scope = if abi.is_some() {
            StorageScope::External
        } else {
            StorageScope::Device
        };
        self.templates.push((
            operand,
            StorageTemplate {
                id,
                scope,
                replication: Replication::Once,
                bytes: Sym::constant(i64::try_from(bytes).unwrap()),
                alignment: 8,
                layout: CpuTemplateLayout::ScalarSlot { dtype, words },
                provenance: PhysicalStorageProvenance {
                    logical_storage: None,
                    operand,
                    view: None,
                    subrange: PhysicalSubrangeTemplate {
                        byte_offset: Sym::constant(0),
                        bytes: Sym::constant(i64::try_from(bytes).unwrap()),
                    },
                    address: PhysicalAddressTemplate::DenseAffine {
                        byte_offset: Sym::constant(0),
                        byte_strides: vec![],
                    },
                    abi,
                },
            },
        ));
        Ok(ValueTransportTemplate::Storage(NonEmpty::new(id)))
    }

    fn allocate_tensor(
        &mut self,
        tensor: &TensorType,
        logical: Option<StorageRef>,
        operand: Option<OperandId>,
    ) -> Result<ValueTransportTemplate, String> {
        let shape = tensor
            .shape
            .iter()
            .map(|extent| capacity(self.program, extent))
            .collect::<Vec<_>>();
        let elements = shape
            .iter()
            .fold(Sym::constant(1), |total, extent| total.mul(extent));
        let mut planes = Vec::new();
        match &tensor.elem {
            Elem::Dtype(dtype) => planes.push((Vec::new(), *dtype, elements.clone(), None)),
            Elem::Repr(name) => {
                let representation =
                    repr::lookup(name).ok_or_else(|| format!("unknown representation `{name}`"))?;
                for plane in representation.planes() {
                    planes.push((
                        vec![plane.name.to_string()],
                        plane.dtype(),
                        plane.extent(&elements),
                        Some(name.clone()),
                    ));
                }
            }
            Elem::Param(name) => return Err(format!("unresolved CPU element parameter `{name}`")),
        }
        let mut ids = Vec::new();
        for (plane, dtype, plane_elements, representation) in planes {
            let id = self.fresh_storage();
            let bytes = plane_elements.scale(i64::from(dtype.bytes()));
            let (scope, abi) = match &logical {
                Some(StorageRef::Input { port, .. }) => (
                    StorageScope::External,
                    Some(AbiRole::Parameter {
                        ordinal: *port,
                        path: self
                            .graph
                            .inputs
                            .get(*port as usize)
                            .map(|input| input.path.clone())
                            .ok_or("CPU input ABI port is absent")?,
                        representation_plane: plane.first().cloned(),
                    }),
                ),
                Some(StorageRef::Result { port, .. }) => (
                    StorageScope::External,
                    Some(AbiRole::Result {
                        ordinal: *port,
                        path: self
                            .graph
                            .results
                            .get(*port as usize)
                            .map(|result| result.path.clone())
                            .ok_or("CPU result ABI port is absent")?,
                        representation_plane: plane.first().cloned(),
                    }),
                ),
                Some(StorageRef::Local(_)) | None => (StorageScope::Device, None),
            };
            let strides = dense_byte_strides(&shape, dtype)?;
            self.templates.push((
                operand,
                StorageTemplate {
                    id,
                    scope,
                    replication: Replication::Once,
                    bytes: bytes.clone(),
                    alignment: u64::from(dtype.bytes()),
                    layout: CpuTemplateLayout::TensorPlane {
                        dtype,
                        shape: shape.clone(),
                        representation: representation.clone(),
                        plane: plane.clone(),
                    },
                    provenance: PhysicalStorageProvenance {
                        logical_storage: logical.clone(),
                        operand,
                        view: None,
                        subrange: PhysicalSubrangeTemplate {
                            byte_offset: Sym::constant(0),
                            bytes,
                        },
                        address: if let Some(representation) = representation {
                            PhysicalAddressTemplate::Representation {
                                representation,
                                plane: plane.join("."),
                                logical_strides: dense_logical_strides(&shape),
                            }
                        } else {
                            PhysicalAddressTemplate::DenseAffine {
                                byte_offset: Sym::constant(0),
                                byte_strides: strides,
                            }
                        },
                        abi,
                    },
                },
            ));
            ids.push(id);
        }
        Ok(ValueTransportTemplate::Storage(nonempty(ids.into_iter())?))
    }

    fn fresh_storage(&mut self) -> StorageId {
        let id = StorageId(self.next_storage);
        self.next_storage += 1;
        id
    }
}

fn aggregate_transport<'a>(
    ty: &Type,
    leaves: &mut impl Iterator<Item = &'a OperandId>,
    operands: &BTreeMap<OperandId, ValueTransportTemplate>,
) -> Result<ValueTransportTemplate, String> {
    match ty {
        Type::Void => Ok(ValueTransportTemplate::Void),
        Type::Tuple(fields) => Ok(ValueTransportTemplate::Tuple(nonempty(
            fields
                .iter()
                .map(|field| aggregate_transport(field, leaves, operands).map(Box::new))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter(),
        )?)),
        _ => {
            let operand = leaves.next().ok_or_else(|| {
                format!("call result has fewer physical leaves than aggregate field `{ty:?}`")
            })?;
            operands
                .get(operand)
                .cloned()
                .ok_or_else(|| format!("call result leaf operand#{} is absent", operand.0))
        }
    }
}

fn nonempty<T>(mut values: impl Iterator<Item = T>) -> Result<NonEmpty<T>, String> {
    let head = values.next().ok_or("physical bundle is empty")?;
    let mut result = NonEmpty::new(head);
    for value in values {
        result.push(value);
    }
    Ok(result)
}

fn capacity(program: &LogicalProgram, value: &Sym) -> Sym {
    value
        .as_constant()
        .or_else(|| program.extent_capacity(value))
        .map(Sym::constant)
        .unwrap_or_else(|| value.clone())
}

fn dense_logical_strides(shape: &[Sym]) -> Vec<Sym> {
    let mut stride = Sym::constant(1);
    let mut result = vec![Sym::constant(1); shape.len()];
    for (axis, extent) in shape.iter().enumerate().rev() {
        result[axis] = stride.clone();
        stride = stride.mul(extent);
    }
    result
}

fn dense_byte_strides(shape: &[Sym], dtype: DType) -> Result<Vec<Sym>, String> {
    Ok(dense_logical_strides(shape)
        .into_iter()
        .map(|stride| stride.scale(i64::from(dtype.bytes())))
        .collect())
}

fn task_order(graph: &LogicalTaskGraph) -> Result<Vec<LogicalEndpoint>, String> {
    let mut nodes = graph
        .tasks
        .iter()
        .map(|task| LogicalEndpoint::Task(task.id))
        .chain(
            graph
                .calls
                .iter()
                .map(|call| LogicalEndpoint::Call(call.id)),
        )
        .collect::<BTreeSet<_>>();
    let mut order = Vec::new();
    while !nodes.is_empty() {
        let next = nodes
            .iter()
            .find(|candidate| {
                graph.dependencies.iter().all(|dependency| {
                    dependency.to != **candidate || !nodes.contains(&dependency.from)
                })
            })
            .copied()
            .ok_or("logical task graph contains a cycle")?;
        nodes.remove(&next);
        order.push(next);
    }
    Ok(order)
}

fn elaborate_graph(
    program: &LogicalProgram,
    graph: &LogicalTaskGraph,
    limits: &super::mapping::Limits,
) -> Result<exec::PlanAlternative<CpuDialect>, String> {
    let storage = GraphStorage::new(program, graph).build()?;
    let mut builder = ScheduleBuilder::new(graph)?;
    for (operand, template) in &storage.templates {
        if let Some(operand) = operand {
            builder.add_operand_storage(*operand, template.clone())?;
        } else if let Some(StorageRef::Input { port, .. }) = &template.provenance.logical_storage {
            builder.add_input_storage(*port, template.clone())?;
        } else if let Some(StorageRef::Result { port, .. }) = &template.provenance.logical_storage {
            builder.add_result_storage(*port, template.clone())?;
        } else if let Some(logical) = &template.provenance.logical_storage {
            builder.add_effect_storage(logical.clone(), template.clone())?;
        } else {
            builder.add_invocation_storage(template.clone())?;
        }
    }
    for (port, input) in graph.inputs.iter().enumerate() {
        let port = port as u32;
        let transport = if let Some(operand) = graph
            .operands
            .iter()
            .find(|operand| matches!(operand.value, ValueRef::Input(value) if value == port))
        {
            storage.operands[&operand.id].clone()
        } else if matches!(input.ty, Type::Void) {
            ValueTransportTemplate::Void
        } else {
            storage
                .storage
                .get(&StorageRef::Input {
                    port,
                    path: Vec::new(),
                })
                .cloned()
                .ok_or_else(|| format!("CPU input port#{port} has no physical transport"))?
        };
        let obligation = builder.input(port)?;
        builder.bind_input(obligation, transport)?;
    }

    let mut scalar_values = BTreeMap::<ValueBindingKey, ScalarValueId>::new();
    let mut operand_values = BTreeMap::<OperandId, ScalarValueId>::new();
    let mut next_value = 0u32;
    for port in 0..graph.inputs.len() as u32 {
        scalar_values.insert(ValueBindingKey::Input(port), ScalarValueId(next_value));
        next_value += 1;
    }
    for port in 0..graph.results.len() as u32 {
        scalar_values.insert(ValueBindingKey::Result(port), ScalarValueId(next_value));
        next_value += 1;
    }
    for operand in &graph.operands {
        let value = scalar_values
            .entry(ValueBindingKey::from(&operand.value))
            .or_insert_with(|| {
                let value = ScalarValueId(next_value);
                next_value += 1;
                value
            });
        operand_values.insert(operand.id, *value);
    }
    for (ordinal, _) in graph.values.iter().enumerate() {
        scalar_values
            .entry(ValueBindingKey::Local(seismic_lang::logical::LocalValueId(
                ordinal as u32,
            )))
            .or_insert_with(|| {
                let value = ScalarValueId(next_value);
                next_value += 1;
                value
            });
    }
    let scalar_storage = storage
        .storage
        .iter()
        .map(|(logical, transport)| {
            Ok((
                logical.clone(),
                scalar_storage_bindings(transport, &storage.templates)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let views: BTreeMap<_, _> = graph
        .views
        .iter()
        .enumerate()
        .map(|(ordinal, view)| {
            Ok((
                seismic_lang::logical::LocalViewId(ordinal as u32),
                scalar_storage
                    .get(&view.storage)
                    .cloned()
                    .ok_or("CPU view base has no physical storage")?,
            ))
        })
        .collect::<Result<_, String>>()?;
    let order = task_order(graph)?;
    let mut phase = 0u32;
    let mut launch = 0u32;
    let mut previous_phase = None;
    let mut task_launch = BTreeMap::new();
    for endpoint in &order {
        match endpoint {
            LogicalEndpoint::Task(task_id) => {
                let task = graph.task(*task_id).ok_or("scheduled CPU task is absent")?;
                let bindings = ScalarBindings {
                    values: scalar_values.clone(),
                    operands: operand_values.clone(),
                    storage: scalar_storage.clone(),
                    views: views.clone(),
                };
                let lowered = terminal::lower_task(graph, task, &bindings).map_err(|error| {
                    format!(
                        "{error}; CPU physical storage mappings are {:?}; input remap {:?}; inputs {:?}",
                        scalar_storage.keys().collect::<Vec<_>>(),
                        graph.input_remap,
                        graph.inputs,
                    )
                })?;
                for operand in &task.outputs {
                    if let Some(value) = lowered.operand_value(*operand) {
                        operand_values.insert(*operand, value);
                    }
                }
                let mut instructions = Vec::new();
                for (axis, logical_axis) in task.domain.axes.iter().enumerate() {
                    let components = axis_components(logical_axis)?;
                    for (component, binder) in logical_axis.binders.iter().enumerate() {
                        let value = scalar_values[&ValueBindingKey::Local(*binder)];
                        instructions.push(CpuTemplateInstruction::Coordinate {
                            value,
                            axis: axis as u32,
                            component: component as u32,
                            components: components.clone(),
                        });
                    }
                }
                for operand in &task.inputs {
                    instructions.push(CpuTemplateInstruction::Operand {
                        value: operand_values[operand],
                        operand: *operand,
                        ty: graph.operand(*operand).unwrap().ty.clone(),
                        transport: storage.operands[operand].clone(),
                    });
                }
                let logical_views = graph
                    .views
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, view)| (seismic_lang::logical::LocalViewId(index as u32), view))
                    .collect::<BTreeMap<_, _>>();
                instructions.extend(lowered.instructions().iter().cloned().map(|scalar| {
                    let value_types = scalar_values
                        .values()
                        .copied()
                        .chain(operand_values.values().copied())
                        .chain(scalar.result)
                        .filter_map(|value| {
                            lowered.value_type(value).cloned().map(|ty| (value, ty))
                        })
                        .collect();
                    CpuTemplateInstruction::Scalar {
                        scalar,
                        value_types,
                        views: logical_views.clone(),
                    }
                }));
                for operand in &task.outputs {
                    instructions.push(CpuTemplateInstruction::Output {
                        value: operand_values[operand],
                        operand: *operand,
                        ty: graph.operand(*operand).unwrap().ty.clone(),
                        transport: storage.operands[operand].clone(),
                    });
                }
                let task_bindings = task
                    .inputs
                    .iter()
                    .chain(&task.outputs)
                    .map(|operand| OperandTransportTemplate {
                        operand: *operand,
                        transport: storage.operands[operand].clone(),
                    })
                    .collect::<Vec<_>>();
                let mapping = serial_participant_map(task, program)?;
                let step = builder.mapped_task_step(
                    builder.task(*task_id)?,
                    mapping,
                    task_bindings,
                    nonempty(instructions.into_iter()).map_err(|_| {
                        format!("CPU task#{} has no physical instruction", task_id.0)
                    })?,
                )?;
                let used = instruction_storage(&step);
                let groups = binding_groups(&used);
                let launch_id = LaunchId(launch);
                let phase_id = PhaseId(phase);
                builder.add_phase(PhaseTemplate {
                    id: phase_id,
                    predecessors: previous_phase.into_iter().collect(),
                    launches: NonEmpty::new(LaunchTemplate {
                        id: launch_id,
                        geometry: DispatchTemplate {
                            workgroups: [Sym::constant(1), Sym::constant(1), Sym::constant(1)],
                            participants_per_workgroup: [
                                Sym::constant(1),
                                Sym::constant(1),
                                Sym::constant(1),
                            ],
                        },
                        binding_groups: groups,
                        kernel: KernelTemplate {
                            participant_storage: vec![],
                            workgroup_storage: vec![],
                            steps: NonEmpty::new(step),
                        },
                    }),
                })?;
                task_launch.insert(*task_id, launch_id);
                previous_phase = Some(phase_id);
                phase += 1;
                launch += 1;
            }
            LogicalEndpoint::Call(call_id) => {
                let call = graph.call(*call_id).ok_or("scheduled CPU call is absent")?;
                let inputs = call
                    .inputs
                    .iter()
                    .map(|operand| OperandTransportTemplate {
                        operand: *operand,
                        transport: storage.operands[operand].clone(),
                    })
                    .collect();
                let mut result_operands = call.outputs.clone();
                if !result_operands.contains(&call.result) {
                    result_operands.push(call.result);
                }
                let results = result_operands
                    .iter()
                    .map(|operand| OperandTransportTemplate {
                        operand: *operand,
                        transport: storage.operands[operand].clone(),
                    })
                    .collect();
                builder.add_subplan(builder.call(*call_id)?, inputs, results)?;
            }
            LogicalEndpoint::Input(_) | LogicalEndpoint::Output(_) => unreachable!(),
        }
    }
    for dependency in &graph.dependencies {
        let transport = match &dependency.kind {
            LogicalDependencyKind::Control => DependencyTransport::Control,
            LogicalDependencyKind::Value(operand) => DependencyTransport::Value {
                operand: *operand,
                transport: storage.operands[operand].clone(),
            },
            LogicalDependencyKind::Effect(logical) => DependencyTransport::Effect {
                logical_storage: logical.clone(),
                storage: storage_ids(&storage.storage[logical])?,
            },
            LogicalDependencyKind::Ownership(logical) => DependencyTransport::Ownership {
                logical_storage: logical.clone(),
                storage: storage_ids(&storage.storage[logical])?,
            },
        };
        builder.place_launch_boundary(builder.dependency(dependency.id)?, transport)?;
    }
    for output in 0..graph.results.len() as u32 {
        let operand = graph
            .operands
            .iter()
            .find(|operand| matches!(operand.value, ValueRef::Result(port) if port == output))
            .ok_or_else(|| format!("CPU result port#{output} has no operand"))?;
        builder.publish_output(
            builder.output(output)?,
            operand.id,
            storage.operands[&operand.id].clone(),
        )?;
    }
    let _ = limits;
    builder.finish(Sym::constant(i64::from(phase.max(1))))
}

fn storage_ids(transport: &ValueTransportTemplate) -> Result<NonEmpty<StorageId>, String> {
    let mut ids = Vec::new();
    fn collect(value: &ValueTransportTemplate, ids: &mut Vec<StorageId>) -> Result<(), String> {
        match value {
            ValueTransportTemplate::Void => {}
            ValueTransportTemplate::Kernel(_) => {
                return Err("retained CPU value has a kernel-only transport".into());
            }
            ValueTransportTemplate::Storage(values) => ids.extend(values.iter().copied()),
            ValueTransportTemplate::Tuple(values) => {
                for value in values.iter() {
                    collect(value, ids)?;
                }
            }
        }
        Ok(())
    }
    collect(transport, &mut ids)?;
    nonempty(ids.into_iter())
}

fn scalar_storage_bindings(
    transport: &ValueTransportTemplate,
    templates: &[(Option<OperandId>, StorageTemplate<CpuDialect>)],
) -> Result<Vec<ScalarStorageBinding>, String> {
    let ids = storage_ids(transport)?;
    ids.iter()
        .map(|id| {
            let template = templates
                .iter()
                .find(|(_, template)| template.id == *id)
                .ok_or("CPU scalar storage template is absent")?;
            let plane = match &template.1.layout {
                CpuTemplateLayout::TensorPlane { plane, .. } => plane.clone(),
                CpuTemplateLayout::ScalarSlot { .. } => vec![],
            };
            Ok(ScalarStorageBinding {
                storage: *id,
                plane,
            })
        })
        .collect()
}

fn serial_participant_map(
    task: &seismic_lang::logical::LogicalTask,
    program: &LogicalProgram,
) -> Result<ParticipantMap, String> {
    let mut extents = Vec::new();
    let mut axes = Vec::new();
    for (axis, logical) in task.domain.axes.iter().enumerate() {
        let extent = match &logical.source {
            seismic_lang::logical::LogicalAxisSource::Range { lo, hi, .. } => {
                logical_extent(program, lo, hi)?
            }
            seismic_lang::logical::LogicalAxisSource::Coordinates { value, axes } => {
                let Type::Tensor(tensor) = &value.ty else {
                    return Err("coordinate domain source is not a tensor".into());
                };
                let source_axis = *axes.first().ok_or("coordinate domain has no axis")?;
                capacity(program, &tensor.shape[source_axis])
            }
            seismic_lang::logical::LogicalAxisSource::Members { .. } => Sym::constant(1),
        };
        extents.push(extent);
        axes.push(AxisMap::Serial {
            logical_axis: axis as u32,
        });
    }
    Ok(ParticipantMap {
        task: task.id,
        axes,
        logical_extents: extents,
        masks_inactive_participants: false,
    })
}

fn axis_components(axis: &seismic_lang::logical::LogicalAxis) -> Result<Vec<Sym>, String> {
    match &axis.source {
        seismic_lang::logical::LogicalAxisSource::Coordinates { value, axes } => {
            let Type::Tensor(tensor) = &value.ty else {
                return Err("coordinate domain source is not a tensor".into());
            };
            axes.iter()
                .map(|axis| {
                    tensor
                        .shape
                        .get(*axis)
                        .cloned()
                        .ok_or_else(|| "coordinate component axis is out of bounds".into())
                })
                .collect()
        }
        seismic_lang::logical::LogicalAxisSource::Range { lo, hi, .. } => {
            Ok(vec![logical_extent_raw(lo, hi)?])
        }
        seismic_lang::logical::LogicalAxisSource::Members { .. } => Ok(vec![Sym::constant(1)]),
    }
}

fn logical_extent_raw(
    lo: &seismic_lang::logical::LogicalExpr,
    hi: &seismic_lang::logical::LogicalExpr,
) -> Result<Sym, String> {
    fn value(expr: &seismic_lang::logical::LogicalExpr) -> Option<Sym> {
        match &expr.kind {
            seismic_lang::logical::LogicalExprKind::Int(value) => Some(Sym::constant(*value)),
            seismic_lang::logical::LogicalExprKind::Shape(value) => Some(value.clone()),
            _ => None,
        }
    }
    let lo = value(lo).ok_or("CPU range lower bound is not structural")?;
    let hi = value(hi).ok_or("CPU range upper bound is not structural")?;
    Ok(hi.add(&lo.scale(-1)))
}

fn logical_extent(
    program: &LogicalProgram,
    lo: &seismic_lang::logical::LogicalExpr,
    hi: &seismic_lang::logical::LogicalExpr,
) -> Result<Sym, String> {
    Ok(capacity(program, &logical_extent_raw(lo, hi)?))
}

fn instruction_storage(step: &KernelStep<CpuDialect>) -> BTreeMap<StorageId, AccessMode> {
    let mut used = BTreeMap::new();
    if let exec::KernelStepView::MappedTask { instructions, .. } = step.view() {
        for instruction in instructions.iter() {
            for access in CpuDialect::consequences(instruction).accesses {
                used.entry(access.storage)
                    .and_modify(|mode| *mode = merge_access(*mode, access.mode))
                    .or_insert(access.mode);
            }
        }
    }
    used
}

fn merge_access(left: AccessMode, right: AccessMode) -> AccessMode {
    if left == right {
        left
    } else if left == AccessMode::Atomic || right == AccessMode::Atomic {
        AccessMode::Atomic
    } else {
        AccessMode::ReadWrite
    }
}

fn binding_groups(used: &BTreeMap<StorageId, AccessMode>) -> Vec<BindingGroupTemplate> {
    used.iter()
        .enumerate()
        .map(|(slot, (storage, access))| BindingGroupTemplate {
            id: exec::BindingGroupId(slot as u32),
            kind: BindingGroupKind::Direct,
            slot: slot as u32,
            members: NonEmpty::new(BindingTemplate {
                id: BindingId(slot as u32),
                storage: *storage,
                access: *access,
                operand: None,
            }),
        })
        .collect()
}

pub fn elaborate(
    program: &LogicalProgram,
    limits: &super::mapping::Limits,
) -> Result<PlanFamily<CpuDialect>, String> {
    let mut family = PlanFamilyBuilder::from_logical(program)?;
    for graph in &program.task_graphs {
        family.add_alternative(elaborate_graph(program, graph, limits)?)?;
    }
    family.finish()
}

pub struct EncodedLaunch {
    pub launch: exec::ResolvedLaunch<CpuDialect>,
    pub native: crate::NativePhase,
}

pub struct NativeArtifact {
    pub kernel: crate::Kernel,
}

impl Backend for super::mapping::Cpu {
    type Dialect = CpuDialect;
    type EncodedLaunch = EncodedLaunch;
    type NativeArtifact = NativeArtifact;

    fn target(&self) -> &'static str {
        super::mapping::TARGET
    }

    fn capability_fingerprint(&self) -> String {
        self.executable_target.capability_fingerprint.clone()
    }

    fn supports_intrinsic(
        &self,
        intrinsic: &seismic_lang::sir::IntrinsicUse,
    ) -> Result<(), String> {
        Err(format!(
            "the CPU scalar backend does not implement backend intrinsic `{}`",
            intrinsic.id.path()
        ))
    }

    fn target_profile(&self) -> &ExecutableTargetProfile<ScalarCapabilitySet> {
        &self.executable_target
    }

    fn elaborate(&self, logical: &LogicalProgram) -> Result<PlanFamily<CpuDialect>, String> {
        elaborate(logical, &self.limits)
    }

    fn encode_launch(
        &self,
        launch: &exec::ResolvedLaunch<CpuDialect>,
    ) -> Result<Self::EncodedLaunch, String> {
        Ok(EncodedLaunch {
            launch: launch.clone(),
            native: crate::native::encode_launch(launch)?,
        })
    }

    fn assemble(
        &self,
        encoded: EncodedPlan<CpuDialect, EncodedLaunch>,
    ) -> Result<Self::NativeArtifact, String> {
        let execution = crate::native::assemble(encoded)?;
        Ok(NativeArtifact {
            kernel: crate::compile_native(execution)?,
        })
    }
}
