//! CUDA executable dialect and complete task-graph schedule construction.

use seismic_compiler::terminal::{
    self, ResolvedScalarInstruction, ScalarBindings, ScalarInstruction, ScalarStorageBinding,
    ScalarValueId, ValueBindingKey,
};
use seismic_lang::{
    logical::{
        AxisOrder, LocalView, LocalViewId, LocalViewTransform, LogicalAxisSource,
        LogicalDependencyKind, LogicalEndpoint, LogicalExpr, LogicalExprKind, LogicalProgram,
        LogicalTaskGraph, OperandId, StorageRef, Type, ValueRef,
    },
    sym::Sym,
    types::{DType, Elem},
};
use seismic_realization::executable::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CudaDialect;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaTemplateInstruction {
    pub scalar: ScalarInstruction,
    pub capability: Option<CudaCapability>,
    pub operand_values: BTreeMap<OperandId, ScalarValueId>,
    pub value_types: BTreeMap<ScalarValueId, Type>,
    pub axis_binders: Vec<Vec<ScalarValueId>>,
    pub result_values: BTreeMap<u32, ScalarValueId>,
    pub views: BTreeMap<LocalViewId, LocalView>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaResolvedInstruction {
    pub scalar: ResolvedScalarInstruction,
    pub operand_values: BTreeMap<OperandId, ScalarValueId>,
    pub value_types: BTreeMap<ScalarValueId, CudaValueType>,
    pub axis_binders: Vec<Vec<ScalarValueId>>,
    pub result_values: BTreeMap<u32, ScalarValueId>,
    pub views: BTreeMap<LocalViewId, CudaResolvedView>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaResolvedView {
    Identity,
    Reshape { shape: Vec<u64> },
    Transpose { permutation: Vec<u32> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaValueType {
    Scalar(DType),
    Index,
    Range,
    Tensor {
        dtype: DType,
        shape: Vec<u64>,
    },
    Packed {
        representation: String,
        shape: Vec<u64>,
        planes: Vec<(String, DType, u64)>,
    },
    Tuple(Vec<CudaValueType>),
    Capability,
    Void,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaTemplateLayout {
    pub dtype: Option<DType>,
    pub elements: Sym,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaResolvedLayout {
    pub dtype: Option<DType>,
    pub elements: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CudaCapability {
    Scalar,
    LaneIndex,
    ShuffleIndex,
    SubgroupReduction,
    Matrix,
}

impl ExecutableDialect for CudaDialect {
    type TemplateInstruction = CudaTemplateInstruction;
    type ResolvedInstruction = CudaResolvedInstruction;
    type TemplateLayout = CudaTemplateLayout;
    type ResolvedLayout = CudaResolvedLayout;
    type Capability = CudaCapability;

    fn consequences(
        value: &Self::TemplateInstruction,
    ) -> InstructionConsequences<Self::Capability> {
        let source = value.scalar.consequences();
        InstructionConsequences {
            accesses: source.accesses.clone(),
            capability: value.capability.clone(),
            numerical: source.numerical.clone(),
            resources: source.resources.clone(),
        }
    }

    fn resolve_instruction(
        value: &Self::TemplateInstruction,
        symbols: &BTreeMap<String, i64>,
        storage: &BTreeMap<StorageId, ResolvedStorageId>,
    ) -> Result<Self::ResolvedInstruction, String> {
        Ok(CudaResolvedInstruction {
            scalar: terminal::resolve_scalar_instruction(&value.scalar, symbols, storage)?,
            operand_values: value.operand_values.clone(),
            value_types: value
                .value_types
                .iter()
                .map(|(id, ty)| Ok((*id, resolve_type(ty, symbols)?)))
                .collect::<Result<_, String>>()?,
            axis_binders: value.axis_binders.clone(),
            result_values: value.result_values.clone(),
            views: value
                .views
                .iter()
                .map(|(id, view)| {
                    let resolved = match &view.transform {
                        LocalViewTransform::Identity => CudaResolvedView::Identity,
                        LocalViewTransform::Reshape { .. } => CudaResolvedView::Reshape {
                            shape: resolve_shape(&view.shape, symbols)?,
                        },
                        LocalViewTransform::Transpose { permutation } => {
                            CudaResolvedView::Transpose {
                                permutation: permutation.clone(),
                            }
                        }
                    };
                    Ok((*id, resolved))
                })
                .collect::<Result<_, String>>()?,
        })
    }

    fn resolve_layout(
        layout: &Self::TemplateLayout,
        symbols: &BTreeMap<String, i64>,
    ) -> Result<Self::ResolvedLayout, String> {
        Ok(CudaResolvedLayout {
            dtype: layout.dtype,
            elements: u64::try_from(
                layout
                    .elements
                    .eval(&|name| symbols.get(name).copied())
                    .ok_or_else(|| {
                        format!(
                            "CUDA layout has unresolved element count `{}`",
                            layout.elements
                        )
                    })?,
            )
            .map_err(|_| "CUDA layout element count is negative".to_string())?,
        })
    }
}

fn resolve_type(ty: &Type, symbols: &BTreeMap<String, i64>) -> Result<CudaValueType, String> {
    Ok(match ty {
        Type::Scalar(dtype) => CudaValueType::Scalar(*dtype),
        Type::Index { .. } => CudaValueType::Index,
        Type::Range { .. } => CudaValueType::Range,
        Type::Tensor(tensor) => {
            let shape = tensor
                .shape
                .iter()
                .map(|extent| {
                    u64::try_from(
                        extent
                            .eval(&|name| symbols.get(name).copied())
                            .ok_or_else(|| format!("unresolved CUDA extent `{extent}`"))?,
                    )
                    .map_err(|_| "negative CUDA extent".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            match &tensor.elem {
                Elem::Dtype(dtype) => CudaValueType::Tensor {
                    dtype: *dtype,
                    shape,
                },
                Elem::Repr(name) => {
                    let representation = seismic_lang::repr::lookup(name)
                        .ok_or_else(|| format!("unknown CUDA representation `{name}`"))?;
                    let elements = shape
                        .iter()
                        .try_fold(1_u64, |a, b| a.checked_mul(*b))
                        .ok_or("CUDA packed tensor extent overflow")?;
                    CudaValueType::Packed {
                        representation: name.clone(),
                        shape,
                        planes: representation
                            .planes()
                            .into_iter()
                            .map(|plane| {
                                Ok((
                                    plane.name.to_string(),
                                    plane.dtype(),
                                    plane
                                        .storage_elements(elements)
                                        .ok_or("CUDA packed plane extent overflow")?,
                                ))
                            })
                            .collect::<Result<_, String>>()?,
                    }
                }
                Elem::Param(name) => {
                    return Err(format!("unresolved CUDA element parameter `{name}`"))
                }
            }
        }
        Type::Tuple(fields) => CudaValueType::Tuple(
            fields
                .iter()
                .map(|field| resolve_type(field, symbols))
                .collect::<Result<_, _>>()?,
        ),
        Type::CapabilityValue { .. } => CudaValueType::Capability,
        Type::Void => CudaValueType::Void,
    })
}

fn resolve_shape(shape: &[Sym], symbols: &BTreeMap<String, i64>) -> Result<Vec<u64>, String> {
    shape
        .iter()
        .map(|extent| {
            u64::try_from(
                extent
                    .eval(&|name| symbols.get(name).copied())
                    .ok_or_else(|| format!("unresolved CUDA extent `{extent}`"))?,
            )
            .map_err(|_| "negative CUDA extent".to_string())
        })
        .collect()
}

pub fn capability_fingerprint(
    limits: &super::mapping::Limits,
    target: &crate::target::TargetProfile,
) -> String {
    format!(
        "seismic-cuda-executable-v2:{}:threads={}:grid={}:warp={}:scratch={}",
        target.fingerprint(),
        limits.max_threads_per_block,
        limits.max_grid_x,
        limits.warp_size,
        limits.max_scratch_bytes
    )
}

pub fn target_profile(
    limits: &super::mapping::Limits,
    target: &crate::target::TargetProfile,
) -> ExecutableTargetProfile<CudaCapability> {
    ExecutableTargetProfile {
        target: super::mapping::TARGET.into(),
        capability_fingerprint: capability_fingerprint(limits, target),
        toolchain_fingerprint: target.fingerprint().to_string(),
        capabilities: BTreeSet::from([
            CudaCapability::Scalar,
            CudaCapability::LaneIndex,
            CudaCapability::ShuffleIndex,
            CudaCapability::SubgroupReduction,
        ]),
        limits: ExecutableTargetLimits {
            max_allocation_bytes: limits.max_scratch_bytes,
            max_device_bytes: limits.max_scratch_bytes,
            max_workgroup_bytes: 0,
            max_private_bytes_per_participant: limits.max_scratch_bytes,
            max_bindings_per_launch: (crate::native::MAX_KERNEL_PARAMETER_BYTES / 8) as u64,
            max_registers_per_kernel: u64::MAX,
            max_workgroups: [u64::from(limits.max_grid_x), 65_535, 65_535],
            max_participants_per_workgroup: u64::from(limits.max_threads_per_block),
        },
    }
}

struct GraphStorage {
    transports: BTreeMap<OperandId, ValueTransportTemplate>,
    logical: BTreeMap<StorageRef, NonEmpty<(StorageId, Vec<String>)>>,
    next_storage: u32,
}

pub fn elaborate(
    logical: &LogicalProgram,
    limits: &super::mapping::Limits,
) -> Result<PlanFamily<CudaDialect>, String> {
    let mut family = PlanFamilyBuilder::from_logical(logical)?;
    for graph in &logical.task_graphs {
        family.add_alternative(schedule_graph(graph, limits)?)?;
    }
    family.finish()
}

fn schedule_graph(
    graph: &LogicalTaskGraph,
    limits: &super::mapping::Limits,
) -> Result<PlanAlternative<CudaDialect>, String> {
    let mut builder = ScheduleBuilder::new(graph)?;
    let mut storage = GraphStorage {
        transports: BTreeMap::new(),
        logical: BTreeMap::new(),
        next_storage: 0,
    };
    let input_transports = graph
        .inputs
        .iter()
        .enumerate()
        .map(|(port, input)| {
            allocate_input_type(
                &mut builder,
                &mut storage,
                port as u32,
                input.path.clone(),
                &input.ty,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for operand in &graph.operands {
        if let ValueRef::Input(candidate_port) = operand.value {
            let boundary_port = graph
                .input_remap
                .get(candidate_port as usize)
                .copied()
                .unwrap_or(candidate_port);
            storage
                .transports
                .insert(operand.id, input_transports[boundary_port as usize].clone());
            let boundary_storage = StorageRef::Input {
                port: boundary_port,
                path: vec![],
            };
            if let Some(ids) = storage.logical.get(&boundary_storage).cloned() {
                storage
                    .logical
                    .entry(StorageRef::Input {
                        port: candidate_port,
                        path: vec![],
                    })
                    .or_insert(ids);
            }
            continue;
        }
        let inferred_storage = operand.storage.clone().or_else(|| {
            matches!(operand.ty, Type::Tensor(_))
                .then(|| match operand.value {
                    ValueRef::Input(port) => Some(StorageRef::Input { port, path: vec![] }),
                    ValueRef::Result(port) => Some(StorageRef::Result { port, path: vec![] }),
                    ValueRef::Local(_) => None,
                })
                .flatten()
        });
        let transport = allocate_type(
            &mut builder,
            &mut storage,
            operand.id,
            inferred_storage.as_ref(),
            &operand.value,
            &operand.ty,
        )?;
        storage.transports.insert(operand.id, transport);
    }
    for port in 0..graph.inputs.len() as u32 {
        builder.bind_input(
            builder.input(port)?,
            input_transports[port as usize].clone(),
        )?;
    }

    let order = endpoint_order(graph)?;
    let mut previous_phase = None;
    let mut phase_id = 0u32;
    let mut launch_id = 0u32;
    let mut binding_id = 0u32;
    let mut group_id = 0u32;
    let mut instruction_count = 0i64;
    for endpoint in order {
        match endpoint {
            LogicalEndpoint::Task(task_id) => {
                let task = graph.task(task_id).ok_or("schedule names absent task")?;
                let scalar = scalar_program(graph, task, &storage)?;
                instruction_count += scalar.instructions().len() as i64;
                let metadata_values = task
                    .inputs
                    .iter()
                    .chain(task.outputs.iter())
                    .filter_map(|operand| {
                        scalar
                            .operand_value(*operand)
                            .map(|value| (*operand, value))
                    })
                    .collect::<BTreeMap<_, _>>();
                let metadata_types = scalar
                    .instructions()
                    .iter()
                    .filter_map(|instruction| instruction.result)
                    .filter_map(|value| scalar.value_type(value).cloned().map(|ty| (value, ty)))
                    .collect::<BTreeMap<_, _>>();
                let mut metadata_types = metadata_types;
                for operand in task.inputs.iter().chain(task.outputs.iter()) {
                    if let Some(value) = scalar.operand_value(*operand) {
                        metadata_types.insert(value, graph.operands[operand.0 as usize].ty.clone());
                    }
                }
                for (ordinal, ty) in graph.values.iter().enumerate() {
                    metadata_types.insert(
                        ScalarValueId(graph.operands.len() as u32 + ordinal as u32),
                        ty.clone(),
                    );
                }
                let axis_binders = task
                    .domain
                    .axes
                    .iter()
                    .map(|axis| {
                        axis.binders
                            .iter()
                            .map(|binder| ScalarValueId(graph.operands.len() as u32 + binder.0))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let result_values = task
                    .outputs
                    .iter()
                    .filter_map(|operand| match graph.operands[operand.0 as usize].value {
                        ValueRef::Result(port) => {
                            scalar.operand_value(*operand).map(|value| (port, value))
                        }
                        _ => None,
                    })
                    .collect::<BTreeMap<_, _>>();
                let instructions = nonempty(
                    scalar
                        .instructions()
                        .iter()
                        .cloned()
                        .map(|scalar| CudaTemplateInstruction {
                            capability: scalar_capability(&scalar),
                            scalar,
                            operand_values: metadata_values.clone(),
                            value_types: metadata_types.clone(),
                            axis_binders: axis_binders.clone(),
                            result_values: result_values.clone(),
                            views: graph
                                .views
                                .iter()
                                .enumerate()
                                .map(|(index, view)| (LocalViewId(index as u32), view.clone()))
                                .collect(),
                        })
                        .collect(),
                    "CUDA task has no scalar instructions",
                )?;
                let operands = task
                    .inputs
                    .iter()
                    .chain(task.outputs.iter())
                    .copied()
                    .collect::<BTreeSet<_>>();
                let bindings = operands
                    .iter()
                    .map(|operand| OperandTransportTemplate {
                        operand: *operand,
                        transport: storage.transports[operand].clone(),
                    })
                    .collect();
                let mapping = participant_map(task, limits)?;
                let step = builder.mapped_task_step(
                    builder.task(task.id)?,
                    mapping.clone(),
                    bindings,
                    instructions,
                )?;
                let mut native_bindings = Vec::new();
                let mut seen = BTreeSet::new();
                for operand in operands {
                    for storage_id in transport_ids(&storage.transports[&operand]) {
                        if seen.insert(storage_id) {
                            native_bindings.push(BindingGroupTemplate {
                                id: BindingGroupId(group_id),
                                kind: BindingGroupKind::Direct,
                                slot: group_id,
                                members: NonEmpty::new(BindingTemplate {
                                    id: BindingId(binding_id),
                                    storage: storage_id,
                                    access: AccessMode::ReadWrite,
                                    operand: Some(operand),
                                }),
                            });
                            binding_id += 1;
                            group_id += 1;
                        }
                    }
                }
                for access in &task.effects.accesses {
                    let ids = storage
                        .logical
                        .get(&access.storage)
                        .ok_or("task effect has no CUDA physical storage")?;
                    for (storage_id, _) in ids.iter() {
                        if seen.insert(*storage_id) {
                            native_bindings.push(BindingGroupTemplate {
                                id: BindingGroupId(group_id),
                                kind: BindingGroupKind::Direct,
                                slot: group_id,
                                members: NonEmpty::new(BindingTemplate {
                                    id: BindingId(binding_id),
                                    storage: *storage_id,
                                    access: AccessMode::ReadWrite,
                                    operand: None,
                                }),
                            });
                            binding_id += 1;
                            group_id += 1;
                        }
                    }
                }
                for access in scalar
                    .instructions()
                    .iter()
                    .flat_map(|instruction| instruction.consequences().accesses.clone())
                {
                    if seen.insert(access.storage) {
                        native_bindings.push(BindingGroupTemplate {
                            id: BindingGroupId(group_id),
                            kind: BindingGroupKind::Direct,
                            slot: group_id,
                            members: NonEmpty::new(BindingTemplate {
                                id: BindingId(binding_id),
                                storage: access.storage,
                                access: AccessMode::ReadWrite,
                                operand: None,
                            }),
                        });
                        binding_id += 1;
                        group_id += 1;
                    }
                }
                let launch = LaunchTemplate {
                    id: LaunchId(launch_id),
                    geometry: dispatch(&mapping, limits)?,
                    binding_groups: native_bindings,
                    kernel: KernelTemplate {
                        participant_storage: vec![],
                        workgroup_storage: vec![],
                        steps: NonEmpty::new(step),
                    },
                };
                builder.add_phase(PhaseTemplate {
                    id: PhaseId(phase_id),
                    predecessors: previous_phase.into_iter().collect(),
                    launches: NonEmpty::new(launch),
                })?;
                previous_phase = Some(PhaseId(phase_id));
                phase_id += 1;
                launch_id += 1;
            }
            LogicalEndpoint::Call(call_id) => {
                let call = graph.call(call_id).ok_or("schedule names absent call")?;
                let inputs = call
                    .inputs
                    .iter()
                    .map(|operand| OperandTransportTemplate {
                        operand: *operand,
                        transport: storage.transports[operand].clone(),
                    })
                    .collect();
                let mut results = call
                    .outputs
                    .iter()
                    .map(|operand| OperandTransportTemplate {
                        operand: *operand,
                        transport: storage.transports[operand].clone(),
                    })
                    .collect::<Vec<_>>();
                if !call.outputs.contains(&call.result) {
                    results.push(OperandTransportTemplate {
                        operand: call.result,
                        transport: storage.transports[&call.result].clone(),
                    });
                }
                builder.add_subplan(builder.call(call.id)?, inputs, results)?;
            }
            LogicalEndpoint::Input(_) | LogicalEndpoint::Output(_) => {}
        }
    }

    for dependency in &graph.dependencies {
        let obligation = builder.dependency(dependency.id)?;
        let transport = dependency_transport(dependency.kind.clone(), &storage)?;
        builder.place_launch_boundary(obligation, transport)?;
    }
    for port in 0..graph.results.len() as u32 {
        let operand = graph
            .operands
            .iter()
            .find(|operand| operand.value == ValueRef::Result(port))
            .ok_or_else(|| format!("result port#{port} has no operand"))?;
        builder.publish_output(
            builder.output(port)?,
            operand.id,
            storage.transports[&operand.id].clone(),
        )?;
    }
    builder.finish(Sym::constant(instruction_count + i64::from(launch_id)))
}

fn allocate_input_type(
    builder: &mut ScheduleBuilder<CudaDialect>,
    state: &mut GraphStorage,
    port: u32,
    path: Vec<u32>,
    ty: &Type,
) -> Result<ValueTransportTemplate, String> {
    if matches!(ty, Type::Void) {
        return Ok(ValueTransportTemplate::Void);
    }
    if let Type::Tuple(fields) = ty {
        return Ok(ValueTransportTemplate::Tuple(nonempty(
            fields
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let mut field_path = path.clone();
                    field_path.push(index as u32);
                    allocate_input_type(builder, state, port, field_path, field).map(Box::new)
                })
                .collect::<Result<_, _>>()?,
            "empty input tuple transport",
        )?));
    }
    let logical = StorageRef::Input { port, path: vec![] };
    let mut planes = Vec::new();
    if let Type::Tensor(tensor) = ty {
        if let Elem::Repr(name) = &tensor.elem {
            let representation = seismic_lang::repr::lookup(name)
                .ok_or_else(|| format!("unknown CUDA representation `{name}`"))?;
            let elements = tensor.shape.iter().fold(Sym::constant(1), |a, b| a.mul(b));
            for plane in representation.planes() {
                planes.push((
                    vec![plane.name.to_string()],
                    Some(plane.dtype()),
                    plane.extent(&elements),
                    u64::from(plane.dtype().bytes()),
                ));
            }
        }
    }
    if planes.is_empty() {
        let (dtype, elements, _, alignment) = layout(ty)?;
        planes.push((vec![], dtype, elements, alignment));
    }
    let mut ids = Vec::new();
    for (plane, dtype, elements, alignment) in planes {
        let id = StorageId(state.next_storage);
        state.next_storage += 1;
        let bytes = elements.scale(i64::try_from(alignment).map_err(|_| "alignment overflow")?);
        builder.add_input_storage(
            port,
            StorageTemplate {
                id,
                scope: StorageScope::External,
                replication: Replication::Once,
                bytes: bytes.clone(),
                alignment,
                layout: CudaTemplateLayout { dtype, elements },
                provenance: PhysicalStorageProvenance {
                    logical_storage: Some(logical.clone()),
                    operand: None,
                    view: None,
                    subrange: PhysicalSubrangeTemplate {
                        byte_offset: Sym::constant(0),
                        bytes: bytes.clone(),
                    },
                    address: PhysicalAddressTemplate::DenseAffine {
                        byte_offset: Sym::constant(0),
                        byte_strides: vec![Sym::constant(alignment as i64)],
                    },
                    abi: Some(AbiRole::Parameter {
                        ordinal: port,
                        path: path.clone(),
                        representation_plane: plane.first().cloned(),
                    }),
                },
            },
        )?;
        match state.logical.get_mut(&logical) {
            Some(values) => values.push((id, plane.clone())),
            None => {
                state
                    .logical
                    .insert(logical.clone(), NonEmpty::new((id, plane)));
            }
        }
        ids.push(id);
    }
    Ok(ValueTransportTemplate::Storage(nonempty(
        ids,
        "input has no storage planes",
    )?))
}

fn scalar_program(
    graph: &LogicalTaskGraph,
    task: &seismic_lang::logical::LogicalTask,
    storage: &GraphStorage,
) -> Result<terminal::ScalarProgram, String> {
    let mut bindings = ScalarBindings::default();
    for operand in task.inputs.iter().chain(task.outputs.iter()) {
        let value = ScalarValueId(operand.0);
        bindings.operands.insert(*operand, value);
        bindings.values.insert(
            ValueBindingKey::from(&graph.operands[operand.0 as usize].value),
            value,
        );
    }
    for (candidate, common) in graph.input_remap.iter().copied().enumerate() {
        if let Some(value) = bindings
            .values
            .get(&ValueBindingKey::Input(common))
            .copied()
        {
            bindings
                .values
                .insert(ValueBindingKey::Input(candidate as u32), value);
        }
    }
    for (ordinal, _) in graph.values.iter().enumerate() {
        bindings.values.insert(
            ValueBindingKey::Local(seismic_lang::logical::LocalValueId(ordinal as u32)),
            ScalarValueId(graph.operands.len() as u32 + ordinal as u32),
        );
    }
    for (logical, ids) in &storage.logical {
        bindings.storage.insert(
            logical.clone(),
            ids.iter()
                .map(|(id, plane)| ScalarStorageBinding {
                    storage: *id,
                    plane: plane.clone(),
                })
                .collect(),
        );
    }
    for (ordinal, view) in graph.views.iter().enumerate() {
        if let Some(ids) = storage.logical.get(&view.storage) {
            bindings.views.insert(
                LocalViewId(ordinal as u32),
                ids.iter()
                    .map(|(id, plane)| ScalarStorageBinding {
                        storage: *id,
                        plane: plane.clone(),
                    })
                    .collect(),
            );
        }
    }
    terminal::lower_task(graph, task, &bindings).map_err(|error| {
        format!(
            "CUDA scalar lowering task#{} with inputs {:?}: {error}",
            task.id.0, task.inputs
        )
    })
}

fn scalar_capability(value: &ScalarInstruction) -> Option<CudaCapability> {
    use seismic_compiler::terminal::ScalarInstructionKind;
    use seismic_lang::intrinsics::Operation;
    let ScalarInstructionKind::Intrinsic { operation, .. } = value.kind else {
        return None;
    };
    Some(match operation {
        Operation::LaneIndex => CudaCapability::LaneIndex,
        Operation::ShuffleIndex => CudaCapability::ShuffleIndex,
        Operation::SimdSum | Operation::SimdMax | Operation::SimdMin => {
            CudaCapability::SubgroupReduction
        }
        Operation::Matrix
        | Operation::MatrixLoad
        | Operation::MatrixLoadTranspose
        | Operation::MatrixStore
        | Operation::MatrixMultiplyAccumulate
        | Operation::MatrixMatmul
        | Operation::MatrixMatmulAdd => CudaCapability::Matrix,
    })
}

fn allocate_type(
    builder: &mut ScheduleBuilder<CudaDialect>,
    state: &mut GraphStorage,
    operand: OperandId,
    logical_storage: Option<&StorageRef>,
    value: &ValueRef,
    ty: &Type,
) -> Result<ValueTransportTemplate, String> {
    if matches!(ty, Type::Void) {
        return Ok(ValueTransportTemplate::Void);
    }
    if let Type::Tuple(fields) = ty {
        let values = fields
            .iter()
            .map(|field| {
                allocate_type(builder, state, operand, logical_storage, value, field).map(Box::new)
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(ValueTransportTemplate::Tuple(nonempty(
            values,
            "empty tuple transport",
        )?));
    }
    if let Type::Tensor(tensor) = ty {
        if let Elem::Repr(name) = &tensor.elem {
            let representation = seismic_lang::repr::lookup(name)
                .ok_or_else(|| format!("unknown CUDA representation `{name}`"))?;
            let logical_elements = tensor.shape.iter().fold(Sym::constant(1), |a, b| a.mul(b));
            let mut ids = Vec::new();
            for plane in representation.planes() {
                let id = StorageId(state.next_storage);
                state.next_storage += 1;
                let elements = plane.extent(&logical_elements);
                let dtype = plane.dtype();
                let bytes = elements.scale(i64::from(dtype.bytes()));
                let plane_path = vec![plane.name.to_string()];
                let abi = match value {
                    ValueRef::Input(port) => Some(AbiRole::Parameter {
                        ordinal: *port,
                        path: vec![],
                        representation_plane: Some(plane.name.to_string()),
                    }),
                    ValueRef::Result(port) => Some(AbiRole::Result {
                        ordinal: *port,
                        path: vec![],
                        representation_plane: Some(plane.name.to_string()),
                    }),
                    ValueRef::Local(_) => None,
                };
                builder.add_operand_storage(
                    operand,
                    StorageTemplate {
                        id,
                        scope: if abi.is_some() {
                            StorageScope::External
                        } else {
                            StorageScope::Device
                        },
                        replication: Replication::Once,
                        bytes: bytes.clone(),
                        alignment: u64::from(dtype.bytes()),
                        layout: CudaTemplateLayout {
                            dtype: Some(dtype),
                            elements,
                        },
                        provenance: PhysicalStorageProvenance {
                            logical_storage: logical_storage.cloned(),
                            operand: Some(operand),
                            view: None,
                            subrange: PhysicalSubrangeTemplate {
                                byte_offset: Sym::constant(0),
                                bytes: bytes.clone(),
                            },
                            address: PhysicalAddressTemplate::DenseAffine {
                                byte_offset: Sym::constant(0),
                                byte_strides: vec![Sym::constant(i64::from(dtype.bytes()))],
                            },
                            abi,
                        },
                    },
                )?;
                if let Some(logical) = logical_storage {
                    match state.logical.get_mut(logical) {
                        Some(values) => values.push((id, plane_path.clone())),
                        None => {
                            state
                                .logical
                                .insert(logical.clone(), NonEmpty::new((id, plane_path)));
                        }
                    }
                }
                ids.push(id);
            }
            return Ok(ValueTransportTemplate::Storage(nonempty(
                ids,
                "packed representation has no planes",
            )?));
        }
    }
    let (dtype, elements, bytes, alignment) = layout(ty)?;
    let id = StorageId(state.next_storage);
    state.next_storage += 1;
    let abi = match value {
        ValueRef::Input(port) => Some(AbiRole::Parameter {
            ordinal: *port,
            path: vec![],
            representation_plane: None,
        }),
        ValueRef::Result(port) => Some(AbiRole::Result {
            ordinal: *port,
            path: vec![],
            representation_plane: None,
        }),
        ValueRef::Local(_) => None,
    };
    let allocation = StorageTemplate {
        id,
        scope: if abi.is_some() {
            StorageScope::External
        } else {
            StorageScope::Device
        },
        replication: Replication::Once,
        bytes: bytes.clone(),
        alignment,
        layout: CudaTemplateLayout { dtype, elements },
        provenance: PhysicalStorageProvenance {
            logical_storage: logical_storage.cloned(),
            operand: Some(operand),
            view: None,
            subrange: PhysicalSubrangeTemplate {
                byte_offset: Sym::constant(0),
                bytes: bytes.clone(),
            },
            address: PhysicalAddressTemplate::DenseAffine {
                byte_offset: Sym::constant(0),
                byte_strides: vec![Sym::constant(alignment as i64)],
            },
            abi,
        },
    };
    builder.add_operand_storage(operand, allocation)?;
    if let Some(logical) = logical_storage {
        match state.logical.get_mut(logical) {
            Some(ids) => ids.push((id, vec![])),
            None => {
                state
                    .logical
                    .insert(logical.clone(), NonEmpty::new((id, vec![])));
            }
        }
    }
    Ok(ValueTransportTemplate::Storage(NonEmpty::new(id)))
}

fn layout(ty: &Type) -> Result<(Option<DType>, Sym, Sym, u64), String> {
    match ty {
        Type::Scalar(dtype) => Ok((
            Some(*dtype),
            Sym::constant(1),
            Sym::constant(i64::from(dtype.bytes())),
            u64::from(dtype.bytes()),
        )),
        Type::Index { .. } | Type::Range { .. } => {
            Ok((None, Sym::constant(1), Sym::constant(8), 8))
        }
        Type::Tensor(tensor) => {
            let Elem::Dtype(dtype) = tensor.elem else {
                return Err("CUDA packed tensors require explicit plane scheduling".into());
            };
            let elements = tensor.shape.iter().fold(Sym::constant(1), |a, b| a.mul(b));
            Ok((
                Some(dtype),
                elements.clone(),
                elements.scale(i64::from(dtype.bytes())),
                u64::from(dtype.bytes()),
            ))
        }
        Type::CapabilityValue { .. } => Ok((None, Sym::constant(1), Sym::constant(8), 8)),
        Type::Void | Type::Tuple(_) => {
            Err("CUDA cannot allocate void/aggregate as one storage leaf".into())
        }
    }
}

fn participant_map(
    task: &seismic_lang::logical::LogicalTask,
    _limits: &super::mapping::Limits,
) -> Result<ParticipantMap, String> {
    let mut axes = Vec::new();
    let mut extents = Vec::new();
    let mut grid_axis = 0u8;
    for (logical_axis, axis) in task.domain.axes.iter().enumerate() {
        let extent = axis_extent(axis)?;
        extents.push(extent);
        if axis.order == AxisOrder::Ordered {
            axes.push(AxisMap::Serial {
                logical_axis: logical_axis as u32,
            });
        } else {
            if grid_axis >= 3 {
                return Err("CUDA task has more than three independent axes".into());
            }
            axes.push(AxisMap::Grid {
                logical_axis: logical_axis as u32,
                workgroup_axis: grid_axis,
                participant_axis: grid_axis,
                mode: GridMapping::OnePass,
            });
            grid_axis += 1;
        }
    }
    Ok(ParticipantMap {
        task: task.id,
        axes,
        logical_extents: extents,
        masks_inactive_participants: true,
    })
}

fn axis_extent(axis: &seismic_lang::logical::LogicalAxis) -> Result<Sym, String> {
    match &axis.source {
        LogicalAxisSource::Range { lo, hi, .. } => Ok(expr_sym(hi)?.sub(&expr_sym(lo)?)),
        LogicalAxisSource::Members { slice } => Ok(Sym::param(&format!("@site{slice}"))),
        LogicalAxisSource::Coordinates { value, axes } => {
            let Type::Tensor(tensor) = &value.ty else {
                return Err("coordinate axis source is not tensor-shaped".into());
            };
            axes.first()
                .and_then(|axis| tensor.shape.get(*axis))
                .cloned()
                .ok_or_else(|| "coordinate axis is absent".into())
        }
    }
}

fn expr_sym(expr: &LogicalExpr) -> Result<Sym, String> {
    match &expr.kind {
        LogicalExprKind::Int(v) => Ok(Sym::constant(*v)),
        LogicalExprKind::Shape(v) => Ok(v.clone()),
        _ => Err("CUDA task axis has a runtime value without a resolved symbolic bound".into()),
    }
}

fn dispatch(
    mapping: &ParticipantMap,
    limits: &super::mapping::Limits,
) -> Result<DispatchTemplate, String> {
    let mut groups = [Sym::constant(1), Sym::constant(1), Sym::constant(1)];
    let mut participants = [Sym::constant(1), Sym::constant(1), Sym::constant(1)];
    for axis in &mapping.axes {
        if let AxisMap::Grid {
            logical_axis,
            workgroup_axis,
            participant_axis,
            ..
        } = axis
        {
            let extent = mapping.logical_extents[*logical_axis as usize].clone();
            if *participant_axis == 0 {
                let width = Sym::constant(i64::from(limits.max_threads_per_block));
                participants[0] = width.clone();
                groups[*workgroup_axis as usize] =
                    extent.add(&width.sub(&Sym::constant(1))).quot(&width);
            } else {
                groups[*workgroup_axis as usize] = extent;
            }
        }
    }
    Ok(DispatchTemplate {
        workgroups: groups,
        participants_per_workgroup: participants,
    })
}

fn endpoint_order(graph: &LogicalTaskGraph) -> Result<Vec<LogicalEndpoint>, String> {
    let mut nodes = graph
        .tasks
        .iter()
        .map(|t| LogicalEndpoint::Task(t.id))
        .chain(graph.calls.iter().map(|c| LogicalEndpoint::Call(c.id)))
        .collect::<BTreeSet<_>>();
    let mut result = Vec::new();
    while !nodes.is_empty() {
        let ready = nodes
            .iter()
            .copied()
            .find(|node| {
                !graph
                    .dependencies
                    .iter()
                    .any(|d| d.to == *node && nodes.contains(&d.from))
            })
            .ok_or("logical task graph contains a scheduling cycle")?;
        nodes.remove(&ready);
        result.push(ready);
    }
    Ok(result)
}

fn dependency_transport(
    kind: LogicalDependencyKind,
    storage: &GraphStorage,
) -> Result<DependencyTransport, String> {
    Ok(match kind {
        LogicalDependencyKind::Control => DependencyTransport::Control,
        LogicalDependencyKind::Value(operand) => DependencyTransport::Value {
            operand,
            transport: storage.transports[&operand].clone(),
        },
        LogicalDependencyKind::Effect(logical_storage) => DependencyTransport::Effect {
            logical_storage: logical_storage.clone(),
            storage: nonempty(
                storage
                    .logical
                    .get(&logical_storage)
                    .ok_or("effect dependency has no physical storage")?
                    .iter()
                    .map(|(id, _)| *id)
                    .collect(),
                "effect dependency has no physical storage",
            )?,
        },
        LogicalDependencyKind::Ownership(logical_storage) => DependencyTransport::Ownership {
            logical_storage: logical_storage.clone(),
            storage: nonempty(
                storage
                    .logical
                    .get(&logical_storage)
                    .ok_or("ownership dependency has no physical storage")?
                    .iter()
                    .map(|(id, _)| *id)
                    .collect(),
                "ownership dependency has no physical storage",
            )?,
        },
    })
}

fn transport_ids(value: &ValueTransportTemplate) -> Vec<StorageId> {
    match value {
        ValueTransportTemplate::Void => vec![],
        ValueTransportTemplate::Kernel(_) => vec![],
        ValueTransportTemplate::Storage(ids) => ids.iter().copied().collect(),
        ValueTransportTemplate::Tuple(fields) => fields
            .iter()
            .flat_map(|field| transport_ids(field))
            .collect(),
    }
}

fn nonempty<T>(mut values: Vec<T>, message: &str) -> Result<NonEmpty<T>, String> {
    if values.is_empty() {
        return Err(message.into());
    }
    let mut result = NonEmpty::new(values.remove(0));
    for value in values {
        result.push(value);
    }
    Ok(result)
}
