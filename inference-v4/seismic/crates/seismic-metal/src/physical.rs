//! Metal executable planning.
//!
//! Logical task graphs are consumed here exactly once. Native encoding sees
//! only resolved launches; it never reconstructs scheduling, storage, ABI, or
//! participation decisions from a whole logical fragment.

use seismic_compiler::terminal::{
    self, ResolvedScalarInstruction, ScalarBindings, ScalarInstruction, ScalarStorageBinding,
    ScalarValueId, ValueBindingKey,
};
use seismic_lang::{
    abi::{RangeEndpoint, RangeScalar, ScalarParameter},
    logical::{
        AxisOrder, LocalViewId, LogicalAxisSource, LogicalEndpoint, LogicalExpr, LogicalExprKind,
        LogicalTask, LogicalTaskGraph, OperandId, StorageRef, TensorType, Type, ValueRef,
    },
    repr,
    sym::Sym,
    types::{DType, Elem},
};
use seismic_realization::executable::{
    AbiRole, AccessMode, AxisMap, BindingGroupId, BindingGroupKind, BindingGroupTemplate,
    BindingId, BindingTemplate, DependencyTransport, DispatchTemplate, ExecutableDialect,
    GridMapping, InstructionConsequences, InstructionResources, KernelTemplate, LaunchId,
    LaunchTemplate, NonEmpty, OperandTransportTemplate, ParticipantMap, PhaseId, PhaseTemplate,
    PhysicalAddressTemplate, PhysicalStorageProvenance, PhysicalSubrangeTemplate, PlanFamily,
    PlanFamilyBuilder, Replication, ResolvedStorageId, ScheduleBuilder, StorageId, StorageScope,
    StorageTemplate, ValueTransportTemplate,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MetalDialect;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalLayout {
    Dense {
        dtype: DType,
        elements: Sym,
    },
    Scalar {
        parameters: Vec<ScalarParameter>,
    },
    PackedPlane {
        representation: String,
        plane: String,
        dtype: DType,
        elements: Sym,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedMetalLayout {
    Dense {
        dtype: DType,
        elements: u64,
    },
    Scalar {
        parameters: Vec<ScalarParameter>,
        bytes: u64,
    },
    PackedPlane {
        representation: String,
        plane: String,
        dtype: DType,
        elements: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetalCapability {
    Scalar,
    Simdgroup,
    SimdgroupMatrix,
    BFloat,
    Atomic32,
    PackedVector(u8),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetalInstruction {
    pub task: seismic_lang::logical::TaskId,
    pub scalar: ScalarInstruction,
    pub value_types: BTreeMap<ScalarValueId, Type>,
    pub axis_values: Vec<Vec<ScalarValueId>>,
    pub axis_component_extents: Vec<Vec<Sym>>,
    pub operand_values: BTreeMap<OperandId, ScalarValueId>,
    pub input_operands: BTreeSet<OperandId>,
    pub output_operands: BTreeSet<OperandId>,
    pub views: BTreeMap<LocalViewId, seismic_lang::logical::LocalView>,
    pub view_bindings: BTreeMap<LocalViewId, Vec<ScalarStorageBinding>>,
    pub value_bindings: BTreeMap<ScalarValueId, Vec<ScalarStorageBinding>>,
}

// Logical floating literals compare by bit representation.
impl Eq for MetalInstruction {}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedMetalInstruction {
    pub task: seismic_lang::logical::TaskId,
    pub scalar: ResolvedScalarInstruction,
    pub value_types: BTreeMap<ScalarValueId, Type>,
    pub axis_values: Vec<Vec<ScalarValueId>>,
    pub axis_component_extents: Vec<Vec<u64>>,
    pub operand_values: BTreeMap<OperandId, ScalarValueId>,
    pub input_operands: BTreeSet<OperandId>,
    pub output_operands: BTreeSet<OperandId>,
    pub views: BTreeMap<LocalViewId, seismic_lang::logical::LocalView>,
    pub view_bindings:
        BTreeMap<LocalViewId, Vec<seismic_compiler::terminal::ResolvedScalarStorageBinding>>,
    pub value_bindings:
        BTreeMap<ScalarValueId, Vec<seismic_compiler::terminal::ResolvedScalarStorageBinding>>,
    /// Exact physical storages named by this scalar instruction's consequences.
    pub access_bindings: Vec<ResolvedStorageId>,
}

impl ExecutableDialect for MetalDialect {
    type TemplateInstruction = MetalInstruction;
    type ResolvedInstruction = ResolvedMetalInstruction;
    type TemplateLayout = MetalLayout;
    type ResolvedLayout = ResolvedMetalLayout;
    type Capability = MetalCapability;

    fn consequences(
        instruction: &Self::TemplateInstruction,
    ) -> InstructionConsequences<Self::Capability> {
        let scalar = instruction.scalar.consequences();
        InstructionConsequences {
            accesses: scalar.accesses.clone(),
            capability: scalar
                .capability
                .as_ref()
                .and_then(|set| set.0.iter().next())
                .map(|name| metal_capability(name)),
            numerical: scalar.numerical.clone(),
            resources: InstructionResources {
                registers: scalar.resources.registers,
                private_bytes: scalar.resources.private_bytes,
                workgroup_bytes: scalar.resources.workgroup_bytes,
            },
        }
    }

    fn resolve_instruction(
        instruction: &Self::TemplateInstruction,
        symbols: &BTreeMap<String, i64>,
        storage: &BTreeMap<StorageId, ResolvedStorageId>,
    ) -> Result<Self::ResolvedInstruction, String> {
        Ok(ResolvedMetalInstruction {
            task: instruction.task,
            scalar: terminal::resolve_scalar_instruction(&instruction.scalar, symbols, storage)?,
            value_types: instruction.value_types.clone(),
            axis_values: instruction.axis_values.clone(),
            axis_component_extents: instruction
                .axis_component_extents
                .iter()
                .map(|components| {
                    components
                        .iter()
                        .map(|extent| {
                            let value = extent
                                .eval(&|name| symbols.get(name).copied())
                                .ok_or_else(|| format!("unbound Metal axis extent `{extent}`"))?;
                            u64::try_from(value)
                                .map_err(|_| format!("negative Metal axis extent `{value}`"))
                        })
                        .collect()
                })
                .collect::<Result<_, _>>()?,
            operand_values: instruction.operand_values.clone(),
            input_operands: instruction.input_operands.clone(),
            output_operands: instruction.output_operands.clone(),
            views: instruction.views.clone(),
            view_bindings: instruction
                .view_bindings
                .iter()
                .map(|(view, bindings)| {
                    Ok((
                        *view,
                        bindings
                            .iter()
                            .map(|binding| {
                                Ok(seismic_compiler::terminal::ResolvedScalarStorageBinding {
                                    storage: storage.get(&binding.storage).copied().ok_or_else(
                                        || {
                                            format!(
                                                "Metal view names absent storage#{}",
                                                binding.storage.0
                                            )
                                        },
                                    )?,
                                    plane: binding.plane.clone(),
                                })
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                    ))
                })
                .collect::<Result<_, String>>()?,
            value_bindings: instruction
                .value_bindings
                .iter()
                .map(|(value, bindings)| {
                    Ok((
                        *value,
                        bindings
                            .iter()
                            .map(|binding| {
                                Ok(seismic_compiler::terminal::ResolvedScalarStorageBinding {
                                    storage: storage.get(&binding.storage).copied().ok_or_else(
                                        || {
                                            format!(
                                                "Metal value names absent storage#{}",
                                                binding.storage.0
                                            )
                                        },
                                    )?,
                                    plane: binding.plane.clone(),
                                })
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                    ))
                })
                .collect::<Result<_, String>>()?,
            access_bindings: instruction
                .scalar
                .consequences()
                .accesses
                .iter()
                .map(|access| {
                    storage.get(&access.storage).copied().ok_or_else(|| {
                        format!(
                            "Metal instruction access names absent storage#{}",
                            access.storage.0
                        )
                    })
                })
                .collect::<Result<_, _>>()?,
        })
    }

    fn resolve_layout(
        layout: &Self::TemplateLayout,
        symbols: &BTreeMap<String, i64>,
    ) -> Result<Self::ResolvedLayout, String> {
        let resolve = |value: &Sym| {
            let value = value
                .eval(&|name| symbols.get(name).copied())
                .ok_or_else(|| format!("unbound Metal layout extent `{value}`"))?;
            u64::try_from(value).map_err(|_| format!("negative Metal layout extent `{value}`"))
        };
        Ok(match layout {
            MetalLayout::Dense { dtype, elements } => ResolvedMetalLayout::Dense {
                dtype: *dtype,
                elements: resolve(elements)?,
            },
            MetalLayout::Scalar { parameters } => ResolvedMetalLayout::Scalar {
                parameters: parameters.clone(),
                bytes: seismic_lang::abi::ScalarLayout::natural(parameters)?.bytes as u64,
            },
            MetalLayout::PackedPlane {
                representation,
                plane,
                dtype,
                elements,
            } => ResolvedMetalLayout::PackedPlane {
                representation: representation.clone(),
                plane: plane.clone(),
                dtype: *dtype,
                elements: resolve(elements)?,
            },
        })
    }
}

fn metal_capability(name: &str) -> MetalCapability {
    if name == "metal.matrix" || name.starts_with("metal.matrix.") {
        MetalCapability::SimdgroupMatrix
    } else if name == "metal.subgroup" || name.starts_with("metal.subgroup.") {
        MetalCapability::Simdgroup
    } else if name == "metal.atomic32" || name.starts_with("metal.atomic32.") {
        MetalCapability::Atomic32
    } else if name == "metal.bfloat" || name.starts_with("metal.bfloat.") {
        MetalCapability::BFloat
    } else if let Some(width) = name.strip_prefix("metal.packed-vector.") {
        MetalCapability::PackedVector(width.parse().unwrap_or(1))
    } else {
        MetalCapability::Scalar
    }
}

#[derive(Clone)]
struct Plane {
    id: StorageId,
    plane: Vec<String>,
    operand: Option<OperandId>,
    direct: bool,
}

#[derive(Default)]
struct Ids {
    storage: u32,
    binding: u32,
    group: u32,
    phase: u32,
    launch: u32,
}

impl Ids {
    fn storage(&mut self) -> StorageId {
        let id = StorageId(self.storage);
        self.storage += 1;
        id
    }
    fn binding(&mut self) -> BindingId {
        let id = BindingId(self.binding);
        self.binding += 1;
        id
    }
    fn group(&mut self) -> BindingGroupId {
        let id = BindingGroupId(self.group);
        self.group += 1;
        id
    }
    fn phase(&mut self) -> PhaseId {
        let id = PhaseId(self.phase);
        self.phase += 1;
        id
    }
    fn launch(&mut self) -> LaunchId {
        let id = LaunchId(self.launch);
        self.launch += 1;
        id
    }
}

pub fn elaborate(
    logical: &seismic_lang::logical::LogicalProgram,
    max_participants_per_workgroup: u64,
) -> Result<PlanFamily<MetalDialect>, String> {
    let parallel_width = i64::try_from(max_participants_per_workgroup.min(256))
        .map_err(|_| "Metal participant width does not fit i64")?;
    let mut family = PlanFamilyBuilder::from_logical(logical)?;
    for graph in &logical.task_graphs {
        family.add_alternative(schedule_graph(graph, parallel_width)?)?;
    }
    family.finish()
}

fn schedule_graph(
    graph: &LogicalTaskGraph,
    parallel_width: i64,
) -> Result<seismic_realization::executable::PlanAlternative<MetalDialect>, String> {
    let mut builder = ScheduleBuilder::new(graph)?;
    let mut ids = Ids::default();
    let (storage, allocations, transports) = allocate_storage(graph, &mut builder, &mut ids)?;
    for (input, port) in graph.inputs.iter().enumerate() {
        let input = input as u32;
        let tensor = storage.iter().find_map(|(logical, planes)| match logical {
            StorageRef::Input { port: source, path }
                if path.as_slice() == port.path.get(1..).unwrap_or_default()
                    && graph
                        .input_remap
                        .get(*source as usize)
                        .copied()
                        .unwrap_or(*source)
                        == input =>
            {
                Some(ValueTransportTemplate::Storage(
                    nonempty(
                        planes.iter().map(|plane| plane.id).collect(),
                        "Metal input storage plane",
                    )
                    .ok()?,
                ))
            }
            _ => None,
        });
        let transport = tensor
            .or_else(|| {
                graph.operands.iter().find_map(|operand| {
                    let ValueRef::Input(source) = operand.value else {
                        return None;
                    };
                    (graph
                        .input_remap
                        .get(source as usize)
                        .copied()
                        .unwrap_or(source)
                        == input)
                        .then(|| transports.get(&operand.id).cloned())
                        .flatten()
                })
            })
            .ok_or_else(|| {
                format!(
                    "Metal input port#{input} path {:?} has no physical transport",
                    port.path
                )
            })?;
        builder.bind_input(builder.input(input)?, transport)?;
    }
    let scalar = scalar_bindings(graph, &storage);
    let produced_storage_values = graph
        .tasks
        .iter()
        .map(|task| terminal::lower_task(graph, task, &scalar))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flat_map(|program| program.instructions)
        .filter_map(|instruction| {
            let result = instruction.result?;
            let seismic_compiler::terminal::ScalarInstructionKind::Storage { bindings, .. } =
                instruction.kind
            else {
                return None;
            };
            Some((result, bindings))
        })
        .collect::<BTreeMap<_, _>>();
    let bindings = binding_groups(&allocations, &mut ids);
    let order = execution_order(graph)?;
    let mut pending_tasks = Vec::new();
    let mut previous_phase = None;

    for endpoint in order {
        match endpoint {
            LogicalEndpoint::Task(task) => pending_tasks.push(task),
            LogicalEndpoint::Call(call) => {
                flush_tasks(
                    graph,
                    &mut builder,
                    &mut ids,
                    &scalar,
                    &produced_storage_values,
                    &transports,
                    &bindings,
                    &mut pending_tasks,
                    &mut previous_phase,
                    parallel_width,
                )?;
                let logical_call = graph.call(call).ok_or("scheduled call is absent")?;
                let inputs = logical_call
                    .inputs
                    .iter()
                    .map(|operand| operand_binding(*operand, &transports))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut results = logical_call
                    .outputs
                    .iter()
                    .map(|operand| operand_binding(*operand, &transports))
                    .collect::<Result<Vec<_>, _>>()?;
                if !logical_call.outputs.contains(&logical_call.result) {
                    results.push(operand_binding(logical_call.result, &transports)?);
                }
                builder.add_subplan(builder.call(call)?, inputs, results)?;
            }
            LogicalEndpoint::Input(_) | LogicalEndpoint::Output(_) => unreachable!(),
        }
    }
    flush_tasks(
        graph,
        &mut builder,
        &mut ids,
        &scalar,
        &produced_storage_values,
        &transports,
        &bindings,
        &mut pending_tasks,
        &mut previous_phase,
        parallel_width,
    )?;

    for output in 0..graph.results.len() as u32 {
        let operand = output_operand(graph, output)?;
        builder.publish_output(
            builder.output(output)?,
            operand,
            transports
                .get(&operand)
                .cloned()
                .ok_or_else(|| format!("output operand#{} has no transport", operand.0))?,
        )?;
    }
    for dependency in &graph.dependencies {
        let transport = dependency_transport(dependency, &transports, &storage)?;
        builder.place_launch_boundary(builder.dependency(dependency.id)?, transport)?;
    }
    builder.finish(Sym::constant(graph.tasks.len() as i64))
}

#[allow(clippy::too_many_arguments)]
fn flush_tasks(
    graph: &LogicalTaskGraph,
    builder: &mut ScheduleBuilder<MetalDialect>,
    ids: &mut Ids,
    scalar: &ScalarBindings,
    produced_storage_values: &BTreeMap<ScalarValueId, Vec<ScalarStorageBinding>>,
    transports: &BTreeMap<OperandId, ValueTransportTemplate>,
    binding_groups: &[BindingGroupTemplate],
    tasks: &mut Vec<seismic_lang::logical::TaskId>,
    previous_phase: &mut Option<PhaseId>,
    parallel_width: i64,
) -> Result<(), String> {
    if tasks.is_empty() {
        return Ok(());
    }
    for task_id in tasks.drain(..) {
        let task = graph
            .task(task_id)
            .ok_or_else(|| format!("task#{} is absent", task_id.0))?;
        let program = terminal::lower_task(graph, task, scalar)?;
        let axis_values = task
            .domain
            .axes
            .iter()
            .map(|axis| {
                axis.binders
                    .iter()
                    .map(|value| scalar.values[&ValueBindingKey::Local(*value)])
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        let axis_component_extents = task
            .domain
            .axes
            .iter()
            .map(axis_components)
            .collect::<Result<Vec<_>, _>>()?;
        let mut instructions = Vec::new();
        let operand_values = task
            .inputs
            .iter()
            .chain(&task.outputs)
            .filter_map(|operand| {
                program
                    .operand_value(*operand)
                    .map(|value| (*operand, value))
            })
            .collect::<BTreeMap<_, _>>();
        for instruction in program.instructions() {
            let mut value_types = BTreeMap::new();
            for value in scalar
                .values
                .values()
                .chain(scalar.operands.values())
                .copied()
                .chain(instruction.result)
            {
                if let Some(ty) = program.value_type(value) {
                    value_types.insert(value, ty.clone());
                }
            }
            instructions.push(MetalInstruction {
                task: task_id,
                scalar: instruction.clone(),
                value_types,
                axis_values: axis_values.clone(),
                axis_component_extents: axis_component_extents.clone(),
                operand_values: operand_values.clone(),
                input_operands: task.inputs.iter().copied().collect(),
                output_operands: task.outputs.iter().copied().collect(),
                views: graph
                    .views
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, view)| (LocalViewId(index as u32), view))
                    .collect(),
                view_bindings: scalar.views.clone(),
                value_bindings: produced_storage_values
                    .iter()
                    .map(|(value, bindings)| (*value, bindings.clone()))
                    .chain(graph.operands.iter().filter_map(|operand| {
                        let storage = operand.storage.as_ref()?;
                        Some((
                            *scalar.operands.get(&operand.id)?,
                            scalar.storage.get(storage)?.clone(),
                        ))
                    }))
                    .collect(),
            });
        }
        let operand_bindings = task
            .inputs
            .iter()
            .chain(&task.outputs)
            .map(|operand| operand_binding(*operand, transports))
            .collect::<Result<Vec<_>, _>>()?;
        let (mapping, geometry) = participant_map(task, parallel_width)?;
        let step = builder.mapped_task_step(
            builder.task(task_id)?,
            mapping,
            operand_bindings,
            nonempty(instructions, "Metal scalar instruction")?,
        )?;
        let phase = ids.phase();
        builder.add_phase(PhaseTemplate {
            id: phase,
            predecessors: previous_phase.iter().copied().collect(),
            launches: NonEmpty::new(LaunchTemplate {
                id: ids.launch(),
                geometry,
                binding_groups: binding_groups.to_vec(),
                kernel: KernelTemplate {
                    participant_storage: vec![],
                    workgroup_storage: vec![],
                    steps: NonEmpty::new(step),
                },
            }),
        })?;
        *previous_phase = Some(phase);
    }
    Ok(())
}

fn participant_map(
    task: &LogicalTask,
    parallel_width: i64,
) -> Result<(ParticipantMap, DispatchTemplate), String> {
    let logical_extents = task
        .domain
        .axes
        .iter()
        .map(axis_extent)
        .collect::<Result<Vec<_>, _>>()?;
    let mut axes = Vec::new();
    let mut workgroups = [Sym::constant(1), Sym::constant(1), Sym::constant(1)];
    let mut participants = [Sym::constant(1), Sym::constant(1), Sym::constant(1)];
    let mut grid_axis = 0u8;
    for (logical_axis, axis) in task.domain.axes.iter().enumerate() {
        if axis.order == AxisOrder::Independent && grid_axis < 3 {
            let width = if grid_axis == 0 { parallel_width } else { 1 };
            participants[grid_axis as usize] = Sym::constant(width);
            workgroups[grid_axis as usize] = ceil_div(&logical_extents[logical_axis], width);
            axes.push(AxisMap::Grid {
                logical_axis: logical_axis as u32,
                workgroup_axis: grid_axis,
                participant_axis: grid_axis,
                mode: GridMapping::OnePass,
            });
            grid_axis += 1;
        } else {
            axes.push(AxisMap::Serial {
                logical_axis: logical_axis as u32,
            });
        }
    }
    Ok((
        ParticipantMap {
            task: task.id,
            axes,
            logical_extents,
            masks_inactive_participants: true,
        },
        DispatchTemplate {
            workgroups,
            participants_per_workgroup: participants,
        },
    ))
}

fn ceil_div(value: &Sym, divisor: i64) -> Sym {
    Sym::atom(seismic_lang::sym::Atom::Quot(
        Box::new(value.add(&Sym::constant(divisor - 1))),
        Box::new(Sym::constant(divisor)),
    ))
}

fn axis_extent(axis: &seismic_lang::logical::LogicalAxis) -> Result<Sym, String> {
    match &axis.source {
        LogicalAxisSource::Range { lo, hi, runtime } => runtime
            .as_ref()
            .map(expr_sym)
            .unwrap_or_else(|| Ok(expr_sym(hi)?.sub(&expr_sym(lo)?))),
        LogicalAxisSource::Coordinates { value, axes } => {
            let Type::Tensor(tensor) = &value.ty else {
                return Err("coordinate domain source is not a tensor".into());
            };
            axes.iter().try_fold(Sym::constant(1), |total, axis| {
                Ok(total.mul(
                    tensor
                        .shape
                        .get(*axis)
                        .ok_or("coordinate domain axis is out of bounds")?,
                ))
            })
        }
        LogicalAxisSource::Members { slice } => Ok(Sym::constant(i64::from(*slice))),
    }
}

fn axis_components(axis: &seismic_lang::logical::LogicalAxis) -> Result<Vec<Sym>, String> {
    match &axis.source {
        LogicalAxisSource::Coordinates { value, axes } => {
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
        _ => Ok(vec![axis_extent(axis)?]),
    }
}

fn expr_sym(expr: &LogicalExpr) -> Result<Sym, String> {
    match &expr.kind {
        LogicalExprKind::Int(value) => Ok(Sym::constant(*value)),
        LogicalExprKind::Shape(value) => Ok(value.clone()),
        LogicalExprKind::Binary { op, lhs, rhs } => {
            let lhs = expr_sym(lhs)?;
            let rhs = expr_sym(rhs)?;
            use seismic_lang::syntax::ast::BinaryOp;
            match op {
                BinaryOp::Add => Ok(lhs.add(&rhs)),
                BinaryOp::Sub => Ok(lhs.sub(&rhs)),
                BinaryOp::Mul => Ok(lhs.mul(&rhs)),
                _ => Err("logical axis extent is not affine integer arithmetic".into()),
            }
        }
        _ => Err("logical axis extent has no physical symbolic form".into()),
    }
}

fn scalar_bindings(
    graph: &LogicalTaskGraph,
    storage: &BTreeMap<StorageRef, Vec<Plane>>,
) -> ScalarBindings {
    let mut result = ScalarBindings::default();
    let mut next = 0u32;
    let mut bind = |key, result: &mut ScalarBindings| {
        result.values.entry(key).or_insert_with(|| {
            let value = ScalarValueId(next);
            next += 1;
            value
        });
    };
    for port in 0..graph.inputs.len() as u32 {
        bind(ValueBindingKey::Input(port), &mut result);
    }
    for port in 0..graph.results.len() as u32 {
        bind(ValueBindingKey::Result(port), &mut result);
    }
    for local in 0..graph.values.len() as u32 {
        bind(
            ValueBindingKey::Local(seismic_lang::logical::LocalValueId(local)),
            &mut result,
        );
    }
    for operand in &graph.operands {
        if let Some(value) = result
            .values
            .get(&ValueBindingKey::from(&operand.value))
            .copied()
        {
            result.operands.insert(operand.id, value);
        }
    }
    for (logical, planes) in storage {
        result.storage.insert(
            logical.clone(),
            planes
                .iter()
                .map(|plane| ScalarStorageBinding {
                    storage: plane.id,
                    plane: plane.plane.clone(),
                })
                .collect(),
        );
    }
    for (ordinal, view) in graph.views.iter().enumerate() {
        if let Some(planes) = storage.get(&view.storage) {
            result.views.insert(
                LocalViewId(ordinal as u32),
                planes
                    .iter()
                    .map(|plane| ScalarStorageBinding {
                        storage: plane.id,
                        plane: plane.plane.clone(),
                    })
                    .collect(),
            );
        }
    }
    result
}

fn allocate_storage(
    graph: &LogicalTaskGraph,
    builder: &mut ScheduleBuilder<MetalDialect>,
    ids: &mut Ids,
) -> Result<
    (
        BTreeMap<StorageRef, Vec<Plane>>,
        Vec<Plane>,
        BTreeMap<OperandId, ValueTransportTemplate>,
    ),
    String,
> {
    let mut refs = BTreeSet::new();
    refs.extend(
        graph
            .operands
            .iter()
            .filter_map(|operand| operand.storage.clone()),
    );
    refs.extend(
        graph
            .tasks
            .iter()
            .flat_map(|task| task.effects.accesses.iter())
            .map(|access| access.storage.clone()),
    );
    refs.extend(
        graph
            .calls
            .iter()
            .flat_map(|call| call.effects.accesses.iter())
            .map(|access| access.storage.clone()),
    );
    let effects = graph
        .tasks
        .iter()
        .flat_map(|task| task.effects.accesses.iter())
        .chain(
            graph
                .calls
                .iter()
                .flat_map(|call| call.effects.accesses.iter()),
        )
        .map(|access| access.storage.clone())
        .collect::<BTreeSet<_>>();
    let mut by_ref = BTreeMap::new();
    let mut all = Vec::new();
    for logical_ref in refs {
        let tensor = tensor_for_storage(graph, &logical_ref)?;
        let operand = graph
            .operands
            .iter()
            .find(|operand| operand.storage.as_ref() == Some(&logical_ref))
            .map(|operand| operand.id);
        let mut planes = Vec::new();
        for (plane, layout, bytes, alignment, address) in tensor_planes(&tensor)? {
            let id = ids.storage();
            let abi = match &logical_ref {
                StorageRef::Input { port, .. } => {
                    let ordinal = graph
                        .input_remap
                        .get(*port as usize)
                        .copied()
                        .unwrap_or(*port);
                    Some(AbiRole::Parameter {
                        ordinal,
                        path: graph
                            .inputs
                            .get(ordinal as usize)
                            .map(|port| port.path.clone())
                            .ok_or_else(|| format!("input remap names absent port#{ordinal}"))?,
                        representation_plane: plane.first().cloned(),
                    })
                }
                StorageRef::Result { port, .. } => {
                    let ordinal = graph
                        .result_remap
                        .get(*port as usize)
                        .copied()
                        .unwrap_or(*port);
                    Some(AbiRole::Result {
                        ordinal,
                        path: graph
                            .results
                            .get(ordinal as usize)
                            .map(|port| port.path.clone())
                            .ok_or_else(|| format!("result remap names absent port#{ordinal}"))?,
                        representation_plane: plane.first().cloned(),
                    })
                }
                StorageRef::Local(_) => None,
            };
            let template = StorageTemplate {
                id,
                scope: if abi.is_some() {
                    StorageScope::External
                } else {
                    StorageScope::Device
                },
                replication: Replication::Once,
                bytes: bytes.clone(),
                alignment,
                layout,
                provenance: PhysicalStorageProvenance {
                    logical_storage: Some(logical_ref.clone()),
                    operand,
                    view: None,
                    subrange: PhysicalSubrangeTemplate {
                        byte_offset: Sym::constant(0),
                        bytes,
                    },
                    address,
                    abi,
                },
            };
            if let Some(operand) = operand {
                builder.add_operand_storage(operand, template)?;
            } else if effects.contains(&logical_ref) {
                builder.add_effect_storage(logical_ref.clone(), template)?;
            } else {
                return Err(format!(
                    "storage {logical_ref:?} has no operand or effect provenance"
                ));
            }
            let physical = Plane {
                id,
                plane,
                operand,
                direct: false,
            };
            planes.push(physical.clone());
            all.push(physical);
        }
        by_ref.insert(logical_ref, planes);
    }
    let mut transports = BTreeMap::new();
    let aggregate_results = graph
        .calls
        .iter()
        .filter(|call| !call.outputs.contains(&call.result))
        .map(|call| call.result)
        .collect::<BTreeSet<_>>();
    let mut value_transports = BTreeMap::<ValueBindingKey, ValueTransportTemplate>::new();
    for operand in &graph.operands {
        if let Some(storage) = &operand.storage {
            let planes = by_ref
                .get(storage)
                .ok_or_else(|| format!("operand#{} storage was not allocated", operand.id.0))?;
            transports.insert(
                operand.id,
                ValueTransportTemplate::Storage(nonempty(
                    planes.iter().map(|plane| plane.id).collect(),
                    "tensor storage plane",
                )?),
            );
        } else {
            if aggregate_results.contains(&operand.id) {
                continue;
            }
            let key = ValueBindingKey::from(&operand.value);
            let transport = if let Some(existing) = value_transports.get(&key) {
                existing.clone()
            } else {
                let transport = allocate_value_transport(
                    graph,
                    builder,
                    ids,
                    &mut all,
                    operand.id,
                    &operand.value,
                    &operand.ty,
                    &mut Vec::new(),
                )?;
                value_transports.insert(key, transport.clone());
                transport
            };
            transports.insert(operand.id, transport);
        }
    }
    // The aggregate result of a tuple-valued call is a structural view over
    // the exact flattened result transports, never a duplicate allocation.
    for call in &graph.calls {
        if call.outputs.contains(&call.result) {
            continue;
        }
        let ty = &graph
            .operand(call.result)
            .ok_or("call aggregate result operand is absent")?
            .ty;
        if matches!(ty, Type::Void) {
            transports.insert(call.result, ValueTransportTemplate::Void);
            continue;
        }
        let mut leaves =
            call.outputs
                .iter()
                .map(|operand| {
                    transports.get(operand).cloned().ok_or_else(|| {
                        format!("call result operand#{} has no transport", operand.0)
                    })
                })
                .collect::<Result<VecDeque<_>, _>>()?;
        let aggregate = rebuild_tuple_transport(ty, &mut leaves)?;
        if !leaves.is_empty() {
            return Err("call result transport has excess flattened leaves".into());
        }
        transports.insert(call.result, aggregate);
    }
    Ok((by_ref, all, transports))
}

fn allocate_value_transport(
    graph: &LogicalTaskGraph,
    builder: &mut ScheduleBuilder<MetalDialect>,
    ids: &mut Ids,
    all: &mut Vec<Plane>,
    operand: OperandId,
    value: &ValueRef,
    ty: &Type,
    path: &mut Vec<u32>,
) -> Result<ValueTransportTemplate, String> {
    if let Type::Tuple(fields) = ty {
        let mut values = Vec::new();
        for (field, ty) in fields.iter().enumerate() {
            path.push(field as u32);
            values.push(Box::new(allocate_value_transport(
                graph, builder, ids, all, operand, value, ty, path,
            )?));
            path.pop();
        }
        return Ok(ValueTransportTemplate::Tuple(nonempty(
            values,
            "tuple transport",
        )?));
    }
    let descriptions = match ty {
        Type::Tensor(tensor) => tensor_planes(tensor)?,
        Type::Scalar(dtype) => vec![scalar_plane(vec![ScalarParameter::plain(
            value_name(value, path),
            *dtype,
        )])?],
        Type::Index { bound } => vec![scalar_plane(vec![ScalarParameter {
            name: value_name(value, path),
            dtype: DType::I32,
            index_bound: bound
                .as_constant()
                .and_then(|value| u64::try_from(value).ok()),
            range: None,
        }])?],
        Type::Range { bound } => {
            let name = value_name(value, path);
            let bound = bound
                .as_constant()
                .and_then(|value| u64::try_from(value).ok())
                .unwrap_or(u64::MAX);
            vec![scalar_plane(vec![
                ScalarParameter {
                    name: format!("{name}_start"),
                    dtype: DType::I32,
                    index_bound: None,
                    range: Some(RangeScalar {
                        parameter: name.clone(),
                        endpoint: RangeEndpoint::Start,
                        bound,
                    }),
                },
                ScalarParameter {
                    name: format!("{name}_end"),
                    dtype: DType::I32,
                    index_bound: None,
                    range: Some(RangeScalar {
                        parameter: name,
                        endpoint: RangeEndpoint::End,
                        bound,
                    }),
                },
            ])?]
        }
        Type::Void => return Ok(ValueTransportTemplate::Void),
        Type::CapabilityValue { .. } => {
            return Err("capability values have no Metal invocation transport".into())
        }
        Type::Tuple(_) => unreachable!(),
    };
    let mut storages = Vec::new();
    for (plane, layout, bytes, alignment, address) in descriptions {
        let id = ids.storage();
        let abi = match value {
            ValueRef::Input(port) => {
                let ordinal = graph
                    .input_remap
                    .get(*port as usize)
                    .copied()
                    .unwrap_or(*port);
                let mut abi_path = graph
                    .inputs
                    .get(ordinal as usize)
                    .map(|port| port.path.clone())
                    .ok_or_else(|| format!("input remap names absent port#{ordinal}"))?;
                abi_path.extend(path.iter().copied());
                Some(AbiRole::Parameter {
                    ordinal,
                    path: abi_path,
                    representation_plane: plane.first().cloned(),
                })
            }
            ValueRef::Result(port) => {
                let ordinal = graph
                    .result_remap
                    .get(*port as usize)
                    .copied()
                    .unwrap_or(*port);
                let mut abi_path = graph
                    .results
                    .get(ordinal as usize)
                    .map(|port| port.path.clone())
                    .ok_or_else(|| format!("result remap names absent port#{ordinal}"))?;
                abi_path.extend(path.iter().copied());
                Some(AbiRole::Result {
                    ordinal,
                    path: abi_path,
                    representation_plane: plane.first().cloned(),
                })
            }
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
                alignment,
                layout,
                provenance: PhysicalStorageProvenance {
                    logical_storage: None,
                    operand: Some(operand),
                    view: None,
                    subrange: PhysicalSubrangeTemplate {
                        byte_offset: Sym::constant(0),
                        bytes,
                    },
                    address,
                    abi,
                },
            },
        )?;
        all.push(Plane {
            id,
            plane,
            operand: Some(operand),
            direct: matches!(value, ValueRef::Input(_))
                && matches!(
                    ty,
                    Type::Scalar(_) | Type::Index { .. } | Type::Range { .. }
                ),
        });
        storages.push(id);
    }
    Ok(ValueTransportTemplate::Storage(nonempty(
        storages,
        "scalar storage",
    )?))
}

fn scalar_plane(parameters: Vec<ScalarParameter>) -> Result<PlaneDescription, String> {
    let bytes = seismic_lang::abi::ScalarLayout::natural(&parameters)?.bytes as u64;
    Ok((
        vec![],
        MetalLayout::Scalar { parameters },
        Sym::constant(bytes as i64),
        bytes.next_power_of_two().min(8),
        PhysicalAddressTemplate::DenseAffine {
            byte_offset: Sym::constant(0),
            byte_strides: vec![],
        },
    ))
}

fn value_name(value: &ValueRef, path: &[u32]) -> String {
    let base = match value {
        ValueRef::Input(port) => format!("parameter_{port}"),
        ValueRef::Result(port) => format!("result_{port}"),
        ValueRef::Local(local) => format!("local_{}", local.0),
    };
    if path.is_empty() {
        base
    } else {
        format!(
            "{base}_{}",
            path.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("_")
        )
    }
}

fn rebuild_tuple_transport(
    ty: &Type,
    leaves: &mut VecDeque<ValueTransportTemplate>,
) -> Result<ValueTransportTemplate, String> {
    match ty {
        Type::Void => Ok(ValueTransportTemplate::Void),
        Type::Tuple(fields) => {
            let values = fields
                .iter()
                .map(|field| rebuild_tuple_transport(field, leaves).map(Box::new))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ValueTransportTemplate::Tuple(nonempty(
                values,
                "aggregate call result transport",
            )?))
        }
        _ => leaves
            .pop_front()
            .ok_or_else(|| "call result transport has too few flattened leaves".into()),
    }
}

fn tensor_for_storage(
    graph: &LogicalTaskGraph,
    storage: &StorageRef,
) -> Result<TensorType, String> {
    let ty = match storage {
        StorageRef::Input { port, path } => graph
            .inputs
            .iter()
            .find(|candidate| candidate.path == *path)
            .or_else(|| graph.inputs.get(*port as usize))
            .map(|port| port.ty.clone()),
        StorageRef::Result { port, path } => graph
            .results
            .iter()
            .find(|candidate| candidate.path == *path)
            .or_else(|| graph.results.get(*port as usize))
            .map(|port| port.ty.clone()),
        StorageRef::Local(local) => graph
            .storage
            .get(local.0 as usize)
            .map(|value| Type::Tensor(value.ty.clone())),
    }
    .ok_or_else(|| format!("storage {storage:?} has no logical type"))?;
    let Type::Tensor(tensor) = ty else {
        return Err(format!("storage {storage:?} is not a tensor"));
    };
    Ok(tensor)
}

type PlaneDescription = (Vec<String>, MetalLayout, Sym, u64, PhysicalAddressTemplate);

fn tensor_planes(tensor: &TensorType) -> Result<Vec<PlaneDescription>, String> {
    let elements = product(&tensor.shape);
    match &tensor.elem {
        Elem::Dtype(dtype) => Ok(vec![(
            vec![],
            MetalLayout::Dense {
                dtype: *dtype,
                elements: elements.clone(),
            },
            elements.scale(i64::from(dtype.bytes())),
            u64::from(dtype.bytes()),
            PhysicalAddressTemplate::DenseAffine {
                byte_offset: Sym::constant(0),
                byte_strides: byte_strides(&tensor.shape, i64::from(dtype.bytes())),
            },
        )]),
        Elem::Repr(name) => {
            let representation = repr::lookup(name)
                .ok_or_else(|| format!("unknown packed representation `{name}`"))?;
            representation
                .planes()
                .into_iter()
                .map(|plane| {
                    let plane_elements = plane.extent(&elements);
                    Ok((
                        vec![plane.name.into()],
                        MetalLayout::PackedPlane {
                            representation: name.clone(),
                            plane: plane.name.into(),
                            dtype: plane.dtype(),
                            elements: plane_elements.clone(),
                        },
                        plane_elements.scale(i64::from(plane.dtype().bytes())),
                        u64::from(plane.dtype().bytes()),
                        PhysicalAddressTemplate::Representation {
                            representation: name.clone(),
                            plane: plane.name.into(),
                            logical_strides: byte_strides(&tensor.shape, 1),
                        },
                    ))
                })
                .collect()
        }
        Elem::Param(name) => Err(format!("unresolved element parameter `{name}`")),
    }
}

fn product(shape: &[Sym]) -> Sym {
    shape
        .iter()
        .fold(Sym::constant(1), |value, extent| value.mul(extent))
}

fn byte_strides(shape: &[Sym], element_bytes: i64) -> Vec<Sym> {
    let mut stride = Sym::constant(element_bytes);
    let mut result = vec![Sym::constant(0); shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = stride.clone();
        stride = stride.mul(&shape[axis]);
    }
    result
}

fn binding_groups(allocations: &[Plane], ids: &mut Ids) -> Vec<BindingGroupTemplate> {
    if allocations.is_empty() {
        return vec![];
    }
    if allocations.len() <= crate::msl::MAX_KERNEL_BUFFERS {
        allocations
            .iter()
            .enumerate()
            .map(|(slot, allocation)| BindingGroupTemplate {
                id: ids.group(),
                kind: BindingGroupKind::Direct,
                slot: slot as u32,
                members: NonEmpty::new(BindingTemplate {
                    id: ids.binding(),
                    storage: allocation.id,
                    access: AccessMode::ReadWrite,
                    operand: allocation.operand,
                }),
            })
            .collect()
    } else {
        let mut groups = Vec::new();
        let mut slot = 0u32;
        for allocation in allocations.iter().filter(|allocation| allocation.direct) {
            groups.push(BindingGroupTemplate {
                id: ids.group(),
                kind: BindingGroupKind::Direct,
                slot,
                members: NonEmpty::new(BindingTemplate {
                    id: ids.binding(),
                    storage: allocation.id,
                    access: AccessMode::ReadWrite,
                    operand: allocation.operand,
                }),
            });
            slot += 1;
        }
        let mut members = allocations
            .iter()
            .filter(|allocation| !allocation.direct)
            .map(|allocation| BindingTemplate {
                id: ids.binding(),
                storage: allocation.id,
                access: AccessMode::ReadWrite,
                operand: allocation.operand,
            });
        if let Some(first) = members.next() {
            let mut values = NonEmpty::new(first);
            for value in members {
                values.push(value);
            }
            groups.push(BindingGroupTemplate {
                id: ids.group(),
                kind: BindingGroupKind::ArgumentTable,
                slot,
                members: values,
            });
        }
        groups
    }
}

fn execution_order(graph: &LogicalTaskGraph) -> Result<Vec<LogicalEndpoint>, String> {
    let nodes = graph
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
    let mut indegree = nodes
        .iter()
        .map(|node| (*node, 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut edges = BTreeMap::<LogicalEndpoint, BTreeSet<LogicalEndpoint>>::new();
    for dependency in &graph.dependencies {
        if nodes.contains(&dependency.from)
            && nodes.contains(&dependency.to)
            && edges
                .entry(dependency.from)
                .or_default()
                .insert(dependency.to)
        {
            *indegree.get_mut(&dependency.to).unwrap() += 1;
        }
    }
    let mut ready = indegree
        .iter()
        .filter_map(|(node, degree)| (*degree == 0).then_some(*node))
        .collect::<VecDeque<_>>();
    let mut order = Vec::new();
    while let Some(node) = ready.pop_front() {
        order.push(node);
        if let Some(consumers) = edges.get(&node) {
            for consumer in consumers {
                let degree = indegree.get_mut(consumer).unwrap();
                *degree -= 1;
                if *degree == 0 {
                    ready.push_back(*consumer);
                }
            }
        }
    }
    if order.len() != nodes.len() {
        return Err("logical task/call dependencies contain a cycle".into());
    }
    Ok(order)
}

fn output_operand(graph: &LogicalTaskGraph, output: u32) -> Result<OperandId, String> {
    graph
        .dependencies
        .iter()
        .find_map(|dependency| {
            if dependency.to != LogicalEndpoint::Output(output) {
                return None;
            }
            match dependency.kind {
                seismic_lang::logical::LogicalDependencyKind::Value(operand) => Some(operand),
                _ => None,
            }
        })
        .or_else(|| {
            graph
                .operands
                .iter()
                .find(|operand| operand.value == ValueRef::Result(output))
                .map(|operand| operand.id)
        })
        .ok_or_else(|| format!("output#{output} has no value operand"))
}

fn operand_binding(
    operand: OperandId,
    transports: &BTreeMap<OperandId, ValueTransportTemplate>,
) -> Result<OperandTransportTemplate, String> {
    Ok(OperandTransportTemplate {
        operand,
        transport: transports
            .get(&operand)
            .cloned()
            .ok_or_else(|| format!("operand#{} has no physical transport", operand.0))?,
    })
}

fn dependency_transport(
    dependency: &seismic_lang::logical::LogicalDependency,
    transports: &BTreeMap<OperandId, ValueTransportTemplate>,
    storage: &BTreeMap<StorageRef, Vec<Plane>>,
) -> Result<DependencyTransport, String> {
    use seismic_lang::logical::LogicalDependencyKind as K;
    Ok(match &dependency.kind {
        K::Control => DependencyTransport::Control,
        K::Value(operand) => DependencyTransport::Value {
            operand: *operand,
            transport: transports
                .get(operand)
                .cloned()
                .ok_or_else(|| format!("dependency operand#{} has no transport", operand.0))?,
        },
        K::Effect(logical_storage) => DependencyTransport::Effect {
            logical_storage: logical_storage.clone(),
            storage: storage_transport(storage, logical_storage)?,
        },
        K::Ownership(logical_storage) => DependencyTransport::Ownership {
            logical_storage: logical_storage.clone(),
            storage: storage_transport(storage, logical_storage)?,
        },
    })
}

fn storage_transport(
    storage: &BTreeMap<StorageRef, Vec<Plane>>,
    logical: &StorageRef,
) -> Result<NonEmpty<StorageId>, String> {
    nonempty(
        storage
            .get(logical)
            .ok_or_else(|| format!("dependency storage {logical:?} was not allocated"))?
            .iter()
            .map(|plane| plane.id)
            .collect(),
        "dependency storage plane",
    )
}

fn nonempty<T>(mut values: Vec<T>, description: &str) -> Result<NonEmpty<T>, String> {
    if values.is_empty() {
        return Err(format!("{description} list is empty"));
    }
    let mut result = NonEmpty::new(values.remove(0));
    for value in values {
        result.push(value);
    }
    Ok(result)
}
