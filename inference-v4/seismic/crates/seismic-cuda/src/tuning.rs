//! CUDA choices and resource analysis share the retained terminal execution.
//! Instruction order is fixed by the target plan; inter-warp scheduling is a
//! conditional machine behavior, not an emitted compiler scheduling decision.
use crate::{
    DeviceInfo,
    execution::{Execution, Limits},
    model::{self, CudaHardware},
};
use seismic_accounting::{
    schedule,
    selection::{self, Choices, IntegerRange, Objective},
    workload::{DerivationError, DerivationLimits, ScalarWorkload},
};
use seismic_compiler::tuner::{self as compiler, Preparation};
use seismic_lang::{lowered_ir::LoweredIr, normalize::loads};
use seismic_realization::{CallConv, Dispatch, ScalarProgram};
use std::sync::Arc;
mod relaxation;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchChoice {
    pub subgroup: bool,
}
impl Choices for DispatchChoice {
    type Alternative = Dispatch;
    fn len(&self) -> usize {
        if self.subgroup { 1 } else { 2 }
    }
    fn get(&self, index: usize) -> Option<Dispatch> {
        if self.subgroup {
            return (index == 0).then_some(Dispatch::ParallelRoot);
        }
        [Dispatch::Sequential, Dispatch::ParallelRoot]
            .get(index)
            .copied()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldImplementation {
    Thread,
    Subgroup,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldChoice {
    pub site: usize,
}
impl Choices for FoldChoice {
    type Alternative = FoldImplementation;
    fn len(&self) -> usize {
        2
    }
    fn get(&self, index: usize) -> Option<Self::Alternative> {
        [FoldImplementation::Thread, FoldImplementation::Subgroup]
            .get(index)
            .copied()
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockChoice {
    pub phase: usize,
    family: Arc<BlockFamily>,
}
impl BlockChoice {
    pub fn family(&self) -> &BlockFamily {
        &self.family
    }
}

/// Actual selected PTX and unresolved launch dimensions. Resolved earlier phases
/// and the remaining legal intervals stay with the same implementation owner.
#[derive(Clone)]
pub struct BlockFamily {
    phases: Vec<Phase>,
    selected: Vec<u32>,
    limits: Limits,
}
#[derive(Clone)]
struct Phase {
    program: ScalarProgram,
    target: crate::ptx::TargetPlan,
}
impl std::fmt::Debug for BlockFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockFamily")
            .field("phase_count", &self.phases.len())
            .field("selected", &self.selected)
            .field("limits", &self.limits)
            .finish()
    }
}
impl PartialEq for BlockFamily {
    fn eq(&self, other: &Self) -> bool {
        self.selected == other.selected
            && self.limits == other.limits
            && self.phases.len() == other.phases.len()
            && self.phases.iter().zip(&other.phases).all(|(a, b)| {
                a.target == b.target
                    && a.program.buffers == b.program.buffers
                    && a.program.scalars == b.program.scalars
            })
    }
}
impl Eq for BlockFamily {}
impl BlockFamily {
    pub fn phase_count(&self) -> usize {
        self.phases.len()
    }
    pub fn target(&self, phase: usize) -> Option<&crate::ptx::TargetPlan> {
        self.phases.get(phase).map(|p| &p.target)
    }
    pub fn selected_blocks(&self) -> &[u32] {
        &self.selected
    }
    pub fn lanes_per_item(&self, phase: usize) -> Option<u32> {
        self.phases
            .get(phase)
            .map(|p| p.program.participation.lanes())
    }
    /// Items per block; each item owns its retained complete participant group.
    pub fn block_interval(&self, phase: usize) -> Option<std::ops::RangeInclusive<u64>> {
        let program = &self.phases.get(phase)?.program;
        if let Some(&selected) = self.selected.get(phase) {
            let items = u64::from(selected) / u64::from(program.participation.lanes());
            return Some(items..=items);
        }
        Some(
            program
                .work_items
                .div_ceil(u64::from(self.limits.max_grid_x))
                .max(1)
                ..=u64::from(self.limits.max_threads_per_block / program.participation.lanes()),
        )
    }
    fn finish(self) -> Result<Vec<Execution>, String> {
        if self.selected.len() != self.phases.len() {
            return Err("unresolved CUDA block dimensions".into());
        }
        self.phases
            .into_iter()
            .zip(self.selected)
            .map(|(phase, threads)| {
                Execution::from_plan(phase.program, phase.target, threads, self.limits)
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conditions {
    pub device: DeviceInfo,
    pub hardware: CudaHardware,
}
pub struct Backend {
    conditions: Conditions,
}
impl Backend {
    pub fn new(device: &DeviceInfo, hardware: &CudaHardware) -> Result<Self, String> {
        validate_device(device)?;
        if hardware.warp_width != device.warp_size
            || hardware.execution_units != device.multiprocessors as usize
        {
            return Err("CUDA hardware model geometry differs from the execution device".into());
        }
        // Driver allocation APIs guarantee at least 256-byte root alignment;
        // internal allocations do not exist yet and cannot promise more.
        // https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#device-memory-accesses
        if !hardware.internal_alignment.is_power_of_two() || hardware.internal_alignment > 256 {
            return Err("CUDA internal alignment exceeds the allocator guarantee".into());
        }
        Ok(Self {
            conditions: Conditions {
                device: device.clone(),
                hardware: hardware.clone(),
            },
        })
    }
}

fn validate_device(device: &DeviceInfo) -> Result<(), String> {
    if device.max_grid_x == 0
        || device.max_threads_per_block == 0
        || device.warp_size == 0
        || device.multiprocessors == 0
    {
        return Err("CUDA device requires positive execution and dispatch capacities".into());
    }
    if device.compute_capability.0 < 8 {
        return Err("retained CUDA target requires compute capability 8.0 or newer".into());
    }
    Ok(())
}

/// The single preparation path for compiler search and diagnostic choices.
/// Every legal block size in the device's one-dimensional grid form is exposed;
/// load decisions and dispatch can change the phase sequence before sizing it.
pub fn prepare(
    function: &LoweredIr,
    device: &DeviceInfo,
    path: &[usize],
) -> Result<Preparation<Vec<Execution>>, String> {
    if function.backend != "cuda" {
        return Err("CUDA preparation requires CUDA Lowered IR".into());
    }
    validate_device(device)?;
    let (function, consumed) = match loads::expand(function, path)? {
        loads::Expansion::Choice(choice) => {
            return Ok(Preparation::Choice {
                name: format!("load site {} (variable {})", choice.site, choice.variable),
                alternatives: selection::Domain::new(choice)?,
            });
        }
        loads::Expansion::Selected { function, consumed } => (function, consumed),
    };
    let mut path = &path[consumed..];
    let mut selected_folds = Vec::new();
    if device.warp_size == 32 && device.max_threads_per_block >= 32 {
        for site in seismic_lang::reduction::structured::participants::candidates(&function, 32) {
            let choice = FoldChoice { site };
            let Some((&index, remaining)) = path.split_first() else {
                return Ok(Preparation::Choice {
                    name: format!("CUDA fold {site} participant ownership"),
                    alternatives: selection::Domain::new(choice)?,
                });
            };
            if choice
                .get(index)
                .ok_or("CUDA fold ownership choice is outside its domain")?
                == FoldImplementation::Subgroup
            {
                selected_folds.push(site);
            }
            path = remaining;
        }
    }
    let function =
        seismic_lang::reduction::structured::participants::apply(&function, &selected_folds, 32)?
            .function;
    let subgroup = seismic_compiler::subgroup_required(&function);
    if subgroup && device.warp_size != 32 {
        return Err("CUDA subgroup implementation requires a 32-lane warp".into());
    }
    let dispatches = DispatchChoice { subgroup };
    let Some(&dispatch) = path.first() else {
        return Ok(Preparation::Choice {
            name: "work dispatch".into(),
            alternatives: selection::Domain::new(dispatches)?,
        });
    };
    let dispatch = dispatches
        .get(dispatch)
        .ok_or("CUDA dispatch choice is outside its domain")?;
    let sequence = seismic_compiler::scalar_sequence_participants_resolved(
        &function,
        CallConv::SystemV,
        dispatch,
        if subgroup {
            seismic_realization::dispatch::Participation::Subgroup { lanes: 32 }
        } else {
            seismic_realization::dispatch::Participation::Thread
        },
    )?;
    if sequence.phases.is_empty() {
        return Err("CUDA form requires a nonempty phase sequence".into());
    }
    let limits = Limits {
        max_threads_per_block: device.max_threads_per_block,
        max_grid_x: device.max_grid_x,
    };
    let mut family = BlockFamily {
        phases: Vec::new(),
        selected: Vec::new(),
        limits,
    };
    for (phase, program) in sequence.phases.into_iter().enumerate() {
        let minimum = program
            .program
            .work_items
            .div_ceil(u64::from(limits.max_grid_x))
            .max(1);
        let maximum =
            u64::from(limits.max_threads_per_block / program.program.participation.lanes());
        if minimum > maximum {
            return Ok(Preparation::Infeasible(selection::CapacityViolation {
                resource: format!("CUDA phase {phase} work items in one-dimensional grid"),
                required: program.program.work_items,
                available: u64::from(limits.max_grid_x) * maximum,
            }));
        }
        let target = crate::ptx::prepare(&program.program)?;
        family.phases.push(Phase {
            program: program.program,
            target,
        });
    }
    for phase in 0..family.phases.len() {
        let interval = family.block_interval(phase).expect("known CUDA phase");
        let minimum = *interval.start();
        let maximum = *interval.end();
        let Some(&index) = path.get(phase + 1) else {
            let domain = IntegerRange::new(
                BlockChoice {
                    phase,
                    family: Arc::new(family),
                },
                minimum,
                maximum,
            )?;
            return Ok(Preparation::Choice {
                name: format!("CUDA phase {phase} work items per block"),
                alternatives: selection::Domain::new(domain)?,
            });
        };
        let threads = IntegerRange::new(phase, minimum, maximum)?
            .get(index)
            .ok_or("CUDA block choice is outside its domain")?;
        family
            .selected
            .push((threads * u64::from(family.lanes_per_item(phase).unwrap())) as u32);
    }
    if path.len() != family.selected.len() + 1 {
        return Err("unused CUDA execution decisions".into());
    }
    Ok(Preparation::Execution(family.finish()?))
}

impl compiler::Backend for Backend {
    type Execution = Vec<Execution>;
    type Conditions = Conditions;
    fn name(&self) -> &'static str {
        "cuda"
    }
    fn conditions(&self) -> Conditions {
        self.conditions.clone()
    }
    fn description(&self) -> compiler::Description {
        compiler::Description {
            target: format!("{} sm{}.{}", self.conditions.device.name,
                self.conditions.device.compute_capability.0, self.conditions.device.compute_capability.1),
            contracts: self.conditions.hardware.identity.clone(),
            form: "retained PTX; scalar and adjacent-tree subgroup folds, load, dispatch and per-phase block choices; fixed warp instruction order".into(),
            objective: "conditional best feasible device schedule for ordered resident PTX launches; submission excluded; native mapping unqualified".into(),
            timebase: self.conditions.hardware.timebase.clone(),
        }
    }
    fn prepare(
        &self,
        function: &LoweredIr,
        path: &[usize],
    ) -> Result<Preparation<Vec<Execution>>, String> {
        prepare(function, &self.conditions.device, path)
    }
    fn analyze(
        &self,
        execution: &Vec<Execution>,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
    ) -> Result<schedule::Model, DerivationError> {
        model::derive_sequence(execution, &self.conditions.hardware, workload, limits)
    }
    fn relax(
        &self,
        alternatives: &selection::Domain,
        indices: std::ops::Range<usize>,
        _workload: &ScalarWorkload,
    ) -> Result<Option<schedule::Demand>, String> {
        let Some(domain) = alternatives.owner::<IntegerRange<BlockChoice>>() else {
            return Ok(None);
        };
        relaxation::derive(domain, indices, &self.conditions.hardware)
    }
    fn materialize(
        &self,
        execution: &Vec<Execution>,
        objective: &Objective,
    ) -> Result<Vec<Execution>, String> {
        objective
            .model()
            .check_execution_upper(objective.schedule())?;
        Ok(execution.clone())
    }
    fn check_materialization(
        &self,
        source: &Vec<Execution>,
        selected: &Vec<Execution>,
        objective: &Objective,
    ) -> Result<(), String> {
        objective
            .model()
            .check_execution_upper(objective.schedule())?;
        if source.len() != selected.len()
            || source
                .iter()
                .zip(selected)
                .any(|(a, b)| !a.same_implementation(b))
        {
            return Err(
                "CUDA materialization changed retained PTX, launch geometry or invocation storage"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Diagnostic assignments traverse exactly the compiler's dependent family.
pub fn prepare_fixed(
    function: &LoweredIr,
    device: &DeviceInfo,
    options: seismic_realization::ScalarOptions,
    threads: u32,
) -> Result<Vec<Execution>, String> {
    let mut path = Vec::new();
    loop {
        match prepare(function, device, &path)? {
            Preparation::Execution(executions) => return Ok(executions),
            Preparation::Infeasible(reason) => {
                return Err(format!(
                    "CUDA diagnostic assignment is infeasible: {reason:?}"
                ));
            }
            Preparation::Choice { alternatives, .. } => {
                let index = if let Some(choice) = alternatives.owner::<loads::Choice>() {
                    let mode = if options.loads == seismic_realization::LoadStrategy::Materialize {
                        seismic_lang::ir::LoadMode::Materialize
                    } else {
                        seismic_lang::ir::LoadMode::Borrow
                    };
                    choice
                        .modes()
                        .iter()
                        .position(|&m| m == mode)
                        .ok_or("diagnostic load mode is outside the execution family")?
                } else if alternatives.owner::<FoldChoice>().is_some() {
                    0
                } else if let Some(choice) = alternatives.owner::<DispatchChoice>() {
                    (0..choice.len())
                        .find(|&i| choice.get(i) == Some(options.dispatch))
                        .ok_or("diagnostic dispatch is outside the execution family")?
                } else if let Some(choice) = alternatives.owner::<IntegerRange<BlockChoice>>() {
                    let lanes = choice
                        .decision
                        .family()
                        .lanes_per_item(choice.decision.phase)
                        .ok_or("missing CUDA phase")?;
                    if !threads.is_multiple_of(lanes) {
                        return Err("block size must contain complete participant groups".into());
                    }
                    choice
                        .index(u64::from(threads / lanes))
                        .ok_or("diagnostic block size is outside the execution family")?
                } else {
                    return Err("unknown CUDA execution decision".into());
                };
                path.push(index);
            }
        }
    }
}
