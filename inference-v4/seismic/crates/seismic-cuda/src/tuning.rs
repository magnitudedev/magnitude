//! CUDA execution families append their complete decisions to the common model.
//! Target instantiation consumes an assignment; it owns no selection search.
use crate::{
    DeviceInfo,
    execution::{Execution, Limits},
    model::CudaHardware,
};
use seismic_accounting::{
    choices::Choices,
    workload::{DerivationLimits, ScalarWorkload},
};
use seismic_compiler::tuner as compiler;
use seismic_lang::{lowered_ir::LoweredIr, normalize::loads};
use seismic_realization::{CallConv, Dispatch, ScalarProgram};
use std::sync::Arc;
mod export;
pub mod symbolic;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldImplementation {
    Thread,
    Subgroup,
    SubgroupInsertSeed,
    SubgroupWavefront,
    SubgroupWavefrontInsertSeed,
    SubgroupRootSeed,
    SubgroupWavefrontRootSeed,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldChoice {
    pub site: usize,
    pub wavefront: bool,
    pub root_seed: bool,
}
impl Choices for FoldChoice {
    type Alternative = FoldImplementation;
    fn len(&self) -> usize {
        if self.root_seed {
            if self.wavefront { 3 } else { 2 }
        } else if self.wavefront {
            5
        } else {
            3
        }
    }
    fn get(&self, index: usize) -> Option<Self::Alternative> {
        if self.root_seed {
            return [
                FoldImplementation::Thread,
                FoldImplementation::SubgroupRootSeed,
                FoldImplementation::SubgroupWavefrontRootSeed,
            ][..self.len()]
                .get(index)
                .copied();
        }
        [
            FoldImplementation::Thread,
            FoldImplementation::Subgroup,
            FoldImplementation::SubgroupInsertSeed,
            FoldImplementation::SubgroupWavefront,
            FoldImplementation::SubgroupWavefrontInsertSeed,
        ][..self.len()]
            .get(index)
            .copied()
    }
}

/// The admitted ownership implementations at each original reduction site.
/// Eligibility depends on source semantics and device participation capacity.
pub fn fold_choices(function: &LoweredIr, device: &DeviceInfo) -> Vec<FoldChoice> {
    use seismic_lang::reduction::structured::participants;
    if device.warp_size != 32 || device.max_threads_per_block < 32 {
        return Vec::new();
    }
    let roots = participants::root_seed_candidates(function, 32);
    let waves = participants::wavefront_candidates(function, 32);
    participants::candidates(function, 32)
        .into_iter()
        .map(|site| FoldChoice {
            site,
            root_seed: roots.contains(&site),
            wavefront: waves.contains(&site),
        })
        .collect()
}

/// Complete explicit assignment for the target's load, ownership and dispatch
/// decisions. Source decisions already belong to the supplied lowered function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub loads: Vec<seismic_lang::ir::LoadMode>,
    pub folds: Vec<(usize, FoldImplementation)>,
    pub dispatch: Dispatch,
}

/// One immutable terminal body per phase with unresolved legal block dimensions.
/// All block dimensions are resolved together; there is no dependent path state.
#[derive(Clone)]
pub struct BlockFamily {
    phases: Arc<[Phase]>,
    limits: Limits,
}
#[derive(Clone)]
struct Phase {
    program: Arc<ScalarProgram>,
    target: Arc<crate::ptx::TargetPlan>,
}
impl BlockFamily {
    pub fn from_selected(
        function: &LoweredIr,
        device: &DeviceInfo,
        selection: &Selection,
    ) -> Result<Self, String> {
        validate_device(device)?;
        if function.backend != "cuda" {
            return Err("CUDA family requires CUDA source".into());
        }
        seismic_lang::verify::lowered(function, seismic_lang::verify::Stage::Expanded)?;
        let mut function = function.clone();
        seismic_lang::normalize::bind_values(&mut function.body, &mut function.vars);
        loads::resolve(&mut function.body, &selection.loads)?;
        let choices = fold_choices(&function, device);
        if choices.len() != selection.folds.len() {
            return Err("CUDA fold assignment is incomplete".into());
        }
        use seismic_lang::reduction::structured::participants::{self, Completion, SeedPlacement};
        let mut selected = Vec::new();
        for (choice, &(site, implementation)) in choices.iter().zip(&selection.folds) {
            if choice.site != site
                || !(0..choice.len()).any(|i| choice.get(i) == Some(implementation))
            {
                return Err("CUDA fold assignment is outside its source ownership domain".into());
            }
            let placement = match implementation {
                FoldImplementation::Thread => None,
                FoldImplementation::Subgroup => {
                    Some((SeedPlacement::LeadingLeaf, Completion::RetainLeaves))
                }
                FoldImplementation::SubgroupInsertSeed => {
                    Some((SeedPlacement::InsertAfterSegments, Completion::RetainLeaves))
                }
                FoldImplementation::SubgroupWavefront => {
                    Some((SeedPlacement::LeadingLeaf, Completion::CompleteWaves))
                }
                FoldImplementation::SubgroupWavefrontInsertSeed => Some((
                    SeedPlacement::InsertAfterSegments,
                    Completion::CompleteWaves,
                )),
                FoldImplementation::SubgroupRootSeed => {
                    Some((SeedPlacement::AtRoot, Completion::RetainLeaves))
                }
                FoldImplementation::SubgroupWavefrontRootSeed => {
                    Some((SeedPlacement::AtRoot, Completion::CompleteWaves))
                }
            };
            if let Some((seed, completion)) = placement {
                selected.push(participants::Selection {
                    site,
                    seed,
                    completion,
                });
            }
        }
        let function = participants::apply(&function, &selected, 32)?.function;
        let subgroup = seismic_compiler::subgroup_required(&function);
        if subgroup && (device.warp_size != 32 || selection.dispatch != Dispatch::ParallelRoot) {
            return Err(
                "CUDA subgroup execution requires parallel dispatch and a 32-lane warp".into(),
            );
        }
        if selection.dispatch == Dispatch::ParallelRoot {
            if let seismic_realization::phases::Applicability::Unresolved { reason } =
                seismic_realization::phases::assess(&function)?
            {
                return Err(format!("CUDA phase realization: {reason}"));
            }
        }
        let sequence = seismic_compiler::scalar_sequence_participants_resolved(
            &function,
            CallConv::SystemV,
            selection.dispatch,
            if subgroup {
                seismic_realization::dispatch::Participation::Subgroup { lanes: 32 }
            } else {
                seismic_realization::dispatch::Participation::Thread
            },
        )?;
        if sequence.phases.is_empty() {
            return Err("CUDA requires a nonempty phase sequence".into());
        }
        let mut phases = Vec::new();
        for phase in sequence.phases {
            let target = Arc::new(crate::ptx::prepare(&phase.program)?);
            phases.push(Phase {
                program: Arc::new(phase.program),
                target,
            });
        }
        Ok(Self {
            phases: phases.into(),
            limits: Limits {
                max_threads_per_block: device.max_threads_per_block,
                max_grid_x: device.max_grid_x,
            },
        })
    }
    pub fn phase_count(&self) -> usize {
        self.phases.len()
    }
    pub fn target(&self, phase: usize) -> Option<&crate::ptx::TargetPlan> {
        self.phases.get(phase).map(|p| p.target.as_ref())
    }
    pub fn lanes_per_item(&self, phase: usize) -> Option<u32> {
        self.phases
            .get(phase)
            .map(|p| p.program.participation.lanes())
    }
    pub fn block_interval(&self, phase: usize) -> Option<std::ops::RangeInclusive<u64>> {
        let program = &self.phases.get(phase)?.program;
        Some(
            program
                .work_items
                .div_ceil(u64::from(self.limits.max_grid_x))
                .max(1)
                ..=u64::from(self.limits.max_threads_per_block / program.participation.lanes()),
        )
    }
    pub fn realize(&self, threads: &[u32]) -> Result<Vec<Execution>, String> {
        if threads.len() != self.phases.len() {
            return Err("CUDA block assignment must cover every phase".into());
        }
        self.phases
            .iter()
            .zip(threads)
            .map(|(phase, &threads)| {
                Execution::from_plan(
                    phase.program.clone(),
                    phase.target.clone(),
                    threads,
                    self.limits,
                )
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
            return Err("CUDA hardware geometry differs from the execution device".into());
        }
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
        return Err("CUDA requires compute capability 8.0 or newer".into());
    }
    Ok(())
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
            target: format!("{} sm{}.{}", self.conditions.device.name, self.conditions.device.compute_capability.0, self.conditions.device.compute_capability.1),
            contracts: self.conditions.hardware.identity.clone(),
            form: "retained PTX with source load, fold ownership, dispatch and complete per-phase block domains".into(),
            objective: "conditional minimum feasible ordered PTX launch completion; submission excluded; native mapping unqualified".into(),
            scheduling: seismic_accounting::authority::ScheduleInterpretation {
                instruction_order: seismic_accounting::authority::InstructionOrder::FixedByRealization,
                timing: seismic_accounting::authority::TimingSemantics::IdealResourceFeasible,
            }, timebase: self.conditions.hardware.timebase.clone(),
        }
    }
    fn export(
        &self,
        input: compiler::Input<'_>,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
    ) -> Result<compiler::family::Export<Self::Execution>, String> {
        export::export(input, &self.conditions, workload, limits)
    }
}
