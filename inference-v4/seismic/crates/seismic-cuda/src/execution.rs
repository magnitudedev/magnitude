//! Realized scalar CUDA execution, resolved without a driver or native compilation.
//! One work item (piece) per CUDA thread, or per warp of lanes when the body uses
//! participant intrinsics. The block size is supplied by the mapping's launch rule; native
//! registers and occupancy are not modeled.
use seismic_realization::{ScalarProgram, dispatch::GroupDispatch};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_threads_per_block: u32,
    pub max_grid_x: u32,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvocationStorage {
    pub buffer_table_bytes: usize,
    pub scalar_bytes: usize,
    pub scratch_bytes: usize,
    pub status_bytes: usize,
}
#[derive(Clone)]
pub struct Execution {
    program: Arc<ScalarProgram>,
    target: Arc<crate::ptx::TargetPlan>,
    dispatch: GroupDispatch,
    storage: InvocationStorage,
}
impl Execution {
    pub fn new(
        program: ScalarProgram,
        threads_per_block: u32,
        limits: Limits,
    ) -> Result<Self, String> {
        Self::new_for_target(
            program,
            threads_per_block,
            limits,
            crate::target::PtxTarget::SCALAR_BASELINE,
        )
    }

    /// Prepare terminal PTX for the target selected from the execution device's normalized
    /// profile. `new` remains the hardware-independent scalar-baseline constructor used by
    /// existing inspection tools and tests.
    pub fn new_for_target(
        program: ScalarProgram,
        threads_per_block: u32,
        limits: Limits,
        selected_target: crate::target::PtxTarget,
    ) -> Result<Self, String> {
        let target = crate::ptx::prepare(&program, selected_target)?;
        Self::from_plan(
            Arc::new(program),
            Arc::new(target),
            threads_per_block,
            limits,
        )
    }
    /// Resolve launch geometry without replacing the already selected target IR.
    pub(crate) fn from_plan(
        program: Arc<ScalarProgram>,
        target: Arc<crate::ptx::TargetPlan>,
        threads_per_block: u32,
        limits: Limits,
    ) -> Result<Self, String> {
        if threads_per_block == 0 || threads_per_block > limits.max_threads_per_block {
            return Err("CUDA block size exceeds device capability".into());
        }
        let lanes = u64::from(program.participation.lanes());
        if lanes == 0 || !u64::from(threads_per_block).is_multiple_of(lanes) {
            return Err("CUDA block must contain whole logical participant groups".into());
        }
        let dispatch = GroupDispatch::new(
            program.work_items,
            lanes,
            u64::from(threads_per_block) / lanes,
        )?;
        if dispatch.groups > u64::from(limits.max_grid_x) {
            return Err("CUDA domain exceeds one-dimensional grid capability".into());
        }
        let work_items = usize::try_from(dispatch.participating_lanes())
            .map_err(|_| "CUDA work domain exceeds address range")?;
        let storage = InvocationStorage {
            buffer_table_bytes: program
                .buffers
                .len()
                .checked_mul(8)
                .ok_or("CUDA buffer table overflow")?,
            scalar_bytes: program
                .scalars
                .len()
                .checked_mul(8)
                .ok_or("CUDA scalar table overflow")?,
            scratch_bytes: program
                .scratch_bytes
                .checked_mul(work_items)
                .ok_or("CUDA scratch size overflow")?,
            status_bytes: work_items
                .checked_mul(4)
                .ok_or("CUDA status size overflow")?,
        };
        Ok(Self {
            program,
            target,
            dispatch,
            storage,
        })
    }
    pub fn program(&self) -> &ScalarProgram {
        &self.program
    }
    /// The exact terminal implementation retained before printing/native compilation.
    pub fn target_plan(&self) -> &crate::ptx::TargetPlan {
        &self.target
    }
    pub fn dispatch(&self) -> &GroupDispatch {
        &self.dispatch
    }
    pub fn storage(&self) -> &InvocationStorage {
        &self.storage
    }
    pub(crate) fn validate_limits(&self, limits: Limits) -> Result<(), String> {
        if self.dispatch.threads_per_group > u64::from(limits.max_threads_per_block)
            || self.dispatch.groups > u64::from(limits.max_grid_x)
        {
            return Err("selected CUDA dispatch exceeds the execution device's limits".into());
        }
        Ok(())
    }
}

/// The realized execution of one selected entry: its launches in source order, each a
/// retained terminal PTX program with its launch geometry. Physical completion separates
/// consecutive launches; values crossing a launch use invocation-owned buffers appended
/// to the public binding table.
#[derive(Clone)]
pub struct Launches {
    pub name: String,
    pub phases: Vec<Execution>,
}
impl Launches {
    /// PTX text of every launch, printed from the retained terminal programs.
    pub fn ptx(&self) -> Vec<String> {
        self.phases
            .iter()
            .map(|phase| crate::ptx::print(phase.target_plan()))
            .collect()
    }
}
impl std::fmt::Debug for Launches {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let geometry: Vec<(u64, u64)> = self
            .phases
            .iter()
            .map(|p| (p.dispatch().groups, p.dispatch().threads_per_group))
            .collect();
        f.debug_struct("Launches")
            .field("name", &self.name)
            .field("blocks_x_threads", &geometry)
            .finish()
    }
}
