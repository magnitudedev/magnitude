//! Selected scalar CUDA execution, resolved without a driver or native compilation.
//! This baseline assigns one work item to each CUDA thread. It does not model
//! native registers/occupancy or select a performance-optimal block size.
use seismic_realization::{dispatch::GroupDispatch, ScalarProgram};

#[derive(Clone, Copy, Debug)]
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
pub struct Execution {
    program: ScalarProgram,
    dispatch: GroupDispatch,
    storage: InvocationStorage,
}
impl Execution {
    pub fn new(
        program: ScalarProgram,
        threads_per_block: u32,
        limits: Limits,
    ) -> Result<Self, String> {
        if threads_per_block == 0 || threads_per_block > limits.max_threads_per_block {
            return Err("CUDA block size exceeds device capability".into());
        }
        let dispatch = GroupDispatch::new(program.work_items, 1, u64::from(threads_per_block))?;
        if dispatch.groups > u64::from(limits.max_grid_x) {
            return Err("CUDA domain exceeds one-dimensional grid capability".into());
        }
        let work_items = usize::try_from(program.work_items)
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
            dispatch,
            storage,
        })
    }
    pub fn program(&self) -> &ScalarProgram {
        &self.program
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
