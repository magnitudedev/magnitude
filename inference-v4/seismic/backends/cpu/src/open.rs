//! CPU backend composition over one exact production worker pool.

use crate::executor::{Device, Executor};
use crate::workers::Workers;
use crate::Cpu;
use seismic_compiler::errors::TargetError;

pub struct OpenedCpu {
    pub service: Device,
    pub executor: Executor,
    pub device: std::sync::Arc<seismic_native_target::DeviceDescription<Cpu>>,
}

pub fn open_host() -> Result<OpenedCpu, TargetError> {
    let workers =
        Workers::host().map_err(|error| TargetError::DeviceUnavailable(error.to_string()))?;
    open_workers(workers)
}

/// Opens execution over the production worker pool. Profiling is acquired
/// separately from this same executor only by an analytical evaluator.
pub(crate) fn open_workers(workers: Workers) -> Result<OpenedCpu, TargetError> {
    let device = crate::profile::device_for_workers(&workers)?;
    Ok(OpenedCpu {
        service: Device,
        executor: Executor::from_workers(workers),
        device,
    })
}
