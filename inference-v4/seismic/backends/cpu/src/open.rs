//! CPU backend composition over one exact production worker pool.

use crate::executor::{Device, Executor};
use crate::workers::Workers;
use crate::Cpu;
use seismic_compiler::errors::TargetError;
use seismic_compiler::evaluation::AnalyticalEvaluationContext;

pub struct OpenedCpu {
    pub service: Device,
    pub executor: Executor,
    pub device: std::sync::Arc<seismic_target::DeviceDescription<Cpu>>,
    pub analytical: Result<AnalyticalEvaluationContext<Cpu>, TargetError>,
}

pub fn open_host() -> Result<OpenedCpu, TargetError> {
    let workers =
        Workers::host().map_err(|error| TargetError::DeviceUnavailable(error.to_string()))?;
    open_workers(workers)
}

/// Closes execution and analytical state over one exact production worker
/// pool, returning the resulting authorities as named sibling values.
pub(crate) fn open_workers(mut workers: Workers) -> Result<OpenedCpu, TargetError> {
    let device = crate::profile::device_for_workers(&workers)?;
    let analytical = crate::profile::profile_for_workers(&mut workers, device.clone());
    Ok(OpenedCpu {
        service: Device,
        executor: Executor::from_workers(workers),
        device,
        analytical,
    })
}
