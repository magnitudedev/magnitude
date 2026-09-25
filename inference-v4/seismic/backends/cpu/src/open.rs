//! CPU backend composition over one exact production worker pool.

use crate::executor::{Device, Executor};
use crate::workers::Workers;
use crate::Cpu;
use seismic_compiler::errors::TargetError;
use std::sync::{Arc, Mutex, Weak};

pub struct OpenedCpu {
    pub service: Device,
    pub executor: Executor,
    pub device: std::sync::Arc<seismic_native_target::DeviceDescription<Cpu>>,
}

/// The host pool while any opened CPU device holds it. The pool has one
/// participant per physical performance core, so every CPU device of the
/// process shares it: two pools would place two participants on each core.
static HOST: Mutex<Weak<Mutex<Workers>>> = Mutex::new(Weak::new());

pub fn open_host() -> Result<OpenedCpu, TargetError> {
    let workers = {
        let mut host = HOST.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        match host.upgrade() {
            Some(workers) => workers,
            None => {
                let workers = Arc::new(Mutex::new(
                    Workers::host().map_err(|error| TargetError::DeviceUnavailable(error.to_string()))?,
                ));
                *host = Arc::downgrade(&workers);
                workers
            }
        }
    };
    open_workers(workers)
}

/// Opens execution over the production worker pool. Profiling is acquired
/// separately from this same executor only by an analytical evaluator.
pub(crate) fn open_workers(workers: Arc<Mutex<Workers>>) -> Result<OpenedCpu, TargetError> {
    let device = crate::profile::device_for_workers(
        &workers.lock().expect("CPU worker-pool lock poisoned"),
    )?;
    Ok(OpenedCpu {
        service: Device,
        executor: Executor::from_workers(workers),
        device,
    })
}
