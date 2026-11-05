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
/// participant per performance core, so every CPU device of the process
/// shares it: two pools would place two participants on each core.
static HOST: Mutex<Weak<Mutex<Workers>>> = Mutex::new(Weak::new());

/// The host CPU over its pool of the `cores` performance cores the process
/// may run on ([`Workers::host`]): the pool any open CPU device of the
/// process holds already.
pub fn open_host(cores: std::num::NonZeroUsize) -> Result<OpenedCpu, TargetError> {
    let workers = {
        let mut host = HOST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match host.upgrade() {
            Some(workers) => workers,
            None => {
                let workers =
                    Arc::new(Mutex::new(Workers::host(cores).map_err(|error| {
                        TargetError::DeviceUnavailable(error.to_string())
                    })?));
                *host = Arc::downgrade(&workers);
                workers
            }
        }
    };
    open_workers(workers)
}

/// The host CPU over a pool of the process's quota, for this crate's tests,
/// which need no particular pool of it.
#[cfg(test)]
pub(crate) fn open_quota() -> Result<OpenedCpu, TargetError> {
    open_host(std::thread::available_parallelism().map_err(|error| TargetError::DeviceUnavailable(error.to_string()))?)
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
