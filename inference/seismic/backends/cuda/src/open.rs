//! CUDA backend composition over one exact native context and stream.

use crate::{Cuda, Device};
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::evaluation::AnalyticalEvaluationContext;

pub struct OpenedCuda {
    pub service: Device,
    pub device: std::sync::Arc<seismic_native_target::DeviceDescription<Cuda>>,
}

pub fn open(ordinal: u32) -> Result<OpenedCuda, ExecutionError> {
    let service = Device::open(ordinal)?;
    let device = crate::profile::device_for_opened(ordinal)
        .map_err(|error| ExecutionError::SubmissionFailed(error.to_string()))?;
    Ok(OpenedCuda { service, device })
}

/// The device memory this process can use for its own work: the device's
/// total less what the driver reserves, what the operating system withholds
/// from a process, and this process's context. The driver reports exactly
/// that as free memory once the context exists and before the process
/// allocates: under WDDM `cuMemGetInfo` is the process's budget less its own
/// use, whatever other programs hold. The context is released afterwards, so
/// a process that only measures holds no device.
pub fn usable_memory_bytes(ordinal: u32) -> Result<u64, ExecutionError> {
    Ok(Device::open(ordinal)?.memory_info()?.free_bytes)
}

pub fn open_analytical(
    service: &Device,
    device: std::sync::Arc<seismic_native_target::DeviceDescription<Cuda>>,
) -> Result<AnalyticalEvaluationContext<Cuda>, TargetError> {
    crate::profile::profile_for_opened(
        service.ordinal(),
        service.context(),
        service.stream(),
        device,
    )
}
