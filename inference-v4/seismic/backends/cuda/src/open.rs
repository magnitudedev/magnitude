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
