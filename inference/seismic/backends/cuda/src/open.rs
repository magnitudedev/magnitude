//! CUDA backend composition over one exact native context and stream.

use crate::{Cuda, Device, MemoryAccounts};
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::evaluation::AnalyticalEvaluationContext;

pub struct OpenedCuda {
    pub service: Device,
    pub device: std::sync::Arc<seismic_native_target::DeviceDescription<Cuda>>,
}

/// Device `ordinal`, its own allocations charged to `accounts`.
pub fn open(ordinal: u32, accounts: MemoryAccounts) -> Result<OpenedCuda, ExecutionError> {
    let service = Device::open(ordinal, accounts)?;
    let device = crate::profile::device_for_opened(ordinal)
        .map_err(|error| ExecutionError::SubmissionFailed(error.to_string()))?;
    Ok(OpenedCuda { service, device })
}

/// The device memory this process can use for its own work: the device's
/// total less what the driver reserves, what the operating system withholds
/// from a process, and this process's context. It does not depend on other
/// programs. The context is released afterwards, so a process that only
/// measures holds no device.
///
/// Under WDDM `cuMemGetInfo` free, read once the context exists and before
/// the process allocates, is exactly that: the process's budget less its own
/// use, whatever other programs hold. Elsewhere (Linux) it is the whole
/// device's free memory, which other programs' allocations lower, so the
/// figure is NVML's total less its reserve less what this process's context
/// holds, as NVML counts the process once its context, its first, exists:
/// never a fall in the device's free memory, which other programs move
/// while the context opens.
pub fn usable_memory_bytes(ordinal: u32, uuid: [u8; 16]) -> Result<u64, ExecutionError> {
    #[cfg(target_os = "windows")]
    {
        let _ = uuid;
        Ok(crate::executor::memory_info(&context(ordinal)?)?.free_bytes)
    }
    #[cfg(not(target_os = "windows"))]
    {
        // The context's cost is measurable only while this process has no
        // context yet. When NVML (or its v2 memory query, or the device) is
        // unavailable, or a context already exists, the free figure stands.
        let driver = crate::driver::Driver::load().map_err(ExecutionError::SubmissionFailed)?;
        let fresh = !driver
            .primary_context_active(ordinal as std::ffi::c_int)
            .map_err(|error| ExecutionError::SubmissionFailed(error.to_string()))?;
        let nvml = fresh.then(|| crate::nvml::NvmlDevice::open(uuid)).flatten();
        let context = context(ordinal)?;
        let free = crate::executor::memory_info(&context)?.free_bytes;
        let accounting = nvml.as_ref().and_then(|nvml| {
            let sample = nvml.sample()?;
            let context = nvml.process_bytes(std::process::id()).ok()?;
            Some(DeviceAccounting { total: sample.total, reserved: sample.reserved, context })
        });
        drop(context);
        Ok(usable_capacity(free, accounting))
    }
}

/// Ordinal `ordinal`'s primary context, retained: what a measurement of the
/// context's cost opens, allocating nothing.
fn context(ordinal: u32) -> Result<std::sync::Arc<crate::driver::Context>, ExecutionError> {
    let driver = crate::driver::Driver::load().map_err(ExecutionError::SubmissionFailed)?;
    let ordinal = std::ffi::c_int::try_from(ordinal)
        .map_err(|_| ExecutionError::SubmissionFailed("CUDA device ordinal exceeds driver ABI".into()))?;
    crate::driver::Context::retain(driver, ordinal).map_err(|error| ExecutionError::SubmissionFailed(error.to_string()))
}

/// A device's memory accounting across opening this process's context, in
/// one device-wide basis (NVML).
#[cfg_attr(target_os = "windows", allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeviceAccounting {
    total: u64,
    reserved: u64,
    /// What this process's context holds, as NVML counts the process.
    context: u64,
}

/// Usable capacity from the free memory the opened context reports and,
/// when available, the device's accounting: the total less the reserve less
/// the context. It is never below the free memory this process can take now,
/// which bounds a misreported or racing sample (another program freeing
/// memory while the context opens understates the context, never below zero).
/// Without accounting the free figure is the capacity: exact when no other
/// program holds memory, lower by what they hold otherwise.
#[cfg_attr(target_os = "windows", allow(dead_code))]
fn usable_capacity(free_with_context: u64, accounting: Option<DeviceAccounting>) -> u64 {
    let Some(accounting) = accounting else {
        return free_with_context;
    };
    accounting
        .total
        .saturating_sub(accounting.reserved)
        .saturating_sub(accounting.context)
        .max(free_with_context)
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

#[cfg(test)]
mod tests {
    use super::{usable_capacity, DeviceAccounting};

    const MIB: u64 = 1 << 20;

    fn a10g() -> DeviceAccounting {
        DeviceAccounting { total: 23_028 * MIB, reserved: 439 * MIB, context: 262 * MIB }
    }

    /// What the device's free memory reads with the context open, while
    /// other programs hold `others`.
    fn free(others: u64) -> u64 {
        (23_028 - 439 - 262) * MIB - others
    }

    #[test]
    fn capacity_is_total_less_reserve_less_context() {
        assert_eq!(usable_capacity(free(0), Some(a10g())), (23_028 - 439 - 262) * MIB);
    }

    /// Other programs' memory, held before the context opens or taken and
    /// freed while it does, changes neither the context's count nor the
    /// capacity: the count is the process's own.
    #[test]
    fn other_programs_do_not_change_capacity() {
        assert_eq!(usable_capacity(free(5 << 30), Some(a10g())), usable_capacity(free(0), Some(a10g())));
    }

    #[test]
    fn without_accounting_the_free_figure_stands() {
        assert_eq!(usable_capacity(17_000 * MIB, None), 17_000 * MIB);
    }

    /// A misreported count never leaves less than the process can take now.
    #[test]
    fn capacity_is_never_below_free_memory() {
        let overcounted = DeviceAccounting { context: 4 << 30, ..a10g() };
        assert_eq!(usable_capacity(free(0), Some(overcounted)), free(0));
    }
}
