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
/// from a process, and this process's context. It does not depend on other
/// programs. The context is released afterwards, so a process that only
/// measures holds no device.
///
/// Under WDDM `cuMemGetInfo` free, read once the context exists and before
/// the process allocates, is exactly that: the process's budget less its own
/// use, whatever other programs hold. Elsewhere (Linux) it is the whole
/// device's free memory, which other programs' allocations lower, so the
/// figure is NVML's total less its reserve less what this process's context
/// took, the fall in NVML's device-wide free memory across opening it.
pub fn usable_memory_bytes(ordinal: u32, uuid: [u8; 16]) -> Result<u64, ExecutionError> {
    #[cfg(target_os = "windows")]
    {
        let _ = uuid;
        Ok(Device::open(ordinal)?.memory_info()?.free_bytes)
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
        let before = nvml.as_ref().and_then(|nvml| nvml.sample());
        let device = Device::open(ordinal)?;
        let free = device.memory_info()?.free_bytes;
        let after = nvml.as_ref().and_then(|nvml| nvml.sample());
        drop(device);
        let accounting = before.zip(after).map(|(before, after)| DeviceAccounting {
            total: after.total,
            reserved: after.reserved,
            free_before_context: before.free,
            free_after_context: after.free,
        });
        Ok(usable_capacity(free, accounting))
    }
}

/// A device's memory accounting across opening this process's context, in
/// one device-wide basis (NVML).
#[cfg_attr(target_os = "windows", allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeviceAccounting {
    total: u64,
    reserved: u64,
    free_before_context: u64,
    free_after_context: u64,
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
    let context = accounting
        .free_before_context
        .saturating_sub(accounting.free_after_context);
    accounting
        .total
        .saturating_sub(accounting.reserved)
        .saturating_sub(context)
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

    fn a10g(others: u64) -> DeviceAccounting {
        let (total, reserved, context) = (23_028 * MIB, 439 * MIB, 262 * MIB);
        let free = total - reserved - others;
        DeviceAccounting {
            total,
            reserved,
            free_before_context: free,
            free_after_context: free - context,
        }
    }

    #[test]
    fn capacity_is_total_less_reserve_less_context() {
        let accounting = a10g(0);
        assert_eq!(
            usable_capacity(accounting.free_after_context, Some(accounting)),
            (23_028 - 439 - 262) * MIB
        );
    }

    #[test]
    fn other_programs_do_not_change_capacity() {
        let held = a10g(5 << 30);
        assert_eq!(
            usable_capacity(held.free_after_context, Some(held)),
            usable_capacity(a10g(0).free_after_context, Some(a10g(0)))
        );
    }

    #[test]
    fn without_accounting_the_free_figure_stands() {
        assert_eq!(usable_capacity(17_000 * MIB, None), 17_000 * MIB);
    }

    #[test]
    fn capacity_is_never_below_free_memory() {
        let mut racing = a10g(0);
        racing.free_after_context -= 4 << 30;
        let free = racing.free_after_context + (4 << 30);
        assert_eq!(usable_capacity(free, Some(racing)), free);
    }

    #[test]
    fn memory_freed_while_opening_does_not_underflow() {
        let mut racing = a10g(0);
        racing.free_after_context = racing.free_before_context + MIB;
        assert_eq!(usable_capacity(0, Some(racing)), (23_028 - 439) * MIB);
    }
}
