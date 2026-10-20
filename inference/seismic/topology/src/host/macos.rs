//! macOS 15+: `sysctl hw.memsize`, Mach `HOST_VM_INFO64` and the kernel's
//! memory pressure level.
//!
//! macOS does not enforce `RLIMIT_AS`/`RLIMIT_DATA` against allocations, so
//! no process memory limit applies.

use super::{DisplacementCounters, HostSample, KernelPressure, LimitVisibility, ObservationError};
use crate::model::{Reported, ReportedBy};
use std::io;

/// `sysctlbyname(name)` into `buffer`, or its length alone without one: the
/// bytes the OS wrote, or would write.
fn sysctl(name: &str, buffer: Option<&mut [u8]>) -> io::Result<usize> {
    let name =
        std::ffi::CString::new(name).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    let (pointer, mut length): (*mut libc::c_void, usize) = match buffer {
        Some(buffer) => (buffer.as_mut_ptr().cast(), buffer.len()),
        None => (std::ptr::null_mut(), 0),
    };
    let status =
        unsafe { libc::sysctlbyname(name.as_ptr(), pointer, &mut length, std::ptr::null_mut(), 0) };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(length)
}

fn failed(name: &str, error: io::Error) -> String {
    format!("sysctl {name}: {error}")
}

/// Every byte `sysctl` `name` reports, whatever its length.
fn sysctl_bytes(name: &str) -> Result<Vec<u8>, String> {
    let length = sysctl(name, None).map_err(|error| failed(name, error))?;
    let mut bytes = vec![0u8; length];
    let length = sysctl(name, Some(&mut bytes)).map_err(|error| failed(name, error))?;
    bytes.truncate(length);
    Ok(bytes)
}

/// The value `sysctl` `name` reports, which is exactly `N` bytes long.
fn sysctl_exact<const N: usize>(name: &str) -> Result<[u8; N], String> {
    let mut bytes = [0u8; N];
    let length = sysctl(name, Some(&mut bytes)).map_err(|error| failed(name, error))?;
    if length != N {
        return Err(format!("sysctl {name} returned {length} bytes, not {N}"));
    }
    Ok(bytes)
}

pub(super) fn cpu_identity() -> Result<String, String> {
    let mut fields = Vec::new();
    for key in [
        "machdep.cpu.brand_string",
        "hw.model",
        "hw.ncpu",
        "hw.physicalcpu",
    ] {
        fields.push((key, sysctl_bytes(key)?));
    }
    serde_json::to_string(&fields).map_err(|error| error.to_string())
}

/// The OS version and build as Metal's owner reads it: the build ships the
/// Metal compiler.
pub(super) fn os_build() -> Result<String, String> {
    Ok(seismic_metal::profile::operating_system())
}

pub(super) fn capacity() -> Result<Reported<u64>, String> {
    let bytes = u64::from_ne_bytes(sysctl_exact("hw.memsize")?);
    if bytes == 0 {
        return Err("sysctl hw.memsize reported zero bytes".into());
    }
    Ok(Reported::new(bytes, ReportedBy::MacOsMemsize))
}

/// `kern.memorystatus_vm_pressure_level`: 1 normal, 2 warning, 4 critical.
fn kernel_pressure() -> Result<KernelPressure, ObservationError> {
    let name = "kern.memorystatus_vm_pressure_level";
    let level = i32::from_ne_bytes(sysctl_exact(name).map_err(ObservationError::Failed)?);
    match level {
        1 => Ok(KernelPressure::Normal),
        2 => Ok(KernelPressure::Warning),
        4 => Ok(KernelPressure::Critical),
        other => Err(ObservationError::Failed(format!(
            "{name} reported unknown level {other}"
        ))),
    }
}

/// One Mach `HOST_VM_INFO64` reading, and the page size it counts in.
pub(super) struct VmReading {
    pub(super) page_size: u64,
    pub(super) statistics: libc::vm_statistics64,
}

#[allow(deprecated)] // libc's Mach bindings are stable ABI; mach2 is not a dependency.
pub(super) fn vm_reading() -> Result<VmReading, ObservationError> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size = u64::try_from(page_size)
        .ok()
        .filter(|size| *size > 0)
        .ok_or_else(|| ObservationError::Failed(format!("page size query returned {page_size}")))?;
    let mut statistics = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let result = unsafe {
        libc::host_statistics64(
            libc::mach_host_self(),
            libc::HOST_VM_INFO64,
            statistics.as_mut_ptr().cast(),
            &mut count,
        )
    };
    if result != libc::KERN_SUCCESS {
        return Err(ObservationError::Failed(format!(
            "host_statistics64(HOST_VM_INFO64) returned {result}"
        )));
    }
    // A kernel older than this libc fills its own prefix of the structure and
    // reports that count; every field read from it is in the macOS 13 prefix.
    Ok(VmReading {
        page_size,
        statistics: unsafe { statistics.assume_init() },
    })
}

pub(super) fn sample() -> Result<HostSample, ObservationError> {
    sample_of(&vm_reading()?)
}

/// The sample of one `HOST_VM_INFO64` reading.
pub(super) fn sample_of(reading: &VmReading) -> Result<HostSample, ObservationError> {
    let statistics = &reading.statistics;
    let free_pages = u64::from(statistics.free_count);
    let active_pages = u64::from(statistics.active_count);
    let inactive_pages = u64::from(statistics.inactive_count);
    // Neither list says whether its pages are needed: a program's fresh data
    // sits on the inactive list and idle file cache on the active one. Their
    // sum with free pages is everything that is not wired or compressed.
    let headroom = free_pages
        .checked_add(active_pages)
        .and_then(|pages| pages.checked_add(inactive_pages))
        .and_then(|pages| pages.checked_mul(reading.page_size))
        .ok_or_else(|| ObservationError::Failed("Mach headroom overflow".into()))?;
    Ok(HostSample {
        headroom: Reported::new(headroom, ReportedBy::MachMovablePages),
        limits: Vec::new(),
        limit_visibility: LimitVisibility::Complete,
        displacement: Some(DisplacementCounters {
            compressed_pages: statistics.compressions,
            decompressed_pages: statistics.decompressions,
            swapped_out_pages: statistics.swapouts,
            swapped_in_pages: statistics.swapins,
        }),
        kernel_pressure: Some(kernel_pressure()?),
    })
}
