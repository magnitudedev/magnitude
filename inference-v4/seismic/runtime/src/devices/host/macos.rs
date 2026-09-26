//! macOS 13+: `sysctl hw.memsize` and Mach `HOST_VM_INFO64`.
//!
//! macOS does not enforce `RLIMIT_AS`/`RLIMIT_DATA` against allocations, so
//! no process memory limit applies.

use super::{
    HeadroomBasis, HeadroomEstimate, HostCapacity, HostMeasurements, HostMemoryStatus,
    LimitVisibility,
};
use crate::devices::{CapacityBasis, ObservationError};
use std::time::SystemTime;

pub(super) fn capacity() -> Result<HostCapacity, String> {
    let mut bytes = 0_u64;
    let mut length = std::mem::size_of::<u64>();
    let status = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || length != std::mem::size_of::<u64>() || bytes == 0 {
        return Err(format!(
            "sysctl hw.memsize returned status {status} with {length} bytes"
        ));
    }
    Ok(HostCapacity {
        bytes,
        basis: CapacityBasis::InstalledRam,
    })
}

#[allow(deprecated)] // libc's Mach bindings are stable ABI; mach2 is not a dependency.
pub(super) fn status() -> Result<HostMemoryStatus, ObservationError> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size_bytes = u64::try_from(page_size)
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
    let sampled_at = SystemTime::now();
    // A kernel older than this libc fills its own prefix of the structure and
    // reports that count; every field read below is in the macOS 13 prefix.
    let statistics = unsafe { statistics.assume_init() };
    let free_pages = u64::from(statistics.free_count);
    let inactive_pages = u64::from(statistics.inactive_count);
    let headroom = free_pages
        .checked_add(inactive_pages)
        .and_then(|pages| pages.checked_mul(page_size_bytes))
        .ok_or_else(|| ObservationError::Failed("Mach headroom overflow".into()))?;
    Ok(HostMemoryStatus {
        sampled_at,
        measurements: HostMeasurements::MacOs {
            page_size_bytes,
            free_pages,
            speculative_pages: u64::from(statistics.speculative_count),
            active_pages: u64::from(statistics.active_count),
            inactive_pages,
            wired_pages: u64::from(statistics.wire_count),
            purgeable_pages: u64::from(statistics.purgeable_count),
            compressor_pages: u64::from(statistics.compressor_page_count),
            external_pages: u64::from(statistics.external_page_count),
            internal_pages: u64::from(statistics.internal_page_count),
        },
        headroom: HeadroomEstimate {
            bytes: headroom,
            basis: HeadroomBasis::MachFreeAndInactivePages,
        },
        limits: Vec::new(),
        limit_visibility: LimitVisibility::Complete,
    })
}
