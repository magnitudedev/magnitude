//! macOS 15+: `sysctl hw.memsize` and Mach `HOST_VM_INFO64`.
//!
//! macOS does not enforce `RLIMIT_AS`/`RLIMIT_DATA` against allocations, so
//! no process memory limit applies.

use super::{HostHeld, HostStatus, LimitVisibility, ObservationError};
use seismic_topology::{Reported, ReportedBy};
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

/// The integer `sysctl` `name` reports, none where the OS defines no such
/// name, reports no value for it, or reports zero; any other failure is an
/// error.
fn sysctl_u64(name: &str) -> Result<Option<u64>, String> {
    let mut bytes = [0u8; 8];
    let length = match sysctl(name, Some(&mut bytes)) {
        Ok(length) => length,
        // `ENOENT`: the OS defines no such name. `EINVAL`: it defines the
        // name but reports no value for this machine, as macOS 15 does for
        // `hw.perflevel0.l3cachesize` and `hw.l3cachesize` on an M4 Pro
        // (read on an M4 Pro, where `hw.nonexistent` is `ENOENT`), though
        // the buffer holds any 32- or 64-bit value.
        Err(error) if matches!(error.raw_os_error(), Some(libc::ENOENT | libc::EINVAL)) => {
            return Ok(None);
        }
        Err(error) => return Err(failed(name, error)),
    };
    // The values are 32- or 64-bit integers by name and OS release.
    let value = match length {
        4 => u64::from(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        8 => u64::from_ne_bytes(bytes),
        _ => return Err(format!("sysctl {name} returned {length} bytes")),
    };
    Ok((value > 0).then_some(value))
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

/// The last-level caches of every core type (`hw.perflevel<n>.*` for each
/// of `hw.nperflevels`), summed over their instances: per core type, the
/// L3 where the OS reports one, else the L2, of `l<k>cachesize` bytes, one
/// instance per `cpusperl<k>` of its `logicalcpu`. A core type whose cache
/// or sharing the OS reports none of leaves the caches unknown; any other
/// failure is an error.
pub(super) fn last_level_cache() -> Result<Option<Reported<u64>>, String> {
    let Some(levels) = sysctl_u64("hw.nperflevels")? else {
        return Ok(None);
    };
    let mut total = 0_u64;
    for level in 0..levels {
        let field = |name: &str| sysctl_u64(&format!("hw.perflevel{level}.{name}"));
        let Some(logical) = field("logicalcpu")? else {
            return Ok(None);
        };
        let mut cache = None;
        for (size, sharing) in [("l3cachesize", "cpusperl3"), ("l2cachesize", "cpusperl2")] {
            if let Some(bytes) = field(size)? {
                cache = Some((bytes, field(sharing)?));
                break;
            }
        }
        let Some((bytes, Some(sharing))) = cache else {
            return Ok(None);
        };
        total += bytes * logical.div_ceil(sharing);
    }
    Ok(Some(Reported::new(total, ReportedBy::MacOsSysctlCache)))
}

pub(super) fn capacity() -> Result<Reported<u64>, String> {
    let bytes = u64::from_ne_bytes(sysctl_exact("hw.memsize")?);
    if bytes == 0 {
        return Err("sysctl hw.memsize reported zero bytes".into());
    }
    Ok(Reported::new(bytes, ReportedBy::MacOsMemsize))
}

/// One Mach `HOST_VM_INFO64` reading, the page size it counts in, and the
/// machine's free-page target.
pub(super) struct VmReading {
    pub(super) page_size: u64,
    pub(super) statistics: libc::vm_statistics64,
    /// `vm.vm_page_free_target`: the free pages the pageout daemon keeps,
    /// compressing, swapping and evicting other pages whenever free pages
    /// fall below it.
    pub(super) free_target: u64,
}

/// The machine's `vm.vm_page_free_target`, read once: the kernel sets it at
/// boot from the machine's memory.
fn free_target() -> Result<u64, ObservationError> {
    static TARGET: std::sync::OnceLock<Result<u64, String>> = std::sync::OnceLock::new();
    TARGET
        .get_or_init(|| {
            sysctl_u64("vm.vm_page_free_target")?
                .ok_or_else(|| "sysctl vm.vm_page_free_target reports no value".to_string())
        })
        .clone()
        .map_err(ObservationError::Failed)
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
        free_target: free_target()?,
    })
}

pub(super) fn sample() -> Result<HostStatus, ObservationError> {
    sample_of(&vm_reading()?)
}

/// The sample of one `HOST_VM_INFO64` reading.
pub(super) fn sample_of(reading: &VmReading) -> Result<HostStatus, ObservationError> {
    let statistics = &reading.statistics;
    // What can be taken without the OS reclaiming anything: free pages
    // (`free_count`, which counts speculative read-ahead pages too, less
    // `speculative_count`; vm_stat's "Pages free" is that difference
    // already) and volatile purgeable pages (`purgeable_count`, purged at
    // no one's cost), above the free-page target (`vm.vm_page_free_target`):
    // macOS refills free pages up to that target by compressing, swapping
    // and evicting other pages, so pages taken below it are never free
    // without displacing someone (a bank that took them grew from 65 to 195
    // slots while free pages stayed at the target, compressing a 24.7 GB
    // program's 1.58M anonymous pages to 43k). File-backed pages
    // (`external_page_count`) are not counted: macOS reclaims by page age
    // across anonymous and file-backed pages alike, so taking them under
    // pressure also compresses or swaps idle programs' anonymous pages.
    let free_pages = u64::from(statistics.free_count)
        .checked_sub(u64::from(statistics.speculative_count))
        .ok_or_else(|| {
            ObservationError::Failed("Mach speculative pages exceed free pages".into())
        })?;
    let allocatable = (free_pages + u64::from(statistics.purgeable_count))
        .saturating_sub(reading.free_target)
        .checked_mul(reading.page_size)
        .ok_or_else(|| ObservationError::Failed("Mach allocatable overflow".into()))?;
    // The file-backed pages a load may count once (`Reclaim::CleanFileCache`).
    let reclaimable = u64::from(statistics.external_page_count)
        .checked_mul(reading.page_size)
        .ok_or_else(|| ObservationError::Failed("Mach file-backed overflow".into()))?;
    Ok(HostStatus {
        allocatable: Reported::new(allocatable, ReportedBy::MachAllocatablePages),
        reclaimable,
        limits: Vec::new(),
        limit_visibility: LimitVisibility::Complete,
        held: Some(HostHeld {
            wired_bytes: pages_bytes(reading, statistics.wire_count, "wired")?,
            compressed_bytes: pages_bytes(reading, statistics.compressor_page_count, "compressor")?,
        }),
    })
}

/// The bytes of `pages` pages of `reading`'s size; an error naming `what`
/// past the bytes a `u64` counts.
fn pages_bytes(reading: &VmReading, pages: u32, what: &str) -> Result<u64, ObservationError> {
    u64::from(pages)
        .checked_mul(reading.page_size)
        .ok_or_else(|| ObservationError::Failed(format!("Mach {what} bytes overflow")))
}

/// One memory node of every core: per performance level (`hw.nperflevels`,
/// level 0 the performance cores), its `physicalcpu` cores of
/// `logicalcpu / physicalcpu` threads each; with no levels (Intel), the
/// `hw.physicalcpu` cores of `hw.logicalcpu` threads, every one a
/// performance core. A CPU is its ordinal, performance cores first, and the
/// process may run on every one ([`crate::ProcessorsReportedBy::MacOsSysctl`]).
pub(super) fn processors() -> Result<crate::Processors, String> {
    let reported =
        |name: &str| sysctl_u64(name)?.ok_or_else(|| format!("sysctl {name} reports no value"));
    let levels = match sysctl_u64("hw.nperflevels")? {
        Some(levels) => (0..levels)
            .map(|level| {
                let field = |name: &str| reported(&format!("hw.perflevel{level}.{name}"));
                Ok((field("physicalcpu")?, field("logicalcpu")?, level == 0))
            })
            .collect::<Result<Vec<_>, String>>()?,
        None => vec![(
            reported("hw.physicalcpu")?,
            reported("hw.logicalcpu")?,
            true,
        )],
    };
    let mut cores = Vec::new();
    let mut next = 0usize;
    for (physical, logical, performance) in levels {
        if logical % physical != 0 {
            return Err(format!("{logical} logical CPUs over {physical} cores"));
        }
        let threads = usize::try_from(logical / physical).map_err(|error| error.to_string())?;
        for _ in 0..physical {
            cores.push(crate::Core {
                threads: (next..next + threads).collect(),
                performance,
            });
            next += threads;
        }
    }
    crate::Processors::new(
        vec![crate::MemoryNode {
            id: 0,
            memory: true,
            cores,
        }],
        (0..next).collect(),
        crate::ProcessorsReportedBy::MacOsSysctl,
    )
}

/// macOS places no thread on a core it names: its scheduler places every
/// thread, so nothing is placed.
pub(super) fn place_process(_cpus: &[usize]) -> Result<(), String> {
    Ok(())
}

/// macOS places no thread on a core it names: its scheduler places every
/// thread, so nothing is placed.
pub(super) fn pin(_cpus: &[usize]) -> Result<(), String> {
    Ok(())
}

/// macOS reads no device's memory node.
pub(super) fn device_node(_location: crate::PciLocation) -> Result<crate::DeviceNode, String> {
    Ok(crate::DeviceNode::Unknown)
}
