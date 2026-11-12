//! macOS 15+: `sysctl hw.memsize` and the kernel's live `vm.page_*` counts.
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

/// One reading of the VM's page counts, the page size they count in, and
/// the machine's free-page target. The counts are the kernel's live
/// `vm.page_*` sysctls: `host_statistics64` serves an ordinary process a
/// cached figure past a few calls a second (an M4 Pro's refreshed once a
/// second against 10 ms reads, 2026-11-12), which no sample may pair with
/// its instant.
pub(super) struct VmReading {
    pub(super) page_size: u64,
    /// `vm.page_free_count`: free pages, read-ahead apart.
    pub(super) free: u64,
    /// `vm.page_purgeable_count`: volatile purgeable pages.
    pub(super) purgeable: u64,
    /// `vm.page_pageable_external_count`: file-backed pages.
    pub(super) external: u64,
    /// `vm.compressor_bytes_used`: the size of the data the compressor
    /// stores, not the physical pages that hold it.
    pub(super) compressed_bytes: u64,
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

/// The count `sysctl` `name` reports, zero included: a 32- or 64-bit
/// integer by name and OS release.
///
/// # Panics
/// Where this macOS defines no such name: the reader's names are each
/// release's from 15 on, so one missing is the invariant's violation, never
/// a zero or another source's figure.
fn sysctl_count(name: &str) -> Result<u64, ObservationError> {
    let mut bytes = [0u8; 8];
    let length = match sysctl(name, Some(&mut bytes)) {
        Ok(length) => length,
        Err(error) if error.raw_os_error() == Some(libc::ENOENT) => {
            let version = sysctl_bytes("kern.osproductversion").map(|bytes| String::from_utf8_lossy(&bytes).trim_end_matches('\0').to_owned());
            panic!("invariant violated: macOS {} defines no sysctl {name}", version.unwrap_or_else(|error| format!("of unknown version ({error})")))
        }
        Err(error) => return Err(ObservationError::Failed(failed(name, error))),
    };
    match length {
        4 => Ok(u64::from(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))),
        8 => Ok(u64::from_ne_bytes(bytes)),
        _ => Err(ObservationError::Failed(format!("sysctl {name} returned {length} bytes"))),
    }
}

pub(super) fn vm_reading() -> Result<VmReading, ObservationError> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size = u64::try_from(page_size)
        .ok()
        .filter(|size| *size > 0)
        .ok_or_else(|| ObservationError::Failed(format!("page size query returned {page_size}")))?;
    Ok(VmReading {
        page_size,
        free: sysctl_count("vm.page_free_count")?,
        purgeable: sysctl_count("vm.page_purgeable_count")?,
        external: sysctl_count("vm.page_pageable_external_count")?,
        compressed_bytes: sysctl_count("vm.compressor_bytes_used")?,
        free_target: free_target()?,
    })
}

pub(super) fn sample() -> Result<HostStatus, ObservationError> {
    sample_of(&vm_reading()?)
}

/// The sample of one reading of the VM's page counts.
pub(super) fn sample_of(reading: &VmReading) -> Result<HostStatus, ObservationError> {
    // What can be taken without the OS reclaiming anything: free pages
    // (`vm.page_free_count`, read-ahead pages apart, as vm_stat's "Pages
    // free") and volatile purgeable pages, purged at no one's cost, above
    // the free-page target (`vm.vm_page_free_target`): macOS refills free
    // pages up to that target by compressing, swapping and evicting other
    // pages, so pages taken below it are never free without displacing
    // someone (a bank that took them grew from 65 to 195 slots while free
    // pages stayed at the target, compressing a 24.7 GB program's 1.58M
    // anonymous pages to 43k). File-backed pages are not counted: macOS
    // reclaims by page age across anonymous and file-backed pages alike, so
    // taking them under pressure also compresses or swaps idle programs'
    // anonymous pages.
    let allocatable = (reading.free + reading.purgeable)
        .saturating_sub(reading.free_target)
        .checked_mul(reading.page_size)
        .ok_or_else(|| ObservationError::Failed("macOS free pages overflow".into()))?;
    // The file-backed pages a load may count once (`Reclaim::CleanFileCache`).
    let reclaimable = reading
        .external
        .checked_mul(reading.page_size)
        .ok_or_else(|| ObservationError::Failed("macOS file-backed pages overflow".into()))?;
    Ok(HostStatus {
        allocatable: Reported::new(allocatable, ReportedBy::MacOsFreePages),
        reclaimable,
        limits: Vec::new(),
        limit_visibility: LimitVisibility::Complete,
        held: Some(HostHeld { compressed_bytes: reading.compressed_bytes }),
    })
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
