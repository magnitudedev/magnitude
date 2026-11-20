//! macOS 15+: `sysctl hw.memsize`, the kernel's live `vm.*` counts, the
//! counts only `host_statistics64` reads, and the task's own paging, each
//! read through [`Sys`].

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use super::{read, resolve};
use seismic_topology::{CacheSize, Reported, ReportedBy};
use std::num::NonZeroUsize;

pub(crate) use memory::TaskVm;

/// Every OS read discovery makes on macOS, its one owner. A `sysctl` name,
/// `system_profiler`, and `available_parallelism` may not answer ([`read`]);
/// the Mach calls and the page size answer on every release from macOS 15.
pub(crate) trait Sys {
    /// Every byte `sysctlbyname(name)` reports; none where the OS defines
    /// no such name on this machine (`ENOENT`: an Intel Mac's
    /// `iogpu.wired_limit_mb`), defines it but reports no value for this
    /// machine (`EINVAL`: `hw.perflevel0.l3cachesize` on an M4 Pro), or
    /// refuses the read.
    fn sysctl(&self, name: &str) -> Option<Vec<u8>>;
    /// The VM's counts (`HOST_VM_INFO64`), as `host_statistics64` serves
    /// them: fresh, or a snapshot taken within `STATISTICS_WINDOW`.
    fn vm_statistics(&self) -> libc::vm_statistics64;
    /// This process's task's `task_vm_info`, through revision 6 at least.
    fn task_vm(&self) -> TaskVm;
    /// The host page size, through [`seismic_memory::host_page`].
    fn page(&self) -> u64;
    /// The CPUs the process may keep busy at once
    /// (`seismic_threads::quota`, their one owner: `available_parallelism`,
    /// its affinity under a cgroup's CPU quota, or one).
    fn parallelism(&self) -> NonZeroUsize;
    /// `/usr/sbin/system_profiler SPHardwareDataType -json` stdout bytes;
    /// none where the binary is missing, refused, or exits non-zero.
    fn hardware_profiler_json(&self) -> Option<Vec<u8>>;
}

/// The running OS's readings.
pub(crate) struct LiveSys;

/// `sysctlbyname(name)` into `buffer`, or its length alone without one: the
/// bytes the OS wrote, or would write.
fn sysctl_into(name: &std::ffi::CStr, buffer: Option<&mut [u8]>) -> std::io::Result<usize> {
    let (pointer, mut length): (*mut libc::c_void, usize) = match buffer {
        Some(buffer) => (buffer.as_mut_ptr().cast(), buffer.len()),
        None => (std::ptr::null_mut(), 0),
    };
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            pointer,
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    match status {
        0 => Ok(length),
        _ => Err(std::io::Error::last_os_error()),
    }
}

impl Sys for LiveSys {
    fn sysctl(&self, name: &str) -> Option<Vec<u8>> {
        let text = std::ffi::CString::new(name)
            .unwrap_or_else(|_| panic!("invariant violated: sysctl name {name:?} holds a NUL"));
        // `EINVAL`: a name the OS defines and reports no value of here.
        let answered = |result: std::io::Result<usize>| match result {
            Err(error) if error.raw_os_error() == Some(libc::EINVAL) => None,
            result => read(format_args!("sysctl {name}"), result),
        };
        let mut bytes = vec![0u8; answered(sysctl_into(&text, None))?];
        let length = answered(sysctl_into(&text, Some(&mut bytes)))?;
        bytes.truncate(length);
        Some(bytes)
    }

    fn vm_statistics(&self) -> libc::vm_statistics64 {
        let mut statistics: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        let mut count = libc::HOST_VM_INFO64_COUNT;
        #[allow(deprecated)]
        let host = unsafe { libc::mach_host_self() };
        let status = unsafe {
            libc::host_statistics64(
                host,
                libc::HOST_VM_INFO64,
                (&mut statistics as *mut libc::vm_statistics64).cast(),
                &mut count,
            )
        };
        assert_eq!(
            status,
            libc::KERN_SUCCESS,
            "invariant violated: host_statistics64 HOST_VM_INFO64 returned {status}"
        );
        statistics
    }

    fn task_vm(&self) -> TaskVm {
        let (vm, count) = seismic_native_target::process::macos::task_vm_info_with_count()
            .unwrap_or_else(|status| {
                panic!("invariant violated: task_info TASK_VM_INFO returned {status}")
            });
        assert!(
            count >= seismic_native_target::process::macos::TASK_VM_INFO_REV6_COUNT,
            "invariant violated: task_vm_info has {count} words, without decompressions and ledger_swapins"
        );
        vm
    }

    fn page(&self) -> u64 {
        static PAGE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        *PAGE.get_or_init(|| {
            seismic_memory::host_page()
                .unwrap_or_else(|error| panic!("invariant violated: the host page: {error}"))
        })
    }

    fn parallelism(&self) -> NonZeroUsize {
        seismic_threads::quota()
    }

    fn hardware_profiler_json(&self) -> Option<Vec<u8>> {
        static PROFILER: std::sync::OnceLock<Option<Vec<u8>>> = std::sync::OnceLock::new();
        PROFILER
            .get_or_init(|| {
                let output = read(
                    "/usr/sbin/system_profiler SPHardwareDataType -json",
                    std::process::Command::new("/usr/sbin/system_profiler")
                        .args(["SPHardwareDataType", "-json"])
                        .output(),
                )?;
                output.status.success().then_some(output.stdout)
            })
            .clone()
    }
}

/// The count `sysctl` `name` reports, zero included: a 32- or 64-bit
/// integer by name and OS release.
///
/// # Panics
/// Where it reports another width.
pub(super) fn sysctl_count(sys: &impl Sys, name: &str) -> Option<u64> {
    let bytes = sys.sysctl(name)?;
    Some(match *bytes.as_slice() {
        [a, b, c, d] => u64::from(u32::from_ne_bytes([a, b, c, d])),
        [a, b, c, d, e, f, g, h] => u64::from_ne_bytes([a, b, c, d, e, f, g, h]),
        _ => panic!(
            "invariant violated: sysctl {name} returned {} bytes",
            bytes.len()
        ),
    })
}

/// The count `sysctl` `name` reports, none where it reports none or zero.
pub(super) fn sysctl_u64(sys: &impl Sys, name: &str) -> Option<u64> {
    sysctl_count(sys, name).filter(|value| *value > 0)
}

/// A count the OS reports on every release from macOS 15.
///
/// # Panics
/// Where it reports none.
pub(super) fn sysctl_always(sys: &impl Sys, name: &str) -> u64 {
    sysctl_count(sys, name)
        .unwrap_or_else(|| panic!("invariant violated: macOS reads no sysctl {name}"))
}

/// A C-string `sysctl` value, trimmed of trailing NULs and whitespace; none
/// where unreported or empty.
pub(super) fn sysctl_string(sys: &impl Sys, name: &str) -> Option<String> {
    let bytes = sys.sysctl(name)?;
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let text = String::from_utf8_lossy(&bytes[..end]).trim().to_owned();
    (!text.is_empty()).then_some(text)
}

/// Human-readable CPU brand string from `machdep.cpu.brand_string`, falling
/// back to `"CPU (<arch>)"`.
pub(super) fn cpu_model(sys: &impl Sys) -> String {
    resolve(
        &[((), &|| sysctl_string(sys, "machdep.cpu.brand_string"))],
        (),
        || format!("CPU ({})", std::env::consts::ARCH),
    )
    .0
}

/// `sysctl`'s brand string, model and CPU counts, each the OS reports; the
/// architecture alone (`unknown;arch=<arch>`) where it reports none.
pub(super) fn cpu_identity(sys: &impl Sys) -> String {
    resolve(
        &[((), &|| {
            let fields = [
                "machdep.cpu.brand_string",
                "hw.model",
                "hw.ncpu",
                "hw.physicalcpu",
            ]
            .into_iter()
            .filter_map(|key| Some((key, sys.sysctl(key)?)))
            .collect::<Vec<_>>();
            (!fields.is_empty()).then(|| {
                serde_json::to_string(&fields).unwrap_or_else(|error| {
                    panic!("invariant violated: sysctl fields as JSON: {error}")
                })
            })
        })],
        (),
        || format!("unknown;arch={}", std::env::consts::ARCH),
    )
    .0
}

/// The OS version and build from `kern.osproductversion` and `kern.osversion`
/// (`"macOS <product> (<build>)"`), falling back to `"macOS"`.
pub(super) fn os_build(sys: &impl Sys) -> String {
    resolve(
        &[
            ((), &|| {
                let product = sysctl_string(sys, "kern.osproductversion")?;
                let build = sysctl_string(sys, "kern.osversion")?;
                Some(format!("macOS {product} ({build})"))
            }),
            ((), &|| {
                sysctl_string(sys, "kern.osproductversion").map(|product| format!("macOS {product}"))
            }),
            ((), &|| {
                sysctl_string(sys, "kern.osversion").map(|build| format!("macOS ({build})"))
            }),
        ],
        (),
        || "macOS".into(),
    )
    .0
}

/// The CPU's reported cache: the highest level each core type reports
/// (`hw.perflevel<n>.*` for each of `hw.nperflevels`), summed over the core
/// types and their instances, the L3 where the OS reports one, else the
/// L2, of `l<k>cachesize` bytes, one instance per `cpusperl<k>` of its
/// `logicalcpu`. Where a core type's cache or sharing is not reported, the
/// largest L2 any reports (a core type's or `hw.l2cachesize`); else
/// [`CacheSize::UNREPORTED`].
///
/// `is_last_level` is `false` on macOS because Apple Silicon's system-level
/// cache (SLC) sits behind L2 and is not reported by sysctl.
pub(super) fn last_level_cache(sys: &impl Sys) -> CacheSize {
    let levels = resolve(&[((), &|| sysctl_u64(sys, "hw.nperflevels"))], (), || 0).0;
    let field = |level: u64, name: &str| sysctl_u64(sys, &format!("hw.perflevel{level}.{name}"));
    let highest = |level: u64| {
        let logical = field(level, "logicalcpu")?;
        let (bytes, sharing) = [("l3cachesize", "cpusperl3"), ("l2cachesize", "cpusperl2")]
            .into_iter()
            .find_map(|(size, sharing)| Some((field(level, size)?, field(level, sharing))))?;
        Some(bytes * logical.div_ceil(sharing?))
    };
    let bytes = resolve(
        &[
            ((), &|| {
                (levels > 0)
                    .then(|| (0..levels).map(highest).sum::<Option<u64>>())
                    .flatten()
            }),
            ((), &|| {
                (0..levels)
                    .filter_map(|level| field(level, "l2cachesize"))
                    .chain(sysctl_u64(sys, "hw.l2cachesize"))
                    .max()
            }),
        ],
        (),
        || 0,
    )
    .0;
    CacheSize {
        bytes,
        is_last_level: false,
    }
}

/// `hw.memsize`: installed RAM, which every macOS reports.
///
/// # Panics
/// Where it reports none, zero, or not 8 bytes.
pub(super) fn capacity(sys: &impl Sys) -> Reported<u64> {
    let bytes = sys
        .sysctl("hw.memsize")
        .unwrap_or_else(|| panic!("invariant violated: macOS reads no sysctl hw.memsize"));
    let bytes = <[u8; 8]>::try_from(bytes.as_slice()).unwrap_or_else(|_| {
        panic!(
            "invariant violated: sysctl hw.memsize returned {} bytes, not 8",
            bytes.len()
        )
    });
    match u64::from_ne_bytes(bytes) {
        0 => panic!("invariant violated: sysctl hw.memsize reported zero bytes"),
        bytes => Reported::new(bytes, ReportedBy::MacOsMemsize),
    }
}

/// macOS reports no condition of the host for the hardware report.
pub(super) fn diagnostics(_sys: &impl Sys) -> Vec<seismic_topology::Diagnostic> {
    Vec::new()
}

mod memory;
mod topology;
#[cfg(test)]
mod tests;

pub(crate) use memory::sample;
pub(crate) use topology::{device_node, firmware, pin, place_process, processors};
