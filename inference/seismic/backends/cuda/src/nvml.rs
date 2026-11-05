//! Minimal dynamically loaded NVML: one device's memory accounting
//! (`nvmlDeviceGetMemoryInfo_v2`), which separates the driver's reserve from
//! what processes use, and what it holds for one process
//! (`nvmlDeviceGetComputeRunningProcesses_v3`). Like the CUDA driver it
//! needs no headers or libraries to build; a host without NVML or without an
//! entry point yields `None` for what it would read. The device is found by
//! its CUDA UUID, so CUDA and NVML enumeration order and
//! `CUDA_VISIBLE_DEVICES` do not matter.

use libloading::Library;
use std::ffi::{c_char, c_int, c_uint, c_void, CString};

type Return = c_int;
type DeviceHandle = *mut c_void;

/// `nvmlMemory_v2_t`.
#[repr(C)]
#[derive(Default)]
struct MemoryV2 {
    version: c_uint,
    total: u64,
    reserved: u64,
    free: u64,
    used: u64,
}

/// `nvmlMemory_v2`: the structure's size with API version 2 in the top byte.
const MEMORY_V2_VERSION: c_uint = std::mem::size_of::<MemoryV2>() as c_uint | (2 << 24);

/// One NVML memory sample of a whole device: `total = reserved + used +
/// free`, where `used` counts every process's allocations and contexts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MemorySample {
    pub total: u64,
    pub reserved: u64,
    pub free: u64,
}

/// `nvmlProcessInfo_t`, as `nvmlDeviceGetComputeRunningProcesses_v3` fills
/// it.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcessInfo {
    pid: c_uint,
    used_gpu_memory: u64,
    gpu_instance_id: c_uint,
    compute_instance_id: c_uint,
}

/// `NVML_ERROR_INSUFFICIENT_SIZE`: more processes than the buffer holds;
/// the count says how many.
const INSUFFICIENT_SIZE: Return = 7;
/// `NVML_VALUE_NOT_AVAILABLE`, as a process's used memory.
const NOT_AVAILABLE: u64 = u64::MAX;

type RunningProcesses = unsafe extern "C" fn(DeviceHandle, *mut c_uint, *mut ProcessInfo) -> Return;

/// An initialized NVML session bound to one device; shut down on drop.
pub(crate) struct NvmlDevice {
    shutdown: unsafe extern "C" fn() -> Return,
    memory_info: unsafe extern "C" fn(DeviceHandle, *mut MemoryV2) -> Return,
    running_processes: Option<RunningProcesses>,
    handle: DeviceHandle,
    _library: Library,
}

impl NvmlDevice {
    /// The NVML device whose UUID is `uuid` (`cuDeviceGetUuid_v2`), as a
    /// whole GPU or a MIG partition.
    pub fn open(uuid: [u8; 16]) -> Option<Self> {
        // NVML symbols have the documented C ABI; the library is kept in the
        // returned value for as long as its function pointers are used.
        unsafe {
            let library = Library::new("libnvidia-ml.so.1").ok()?;
            let init = *library
                .get::<unsafe extern "C" fn() -> Return>(b"nvmlInit_v2\0")
                .ok()?;
            let shutdown = *library
                .get::<unsafe extern "C" fn() -> Return>(b"nvmlShutdown\0")
                .ok()?;
            let by_uuid = *library
                .get::<unsafe extern "C" fn(*const c_char, *mut DeviceHandle) -> Return>(
                    b"nvmlDeviceGetHandleByUUID\0",
                )
                .ok()?;
            let memory_info = *library
                .get::<unsafe extern "C" fn(DeviceHandle, *mut MemoryV2) -> Return>(
                    b"nvmlDeviceGetMemoryInfo_v2\0",
                )
                .ok()?;
            let running_processes = library
                .get::<RunningProcesses>(b"nvmlDeviceGetComputeRunningProcesses_v3\0")
                .ok()
                .map(|symbol| *symbol);
            if init() != 0 {
                return None;
            }
            let mut handle = std::ptr::null_mut();
            let found = ["GPU", "MIG"].into_iter().any(|prefix| {
                let text = CString::new(uuid_text(prefix, uuid)).expect("UUID text has no NUL");
                by_uuid(text.as_ptr(), &mut handle) == 0
            });
            if !found {
                shutdown();
                return None;
            }
            Some(Self {
                shutdown,
                memory_info,
                running_processes,
                handle,
                _library: library,
            })
        }
    }

    /// What the device's driver holds for process `pid` (its context and
    /// allocations), as NVML counts it.
    pub fn process_bytes(&self, pid: u32) -> Result<u64, ProcessMemoryError> {
        let query = self.running_processes.ok_or(ProcessMemoryError::Unsupported)?;
        // A count of zero asks how many processes there are; a process may
        // start between two queries, so the room asked for is the last count.
        let mut processes = Vec::new();
        loop {
            let mut count = c_uint::try_from(processes.len()).expect("a process count NVML gave");
            match unsafe { query(self.handle, &mut count, processes.as_mut_ptr()) } {
                0 => {
                    processes.truncate(count as usize);
                    break;
                }
                INSUFFICIENT_SIZE => processes.resize(count as usize, ProcessInfo::default()),
                code => return Err(ProcessMemoryError::Query(code)),
            }
        }
        own_bytes(&processes, pid)
    }

    pub fn sample(&self) -> Option<MemorySample> {
        let mut memory = MemoryV2 {
            version: MEMORY_V2_VERSION,
            ..MemoryV2::default()
        };
        if unsafe { (self.memory_info)(self.handle, &mut memory) } != 0 {
            return None;
        }
        Some(MemorySample {
            total: memory.total,
            reserved: memory.reserved,
            free: memory.free,
        })
    }
}

/// Why NVML gave no count of a process's memory on a device.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ProcessMemoryError {
    /// NVML is not loaded, or has no device of this UUID.
    Unavailable,
    /// NVML lacks `nvmlDeviceGetComputeRunningProcesses_v3`.
    Unsupported,
    /// The query failed with this NVML return code.
    Query(Return),
    /// NVML lists no process `pid` on the device, though it holds a context
    /// there: the process knows itself by a PID NVML does not use (another
    /// PID namespace).
    Unlisted { pid: u32 },
    /// NVML lists the process without its memory.
    Unreported { pid: u32 },
}

impl std::fmt::Display for ProcessMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable => write!(f, "NVML is unavailable for the device"),
            Self::Unsupported => write!(f, "NVML lacks nvmlDeviceGetComputeRunningProcesses_v3"),
            Self::Query(code) => write!(f, "NVML's process query failed ({code})"),
            Self::Unlisted { pid } => write!(
                f,
                "NVML lists no process {pid} on a device it holds a context on: its PID namespace is not NVML's"
            ),
            Self::Unreported { pid } => write!(f, "NVML does not report process {pid}'s memory"),
        }
    }
}

/// What NVML's `processes` say process `pid` holds.
fn own_bytes(processes: &[ProcessInfo], pid: u32) -> Result<u64, ProcessMemoryError> {
    let process = processes
        .iter()
        .find(|process| process.pid == pid)
        .ok_or(ProcessMemoryError::Unlisted { pid })?;
    match process.used_gpu_memory {
        NOT_AVAILABLE => Err(ProcessMemoryError::Unreported { pid }),
        bytes => Ok(bytes),
    }
}

/// What this process's context holds on the device `uuid`, as NVML counts
/// the process there: read once, while the process holds the context and no
/// allocation on the device (`seismic_native_target::process::DriverMemory`).
pub(crate) fn context_bytes(uuid: [u8; 16]) -> Result<u64, ProcessMemoryError> {
    NvmlDevice::open(uuid)
        .ok_or(ProcessMemoryError::Unavailable)?
        .process_bytes(std::process::id())
}

impl Drop for NvmlDevice {
    fn drop(&mut self) {
        unsafe {
            (self.shutdown)();
        }
    }
}

/// NVML's UUID text: `<prefix>-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
fn uuid_text(prefix: &str, uuid: [u8; 16]) -> String {
    let hex = |range: std::ops::Range<usize>| {
        uuid[range]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };
    format!(
        "{prefix}-{}-{}-{}-{}-{}",
        hex(0..4),
        hex(4..6),
        hex(6..8),
        hex(8..10),
        hex(10..16)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_v2_version_matches_the_nvml_header() {
        assert_eq!(std::mem::size_of::<MemoryV2>(), 40);
        assert_eq!(MEMORY_V2_VERSION, 0x0200_0028);
    }

    #[test]
    fn a_process_reads_its_own_count_or_why_it_has_none() {
        let process = |pid, used_gpu_memory| ProcessInfo {
            pid,
            used_gpu_memory,
            ..ProcessInfo::default()
        };
        let listed = [process(7, 100), process(9, NOT_AVAILABLE), process(11, 5)];
        assert_eq!(own_bytes(&listed, 11), Ok(5));
        assert_eq!(own_bytes(&listed, 9), Err(ProcessMemoryError::Unreported { pid: 9 }));
        assert_eq!(own_bytes(&listed, 8), Err(ProcessMemoryError::Unlisted { pid: 8 }));
        assert_eq!(own_bytes(&[], 8), Err(ProcessMemoryError::Unlisted { pid: 8 }));
    }

    #[test]
    fn process_info_matches_the_nvml_header() {
        assert_eq!(std::mem::size_of::<ProcessInfo>(), 24);
        assert_eq!(std::mem::offset_of!(ProcessInfo, used_gpu_memory), 8);
    }

    #[test]
    fn uuid_text_is_nvml_formatted() {
        let uuid = [
            0x6e, 0x2b, 0x0e, 0x1c, 0x3a, 0x4f, 0x5d, 0x6e, 0x7f, 0x80, 0x91, 0xa2, 0xb3, 0xc4,
            0xd5, 0xe6,
        ];
        assert_eq!(
            uuid_text("GPU", uuid),
            "GPU-6e2b0e1c-3a4f-5d6e-7f80-91a2b3c4d5e6"
        );
    }
}
